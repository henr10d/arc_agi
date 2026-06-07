"""Minimal ONNX for ARC task194: make a 2x2 rotation mosaic of a 3x3 input.

Task rule: the input is a 3x3 grid. The 6x6 output is four 3x3 quadrants:
top-left is the original input, top-right is the input rotated 90 degrees
clockwise, bottom-left is the input rotated 90 degrees counterclockwise, and
bottom-right is the input rotated 180 degrees. The 6x6 result is padded to the
30x30 NeuroGolf one-hot tensor contract.

ONNX: slice the compact 3x3 one-hot core, flatten it, gather with a 6x6
index tensor so the mosaic is materialized directly, and pad to the output.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task194"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
CORE = 3
OUT = 6
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def solve(grid: np.ndarray) -> np.ndarray:
    """Return [[original, clockwise], [counterclockwise, 180-degree rotation]]."""
    g = np.asarray(grid, dtype=np.int64)
    return np.block([[g, np.rot90(g, -1)], [np.rot90(g, 1), np.rot90(g, 2)]])


def _grid_to_onehot(grid: List[List[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _mosaic_indices() -> np.ndarray:
    idx: list[int] = []
    for r in range(OUT):
        for c in range(OUT):
            qr, qc = r // CORE, c // CORE
            rr, cc = r % CORE, c % CORE
            if qr == 0 and qc == 0:
                src_r, src_c = rr, cc
            elif qr == 0 and qc == 1:
                src_r, src_c = CORE - 1 - cc, rr
            elif qr == 1 and qc == 0:
                src_r, src_c = cc, CORE - 1 - rr
            else:
                src_r, src_c = CORE - 1 - rr, CORE - 1 - cc
            idx.append(src_r * CORE + src_c)
    return np.asarray(idx, dtype=np.int64)


def build_gather_model() -> onnx.ModelProto:
    """Lowest-cost candidate: flatten the 3x3 core and gather a 6x6 mosaic."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    starts = _i64(inits, [0, 0], "starts")
    ends = _i64(inits, [CORE, CORE], "ends")
    axes = _i64(inits, [2, 3], "axes")
    flat_shape = _i64(inits, [1, C, CORE * CORE], "flat_shape")
    gather_idx = _i64(inits, _mosaic_indices().reshape(OUT, OUT), "gather_idx")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts, ends, axes], ["core"]),
            helper.make_node("Reshape", ["core", flat_shape], ["flat"]),
            helper.make_node("Gather", ["flat", gather_idx], ["mosaic"], axis=2),
            helper.make_node("Pad", ["mosaic"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - OUT, W - OUT]),
        ]
    )

    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_quadrant_model() -> onnx.ModelProto:
    """Reference candidate using explicit transpose, flips, and concatenation."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    starts = _i64(inits, [0, 0], "starts")
    ends = _i64(inits, [CORE, CORE], "ends")
    axes = _i64(inits, [2, 3], "axes")
    rev = _i64(inits, [2, 1, 0], "rev")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts, ends, axes], ["core"]),
            helper.make_node("Transpose", ["core"], ["t"], perm=[0, 1, 3, 2]),
            helper.make_node("Gather", ["t", rev], ["cw"], axis=3),
            helper.make_node("Gather", ["t", rev], ["ccw"], axis=2),
            helper.make_node("Gather", ["core", rev], ["vf"], axis=2),
            helper.make_node("Gather", ["vf", rev], ["r180"], axis=3),
            helper.make_node("Concat", ["core", "cw"], ["top"], axis=3),
            helper.make_node("Concat", ["ccw", "r180"], ["bottom"], axis=3),
            helper.make_node("Concat", ["top", "bottom"], ["mosaic"], axis=2),
            helper.make_node("Pad", ["mosaic"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - OUT, W - OUT]),
        ]
    )

    graph = helper.make_graph(nodes, f"{TASK_ID}_quadrants", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> tuple[int, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    bad = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data.get(split, []):
            inp = np.asarray(ex["input"], dtype=np.int64)
            exp = np.asarray(ex["output"], dtype=np.int64)
            if max(inp.shape) > H or max(exp.shape) > H:
                continue
            pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(ex["input"])))[: exp.shape[0], : exp.shape[1]]
            bad += int(not np.array_equal(pred, exp))
            total += 1
    return total - bad, total


def candidate_train_accuracy() -> Dict[str, tuple[int, int]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    transforms: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "2x_nearest": lambda g: np.repeat(np.repeat(g, 2, axis=0), 2, axis=1),
        "original_hflip_vflip_rot180": lambda g: np.block([[g, np.fliplr(g)], [np.flipud(g), np.rot90(g, 2)]]),
        "original_vflip_hflip_rot180": lambda g: np.block([[g, np.flipud(g)], [np.fliplr(g), np.rot90(g, 2)]]),
        "original_rot90cw_rot90ccw_rot180": solve,
    }
    out: Dict[str, tuple[int, int]] = {}
    examples = data.get("train", [])
    for name, fn in transforms.items():
        good = 0
        for ex in examples:
            good += int(np.array_equal(fn(np.asarray(ex["input"], dtype=np.int64)), np.asarray(ex["output"], dtype=np.int64)))
        out[name] = (good, len(examples))
    return out


def _model_stats(model: onnx.ModelProto, path: Path) -> Dict[str, int]:
    return {
        "bytes": path.stat().st_size,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "initializer_params": sum(int(np.prod(t.dims)) for t in model.graph.initializer),
    }


def main() -> None:
    candidates = {
        "gather": build_gather_model(),
        "quadrants": build_quadrant_model(),
    }
    tmp_scores = []
    with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}_candidates_") as tmp:
        tmp_dir = Path(tmp)
        for name, model in candidates.items():
            candidate_dir = tmp_dir / name
            candidate_dir.mkdir()
            path = candidate_dir / f"{TASK_ID}.onnx"
            onnx.save(model, path)
            correct, total = validate_json(model)
            result = score_file(path)
            tmp_scores.append((name, model, path, correct, total, result))

        valid = [item for item in tmp_scores if item[3] == item[4] and item[5]["valid"]]
        if not valid:
            raise RuntimeError("no fully correct valid candidate")
        best = min(valid, key=lambda item: int(item[5]["cost"]))
        best_name, best_model, _, _, _, best_result = best
        onnx.save(best_model, BEST_PATH)

        print("Hypothesis train accuracy:")
        for name, (good, total) in candidate_train_accuracy().items():
            print(f"  {name}: {good}/{total}")
        print()
        for name, model, path, correct, total, result in tmp_scores:
            stats = _model_stats(model, path)
            score = result["score"] if result["valid"] else None
            print(
                f"{name}: correct={correct}/{total} cost={result['cost']} "
                f"score={score} nodes={stats['nodes']} tensors={stats['initializers']} "
                f"params={result['params']} memory={result['memory']}"
            )
        print()
        print(f"Saved best model: {BEST_PATH} (from {best_name}, cost={best_result['cost']})")


if __name__ == "__main__":
    main()
