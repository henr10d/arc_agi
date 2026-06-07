"""Build ONNX for ARC task172: vertical palindrome from a 3x3 grid.

Task rule: every train/test/arc-gen example is a 3x3 grid. The output is a
6x3 grid formed by copying the input rows, then appending them in reverse
order, so rows ``A, B, C`` become ``A, B, C, C, B, A``. Padding outside the
6x3 output area remains all-zero in the NeuroGolf 30x30 one-hot tensor.

ONNX approach: crop the one-hot input to [1,10,3,3], gather rows
``[0, 1, 2, 2, 1, 0]`` on the height axis, and make the final Pad node write
the competition output directly.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task172"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
IN_H = IN_W = 3
OUT_H = 6
SHAPE = [1, C, H, W]
ROW_ORDER = [0, 1, 2, 2, 1, 0]
OPSET = 10
IR_VERSION = 10


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference solver: return the 6x3 vertically mirrored palindrome."""
    arr = np.asarray(grid, dtype=np.int64)
    return arr[ROW_ORDER, :]


def _examples() -> list[tuple[str, int, dict[str, list[list[int]]]]]:
    data = json.loads(DATA_PATH.read_text())
    return [
        (split, idx, example)
        for split in ("train", "test", "arc-gen")
        for idx, example in enumerate(data.get(split, []))
    ]


def _onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    arr = np.asarray(grid, dtype=np.int64)
    for r, row in enumerate(arr):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def _grid_from_onehot(arr: np.ndarray, h: int = OUT_H, w: int = IN_W) -> np.ndarray:
    return np.argmax(arr[0, :, :h, :w], axis=0).astype(np.int64)


def _init(inits: list[onnx.TensorProto], name: str, arr: np.ndarray) -> str:
    inits.append(numpy_helper.from_array(arr, name=name))
    return name


def _i64(inits: list[onnx.TensorProto], name: str, vals: list[int] | np.ndarray) -> str:
    return _init(inits, name, np.asarray(vals, dtype=np.int64))


def _make_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], graph_name: str) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        graph_name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_gather_model() -> onnx.ModelProto:
    """Lowest-cost candidate found: crop compact 3x3 core, gather six rows, pad."""
    inits: list[onnx.TensorProto] = []
    _i64(inits, "crop_st", [0, 0, 0, 0])
    _i64(inits, "crop_en", [1, C, IN_H, IN_W])
    _i64(inits, "row_idx", ROW_ORDER)

    nodes = [
        helper.make_node("Slice", ["input", "crop_st", "crop_en"], ["core"]),
        helper.make_node("Gather", ["core", "row_idx"], ["out6"], axis=2),
        helper.make_node("Pad", ["out6"], ["output"], pads=[0, 0, 0, 0, 0, 0, H - OUT_H, W - IN_W]),
    ]
    return _make_model(nodes, inits, "task172_gather")


def build_concat_slices_model() -> onnx.ModelProto:
    """Tie candidate by cost: slice rows 0/1/2 separately and concatenate repeats."""
    inits: list[onnx.TensorProto] = []
    _i64(inits, "axes_hw", [2, 3])
    _i64(inits, "r0_st", [0, 0])
    _i64(inits, "r0_en", [1, IN_W])
    _i64(inits, "r1_st", [1, 0])
    _i64(inits, "r1_en", [2, IN_W])
    _i64(inits, "r2_st", [2, 0])
    _i64(inits, "r2_en", [3, IN_W])

    nodes = [
        helper.make_node("Slice", ["input", "r0_st", "r0_en", "axes_hw"], ["r0"]),
        helper.make_node("Slice", ["input", "r1_st", "r1_en", "axes_hw"], ["r1"]),
        helper.make_node("Slice", ["input", "r2_st", "r2_en", "axes_hw"], ["r2"]),
        helper.make_node("Concat", ["r0", "r1", "r2", "r2", "r1", "r0"], ["out6"], axis=2),
        helper.make_node("Pad", ["out6"], ["output"], pads=[0, 0, 0, 0, 0, 0, H - OUT_H, W - IN_W]),
    ]
    return _make_model(nodes, inits, "task172_concat_slices")


def build_split_concat_model() -> onnx.ModelProto:
    """Higher-memory baseline: crop 3x3, split rows, concatenate palindrome."""
    inits: list[onnx.TensorProto] = []
    _i64(inits, "crop_st", [0, 0, 0, 0])
    _i64(inits, "crop_en", [1, C, IN_H, IN_W])

    nodes = [
        helper.make_node("Slice", ["input", "crop_st", "crop_en"], ["core"]),
        helper.make_node("Split", ["core"], ["r0", "r1", "r2"], axis=2, split=[1, 1, 1]),
        helper.make_node("Concat", ["r0", "r1", "r2", "r2", "r1", "r0"], ["out6"], axis=2),
        helper.make_node("Pad", ["out6"], ["output"], pads=[0, 0, 0, 0, 0, 0, H - OUT_H, W - IN_W]),
    ]
    return _make_model(nodes, inits, "task172_split_concat")


def check_hypotheses() -> dict[str, bool]:
    """Validate the prompt's alternatives against training examples."""
    candidates: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "vertical reflection": lambda x: x[ROW_ORDER, :],
        "bottom row duplicated then reversed": lambda x: np.concatenate([x, x[::-1]], axis=0),
        "generic palindrome construction": lambda x: np.concatenate([x, x[::-1]], axis=0),
    }
    results: dict[str, bool] = {}
    for name, fn in candidates.items():
        ok = True
        for split, _, example in _examples():
            if split != "train":
                continue
            inp = np.asarray(example["input"], dtype=np.int64)
            expected = np.asarray(example["output"], dtype=np.int64)
            ok = ok and inp.shape == (IN_H, IN_W) and expected.shape == (OUT_H, IN_W)
            ok = ok and np.array_equal(fn(inp), expected)
        results[name] = ok
    return results


def verify_reference() -> None:
    for split, idx, example in _examples():
        inp = np.asarray(example["input"], dtype=np.int64)
        expected = np.asarray(example["output"], dtype=np.int64)
        got = solve(inp)
        if inp.shape != (IN_H, IN_W) or expected.shape != (OUT_H, IN_W):
            raise AssertionError(f"{split}[{idx}] unexpected shape: {inp.shape} -> {expected.shape}")
        if not np.array_equal(got, expected):
            raise AssertionError(f"{split}[{idx}] reference mismatch:\n{got}\nexpected:\n{expected}")


def verify_model(model: onnx.ModelProto) -> None:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    for split, idx, example in _examples():
        inp = convert_to_numpy(example, "input")
        expected = _onehot(example["output"])
        if inp is None:
            continue
        pred = session.run(["output"], {"input": inp})[0]
        if pred.shape != tuple(SHAPE):
            raise AssertionError(f"{split}[{idx}] bad tensor shape {pred.shape}")
        if not np.array_equal(pred > 0.0, expected > 0.0):
            got_grid = _grid_from_onehot(pred)
            raise AssertionError(f"{split}[{idx}] model mismatch:\n{got_grid}\nexpected:\n{np.asarray(example['output'])}")


def benchmark_candidates() -> list[tuple[str, dict[str, object], onnx.ModelProto]]:
    builders = [
        ("gather", build_gather_model),
        ("concat_slices", build_concat_slices_model),
        ("split_concat", build_split_concat_model),
    ]
    results: list[tuple[str, dict[str, object], onnx.ModelProto]] = []
    with tempfile.TemporaryDirectory(prefix="task172_") as tmp:
        tmp_dir = Path(tmp)
        for name, builder in builders:
            model = builder()
            verify_model(model)
            path = tmp_dir / f"{TASK_ID}_{name}.onnx"
            onnx.save(model, str(path))
            scored = score_file(path)
            if not scored["valid"]:
                raise AssertionError(f"{name} scored invalid: {scored['error']}")
            results.append((name, scored, model))
    return results


def save_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    verify_reference()
    results = benchmark_candidates()
    best_name, best_score, best_model = min(results, key=lambda item: (int(item[1]["cost"]), len(item[2].graph.node)))
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(best_model, str(path))
    final_score = score_file(path)
    print("Hypotheses:", check_hypotheses())
    for name, scored, model in results:
        print(
            f"{name}: cost={scored['cost']} memory={scored['memory']} "
            f"params={scored['params']} nodes={len(model.graph.node)} score={float(scored['score']):.6f}"
        )
    print(
        f"saved {path} using {best_name}: cost={final_score['cost']} "
        f"memory={final_score['memory']} params={final_score['params']} score={float(final_score['score']):.6f}"
    )
    return best_model


def main() -> None:
    save_model()


if __name__ == "__main__":
    main()
