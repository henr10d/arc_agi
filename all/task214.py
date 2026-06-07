"""Minimal ONNX for ARC task214: rotate a 3x3 tile across three panels.

Task rule: the input is a 3x11 grid whose first three columns contain a
colored 3x3 tile. Columns 3 and 7 are gray separators. The output preserves the
first tile, fills columns 4-6 with the tile rotated 90 degrees clockwise, and
fills columns 8-10 with the tile rotated 180 degrees. Everything outside the
3x11 task grid remains padded out of the NeuroGolf 30x30 one-hot tensor.
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

TASK_ID = "task214"
BEST_PATH = OUT_DIR / "task214.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
GH = 3
GW = 11
TILE = 3
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    tile = g[:TILE, :TILE]
    out = np.zeros((GH, GW), dtype=np.int64)
    out[:, :3] = tile
    out[:, 3] = 5
    out[:, 4:7] = np.rot90(tile, k=-1)
    out[:, 7] = 5
    out[:, 8:11] = np.rot90(tile, k=2)
    return out


def candidate_generators() -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    def distinct_cycle(g: np.ndarray) -> np.ndarray:
        out = g.copy()
        for r in range(GH):
            prefix = [int(v) for v in g[r, :3] if v not in (0, 5)]
            colors = list(dict.fromkeys(prefix))
            fill = [colors[i % len(colors)] for i in range(6)]
            out[r, 4:7] = fill[:3]
            out[r, 8:11] = fill[3:]
        return out

    def prefix_cycle(g: np.ndarray) -> np.ndarray:
        out = g.copy()
        for r in range(GH):
            colors = [int(v) for v in g[r, :3] if v not in (0, 5)]
            fill = [colors[i % len(colors)] for i in range(6)]
            out[r, 4:7] = fill[:3]
            out[r, 8:11] = fill[3:]
        return out

    def frequency_weighted(g: np.ndarray) -> np.ndarray:
        out = g.copy()
        for r in range(GH):
            prefix = [int(v) for v in g[r, :3] if v not in (0, 5)]
            order = list(dict.fromkeys(prefix))
            weighted = [color for color in order for _ in range(prefix.count(color))]
            fill = [weighted[i % len(weighted)] for i in range(6)]
            out[r, 4:7] = fill[:3]
            out[r, 8:11] = fill[3:]
        return out

    def shortest_row_motif(g: np.ndarray) -> np.ndarray:
        out = g.copy()
        for r in range(GH):
            seq = list(map(int, g[r, :3]))
            fill = [seq[i % len(seq)] for i in range(6)]
            out[r, 4:7] = fill[:3]
            out[r, 8:11] = fill[3:]
        return out

    return {
        "distinct_cycle": distinct_cycle,
        "prefix_cycle": prefix_cycle,
        "frequency_weighted": frequency_weighted,
        "shortest_row_motif": shortest_row_motif,
        "tile_rotations": solve,
    }


def validate_candidates() -> Dict[str, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    errors = {name: 0 for name in candidate_generators()}
    for ex in data["train"]:
        inp = np.asarray(ex["input"], dtype=np.int64)
        exp = np.asarray(ex["output"], dtype=np.int64)
        for name, fn in candidate_generators().items():
            if not np.array_equal(fn(inp), exp):
                errors[name] += 1
    return errors


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _grid_to_onehot(grid: List[List[int]] | np.ndarray) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    arr = np.asarray(grid, dtype=np.int64)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def build_direct_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, [0, 1, 2, 3], "axes")
    tile_st = _i64(inits, [0, 0, 0, 0], "tile_st")
    tile_en = _i64(inits, [1, C, TILE, TILE], "tile_en")
    sep_st = _i64(inits, [0, 0, 0, 3], "sep_st")
    sep_en = _i64(inits, [1, C, GH, 4], "sep_en")
    rev = _i64(inits, [2, 1, 0], "rev")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, tile_st, tile_en, axes], ["tile"]),
            helper.make_node("Slice", [IN_NAME, sep_st, sep_en, axes], ["sep"]),
            helper.make_node("Transpose", ["tile"], ["tile_t"], perm=[0, 1, 3, 2]),
            helper.make_node("Gather", ["tile_t", rev], ["rot90"], axis=3),
            helper.make_node("Gather", ["tile", rev], ["rev_rows"], axis=2),
            helper.make_node("Gather", ["rev_rows", rev], ["rot180"], axis=3),
            helper.make_node("Concat", ["tile", "sep", "rot90", "sep", "rot180"], ["out11"], axis=3),
            helper.make_node("Pad", ["out11"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - GH, W - GW]),
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


def build_argmax_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    tile_axes = _i64(inits, [2, 3], "tile_axes")
    tile_st = _i64(inits, [0, 0], "tile_st")
    tile_en = _i64(inits, [TILE, TILE], "tile_en")
    rev = _i64(inits, [2, 1, 0], "rev")
    sep = _i64(inits, np.full((1, 1, GH, 1), 5, dtype=np.int64), "sep")
    channels = _i64(inits, np.arange(C, dtype=np.int64).reshape(1, C, 1, 1), "channels")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, tile_st, tile_en, tile_axes], ["tile_oh"]),
            helper.make_node("ArgMax", ["tile_oh"], ["tile"], axis=1, keepdims=1),
            helper.make_node("Transpose", ["tile"], ["tile_t"], perm=[0, 1, 3, 2]),
            helper.make_node("Gather", ["tile_t", rev], ["rot90"], axis=3),
            helper.make_node("Gather", ["tile", rev], ["rev_rows"], axis=2),
            helper.make_node("Gather", ["rev_rows", rev], ["rot180"], axis=3),
            helper.make_node("Concat", ["tile", "sep", "rot90", "sep", "rot180"], ["idx11"], axis=3),
            helper.make_node("Equal", ["idx11", "channels"], ["out11b"]),
            helper.make_node("Cast", ["out11b"], ["out11"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out11"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - GH, W - GW]),
        ]
    )

    graph = helper.make_graph(nodes, f"{TASK_ID}_argmax", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_int32_index_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    tile_axes = _i64(inits, [2, 3], "tile_axes")
    tile_st = _i64(inits, [0, 0], "tile_st")
    tile_en = _i64(inits, [TILE, TILE], "tile_en")
    rev = _i64(inits, [2, 1, 0], "rev")
    sep = _init(inits, np.full((1, 1, GH, 1), 5, dtype=np.int32), "sep")
    channels = _init(inits, np.arange(C, dtype=np.int32).reshape(1, C, 1, 1), "channels")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, tile_st, tile_en, tile_axes], ["tile_oh"]),
            helper.make_node("ArgMax", ["tile_oh"], ["tile_i64"], axis=1, keepdims=1),
            helper.make_node("Cast", ["tile_i64"], ["tile"], to=TensorProto.INT32),
            helper.make_node("Transpose", ["tile"], ["tile_t"], perm=[0, 1, 3, 2]),
            helper.make_node("Gather", ["tile_t", rev], ["rot90"], axis=3),
            helper.make_node("Gather", ["tile", rev], ["rev_rows"], axis=2),
            helper.make_node("Gather", ["rev_rows", rev], ["rot180"], axis=3),
            helper.make_node("Concat", ["tile", "sep", "rot90", "sep", "rot180"], ["idx11"], axis=3),
            helper.make_node("Equal", ["idx11", "channels"], ["out11b"]),
            helper.make_node("Cast", ["out11b"], ["out11"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out11"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - GH, W - GW]),
        ]
    )

    graph = helper.make_graph(nodes, f"{TASK_ID}_int32_index", [x_info], [y_info], initializer=inits)
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


def validate_model(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data.get(split, []):
            inp = np.asarray(ex["input"], dtype=np.int64)
            exp = np.asarray(ex["output"], dtype=np.int64)
            if max(inp.shape) > H:
                continue
            pred_oh = _run_onnx(model, _grid_to_onehot(inp))
            pred = _onehot_to_grid(pred_oh)[: exp.shape[0], : exp.shape[1]]
            expected_oh = _grid_to_onehot(exp)
            if not np.array_equal(pred, exp) or not np.array_equal(pred_oh > 0.0, expected_oh > 0.0):
                bad += 1
    return bad


def main() -> None:
    candidate_errors = validate_candidates()
    assert candidate_errors["tile_rotations"] == 0, candidate_errors

    models = {
        "direct": build_direct_model(),
        "argmax": build_argmax_model(),
        "int32_index": build_int32_index_model(),
    }
    with tempfile.TemporaryDirectory() as tmp:
        scored = []
        for name, model in models.items():
            bad = validate_model(model)
            assert bad == 0, f"{name} model mismatches {bad} examples"
            path = Path(tmp) / f"{TASK_ID}_{name}.onnx"
            onnx.save(model, path)
            result = score_file(path)
            assert result["valid"], result["error"]
            scored.append((int(result["cost"]), name, model, result))

    scored.sort(key=lambda item: item[0])
    _, best_name, best_model, best_result = scored[0]
    onnx.save(best_model, BEST_PATH)
    final_result = score_file(BEST_PATH)
    assert final_result["valid"], final_result["error"]

    print(f"candidate train errors: {candidate_errors}")
    print(f"selected: {best_name}")
    print(
        f"score: cost={final_result['cost']} memory={final_result['memory']} "
        f"params={final_result['params']} points={final_result['score']:.6f}"
    )


if __name__ == "__main__":
    main()
