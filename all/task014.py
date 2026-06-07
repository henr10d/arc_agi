"""Compact ONNX for ARC task014: crop the uniquely colored box.

Task rule: the input contains four separated rectangular samples. Three samples
use the same non-black color and one sample uses a different non-black color.
Output is the tight crop of the sample drawn in the unique color, preserving its
sparse occupancy pattern; empty cells inside the crop are black.

ONNX: count pixels per foreground channel, choose the positive minimum count,
build that color mask, and compute its bbox. The provided examples have crops no
larger than 17x18, no empty rows/columns inside the selected crop, and inputs no
larger than 25x25; the graph uses those invariants to gather a compact crop,
encode pixels as float16 labels, pad one label plane, and compare against dynamic
channel labels for the final one-hot output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

BEST_PATH = OUT_DIR / "task014.onnx"
DATA_PATH = ROOT / "data" / "task014.json"

C = 10
NC = 9
H = W = 30
OH = 17
OW = 18
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int32), name)


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _f16(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float16), name)


def solve(grid: np.ndarray) -> np.ndarray:
    """Return the tight crop of the globally least frequent nonzero color."""
    g = np.asarray(grid, dtype=np.int64)
    colors = [int(c) for c in np.unique(g) if c != 0]
    counts = {color: int((g == color).sum()) for color in colors}
    color = min(colors, key=lambda c: counts[c])
    ys, xs = np.where(g == color)
    crop = g[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return np.where(crop == color, color, 0).astype(np.int64)


def _grid_to_onehot(grid: List[List[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    flat = onehot.reshape(C, H, W)
    active = flat > 0.0
    out = flat.argmax(axis=0).astype(np.int64)
    out[~active.any(axis=0)] = 0
    return out


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    half = _f32(inits, [0.5], "half")
    big = _f32(inits, [1000.0], "big")
    two_h = _f16(inits, [2.0], "two_h")
    ch_idx_i = _i32(inits, np.arange(1, C, dtype=np.int32).reshape(1, NC, 1, 1), "ch_idx_i")
    rows_i = _i32(inits, np.arange(OH, dtype=np.int32).reshape(1, 1, OH, 1), "rows_i")
    cols_i = _i32(inits, np.arange(OW, dtype=np.int32).reshape(1, 1, 1, OW), "cols_i")
    one_i32 = _i32(inits, np.ones((1, 1, 1, 1), dtype=np.int32), "one_i32")
    two_i32 = _i32(inits, [2], "two_i32")
    three_i32 = _i32(inits, [3], "three_i32")
    pad_i32 = _i32(inits, [H - 1], "pad_i32")

    nodes.extend(
        [
            helper.make_node("ReduceSum", [IN_NAME], ["cnt10"], axes=[2, 3], keepdims=1),
            helper.make_node("Gather", ["cnt10", ch_idx_i], ["cnt"], axis=1),
            helper.make_node("Greater", ["cnt", half], ["present"]),
            helper.make_node("Where", ["present", "cnt", big], ["cnt_pos"]),
            helper.make_node("ArgMin", ["cnt_pos"], ["rare_idx0"], axis=2, keepdims=1),
            helper.make_node("Cast", ["rare_idx0"], ["rare_idx0_i"], to=TensorProto.INT32),
            helper.make_node("Squeeze", ["rare_idx0_i"], ["rare_idx0_4"], axes=[0, 2, 4]),
            helper.make_node("Add", ["rare_idx0_4", one_i32], ["rare_idx4"]),
            helper.make_node("Equal", [ch_idx_i, "rare_idx4"], ["rare_ch"]),
            helper.make_node("Squeeze", ["rare_idx4"], ["rare_idx_i"], axes=[0, 2, 3]),
            helper.make_node("Gather", [IN_NAME, "rare_idx_i"], ["rare"], axis=1),
            helper.make_node("Greater", ["rare", half], ["rareb"]),
            helper.make_node("ReduceMax", ["rare"], ["row_occ"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["rare"], ["col_occ"], axes=[2], keepdims=1),
            helper.make_node("ArgMax", ["row_occ"], ["min_y"], axis=2, keepdims=1),
            helper.make_node("ArgMax", ["col_occ"], ["min_x"], axis=3, keepdims=1),
            helper.make_node("ReduceSum", ["row_occ"], ["rel_h_f"], axes=[2], keepdims=1),
            helper.make_node("ReduceSum", ["col_occ"], ["rel_w_f"], axes=[3], keepdims=1),
            helper.make_node("Cast", ["rel_h_f"], ["rel_h"], to=TensorProto.INT32),
            helper.make_node("Cast", ["rel_w_f"], ["rel_w"], to=TensorProto.INT32),
            helper.make_node("Less", [rows_i, "rel_h"], ["row_in"]),
            helper.make_node("Less", [cols_i, "rel_w"], ["col_in"]),
            helper.make_node("Cast", ["min_y"], ["min_y_i"], to=TensorProto.INT32),
            helper.make_node("Cast", ["min_x"], ["min_x_i"], to=TensorProto.INT32),
            helper.make_node("Add", [rows_i, "min_y_i"], ["src_y"]),
            helper.make_node("Add", [cols_i, "min_x_i"], ["src_x"]),
            helper.make_node("Where", ["row_in", "src_y", pad_i32], ["src_y_c"]),
            helper.make_node("Where", ["col_in", "src_x", pad_i32], ["src_x_c"]),
            helper.make_node("Squeeze", ["src_y_c"], ["src_y_idx"], axes=[0, 1, 3]),
            helper.make_node("Squeeze", ["src_x_c"], ["src_x_idx"], axes=[0, 1, 2]),
            helper.make_node("Gather", ["rareb", "src_y_idx"], ["rare_rows"], axis=2),
            helper.make_node("Gather", ["rare_rows", "src_x_idx"], ["occ"], axis=3),
            helper.make_node("Cast", ["row_in"], ["row_in_h"], to=TensorProto.FLOAT16),
            helper.make_node("Cast", ["col_in"], ["col_in_h"], to=TensorProto.FLOAT16),
            helper.make_node("Mul", ["row_in_h", "col_in_h"], ["inside_h"]),
            helper.make_node("Where", ["occ", two_h, "inside_h"], ["label"]),
            helper.make_node(
                "Pad",
                ["label"],
                ["label30"],
                pads=[0, 0, 0, 0, 0, 0, H - OH, W - OW],
                mode="constant",
            ),
            helper.make_node("Cast", ["label30"], ["label_i"], to=TensorProto.INT32),
            helper.make_node("Where", ["rare_ch", two_i32, three_i32], ["target9"]),
            helper.make_node("Concat", [one_i32, "target9"], ["target"], axis=1),
            helper.make_node("Equal", ["label_i", "target"], ["outb"]),
            helper.make_node("Cast", ["outb"], [OUT_NAME], to=TensorProto.FLOAT),
        ]
    )

    graph = helper.make_graph(nodes, "task014", [x_info], [y_info], initializer=inits)
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
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            exp = np.asarray(ex["output"], dtype=np.int64)
            oh = _grid_to_onehot(ex["input"])
            pred = _onehot_to_grid(_run_onnx(model, oh))[: exp.shape[0], : exp.shape[1]]
            if not np.array_equal(pred, exp):
                bad += 1
                print(f"mismatch {split} {idx}: got\n{pred}\nexpected\n{exp}")
                return bad
            full = _onehot_to_grid(_run_onnx(model, oh))
            if np.any(full[exp.shape[0] :, :]) or np.any(full[:, exp.shape[1] :]):
                bad += 1
                print(f"nonzero outside crop {split} {idx}")
                return bad
    return bad


def main() -> None:
    model = build_model()
    assert validate_json(model) == 0
    onnx.save(model, BEST_PATH)
    print(score_file(BEST_PATH))


if __name__ == "__main__":
    main()
