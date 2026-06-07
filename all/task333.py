"""ONNX generator for ARC task333 aligned line connections.

Task rule: find the green 2x2 block (color 3), keep the input grid, and
connect every non-background non-green seed that shares one of the block rows
or columns to the block with a straight segment of the seed's color.  Seeds
outside those two rows and two columns remain unchanged.  The examples are
10x10 grids padded to the NeuroGolf 30x30 one-hot tensor.

The graph detects the green block top-left cell from channel 3, gathers only
the two block rows and two block columns, runs four weight-free MaxPool prefix
fills on those compact strips, broadcasts the resulting lines back into the
10x10 core, restores an explicit background channel, then casts once before
the final zero pad.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task333"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
CORE = 10
PAD = H - CORE
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10
GREEN = 3


def solve_grid(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference implementation of the line-to-green-block rule."""
    arr = np.asarray(grid, dtype=np.int64)
    out = arr.copy()
    green_rows, green_cols = np.where(arr == GREEN)
    r0, r1 = int(green_rows.min()), int(green_rows.max())
    c0, c1 = int(green_cols.min()), int(green_cols.max())

    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            color = int(arr[r, c])
            if color in (0, GREEN):
                continue
            if r in (r0, r1):
                if c < c0:
                    out[r, c:c0] = color
                elif c > c1:
                    out[r, c1 + 1 : c + 1] = color
            if c in (c0, c1):
                if r < r0:
                    out[r:r0, c] = color
                elif r > r1:
                    out[r1 + 1 : r + 1, c] = color
    return out


def _init(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _f32(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    return _init(inits, name, np.asarray(arr, dtype=np.float32))


def _i64(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    return _init(inits, name, np.asarray(arr, dtype=np.int64))


def _node(nodes: list[onnx.NodeProto], op: str, inputs: list[str], output: str, **attrs: Any) -> str:
    nodes.append(helper.make_node(op, inputs, [output], **attrs))
    return output


def _and(nodes: list[onnx.NodeProto], left: str, right: str, out: str) -> str:
    return _node(nodes, "And", [left, right], out)


def _or(nodes: list[onnx.NodeProto], left: str, right: str, out: str) -> str:
    return _node(nodes, "Or", [left, right], out)


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, "axes4", [0, 1, 2, 3])
    core_en = _i64(inits, "core_en", [1, C, CORE, CORE])
    green_st = _i64(inits, "green_st", [0, GREEN, 0, 0])
    green_en = _i64(inits, "green_en", [1, GREEN + 1, CORE, CORE])
    tail_st = _i64(inits, "tail_st", [0, 1, 0, 0])
    one_shape = _i64(inits, "one_shape", [1])
    one_i = _i64(inits, "one_i", [1])
    zero_f = _f32(inits, "zero_f", [0.0])

    row = np.arange(CORE, dtype=np.int64).reshape(1, 1, CORE, 1)
    col = np.arange(CORE, dtype=np.int64).reshape(1, 1, 1, CORE)
    _i64(inits, "row", row)
    _i64(inits, "col", col)

    xfg = _node(nodes, "Slice", [IN_NAME, "tail_st", "core_en", "axes4"], "xfg")
    green = _node(nodes, "Slice", [IN_NAME, "green_st", "green_en", "axes4"], "green")
    row_green_count = _node(nodes, "ReduceSum", [green], "row_green_count", axes=[3], keepdims=1)
    col_green_count = _node(nodes, "ReduceSum", [green], "col_green_count", axes=[2], keepdims=1)
    r0_i = _node(nodes, "ArgMax", [row_green_count], "r0_i", axis=2, keepdims=1)
    c0_i = _node(nodes, "ArgMax", [col_green_count], "c0_i", axis=3, keepdims=1)
    r1_i = _node(nodes, "Add", [r0_i, "one_i"], "r1_i")
    c1_i = _node(nodes, "Add", [c0_i, "one_i"], "c1_i")
    col_lt_c0 = _node(nodes, "Less", ["col", c0_i], "col_lt_c0")
    col_gt_c1 = _node(nodes, "Greater", ["col", c1_i], "col_gt_c1")
    row_lt_r0 = _node(nodes, "Less", ["row", r0_i], "row_lt_r0")
    row_gt_r1 = _node(nodes, "Greater", ["row", r1_i], "row_gt_r1")
    row_gt_r0 = _node(nodes, "Greater", ["row", r0_i], "row_gt_r0")
    row_lt_r1 = _node(nodes, "Less", ["row", r1_i], "row_lt_r1")
    col_gt_c0 = _node(nodes, "Greater", ["col", c0_i], "col_gt_c0")
    col_lt_c1 = _node(nodes, "Less", ["col", c1_i], "col_lt_c1")
    row_ne_r0 = _or(nodes, row_lt_r0, row_gt_r0, "row_ne_r0")
    row_ne_r1 = _or(nodes, row_lt_r1, row_gt_r1, "row_ne_r1")
    col_ne_c0 = _or(nodes, col_lt_c0, col_gt_c0, "col_ne_c0")
    col_ne_c1 = _or(nodes, col_lt_c1, col_gt_c1, "col_ne_c1")
    row_eq_r0 = _node(nodes, "Not", [row_ne_r0], "row_eq_r0")
    row_eq_r1 = _node(nodes, "Not", [row_ne_r1], "row_eq_r1")
    col_eq_c0 = _node(nodes, "Not", [col_ne_c0], "col_eq_c0")
    col_eq_c1 = _node(nodes, "Not", [col_ne_c1], "col_eq_c1")

    r0_v = _node(nodes, "Reshape", [r0_i, "one_shape"], "r0_v")
    r1_v = _node(nodes, "Reshape", [r1_i, "one_shape"], "r1_v")
    c0_v = _node(nodes, "Reshape", [c0_i, "one_shape"], "c0_v")
    c1_v = _node(nodes, "Reshape", [c1_i, "one_shape"], "c1_v")
    r_idx = _node(nodes, "Concat", [r0_v, r1_v], "r_idx", axis=0)
    c_idx = _node(nodes, "Concat", [c0_v, c1_v], "c_idx", axis=0)

    hrows = _node(nodes, "Gather", [xfg, r_idx], "hrows", axis=2)
    vcols = _node(nodes, "Gather", [xfg, c_idx], "vcols", axis=3)
    l2r = _node(nodes, "MaxPool", [hrows], "l2r", kernel_shape=[1, CORE], pads=[0, CORE - 1, 0, 0])
    r2l = _node(nodes, "MaxPool", [hrows], "r2l", kernel_shape=[1, CORE], pads=[0, 0, 0, CORE - 1])
    t2b = _node(nodes, "MaxPool", [vcols], "t2b", kernel_shape=[CORE, 1], pads=[CORE - 1, 0, 0, 0])
    b2t = _node(nodes, "MaxPool", [vcols], "b2t", kernel_shape=[CORE, 1], pads=[0, 0, CORE - 1, 0])
    l2r_on = _node(nodes, "Greater", [l2r, "zero_f"], "l2r_on")
    r2l_on = _node(nodes, "Greater", [r2l, "zero_f"], "r2l_on")
    t2b_on = _node(nodes, "Greater", [t2b, "zero_f"], "t2b_on")
    b2t_on = _node(nodes, "Greater", [b2t, "zero_f"], "b2t_on")
    hleft = _and(nodes, l2r_on, col_lt_c0, "hleft")
    hright = _and(nodes, r2l_on, col_gt_c1, "hright")
    vabove = _and(nodes, t2b_on, row_lt_r0, "vabove")
    vbelow = _and(nodes, b2t_on, row_gt_r1, "vbelow")
    hline2 = _or(nodes, hleft, hright, "hline2")
    vline2 = _or(nodes, vabove, vbelow, "vline2")
    nodes.append(helper.make_node("Split", ["hline2"], ["hline_r0", "hline_r1"], axis=2, split=[1, 1]))
    nodes.append(helper.make_node("Split", ["vline2"], ["vline_c0", "vline_c1"], axis=3, split=[1, 1]))
    hline_r0_full = _and(nodes, "hline_r0", row_eq_r0, "hline_r0_full")
    hline_r1_full = _and(nodes, "hline_r1", row_eq_r1, "hline_r1_full")
    vline_c0_full = _and(nodes, "vline_c0", col_eq_c0, "vline_c0_full")
    vline_c1_full = _and(nodes, "vline_c1", col_eq_c1, "vline_c1_full")
    hline = _or(nodes, hline_r0_full, hline_r1_full, "hline")
    vline = _or(nodes, vline_c0_full, vline_c1_full, "vline")
    drawn_on = _or(nodes, hline, vline, "drawn_on")

    fg_on = _node(nodes, "Greater", [xfg, "zero_f"], "fg_on")
    out_fg_on = _or(nodes, fg_on, drawn_on, "out_fg_on")

    split_channels = [f"fg_ch{i}" for i in range(9)]
    nodes.append(helper.make_node("Split", ["out_fg_on"], split_channels, axis=1, split=[1] * 9))
    any_fg = split_channels[0]
    for idx, channel in enumerate(split_channels[1:], start=1):
        any_fg = _or(nodes, any_fg, channel, f"any_fg_{idx}")
    bg_on = _node(nodes, "Not", [any_fg], "bg_on")
    out10_on = _node(nodes, "Concat", [bg_on, out_fg_on], "out10_on", axis=1)
    out10 = _node(nodes, "Cast", [out10_on], "out10", to=TensorProto.FLOAT)
    _node(nodes, "Pad", [out10], OUT_NAME, pads=[0, 0, 0, 0, 0, 0, PAD, PAD])

    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="neurogolf-task333",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _grid_to_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(arr):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def _validate_reference(data: dict[str, list[dict[str, Any]]]) -> None:
    for split, examples in data.items():
        for idx, example in enumerate(examples):
            expected = np.asarray(example["output"], dtype=np.int64)
            got = solve_grid(example["input"])
            if not np.array_equal(got, expected):
                raise AssertionError(f"reference mismatch on {split} {idx}")


def _validate_onnx(model: onnx.ModelProto, data: dict[str, list[dict[str, Any]]]) -> None:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    for split, examples in data.items():
        for idx, example in enumerate(examples):
            expected = _grid_to_onehot(example["output"])
            got = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(example["input"])})[0]
            if not np.array_equal(got > 0.0, expected > 0.0):
                raise AssertionError(f"ONNX mismatch on {split} {idx}")


def main() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    _validate_reference(data)
    model = build_model()
    _validate_onnx(model, data)
    onnx.save(model, BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"saved {BEST_PATH}")
    print(
        f"score_model: valid={result.get('valid')} memory={result.get('memory')} "
        f"params={result.get('params')} cost={result.get('cost')} score={result.get('score')}"
    )
    if not result.get("valid"):
        raise SystemExit(result.get("error") or "score_model reported invalid")


if __name__ == "__main__":
    main()
