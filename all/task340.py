"""ONNX generator for ARC task340: move frame-colored markers to the frame.

Task rule: the input has a rectangular colored frame with black corners and
scattered interior colored cells.  Preserve the frame exactly.  Interior cells
whose color matches the top, bottom, left, or right frame color are not kept in
place; instead they are copied to the matching inner frame line at the same
column or row: top markers go to row 1, bottom markers to row h-2, left markers
to column 1, and right markers to column w-2.  Non-frame-colored interior cells
disappear, and all other interior cells become background.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task340"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task340.onnx"
DATA_PATH = ROOT / "data" / "task340.json"

C = 10
NC = C - 1
H = W = 30
SH = SW = 20
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference implementation for validating the ONNX graph."""
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    out = np.zeros_like(g)
    out[0, :] = g[0, :]
    out[h - 1, :] = g[h - 1, :]
    out[:, 0] = g[:, 0]
    out[:, w - 1] = g[:, w - 1]

    top, bottom, left, right = g[0, 1], g[h - 1, 1], g[1, 0], g[1, w - 1]
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            color = g[r, c]
            if color == top:
                out[1, c] = color
            elif color == bottom:
                out[h - 2, c] = color
            elif color == left:
                out[r, 1] = color
            elif color == right:
                out[r, w - 2] = color
    return out


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _i32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int32), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _f16(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float16), name)


def _bool(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.bool_), name)


def _node(nodes: List[onnx.NodeProto], op: str, inputs: list[str], output: str, **attrs: Any) -> str:
    nodes.append(helper.make_node(op, inputs, [output], **attrs))
    return output


def _and(nodes: List[onnx.NodeProto], left: str, right: str, out: str) -> str:
    return _node(nodes, "And", [left, right], out)


def _or(nodes: List[onnx.NodeProto], left: str, right: str, out: str) -> str:
    return _node(nodes, "Or", [left, right], out)


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    axis_row = _i64(inits, [2], "axis_row")
    axis_col = _i64(inits, [3], "axis_col")
    st_crop = _i64(inits, [0, 1, 0, 0], "st_crop")
    en_crop = _i64(inits, [1, C, SH, SW], "en_crop")
    st_top = _i64(inits, [0, 0, 0, 1], "st_top")
    en_top = _i64(inits, [1, NC, 1, 2], "en_top")
    st_left = _i64(inits, [0, 0, 1, 0], "st_left")
    en_left = _i64(inits, [1, NC, 2, 1], "en_left")
    st_col1 = _i64(inits, [0, 0, 0, 1], "st_col1")
    en_col1 = _i64(inits, [1, NC, SH, 2], "en_col1")
    st_row1 = _i64(inits, [0, 0, 1, 0], "st_row1")
    en_row1 = _i64(inits, [1, NC, 2, SW], "en_row1")
    s0 = _i64(inits, [0], "s0")
    s1 = _i64(inits, [1], "s1")
    e19 = _i64(inits, [SH - 1], "e19")
    e20 = _i64(inits, [SH], "e20")
    vec_shape = _i64(inits, [1, NC, 1, 1], "vec_shape")
    half = _f16(inits, [0.5], "half")
    zero_f16 = _f16(inits, [0.0], "zero_f16")
    color_ids9 = _f16(inits, np.arange(1, C, dtype=np.float16).reshape(1, NC, 1, 1), "color_ids9")
    color_ids10_i = _i32(inits, np.arange(C, dtype=np.int32).reshape(1, C, 1, 1), "color_ids10_i")
    zero_i32 = _i32(inits, [0], "zero_i32")
    neg_one_i32 = _i32(inits, [-1], "neg_one_i32")
    row_ids = _f16(inits, np.arange(SH, dtype=np.float16).reshape(1, 1, SH, 1), "row_ids")
    col_ids = _f16(inits, np.arange(SW, dtype=np.float16).reshape(1, 1, 1, SW), "col_ids")
    false_row = _bool(inits, np.zeros((1, 1, 1, 1), dtype=np.bool_), "false_row")

    _node(nodes, "Slice", [IN_NAME, st_crop, en_crop, axes4], "xfg")
    _node(nodes, "Cast", ["xfg"], "xh", to=TensorProto.FLOAT16)
    _node(nodes, "ReduceMax", ["xh"], "fg_any_f", axes=[1], keepdims=1)
    _node(nodes, "Greater", ["fg_any_f", half], "fg_any")
    _node(nodes, "ReduceMax", ["fg_any_f"], "row_f", axes=[3], keepdims=1)
    _node(nodes, "ReduceMax", ["fg_any_f"], "col_f", axes=[2], keepdims=1)
    _node(nodes, "Greater", ["row_f", half], "row_occ")
    _node(nodes, "Greater", ["col_f", half], "col_occ")
    _and(nodes, "row_occ", "col_occ", "valid")
    _node(nodes, "Mul", ["row_f", row_ids], "weighted_rows")
    _node(nodes, "Mul", ["col_f", col_ids], "weighted_cols")
    _node(nodes, "ArgMax", ["weighted_rows"], "bottom_idx", axis=2, keepdims=0)
    _node(nodes, "ArgMax", ["weighted_cols"], "right_idx", axis=3, keepdims=0)

    _node(nodes, "Slice", ["row_occ", s0, e19, axis_row], "row_prev_src")
    _node(nodes, "Concat", [false_row, "row_prev_src"], "row_prev", axis=2)
    _node(nodes, "Not", ["row_prev"], "not_row_prev")
    _and(nodes, "row_occ", "not_row_prev", "top_row")

    _node(nodes, "Slice", ["top_row", s0, e19, axis_row], "top_shift_src")
    _node(nodes, "Concat", [false_row, "top_shift_src"], "row1", axis=2)

    _node(nodes, "Slice", ["row_occ", s1, e20, axis_row], "row_next_src")
    _node(nodes, "Concat", ["row_next_src", false_row], "row_next", axis=2)
    _node(nodes, "Not", ["row_next"], "not_row_next")
    _and(nodes, "row_occ", "not_row_next", "bottom_row")

    _node(nodes, "Slice", ["bottom_row", s1, e20, axis_row], "bottom_shift_src")
    _node(nodes, "Concat", ["bottom_shift_src", false_row], "bottom_inner", axis=2)

    _node(nodes, "Slice", ["col_occ", s0, e19, axis_col], "col_prev_src")
    _node(nodes, "Concat", [false_row, "col_prev_src"], "col_prev", axis=3)
    _node(nodes, "Not", ["col_prev"], "not_col_prev")
    _and(nodes, "col_occ", "not_col_prev", "left_col")

    _node(nodes, "Slice", ["left_col", s0, e19, axis_col], "left_shift_src")
    _node(nodes, "Concat", [false_row, "left_shift_src"], "col1", axis=3)

    _node(nodes, "Slice", ["col_occ", s1, e20, axis_col], "col_next_src")
    _node(nodes, "Concat", ["col_next_src", false_row], "col_next", axis=3)
    _node(nodes, "Not", ["col_next"], "not_col_next")
    _and(nodes, "col_occ", "not_col_next", "right_col")

    _node(nodes, "Slice", ["right_col", s1, e20, axis_col], "right_shift_src")
    _node(nodes, "Concat", ["right_shift_src", false_row], "right_inner", axis=3)

    row_border = _or(nodes, "top_row", "bottom_row", "row_border")
    col_border = _or(nodes, "left_col", "right_col", "col_border")
    border_line = _or(nodes, row_border, col_border, "border_line")
    _and(nodes, "valid", "border_line", "border")
    _node(nodes, "Not", ["border_line"], "not_border_line")
    _and(nodes, "valid", "not_border_line", "interior")
    _node(nodes, "Where", ["interior", "xh", zero_f16], "interior_colors")
    _node(nodes, "ReduceMax", ["interior_colors"], "interior_cols", axes=[2], keepdims=1)
    _node(nodes, "ReduceMax", ["interior_colors"], "interior_rows", axes=[3], keepdims=1)

    _node(nodes, "Slice", ["xh", st_top, en_top, axes4], "top_vec")
    _node(nodes, "Slice", ["xh", st_left, en_left, axes4], "left_vec")

    _node(nodes, "Slice", ["xh", st_col1, en_col1, axes4], "x_col1")
    _node(nodes, "Gather", ["x_col1", "bottom_idx"], "bottom_gather", axis=2)
    _node(nodes, "Reshape", ["bottom_gather", vec_shape], "bottom_vec")
    _node(nodes, "Slice", ["xh", st_row1, en_row1, axes4], "x_row1")
    _node(nodes, "Gather", ["x_row1", "right_idx"], "right_gather", axis=3)
    _node(nodes, "Reshape", ["right_gather", vec_shape], "right_vec")

    _node(nodes, "Mul", ["interior_cols", "top_vec"], "top_reduced")
    _node(nodes, "Mul", ["interior_cols", "bottom_vec"], "bottom_reduced")
    _node(nodes, "Mul", ["interior_rows", "left_vec"], "left_reduced")
    _node(nodes, "Mul", ["interior_rows", "right_vec"], "right_reduced")

    _node(nodes, "ReduceMax", ["top_reduced"], "top_area_cols_f", axes=[1], keepdims=1)
    _node(nodes, "ReduceMax", ["bottom_reduced"], "bottom_area_cols_f", axes=[1], keepdims=1)
    _node(nodes, "ReduceMax", ["left_reduced"], "left_area_rows_f", axes=[1], keepdims=1)
    _node(nodes, "ReduceMax", ["right_reduced"], "right_area_rows_f", axes=[1], keepdims=1)
    _node(nodes, "Greater", ["top_area_cols_f", half], "top_area_cols")
    _node(nodes, "Greater", ["bottom_area_cols_f", half], "bottom_area_cols")
    _node(nodes, "Greater", ["left_area_rows_f", half], "left_area_rows")
    _node(nodes, "Greater", ["right_area_rows_f", half], "right_area_rows")
    top_area = _and(nodes, "top_area_cols", "row1", "top_area")
    bottom_area = _and(nodes, "bottom_area_cols", "bottom_inner", "bottom_area")
    left_area = _and(nodes, "left_area_rows", "col1", "left_area")
    right_area = _and(nodes, "right_area_rows", "right_inner", "right_area")

    _node(nodes, "Not", ["col_border"], "not_col_border")
    _node(nodes, "Not", ["row_border"], "not_row_border")
    _and(nodes, "top_row", "not_col_border", "top_seg0")
    _and(nodes, "bottom_row", "not_col_border", "bottom_seg0")
    _and(nodes, "left_col", "not_row_border", "left_seg0")
    _and(nodes, "right_col", "not_row_border", "right_seg0")
    top_seg = _and(nodes, "valid", "top_seg0", "top_seg")
    bottom_seg = _and(nodes, "valid", "bottom_seg0", "bottom_seg")
    left_seg = _and(nodes, "valid", "left_seg0", "left_seg")
    right_seg = _and(nodes, "valid", "right_seg0", "right_seg")

    top_mask = _or(nodes, top_seg, top_area, "top_mask")
    bottom_mask = _or(nodes, bottom_seg, bottom_area, "bottom_mask")
    left_mask = _or(nodes, left_seg, left_area, "left_mask")
    right_mask = _or(nodes, right_seg, right_area, "right_mask")

    _node(nodes, "Mul", ["top_vec", color_ids9], "top_color_parts")
    _node(nodes, "Mul", ["bottom_vec", color_ids9], "bottom_color_parts")
    _node(nodes, "Mul", ["left_vec", color_ids9], "left_color_parts")
    _node(nodes, "Mul", ["right_vec", color_ids9], "right_color_parts")
    _node(nodes, "ReduceMax", ["top_color_parts"], "top_color", axes=[1], keepdims=1)
    _node(nodes, "ReduceMax", ["bottom_color_parts"], "bottom_color", axes=[1], keepdims=1)
    _node(nodes, "ReduceMax", ["left_color_parts"], "left_color", axes=[1], keepdims=1)
    _node(nodes, "ReduceMax", ["right_color_parts"], "right_color", axes=[1], keepdims=1)

    _node(nodes, "Cast", ["top_color"], "top_color_i", to=TensorProto.INT32)
    _node(nodes, "Cast", ["bottom_color"], "bottom_color_i", to=TensorProto.INT32)
    _node(nodes, "Cast", ["left_color"], "left_color_i", to=TensorProto.INT32)
    _node(nodes, "Cast", ["right_color"], "right_color_i", to=TensorProto.INT32)
    _node(nodes, "Not", ["valid"], "invalid")
    _node(nodes, "Where", ["invalid", neg_one_i32, zero_i32], "base_grid")
    _node(nodes, "Where", [top_mask, "top_color_i", "base_grid"], "top_grid")
    _node(nodes, "Where", [bottom_mask, "bottom_color_i", "top_grid"], "bottom_grid")
    _node(nodes, "Where", [left_mask, "left_color_i", "bottom_grid"], "left_grid")
    _node(nodes, "Where", [right_mask, "right_color_i", "left_grid"], "color_grid")
    _node(nodes, "Equal", ["color_grid", color_ids10_i], "out_bool")
    _node(nodes, "Cast", ["out_bool"], "out20", to=TensorProto.FLOAT)
    _node(nodes, "Pad", ["out20"], OUT_NAME, pads=[0, 0, 0, 0, 0, 0, H - SH, W - SW])

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


def _run(model_path: Path, arr: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: arr})[0]


def verify_model(model_path: Path) -> tuple[bool, str]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    count = 0
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            inp = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if inp is None or expected is None:
                continue
            actual = _run(model_path, inp)
            if not np.array_equal(actual > 0.0, expected > 0.0):
                return False, f"{split} example {idx} mismatch"
            ref = solve(example["input"])
            if not np.array_equal(ref, np.asarray(example["output"], dtype=np.int64)):
                return False, f"reference solver mismatch on {split} example {idx}"
            count += 1
    return True, f"{count} examples matched"


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    ok, note = verify_model(BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote:   {BEST_PATH}")
    print(f"verify:  {'PASS' if ok else 'FAIL'} ({note})")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"nodes:   {len(model.graph.node)}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    print(f"score:   {result['score']}")

    if not ok or not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
