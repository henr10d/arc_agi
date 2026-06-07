"""ONNX for ARC task290: crop a centered two-color square and swap its colors.

Task rule: the input contains one non-black solid square, embedded in black
padding.  The square has an outer color and a centered 1x1 or 2x2 inner square
of a second color.  The output is only the tight square crop, moved to the
top-left of the NeuroGolf output tensor, with the two non-black colors swapped.
All cells outside the compact crop remain inactive padding.

ONNX approach: find the non-black bounding box from the one-hot tensor, infer
the centered inner color and the remaining outer color, then render the swapped
square at the origin.  The graph keeps most geometry as 1-channel bool masks and
casts only the final one-hot result to float.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, List, Sequence

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task290"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task290.onnx"
DATA_PATH = ROOT / "data" / "task290.json"

C = 10
H = W = 30
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10
HALF = 0.5

Example = tuple[str, int, np.ndarray, np.ndarray]


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _bool(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.bool_), name)


def _make_model(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], name: str) -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(nodes, name, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _sub(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], a: str, b: str | float, out: str) -> str:
    if not isinstance(b, str):
        b = _f32(inits, [b], f"{out}_c")
    nodes.append(helper.make_node("Sub", [a, b], [out]))
    return out


def _add(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], a: str, b: str | float, out: str) -> str:
    if not isinstance(b, str):
        b = _f32(inits, [b], f"{out}_c")
    nodes.append(helper.make_node("Add", [a, b], [out]))
    return out


def _mul(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], a: str, b: str | float, out: str) -> str:
    if not isinstance(b, str):
        b = _f32(inits, [b], f"{out}_c")
    nodes.append(helper.make_node("Mul", [a, b], [out]))
    return out


def _less(nodes: List[onnx.NodeProto], a: str, b: str, out: str) -> str:
    nodes.append(helper.make_node("Less", [a, b], [out]))
    return out


def _not(nodes: List[onnx.NodeProto], a: str, out: str) -> str:
    nodes.append(helper.make_node("Not", [a], [out]))
    return out


def _and(nodes: List[onnx.NodeProto], a: str, b: str, out: str) -> str:
    nodes.append(helper.make_node("And", [a, b], [out]))
    return out


def _or(nodes: List[onnx.NodeProto], a: str, b: str, out: str) -> str:
    nodes.append(helper.make_node("Or", [a, b], [out]))
    return out


def _ge(nodes: List[onnx.NodeProto], a: str, b: str, out: str) -> str:
    return _not(nodes, _less(nodes, a, b, f"{out}_lt"), out)


def _le(nodes: List[onnx.NodeProto], a: str, b: str, out: str) -> str:
    return _not(nodes, _less(nodes, b, a, f"{out}_gt"), out)


def _eq(nodes: List[onnx.NodeProto], a: str, b: str, out: str) -> str:
    ge = _ge(nodes, a, b, f"{out}_ge")
    le = _le(nodes, a, b, f"{out}_le")
    return _and(nodes, ge, le, out)


def load_examples() -> list[Example]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    examples: list[Example] = []
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            examples.append(
                (
                    split,
                    idx,
                    np.asarray(ex["input"], dtype=np.int64),
                    np.asarray(ex["output"], dtype=np.int64),
                )
            )
    return examples


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference implementation using the task rule from the JSON examples."""
    g = np.asarray(grid, dtype=np.int64)
    pts = np.argwhere(g != 0)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0) + 1
    crop = g[r0:r1, c0:c1]

    values = [int(v) for v in np.unique(crop) if v != 0]
    border_counts = {
        color: int(np.sum(crop[0, :] == color) + np.sum(crop[-1, :] == color) + np.sum(crop[:, 0] == color) + np.sum(crop[:, -1] == color))
        for color in values
    }
    inner_candidates = [color for color in values if border_counts[color] == 0]
    inner = inner_candidates[0] if len(inner_candidates) == 1 else min(values, key=lambda color: border_counts[color])
    outer = next(color for color in values if color != inner)

    out = np.full(crop.shape, inner, dtype=np.int64)
    out[crop == inner] = outer
    return out


def _grid_to_onehot(grid: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _expected_onehot(grid: np.ndarray) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            out[0, int(grid[r, c]), r, c] = 1.0
    return out


def _run_onnx(model: onnx.ModelProto, grid: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: _grid_to_onehot(grid)})[0]


def _common_geometry(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto]) -> dict[str, str]:
    rows = _f32(inits, np.arange(H, dtype=np.float32).reshape(1, 1, H, 1), "rows")
    cols = _f32(inits, np.arange(W, dtype=np.float32).reshape(1, 1, 1, W), "cols")
    big = _f32(inits, [100.0], "big")
    neg = _f32(inits, [-1.0], "neg")
    half = _f32(inits, [HALF], "half")
    fg_flag = _bool(inits, (np.arange(C).reshape(1, C, 1, 1) > 0), "fg_flag")
    bg_starts = _i64(inits, [0, 0, 0, 0], "bg_starts")
    bg_ends = _i64(inits, [1, 1, H, W], "bg_ends")
    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")

    nodes.extend(
        [
            helper.make_node("ReduceSum", [IN_NAME], ["active_f"], axes=[1], keepdims=1),
            helper.make_node("Slice", [IN_NAME, bg_starts, bg_ends, axes4], ["bg_f"]),
            helper.make_node("Sub", ["active_f", "bg_f"], ["nonzero_f"]),
            helper.make_node("Greater", ["nonzero_f", half], ["nonzero_b"]),
            helper.make_node("Where", ["nonzero_b", rows, big], ["rows_for_min"]),
            helper.make_node("Where", ["nonzero_b", cols, big], ["cols_for_min"]),
            helper.make_node("Where", ["nonzero_b", rows, neg], ["rows_for_max"]),
            helper.make_node("Where", ["nonzero_b", cols, neg], ["cols_for_max"]),
            helper.make_node("ReduceMin", ["rows_for_min"], ["rmin"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMin", ["cols_for_min"], ["cmin"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMax", ["rows_for_max"], ["rmax"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMax", ["cols_for_max"], ["cmax"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMax", [IN_NAME], ["present_f"], axes=[2, 3], keepdims=1),
            helper.make_node("Greater", ["present_f", half], ["present_b0"]),
            helper.make_node("And", ["present_b0", fg_flag], ["present_b"]),
        ]
    )

    size = _add(nodes, inits, _sub(nodes, inits, "rmax", "rmin", "height_m1"), 1.0, "size")
    low = _sub(nodes, inits, size, 2.0, "center_low")
    two_rows = _mul(nodes, inits, rows, 2.0, "two_rows")
    two_cols = _mul(nodes, inits, cols, 2.0, "two_cols")
    row_active = _less(nodes, rows, size, "row_active")
    col_active = _less(nodes, cols, size, "col_active")
    active_square = _and(nodes, row_active, col_active, "active_square")
    row_inner = _and(nodes, _ge(nodes, two_rows, low, "row_inner_ge"), _le(nodes, two_rows, size, "row_inner_le"), "row_inner")
    col_inner = _and(nodes, _ge(nodes, two_cols, low, "col_inner_ge"), _le(nodes, two_cols, size, "col_inner_le"), "col_inner")
    inner_at_origin = _and(nodes, row_inner, col_inner, "inner_at_origin")
    not_inner_at_origin = _not(nodes, inner_at_origin, "not_inner_at_origin")
    active_non_inner = _and(nodes, active_square, not_inner_at_origin, "active_non_inner")

    return {
        "rows": rows,
        "cols": cols,
        "half": half,
        "rmin": "rmin",
        "cmin": "cmin",
        "rmax": "rmax",
        "cmax": "cmax",
        "present_b": "present_b",
        "nonzero_b": "nonzero_b",
        "active_non_inner": active_non_inner,
        "inner_at_origin": inner_at_origin,
    }


def _finish_render(nodes: List[onnx.NodeProto], inner_color: str, outer_color: str, geom: dict[str, str]) -> None:
    nodes.extend(
        [
            helper.make_node("And", [inner_color, geom["active_non_inner"]], ["inner_fill"]),
            helper.make_node("And", [outer_color, geom["inner_at_origin"]], ["outer_fill"]),
            helper.make_node("Or", ["inner_fill", "outer_fill"], ["out_b"]),
            helper.make_node("Cast", ["out_b"], [OUT_NAME], to=TensorProto.FLOAT),
        ]
    )


def build_center_color_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    geom = _common_geometry(nodes, inits)

    sum_minmax_r = _add(nodes, inits, geom["rmin"], geom["rmax"], "sum_minmax_r")
    sum_minmax_c = _add(nodes, inits, geom["cmin"], geom["cmax"], "sum_minmax_c")
    low_r = _sub(nodes, inits, sum_minmax_r, 1.0, "orig_center_low_r")
    low_c = _sub(nodes, inits, sum_minmax_c, 1.0, "orig_center_low_c")
    high_r = _add(nodes, inits, sum_minmax_r, 1.0, "orig_center_high_r")
    high_c = _add(nodes, inits, sum_minmax_c, 1.0, "orig_center_high_c")
    two_rows = _mul(nodes, inits, geom["rows"], 2.0, "orig_two_rows")
    two_cols = _mul(nodes, inits, geom["cols"], 2.0, "orig_two_cols")

    center_rows = _and(nodes, _ge(nodes, two_rows, low_r, "orig_center_rows_ge"), _le(nodes, two_rows, high_r, "orig_center_rows_le"), "orig_center_rows")
    center_cols = _and(nodes, _ge(nodes, two_cols, low_c, "orig_center_cols_ge"), _le(nodes, two_cols, high_c, "orig_center_cols_le"), "orig_center_cols")
    center_mask = _and(nodes, _and(nodes, center_rows, center_cols, "orig_center_square"), geom["nonzero_b"], "orig_center_mask")

    nodes.extend(
        [
            helper.make_node("Cast", [center_mask], ["orig_center_mask_f"], to=TensorProto.FLOAT),
            helper.make_node("Mul", [IN_NAME, "orig_center_mask_f"], ["center_cells"]),
            helper.make_node("ReduceMax", ["center_cells"], ["inner_color_f"], axes=[2, 3], keepdims=1),
            helper.make_node("Greater", ["inner_color_f", geom["half"]], ["inner_color"]),
            helper.make_node("Not", ["inner_color"], ["not_inner_color"]),
            helper.make_node("And", [geom["present_b"], "not_inner_color"], ["outer_color"]),
        ]
    )
    _finish_render(nodes, "inner_color", "outer_color", geom)
    return _make_model(nodes, inits, "task290_center_color")


def build_border_color_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    geom = _common_geometry(nodes, inits)

    r_is_min = _eq(nodes, geom["rows"], geom["rmin"], "r_is_min")
    r_is_max = _eq(nodes, geom["rows"], geom["rmax"], "r_is_max")
    c_is_min = _eq(nodes, geom["cols"], geom["cmin"], "c_is_min")
    c_is_max = _eq(nodes, geom["cols"], geom["cmax"], "c_is_max")
    row_border = _or(nodes, r_is_min, r_is_max, "row_border")
    col_border = _or(nodes, c_is_min, c_is_max, "col_border")
    border_mask = _and(nodes, _or(nodes, row_border, col_border, "bbox_border"), geom["nonzero_b"], "border_mask")

    nodes.extend(
        [
            helper.make_node("Cast", [border_mask], ["border_mask_f"], to=TensorProto.FLOAT),
            helper.make_node("Mul", [IN_NAME, "border_mask_f"], ["border_cells"]),
            helper.make_node("ReduceMax", ["border_cells"], ["outer_color_f"], axes=[2, 3], keepdims=1),
            helper.make_node("Greater", ["outer_color_f", geom["half"]], ["outer_color"]),
            helper.make_node("Not", ["outer_color"], ["not_outer_color"]),
            helper.make_node("And", [geom["present_b"], "not_outer_color"], ["inner_color"]),
        ]
    )
    _finish_render(nodes, "inner_color", "outer_color", geom)
    return _make_model(nodes, inits, "task290_border_color")


def build_area_count_onehot_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    rows = _f32(inits, np.arange(H, dtype=np.float32).reshape(1, 1, H, 1), "rows")
    cols = _f32(inits, np.arange(W, dtype=np.float32).reshape(1, 1, 1, W), "cols")
    half = _f32(inits, [HALF], "half")
    max_inner = _f32(inits, [4.5], "max_inner")
    area4 = _f32(inits, [3.0], "size_base")
    t12 = _f32(inits, [12.5], "area_gt_3")
    t20 = _f32(inits, [20.5], "area_gt_4")
    t30 = _f32(inits, [30.5], "area_gt_5")
    fg_flag = _bool(inits, (np.arange(C).reshape(1, C, 1, 1) > 0), "fg_flag")
    fg_starts = _i64(inits, [1], "fg_starts")
    fg_ends = _i64(inits, [C], "fg_ends")
    axis_channel = _i64(inits, [1], "axis_channel")
    depth = _i64(inits, np.array(10, dtype=np.int64), "depth")
    values = _f32(inits, [0.0, 1.0], "onehot_values")
    off_index = _i64(inits, np.array(C, dtype=np.int64), "off_index")

    nodes.extend(
        [
            helper.make_node("ReduceSum", [IN_NAME], ["color_counts"], axes=[2, 3], keepdims=1),
            helper.make_node("Slice", ["color_counts", fg_starts, fg_ends, axis_channel], ["fg_counts"]),
            helper.make_node("ReduceSum", ["fg_counts"], ["area"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["color_counts", half], ["present_b0"]),
            helper.make_node("Less", ["color_counts", max_inner], ["small_count"]),
            helper.make_node("And", ["present_b0", "small_count"], ["small_present"]),
            helper.make_node("And", ["small_present", fg_flag], ["inner_color"]),
            helper.make_node("Greater", ["color_counts", max_inner], ["large_count"]),
            helper.make_node("And", ["large_count", fg_flag], ["outer_color"]),
            helper.make_node("Greater", ["area", t12], ["area_gt12"]),
            helper.make_node("Greater", ["area", t20], ["area_gt20"]),
            helper.make_node("Greater", ["area", t30], ["area_gt30"]),
            helper.make_node("Cast", ["area_gt12"], ["area_gt12_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["area_gt20"], ["area_gt20_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["area_gt30"], ["area_gt30_f"], to=TensorProto.FLOAT),
            helper.make_node("Add", [area4, "area_gt12_f"], ["size4"]),
            helper.make_node("Add", ["size4", "area_gt20_f"], ["size5"]),
            helper.make_node("Add", ["size5", "area_gt30_f"], ["size"]),
        ]
    )

    low = _sub(nodes, inits, "size", 2.0, "center_low")
    two_rows = _mul(nodes, inits, rows, 2.0, "two_rows")
    two_cols = _mul(nodes, inits, cols, 2.0, "two_cols")
    row_active = _less(nodes, rows, "size", "row_active")
    col_active = _less(nodes, cols, "size", "col_active")
    active_square = _and(nodes, row_active, col_active, "active_square")
    row_inner = _and(nodes, _ge(nodes, two_rows, low, "row_inner_ge"), _le(nodes, two_rows, "size", "row_inner_le"), "row_inner")
    col_inner = _and(nodes, _ge(nodes, two_cols, low, "col_inner_ge"), _le(nodes, two_cols, "size", "col_inner_le"), "col_inner")
    inner_at_origin = _and(nodes, row_inner, col_inner, "inner_at_origin")

    nodes.extend(
        [
            helper.make_node("Cast", ["inner_color"], ["inner_color_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["outer_color"], ["outer_color_f"], to=TensorProto.FLOAT),
            helper.make_node("ArgMax", ["inner_color_f"], ["inner_idx"], axis=1, keepdims=0),
            helper.make_node("ArgMax", ["outer_color_f"], ["outer_idx"], axis=1, keepdims=0),
            helper.make_node("Squeeze", [inner_at_origin], ["inner_mask"], axes=[1]),
            helper.make_node("Squeeze", [active_square], ["active_mask"], axes=[1]),
            helper.make_node("Where", ["inner_mask", "outer_idx", "inner_idx"], ["color_grid"]),
            helper.make_node("Where", ["active_mask", "color_grid", off_index], ["index_grid"]),
            helper.make_node("OneHot", ["index_grid", depth, values], [OUT_NAME], axis=1),
        ]
    )
    return _make_model(nodes, inits, "task290_area_count_onehot")


def build_area_count_compact_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    rows = _f32(inits, np.arange(6, dtype=np.float32).reshape(1, 1, 6, 1), "rows6")
    cols = _f32(inits, np.arange(6, dtype=np.float32).reshape(1, 1, 1, 6), "cols6")
    half = _f32(inits, [HALF], "half")
    max_inner = _f32(inits, [4.5], "max_inner")
    area4 = _f32(inits, [3.0], "size_base")
    t12 = _f32(inits, [12.5], "area_gt_3")
    t20 = _f32(inits, [20.5], "area_gt_4")
    t30 = _f32(inits, [30.5], "area_gt_5")
    fg_flag = _bool(inits, (np.arange(C).reshape(1, C, 1, 1) > 0), "fg_flag")
    fg_starts = _i64(inits, [1], "fg_starts")
    fg_ends = _i64(inits, [C], "fg_ends")
    axis_channel = _i64(inits, [1], "axis_channel")
    depth = _i64(inits, np.array(C, dtype=np.int64), "depth")
    values = _f32(inits, [0.0, 1.0], "onehot_values")
    off_index = _i64(inits, np.array(C, dtype=np.int64), "off_index")

    nodes.extend(
        [
            helper.make_node("ReduceSum", [IN_NAME], ["color_counts"], axes=[2, 3], keepdims=1),
            helper.make_node("Slice", ["color_counts", fg_starts, fg_ends, axis_channel], ["fg_counts"]),
            helper.make_node("ReduceSum", ["fg_counts"], ["area"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["color_counts", half], ["present_b0"]),
            helper.make_node("Less", ["color_counts", max_inner], ["small_count"]),
            helper.make_node("And", ["present_b0", "small_count"], ["small_present"]),
            helper.make_node("And", ["small_present", fg_flag], ["inner_color"]),
            helper.make_node("Greater", ["color_counts", max_inner], ["large_count"]),
            helper.make_node("And", ["large_count", fg_flag], ["outer_color"]),
            helper.make_node("Greater", ["area", t12], ["area_gt12"]),
            helper.make_node("Greater", ["area", t20], ["area_gt20"]),
            helper.make_node("Greater", ["area", t30], ["area_gt30"]),
            helper.make_node("Cast", ["area_gt12"], ["area_gt12_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["area_gt20"], ["area_gt20_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["area_gt30"], ["area_gt30_f"], to=TensorProto.FLOAT),
            helper.make_node("Add", [area4, "area_gt12_f"], ["size4"]),
            helper.make_node("Add", ["size4", "area_gt20_f"], ["size5"]),
            helper.make_node("Add", ["size5", "area_gt30_f"], ["size"]),
        ]
    )

    low = _sub(nodes, inits, "size", 2.0, "center_low")
    two_rows = _mul(nodes, inits, rows, 2.0, "two_rows")
    two_cols = _mul(nodes, inits, cols, 2.0, "two_cols")
    row_active = _less(nodes, rows, "size", "row_active")
    col_active = _less(nodes, cols, "size", "col_active")
    active_square = _and(nodes, row_active, col_active, "active_square")
    row_inner = _and(nodes, _ge(nodes, two_rows, low, "row_inner_ge"), _le(nodes, two_rows, "size", "row_inner_le"), "row_inner")
    col_inner = _and(nodes, _ge(nodes, two_cols, low, "col_inner_ge"), _le(nodes, two_cols, "size", "col_inner_le"), "col_inner")
    inner_at_origin = _and(nodes, row_inner, col_inner, "inner_at_origin")

    nodes.extend(
        [
            helper.make_node("Cast", ["inner_color"], ["inner_color_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["outer_color"], ["outer_color_f"], to=TensorProto.FLOAT),
            helper.make_node("ArgMax", ["inner_color_f"], ["inner_idx"], axis=1, keepdims=0),
            helper.make_node("ArgMax", ["outer_color_f"], ["outer_idx"], axis=1, keepdims=0),
            helper.make_node("Squeeze", [inner_at_origin], ["inner_mask"], axes=[1]),
            helper.make_node("Squeeze", [active_square], ["active_mask"], axes=[1]),
            helper.make_node("Where", ["inner_mask", "outer_idx", "inner_idx"], ["color_grid"]),
            helper.make_node("Where", ["active_mask", "color_grid", off_index], ["index_grid"]),
            helper.make_node("OneHot", ["index_grid", depth, values], ["out6"], axis=1),
            helper.make_node("Pad", ["out6"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - 6, W - 6]),
        ]
    )
    return _make_model(nodes, inits, "task290_area_count_compact")


def validate_reference(examples: list[Example]) -> None:
    for split, idx, inp, expected in examples:
        pred = solve(inp)
        if not np.array_equal(pred, expected):
            raise AssertionError(f"reference failed {split}[{idx}]:\n{pred}\n{expected}")


def validate_json(model: onnx.ModelProto, examples: list[Example]) -> None:
    for split, idx, inp, expected_grid in examples:
        pred = _run_onnx(model, inp)
        expected = _expected_onehot(expected_grid)
        if not np.array_equal(pred > 0.0, expected > 0.0):
            decoded = (pred[0, :, : expected_grid.shape[0], : expected_grid.shape[1]] > 0.0).argmax(axis=0)
            diff = int(np.sum(decoded != expected_grid))
            raise AssertionError(f"ONNX failed {split}[{idx}] with {diff} mismatched cropped cells")


def _score_candidate(label: str, build: Callable[[], onnx.ModelProto], examples: list[Example]) -> tuple[int, float, onnx.ModelProto]:
    model = build()
    validate_json(model, examples)
    with tempfile.NamedTemporaryFile(suffix=f"_{TASK_ID}.onnx", dir=OUT_DIR, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        onnx.save(model, tmp_path)
        result = score_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not result["valid"]:
        raise AssertionError(f"{label} invalid: {result['error']}")
    assert result["cost"] is not None and result["score"] is not None
    print(
        f"{label}: nodes={len(model.graph.node)} memory={result['memory']} "
        f"params={result['params']} cost={result['cost']} score={result['score']:.6f}"
    )
    return int(result["cost"]), float(result["score"]), model


def main() -> None:
    examples = load_examples()
    validate_reference(examples)
    candidates = [
        _score_candidate("area-count-compact", build_area_count_compact_model, examples),
        _score_candidate("area-count-onehot", build_area_count_onehot_model, examples),
        _score_candidate("center-color", build_center_color_model, examples),
        _score_candidate("border-color", build_border_color_model, examples),
    ]
    _cost, _score, best = min(candidates, key=lambda item: item[0])
    onnx.save(best, BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"nodes:   {len(best.graph.node)}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
