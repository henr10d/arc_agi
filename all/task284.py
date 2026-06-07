"""ONNX for ARC task284: expand two aligned seed pixels into a linked symbol.

Task rule: the input has exactly two non-black seed pixels on the same row or
the same column.  If they share a column, keep the top seed color above and the
bottom seed color below, drawing a 5-wide stacked pair of squared U/T shapes
around the midpoint.  If they share a row, use the same construction rotated:
the left seed color fills the left half and the right seed color fills the
right half.  The original grid extent is preserved; padding outside it remains
all-zero for the competition tensor contract.

ONNX approach: derive the two seed extrema directly from tiny coordinate
grids, build the two bool spatial masks with coordinate comparisons, broadcast
the two seed color vectors onto those masks, and add a black channel only over
the original input extent.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task284"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task284.onnx"
DATA_PATH = ROOT / "data" / "task284.json"

C = 10
H = W = 30
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


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


def load_examples() -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    examples: list[tuple[str, int, np.ndarray, np.ndarray]] = []
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
    """Reference implementation of the validated aligned-seed construction."""
    g = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(g)
    seeds = np.argwhere(g > 0)
    assert seeds.shape[0] == 2
    (r0, c0), (r1, c1) = seeds
    color0 = int(g[r0, c0])
    color1 = int(g[r1, c1])

    if c0 == c1:
        if r1 < r0:
            r0, c0, r1, c1 = r1, c1, r0, c0
            color0, color1 = color1, color0
        ra = (r0 + r1 - 3) // 2
        rb = ra + 3
        c = c0
        out[r0 : ra + 1, c] = color0
        out[ra, c - 2 : c + 3] = color0
        out[ra : ra + 2, c - 2] = color0
        out[ra : ra + 2, c + 2] = color0
        out[rb : r1 + 1, c] = color1
        out[rb, c - 2 : c + 3] = color1
        out[rb - 1 : rb + 1, c - 2] = color1
        out[rb - 1 : rb + 1, c + 2] = color1
    else:
        if c1 < c0:
            r0, c0, r1, c1 = r1, c1, r0, c0
            color0, color1 = color1, color0
        ca = (c0 + c1 - 3) // 2
        cb = ca + 3
        r = r0
        out[r, c0 : ca + 1] = color0
        out[r - 2 : r + 3, ca] = color0
        out[r - 2, ca : ca + 2] = color0
        out[r + 2, ca : ca + 2] = color0
        out[r, cb : c1 + 1] = color1
        out[r - 2 : r + 3, cb] = color1
        out[r - 2, cb - 1 : cb + 1] = color1
        out[r + 2, cb - 1 : cb + 1] = color1
    return out


def _grid_to_onehot(grid: np.ndarray) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            out[0, int(grid[r, c]), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    active = onehot > 0.0
    return active.argmax(axis=1)[0].astype(np.int64)


def _run_onnx(model: onnx.ModelProto, grid: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: _grid_to_onehot(grid)})[0]


def _add(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    a: str,
    b: str | float,
    out: str,
) -> str:
    if not isinstance(b, str):
        b = _f32(inits, [b], f"{out}_c")
    nodes.append(helper.make_node("Add", [a, b], [out]))
    return out


def _sub(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    a: str,
    b: str | float,
    out: str,
) -> str:
    if not isinstance(b, str):
        b = _f32(inits, [b], f"{out}_c")
    nodes.append(helper.make_node("Sub", [a, b], [out]))
    return out


def _div(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    a: str,
    b: str | float,
    out: str,
) -> str:
    if not isinstance(b, str):
        b = _f32(inits, [b], f"{out}_c")
    nodes.append(helper.make_node("Div", [a, b], [out]))
    return out


def _eq(nodes: List[onnx.NodeProto], a: str, b: str, out: str) -> str:
    ge = _ge(nodes, a, b, f"{out}_ge")
    le = _le(nodes, a, b, f"{out}_le")
    return _and(nodes, ge, le, out)


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


def _between(nodes: List[onnx.NodeProto], x: str, lo: str, hi: str, out: str) -> str:
    ge = _ge(nodes, x, lo, f"{out}_ge")
    le = _le(nodes, x, hi, f"{out}_le")
    return _and(nodes, ge, le, out)


def _or_many(nodes: List[onnx.NodeProto], names: list[str], prefix: str) -> str:
    current = names[0]
    for idx, name in enumerate(names[1:], start=1):
        current = _or(nodes, current, name, f"{prefix}_{idx}")
    return current


def build_coordinate_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    rows = _f32(inits, np.arange(H, dtype=np.float32).reshape(1, 1, H, 1), "rows")
    cols = _f32(inits, np.arange(W, dtype=np.float32).reshape(1, 1, 1, W), "cols")
    big = _f32(inits, [100.0], "big")
    neg = _f32(inits, [-1.0], "neg")
    half = _f32(inits, [0.5], "half")
    bg_flag = _bool(inits, np.arange(C).reshape(1, C, 1, 1) == 0, "bg_flag")

    starts = _i64(inits, [0, 1, 0, 0], "nonzero_starts")
    ends = _i64(inits, [1, C, H, W], "nonzero_ends")
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts, ends], ["nonzero_channels"]),
            helper.make_node("ReduceSum", ["nonzero_channels"], ["nonzero_f"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["nonzero_f", half], ["nonzero_b"]),
            helper.make_node("ReduceSum", [IN_NAME], ["interior_f"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["interior_f", half], ["interior_b"]),
            helper.make_node("Where", ["nonzero_b", rows, big], ["rows_for_min"]),
            helper.make_node("Where", ["nonzero_b", cols, big], ["cols_for_min"]),
            helper.make_node("Where", ["nonzero_b", rows, neg], ["rows_for_max"]),
            helper.make_node("Where", ["nonzero_b", cols, neg], ["cols_for_max"]),
            helper.make_node("ReduceMin", ["rows_for_min"], ["rmin"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMin", ["cols_for_min"], ["cmin"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMax", ["rows_for_max"], ["rmax"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMax", ["cols_for_max"], ["cmax"], axes=[2, 3], keepdims=1),
        ]
    )

    is_v = _less(nodes, "rmin", "rmax", "is_v")
    is_h = _less(nodes, "cmin", "cmax", "is_h")

    r_sum = _add(nodes, inits, "rmin", "rmax", "r_sum")
    r_inner = _div(nodes, inits, _sub(nodes, inits, r_sum, 3.0, "r_sum_m3"), 2.0, "r_a")
    r_b = _add(nodes, inits, r_inner, 3.0, "r_b")
    c_sum = _add(nodes, inits, "cmin", "cmax", "c_sum")
    c_inner = _div(nodes, inits, _sub(nodes, inits, c_sum, 3.0, "c_sum_m3"), 2.0, "c_a")
    c_b = _add(nodes, inits, c_inner, 3.0, "c_b")

    r_m2 = _sub(nodes, inits, "rmin", 2.0, "r_m2")
    r_p2 = _add(nodes, inits, "rmin", 2.0, "r_p2")
    c_m2 = _sub(nodes, inits, "cmin", 2.0, "c_m2")
    c_p2 = _add(nodes, inits, "cmin", 2.0, "c_p2")
    r_a_p1 = _add(nodes, inits, r_inner, 1.0, "r_a_p1")
    r_b_m1 = _sub(nodes, inits, r_b, 1.0, "r_b_m1")
    c_a_p1 = _add(nodes, inits, c_inner, 1.0, "c_a_p1")
    c_b_m1 = _sub(nodes, inits, c_b, 1.0, "c_b_m1")

    v_center_col = _eq(nodes, cols, "cmin", "v_center_col")
    v_left_col = _eq(nodes, cols, c_m2, "v_left_col")
    v_right_col = _eq(nodes, cols, c_p2, "v_right_col")
    v_side_cols = _or(nodes, v_left_col, v_right_col, "v_side_cols")
    v_bar_cols = _between(nodes, cols, c_m2, c_p2, "v_bar_cols")
    v_a_stem_rows = _between(nodes, rows, "rmin", r_inner, "v_a_stem_rows")
    v_b_stem_rows = _between(nodes, rows, r_b, "rmax", "v_b_stem_rows")
    v_a_side_rows = _between(nodes, rows, r_inner, r_a_p1, "v_a_side_rows")
    v_b_side_rows = _between(nodes, rows, r_b_m1, r_b, "v_b_side_rows")
    v_a_bar_row = _eq(nodes, rows, r_inner, "v_a_bar_row")
    v_b_bar_row = _eq(nodes, rows, r_b, "v_b_bar_row")
    v_a = _or_many(
        nodes,
        [
            _and(nodes, v_center_col, v_a_stem_rows, "v_a_stem"),
            _and(nodes, v_a_bar_row, v_bar_cols, "v_a_bar"),
            _and(nodes, v_side_cols, v_a_side_rows, "v_a_sides"),
        ],
        "v_a",
    )
    v_b = _or_many(
        nodes,
        [
            _and(nodes, v_center_col, v_b_stem_rows, "v_b_stem"),
            _and(nodes, v_b_bar_row, v_bar_cols, "v_b_bar"),
            _and(nodes, v_side_cols, v_b_side_rows, "v_b_sides"),
        ],
        "v_b",
    )

    h_center_row = _eq(nodes, rows, "rmin", "h_center_row")
    h_top_row = _eq(nodes, rows, r_m2, "h_top_row")
    h_bottom_row = _eq(nodes, rows, r_p2, "h_bottom_row")
    h_edge_rows = _or(nodes, h_top_row, h_bottom_row, "h_edge_rows")
    h_bar_rows = _between(nodes, rows, r_m2, r_p2, "h_bar_rows")
    h_a_stem_cols = _between(nodes, cols, "cmin", c_inner, "h_a_stem_cols")
    h_b_stem_cols = _between(nodes, cols, c_b, "cmax", "h_b_stem_cols")
    h_a_top_cols = _between(nodes, cols, c_inner, c_a_p1, "h_a_top_cols")
    h_b_top_cols = _between(nodes, cols, c_b_m1, c_b, "h_b_top_cols")
    h_a_bar_col = _eq(nodes, cols, c_inner, "h_a_bar_col")
    h_b_bar_col = _eq(nodes, cols, c_b, "h_b_bar_col")
    h_a = _or_many(
        nodes,
        [
            _and(nodes, h_center_row, h_a_stem_cols, "h_a_stem"),
            _and(nodes, h_a_bar_col, h_bar_rows, "h_a_bar"),
            _and(nodes, h_edge_rows, h_a_top_cols, "h_a_caps"),
        ],
        "h_a",
    )
    h_b = _or_many(
        nodes,
        [
            _and(nodes, h_center_row, h_b_stem_cols, "h_b_stem"),
            _and(nodes, h_b_bar_col, h_bar_rows, "h_b_bar"),
            _and(nodes, h_edge_rows, h_b_top_cols, "h_b_caps"),
        ],
        "h_b",
    )

    a_shape = _or(nodes, _and(nodes, is_v, v_a, "a_v"), _and(nodes, is_h, h_a, "a_h"), "a_shape")
    b_shape = _or(nodes, _and(nodes, is_v, v_b, "b_v"), _and(nodes, is_h, h_b, "b_h"), "b_shape")

    a_pos = _or(
        nodes,
        _and(nodes, is_v, _and(nodes, "nonzero_b", _eq(nodes, rows, "rmin", "a_pos_v_row"), "a_pos_v"), "a_pos_v_oriented"),
        _and(nodes, is_h, _and(nodes, "nonzero_b", _eq(nodes, cols, "cmin", "a_pos_h_col"), "a_pos_h"), "a_pos_h_oriented"),
        "a_pos",
    )
    b_pos = _or(
        nodes,
        _and(nodes, is_v, _and(nodes, "nonzero_b", _eq(nodes, rows, "rmax", "b_pos_v_row"), "b_pos_v"), "b_pos_v_oriented"),
        _and(nodes, is_h, _and(nodes, "nonzero_b", _eq(nodes, cols, "cmax", "b_pos_h_col"), "b_pos_h"), "b_pos_h_oriented"),
        "b_pos",
    )

    nodes.extend(
        [
            helper.make_node("Cast", ["a_pos"], ["a_pos_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["b_pos"], ["b_pos_f"], to=TensorProto.FLOAT),
            helper.make_node("Mul", [IN_NAME, "a_pos_f"], ["a_color_pick"]),
            helper.make_node("Mul", [IN_NAME, "b_pos_f"], ["b_color_pick"]),
            helper.make_node("ReduceMax", ["a_color_pick"], ["a_color_f"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMax", ["b_color_pick"], ["b_color_f"], axes=[2, 3], keepdims=1),
            helper.make_node("Greater", ["a_color_f", half], ["a_color_b"]),
            helper.make_node("Greater", ["b_color_f", half], ["b_color_b"]),
            helper.make_node("And", ["a_color_b", a_shape], ["a_channels"]),
            helper.make_node("And", ["b_color_b", b_shape], ["b_channels"]),
            helper.make_node("Or", ["a_channels", "b_channels"], ["color_channels"]),
            helper.make_node("Or", [a_shape, b_shape], ["painted"]),
            helper.make_node("Not", ["painted"], ["not_painted"]),
            helper.make_node("And", ["interior_b", "not_painted"], ["bg_spatial"]),
            helper.make_node("And", [bg_flag, "bg_spatial"], ["bg_channels"]),
            helper.make_node("Or", ["color_channels", "bg_channels"], ["out_b"]),
            helper.make_node("Cast", ["out_b"], [OUT_NAME], to=TensorProto.FLOAT),
        ]
    )
    return _make_model(nodes, inits, "task284_coordinate_geometry")


def build_argmax_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    rows = _f32(inits, np.arange(H, dtype=np.float32).reshape(1, 1, H, 1), "rows")
    cols = _f32(inits, np.arange(W, dtype=np.float32).reshape(1, 1, 1, W), "cols")
    half = _f32(inits, [0.5], "half")
    zero_i = _i64(inits, [0], "zero_i")
    zero_f = _f32(inits, [0.0], "zero_f")
    neg_one_f = _f32(inits, [-1.0], "neg_one_f")
    channel_ids = _i64(inits, np.arange(C, dtype=np.int64).reshape(1, C, 1, 1), "channel_ids")
    bg_starts = _i64(inits, [0, 0, 0, 0], "bg_starts")
    bg_ends = _i64(inits, [1, 1, H, W], "bg_ends")

    nodes.extend(
        [
            helper.make_node("ArgMax", [IN_NAME], ["color_idx"], axis=1, keepdims=1),
            helper.make_node("Greater", ["color_idx", zero_i], ["nonzero_b"]),
            helper.make_node("Cast", ["nonzero_b"], ["nonzero_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["color_idx"], ["color_idx_f"], to=TensorProto.FLOAT),
            helper.make_node("Slice", [IN_NAME, bg_starts, bg_ends], ["bg_f"]),
            helper.make_node("Greater", ["bg_f", half], ["bg_spatial_in"]),
            helper.make_node("Mul", ["nonzero_f", rows], ["seed_rows"]),
            helper.make_node("Mul", ["nonzero_f", cols], ["seed_cols"]),
            helper.make_node("ReduceSum", ["seed_rows"], ["r_seed_sum"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceSum", ["seed_cols"], ["c_seed_sum"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMax", ["seed_rows"], ["rmax"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMax", ["seed_cols"], ["cmax"], axes=[2, 3], keepdims=1),
            helper.make_node("Sub", ["r_seed_sum", "rmax"], ["rmin"]),
            helper.make_node("Sub", ["c_seed_sum", "cmax"], ["cmin"]),
        ]
    )

    is_v = _less(nodes, "rmin", "rmax", "is_v")
    is_h = _less(nodes, "cmin", "cmax", "is_h")

    r_sum = _add(nodes, inits, "rmin", "rmax", "r_sum")
    r_inner = _div(nodes, inits, _sub(nodes, inits, r_sum, 3.0, "r_sum_m3"), 2.0, "r_a")
    r_b = _add(nodes, inits, r_inner, 3.0, "r_b")
    c_sum = _add(nodes, inits, "cmin", "cmax", "c_sum")
    c_inner = _div(nodes, inits, _sub(nodes, inits, c_sum, 3.0, "c_sum_m3"), 2.0, "c_a")
    c_b = _add(nodes, inits, c_inner, 3.0, "c_b")

    r_m2 = _sub(nodes, inits, "rmin", 2.0, "r_m2")
    r_p2 = _add(nodes, inits, "rmin", 2.0, "r_p2")
    c_m2 = _sub(nodes, inits, "cmin", 2.0, "c_m2")
    c_p2 = _add(nodes, inits, "cmin", 2.0, "c_p2")
    r_a_p1 = _add(nodes, inits, r_inner, 1.0, "r_a_p1")
    r_b_m1 = _sub(nodes, inits, r_b, 1.0, "r_b_m1")
    c_a_p1 = _add(nodes, inits, c_inner, 1.0, "c_a_p1")
    c_b_m1 = _sub(nodes, inits, c_b, 1.0, "c_b_m1")

    v_center_col = _eq(nodes, cols, "cmin", "v_center_col")
    v_left_col = _eq(nodes, cols, c_m2, "v_left_col")
    v_right_col = _eq(nodes, cols, c_p2, "v_right_col")
    v_side_cols = _or(nodes, v_left_col, v_right_col, "v_side_cols")
    v_bar_cols = _between(nodes, cols, c_m2, c_p2, "v_bar_cols")
    v_a_stem_rows = _between(nodes, rows, "rmin", r_inner, "v_a_stem_rows")
    v_b_stem_rows = _between(nodes, rows, r_b, "rmax", "v_b_stem_rows")
    v_a_side_rows = _between(nodes, rows, r_inner, r_a_p1, "v_a_side_rows")
    v_b_side_rows = _between(nodes, rows, r_b_m1, r_b, "v_b_side_rows")
    v_a_bar_row = _eq(nodes, rows, r_inner, "v_a_bar_row")
    v_b_bar_row = _eq(nodes, rows, r_b, "v_b_bar_row")
    v_a = _or_many(
        nodes,
        [
            _and(nodes, v_center_col, v_a_stem_rows, "v_a_stem"),
            _and(nodes, v_a_bar_row, v_bar_cols, "v_a_bar"),
            _and(nodes, v_side_cols, v_a_side_rows, "v_a_sides"),
        ],
        "v_a",
    )
    v_b = _or_many(
        nodes,
        [
            _and(nodes, v_center_col, v_b_stem_rows, "v_b_stem"),
            _and(nodes, v_b_bar_row, v_bar_cols, "v_b_bar"),
            _and(nodes, v_side_cols, v_b_side_rows, "v_b_sides"),
        ],
        "v_b",
    )

    h_center_row = _eq(nodes, rows, "rmin", "h_center_row")
    h_top_row = _eq(nodes, rows, r_m2, "h_top_row")
    h_bottom_row = _eq(nodes, rows, r_p2, "h_bottom_row")
    h_edge_rows = _or(nodes, h_top_row, h_bottom_row, "h_edge_rows")
    h_bar_rows = _between(nodes, rows, r_m2, r_p2, "h_bar_rows")
    h_a_stem_cols = _between(nodes, cols, "cmin", c_inner, "h_a_stem_cols")
    h_b_stem_cols = _between(nodes, cols, c_b, "cmax", "h_b_stem_cols")
    h_a_top_cols = _between(nodes, cols, c_inner, c_a_p1, "h_a_top_cols")
    h_b_top_cols = _between(nodes, cols, c_b_m1, c_b, "h_b_top_cols")
    h_a_bar_col = _eq(nodes, cols, c_inner, "h_a_bar_col")
    h_b_bar_col = _eq(nodes, cols, c_b, "h_b_bar_col")
    h_a = _or_many(
        nodes,
        [
            _and(nodes, h_center_row, h_a_stem_cols, "h_a_stem"),
            _and(nodes, h_a_bar_col, h_bar_rows, "h_a_bar"),
            _and(nodes, h_edge_rows, h_a_top_cols, "h_a_caps"),
        ],
        "h_a",
    )
    h_b = _or_many(
        nodes,
        [
            _and(nodes, h_center_row, h_b_stem_cols, "h_b_stem"),
            _and(nodes, h_b_bar_col, h_bar_rows, "h_b_bar"),
            _and(nodes, h_edge_rows, h_b_top_cols, "h_b_caps"),
        ],
        "h_b",
    )

    a_shape = _or(nodes, _and(nodes, is_v, v_a, "a_v"), _and(nodes, is_h, h_a, "a_h"), "a_shape")
    b_shape = _or(nodes, _and(nodes, is_v, v_b, "b_v"), _and(nodes, is_h, h_b, "b_h"), "b_shape")

    a_pos = _or(
        nodes,
        _and(nodes, is_v, _and(nodes, "nonzero_b", _eq(nodes, rows, "rmin", "a_pos_v_row"), "a_pos_v"), "a_pos_v_oriented"),
        _and(nodes, is_h, _and(nodes, "nonzero_b", _eq(nodes, cols, "cmin", "a_pos_h_col"), "a_pos_h"), "a_pos_h_oriented"),
        "a_pos",
    )
    b_pos = _or(
        nodes,
        _and(nodes, is_v, _and(nodes, "nonzero_b", _eq(nodes, rows, "rmax", "b_pos_v_row"), "b_pos_v"), "b_pos_v_oriented"),
        _and(nodes, is_h, _and(nodes, "nonzero_b", _eq(nodes, cols, "cmax", "b_pos_h_col"), "b_pos_h"), "b_pos_h_oriented"),
        "b_pos",
    )

    nodes.extend(
        [
            helper.make_node("Where", ["a_pos", "color_idx_f", zero_f], ["a_color_grid"]),
            helper.make_node("Where", ["b_pos", "color_idx_f", zero_f], ["b_color_grid"]),
            helper.make_node("ReduceMax", ["a_color_grid"], ["a_color_f"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMax", ["b_color_grid"], ["b_color_f"], axes=[2, 3], keepdims=1),
            helper.make_node("Where", [a_shape, "a_color_f", zero_f], ["a_painted_color"]),
            helper.make_node("Where", [b_shape, "b_color_f", "a_painted_color"], ["painted_color"]),
            helper.make_node("Or", [a_shape, b_shape], ["painted"]),
            helper.make_node("Or", ["bg_spatial_in", "painted"], ["valid_spatial"]),
            helper.make_node("Where", ["valid_spatial", "painted_color", neg_one_f], ["output_color_f"]),
            helper.make_node("Cast", ["output_color_f"], ["output_color_i"], to=TensorProto.INT64),
            helper.make_node("Equal", [channel_ids, "output_color_i"], ["out_b"]),
            helper.make_node("Cast", ["out_b"], [OUT_NAME], to=TensorProto.FLOAT),
        ]
    )
    return _make_model(nodes, inits, "task284_argmax_geometry")


def validate_reference(examples: list[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    for split, idx, inp, expected in examples:
        pred = solve(inp)
        if not np.array_equal(pred, expected):
            diff = int(np.sum(pred != expected))
            raise AssertionError(f"reference failed {split}[{idx}] with {diff} mismatches")


def validate_json(model: onnx.ModelProto, examples: list[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    for split, idx, inp, expected in examples:
        pred_oh = _run_onnx(model, inp)
        active = pred_oh[0, :, : expected.shape[0], : expected.shape[1]] > 0.0
        decoded = active.argmax(axis=0).astype(np.int64)
        if not np.array_equal(decoded, expected) or not np.all(active.sum(axis=0) == 1):
            diff = int(np.sum(decoded != expected))
            raise AssertionError(f"{split}[{idx}] failed with {diff} mismatched cells")
        outside_rows = pred_oh[0, :, expected.shape[0] :, :] > 0.0
        outside_cols = pred_oh[0, :, :, expected.shape[1] :] > 0.0
        if outside_rows.any() or outside_cols.any():
            raise AssertionError(f"{split}[{idx}] wrote outside the original grid")


def _score_candidate(label: str, build: Callable[[], onnx.ModelProto], examples: list[tuple[str, int, np.ndarray, np.ndarray]]) -> tuple[int, float, onnx.ModelProto]:
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
        _score_candidate("coordinate-geometry", build_coordinate_model, examples),
        _score_candidate("argmax-geometry", build_argmax_model, examples),
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
