"""Build an ONNX solution for ARC task377's nested rectangle hierarchy.

Task rule: the input is a stack of solid rectangles drawn inside one another,
possibly reusing an earlier color at a deeper level. Starting from the whole
grid, record the color at the current rectangle's top-left corner, then find
the bounding box of all cells inside it that differ from that color. Repeating
this recovers the outer-to-inner layer colors. The output is the small
concentric-ring grid whose side length is ``2 * layers - 1``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TASK_ID = "task377"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task377.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

IN_NAME = "input"
OUT_NAME = "output"
C = 10
H = W = 30
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10
MAX_LAYERS = 5


def solve(grid: list[list[int]]) -> list[list[int]]:
    """Reference implementation of the rectangle hierarchy rule."""
    arr = np.asarray(grid, dtype=np.int64)
    top, left = 0, 0
    bottom, right = arr.shape[0] - 1, arr.shape[1] - 1
    layers: list[int] = []

    for _ in range(MAX_LAYERS):
        color = int(arr[top, left])
        layers.append(color)
        sub = arr[top : bottom + 1, left : right + 1]
        mask = sub != color
        if not bool(mask.any()):
            break
        rows, cols = np.where(mask)
        bottom = top + int(rows.max())
        right = left + int(cols.max())
        top = top + int(rows.min())
        left = left + int(cols.min())

    side = 2 * len(layers) - 1
    return [
        [layers[min(r, c, side - 1 - r, side - 1 - c)] for c in range(side)]
        for r in range(side)
    ]


def _add_init(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _node(nodes: list[onnx.NodeProto], op: str, inputs: list[str], output: str, **attrs: Any) -> str:
    nodes.append(helper.make_node(op, inputs, [output], **attrs))
    return output


def _and(nodes: list[onnx.NodeProto], a: str, b: str, name: str) -> str:
    return _node(nodes, "And", [a, b], name)


def _or(nodes: list[onnx.NodeProto], a: str, b: str, name: str) -> str:
    return _node(nodes, "Or", [a, b], name)


def _not(nodes: list[onnx.NodeProto], x: str, name: str) -> str:
    return _node(nodes, "Not", [x], name)


def _ge(nodes: list[onnx.NodeProto], a: str, b: str, name: str) -> str:
    lt = _node(nodes, "Less", [a, b], f"{name}_lt")
    return _not(nodes, lt, name)


def _le(nodes: list[onnx.NodeProto], a: str, b: str, name: str) -> str:
    gt = _node(nodes, "Greater", [a, b], f"{name}_gt")
    return _not(nodes, gt, name)


def _cell_color(
    nodes: list[onnx.NodeProto],
    row_idx: str,
    col_idx: str,
    top: str,
    left: str,
    prefix: str,
) -> str:
    row_eq = _node(nodes, "Equal", [row_idx, top], f"{prefix}_row_eq")
    col_eq = _node(nodes, "Equal", [col_idx, left], f"{prefix}_col_eq")
    cell = _and(nodes, row_eq, col_eq, f"{prefix}_cell")
    cell_f = _node(nodes, "Cast", [cell], f"{prefix}_cell_f", to=TensorProto.FLOAT)
    picked = _node(nodes, "Mul", [IN_NAME, cell_f], f"{prefix}_picked")
    return _node(nodes, "ReduceMax", [picked], f"{prefix}_color", axes=[2, 3], keepdims=1)


def _cell_color_id(
    nodes: list[onnx.NodeProto],
    color_img: str,
    row_idx: str,
    col_idx: str,
    top: str,
    left: str,
    zero_i: str,
    prefix: str,
) -> str:
    row_eq = _node(nodes, "Equal", [row_idx, top], f"{prefix}_row_eq")
    col_eq = _node(nodes, "Equal", [col_idx, left], f"{prefix}_col_eq")
    cell = _and(nodes, row_eq, col_eq, f"{prefix}_cell")
    picked = _node(nodes, "Where", [cell, color_img, zero_i], f"{prefix}_picked")
    return _node(nodes, "ReduceMax", [picked], f"{prefix}_color", axes=[2, 3], keepdims=1)


def _cell_color_gather(
    nodes: list[onnx.NodeProto],
    flat_colors: str,
    active: str,
    top: str,
    left: str,
    zero_i: str,
    thirty_i: str,
    prefix: str,
) -> str:
    row_offset = _node(nodes, "Mul", [top, thirty_i], f"{prefix}_row_offset")
    raw_index = _node(nodes, "Add", [row_offset, left], f"{prefix}_raw_index")
    index = _node(nodes, "Where", [active, raw_index, zero_i], f"{prefix}_index")
    return _node(nodes, "Gather", [flat_colors, index], f"{prefix}_color", axis=0)


def _inside_rect(
    nodes: list[onnx.NodeProto],
    row_idx: str,
    col_idx: str,
    top: str,
    left: str,
    bottom: str,
    right: str,
    prefix: str,
) -> str:
    row_ge = _ge(nodes, row_idx, top, f"{prefix}_row_ge")
    row_le = _le(nodes, row_idx, bottom, f"{prefix}_row_le")
    col_ge = _ge(nodes, col_idx, left, f"{prefix}_col_ge")
    col_le = _le(nodes, col_idx, right, f"{prefix}_col_le")
    rows = _and(nodes, row_ge, row_le, f"{prefix}_rows")
    cols = _and(nodes, col_ge, col_le, f"{prefix}_cols")
    return _and(nodes, rows, cols, f"{prefix}_inside")


def _next_box(
    nodes: list[onnx.NodeProto],
    row_idx: str,
    col_idx: str,
    row_idx_line: str,
    col_idx_line: str,
    valid: str,
    zero_f: str,
    zero_i: str,
    thirty_i: str,
    active: str,
    top: str,
    left: str,
    bottom: str,
    right: str,
    color: str,
    prefix: str,
) -> tuple[str, str, str, str, str]:
    inside = _inside_rect(nodes, row_idx, col_idx, top, left, bottom, right, f"{prefix}_box")
    color_hits = _node(nodes, "Mul", [IN_NAME, color], f"{prefix}_hits")
    hit_sum = _node(nodes, "ReduceSum", [color_hits], f"{prefix}_hit_sum", axes=[1], keepdims=1)
    same = _node(nodes, "Greater", [hit_sum, zero_f], f"{prefix}_same")
    diff_color = _not(nodes, same, f"{prefix}_diff_color")
    diff = _and(nodes, _and(nodes, inside, valid, f"{prefix}_inside_valid"), diff_color, f"{prefix}_diff0")
    diff = _and(nodes, diff, active, f"{prefix}_diff")
    diff_i = _node(nodes, "Cast", [diff], f"{prefix}_diff_i", to=TensorProto.INT64)

    any_i = _node(nodes, "ReduceMax", [diff_i], f"{prefix}_any_i", axes=[2, 3], keepdims=1)
    any_b = _node(nodes, "Greater", [any_i, zero_i], f"{prefix}_any")
    next_active = _and(nodes, active, any_b, f"{prefix}_next_active")

    row_present = _node(nodes, "ReduceMax", [diff_i], f"{prefix}_row_present", axes=[3], keepdims=1)
    row_present_b = _node(nodes, "Greater", [row_present, zero_i], f"{prefix}_row_present_b")
    row_weighted = _node(nodes, "Mul", [row_present, row_idx_line], f"{prefix}_row_weighted")
    next_bottom = _node(nodes, "ReduceMax", [row_weighted], f"{prefix}_bottom", axes=[2, 3], keepdims=1)
    row_min_vals = _node(nodes, "Where", [row_present_b, row_idx_line, thirty_i], f"{prefix}_row_min_vals")
    next_top = _node(nodes, "ReduceMin", [row_min_vals], f"{prefix}_top", axes=[2, 3], keepdims=1)

    col_present = _node(nodes, "ReduceMax", [diff_i], f"{prefix}_col_present", axes=[2], keepdims=1)
    col_present_b = _node(nodes, "Greater", [col_present, zero_i], f"{prefix}_col_present_b")
    col_weighted = _node(nodes, "Mul", [col_present, col_idx_line], f"{prefix}_col_weighted")
    next_right = _node(nodes, "ReduceMax", [col_weighted], f"{prefix}_right", axes=[2, 3], keepdims=1)
    col_min_vals = _node(nodes, "Where", [col_present_b, col_idx_line, thirty_i], f"{prefix}_col_min_vals")
    next_left = _node(nodes, "ReduceMin", [col_min_vals], f"{prefix}_left", axes=[2, 3], keepdims=1)

    return next_active, next_top, next_left, next_bottom, next_right


def _next_box_id(
    nodes: list[onnx.NodeProto],
    color_img: str,
    row_idx: str,
    col_idx: str,
    valid: str,
    zero_i: str,
    thirty_i: str,
    active: str,
    top: str,
    left: str,
    bottom: str,
    right: str,
    color: str,
    prefix: str,
) -> tuple[str, str, str, str, str]:
    inside = _inside_rect(nodes, row_idx, col_idx, top, left, bottom, right, f"{prefix}_box")
    same = _node(nodes, "Equal", [color_img, color], f"{prefix}_same")
    diff_color = _not(nodes, same, f"{prefix}_diff_color")
    diff = _and(nodes, _and(nodes, inside, valid, f"{prefix}_inside_valid"), diff_color, f"{prefix}_diff0")
    diff = _and(nodes, diff, active, f"{prefix}_diff")
    diff_i = _node(nodes, "Cast", [diff], f"{prefix}_diff_i", to=TensorProto.INT64)

    any_i = _node(nodes, "ReduceMax", [diff_i], f"{prefix}_any_i", axes=[2, 3], keepdims=1)
    any_b = _node(nodes, "Greater", [any_i, zero_i], f"{prefix}_any")
    next_active = _and(nodes, active, any_b, f"{prefix}_next_active")

    row_present = _node(nodes, "ReduceMax", [diff_i], f"{prefix}_row_present", axes=[3], keepdims=1)
    row_present_b = _node(nodes, "Greater", [row_present, zero_i], f"{prefix}_row_present_b")
    row_weighted = _node(nodes, "Mul", [row_present, row_idx], f"{prefix}_row_weighted")
    next_bottom = _node(nodes, "ReduceMax", [row_weighted], f"{prefix}_bottom", axes=[2, 3], keepdims=1)
    row_min_vals = _node(nodes, "Where", [row_present_b, row_idx, thirty_i], f"{prefix}_row_min_vals")
    next_top = _node(nodes, "ReduceMin", [row_min_vals], f"{prefix}_top", axes=[2, 3], keepdims=1)

    col_present = _node(nodes, "ReduceMax", [diff_i], f"{prefix}_col_present", axes=[2], keepdims=1)
    col_present_b = _node(nodes, "Greater", [col_present, zero_i], f"{prefix}_col_present_b")
    col_weighted = _node(nodes, "Mul", [col_present, col_idx], f"{prefix}_col_weighted")
    next_right = _node(nodes, "ReduceMax", [col_weighted], f"{prefix}_right", axes=[2, 3], keepdims=1)
    col_min_vals = _node(nodes, "Where", [col_present_b, col_idx, thirty_i], f"{prefix}_col_min_vals")
    next_left = _node(nodes, "ReduceMin", [col_min_vals], f"{prefix}_left", axes=[2, 3], keepdims=1)

    return next_active, next_top, next_left, next_bottom, next_right


def _next_box_i32(
    nodes: list[onnx.NodeProto],
    color_img: str,
    row_idx: str,
    col_idx: str,
    valid: str,
    zero_i32: str,
    zero_i: str,
    thirty_i: str,
    active: str,
    top: str,
    left: str,
    bottom: str,
    right: str,
    color: str,
    prefix: str,
) -> tuple[str, str, str, str, str]:
    inside = _inside_rect(nodes, row_idx, col_idx, top, left, bottom, right, f"{prefix}_box")
    same = _node(nodes, "Equal", [color_img, color], f"{prefix}_same")
    diff_color = _not(nodes, same, f"{prefix}_diff_color")
    diff = _and(nodes, _and(nodes, inside, valid, f"{prefix}_inside_valid"), diff_color, f"{prefix}_diff0")
    diff = _and(nodes, diff, active, f"{prefix}_diff")
    diff_i32 = _node(nodes, "Cast", [diff], f"{prefix}_diff_i32", to=TensorProto.INT32)

    any_i32 = _node(nodes, "ReduceMax", [diff_i32], f"{prefix}_any_i32", axes=[2, 3], keepdims=1)
    any_b = _node(nodes, "Greater", [any_i32, zero_i32], f"{prefix}_any")
    next_active = _and(nodes, active, any_b, f"{prefix}_next_active")

    row_present_i32 = _node(nodes, "ReduceMax", [diff_i32], f"{prefix}_row_present_i32", axes=[3], keepdims=1)
    row_present_b = _node(nodes, "Greater", [row_present_i32, zero_i32], f"{prefix}_row_present_b")
    row_present = _node(nodes, "Cast", [row_present_i32], f"{prefix}_row_present", to=TensorProto.INT64)
    row_weighted = _node(nodes, "Mul", [row_present, row_idx], f"{prefix}_row_weighted")
    next_bottom = _node(nodes, "ReduceMax", [row_weighted], f"{prefix}_bottom", axes=[2, 3], keepdims=1)
    row_min_vals = _node(nodes, "Where", [row_present_b, row_idx, thirty_i], f"{prefix}_row_min_vals")
    next_top = _node(nodes, "ReduceMin", [row_min_vals], f"{prefix}_top", axes=[2, 3], keepdims=1)

    col_present_i32 = _node(nodes, "ReduceMax", [diff_i32], f"{prefix}_col_present_i32", axes=[2], keepdims=1)
    col_present_b = _node(nodes, "Greater", [col_present_i32, zero_i32], f"{prefix}_col_present_b")
    col_present = _node(nodes, "Cast", [col_present_i32], f"{prefix}_col_present", to=TensorProto.INT64)
    col_weighted = _node(nodes, "Mul", [col_present, col_idx], f"{prefix}_col_weighted")
    next_right = _node(nodes, "ReduceMax", [col_weighted], f"{prefix}_right", axes=[2, 3], keepdims=1)
    col_min_vals = _node(nodes, "Where", [col_present_b, col_idx, thirty_i], f"{prefix}_col_min_vals")
    next_left = _node(nodes, "ReduceMin", [col_min_vals], f"{prefix}_left", axes=[2, 3], keepdims=1)

    return next_active, next_top, next_left, next_bottom, next_right


def _initial_bounds(
    nodes: list[onnx.NodeProto],
    valid_sum: str,
    row_idx: str,
    col_idx: str,
    zero_f: str,
    zero_i: str,
) -> tuple[str, str]:
    row_present_f = _node(nodes, "ReduceMax", [valid_sum], "init_row_present_f", axes=[3], keepdims=1)
    row_present_b = _node(nodes, "Greater", [row_present_f, zero_f], "init_row_present_b")
    row_vals = _node(nodes, "Where", [row_present_b, row_idx, zero_i], "init_row_vals")
    bottom = _node(nodes, "ReduceMax", [row_vals], "init_bottom", axes=[2, 3], keepdims=1)

    col_present_f = _node(nodes, "ReduceMax", [valid_sum], "init_col_present_f", axes=[2], keepdims=1)
    col_present_b = _node(nodes, "Greater", [col_present_f, zero_f], "init_col_present_b")
    col_vals = _node(nodes, "Where", [col_present_b, col_idx, zero_i], "init_col_vals")
    right = _node(nodes, "ReduceMax", [col_vals], "init_right", axes=[2, 3], keepdims=1)
    return bottom, right


def _initial_bounds_from_color_img(
    nodes: list[onnx.NodeProto],
    color_img: str,
    row_idx: str,
    col_idx: str,
    zero_i32: str,
    zero_i: str,
) -> tuple[str, str]:
    row_present_i32 = _node(nodes, "ReduceMax", [color_img], "init_row_present_i32", axes=[3], keepdims=1)
    row_present_b = _node(nodes, "Greater", [row_present_i32, zero_i32], "init_row_present_b")
    row_vals = _node(nodes, "Where", [row_present_b, row_idx, zero_i], "init_row_vals")
    bottom = _node(nodes, "ReduceMax", [row_vals], "init_bottom", axes=[2, 3], keepdims=1)

    col_present_i32 = _node(nodes, "ReduceMax", [color_img], "init_col_present_i32", axes=[2], keepdims=1)
    col_present_b = _node(nodes, "Greater", [col_present_i32, zero_i32], "init_col_present_b")
    col_vals = _node(nodes, "Where", [col_present_b, col_idx, zero_i], "init_col_vals")
    right = _node(nodes, "ReduceMax", [col_vals], "init_right", axes=[2, 3], keepdims=1)
    return bottom, right


def _next_box_i32_bounded(
    nodes: list[onnx.NodeProto],
    color_img: str,
    row_idx: str,
    col_idx: str,
    zero_i32: str,
    zero_i: str,
    thirty_i: str,
    active: str,
    top: str,
    left: str,
    bottom: str,
    right: str,
    color: str,
    prefix: str,
) -> tuple[str, str, str, str, str]:
    inside = _inside_rect(nodes, row_idx, col_idx, top, left, bottom, right, f"{prefix}_box")
    same = _node(nodes, "Equal", [color_img, color], f"{prefix}_same")
    diff_color = _not(nodes, same, f"{prefix}_diff_color")
    diff = _and(nodes, _and(nodes, inside, diff_color, f"{prefix}_diff0"), active, f"{prefix}_diff")
    diff_i32 = _node(nodes, "Cast", [diff], f"{prefix}_diff_i32", to=TensorProto.INT32)

    any_i32 = _node(nodes, "ReduceMax", [diff_i32], f"{prefix}_any_i32", axes=[2, 3], keepdims=1)
    any_b = _node(nodes, "Greater", [any_i32, zero_i32], f"{prefix}_any")
    next_active = _and(nodes, active, any_b, f"{prefix}_next_active")

    row_present_i32 = _node(nodes, "ReduceMax", [diff_i32], f"{prefix}_row_present_i32", axes=[3], keepdims=1)
    row_present_b = _node(nodes, "Greater", [row_present_i32, zero_i32], f"{prefix}_row_present_b")
    row_present = _node(nodes, "Cast", [row_present_i32], f"{prefix}_row_present", to=TensorProto.INT64)
    row_weighted = _node(nodes, "Mul", [row_present, row_idx], f"{prefix}_row_weighted")
    next_bottom = _node(nodes, "ReduceMax", [row_weighted], f"{prefix}_bottom", axes=[2, 3], keepdims=1)
    row_min_vals = _node(nodes, "Where", [row_present_b, row_idx, thirty_i], f"{prefix}_row_min_vals")
    next_top = _node(nodes, "ReduceMin", [row_min_vals], f"{prefix}_top", axes=[2, 3], keepdims=1)

    col_present_i32 = _node(nodes, "ReduceMax", [diff_i32], f"{prefix}_col_present_i32", axes=[2], keepdims=1)
    col_present_b = _node(nodes, "Greater", [col_present_i32, zero_i32], f"{prefix}_col_present_b")
    col_present = _node(nodes, "Cast", [col_present_i32], f"{prefix}_col_present", to=TensorProto.INT64)
    col_weighted = _node(nodes, "Mul", [col_present, col_idx], f"{prefix}_col_weighted")
    next_right = _node(nodes, "ReduceMax", [col_weighted], f"{prefix}_right", axes=[2, 3], keepdims=1)
    col_min_vals = _node(nodes, "Where", [col_present_b, col_idx, thirty_i], f"{prefix}_col_min_vals")
    next_left = _node(nodes, "ReduceMin", [col_min_vals], f"{prefix}_left", axes=[2, 3], keepdims=1)

    return next_active, next_top, next_left, next_bottom, next_right


def _ring_mask(
    nodes: list[onnx.NodeProto],
    row_idx: str,
    col_idx: str,
    active: str,
    n_layers: str,
    two_i: str,
    level_i: str,
    level: int,
    prefix: str,
) -> str:
    two_n = _node(nodes, "Add", [n_layers, n_layers], f"{prefix}_two_n")
    end_base = _node(nodes, "Sub", [two_n, two_i], f"{prefix}_end_base")
    end = _node(nodes, "Sub", [end_base, level_i], f"{prefix}_end")

    row_ge = _ge(nodes, row_idx, level_i, f"{prefix}_row_ge")
    col_ge = _ge(nodes, col_idx, level_i, f"{prefix}_col_ge")
    row_le = _le(nodes, row_idx, end, f"{prefix}_row_le")
    col_le = _le(nodes, col_idx, end, f"{prefix}_col_le")
    inside = _and(
        nodes,
        _and(nodes, row_ge, row_le, f"{prefix}_rows"),
        _and(nodes, col_ge, col_le, f"{prefix}_cols"),
        f"{prefix}_inside",
    )

    row_start = _node(nodes, "Equal", [row_idx, level_i], f"{prefix}_row_start")
    col_start = _node(nodes, "Equal", [col_idx, level_i], f"{prefix}_col_start")
    row_end = _node(nodes, "Equal", [row_idx, end], f"{prefix}_row_end")
    col_end = _node(nodes, "Equal", [col_idx, end], f"{prefix}_col_end")
    border = _or(
        nodes,
        _or(nodes, row_start, col_start, f"{prefix}_border_a"),
        _or(nodes, row_end, col_end, f"{prefix}_border_b"),
        f"{prefix}_border",
    )
    return _and(nodes, _and(nodes, inside, border, f"{prefix}_ring0"), active, f"{prefix}_ring")


def _ring_index_table() -> np.ndarray:
    table = np.full((MAX_LAYERS, 1, 9, 9), 10, dtype=np.uint8)
    for layer_count in range(1, MAX_LAYERS + 1):
        side = 2 * layer_count - 1
        for r in range(side):
            for c in range(side):
                table[layer_count - 1, 0, r, c] = min(r, c, side - 1 - r, side - 1 - c)
    return table


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    row_line = np.arange(H, dtype=np.int64).reshape(1, 1, H, 1)
    col_line = np.arange(W, dtype=np.int64).reshape(1, 1, 1, W)
    row_idx = _add_init(inits, "row_idx", row_line)
    col_idx = _add_init(inits, "col_idx", col_line)
    zero_i = _add_init(inits, "zero_i", np.zeros((1, 1, 1, 1), dtype=np.int64))
    two_i = _add_init(inits, "two_i", np.full((1, 1, 1, 1), 2, dtype=np.int64))
    twenty_nine_i = _add_init(inits, "twenty_nine_i", np.full((1, 1, 1, 1), 29, dtype=np.int64))
    thirty_i = _add_init(inits, "thirty_i", np.full((1, 1, 1, 1), 30, dtype=np.int64))
    zero_f = _add_init(inits, "zero_f", np.zeros((1, 1, 1, 1), dtype=np.float32))
    true_b = _add_init(inits, "true_b", np.ones((1, 1, 1, 1), dtype=bool))
    level_consts = [
        _add_init(inits, f"level_{idx}", np.full((1, 1, 1, 1), idx, dtype=np.int64))
        for idx in range(MAX_LAYERS)
    ]

    valid_sum = _node(nodes, "ReduceSum", [IN_NAME], "valid_sum", axes=[1], keepdims=1)
    valid = _node(nodes, "Greater", [valid_sum, zero_f], "valid")

    active = true_b
    top = zero_i
    left = zero_i
    bottom = twenty_nine_i
    right = twenty_nine_i
    colors: list[str] = []
    actives: list[str] = []

    for level in range(MAX_LAYERS):
        color = _cell_color(nodes, row_idx, col_idx, top, left, f"l{level}")
        colors.append(color)
        actives.append(active)
        if level < MAX_LAYERS - 1:
            active, top, left, bottom, right = _next_box(
                nodes,
                row_idx,
                col_idx,
                row_idx,
                col_idx,
                valid,
                zero_f,
                zero_i,
                thirty_i,
                active,
                top,
                left,
                bottom,
                right,
                color,
                f"l{level}",
            )

    active_ints = [
        _node(nodes, "Cast", [active_name], f"active_{idx}_i", to=TensorProto.INT64)
        for idx, active_name in enumerate(actives)
    ]
    n_layers = active_ints[0]
    for idx, active_i in enumerate(active_ints[1:], start=1):
        n_layers = _node(nodes, "Add", [n_layers, active_i], f"n_layers_{idx}")

    layers: list[str] = []
    for level, (color, active_name) in enumerate(zip(colors, actives)):
        ring = _ring_mask(
            nodes,
            row_idx,
            col_idx,
            active_name,
            n_layers,
            two_i,
            level_consts[level],
            level,
            f"r{level}",
        )
        ring_f = _node(nodes, "Cast", [ring], f"r{level}_f", to=TensorProto.FLOAT)
        layers.append(_node(nodes, "Mul", [color, ring_f], f"out_layer_{level}"))

    out = layers[0]
    for idx, layer in enumerate(layers[1:], start=1):
        name = OUT_NAME if idx == len(layers) - 1 else f"out_sum_{idx}"
        out = _node(nodes, "Add", [out, layer], name)

    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="neurogolf-task377",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_color_id_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    row_line = np.arange(H, dtype=np.int64).reshape(1, 1, H, 1)
    col_line = np.arange(W, dtype=np.int64).reshape(1, 1, 1, W)
    row_idx = _add_init(inits, "row_idx", row_line)
    col_idx = _add_init(inits, "col_idx", col_line)
    zero_i = _add_init(inits, "zero_i", np.zeros((1, 1, 1, 1), dtype=np.int64))
    two_i = _add_init(inits, "two_i", np.full((1, 1, 1, 1), 2, dtype=np.int64))
    twenty_nine_i = _add_init(inits, "twenty_nine_i", np.full((1, 1, 1, 1), 29, dtype=np.int64))
    thirty_i = _add_init(inits, "thirty_i", np.full((1, 1, 1, 1), 30, dtype=np.int64))
    zero_f = _add_init(inits, "zero_f", np.zeros((1, 1, 1, 1), dtype=np.float32))
    true_b = _add_init(inits, "true_b", np.ones((1, 1, 1, 1), dtype=bool))
    color_lut = _add_init(inits, "color_lut", np.arange(C, dtype=np.int64).reshape(1, C, 1, 1))
    level_consts = [
        _add_init(inits, f"level_{idx}", np.full((1, 1, 1, 1), idx, dtype=np.int64))
        for idx in range(MAX_LAYERS)
    ]

    color_img = _node(nodes, "ArgMax", [IN_NAME], "color_img", axis=1, keepdims=1)
    valid_sum = _node(nodes, "ReduceSum", [IN_NAME], "valid_sum", axes=[1], keepdims=1)
    valid = _node(nodes, "Greater", [valid_sum, zero_f], "valid")

    active = true_b
    top = zero_i
    left = zero_i
    bottom = twenty_nine_i
    right = twenty_nine_i
    colors: list[str] = []
    actives: list[str] = []

    for level in range(MAX_LAYERS):
        color = _cell_color_id(nodes, color_img, row_idx, col_idx, top, left, zero_i, f"i{level}")
        colors.append(color)
        actives.append(active)
        if level < MAX_LAYERS - 1:
            active, top, left, bottom, right = _next_box_id(
                nodes,
                color_img,
                row_idx,
                col_idx,
                valid,
                zero_i,
                thirty_i,
                active,
                top,
                left,
                bottom,
                right,
                color,
                f"i{level}",
            )

    active_ints = [
        _node(nodes, "Cast", [active_name], f"id_active_{idx}_i", to=TensorProto.INT64)
        for idx, active_name in enumerate(actives)
    ]
    n_layers = active_ints[0]
    for idx, active_i in enumerate(active_ints[1:], start=1):
        n_layers = _node(nodes, "Add", [n_layers, active_i], f"id_n_layers_{idx}")

    rings: list[str] = []
    layer_ids: list[str] = []
    for level, (color, active_name) in enumerate(zip(colors, actives)):
        ring = _ring_mask(
            nodes,
            row_idx,
            col_idx,
            active_name,
            n_layers,
            two_i,
            level_consts[level],
            level,
            f"id_r{level}",
        )
        rings.append(ring)
        layer_ids.append(_node(nodes, "Where", [ring, color, zero_i], f"id_layer_{level}"))

    out_ids = layer_ids[0]
    for idx, layer in enumerate(layer_ids[1:], start=1):
        out_ids = _node(nodes, "Add", [out_ids, layer], f"id_sum_{idx}")

    out_valid = rings[0]
    for idx, ring in enumerate(rings[1:], start=1):
        out_valid = _or(nodes, out_valid, ring, f"id_valid_{idx}")
    onehot = _node(nodes, "Equal", [out_ids, color_lut], "id_onehot")
    onehot_valid = _and(nodes, onehot, out_valid, "id_onehot_valid")
    _node(nodes, "Cast", [onehot_valid], OUT_NAME, to=TensorProto.FLOAT)

    graph = helper.make_graph(nodes, f"{TASK_ID}_color_id", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="neurogolf-task377",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_color_i32_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    row_line = np.arange(H, dtype=np.int64).reshape(1, 1, H, 1)
    col_line = np.arange(W, dtype=np.int64).reshape(1, 1, 1, W)
    row_idx = _add_init(inits, "row_idx", row_line)
    col_idx = _add_init(inits, "col_idx", col_line)
    zero_i = _add_init(inits, "zero_i", np.zeros((1, 1, 1, 1), dtype=np.int64))
    zero_i32 = _add_init(inits, "zero_i32", np.zeros((1, 1, 1, 1), dtype=np.int32))
    sentinel_i32 = _add_init(inits, "sentinel_i32", np.full((1, 1, 1, 1), 10, dtype=np.int32))
    two_i = _add_init(inits, "two_i", np.full((1, 1, 1, 1), 2, dtype=np.int64))
    twenty_nine_i = _add_init(inits, "twenty_nine_i", np.full((1, 1, 1, 1), 29, dtype=np.int64))
    thirty_i = _add_init(inits, "thirty_i", np.full((1, 1, 1, 1), 30, dtype=np.int64))
    zero_f = _add_init(inits, "zero_f", np.zeros((1, 1, 1, 1), dtype=np.float32))
    true_b = _add_init(inits, "true_b", np.ones((1, 1, 1, 1), dtype=bool))
    color_lut = _add_init(inits, "color_lut", np.arange(C, dtype=np.int32).reshape(1, C, 1, 1))
    flat_shape = _add_init(inits, "flat_shape", np.asarray([H * W], dtype=np.int64))
    level_consts = [
        _add_init(inits, f"level_{idx}", np.full((1, 1, 1, 1), idx, dtype=np.int64))
        for idx in range(MAX_LAYERS)
    ]

    color_argmax = _node(nodes, "ArgMax", [IN_NAME], "color_argmax", axis=1, keepdims=1)
    color_img = _node(nodes, "Cast", [color_argmax], "color_img_i32", to=TensorProto.INT32)
    flat_colors = _node(nodes, "Reshape", [color_img, flat_shape], "flat_colors")
    valid_sum = _node(nodes, "ReduceSum", [IN_NAME], "valid_sum", axes=[1], keepdims=1)
    valid = _node(nodes, "Greater", [valid_sum, zero_f], "valid")

    active = true_b
    top = zero_i
    left = zero_i
    bottom = twenty_nine_i
    right = twenty_nine_i
    colors: list[str] = []
    actives: list[str] = []

    for level in range(MAX_LAYERS):
        color = _cell_color_gather(nodes, flat_colors, active, top, left, zero_i, thirty_i, f"u{level}")
        colors.append(color)
        actives.append(active)
        if level < MAX_LAYERS - 1:
            active, top, left, bottom, right = _next_box_i32(
                nodes,
                color_img,
                row_idx,
                col_idx,
                valid,
                zero_i32,
                zero_i,
                thirty_i,
                active,
                top,
                left,
                bottom,
                right,
                color,
                f"u{level}",
            )

    active_ints = [
        _node(nodes, "Cast", [active_name], f"u_active_{idx}_i", to=TensorProto.INT64)
        for idx, active_name in enumerate(actives)
    ]
    n_layers = active_ints[0]
    for idx, active_i in enumerate(active_ints[1:], start=1):
        n_layers = _node(nodes, "Add", [n_layers, active_i], f"u_n_layers_{idx}")

    rings: list[str] = []
    out_ids = ""
    for level, (color, active_name) in enumerate(zip(colors, actives)):
        ring = _ring_mask(
            nodes,
            row_idx,
            col_idx,
            active_name,
            n_layers,
            two_i,
            level_consts[level],
            level,
            f"u_r{level}",
        )
        rings.append(ring)
        if level == 0:
            out_ids = _node(nodes, "Where", [ring, color, sentinel_i32], "u_ids_0")
        else:
            out_ids = _node(nodes, "Where", [ring, color, out_ids], f"u_ids_{level}")

    onehot = _node(nodes, "Equal", [out_ids, color_lut], "u_onehot")
    _node(nodes, "Cast", [onehot], OUT_NAME, to=TensorProto.FLOAT)

    graph = helper.make_graph(nodes, f"{TASK_ID}_color_i32", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="neurogolf-task377",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_color_i32_table_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    row_line = np.arange(H, dtype=np.int64).reshape(1, 1, H, 1)
    col_line = np.arange(W, dtype=np.int64).reshape(1, 1, 1, W)
    row_idx = _add_init(inits, "row_idx", row_line)
    col_idx = _add_init(inits, "col_idx", col_line)
    zero_i = _add_init(inits, "zero_i", np.zeros((1, 1, 1, 1), dtype=np.int64))
    one_i = _add_init(inits, "one_i", np.ones((1, 1, 1, 1), dtype=np.int64))
    thirty_i = _add_init(inits, "thirty_i", np.full((1, 1, 1, 1), 30, dtype=np.int64))
    true_b = _add_init(inits, "true_b", np.ones((1, 1, 1, 1), dtype=bool))
    zero_i32 = _add_init(inits, "zero_i32", np.zeros((1, 1, 1, 1), dtype=np.int32))
    sentinel_tail = _add_init(inits, "sentinel_tail", np.full((6,), 10, dtype=np.int32))
    color_lut = _add_init(inits, "color_lut", np.arange(C, dtype=np.int32).reshape(1, C, 1, 1))
    flat_shape = _add_init(inits, "flat_shape", np.asarray([H * W], dtype=np.int64))
    one_shape = _add_init(inits, "one_shape", np.asarray([1], dtype=np.int64))
    ring_table = _add_init(inits, "ring_table", _ring_index_table().astype(np.int32))

    color_argmax = _node(nodes, "ArgMax", [IN_NAME], "v_color_argmax", axis=1, keepdims=1)
    color_img = _node(nodes, "Cast", [color_argmax], "v_color_img_i32", to=TensorProto.INT32)
    flat_colors = _node(nodes, "Reshape", [color_img, flat_shape], "v_flat_colors")

    active = true_b
    top = zero_i
    left = zero_i
    bottom, right = _initial_bounds_from_color_img(nodes, color_img, row_idx, col_idx, zero_i32, zero_i)
    colors: list[str] = []
    actives: list[str] = []

    for level in range(MAX_LAYERS):
        color = _cell_color_gather(nodes, flat_colors, active, top, left, zero_i, thirty_i, f"v{level}")
        colors.append(color)
        actives.append(active)
        if level < MAX_LAYERS - 1:
            active, top, left, bottom, right = _next_box_i32_bounded(
                nodes,
                color_img,
                row_idx,
                col_idx,
                zero_i32,
                zero_i,
                thirty_i,
                active,
                top,
                left,
                bottom,
                right,
                color,
                f"v{level}",
            )

    active_ints = [
        _node(nodes, "Cast", [active_name], f"v_active_{idx}_i", to=TensorProto.INT64)
        for idx, active_name in enumerate(actives)
    ]
    n_layers = active_ints[0]
    for idx, active_i in enumerate(active_ints[1:], start=1):
        n_layers = _node(nodes, "Add", [n_layers, active_i], f"v_n_layers_{idx}")

    n_layers_zero_based = _node(nodes, "Sub", [n_layers, one_i], "v_n_layers_zero_based")
    n_index = _node(nodes, "Reshape", [n_layers_zero_based, one_shape], "v_n_index")
    ring_ids = _node(nodes, "Gather", [ring_table, n_index], "v_ring_ids", axis=0)

    color_vecs = [
        _node(nodes, "Reshape", [color, one_shape], f"v_color_vec_{level}")
        for level, color in enumerate(colors)
    ]
    palette = _node(nodes, "Concat", color_vecs + [sentinel_tail], "v_palette", axis=0)
    out_ids = _node(nodes, "Gather", [palette, ring_ids], "v_ids", axis=0)

    onehot = _node(nodes, "Equal", [out_ids, color_lut], "v_onehot9")
    onehot_f = _node(nodes, "Cast", [onehot], "v_onehot9f", to=TensorProto.FLOAT)
    nodes.append(
        helper.make_node(
            "Pad",
            [onehot_f],
            [OUT_NAME],
            pads=[0, 0, 0, 0, 0, 0, H - 9, W - 9],
            value=0.0,
        )
    )

    graph = helper.make_graph(nodes, f"{TASK_ID}_color_i32_table", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="neurogolf-task377",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _expected_onehot(grid: list[list[int]]) -> np.ndarray:
    return _grid_to_onehot(grid)


def verify_reference() -> None:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            pred = solve(example["input"])
            if pred != example["output"]:
                raise AssertionError(f"reference mismatch on {split}[{idx}]")


def verify_model(model: onnx.ModelProto) -> None:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    sess = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            got = sess.run([OUT_NAME], {IN_NAME: _grid_to_onehot(example["input"])})[0]
            expected = _expected_onehot(example["output"])
            if not np.array_equal(got > 0.0, expected > 0.0):
                raise AssertionError(f"ONNX mismatch on {split}[{idx}]")


def main() -> None:
    verify_reference()

    from score_model import score_file  # noqa: WPS433

    candidates = [
        ("onehot_layers", build_model()),
        ("color_id", build_color_id_model()),
        ("color_i32", build_color_i32_model()),
        ("color_i32_table", build_color_i32_table_model()),
    ]
    results: list[tuple[str, onnx.ModelProto, dict[str, Any]]] = []

    for name, model in candidates:
        verify_model(model)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / BEST_PATH.name
            onnx.save(model, tmp_path)
            score = score_file(tmp_path)
        results.append((name, model, score))
        print(
            name,
            f"valid={score['valid']}",
            f"memory={score['memory']}",
            f"params={score['params']}",
            f"cost={score['cost']}",
            f"score={score['score']}",
        )

    valid = [item for item in results if item[2]["valid"]]
    if not valid:
        raise RuntimeError("no valid candidate models")
    best_name, best_model, best_score = min(valid, key=lambda item: int(item[2]["cost"]))
    onnx.save(best_model, BEST_PATH)
    print(f"saved {BEST_PATH}")
    print(
        "selected:",
        best_name,
        f"memory={best_score['memory']}",
        f"params={best_score['params']}",
        f"cost={best_score['cost']}",
        f"score={best_score['score']}",
    )


if __name__ == "__main__":
    main()
