"""Minimal ONNX for NeuroGolf ARC task086 (ring swap + side arms).

Task rule: each input has one or two axis-aligned square/ring objects on black.
Each object has outer color A (one-cell ring) and inner center color B (1x1 or 2x2).
Output per object: swap A and B inside the original square, then add four arms of
thickness equal to the center span (1 or 2) using the original outer color A.
Two separated objects are split by an all-black row or column inside the global fg bbox.

ONNX: crop to 12x12, ArgMax to uint8 class grid, bool fg + dynamic bbox/separator,
apply the transform twice for split regions when needed, one-hot only at the end.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import (  # noqa: E402
    calculate_memory,
    convert_to_numpy,
    sanitize_model,
    score_file,
)

TASK_ID = "task086"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task086.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

G = 12
C = 10
H = W = 30
PAD = H - G
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.value_infos: list[onnx.ValueInfoProto] = []
        self.counter = 0

    def name(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def init(self, name: str, array: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(array, name))
        return name

    def vi(self, name: str, dtype: int, shape: tuple[int, ...]) -> str:
        self.value_infos.append(helper.make_tensor_value_info(name, dtype, list(shape)))
        return name

    def node(
        self,
        op_type: str,
        inputs: list[str],
        dtype: int,
        shape: tuple[int, ...],
        prefix: str,
        **attrs: object,
    ) -> str:
        out = self.vi(self.name(prefix), dtype, shape)
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out

    def where_bool(
        self,
        cond: str,
        a: str,
        b: str,
        shape: tuple[int, ...],
        prefix: str,
    ) -> str:
        a_f = self.node("Cast", [a], TensorProto.FLOAT, shape, f"{prefix}_af", to=TensorProto.FLOAT)
        b_f = self.node("Cast", [b], TensorProto.FLOAT, shape, f"{prefix}_bf", to=TensorProto.FLOAT)
        w_f = self.node("Where", [cond, a_f, b_f], TensorProto.FLOAT, shape, f"{prefix}_wf")
        return self.node("Greater", [w_f, "zero_f"], TensorProto.BOOL, shape, f"{prefix}_wb")


def add_common_inits(b: Builder, *, grid: int) -> None:
    b.init("zero_f", np.array([0.0], dtype=np.float32))
    b.init("one_i", np.array([1], dtype=np.int64))
    b.init("last_i", np.array([grid - 1], dtype=np.int64))
    b.init("grid_i", np.array([grid], dtype=np.int64))
    b.init("rev", np.arange(grid - 1, -1, -1, dtype=np.int64))
    b.init("rows_i", np.arange(grid, dtype=np.int64).reshape(1, 1, grid, 1))
    b.init("cols_i", np.arange(grid, dtype=np.int64).reshape(1, 1, 1, grid))
    b.init("crop_starts", np.array([0, 0, 0, 0], dtype=np.int64))
    b.init("crop_ends", np.array([1, C, grid, grid], dtype=np.int64))
    b.init("crop_axes", np.array([0, 1, 2, 3], dtype=np.int64))
    b.init("shape_g2", np.array([1, 1, grid, grid], dtype=np.int64))
    b.init("shape_flat", np.array([grid * grid], dtype=np.int64))
    b.init("shape_1", np.array([1], dtype=np.int64))
    b.init("color_ids", np.arange(C, dtype=np.uint8).reshape(1, C, 1, 1))


def bbox_from_mask(
    b: Builder,
    mask_f: str,
    prefix: str,
) -> tuple[str, str, str, str]:
    """Tight bbox on float mask [1,1,G,G] with any positive cell."""
    row_has = b.node("ReduceMax", [mask_f], TensorProto.FLOAT, (1, 1, G, 1), f"{prefix}_row_has", axes=[3], keepdims=1)
    col_has = b.node("ReduceMax", [mask_f], TensorProto.FLOAT, (1, 1, 1, G), f"{prefix}_col_has", axes=[2], keepdims=1)
    rmin = b.node("ArgMax", [row_has], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_rmin", axis=2, keepdims=1)
    cmin = b.node("ArgMax", [col_has], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_cmin", axis=3, keepdims=1)
    row_rev = b.node("Gather", [row_has, "rev"], TensorProto.FLOAT, (1, 1, G, 1), f"{prefix}_row_rev", axis=2)
    col_rev = b.node("Gather", [col_has, "rev"], TensorProto.FLOAT, (1, 1, 1, G), f"{prefix}_col_rev", axis=3)
    rrev = b.node("ArgMax", [row_rev], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_rrev", axis=2, keepdims=1)
    crev = b.node("ArgMax", [col_rev], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_crev", axis=3, keepdims=1)
    rmax = b.node("Sub", ["last_i", rrev], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_rmax")
    cmax = b.node("Sub", ["last_i", crev], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_cmax")
    return rmin, rmax, cmin, cmax


def gather_color(
    b: Builder,
    colors: str,
    row: str,
    col: str,
    prefix: str,
) -> str:
    row_off = b.node("Mul", [row, "grid_i"], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_row_off")
    idx = b.node("Add", [row_off, col], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_idx")
    idx_flat = b.node("Reshape", [idx, "shape_1"], TensorProto.INT64, (1,), f"{prefix}_idx_flat")
    flat = b.node("Reshape", [colors, "shape_flat"], TensorProto.UINT8, (G * G,), f"{prefix}_flat")
    picked = b.node("Gather", [flat, idx_flat], TensorProto.UINT8, (1,), f"{prefix}_picked", axis=0)
    return b.node("Reshape", [picked, "shape_1"], TensorProto.UINT8, (1,), f"{prefix}_color")


def apply_region(
    b: Builder,
    colors: str,
    fg: str,
    region: str,
    prefix: str,
    *,
    require_active: bool = True,
) -> str:
    """Transform one region; inactive cells keep ``colors``."""
    masked_bool = b.node("And", [fg, region], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_masked_bool")
    masked = b.node("Cast", [masked_bool], TensorProto.FLOAT, (1, 1, G, G), f"{prefix}_masked", to=TensorProto.FLOAT)
    if require_active:
        active_f = b.node("ReduceMax", [masked], TensorProto.FLOAT, (1, 1, 1, 1), f"{prefix}_active_f", axes=[2, 3], keepdims=1)
        active = b.node("Greater", [active_f, "zero_f"], TensorProto.BOOL, (1, 1, 1, 1), f"{prefix}_active")

    rmin, rmax, cmin, cmax = bbox_from_mask(b, masked, f"{prefix}_bb")

    rlo = b.node("Add", [rmin, "one_i"], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_rlo")
    clo = b.node("Add", [cmin, "one_i"], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_clo")

    color_a = gather_color(b, colors, rmin, cmin, f"{prefix}_a")
    color_b = gather_color(b, colors, rlo, clo, f"{prefix}_b")

    span = b.node("Sub", [rmax, rmin], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_span")
    arm = b.node("Sub", [span, "one_i"], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_arm")

    rmin_arm = b.node("Sub", [rmin, arm], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_rmin_arm")
    rmax_arm = b.node("Add", [rmax, arm], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_rmax_arm")
    cmin_arm = b.node("Sub", [cmin, arm], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_cmin_arm")
    cmax_arm = b.node("Add", [cmax, arm], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_cmax_arm")

    row_lt_rmin = b.node("Less", ["rows_i", rmin], TensorProto.BOOL, (1, 1, G, 1), f"{prefix}_row_lt_rmin")
    row_ge_rmin = b.node("Not", [row_lt_rmin], TensorProto.BOOL, (1, 1, G, 1), f"{prefix}_row_ge_rmin")
    row_gt_rmax = b.node("Greater", ["rows_i", rmax], TensorProto.BOOL, (1, 1, G, 1), f"{prefix}_row_gt_rmax")
    row_le_rmax = b.node("Not", [row_gt_rmax], TensorProto.BOOL, (1, 1, G, 1), f"{prefix}_row_le_rmax")
    col_lt_cmin = b.node("Less", ["cols_i", cmin], TensorProto.BOOL, (1, 1, 1, G), f"{prefix}_col_lt_cmin")
    col_ge_cmin = b.node("Not", [col_lt_cmin], TensorProto.BOOL, (1, 1, 1, G), f"{prefix}_col_ge_cmin")
    col_gt_cmax = b.node("Greater", ["cols_i", cmax], TensorProto.BOOL, (1, 1, 1, G), f"{prefix}_col_gt_cmax")
    col_le_cmax = b.node("Not", [col_gt_cmax], TensorProto.BOOL, (1, 1, 1, G), f"{prefix}_col_le_cmax")
    in_rc = b.node("And", [row_ge_rmin, row_le_rmax], TensorProto.BOOL, (1, 1, G, 1), f"{prefix}_in_rc")
    in_cc = b.node("And", [col_ge_cmin, col_le_cmax], TensorProto.BOOL, (1, 1, 1, G), f"{prefix}_in_cc")
    in_sq = b.node("And", [in_rc, in_cc], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_in_sq")

    row_lt_rmin_arm = b.node("Less", ["rows_i", rmin_arm], TensorProto.BOOL, (1, 1, G, 1), f"{prefix}_row_lt_rmin_arm")
    row_ge_rmin_arm = b.node("Not", [row_lt_rmin_arm], TensorProto.BOOL, (1, 1, G, 1), f"{prefix}_row_ge_rmin_arm")
    top_arm_rc = b.node("And", [row_ge_rmin_arm, row_lt_rmin], TensorProto.BOOL, (1, 1, G, 1), f"{prefix}_top_arm_rc")
    top_arm = b.node("And", [top_arm_rc, in_cc], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_top_arm")

    row_gt_rmax_arm = b.node("Greater", ["rows_i", rmax_arm], TensorProto.BOOL, (1, 1, G, 1), f"{prefix}_row_gt_rmax_arm")
    row_le_rmax_arm = b.node("Not", [row_gt_rmax_arm], TensorProto.BOOL, (1, 1, G, 1), f"{prefix}_row_le_rmax_arm")
    bot_arm_rc = b.node("And", [row_gt_rmax, row_le_rmax_arm], TensorProto.BOOL, (1, 1, G, 1), f"{prefix}_bot_arm_rc")
    bot_arm = b.node("And", [bot_arm_rc, in_cc], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_bot_arm")

    col_lt_cmin_arm = b.node("Less", ["cols_i", cmin_arm], TensorProto.BOOL, (1, 1, 1, G), f"{prefix}_col_lt_cmin_arm")
    col_ge_cmin_arm = b.node("Not", [col_lt_cmin_arm], TensorProto.BOOL, (1, 1, 1, G), f"{prefix}_col_ge_cmin_arm")
    left_arm_cc = b.node("And", [col_ge_cmin_arm, col_lt_cmin], TensorProto.BOOL, (1, 1, 1, G), f"{prefix}_left_arm_cc")
    left_arm = b.node("And", [in_rc, left_arm_cc], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_left_arm")

    col_gt_cmax_arm = b.node("Greater", ["cols_i", cmax_arm], TensorProto.BOOL, (1, 1, 1, G), f"{prefix}_col_gt_cmax_arm")
    col_le_cmax_arm = b.node("Not", [col_gt_cmax_arm], TensorProto.BOOL, (1, 1, 1, G), f"{prefix}_col_le_cmax_arm")
    right_arm_cc = b.node("And", [col_gt_cmax, col_le_cmax_arm], TensorProto.BOOL, (1, 1, 1, G), f"{prefix}_right_arm_cc")
    right_arm = b.node("And", [in_rc, right_arm_cc], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_right_arm")

    arm_tb = b.node("Or", [top_arm, bot_arm], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_arm_tb")
    arm_lr = b.node("Or", [left_arm, right_arm], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_arm_lr")
    arm_mask = b.node("Or", [arm_tb, arm_lr], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_arm_mask")

    a_i = b.node("Cast", [color_a], TensorProto.INT64, (1,), f"{prefix}_a_i", to=TensorProto.INT64)
    b_i = b.node("Cast", [color_b], TensorProto.INT64, (1,), f"{prefix}_b_i", to=TensorProto.INT64)
    colors_i = b.node("Cast", [colors], TensorProto.INT64, (1, 1, G, G), f"{prefix}_colors_i", to=TensorProto.INT64)

    eq_a = b.node("Equal", [colors_i, a_i], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_eq_a")
    eq_b = b.node("Equal", [colors_i, b_i], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_eq_b")

    swap_a = b.node("And", [in_sq, eq_a], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_swap_a")
    swap_b = b.node("And", [in_sq, eq_b], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_swap_b")

    step1 = b.node("Where", [swap_a, color_b, colors], TensorProto.UINT8, (1, 1, G, G), f"{prefix}_step1")
    step2 = b.node("Where", [swap_b, color_a, step1], TensorProto.UINT8, (1, 1, G, G), f"{prefix}_step2")
    step3 = b.node("Where", [arm_mask, color_a, step2], TensorProto.UINT8, (1, 1, G, G), f"{prefix}_step3")

    touched = b.node("Or", [in_sq, arm_mask], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_touched")
    if require_active:
        use_new = b.node("And", [active, touched], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_use_new")
    else:
        use_new = touched
    return b.node("Where", [use_new, step3, colors], TensorProto.UINT8, (1, 1, G, G), f"{prefix}_out")


def build_model(*, grid: int = G, fg_channels_only: bool = False) -> onnx.ModelProto:
    global G
    G = grid
    b = Builder()
    add_common_inits(b, grid=grid)

    inp = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    out = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    if fg_channels_only:
        b.init("fg_starts", np.array([1], dtype=np.int64))
        b.init("fg_ends", np.array([C], dtype=np.int64))
        b.init("axes_fg", np.array([1], dtype=np.int64))

        crop10 = b.node(
            "Slice",
            [IN_NAME, "crop_starts", "crop_ends", "crop_axes"],
            TensorProto.FLOAT,
            (1, C, grid, grid),
            "crop10",
        )
        in_any_f = b.node("ReduceMax", [crop10], TensorProto.FLOAT, (1, 1, grid, grid), "in_any_f", axes=[1], keepdims=1)
        _, rmax_in, _, cmax_in = bbox_from_mask(b, in_any_f, "canvas")
        row_gt_rmax_in = b.node("Greater", ["rows_i", rmax_in], TensorProto.BOOL, (1, 1, grid, 1), "row_gt_rmax_in")
        col_gt_cmax_in = b.node("Greater", ["cols_i", cmax_in], TensorProto.BOOL, (1, 1, 1, grid), "col_gt_cmax_in")
        row_in_canvas = b.node("Not", [row_gt_rmax_in], TensorProto.BOOL, (1, 1, grid, 1), "row_in_canvas")
        col_in_canvas = b.node("Not", [col_gt_cmax_in], TensorProto.BOOL, (1, 1, 1, grid), "col_in_canvas")
        canvas = b.node("And", [row_in_canvas, col_in_canvas], TensorProto.BOOL, (1, 1, grid, grid), "canvas")

        crop = b.node(
            "Slice",
            [crop10, "fg_starts", "fg_ends", "axes_fg"],
            TensorProto.FLOAT,
            (1, C - 1, grid, grid),
            "crop_fg",
        )
        colors_i0 = b.node("ArgMax", [crop], TensorProto.INT64, (1, 1, grid, grid), "colors_i0", axis=1, keepdims=1)
        colors_1b = b.node("Add", [colors_i0, "one_i"], TensorProto.INT64, (1, 1, grid, grid), "colors_1b")
        has_fg = b.node("Greater", [crop, "zero_f"], TensorProto.BOOL, (1, C - 1, grid, grid), "has_fg")
        has_fg_f = b.node("Cast", [has_fg], TensorProto.FLOAT, (1, C - 1, grid, grid), "has_fg_f", to=TensorProto.FLOAT)
        any_fg_f = b.node("ReduceMax", [has_fg_f], TensorProto.FLOAT, (1, 1, grid, grid), "any_fg_f", axes=[1], keepdims=1)
        any_fg = b.node("Greater", [any_fg_f, "zero_f"], TensorProto.BOOL, (1, 1, grid, grid), "any_fg")
        any_i = b.node("Cast", [any_fg], TensorProto.INT64, (1, 1, grid, grid), "any_i", to=TensorProto.INT64)
        colors_i = b.node("Mul", [colors_1b, any_i], TensorProto.INT64, (1, 1, grid, grid), "colors_i")
        colors = b.node("Cast", [colors_i], TensorProto.UINT8, (1, 1, grid, grid), "colors", to=TensorProto.UINT8)
    else:
        crop = b.node(
            "Slice",
            [IN_NAME, "crop_starts", "crop_ends", "crop_axes"],
            TensorProto.FLOAT,
            (1, C, grid, grid),
            "crop",
        )
        in_any_f = b.node("ReduceMax", [crop], TensorProto.FLOAT, (1, 1, grid, grid), "in_any_f", axes=[1], keepdims=1)
        _, rmax_in, _, cmax_in = bbox_from_mask(b, in_any_f, "canvas")
        row_gt_rmax_in = b.node("Greater", ["rows_i", rmax_in], TensorProto.BOOL, (1, 1, grid, 1), "row_gt_rmax_in")
        col_gt_cmax_in = b.node("Greater", ["cols_i", cmax_in], TensorProto.BOOL, (1, 1, 1, grid), "col_gt_cmax_in")
        row_in_canvas = b.node("Not", [row_gt_rmax_in], TensorProto.BOOL, (1, 1, grid, 1), "row_in_canvas")
        col_in_canvas = b.node("Not", [col_gt_cmax_in], TensorProto.BOOL, (1, 1, 1, grid), "col_in_canvas")
        canvas = b.node("And", [row_in_canvas, col_in_canvas], TensorProto.BOOL, (1, 1, grid, grid), "canvas")

        colors_i = b.node("ArgMax", [crop], TensorProto.INT64, (1, grid, grid), "colors_i", axis=1, keepdims=0)
        colors_r = b.node("Reshape", [colors_i, "shape_g2"], TensorProto.INT64, (1, 1, grid, grid), "colors_r")
        colors = b.node("Cast", [colors_r], TensorProto.UINT8, (1, 1, grid, grid), "colors", to=TensorProto.UINT8)

    b.init("zero_u8", np.array([0], dtype=np.uint8))
    fg = b.node("Greater", [colors, "zero_u8"], TensorProto.BOOL, (1, 1, grid, grid), "fg")
    fg_f = b.node("Cast", [fg], TensorProto.FLOAT, (1, 1, grid, grid), "fg_f", to=TensorProto.FLOAT)

    rmin, rmax, cmin, cmax = bbox_from_mask(b, fg_f, "glob")

    row_sum = b.node("ReduceSum", [fg_f], TensorProto.FLOAT, (1, 1, grid, 1), "row_sum", axes=[3], keepdims=1)
    col_sum = b.node("ReduceSum", [fg_f], TensorProto.FLOAT, (1, 1, 1, grid), "col_sum", axes=[2], keepdims=1)

    row_has = b.node("Greater", [row_sum, "zero_f"], TensorProto.BOOL, (1, 1, grid, 1), "row_has")
    col_has = b.node("Greater", [col_sum, "zero_f"], TensorProto.BOOL, (1, 1, 1, grid), "col_has")
    row_empty = b.node("Not", [row_has], TensorProto.BOOL, (1, 1, grid, 1), "row_empty")
    col_empty = b.node("Not", [col_has], TensorProto.BOOL, (1, 1, 1, grid), "col_empty")

    row_gt_rmin = b.node("Greater", ["rows_i", rmin], TensorProto.BOOL, (1, 1, grid, 1), "row_gt_rmin")
    row_lt_rmax = b.node("Less", ["rows_i", rmax], TensorProto.BOOL, (1, 1, grid, 1), "row_lt_rmax")
    row_interior = b.node("And", [row_gt_rmin, row_lt_rmax], TensorProto.BOOL, (1, 1, grid, 1), "row_interior")
    row_sep_cand = b.node("And", [row_empty, row_interior], TensorProto.BOOL, (1, 1, grid, 1), "row_sep_cand")

    col_gt_cmin = b.node("Greater", ["cols_i", cmin], TensorProto.BOOL, (1, 1, 1, grid), "col_gt_cmin")
    col_lt_cmax = b.node("Less", ["cols_i", cmax], TensorProto.BOOL, (1, 1, 1, grid), "col_lt_cmax")
    col_interior = b.node("And", [col_gt_cmin, col_lt_cmax], TensorProto.BOOL, (1, 1, 1, grid), "col_interior")
    col_sep_cand = b.node("And", [col_empty, col_interior], TensorProto.BOOL, (1, 1, 1, grid), "col_sep_cand")

    row_sep_f = b.node("Cast", [row_sep_cand], TensorProto.FLOAT, (1, 1, grid, 1), "row_sep_f", to=TensorProto.FLOAT)
    has_row_sep_f = b.node("ReduceMax", [row_sep_f], TensorProto.FLOAT, (1, 1, 1, 1), "has_row_sep_f", axes=[2], keepdims=1)
    has_row_sep = b.node("Greater", [has_row_sep_f, "zero_f"], TensorProto.BOOL, (1, 1, 1, 1), "has_row_sep")

    sep_row = b.node("ArgMax", [row_sep_f], TensorProto.INT64, (1, 1, 1, 1), "sep_row", axis=2, keepdims=1)

    col_sep_f = b.node("Cast", [col_sep_cand], TensorProto.FLOAT, (1, 1, 1, grid), "col_sep_f", to=TensorProto.FLOAT)
    has_col_sep_f = b.node("ReduceMax", [col_sep_f], TensorProto.FLOAT, (1, 1, 1, 1), "has_col_sep_f", axes=[3], keepdims=1)
    has_col_sep = b.node("Greater", [has_col_sep_f, "zero_f"], TensorProto.BOOL, (1, 1, 1, 1), "has_col_sep")

    sep_col = b.node("ArgMax", [col_sep_f], TensorProto.INT64, (1, 1, 1, 1), "sep_col", axis=3, keepdims=1)

    row_lt_sep = b.node("Less", ["rows_i", sep_row], TensorProto.BOOL, (1, 1, grid, 1), "row_lt_sep")
    row_gt_sep = b.node("Greater", ["rows_i", sep_row], TensorProto.BOOL, (1, 1, grid, 1), "row_gt_sep")
    col_lt_sep = b.node("Less", ["cols_i", sep_col], TensorProto.BOOL, (1, 1, 1, grid), "col_lt_sep")
    col_gt_sep = b.node("Greater", ["cols_i", sep_col], TensorProto.BOOL, (1, 1, 1, grid), "col_gt_sep")

    all_rows = b.init("all_true_rows", np.ones((1, 1, grid, 1), dtype=np.bool_))
    all_cols = b.init("all_true_cols", np.ones((1, 1, 1, grid), dtype=np.bool_))
    false_rows = b.init("false_rows", np.zeros((1, 1, grid, 1), dtype=np.bool_))
    false_cols = b.init("false_cols", np.zeros((1, 1, 1, grid), dtype=np.bool_))

    not_row_sep = b.node("Not", [has_row_sep], TensorProto.BOOL, (1, 1, 1, 1), "not_row_sep")
    use_col = b.node("And", [not_row_sep, has_col_sep], TensorProto.BOOL, (1, 1, 1, 1), "use_col")

    region1_rows_row = b.where_bool(has_row_sep, row_lt_sep, all_rows, (1, 1, grid, 1), "region1_rows_row")
    region2_rows_row = b.where_bool(has_row_sep, row_gt_sep, false_rows, (1, 1, grid, 1), "region2_rows_row")

    region1_cols_col = b.where_bool(use_col, col_lt_sep, all_cols, (1, 1, 1, grid), "region1_cols_col")
    region2_cols_col = b.where_bool(use_col, col_gt_sep, false_cols, (1, 1, 1, grid), "region2_cols_col")

    region1_rows = b.where_bool(use_col, all_rows, region1_rows_row, (1, 1, grid, 1), "region1_rows")
    region2_rows = b.where_bool(use_col, all_rows, region2_rows_row, (1, 1, grid, 1), "region2_rows")
    region1_cols = b.where_bool(has_row_sep, all_cols, region1_cols_col, (1, 1, 1, grid), "region1_cols")
    region2_cols = b.where_bool(has_row_sep, all_cols, region2_cols_col, (1, 1, 1, grid), "region2_cols")

    region1 = b.node("And", [region1_rows, region1_cols], TensorProto.BOOL, (1, 1, grid, grid), "region1")
    region2 = b.node("And", [region2_rows, region2_cols], TensorProto.BOOL, (1, 1, grid, grid), "region2")

    out1 = apply_region(b, colors, fg, region1, "r1", require_active=False)
    out2 = apply_region(b, out1, fg, region2, "r2")

    color_ids_i = b.node("Cast", ["color_ids"], TensorProto.INT64, (1, C, 1, 1), "color_ids_i", to=TensorProto.INT64)
    out2_i = b.node("Cast", [out2], TensorProto.INT64, (1, 1, grid, grid), "out2_i", to=TensorProto.INT64)
    raw_bool = b.node("Equal", [color_ids_i, out2_i], TensorProto.BOOL, (1, C, grid, grid), "raw_bool")
    solved_bool = b.node("And", [raw_bool, canvas], TensorProto.BOOL, (1, C, grid, grid), "solved_bool")
    final_float = b.node("Cast", [solved_bool], TensorProto.FLOAT, (1, C, grid, grid), "final_float", to=TensorProto.FLOAT)
    b.nodes.append(
        helper.make_node(
            "Pad",
            [final_float],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, PAD, PAD],
            value=0.0,
        )
    )

    graph = helper.make_graph(b.nodes, TASK_ID, [inp], [out], b.initializers, value_info=b.value_infos)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    return model


def _load_examples() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _check_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]], list[str]]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return False, {}, ["sanitize failed"]
    session = ort.InferenceSession(sanitized.SerializeToString(), providers=["CPUExecutionProvider"])
    task = _load_examples()
    counts: dict[str, tuple[int, int]] = {}
    failures: list[str] = []
    for split in ("train", "test", "arc-gen"):
        ok_count = 0
        examples = task.get(split, [])
        for idx, example in enumerate(examples):
            x = convert_to_numpy(example, "input")
            y = convert_to_numpy(example, "output")
            if x is None or y is None:
                continue
            pred = session.run([OUT_NAME], {IN_NAME: x})[0]
            if not np.array_equal(pred > 0.0, y > 0.0):
                failures.append(f"{split}[{idx}]")
                return False, {**counts, split: (ok_count, len(examples))}, failures
            ok_count += 1
        counts[split] = (ok_count, len(examples))
    return True, counts, failures


def _profile_largest_internal(model: onnx.ModelProto) -> tuple[int | None, str | None, int | None]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return None, None, None
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(Path(tempfile.gettempdir()) / f"ng_{TASK_ID}")
    session = ort.InferenceSession(sanitized.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    for split_examples in _load_examples().values():
        for example in split_examples:
            x = convert_to_numpy(example, "input")
            if x is not None:
                session.run([OUT_NAME], {IN_NAME: x})
    trace_path = session.end_profiling()
    memory = calculate_memory(sanitized, trace_path)
    graph = onnx.shape_inference.infer_shapes(sanitized, strict_mode=True).graph
    largest_name: str | None = None
    largest_bytes = -1
    for value in graph.value_info:
        if value.name == OUT_NAME or not value.type.HasField("tensor_type"):
            continue
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        n = 1
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                n *= dim.dim_value
        itemsize = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)).itemsize
        size = int(n * itemsize)
        if size > largest_bytes:
            largest_name = value.name
            largest_bytes = size
    return memory, largest_name, largest_bytes


def _format_counts(counts: dict[str, tuple[int, int]]) -> str:
    return ", ".join(f"{split}={ok}/{total}" for split, (ok, total) in counts.items())


def main() -> None:
    builders: list[tuple[str, Callable[[], onnx.ModelProto]]] = [
        ("argmax12", lambda: build_model(grid=12, fg_channels_only=False)),
    ]
    results: list[tuple[int, str, onnx.ModelProto, dict, str | None, int | None, dict[str, tuple[int, int]]]] = []
    for label, builder in builders:
        model = builder()
        tmp_path = OUT_DIR / f"{TASK_ID}_{label}.onnx"
        onnx.save(model, tmp_path)
        correct, counts, failures = _check_correct(model)
        scored = score_file(tmp_path)
        memory, largest_name, largest_bytes = _profile_largest_internal(model)
        score_text = f"{scored['score']:.6f}" if scored.get("score") is not None else "INVALID"
        print(
            f"{label:<12} correct={correct} ({_format_counts(counts)}) "
            f"memory={scored['memory']} params={scored['params']} cost={scored['cost']} "
            f"score={score_text} largest={largest_name}:{largest_bytes} "
            f"fail={failures[:3]}"
        )
        if correct and scored["valid"]:
            results.append((int(scored["cost"]), label, model, scored, largest_name, largest_bytes, counts))
        tmp_path.unlink(missing_ok=True)

    if not results:
        raise SystemExit("no valid correct variants")

    _, label, model, scored, largest_name, largest_bytes, counts = min(results, key=lambda item: item[0])
    onnx.save(model, BEST_PATH)
    print()
    print(f"kept:    {label}")
    print(f"wrote:   {BEST_PATH}")
    print(f"passes:  {_format_counts(counts)}")
    print(f"memory:  {scored['memory']}")
    print(f"params:  {scored['params']}")
    print(f"cost:    {scored['cost']}")
    print(f"score:   {scored['score']:.6f}")
    print(f"largest: {largest_name} ({largest_bytes} bytes)")


if __name__ == "__main__":
    main()
