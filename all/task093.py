"""Minimal ONNX for NeuroGolf ARC task093 (marker projection onto gray band).

Task rule (14x14 crop, padded to 30x30 one-hot I/O):
- One gray (color 5) horizontal or vertical thick band spans the grid.
- Same-color marker pixels lie off the band.
- Output keeps the band, removes markers, and fills gray extension cells
  adjacent to the band toward each marker:
  * horizontal band: in column c with n markers above (below) the band,
    rows top-1..top-n (bottom+1..bottom+n) become gray;
  * vertical band: in row r with n markers left (right) of the band,
    cols left-1..left-n (right+1..right+n) become gray.
- Orientation: horizontal when max gray cells per row > max gray cells per column.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import (  # noqa: E402
    calculate_memory,
    calculate_params,
    convert_to_numpy,
    load_task_examples,
    sanitize_model,
    score_file,
)

TASK_ID = "task093"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task093.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

G = 14
C = 10
H = W = 30
PAD = H - G
GRAY = 5
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.value_infos: list[onnx.ValueInfoProto] = []
        self.false_14: str | None = None
        self.false_14_f: str | None = None
        self.last_where_float: str | None = None
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
        self.last_where_float = w_f
        return self.node("Greater", [w_f, "zero_f"], TensorProto.BOOL, shape, f"{prefix}_wb")

    def false14(self) -> str:
        if self.false_14 is None:
            self.false_14 = self.init("false14", np.zeros((1, 1, G, G), dtype=np.bool_))
        return self.false_14

    def false14f(self) -> str:
        if self.false_14_f is None:
            self.false_14_f = self.init("false14f", np.zeros((1, 1, G, G), dtype=np.float32))
        return self.false_14_f


def add_common_inits(
    b: Builder,
    *,
    from_crop: bool,
    onehot_equal: bool = True,
    marker_from_bg: bool = False,
) -> None:
    b.init("zero_f", np.array([0.0], dtype=np.float32))
    b.init("last_i", np.array([G - 1], dtype=np.int64))
    b.init("rev", np.arange(G - 1, -1, -1, dtype=np.int64))
    b.init("rows_i", np.arange(G, dtype=np.int64).reshape(1, 1, G, 1))
    b.init("cols_i", np.arange(G, dtype=np.int64).reshape(1, 1, 1, G))
    b.init("crop_axes", np.array([0, 1, 2, 3], dtype=np.int64))
    if onehot_equal:
        b.init("five_f", np.array([5.0], dtype=np.float32))
        b.init("color_ids", np.arange(C, dtype=np.int64).reshape(1, C, 1, 1))
    if from_crop:
        b.init("axes_c", np.array([1], dtype=np.int64))
        b.init("crop_starts", np.array([0, 0, 0, 0], dtype=np.int64))
        b.init("crop_ends", np.array([1, C, G, G], dtype=np.int64))
        b.init("gray_starts", np.array([GRAY], dtype=np.int64))
        b.init("gray_ends", np.array([GRAY + 1], dtype=np.int64))
        b.init("m1_starts", np.array([1], dtype=np.int64))
        b.init("m1_ends", np.array([5], dtype=np.int64))
        b.init("m2_starts", np.array([6], dtype=np.int64))
        b.init("m2_ends", np.array([C], dtype=np.int64))
    elif marker_from_bg:
        b.init("gray_in_starts", np.array([0, GRAY, 0, 0], dtype=np.int64))
        b.init("gray_in_ends", np.array([1, GRAY + 1, G, G], dtype=np.int64))
        b.init("bg_in_starts", np.array([0, 0, 0, 0], dtype=np.int64))
        b.init("bg_in_ends", np.array([1, 1, G, G], dtype=np.int64))
    else:
        b.init("gray_in_starts", np.array([0, GRAY, 0, 0], dtype=np.int64))
        b.init("gray_in_ends", np.array([1, GRAY + 1, G, G], dtype=np.int64))
        b.init("m1_in_starts", np.array([0, 1, 0, 0], dtype=np.int64))
        b.init("m1_in_ends", np.array([1, 5, G, G], dtype=np.int64))
        b.init("m2_in_starts", np.array([0, 6, 0, 0], dtype=np.int64))
        b.init("m2_in_ends", np.array([1, C, G, G], dtype=np.int64))


def bbox_edges(
    b: Builder,
    mask_f: str,
    prefix: str,
) -> tuple[str, str, str, str]:
    row_has = b.node("ReduceMax", [mask_f], TensorProto.FLOAT, (1, 1, G, 1), f"{prefix}_row_has", axes=[3], keepdims=1)
    col_has = b.node("ReduceMax", [mask_f], TensorProto.FLOAT, (1, 1, 1, G), f"{prefix}_col_has", axes=[2], keepdims=1)
    top = b.node("ArgMax", [row_has], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_top", axis=2, keepdims=1)
    left = b.node("ArgMax", [col_has], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_left", axis=3, keepdims=1)
    row_rev = b.node("Gather", [row_has, "rev"], TensorProto.FLOAT, (1, 1, G, 1), f"{prefix}_row_rev", axis=2)
    col_rev = b.node("Gather", [col_has, "rev"], TensorProto.FLOAT, (1, 1, 1, G), f"{prefix}_col_rev", axis=3)
    rrev = b.node("ArgMax", [row_rev], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_rrev", axis=2, keepdims=1)
    crev = b.node("ArgMax", [col_rev], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_crev", axis=3, keepdims=1)
    bottom = b.node("Sub", ["last_i", rrev], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_bottom")
    right = b.node("Sub", ["last_i", crev], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_right")
    return top, bottom, left, right


def bbox_edges_from_counts(
    b: Builder,
    row_count_f: str,
    col_count_f: str,
    prefix: str,
) -> tuple[str, str, str, str]:
    top = b.node("ArgMax", [row_count_f], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_top", axis=2, keepdims=1)
    left = b.node("ArgMax", [col_count_f], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_left", axis=3, keepdims=1)
    row_rev = b.node("Gather", [row_count_f, "rev"], TensorProto.FLOAT, (1, 1, G, 1), f"{prefix}_row_rev", axis=2)
    col_rev = b.node("Gather", [col_count_f, "rev"], TensorProto.FLOAT, (1, 1, 1, G), f"{prefix}_col_rev", axis=3)
    rrev = b.node("ArgMax", [row_rev], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_rrev", axis=2, keepdims=1)
    crev = b.node("ArgMax", [col_rev], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_crev", axis=3, keepdims=1)
    bottom = b.node("Sub", ["last_i", rrev], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_bottom")
    right = b.node("Sub", ["last_i", crev], TensorProto.INT64, (1, 1, 1, 1), f"{prefix}_right")
    return top, bottom, left, right


def ge_float(b: Builder, a: str, b_name: str, shape: tuple[int, ...], prefix: str) -> str:
    lt = b.node("Less", [a, b_name], TensorProto.BOOL, shape, f"{prefix}_lt")
    return b.node("Not", [lt], TensorProto.BOOL, shape, f"{prefix}_ge")


def _marker_and_gray(
    b: Builder,
    *,
    from_crop: bool,
    marker_from_bg: bool = False,
) -> tuple[str, str]:
    if from_crop:
        crop = b.node(
            "Slice",
            [IN_NAME, "crop_starts", "crop_ends", "crop_axes"],
            TensorProto.FLOAT,
            (1, C, G, G),
            "crop",
        )
        gray_f = b.node(
            "Slice",
            [crop, "gray_starts", "gray_ends", "axes_c"],
            TensorProto.FLOAT,
            (1, 1, G, G),
            "gray_f",
        )
        m1 = b.node("Slice", [crop, "m1_starts", "m1_ends", "axes_c"], TensorProto.FLOAT, (1, 4, G, G), "m1")
        m2 = b.node("Slice", [crop, "m2_starts", "m2_ends", "axes_c"], TensorProto.FLOAT, (1, 4, G, G), "m2")
    elif marker_from_bg:
        gray_f = b.node(
            "Slice",
            [IN_NAME, "gray_in_starts", "gray_in_ends", "crop_axes"],
            TensorProto.FLOAT,
            (1, 1, G, G),
            "gray_f",
        )
    else:
        gray_f = b.node(
            "Slice",
            [IN_NAME, "gray_in_starts", "gray_in_ends", "crop_axes"],
            TensorProto.FLOAT,
            (1, 1, G, G),
            "gray_f",
        )
        m1 = b.node(
            "Slice",
            [IN_NAME, "m1_in_starts", "m1_in_ends", "crop_axes"],
            TensorProto.FLOAT,
            (1, 4, G, G),
            "m1",
        )
        m2 = b.node(
            "Slice",
            [IN_NAME, "m2_in_starts", "m2_in_ends", "crop_axes"],
            TensorProto.FLOAT,
            (1, 4, G, G),
            "m2",
        )

    g = b.node("Greater", [gray_f, "zero_f"], TensorProto.BOOL, (1, 1, G, G), "g")
    if marker_from_bg:
        bg_f = b.node(
            "Slice",
            [IN_NAME, "bg_in_starts", "bg_in_ends", "crop_axes"],
            TensorProto.FLOAT,
            (1, 1, G, G),
            "bg_f",
        )
        bg = b.node("Greater", [bg_f, "zero_f"], TensorProto.BOOL, (1, 1, G, G), "bg")
        not_bg = b.node("Not", [bg], TensorProto.BOOL, (1, 1, G, G), "not_bg")
        not_g = b.node("Not", [g], TensorProto.BOOL, (1, 1, G, G), "not_g")
        m = b.node("And", [not_bg, not_g], TensorProto.BOOL, (1, 1, G, G), "m")
        return gray_f, g, m

    m1_any = b.node("ReduceMax", [m1], TensorProto.FLOAT, (1, 1, G, G), "m1_any", axes=[1], keepdims=1)
    m2_any = b.node("ReduceMax", [m2], TensorProto.FLOAT, (1, 1, G, G), "m2_any", axes=[1], keepdims=1)
    m1_b = b.node("Greater", [m1_any, "zero_f"], TensorProto.BOOL, (1, 1, G, G), "m1_b")
    m2_b = b.node("Greater", [m2_any, "zero_f"], TensorProto.BOOL, (1, 1, G, G), "m2_b")
    m = b.node("Or", [m1_b, m2_b], TensorProto.BOOL, (1, 1, G, G), "m")
    return gray_f, g, m


def build_core(b: Builder, *, from_crop: bool = True, marker_from_bg: bool = False) -> str:
    """Return bool gray output [1,1,G,G]."""
    gray_f, g, m = _marker_and_gray(b, from_crop=from_crop, marker_from_bg=marker_from_bg)

    row_count_f = b.node("ReduceSum", [gray_f], TensorProto.FLOAT, (1, 1, G, 1), "row_count_f", axes=[3], keepdims=1)
    col_count_f = b.node("ReduceSum", [gray_f], TensorProto.FLOAT, (1, 1, 1, G), "col_count_f", axes=[2], keepdims=1)
    top, bottom, left, right = bbox_edges_from_counts(b, row_count_f, col_count_f, "bb")
    max_row = b.node("ReduceMax", [row_count_f], TensorProto.FLOAT, (1, 1, 1, 1), "max_row", axes=[2], keepdims=1)
    max_col = b.node("ReduceMax", [col_count_f], TensorProto.FLOAT, (1, 1, 1, 1), "max_col", axes=[3], keepdims=1)
    horizontal = b.node("Greater", [max_row, max_col], TensorProto.BOOL, (1, 1, 1, 1), "horizontal")

    row_lt_top = b.node("Less", ["rows_i", top], TensorProto.BOOL, (1, 1, G, 1), "row_lt_top")
    row_gt_bot = b.node("Greater", ["rows_i", bottom], TensorProto.BOOL, (1, 1, G, 1), "row_gt_bot")
    above_m = b.node("And", [m, row_lt_top], TensorProto.BOOL, (1, 1, G, G), "above_m")
    below_m = b.node("And", [m, row_gt_bot], TensorProto.BOOL, (1, 1, G, G), "below_m")
    above_m_f = b.node("Cast", [above_m], TensorProto.FLOAT, (1, 1, G, G), "above_m_f", to=TensorProto.FLOAT)
    below_m_f = b.node("Cast", [below_m], TensorProto.FLOAT, (1, 1, G, G), "below_m_f", to=TensorProto.FLOAT)
    col_above_f = b.node("ReduceSum", [above_m_f], TensorProto.FLOAT, (1, 1, 1, G), "col_above_f", axes=[2], keepdims=1)
    col_below_f = b.node("ReduceSum", [below_m_f], TensorProto.FLOAT, (1, 1, 1, G), "col_below_f", axes=[2], keepdims=1)

    need_above_i = b.node("Sub", [top, "rows_i"], TensorProto.INT64, (1, 1, G, 1), "need_above_i")
    need_below_i = b.node("Sub", ["rows_i", bottom], TensorProto.INT64, (1, 1, G, 1), "need_below_i")
    need_above_f = b.node("Cast", [need_above_i], TensorProto.FLOAT, (1, 1, G, 1), "need_above_f", to=TensorProto.FLOAT)
    need_below_f = b.node("Cast", [need_below_i], TensorProto.FLOAT, (1, 1, G, 1), "need_below_f", to=TensorProto.FLOAT)

    above_ok = ge_float(b, col_above_f, need_above_f, (1, 1, G, G), "above_ok")
    below_ok = ge_float(b, col_below_f, need_below_f, (1, 1, G, G), "below_ok")
    above_zone = b.node("And", [row_lt_top, above_ok], TensorProto.BOOL, (1, 1, G, G), "above_zone")
    below_zone = b.node("And", [row_gt_bot, below_ok], TensorProto.BOOL, (1, 1, G, G), "below_zone")
    h_proj = b.node("Or", [above_zone, below_zone], TensorProto.BOOL, (1, 1, G, G), "h_proj")
    gray_h = b.node("Or", [g, h_proj], TensorProto.BOOL, (1, 1, G, G), "gray_h")

    col_lt_left = b.node("Less", ["cols_i", left], TensorProto.BOOL, (1, 1, 1, G), "col_lt_left")
    col_gt_right = b.node("Greater", ["cols_i", right], TensorProto.BOOL, (1, 1, 1, G), "col_gt_right")
    left_m = b.node("And", [m, col_lt_left], TensorProto.BOOL, (1, 1, G, G), "left_m")
    right_m = b.node("And", [m, col_gt_right], TensorProto.BOOL, (1, 1, G, G), "right_m")
    left_m_f = b.node("Cast", [left_m], TensorProto.FLOAT, (1, 1, G, G), "left_m_f", to=TensorProto.FLOAT)
    right_m_f = b.node("Cast", [right_m], TensorProto.FLOAT, (1, 1, G, G), "right_m_f", to=TensorProto.FLOAT)
    row_left_f = b.node("ReduceSum", [left_m_f], TensorProto.FLOAT, (1, 1, G, 1), "row_left_f", axes=[3], keepdims=1)
    row_right_f = b.node("ReduceSum", [right_m_f], TensorProto.FLOAT, (1, 1, G, 1), "row_right_f", axes=[3], keepdims=1)

    need_left_i = b.node("Sub", [left, "cols_i"], TensorProto.INT64, (1, 1, 1, G), "need_left_i")
    need_right_i = b.node("Sub", ["cols_i", right], TensorProto.INT64, (1, 1, 1, G), "need_right_i")
    need_left_f = b.node("Cast", [need_left_i], TensorProto.FLOAT, (1, 1, 1, G), "need_left_f", to=TensorProto.FLOAT)
    need_right_f = b.node("Cast", [need_right_i], TensorProto.FLOAT, (1, 1, 1, G), "need_right_f", to=TensorProto.FLOAT)

    left_ok = ge_float(b, row_left_f, need_left_f, (1, 1, G, G), "left_ok")
    right_ok = ge_float(b, row_right_f, need_right_f, (1, 1, G, G), "right_ok")
    left_zone = b.node("And", [col_lt_left, left_ok], TensorProto.BOOL, (1, 1, G, G), "left_zone")
    right_zone = b.node("And", [col_gt_right, right_ok], TensorProto.BOOL, (1, 1, G, G), "right_zone")
    v_proj = b.node("Or", [left_zone, right_zone], TensorProto.BOOL, (1, 1, G, G), "v_proj")
    gray_v = b.node("Or", [g, v_proj], TensorProto.BOOL, (1, 1, G, G), "gray_v")

    return b.where_bool(horizontal, gray_h, gray_v, (1, 1, G, G), "gray_out")


def build_model_onehot(*, opset: int, ir_version: int, from_crop: bool = True) -> onnx.ModelProto:
    b = Builder()
    add_common_inits(b, from_crop=from_crop)

    inp = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    out = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    gray_out = build_core(b, from_crop=from_crop)

    gray_f = b.node("Cast", [gray_out], TensorProto.FLOAT, (1, 1, G, G), "gray_f", to=TensorProto.FLOAT)
    gray5_f = b.node("Mul", [gray_f, "five_f"], TensorProto.FLOAT, (1, 1, G, G), "gray5_f")
    gray_i = b.node("Cast", [gray5_f], TensorProto.INT64, (1, 1, G, G), "gray_i", to=TensorProto.INT64)
    one_hot = b.node("Equal", ["color_ids", gray_i], TensorProto.BOOL, (1, C, G, G), "one_hot")
    final_float = b.node("Cast", [one_hot], TensorProto.FLOAT, (1, C, G, G), "final_float", to=TensorProto.FLOAT)

    if opset >= 11:
        b.init("pad_tensor", np.array([0, 0, 0, 0, 0, 0, PAD, PAD], dtype=np.int64))
        b.nodes.append(
            helper.make_node("Pad", [final_float, "pad_tensor", "zero_f"], [OUT_NAME], mode="constant")
        )
    else:
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
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = ir_version
    onnx.checker.check_model(model)
    return model


def build_model_two_planes(*, from_crop: bool = False) -> onnx.ModelProto:
    b = Builder()
    add_common_inits(b, from_crop=from_crop, onehot_equal=False)

    inp = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    out = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    gray_out = build_core(b, from_crop=from_crop)
    bg_out = b.node("Not", [gray_out], TensorProto.BOOL, (1, 1, G, G), "bg_out")
    false14 = b.false14()
    one_hot = b.node(
        "Concat",
        [bg_out, false14, false14, false14, false14, gray_out, false14, false14, false14, false14],
        TensorProto.BOOL,
        (1, C, G, G),
        "one_hot",
        axis=1,
    )
    final_float = b.node("Cast", [one_hot], TensorProto.FLOAT, (1, C, G, G), "final_float", to=TensorProto.FLOAT)
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
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def build_model_bg_marker() -> onnx.ModelProto:
    b = Builder()
    add_common_inits(b, from_crop=False, onehot_equal=False, marker_from_bg=True)

    inp = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    out = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    gray_out = build_core(b, from_crop=False, marker_from_bg=True)
    bg_out = b.node("Not", [gray_out], TensorProto.BOOL, (1, 1, G, G), "bg_out")
    false14 = b.false14()
    one_hot = b.node(
        "Concat",
        [bg_out, false14, false14, false14, false14, gray_out, false14, false14, false14, false14],
        TensorProto.BOOL,
        (1, C, G, G),
        "one_hot",
        axis=1,
    )
    final_float = b.node("Cast", [one_hot], TensorProto.FLOAT, (1, C, G, G), "final_float", to=TensorProto.FLOAT)
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
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def build_model_bg_marker_float_planes() -> onnx.ModelProto:
    b = Builder()
    add_common_inits(b, from_crop=False, onehot_equal=False, marker_from_bg=True)

    inp = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    out = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    gray_out = build_core(b, from_crop=False, marker_from_bg=True)
    bg_out = b.node("Not", [gray_out], TensorProto.BOOL, (1, 1, G, G), "bg_out")
    bg_f = b.node("Cast", [bg_out], TensorProto.FLOAT, (1, 1, G, G), "bg_f", to=TensorProto.FLOAT)
    gray_f = b.last_where_float
    if gray_f is None:
        gray_f = b.node("Cast", [gray_out], TensorProto.FLOAT, (1, 1, G, G), "gray_f", to=TensorProto.FLOAT)
    false14f = b.false14f()
    final_float = b.node(
        "Concat",
        [bg_f, false14f, false14f, false14f, false14f, gray_f, false14f, false14f, false14f, false14f],
        TensorProto.FLOAT,
        (1, C, G, G),
        "final_float",
        axis=1,
    )
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
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def solve_reference(grid: list[list[int]]) -> np.ndarray:
    """Reference solver for verification."""
    g = np.array(grid, dtype=np.int64)
    g_mask = g == GRAY
    m_mask = (g != 0) & (g != GRAY)
    row_count = g_mask.sum(axis=1)
    col_count = g_mask.sum(axis=0)
    horizontal = row_count.max() > col_count.max()
    out = np.zeros_like(g)
    out[g_mask] = GRAY
    if horizontal:
        top = int(np.where(row_count > 0)[0].min())
        bottom = int(np.where(row_count > 0)[0].max())
        for c in range(G):
            n_above = int(m_mask[:top, c].sum())
            n_below = int(m_mask[bottom + 1 :, c].sum())
            for i in range(1, n_above + 1):
                out[top - i, c] = GRAY
            for i in range(1, n_below + 1):
                out[bottom + i, c] = GRAY
    else:
        left = int(np.where(col_count > 0)[0].min())
        right = int(np.where(col_count > 0)[0].max())
        for r in range(G):
            n_left = int(m_mask[r, :left].sum())
            n_right = int(m_mask[r, right + 1 :].sum())
            for i in range(1, n_left + 1):
                out[r, left - i] = GRAY
            for i in range(1, n_right + 1):
                out[r, right + i] = GRAY
    return out


def _manual_examples() -> list[dict[str, list[list[int]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        task = json.load(fh)
    examples = [
        task["train"][0],
        task["test"][0],
    ]
    return examples


def _load_task() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _grid_from_onehot(arr: np.ndarray) -> np.ndarray:
    return arr[0, :, :G, :G].argmax(axis=0).astype(np.int64)


def _check_manual_examples(model: onnx.ModelProto) -> bool:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return False
    session = ort.InferenceSession(sanitized.SerializeToString(), providers=["CPUExecutionProvider"])
    ok = True
    for idx, ex in enumerate(_manual_examples()):
        inp = convert_to_numpy(ex, "input")
        expected = convert_to_numpy(ex, "output")
        assert inp is not None and expected is not None
        pred = session.run([OUT_NAME], {IN_NAME: inp})[0]
        match = np.array_equal(pred > 0.0, expected > 0.0)
        ok = ok and match
        print(f"manual[{idx}] match={match}")
        if not match:
            print("expected:\n", _grid_from_onehot(expected))
            print("predicted:\n", _grid_from_onehot(pred))
    return ok


def _check_task(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]], list[str]]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return False, {}, ["sanitize failed"]
    session = ort.InferenceSession(sanitized.SerializeToString(), providers=["CPUExecutionProvider"])
    task = _load_task()
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


def _infer_summary(model: onnx.ModelProto) -> tuple[int, int, list[tuple[str, tuple[int, ...], int]]]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return 0, 0, []
    inferred = onnx.shape_inference.infer_shapes(sanitized, strict_mode=True)
    params = calculate_params(sanitized) or 0
    shapes: list[tuple[str, tuple[int, ...], int]] = []
    for info in inferred.graph.value_info:
        if not info.type.HasField("tensor_type") or not info.type.tensor_type.HasField("shape"):
            continue
        dims: list[int] = []
        for dim in info.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
        if dims:
            dtype = onnx.helper.tensor_dtype_to_np_dtype(info.type.tensor_type.elem_type)
            nbytes = int(np.prod(dims) * np.dtype(dtype).itemsize)
            shapes.append((info.name, tuple(dims), nbytes))
    shapes.sort(key=lambda item: item[2], reverse=True)
    return len(inferred.graph.value_info), params, shapes[:8]


def _profile_largest_internal(model: onnx.ModelProto) -> tuple[int | None, str | None, int | None]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return None, None, None
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(Path(tempfile.gettempdir()) / f"ng_{TASK_ID}")
    session = ort.InferenceSession(sanitized.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    for arr in load_task_examples(BEST_PATH):
        session.run([OUT_NAME], {IN_NAME: arr})
    trace_path = session.end_profiling()
    return calculate_memory(sanitized, trace_path), trace_path, None


def _format_counts(counts: dict[str, tuple[int, int]]) -> str:
    return ", ".join(f"{split}={ok}/{total}" for split, (ok, total) in counts.items())


def main() -> None:
    variant_builders: list[tuple[str, Callable[[], onnx.ModelProto]]] = [
        ("bg_marker", lambda: build_model_bg_marker()),
        ("bg_float", lambda: build_model_bg_marker_float_planes()),
        ("two_planes", lambda: build_model_two_planes(from_crop=False)),
        ("equal_op10", lambda: build_model_onehot(opset=10, ir_version=10, from_crop=True)),
        ("direct_op10", lambda: build_model_onehot(opset=10, ir_version=10, from_crop=False)),
        ("equal_op11", lambda: build_model_onehot(opset=11, ir_version=10, from_crop=False)),
        ("equal_op12", lambda: build_model_onehot(opset=12, ir_version=10, from_crop=False)),
    ]

    results: list[tuple[int, str, onnx.ModelProto, dict[str, Any], dict[str, tuple[int, int]]]] = []
    for label, builder in variant_builders:
        try:
            model = builder()
        except Exception as exc:
            print(f"{label:<14} build failed: {exc}")
            continue

        tmp_path = OUT_DIR / f"{TASK_ID}_{label}.onnx"
        onnx.save(model, tmp_path)
        manual_ok = _check_manual_examples(model)
        correct, counts, failures = _check_task(model)
        scored = score_file(tmp_path)
        n_vi, params, top_shapes = _infer_summary(model)
        score_text = f"{scored['score']:.6f}" if scored.get("score") is not None else "INVALID"
        print(
            f"{label:<14} manual={manual_ok} task={correct} ({_format_counts(counts)}) "
            f"memory={scored['memory']} params={scored['params']} cost={scored['cost']} "
            f"score={score_text} value_infos={n_vi} fail={failures[:2]}"
        )
        if top_shapes:
            shape_preview = ", ".join(f"{name}{dims}:{n}b" for name, dims, n in top_shapes[:3])
            print(f"               largest internals: {shape_preview}")
        if manual_ok and correct and scored.get("valid"):
            results.append((int(scored["cost"]), label, model, scored, counts))
        tmp_path.unlink(missing_ok=True)
        print()

    if not results:
        raise SystemExit("no valid correct variants")

    cost, label, model, scored, counts = min(results, key=lambda item: item[0])
    onnx.save(model, BEST_PATH)
    n_vi, params, top_shapes = _infer_summary(model)
    print(f"kept:           {label}")
    print(f"wrote:          {BEST_PATH}")
    print(f"passes:         {_format_counts(counts)}")
    print(f"initializer:    {params} scalar/element params")
    print(f"value_infos:    {n_vi} internal tensors with inferred shapes")
    print(f"memory:         {scored['memory']} bytes")
    print(f"cost:           {scored['cost']}")
    print(f"score:          {scored['score']:.6f}")
    print(f"manual_points:  {max(1.0, 25.0 - math.log(max(1.0, float(scored['cost'])))):.6f}")
    print("top internal tensors:")
    for name, dims, nbytes in top_shapes:
        print(f"  {name:<22} {dims}  {nbytes} bytes")


if __name__ == "__main__":
    main()
