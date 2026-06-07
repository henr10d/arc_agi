"""ONNX solver for NeuroGolf task131: slide the green pattern to the red line.

Task rule: the grid contains background 0, a green pattern (3), and one full
red line (2), either horizontal or vertical. Move the entire green pattern
toward the red line until it touches it, preserve the red line, and draw a cyan
line (8) parallel to the red line on the opposite side of the moved pattern.
The output keeps the input grid size; padded cells outside the ARC grid remain
all-zero in the competition 30x30 wrapper.

ONNX approach: all task examples fit in the top-left 18x18 envelope. Slice only
the background/red/green channels in that crop, infer the red orientation and
green bounding box there, compute a scalar row/column shift, gather the moved
green mask from a flattened 18x18 mask with a computed source-index grid, then
pad the finished boolean planes to 30x30 and cast only the final graph output
to float.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402


TASK_NUM = "131"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
IR_VERSION = 10


@dataclass(frozen=True)
class Variant:
    name: str
    build: Callable[[], onnx.ModelProto]


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []

    def init(self, name: str, value: Any, dtype: np.dtype[Any] | type[np.generic]) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def i64(self, name: str, value: Any) -> str:
        return self.init(name, value, np.int64)

    def i32(self, name: str, value: Any) -> str:
        return self.init(name, value, np.int32)

    def f32(self, name: str, value: Any) -> str:
        return self.init(name, value, np.float32)

    def bool(self, name: str, value: Any) -> str:
        return self.init(name, value, np.bool_)

    def node(self, op_type: str, inputs: list[str], output: str, **attrs: Any) -> str:
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        return output


def make_model(nodes: list[onnx.NodeProto], initializers: list[onnx.TensorProto], *, name: str) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 10)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def scalar_from_channel(b: Builder, channel: int, name: str) -> tuple[str, str]:
    idx = b.i64(f"{name}_idx", [channel])
    gathered = b.node("Gather", [IN_NAME, idx], f"{name}_g", axis=1)
    mask = b.node("Greater", [gathered, "half_f"], f"{name}_b")
    return gathered, mask


def bbox_from_plane(b: Builder, plane_f: str, prefix: str) -> tuple[str, str, str, str]:
    row_has = b.node("ReduceMax", [plane_f], f"{prefix}_row_has", axes=[1, 3], keepdims=0)
    col_has = b.node("ReduceMax", [plane_f], f"{prefix}_col_has", axes=[1, 2], keepdims=0)
    top = b.node("ArgMax", [row_has], f"{prefix}_top", axis=1, keepdims=0)
    left = b.node("ArgMax", [col_has], f"{prefix}_left", axis=1, keepdims=0)
    row_rev = b.node("Gather", [row_has, "rev30"], f"{prefix}_row_rev", axis=1)
    col_rev = b.node("Gather", [col_has, "rev30"], f"{prefix}_col_rev", axis=1)
    bottom_rev = b.node("ArgMax", [row_rev], f"{prefix}_bottom_rev", axis=1, keepdims=0)
    right_rev = b.node("ArgMax", [col_rev], f"{prefix}_right_rev", axis=1, keepdims=0)
    bottom = b.node("Sub", ["last_i", bottom_rev], f"{prefix}_bottom")
    right = b.node("Sub", ["last_i", right_rev], f"{prefix}_right")
    return top, bottom, left, right


def bbox_from_plane_18(b: Builder, plane_f: str, prefix: str) -> tuple[str, str, str, str]:
    row_has = b.node("ReduceMax", [plane_f], f"{prefix}_row_has", axes=[1, 3], keepdims=0)
    col_has = b.node("ReduceMax", [plane_f], f"{prefix}_col_has", axes=[1, 2], keepdims=0)
    top = b.node("ArgMax", [row_has], f"{prefix}_top", axis=1, keepdims=0)
    left = b.node("ArgMax", [col_has], f"{prefix}_left", axis=1, keepdims=0)
    row_rev = b.node("Gather", [row_has, "rev18"], f"{prefix}_row_rev", axis=1)
    col_rev = b.node("Gather", [col_has, "rev18"], f"{prefix}_col_rev", axis=1)
    bottom_rev = b.node("ArgMax", [row_rev], f"{prefix}_bottom_rev", axis=1, keepdims=0)
    right_rev = b.node("ArgMax", [col_rev], f"{prefix}_right_rev", axis=1, keepdims=0)
    bottom = b.node("Sub", ["last_i", bottom_rev], f"{prefix}_bottom")
    right = b.node("Sub", ["last_i", right_rev], f"{prefix}_right")
    return top, bottom, left, right


def slice_channel_crop(b: Builder, channel: int, prefix: str) -> tuple[str, str]:
    crop_f = b.node(
        "Slice",
        [IN_NAME, f"{prefix}_starts", f"{prefix}_ends"],
        f"{prefix}_crop_f",
    )
    crop_b = b.node("Greater", [crop_f, "half_f"], f"{prefix}_crop_b")
    return crop_f, crop_b


def expand18_to30(b: Builder, plane4: str, prefix: str) -> str:
    wide = b.node("Concat", [plane4, "false18x12"], f"{prefix}_wide", axis=3)
    return b.node("Concat", [wide, "false12x30"], f"{prefix}_full", axis=2)


def build_crop18_index_gather() -> onnx.ModelProto:
    b = Builder()
    b.f32("half_f", [0.5])
    b.i64("zero_i", np.array(0, dtype=np.int64))
    b.i64("one_i", np.array(1, dtype=np.int64))
    b.i64("last_i", np.array(17, dtype=np.int64))
    b.i64("rev18", np.arange(17, -1, -1, dtype=np.int64))
    b.i64("flat324_shape", [324])
    b.i64("rows18", np.arange(18, dtype=np.int64).reshape(18, 1))
    b.i64("cols18", np.arange(18, dtype=np.int64).reshape(1, 18))
    b.i32("neg_one_s", np.array(-1, dtype=np.int32))
    b.i32("eighteen_s", np.array(18, dtype=np.int32))
    b.i32("miss_s", np.array(324, dtype=np.int32))
    b.i32("rows18_s", np.arange(18, dtype=np.int32).reshape(18, 1))
    b.i32("cols18_s", np.arange(18, dtype=np.int32).reshape(1, 18))
    b.bool("false324", np.array([False], dtype=np.bool_))
    b.bool("false18x12", np.zeros((1, 1, 18, 12), dtype=np.bool_))
    b.bool("false12x30", np.zeros((1, 1, 12, 30), dtype=np.bool_))
    b.bool("false4d", np.zeros((1, 1, 30, 30), dtype=np.bool_))

    for prefix, channel in (("bg", 0), ("red", 2), ("green", 3)):
        b.i64(f"{prefix}_starts", [0, channel, 0, 0])
        b.i64(f"{prefix}_ends", [1, channel + 1, 18, 18])

    bg_f, bg_b = slice_channel_crop(b, 0, "bg")
    red_f, red_b = slice_channel_crop(b, 2, "red")
    green_f, green_b = slice_channel_crop(b, 3, "green")

    bg_2d = b.node("Squeeze", [bg_b], "bg_2d", axes=[0, 1])
    red_2d = b.node("Squeeze", [red_b], "red_2d", axes=[0, 1])
    green_2d = b.node("Squeeze", [green_b], "green_2d", axes=[0, 1])
    valid = b.node("Or", [b.node("Or", [bg_2d, red_2d], "bg_or_red"), green_2d], "valid")

    row_sum = b.node("ReduceSum", [red_f], "red_row_sum", axes=[1, 3], keepdims=0)
    col_sum = b.node("ReduceSum", [red_f], "red_col_sum", axes=[1, 2], keepdims=0)
    max_row = b.node("ReduceMax", [row_sum], "red_max_row", axes=[1], keepdims=0)
    max_col = b.node("ReduceMax", [col_sum], "red_max_col", axes=[1], keepdims=0)
    horizontal = b.node("Greater", [max_row, max_col], "horizontal")
    red_row = b.node("ArgMax", [row_sum], "red_row", axis=1, keepdims=0)
    red_col = b.node("ArgMax", [col_sum], "red_col", axis=1, keepdims=0)

    green_top, green_bottom, green_left, green_right = bbox_from_plane_18(b, green_f, "green")
    above = b.node("Less", [green_bottom, red_row], "green_above")
    left_side = b.node("Less", [green_right, red_col], "green_left_side")

    red_row_m1 = b.node("Sub", [red_row, "one_i"], "red_row_m1")
    red_row_p1 = b.node("Add", [red_row, "one_i"], "red_row_p1")
    red_col_m1 = b.node("Sub", [red_col, "one_i"], "red_col_m1")
    red_col_p1 = b.node("Add", [red_col, "one_i"], "red_col_p1")
    dy_above = b.node("Sub", [red_row_m1, green_bottom], "dy_above")
    dy_below = b.node("Sub", [red_row_p1, green_top], "dy_below")
    dx_left = b.node("Sub", [red_col_m1, green_right], "dx_left")
    dx_right = b.node("Sub", [red_col_p1, green_left], "dx_right")
    not_h = b.node("Not", [horizontal], "not_h")
    dy_h = b.node("Where", [above, dy_above, dy_below], "dy_h")
    dx_v = b.node("Where", [left_side, dx_left, dx_right], "dx_v")
    dy = b.node("Where", [horizontal, dy_h, "zero_i"], "dy")
    dx = b.node("Where", [horizontal, "zero_i", dx_v], "dx")

    dy_src = b.node("Cast", [dy], "dy_src", to=TensorProto.INT32)
    dx_src = b.node("Cast", [dx], "dx_src", to=TensorProto.INT32)
    src_r = b.node("Sub", ["rows18_s", dy_src], "src_r")
    src_c = b.node("Sub", ["cols18_s", dx_src], "src_c")
    src_r_ok = b.node("And", [
        b.node("Greater", [src_r, "neg_one_s"], "src_r_ge0"),
        b.node("Less", [src_r, "eighteen_s"], "src_r_lt18"),
    ], "src_r_ok")
    src_c_ok = b.node("And", [
        b.node("Greater", [src_c, "neg_one_s"], "src_c_ge0"),
        b.node("Less", [src_c, "eighteen_s"], "src_c_lt18"),
    ], "src_c_ok")
    src_ok = b.node("And", [src_r_ok, src_c_ok], "src_ok")
    src_idx_raw = b.node("Add", [b.node("Mul", [src_r, "eighteen_s"], "src_r18"), src_c], "src_idx_raw")
    src_idx = b.node("Where", [src_ok, src_idx_raw, "miss_s"], "src_idx")
    green_flat = b.node("Reshape", [green_2d, "flat324_shape"], "green_flat")
    green_flat_pad = b.node("Concat", [green_flat, "false324"], "green_flat_pad", axis=0)
    moved_green = b.node("Gather", [green_flat_pad, src_idx], "moved_green", axis=0)

    moved_top = b.node("Add", [green_top, dy], "moved_top")
    moved_bottom = b.node("Add", [green_bottom, dy], "moved_bottom")
    moved_left = b.node("Add", [green_left, dx], "moved_left")
    moved_right = b.node("Add", [green_right, dx], "moved_right")
    cyan_row = b.node("Where", [
        above,
        b.node("Sub", [moved_top, "one_i"], "moved_top_m1"),
        b.node("Add", [moved_bottom, "one_i"], "moved_bottom_p1"),
    ], "cyan_row")
    cyan_col = b.node("Where", [
        left_side,
        b.node("Sub", [moved_left, "one_i"], "moved_left_m1"),
        b.node("Add", [moved_right, "one_i"], "moved_right_p1"),
    ], "cyan_col")

    row_bg = b.node("ReduceMax", [bg_f], "row_bg", axes=[1, 3], keepdims=0)
    col_bg = b.node("ReduceMax", [bg_f], "col_bg", axes=[1, 2], keepdims=0)
    row_valid_f = b.node("Add", [b.node("Add", [row_bg, row_sum], "row_bg_red"), "green_row_has"], "row_valid_f")
    col_valid_f = b.node("Add", [b.node("Add", [col_bg, col_sum], "col_bg_red"), "green_col_has"], "col_valid_f")
    row_valid = b.node("Greater", [row_valid_f, "half_f"], "row_valid")
    col_valid = b.node("Greater", [col_valid_f, "half_f"], "col_valid")
    row_valid_2d = b.node("Unsqueeze", [b.node("Squeeze", [row_valid], "row_valid_1d", axes=[0])], "row_valid_2d", axes=[1])
    col_valid_1d = b.node("Squeeze", [col_valid], "col_valid_1d", axes=[0])

    cyan_h = b.node("And", [b.node("Equal", ["rows18", cyan_row], "cyan_h_row"), col_valid_1d], "cyan_h")
    cyan_v = b.node("And", [row_valid_2d, b.node("Equal", ["cols18", cyan_col], "cyan_v_col")], "cyan_v")
    cyan = b.node("Or", [
        b.node("And", [horizontal, cyan_h], "cyan_h_on"),
        b.node("And", [not_h, cyan_v], "cyan_v_on"),
    ], "cyan")

    occupied = b.node("Or", [b.node("Or", [red_2d, moved_green], "red_or_green"), cyan], "occupied")
    bg = b.node("And", [valid, b.node("Not", [occupied], "not_occupied")], "bg")
    green_valid = b.node("And", [moved_green, valid], "green_valid")
    cyan_valid = b.node("And", [cyan, valid], "cyan_valid")

    bg4 = b.node("Unsqueeze", [bg], "bg4", axes=[0, 1])
    red4 = b.node("Unsqueeze", [red_2d], "red4", axes=[0, 1])
    green4 = b.node("Unsqueeze", [green_valid], "green4", axes=[0, 1])
    cyan4 = b.node("Unsqueeze", [cyan_valid], "cyan4", axes=[0, 1])
    bg_full = expand18_to30(b, bg4, "bg")
    red_full = expand18_to30(b, red4, "red")
    green_full = expand18_to30(b, green4, "green")
    cyan_full = expand18_to30(b, cyan4, "cyan")
    planes = ["bg_full", "false4d", "red_full", "green_full", "false4d", "false4d", "false4d", "false4d", "cyan_full", "false4d"]
    out_bool = b.node("Concat", planes, "out_bool", axis=1)
    b.node("Cast", [out_bool], OUT_NAME, to=TensorProto.FLOAT)
    return make_model(b.nodes, b.initializers, name="task131_crop18_index_gather")


def build_generic_index_gather(*, source_i32: bool = False) -> onnx.ModelProto:
    b = Builder()
    b.f32("half_f", [0.5])
    b.i64("one_i", np.array(1, dtype=np.int64))
    b.i64("last_i", np.array(29, dtype=np.int64))
    b.i64("rev30", np.arange(29, -1, -1, dtype=np.int64))
    b.i64("flat900_shape", [900])
    b.i64("rows30", np.arange(30, dtype=np.int64).reshape(30, 1))
    b.i64("cols30", np.arange(30, dtype=np.int64).reshape(1, 30))
    if source_i32:
        b.i32("neg_one_s", np.array(-1, dtype=np.int32))
        b.i32("thirty_s", np.array(30, dtype=np.int32))
        b.i32("miss_s", np.array(900, dtype=np.int32))
        b.i32("rows30_s", np.arange(30, dtype=np.int32).reshape(30, 1))
        b.i32("cols30_s", np.arange(30, dtype=np.int32).reshape(1, 30))
    else:
        b.i64("neg_one_i", np.array(-1, dtype=np.int64))
        b.i64("thirty_i", np.array(30, dtype=np.int64))
        b.i64("miss_i", np.array(900, dtype=np.int64))
    b.bool("false900", np.array([False], dtype=np.bool_))
    b.bool("false4d", np.zeros((1, 1, 30, 30), dtype=np.bool_))

    red_f, red_b = scalar_from_channel(b, 2, "red")
    green_f, green_b = scalar_from_channel(b, 3, "green")

    red_2d = b.node("Squeeze", [red_b], "red_2d", axes=[0, 1])
    row_sum = b.node("ReduceSum", [red_f], "red_row_sum", axes=[1, 3], keepdims=0)
    col_sum = b.node("ReduceSum", [red_f], "red_col_sum", axes=[1, 2], keepdims=0)
    max_row = b.node("ReduceMax", [row_sum], "red_max_row", axes=[1], keepdims=0)
    max_col = b.node("ReduceMax", [col_sum], "red_max_col", axes=[1], keepdims=0)
    horizontal = b.node("Greater", [max_row, max_col], "horizontal")
    red_row = b.node("ArgMax", [row_sum], "red_row", axis=1, keepdims=0)
    red_col = b.node("ArgMax", [col_sum], "red_col", axis=1, keepdims=0)

    green_top, green_bottom, green_left, green_right = bbox_from_plane(b, green_f, "green")
    above = b.node("Less", [green_bottom, red_row], "green_above")
    left_side = b.node("Less", [green_right, red_col], "green_left_side")

    red_row_m1 = b.node("Sub", [red_row, "one_i"], "red_row_m1")
    red_row_p1 = b.node("Add", [red_row, "one_i"], "red_row_p1")
    red_col_m1 = b.node("Sub", [red_col, "one_i"], "red_col_m1")
    red_col_p1 = b.node("Add", [red_col, "one_i"], "red_col_p1")
    dy_above = b.node("Sub", [red_row_m1, green_bottom], "dy_above")
    dy_below = b.node("Sub", [red_row_p1, green_top], "dy_below")
    dx_left = b.node("Sub", [red_col_m1, green_right], "dx_left")
    dx_right = b.node("Sub", [red_col_p1, green_left], "dx_right")
    above_i = b.node("Cast", [above], "above_i", to=TensorProto.INT64)
    not_above = b.node("Not", [above], "not_above")
    not_above_i = b.node("Cast", [not_above], "not_above_i", to=TensorProto.INT64)
    left_i = b.node("Cast", [left_side], "left_i", to=TensorProto.INT64)
    not_left = b.node("Not", [left_side], "not_left")
    not_left_i = b.node("Cast", [not_left], "not_left_i", to=TensorProto.INT64)
    h_i = b.node("Cast", [horizontal], "h_i", to=TensorProto.INT64)
    not_h = b.node("Not", [horizontal], "not_h")
    not_h_i = b.node("Cast", [not_h], "not_h_i", to=TensorProto.INT64)
    dy_a_part = b.node("Mul", [above_i, dy_above], "dy_a_part")
    dy_b_part = b.node("Mul", [not_above_i, dy_below], "dy_b_part")
    dx_l_part = b.node("Mul", [left_i, dx_left], "dx_l_part")
    dx_r_part = b.node("Mul", [not_left_i, dx_right], "dx_r_part")
    dy_h = b.node("Add", [dy_a_part, dy_b_part], "dy_h")
    dx_v = b.node("Add", [dx_l_part, dx_r_part], "dx_v")
    dy = b.node("Mul", [h_i, dy_h], "dy")
    dx = b.node("Mul", [not_h_i, dx_v], "dx")

    if source_i32:
        dy_src = b.node("Cast", [dy], "dy_src", to=TensorProto.INT32)
        dx_src = b.node("Cast", [dx], "dx_src", to=TensorProto.INT32)
        src_rows = "rows30_s"
        src_cols = "cols30_s"
        src_neg_one = "neg_one_s"
        src_thirty = "thirty_s"
        src_miss = "miss_s"
        src_cast_to = TensorProto.INT32
    else:
        dy_src = dy
        dx_src = dx
        src_rows = "rows30"
        src_cols = "cols30"
        src_neg_one = "neg_one_i"
        src_thirty = "thirty_i"
        src_miss = "miss_i"
        src_cast_to = TensorProto.INT64
    src_r = b.node("Sub", [src_rows, dy_src], "src_r")
    src_c = b.node("Sub", [src_cols, dx_src], "src_c")
    src_r_ge0 = b.node("Greater", [src_r, src_neg_one], "src_r_ge0")
    src_c_ge0 = b.node("Greater", [src_c, src_neg_one], "src_c_ge0")
    src_r_lt30 = b.node("Less", [src_r, src_thirty], "src_r_lt30")
    src_c_lt30 = b.node("Less", [src_c, src_thirty], "src_c_lt30")
    src_r_ok = b.node("And", [src_r_ge0, src_r_lt30], "src_r_ok")
    src_c_ok = b.node("And", [src_c_ge0, src_c_lt30], "src_c_ok")
    src_ok = b.node("And", [src_r_ok, src_c_ok], "src_ok")
    src_r30 = b.node("Mul", [src_r, src_thirty], "src_r30")
    src_idx_raw = b.node("Add", [src_r30, src_c], "src_idx_raw")
    src_ok_i = b.node("Cast", [src_ok], "src_ok_i", to=src_cast_to)
    src_bad = b.node("Not", [src_ok], "src_bad")
    src_bad_i = b.node("Cast", [src_bad], "src_bad_i", to=src_cast_to)
    src_idx_good = b.node("Mul", [src_ok_i, src_idx_raw], "src_idx_good")
    src_idx_bad = b.node("Mul", [src_bad_i, src_miss], "src_idx_bad")
    src_idx_typed = b.node("Add", [src_idx_good, src_idx_bad], "src_idx_typed")
    src_idx = src_idx_typed
    green_flat = b.node("Reshape", [green_b, "flat900_shape"], "green_flat")
    green_flat_pad = b.node("Concat", [green_flat, "false900"], "green_flat_pad", axis=0)
    moved_green = b.node("Gather", [green_flat_pad, src_idx], "moved_green", axis=0)

    moved_top = b.node("Add", [green_top, dy], "moved_top")
    moved_bottom = b.node("Add", [green_bottom, dy], "moved_bottom")
    moved_left = b.node("Add", [green_left, dx], "moved_left")
    moved_right = b.node("Add", [green_right, dx], "moved_right")
    moved_top_m1 = b.node("Sub", [moved_top, "one_i"], "moved_top_m1")
    moved_bottom_p1 = b.node("Add", [moved_bottom, "one_i"], "moved_bottom_p1")
    moved_left_m1 = b.node("Sub", [moved_left, "one_i"], "moved_left_m1")
    moved_right_p1 = b.node("Add", [moved_right, "one_i"], "moved_right_p1")
    cyan_row_top = b.node("Mul", [above_i, moved_top_m1], "cyan_row_top")
    cyan_row_bottom = b.node("Mul", [not_above_i, moved_bottom_p1], "cyan_row_bottom")
    cyan_row = b.node("Add", [cyan_row_top, cyan_row_bottom], "cyan_row")
    cyan_col_left = b.node("Mul", [left_i, moved_left_m1], "cyan_col_left")
    cyan_col_right = b.node("Mul", [not_left_i, moved_right_p1], "cyan_col_right")
    cyan_col = b.node("Add", [cyan_col_left, cyan_col_right], "cyan_col")

    active_3d = b.node("ReduceMax", [IN_NAME], "active_3d", axes=[1], keepdims=0)
    active_f = b.node("Squeeze", [active_3d], "active_f", axes=[0])
    valid = b.node("Greater", [active_f, "half_f"], "valid_b")
    row_valid_f = b.node("ReduceMax", [active_f], "row_valid_f", axes=[1], keepdims=0)
    col_valid_f = b.node("ReduceMax", [active_f], "col_valid_f", axes=[0], keepdims=0)
    row_valid = b.node("Greater", [row_valid_f, "half_f"], "row_valid_b")
    col_valid = b.node("Greater", [col_valid_f, "half_f"], "col_valid_b")
    row_valid_2d = b.node("Unsqueeze", [row_valid], "row_valid_2d", axes=[1])

    cyan_h_row = b.node("Equal", ["rows30", cyan_row], "cyan_h_row")
    cyan_h = b.node("And", [cyan_h_row, col_valid], "cyan_h")
    cyan_v_col = b.node("Equal", ["cols30", cyan_col], "cyan_v_col")
    cyan_v = b.node("And", [row_valid_2d, cyan_v_col], "cyan_v")
    cyan_h_on = b.node("And", [horizontal, cyan_h], "cyan_h_on")
    cyan_v_on = b.node("And", [not_h, cyan_v], "cyan_v_on")
    cyan = b.node("Or", [cyan_h_on, cyan_v_on], "cyan")

    red_or_green = b.node("Or", [red_2d, moved_green], "red_or_green")
    occupied_tmp = b.node("Or", [red_or_green, cyan], "occupied_tmp")
    bg_not_occ = b.node("Not", [occupied_tmp], "bg_not_occ")
    bg = b.node("And", [valid, bg_not_occ], "bg")
    cyan_valid = b.node("And", [cyan, valid], "cyan_valid")
    moved_green_valid = b.node("And", [moved_green, valid], "moved_green_valid")

    bg4 = b.node("Unsqueeze", ["bg"], "bg4", axes=[0, 1])
    green4 = b.node("Unsqueeze", [moved_green_valid], "green4", axes=[0, 1])
    cyan4 = b.node("Unsqueeze", [cyan_valid], "cyan4", axes=[0, 1])
    planes = ["bg4", "false4d", red_b, "green4", "false4d", "false4d", "false4d", "false4d", "cyan4", "false4d"]
    out_bool = b.node("Concat", planes, "out_bool", axis=1)
    b.node("Cast", [out_bool], OUT_NAME, to=TensorProto.FLOAT)
    suffix = "i32" if source_i32 else "i64"
    return make_model(b.nodes, b.initializers, name=f"task131_generic_index_gather_{suffix}")


def build_generic_index_gather_i64() -> onnx.ModelProto:
    return build_generic_index_gather(source_i32=False)


def build_generic_index_gather_i32() -> onnx.ModelProto:
    return build_generic_index_gather(source_i32=True)


def variants() -> list[Variant]:
    return [
        Variant("crop18_index_gather", build_crop18_index_gather),
        Variant("generic_index_gather_i32", build_generic_index_gather_i32),
        Variant("generic_index_gather_i64", build_generic_index_gather_i64),
    ]


def load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def solve_grid(grid: list[list[int]]) -> list[list[int]]:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(arr)
    red_pos = np.argwhere(arr == 2)
    green_pos = np.argwhere(arr == 3)
    red_rows = np.unique(red_pos[:, 0])
    red_cols = np.unique(red_pos[:, 1])
    horizontal = len(red_rows) == 1
    rr = int(red_rows[0])
    cc = int(red_cols[0])
    top, left = green_pos.min(axis=0)
    bottom, right = green_pos.max(axis=0)
    dy = dx = 0
    if horizontal:
        if bottom < rr:
            dy = rr - 1 - int(bottom)
            cyan_row = int(top) + dy - 1
        else:
            dy = rr + 1 - int(top)
            cyan_row = int(bottom) + dy + 1
        out[cyan_row, :] = 8
    else:
        if right < cc:
            dx = cc - 1 - int(right)
            cyan_col = int(left) + dx - 1
        else:
            dx = cc + 1 - int(left)
            cyan_col = int(right) + dx + 1
        out[:, cyan_col] = 8
    for r, c in green_pos:
        out[int(r) + dy, int(c) + dx] = 3
    for r, c in red_pos:
        out[int(r), int(c)] = 2
    return out.tolist()


def verify_reference() -> None:
    for split, examples in load_task_data().items():
        for idx, example in enumerate(examples):
            got = solve_grid(example["input"])
            if got != example["output"]:
                raise AssertionError(f"reference mismatch {split} #{idx}")


def verify_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]], str | None]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    all_ok = True
    first_error: str | None = None
    splits: dict[str, tuple[int, int]] = {}
    for split, examples in load_task_data().items():
        passed = 0
        checked = 0
        for idx, example in enumerate(examples):
            inp = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if inp is None or expected is None:
                continue
            pred = session.run([OUT_NAME], {IN_NAME: inp})[0]
            checked += 1
            if np.array_equal(pred > 0.0, expected > 0.0):
                passed += 1
            else:
                all_ok = False
                if first_error is None:
                    first_error = f"{split} #{idx}"
        splits[split] = (passed, checked)
    return all_ok, splits, first_error


def write_model(model: onnx.ModelProto, path: Path) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(path))


def benchmark_variants() -> tuple[dict[str, dict[str, Any]], str, dict[str, onnx.ModelProto]]:
    verify_reference()
    built = {variant.name: variant.build() for variant in variants()}
    results: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}_") as tmp:
        tmp_path = Path(tmp)
        for name, model in built.items():
            ok, splits, first_error = verify_correct(model)
            path = tmp_path / f"{TASK_ID}.onnx"
            write_model(model, path)
            result = score_file(path)
            result["correct"] = ok
            result["splits"] = splits
            result["first_error"] = first_error
            results[name] = result

    def sort_key(item: tuple[str, dict[str, Any]]) -> int:
        result = item[1]
        if not result["valid"] or not result["correct"]:
            return 10**18
        return int(result["cost"])

    best_name, best_result = min(results.items(), key=sort_key)
    if sort_key((best_name, best_result)) >= 10**18:
        raise RuntimeError(f"no valid correct variant: {results}")
    return results, best_name, built


def print_benchmark(results: dict[str, dict[str, Any]], best_name: str) -> None:
    print(f"{'variant':<24} {'ok':<5} {'valid':<6} {'memory':>8} {'params':>7} {'cost':>8} {'score':>10}")
    for name, result in sorted(results.items(), key=lambda item: (item[0] != best_name, item[0])):
        score = result["score"]
        score_text = f"{score:.6f}" if isinstance(score, float) else "None"
        print(
            f"{name:<24} {str(result['correct']):<5} {str(result['valid']):<6} "
            f"{str(result['memory']):>8} {str(result['params']):>7} "
            f"{str(result['cost']):>8} {score_text:>10}"
        )
        if not result["correct"]:
            print(f"  splits: {result['splits']} first_error={result['first_error']}")
        if result["error"]:
            print(f"  error: {str(result['error']).strip()}")
    print(f"best: {best_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and benchmark task131 ONNX variants.")
    parser.add_argument("--benchmark-only", action="store_true", help="print variant scores without writing best model")
    args = parser.parse_args()

    results, best_name, built = benchmark_variants()
    print_benchmark(results, best_name)
    if not args.benchmark_only:
        OUT_DIR.mkdir(exist_ok=True)
        write_model(built[best_name], BEST_PATH)
        print(f"wrote: {BEST_PATH}")


if __name__ == "__main__":
    main()
