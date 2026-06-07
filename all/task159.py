"""Direct ONNX generator for NeuroGolf task159.

Task rule: find the red hollow square frame and the separate 3x3 non-red
object. The output is a copy of the frame, with the 3x3 object scaled by an
integer factor so it fills the frame interior: 5x5 frames use the object as-is,
8x8 frames expand each object cell to 2x2, 11x11 to 3x3, and 14x14 to 4x4.
The object color is preserved, red border cells stay red, black object cells
become black interior cells, and cells outside the variable-size output remain
zero-padded in the required 30x30 competition tensor.
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


TASK_NUM = "159"
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
        self.value_infos: list[onnx.ValueInfoProto] = []

    def init(self, name: str, array: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(array, name))
        return name

    def init_i64(self, name: str, values: list[int], shape: tuple[int, ...] | None = None) -> str:
        arr = np.asarray(values, dtype=np.int64)
        if shape is not None:
            arr = arr.reshape(shape)
        return self.init(name, arr)

    def init_f32(self, name: str, array: np.ndarray) -> str:
        return self.init(name, np.asarray(array, dtype=np.float32))

    def init_bool(self, name: str, array: np.ndarray) -> str:
        return self.init(name, np.asarray(array, dtype=np.bool_))

    def node(self, op_type: str, inputs: list[str], output: str, **attrs: Any) -> str:
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        return output

    def value_info(self, name: str, elem_type: int, shape: list[int]) -> None:
        self.value_infos.append(helper.make_tensor_value_info(name, elem_type, shape))


def make_model(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    value_infos: list[onnx.ValueInfoProto] | None = None,
) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        initializers,
        value_info=value_infos or [],
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


def frame_masks(size: int) -> tuple[np.ndarray, np.ndarray]:
    border = np.zeros((1, 1, 30, 30), dtype=np.bool_)
    interior = np.zeros((1, 1, 30, 30), dtype=np.bool_)
    border[:, :, 0, :size] = True
    border[:, :, size - 1, :size] = True
    border[:, :, :size, 0] = True
    border[:, :, :size, size - 1] = True
    interior[:, :, 1 : size - 1, 1 : size - 1] = True
    return border, interior


def compact_frame_masks(size: int) -> tuple[np.ndarray, np.ndarray]:
    border = np.zeros((1, 1, 14, 14), dtype=np.bool_)
    interior = np.zeros((1, 1, 14, 14), dtype=np.bool_)
    border[:, :, 0, :size] = True
    border[:, :, size - 1, :size] = True
    border[:, :, :size, 0] = True
    border[:, :, :size, size - 1] = True
    interior[:, :, 1 : size - 1, 1 : size - 1] = True
    return border, interior


def scaled_indices(scale: int) -> list[int]:
    interior = [idx for idx in range(3) for _ in range(scale)]
    return [0] + interior + [2] * (29 - len(interior))


def compact_scaled_indices(scale: int) -> list[int]:
    interior = [idx for idx in range(3) for _ in range(scale)]
    return [0] + interior + [2] * (13 - len(interior))


def add_object_crop(b: Builder) -> str:
    b.init_f32("zero_f", np.array([0.0], dtype=np.float32))
    b.init_f32("one_half_f", np.array([1.5], dtype=np.float32))
    b.init_f32("two_f", np.array([2.0], dtype=np.float32))
    b.init_f32("two_half_f", np.array([2.5], dtype=np.float32))
    b.init_i64("three_i", [3])
    b.init_i64("slice_prefix", [0, 0])
    b.init_i64("slice_end_prefix", [1, 10])
    b.init_i64("slice_axes", [0, 1, 2, 3])

    color_i = b.node("ArgMax", [IN_NAME], "color_i", axis=1, keepdims=0)
    color = b.node("Cast", [color_i], "color", to=TensorProto.FLOAT)
    red_lo = b.node("Greater", [color, "one_half_f"], "red_lo")
    red_hi = b.node("Less", [color, "two_half_f"], "red_hi")
    red = b.node("And", [red_lo, red_hi], "red")
    red_f = b.node("Cast", [red], "red_f", to=TensorProto.FLOAT)
    red_count = b.node("ReduceSum", [red_f], "red_count", axes=[1, 2], keepdims=0)

    gt0 = b.node("Greater", [color, "zero_f"], "gt0")
    lt2 = b.node("Less", [color, "two_f"], "lt2")
    gt2 = b.node("Greater", [color, "two_f"], "gt2")
    obj_low = b.node("And", [gt0, lt2], "obj_low")
    obj_has = b.node("Or", [obj_low, gt2], "obj_has")
    obj = b.node("Cast", [obj_has], "obj", to=TensorProto.FLOAT)
    row_score = b.node("ReduceSum", [obj], "row_score", axes=[2], keepdims=0)
    col_score = b.node("ReduceSum", [obj], "col_score", axes=[1], keepdims=0)
    row_has = b.node("Greater", [row_score, "zero_f"], "row_has")
    col_has = b.node("Greater", [col_score, "zero_f"], "col_has")
    row_has_f = b.node("Cast", [row_has], "row_has_f", to=TensorProto.FLOAT)
    col_has_f = b.node("Cast", [col_has], "col_has_f", to=TensorProto.FLOAT)
    row0 = b.node("ArgMax", [row_has_f], "row0", axis=1, keepdims=0)
    col0 = b.node("ArgMax", [col_has_f], "col0", axis=1, keepdims=0)
    row1 = b.node("Add", [row0, "three_i"], "row1")
    col1 = b.node("Add", [col0, "three_i"], "col1")
    starts = b.node("Concat", ["slice_prefix", row0, col0], "crop_starts", axis=0)
    ends = b.node("Concat", ["slice_end_prefix", row1, col1], "crop_ends", axis=0)
    crop = b.node("Slice", [IN_NAME, starts, ends, "slice_axes"], "obj_crop")
    b.value_info(crop, TensorProto.FLOAT, [1, 10, 3, 3])
    return red_count + "|" + crop


def add_object_crop_int_labels(b: Builder) -> str:
    b.init_i64("zero_i", [0])
    b.init_i64("two_i", [2])
    b.init_i64("three_i", [3])
    b.init_f32("zero_f", np.array([0.0], dtype=np.float32))
    b.init_i64("slice_prefix", [0, 0])
    b.init_i64("slice_end_prefix", [1, 10])
    b.init_i64("slice_axes", [0, 1, 2, 3])

    color_i = b.node("ArgMax", [IN_NAME], "color_i", axis=1, keepdims=0)
    red = b.node("Equal", [color_i, "two_i"], "red")
    red_f = b.node("Cast", [red], "red_f", to=TensorProto.FLOAT)
    red_count = b.node("ReduceSum", [red_f], "red_count", axes=[1, 2], keepdims=0)

    gt0 = b.node("Greater", [color_i, "zero_i"], "gt0")
    not_red = b.node("Not", [red], "not_red")
    obj_has = b.node("And", [gt0, not_red], "obj_has")
    obj = b.node("Cast", [obj_has], "obj", to=TensorProto.FLOAT)
    row_score = b.node("ReduceSum", [obj], "row_score", axes=[2], keepdims=0)
    col_score = b.node("ReduceSum", [obj], "col_score", axes=[1], keepdims=0)
    row_has = b.node("Greater", [row_score, "zero_f"], "row_has")
    col_has = b.node("Greater", [col_score, "zero_f"], "col_has")
    row_has_f = b.node("Cast", [row_has], "row_has_f", to=TensorProto.FLOAT)
    col_has_f = b.node("Cast", [col_has], "col_has_f", to=TensorProto.FLOAT)
    row0 = b.node("ArgMax", [row_has_f], "row0", axis=1, keepdims=0)
    col0 = b.node("ArgMax", [col_has_f], "col0", axis=1, keepdims=0)
    row1 = b.node("Add", [row0, "three_i"], "row1")
    col1 = b.node("Add", [col0, "three_i"], "col1")
    starts = b.node("Concat", ["slice_prefix", row0, col0], "crop_starts", axis=0)
    ends = b.node("Concat", ["slice_end_prefix", row1, col1], "crop_ends", axis=0)
    crop = b.node("Slice", [IN_NAME, starts, ends, "slice_axes"], "obj_crop")
    b.value_info(color_i, TensorProto.INT64, [1, 30, 30])
    b.value_info(crop, TensorProto.FLOAT, [1, 10, 3, 3])
    return red_count + "|" + crop


def add_scaled_candidate(b: Builder, crop: str, size: int) -> str:
    scale = (size - 2) // 3
    b.init_i64(f"row_idx_{size}", scaled_indices(scale))
    b.init_i64(f"col_idx_{size}", scaled_indices(scale))
    border, interior = frame_masks(size)
    b.init_bool(f"border_{size}", border)
    b.init_bool(f"interior_{size}", interior)

    rows = b.node("Gather", [crop, f"row_idx_{size}"], f"rows_{size}", axis=2)
    scaled = b.node("Gather", [rows, f"col_idx_{size}"], f"scaled_{size}", axis=3)
    inside = b.node("Where", [f"interior_{size}", scaled, "zero_f"], f"inside_{size}")
    cand = b.node("Where", [f"border_{size}", "red_vec", inside], f"cand_{size}")
    b.value_info(rows, TensorProto.FLOAT, [1, 10, 30, 3])
    b.value_info(scaled, TensorProto.FLOAT, FULL_SHAPE)
    b.value_info(inside, TensorProto.FLOAT, FULL_SHAPE)
    b.value_info(cand, TensorProto.FLOAT, FULL_SHAPE)
    return cand


def build_four_size_selector() -> onnx.ModelProto:
    b = Builder()
    b.init_f32(
        "red_vec",
        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 10, 1, 1),
    )
    b.init_f32("count_22", np.array([22.0], dtype=np.float32))
    b.init_f32("count_34", np.array([34.0], dtype=np.float32))
    b.init_f32("count_46", np.array([46.0], dtype=np.float32))

    packed = add_object_crop(b)
    red_count, crop = packed.split("|")

    cand5 = add_scaled_candidate(b, crop, 5)
    cand8 = add_scaled_candidate(b, crop, 8)
    cand11 = add_scaled_candidate(b, crop, 11)
    cand14 = add_scaled_candidate(b, crop, 14)

    lt22 = b.node("Less", [red_count, "count_22"], "lt22")
    lt34 = b.node("Less", [red_count, "count_34"], "lt34")
    lt46 = b.node("Less", [red_count, "count_46"], "lt46")
    choose11 = b.node("Where", [lt46, cand11, cand14], "choose11")
    choose8 = b.node("Where", [lt34, cand8, choose11], "choose8")
    b.node("Where", [lt22, cand5, choose8], OUT_NAME)
    b.value_info(choose11, TensorProto.FLOAT, FULL_SHAPE)
    b.value_info(choose8, TensorProto.FLOAT, FULL_SHAPE)
    return make_model(b.nodes, b.initializers, b.value_infos)


def add_size_conditions(b: Builder, red_count: str) -> tuple[str, str, str]:
    b.init_f32("count_22", np.array([22.0], dtype=np.float32))
    b.init_f32("count_34", np.array([34.0], dtype=np.float32))
    b.init_f32("count_46", np.array([46.0], dtype=np.float32))
    lt22 = b.node("Less", [red_count, "count_22"], "lt22")
    lt34 = b.node("Less", [red_count, "count_34"], "lt34")
    lt46 = b.node("Less", [red_count, "count_46"], "lt46")
    return lt22, lt34, lt46


def build_dynamic_index_selector() -> onnx.ModelProto:
    b = Builder()
    b.init_f32(
        "red_vec",
        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 10, 1, 1),
    )

    packed = add_object_crop(b)
    red_count, crop = packed.split("|")
    lt22, lt34, lt46 = add_size_conditions(b, red_count)

    for size in (5, 8, 11, 14):
        scale = (size - 2) // 3
        b.init_f32(f"idx_{size}", np.asarray(scaled_indices(scale), dtype=np.float32))
        border, interior = frame_masks(size)
        b.init_f32(f"border_{size}", border.astype(np.float32))
        b.init_f32(f"interior_{size}", interior.astype(np.float32))

    idx11 = b.node("Where", [lt46, "idx_11", "idx_14"], "idx11_f")
    idx8 = b.node("Where", [lt34, "idx_8", idx11], "idx8_f")
    idx_f = b.node("Where", [lt22, "idx_5", idx8], "idx_f")
    idx = b.node("Cast", [idx_f], "idx", to=TensorProto.INT64)
    border11 = b.node("Where", [lt46, "border_11", "border_14"], "border11")
    border8 = b.node("Where", [lt34, "border_8", border11], "border8")
    border_f = b.node("Where", [lt22, "border_5", border8], "border_f")
    interior11 = b.node("Where", [lt46, "interior_11", "interior_14"], "interior11")
    interior8 = b.node("Where", [lt34, "interior_8", interior11], "interior8")
    interior_f = b.node("Where", [lt22, "interior_5", interior8], "interior_f")
    border = b.node("Greater", [border_f, "zero_f"], "border")
    interior = b.node("Greater", [interior_f, "zero_f"], "interior")

    rows = b.node("Gather", [crop, idx], "rows", axis=2)
    scaled = b.node("Gather", [rows, idx], "scaled", axis=3)
    inside = b.node("Where", [interior, scaled, "zero_f"], "inside")
    b.node("Where", [border, "red_vec", inside], OUT_NAME)

    b.value_info(idx11, TensorProto.FLOAT, [30])
    b.value_info(idx8, TensorProto.FLOAT, [30])
    b.value_info(idx_f, TensorProto.FLOAT, [30])
    b.value_info(idx, TensorProto.INT64, [30])
    for name in (border11, border8, border_f, interior11, interior8, interior_f):
        b.value_info(name, TensorProto.FLOAT, [1, 1, 30, 30])
    for name in (border, interior):
        b.value_info(name, TensorProto.BOOL, [1, 1, 30, 30])
    b.value_info(rows, TensorProto.FLOAT, [1, 10, 30, 3])
    b.value_info(scaled, TensorProto.FLOAT, FULL_SHAPE)
    b.value_info(inside, TensorProto.FLOAT, FULL_SHAPE)
    return make_model(b.nodes, b.initializers, b.value_infos)


def build_compact_index_selector() -> onnx.ModelProto:
    b = Builder()
    b.init_f32(
        "red_vec",
        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 10, 1, 1),
    )
    packed = add_object_crop_int_labels(b)
    red_count, crop = packed.split("|")
    lt22, lt34, lt46 = add_size_conditions(b, red_count)

    for size in (5, 8, 11, 14):
        scale = (size - 2) // 3
        b.init_f32(f"cidx_{size}", np.asarray(compact_scaled_indices(scale), dtype=np.float32))
        border, interior = compact_frame_masks(size)
        b.init_f32(f"cborder_{size}", border.astype(np.float32))
        b.init_f32(f"cinterior_{size}", interior.astype(np.float32))

    idx11 = b.node("Where", [lt46, "cidx_11", "cidx_14"], "idx11_f")
    idx8 = b.node("Where", [lt34, "cidx_8", idx11], "idx8_f")
    idx_f = b.node("Where", [lt22, "cidx_5", idx8], "idx_f")
    idx = b.node("Cast", [idx_f], "idx", to=TensorProto.INT64)
    border11 = b.node("Where", [lt46, "cborder_11", "cborder_14"], "border11")
    border8 = b.node("Where", [lt34, "cborder_8", border11], "border8")
    border_f = b.node("Where", [lt22, "cborder_5", border8], "border_f")
    interior11 = b.node("Where", [lt46, "cinterior_11", "cinterior_14"], "interior11")
    interior8 = b.node("Where", [lt34, "cinterior_8", interior11], "interior8")
    interior_f = b.node("Where", [lt22, "cinterior_5", interior8], "interior_f")
    border = b.node("Greater", [border_f, "zero_f"], "border")
    interior = b.node("Greater", [interior_f, "zero_f"], "interior")

    rows = b.node("Gather", [crop, idx], "rows", axis=2)
    scaled = b.node("Gather", [rows, idx], "scaled", axis=3)
    inside = b.node("Where", [interior, scaled, "zero_f"], "inside")
    compact = b.node("Where", [border, "red_vec", inside], "compact")
    b.node("Pad", [compact], OUT_NAME, mode="constant", pads=[0, 0, 0, 0, 0, 0, 16, 16], value=0.0)

    b.value_info(idx11, TensorProto.FLOAT, [14])
    b.value_info(idx8, TensorProto.FLOAT, [14])
    b.value_info(idx_f, TensorProto.FLOAT, [14])
    b.value_info(idx, TensorProto.INT64, [14])
    for name in (border11, border8, border_f, interior11, interior8, interior_f):
        b.value_info(name, TensorProto.FLOAT, [1, 1, 14, 14])
    for name in (border, interior):
        b.value_info(name, TensorProto.BOOL, [1, 1, 14, 14])
    b.value_info(rows, TensorProto.FLOAT, [1, 10, 14, 3])
    b.value_info(scaled, TensorProto.FLOAT, [1, 10, 14, 14])
    b.value_info(inside, TensorProto.FLOAT, [1, 10, 14, 14])
    b.value_info(compact, TensorProto.FLOAT, [1, 10, 14, 14])
    return make_model(b.nodes, b.initializers, b.value_infos)


def build_label_onehot_selector() -> onnx.ModelProto:
    b = Builder()
    b.init_i64("zero_i", [0])
    b.init_i64("two_i", [2])
    b.init_i64("three_i", [3])
    b.init("zero_h", np.array([0.0], dtype=np.float16))
    b.init("outside_h", np.array([10.0], dtype=np.float16))
    b.init("two_h", np.array([2.0], dtype=np.float16))
    b.init("count_22_h", np.array([22.0], dtype=np.float16))
    b.init("count_34_h", np.array([34.0], dtype=np.float16))
    b.init("count_46_h", np.array([46.0], dtype=np.float16))
    b.init_i64("depth_10", [10])
    b.init_f32("onehot_values", np.array([0.0, 1.0], dtype=np.float32))
    b.init_i64("label_slice_prefix", [0])
    b.init_i64("label_slice_end_prefix", [1])
    b.init_i64("label_slice_axes", [0, 1, 2])

    color_i = b.node("ArgMax", [IN_NAME], "color_i", axis=1, keepdims=0)
    red = b.node("Equal", [color_i, "two_i"], "red")
    red_f = b.node("Cast", [red], "red_f", to=TensorProto.FLOAT16)
    red_count = b.node("ReduceSum", [red_f], "red_count", axes=[1, 2], keepdims=0)
    lt22 = b.node("Less", [red_count, "count_22_h"], "lt22")
    lt34 = b.node("Less", [red_count, "count_34_h"], "lt34")
    lt46 = b.node("Less", [red_count, "count_46_h"], "lt46")

    gt0 = b.node("Greater", [color_i, "zero_i"], "gt0")
    not_red = b.node("Not", [red], "not_red")
    obj_has = b.node("And", [gt0, not_red], "obj_has")
    obj = b.node("Cast", [obj_has], "obj", to=TensorProto.FLOAT16)
    row_score = b.node("ReduceSum", [obj], "row_score", axes=[2], keepdims=0)
    col_score = b.node("ReduceSum", [obj], "col_score", axes=[1], keepdims=0)
    row_has = b.node("Greater", [row_score, "zero_h"], "row_has")
    col_has = b.node("Greater", [col_score, "zero_h"], "col_has")
    row_has_f = b.node("Cast", [row_has], "row_has_f", to=TensorProto.FLOAT16)
    col_has_f = b.node("Cast", [col_has], "col_has_f", to=TensorProto.FLOAT16)
    row0 = b.node("ArgMax", [row_has_f], "row0", axis=1, keepdims=0)
    col0 = b.node("ArgMax", [col_has_f], "col0", axis=1, keepdims=0)
    row1 = b.node("Add", [row0, "three_i"], "row1")
    col1 = b.node("Add", [col0, "three_i"], "col1")
    starts = b.node("Concat", ["label_slice_prefix", row0, col0], "label_crop_starts", axis=0)
    ends = b.node("Concat", ["label_slice_end_prefix", row1, col1], "label_crop_ends", axis=0)
    crop = b.node("Slice", [color_i, starts, ends, "label_slice_axes"], "label_crop")

    for size in (5, 8, 11, 14):
        scale = (size - 2) // 3
        b.init(f"lidx_{size}", np.asarray(compact_scaled_indices(scale), dtype=np.float16))
        border, interior = compact_frame_masks(size)
        b.init(f"lborder_{size}", border.reshape(1, 14, 14).astype(np.float16))
        b.init(f"linterior_{size}", interior.reshape(1, 14, 14).astype(np.float16))

    idx11 = b.node("Where", [lt46, "lidx_11", "lidx_14"], "lidx11_f")
    idx8 = b.node("Where", [lt34, "lidx_8", idx11], "lidx8_f")
    idx_f = b.node("Where", [lt22, "lidx_5", idx8], "lidx_f")
    idx = b.node("Cast", [idx_f], "lidx", to=TensorProto.INT64)
    border11 = b.node("Where", [lt46, "lborder_11", "lborder_14"], "lborder11")
    border8 = b.node("Where", [lt34, "lborder_8", border11], "lborder8")
    border_f = b.node("Where", [lt22, "lborder_5", border8], "lborder_f")
    interior11 = b.node("Where", [lt46, "linterior_11", "linterior_14"], "linterior11")
    interior8 = b.node("Where", [lt34, "linterior_8", interior11], "linterior8")
    interior_f = b.node("Where", [lt22, "linterior_5", interior8], "linterior_f")
    border = b.node("Greater", [border_f, "zero_h"], "lborder")
    interior = b.node("Greater", [interior_f, "zero_h"], "linterior")

    rows = b.node("Gather", [crop, idx], "label_rows", axis=1)
    scaled_i = b.node("Gather", [rows, idx], "label_scaled_i", axis=2)
    scaled_f = b.node("Cast", [scaled_i], "label_scaled_f", to=TensorProto.FLOAT16)
    inside = b.node("Where", [interior, scaled_f, "outside_h"], "label_inside")
    label_f = b.node("Where", [border, "two_h", inside], "label_f")
    label_i = b.node("Cast", [label_f], "label_i", to=TensorProto.INT64)
    compact = b.node("OneHot", [label_i, "depth_10", "onehot_values"], "onehot", axis=1)
    b.node("Pad", [compact], OUT_NAME, mode="constant", pads=[0, 0, 0, 0, 0, 0, 16, 16], value=0.0)

    b.value_info(color_i, TensorProto.INT64, [1, 30, 30])
    b.value_info(crop, TensorProto.INT64, [1, 3, 3])
    b.value_info(idx11, TensorProto.FLOAT16, [14])
    b.value_info(idx8, TensorProto.FLOAT16, [14])
    b.value_info(idx_f, TensorProto.FLOAT16, [14])
    b.value_info(idx, TensorProto.INT64, [14])
    for name in (border11, border8, border_f, interior11, interior8, interior_f):
        b.value_info(name, TensorProto.FLOAT16, [1, 14, 14])
    for name in (border, interior):
        b.value_info(name, TensorProto.BOOL, [1, 14, 14])
    b.value_info(rows, TensorProto.INT64, [1, 14, 3])
    b.value_info(scaled_i, TensorProto.INT64, [1, 14, 14])
    b.value_info(scaled_f, TensorProto.FLOAT16, [1, 14, 14])
    b.value_info(inside, TensorProto.FLOAT16, [1, 14, 14])
    b.value_info(label_f, TensorProto.FLOAT16, [1, 14, 14])
    b.value_info(label_i, TensorProto.INT64, [1, 14, 14])
    b.value_info(compact, TensorProto.FLOAT, [1, 10, 14, 14])
    return make_model(b.nodes, b.initializers, b.value_infos)


def variants() -> list[Variant]:
    return [
        Variant("compact_index_selector", build_compact_index_selector),
        Variant("label_onehot_selector", build_label_onehot_selector),
        Variant("dynamic_index_selector", build_dynamic_index_selector),
        Variant("four_size_selector", build_four_size_selector),
    ]


def load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def verify_correct(model: onnx.ModelProto, *, split_filter: set[str] | None = None) -> tuple[bool, dict[str, tuple[int, int]]]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    all_ok = True
    splits: dict[str, tuple[int, int]] = {}
    for split, examples in load_task_data().items():
        if split_filter is not None and split not in split_filter:
            continue
        passed = 0
        checked = 0
        for example in examples:
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
        splits[split] = (passed, checked)
    return all_ok, splits


def write_model(model: onnx.ModelProto, path: Path) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(path))


def benchmark_variants() -> tuple[dict[str, dict[str, Any]], str, dict[str, onnx.ModelProto]]:
    built = {variant.name: variant.build() for variant in variants()}
    results: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}_") as tmp:
        tmp_path = Path(tmp)
        for name, model in built.items():
            ok, splits = verify_correct(model)
            path = tmp_path / f"{TASK_ID}.onnx"
            write_model(model, path)
            result = score_file(path)
            result["correct"] = ok
            result["splits"] = splits
            results[name] = result

    best_name = min(
        results,
        key=lambda name: int(results[name]["cost"]) if results[name]["valid"] and results[name]["correct"] else 10**18,
    )
    if not results[best_name]["valid"] or not results[best_name]["correct"]:
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
            print(f"  splits: {result['splits']}")
        if result["error"]:
            print(f"  error: {str(result['error']).strip()}")
    print(f"best: {best_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and benchmark task159 ONNX variants.")
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
