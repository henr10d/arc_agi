"""Compact ONNX for NeuroGolf task082: repeat a shifted first-row pattern.

Task rule: the meaningful grid has height 6 and variable width up to 30.
Only the first input row contains non-background colors. Output rows 0, 2,
and 4 copy that first row exactly. Output rows 1, 3, and 5 place each
non-background first-row color one column left and one column right, clipped
to the example's real width; all other meaningful cells are background 0 and
padding outside the real grid stays all-zero.

ONNX approach: work only on row 0 as a compact [1, C, 1, 15] tensor, derive a
valid-column mask from one-hot activity, shift channels 1..9 horizontally, build
channel 0 from the valid mask, concatenate the six rows, and pad height only at
the graph output.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402


TASK_NUM = "082"
TASK_ID = f"task{TASK_NUM}"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
OPSET = 10
IR_VERSION = 10
WORK_W = 15
SHIFT_W = WORK_W - 1


def init_i64(name: str, values: list[int] | np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name)


def init_f32(name: str, values: list[float] | np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.float32), name)


def vi(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


class Graph:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self.value_infos: list[onnx.ValueInfoProto] = []

    def add_init(self, tensor: onnx.TensorProto) -> str:
        self.inits.append(tensor)
        return tensor.name

    def add_node(
        self,
        op_type: str,
        inputs: list[str],
        output: str,
        dtype: int,
        shape: list[int],
        **attrs: Any,
    ) -> str:
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        self.value_infos.append(vi(output, dtype, shape))
        return output


def base_constants(g: Graph, *, include_zero: bool = True) -> None:
    g.add_init(init_i64("row_starts", [0, 0, 0, 0]))
    g.add_init(init_i64("row_ends", [1, 10, 1, WORK_W]))
    g.add_init(init_i64("nz_starts", [0, 1, 0, 0]))
    g.add_init(init_i64("nz_ends", [1, 10, 1, WORK_W]))
    g.add_init(init_i64("left_starts", [0, 0, 0, 1]))
    g.add_init(init_i64("left_ends", [1, 9, 1, WORK_W]))
    g.add_init(init_i64("right_starts", [0, 0, 0, 0]))
    g.add_init(init_i64("right_ends", [1, 9, 1, SHIFT_W]))
    g.add_init(init_i64("axes4", [0, 1, 2, 3]))
    if include_zero:
        g.add_init(init_f32("zero", [0.0]))


def shifted_nz_pad(g: Graph) -> str:
    nz = g.add_node(
        "Slice",
        ["row", "nz_starts", "nz_ends", "axes4"],
        "nz",
        TensorProto.FLOAT,
        [1, 9, 1, WORK_W],
    )
    left_src = g.add_node(
        "Slice",
        [nz, "left_starts", "left_ends", "axes4"],
        "left_src",
        TensorProto.FLOAT,
        [1, 9, 1, SHIFT_W],
    )
    left = g.add_node(
        "Pad",
        [left_src],
        "left",
        TensorProto.FLOAT,
        [1, 9, 1, WORK_W],
        pads=[0, 0, 0, 0, 0, 0, 0, 1],
    )
    right_src = g.add_node(
        "Slice",
        [nz, "right_starts", "right_ends", "axes4"],
        "right_src",
        TensorProto.FLOAT,
        [1, 9, 1, SHIFT_W],
    )
    right = g.add_node(
        "Pad",
        [right_src],
        "right",
        TensorProto.FLOAT,
        [1, 9, 1, WORK_W],
        pads=[0, 0, 0, 1, 0, 0, 0, 0],
    )
    return g.add_node("Add", [left, right], "odd_nz_raw", TensorProto.FLOAT, [1, 9, 1, WORK_W])


def shifted_nz_concat_bool(g: Graph, source: str = "row_b", source_is_nz: bool = False) -> str:
    g.add_init(numpy_helper.from_array(np.zeros((1, 9, 1, 1), dtype=bool), "false_col9"))
    if source_is_nz:
        nz = source
    else:
        nz = g.add_node(
            "Slice",
            [source, "nz_starts", "nz_ends", "axes4"],
            "nz_b",
            TensorProto.BOOL,
            [1, 9, 1, WORK_W],
        )
    left_src = g.add_node(
        "Slice",
        [nz, "left_starts", "left_ends", "axes4"],
        "left_src_b",
        TensorProto.BOOL,
        [1, 9, 1, SHIFT_W],
    )
    left = g.add_node(
        "Concat",
        [left_src, "false_col9"],
        "left_b",
        TensorProto.BOOL,
        [1, 9, 1, WORK_W],
        axis=3,
    )
    right_src = g.add_node(
        "Slice",
        [nz, "right_starts", "right_ends", "axes4"],
        "right_src_b",
        TensorProto.BOOL,
        [1, 9, 1, SHIFT_W],
    )
    right = g.add_node(
        "Concat",
        ["false_col9", right_src],
        "right_b",
        TensorProto.BOOL,
        [1, 9, 1, WORK_W],
        axis=3,
    )
    return g.add_node("Or", [left, right], "odd_nz_raw_b", TensorProto.BOOL, [1, 9, 1, WORK_W])


def shifted_nz_concat(g: Graph) -> str:
    g.add_init(init_f32("zero_col9", np.zeros((1, 9, 1, 1), dtype=np.float32)))
    nz = g.add_node(
        "Slice",
        ["row", "nz_starts", "nz_ends", "axes4"],
        "nz",
        TensorProto.FLOAT,
        [1, 9, 1, WORK_W],
    )
    left_src = g.add_node(
        "Slice",
        [nz, "left_starts", "left_ends", "axes4"],
        "left_src",
        TensorProto.FLOAT,
        [1, 9, 1, SHIFT_W],
    )
    left = g.add_node(
        "Concat",
        [left_src, "zero_col9"],
        "left",
        TensorProto.FLOAT,
        [1, 9, 1, WORK_W],
        axis=3,
    )
    right_src = g.add_node(
        "Slice",
        [nz, "right_starts", "right_ends", "axes4"],
        "right_src",
        TensorProto.FLOAT,
        [1, 9, 1, SHIFT_W],
    )
    right = g.add_node(
        "Concat",
        ["zero_col9", right_src],
        "right",
        TensorProto.FLOAT,
        [1, 9, 1, WORK_W],
        axis=3,
    )
    return g.add_node("Add", [left, right], "odd_nz_raw", TensorProto.FLOAT, [1, 9, 1, WORK_W])


def build_bool_model() -> onnx.ModelProto:
    g = Graph()
    base_constants(g)

    inp = vi("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = vi("output", TensorProto.FLOAT, [1, 10, 30, 30])

    row = g.add_node(
        "Slice",
        ["input", "row_starts", "row_ends", "axes4"],
        "row",
        TensorProto.FLOAT,
        [1, 10, 1, WORK_W],
    )
    valid_sum = g.add_node("ReduceSum", [row], "valid_sum", TensorProto.FLOAT, [1, 1, 1, WORK_W], axes=[1], keepdims=1)
    valid = g.add_node("Greater", [valid_sum, "zero"], "valid", TensorProto.BOOL, [1, 1, 1, WORK_W])
    row_b = g.add_node("Greater", [row, "zero"], "row_b", TensorProto.BOOL, [1, 10, 1, WORK_W])

    odd_nz_raw = shifted_nz_concat_bool(g)
    odd_nz_raw_f = g.add_node(
        "Cast",
        [odd_nz_raw],
        "odd_nz_raw_f",
        TensorProto.FLOAT,
        [1, 9, 1, WORK_W],
        to=TensorProto.FLOAT,
    )
    odd_nz_sum = g.add_node(
        "ReduceSum",
        [odd_nz_raw_f],
        "odd_nz_sum",
        TensorProto.FLOAT,
        [1, 1, 1, WORK_W],
        axes=[1],
        keepdims=1,
    )
    odd_has_nz = g.add_node("Greater", [odd_nz_sum, "zero"], "odd_has_nz", TensorProto.BOOL, [1, 1, 1, WORK_W])
    odd_bg_not_nz = g.add_node("Not", [odd_has_nz], "odd_bg_not_nz", TensorProto.BOOL, [1, 1, 1, WORK_W])
    odd_bg = g.add_node("And", [valid, odd_bg_not_nz], "odd_bg_b", TensorProto.BOOL, [1, 1, 1, WORK_W])
    odd_nz = g.add_node("And", [odd_nz_raw, valid], "odd_nz_b", TensorProto.BOOL, [1, 9, 1, WORK_W])
    odd = g.add_node("Concat", [odd_bg, odd_nz], "odd_b", TensorProto.BOOL, [1, 10, 1, WORK_W], axis=1)
    rows6_b = g.add_node(
        "Concat",
        [row_b, odd, row_b, odd, row_b, odd],
        "rows6_b",
        TensorProto.BOOL,
        [1, 10, 6, WORK_W],
        axis=2,
    )
    rows6 = g.add_node("Cast", [rows6_b], "rows6", TensorProto.FLOAT, [1, 10, 6, WORK_W], to=TensorProto.FLOAT)
    g.nodes.append(helper.make_node("Pad", [rows6], ["output"], pads=[0, 0, 0, 0, 0, 0, 24, 30 - WORK_W]))

    graph = helper.make_graph(
        g.nodes,
        f"{TASK_ID}_bool_pad_shift",
        [inp],
        [out],
        initializer=g.inits,
        value_info=g.value_infos,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    return model


def build_bool_odd_model() -> onnx.ModelProto:
    g = Graph()
    base_constants(g)

    inp = vi("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = vi("output", TensorProto.FLOAT, [1, 10, 30, 30])

    row = g.add_node(
        "Slice",
        ["input", "row_starts", "row_ends", "axes4"],
        "row",
        TensorProto.FLOAT,
        [1, 10, 1, WORK_W],
    )
    valid_sum = g.add_node("ReduceSum", [row], "valid_sum", TensorProto.FLOAT, [1, 1, 1, WORK_W], axes=[1], keepdims=1)
    valid = g.add_node("Greater", [valid_sum, "zero"], "valid", TensorProto.BOOL, [1, 1, 1, WORK_W])
    nz_f = g.add_node("Slice", [row, "nz_starts", "nz_ends", "axes4"], "nz_f", TensorProto.FLOAT, [1, 9, 1, WORK_W])
    nz_b = g.add_node("Greater", [nz_f, "zero"], "nz_b_full", TensorProto.BOOL, [1, 9, 1, WORK_W])

    odd_nz_raw = shifted_nz_concat_bool(g, "nz_b_full", source_is_nz=True)
    odd_nz_raw_f = g.add_node(
        "Cast",
        [odd_nz_raw],
        "odd_nz_raw_f",
        TensorProto.FLOAT,
        [1, 9, 1, WORK_W],
        to=TensorProto.FLOAT,
    )
    odd_nz_sum = g.add_node(
        "ReduceSum",
        [odd_nz_raw_f],
        "odd_nz_sum",
        TensorProto.FLOAT,
        [1, 1, 1, WORK_W],
        axes=[1],
        keepdims=1,
    )
    odd_has_nz = g.add_node("Greater", [odd_nz_sum, "zero"], "odd_has_nz", TensorProto.BOOL, [1, 1, 1, WORK_W])
    odd_bg_not_nz = g.add_node("Not", [odd_has_nz], "odd_bg_not_nz", TensorProto.BOOL, [1, 1, 1, WORK_W])
    odd_bg = g.add_node("And", [valid, odd_bg_not_nz], "odd_bg_b", TensorProto.BOOL, [1, 1, 1, WORK_W])
    odd_nz = g.add_node("And", [odd_nz_raw, valid], "odd_nz_b", TensorProto.BOOL, [1, 9, 1, WORK_W])
    odd_b = g.add_node("Concat", [odd_bg, odd_nz], "odd_b", TensorProto.BOOL, [1, 10, 1, WORK_W], axis=1)
    odd = g.add_node("Cast", [odd_b], "odd", TensorProto.FLOAT, [1, 10, 1, WORK_W], to=TensorProto.FLOAT)
    rows6 = g.add_node(
        "Concat",
        [row, odd, row, odd, row, odd],
        "rows6",
        TensorProto.FLOAT,
        [1, 10, 6, WORK_W],
        axis=2,
    )
    g.nodes.append(helper.make_node("Pad", [rows6], ["output"], pads=[0, 0, 0, 0, 0, 0, 24, 30 - WORK_W]))

    graph = helper.make_graph(
        g.nodes,
        f"{TASK_ID}_bool_odd_float_rows",
        [inp],
        [out],
        initializer=g.inits,
        value_info=g.value_infos,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    return model


def build_bool_ch0_model() -> onnx.ModelProto:
    g = Graph()
    base_constants(g)
    g.add_init(init_i64("ch0_start", [0]))
    g.add_init(init_i64("ch0_end", [1]))
    g.add_init(init_i64("axis_c", [1]))
    g.add_init(numpy_helper.from_array(np.zeros((1, 1, 1, 1), dtype=bool), "false_col1"))
    g.add_init(numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=bool), "true_col1"))

    inp = vi("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = vi("output", TensorProto.FLOAT, [1, 10, 30, 30])

    row = g.add_node(
        "Slice",
        ["input", "row_starts", "row_ends", "axes4"],
        "row",
        TensorProto.FLOAT,
        [1, 10, 1, WORK_W],
    )
    valid_sum = g.add_node("ReduceSum", [row], "valid_sum", TensorProto.FLOAT, [1, 1, 1, WORK_W], axes=[1], keepdims=1)
    valid = g.add_node("Greater", [valid_sum, "zero"], "valid", TensorProto.BOOL, [1, 1, 1, WORK_W])
    row_b = g.add_node("Greater", [row, "zero"], "row_b", TensorProto.BOOL, [1, 10, 1, WORK_W])

    odd_nz = shifted_nz_concat_bool(g)

    ch0 = g.add_node("Slice", [row_b, "ch0_start", "ch0_end", "axis_c"], "ch0_b", TensorProto.BOOL, [1, 1, 1, WORK_W])
    left_bg_src = g.add_node(
        "Slice",
        [ch0, "left_starts", "left_ends", "axes4"],
        "left_bg_src",
        TensorProto.BOOL,
        [1, 1, 1, SHIFT_W],
    )
    left_bg = g.add_node(
        "Concat",
        [left_bg_src, "true_col1"],
        "left_ch0_bg",
        TensorProto.BOOL,
        [1, 1, 1, WORK_W],
        axis=3,
    )
    left_valid_src = g.add_node(
        "Slice",
        [valid, "left_starts", "left_ends", "axes4"],
        "left_valid_src",
        TensorProto.BOOL,
        [1, 1, 1, SHIFT_W],
    )
    left_valid = g.add_node(
        "Concat",
        [left_valid_src, "false_col1"],
        "left_valid",
        TensorProto.BOOL,
        [1, 1, 1, WORK_W],
        axis=3,
    )
    left_invalid = g.add_node("Not", [left_valid], "left_invalid", TensorProto.BOOL, [1, 1, 1, WORK_W])
    left_bg = g.add_node("Or", [left_bg, left_invalid], "left_bg", TensorProto.BOOL, [1, 1, 1, WORK_W])
    right_bg_src = g.add_node(
        "Slice",
        [ch0, "right_starts", "right_ends", "axes4"],
        "right_bg_src",
        TensorProto.BOOL,
        [1, 1, 1, SHIFT_W],
    )
    right_bg = g.add_node(
        "Concat",
        ["true_col1", right_bg_src],
        "right_ch0_bg",
        TensorProto.BOOL,
        [1, 1, 1, WORK_W],
        axis=3,
    )
    right_valid_src = g.add_node(
        "Slice",
        [valid, "right_starts", "right_ends", "axes4"],
        "right_valid_src",
        TensorProto.BOOL,
        [1, 1, 1, SHIFT_W],
    )
    right_valid = g.add_node(
        "Concat",
        ["false_col1", right_valid_src],
        "right_valid",
        TensorProto.BOOL,
        [1, 1, 1, WORK_W],
        axis=3,
    )
    right_invalid = g.add_node("Not", [right_valid], "right_invalid", TensorProto.BOOL, [1, 1, 1, WORK_W])
    right_bg = g.add_node("Or", [right_bg, right_invalid], "right_bg", TensorProto.BOOL, [1, 1, 1, WORK_W])
    both_bg = g.add_node("And", [left_bg, right_bg], "both_bg", TensorProto.BOOL, [1, 1, 1, WORK_W])
    odd_bg = g.add_node("And", [valid, both_bg], "odd_bg_b", TensorProto.BOOL, [1, 1, 1, WORK_W])
    odd = g.add_node("Concat", [odd_bg, odd_nz], "odd_b", TensorProto.BOOL, [1, 10, 1, WORK_W], axis=1)
    rows6_b = g.add_node(
        "Concat",
        [row_b, odd, row_b, odd, row_b, odd],
        "rows6_b",
        TensorProto.BOOL,
        [1, 10, 6, WORK_W],
        axis=2,
    )
    rows6 = g.add_node("Cast", [rows6_b], "rows6", TensorProto.FLOAT, [1, 10, 6, WORK_W], to=TensorProto.FLOAT)
    g.nodes.append(helper.make_node("Pad", [rows6], ["output"], pads=[0, 0, 0, 0, 0, 0, 24, 30 - WORK_W]))

    graph = helper.make_graph(
        g.nodes,
        f"{TASK_ID}_bool_ch0_bg",
        [inp],
        [out],
        initializer=g.inits,
        value_info=g.value_infos,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    return model


def build_float_rows_ch0_model() -> onnx.ModelProto:
    g = Graph()
    base_constants(g)
    g.add_init(init_i64("ch0_start", [0]))
    g.add_init(init_i64("ch0_end", [1]))
    g.add_init(init_i64("axis_c", [1]))
    g.add_init(numpy_helper.from_array(np.zeros((1, 1, 1, 1), dtype=bool), "false_col1"))
    g.add_init(numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=bool), "true_col1"))

    inp = vi("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = vi("output", TensorProto.FLOAT, [1, 10, 30, 30])

    row = g.add_node(
        "Slice",
        ["input", "row_starts", "row_ends", "axes4"],
        "row",
        TensorProto.FLOAT,
        [1, 10, 1, WORK_W],
    )
    valid_sum = g.add_node("ReduceSum", [row], "valid_sum", TensorProto.FLOAT, [1, 1, 1, WORK_W], axes=[1], keepdims=1)
    valid = g.add_node("Greater", [valid_sum, "zero"], "valid", TensorProto.BOOL, [1, 1, 1, WORK_W])
    row_b = g.add_node("Greater", [row, "zero"], "row_b", TensorProto.BOOL, [1, 10, 1, WORK_W])

    odd_nz = shifted_nz_concat_bool(g)

    ch0 = g.add_node("Slice", [row_b, "ch0_start", "ch0_end", "axis_c"], "ch0_b", TensorProto.BOOL, [1, 1, 1, WORK_W])
    left_bg_src = g.add_node(
        "Slice",
        [ch0, "left_starts", "left_ends", "axes4"],
        "left_bg_src",
        TensorProto.BOOL,
        [1, 1, 1, SHIFT_W],
    )
    left_bg = g.add_node(
        "Concat",
        [left_bg_src, "true_col1"],
        "left_ch0_bg",
        TensorProto.BOOL,
        [1, 1, 1, WORK_W],
        axis=3,
    )
    left_valid_src = g.add_node(
        "Slice",
        [valid, "left_starts", "left_ends", "axes4"],
        "left_valid_src",
        TensorProto.BOOL,
        [1, 1, 1, SHIFT_W],
    )
    left_valid = g.add_node(
        "Concat",
        [left_valid_src, "false_col1"],
        "left_valid",
        TensorProto.BOOL,
        [1, 1, 1, WORK_W],
        axis=3,
    )
    left_invalid = g.add_node("Not", [left_valid], "left_invalid", TensorProto.BOOL, [1, 1, 1, WORK_W])
    left_bg = g.add_node("Or", [left_bg, left_invalid], "left_bg", TensorProto.BOOL, [1, 1, 1, WORK_W])
    right_bg_src = g.add_node(
        "Slice",
        [ch0, "right_starts", "right_ends", "axes4"],
        "right_bg_src",
        TensorProto.BOOL,
        [1, 1, 1, SHIFT_W],
    )
    right_bg = g.add_node(
        "Concat",
        ["true_col1", right_bg_src],
        "right_ch0_bg",
        TensorProto.BOOL,
        [1, 1, 1, WORK_W],
        axis=3,
    )
    right_valid_src = g.add_node(
        "Slice",
        [valid, "right_starts", "right_ends", "axes4"],
        "right_valid_src",
        TensorProto.BOOL,
        [1, 1, 1, SHIFT_W],
    )
    right_valid = g.add_node(
        "Concat",
        ["false_col1", right_valid_src],
        "right_valid",
        TensorProto.BOOL,
        [1, 1, 1, WORK_W],
        axis=3,
    )
    right_invalid = g.add_node("Not", [right_valid], "right_invalid", TensorProto.BOOL, [1, 1, 1, WORK_W])
    right_bg = g.add_node("Or", [right_bg, right_invalid], "right_bg", TensorProto.BOOL, [1, 1, 1, WORK_W])
    both_bg = g.add_node("And", [left_bg, right_bg], "both_bg", TensorProto.BOOL, [1, 1, 1, WORK_W])
    odd_bg = g.add_node("And", [valid, both_bg], "odd_bg_b", TensorProto.BOOL, [1, 1, 1, WORK_W])
    odd_b = g.add_node("Concat", [odd_bg, odd_nz], "odd_b", TensorProto.BOOL, [1, 10, 1, WORK_W], axis=1)
    odd = g.add_node("Cast", [odd_b], "odd", TensorProto.FLOAT, [1, 10, 1, WORK_W], to=TensorProto.FLOAT)
    rows6 = g.add_node(
        "Concat",
        [row, odd, row, odd, row, odd],
        "rows6",
        TensorProto.FLOAT,
        [1, 10, 6, WORK_W],
        axis=2,
    )
    g.nodes.append(helper.make_node("Pad", [rows6], ["output"], pads=[0, 0, 0, 0, 0, 0, 24, 30 - WORK_W]))

    graph = helper.make_graph(
        g.nodes,
        f"{TASK_ID}_float_rows_ch0_bg",
        [inp],
        [out],
        initializer=g.inits,
        value_info=g.value_infos,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    return model


def build_float_rows_colored_model() -> onnx.ModelProto:
    g = Graph()
    base_constants(g)
    g.add_init(init_i64("ch0_start", [0]))
    g.add_init(init_i64("ch0_end", [1]))
    g.add_init(init_i64("axis_c", [1]))
    g.add_init(numpy_helper.from_array(np.zeros((1, 1, 1, 1), dtype=bool), "false_col1"))

    inp = vi("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = vi("output", TensorProto.FLOAT, [1, 10, 30, 30])

    row = g.add_node(
        "Slice",
        ["input", "row_starts", "row_ends", "axes4"],
        "row",
        TensorProto.FLOAT,
        [1, 10, 1, WORK_W],
    )
    valid_sum = g.add_node("ReduceSum", [row], "valid_sum", TensorProto.FLOAT, [1, 1, 1, WORK_W], axes=[1], keepdims=1)
    valid = g.add_node("Greater", [valid_sum, "zero"], "valid", TensorProto.BOOL, [1, 1, 1, WORK_W])
    row_b = g.add_node("Greater", [row, "zero"], "row_b", TensorProto.BOOL, [1, 10, 1, WORK_W])

    odd_nz = shifted_nz_concat_bool(g)

    ch0 = g.add_node("Slice", [row_b, "ch0_start", "ch0_end", "axis_c"], "ch0_b", TensorProto.BOOL, [1, 1, 1, WORK_W])
    not_ch0 = g.add_node("Not", [ch0], "not_ch0", TensorProto.BOOL, [1, 1, 1, WORK_W])
    colored = g.add_node("And", [valid, not_ch0], "colored", TensorProto.BOOL, [1, 1, 1, WORK_W])
    left_colored_src = g.add_node(
        "Slice",
        [colored, "left_starts", "left_ends", "axes4"],
        "left_colored_src",
        TensorProto.BOOL,
        [1, 1, 1, SHIFT_W],
    )
    left_colored = g.add_node(
        "Concat",
        [left_colored_src, "false_col1"],
        "left_colored",
        TensorProto.BOOL,
        [1, 1, 1, WORK_W],
        axis=3,
    )
    right_colored_src = g.add_node(
        "Slice",
        [colored, "right_starts", "right_ends", "axes4"],
        "right_colored_src",
        TensorProto.BOOL,
        [1, 1, 1, SHIFT_W],
    )
    right_colored = g.add_node(
        "Concat",
        ["false_col1", right_colored_src],
        "right_colored",
        TensorProto.BOOL,
        [1, 1, 1, WORK_W],
        axis=3,
    )
    odd_has_colored = g.add_node("Or", [left_colored, right_colored], "odd_has_colored", TensorProto.BOOL, [1, 1, 1, WORK_W])
    odd_bg_not_colored = g.add_node("Not", [odd_has_colored], "odd_bg_not_colored", TensorProto.BOOL, [1, 1, 1, WORK_W])
    odd_bg = g.add_node("And", [valid, odd_bg_not_colored], "odd_bg_b", TensorProto.BOOL, [1, 1, 1, WORK_W])
    odd_b = g.add_node("Concat", [odd_bg, odd_nz], "odd_b", TensorProto.BOOL, [1, 10, 1, WORK_W], axis=1)
    odd = g.add_node("Cast", [odd_b], "odd", TensorProto.FLOAT, [1, 10, 1, WORK_W], to=TensorProto.FLOAT)
    rows6 = g.add_node(
        "Concat",
        [row, odd, row, odd, row, odd],
        "rows6",
        TensorProto.FLOAT,
        [1, 10, 6, WORK_W],
        axis=2,
    )
    g.nodes.append(helper.make_node("Pad", [rows6], ["output"], pads=[0, 0, 0, 0, 0, 0, 24, 30 - WORK_W]))

    graph = helper.make_graph(
        g.nodes,
        f"{TASK_ID}_float_rows_colored_bg",
        [inp],
        [out],
        initializer=g.inits,
        value_info=g.value_infos,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    return model


def build_cast_rows_colored_model() -> onnx.ModelProto:
    g = Graph()
    base_constants(g, include_zero=False)
    g.add_init(init_i64("ch0_start", [0]))
    g.add_init(init_i64("ch0_end", [1]))
    g.add_init(init_i64("axis_c", [1]))
    g.add_init(numpy_helper.from_array(np.zeros((1, 1, 1, 1), dtype=bool), "false_col1"))

    inp = vi("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = vi("output", TensorProto.FLOAT, [1, 10, 30, 30])

    row = g.add_node(
        "Slice",
        ["input", "row_starts", "row_ends", "axes4"],
        "row",
        TensorProto.FLOAT,
        [1, 10, 1, WORK_W],
    )
    valid_sum = g.add_node("ReduceSum", [row], "valid_sum", TensorProto.FLOAT, [1, 1, 1, WORK_W], axes=[1], keepdims=1)
    valid = g.add_node("Cast", [valid_sum], "valid", TensorProto.BOOL, [1, 1, 1, WORK_W], to=TensorProto.BOOL)
    row_b = g.add_node("Cast", [row], "row_b", TensorProto.BOOL, [1, 10, 1, WORK_W], to=TensorProto.BOOL)

    odd_nz = shifted_nz_concat_bool(g)

    ch0 = g.add_node("Slice", [row_b, "ch0_start", "ch0_end", "axis_c"], "ch0_b", TensorProto.BOOL, [1, 1, 1, WORK_W])
    not_ch0 = g.add_node("Not", [ch0], "not_ch0", TensorProto.BOOL, [1, 1, 1, WORK_W])
    colored = g.add_node("And", [valid, not_ch0], "colored", TensorProto.BOOL, [1, 1, 1, WORK_W])
    left_colored_src = g.add_node(
        "Slice",
        [colored, "left_starts", "left_ends", "axes4"],
        "left_colored_src",
        TensorProto.BOOL,
        [1, 1, 1, SHIFT_W],
    )
    left_colored = g.add_node(
        "Concat",
        [left_colored_src, "false_col1"],
        "left_colored",
        TensorProto.BOOL,
        [1, 1, 1, WORK_W],
        axis=3,
    )
    right_colored_src = g.add_node(
        "Slice",
        [colored, "right_starts", "right_ends", "axes4"],
        "right_colored_src",
        TensorProto.BOOL,
        [1, 1, 1, SHIFT_W],
    )
    right_colored = g.add_node(
        "Concat",
        ["false_col1", right_colored_src],
        "right_colored",
        TensorProto.BOOL,
        [1, 1, 1, WORK_W],
        axis=3,
    )
    odd_has_colored = g.add_node("Or", [left_colored, right_colored], "odd_has_colored", TensorProto.BOOL, [1, 1, 1, WORK_W])
    odd_bg_not_colored = g.add_node("Not", [odd_has_colored], "odd_bg_not_colored", TensorProto.BOOL, [1, 1, 1, WORK_W])
    odd_bg = g.add_node("And", [valid, odd_bg_not_colored], "odd_bg_b", TensorProto.BOOL, [1, 1, 1, WORK_W])
    odd_b = g.add_node("Concat", [odd_bg, odd_nz], "odd_b", TensorProto.BOOL, [1, 10, 1, WORK_W], axis=1)
    odd = g.add_node("Cast", [odd_b], "odd", TensorProto.FLOAT, [1, 10, 1, WORK_W], to=TensorProto.FLOAT)
    rows6 = g.add_node(
        "Concat",
        [row, odd, row, odd, row, odd],
        "rows6",
        TensorProto.FLOAT,
        [1, 10, 6, WORK_W],
        axis=2,
    )
    g.nodes.append(helper.make_node("Pad", [rows6], ["output"], pads=[0, 0, 0, 0, 0, 0, 24, 30 - WORK_W]))

    graph = helper.make_graph(
        g.nodes,
        f"{TASK_ID}_cast_rows_colored_bg",
        [inp],
        [out],
        initializer=g.inits,
        value_info=g.value_infos,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    return model


def build_model(variant: str) -> onnx.ModelProto:
    if variant == "bool_pad_shift":
        return build_bool_model()
    if variant == "bool_odd_float_rows":
        return build_bool_odd_model()
    if variant == "bool_ch0_bg":
        return build_bool_ch0_model()
    if variant == "float_rows_ch0_bg":
        return build_float_rows_ch0_model()
    if variant == "float_rows_colored_bg":
        return build_float_rows_colored_model()
    if variant == "cast_rows_colored_bg":
        return build_cast_rows_colored_model()

    g = Graph()
    base_constants(g)

    inp = vi("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = vi("output", TensorProto.FLOAT, [1, 10, 30, 30])

    row = g.add_node(
        "Slice",
        ["input", "row_starts", "row_ends", "axes4"],
        "row",
        TensorProto.FLOAT,
        [1, 10, 1, WORK_W],
    )
    valid_sum = g.add_node("ReduceSum", [row], "valid_sum", TensorProto.FLOAT, [1, 1, 1, WORK_W], axes=[1], keepdims=1)
    valid = g.add_node("Greater", [valid_sum, "zero"], "valid", TensorProto.BOOL, [1, 1, 1, WORK_W])
    valid_f = g.add_node("Cast", [valid], "valid_f", TensorProto.FLOAT, [1, 1, 1, WORK_W], to=TensorProto.FLOAT)

    odd_nz_raw = shifted_nz_concat(g) if variant == "concat_shift" else shifted_nz_pad(g)
    odd_nz = g.add_node("Mul", [odd_nz_raw, valid_f], "odd_nz", TensorProto.FLOAT, [1, 9, 1, WORK_W])
    odd_nz_sum = g.add_node(
        "ReduceSum",
        [odd_nz],
        "odd_nz_sum",
        TensorProto.FLOAT,
        [1, 1, 1, WORK_W],
        axes=[1],
        keepdims=1,
    )
    odd_has_nz = g.add_node("Greater", [odd_nz_sum, "zero"], "odd_has_nz", TensorProto.BOOL, [1, 1, 1, WORK_W])
    odd_has_nz_f = g.add_node(
        "Cast",
        [odd_has_nz],
        "odd_has_nz_f",
        TensorProto.FLOAT,
        [1, 1, 1, WORK_W],
        to=TensorProto.FLOAT,
    )
    odd_bg = g.add_node("Sub", [valid_f, odd_has_nz_f], "odd_bg", TensorProto.FLOAT, [1, 1, 1, WORK_W])
    odd = g.add_node("Concat", [odd_bg, odd_nz], "odd", TensorProto.FLOAT, [1, 10, 1, WORK_W], axis=1)

    rows6 = g.add_node(
        "Concat",
        [row, odd, row, odd, row, odd],
        "rows6",
        TensorProto.FLOAT,
        [1, 10, 6, WORK_W],
        axis=2,
    )
    if variant == "concat_final":
        g.add_init(init_f32("right_zero", np.zeros((1, 10, 6, 30 - WORK_W), dtype=np.float32)))
        g.add_init(init_f32("tail_zero", np.zeros((1, 10, 24, 30), dtype=np.float32)))
        rows6_wide = g.add_node(
            "Concat",
            [rows6, "right_zero"],
            "rows6_wide",
            TensorProto.FLOAT,
            [1, 10, 6, 30],
            axis=3,
        )
        g.nodes.append(helper.make_node("Concat", [rows6_wide, "tail_zero"], ["output"], axis=2))
    else:
        g.nodes.append(helper.make_node("Pad", [rows6], ["output"], pads=[0, 0, 0, 0, 0, 0, 24, 30 - WORK_W]))

    graph = helper.make_graph(
        g.nodes,
        f"{TASK_ID}_{variant}",
        [inp],
        [out],
        initializer=g.inits,
        value_info=g.value_infos,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    return model


def load_examples() -> list[dict[str, list[list[int]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [example for split in ("train", "test", "arc-gen") for example in data.get(split, [])]


def verify_model(model: onnx.ModelProto) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    for idx, example in enumerate(load_examples()):
        x = convert_to_numpy(example, "input")
        y = convert_to_numpy(example, "output")
        if x is None or y is None:
            continue
        pred = sess.run(["output"], {"input": x})[0]
        actual = pred > 0.0
        expected = y > 0.0
        if not np.array_equal(actual, expected):
            coords = np.argwhere(actual != expected)
            first = tuple(int(v) for v in coords[0])
            raise AssertionError(f"example {idx} mismatch at {first}: pred={pred[first]:.3g}, expected={y[first]:.3g}")


def evaluate_variant(variant: str) -> dict[str, Any]:
    model = build_model(variant)
    verify_model(model)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{TASK_ID}.onnx"
        onnx.save(model, path)
        result = score_file(path)
        if not result["valid"]:
            raise RuntimeError(f"{variant} invalid: {result['error']}")
        result["variant"] = variant
        result["model"] = model
        return result


def main() -> None:
    variants = [
        "pad_shift",
        "concat_shift",
        "concat_final",
        "bool_pad_shift",
        "bool_odd_float_rows",
        "bool_ch0_bg",
        "float_rows_ch0_bg",
        "float_rows_colored_bg",
        "cast_rows_colored_bg",
    ]
    results = [evaluate_variant(variant) for variant in variants]
    best = min(results, key=lambda item: (int(item["cost"]), int(item["params"]), str(item["variant"])))
    onnx.save(best["model"], BEST_PATH)

    print(f"wrote {BEST_PATH}")
    for result in results:
        print(
            f"{result['variant']}: memory={result['memory']} params={result['params']} "
            f"cost={result['cost']} score={float(result['score']):.6f}"
        )
    print(
        f"best={best['variant']} memory={best['memory']} params={best['params']} "
        f"cost={best['cost']} score={float(best['score']):.6f}"
    )


if __name__ == "__main__":
    main()
