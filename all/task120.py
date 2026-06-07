"""Compact ONNX solver for ARC task120.

Task rule: each example contains solid non-black rectangles on a black
background, with grids up to 16x15 in the provided train/test/arc-gen set.
Keep every rectangle in place, preserve its original colored border, and
recolor cells strictly inside a non-black rectangle to cyan/color 8. Background
cells remain black, and padded cells outside each example's actual grid remain
all-zero.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task120"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
OPSET = 10
ACTIVE_H = 16
ACTIVE_W = 15


def _init(array: np.ndarray | list[int] | list[bool], name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(array), name=name)


def _model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _common_bool_graph(
    cast_full_input: bool,
    pad_bool_before_cast: bool,
    where_output: bool = False,
) -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        _init([0, 0, 0, 0], "s0"),
        _init([1, 10, ACTIVE_H, ACTIVE_W], "e_active"),
        _init([0, 1, 2, 3], "axes4"),
        _init([0, 1, 0, 0], "s_fg"),
        _init([1, 10, ACTIVE_H, ACTIVE_W], "e_fg"),
        _init([0, 0, 0, 0], "s_up_src"),
        _init([1, 1, ACTIVE_H - 1, ACTIVE_W], "e_up_src"),
        _init([0, 0, 1, 0], "s_down_src"),
        _init([1, 1, ACTIVE_H, ACTIVE_W], "e_down_src"),
        _init([0, 0, 0, 0], "s_left_src"),
        _init([1, 1, ACTIVE_H, ACTIVE_W - 1], "e_left_src"),
        _init([0, 0, 0, 1], "s_right_src"),
        _init([1, 1, ACTIVE_H, ACTIVE_W], "e_right_src"),
        _init(np.zeros((1, 1, 1, ACTIVE_W), dtype=np.bool_), "false_row"),
        _init(np.zeros((1, 1, ACTIVE_H, 1), dtype=np.bool_), "false_col"),
    ]
    if where_output:
        cyan = np.zeros((1, 10, 1, 1), dtype=np.bool_)
        cyan[0, 8, 0, 0] = True
        inits.extend(
            [
                _init(np.zeros((1, 10, 1, 1), dtype=np.bool_), "false10"),
                _init(cyan, "cyan10"),
            ]
        )
    else:
        inits.extend(
            [
                _init([0, 0, 0, 0], "s_pre8"),
                _init([1, 8, ACTIVE_H, ACTIVE_W], "e_pre8"),
                _init([0, 8, 0, 0], "s_c8"),
                _init([1, 9, ACTIVE_H, ACTIVE_W], "e_c8"),
                _init([0, 9, 0, 0], "s_c9"),
                _init([1, 10, ACTIVE_H, ACTIVE_W], "e_c9"),
            ]
        )
    nodes: list[onnx.NodeProto] = []

    if cast_full_input:
        nodes.extend(
            [
                helper.make_node("Cast", [IN_NAME], ["input_b"], to=TensorProto.BOOL),
                helper.make_node("Slice", ["input_b", "s0", "e_active", "axes4"], ["active_b"]),
            ]
        )
    else:
        inits.append(_init(np.asarray([0.0], dtype=np.float32), "zero"))
        nodes.extend(
            [
                helper.make_node("Slice", [IN_NAME, "s0", "e_active", "axes4"], ["active_f"]),
                helper.make_node("Greater", ["active_f", "zero"], ["active_b"]),
            ]
        )

    nodes.extend(
        [
            helper.make_node("Slice", ["active_b", "s_fg", "e_fg", "axes4"], ["fg_b"]),
            helper.make_node(
                "Split",
                ["fg_b"],
                ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8_fg", "c9_fg"],
                axis=1,
                split=[1] * 9,
            ),
            helper.make_node("Or", ["c1", "c2"], ["nz12"]),
            helper.make_node("Or", ["nz12", "c3"], ["nz123"]),
            helper.make_node("Or", ["nz123", "c4"], ["nz1234"]),
            helper.make_node("Or", ["nz1234", "c5"], ["nz12345"]),
            helper.make_node("Or", ["nz12345", "c6"], ["nz123456"]),
            helper.make_node("Or", ["nz123456", "c7"], ["nz1234567"]),
            helper.make_node("Or", ["nz1234567", "c8_fg"], ["nz12345678"]),
            helper.make_node("Or", ["nz12345678", "c9_fg"], ["nz"]),
            helper.make_node("Slice", ["nz", "s_up_src", "e_up_src", "axes4"], ["up_src"]),
            helper.make_node("Concat", ["false_row", "up_src"], ["up"], axis=2),
            helper.make_node("Slice", ["nz", "s_down_src", "e_down_src", "axes4"], ["down_src"]),
            helper.make_node("Concat", ["down_src", "false_row"], ["down"], axis=2),
            helper.make_node("Slice", ["nz", "s_left_src", "e_left_src", "axes4"], ["left_src"]),
            helper.make_node("Concat", ["false_col", "left_src"], ["left"], axis=3),
            helper.make_node("Slice", ["nz", "s_right_src", "e_right_src", "axes4"], ["right_src"]),
            helper.make_node("Concat", ["right_src", "false_col"], ["right"], axis=3),
            helper.make_node("And", ["nz", "up"], ["i0"]),
            helper.make_node("And", ["i0", "down"], ["i1"]),
            helper.make_node("And", ["i1", "left"], ["i2"]),
            helper.make_node("And", ["i2", "right"], ["interior"]),
        ]
    )
    if where_output:
        nodes.extend(
            [
                helper.make_node("Where", ["interior", "false10", "active_b"], ["cleared"]),
                helper.make_node("Where", ["interior", "cyan10", "cleared"], ["out_active_b"]),
            ]
        )
    else:
        nodes.extend(
            [
                helper.make_node("Not", ["interior"], ["not_interior"]),
            helper.make_node("Slice", ["active_b", "s_pre8", "e_pre8", "axes4"], ["pre8"]),
            helper.make_node("And", ["pre8", "not_interior"], ["pre8_out"]),
            helper.make_node("Slice", ["active_b", "s_c8", "e_c8", "axes4"], ["c8_in"]),
            helper.make_node("Or", ["c8_in", "interior"], ["c8_out"]),
            helper.make_node("Slice", ["active_b", "s_c9", "e_c9", "axes4"], ["c9_in"]),
            helper.make_node("And", ["c9_in", "not_interior"], ["c9_out"]),
            helper.make_node("Concat", ["pre8_out", "c8_out", "c9_out"], ["out_active_b"], axis=1),
            ]
        )
    if pad_bool_before_cast:
        nodes.extend(
            [
                helper.make_node(
                    "Pad",
                    ["out_active_b"],
                    ["out_padded_b"],
                    pads=[0, 0, 0, 0, 0, 0, 30 - ACTIVE_H, 30 - ACTIVE_W],
                ),
                helper.make_node("Cast", ["out_padded_b"], [OUT_NAME], to=TensorProto.FLOAT),
            ]
        )
        suffix = "boolpad"
    else:
        nodes.extend(
            [
                helper.make_node("Cast", ["out_active_b"], ["out_active_f"], to=TensorProto.FLOAT),
                helper.make_node(
                    "Pad",
                    ["out_active_f"],
                    [OUT_NAME],
                    pads=[0, 0, 0, 0, 0, 0, 30 - ACTIVE_H, 30 - ACTIVE_W],
                ),
            ]
        )
        suffix = "floatpad"
    prefix = "castfull" if cast_full_input else "slicefirst"
    if where_output:
        prefix += "_where"
    return _model(nodes, inits, f"{TASK_ID}_{prefix}_{suffix}")


def build_castfull_boolpad_model() -> onnx.ModelProto:
    return _common_bool_graph(cast_full_input=True, pad_bool_before_cast=True)


def build_castfull_floatpad_model() -> onnx.ModelProto:
    return _common_bool_graph(cast_full_input=True, pad_bool_before_cast=False)


def build_castfull_where_floatpad_model() -> onnx.ModelProto:
    return _common_bool_graph(cast_full_input=True, pad_bool_before_cast=False, where_output=True)


def build_slicefirst_floatpad_model() -> onnx.ModelProto:
    return _common_bool_graph(cast_full_input=False, pad_bool_before_cast=False)


def build_argmax_floatpad_model(use_casts: bool = False) -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        _init([0, 0, 0, 0], "s0"),
        _init([1, 10, ACTIVE_H, ACTIVE_W], "e_active"),
        _init([0, 1, 2, 3], "axes4"),
        _init([0, 0, 0, 0], "s_up_src"),
        _init([1, 1, ACTIVE_H - 1, ACTIVE_W], "e_up_src"),
        _init([0, 0, 1, 0], "s_down_src"),
        _init([1, 1, ACTIVE_H, ACTIVE_W], "e_down_src"),
        _init([0, 0, 0, 0], "s_left_src"),
        _init([1, 1, ACTIVE_H, ACTIVE_W - 1], "e_left_src"),
        _init([0, 0, 0, 1], "s_right_src"),
        _init([1, 1, ACTIVE_H, ACTIVE_W], "e_right_src"),
        _init(np.zeros((1, 1, 1, ACTIVE_W), dtype=np.bool_), "false_row"),
        _init(np.zeros((1, 1, ACTIVE_H, 1), dtype=np.bool_), "false_col"),
        _init([0, 0, 0, 0], "s_pre8"),
        _init([1, 8, ACTIVE_H, ACTIVE_W], "e_pre8"),
        _init([0, 8, 0, 0], "s_c8"),
        _init([1, 9, ACTIVE_H, ACTIVE_W], "e_c8"),
        _init([0, 9, 0, 0], "s_c9"),
        _init([1, 10, ACTIVE_H, ACTIVE_W], "e_c9"),
    ]
    if not use_casts:
        inits.extend(
            [
                _init(np.asarray([0.0], dtype=np.float32), "zero_f"),
                _init(np.asarray([0], dtype=np.int64), "zero_i"),
            ]
        )
    nodes = [
        helper.make_node("Slice", [IN_NAME, "s0", "e_active", "axes4"], ["active_f"]),
        helper.make_node("ArgMax", ["active_f"], ["color"], axis=1, keepdims=1),
    ]
    if use_casts:
        nodes.extend(
            [
                helper.make_node("Cast", ["active_f"], ["active_b"], to=TensorProto.BOOL),
                helper.make_node("Cast", ["color"], ["nz"], to=TensorProto.BOOL),
            ]
        )
    else:
        nodes.extend(
            [
                helper.make_node("Greater", ["active_f", "zero_f"], ["active_b"]),
                helper.make_node("Greater", ["color", "zero_i"], ["nz"]),
            ]
        )
    nodes.extend(
        [
        helper.make_node("Slice", ["nz", "s_up_src", "e_up_src", "axes4"], ["up_src"]),
        helper.make_node("Concat", ["false_row", "up_src"], ["up"], axis=2),
        helper.make_node("Slice", ["nz", "s_down_src", "e_down_src", "axes4"], ["down_src"]),
        helper.make_node("Concat", ["down_src", "false_row"], ["down"], axis=2),
        helper.make_node("Slice", ["nz", "s_left_src", "e_left_src", "axes4"], ["left_src"]),
        helper.make_node("Concat", ["false_col", "left_src"], ["left"], axis=3),
        helper.make_node("Slice", ["nz", "s_right_src", "e_right_src", "axes4"], ["right_src"]),
        helper.make_node("Concat", ["right_src", "false_col"], ["right"], axis=3),
        helper.make_node("And", ["nz", "up"], ["i0"]),
        helper.make_node("And", ["i0", "down"], ["i1"]),
        helper.make_node("And", ["i1", "left"], ["i2"]),
        helper.make_node("And", ["i2", "right"], ["interior"]),
        helper.make_node("Not", ["interior"], ["not_interior"]),
        helper.make_node("Slice", ["active_b", "s_pre8", "e_pre8", "axes4"], ["pre8"]),
        helper.make_node("And", ["pre8", "not_interior"], ["pre8_out"]),
        helper.make_node("Slice", ["active_b", "s_c8", "e_c8", "axes4"], ["c8_in"]),
        helper.make_node("Or", ["c8_in", "interior"], ["c8_out"]),
        helper.make_node("Slice", ["active_b", "s_c9", "e_c9", "axes4"], ["c9_in"]),
        helper.make_node("And", ["c9_in", "not_interior"], ["c9_out"]),
        helper.make_node("Concat", ["pre8_out", "c8_out", "c9_out"], ["out_active_b"], axis=1),
        helper.make_node("Cast", ["out_active_b"], ["out_active_f"], to=TensorProto.FLOAT),
        helper.make_node(
            "Pad",
            ["out_active_f"],
            [OUT_NAME],
            pads=[0, 0, 0, 0, 0, 0, 30 - ACTIVE_H, 30 - ACTIVE_W],
        ),
        ]
    )
    suffix = "argmax_cast_floatpad" if use_casts else "argmax_floatpad"
    return _model(nodes, inits, f"{TASK_ID}_{suffix}")


def build_argmax_cast_floatpad_model() -> onnx.ModelProto:
    return build_argmax_floatpad_model(use_casts=True)


def build_color123_floatpad_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        _init([0, 0, 0, 0], "s0"),
        _init([1, 4, ACTIVE_H, ACTIVE_W], "e_active4"),
        _init([0, 1, 2, 3], "axes4"),
        _init([0, 0, 0, 0], "s_up_src"),
        _init([1, 1, ACTIVE_H - 1, ACTIVE_W], "e_up_src"),
        _init([0, 0, 1, 0], "s_down_src"),
        _init([1, 1, ACTIVE_H, ACTIVE_W], "e_down_src"),
        _init([0, 0, 0, 0], "s_left_src"),
        _init([1, 1, ACTIVE_H, ACTIVE_W - 1], "e_left_src"),
        _init([0, 0, 0, 1], "s_right_src"),
        _init([1, 1, ACTIVE_H, ACTIVE_W], "e_right_src"),
        _init(np.zeros((1, 1, 1, ACTIVE_W), dtype=np.bool_), "false_row"),
        _init(np.zeros((1, 1, ACTIVE_H, 1), dtype=np.bool_), "false_col"),
    ]
    nodes = [
        helper.make_node("Slice", [IN_NAME, "s0", "e_active4", "axes4"], ["active4_f"]),
        helper.make_node("Cast", ["active4_f"], ["active4_b"], to=TensorProto.BOOL),
        helper.make_node("Split", ["active4_b"], ["c0", "c1", "c2", "c3"], axis=1, split=[1] * 4),
        helper.make_node("Or", ["c1", "c2"], ["nz12"]),
        helper.make_node("Or", ["nz12", "c3"], ["nz"]),
        helper.make_node("Slice", ["nz", "s_up_src", "e_up_src", "axes4"], ["up_src"]),
        helper.make_node("Concat", ["false_row", "up_src"], ["up"], axis=2),
        helper.make_node("Slice", ["nz", "s_down_src", "e_down_src", "axes4"], ["down_src"]),
        helper.make_node("Concat", ["down_src", "false_row"], ["down"], axis=2),
        helper.make_node("Slice", ["nz", "s_left_src", "e_left_src", "axes4"], ["left_src"]),
        helper.make_node("Concat", ["false_col", "left_src"], ["left"], axis=3),
        helper.make_node("Slice", ["nz", "s_right_src", "e_right_src", "axes4"], ["right_src"]),
        helper.make_node("Concat", ["right_src", "false_col"], ["right"], axis=3),
        helper.make_node("And", ["nz", "up"], ["i0"]),
        helper.make_node("And", ["i0", "down"], ["i1"]),
        helper.make_node("And", ["i1", "left"], ["i2"]),
        helper.make_node("And", ["i2", "right"], ["interior"]),
        helper.make_node("Not", ["interior"], ["not_interior"]),
        helper.make_node("And", ["c0", "c1"], ["zero"]),
        helper.make_node("And", ["c1", "not_interior"], ["c1_out"]),
        helper.make_node("And", ["c2", "not_interior"], ["c2_out"]),
        helper.make_node("And", ["c3", "not_interior"], ["c3_out"]),
        helper.make_node(
            "Concat",
            ["c0", "c1_out", "c2_out", "c3_out", "zero", "zero", "zero", "zero", "interior", "zero"],
            ["out_active_b"],
            axis=1,
        ),
        helper.make_node("Cast", ["out_active_b"], ["out_active_f"], to=TensorProto.FLOAT),
        helper.make_node(
            "Pad",
            ["out_active_f"],
            [OUT_NAME],
            pads=[0, 0, 0, 0, 0, 0, 30 - ACTIVE_H, 30 - ACTIVE_W],
        ),
    ]
    return _model(nodes, inits, f"{TASK_ID}_color123_floatpad")


def build_direct123_floatpad_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        _init([0, 0, 0, 0], "s_c0"),
        _init([1, 1, ACTIVE_H, ACTIVE_W], "e_c0"),
        _init([0, 1, 0, 0], "s_c1"),
        _init([1, 2, ACTIVE_H, ACTIVE_W], "e_c1"),
        _init([0, 2, 0, 0], "s_c2"),
        _init([1, 3, ACTIVE_H, ACTIVE_W], "e_c2"),
        _init([0, 3, 0, 0], "s_c3"),
        _init([1, 4, ACTIVE_H, ACTIVE_W], "e_c3"),
        _init([0, 1, 2, 3], "axes4"),
        _init([0, 0, 0, 0], "s_up_src"),
        _init([1, 1, ACTIVE_H - 1, ACTIVE_W], "e_up_src"),
        _init([0, 0, 1, 0], "s_down_src"),
        _init([1, 1, ACTIVE_H, ACTIVE_W], "e_down_src"),
        _init([0, 0, 0, 0], "s_left_src"),
        _init([1, 1, ACTIVE_H, ACTIVE_W - 1], "e_left_src"),
        _init([0, 0, 0, 1], "s_right_src"),
        _init([1, 1, ACTIVE_H, ACTIVE_W], "e_right_src"),
        _init(np.zeros((1, 1, 1, ACTIVE_W), dtype=np.bool_), "false_row"),
        _init(np.zeros((1, 1, ACTIVE_H, 1), dtype=np.bool_), "false_col"),
    ]
    nodes = [
        helper.make_node("Slice", [IN_NAME, "s_c0", "e_c0", "axes4"], ["c0_f"]),
        helper.make_node("Cast", ["c0_f"], ["c0"], to=TensorProto.BOOL),
        helper.make_node("Slice", [IN_NAME, "s_c1", "e_c1", "axes4"], ["c1_f"]),
        helper.make_node("Cast", ["c1_f"], ["c1"], to=TensorProto.BOOL),
        helper.make_node("Slice", [IN_NAME, "s_c2", "e_c2", "axes4"], ["c2_f"]),
        helper.make_node("Cast", ["c2_f"], ["c2"], to=TensorProto.BOOL),
        helper.make_node("Slice", [IN_NAME, "s_c3", "e_c3", "axes4"], ["c3_f"]),
        helper.make_node("Cast", ["c3_f"], ["c3"], to=TensorProto.BOOL),
        helper.make_node("Or", ["c1", "c2"], ["nz12"]),
        helper.make_node("Or", ["nz12", "c3"], ["nz"]),
        helper.make_node("Slice", ["nz", "s_up_src", "e_up_src", "axes4"], ["up_src"]),
        helper.make_node("Concat", ["false_row", "up_src"], ["up"], axis=2),
        helper.make_node("Slice", ["nz", "s_down_src", "e_down_src", "axes4"], ["down_src"]),
        helper.make_node("Concat", ["down_src", "false_row"], ["down"], axis=2),
        helper.make_node("Slice", ["nz", "s_left_src", "e_left_src", "axes4"], ["left_src"]),
        helper.make_node("Concat", ["false_col", "left_src"], ["left"], axis=3),
        helper.make_node("Slice", ["nz", "s_right_src", "e_right_src", "axes4"], ["right_src"]),
        helper.make_node("Concat", ["right_src", "false_col"], ["right"], axis=3),
        helper.make_node("And", ["nz", "up"], ["i0"]),
        helper.make_node("And", ["i0", "down"], ["i1"]),
        helper.make_node("And", ["i1", "left"], ["i2"]),
        helper.make_node("And", ["i2", "right"], ["interior"]),
        helper.make_node("Not", ["interior"], ["not_interior"]),
        helper.make_node("And", ["c0", "c1"], ["zero"]),
        helper.make_node("And", ["c1", "not_interior"], ["c1_out"]),
        helper.make_node("And", ["c2", "not_interior"], ["c2_out"]),
        helper.make_node("And", ["c3", "not_interior"], ["c3_out"]),
        helper.make_node(
            "Concat",
            ["c0", "c1_out", "c2_out", "c3_out", "zero", "zero", "zero", "zero", "interior", "zero"],
            ["out_active_b"],
            axis=1,
        ),
        helper.make_node("Cast", ["out_active_b"], ["out_active_f"], to=TensorProto.FLOAT),
        helper.make_node(
            "Pad",
            ["out_active_f"],
            [OUT_NAME],
            pads=[0, 0, 0, 0, 0, 0, 30 - ACTIVE_H, 30 - ACTIVE_W],
        ),
    ]
    return _model(nodes, inits, f"{TASK_ID}_direct123_floatpad")


def build_conv123_floatpad_model() -> onnx.ModelProto:
    kernel = np.asarray(
        [[[[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]]],
        dtype=np.float32,
    )
    inits: list[onnx.TensorProto] = [
        _init([0, 0, 0, 0], "s_c0"),
        _init([1, 1, ACTIVE_H, ACTIVE_W], "e_c0"),
        _init([0, 1, 0, 0], "s_c1"),
        _init([1, 2, ACTIVE_H, ACTIVE_W], "e_c1"),
        _init([0, 2, 0, 0], "s_c2"),
        _init([1, 3, ACTIVE_H, ACTIVE_W], "e_c2"),
        _init([0, 3, 0, 0], "s_c3"),
        _init([1, 4, ACTIVE_H, ACTIVE_W], "e_c3"),
        _init([0, 1, 2, 3], "axes4"),
        _init(kernel, "cross_kernel"),
        _init(np.asarray([4.5], dtype=np.float32), "four_half"),
    ]
    nodes = [
        helper.make_node("Slice", [IN_NAME, "s_c0", "e_c0", "axes4"], ["c0_f"]),
        helper.make_node("Cast", ["c0_f"], ["c0"], to=TensorProto.BOOL),
        helper.make_node("Slice", [IN_NAME, "s_c1", "e_c1", "axes4"], ["c1_f"]),
        helper.make_node("Cast", ["c1_f"], ["c1"], to=TensorProto.BOOL),
        helper.make_node("Slice", [IN_NAME, "s_c2", "e_c2", "axes4"], ["c2_f"]),
        helper.make_node("Cast", ["c2_f"], ["c2"], to=TensorProto.BOOL),
        helper.make_node("Slice", [IN_NAME, "s_c3", "e_c3", "axes4"], ["c3_f"]),
        helper.make_node("Cast", ["c3_f"], ["c3"], to=TensorProto.BOOL),
        helper.make_node("Add", ["c1_f", "c2_f"], ["nz12_f"]),
        helper.make_node("Add", ["nz12_f", "c3_f"], ["nz_f"]),
        helper.make_node("Conv", ["nz_f", "cross_kernel"], ["cross_count"], pads=[1, 1, 1, 1]),
        helper.make_node("Greater", ["cross_count", "four_half"], ["interior"]),
        helper.make_node("Not", ["interior"], ["not_interior"]),
        helper.make_node("And", ["c0", "c1"], ["zero"]),
        helper.make_node("And", ["c1", "not_interior"], ["c1_out"]),
        helper.make_node("And", ["c2", "not_interior"], ["c2_out"]),
        helper.make_node("And", ["c3", "not_interior"], ["c3_out"]),
        helper.make_node(
            "Concat",
            ["c0", "c1_out", "c2_out", "c3_out", "zero", "zero", "zero", "zero", "interior", "zero"],
            ["out_active_b"],
            axis=1,
        ),
        helper.make_node("Cast", ["out_active_b"], ["out_active_f"], to=TensorProto.FLOAT),
        helper.make_node(
            "Pad",
            ["out_active_f"],
            [OUT_NAME],
            pads=[0, 0, 0, 0, 0, 0, 30 - ACTIVE_H, 30 - ACTIVE_W],
        ),
    ]
    return _model(nodes, inits, f"{TASK_ID}_conv123_floatpad")


def _expected_onehot(grid: list[list[int]]) -> np.ndarray:
    return convert_to_numpy({"input": grid}, "input")


def verify_model(path: Path) -> tuple[bool, str]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        return False, f"ORT load failed: {exc}"
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            inp = convert_to_numpy(example, "input")
            exp = _expected_onehot(example["output"])
            if inp is None or exp is None:
                continue
            got = session.run([OUT_NAME], {IN_NAME: inp})[0] > 0.0
            if not np.array_equal(got, exp.astype(bool)):
                return False, f"{split}[{idx}] failed"
    return True, "ok"


def main() -> None:
    builders: list[tuple[str, Callable[[], onnx.ModelProto]]] = [
        ("conv123_floatpad", build_conv123_floatpad_model),
        ("direct123_floatpad", build_direct123_floatpad_model),
        ("color123_floatpad", build_color123_floatpad_model),
        ("castfull_boolpad", build_castfull_boolpad_model),
        ("castfull_where_floatpad", build_castfull_where_floatpad_model),
        ("castfull_floatpad", build_castfull_floatpad_model),
        ("argmax_floatpad", build_argmax_floatpad_model),
        ("argmax_cast_floatpad", build_argmax_cast_floatpad_model),
        ("slicefirst_floatpad", build_slicefirst_floatpad_model),
    ]

    results: list[tuple[int, str, Path, dict[str, object]]] = []
    for name, build in builders:
        path = OUT_DIR / f"{TASK_ID}_{name}.onnx"
        model = build()
        onnx.save(model, path)
        valid, message = verify_model(path)
        if not valid:
            print(f"{name}: invalid ({message})")
            continue
        result = score_file(path)
        if result["valid"]:
            cost = int(result["cost"])
            results.append((cost, name, path, result))
            print(
                f"{name}: memory={result['memory']} params={result['params']} "
                f"cost={result['cost']} score={float(result['score']):.6f}"
            )
        else:
            print(f"{name}: score invalid ({result['error']})")

    if not results:
        raise SystemExit("no valid task120 variant")

    _, best_name, best_path, best_result = min(results, key=lambda item: item[0])
    shutil.copyfile(best_path, BEST_PATH)
    print()
    print(
        f"best={best_name} -> {BEST_PATH.name}: memory={best_result['memory']} "
        f"params={best_result['params']} cost={best_result['cost']} "
        f"score={float(best_result['score']):.6f}"
    )


if __name__ == "__main__":
    main()
