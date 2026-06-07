"""Minimal ONNX for ARC task091: crop between two vertical gray bars.

Task rule: the input contains exactly two vertical gray (color 5) bars. Let their
columns be c0 < c1 and rows span r0..r1. Output is the exact subgrid
input[r0-1 : r1+2, c0 : c1+1], including the bar columns and one padding row
above and below the bars. All interior colors are preserved unchanged.

ONNX: detect gray in the compact 15x15 task window, reduce to bbox
(r0,r1,c0,c1), gather only the color-5 and color-8 masks into a fixed 14x14
canvas, assemble the 0/5/8 one-hot crop as bool, cast once, then Pad to the
required 30x30 one-hot output.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import (  # noqa: E402
    calculate_memory,
    calculate_params,
    convert_to_numpy,
    load_task_examples,
    sanitize_model,
    score_file,
)

TASK_ID = "task091"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task091.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
SH = 15
SW = 15
OH = 14
OW = 14
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10
GRAY_CH = 5


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference crop between gray vertical bars."""
    g = np.asarray(grid, dtype=np.int64)
    rows, cols = np.where(g == GRAY_CH)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = sorted({int(c) for c in cols})
    return g[r0 - 1 : r1 + 2, c0 : c1 + 1]


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


def _common_inits(inits: List[onnx.TensorProto], *, with_rev: bool = False) -> dict[str, Any]:
    pad_list = [0, 0, 0, 0, 0, 0, H - OH, W - OW]
    names: dict[str, Any] = {
        "axes4": _i64(inits, [0, 1, 2, 3], "axes4"),
        "crop_st": _i64(inits, [0, 0, 0, 0], "crop_st"),
        "crop_en": _i64(inits, [1, C, SH, SW], "crop_en"),
        "gray_st": _i64(inits, [0, GRAY_CH, 0, 0], "gray_st"),
        "gray_en": _i64(inits, [1, GRAY_CH + 1, SH, SW], "gray_en"),
        "flat_shape": _i64(inits, [1, C, SH * SW], "flat_shape"),
        "rows": _f32(inits, np.arange(SH, dtype=np.float32).reshape(1, 1, SH, 1), "rows"),
        "cols": _f32(inits, np.arange(SW, dtype=np.float32).reshape(1, 1, 1, SW), "cols"),
        "out_rows": _f32(inits, np.arange(OH, dtype=np.float32).reshape(OH, 1), "out_rows"),
        "out_cols": _f32(inits, np.arange(OW, dtype=np.float32).reshape(1, OW), "out_cols"),
        "channels": _i64(inits, np.arange(C, dtype=np.int64).reshape(1, C, 1, 1), "channels"),
        "big": _f32(inits, [99.0], "big"),
        "half": _f32(inits, [0.5], "half"),
        "zero": _f32(inits, [0.0], "zero"),
        "zero_i": _i64(inits, [0], "zero_i"),
        "one": _f32(inits, [1.0], "one"),
        "three": _f32(inits, [3.0], "three"),
        "stride": _f32(inits, [float(SW)], "stride"),
        "sq4": _i64(inits, [0, 1, 2, 3], "sq4"),
        "pad_list": pad_list,
        "last_i": _i64(inits, [SH - 1], "last_i"),
    }
    if with_rev:
        names["last_c"] = _i64(inits, [SW - 1], "last_c")
        names["rev"] = _i64(inits, np.arange(SH - 1, -1, -1, dtype=np.int64), "rev")
        names["rev_c"] = _i64(inits, np.arange(SW - 1, -1, -1, dtype=np.int64), "rev_c")
    return names


def _bbox_reduce(nodes: List[onnx.NodeProto], gray: str, names: dict[str, str]) -> tuple[str, str, str, str]:
    """Min/max row/col of gray occupancy via ReduceMin/Max (task031 style)."""
    nodes.extend(
        [
            helper.make_node("Greater", [gray, names["half"]], ["grayb"]),
            helper.make_node("ReduceMax", [gray], ["row_occ"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", [gray], ["col_occ"], axes=[2], keepdims=1),
            helper.make_node("Mul", ["row_occ", names["rows"]], ["row_w"]),
            helper.make_node("Mul", ["col_occ", names["cols"]], ["col_w"]),
            helper.make_node("ReduceMax", ["row_w"], ["max_y4"], axes=[2], keepdims=1),
            helper.make_node("ReduceMax", ["col_w"], ["max_x4"], axes=[3], keepdims=1),
            helper.make_node("Greater", ["row_occ", names["half"]], ["rowb"]),
            helper.make_node("Greater", ["col_occ", names["half"]], ["colb"]),
            helper.make_node("Where", ["rowb", names["rows"], names["big"]], ["row_min_src"]),
            helper.make_node("Where", ["colb", names["cols"], names["big"]], ["col_min_src"]),
            helper.make_node("ReduceMin", ["row_min_src"], ["min_y4"], axes=[2], keepdims=1),
            helper.make_node("ReduceMin", ["col_min_src"], ["min_x4"], axes=[3], keepdims=1),
            helper.make_node("Squeeze", ["min_y4"], ["rmin"], axes=[0, 1, 2, 3]),
            helper.make_node("Squeeze", ["min_x4"], ["cmin"], axes=[0, 1, 2, 3]),
            helper.make_node("Squeeze", ["max_y4"], ["rmax"], axes=[0, 1, 2, 3]),
            helper.make_node("Squeeze", ["max_x4"], ["cmax"], axes=[0, 1, 2, 3]),
        ]
    )
    return "rmin", "rmax", "cmin", "cmax"


def _gather_crop_nodes(
    nodes: List[onnx.NodeProto],
    flat_src: str,
    rmin: str,
    rmax: str,
    cmin: str,
    cmax: str,
    names: dict[str, str],
    *,
    emit_onehot: bool,
) -> None:
    """Fill fixed OHxOW canvas from bbox; optional one-hot via Equal."""
    nodes.extend(
        [
            helper.make_node("Sub", [rmax, rmin], ["span_r"]),
            helper.make_node("Sub", [cmax, cmin], ["span_c"]),
            helper.make_node("Add", ["span_r", names["three"]], ["height"]),
            helper.make_node("Add", ["span_c", names["one"]], ["width"]),
            helper.make_node("Sub", [rmin, names["one"]], ["row0"]),
            helper.make_node("Less", [names["out_rows"], "height"], ["valid_y"]),
            helper.make_node("Less", [names["out_cols"], "width"], ["valid_x"]),
            helper.make_node("And", ["valid_y", "valid_x"], ["valid"]),
            helper.make_node("Add", [names["out_rows"], "row0"], ["abs_y"]),
            helper.make_node("Add", [names["out_cols"], cmin], ["abs_x"]),
            helper.make_node("Mul", ["abs_y", names["stride"]], ["row_base"]),
            helper.make_node("Add", ["row_base", "abs_x"], ["idxf"]),
            helper.make_node("Where", ["valid", "idxf", names["zero"]], ["safe_idxf"]),
            helper.make_node("Cast", ["safe_idxf"], ["idx"], to=TensorProto.INT64),
            helper.make_node("Gather", [flat_src, "idx"], ["gathered"], axis=2),
        ]
    )
    if emit_onehot:
        nodes.extend(
            [
                helper.make_node("Greater", ["gathered", names["half"]], ["fgb"]),
                helper.make_node("ArgMax", ["gathered"], ["color"], axis=1, keepdims=0),
                helper.make_node("Cast", ["color"], ["color_i"], to=TensorProto.INT64),
                helper.make_node("Where", ["fgb", "color_i", names["zero_i"]], ["color_grid"]),
                helper.make_node("Equal", [names["channels"], "color_grid"], ["onehot"]),
                helper.make_node("Unsqueeze", ["valid"], ["valid4"], axes=[0, 1]),
                helper.make_node("And", ["onehot", "valid4"], ["onehot_valid"]),
                helper.make_node("Cast", ["onehot_valid"], ["out_crop"], to=TensorProto.FLOAT),
            ]
        )
    else:
        nodes.extend(
            [
                helper.make_node("Unsqueeze", ["valid"], ["valid4"], axes=[0, 1]),
                helper.make_node("Cast", ["valid4"], ["validf"], to=TensorProto.FLOAT),
                helper.make_node("Mul", ["gathered", "validf"], ["out_crop"]),
            ]
        )
    nodes.append(helper.make_node("Pad", ["out_crop"], [OUT_NAME], mode="constant", pads=names["pad_list"]))


def build_gather_reduce_model() -> onnx.ModelProto:
    """Variant A: ReduceMin/Max bbox + Gather from 10-channel crop."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    n = _common_inits(inits)

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, n["crop_st"], n["crop_en"], n["axes4"]], ["crop"]),
            helper.make_node("Slice", ["crop", n["gray_st"], n["gray_en"], n["axes4"]], ["gray"]),
        ]
    )
    rmin, rmax, cmin, cmax = _bbox_reduce(nodes, "gray", n)
    nodes.append(helper.make_node("Reshape", ["crop", n["flat_shape"]], ["flat"]))
    _gather_crop_nodes(nodes, "flat", rmin, rmax, cmin, cmax, n, emit_onehot=False)
    return _make_model(nodes, inits, f"{TASK_ID}_gather_reduce")


def build_gather_argmax_model() -> onnx.ModelProto:
    """Variant B: ArgMax/rev bbox + Gather (compact scalar extrema)."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    n = _common_inits(inits, with_rev=True)

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, n["crop_st"], n["crop_en"], n["axes4"]], ["crop"]),
            helper.make_node("Slice", ["crop", n["gray_st"], n["gray_en"], n["axes4"]], ["gray"]),
            helper.make_node("Greater", ["gray", n["half"]], ["grayb"]),
            helper.make_node("Cast", ["grayb"], ["grayf"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", ["grayf"], ["row_has"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["grayf"], ["col_has"], axes=[2], keepdims=1),
            helper.make_node("ArgMax", ["row_has"], ["rmin4"], axis=2, keepdims=1),
            helper.make_node("ArgMax", ["col_has"], ["cmin4"], axis=3, keepdims=1),
            helper.make_node("Gather", ["row_has", n["rev"]], ["row_rev"], axis=2),
            helper.make_node("Gather", ["col_has", n["rev_c"]], ["col_rev"], axis=3),
            helper.make_node("ArgMax", ["row_rev"], ["rrev4"], axis=2, keepdims=1),
            helper.make_node("ArgMax", ["col_rev"], ["crev4"], axis=3, keepdims=1),
            helper.make_node("Sub", [n["last_i"], "rrev4"], ["rmax4"]),
            helper.make_node("Sub", [n["last_c"], "crev4"], ["cmax4"]),
            helper.make_node("Squeeze", ["rmin4"], ["rmin"], axes=[0, 1, 2, 3]),
            helper.make_node("Squeeze", ["cmin4"], ["cmin"], axes=[0, 1, 2, 3]),
            helper.make_node("Squeeze", ["rmax4"], ["rmax"], axes=[0, 1, 2, 3]),
            helper.make_node("Squeeze", ["cmax4"], ["cmax"], axes=[0, 1, 2, 3]),
            helper.make_node("Cast", ["rmin"], ["rmin_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["rmax"], ["rmax_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["cmin"], ["cmin_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["cmax"], ["cmax_f"], to=TensorProto.FLOAT),
            helper.make_node("Reshape", ["crop", n["flat_shape"]], ["flat"]),
        ]
    )
    _gather_crop_nodes(nodes, "flat", "rmin_f", "rmax_f", "cmin_f", "cmax_f", n, emit_onehot=False)
    return _make_model(nodes, inits, f"{TASK_ID}_gather_argmax")


def build_color_gather_model() -> onnx.ModelProto:
    """Variant C: ArgMax color grid + scalar Gather + Equal one-hot."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    n = _common_inits(inits)
    n["flat_color_shape"] = _i64(inits, [1, 1, SH * SW], "flat_color_shape")
    n["shape_ohow"] = _i64(inits, [1, 1, OH, OW], "shape_ohow")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, n["crop_st"], n["crop_en"], n["axes4"]], ["crop"]),
            helper.make_node("Slice", ["crop", n["gray_st"], n["gray_en"], n["axes4"]], ["gray"]),
            helper.make_node("ArgMax", ["crop"], ["color_hw"], axis=1, keepdims=0),
            helper.make_node("Reshape", ["color_hw", n["flat_color_shape"]], ["flat_color"]),
        ]
    )
    rmin, rmax, cmin, cmax = _bbox_reduce(nodes, "gray", n)
    nodes.extend(
        [
            helper.make_node("Sub", [rmax, rmin], ["span_r"]),
            helper.make_node("Sub", [cmax, cmin], ["span_c"]),
            helper.make_node("Add", ["span_r", n["three"]], ["height"]),
            helper.make_node("Add", ["span_c", n["one"]], ["width"]),
            helper.make_node("Sub", [rmin, n["one"]], ["row0"]),
            helper.make_node("Less", [n["out_rows"], "height"], ["valid_y"]),
            helper.make_node("Less", [n["out_cols"], "width"], ["valid_x"]),
            helper.make_node("And", ["valid_y", "valid_x"], ["valid"]),
            helper.make_node("Add", [n["out_rows"], "row0"], ["abs_y"]),
            helper.make_node("Add", [n["out_cols"], cmin], ["abs_x"]),
            helper.make_node("Mul", ["abs_y", n["stride"]], ["row_base"]),
            helper.make_node("Add", ["row_base", "abs_x"], ["idxf"]),
            helper.make_node("Where", ["valid", "idxf", n["zero"]], ["safe_idxf"]),
            helper.make_node("Cast", ["safe_idxf"], ["idx"], to=TensorProto.INT64),
            helper.make_node("Gather", ["flat_color", "idx"], ["picked"], axis=2),
            helper.make_node("Cast", ["picked"], ["picked_i"], to=TensorProto.INT64),
            helper.make_node("Reshape", ["picked_i", n["shape_ohow"]], ["color_grid"]),
            helper.make_node("Equal", [n["channels"], "color_grid"], ["onehot"]),
            helper.make_node("Unsqueeze", ["valid"], ["valid4"], axes=[0, 1]),
            helper.make_node("And", ["onehot", "valid4"], ["onehot_valid"]),
            helper.make_node("Cast", ["onehot_valid"], ["out_crop"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out_crop"], [OUT_NAME], mode="constant", pads=n["pad_list"]),
        ]
    )
    return _make_model(nodes, inits, f"{TASK_ID}_color_gather")


def build_mask_gather_model() -> onnx.ModelProto:
    """Variant D: gather only color-5/color-8 masks, then assemble one-hot."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    n = _common_inits(inits)
    n["eight_st"] = _i64(inits, [0, 8, 0, 0], "eight_st")
    n["eight_en"] = _i64(inits, [1, 9, SH, SW], "eight_en")
    n["flat_mask_shape"] = _i64(inits, [1, 1, SH * SW], "flat_mask_shape")
    n["zero_crop"] = _f32(inits, np.zeros((1, 1, OH, OW), dtype=np.float32), "zero_crop")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, n["gray_st"], n["gray_en"], n["axes4"]], ["gray"]),
            helper.make_node("Slice", [IN_NAME, n["eight_st"], n["eight_en"], n["axes4"]], ["eight"]),
        ]
    )
    rmin, rmax, cmin, cmax = _bbox_reduce(nodes, "gray", n)
    nodes.extend(
        [
            helper.make_node("Reshape", ["gray", n["flat_mask_shape"]], ["flat_gray"]),
            helper.make_node("Reshape", ["eight", n["flat_mask_shape"]], ["flat_eight"]),
            helper.make_node("Sub", [rmax, rmin], ["span_r"]),
            helper.make_node("Sub", [cmax, cmin], ["span_c"]),
            helper.make_node("Add", ["span_r", n["three"]], ["height"]),
            helper.make_node("Add", ["span_c", n["one"]], ["width"]),
            helper.make_node("Sub", [rmin, n["one"]], ["row0"]),
            helper.make_node("Less", [n["out_rows"], "height"], ["valid_y"]),
            helper.make_node("Less", [n["out_cols"], "width"], ["valid_x"]),
            helper.make_node("And", ["valid_y", "valid_x"], ["valid"]),
            helper.make_node("Add", [n["out_rows"], "row0"], ["abs_y"]),
            helper.make_node("Add", [n["out_cols"], cmin], ["abs_x"]),
            helper.make_node("Mul", ["abs_y", n["stride"]], ["row_base"]),
            helper.make_node("Add", ["row_base", "abs_x"], ["idxf"]),
            helper.make_node("Where", ["valid", "idxf", n["zero"]], ["safe_idxf"]),
            helper.make_node("Cast", ["safe_idxf"], ["idx"], to=TensorProto.INT64),
            helper.make_node("Gather", ["flat_gray", "idx"], ["gray_pick"], axis=2),
            helper.make_node("Gather", ["flat_eight", "idx"], ["eight_pick"], axis=2),
            helper.make_node("Unsqueeze", ["valid"], ["valid4"], axes=[0, 1]),
            helper.make_node("Cast", ["valid4"], ["validf"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["gray_pick", "validf"], ["out_gray"]),
            helper.make_node("Mul", ["eight_pick", "validf"], ["out_eight"]),
            helper.make_node("Sub", ["validf", "out_gray"], ["not_gray"]),
            helper.make_node("Sub", ["not_gray", "out_eight"], ["out_zero"]),
            helper.make_node(
                "Concat",
                [
                    "out_zero",
                    n["zero_crop"],
                    n["zero_crop"],
                    n["zero_crop"],
                    n["zero_crop"],
                    "out_gray",
                    n["zero_crop"],
                    n["zero_crop"],
                    "out_eight",
                    n["zero_crop"],
                ],
                ["out_crop"],
                axis=1,
            ),
            helper.make_node("Pad", ["out_crop"], [OUT_NAME], mode="constant", pads=n["pad_list"]),
        ]
    )
    return _make_model(nodes, inits, f"{TASK_ID}_mask_gather")


def build_bool_mask_gather_model() -> onnx.ModelProto:
    """Variant E: bool color masks until the final crop cast."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    n = _common_inits(inits)
    n["eight_st"] = _i64(inits, [0, 8, 0, 0], "eight_st")
    n["eight_en"] = _i64(inits, [1, 9, SH, SW], "eight_en")
    n["flat_mask_shape"] = _i64(inits, [1, 1, SH * SW], "flat_mask_shape")
    n["zero_bool_crop"] = _init(inits, np.zeros((1, 1, OH, OW), dtype=np.bool_), "zero_bool_crop")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, n["gray_st"], n["gray_en"], n["axes4"]], ["gray"]),
            helper.make_node("Slice", [IN_NAME, n["eight_st"], n["eight_en"], n["axes4"]], ["eight"]),
        ]
    )
    rmin, rmax, cmin, cmax = _bbox_reduce(nodes, "gray", n)
    nodes.extend(
        [
            helper.make_node("Greater", ["eight", n["half"]], ["eightb"]),
            helper.make_node("Reshape", ["grayb", n["flat_mask_shape"]], ["flat_grayb"]),
            helper.make_node("Reshape", ["eightb", n["flat_mask_shape"]], ["flat_eightb"]),
            helper.make_node("Sub", [rmax, rmin], ["span_r"]),
            helper.make_node("Sub", [cmax, cmin], ["span_c"]),
            helper.make_node("Add", ["span_r", n["three"]], ["height"]),
            helper.make_node("Add", ["span_c", n["one"]], ["width"]),
            helper.make_node("Sub", [rmin, n["one"]], ["row0"]),
            helper.make_node("Less", [n["out_rows"], "height"], ["valid_y"]),
            helper.make_node("Less", [n["out_cols"], "width"], ["valid_x"]),
            helper.make_node("And", ["valid_y", "valid_x"], ["valid"]),
            helper.make_node("Add", [n["out_rows"], "row0"], ["abs_y"]),
            helper.make_node("Add", [n["out_cols"], cmin], ["abs_x"]),
            helper.make_node("Mul", ["abs_y", n["stride"]], ["row_base"]),
            helper.make_node("Add", ["row_base", "abs_x"], ["idxf"]),
            helper.make_node("Where", ["valid", "idxf", n["zero"]], ["safe_idxf"]),
            helper.make_node("Cast", ["safe_idxf"], ["idx"], to=TensorProto.INT64),
            helper.make_node("Gather", ["flat_grayb", "idx"], ["gray_pick"], axis=2),
            helper.make_node("Gather", ["flat_eightb", "idx"], ["eight_pick"], axis=2),
            helper.make_node("Unsqueeze", ["valid"], ["valid4"], axes=[0, 1]),
            helper.make_node("And", ["gray_pick", "valid4"], ["out_gray"]),
            helper.make_node("And", ["eight_pick", "valid4"], ["out_eight"]),
            helper.make_node("Or", ["out_gray", "out_eight"], ["nonzero"]),
            helper.make_node("Not", ["nonzero"], ["is_zero_unmasked"]),
            helper.make_node("And", ["is_zero_unmasked", "valid4"], ["out_zero"]),
            helper.make_node(
                "Concat",
                [
                    "out_zero",
                    n["zero_bool_crop"],
                    n["zero_bool_crop"],
                    n["zero_bool_crop"],
                    n["zero_bool_crop"],
                    "out_gray",
                    n["zero_bool_crop"],
                    n["zero_bool_crop"],
                    "out_eight",
                    n["zero_bool_crop"],
                ],
                ["out_crop_bool"],
                axis=1,
            ),
            helper.make_node("Cast", ["out_crop_bool"], ["out_crop"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out_crop"], [OUT_NAME], mode="constant", pads=n["pad_list"]),
        ]
    )
    return _make_model(nodes, inits, f"{TASK_ID}_bool_mask_gather")


def build_dynamic_slice_model() -> onnx.ModelProto | None:
    """Variant F: dynamic Slice to max OHxOW then mask (may fail strict shape infer)."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    n = _common_inits(inits)
    n["one_i"] = _i64(inits, [1], "one_i")
    n["three_i"] = _i64(inits, [3], "three_i")
    n["axes2"] = _i64(inits, [2, 3], "axes2")
    n["ch_en"] = _i64(inits, [C], "ch_en")
    n["oh_i"] = _i64(inits, [OH], "oh_i")
    n["ow_i"] = _i64(inits, [OW], "ow_i")
    n["out_rows_i"] = _i64(inits, np.arange(OH, dtype=np.int64).reshape(OH, 1), "out_rows_i")
    n["out_cols_i"] = _i64(inits, np.arange(OW, dtype=np.int64).reshape(1, OW), "out_cols_i")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, n["crop_st"], n["crop_en"], n["axes4"]], ["crop"]),
            helper.make_node("Slice", ["crop", n["gray_st"], n["gray_en"], n["axes4"]], ["gray"]),
        ]
    )
    rmin, rmax, cmin, cmax = _bbox_reduce(nodes, "gray", n)
    nodes.extend(
        [
            helper.make_node("Cast", [rmin], ["rmin_i"], to=TensorProto.INT64),
            helper.make_node("Cast", [rmax], ["rmax_i"], to=TensorProto.INT64),
            helper.make_node("Cast", [cmin], ["cmin_i"], to=TensorProto.INT64),
            helper.make_node("Cast", [cmax], ["cmax_i"], to=TensorProto.INT64),
            helper.make_node("Sub", ["rmin_i", n["one_i"]], ["row0_i"]),
            helper.make_node("Add", ["rmax_i", n["one_i"]], ["row1_i"]),
            helper.make_node("Add", ["cmax_i", n["one_i"]], ["col1_i"]),
            helper.make_node("Unsqueeze", ["row0_i"], ["row0_4"], axes=[0, 1]),
            helper.make_node("Unsqueeze", [n["zero_i"]], ["z4"], axes=[0, 1]),
            helper.make_node("Unsqueeze", ["row1_i"], ["row1_4"], axes=[0, 1]),
            helper.make_node("Unsqueeze", ["col1_i"], ["col1_4"], axes=[0, 1]),
            helper.make_node("Unsqueeze", ["cmin_i"], ["c0_4"], axes=[0, 1]),
            helper.make_node("Concat", ["z4", "z4", "row0_4", "c0_4"], ["starts"], axis=0),
            helper.make_node("Concat", [n["one_i"], n["ch_en"], "row1_4", "col1_4"], ["ends"], axis=0),
            helper.make_node("Slice", ["crop", "starts", "ends", n["axes4"]], ["sliced"]),
            helper.make_node("Sub", ["rmax_i", "rmin_i"], ["span_r"]),
            helper.make_node("Sub", ["cmax_i", "cmin_i"], ["span_c"]),
            helper.make_node("Add", ["span_r", n["three_i"]], ["height_i"]),
            helper.make_node("Add", ["span_c", n["one_i"]], ["width_i"]),
            helper.make_node("Less", [n["out_rows_i"], "height_i"], ["valid_y"]),
            helper.make_node("Less", [n["out_cols_i"], "width_i"], ["valid_x"]),
            helper.make_node("And", ["valid_y", "valid_x"], ["valid"]),
            helper.make_node("Unsqueeze", ["valid"], ["valid4"], axes=[0, 1]),
            helper.make_node("Cast", ["valid4"], ["validf"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["sliced", "validf"], ["out_crop"]),
            helper.make_node("Pad", ["out_crop"], [OUT_NAME], mode="constant", pads=n["pad_list"]),
        ]
    )
    try:
        model = _make_model(nodes, inits, f"{TASK_ID}_dynamic_slice")
        onnx.shape_inference.infer_shapes(model, strict_mode=True)
        return model
    except Exception:
        return None


def _load_examples() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _check_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return False, {}
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        sanitized.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    examples = _load_examples()
    counts: dict[str, tuple[int, int]] = {}
    all_ok = True
    for split in ("train", "test", "arc-gen"):
        passed = 0
        total = 0
        for example in examples.get(split, []):
            input_arr = convert_to_numpy(example, "input")
            expected_arr = convert_to_numpy(example, "output")
            if input_arr is None or expected_arr is None:
                continue
            total += 1
            out = session.run([OUT_NAME], {IN_NAME: input_arr})[0]
            if np.array_equal((out > 0.0).astype(np.float32), expected_arr):
                passed += 1
            else:
                all_ok = False
        counts[split] = (passed, total)
    return all_ok, counts


def _profile_largest_internal(model: onnx.ModelProto) -> tuple[int | None, str | None, int | None]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return None, None, None

    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(Path(tempfile.gettempdir()) / f"ng_{TASK_ID}_largest")
    session = ort.InferenceSession(
        sanitized.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    for arr in load_task_examples(BEST_PATH):
        session.run([OUT_NAME], {IN_NAME: arr})
    trace_path = session.end_profiling()

    graph = onnx.shape_inference.infer_shapes(sanitized, strict_mode=True).graph
    outputs_by_node = {node.name: list(node.output) for node in graph.node}
    dtypes = {
        info.name: onnx.helper.tensor_dtype_to_np_dtype(info.type.tensor_type.elem_type)
        for info in list(graph.value_info) + list(graph.output)
        if info.type.HasField("tensor_type")
    }
    largest_name: str | None = None
    largest_bytes = -1
    with open(trace_path, encoding="utf-8") as fh:
        trace = json.load(fh)
    for event in trace:
        if event.get("cat") != "Node" or "output_type_shape" not in event.get("args", {}):
            continue
        node_name = event.get("name", "").replace("_kernel_time", "")
        for idx, shape_dict in enumerate(event["args"]["output_type_shape"]):
            outputs = outputs_by_node.get(node_name, [])
            if idx >= len(outputs):
                continue
            output_name = outputs[idx]
            if output_name == OUT_NAME or output_name not in dtypes:
                continue
            itemsize = np.dtype(dtypes[output_name]).itemsize
            size = itemsize * sum(math.prod(dims) for dims in shape_dict.values())
            if size > largest_bytes:
                largest_name = output_name
                largest_bytes = int(size)
    memory = calculate_memory(sanitized, trace_path)
    return memory, largest_name, largest_bytes


def _format_counts(counts: dict[str, tuple[int, int]]) -> str:
    return ", ".join(f"{split} {ok}/{total}" for split, (ok, total) in counts.items())


def _infer_summary(model: onnx.ModelProto) -> tuple[int, int]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return 0, 0
    inferred = onnx.shape_inference.infer_shapes(sanitized, strict_mode=True)
    n_tensors = len(inferred.graph.value_info)
    params = calculate_params(sanitized) or 0
    return n_tensors, params


def main() -> None:
    variant_builders: list[tuple[str, Any]] = [
        ("gather_reduce", build_gather_reduce_model),
        ("gather_argmax", build_gather_argmax_model),
        ("color_gather", build_color_gather_model),
        ("mask_gather", build_mask_gather_model),
        ("bool_mask_gather", build_bool_mask_gather_model),
        ("dynamic_slice", build_dynamic_slice_model),
    ]

    results: list[tuple[int, str, onnx.ModelProto, dict[str, Any], str | None, int | None, dict[str, tuple[int, int]]]] = []
    for label, builder in variant_builders:
        model = builder()
        if model is None:
            print(f"{label:<18} skipped (build or shape inference failed)")
            continue
        tmp_path = OUT_DIR / f"{TASK_ID}_{label}.onnx"
        onnx.save(model, tmp_path)
        correct, counts = _check_correct(model)
        scored = score_file(tmp_path)
        n_vi, params = _infer_summary(model)
        memory, largest_name, largest_bytes = _profile_largest_internal(model)
        score_text = f"{scored['score']:.6f}" if scored["score"] is not None else "None"
        print(
            f"{label:<18} correct={correct} ({_format_counts(counts)}) "
            f"memory={scored['memory']} params={scored['params']} cost={scored['cost']} "
            f"score={score_text} value_infos={n_vi} largest={largest_name}:{largest_bytes}"
        )
        if correct and scored.get("valid"):
            results.append((int(scored["cost"] or 10**9), label, model, scored, largest_name, largest_bytes, counts))
        tmp_path.unlink(missing_ok=True)

    if not results:
        raise SystemExit("no valid correct variants")

    _, label, model, scored, largest_name, largest_bytes, counts = min(results, key=lambda item: item[0])
    onnx.save(model, BEST_PATH)
    print()
    print(f"kept:    {label}")
    print(f"wrote:   {BEST_PATH}")
    print(f"valid:   {scored['valid']}")
    print(f"passes:  {_format_counts(counts)}")
    print(f"memory:  {scored['memory']}")
    print(f"params:  {scored['params']}")
    print(f"cost:    {scored['cost']}")
    print(f"score:   {scored['score']:.6f}")
    print(f"largest: {largest_name} ({largest_bytes} bytes)")


if __name__ == "__main__":
    main()
