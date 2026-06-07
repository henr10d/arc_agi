"""Minimal ONNX for ARC task065: return the quadrant with the anomaly.

Task rule: the input is an odd square with a full separator cross through the
center row and column.  The cross splits the grid into four equal quadrants.
Exactly one quadrant contains a cell whose color differs from the common
quadrant background.  Output that quadrant only, without the separator cross,
preserving the anomalous cell position.  The bundled examples range from 3x3 to
15x15 inputs, so the ONNX graph works on the top-left 15x15 active window and
pads the selected 1x1..7x7 quadrant into the standard 30x30 NeuroGolf output.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import (  # noqa: E402
    calculate_memory,
    calculate_params,
    convert_to_numpy,
    sanitize_model,
    score_file,
)

TASK_ID = "task065"
BEST_PATH = OUT_DIR / "task065.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
MAX_N = 15
MAX_Q = 7
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def solve_grid(grid: list[list[int]]) -> np.ndarray:
    """Numpy reference solver for local validation."""
    arr = np.asarray(grid, dtype=np.int64)
    n = arr.shape[0]
    q = (n - 1) // 2
    quads = [
        arr[:q, :q],
        arr[:q, q + 1 : n],
        arr[q + 1 : n, :q],
        arr[q + 1 : n, q + 1 : n],
    ]
    corners = [int(part[0, 0]) for part in quads]
    scores = [
        int(np.count_nonzero(part != corner))
        + sum(int(corner != other) for j, other in enumerate(corners) if j != i)
        for i, (part, corner) in enumerate(zip(quads, corners))
    ]
    return quads[int(np.argmax(scores))]


def _grid_to_onehot(grid: Iterable[Iterable[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


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


def _add_common_prefix(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    *,
    direct_onehot_scores: bool,
) -> None:
    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    s15 = _i64(inits, [0, 0, 0, 0], "s15")
    e15 = _i64(inits, [1, C, MAX_N, MAX_N], "e15")
    zero = _f32(inits, [0.0], "zero")
    one = _f32(inits, [1.0], "one")
    two = _f32(inits, [2.0], "two")
    shape_flat = _i64(inits, [MAX_N * MAX_N], "shape_flat")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, s15, e15, axes4], ["x15"]),
            helper.make_node("ReduceSum", ["x15"], ["cell_sum"], axes=[1], keepdims=0),
            helper.make_node("Greater", ["cell_sum", zero], ["valid15"]),
            helper.make_node("Cast", ["valid15"], ["valid15_f"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", ["valid15_f"], ["row_any"], axes=[2], keepdims=1),
            helper.make_node("ReduceSum", ["row_any"], ["n"], axes=[1, 2], keepdims=0),
            helper.make_node("Sub", ["n", one], ["n_minus_1"]),
            helper.make_node("Div", ["n_minus_1", two], ["q"]),
            helper.make_node("Add", ["q", one], ["q_plus_1"]),
            helper.make_node("Cast", ["q"], ["q_i"], to=TensorProto.INT64),
            helper.make_node("ArgMax", ["x15"], ["color15"], axis=1, keepdims=0),
            helper.make_node("Reshape", ["color15", shape_flat], ["color_flat"]),
        ]
    )
    if direct_onehot_scores:
        nodes.append(helper.make_node("Greater", ["x15", zero], ["x15_bool"]))


def _add_compact_prefix(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto]) -> None:
    axes4 = _i64(inits, [0, 1, 2, 3], "compact_axes4")
    s15 = _i64(inits, [0, 0, 0, 0], "compact_s15")
    e15 = _i64(inits, [1, C, MAX_N, MAX_N], "compact_e15")
    erow = _i64(inits, [1, C, 1, MAX_N], "compact_erow")
    zero = _f32(inits, [0.0], "zero")
    one = _f32(inits, [1.0], "one")
    two = _f32(inits, [2.0], "two")
    shape_flat = _i64(inits, [MAX_N * MAX_N], "compact_shape_flat")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, s15, e15, axes4], ["x15"]),
            helper.make_node("Slice", [IN_NAME, s15, erow, axes4], ["row0"]),
            helper.make_node("ReduceSum", ["row0"], ["row0_cell_sum"], axes=[1], keepdims=0),
            helper.make_node("Greater", ["row0_cell_sum", zero], ["row0_valid"]),
            helper.make_node("Cast", ["row0_valid"], ["row0_valid_f"], to=TensorProto.FLOAT),
            helper.make_node("ReduceSum", ["row0_valid_f"], ["n"], axes=[1, 2], keepdims=0),
            helper.make_node("Sub", ["n", one], ["n_minus_1"]),
            helper.make_node("Div", ["n_minus_1", two], ["q"]),
            helper.make_node("Add", ["q", one], ["q_plus_1"]),
            helper.make_node("Cast", ["q"], ["q_i"], to=TensorProto.INT64),
            helper.make_node("ArgMax", ["x15"], ["color15"], axis=1, keepdims=0),
            helper.make_node("Reshape", ["color15", shape_flat], ["color_flat"]),
        ]
    )


def _score_quadrants(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    *,
    direct_onehot_scores: bool,
) -> None:
    zero = "zero"
    fifteen_i = _i64(inits, [MAX_N], "score_fifteen_i")
    zero_i = _i64(inits, [0], "score_zero_i")
    if direct_onehot_scores:
        colors = _i64(inits, np.arange(C, dtype=np.int64).reshape(1, C, 1, 1), "score_colors")
        nine = _f32(inits, [9.0], "score_nine")

    nodes.extend(
        [
            helper.make_node("Cast", ["q_plus_1"], ["qpi"], to=TensorProto.INT64),
            helper.make_node("Mul", ["qpi", fifteen_i], ["bl_idx"]),
            helper.make_node("Add", ["bl_idx", "qpi"], ["br_idx"]),
            helper.make_node("Gather", ["color_flat", zero_i], ["tl_corner"], axis=0),
            helper.make_node("Gather", ["color_flat", "qpi"], ["tr_corner"], axis=0),
            helper.make_node("Gather", ["color_flat", "bl_idx"], ["bl_corner"], axis=0),
            helper.make_node("Gather", ["color_flat", "br_idx"], ["br_corner"], axis=0),
        ]
    )

    y15 = _i64(inits, np.arange(MAX_N, dtype=np.int64).reshape(1, MAX_N, 1), "y15")
    x15c = _i64(inits, np.arange(MAX_N, dtype=np.int64).reshape(1, 1, MAX_N), "x15c")
    nodes.extend(
        [
            helper.make_node("Equal", [y15, "q_i"], ["is_mid_y"]),
            helper.make_node("Equal", [x15c, "q_i"], ["is_mid_x"]),
            helper.make_node("Or", ["is_mid_y", "is_mid_x"], ["is_cross"]),
            helper.make_node("Not", ["is_cross"], ["not_cross"]),
            helper.make_node("Less", [y15, "q_i"], ["top_band"]),
            helper.make_node("Greater", [y15, "q_i"], ["bottom_band"]),
            helper.make_node("Less", [x15c, "q_i"], ["left_band"]),
            helper.make_node("Greater", [x15c, "q_i"], ["right_band"]),
        ]
    )
    for name, row_band, col_band in (
        ("tl", "top_band", "left_band"),
        ("tr", "top_band", "right_band"),
        ("bl", "bottom_band", "left_band"),
        ("br", "bottom_band", "right_band"),
    ):
        if direct_onehot_scores:
            nodes.extend(
                [
                    helper.make_node("Equal", [colors, f"{name}_corner"], [f"{name}_corner_oh"]),
                    helper.make_node("Equal", ["x15_bool", f"{name}_corner_oh"], [f"{name}_eq_ch"]),
                    helper.make_node("Cast", [f"{name}_eq_ch"], [f"{name}_eq_ch_f"], to=TensorProto.FLOAT),
                    helper.make_node("ReduceSum", [f"{name}_eq_ch_f"], [f"{name}_same_count"], axes=[1], keepdims=0),
                    helper.make_node("Greater", [f"{name}_same_count", nine], [f"{name}_same"]),
                    helper.make_node("Not", [f"{name}_same"], [f"{name}_diff"]),
                    helper.make_node("And", [row_band, col_band], [f"{name}_band"]),
                    helper.make_node("And", [f"{name}_diff", "valid15"], [f"{name}_diff_valid"]),
                    helper.make_node("And", [f"{name}_diff_valid", "not_cross"], [f"{name}_anom0"]),
                    helper.make_node("And", [f"{name}_anom0", f"{name}_band"], [f"{name}_anom"]),
                    helper.make_node("Cast", [f"{name}_anom"], [f"{name}_f"], to=TensorProto.FLOAT),
                    helper.make_node("ReduceSum", [f"{name}_f"], [f"{name}_cell_score"], axes=[1, 2], keepdims=0),
                ]
            )
        else:
            nodes.extend(
                [
                    helper.make_node("Equal", ["color15", f"{name}_corner"], [f"{name}_same"]),
                    helper.make_node("Not", [f"{name}_same"], [f"{name}_diff"]),
                    helper.make_node("And", [row_band, col_band], [f"{name}_band"]),
                    helper.make_node("And", [f"{name}_diff", "valid15"], [f"{name}_diff_valid"]),
                    helper.make_node("And", [f"{name}_diff_valid", "not_cross"], [f"{name}_anom0"]),
                    helper.make_node("And", [f"{name}_anom0", f"{name}_band"], [f"{name}_anom"]),
                    helper.make_node("Cast", [f"{name}_anom"], [f"{name}_f"], to=TensorProto.FLOAT),
                    helper.make_node("ReduceSum", [f"{name}_f"], [f"{name}_cell_score"], axes=[1, 2], keepdims=0),
                ]
            )

    for name in ("tl", "tr", "bl", "br"):
        others = [other for other in ("tl", "tr", "bl", "br") if other != name]
        nodes.extend(
            [
                helper.make_node("Equal", [f"{name}_corner", f"{others[0]}_corner"], [f"{name}_eq0"]),
                helper.make_node("Equal", [f"{name}_corner", f"{others[1]}_corner"], [f"{name}_eq1"]),
                helper.make_node("Equal", [f"{name}_corner", f"{others[2]}_corner"], [f"{name}_eq2"]),
                helper.make_node("Not", [f"{name}_eq0"], [f"{name}_ne0"]),
                helper.make_node("Not", [f"{name}_eq1"], [f"{name}_ne1"]),
                helper.make_node("Not", [f"{name}_eq2"], [f"{name}_ne2"]),
                helper.make_node("Cast", [f"{name}_ne0"], [f"{name}_ne0_f"], to=TensorProto.FLOAT),
                helper.make_node("Cast", [f"{name}_ne1"], [f"{name}_ne1_f"], to=TensorProto.FLOAT),
                helper.make_node("Cast", [f"{name}_ne2"], [f"{name}_ne2_f"], to=TensorProto.FLOAT),
                helper.make_node("Add", [f"{name}_ne0_f", f"{name}_ne1_f"], [f"{name}_corner_score0"]),
                helper.make_node("Add", [f"{name}_corner_score0", f"{name}_ne2_f"], [f"{name}_corner_score"]),
                helper.make_node("Add", [f"{name}_cell_score", f"{name}_corner_score"], [f"{name}_score"]),
            ]
        )

    nodes.extend(
        [
            helper.make_node("Greater", ["br_score", "tl_score"], ["br_gt_tl"]),
            helper.make_node("Greater", ["br_score", "tr_score"], ["br_gt_tr"]),
            helper.make_node("Greater", ["br_score", "bl_score"], ["br_gt_bl"]),
            helper.make_node("And", ["br_gt_tl", "br_gt_tr"], ["br_gt_tltr"]),
            helper.make_node("And", ["br_gt_tltr", "br_gt_bl"], ["br_best"]),
            helper.make_node("Greater", ["bl_score", "tl_score"], ["bl_gt_tl"]),
            helper.make_node("Greater", ["bl_score", "tr_score"], ["bl_gt_tr"]),
            helper.make_node("Greater", ["bl_score", "br_score"], ["bl_gt_br"]),
            helper.make_node("And", ["bl_gt_tl", "bl_gt_tr"], ["bl_gt_tltr"]),
            helper.make_node("And", ["bl_gt_tltr", "bl_gt_br"], ["bl_best"]),
            helper.make_node("Greater", ["tr_score", "tl_score"], ["tr_gt_tl"]),
            helper.make_node("Greater", ["tr_score", "bl_score"], ["tr_gt_bl"]),
            helper.make_node("Greater", ["tr_score", "br_score"], ["tr_gt_br"]),
            helper.make_node("And", ["tr_gt_tl", "tr_gt_bl"], ["tr_gt_tlbl"]),
            helper.make_node("And", ["tr_gt_tlbl", "tr_gt_br"], ["tr_best"]),
            helper.make_node("Or", ["br_best", "bl_best"], ["bottom_best"]),
            helper.make_node("Or", ["bottom_best", "tr_best"], ["not_tl_best"]),
            helper.make_node("Not", ["not_tl_best"], ["tl_best"]),
            helper.make_node("Or", ["tl_best", "tr_best"], ["pick_top"]),
            helper.make_node("Or", ["tl_best", "bl_best"], ["pick_left"]),
            helper.make_node("Where", ["pick_top", zero, "q_plus_1"], ["row_off"]),
            helper.make_node("Where", ["pick_left", zero, "q_plus_1"], ["col_off"]),
        ]
    )


def _add_compact_output(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto]) -> None:
    y7 = _f32(inits, np.arange(MAX_Q, dtype=np.float32).reshape(1, MAX_Q, 1), "y7")
    x7c = _f32(inits, np.arange(MAX_Q, dtype=np.float32).reshape(1, 1, MAX_Q), "x7c")
    colors = _i64(inits, np.arange(C, dtype=np.int64).reshape(1, C, 1, 1), "colors")
    fifteen_i = _i64(inits, [MAX_N], "fifteen_i")
    neg_i = _i64(inits, [-1], "neg_i")

    nodes.extend(
        [
            helper.make_node("Less", [y7, "q"], ["out_y_ok"]),
            helper.make_node("Less", [x7c, "q"], ["out_x_ok"]),
            helper.make_node("And", ["out_y_ok", "out_x_ok"], ["out_valid"]),
            helper.make_node("Mul", [x7c, "zero"], ["x7_zero"]),
            helper.make_node("Mul", [y7, "zero"], ["y7_zero"]),
            helper.make_node("Add", [y7, "row_off"], ["sy_base"]),
            helper.make_node("Add", [x7c, "col_off"], ["sx_base"]),
            helper.make_node("Add", ["sy_base", "x7_zero"], ["sy_full"]),
            helper.make_node("Add", ["sx_base", "y7_zero"], ["sx_full"]),
            helper.make_node("Where", ["out_valid", "sy_full", "x7_zero"], ["sy_safe_f"]),
            helper.make_node("Where", ["out_valid", "sx_full", "y7_zero"], ["sx_safe_f"]),
            helper.make_node("Cast", ["sy_safe_f"], ["sy_i"], to=TensorProto.INT64),
            helper.make_node("Cast", ["sx_safe_f"], ["sx_i"], to=TensorProto.INT64),
            helper.make_node("Mul", ["sy_i", fifteen_i], ["sy_lin"]),
            helper.make_node("Add", ["sy_lin", "sx_i"], ["lin_idx"]),
            helper.make_node("Gather", ["color_flat", "lin_idx"], ["gathered"], axis=0),
            helper.make_node("Where", ["out_valid", "gathered", neg_i], ["safe_color"]),
            helper.make_node("Unsqueeze", ["safe_color"], ["safe_color_ch"], axes=[1]),
            helper.make_node("Equal", ["safe_color_ch", colors], ["out7_bool"]),
            helper.make_node("Cast", ["out7_bool"], ["out7"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out7"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - MAX_Q, W - MAX_Q]),
        ]
    )


def _add_compact_quadrants(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto]) -> None:
    y7 = _i64(inits, np.arange(MAX_Q, dtype=np.int64).reshape(1, MAX_Q, 1), "cq_y7")
    x7c = _i64(inits, np.arange(MAX_Q, dtype=np.int64).reshape(1, 1, MAX_Q), "cq_x7")
    fifteen_i = _i64(inits, [MAX_N], "cq_fifteen_i")

    nodes.extend(
        [
            helper.make_node("Cast", ["q_plus_1"], ["qpi"], to=TensorProto.INT64),
            helper.make_node("Less", [y7, "q_i"], ["cq_y_ok"]),
            helper.make_node("Less", [x7c, "q_i"], ["cq_x_ok"]),
            helper.make_node("And", ["cq_y_ok", "cq_x_ok"], ["cq_valid"]),
            helper.make_node("Mul", [y7, fifteen_i], ["cq_y_lin0"]),
        ]
    )

    for name, row_off, col_off in (
        ("tl", None, None),
        ("tr", None, "qpi"),
        ("bl", "qpi", None),
        ("br", "qpi", "qpi"),
    ):
        sy = y7
        sx = x7c
        if row_off is not None:
            sy = f"{name}_sy"
            nodes.append(helper.make_node("Add", [y7, row_off], [sy]))
        if col_off is not None:
            sx = f"{name}_sx"
            nodes.append(helper.make_node("Add", [x7c, col_off], [sx]))
        ylin = "cq_y_lin0"
        if row_off is not None:
            ylin = f"{name}_ylin"
            nodes.append(helper.make_node("Mul", [sy, fifteen_i], [ylin]))
        nodes.extend(
            [
                helper.make_node("Add", [ylin, sx], [f"{name}_lin"]),
                helper.make_node("Gather", ["color_flat", f"{name}_lin"], [f"{name}_raw"], axis=0),
            ]
        )


def _score_compact_quadrants(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto]) -> None:
    zero_i = _i64(inits, [0], "compact_score_zero_i")

    nodes.extend(
        [
            helper.make_node("Mul", ["qpi", _i64(inits, [MAX_N], "compact_score_fifteen_i")], ["bl_idx"]),
            helper.make_node("Add", ["bl_idx", "qpi"], ["br_idx"]),
            helper.make_node("Gather", ["color_flat", zero_i], ["tl_corner"], axis=0),
            helper.make_node("Gather", ["color_flat", "qpi"], ["tr_corner"], axis=0),
            helper.make_node("Gather", ["color_flat", "bl_idx"], ["bl_corner"], axis=0),
            helper.make_node("Gather", ["color_flat", "br_idx"], ["br_corner"], axis=0),
        ]
    )

    for name in ("tl", "tr", "bl", "br"):
        nodes.extend(
            [
                helper.make_node("Equal", [f"{name}_raw", f"{name}_corner"], [f"{name}_same"]),
                helper.make_node("Not", [f"{name}_same"], [f"{name}_diff"]),
                helper.make_node("And", [f"{name}_diff", "cq_valid"], [f"{name}_anom"]),
                helper.make_node("Cast", [f"{name}_anom"], [f"{name}_f"], to=TensorProto.FLOAT),
                helper.make_node("ReduceSum", [f"{name}_f"], [f"{name}_cell_score"], axes=[1, 2], keepdims=0),
            ]
        )

    for name in ("tl", "tr", "bl", "br"):
        others = [other for other in ("tl", "tr", "bl", "br") if other != name]
        nodes.extend(
            [
                helper.make_node("Equal", [f"{name}_corner", f"{others[0]}_corner"], [f"{name}_eq0"]),
                helper.make_node("Equal", [f"{name}_corner", f"{others[1]}_corner"], [f"{name}_eq1"]),
                helper.make_node("Equal", [f"{name}_corner", f"{others[2]}_corner"], [f"{name}_eq2"]),
                helper.make_node("Not", [f"{name}_eq0"], [f"{name}_ne0"]),
                helper.make_node("Not", [f"{name}_eq1"], [f"{name}_ne1"]),
                helper.make_node("Not", [f"{name}_eq2"], [f"{name}_ne2"]),
                helper.make_node("Cast", [f"{name}_ne0"], [f"{name}_ne0_f"], to=TensorProto.FLOAT),
                helper.make_node("Cast", [f"{name}_ne1"], [f"{name}_ne1_f"], to=TensorProto.FLOAT),
                helper.make_node("Cast", [f"{name}_ne2"], [f"{name}_ne2_f"], to=TensorProto.FLOAT),
                helper.make_node("Add", [f"{name}_ne0_f", f"{name}_ne1_f"], [f"{name}_corner_score0"]),
                helper.make_node("Add", [f"{name}_corner_score0", f"{name}_ne2_f"], [f"{name}_corner_score"]),
                helper.make_node("Add", [f"{name}_cell_score", f"{name}_corner_score"], [f"{name}_score"]),
            ]
        )

    nodes.extend(
        [
            helper.make_node("Greater", ["br_score", "tl_score"], ["br_gt_tl"]),
            helper.make_node("Greater", ["br_score", "tr_score"], ["br_gt_tr"]),
            helper.make_node("Greater", ["br_score", "bl_score"], ["br_gt_bl"]),
            helper.make_node("And", ["br_gt_tl", "br_gt_tr"], ["br_gt_tltr"]),
            helper.make_node("And", ["br_gt_tltr", "br_gt_bl"], ["br_best"]),
            helper.make_node("Greater", ["bl_score", "tl_score"], ["bl_gt_tl"]),
            helper.make_node("Greater", ["bl_score", "tr_score"], ["bl_gt_tr"]),
            helper.make_node("Greater", ["bl_score", "br_score"], ["bl_gt_br"]),
            helper.make_node("And", ["bl_gt_tl", "bl_gt_tr"], ["bl_gt_tltr"]),
            helper.make_node("And", ["bl_gt_tltr", "bl_gt_br"], ["bl_best"]),
            helper.make_node("Greater", ["tr_score", "tl_score"], ["tr_gt_tl"]),
            helper.make_node("Greater", ["tr_score", "bl_score"], ["tr_gt_bl"]),
            helper.make_node("Greater", ["tr_score", "br_score"], ["tr_gt_br"]),
            helper.make_node("And", ["tr_gt_tl", "tr_gt_bl"], ["tr_gt_tlbl"]),
            helper.make_node("And", ["tr_gt_tlbl", "tr_gt_br"], ["tr_best"]),
            helper.make_node("Or", ["br_best", "bl_best"], ["bottom_best"]),
            helper.make_node("Or", ["bottom_best", "tr_best"], ["not_tl_best"]),
            helper.make_node("Not", ["not_tl_best"], ["tl_best"]),
        ]
    )


def _add_compact_selected_output(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto]) -> None:
    colors = _i64(inits, np.arange(C, dtype=np.int64).reshape(1, C, 1, 1), "compact_colors")

    nodes.extend(
        [
            helper.make_node("Where", ["tr_best", "tr_raw", "tl_raw"], ["sel_tr_tl"]),
            helper.make_node("Where", ["bl_best", "bl_raw", "sel_tr_tl"], ["sel_bl"]),
            helper.make_node("Where", ["br_best", "br_raw", "sel_bl"], ["safe_color"]),
            helper.make_node("Unsqueeze", ["safe_color"], ["safe_color_ch"], axes=[1]),
            helper.make_node("Equal", ["safe_color_ch", colors], ["out7_bool"]),
            helper.make_node("Unsqueeze", ["cq_valid"], ["out_valid_ch"], axes=[1]),
            helper.make_node("And", ["out7_bool", "out_valid_ch"], ["out7_valid"]),
            helper.make_node("Cast", ["out7_valid"], ["out7"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out7"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - MAX_Q, W - MAX_Q]),
        ]
    )


def build_compact_model() -> onnx.ModelProto:
    """Build the lower-memory variant that scores compact 7x7 quadrants."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    _add_compact_prefix(nodes, inits)
    _add_compact_quadrants(nodes, inits)
    _score_compact_quadrants(nodes, inits)
    _add_compact_selected_output(nodes, inits)
    return _make_model(nodes, inits, f"{TASK_ID}_compact_quadrants")


def build_model(*, direct_onehot_scores: bool = False) -> onnx.ModelProto:
    """Build either the ArgMax-score variant or the direct one-hot-score variant."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    _add_common_prefix(nodes, inits, direct_onehot_scores=direct_onehot_scores)
    _score_quadrants(nodes, inits, direct_onehot_scores=direct_onehot_scores)
    _add_compact_output(nodes, inits)
    suffix = "onehot_scores" if direct_onehot_scores else "argmax_scores"
    return _make_model(nodes, inits, f"{TASK_ID}_{suffix}")


def _load_examples() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def validate_numpy() -> None:
    for split, examples in _load_examples().items():
        for idx, example in enumerate(examples):
            got = solve_grid(example["input"])
            want = np.asarray(example["output"], dtype=np.int64)
            if not np.array_equal(got, want):
                raise AssertionError(f"numpy failed {split}[{idx}]")


def validate_model(model: onnx.ModelProto) -> dict[str, tuple[int, int]]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise AssertionError("model failed sanitization")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        sanitized.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    counts: dict[str, tuple[int, int]] = {}
    for split, examples in _load_examples().items():
        passed = 0
        total = 0
        for idx, example in enumerate(examples):
            x = convert_to_numpy(example, "input")
            y = convert_to_numpy(example, "output")
            if x is None or y is None:
                continue
            total += 1
            out = session.run([OUT_NAME], {IN_NAME: x})[0]
            pred = (out > 0.0).astype(np.float32)
            if np.array_equal(pred, y):
                passed += 1
            else:
                raise AssertionError(f"model failed {split}[{idx}]")
        counts[split] = (passed, total)
    return counts


def measure_model(model: onnx.ModelProto, name: str) -> dict[str, Any]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise AssertionError(f"{name}: sanitizer rejected model")
    inputs = [
        arr
        for split in _load_examples().values()
        for example in split
        if (arr := convert_to_numpy(example, "input")) is not None
    ]
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(Path(tempfile.gettempdir()) / f"ng_{TASK_ID}_{name}")
    session = ort.InferenceSession(
        sanitized.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    for arr in inputs:
        session.run([OUT_NAME], {IN_NAME: arr})
    trace = session.end_profiling()
    memory = calculate_memory(sanitized, trace)
    params = calculate_params(sanitized)
    if memory is None or params is None:
        raise AssertionError(f"{name}: failed memory/param measurement")
    cost = int(memory + params)
    return {
        "name": name,
        "memory": int(memory),
        "params": int(params),
        "cost": cost,
        "score": max(1.0, 25.0 - math.log(max(1.0, float(cost)))),
    }


def main() -> None:
    validate_numpy()
    candidates = [
        ("compact_quadrants", build_compact_model()),
        ("argmax_scores", build_model(direct_onehot_scores=False)),
        ("onehot_scores", build_model(direct_onehot_scores=True)),
    ]
    results: list[tuple[dict[str, Any], onnx.ModelProto]] = []
    for name, model in candidates:
        counts = validate_model(model)
        stats = measure_model(model, name)
        print(
            f"{name}: valid {counts}; memory={stats['memory']} params={stats['params']} "
            f"cost={stats['cost']} score={stats['score']:.6f}"
        )
        results.append((stats, model))

    best_stats, best_model = min(results, key=lambda item: int(item[0]["cost"]))
    onnx.save(best_model, BEST_PATH)
    report = score_file(BEST_PATH)
    print(f"saved {BEST_PATH}")
    print(
        f"best={best_stats['name']} scorer_valid={report['valid']} "
        f"memory={report['memory']} params={report['params']} "
        f"cost={report['cost']} score={report['score']:.6f}"
    )


if __name__ == "__main__":
    main()
