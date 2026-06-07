"""Generate NeuroGolf task107 ONNX models.

Task rule: the 5x5 input is expanded by a scale factor equal to the number of
distinct colors in the input minus one. Each input cell becomes an s by s block
inside the top-left 5s by 5s output area, with the rest of the 30x30 Kaggle
canvas left empty. A 2x2 non-background block appears at one of three positions
near the top-left; red diagonal rays are drawn through the surrounding
background from the four outer corners of the scaled block.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task107"
OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
IN = 5
SHAPE = [1, C, H, W]
SCALES = [2, 3, 4, 5, 6]
STARTS = [(0, 1), (1, 0), (1, 1)]
IR_VERSION = 10


def _init(inits: list[onnx.TensorProto], name: str, arr: np.ndarray) -> str:
    inits.append(numpy_helper.from_array(arr, name=name))
    return name


def _i64(inits: list[onnx.TensorProto], name: str, vals: list[int] | np.ndarray) -> str:
    return _init(inits, name, np.asarray(vals, dtype=np.int64))


def _f32(inits: list[onnx.TensorProto], name: str, vals: list[float] | np.ndarray) -> str:
    return _init(inits, name, np.asarray(vals, dtype=np.float32))


def _bool(inits: list[onnx.TensorProto], name: str, vals: np.ndarray) -> str:
    return _init(inits, name, np.asarray(vals, dtype=bool))


def _onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros((1, C, H, W), dtype=np.float32)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def solve_grid(inp: list[list[int]] | np.ndarray) -> np.ndarray:
    x = np.asarray(inp, dtype=np.int64)
    s = len(set(int(v) for v in x.ravel())) - 1
    out = np.zeros((H, W), dtype=np.int64)
    for r in range(IN):
        for c in range(IN):
            out[r * s : (r + 1) * s, c * s : (c + 1) * s] = x[r, c]

    coords = [(r, c) for r in range(4) for c in range(4) if x[r, c] != 0]
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    r1, c1 = r0 + 2, c0 + 2
    size = 5 * s
    for rr, cc, dr, dc in (
        (r0 * s - 1, c0 * s - 1, -1, -1),
        (r0 * s - 1, c1 * s, -1, 1),
        (r1 * s, c0 * s - 1, 1, -1),
        (r1 * s, c1 * s, 1, 1),
    ):
        while 0 <= rr < size and 0 <= cc < size:
            if out[rr, cc] == 0:
                out[rr, cc] = 2
            rr += dr
            cc += dc
    return out


def _red_mask(s: int, r0: int, c0: int) -> np.ndarray:
    mask = np.zeros((1, 1, H, W), dtype=bool)
    size = 5 * s
    r1, c1 = r0 + 2, c0 + 2
    for rr, cc, dr, dc in (
        (r0 * s - 1, c0 * s - 1, -1, -1),
        (r0 * s - 1, c1 * s, -1, 1),
        (r1 * s, c0 * s - 1, 1, -1),
        (r1 * s, c1 * s, 1, 1),
    ):
        while 0 <= rr < size and 0 <= cc < size:
            mask[0, 0, rr, cc] = True
            rr += dr
            cc += dc
    return mask


def _active_mask(s: int) -> np.ndarray:
    mask = np.zeros((1, 1, H, W), dtype=bool)
    mask[:, :, : 5 * s, : 5 * s] = True
    return mask


def _scale_indices(s: int) -> np.ndarray:
    idx = np.zeros((H,), dtype=np.int64)
    for i in range(5 * s):
        idx[i] = i // s
    return idx


def _start_predicates(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], core: str) -> dict[tuple[int, int], str]:
    axes = _i64(inits, "axes_all", [0, 1, 2, 3])
    zero = _f32(inits, "zero", [0.0])
    preds: dict[tuple[int, int], str] = {}
    for r, c in STARTS:
        st = _i64(inits, f"st_{r}_{c}", [0, 1, r, c])
        en = _i64(inits, f"en_{r}_{c}", [1, C, r + 1, c + 1])
        nodes.append(helper.make_node("Slice", [core, st, en, axes], [f"cell_{r}_{c}"]))
        nodes.append(helper.make_node("ReduceSum", [f"cell_{r}_{c}"], [f"sum_{r}_{c}"], keepdims=0))
        nodes.append(helper.make_node("Greater", [f"sum_{r}_{c}", zero], [f"pos_{r}_{c}"]))
        preds[(r, c)] = f"pos_{r}_{c}"
    return preds


def build_gather_branch_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    x_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, "axes", [0, 1, 2, 3])
    st = _i64(inits, "st", [0, 0, 0, 0])
    en = _i64(inits, "en", [1, C, IN, IN])
    zero = _f32(inits, "z", [0.0])
    one = _f32(inits, "o", np.array(1.0, dtype=np.float32))
    red_channel = _f32(inits, "red_ch", np.array([[[[0.0]], [[0.0]], [[1.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]]]], dtype=np.float32))

    nodes.append(helper.make_node("Slice", ["input", st, en, axes], ["core"]))
    nodes.append(helper.make_node("ReduceSum", ["core"], ["color_sum"], axes=[2, 3], keepdims=0))
    nodes.append(helper.make_node("Greater", ["color_sum", zero], ["color_used"]))
    nodes.append(helper.make_node("Cast", ["color_used"], ["color_used_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("ReduceSum", ["color_used_f"], ["ncolors"], keepdims=0))
    nodes.append(helper.make_node("Sub", ["ncolors", one], ["scale_f"]))
    nodes.append(helper.make_node("Cast", ["scale_f"], ["scale_i"], to=TensorProto.INT64))

    starts = _start_predicates(nodes, inits, "core")

    scale_outputs: list[str] = []
    for s in SCALES:
        rows = _i64(inits, f"rows{s}", _scale_indices(s))
        cols = _i64(inits, f"cols{s}", _scale_indices(s))
        active = _bool(inits, f"active{s}", _active_mask(s))
        ch0_en = _i64(inits, f"ch0_en{s}", [1, 1, H, W])
        nodes.append(helper.make_node("Gather", ["core", rows], [f"row{s}"], axis=2))
        nodes.append(helper.make_node("Gather", [f"row{s}", cols], [f"scaled{s}"], axis=3))
        nodes.append(helper.make_node("Where", [active, f"scaled{s}", zero], [f"active_scaled{s}"]))
        nodes.append(helper.make_node("Slice", [f"active_scaled{s}", st, ch0_en, axes], [f"bg{s}"]))
        nodes.append(helper.make_node("Greater", [f"bg{s}", zero], [f"is_bg{s}"]))

        red_masks: list[str] = []
        for r, c in STARTS:
            rm = _bool(inits, f"red{s}_{r}_{c}", _red_mask(s, r, c))
            nodes.append(helper.make_node("And", [rm, f"is_bg{s}"], [f"redcond{s}_{r}_{c}"]))
            nodes.append(helper.make_node("Where", [f"redcond{s}_{r}_{c}", red_channel, f"active_scaled{s}"], [f"redout{s}_{r}_{c}"]))
            red_masks.append(f"redout{s}_{r}_{c}")
        nodes.append(helper.make_node("Where", [starts[(1, 0)], red_masks[1], red_masks[2]], [f"starttmp{s}"]))
        nodes.append(helper.make_node("Where", [starts[(0, 1)], red_masks[0], f"starttmp{s}"], [f"branch{s}"]))
        scale_outputs.append(f"branch{s}")

    cur = scale_outputs[-1]
    for s, name in reversed(list(zip(SCALES[:-1], scale_outputs[:-1]))):
        sf = _i64(inits, f"sf{s}", [s])
        nodes.append(helper.make_node("Equal", ["scale_i", sf], [f"is{s}"]))
        nodes.append(helper.make_node("Where", [f"is{s}", name, cur], [f"sel{s}"]))
        cur = f"sel{s}"
    nodes.append(helper.make_node("Identity", [cur], ["output"]))

    graph = helper.make_graph(nodes, "task107_gather", [x_info], [y_info], initializer=inits)
    model = helper.make_model(graph, producer_name="", ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 10)])
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_dynamic_gather_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    x_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, "axes", [0, 1, 2, 3])
    st = _i64(inits, "st", [0, 0, 0, 0])
    en = _i64(inits, "en", [1, C, IN, IN])
    ch0_en = _i64(inits, "ch0_en", [1, 1, H, W])
    zero = _f32(inits, "z", [0.0])
    one = _f32(inits, "o", [1.0])
    red_channel = _f32(inits, "red_ch", np.array([[[[0.0]], [[0.0]], [[1.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]]]], dtype=np.float32))

    nodes.append(helper.make_node("Slice", ["input", st, en, axes], ["core"]))
    nodes.append(helper.make_node("ReduceSum", ["core"], ["color_sum"], axes=[2, 3], keepdims=0))
    nodes.append(helper.make_node("Greater", ["color_sum", zero], ["color_used"]))
    nodes.append(helper.make_node("Cast", ["color_used"], ["color_used_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("ReduceSum", ["color_used_f"], ["ncolors"], keepdims=0))
    nodes.append(helper.make_node("Sub", ["ncolors", one], ["scale_f"]))
    nodes.append(helper.make_node("Cast", ["scale_f"], ["scale_i"], to=TensorProto.INT64))
    starts = _start_predicates(nodes, inits, "core")
    two_i = _i64(inits, "two_i", np.array(2, dtype=np.int64))
    rows_table = _i64(inits, "rows_table", np.stack([_scale_indices(s) for s in SCALES], axis=0))
    red_table = _bool(
        inits,
        "red_table",
        np.stack([[ _red_mask(s, r, c) for r, c in STARTS] for s in SCALES], axis=0),
    )
    pos01_f = "pos_0_1_f"
    pos10_f = "pos_1_0_f"
    nodes.append(helper.make_node("Cast", [starts[(0, 1)]], [pos01_f], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Cast", [starts[(1, 0)]], [pos10_f], to=TensorProto.FLOAT))
    two_f = _f32(inits, "two_f", np.array(2.0, dtype=np.float32))
    nodes.append(helper.make_node("Mul", [pos01_f, two_f], ["pos01x2"]))
    nodes.append(helper.make_node("Sub", [two_f, "pos01x2"], ["start_base"]))
    nodes.append(helper.make_node("Sub", ["start_base", pos10_f], ["start_f"]))
    nodes.append(helper.make_node("Cast", ["start_f"], ["start_i_v"], to=TensorProto.INT64))
    nodes.append(helper.make_node("Squeeze", ["start_i_v"], ["start_i"], axes=[0]))
    nodes.append(helper.make_node("Sub", ["scale_i", two_i], ["scale_idx_v"]))
    nodes.append(helper.make_node("Squeeze", ["scale_idx_v"], ["scale_idx"], axes=[0]))
    nodes.append(helper.make_node("Gather", [rows_table, "scale_idx"], ["row_idx"], axis=0))
    nodes.append(helper.make_node("Gather", [red_table, "scale_idx"], ["red_scale"], axis=0))
    nodes.append(helper.make_node("Gather", ["red_scale", "start_i"], ["red_mask"], axis=0))

    five_f = _f32(inits, "five_f", np.array(5.0, dtype=np.float32))
    y_pos = _f32(inits, "y_pos", np.arange(H, dtype=np.float32).reshape(1, 1, H, 1))
    x_pos = _f32(inits, "x_pos", np.arange(W, dtype=np.float32).reshape(1, 1, 1, W))
    nodes.append(helper.make_node("Mul", ["scale_f", five_f], ["size_f"]))
    nodes.append(helper.make_node("Less", [y_pos, "size_f"], ["y_active"]))
    nodes.append(helper.make_node("Less", [x_pos, "size_f"], ["x_active"]))
    nodes.append(helper.make_node("And", ["y_active", "x_active"], ["active_mask"]))

    nodes.append(helper.make_node("Gather", ["core", "row_idx"], ["row_scaled"], axis=2))
    nodes.append(helper.make_node("Gather", ["row_scaled", "row_idx"], ["scaled"], axis=3))
    nodes.append(helper.make_node("Where", ["active_mask", "scaled", zero], ["active_scaled"]))
    nodes.append(helper.make_node("Slice", ["active_scaled", st, ch0_en, axes], ["bg"]))
    nodes.append(helper.make_node("Greater", ["bg", zero], ["is_bg"]))

    nodes.append(helper.make_node("And", ["red_mask", "is_bg"], ["red_cond"]))
    nodes.append(helper.make_node("Where", ["red_cond", red_channel, "active_scaled"], ["output"]))

    graph = helper.make_graph(nodes, "task107_dynamic_gather", [x_info], [y_info], initializer=inits)
    model = helper.make_model(graph, producer_name="", ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 10)])
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_dynamic_bool_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    x_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, "axes", [0, 1, 2, 3])
    st = _i64(inits, "st", [0, 0, 0, 0])
    en = _i64(inits, "en", [1, C, IN, IN])
    ch0_en = _i64(inits, "ch0_en", [1, 1, H, W])
    zero = _f32(inits, "z", [0.0])
    one = _f32(inits, "o", np.array(1.0, dtype=np.float32))
    nodes.append(helper.make_node("Slice", ["input", st, en, axes], ["core"]))
    nodes.append(helper.make_node("Greater", ["core", zero], ["coreb"]))
    nodes.append(helper.make_node("ReduceSum", ["core"], ["color_sum"], axes=[2, 3], keepdims=0))
    nodes.append(helper.make_node("Greater", ["color_sum", zero], ["color_used"]))
    nodes.append(helper.make_node("Cast", ["color_used"], ["color_used_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("ReduceSum", ["color_used_f"], ["ncolors"], keepdims=0))
    nodes.append(helper.make_node("Sub", ["ncolors", one], ["scale_f"]))
    nodes.append(helper.make_node("Cast", ["scale_f"], ["scale_i"], to=TensorProto.INT64))
    starts = _start_predicates(nodes, inits, "core")

    two_i = _i64(inits, "two_i", np.array(2, dtype=np.int64))
    rows_table = _i64(inits, "rows_table", np.stack([_scale_indices(s) for s in SCALES], axis=0))
    red_table = _bool(
        inits,
        "red_table",
        np.stack([[_red_mask(s, r, c) for r, c in STARTS] for s in SCALES], axis=0),
    )
    nodes.append(helper.make_node("Cast", [starts[(0, 1)]], ["pos_0_1_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Cast", [starts[(1, 0)]], ["pos_1_0_f"], to=TensorProto.FLOAT))
    two_f = _f32(inits, "two_f", np.array(2.0, dtype=np.float32))
    nodes.append(helper.make_node("Mul", ["pos_0_1_f", two_f], ["pos01x2"]))
    nodes.append(helper.make_node("Sub", [two_f, "pos01x2"], ["start_base"]))
    nodes.append(helper.make_node("Sub", ["start_base", "pos_1_0_f"], ["start_f"]))
    nodes.append(helper.make_node("Cast", ["start_f"], ["start_i_v"], to=TensorProto.INT64))
    nodes.append(helper.make_node("Squeeze", ["start_i_v"], ["start_i"], axes=[0]))
    nodes.append(helper.make_node("Sub", ["scale_i", two_i], ["scale_idx"]))
    nodes.append(helper.make_node("Gather", [rows_table, "scale_idx"], ["row_idx"], axis=0))
    nodes.append(helper.make_node("Gather", [red_table, "scale_idx"], ["red_scale"], axis=0))
    nodes.append(helper.make_node("Gather", ["red_scale", "start_i"], ["red_mask"], axis=0))

    five_f = _f32(inits, "five_f", np.array(5.0, dtype=np.float32))
    y_pos = _f32(inits, "y_pos", np.arange(H, dtype=np.float32).reshape(1, 1, H, 1))
    x_pos = _f32(inits, "x_pos", np.arange(W, dtype=np.float32).reshape(1, 1, 1, W))
    nodes.append(helper.make_node("Mul", ["scale_f", five_f], ["size_f"]))
    nodes.append(helper.make_node("Less", [y_pos, "size_f"], ["y_active"]))
    nodes.append(helper.make_node("Less", [x_pos, "size_f"], ["x_active"]))
    nodes.append(helper.make_node("And", ["y_active", "x_active"], ["active_mask"]))

    nodes.append(helper.make_node("Gather", ["coreb", "row_idx"], ["row_scaled"], axis=2))
    nodes.append(helper.make_node("Gather", ["row_scaled", "row_idx"], ["scaled"], axis=3))
    ch1_st = _i64(inits, "ch1_st", [0, 1, 0, 0])
    ch1_en = _i64(inits, "ch1_en", [1, 2, H, W])
    ch2_st = _i64(inits, "ch2_st", [0, 2, 0, 0])
    ch2_en = _i64(inits, "ch2_en", [1, 3, H, W])
    ch3_st = _i64(inits, "ch3_st", [0, 3, 0, 0])
    ch3_en = _i64(inits, "ch3_en", [1, C, H, W])
    nodes.append(helper.make_node("Slice", ["scaled", st, ch0_en, axes], ["ch0_raw"]))
    nodes.append(helper.make_node("And", ["ch0_raw", "active_mask"], ["ch0_active"]))
    nodes.append(helper.make_node("And", ["red_mask", "ch0_active"], ["red_cond"]))
    nodes.append(helper.make_node("Not", ["red_cond"], ["not_red_cond"]))
    nodes.append(helper.make_node("And", ["ch0_active", "not_red_cond"], ["ch0_keep"]))
    nodes.append(helper.make_node("Slice", ["scaled", ch1_st, ch1_en, axes], ["ch1"]))
    nodes.append(helper.make_node("And", ["ch1", "active_mask"], ["ch1_active"]))
    nodes.append(helper.make_node("Slice", ["scaled", ch2_st, ch2_en, axes], ["ch2"]))
    nodes.append(helper.make_node("And", ["ch2", "active_mask"], ["ch2_active"]))
    nodes.append(helper.make_node("Or", ["ch2_active", "red_cond"], ["ch2_red"]))
    nodes.append(helper.make_node("Slice", ["scaled", ch3_st, ch3_en, axes], ["ch3_9"]))
    nodes.append(helper.make_node("And", ["ch3_9", "active_mask"], ["ch3_9_active"]))
    nodes.append(helper.make_node("Concat", ["ch0_keep", "ch1_active", "ch2_red", "ch3_9_active"], ["outb"], axis=1))
    nodes.append(helper.make_node("Cast", ["outb"], ["output"], to=TensorProto.FLOAT))

    graph = helper.make_graph(nodes, "task107_dynamic_bool", [x_info], [y_info], initializer=inits)
    model = helper.make_model(graph, producer_name="", ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 10)])
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_resize_mask_model() -> onnx.ModelProto:
    # Nearest Resize is compact in nodes, but opset-10 Resize needs scales and
    # produces fixed 30x30 candidates. It is kept as a measured alternative.
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    x_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, "axes", [0, 1, 2, 3])
    st = _i64(inits, "st", [0, 0, 0, 0])
    en = _i64(inits, "en", [1, C, IN, IN])
    zero = _f32(inits, "z", [0.0])
    one = _f32(inits, "o", [1.0])
    red_channel = _f32(inits, "red_ch", np.array([[[[0.0]], [[0.0]], [[1.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]]]], dtype=np.float32))

    nodes.append(helper.make_node("Slice", ["input", st, en, axes], ["core"]))
    nodes.append(helper.make_node("ReduceSum", ["core"], ["color_sum"], axes=[2, 3], keepdims=0))
    nodes.append(helper.make_node("Greater", ["color_sum", zero], ["color_used"]))
    nodes.append(helper.make_node("Cast", ["color_used"], ["color_used_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("ReduceSum", ["color_used_f"], ["ncolors"], keepdims=0))
    nodes.append(helper.make_node("Sub", ["ncolors", one], ["scale_f"]))
    nodes.append(helper.make_node("Cast", ["scale_f"], ["scale_i"], to=TensorProto.INT64))
    starts = _start_predicates(nodes, inits, "core")

    branches: list[str] = []
    for s in SCALES:
        scales = _f32(inits, f"scales{s}", [1.0, 1.0, float(s), float(s)])
        active = _bool(inits, f"active{s}", _active_mask(s))
        ch0_en = _i64(inits, f"ch0_en{s}", [1, 1, H, W])
        pads = [0, 0, 0, 0, 0, 0, H - 5 * s, W - 5 * s]
        nodes.append(helper.make_node("Resize", ["core", scales], [f"res{s}"], mode="nearest"))
        nodes.append(helper.make_node("Pad", [f"res{s}"], [f"pad{s}"], pads=pads))
        nodes.append(helper.make_node("Where", [active, f"pad{s}", zero], [f"active_scaled{s}"]))
        nodes.append(helper.make_node("Slice", [f"active_scaled{s}", st, ch0_en, axes], [f"bg{s}"]))
        nodes.append(helper.make_node("Greater", [f"bg{s}", zero], [f"is_bg{s}"]))
        red_masks: list[str] = []
        for r, c in STARTS:
            rm = _bool(inits, f"red{s}_{r}_{c}", _red_mask(s, r, c))
            nodes.append(helper.make_node("And", [rm, f"is_bg{s}"], [f"redcond{s}_{r}_{c}"]))
            nodes.append(helper.make_node("Where", [f"redcond{s}_{r}_{c}", red_channel, f"active_scaled{s}"], [f"redout{s}_{r}_{c}"]))
            red_masks.append(f"redout{s}_{r}_{c}")
        nodes.append(helper.make_node("Where", [starts[(1, 0)], red_masks[1], red_masks[2]], [f"starttmp{s}"]))
        nodes.append(helper.make_node("Where", [starts[(0, 1)], red_masks[0], f"starttmp{s}"], [f"branch{s}"]))
        branches.append(f"branch{s}")

    cur = branches[-1]
    for s, name in reversed(list(zip(SCALES[:-1], branches[:-1]))):
        sf = _i64(inits, f"sf{s}", [s])
        nodes.append(helper.make_node("Equal", ["scale_i", sf], [f"is{s}"]))
        nodes.append(helper.make_node("Where", [f"is{s}", name, cur], [f"sel{s}"]))
        cur = f"sel{s}"
    nodes.append(helper.make_node("Identity", [cur], ["output"]))

    graph = helper.make_graph(nodes, "task107_resize", [x_info], [y_info], initializer=inits)
    model = helper.make_model(graph, producer_name="", ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 10)])
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_manual_constant_model() -> onnx.ModelProto:
    # A high-parameter but low-logic fallback: precompute coordinate indices as
    # in the Gather model, but use one 30x30 red label map per scale/start.
    return build_gather_branch_model()


def validate(path: Path) -> None:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for i, ex in enumerate(data[split]):
            pred = session.run(["output"], {"input": _onehot(ex["input"])})[0]
            expected = _onehot(ex["output"])
            if not np.array_equal(pred > 0.0, expected > 0.0):
                got = np.argmax(pred[0], axis=0)
                exp = np.argmax(expected[0], axis=0)
                diff = np.argwhere((pred > 0.0) != (expected > 0.0))
                raise AssertionError(f"{split} {i} failed at {diff[:5].tolist()} got={got[:6,:6]} exp={exp[:6,:6]}")


def build_and_measure() -> tuple[str, dict[str, object]]:
    builders: list[tuple[str, Callable[[], onnx.ModelProto]]] = [
        ("A_resize_nearest_masks", build_resize_mask_model),
        ("B_gather_indices_masks", build_gather_branch_model),
        ("C_manual_concat_proxy", build_manual_constant_model),
        ("D_dynamic_gather_selected_masks", build_dynamic_gather_model),
        ("E_dynamic_bool_internals", build_dynamic_bool_model),
    ]
    results: list[tuple[str, Path, dict[str, object]]] = []
    with tempfile.TemporaryDirectory(prefix="task107_") as td:
        tmp = Path(td)
        for name, builder in builders:
            path = tmp / f"{TASK_ID}_{name}.onnx"
            onnx.save(builder(), path)
            validate(path)
            result = score_file(path)
            results.append((name, path, result))
            print(
                f"{name}: memory={result['memory']} params={result['params']} "
                f"cost={result['cost']} score={result['score']}"
            )

        valid = [item for item in results if item[2]["valid"]]
        if not valid:
            raise RuntimeError("no valid variants")
        name, path, result = min(valid, key=lambda item: int(item[2]["cost"]))
        BEST_PATH.write_bytes(path.read_bytes())
        return name, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure", action="store_true", help="build all variants and keep the cheapest")
    args = parser.parse_args()
    if args.measure:
        name, result = build_and_measure()
    else:
        onnx.save(build_dynamic_bool_model(), BEST_PATH)
        validate(BEST_PATH)
        result = score_file(BEST_PATH)
        name = "E_dynamic_bool_internals"
    print(
        f"kept {name} at {BEST_PATH}: memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={float(result['score']):.6f}"
    )


if __name__ == "__main__":
    main()
