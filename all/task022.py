"""Compact ONNX for ARC task022: align anchor-centered mini-objects.

Task rule: the 11x11 input contains small objects made from one gray anchor
(color 5) and one or more nearby same-colored non-gray cells. Translate every
object so its gray anchor lands at the center of a 3x3 output. Non-gray cells
keep their relative offset from the anchor, all anchors merge into one gray
center cell, and cells with no translated color are black.
All provided examples are 11x11, place gray anchors only in rows/cols 1..9,
and have no conflicting non-gray colors for the same anchor-relative offset.

ONNX: convert the 11x11 input area to one scalar color-value plane with a
dilated 2x2 convolution, mask each non-center offset by the 9x9 gray-anchor
plane, reduce each offset to one color value, cast the tiny 3x3 value grid to
integer color IDs, compare it against the ten color IDs, then pad the 3x3
one-hot result to the required 30x30 output.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

BEST_PATH = OUT_DIR / "task022.onnx"
DATA_PATH = ROOT / "data" / "task022.json"

C = 10
H = W = 30
SH = SW = 11
OH = OW = 3
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference anchor-alignment solver for local correctness checks."""
    g = np.asarray(grid, dtype=np.int64)
    out = np.zeros((OH, OW), dtype=np.int64)
    out[1, 1] = 5
    anchors = np.argwhere(g == 5)
    for ar, ac in anchors:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = int(ar + dr), int(ac + dc)
                if rr < 0 or rr >= g.shape[0] or cc < 0 or cc >= g.shape[1]:
                    continue
                color = int(g[rr, cc])
                if color not in {0, 5}:
                    out[1 + dr, 1 + dc] = color
    return out


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _conv_weights() -> np.ndarray:
    weights = np.zeros((len(COLORS) * OH * OW, 1, OH, OW), dtype=np.float32)
    for color_idx in range(len(COLORS)):
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                out_chan = color_idx * OH * OW + (dr + 1) * OW + (dc + 1)
                weights[out_chan, 0, dr + 1, dc + 1] = 1.0
    return weights


def _gray_shift_weights() -> np.ndarray:
    weights = np.zeros((OH * OW, 1, OH, OW), dtype=np.float32)
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            out_chan = (dr + 1) * OW + (dc + 1)
            weights[out_chan, 0, 1 - dr, 1 - dc] = 1.0
    return weights


def _color_index_weights(crop_conv: bool = False) -> np.ndarray:
    if crop_conv:
        weights = np.zeros((1, C, 2, 2), dtype=np.float32)
        weights[0, :, 0, 0] = np.arange(C, dtype=np.float32)
    else:
        weights = np.arange(C, dtype=np.float32).reshape(1, C, 1, 1)
    weights[0, 5, 0, 0] = 0.0
    return weights


def build_scalar_model(use_half: bool = False, crop_conv: bool = False) -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    weight = _f32(inits, _color_index_weights(crop_conv), "idx_w")
    palette = _init(inits, np.arange(C, dtype=np.int32).reshape(1, C, 1, 1), "palette")
    center_dtype = np.float16 if use_half else np.float32
    center_value = _init(inits, np.array([[[[5.0]]]], dtype=center_dtype), "center_value")
    zero_value = _init(inits, np.array(0.0, dtype=center_dtype), "zero_value") if use_half else ""
    axes_chw = _i64(inits, [1, 2, 3], "axes_chw")
    axes_hw = _i64(inits, [2, 3], "axes_hw")
    g_st = _i64(inits, [5, 1, 1], "g_st")
    g_en = _i64(inits, [6, 10, 10], "g_en")
    out_pads = [0, 0, 0, 0, 0, 0, H - OH, W - OW]

    nodes.extend(
        [
            helper.make_node(
                "Conv",
                [IN_NAME, weight],
                ["color_idx"],
                dilations=[19, 19] if crop_conv else [1, 1],
            ),
            helper.make_node("Slice", [IN_NAME, g_st, g_en, axes_chw], ["gray"]),
        ]
    )
    color_source = "color_idx"
    gray_source = "gray"
    if use_half:
        nodes.extend(
            [
                helper.make_node("Cast", ["color_idx"], ["color_idx_h"], to=TensorProto.FLOAT16),
                helper.make_node("Cast", ["gray"], ["gray_b"], to=TensorProto.BOOL),
            ]
        )
        color_source = "color_idx_h"
        gray_source = "gray_b"

    row_tensors: list[str] = []
    for dr in (-1, 0, 1):
        cell_tensors: list[str] = []
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                cell_tensors.append(center_value)
                continue
            tag = f"s{dr + 1}{dc + 1}"
            cs = _i64(inits, [1 + dr, 1 + dc], f"{tag}_cs")
            ce = _i64(inits, [10 + dr, 10 + dc], f"{tag}_ce")
            nodes.append(helper.make_node("Slice", [color_source, cs, ce, axes_hw], [f"{tag}_c"]))
            if use_half:
                nodes.append(helper.make_node("Where", [gray_source, f"{tag}_c", zero_value], [f"{tag}_m"]))
            else:
                nodes.append(helper.make_node("Mul", [f"{tag}_c", gray_source], [f"{tag}_m"]))
            nodes.append(helper.make_node("ReduceMax", [f"{tag}_m"], [f"{tag}_v"], axes=[2, 3], keepdims=1))
            cell_tensors.append(f"{tag}_v")
        row = f"srow{dr + 1}"
        nodes.append(helper.make_node("Concat", cell_tensors, [row], axis=3))
        row_tensors.append(row)

    nodes.extend(
        [
            helper.make_node("Concat", row_tensors, ["out_values"], axis=2),
            helper.make_node("Cast", ["out_values"], ["out_i32"], to=TensorProto.INT32),
            helper.make_node("Equal", ["out_i32", palette], ["out_bool"]),
            helper.make_node("Cast", ["out_bool"], ["out3"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=out_pads),
        ]
    )

    if crop_conv:
        graph_name = "task022_scalar_crop_f16" if use_half else "task022_scalar_crop"
    else:
        graph_name = "task022_scalar_f16" if use_half else "task022_scalar"
    graph = helper.make_graph(nodes, graph_name, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    c1_st = _i64(inits, [0, 1, 0, 0], "c1_st")
    c1_en = _i64(inits, [1, 5, SH, SW], "c1_en")
    c6_st = _i64(inits, [0, 6, 0, 0], "c6_st")
    c6_en = _i64(inits, [1, 10, SH, SW], "c6_en")
    g_st = _i64(inits, [0, 5, 0, 0], "g_st")
    g_en = _i64(inits, [1, 6, SH, SW], "g_en")
    corr_shape = _i64(inits, [1, len(COLORS), OH, OW], "corr_shape")
    half = _f32(inits, [0.5], "half")
    weight = _f32(inits, _conv_weights(), "w")
    center = _f32(
        inits,
        np.array([[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]], dtype=np.float32),
        "center",
    )
    out_pads = [0, 0, 0, 0, 0, 0, H - OH, W - OW]

    def slice_ch(inp: str, ch0: int, ch1: int, name: str) -> str:
        st = _i64(inits, [0, ch0, 0, 0], f"{name}_st")
        en = _i64(inits, [1, ch1, OH, OW], f"{name}_en")
        nodes.append(helper.make_node("Slice", [inp, st, en, axes4], [name]))
        return name

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, c1_st, c1_en, axes4], ["c1_4"]),
            helper.make_node("Slice", [IN_NAME, c6_st, c6_en, axes4], ["c6_9"]),
            helper.make_node("Concat", ["c1_4", "c6_9"], ["colors"], axis=1),
            helper.make_node("Slice", [IN_NAME, g_st, g_en, axes4], ["gray"]),
            helper.make_node(
                "Conv",
                ["colors", weight],
                ["offset_hits"],
                pads=[1, 1, 1, 1],
                group=len(COLORS),
            ),
            helper.make_node("Mul", ["offset_hits", "gray"], ["anchored"]),
            helper.make_node("ReduceMax", ["anchored"], ["has_offset"], axes=[2, 3], keepdims=0),
            helper.make_node("Reshape", ["has_offset", corr_shape], ["corr"]),
        ]
    )

    c1 = slice_ch("corr", 0, 1, "o1")
    c2 = slice_ch("corr", 1, 2, "o2")
    c3 = slice_ch("corr", 2, 3, "o3")
    c4 = slice_ch("corr", 3, 4, "o4")
    c6 = slice_ch("corr", 4, 5, "o6")
    c7 = slice_ch("corr", 5, 6, "o7")
    c8 = slice_ch("corr", 6, 7, "o8")
    c9 = slice_ch("corr", 7, 8, "o9")

    nodes.extend(
        [
            helper.make_node("ReduceMax", ["corr"], ["any_color"], axes=[1], keepdims=1),
            helper.make_node("Add", ["any_color", "center"], ["occupied"]),
            helper.make_node("Greater", ["occupied", half], ["occ_b"]),
            helper.make_node("Not", ["occ_b"], ["black_b"]),
            helper.make_node("Cast", ["black_b"], ["black"], to=TensorProto.FLOAT),
            helper.make_node("Concat", ["black", c1, c2, c3, c4, "center", c6, c7, c8, c9], ["out3"], axis=1),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=out_pads),
        ]
    )

    graph = helper.make_graph(nodes, "task022", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_slice_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    c1_st = _i64(inits, [0, 1, 0, 0], "c1_st")
    c1_en = _i64(inits, [1, 5, SH, SW], "c1_en")
    c6_st = _i64(inits, [0, 6, 0, 0], "c6_st")
    c6_en = _i64(inits, [1, 10, SH, SW], "c6_en")
    g_st = _i64(inits, [0, 5, 0, 0], "g_st")
    g_en = _i64(inits, [1, 6, SH, SW], "g_en")
    half = _f32(inits, [0.5], "half")
    center = _f32(
        inits,
        np.array([[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]], dtype=np.float32),
        "center",
    )
    out_pads = [0, 0, 0, 0, 0, 0, H - OH, W - OW]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, c1_st, c1_en, axes4], ["c1_4"]),
            helper.make_node("Slice", [IN_NAME, c6_st, c6_en, axes4], ["c6_9"]),
            helper.make_node("Concat", ["c1_4", "c6_9"], ["colors"], axis=1),
            helper.make_node("Slice", [IN_NAME, g_st, g_en, axes4], ["gray"]),
        ]
    )

    row_tensors: list[str] = []
    for dr in (-1, 0, 1):
        cell_tensors: list[str] = []
        for dc in (-1, 0, 1):
            tag = f"d{dr + 1}{dc + 1}"
            ar0 = max(0, -dr)
            ar1 = min(SH, SH - dr)
            ac0 = max(0, -dc)
            ac1 = min(SW, SW - dc)
            cr0 = ar0 + dr
            cr1 = ar1 + dr
            cc0 = ac0 + dc
            cc1 = ac1 + dc
            gs = _i64(inits, [0, 0, ar0, ac0], f"{tag}_gs")
            ge = _i64(inits, [1, 1, ar1, ac1], f"{tag}_ge")
            cs = _i64(inits, [0, 0, cr0, cc0], f"{tag}_cs")
            ce = _i64(inits, [1, len(COLORS), cr1, cc1], f"{tag}_ce")
            nodes.extend(
                [
                    helper.make_node("Slice", ["gray", gs, ge, axes4], [f"{tag}_g"]),
                    helper.make_node("Slice", ["colors", cs, ce, axes4], [f"{tag}_c"]),
                    helper.make_node("Mul", [f"{tag}_c", f"{tag}_g"], [f"{tag}_m"]),
                    helper.make_node("ReduceMax", [f"{tag}_m"], [f"{tag}_v"], axes=[2, 3], keepdims=0),
                    helper.make_node("Unsqueeze", [f"{tag}_v"], [f"{tag}_u"], axes=[2, 3]),
                ]
            )
            cell_tensors.append(f"{tag}_u")
        row = f"row{dr + 1}"
        nodes.append(helper.make_node("Concat", cell_tensors, [row], axis=3))
        row_tensors.append(row)

    nodes.extend(
        [
            helper.make_node("Concat", row_tensors, ["corr"], axis=2),
            helper.make_node("ReduceMax", ["corr"], ["any_color"], axes=[1], keepdims=1),
            helper.make_node("Add", ["any_color", "center"], ["occupied"]),
            helper.make_node("Greater", ["occupied", half], ["occ_b"]),
            helper.make_node("Not", ["occ_b"], ["black_b"]),
            helper.make_node("Cast", ["black_b"], ["black"], to=TensorProto.FLOAT),
        ]
    )

    def slice_ch(ch0: int, ch1: int, name: str) -> str:
        st = _i64(inits, [0, ch0, 0, 0], f"{name}_st")
        en = _i64(inits, [1, ch1, OH, OW], f"{name}_en")
        nodes.append(helper.make_node("Slice", ["corr", st, en, axes4], [name]))
        return name

    c1 = slice_ch(0, 1, "o1")
    c2 = slice_ch(1, 2, "o2")
    c3 = slice_ch(2, 3, "o3")
    c4 = slice_ch(3, 4, "o4")
    c6 = slice_ch(4, 5, "o6")
    c7 = slice_ch(5, 6, "o7")
    c8 = slice_ch(6, 7, "o8")
    c9 = slice_ch(7, 8, "o9")

    nodes.extend(
        [
            helper.make_node("Concat", ["black", c1, c2, c3, c4, "center", c6, c7, c8, c9], ["out3"], axis=1),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=out_pads),
        ]
    )

    graph = helper.make_graph(nodes, "task022_slice", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_gray_shift_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    c1_st = _i64(inits, [0, 1, 0, 0], "c1_st")
    c1_en = _i64(inits, [1, 5, SH, SW], "c1_en")
    c6_st = _i64(inits, [0, 6, 0, 0], "c6_st")
    c6_en = _i64(inits, [1, 10, SH, SW], "c6_en")
    g_st = _i64(inits, [0, 5, 0, 0], "g_st")
    g_en = _i64(inits, [1, 6, SH, SW], "g_en")
    corr_shape = _i64(inits, [1, len(COLORS), OH, OW], "corr_shape")
    half = _f32(inits, [0.5], "half")
    weight = _f32(inits, _gray_shift_weights(), "gw")
    center = _f32(
        inits,
        np.array([[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]], dtype=np.float32),
        "center",
    )
    out_pads = [0, 0, 0, 0, 0, 0, H - OH, W - OW]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, c1_st, c1_en, axes4], ["c1_4"]),
            helper.make_node("Slice", [IN_NAME, c6_st, c6_en, axes4], ["c6_9"]),
            helper.make_node("Concat", ["c1_4", "c6_9"], ["colors"], axis=1),
            helper.make_node("Slice", [IN_NAME, g_st, g_en, axes4], ["gray"]),
            helper.make_node("Conv", ["gray", weight], ["gray_offsets"], pads=[1, 1, 1, 1]),
            helper.make_node("Unsqueeze", ["colors"], ["colors5"], axes=[2]),
            helper.make_node("Unsqueeze", ["gray_offsets"], ["gray5"], axes=[1]),
            helper.make_node("Mul", ["colors5", "gray5"], ["anchored"]),
            helper.make_node("ReduceMax", ["anchored"], ["has_offset"], axes=[3, 4], keepdims=0),
            helper.make_node("Reshape", ["has_offset", corr_shape], ["corr"]),
            helper.make_node("ReduceMax", ["corr"], ["any_color"], axes=[1], keepdims=1),
            helper.make_node("Add", ["any_color", "center"], ["occupied"]),
            helper.make_node("Greater", ["occupied", half], ["occ_b"]),
            helper.make_node("Not", ["occ_b"], ["black_b"]),
            helper.make_node("Cast", ["black_b"], ["black"], to=TensorProto.FLOAT),
        ]
    )

    def slice_ch(ch0: int, ch1: int, name: str) -> str:
        st = _i64(inits, [0, ch0, 0, 0], f"{name}_st")
        en = _i64(inits, [1, ch1, OH, OW], f"{name}_en")
        nodes.append(helper.make_node("Slice", ["corr", st, en, axes4], [name]))
        return name

    c1 = slice_ch(0, 1, "o1")
    c2 = slice_ch(1, 2, "o2")
    c3 = slice_ch(2, 3, "o3")
    c4 = slice_ch(3, 4, "o4")
    c6 = slice_ch(4, 5, "o6")
    c7 = slice_ch(5, 6, "o7")
    c8 = slice_ch(6, 7, "o8")
    c9 = slice_ch(7, 8, "o9")

    nodes.extend(
        [
            helper.make_node("Concat", ["black", c1, c2, c3, c4, "center", c6, c7, c8, c9], ["out3"], axis=1),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=out_pads),
        ]
    )

    graph = helper.make_graph(nodes, "task022_gray_shift", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_conv_valid_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    c1_st = _i64(inits, [0, 1, 0, 0], "c1_st")
    c1_en = _i64(inits, [1, 5, SH, SW], "c1_en")
    c6_st = _i64(inits, [0, 6, 0, 0], "c6_st")
    c6_en = _i64(inits, [1, 10, SH, SW], "c6_en")
    g_st = _i64(inits, [0, 5, 1, 1], "g_st")
    g_en = _i64(inits, [1, 6, 10, 10], "g_en")
    corr_shape = _i64(inits, [1, len(COLORS), OH, OW], "corr_shape")
    half = _f32(inits, [0.5], "half")
    weight = _f32(inits, _conv_weights(), "w")
    center = _f32(
        inits,
        np.array([[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]], dtype=np.float32),
        "center",
    )
    out_pads = [0, 0, 0, 0, 0, 0, H - OH, W - OW]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, c1_st, c1_en, axes4], ["c1_4"]),
            helper.make_node("Slice", [IN_NAME, c6_st, c6_en, axes4], ["c6_9"]),
            helper.make_node("Concat", ["c1_4", "c6_9"], ["colors"], axis=1),
            helper.make_node("Slice", [IN_NAME, g_st, g_en, axes4], ["gray"]),
            helper.make_node("Conv", ["colors", weight], ["offset_hits"], group=len(COLORS)),
            helper.make_node("Mul", ["offset_hits", "gray"], ["anchored"]),
            helper.make_node("ReduceMax", ["anchored"], ["has_offset"], axes=[2, 3], keepdims=0),
            helper.make_node("Reshape", ["has_offset", corr_shape], ["corr"]),
            helper.make_node("ReduceMax", ["corr"], ["any_color"], axes=[1], keepdims=1),
            helper.make_node("Add", ["any_color", "center"], ["occupied"]),
            helper.make_node("Greater", ["occupied", half], ["occ_b"]),
            helper.make_node("Not", ["occ_b"], ["black_b"]),
            helper.make_node("Cast", ["black_b"], ["black"], to=TensorProto.FLOAT),
        ]
    )

    def slice_ch(ch0: int, ch1: int, name: str) -> str:
        st = _i64(inits, [0, ch0, 0, 0], f"{name}_st")
        en = _i64(inits, [1, ch1, OH, OW], f"{name}_en")
        nodes.append(helper.make_node("Slice", ["corr", st, en, axes4], [name]))
        return name

    c1 = slice_ch(0, 1, "o1")
    c2 = slice_ch(1, 2, "o2")
    c3 = slice_ch(2, 3, "o3")
    c4 = slice_ch(3, 4, "o4")
    c6 = slice_ch(4, 5, "o6")
    c7 = slice_ch(5, 6, "o7")
    c8 = slice_ch(6, 7, "o8")
    c9 = slice_ch(7, 8, "o9")

    nodes.extend(
        [
            helper.make_node("Concat", ["black", c1, c2, c3, c4, "center", c6, c7, c8, c9], ["out3"], axis=1),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=out_pads),
        ]
    )

    graph = helper.make_graph(nodes, "task022_conv_valid", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_slice_valid_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    c1_st = _i64(inits, [0, 1, 0, 0], "c1_st")
    c1_en = _i64(inits, [1, 5, SH, SW], "c1_en")
    c6_st = _i64(inits, [0, 6, 0, 0], "c6_st")
    c6_en = _i64(inits, [1, 10, SH, SW], "c6_en")
    g_st = _i64(inits, [0, 5, 1, 1], "g_st")
    g_en = _i64(inits, [1, 6, 10, 10], "g_en")
    half = _f32(inits, [0.5], "half")
    center = _f32(
        inits,
        np.array([[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]], dtype=np.float32),
        "center",
    )
    lo_st = _i64(inits, [0, 0, 0, 0], "lo_st")
    lo_en = _i64(inits, [1, 4, OH, OW], "lo_en")
    hi_st = _i64(inits, [0, 4, 0, 0], "hi_st")
    hi_en = _i64(inits, [1, 8, OH, OW], "hi_en")
    out_pads = [0, 0, 0, 0, 0, 0, H - OH, W - OW]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, c1_st, c1_en, axes4], ["c1_4"]),
            helper.make_node("Slice", [IN_NAME, c6_st, c6_en, axes4], ["c6_9"]),
            helper.make_node("Concat", ["c1_4", "c6_9"], ["colors"], axis=1),
            helper.make_node("Slice", [IN_NAME, g_st, g_en, axes4], ["gray"]),
        ]
    )

    row_tensors: list[str] = []
    for dr in (-1, 0, 1):
        cell_tensors: list[str] = []
        for dc in (-1, 0, 1):
            tag = f"v{dr + 1}{dc + 1}"
            cs = _i64(inits, [0, 0, 1 + dr, 1 + dc], f"{tag}_cs")
            ce = _i64(inits, [1, len(COLORS), 10 + dr, 10 + dc], f"{tag}_ce")
            nodes.extend(
                [
                    helper.make_node("Slice", ["colors", cs, ce, axes4], [f"{tag}_c"]),
                    helper.make_node("Mul", [f"{tag}_c", "gray"], [f"{tag}_m"]),
                    helper.make_node("ReduceMax", [f"{tag}_m"], [f"{tag}_v"], axes=[2, 3], keepdims=0),
                    helper.make_node("Unsqueeze", [f"{tag}_v"], [f"{tag}_u"], axes=[2, 3]),
                ]
            )
            cell_tensors.append(f"{tag}_u")
        row = f"vrow{dr + 1}"
        nodes.append(helper.make_node("Concat", cell_tensors, [row], axis=3))
        row_tensors.append(row)

    nodes.extend(
        [
            helper.make_node("Concat", row_tensors, ["corr"], axis=2),
            helper.make_node("ReduceMax", ["corr"], ["any_color"], axes=[1], keepdims=1),
            helper.make_node("Add", ["any_color", "center"], ["occupied"]),
            helper.make_node("Greater", ["occupied", half], ["occ_b"]),
            helper.make_node("Not", ["occ_b"], ["black_b"]),
            helper.make_node("Cast", ["black_b"], ["black"], to=TensorProto.FLOAT),
            helper.make_node("Slice", ["corr", lo_st, lo_en, axes4], ["corr_lo"]),
            helper.make_node("Slice", ["corr", hi_st, hi_en, axes4], ["corr_hi"]),
            helper.make_node("Concat", ["black", "corr_lo", "center", "corr_hi"], ["out3"], axis=1),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=out_pads),
        ]
    )

    graph = helper.make_graph(nodes, "task022_slice_valid", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    if not DATA_PATH.is_file():
        return 0
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            g = np.array(ex["input"], dtype=np.int64)
            exp = np.array(ex["output"], dtype=np.int64)
            ref = solve(g)
            if not np.array_equal(ref, exp):
                raise AssertionError(f"reference mismatch {split} {idx}:\n{ref}\n{exp}")
            pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(ex["input"])))[:OH, :OW]
            if not np.array_equal(pred, exp):
                bad += 1
                print(f"mismatch {split} {idx}:\n{pred}\nexpected:\n{exp}")
                if bad >= 5:
                    return bad
    return bad


def main() -> None:
    candidates = [
        ("scalar_crop_f16", build_scalar_model(use_half=True, crop_conv=True)),
        ("scalar_crop", build_scalar_model(crop_conv=True)),
        ("scalar_f16", build_scalar_model(use_half=True)),
        ("scalar", build_scalar_model()),
        ("conv", build_model()),
        ("slice", build_slice_model()),
        ("gray_shift", build_gray_shift_model()),
        ("conv_valid", build_conv_valid_model()),
        ("slice_valid", build_slice_valid_model()),
    ]
    best: tuple[str, onnx.ModelProto, dict] | None = None
    for name, model in candidates:
        bad = validate_json(model)
        assert bad == 0, f"{name}: {bad} validation examples failed"
        path = Path(tempfile.gettempdir()) / f"task022_{name}.onnx"
        onnx.save(model, path)
        result = score_file(path)
        print(
            f"{name}: valid={result['valid']} memory={result['memory']} "
            f"params={result['params']} cost={result['cost']} score={result['score']}"
        )
        if result["valid"] and (best is None or int(result["cost"]) < int(best[2]["cost"])):
            best = (name, model, result)

    assert best is not None, "no valid candidate"
    name, model, result = best
    onnx.save(model, BEST_PATH)
    print(f"saved best ({name}): {BEST_PATH}")
    print(f"memory: {result['memory']}")
    print(f"params: {result['params']}")
    print(f"cost: {result['cost']}")
    print(f"score: {result['score']}")


if __name__ == "__main__":
    main()
