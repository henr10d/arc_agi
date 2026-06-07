"""Minimal ONNX for ARC task001 using Kaggle one-hot I/O."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))
BEST_PATH = OUT_DIR / "task001.onnx"
DATA_PATH = ROOT / "data" / "task001.json"

C = 10
H = W = 30
CORE = 3
OUT = CORE * CORE
PAD = H - OUT
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
OPSET_BOOL_PAD = 13
IR_VERSION = 10


def solve(x: np.ndarray) -> np.ndarray:
    """Reference: paste full input into each 3x3 block where input[i, j] != 0."""
    x = np.asarray(x)
    core = x[:CORE, :CORE]
    out = np.zeros((OUT, OUT), dtype=core.dtype)
    for i in range(CORE):
        for j in range(CORE):
            if core[i, j] != 0:
                out[i * CORE : (i + 1) * CORE, j * CORE : (j + 1) * CORE] = core
    return out


def _i64(inits: List[onnx.TensorProto], vals: List[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))
    return name


def _f32(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name))
    return name


def _i64_arr(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.int64), name=name))
    return name


def _bool_arr(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=bool), name=name))
    return name


def _grid_to_onehot(grid: np.ndarray | List[List[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            color = int(arr[r, c])
            if 0 <= color < C:
                out[0, color, r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _expected_onehot(grid: np.ndarray | List[List[int]]) -> np.ndarray:
    # Matches Kaggle's convert_to_numpy(): cells outside the task's output grid
    # stay all-zero, while cells inside the output grid are one-hot.
    return _grid_to_onehot(np.asarray(grid, dtype=np.int64))


def build_onnx_model() -> onnx.ModelProto:
    """
    One-hot version of the 3x3 stencil tiling graph with lower official memory.

    The core logic runs as bool tensors, then only the compact 9x9 result is cast
    to float before the final Pad. NeuroGolf excludes graph output from memory
    scoring, so this avoids materializing internal [1, 10, 30, 30] tensors and
    reduces the 9x9 intermediates from float32 to bool where possible.
    """
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    ax = _i64(inits, [0, 1, 2, 3], "a")
    st = _i64(inits, [0, 0, 0, 0], "s")
    en = _i64(inits, [1, C, CORE, CORE], "e")
    e0 = _i64(inits, [1, 1, CORE, CORE], "f")
    zero = _f32(inits, [0.0], "z")
    rep4 = _i64(inits, [1, 1, CORE, CORE], "r")
    r6m = _i64(inits, [1, 1, CORE, 1, CORE, 1], "b")
    rep6 = _i64(inits, [1, 1, 1, CORE, 1, CORE], "c")
    s9m = _i64(inits, [1, 1, OUT, OUT], "d")
    st1 = _i64(inits, [0, 1, 0, 0], "t")
    e9 = _i64(inits, [1, C, OUT, OUT], "u")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st, en, ax], ["core"]),
            helper.make_node("Greater", ["core", zero], ["coreb"]),
            helper.make_node("Slice", ["coreb", st, e0, ax], ["zerob"]),
            helper.make_node("Not", ["zerob"], ["mask"]),
            helper.make_node("Tile", ["coreb", rep4], ["body"]),
            helper.make_node("Reshape", ["mask", r6m], ["m6"]),
            helper.make_node("Tile", ["m6", rep6], ["mt"]),
            helper.make_node("Reshape", ["mt", s9m], ["m9"]),
            helper.make_node("Slice", ["body", st1, e9, ax], ["fgbody"]),
            helper.make_node("And", ["fgbody", "m9"], ["fg9"]),
            helper.make_node("Slice", ["body", st, s9m, ax], ["body0"]),
            helper.make_node("Not", ["m9"], ["notm"]),
            helper.make_node("Or", ["body0", "notm"], ["out0"]),
            helper.make_node("Concat", ["out0", "fg9"], ["out9b"], axis=1),
            helper.make_node("Cast", ["out9b"], ["out9"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out9"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]),
        ]
    )

    graph = helper.make_graph(nodes, "g", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_competition_safe_model() -> onnx.ModelProto:
    """
    Opset-10 float-output model for the stricter Kaggle-style checker.

    The checker compares ``output > 0`` against a full one-hot target, so
    background cells must explicitly have channel 0 positive. The low-memory
    argmax-only shortcut is not valid for submission.
    """
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    ax = _i64(inits, [0, 1, 2, 3], "a")
    st = _i64(inits, [0, 0, 0, 0], "s")
    en = _i64(inits, [1, C, CORE, CORE], "e")
    e0 = _i64(inits, [1, 1, CORE, CORE], "f")
    efg = _i64(inits, [1, C, H, W], "g")
    ec0 = _i64(inits, [1, 1, H, W], "h")
    one = _f32(inits, [1.0], "o")
    zero = _f32(inits, [0.0], "z")
    rep4 = _i64(inits, [1, 1, CORE, CORE], "r")
    r6m = _i64(inits, [1, 1, CORE, 1, CORE, 1], "b")
    rep6 = _i64(inits, [1, 1, 1, CORE, 1, CORE], "c")
    s9m = _i64(inits, [1, 1, OUT, OUT], "d")
    row_mask = np.pad(
        np.ones((1, 1, OUT, 1), dtype=np.float32),
        ((0, 0), (0, 0), (0, PAD), (0, 0)),
    )
    col_mask = np.pad(
        np.ones((1, 1, 1, OUT), dtype=np.float32),
        ((0, 0), (0, 0), (0, 0), (0, PAD)),
    )
    _f32(inits, row_mask, "rm")
    _f32(inits, col_mask, "cm")
    st1 = _i64(inits, [0, 1, 0, 0], "t")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st, en, ax], ["core"]),
            helper.make_node("Slice", ["core", st, e0, ax], ["ch0"]),
            helper.make_node("Sub", [one, "ch0"], ["mask"]),
            helper.make_node("Tile", ["core", rep4], ["body"]),
            helper.make_node("Reshape", ["mask", r6m], ["m6"]),
            helper.make_node("Tile", ["m6", rep6], ["mt"]),
            helper.make_node("Reshape", ["mt", s9m], ["m9"]),
            helper.make_node("Mul", ["body", "m9"], ["o9"]),
            helper.make_node("Pad", ["o9"], ["pad"], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]),
            helper.make_node("Slice", ["pad", st, efg, ax], ["fg"]),
            helper.make_node("ReduceMax", ["fg"], ["mx"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["mx", zero], ["fa"]),
            helper.make_node("Not", ["fa"], ["nf"]),
            helper.make_node("Mul", ["rm", "cm"], ["rg"]),
            helper.make_node("Greater", ["rg", zero], ["ir"]),
            helper.make_node("And", ["nf", "ir"], ["bg"]),
            helper.make_node("Slice", ["pad", st, ec0, ax], ["c0"]),
            helper.make_node("Where", ["bg", one, "c0"], ["c0f"]),
            helper.make_node("Slice", ["pad", st1, efg, ax], ["tl"]),
            helper.make_node("Concat", ["c0f", "tl"], [OUT_NAME], axis=1),
        ]
    )

    graph = helper.make_graph(nodes, "g_competition_safe", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_bool_output_model() -> onnx.ModelProto:
    """Current graph with bool output and no internal float cast."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.BOOL, SHAPE)

    ax = _i64(inits, [0, 1, 2, 3], "a")
    st = _i64(inits, [0, 0, 0, 0], "s")
    en = _i64(inits, [1, C, CORE, CORE], "e")
    e0 = _i64(inits, [1, 1, CORE, CORE], "f")
    zero = _f32(inits, [0.0], "z")
    rep4 = _i64(inits, [1, 1, CORE, CORE], "r")
    r6m = _i64(inits, [1, 1, CORE, 1, CORE, 1], "b")
    rep6 = _i64(inits, [1, 1, 1, CORE, 1, CORE], "c")
    s9m = _i64(inits, [1, 1, OUT, OUT], "d")
    st1 = _i64(inits, [0, 1, 0, 0], "t")
    e9 = _i64(inits, [1, C, OUT, OUT], "u")
    pads = _i64(inits, [0, 0, 0, 0, 0, 0, PAD, PAD], "p")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st, en, ax], ["core"]),
            helper.make_node("Greater", ["core", zero], ["coreb"]),
            helper.make_node("Slice", ["coreb", st, e0, ax], ["zerob"]),
            helper.make_node("Not", ["zerob"], ["mask"]),
            helper.make_node("Tile", ["coreb", rep4], ["body"]),
            helper.make_node("Reshape", ["mask", r6m], ["m6"]),
            helper.make_node("Tile", ["m6", rep6], ["mt"]),
            helper.make_node("Reshape", ["mt", s9m], ["m9"]),
            helper.make_node("Slice", ["body", st1, e9, ax], ["fgbody"]),
            helper.make_node("And", ["fgbody", "m9"], ["fg9"]),
            helper.make_node("Slice", ["body", st, s9m, ax], ["body0"]),
            helper.make_node("Not", ["m9"], ["notm"]),
            helper.make_node("Or", ["body0", "notm"], ["out0"]),
            helper.make_node("Concat", ["out0", "fg9"], ["out9b"], axis=1),
            helper.make_node("Pad", ["out9b", pads], [OUT_NAME]),
        ]
    )

    graph = helper.make_graph(nodes, "g_bool", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET_BOOL_PAD)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_channel_selector_model() -> onnx.ModelProto:
    """
    Avoid channel-0/tail slicing and Concat.

    For each compact 9x9 position, active blocks use the tiled input body;
    inactive blocks use a broadcastable channel-0 selector. The final Pad is
    still required, so this cannot hit score 19, but it removes several large
    intermediates from the current graph.
    """
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.BOOL, SHAPE)

    ax = _i64(inits, [0, 1, 2, 3], "a")
    st = _i64(inits, [0, 0, 0, 0], "s")
    en = _i64(inits, [1, C, CORE, CORE], "e")
    e0 = _i64(inits, [1, 1, CORE, CORE], "f")
    zero = _f32(inits, [0.0], "z")
    rep4 = _i64(inits, [1, 1, CORE, CORE], "r")
    r6m = _i64(inits, [1, 1, CORE, 1, CORE, 1], "b")
    rep6 = _i64(inits, [1, 1, 1, CORE, 1, CORE], "c")
    s9m = _i64(inits, [1, 1, OUT, OUT], "d")
    ch0 = _bool_arr(inits, np.arange(C).reshape(1, C, 1, 1) == 0, "q")
    pads = _i64(inits, [0, 0, 0, 0, 0, 0, PAD, PAD], "p")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st, en, ax], ["core"]),
            helper.make_node("Greater", ["core", zero], ["coreb"]),
            helper.make_node("Slice", ["coreb", st, e0, ax], ["zerob"]),
            helper.make_node("Not", ["zerob"], ["mask"]),
            helper.make_node("Tile", ["coreb", rep4], ["body"]),
            helper.make_node("Reshape", ["mask", r6m], ["m6"]),
            helper.make_node("Tile", ["m6", rep6], ["mt"]),
            helper.make_node("Reshape", ["mt", s9m], ["m9"]),
            helper.make_node("And", ["body", "m9"], ["fg"]),
            helper.make_node("Not", ["m9"], ["notm"]),
            helper.make_node("And", [ch0, "notm"], ["bg"]),
            helper.make_node("Or", ["fg", "bg"], ["out9b"]),
            helper.make_node("Pad", ["out9b", pads], [OUT_NAME]),
        ]
    )

    graph = helper.make_graph(nodes, "g_selector", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET_BOOL_PAD)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_masked_body_model() -> onnx.ModelProto:
    """
    Best local variant found: all-zero logits are acceptable for color 0.

    That lets inactive blocks and the padded area remain all false instead of
    explicitly setting channel 0. The only compact 9x9 tensors left are the
    tiled body, the block mask, and their final AND before bool Pad.
    """
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.BOOL, SHAPE)

    ax = _i64(inits, [0, 1, 2, 3], "a")
    st = _i64(inits, [0, 0, 0, 0], "s")
    en = _i64(inits, [1, C, CORE, CORE], "e")
    e0 = _i64(inits, [1, 1, CORE, CORE], "f")
    zero = _f32(inits, [0.0], "z")
    rep4 = _i64(inits, [1, 1, CORE, CORE], "r")
    r6m = _i64(inits, [1, 1, CORE, 1, CORE, 1], "b")
    rep6 = _i64(inits, [1, 1, 1, CORE, 1, CORE], "c")
    s9m = _i64(inits, [1, 1, OUT, OUT], "d")
    pads = _i64(inits, [0, 0, 0, 0, 0, 0, PAD, PAD], "p")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st, en, ax], ["core"]),
            helper.make_node("Greater", ["core", zero], ["coreb"]),
            helper.make_node("Slice", ["coreb", st, e0, ax], ["zerob"]),
            helper.make_node("Not", ["zerob"], ["mask"]),
            helper.make_node("Tile", ["coreb", rep4], ["body"]),
            helper.make_node("Reshape", ["mask", r6m], ["m6"]),
            helper.make_node("Tile", ["m6", rep6], ["mt"]),
            helper.make_node("Reshape", ["mt", s9m], ["m9"]),
            helper.make_node("And", ["body", "m9"], ["out9b"]),
            helper.make_node("Pad", ["out9b", pads], [OUT_NAME]),
        ]
    )

    graph = helper.make_graph(nodes, "g_masked_body", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET_BOOL_PAD)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_broadcast_channel_model() -> onnx.ModelProto:
    """
    Compact color-ID experiment.

    This proves the ArgMax/color-ID route is not enough: the [1,1,9,9] int64
    ID tensor alone is already 648 bytes before OneHot expands it again.
    """
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    ax = _i64(inits, [0, 1, 2, 3], "a")
    st = _i64(inits, [0, 0, 0, 0], "s")
    en = _i64(inits, [1, C, CORE, CORE], "e")
    rep4 = _i64(inits, [1, 1, CORE, CORE], "r")
    r6m = _i64(inits, [1, 1, CORE, 1, CORE, 1], "b")
    rep6 = _i64(inits, [1, 1, 1, CORE, 1, CORE], "c")
    s9m = _i64(inits, [1, 1, OUT, OUT], "d")
    zero = _i64_arr(inits, np.asarray([0], dtype=np.int64), "z")
    depth = _i64_arr(inits, np.asarray(C, dtype=np.int64), "h")
    vals = _f32(inits, [0.0, 1.0], "v")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st, en, ax], ["core"]),
            helper.make_node("ArgMax", ["core"], ["ids"], axis=1, keepdims=1),
            helper.make_node("Greater", ["ids", zero], ["active"]),
            helper.make_node("Tile", ["ids", rep4], ["ids9"]),
            helper.make_node("Reshape", ["active", r6m], ["m6"]),
            helper.make_node("Tile", ["m6", rep6], ["mt"]),
            helper.make_node("Reshape", ["mt", s9m], ["m9"]),
            helper.make_node("Where", ["m9", "ids9", zero], ["yids"]),
            helper.make_node("OneHot", ["yids", depth, vals], ["oh5"], axis=-1),
            helper.make_node("Squeeze", ["oh5"], ["oh4nhwc"], axes=[1]),
            helper.make_node("Transpose", ["oh4nhwc"], ["out9"], perm=[0, 3, 1, 2]),
            helper.make_node("Pad", ["out9"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]),
        ]
    )

    graph = helper.make_graph(nodes, "g_ids", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_final_broadcast_model() -> onnx.ModelProto:
    """
    Final ConvTranspose expansion experiment.

    The final node directly produces [1,10,30,30], avoiding Pad after 9x9.
    It still needs a 90-channel pairwise tensor and selector weights, so the
    memory/params are much worse than the target.
    """
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    ax = _i64(inits, [0, 1, 2, 3], "a")
    st = _i64(inits, [0, 0, 0, 0], "s")
    en = _i64(inits, [1, C, CORE, CORE], "e")
    e0 = _i64(inits, [1, 1, CORE, CORE], "f")
    zero = _f32(inits, [0.0], "z")
    r_core = _i64(inits, [1, C, 1, 1, CORE, CORE], "rc")
    r_mask = _i64(inits, [1, 1, CORE, CORE, 1, 1], "rm")
    r_pair = _i64(inits, [1, C * CORE * CORE, CORE, CORE], "rp")
    ch0 = _f32(inits, (np.arange(C).reshape(1, C, 1, 1, 1, 1) == 0).astype(np.float32), "q")

    weight = np.zeros((C * CORE * CORE, 1, CORE, CORE), dtype=np.float32)
    for color in range(C):
        for rr in range(CORE):
            for cc in range(CORE):
                weight[color * OUT + rr * CORE + cc, 0, rr, cc] = 1.0
    wt = _f32(inits, weight, "w")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st, en, ax], ["core"]),
            helper.make_node("Greater", ["core", zero], ["coreb"]),
            helper.make_node("Slice", ["coreb", st, e0, ax], ["zerob"]),
            helper.make_node("Not", ["zerob"], ["mask"]),
            helper.make_node("Reshape", ["core", r_core], ["core6"]),
            helper.make_node("Reshape", ["mask", r_mask], ["mask6"]),
            helper.make_node("Where", ["mask6", "core6", ch0], ["pair6"]),
            helper.make_node("Transpose", ["pair6"], ["pair6t"], perm=[0, 1, 4, 5, 2, 3]),
            helper.make_node("Reshape", ["pair6t", r_pair], ["pair90"]),
            helper.make_node(
                "ConvTranspose",
                ["pair90", wt],
                [OUT_NAME],
                group=C,
                strides=[CORE, CORE],
                kernel_shape=[CORE, CORE],
                output_shape=[1, C, H, W],
            ),
        ]
    )

    graph = helper.make_graph(nodes, "g_final_convtranspose", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def model_stats(model: onnx.ModelProto, path: Path | None = None) -> Dict[str, int]:
    params = sum(int(np.prod(list(t.dims))) for t in model.graph.initializer)
    return {
        "bytes": path.stat().st_size if path and path.is_file() else len(model.SerializeToString()),
        "nodes": len(model.graph.node),
        "inits": len(model.graph.initializer),
        "params": params,
        "opset": model.opset_import[0].version,
        "ir": model.ir_version,
    }


def save_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    model = build_onnx_model()
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return model


def _dtype_size(elem_type: int) -> tuple[str, int]:
    try:
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(elem_type)
        return str(np.dtype(np_dtype)), int(np.dtype(np_dtype).itemsize)
    except Exception:
        if elem_type == TensorProto.BOOL:
            return "bool", 1
        return f"type{elem_type}", 0


def inferred_internal_tensors(model: onnx.ModelProto) -> tuple[int | None, list[dict[str, Any]], str | None]:
    """Return shape-inference internal memory and per-tensor details."""
    try:
        graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    except Exception as exc:
        return None, [], str(exc)

    rows: list[dict[str, Any]] = []
    total = 0
    tensor_map = {t.name: t for t in list(graph.input) + list(graph.value_info) + list(graph.output)}
    io_names = {t.name for t in list(graph.input) + list(graph.output)}
    for node in graph.node:
        for name in node.output:
            if not name or name in io_names:
                continue
            item = tensor_map.get(name)
            if item is None or not item.type.HasField("tensor_type"):
                return None, rows, f"missing inferred tensor info for {name}"
            tensor_type = item.type.tensor_type
            if not tensor_type.HasField("shape"):
                return None, rows, f"missing inferred shape for {name}"
            shape: list[int] = []
            n = 1
            for dim in tensor_type.shape.dim:
                if dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0:
                    return None, rows, f"dynamic/invalid inferred shape for {name}"
                shape.append(int(dim.dim_value))
                n *= int(dim.dim_value)
            dtype, itemsize = _dtype_size(tensor_type.elem_type)
            mem = n * itemsize
            rows.append({"name": name, "shape": shape, "dtype": dtype, "bytes": mem})
            total += mem
    rows.sort(key=lambda row: int(row["bytes"]), reverse=True)
    return total, rows, None


def official_score(path: Path) -> dict[str, Any]:
    from score_model import score_file

    return score_file(path)


def _score(cost: int | float) -> float:
    return max(1.0, 25.0 - math.log(max(1.0, float(cost))))


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def _pad_grid(grid: np.ndarray) -> np.ndarray:
    out = np.zeros((H, W), dtype=np.int64)
    out[: grid.shape[0], : grid.shape[1]] = grid
    return out


def _is_competition_compatible(model: onnx.ModelProto) -> tuple[bool, str]:
    if model.opset_import[0].version != OPSET:
        return False, f"opset {model.opset_import[0].version} != {OPSET}"
    out_type = model.graph.output[0].type.tensor_type.elem_type
    if out_type != TensorProto.FLOAT:
        return False, f"output type {out_type} is not FLOAT"
    return True, "PASS"


def _strict_onehot_matches(pred: np.ndarray, expected: np.ndarray) -> bool:
    return pred.shape == expected.shape and np.array_equal(pred > 0, expected > 0)


def validate_model(
    model: onnx.ModelProto,
    path: Path | None = None,
    *,
    random_cases: int = 200,
    strict_onehot: bool = True,
) -> tuple[bool, str]:
    try:
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    except Exception as exc:
        return False, f"ORT load failed: {exc}"

    rng = np.random.default_rng(0)
    for _ in range(random_cases):
        x = rng.integers(0, C, size=(CORE, CORE), dtype=np.int64)
        ref = solve(x)
        pred = sess.run([OUT_NAME], {IN_NAME: _grid_to_onehot(x)})[0]
        if strict_onehot:
            if not _strict_onehot_matches(pred, _expected_onehot(ref)):
                return False, f"random strict one-hot mismatch for input {x.tolist()}"
        else:
            y = _onehot_to_grid(pred[0])[:OUT, :OUT]
            if not np.array_equal(ref, y):
                return False, f"random mismatch for input {x.tolist()}"

    if DATA_PATH.is_file():
        with DATA_PATH.open(encoding="utf-8") as fh:
            data = json.load(fh)
        for split in ("train", "test", "arc-gen"):
            for idx, ex in enumerate(data[split]):
                x = np.asarray(ex["input"], dtype=np.int64)
                exp = np.asarray(ex["output"], dtype=np.int64)
                pred = sess.run([OUT_NAME], {IN_NAME: _grid_to_onehot(x)})[0]
                if strict_onehot:
                    if not _strict_onehot_matches(pred, _expected_onehot(exp)):
                        return False, f"{split}#{idx} strict one-hot mismatch"
                else:
                    exp = _pad_grid(exp)
                    grid = _onehot_to_grid(pred[0])
                    if not np.array_equal(exp, grid):
                        return False, f"{split}#{idx} mismatch"

    if path is not None:
        try:
            from train_arc import validate_task_onnx

            ok, _ = validate_task_onnx("task001", path, save_viz=False)
            if not ok:
                return False, "train_arc validation failed"
        except Exception as exc:
            return False, f"train_arc validation error: {exc}"

    return True, "PASS"


def variant_builders() -> dict[str, Callable[[], onnx.ModelProto]]:
    return {
        "competition_safe": build_competition_safe_model,
        "current": build_onnx_model,
        "bool_output": build_bool_output_model,
        "broadcast_channel": build_broadcast_channel_model,
        "final_broadcast": build_final_broadcast_model,
        "channel_selector": build_channel_selector_model,
        "masked_body": build_masked_body_model,
    }


def run_experiments() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for name, builder in variant_builders().items():
        path = OUT_DIR / f"task001_variant_{name}.onnx"
        result: dict[str, Any] = {"name": name, "path": path, "valid": False}
        try:
            model = builder()
            path.parent.mkdir(parents=True, exist_ok=True)
            onnx.save(model, str(path))
            result.update(model_stats(model, path))
            inferred_mem, tensors, infer_error = inferred_internal_tensors(model)
            result["inferred_memory"] = inferred_mem
            result["top_tensors"] = tensors[:20]
            result["infer_error"] = infer_error
            ok, message = validate_model(model, path)
            result["valid"] = ok
            result["validation"] = message
            scored = official_score(path)
            result["official_memory"] = scored.get("memory")
            result["official_params"] = scored.get("params")
            result["official_cost"] = scored.get("cost")
            result["official_score"] = scored.get("score")
            result["score_error"] = scored.get("error")
        except Exception as exc:
            result["validation"] = f"build/check failed: {exc}"
        results.append(result)

    valid_results = [
        row
        for row in results
        if row.get("valid")
        and row.get("official_cost") is not None
        and row.get("official_score") is not None
    ]
    if valid_results:
        best = min(valid_results, key=lambda row: int(row["official_cost"]))
        onnx.save(onnx.load(str(best["path"])), str(BEST_PATH))
    return results


def print_experiment_report(results: list[dict[str, Any]]) -> None:
    print("\nVariant comparison")
    print(
        f"{'variant':<19} {'pass':<5} {'bytes':>6} {'nodes':>5} {'params':>6} "
        f"{'infer_mem':>9} {'official_mem':>12} {'cost':>6} {'score':>9}"
    )
    for row in results:
        score = row.get("official_score")
        print(
            f"{row['name']:<19} {str(row.get('valid')):<5} "
            f"{str(row.get('bytes', '-')):>6} {str(row.get('nodes', '-')):>5} "
            f"{str(row.get('official_params', row.get('params', '-'))):>6} "
            f"{str(row.get('inferred_memory', '-')):>9} "
            f"{str(row.get('official_memory', '-')):>12} "
            f"{str(row.get('official_cost', '-')):>6} "
            f"{score:.6f}" if isinstance(score, float) else
            f"{row['name']:<19} {str(row.get('valid')):<5} "
            f"{str(row.get('bytes', '-')):>6} {str(row.get('nodes', '-')):>5} "
            f"{str(row.get('official_params', row.get('params', '-'))):>6} "
            f"{str(row.get('inferred_memory', '-')):>9} "
            f"{str(row.get('official_memory', '-')):>12} "
            f"{str(row.get('official_cost', '-')):>6} {'INVALID':>9}"
        )
        if not row.get("valid") or row.get("score_error"):
            print(f"  note: {row.get('validation')} {row.get('score_error') or ''}".rstrip())

    for row in results:
        print(f"\nTop internal tensors: {row['name']}")
        tensors = row.get("top_tensors") or []
        if not tensors:
            print(f"  {row.get('infer_error') or 'no inferred tensor details'}")
            continue
        for tensor in tensors[:20]:
            print(
                f"  {tensor['bytes']:>5}  {tensor['dtype']:<7} "
                f"{str(tensor['shape']):<22} {tensor['name']}"
            )

    target_cost = int(math.floor(math.exp(6)))
    best = min(
        (row for row in results if row.get("valid") and row.get("official_cost") is not None),
        key=lambda row: int(row["official_cost"]),
        default=None,
    )
    print(f"\nScore 19 threshold: memory+params <= floor(exp(6)) = {target_cost}")
    if best is not None:
        print(
            f"Best Kaggle-utility-valid variant here: {best['name']} "
            f"cost={best.get('official_cost')} score={best.get('official_score'):.6f}"
        )
    print(
        "Loop/Scan was not saved as a candidate: train_arc.py bans LOOP/SCAN, "
        "and score_model.py rejects graph attributes, so a control-flow model is "
        "invalid under the local official-style scorer."
    )


def test() -> None:
    model = save_model()
    stats = model_stats(model, BEST_PATH)
    print(
        f"bytes={stats['bytes']} nodes={stats['nodes']} inits={stats['inits']} "
        f"params={stats['params']} opset={stats['opset']} ir={stats['ir']}"
    )

    rng = np.random.default_rng(0)
    for _ in range(200):
        x = rng.integers(0, C, size=(CORE, CORE), dtype=np.int64)
        y = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(x))[0])[:OUT, :OUT]
        ref = solve(x)
        if not np.array_equal(ref, y):
            raise SystemExit(f"random mismatch for input:\n{x}\nref:\n{ref}\nout:\n{y}")
    print("random 3x3 grids (0..9): PASS")

    if DATA_PATH.is_file():
        with DATA_PATH.open(encoding="utf-8") as fh:
            data = json.load(fh)
        for split in ("train", "test", "arc-gen"):
            bad = 0
            for ex in data[split]:
                x = np.asarray(ex["input"], dtype=np.int64)
                exp = _pad_grid(np.asarray(ex["output"], dtype=np.int64))
                pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(x))[0])
                if not np.array_equal(exp, pred):
                    bad += 1
            print(f"task001.json {split}: {'PASS' if bad == 0 else f'FAIL ({bad} wrong)'}")

    from train_arc import validate_task_onnx

    ok, _ = validate_task_onnx("task001", BEST_PATH, save_viz=False)
    print(f"train_arc validation: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


def main() -> None:
    results = run_experiments()
    print_experiment_report(results)
    print(f"\nsaved best passing variant to {BEST_PATH}")


if __name__ == "__main__":
    main()
