"""Build ONNX for ARC task232 using rightward alternating row stripes.

Task rule: each non-black input pixel is an isolated seed.  On that seed's
row, copy the seed color at the seed column and every other cell to the right;
fill the alternating intervening cells with gray (color 5).  Rows act
independently, the output keeps the input grid size, and padded cells outside
the task grid remain empty.

The best graph below scans the 14-column task envelope from left to right with
small bool state tensors, then casts only the compact 14x14 result to float
before padding to the required 30x30 NeuroGolf interface.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

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

TASK_ID = "task232"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task232.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
N = 14
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def solve(grid: list[list[int]]) -> list[list[int]]:
    """Reference implementation used to validate the inferred ARC rule."""
    out = [[0 for _ in row] for row in grid]
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if color == 0:
                continue
            for j in range(c, len(row)):
                out[r][j] = color if (j - c) % 2 == 0 else 5
    return out


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _make_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
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


def build_scan_model(*, split_input: bool = True) -> onnx.ModelProto:
    """Column scan with bool propagation state and no large learned weights."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    axes = _i64(inits, [0, 1, 2, 3], "axes")
    core_st = _i64(inits, [0, 0, 0, 0], "core_st")
    fg_st = _i64(inits, [0, 1, 0, 0], "fg_st")
    fg_en = _i64(inits, [1, C, N, N], "fg_en")
    bg_en = _i64(inits, [1, 1, N, N], "bg_en")
    zero = _f32(inits, [0.0], "zero")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, core_st, bg_en, axes], ["bg0f"]),
            helper.make_node("Cast", ["bg0f"], ["bg0"], to=TensorProto.BOOL),
            helper.make_node("Slice", [IN_NAME, fg_st, fg_en, axes], ["fgf"]),
            helper.make_node("ReduceSum", ["fgf"], ["fg_any"], axes=[1], keepdims=1),
            helper.make_node("Add", ["bg0f", "fg_any"], ["inside_sum"]),
            helper.make_node("Greater", ["inside_sum", zero], ["inside"]),
        ]
    )

    if split_input:
        seeds_f = [f"seedf_{i}" for i in range(N)]
        nodes.append(helper.make_node("Split", ["fgf"], seeds_f, axis=3, split=[1] * N))
    else:
        col_axis = _i64(inits, [3], "col_axis")
        starts = [_i64(inits, [i], f"col_s{i}") for i in range(N)]
        ends = [_i64(inits, [i + 1], f"col_e{i}") for i in range(N)]
        seeds_f = []
        for i in range(N):
            name = f"seedf_{i}"
            nodes.append(helper.make_node("Slice", ["fgf", starts[i], ends[i], col_axis], [name]))
            seeds_f.append(name)

    prop_prev: str | None = None
    active_prev: str | None = None
    phase_prev: str | None = None
    color_cols: list[str] = []
    gray_cols: list[str] = []
    active_cols: list[str] = []

    for i, seed_f in enumerate(seeds_f):
        seed = f"seed_{i}"
        seed_sum = f"seed_sum_{i}"
        seed_any = f"seed_any_{i}"
        prop = f"prop_{i}"
        active = f"active_{i}"
        phase = f"phase_{i}"
        color = f"color_{i}"
        not_phase = f"not_phase_{i}"
        gray = f"gray_{i}"
        nodes.extend(
            [
                helper.make_node("Cast", [seed_f], [seed], to=TensorProto.BOOL),
                helper.make_node("ReduceSum", [seed_f], [seed_sum], axes=[1], keepdims=1),
                helper.make_node("Greater", [seed_sum, zero], [seed_any]),
            ]
        )
        if prop_prev is None:
            nodes.extend(
                [
                    helper.make_node("Identity", [seed], [prop]),
                    helper.make_node("Identity", [seed_any], [active]),
                    helper.make_node("Identity", [seed_any], [phase]),
                ]
            )
        else:
            toggled = f"toggled_{i}"
            nodes.extend(
                [
                    helper.make_node("Or", [prop_prev, seed], [prop]),
                    helper.make_node("Or", [active_prev, seed_any], [active]),
                    helper.make_node("Not", [phase_prev], [toggled]),
                    helper.make_node("And", [active_prev, toggled], [f"phase_old_{i}"]),
                    helper.make_node("Or", [seed_any, f"phase_old_{i}"], [phase]),
                ]
            )
        nodes.extend(
            [
                helper.make_node("And", [prop, phase], [color]),
                helper.make_node("Not", [phase], [not_phase]),
                helper.make_node("And", [active, not_phase], [gray]),
            ]
        )
        color_cols.append(color)
        gray_cols.append(gray)
        active_cols.append(active)
        prop_prev = prop
        active_prev = active
        phase_prev = phase

    c0 = _i64(inits, [0, 0, 0, 0], "c0")
    c4 = _i64(inits, [1, 4, N, N], "c4")
    c5 = _i64(inits, [0, 5, 0, 0], "c5")
    c9 = _i64(inits, [1, 9, N, N], "c9")

    nodes.extend(
        [
            helper.make_node("Concat", color_cols, ["color14_raw"], axis=3),
            helper.make_node("Concat", gray_cols, ["gray14_raw"], axis=3),
            helper.make_node("Concat", active_cols, ["active14"], axis=3),
            helper.make_node("And", ["color14_raw", "inside"], ["color14"]),
            helper.make_node("And", ["gray14_raw", "inside"], ["gray14"]),
            helper.make_node("Not", ["active14"], ["not_active14"]),
            helper.make_node("And", ["bg0", "not_active14"], ["bg14"]),
            helper.make_node("Slice", ["color14", c0, c4, axes], ["color1_4"]),
            helper.make_node("Slice", ["color14", c5, c9, axes], ["color6_9"]),
            helper.make_node("Concat", ["bg14", "color1_4", "gray14", "color6_9"], ["out14b"], axis=1),
            helper.make_node("Cast", ["out14b"], ["out14"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out14"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - N, W - N]),
        ]
    )
    return _make_model(nodes, inits, f"{TASK_ID}_scan_{'split' if split_input else 'slice'}")


def build_matmul_model() -> onnx.ModelProto:
    """Direct rightward/parity propagation using fixed 14x14 masks."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    axes = _i64(inits, [0, 1, 2, 3], "axes")
    core_st = _i64(inits, [0, 0, 0, 0], "core_st")
    fg_st = _i64(inits, [0, 1, 0, 0], "fg_st")
    fg_en = _i64(inits, [1, C, N, N], "fg_en")
    bg_en = _i64(inits, [1, 1, N, N], "bg_en")
    zero = _f32(inits, [0.0], "zero")

    even = np.zeros((N, N), dtype=np.float32)
    odd = np.zeros((N, N), dtype=np.float32)
    for c in range(N):
        for j in range(c, N):
            if (j - c) % 2 == 0:
                even[c, j] = 1.0
            else:
                odd[c, j] = 1.0
    _f32(inits, even, "even_mask")
    _f32(inits, odd, "odd_mask")

    c0 = _i64(inits, [0, 0, 0, 0], "c0")
    c4 = _i64(inits, [1, 4, N, N], "c4")
    c5 = _i64(inits, [0, 5, 0, 0], "c5")
    c9 = _i64(inits, [1, 9, N, N], "c9")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, core_st, bg_en, axes], ["bg0f"]),
            helper.make_node("Slice", [IN_NAME, fg_st, fg_en, axes], ["fg"]),
            helper.make_node("ReduceSum", ["fg"], ["seed_any"], axes=[1], keepdims=1),
            helper.make_node("Add", ["bg0f", "seed_any"], ["inside_sum"]),
            helper.make_node("Greater", ["inside_sum", zero], ["inside"]),
            helper.make_node("MatMul", ["fg", "even_mask"], ["color14_raw"]),
            helper.make_node("MatMul", ["seed_any", "odd_mask"], ["gray14_raw"]),
            helper.make_node("ReduceSum", ["color14_raw"], ["even_any"], axes=[1], keepdims=1),
            helper.make_node("Add", ["even_any", "gray14_raw"], ["active_f"]),
            helper.make_node("Greater", ["active_f", zero], ["active14"]),
            helper.make_node("Not", ["active14"], ["not_active14"]),
            helper.make_node("Cast", ["bg0f"], ["bg0"], to=TensorProto.BOOL),
            helper.make_node("And", ["bg0", "not_active14"], ["bg14"]),
            helper.make_node("Greater", ["color14_raw", zero], ["color14_unmasked"]),
            helper.make_node("Greater", ["gray14_raw", zero], ["gray14_unmasked"]),
            helper.make_node("And", ["color14_unmasked", "inside"], ["color14"]),
            helper.make_node("And", ["gray14_unmasked", "inside"], ["gray14"]),
            helper.make_node("Slice", ["color14", c0, c4, axes], ["color1_4"]),
            helper.make_node("Slice", ["color14", c5, c9, axes], ["color6_9"]),
            helper.make_node("Concat", ["bg14", "color1_4", "gray14", "color6_9"], ["out14b"], axis=1),
            helper.make_node("Cast", ["out14b"], ["out14"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out14"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - N, W - N]),
        ]
    )
    return _make_model(nodes, inits, f"{TASK_ID}_matmul")


def build_row_broadcast_model() -> onnx.ModelProto:
    """Propagate one-channel parity masks, then broadcast each row's seed color."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    axes = _i64(inits, [0, 1, 2, 3], "axes")
    core_st = _i64(inits, [0, 0, 0, 0], "core_st")
    fg_st = _i64(inits, [0, 1, 0, 0], "fg_st")
    fg_en = _i64(inits, [1, C, N, N], "fg_en")
    bg_en = _i64(inits, [1, 1, N, N], "bg_en")
    reps = _i64(inits, [1, 1, 1, N], "reps")
    zero = _f32(inits, [0.0], "zero")

    even = np.zeros((N, N), dtype=np.float32)
    odd = np.zeros((N, N), dtype=np.float32)
    for c in range(N):
        for j in range(c, N):
            if (j - c) % 2 == 0:
                even[c, j] = 1.0
            else:
                odd[c, j] = 1.0
    _f32(inits, even, "even_mask")
    _f32(inits, odd, "odd_mask")

    c0 = _i64(inits, [0, 0, 0, 0], "c0")
    c4 = _i64(inits, [1, 4, N, N], "c4")
    c5 = _i64(inits, [0, 5, 0, 0], "c5")
    c9 = _i64(inits, [1, 9, N, N], "c9")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, core_st, bg_en, axes], ["bg0f"]),
            helper.make_node("Slice", [IN_NAME, fg_st, fg_en, axes], ["fg"]),
            helper.make_node("ReduceSum", ["fg"], ["seed_any"], axes=[1], keepdims=1),
            helper.make_node("Add", ["bg0f", "seed_any"], ["inside_sum"]),
            helper.make_node("Greater", ["inside_sum", zero], ["inside"]),
            helper.make_node("ReduceSum", ["fg"], ["row_color_sum"], axes=[3], keepdims=1),
            helper.make_node("Greater", ["row_color_sum", zero], ["row_color"]),
            helper.make_node("Tile", ["row_color", reps], ["color_tile"]),
            helper.make_node("MatMul", ["seed_any", "even_mask"], ["even_phase_f"]),
            helper.make_node("MatMul", ["seed_any", "odd_mask"], ["gray14_raw"]),
            helper.make_node("Greater", ["even_phase_f", zero], ["even_phase"]),
            helper.make_node("Greater", ["gray14_raw", zero], ["gray14_unmasked"]),
            helper.make_node("And", ["even_phase", "inside"], ["even_inside"]),
            helper.make_node("Or", ["even_phase", "gray14_unmasked"], ["active14"]),
            helper.make_node("Not", ["active14"], ["not_active14"]),
            helper.make_node("Cast", ["bg0f"], ["bg0"], to=TensorProto.BOOL),
            helper.make_node("And", ["bg0", "not_active14"], ["bg14"]),
            helper.make_node("And", ["color_tile", "even_inside"], ["color14"]),
            helper.make_node("And", ["gray14_unmasked", "inside"], ["gray14"]),
            helper.make_node("Slice", ["color14", c0, c4, axes], ["color1_4"]),
            helper.make_node("Slice", ["color14", c5, c9, axes], ["color6_9"]),
            helper.make_node("Concat", ["bg14", "color1_4", "gray14", "color6_9"], ["out14b"], axis=1),
            helper.make_node("Cast", ["out14b"], ["out14"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out14"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - N, W - N]),
        ]
    )
    return _make_model(nodes, inits, f"{TASK_ID}_row_broadcast")


def build_seed8_row_broadcast_model(*, split_color_blocks: bool = False) -> onnx.ModelProto:
    """Row-broadcast variant specialized to observed seed columns 0..7."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    S = 8
    axes = _i64(inits, [1, 2, 3], "axes")
    core_st = _i64(inits, [0, 0, 0], "core_st")
    bg_en = _i64(inits, [1, N, N], "bg_en")
    zero = _f32(inits, [0.0], "zero")

    even = np.zeros((S, N), dtype=np.float32)
    odd = np.zeros((S, N), dtype=np.float32)
    for c in range(S):
        for j in range(c, N):
            if (j - c) % 2 == 0:
                even[c, j] = 1.0
            else:
                odd[c, j] = 1.0
    _f32(inits, even, "even_mask8")
    _f32(inits, odd, "odd_mask8")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, core_st, bg_en, axes], ["bg0f"]),
            helper.make_node("Cast", ["bg0f"], ["bg0"], to=TensorProto.BOOL),
        ]
    )

    if split_color_blocks:
        fg1_st = _i64(inits, [1, 0, 0], "fg1_st")
        fg1_en = _i64(inits, [5, N, S], "fg1_en")
        fg6_st = _i64(inits, [6, 0, 0], "fg6_st")
        fg6_en = _i64(inits, [10, N, S], "fg6_en")
        reps4 = _i64(inits, [1, 1, 1, N], "reps4")
        nodes.extend(
            [
                helper.make_node("Slice", [IN_NAME, fg1_st, fg1_en, axes], ["fg1_4"]),
                helper.make_node("Slice", [IN_NAME, fg6_st, fg6_en, axes], ["fg6_9"]),
                helper.make_node("ReduceSum", ["fg1_4"], ["seed_any_1"], axes=[1], keepdims=1),
                helper.make_node("ReduceSum", ["fg6_9"], ["seed_any_6"], axes=[1], keepdims=1),
                helper.make_node("Add", ["seed_any_1", "seed_any_6"], ["seed_any"]),
                helper.make_node("ReduceSum", ["fg1_4"], ["row_color_sum1"], axes=[3], keepdims=1),
                helper.make_node("ReduceSum", ["fg6_9"], ["row_color_sum6"], axes=[3], keepdims=1),
                helper.make_node("Greater", ["row_color_sum1", zero], ["row_color1"]),
                helper.make_node("Greater", ["row_color_sum6", zero], ["row_color6"]),
                helper.make_node("Tile", ["row_color1", reps4], ["color_tile1"]),
                helper.make_node("Tile", ["row_color6", reps4], ["color_tile6"]),
            ]
        )
        color_tile_inputs = ("color_tile1", "color_tile6")
    else:
        fg_st = _i64(inits, [1, 0, 0], "fg_st")
        fg_en = _i64(inits, [C, N, S], "fg_en")
        reps = _i64(inits, [1, 1, 1, N], "reps")
        axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
        c0 = _i64(inits, [0, 0, 0, 0], "c0")
        c4 = _i64(inits, [1, 4, N, N], "c4")
        c5 = _i64(inits, [0, 5, 0, 0], "c5")
        c9 = _i64(inits, [1, 9, N, N], "c9")
        nodes.extend(
            [
                helper.make_node("Slice", [IN_NAME, fg_st, fg_en, axes], ["fg"]),
                helper.make_node("ReduceSum", ["fg"], ["seed_any"], axes=[1], keepdims=1),
                helper.make_node("ReduceSum", ["fg"], ["row_color_sum"], axes=[3], keepdims=1),
                helper.make_node("Greater", ["row_color_sum", zero], ["row_color"]),
                helper.make_node("Tile", ["row_color", reps], ["color_tile"]),
                helper.make_node("Slice", ["color_tile", c0, c4, axes4], ["color_tile1"]),
                helper.make_node("Slice", ["color_tile", c5, c9, axes4], ["color_tile6"]),
            ]
        )
        color_tile_inputs = ("color_tile1", "color_tile6")

    nodes.extend(
        [
            helper.make_node("Greater", ["seed_any", zero], ["seed_any_b"]),
            helper.make_node("Pad", ["seed_any_b"], ["seed_any14"], pads=[0, 0, 0, 0, 0, 0, 0, N - S]),
            helper.make_node("Or", ["bg0", "seed_any14"], ["inside"]),
            helper.make_node("MatMul", ["seed_any", "even_mask8"], ["even_phase_f"]),
            helper.make_node("MatMul", ["seed_any", "odd_mask8"], ["gray14_raw"]),
            helper.make_node("Greater", ["even_phase_f", zero], ["even_phase"]),
            helper.make_node("Greater", ["gray14_raw", zero], ["gray14_unmasked"]),
            helper.make_node("And", ["even_phase", "inside"], ["even_inside"]),
            helper.make_node("Or", ["even_phase", "gray14_unmasked"], ["active14"]),
            helper.make_node("Not", ["active14"], ["not_active14"]),
            helper.make_node("And", ["bg0", "not_active14"], ["bg14"]),
            helper.make_node("And", [color_tile_inputs[0], "even_inside"], ["color1_4"]),
            helper.make_node("And", [color_tile_inputs[1], "even_inside"], ["color6_9"]),
            helper.make_node("And", ["gray14_unmasked", "inside"], ["gray14"]),
            helper.make_node("Concat", ["bg14", "color1_4", "gray14", "color6_9"], ["out14b"], axis=1),
            helper.make_node("Cast", ["out14b"], ["out14"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out14"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - N, W - N]),
        ]
    )
    suffix = "blocks" if split_color_blocks else "seed8"
    return _make_model(nodes, inits, f"{TASK_ID}_{suffix}")


def build_seed8_column_scan_model() -> onnx.ModelProto:
    """Scan 14 one-column stripes, carrying seed color and parity state."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    S = 8
    axes = _i64(inits, [0, 1, 2, 3], "axes")
    core_st = _i64(inits, [0, 0, 0, 0], "core_st")
    bg_en = _i64(inits, [1, 1, N, N], "bg_en")
    fg1_st = _i64(inits, [0, 1, 0, 0], "fg1_st")
    fg1_en = _i64(inits, [1, 5, N, S], "fg1_en")
    fg6_st = _i64(inits, [0, 6, 0, 0], "fg6_st")
    fg6_en = _i64(inits, [1, 10, N, S], "fg6_en")
    zero = _f32(inits, [0.0], "zero")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, core_st, bg_en, axes], ["bg0f"]),
            helper.make_node("Cast", ["bg0f"], ["bg0"], to=TensorProto.BOOL),
            helper.make_node("Slice", [IN_NAME, fg1_st, fg1_en, axes], ["fg1_4"]),
            helper.make_node("Slice", [IN_NAME, fg6_st, fg6_en, axes], ["fg6_9"]),
        ]
    )

    bg_cols = [f"bgc_{i}" for i in range(N)]
    seed1_cols_f = [f"seed1f_{i}" for i in range(S)]
    seed6_cols_f = [f"seed6f_{i}" for i in range(S)]
    nodes.extend(
        [
            helper.make_node("Split", ["bg0"], bg_cols, axis=3, split=[1] * N),
            helper.make_node("Split", ["fg1_4"], seed1_cols_f, axis=3, split=[1] * S),
            helper.make_node("Split", ["fg6_9"], seed6_cols_f, axis=3, split=[1] * S),
        ]
    )

    color1_prev: str | None = None
    color6_prev: str | None = None
    active_prev: str | None = None
    phase_prev: str | None = None
    out_cols: list[str] = []

    for i in range(N):
        bg_col = bg_cols[i]
        if i < S:
            seed1 = f"seed1_{i}"
            seed6 = f"seed6_{i}"
            seed1_sum = f"seed1_sum_{i}"
            seed6_sum = f"seed6_sum_{i}"
            seed_sum = f"seed_sum_{i}"
            seed_any = f"seed_any_{i}"
            nodes.extend(
                [
                    helper.make_node("Cast", [seed1_cols_f[i]], [seed1], to=TensorProto.BOOL),
                    helper.make_node("Cast", [seed6_cols_f[i]], [seed6], to=TensorProto.BOOL),
                    helper.make_node("ReduceSum", [seed1_cols_f[i]], [seed1_sum], axes=[1], keepdims=1),
                    helper.make_node("ReduceSum", [seed6_cols_f[i]], [seed6_sum], axes=[1], keepdims=1),
                    helper.make_node("Add", [seed1_sum, seed6_sum], [seed_sum]),
                    helper.make_node("Greater", [seed_sum, zero], [seed_any]),
                ]
            )
            if color1_prev is None:
                color1 = seed1
                color6 = seed6
                active = seed_any
                phase = seed_any
            else:
                color1 = f"color1_state_{i}"
                color6 = f"color6_state_{i}"
                active = f"active_{i}"
                toggled = f"toggled_{i}"
                phase_old = f"phase_old_{i}"
                phase = f"phase_{i}"
                nodes.extend(
                    [
                        helper.make_node("Or", [color1_prev, seed1], [color1]),
                        helper.make_node("Or", [color6_prev, seed6], [color6]),
                        helper.make_node("Or", [active_prev, seed_any], [active]),
                        helper.make_node("Not", [phase_prev], [toggled]),
                        helper.make_node("And", [active_prev, toggled], [phase_old]),
                        helper.make_node("Or", [seed_any, phase_old], [phase]),
                    ]
                )
            inside_col = f"inside_col_{i}"
            nodes.append(helper.make_node("Or", [bg_col, seed_any], [inside_col]))
        else:
            assert color1_prev is not None and color6_prev is not None and active_prev is not None and phase_prev is not None
            color1 = color1_prev
            color6 = color6_prev
            active = active_prev
            toggled = f"toggled_{i}"
            phase = f"phase_{i}"
            nodes.extend(
                [
                    helper.make_node("Not", [phase_prev], [toggled]),
                    helper.make_node("And", [active_prev, toggled], [phase]),
                ]
            )
            inside_col = bg_col

        not_active = f"not_active_{i}"
        color1_raw = f"color1_raw_{i}"
        color6_raw = f"color6_raw_{i}"
        color1_out = f"color1_out_{i}"
        color6_out = f"color6_out_{i}"
        not_phase = f"not_phase_{i}"
        gray_raw = f"gray_raw_{i}"
        gray_out = f"gray_out_{i}"
        bg_out = f"bg_out_{i}"
        out_col = f"out_col_{i}"
        nodes.extend(
            [
                helper.make_node("Not", [active], [not_active]),
                helper.make_node("And", [bg_col, not_active], [bg_out]),
                helper.make_node("And", [color1, phase], [color1_raw]),
                helper.make_node("And", [color6, phase], [color6_raw]),
                helper.make_node("And", [color1_raw, inside_col], [color1_out]),
                helper.make_node("And", [color6_raw, inside_col], [color6_out]),
                helper.make_node("Not", [phase], [not_phase]),
                helper.make_node("And", [active, not_phase], [gray_raw]),
                helper.make_node("And", [gray_raw, inside_col], [gray_out]),
                helper.make_node("Concat", [bg_out, color1_out, gray_out, color6_out], [out_col], axis=1),
            ]
        )
        out_cols.append(out_col)
        color1_prev = color1
        color6_prev = color6
        active_prev = active
        phase_prev = phase

    nodes.extend(
        [
            helper.make_node("Concat", out_cols, ["out14b"], axis=3),
            helper.make_node("Cast", ["out14b"], ["out14"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out14"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - N, W - N]),
        ]
    )
    return _make_model(nodes, inits, f"{TASK_ID}_seed8_column_scan")


def _load_task() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _verify_rule() -> dict[str, tuple[int, int]]:
    data = _load_task()
    counts: dict[str, tuple[int, int]] = {}
    for split in ("train", "test", "arc-gen"):
        total = 0
        passed = 0
        for example in data.get(split, []):
            total += 1
            if solve(example["input"]) == example["output"]:
                passed += 1
        counts[split] = (passed, total)
    return counts


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
    counts: dict[str, tuple[int, int]] = {}
    all_ok = True
    for split, examples in _load_task().items():
        passed = 0
        total = 0
        for example in examples:
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


def main() -> None:
    rule_counts = _verify_rule()
    print(f"rule:    {_format_counts(rule_counts)}")
    if any(ok != total for ok, total in rule_counts.values()):
        raise SystemExit("hypothesis does not match the JSON examples")

    variants: Iterable[tuple[str, onnx.ModelProto]] = (
        ("scan_split_bool", build_scan_model(split_input=True)),
        ("scan_slice_bool", build_scan_model(split_input=False)),
        ("matmul_masks", build_matmul_model()),
        ("row_broadcast", build_row_broadcast_model()),
        ("seed8_row_broadcast", build_seed8_row_broadcast_model(split_color_blocks=False)),
        ("seed8_color_blocks", build_seed8_row_broadcast_model(split_color_blocks=True)),
        ("seed8_column_scan", build_seed8_column_scan_model()),
    )

    results: list[tuple[int, str, onnx.ModelProto, dict[str, Any], str | None, int | None, dict[str, tuple[int, int]]]] = []
    for label, model in variants:
        tmp_path = OUT_DIR / f"{TASK_ID}_{label}.onnx"
        onnx.save(model, tmp_path)
        try:
            correct, counts = _check_correct(model)
            scored = score_file(tmp_path)
            memory, largest_name, largest_bytes = _profile_largest_internal(model)
            score_text = f"{scored['score']:.6f}" if scored["score"] is not None else "INVALID"
            print(
                f"{label:<18} correct={correct} ({_format_counts(counts)}) "
                f"memory={scored['memory']} params={scored['params']} cost={scored['cost']} "
                f"score={score_text} largest={largest_name}:{largest_bytes}"
            )
            if correct and scored["valid"]:
                results.append((int(scored["cost"]), label, model, scored, largest_name, largest_bytes, counts))
        except Exception as exc:
            print(f"{label:<18} invalid/error={type(exc).__name__}: {exc}")
        tmp_path.unlink(missing_ok=True)

    if not results:
        raise SystemExit("no valid correct variants")

    _, label, model, scored, largest_name, largest_bytes, counts = min(results, key=lambda item: item[0])
    onnx.save(model, BEST_PATH)
    params = calculate_params(model)
    print()
    print(f"kept:    {label}")
    print(f"wrote:   {BEST_PATH}")
    print(f"passes:  {_format_counts(counts)}")
    print(f"memory:  {scored['memory']}")
    print(f"params:  {scored['params']} (raw {params})")
    print(f"cost:    {scored['cost']}")
    print(f"score:   {scored['score']:.6f}")
    print(f"largest: {largest_name} ({largest_bytes} bytes)")


if __name__ == "__main__":
    main()
