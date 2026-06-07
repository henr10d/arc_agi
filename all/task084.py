"""Minimal ONNX for NeuroGolf task084 dynamic border/diagonal rule.

Task rule: each input is a square black grid whose left column is filled with
one non-zero color. Preserve that colored left border, draw red (color 2) on
the anti-diagonal from the top-right corner down to column 1, and draw yellow
(color 4) across the bottom row from column 1 to the right edge. Everything
else inside the square stays black; padded cells outside the square stay empty.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "task084"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task084.onnx"

C = 10
H = W = 15
H_TASK = W_TASK = 21
H_FULL = W_FULL = 30
SHAPE = [1, C, H_FULL, W_FULL]
CORE_SHAPE = [1, C, H, W]
PAD_ATTR = [0, 0, 0, 0, 0, 0, H_FULL - H, W_FULL - W]
PAD_TASK_ATTR = [0, 0, 0, 0, 0, 0, H_FULL - H_TASK, W_FULL - W_TASK]
IR_VERSION = 10


def onehot(grid: np.ndarray) -> np.ndarray:
    y = np.zeros((1, C, H_FULL, W_FULL), dtype=np.float32)
    gh, gw = grid.shape
    for r in range(min(gh, H)):
        for c in range(min(gw, W)):
            y[0, int(grid[r, c]), r, c] = 1.0
    return y


def random_input(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    gh = int(rng.integers(3, H + 1))
    gw = gh
    color = int(rng.integers(1, C))
    grid = np.zeros((gh, gw), dtype=np.int64)
    grid[:, 0] = color
    return grid_to_onehot(grid), grid


def solve_grid(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    """Reference: size-dependent border, yellow bottom row, red anti-diagonal."""
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    out = np.zeros_like(g)
    out[:, 0] = g[:, 0]
    if h >= 2 and w >= 2:
        out[h - 1, 1:w] = 4
        for r in range(h - 1):
            c = w - 1 - r
            if c >= 1:
                out[r, c] = 2
    return out


def grid_to_onehot(grid: np.ndarray) -> np.ndarray:
    y = np.zeros((1, C, H_FULL, W_FULL), dtype=np.float32)
    gh, gw = grid.shape
    for r in range(gh):
        for c in range(gw):
            y[0, int(grid[r, c]), r, c] = 1.0
    return y


def expected_output(color: int, *, gh: int = H, gw: int = W) -> np.ndarray:
    grid = np.zeros((gh, gw), dtype=np.int64)
    grid[:, 0] = color
    return grid_to_onehot(solve_grid(grid))


def init(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array, name=name)


def tensor(name: str, dtype: int, shape: Iterable[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, list(shape))


def model_core(
    name: str,
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    value_infos: list[onnx.ValueInfoProto],
    *,
    opset: int,
) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        name,
        [tensor("input", TensorProto.FLOAT, CORE_SHAPE)],
        [tensor("output", TensorProto.FLOAT, CORE_SHAPE)],
        initializer=inits,
        value_info=value_infos,
    )
    m = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", opset)])
    onnx.checker.check_model(m, full_check=True)
    return m


def wrap_30x30(m_core: onnx.ModelProto, name: str) -> onnx.ModelProto:
    """Slice top-left 15x15 from competition input, run core graph, pad to 30x30."""
    core_opset = max(int(o.version) for o in m_core.opset_import)
    inits = [
        init("io_starts", np.array([0, 0, 0, 0], dtype=np.int64)),
        init("io_ends", np.array([1, C, H, W], dtype=np.int64)),
        init("io_axes", np.array([0, 1, 2, 3], dtype=np.int64)),
    ]
    if core_opset >= 11:
        inits.append(init("io_pads", np.array(PAD_ATTR, dtype=np.int64)))
    inits.extend(list(m_core.graph.initializer))
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "io_starts", "io_ends", "io_axes"], ["core"])
    ]
    for node in m_core.graph.node:
        node_inputs = ["core" if x == "input" else x for x in node.input]
        node_outputs = ["core_out" if x == "output" else x for x in node.output]
        kwargs: dict[str, Any] = {}
        for attr in node.attribute:
            kwargs[attr.name] = onnx.helper.get_attribute_value(attr)
        nodes.append(
            helper.make_node(node.op_type, node_inputs, node_outputs, name=node_outputs[0], **kwargs)
        )
    if core_opset >= 11:
        nodes.append(helper.make_node("Pad", ["core_out", "io_pads"], ["output"], mode="constant"))
    else:
        nodes.append(
            helper.make_node("Pad", ["core_out"], ["output"], mode="constant", pads=PAD_ATTR)
        )
    graph = helper.make_graph(
        nodes,
        f"{name}_30x30",
        [tensor("input", TensorProto.FLOAT, SHAPE)],
        [tensor("output", TensorProto.FLOAT, SHAPE)],
        initializer=inits,
        value_info=list(m_core.graph.value_info),
    )
    m = helper.make_model(
        graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", core_opset)]
    )
    onnx.checker.check_model(m, full_check=True)
    return m


def variant_scatternd(opset: int = 11) -> onnx.ModelProto:
    """Best sparse update graph: 56 scalar writes into the input tensor."""
    indices: list[list[int]] = []
    updates: list[float] = []

    def write(ch: int, r: int, c: int, value: float) -> None:
        indices.append([0, ch, r, c])
        updates.append(value)

    for r in range(14):
        c = 14 - r
        write(0, r, c, 0.0)
        write(2, r, c, 1.0)
    for c in range(1, 15):
        write(0, 14, c, 0.0)
        write(4, 14, c, 1.0)

    inits = [
        init("scatternd_indices", np.asarray(indices, dtype=np.int64)),
        init("scatternd_updates", np.asarray(updates, dtype=np.float32)),
    ]
    nodes = [helper.make_node("ScatterND", ["input", "scatternd_indices", "scatternd_updates"], ["output"])]
    return model_core("task084_scatternd", nodes, inits, [], opset=opset)


def fixed_right_onehot() -> np.ndarray:
    grid = np.zeros((H, W - 1), dtype=np.int64)
    grid[14, :] = 4
    for r in range(14):
        grid[r, 13 - r] = 2
    y = np.zeros((1, C, H, W - 1), dtype=np.float32)
    for r in range(H):
        for c in range(W - 1):
            y[0, int(grid[r, c]), r, c] = 1.0
    return y


def variant_slice_concat(opset: int = 10) -> onnx.ModelProto:
    inits = [
        init("starts", np.array([0, 0, 0, 0], dtype=np.int64)),
        init("ends", np.array([1, C, H, 1], dtype=np.int64)),
        init("axes", np.array([0, 1, 2, 3], dtype=np.int64)),
        init("right", fixed_right_onehot()),
    ]
    vis = [tensor("left", TensorProto.FLOAT, [1, C, H, 1])]
    nodes = [
        helper.make_node("Slice", ["input", "starts", "ends", "axes"], ["left"]),
        helper.make_node("Concat", ["left", "right"], ["output"], axis=3),
    ]
    return model_core("task084_slice_concat", nodes, inits, vis, opset=opset)


def fixed_right_ids() -> np.ndarray:
    grid = np.zeros((1, H, W - 1), dtype=np.int64)
    grid[:, 14, :] = 4
    for r in range(14):
        grid[:, r, 13 - r] = 2
    return grid


def variant_index_onehot(opset: int = 11) -> onnx.ModelProto:
    inits = [
        init("starts", np.array([0, 0, 0, 0], dtype=np.int64)),
        init("ends", np.array([1, C, H, 1], dtype=np.int64)),
        init("axes", np.array([0, 1, 2, 3], dtype=np.int64)),
        init("left_axes", np.array([2], dtype=np.int64)),
        init("right_ids", fixed_right_ids()),
        init("depth", np.array(10, dtype=np.int64)),
        init("values", np.array([0.0, 1.0], dtype=np.float32)),
    ]
    vis = [
        tensor("left_oh", TensorProto.FLOAT, [1, C, H, 1]),
        tensor("left_ids_3d", TensorProto.INT64, [1, H, 1]),
        tensor("left_ids", TensorProto.INT64, [1, H]),
        tensor("ids", TensorProto.INT64, [1, H, W]),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "starts", "ends", "axes"], ["left_oh"]),
        helper.make_node("ArgMax", ["left_oh"], ["left_ids_3d"], axis=1, keepdims=0),
        helper.make_node("Squeeze", ["left_ids_3d", "left_axes"], ["left_ids"]),
        helper.make_node("Unsqueeze", ["left_ids", "left_axes"], ["left_col_ids"]),
        helper.make_node("Concat", ["left_col_ids", "right_ids"], ["ids"], axis=2),
        helper.make_node("OneHot", ["ids", "depth", "values"], ["output"], axis=1),
    ]
    vis.append(tensor("left_col_ids", TensorProto.INT64, [1, H, 1]))
    return model_core("task084_index_onehot", nodes, inits, vis, opset=opset)


def variant_equal_where(opset: int = 10) -> onnx.ModelProto:
    right = fixed_right_onehot()
    mask = np.zeros((1, 1, H, W), dtype=bool)
    mask[:, :, :, 0] = True
    const = np.zeros((1, C, H, W), dtype=np.float32)
    const[:, :, :, 1:] = right
    inits = [
        init("left_mask", mask),
        init("const_out", const),
    ]
    vis = []
    nodes = [helper.make_node("Where", ["left_mask", "input", "const_out"], ["output"])]
    return model_core("task084_equal_where", nodes, inits, vis, opset=opset)


def variant_channel_masks(opset: int = 10) -> onnx.ModelProto:
    """Dynamic 30x30 graph: infer square size and patch only affected channels."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = [
        init("row0_starts", np.array([0, 0, 0, 1], dtype=np.int64)),
        init("row0_ends", np.array([1, 1, 1, W_FULL], dtype=np.int64)),
        init("axes4", np.array([0, 1, 2, 3], dtype=np.int64)),
        init("rows_i", np.arange(H_FULL, dtype=np.int64).reshape(1, 1, H_FULL, 1)),
        init("cols_i", np.arange(W_FULL, dtype=np.int64).reshape(1, 1, 1, W_FULL)),
        init("zero_i", np.array(0, dtype=np.int64)),
        init("one_i", np.array(1, dtype=np.int64)),
        init("zero_f", np.array(0.0, dtype=np.float32)),
        init("one_f", np.array(1.0, dtype=np.float32)),
    ]
    vis: list[onnx.ValueInfoProto] = [
        tensor("row0_bg", TensorProto.FLOAT, [1, 1, 1, W_FULL - 1]),
        tensor("w_minus1", TensorProto.FLOAT, [1, 1, 1, 1]),
        tensor("w_minus1_i", TensorProto.INT64, [1, 1, 1, 1]),
        tensor("n", TensorProto.INT64, [1, 1, 1, 1]),
        tensor("n_minus1", TensorProto.INT64, [1, 1, 1, 1]),
        tensor("on_bottom", TensorProto.BOOL, [1, 1, H_FULL, 1]),
        tensor("cols_ge1", TensorProto.BOOL, [1, 1, 1, W_FULL]),
        tensor("cols_lt_n", TensorProto.BOOL, [1, 1, 1, W_FULL]),
        tensor("yellow0", TensorProto.BOOL, [1, 1, H_FULL, W_FULL]),
        tensor("yellow_m", TensorProto.BOOL, [1, 1, H_FULL, W_FULL]),
        tensor("red_col", TensorProto.INT64, [1, 1, H_FULL, 1]),
        tensor("red0", TensorProto.BOOL, [1, 1, H_FULL, W_FULL]),
        tensor("above_bottom", TensorProto.BOOL, [1, 1, H_FULL, 1]),
        tensor("red_m", TensorProto.BOOL, [1, 1, H_FULL, W_FULL]),
        tensor("clear_m", TensorProto.BOOL, [1, 1, H_FULL, W_FULL]),
    ]

    nodes.extend(
        [
            helper.make_node("Slice", ["input", "row0_starts", "row0_ends", "axes4"], ["row0_bg"]),
            helper.make_node("ReduceSum", ["row0_bg"], ["w_minus1"], axes=[3], keepdims=1),
            helper.make_node("Cast", ["w_minus1"], ["w_minus1_i"], to=TensorProto.INT64),
            helper.make_node("Add", ["w_minus1_i", "one_i"], ["n"]),
            helper.make_node("Sub", ["n", "one_i"], ["n_minus1"]),
            helper.make_node("Equal", ["rows_i", "n_minus1"], ["on_bottom"]),
            helper.make_node("Greater", ["cols_i", "zero_i"], ["cols_ge1"]),
            helper.make_node("Less", ["cols_i", "n"], ["cols_lt_n"]),
            helper.make_node("And", ["on_bottom", "cols_ge1"], ["yellow0"]),
            helper.make_node("And", ["yellow0", "cols_lt_n"], ["yellow_m"]),
            helper.make_node("Sub", ["n_minus1", "rows_i"], ["red_col"]),
            helper.make_node("Equal", ["cols_i", "red_col"], ["red0"]),
            helper.make_node("Less", ["rows_i", "n_minus1"], ["above_bottom"]),
            helper.make_node("And", ["red0", "above_bottom"], ["red_m"]),
            helper.make_node("Or", ["red_m", "yellow_m"], ["clear_m"]),
            helper.make_node(
                "Split",
                ["input"],
                [f"ch{c}" for c in range(C)],
                axis=1,
                split=[1] * C,
            ),
            helper.make_node("Where", ["clear_m", "zero_f", "ch0"], ["out0"]),
            helper.make_node("Where", ["red_m", "one_f", "ch2"], ["out2"]),
            helper.make_node("Where", ["yellow_m", "one_f", "ch4"], ["out4"]),
            helper.make_node(
                "Concat",
                ["out0", "ch1", "out2", "ch3", "out4", "ch5", "ch6", "ch7", "ch8", "ch9"],
                ["output"],
                axis=1,
            ),
        ]
    )
    vis.extend(tensor(f"ch{c}", TensorProto.FLOAT, [1, 1, H_FULL, W_FULL]) for c in range(C))
    vis.extend(
        [
            tensor("out0", TensorProto.FLOAT, [1, 1, H_FULL, W_FULL]),
            tensor("out2", TensorProto.FLOAT, [1, 1, H_FULL, W_FULL]),
            tensor("out4", TensorProto.FLOAT, [1, 1, H_FULL, W_FULL]),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task084_channel_masks",
        [tensor("input", TensorProto.FLOAT, SHAPE)],
        [tensor("output", TensorProto.FLOAT, SHAPE)],
        initializer=inits,
        value_info=vis,
    )
    m = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", opset)])
    onnx.checker.check_model(m, full_check=True)
    return m


def variant_overlay_where(opset: int = 10) -> onnx.ModelProto:
    """Dynamic 30x30 graph: clear affected cells, then final-overlay red/yellow."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = [
        init("row0_starts", np.array([0, 0, 0, 1], dtype=np.int64)),
        init("row0_ends", np.array([1, 1, 1, W_FULL], dtype=np.int64)),
        init("axes4", np.array([0, 1, 2, 3], dtype=np.int64)),
        init("rows_i", np.arange(H_FULL, dtype=np.int64).reshape(1, 1, H_FULL, 1)),
        init("cols_i", np.arange(W_FULL, dtype=np.int64).reshape(1, 1, 1, W_FULL)),
        init("zero_i", np.array(0, dtype=np.int64)),
        init("one_i", np.array(1, dtype=np.int64)),
        init("zero_f", np.array(0.0, dtype=np.float32)),
        init("one_f", np.array(1.0, dtype=np.float32)),
    ]
    vis: list[onnx.ValueInfoProto] = [
        tensor("row0_bg", TensorProto.FLOAT, [1, 1, 1, W_FULL - 1]),
        tensor("w_minus1", TensorProto.FLOAT, [1, 1, 1, 1]),
        tensor("w_minus1_i", TensorProto.INT64, [1, 1, 1, 1]),
        tensor("n", TensorProto.INT64, [1, 1, 1, 1]),
        tensor("n_minus1", TensorProto.INT64, [1, 1, 1, 1]),
        tensor("on_bottom", TensorProto.BOOL, [1, 1, H_FULL, 1]),
        tensor("cols_ge1", TensorProto.BOOL, [1, 1, 1, W_FULL]),
        tensor("cols_lt_n", TensorProto.BOOL, [1, 1, 1, W_FULL]),
        tensor("yellow0", TensorProto.BOOL, [1, 1, H_FULL, W_FULL]),
        tensor("yellow_m", TensorProto.BOOL, [1, 1, H_FULL, W_FULL]),
        tensor("red_col", TensorProto.INT64, [1, 1, H_FULL, 1]),
        tensor("red0", TensorProto.BOOL, [1, 1, H_FULL, W_FULL]),
        tensor("above_bottom", TensorProto.BOOL, [1, 1, H_FULL, 1]),
        tensor("red_m", TensorProto.BOOL, [1, 1, H_FULL, W_FULL]),
        tensor("clear_m", TensorProto.BOOL, [1, 1, H_FULL, W_FULL]),
        tensor("cleared", TensorProto.FLOAT, [1, C, H_FULL, W_FULL]),
        tensor("false_m", TensorProto.BOOL, [1, 1, H_FULL, W_FULL]),
        tensor("line_ch", TensorProto.BOOL, [1, C, H_FULL, W_FULL]),
    ]

    nodes.extend(
        [
            helper.make_node("Slice", ["input", "row0_starts", "row0_ends", "axes4"], ["row0_bg"]),
            helper.make_node("ReduceSum", ["row0_bg"], ["w_minus1"], axes=[3], keepdims=1),
            helper.make_node("Cast", ["w_minus1"], ["w_minus1_i"], to=TensorProto.INT64),
            helper.make_node("Add", ["w_minus1_i", "one_i"], ["n"]),
            helper.make_node("Sub", ["n", "one_i"], ["n_minus1"]),
            helper.make_node("Equal", ["rows_i", "n_minus1"], ["on_bottom"]),
            helper.make_node("Greater", ["cols_i", "zero_i"], ["cols_ge1"]),
            helper.make_node("Less", ["cols_i", "n"], ["cols_lt_n"]),
            helper.make_node("And", ["on_bottom", "cols_ge1"], ["yellow0"]),
            helper.make_node("And", ["yellow0", "cols_lt_n"], ["yellow_m"]),
            helper.make_node("Sub", ["n_minus1", "rows_i"], ["red_col"]),
            helper.make_node("Equal", ["cols_i", "red_col"], ["red0"]),
            helper.make_node("Less", ["rows_i", "n_minus1"], ["above_bottom"]),
            helper.make_node("And", ["red0", "above_bottom"], ["red_m"]),
            helper.make_node("Or", ["red_m", "yellow_m"], ["clear_m"]),
            helper.make_node("Where", ["clear_m", "zero_f", "input"], ["cleared"]),
            helper.make_node("And", ["red_m", "yellow_m"], ["false_m"]),
            helper.make_node(
                "Concat",
                [
                    "false_m",
                    "false_m",
                    "red_m",
                    "false_m",
                    "yellow_m",
                    "false_m",
                    "false_m",
                    "false_m",
                    "false_m",
                    "false_m",
                ],
                ["line_ch"],
                axis=1,
            ),
            helper.make_node("Where", ["line_ch", "one_f", "cleared"], ["output"]),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task084_overlay_where",
        [tensor("input", TensorProto.FLOAT, SHAPE)],
        [tensor("output", TensorProto.FLOAT, SHAPE)],
        initializer=inits,
        value_info=vis,
    )
    m = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", opset)])
    onnx.checker.check_model(m, full_check=True)
    return m


def variant_overlay_where_21(opset: int = 10) -> onnx.ModelProto:
    """Specialized dynamic graph for observed task sizes up to 21x21."""
    h = H_TASK
    w = W_TASK
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = [
        init("core_starts", np.array([0, 0, 0, 0], dtype=np.int64)),
        init("core_ends", np.array([1, C, h, w], dtype=np.int64)),
        init("row0_starts", np.array([0, 0, 0, 1], dtype=np.int64)),
        init("row0_ends", np.array([1, 1, 1, w], dtype=np.int64)),
        init("axes4", np.array([0, 1, 2, 3], dtype=np.int64)),
        init("rows_i21", np.arange(h, dtype=np.int64).reshape(1, 1, h, 1)),
        init("cols_i21", np.arange(w, dtype=np.int64).reshape(1, 1, 1, w)),
        init("zero_i", np.array(0, dtype=np.int64)),
        init("one_i", np.array(1, dtype=np.int64)),
        init("zero_f", np.array(0.0, dtype=np.float32)),
        init("one_f", np.array(1.0, dtype=np.float32)),
    ]
    vis: list[onnx.ValueInfoProto] = [
        tensor("core", TensorProto.FLOAT, [1, C, h, w]),
        tensor("row0_bg", TensorProto.FLOAT, [1, 1, 1, w - 1]),
        tensor("w_minus1", TensorProto.FLOAT, [1, 1, 1, 1]),
        tensor("w_minus1_i", TensorProto.INT64, [1, 1, 1, 1]),
        tensor("n", TensorProto.INT64, [1, 1, 1, 1]),
        tensor("n_minus1", TensorProto.INT64, [1, 1, 1, 1]),
        tensor("on_bottom", TensorProto.BOOL, [1, 1, h, 1]),
        tensor("cols_ge1", TensorProto.BOOL, [1, 1, 1, w]),
        tensor("cols_lt_n", TensorProto.BOOL, [1, 1, 1, w]),
        tensor("yellow0", TensorProto.BOOL, [1, 1, h, w]),
        tensor("yellow_m", TensorProto.BOOL, [1, 1, h, w]),
        tensor("red_col", TensorProto.INT64, [1, 1, h, 1]),
        tensor("red0", TensorProto.BOOL, [1, 1, h, w]),
        tensor("above_bottom", TensorProto.BOOL, [1, 1, h, 1]),
        tensor("red_m", TensorProto.BOOL, [1, 1, h, w]),
        tensor("clear_m", TensorProto.BOOL, [1, 1, h, w]),
        tensor("cleared", TensorProto.FLOAT, [1, C, h, w]),
        tensor("false_m", TensorProto.BOOL, [1, 1, h, w]),
        tensor("line_ch", TensorProto.BOOL, [1, C, h, w]),
        tensor("core_out", TensorProto.FLOAT, [1, C, h, w]),
    ]

    nodes.extend(
        [
            helper.make_node("Slice", ["input", "core_starts", "core_ends", "axes4"], ["core"]),
            helper.make_node("Slice", ["core", "row0_starts", "row0_ends", "axes4"], ["row0_bg"]),
            helper.make_node("ReduceSum", ["row0_bg"], ["w_minus1"], axes=[3], keepdims=1),
            helper.make_node("Cast", ["w_minus1"], ["w_minus1_i"], to=TensorProto.INT64),
            helper.make_node("Add", ["w_minus1_i", "one_i"], ["n"]),
            helper.make_node("Sub", ["n", "one_i"], ["n_minus1"]),
            helper.make_node("Equal", ["rows_i21", "n_minus1"], ["on_bottom"]),
            helper.make_node("Greater", ["cols_i21", "zero_i"], ["cols_ge1"]),
            helper.make_node("Less", ["cols_i21", "n"], ["cols_lt_n"]),
            helper.make_node("And", ["on_bottom", "cols_ge1"], ["yellow0"]),
            helper.make_node("And", ["yellow0", "cols_lt_n"], ["yellow_m"]),
            helper.make_node("Sub", ["n_minus1", "rows_i21"], ["red_col"]),
            helper.make_node("Equal", ["cols_i21", "red_col"], ["red0"]),
            helper.make_node("Less", ["rows_i21", "n_minus1"], ["above_bottom"]),
            helper.make_node("And", ["red0", "above_bottom"], ["red_m"]),
            helper.make_node("Or", ["red_m", "yellow_m"], ["clear_m"]),
            helper.make_node("Where", ["clear_m", "zero_f", "core"], ["cleared"]),
            helper.make_node("And", ["red_m", "yellow_m"], ["false_m"]),
            helper.make_node(
                "Concat",
                [
                    "false_m",
                    "false_m",
                    "red_m",
                    "false_m",
                    "yellow_m",
                    "false_m",
                    "false_m",
                    "false_m",
                    "false_m",
                    "false_m",
                ],
                ["line_ch"],
                axis=1,
            ),
            helper.make_node("Where", ["line_ch", "one_f", "cleared"], ["core_out"]),
            helper.make_node("Pad", ["core_out"], ["output"], mode="constant", pads=PAD_TASK_ATTR),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task084_overlay_where_21",
        [tensor("input", TensorProto.FLOAT, SHAPE)],
        [tensor("output", TensorProto.FLOAT, SHAPE)],
        initializer=inits,
        value_info=vis,
    )
    m = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", opset)])
    onnx.checker.check_model(m, full_check=True)
    return m


def _ids_to_onehot(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    ids: str,
    prefix: str,
    *,
    out_name: str = "output",
) -> str:
    ch: list[str] = []
    for color in range(C):
        inits.append(init(f"{prefix}_cv{color}", np.array(color, dtype=np.int64)))
        out = f"{prefix}_bc{color}"
        ch.append(out)
        nodes.append(helper.make_node("Equal", [ids, f"{prefix}_cv{color}"], [out]))
    out10b = f"{prefix}_ohb"
    nodes.extend(
        [
            helper.make_node("Concat", ch, [out10b], axis=1),
            helper.make_node("Cast", [out10b], [out_name], to=TensorProto.FLOAT),
        ]
    )
    return out_name


def variant_extent(opset: int = 10) -> onnx.ModelProto:
    """Full 30x30: infer h/w from content, apply rule, mask padding."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    vis: list[onnx.ValueInfoProto] = []

    rows = np.arange(H_FULL, dtype=np.int64).reshape(1, 1, H_FULL, 1)
    cols = np.arange(W_FULL, dtype=np.int64).reshape(1, 1, 1, W_FULL)
    inits.extend(
        [
            init("rows", rows),
            init("cols", cols),
            init("one", np.array(1, dtype=np.int64)),
            init("zero", np.array(0, dtype=np.int64)),
            init("two", np.array(2, dtype=np.int64)),
            init("four", np.array(4, dtype=np.int64)),
        ]
    )

    inits.extend(
        [
            init("col0_starts", np.array([0, 0, 0, 0], dtype=np.int64)),
            init("col0_ends", np.array([1, 1, H_FULL, 1], dtype=np.int64)),
            init("col0_axes", np.array([0, 1, 2, 3], dtype=np.int64)),
        ]
    )

    nodes.extend(
        [
            helper.make_node("ArgMax", ["input"], ["ids"], axis=1, keepdims=1),
            helper.make_node(
                "Slice",
                ["ids", "col0_starts", "col0_ends", "col0_axes"],
                ["left_col"],
            ),
            helper.make_node("Greater", ["left_col", "zero"], ["left_nz"]),
            helper.make_node("Cast", ["left_nz"], ["left_i"], to=TensorProto.INT64),
            helper.make_node("Mul", ["left_i", "rows"], ["row_w"]),
            helper.make_node("ReduceMax", ["row_w"], ["row_max"], axes=[2], keepdims=0),
            helper.make_node("Add", ["row_max", "one"], ["h"]),
        ]
    )
    vis.extend(
        [
            tensor("ids", TensorProto.INT64, [1, 1, H_FULL, W_FULL]),
            tensor("left_col", TensorProto.INT64, [1, 1, H_FULL, 1]),
        ]
    )

    nodes.extend(
        [
            helper.make_node("Sub", ["h", "one"], ["hm1"]),
            helper.make_node("Equal", ["rows", "hm1"], ["on_bottom"]),
            helper.make_node("Greater", ["cols", "zero"], ["ge1"]),
            helper.make_node("Less", ["cols", "h"], ["ltw"]),
            helper.make_node("And", ["on_bottom", "ge1"], ["y0"]),
            helper.make_node("And", ["y0", "ltw"], ["yellow_m"]),
            helper.make_node("Sub", ["hm1", "rows"], ["red_c"]),
            helper.make_node("Equal", ["cols", "red_c"], ["on_red_c"]),
            helper.make_node("Less", ["rows", "hm1"], ["above_bottom"]),
            helper.make_node("And", ["on_red_c", "above_bottom"], ["r0"]),
            helper.make_node("And", ["r0", "ge1"], ["red_m"]),
            helper.make_node("Where", ["yellow_m", "four", "ids"], ["ids_y"]),
            helper.make_node("Where", ["red_m", "two", "ids_y"], ["ids_out"]),
            helper.make_node("Less", ["rows", "h"], ["in_h"]),
            helper.make_node("Less", ["cols", "h"], ["in_w"]),
            helper.make_node("And", ["in_h", "in_w"], ["valid_m"]),
            helper.make_node("Where", ["valid_m", "ids_out", "zero"], ["ids_masked"]),
        ]
    )
    _ids_to_onehot(nodes, inits, "ids_masked", "oh", out_name="oh")
    nodes.extend(
        [
            helper.make_node("Cast", ["valid_m"], ["valid_f"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["oh", "valid_f"], ["output"]),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task084_extent",
        [tensor("input", TensorProto.FLOAT, SHAPE)],
        [tensor("output", TensorProto.FLOAT, SHAPE)],
        initializer=inits,
        value_info=vis,
    )
    m = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", opset)])
    onnx.checker.check_model(m, full_check=True)
    return m


def validate_task_json(m: onnx.ModelProto) -> tuple[bool, str]:
    task_path = ROOT / "data" / f"{TASK_ID}.json"
    with task_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data.get(split, []):
            total += 1
            inp = grid_to_onehot(np.array(ex["input"], dtype=np.int64))
            want = grid_to_onehot(solve_grid(np.array(ex["input"], dtype=np.int64)))
            got = sess.run(["output"], {"input": inp})[0]
            if np.array_equal((got > 0.0).astype(np.float32), want):
                passed += 1
    return passed == total, f"{passed}/{total}"


def sanitize_for_measure(m: onnx.ModelProto) -> onnx.ModelProto | None:
    m = onnx.ModelProto().FromString(m.SerializeToString())
    for node in m.graph.node:
        if not node.output:
            return None
        node.name = node.output[0]
    return m


def params_count(m: onnx.ModelProto) -> int:
    total = 0
    for item in m.graph.initializer:
        total += int(math.prod(item.dims))
    for node in m.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                total += int(math.prod(attr.t.dims))
            elif attr.name in {"value_floats", "value_ints", "value_strings"}:
                total += len(getattr(attr, attr.name.split("_", 1)[1]))
    return total


def profile_trace(m: onnx.ModelProto, inputs: list[np.ndarray], stem: str) -> tuple[str | None, str | None]:
    opts = ort.SessionOptions()
    opts.enable_profiling = True
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.profile_file_prefix = str(Path(tempfile.gettempdir()) / f"ng_{stem}")
    try:
        sess = ort.InferenceSession(m.SerializeToString(), sess_options=opts, providers=["CPUExecutionProvider"])
        for x in inputs:
            sess.run(["output"], {"input": x})
        return sess.end_profiling(), None
    except Exception:
        return None, traceback.format_exc()


def inferred_shapes(m: onnx.ModelProto) -> dict[str, tuple[str, list[int]]]:
    graph = onnx.shape_inference.infer_shapes(m, strict_mode=True).graph
    out: dict[str, tuple[str, list[int]]] = {}
    for vi in list(graph.value_info) + list(graph.output):
        tt = vi.type.tensor_type
        dtype = TensorProto.DataType.Name(tt.elem_type).lower()
        shape = [int(d.dim_value) for d in tt.shape.dim]
        out[vi.name] = (dtype, shape)
    return out


def memory_count(m: onnx.ModelProto, trace_path: str) -> int:
    graph = onnx.shape_inference.infer_shapes(m, strict_mode=True).graph
    io = {x.name for x in list(graph.input) + list(graph.output)}
    by_name = {x.name: x for x in list(graph.input) + list(graph.value_info) + list(graph.output)}
    mem: dict[str, int] = {}
    dtype: dict[str, np.dtype[Any]] = {}
    for name, vi in by_name.items():
        if name in io:
            continue
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            continue
        n = 1
        for d in tt.shape.dim:
            if not d.HasField("dim_value") or d.dim_value <= 0:
                raise ValueError(f"non-static shape for {name}")
            n *= int(d.dim_value)
        dt = onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)
        dtype[name] = np.dtype(dt)
        mem[name] = int(n * np.dtype(dt).itemsize)

    node_outputs = {node.name: list(node.output) for node in graph.node}
    with open(trace_path, encoding="utf-8") as fh:
        trace = json.load(fh)
    for event in trace:
        if event.get("cat") != "Node":
            continue
        node = event.get("name", "").replace("_kernel_time", "")
        for idx, shape_dict in enumerate(event.get("args", {}).get("output_type_shape", [])):
            if node not in node_outputs or idx >= len(node_outputs[node]):
                continue
            out_name = node_outputs[node][idx]
            if out_name not in dtype:
                continue
            size = 0
            for dims in shape_dict.values():
                size += int(math.prod(dims)) * dtype[out_name].itemsize
            mem[out_name] = max(mem[out_name], size)
    return int(sum(mem.values()))


def validate(m: onnx.ModelProto, trials: int = 64) -> None:
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(84)
    for _ in range(trials):
        x, grid = random_input(rng)
        got = sess.run(["output"], {"input": x})[0]
        want = grid_to_onehot(solve_grid(grid))
        if not np.array_equal(got > 0.0, want > 0.0):
            raise AssertionError("validation mismatch")


def score_model(name: str, m: onnx.ModelProto) -> dict[str, Any]:
    sanitized = sanitize_for_measure(m)
    if sanitized is None:
        return {"name": name, "valid": False, "error": "sanitize failed"}
    inputs = [random_input(np.random.default_rng(seed))[0] for seed in range(8)]
    trace_path, error = profile_trace(sanitized, inputs, name)
    if error or trace_path is None:
        return {"name": name, "valid": False, "error": error}
    try:
        memory = memory_count(sanitized, trace_path)
        params = params_count(sanitized)
        validate(sanitized)
    except Exception:
        return {"name": name, "valid": False, "error": traceback.format_exc()}
    cost = memory + params
    return {
        "name": name,
        "valid": True,
        "memory": memory,
        "params": params,
        "cost": cost,
        "score": max(1.0, 25.0 - math.log(max(1.0, cost))),
        "shapes": inferred_shapes(sanitized),
    }


def variants() -> list[tuple[str, onnx.ModelProto]]:
    return [("G_overlay_where_opset10", variant_overlay_where(10))]


def print_report(results: list[dict[str, Any]]) -> None:
    for r in results:
        if not r["valid"]:
            print(f"{r['name']}: INVALID")
            print(str(r.get("error", "")).strip())
            continue
        print(
            f"{r['name']}: memory={r['memory']} params={r['params']} "
            f"cost={r['cost']} score={r['score']:.6f}"
        )
    best = min((r for r in results if r["valid"]), key=lambda r: int(r["cost"]))
    print()
    print(f"best: {best['name']}")
    print("internal inferred tensors:")
    for name, (dtype, shape) in sorted(best["shapes"].items()):
        if name != "output":
            print(f"  {name}: {dtype} {shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the cheapest task084 ONNX model (30x30 I/O).")
    parser.add_argument("--out", type=Path, default=BEST_PATH)
    parser.add_argument("--keep-variants", action="store_true")
    args = parser.parse_args()

    built = variants()
    results = [score_model(name, m) for name, m in built]
    print_report(results)
    valid = [r for r in results if r["valid"]]
    if not valid:
        raise SystemExit("no valid variants")
    correct: list[tuple[str, onnx.ModelProto, dict[str, Any]]] = []
    for name, m in built:
        r = next(x for x in results if x["name"] == name)
        if not r["valid"]:
            continue
        ok, summary = validate_task_json(m)
        print(f"{name}: task_json {summary}")
        if ok:
            correct.append((name, m, r))
    if not correct:
        raise SystemExit("no variant passes all task examples")
    best_name, best_model, _ = min(correct, key=lambda t: int(t[2]["cost"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(best_model, args.out)
    print(f"wrote: {args.out}")

    if args.keep_variants:
        for name, m in built:
            onnx.save(m, args.out.with_name(f"{name}.onnx"))


if __name__ == "__main__":
    main()
