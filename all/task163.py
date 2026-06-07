"""Compact ONNX generator for NeuroGolf task163.

Task rule: the 11x11 input is split by gray separator rows/columns at indices
3 and 7 into nine 3x3 regions. Find the unique region containing color 4.
Copy that whole 3x3 region to the output region whose 3x3 position matches
the local position of the color-4 pixel inside the source region. Keep the
gray separator lattice unchanged and make every other content cell black.

ONNX: for each of the nine local cell positions, take a stepped 3x3 slice over
the nine regions and run a tiny shared 1x1 Conv to convert one-hot colors into
single color codes. MatMul selectors from the color-4 channel choose the source
region for each possible target block. The graph assembles a 3D 11x11 color-code
lattice, expands it with OneHot, then pads to the required 30x30 output.
"""

from __future__ import annotations

import json
import shutil
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

TASK_NUM = "163"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task163.onnx"
ROOT_SOLUTION_PATH = ROOT / "solution.onnx"

IN_NAME = "input"
OUT_NAME = "output"
C = 10
NC = 9
GH = GW = 11
H = W = 30
SHAPE = [1, C, H, W]
IR_VERSION = 10
OPSET = 10


@dataclass(frozen=True)
class Variant:
    name: str
    build: Callable[[], onnx.ModelProto]


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self._n = 0

    def name(self, prefix: str = "t") -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    def init(self, name: str, arr: np.ndarray) -> str:
        self.inits.append(numpy_helper.from_array(arr, name))
        return name

    def i64(self, name: str, values: list[int]) -> str:
        return self.init(name, np.asarray(values, dtype=np.int64))

    def f32(self, name: str, values: list[float]) -> str:
        return self.init(name, np.asarray(values, dtype=np.float32))

    def node(self, op_type: str, inputs: list[str], output: str | None = None, **attrs: Any) -> str:
        out = output or self.name()
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out


def make_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto]) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        TASK_ID,
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
    onnx.checker.check_model(model, full_check=True)
    return model


def gray_band(shape: tuple[int, int, int, int]) -> np.ndarray:
    arr = np.zeros(shape, dtype=np.float32)
    arr[:, 4:5, :, :] = 1.0
    return arr


def gray_code(shape: tuple[int, int, int, int]) -> np.ndarray:
    return np.full(shape, 5.0, dtype=np.float32)


def build_color_code_matmul() -> onnx.ModelProto:
    b = Builder()
    axes = b.i64("axes", [1, 2, 3])
    steps = b.i64("steps", [1, 4, 4])
    flat_shape = b.i64("flat_shape", [1, 1, 1, 9])
    sel_shape = b.i64("sel_shape", [1, 1, 9, 1])
    patch_shape = b.i64("patch_shape", [1, 1, 3, 3])
    color_values = b.init("color_values", np.arange(10, dtype=np.int64).reshape(1, 10, 1, 1))
    gray_col = b.init("gray_col_code", gray_code((1, 1, 3, 1)))
    gray_row = b.init("gray_row_code", gray_code((1, 1, 1, GW)))

    local_cells: list[str] = []
    for rr in range(3):
        for cc in range(3):
            st = b.i64(f"d_st_{rr}{cc}", [0, rr, cc])
            en = b.i64(f"d_en_{rr}{cc}", [10, rr + 9, cc + 9])
            sampled = b.node("Slice", [IN_NAME, st, en, axes, steps], f"d_s_{rr}{cc}")
            arg = b.node("ArgMax", [sampled], f"arg_{rr}{cc}", axis=1, keepdims=1)
            arg_f = b.node("Cast", [arg], f"arg_f_{rr}{cc}", to=TensorProto.FLOAT)
            local_cells.append(b.node("Reshape", [arg_f, flat_shape], f"d_f_{rr}{cc}"))
    data = b.node("Concat", local_cells, "data_code", axis=2)

    patches: list[str] = []
    for rr in range(3):
        for cc in range(3):
            st = b.i64(f"s_st_{rr}{cc}", [4, rr, cc])
            en = b.i64(f"s_en_{rr}{cc}", [5, rr + 9, cc + 9])
            sel = b.node("Slice", [IN_NAME, st, en, axes, steps], f"sel_{rr}{cc}")
            sel_flat = b.node("Reshape", [sel, sel_shape], f"sf_{rr}{cc}")
            picked = b.node("MatMul", [data, sel_flat], f"pick_{rr}{cc}")
            patches.append(b.node("Reshape", [picked, patch_shape], f"patch_{rr}{cc}"))

    rows: list[str] = []
    for rr in range(3):
        row_parts = [patches[rr * 3], gray_col, patches[rr * 3 + 1], gray_col, patches[rr * 3 + 2]]
        rows.append(b.node("Concat", row_parts, f"row_{rr}", axis=3))
    code = b.node("Concat", [rows[0], gray_row, rows[1], gray_row, rows[2]], "code", axis=2)
    code_i = b.node("Cast", [code], "code_i", to=TensorProto.INT64)
    onehot_b = b.node("Equal", [code_i, color_values], "onehot_b")
    core = b.node("Cast", [onehot_b], "core", to=TensorProto.FLOAT)
    b.nodes.append(
        helper.make_node(
            "Pad",
            [core],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - GH, W - GW],
        )
    )
    return make_model(b.nodes, b.inits)


def build_conv_code_matmul() -> onnx.ModelProto:
    b = Builder()
    axes_sel = b.i64("axes_sel", [1, 2, 3])
    axes_sp = b.i64("axes_sp", [2, 3])
    steps_sel = b.i64("steps_sel", [1, 4, 4])
    steps_sp = b.i64("steps_sp", [4, 4])
    flat_shape = b.i64("flat_shape", [1, 1, 1, 9])
    sel_shape = b.i64("sel_shape", [1, 1, 9, 1])
    patch_shape = b.i64("patch_shape", [1, 1, 3, 3])
    color_values = b.init("color_values", np.arange(10, dtype=np.int64).reshape(1, 10, 1, 1))
    gray_col = b.init("gray_col_code", gray_code((1, 1, 3, 1)))
    gray_row = b.init("gray_row_code", gray_code((1, 1, 1, GW)))
    weight = b.init("code_weight", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))

    code30 = b.node("Conv", [IN_NAME, weight], "code30", kernel_shape=[1, 1])
    local_cells: list[str] = []
    for rr in range(3):
        for cc in range(3):
            st = b.i64(f"d_st_{rr}{cc}", [rr, cc])
            en = b.i64(f"d_en_{rr}{cc}", [rr + 9, cc + 9])
            sampled = b.node("Slice", [code30, st, en, axes_sp, steps_sp], f"d_s_{rr}{cc}")
            local_cells.append(b.node("Reshape", [sampled, flat_shape], f"d_f_{rr}{cc}"))
    data = b.node("Concat", local_cells, "data_code", axis=2)

    patches: list[str] = []
    for rr in range(3):
        for cc in range(3):
            st = b.i64(f"s_st_{rr}{cc}", [4, rr, cc])
            en = b.i64(f"s_en_{rr}{cc}", [5, rr + 9, cc + 9])
            sel = b.node("Slice", [IN_NAME, st, en, axes_sel, steps_sel], f"sel_{rr}{cc}")
            sel_flat = b.node("Reshape", [sel, sel_shape], f"sf_{rr}{cc}")
            picked = b.node("MatMul", [data, sel_flat], f"pick_{rr}{cc}")
            patches.append(b.node("Reshape", [picked, patch_shape], f"patch_{rr}{cc}"))

    rows: list[str] = []
    for rr in range(3):
        row_parts = [patches[rr * 3], gray_col, patches[rr * 3 + 1], gray_col, patches[rr * 3 + 2]]
        rows.append(b.node("Concat", row_parts, f"row_{rr}", axis=3))
    code = b.node("Concat", [rows[0], gray_row, rows[1], gray_row, rows[2]], "code", axis=2)
    code_i = b.node("Cast", [code], "code_i", to=TensorProto.INT64)
    onehot_b = b.node("Equal", [code_i, color_values], "onehot_b")
    core = b.node("Cast", [onehot_b], "core", to=TensorProto.FLOAT)
    b.nodes.append(
        helper.make_node(
            "Pad",
            [core],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - GH, W - GW],
        )
    )
    return make_model(b.nodes, b.inits)


def build_conv_code_onehot() -> onnx.ModelProto:
    b = Builder()
    axes_sel = b.i64("axes_sel", [1, 2, 3])
    axes_sp = b.i64("axes_sp", [2, 3])
    steps_sel = b.i64("steps_sel", [1, 4, 4])
    steps_sp = b.i64("steps_sp", [4, 4])
    flat_shape = b.i64("flat_shape", [1, 1, 1, 9])
    sel_shape = b.i64("sel_shape", [1, 1, 9, 1])
    patch_shape = b.i64("patch_shape", [1, 1, 3, 3])
    depth = b.init("depth", np.asarray(10, dtype=np.int64))
    onehot_values = b.init("onehot_values", np.asarray([0.0, 1.0], dtype=np.float32))
    gray_col = b.init("gray_col_code", gray_code((1, 1, 3, 1)))
    gray_row = b.init("gray_row_code", gray_code((1, 1, 1, GW)))
    weight = b.init("code_weight", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))

    code30 = b.node("Conv", [IN_NAME, weight], "code30", kernel_shape=[1, 1])
    local_cells: list[str] = []
    for rr in range(3):
        for cc in range(3):
            st = b.i64(f"d_st_{rr}{cc}", [rr, cc])
            en = b.i64(f"d_en_{rr}{cc}", [rr + 9, cc + 9])
            sampled = b.node("Slice", [code30, st, en, axes_sp, steps_sp], f"d_s_{rr}{cc}")
            local_cells.append(b.node("Reshape", [sampled, flat_shape], f"d_f_{rr}{cc}"))
    data = b.node("Concat", local_cells, "data_code", axis=2)

    patches: list[str] = []
    for rr in range(3):
        for cc in range(3):
            st = b.i64(f"s_st_{rr}{cc}", [4, rr, cc])
            en = b.i64(f"s_en_{rr}{cc}", [5, rr + 9, cc + 9])
            sel = b.node("Slice", [IN_NAME, st, en, axes_sel, steps_sel], f"sel_{rr}{cc}")
            sel_flat = b.node("Reshape", [sel, sel_shape], f"sf_{rr}{cc}")
            picked = b.node("MatMul", [data, sel_flat], f"pick_{rr}{cc}")
            patches.append(b.node("Reshape", [picked, patch_shape], f"patch_{rr}{cc}"))

    rows: list[str] = []
    for rr in range(3):
        row_parts = [patches[rr * 3], gray_col, patches[rr * 3 + 1], gray_col, patches[rr * 3 + 2]]
        rows.append(b.node("Concat", row_parts, f"row_{rr}", axis=3))
    code = b.node("Concat", [rows[0], gray_row, rows[1], gray_row, rows[2]], "code", axis=2)
    code_3d = b.node("Squeeze", [code], "code_3d", axes=[1])
    code_i = b.node("Cast", [code_3d], "code_i", to=TensorProto.INT64)
    core = b.node("OneHot", [code_i, depth, onehot_values], "core", axis=1)
    b.nodes.append(
        helper.make_node(
            "Pad",
            [core],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - GH, W - GW],
        )
    )
    return make_model(b.nodes, b.inits)


def build_conv_code_onehot_3d() -> onnx.ModelProto:
    b = Builder()
    axes_sel = b.i64("axes_sel", [1, 2, 3])
    axes_sp = b.i64("axes_sp", [2, 3])
    steps_sel = b.i64("steps_sel", [1, 4, 4])
    steps_sp = b.i64("steps_sp", [4, 4])
    flat_shape = b.i64("flat_shape", [1, 1, 1, 9])
    sel_shape = b.i64("sel_shape", [1, 1, 9, 1])
    patch_shape = b.i64("patch_shape", [1, 3, 3])
    depth = b.init("depth", np.asarray(10, dtype=np.int64))
    onehot_values = b.init("onehot_values", np.asarray([0.0, 1.0], dtype=np.float32))
    gray_col = b.init("gray_col_code", np.full((1, 3, 1), 5.0, dtype=np.float32))
    gray_row = b.init("gray_row_code", np.full((1, 1, GW), 5.0, dtype=np.float32))
    weight = b.init("code_weight", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))

    code30 = b.node("Conv", [IN_NAME, weight], "code30", kernel_shape=[1, 1])
    local_cells: list[str] = []
    for rr in range(3):
        for cc in range(3):
            st = b.i64(f"d_st_{rr}{cc}", [rr, cc])
            en = b.i64(f"d_en_{rr}{cc}", [rr + 9, cc + 9])
            sampled = b.node("Slice", [code30, st, en, axes_sp, steps_sp], f"d_s_{rr}{cc}")
            local_cells.append(b.node("Reshape", [sampled, flat_shape], f"d_f_{rr}{cc}"))
    data = b.node("Concat", local_cells, "data_code", axis=2)

    patches: list[str] = []
    for rr in range(3):
        for cc in range(3):
            st = b.i64(f"s_st_{rr}{cc}", [4, rr, cc])
            en = b.i64(f"s_en_{rr}{cc}", [5, rr + 9, cc + 9])
            sel = b.node("Slice", [IN_NAME, st, en, axes_sel, steps_sel], f"sel_{rr}{cc}")
            sel_flat = b.node("Reshape", [sel, sel_shape], f"sf_{rr}{cc}")
            picked = b.node("MatMul", [data, sel_flat], f"pick_{rr}{cc}")
            patches.append(b.node("Reshape", [picked, patch_shape], f"patch_{rr}{cc}"))

    rows: list[str] = []
    for rr in range(3):
        row_parts = [patches[rr * 3], gray_col, patches[rr * 3 + 1], gray_col, patches[rr * 3 + 2]]
        rows.append(b.node("Concat", row_parts, f"row_{rr}", axis=2))
    code = b.node("Concat", [rows[0], gray_row, rows[1], gray_row, rows[2]], "code", axis=1)
    code_i = b.node("Cast", [code], "code_i", to=TensorProto.INT64)
    core = b.node("OneHot", [code_i, depth, onehot_values], "core", axis=1)
    b.nodes.append(
        helper.make_node(
            "Pad",
            [core],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - GH, W - GW],
        )
    )
    return make_model(b.nodes, b.inits)


def build_sliced_conv_onehot_3d() -> onnx.ModelProto:
    b = Builder()
    axes_sel = b.i64("axes_sel", [1, 2, 3])
    axes_sp = b.i64("axes_sp", [2, 3])
    steps_sel = b.i64("steps_sel", [1, 4, 4])
    steps_sp = b.i64("steps_sp", [4, 4])
    flat_shape = b.i64("flat_shape", [1, 1, 1, 9])
    sel_shape = b.i64("sel_shape", [1, 1, 9, 1])
    patch_shape = b.i64("patch_shape", [1, 3, 3])
    depth = b.init("depth", np.asarray(10, dtype=np.int64))
    onehot_values = b.init("onehot_values", np.asarray([0.0, 1.0], dtype=np.float32))
    gray_col = b.init("gray_col_code", np.full((1, 3, 1), 5.0, dtype=np.float32))
    gray_row = b.init("gray_row_code", np.full((1, 1, GW), 5.0, dtype=np.float32))
    weight = b.init("code_weight", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))

    local_cells: list[str] = []
    for rr in range(3):
        for cc in range(3):
            st = b.i64(f"d_st_{rr}{cc}", [rr, cc])
            en = b.i64(f"d_en_{rr}{cc}", [rr + 9, cc + 9])
            sampled = b.node("Slice", [IN_NAME, st, en, axes_sp, steps_sp], f"d_s_{rr}{cc}")
            coded = b.node("Conv", [sampled, weight], f"d_code_{rr}{cc}", kernel_shape=[1, 1])
            local_cells.append(b.node("Reshape", [coded, flat_shape], f"d_f_{rr}{cc}"))
    data = b.node("Concat", local_cells, "data_code", axis=2)

    patches: list[str] = []
    for rr in range(3):
        for cc in range(3):
            st = b.i64(f"s_st_{rr}{cc}", [4, rr, cc])
            en = b.i64(f"s_en_{rr}{cc}", [5, rr + 9, cc + 9])
            sel = b.node("Slice", [IN_NAME, st, en, axes_sel, steps_sel], f"sel_{rr}{cc}")
            sel_flat = b.node("Reshape", [sel, sel_shape], f"sf_{rr}{cc}")
            picked = b.node("MatMul", [data, sel_flat], f"pick_{rr}{cc}")
            patches.append(b.node("Reshape", [picked, patch_shape], f"patch_{rr}{cc}"))

    rows: list[str] = []
    for rr in range(3):
        row_parts = [patches[rr * 3], gray_col, patches[rr * 3 + 1], gray_col, patches[rr * 3 + 2]]
        rows.append(b.node("Concat", row_parts, f"row_{rr}", axis=2))
    code = b.node("Concat", [rows[0], gray_row, rows[1], gray_row, rows[2]], "code", axis=1)
    code_i = b.node("Cast", [code], "code_i", to=TensorProto.INT64)
    core = b.node("OneHot", [code_i, depth, onehot_values], "core", axis=1)
    b.nodes.append(
        helper.make_node(
            "Pad",
            [core],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - GH, W - GW],
        )
    )
    return make_model(b.nodes, b.inits)


def build_nonbg_compact() -> onnx.ModelProto:
    b = Builder()
    axes = b.i64("axes", [1, 2, 3])
    steps = b.i64("steps", [1, 4, 4])
    flat_shape = b.i64("flat_shape", [1, NC, 1, 9])
    sel_shape = b.i64("sel_shape", [1, 1, 1, 9])
    patch_shape = b.i64("patch_shape", [1, NC, 3, 3])
    zero = b.f32("zero", [0.5])
    gray_col = b.init("gray_col", gray_band((1, NC, 3, 1)))
    gray_row = b.init("gray_row", gray_band((1, NC, 1, GW)))

    local_cells: list[str] = []
    for rr in range(3):
        for cc in range(3):
            st = b.i64(f"d_st_{rr}{cc}", [1, rr, cc])
            en = b.i64(f"d_en_{rr}{cc}", [10, rr + 9, cc + 9])
            sampled = b.node("Slice", [IN_NAME, st, en, axes, steps], f"d_s_{rr}{cc}")
            local_cells.append(b.node("Reshape", [sampled, flat_shape], f"d_f_{rr}{cc}"))
    data = b.node("Concat", local_cells, "data", axis=2)

    patches: list[str] = []
    for rr in range(3):
        for cc in range(3):
            st = b.i64(f"s_st_{rr}{cc}", [4, rr, cc])
            en = b.i64(f"s_en_{rr}{cc}", [5, rr + 9, cc + 9])
            sel = b.node("Slice", [IN_NAME, st, en, axes, steps], f"sel_{rr}{cc}")
            sel_flat = b.node("Reshape", [sel, sel_shape], f"sf_{rr}{cc}")
            picked = b.node("Mul", [data, sel_flat], f"pick_{rr}{cc}")
            reduced = b.node("ReduceSum", [picked], f"red_{rr}{cc}", axes=[3], keepdims=1)
            patches.append(b.node("Reshape", [reduced, patch_shape], f"patch_{rr}{cc}"))

    rows: list[str] = []
    for rr in range(3):
        row_parts = [patches[rr * 3], gray_col, patches[rr * 3 + 1], gray_col, patches[rr * 3 + 2]]
        rows.append(b.node("Concat", row_parts, f"row_{rr}", axis=3))
    nonbg = b.node("Concat", [rows[0], gray_row, rows[1], gray_row, rows[2]], "nonbg", axis=2)
    occ = b.node("ReduceSum", [nonbg], "occ", axes=[1], keepdims=1)
    bg_b = b.node("Less", [occ, zero], "bg_b")
    bg = b.node("Cast", [bg_b], "bg", to=TensorProto.FLOAT)
    core = b.node("Concat", [bg, nonbg], "core", axis=1)
    b.nodes.append(
        helper.make_node(
            "Pad",
            [core],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - GH, W - GW],
        )
    )
    return make_model(b.nodes, b.inits)


def build_nonbg_matmul() -> onnx.ModelProto:
    b = Builder()
    axes = b.i64("axes", [1, 2, 3])
    steps = b.i64("steps", [1, 4, 4])
    flat_shape = b.i64("flat_shape", [1, NC, 1, 9])
    sel_shape = b.i64("sel_shape", [1, 1, 9, 1])
    patch_shape = b.i64("patch_shape", [1, NC, 3, 3])
    zero = b.f32("zero", [0.5])
    gray_col = b.init("gray_col", gray_band((1, NC, 3, 1)))
    gray_row = b.init("gray_row", gray_band((1, NC, 1, GW)))

    local_cells: list[str] = []
    for rr in range(3):
        for cc in range(3):
            st = b.i64(f"d_st_{rr}{cc}", [1, rr, cc])
            en = b.i64(f"d_en_{rr}{cc}", [10, rr + 9, cc + 9])
            sampled = b.node("Slice", [IN_NAME, st, en, axes, steps], f"d_s_{rr}{cc}")
            local_cells.append(b.node("Reshape", [sampled, flat_shape], f"d_f_{rr}{cc}"))
    data = b.node("Concat", local_cells, "data", axis=2)

    patches: list[str] = []
    for rr in range(3):
        for cc in range(3):
            st = b.i64(f"s_st_{rr}{cc}", [4, rr, cc])
            en = b.i64(f"s_en_{rr}{cc}", [5, rr + 9, cc + 9])
            sel = b.node("Slice", [IN_NAME, st, en, axes, steps], f"sel_{rr}{cc}")
            sel_flat = b.node("Reshape", [sel, sel_shape], f"sf_{rr}{cc}")
            picked = b.node("MatMul", [data, sel_flat], f"pick_{rr}{cc}")
            patches.append(b.node("Reshape", [picked, patch_shape], f"patch_{rr}{cc}"))

    rows: list[str] = []
    for rr in range(3):
        row_parts = [patches[rr * 3], gray_col, patches[rr * 3 + 1], gray_col, patches[rr * 3 + 2]]
        rows.append(b.node("Concat", row_parts, f"row_{rr}", axis=3))
    nonbg = b.node("Concat", [rows[0], gray_row, rows[1], gray_row, rows[2]], "nonbg", axis=2)
    occ = b.node("ReduceSum", [nonbg], "occ", axes=[1], keepdims=1)
    bg_b = b.node("Less", [occ, zero], "bg_b")
    bg = b.node("Cast", [bg_b], "bg", to=TensorProto.FLOAT)
    core = b.node("Concat", [bg, nonbg], "core", axis=1)
    b.nodes.append(
        helper.make_node(
            "Pad",
            [core],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - GH, W - GW],
        )
    )
    return make_model(b.nodes, b.inits)


def variants() -> list[Variant]:
    return [
        Variant("sliced_conv_onehot_3d", build_sliced_conv_onehot_3d),
        Variant("conv_code_onehot_3d", build_conv_code_onehot_3d),
        Variant("conv_code_onehot", build_conv_code_onehot),
        Variant("conv_code_matmul", build_conv_code_matmul),
        Variant("color_code_matmul", build_color_code_matmul),
        Variant("nonbg_matmul", build_nonbg_matmul),
        Variant("nonbg_compact", build_nonbg_compact),
    ]


def load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def verify_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    all_ok = True
    splits: dict[str, tuple[int, int]] = {}
    for split in ("train", "test", "arc-gen"):
        passed = 0
        checked = 0
        for example in load_task_data().get(split, []):
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


def benchmark() -> tuple[str, dict[str, dict[str, Any]], dict[str, onnx.ModelProto]]:
    built = {variant.name: variant.build() for variant in variants()}
    results: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}_") as tmp:
        path = Path(tmp) / f"{TASK_ID}.onnx"
        for name, model in built.items():
            ok, splits = verify_correct(model)
            write_model(model, path)
            result = score_file(path)
            result["correct"] = ok
            result["splits"] = splits
            results[name] = result

    def key(item: tuple[str, dict[str, Any]]) -> int:
        result = item[1]
        if not result["valid"] or not result["correct"]:
            return 10**18
        return int(result["cost"])

    best_name, best_result = min(results.items(), key=key)
    if key((best_name, best_result)) >= 10**18:
        raise RuntimeError(f"no valid correct variant: {results}")
    return best_name, results, built


def print_results(best_name: str, results: dict[str, dict[str, Any]]) -> None:
    print(f"{'variant':<18} {'ok':<5} {'valid':<6} {'memory':>8} {'params':>7} {'cost':>8} {'score':>10}")
    for name, result in sorted(results.items(), key=lambda item: (item[0] != best_name, item[0])):
        score = result["score"]
        score_text = f"{score:.6f}" if isinstance(score, float) else "None"
        print(
            f"{name:<18} {str(result['correct']):<5} {str(result['valid']):<6} "
            f"{str(result['memory']):>8} {str(result['params']):>7} "
            f"{str(result['cost']):>8} {score_text:>10}"
        )


def main() -> None:
    best_name, results, built = benchmark()
    print_results(best_name, results)
    write_model(built[best_name], BEST_PATH)
    shutil.copyfile(BEST_PATH, ROOT_SOLUTION_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"wrote {ROOT_SOLUTION_PATH}")


if __name__ == "__main__":
    main()
