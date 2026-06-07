"""Build optimized ONNX solvers for NeuroGolf task305.

Task rule: each 16x16 grid is a diagonal cyclic texture with black holes.
The completed output has color ((row + col) mod P) + 1 at every 16x16 cell,
where P is the largest nonzero color present in the input (4, 5, 6, 7, or 8).
Cells outside the 16x16 task grid remain all-zero padding in the 30x30
NeuroGolf tensor.
"""

from __future__ import annotations

import copy
import json
import math
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import (
    calculate_memory,
    calculate_params,
    convert_to_numpy,
    sanitize_model,
    score,
)

TASK_ID = "task305"
TASK_NUM = 305
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
OPSET = 10
IR_VERSION = 10
SHAPE = [1, 10, 30, 30]
CORE = 16


def _load_examples() -> list[dict[str, Any]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    examples: list[dict[str, Any]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            examples.append({"split": split, "idx": idx, "example": example})
    return examples


def _expected_tensor(example: dict[str, list[list[int]]]) -> np.ndarray:
    return convert_to_numpy(example, "output")


def _i64(inits: list[onnx.TensorProto], name: str, vals: list[int]) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))
    return name


def _mask_for_period(period: int, *, full: bool, dtype: np.dtype[Any]) -> np.ndarray:
    h = w = 30 if full else CORE
    arr = np.zeros((1, 10, h, w), dtype=dtype)
    for r in range(CORE):
        for c in range(CORE):
            color = ((r + c) % period) + 1
            arr[0, color, r, c] = 1
    return arr


def _make_model(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    value_infos: list[onnx.ValueInfoProto],
    name: str,
) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)],
        initializer=inits,
        value_info=value_infos,
    )
    model = helper.make_model(
        graph,
        producer_name="task305",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _period_flags(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    periods: range = range(5, 9),
) -> dict[int, str]:
    slice_axis = _i64(inits, "axis_channel_sum", [0])
    threshold = "zero_f"
    inits.append(numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), name=threshold))
    nodes.append(helper.make_node("ReduceSum", ["input"], ["channel_counts"], axes=[0, 2, 3], keepdims=0))
    flags: dict[int, str] = {}
    for period in periods:
        starts = _i64(inits, f"ch{period}_starts", [period])
        ends = _i64(inits, f"ch{period}_ends", [period + 1])
        nodes.extend(
            [
                helper.make_node("Slice", ["channel_counts", starts, ends, slice_axis], [f"sum{period}"]),
                helper.make_node("Greater", [f"sum{period}", threshold], [f"has{period}"]),
            ]
        )
        flags[period] = f"has{period}"
    return flags


def _period_from_gathered_flags(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto]) -> str:
    inits.extend(
        [
            numpy_helper.from_array(np.asarray([5, 6, 7, 8], dtype=np.int64), name="period_indices"),
            numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), name="zero_f"),
            numpy_helper.from_array(np.asarray(4, dtype=np.int32), name="four"),
        ]
    )
    nodes.extend(
        [
            helper.make_node("ReduceSum", ["input"], ["channel_counts"], axes=[0, 2, 3], keepdims=0),
            helper.make_node("Gather", ["channel_counts", "period_indices"], ["period_counts"], axis=0),
            helper.make_node("Greater", ["period_counts", "zero_f"], ["period_flags"]),
            helper.make_node("Cast", ["period_flags"], ["period_increments"], to=TensorProto.INT32),
            helper.make_node("ReduceSum", ["period_increments"], ["period_delta"], axes=[0], keepdims=1),
            helper.make_node("Add", ["four", "period_delta"], ["period"]),
        ]
    )
    return "period"


def build_mod_formula() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    flags = _period_flags(nodes, inits)
    diag = np.fromfunction(lambda r, c: r + c, (CORE, CORE), dtype=int).astype(np.int32)
    channels = np.arange(10, dtype=np.int32).reshape(10, 1, 1)
    inits.extend(
        [
            numpy_helper.from_array(diag, "diag"),
            numpy_helper.from_array(channels, "channels"),
            numpy_helper.from_array(np.asarray(4, dtype=np.int32), "four"),
            numpy_helper.from_array(np.asarray(1, dtype=np.int32), "one_i32"),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Cast", [flags[5]], ["p5_i"], to=TensorProto.INT32),
            helper.make_node("Cast", [flags[6]], ["p6_i"], to=TensorProto.INT32),
            helper.make_node("Cast", [flags[7]], ["p7_i"], to=TensorProto.INT32),
            helper.make_node("Cast", [flags[8]], ["p8_i"], to=TensorProto.INT32),
            helper.make_node("Add", ["four", "p5_i"], ["p5"]),
            helper.make_node("Add", ["p5", "p6_i"], ["p6"]),
            helper.make_node("Add", ["p6", "p7_i"], ["p7"]),
            helper.make_node("Add", ["p7", "p8_i"], ["period"]),
            helper.make_node("Mod", ["diag", "period"], ["mod"]),
            helper.make_node("Add", ["mod", "one_i32"], ["colors"]),
            helper.make_node("Equal", ["channels", "colors"], ["core_bool_3d"]),
            helper.make_node("Cast", ["core_bool_3d"], ["core_f_3d"], to=TensorProto.FLOAT),
            helper.make_node("Unsqueeze", ["core_f_3d"], ["core_f"], axes=[0]),
            helper.make_node(
                "Pad",
                ["core_f"],
                ["output"],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, 30 - CORE, 30 - CORE],
            ),
        ]
    )
    return _make_model(nodes, inits, [], "task305_mod_formula")


def build_mod_formula_cast_after_unsqueeze() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    flags = _period_flags(nodes, inits)
    diag = np.fromfunction(lambda r, c: r + c, (CORE, CORE), dtype=int).astype(np.int32)
    channels = np.arange(10, dtype=np.int32).reshape(10, 1, 1)
    inits.extend(
        [
            numpy_helper.from_array(diag, "diag"),
            numpy_helper.from_array(channels, "channels"),
            numpy_helper.from_array(np.asarray(4, dtype=np.int32), "four"),
            numpy_helper.from_array(np.asarray(1, dtype=np.int32), "one_i32"),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Cast", [flags[5]], ["p5_i"], to=TensorProto.INT32),
            helper.make_node("Cast", [flags[6]], ["p6_i"], to=TensorProto.INT32),
            helper.make_node("Cast", [flags[7]], ["p7_i"], to=TensorProto.INT32),
            helper.make_node("Cast", [flags[8]], ["p8_i"], to=TensorProto.INT32),
            helper.make_node("Add", ["four", "p5_i"], ["p5"]),
            helper.make_node("Add", ["p5", "p6_i"], ["p6"]),
            helper.make_node("Add", ["p6", "p7_i"], ["p7"]),
            helper.make_node("Add", ["p7", "p8_i"], ["period"]),
            helper.make_node("Mod", ["diag", "period"], ["mod"]),
            helper.make_node("Add", ["mod", "one_i32"], ["colors"]),
            helper.make_node("Equal", ["channels", "colors"], ["core_bool_3d"]),
            helper.make_node("Unsqueeze", ["core_bool_3d"], ["core_bool"], axes=[0]),
            helper.make_node("Cast", ["core_bool"], ["core_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["core_f"],
                ["output"],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, 30 - CORE, 30 - CORE],
            ),
        ]
    )
    return _make_model(nodes, inits, [], "task305_mod_formula_cast_after_unsqueeze")


def build_mod_formula_4d_equal() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    flags = _period_flags(nodes, inits)
    diag = np.fromfunction(lambda r, c: r + c, (CORE, CORE), dtype=int).astype(np.int32).reshape(1, 1, CORE, CORE)
    channels = np.arange(10, dtype=np.int32).reshape(1, 10, 1, 1)
    inits.extend(
        [
            numpy_helper.from_array(diag, "diag"),
            numpy_helper.from_array(channels, "channels"),
            numpy_helper.from_array(np.asarray(4, dtype=np.int32), "four"),
            numpy_helper.from_array(np.asarray(1, dtype=np.int32), "one_i32"),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Cast", [flags[5]], ["p5_i"], to=TensorProto.INT32),
            helper.make_node("Cast", [flags[6]], ["p6_i"], to=TensorProto.INT32),
            helper.make_node("Cast", [flags[7]], ["p7_i"], to=TensorProto.INT32),
            helper.make_node("Cast", [flags[8]], ["p8_i"], to=TensorProto.INT32),
            helper.make_node("Add", ["four", "p5_i"], ["p5"]),
            helper.make_node("Add", ["p5", "p6_i"], ["p6"]),
            helper.make_node("Add", ["p6", "p7_i"], ["p7"]),
            helper.make_node("Add", ["p7", "p8_i"], ["period"]),
            helper.make_node("Mod", ["diag", "period"], ["mod"]),
            helper.make_node("Add", ["mod", "one_i32"], ["colors"]),
            helper.make_node("Equal", ["channels", "colors"], ["core_bool"]),
            helper.make_node("Cast", ["core_bool"], ["core_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["core_f"],
                ["output"],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, 30 - CORE, 30 - CORE],
            ),
        ]
    )
    return _make_model(nodes, inits, [], "task305_mod_formula_4d_equal")


def build_mod_formula_no_color_add() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    flags = _period_flags(nodes, inits)
    diag = np.fromfunction(lambda r, c: r + c, (CORE, CORE), dtype=int).astype(np.int32).reshape(1, 1, CORE, CORE)
    channel_mod = (np.arange(10, dtype=np.int32) - 1).reshape(1, 10, 1, 1)
    inits.extend(
        [
            numpy_helper.from_array(diag, "diag"),
            numpy_helper.from_array(channel_mod, "channel_mod"),
            numpy_helper.from_array(np.asarray(4, dtype=np.int32), "four"),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Cast", [flags[5]], ["p5_i"], to=TensorProto.INT32),
            helper.make_node("Cast", [flags[6]], ["p6_i"], to=TensorProto.INT32),
            helper.make_node("Cast", [flags[7]], ["p7_i"], to=TensorProto.INT32),
            helper.make_node("Cast", [flags[8]], ["p8_i"], to=TensorProto.INT32),
            helper.make_node("Add", ["four", "p5_i"], ["p5"]),
            helper.make_node("Add", ["p5", "p6_i"], ["p6"]),
            helper.make_node("Add", ["p6", "p7_i"], ["p7"]),
            helper.make_node("Add", ["p7", "p8_i"], ["period"]),
            helper.make_node("Mod", ["diag", "period"], ["mod"]),
            helper.make_node("Equal", ["channel_mod", "mod"], ["core_bool"]),
            helper.make_node("Cast", ["core_bool"], ["core_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["core_f"],
                ["output"],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, 30 - CORE, 30 - CORE],
            ),
        ]
    )
    return _make_model(nodes, inits, [], "task305_mod_formula_no_color_add")


def build_mod_formula_8_color_channels() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    flags = _period_flags(nodes, inits)
    diag = np.fromfunction(lambda r, c: r + c, (CORE, CORE), dtype=int).astype(np.int32).reshape(1, 1, CORE, CORE)
    channel_mod = np.arange(8, dtype=np.int32).reshape(1, 8, 1, 1)
    inits.extend(
        [
            numpy_helper.from_array(diag, "diag"),
            numpy_helper.from_array(channel_mod, "channel_mod"),
            numpy_helper.from_array(np.asarray(4, dtype=np.int32), "four"),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Cast", [flags[5]], ["p5_i"], to=TensorProto.INT32),
            helper.make_node("Cast", [flags[6]], ["p6_i"], to=TensorProto.INT32),
            helper.make_node("Cast", [flags[7]], ["p7_i"], to=TensorProto.INT32),
            helper.make_node("Cast", ["has8"], ["p8_i"], to=TensorProto.INT32),
            helper.make_node("Add", ["four", "p5_i"], ["p5"]),
            helper.make_node("Add", ["p5", "p6_i"], ["p6"]),
            helper.make_node("Add", ["p6", "p7_i"], ["p7"]),
            helper.make_node("Add", ["p7", "p8_i"], ["period"]),
            helper.make_node("Mod", ["diag", "period"], ["mod"]),
            helper.make_node("Equal", ["channel_mod", "mod"], ["core_bool"]),
            helper.make_node("Cast", ["core_bool"], ["core_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["core_f"],
                ["output"],
                mode="constant",
                pads=[0, 1, 0, 0, 0, 1, 30 - CORE, 30 - CORE],
            ),
        ]
    )
    return _make_model(nodes, inits, [], "task305_mod_formula_8_color_channels")


def build_mod_formula_8_color_channels_gather_period() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    period = _period_from_gathered_flags(nodes, inits)
    diag = np.fromfunction(lambda r, c: r + c, (CORE, CORE), dtype=int).astype(np.int32).reshape(1, 1, CORE, CORE)
    channel_mod = np.arange(8, dtype=np.int32).reshape(1, 8, 1, 1)
    inits.extend(
        [
            numpy_helper.from_array(diag, "diag"),
            numpy_helper.from_array(channel_mod, "channel_mod"),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Mod", ["diag", period], ["mod"]),
            helper.make_node("Equal", ["channel_mod", "mod"], ["core_bool"]),
            helper.make_node("Cast", ["core_bool"], ["core_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["core_f"],
                ["output"],
                mode="constant",
                pads=[0, 1, 0, 0, 0, 1, 30 - CORE, 30 - CORE],
            ),
        ]
    )
    return _make_model(nodes, inits, [], "task305_mod_formula_8_color_channels_gather_period")


def build_onehot_8_color_channels() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    period = _period_from_gathered_flags(nodes, inits)
    diag = np.fromfunction(lambda _b, r, c: r + c, (1, CORE, CORE), dtype=int).astype(np.int32)
    inits.extend(
        [
            numpy_helper.from_array(diag, "diag"),
            numpy_helper.from_array(np.asarray(8, dtype=np.int32), "onehot_depth"),
            numpy_helper.from_array(np.asarray([0.0, 1.0], dtype=np.float32), "onehot_values"),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Mod", ["diag", period], ["color_indices"]),
            helper.make_node(
                "OneHot",
                ["color_indices", "onehot_depth", "onehot_values"],
                ["core_f"],
                axis=1,
            ),
            helper.make_node(
                "Pad",
                ["core_f"],
                ["output"],
                mode="constant",
                pads=[0, 1, 0, 0, 0, 1, 30 - CORE, 30 - CORE],
            ),
        ]
    )
    return _make_model(nodes, inits, [], "task305_onehot_8_color_channels")


def build_mod_formula_bool_pad() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    flags = _period_flags(nodes, inits)
    diag = np.fromfunction(lambda r, c: r + c, (CORE, CORE), dtype=int).astype(np.int64)
    channels = np.arange(10, dtype=np.int64).reshape(10, 1, 1)
    inits.extend(
        [
            numpy_helper.from_array(diag, "diag"),
            numpy_helper.from_array(channels, "channels"),
            numpy_helper.from_array(np.asarray(4, dtype=np.int64), "four"),
            numpy_helper.from_array(np.asarray(1, dtype=np.int64), "one_i64"),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Cast", [flags[5]], ["p5_i"], to=TensorProto.INT64),
            helper.make_node("Cast", [flags[6]], ["p6_i"], to=TensorProto.INT64),
            helper.make_node("Cast", [flags[7]], ["p7_i"], to=TensorProto.INT64),
            helper.make_node("Cast", [flags[8]], ["p8_i"], to=TensorProto.INT64),
            helper.make_node("Add", ["four", "p5_i"], ["p5"]),
            helper.make_node("Add", ["p5", "p6_i"], ["p6"]),
            helper.make_node("Add", ["p6", "p7_i"], ["p7"]),
            helper.make_node("Add", ["p7", "p8_i"], ["period"]),
            helper.make_node("Mod", ["diag", "period"], ["mod"]),
            helper.make_node("Add", ["mod", "one_i64"], ["colors"]),
            helper.make_node("Equal", ["channels", "colors"], ["core_bool_3d"]),
            helper.make_node("Unsqueeze", ["core_bool_3d"], ["core_bool"], axes=[0]),
            helper.make_node(
                "Pad",
                ["core_bool"],
                ["out_bool"],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, 30 - CORE, 30 - CORE],
            ),
            helper.make_node("Cast", ["out_bool"], ["output"], to=TensorProto.FLOAT),
        ]
    )
    return _make_model(nodes, inits, [], "task305_mod_formula_bool_pad")


def build_compact_bool_select() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    value_infos: list[onnx.ValueInfoProto] = []
    flags = _period_flags(nodes, inits)

    for period in range(4, 9):
        inits.append(
            numpy_helper.from_array(_mask_for_period(period, full=False, dtype=np.bool_), f"mask{period}")
        )
    nodes.extend(
        [
            helper.make_node("Where", [flags[5], "mask5", "mask4"], ["sel5"]),
            helper.make_node("Where", [flags[6], "mask6", "sel5"], ["sel6"]),
            helper.make_node("Where", [flags[7], "mask7", "sel6"], ["sel7"]),
            helper.make_node("Where", [flags[8], "mask8", "sel7"], ["core_bool"]),
            helper.make_node("Cast", ["core_bool"], ["core_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["core_f"],
                ["output"],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, 30 - CORE, 30 - CORE],
            ),
        ]
    )
    return _make_model(nodes, inits, value_infos, "task305_compact_bool_select")


def build_compact_float_select() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    flags = _period_flags(nodes, inits)
    for period in range(4, 9):
        inits.append(
            numpy_helper.from_array(_mask_for_period(period, full=False, dtype=np.float32), f"mask{period}")
        )
    nodes.extend(
        [
            helper.make_node("Where", [flags[5], "mask5", "mask4"], ["sel5"]),
            helper.make_node("Where", [flags[6], "mask6", "sel5"], ["sel6"]),
            helper.make_node("Where", [flags[7], "mask7", "sel6"], ["sel7"]),
            helper.make_node("Where", [flags[8], "mask8", "sel7"], ["core_f"]),
            helper.make_node(
                "Pad",
                ["core_f"],
                ["output"],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, 30 - CORE, 30 - CORE],
            ),
        ]
    )
    return _make_model(nodes, inits, [], "task305_compact_float_select")


def build_full_bool_select() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    flags = _period_flags(nodes, inits)
    for period in range(4, 9):
        inits.append(numpy_helper.from_array(_mask_for_period(period, full=True, dtype=np.bool_), f"mask{period}"))
    nodes.extend(
        [
            helper.make_node("Where", [flags[5], "mask5", "mask4"], ["sel5"]),
            helper.make_node("Where", [flags[6], "mask6", "sel5"], ["sel6"]),
            helper.make_node("Where", [flags[7], "mask7", "sel6"], ["sel7"]),
            helper.make_node("Where", [flags[8], "mask8", "sel7"], ["out_bool"]),
            helper.make_node("Cast", ["out_bool"], ["output"], to=TensorProto.FLOAT),
        ]
    )
    return _make_model(nodes, inits, [], "task305_full_bool_select")


def build_full_float_select() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    flags = _period_flags(nodes, inits)
    for period in range(4, 9):
        inits.append(
            numpy_helper.from_array(_mask_for_period(period, full=True, dtype=np.float32), f"mask{period}")
        )
    nodes.extend(
        [
            helper.make_node("Where", [flags[5], "mask5", "mask4"], ["sel5"]),
            helper.make_node("Where", [flags[6], "mask6", "sel5"], ["sel6"]),
            helper.make_node("Where", [flags[7], "mask7", "sel6"], ["sel7"]),
            helper.make_node("Where", [flags[8], "mask8", "sel7"], ["output"]),
        ]
    )
    return _make_model(nodes, inits, [], "task305_full_float_select")


def _largest_tensor(model: onnx.ModelProto, trace_path: str) -> str:
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    tensor_map = {t.name: t for t in list(graph.input) + list(graph.value_info) + list(graph.output)}
    best = ("n/a", 0, "", "")
    for name, item in tensor_map.items():
        if name in {"input", "output"} or not item.type.HasField("tensor_type"):
            continue
        tt = item.type.tensor_type
        dims = [d.dim_value for d in tt.shape.dim]
        dtype = onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)
        bytes_ = int(math.prod(dims) * np.dtype(dtype).itemsize)
        if bytes_ > best[1]:
            best = (name, bytes_, str(np.dtype(dtype)), str(dims))
    with open(trace_path, encoding="utf-8") as fh:
        trace = json.load(fh)
    node_outputs = {node.name: list(node.output) for node in graph.node}
    for event in trace:
        if event.get("cat") != "Node" or "output_type_shape" not in event.get("args", {}):
            continue
        node_name = event.get("name", "").replace("_kernel_time", "")
        for idx, shape_dict in enumerate(event["args"]["output_type_shape"]):
            outputs = node_outputs.get(node_name, [])
            if idx >= len(outputs) or outputs[idx] in {"output"}:
                continue
            name = outputs[idx]
            item = tensor_map.get(name)
            if item is None or not item.type.HasField("tensor_type"):
                continue
            dtype = onnx.helper.tensor_dtype_to_np_dtype(item.type.tensor_type.elem_type)
            bytes_ = int(np.dtype(dtype).itemsize * sum(math.prod(dims) for dims in shape_dict.values()))
            if bytes_ > best[1]:
                best = (name, bytes_, str(np.dtype(dtype)), str(list(shape_dict.values())))
    return f"{best[0]} {best[1]} B {best[2]} {best[3]}"


def _run_profile(model: onnx.ModelProto, label: str, inputs: list[np.ndarray]) -> tuple[str | None, str | None]:
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(Path(tempfile.gettempdir()) / f"ng_{TASK_ID}_{label}")
    try:
        session = ort.InferenceSession(
            model.SerializeToString(),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        for arr in inputs:
            session.run(["output"], {"input": arr})
        return session.end_profiling(), None
    except Exception:
        return None, traceback.format_exc()


def evaluate(label: str, model: onnx.ModelProto, examples: list[dict[str, Any]]) -> dict[str, Any]:
    inputs = [convert_to_numpy(item["example"], "input") for item in examples]
    expected = [_expected_tensor(item["example"]) for item in examples]
    result: dict[str, Any] = {
        "label": label,
        "correct": False,
        "memory": None,
        "params": None,
        "cost": None,
        "score": None,
        "largest": "n/a",
        "error": "",
        "model": model,
    }
    try:
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        for item, arr, exp in zip(examples, inputs, expected, strict=True):
            pred = session.run(["output"], {"input": arr})[0]
            if not np.array_equal(pred > 0.0, exp > 0.0):
                result["error"] = f"wrong output on {item['split']}[{item['idx']}]"
                return result
        result["correct"] = True
    except Exception:
        result["error"] = traceback.format_exc()
        return result

    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        result["error"] = "sanitize failed"
        return result
    trace_path, error = _run_profile(sanitized, label, inputs)
    if trace_path is None:
        result["error"] = error or "profile failed"
        return result
    memory = calculate_memory(sanitized, trace_path)
    params = calculate_params(sanitized)
    if memory is None or params is None:
        result["error"] = "measurement failed"
        return result
    result["memory"] = memory
    result["params"] = params
    result["cost"] = memory + params
    result["score"] = score(memory + params)
    result["largest"] = _largest_tensor(sanitized, trace_path)
    return result


def print_rule_summary(examples: list[dict[str, Any]]) -> None:
    print("Inferred task305 rule:")
    print("  output[r,c] = ((r + c) mod P) + 1 on the 16x16 grid")
    print("  P = largest nonzero input color; zeros are missing cells to fill")
    for item in examples[:5]:
        grid = np.asarray(item["example"]["input"])
        out = np.asarray(item["example"]["output"])
        period = int(grid.max())
        predicted = np.fromfunction(lambda r, c: ((r + c) % period) + 1, out.shape, dtype=int)
        print(
            f"  {item['split']}[{item['idx']}]: shape={grid.shape}, "
            f"P={period}, rule_ok={bool(np.array_equal(predicted, out))}"
        )
    print(f"  checked examples: {len(examples)}")


def main() -> None:
    examples = _load_examples()
    print_rule_summary(examples)
    candidates = [
        ("onehot_8_color_channels", build_onehot_8_color_channels()),
        ("mod_formula_8_color_channels_gather_period", build_mod_formula_8_color_channels_gather_period()),
        ("mod_formula_8_color_channels", build_mod_formula_8_color_channels()),
        ("mod_formula_no_color_add", build_mod_formula_no_color_add()),
        ("mod_formula_4d_equal", build_mod_formula_4d_equal()),
        ("mod_formula_cast_after_unsqueeze", build_mod_formula_cast_after_unsqueeze()),
        ("mod_formula_bool_pad", build_mod_formula_bool_pad()),
        ("mod_formula", build_mod_formula()),
        ("compact_float_select", build_compact_float_select()),
        ("compact_bool_select", build_compact_bool_select()),
        ("full_bool_select", build_full_bool_select()),
        ("full_float_select", build_full_float_select()),
    ]
    results = [evaluate(label, model, examples) for label, model in candidates]
    results.sort(key=lambda r: (not r["correct"], r["cost"] if r["cost"] is not None else 10**18))

    print()
    print("Ranked candidates:")
    print(f"{'rank':>4}  {'variant':<22} {'ok':<3} {'memory':>8} {'params':>8} {'cost':>8} {'score':>9}  largest tensor")
    for rank, result in enumerate(results, 1):
        print(
            f"{rank:>4}  {result['label']:<22} {str(result['correct']):<3} "
            f"{str(result['memory']):>8} {str(result['params']):>8} "
            f"{str(result['cost']):>8} "
            f"{result['score'] if result['score'] is not None else 0.0:>9.6f}  {result['largest']}"
        )
        if result["error"]:
            print(f"      error: {str(result['error']).splitlines()[0]}")

    best = next((result for result in results if result["correct"] and result["cost"] is not None), None)
    if best is None:
        raise SystemExit("no valid correct candidate")
    onnx.save(best["model"], BEST_PATH)
    print()
    print(f"Saved best model: {BEST_PATH}")


if __name__ == "__main__":
    main()
