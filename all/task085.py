"""Compact ONNX for NeuroGolf task085: stripe rectangle middle rows.

Task rule: the grid contains solid horizontal rectangles on a black
background. Each rectangle is exactly three rows high. Preserve the top and
bottom row of every rectangle, and in the middle row erase every second cell
to black, starting locally from the rectangle's left edge: keep offset 0,
erase offset 1, keep offset 2, and so on. Background and all other cells stay
unchanged.

ONNX approach: compute a compact int32 color grid with ArgMax, identify true
middle rows by matching the same non-zero color above and below, derive each
row's local run-start parity, then use a broadcast Where to select a tiny
black one-hot constant at erased cells while passing the original input through
everywhere else. The task data only uses the top 16 rows, so internal stripe
logic is cropped to 16x30 and padded back to the required 30x30 output mask.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402


TASK_NUM = "085"
TASK_NAME = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_NAME}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_NAME}.onnx"


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.value_infos: list[onnx.ValueInfoProto] = []

    def init(self, name: str, array: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(array, name))
        return name

    def vi(self, name: str, dtype: int, shape: tuple[int, ...]) -> str:
        self.value_infos.append(helper.make_tensor_value_info(name, dtype, list(shape)))
        return name

    def node(
        self,
        op_type: str,
        inputs: list[str],
        output: str,
        dtype: int,
        shape: tuple[int, ...],
        **attrs: Any,
    ) -> str:
        self.vi(output, dtype, shape)
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        return output


def add_constants(b: Builder) -> None:
    for value in (0, 1, 2, 14, 15, 16, 29):
        b.init(f"i{value}", np.array([value], dtype=np.int64))
    b.init("zero_i32", np.array(0, dtype=np.int32))
    b.init("zero_f", np.array(0.0, dtype=np.float32))
    b.init("axes1", np.array([1], dtype=np.int64))
    b.init("axes2", np.array([2], dtype=np.int64))
    b.init("odd_idx", np.arange(1, 30, 2, dtype=np.int64))
    b.init("false_col14", np.zeros((1, 14, 1), dtype=np.bool_))
    b.init("false_row", np.zeros((1, 1, 30), dtype=np.bool_))
    b.init("odd_cols", (np.arange(30) % 2 == 1).reshape(1, 1, 30))
    b.init("black", np.array([1.0] + [0.0] * 9, dtype=np.float32).reshape(1, 10, 1, 1))


def slice_axis(
    b: Builder,
    x: str,
    start: str,
    end: str,
    axis_name: str,
    output: str,
    dtype: int,
    shape: tuple[int, ...],
) -> str:
    return b.node("Slice", [x, start, end, axis_name], output, dtype, shape)


def build_model() -> onnx.ModelProto:
    b = Builder()
    add_constants(b)

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])

    colors_i = b.node("ArgMax", ["input"], "colors_i", TensorProto.INT64, (1, 30, 30), axis=1, keepdims=0)
    colors = b.node("Cast", [colors_i], "colors", TensorProto.INT32, (1, 30, 30), to=TensorProto.INT32)

    up = slice_axis(b, "colors", "i0", "i14", "axes1", "up", TensorProto.INT32, (1, 14, 30))
    cur = slice_axis(b, "colors", "i1", "i15", "axes1", "cur", TensorProto.INT32, (1, 14, 30))
    down = slice_axis(b, "colors", "i2", "i16", "axes1", "down", TensorProto.INT32, (1, 14, 30))
    eq_up = b.node("Equal", [up, cur], "eq_up", TensorProto.BOOL, (1, 14, 30))
    eq_down = b.node("Equal", [cur, down], "eq_down", TensorProto.BOOL, (1, 14, 30))
    mid_same = b.node("And", [eq_up, eq_down], "mid_same", TensorProto.BOOL, (1, 14, 30))
    cur_colored = b.node("Greater", [cur, "zero_i32"], "cur_colored", TensorProto.BOOL, (1, 14, 30))
    mid_core = b.node("And", [mid_same, cur_colored], "mid_core", TensorProto.BOOL, (1, 14, 30))

    prev_core = slice_axis(b, "cur_colored", "i0", "i29", "axes2", "prev_core", TensorProto.BOOL, (1, 14, 29))
    prev = b.node("Concat", ["false_col14", prev_core], "prev", TensorProto.BOOL, (1, 14, 30), axis=2)
    not_prev = b.node("Not", [prev], "not_prev", TensorProto.BOOL, (1, 14, 30))
    starts = b.node("And", [cur_colored, not_prev], "starts", TensorProto.BOOL, (1, 14, 30))
    odd_starts = b.node("Gather", [starts, "odd_idx"], "odd_starts", TensorProto.BOOL, (1, 14, 15), axis=2)
    odd_starts_f = b.node("Cast", [odd_starts], "odd_starts_f", TensorProto.FLOAT, (1, 14, 15), to=TensorProto.FLOAT)
    start_odd_f = b.node("ReduceMax", [odd_starts_f], "start_odd_f", TensorProto.FLOAT, (1, 14, 1), axes=[2], keepdims=1)
    start_odd = b.node("Greater", [start_odd_f, "zero_f"], "start_odd", TensorProto.BOOL, (1, 14, 1))
    local_odd = b.node("Xor", ["odd_cols", start_odd], "local_odd", TensorProto.BOOL, (1, 14, 30))
    erase_core = b.node("And", [mid_core, local_odd], "erase_core", TensorProto.BOOL, (1, 14, 30))
    erase = b.node(
        "Concat",
        ["false_row", "erase_core"] + ["false_row"] * 15,
        "erase",
        TensorProto.BOOL,
        (1, 30, 30),
        axis=1,
    )
    b.nodes.append(helper.make_node("Where", [erase, "black", "input"], ["output"]))

    graph = helper.make_graph(b.nodes, TASK_NAME, [inp], [out], b.initializers, value_info=b.value_infos)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 10)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def reference_grid(grid: list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = arr.copy()
    h, w = arr.shape
    for r in range(1, h - 1):
        c = 0
        while c < w:
            color = arr[r, c]
            same_vertical = color != 0 and arr[r - 1, c] == color and arr[r + 1, c] == color
            if not same_vertical:
                c += 1
                continue
            start = c
            while c < w and arr[r, c] == color and arr[r - 1, c] == color and arr[r + 1, c] == color:
                c += 1
            out[r, start + 1 : c : 2] = 0
    return out


def decode_one_hot(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    active = arr[0, :, :h, :w] > 0.0
    bad = active.sum(axis=0) != 1
    decoded = active.argmax(axis=0).astype(np.int64)
    if bad.any():
        decoded[bad] = -1
    return decoded


def validate_model(path: Path) -> None:
    with TASK_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            expected = np.asarray(example["output"], dtype=np.int64)
            ref = reference_grid(example["input"])
            if not np.array_equal(ref, expected):
                raise AssertionError(f"reference mismatch for {split}[{idx}]")
            arr = convert_to_numpy(example, "input")
            if arr is None:
                continue
            pred = session.run(["output"], {"input": arr})[0]
            got = decode_one_hot(pred, expected.shape[0], expected.shape[1])
            if not np.array_equal(got, expected):
                mismatch = np.argwhere(got != expected)[0].tolist()
                raise AssertionError(f"ONNX mismatch for {split}[{idx}] at {mismatch}")
            total += 1
    print(f"validated examples: {total}")


def inferred_internal_memory(model: onnx.ModelProto) -> int:
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    init_names = {init.name for init in graph.initializer}
    total = 0
    for value_info in graph.value_info:
        if value_info.name in init_names:
            continue
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        elems = 1
        for dim in tensor_type.shape.dim:
            elems *= dim.dim_value
        dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        total += elems * np.dtype(dtype).itemsize
    return int(total)


def params_count(model: onnx.ModelProto) -> int:
    params = 0
    for init in model.graph.initializer:
        params += math.prod(init.dims)
    return int(params)


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    print(f"wrote: {BEST_PATH}")
    print(f"params estimate: {params_count(model)}")
    print(f"shape-inferred internal memory estimate: {inferred_internal_memory(model)}")
    validate_model(BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"official memory: {result['memory']}")
    print(f"official params: {result['params']}")
    print(f"official cost: {result['cost']}")
    print(f"official score: {result['score']}")
    if not result["valid"]:
        raise SystemExit(result["error"])


if __name__ == "__main__":
    main()
