"""Minimal ONNX generator for NeuroGolf task140.

Task rule: every example is a 3x3 grid. The output is the same grid rotated
180 degrees, so output[r, c] = input[2-r, 2-c]. In the NeuroGolf one-hot
tensor this means only the top-left 3x3 visible region is populated; the rest
of the 30x30 tensor remains all-zero padding.

ONNX approach: Slice the compact 3x3 region with negative steps on the two
spatial axes, then Pad directly to the full competition output shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, shape_inference


TASK_ID = "task140"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task140.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
OPSET = 10
I64_MIN = -9223372036854775808


def init_i64(name: str, values: list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name)


def build_slice_model() -> onnx.ModelProto:
    initializers = [
        init_i64("starts", [2, 2]),
        init_i64("ends", [I64_MIN, I64_MIN]),
        init_i64("axes", [2, 3]),
        init_i64("steps", [-1, -1]),
    ]
    nodes = [
        helper.make_node(
            "Slice",
            [IN_NAME, "starts", "ends", "axes", "steps"],
            ["rot3"],
        ),
        helper.make_node(
            "Pad",
            ["rot3"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 27, 27],
            value=0.0,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf-task140-minimal",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def random_one_hot(seed: int = 140) -> np.ndarray:
    rng = np.random.default_rng(seed)
    colors = rng.integers(0, 10, size=(3, 3))
    x = np.zeros(FULL_SHAPE, dtype=np.float32)
    for r in range(3):
        for c in range(3):
            x[0, colors[r, c], r, c] = 1.0
    return x


def grid_to_one_hot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    x = np.zeros(FULL_SHAPE, dtype=np.float32)
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            x[0, int(arr[row, col]), row, col] = 1.0
    return x


def test_runtime(model: onnx.ModelProto) -> None:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    x = random_one_hot()
    y = session.run([OUT_NAME], {IN_NAME: x})[0]
    expected = np.zeros(FULL_SHAPE, dtype=np.float32)
    expected[:, :, :3, :3] = x[:, :, :3, :3][:, :, ::-1, ::-1]
    if not np.array_equal(y, expected):
        raise AssertionError("ONNX output did not match x[:, :, ::-1, ::-1]")
    if not np.all(y[:, :, :3, :3].sum(axis=1) == 1.0):
        raise AssertionError("ONNX output is not exactly one-hot per cell")


def validate_examples(model: onnx.ModelProto) -> None:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    checked = 0
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(data.get(split, [])):
            actual = session.run([OUT_NAME], {IN_NAME: grid_to_one_hot(example["input"])})[0]
            expected = grid_to_one_hot(example["output"])
            if not np.array_equal(actual > 0.0, expected > 0.0):
                raise AssertionError(f"{TASK_ID} failed {split}[{index}]")
            checked += 1
    print(f"validation examples: {checked} passed")


def dtype_itemsize(elem_type: int) -> int:
    sizes = {
        TensorProto.FLOAT: 4,
        TensorProto.UINT8: 1,
        TensorProto.INT8: 1,
        TensorProto.UINT16: 2,
        TensorProto.INT16: 2,
        TensorProto.INT32: 4,
        TensorProto.INT64: 8,
        TensorProto.BOOL: 1,
        TensorProto.DOUBLE: 8,
        TensorProto.UINT32: 4,
        TensorProto.UINT64: 8,
    }
    return sizes[elem_type]


def tensor_numel(value_info: onnx.ValueInfoProto) -> int:
    shape = value_info.type.tensor_type.shape
    numel = 1
    for dim in shape.dim:
        if dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0:
            raise ValueError(f"non-static inferred shape for {value_info.name!r}")
        numel *= dim.dim_value
    return numel


def inferred_internal_tensor_sizes(model: onnx.ModelProto) -> dict[str, int]:
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    graph_names = {IN_NAME, OUT_NAME}
    initializer_names = {init.name for init in inferred.graph.initializer}
    sizes: dict[str, int] = {}
    for value_info in inferred.graph.value_info:
        if value_info.name in graph_names or value_info.name in initializer_names:
            continue
        elem_type = value_info.type.tensor_type.elem_type
        sizes[value_info.name] = tensor_numel(value_info) * dtype_itemsize(elem_type)
    return sizes


def initializer_scalar_count(model: onnx.ModelProto) -> int:
    return sum(int(np.prod(init.dims, dtype=np.int64)) for init in model.graph.initializer)


def print_stats(model: onnx.ModelProto) -> None:
    internal_sizes = inferred_internal_tensor_sizes(model)
    params = initializer_scalar_count(model)
    memory = sum(internal_sizes.values())
    print(f"nodes: {len(model.graph.node)}")
    print(f"initializers: {len(model.graph.initializer)}")
    print(f"initializer/scalar params: {params}")
    print(f"inferred internal tensor sizes: {internal_sizes}")
    print(f"estimated internal memory bytes: {memory}")
    print(f"estimated NeuroGolf cost: {memory + params}")


def main() -> None:
    model = build_slice_model()
    test_runtime(model)
    validate_examples(model)
    onnx.save(model, BEST_PATH)
    print(f"wrote: {BEST_PATH}")
    print_stats(model)
    print("runtime test: passed")


if __name__ == "__main__":
    main()
