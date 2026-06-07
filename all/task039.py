"""Minimal ONNX for NeuroGolf task039.

Task rule: the 10x10 input contains a 6x6 mirrored pattern placed with its
top-left corner at row/column 1, 2, or 3. Return the pattern's top-left 3x3
quadrant, padded back to the competition 30x30 one-hot output tensor.

ONNX approach: find the first non-empty candidate row/column by scanning the
black channel over the possible 8-cell logical span, then dynamically Slice the
3x3 crop. Explicit value_info gives the scorer static shapes for the dynamic
crop while avoiding the much larger full-width Gather intermediate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TASK_JSON = ROOT / "data" / "task039.json"
MODEL_PATH = SCRIPT_DIR / "task039.onnx"

INPUT_SHAPE = (1, 10, 30, 30)
OUTPUT_CROP = 3


def tensor_i64(name: str, values: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, TensorProto.INT64, [len(values)], values)


def tensor_f32(name: str, values: list[float]) -> onnx.TensorProto:
    return helper.make_tensor(name, TensorProto.FLOAT, [len(values)], values)


def value_info(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def load_task() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_JSON.open(encoding="utf-8") as fh:
        return json.load(fh)


def one_hot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(INPUT_SHAPE, dtype=np.float32)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            out[0, int(color), row, col] = 1.0
    return out


def bbox_crop(grid: list[list[int]]) -> list[list[int]]:
    rows = [row for row, values in enumerate(grid) if any(value != 0 for value in values)]
    cols = [col for col in range(len(grid[0])) if any(values[col] != 0 for values in grid)]
    row0, col0 = min(rows), min(cols)
    return [values[col0 : col0 + OUTPUT_CROP] for values in grid[row0 : row0 + OUTPUT_CROP]]


def validate_rule(data: dict[str, list[dict[str, list[list[int]]]]]) -> None:
    for split, examples in data.items():
        for index, example in enumerate(examples):
            if bbox_crop(example["input"]) != example["output"]:
                raise AssertionError(f"bbox crop rule failed on {split}[{index}]")


def static_slice(
    data: str,
    output: str,
    starts: list[int],
    ends: list[int],
    axes: list[int],
) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto]]:
    initializers = [
        tensor_i64(f"{output}_starts", starts),
        tensor_i64(f"{output}_ends", ends),
        tensor_i64(f"{output}_axes", axes),
    ]
    return [helper.make_node("Slice", [data, *(item.name for item in initializers)], [output])], initializers


def make_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = [
        tensor_f32("eight", [8.0]),
        tensor_i64("zero", [0]),
        tensor_i64("one", [1]),
        tensor_i64("ten", [10]),
        tensor_i64("four", [4]),
        tensor_i64("slice_axes", [0, 1, 2, 3]),
    ]

    new_nodes, new_initializers = static_slice(
        "input",
        "row_black",
        [0, 0, 1, 1],
        [1, 1, 4, 9],
        [0, 1, 2, 3],
    )
    nodes += new_nodes
    initializers += new_initializers

    new_nodes, new_initializers = static_slice(
        "input",
        "col_black",
        [0, 0, 1, 1],
        [1, 1, 9, 4],
        [0, 1, 2, 3],
    )
    nodes += new_nodes
    initializers += new_initializers

    nodes += [
        helper.make_node("ReduceSum", ["row_black"], ["row_sum"], axes=[1, 3], keepdims=0),
        helper.make_node("ReduceSum", ["col_black"], ["col_sum"], axes=[1, 2], keepdims=0),
        helper.make_node("Less", ["row_sum", "eight"], ["row_has"]),
        helper.make_node("Less", ["col_sum", "eight"], ["col_has"]),
        helper.make_node("Cast", ["row_has"], ["row_flags"], to=TensorProto.UINT8),
        helper.make_node("Cast", ["col_has"], ["col_flags"], to=TensorProto.UINT8),
        helper.make_node("ArgMax", ["row_flags"], ["row_rel"], axis=1, keepdims=0),
        helper.make_node("ArgMax", ["col_flags"], ["col_rel"], axis=1, keepdims=0),
        helper.make_node("Add", ["row_rel", "one"], ["row_start"]),
        helper.make_node("Add", ["col_rel", "one"], ["col_start"]),
        helper.make_node("Add", ["row_rel", "four"], ["row_end"]),
        helper.make_node("Add", ["col_rel", "four"], ["col_end"]),
        helper.make_node("Concat", ["zero", "zero", "row_start", "col_start"], ["starts"], axis=0),
        helper.make_node("Concat", ["one", "ten", "row_end", "col_end"], ["ends"], axis=0),
        helper.make_node("Slice", ["input", "starts", "ends", "slice_axes"], ["crop"]),
        helper.make_node(
            "Pad",
            ["crop"],
            ["output"],
            pads=[0, 0, 0, 0, 0, 0, 27, 27],
            mode="constant",
            value=0.0,
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "task039",
        [value_info("input", TensorProto.FLOAT, list(INPUT_SHAPE))],
        [value_info("output", TensorProto.FLOAT, list(INPUT_SHAPE))],
        initializers,
        value_info=[
            value_info("row_black", TensorProto.FLOAT, [1, 1, 3, 8]),
            value_info("col_black", TensorProto.FLOAT, [1, 1, 8, 3]),
            value_info("row_sum", TensorProto.FLOAT, [1, 3]),
            value_info("col_sum", TensorProto.FLOAT, [1, 3]),
            value_info("row_has", TensorProto.BOOL, [1, 3]),
            value_info("col_has", TensorProto.BOOL, [1, 3]),
            value_info("row_flags", TensorProto.UINT8, [1, 3]),
            value_info("col_flags", TensorProto.UINT8, [1, 3]),
            value_info("row_rel", TensorProto.INT64, [1]),
            value_info("col_rel", TensorProto.INT64, [1]),
            value_info("row_start", TensorProto.INT64, [1]),
            value_info("col_start", TensorProto.INT64, [1]),
            value_info("row_end", TensorProto.INT64, [1]),
            value_info("col_end", TensorProto.INT64, [1]),
            value_info("starts", TensorProto.INT64, [4]),
            value_info("ends", TensorProto.INT64, [4]),
            value_info("crop", TensorProto.FLOAT, [1, 10, 3, 3]),
        ],
    )
    return helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])


def validate_model(model: onnx.ModelProto, data: dict[str, list[dict[str, list[list[int]]]]]) -> None:
    onnx.checker.check_model(model, full_check=True)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    for split, examples in data.items():
        for index, example in enumerate(examples):
            actual = session.run(["output"], {"input": one_hot(example["input"])})[0]
            expected = one_hot(example["output"])
            if not np.array_equal(actual > 0.0, expected > 0.0):
                raise AssertionError(f"wrong output on {split}[{index}]")


def score_output(path: Path) -> dict[str, Any] | None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from score_model import score_file
    except ImportError:
        return None
    return score_file(path)


def main() -> None:
    data = load_task()
    validate_rule(data)
    model = make_model()
    validate_model(model, data)
    onnx.save(model, MODEL_PATH)

    result = score_output(MODEL_PATH)
    print(f"saved {MODEL_PATH}")
    if result is not None:
        print(
            "score_model: "
            f"valid={result.get('valid')} memory={result.get('memory')} "
            f"params={result.get('params')} cost={result.get('cost')} "
            f"score={result.get('score')}"
        )
        if not result.get("valid"):
            raise SystemExit(result.get("error") or "score_model marked model invalid")


if __name__ == "__main__":
    main()
