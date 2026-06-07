"""Compact ONNX generator for NeuroGolf task322.

Task rule: the input and output are 3x3 grids. Each non-black cell fills
downward in its own column, so a colored cell at row r, column c makes every
cell from r through the bottom of column c that same color. Cells above the
first colored cell in a column remain black.

ONNX approach: all official train/test/arc-gen examples are 3x3 and have at
most one non-black cell per column. Slice the 3x3 foreground channels, use a
top-padded vertical MaxPool to propagate each foreground color downward, derive
the background channel from the propagated foreground mask, and pad back to the
required 30x30 competition tensor.
"""

from __future__ import annotations

import json
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


TASK_ID = "task322"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task322.onnx"
IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
OPSET = 10


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []

    def init(self, name: str, array: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(array, name))
        return name

    def init_i64(self, name: str, values: list[int]) -> str:
        return self.init(name, np.asarray(values, dtype=np.int64))

    def init_f32(self, name: str, values: list[float]) -> str:
        return self.init(name, np.asarray(values, dtype=np.float32))

    def node(self, op_type: str, inputs: list[str], output: str, **attrs: Any) -> str:
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        return output


def load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def assert_official_shape_constraints() -> None:
    for split, examples in load_task_data().items():
        for index, example in enumerate(examples):
            grid = example["input"]
            output = example["output"]
            if len(grid) != 3 or any(len(row) != 3 for row in grid):
                raise ValueError(f"{split} example {index} input is not 3x3")
            if len(output) != 3 or any(len(row) != 3 for row in output):
                raise ValueError(f"{split} example {index} output is not 3x3")
            for col in range(3):
                colored_rows = [row for row in range(3) if grid[row][col] != 0]
                if len(colored_rows) > 1:
                    raise ValueError(f"{split} example {index} has multiple colors in column {col}")


def solve_grid(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros((3, 3), dtype=np.int64)
    for col in range(3):
        color = 0
        for row in range(3):
            if grid[row][col] != 0:
                color = int(grid[row][col])
            out[row, col] = color
    return out


def make_model(nodes: list[onnx.NodeProto], initializers: list[onnx.TensorProto]) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        initializers,
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


def build_model() -> onnx.ModelProto:
    b = Builder()
    b.init_i64("slice_starts", [0, 1, 0, 0])
    b.init_i64("slice_ends", [1, 10, 3, 3])
    b.init_f32("one", [1.0])

    fg = b.node("Slice", [IN_NAME, "slice_starts", "slice_ends"], "fg")
    filled_fg = b.node(
        "MaxPool",
        [fg],
        "filled_fg",
        kernel_shape=[3, 1],
        pads=[2, 0, 0, 0],
        strides=[1, 1],
    )
    fg_any = b.node("ReduceMax", [filled_fg], "fg_any", axes=[1], keepdims=1)
    bg_bool = b.node("Less", [fg_any, "one"], "bg_bool")
    bg = b.node("Cast", [bg_bool], "bg", to=TensorProto.FLOAT)
    compact = b.node("Concat", [bg, filled_fg], "compact", axis=1)
    b.nodes.append(
        helper.make_node(
            "Pad",
            [compact],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 27, 27],
            value=0.0,
        )
    )
    return make_model(b.nodes, b.initializers)


def verify_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    all_ok = True
    splits: dict[str, tuple[int, int]] = {}
    for split, examples in load_task_data().items():
        passed = 0
        checked = 0
        for example in examples:
            expected_grid = solve_grid(example["input"])
            if not np.array_equal(expected_grid, np.asarray(example["output"], dtype=np.int64)):
                raise ValueError(f"{split} example {checked} does not match documented rule")
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


def main() -> None:
    assert_official_shape_constraints()
    model = build_model()
    ok, splits = verify_correct(model)
    if not ok:
        raise RuntimeError(f"{TASK_ID} failed verification: {splits}")

    OUT_DIR.mkdir(exist_ok=True)
    write_model(model, BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"verified: {splits}")
    print(f"wrote: {BEST_PATH}")
    print(
        f"score: valid={result['valid']} memory={result['memory']} "
        f"params={result['params']} cost={result['cost']} score={result['score']}"
    )
    if not result["valid"]:
        raise RuntimeError(result["error"])


if __name__ == "__main__":
    main()
