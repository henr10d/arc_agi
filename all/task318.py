"""Minimal ONNX for ARC task318 using Kaggle one-hot I/O.

Task rule: the 9x4 input has a 4x4 top pattern, a solid separator row, and a
4x4 bottom pattern.  Ignore the separator.  A 4x4 output cell is black exactly
when the corresponding top and bottom cells are both black; every other output
cell is green.

The graph slices only the black one-hot channel from the top and bottom
regions, intersects those compact 4x4 masks, then uses one broadcast Where to
choose between black and green one-hot channel templates.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task318"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task318.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
IN_ROWS = 9
OUT_ROWS = OUT_COLS = 4
BOTTOM_ROW = 5
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


@dataclass(frozen=True)
class Colors:
    black: int
    top: int
    bottom: int
    green: int
    separator: int


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def load_task() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _single_nonzero(values: set[int], label: str) -> int:
    nonzero = values - {0}
    if len(nonzero) != 1:
        raise ValueError(f"expected one non-black {label} color, found {sorted(nonzero)}")
    return next(iter(nonzero))


def infer_colors(data: dict[str, list[dict[str, list[list[int]]]]]) -> Colors:
    """Infer task color IDs from the actual JSON layout and outputs."""
    input_values = {v for examples in data.values() for ex in examples for row in ex["input"] for v in row}
    output_values = {v for examples in data.values() for ex in examples for row in ex["output"] for v in row}
    if 0 not in input_values or 0 not in output_values:
        raise ValueError("task318 examples do not contain black color 0")

    top_values = {v for examples in data.values() for ex in examples for row in ex["input"][:OUT_ROWS] for v in row}
    bottom_values = {
        v
        for examples in data.values()
        for ex in examples
        for row in ex["input"][BOTTOM_ROW:IN_ROWS]
        for v in row
    }
    separator_values = {v for examples in data.values() for ex in examples for v in ex["input"][OUT_ROWS]}
    green_values = output_values - {0}
    if len(separator_values) != 1 or len(green_values) != 1:
        raise ValueError(
            f"unexpected colors: separator={sorted(separator_values)}, output non-black={sorted(green_values)}"
        )

    return Colors(
        black=0,
        top=_single_nonzero(top_values, "top"),
        bottom=_single_nonzero(bottom_values, "bottom"),
        green=next(iter(green_values)),
        separator=next(iter(separator_values)),
    )


def solve(grid: np.ndarray | list[list[int]], colors: Colors) -> np.ndarray:
    """Reference implementation: intersect black cells from the two 4x4 halves."""
    arr = np.asarray(grid, dtype=np.int64)
    top_black = arr[:OUT_ROWS, :OUT_COLS] == colors.black
    bottom_black = arr[BOTTOM_ROW:IN_ROWS, :OUT_COLS] == colors.black
    return np.where(top_black & bottom_black, colors.black, colors.green).astype(np.int64)


def _grid_to_onehot(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            color = int(arr[r, c])
            if 0 <= color < C:
                out[0, color, r, c] = 1.0
    return out


def build_model(colors: Colors) -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, [0, 1, 2, 3], "axes")
    top_starts = _i64(inits, [0, colors.black, 0, 0], "top_starts")
    top_ends = _i64(inits, [1, colors.black + 1, OUT_ROWS, OUT_COLS], "top_ends")
    bottom_starts = _i64(inits, [0, colors.black, BOTTOM_ROW, 0], "bottom_starts")
    bottom_ends = _i64(inits, [1, colors.black + 1, IN_ROWS, OUT_COLS], "bottom_ends")

    black_template = np.zeros((1, C, 1, 1), dtype=np.float32)
    green_template = np.zeros((1, C, 1, 1), dtype=np.float32)
    black_template[0, colors.black, 0, 0] = 1.0
    green_template[0, colors.green, 0, 0] = 1.0
    black = _f32(inits, black_template, "black")
    green = _f32(inits, green_template, "green")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, top_starts, top_ends, axes], ["top_black_f"]),
            helper.make_node("Slice", [IN_NAME, bottom_starts, bottom_ends, axes], ["bottom_black_f"]),
            helper.make_node("Cast", ["top_black_f"], ["top_black"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["bottom_black_f"], ["bottom_black"], to=TensorProto.BOOL),
            helper.make_node("And", ["top_black", "bottom_black"], ["output_black_mask"]),
            helper.make_node("Where", ["output_black_mask", black, green], ["out4"]),
            helper.make_node(
                "Pad",
                ["out4"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - OUT_ROWS, W - OUT_COLS],
            ),
        ]
    )

    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def validate_model(model: onnx.ModelProto, data: dict[str, list[dict[str, list[list[int]]]]], colors: Colors) -> str:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    counts: dict[str, int] = {}

    for split in ("train", "test", "arc-gen"):
        counts[split] = 0
        for index, example in enumerate(data[split]):
            expected_grid = np.asarray(example["output"], dtype=np.int64)
            solved_grid = solve(example["input"], colors)
            if not np.array_equal(solved_grid, expected_grid):
                return f"{split}#{index}: reference rule mismatch"

            expected = _grid_to_onehot(expected_grid) > 0
            pred = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(example["input"])})[0]
            if pred.shape != tuple(SHAPE):
                return f"{split}#{index}: bad output shape {pred.shape}"
            if not np.array_equal(pred > 0, expected):
                return f"{split}#{index}: ONNX output mismatch"
            counts[split] += 1

    return f"PASS train={counts['train']} test={counts['test']} arc-gen={counts['arc-gen']}"


def main() -> None:
    data = load_task()
    colors = infer_colors(data)
    model = build_model(colors)
    message = validate_model(model, data, colors)
    if not message.startswith("PASS"):
        raise SystemExit(message)

    onnx.save(model, BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"colors: {colors}")
    print(f"validation: {message}")
    print(f"path: {BEST_PATH}")
    print(f"memory: {result.get('memory')}")
    print(f"params: {result.get('params')}")
    print(f"cost: {result.get('cost')}")
    print(f"score: {result.get('score')}")
    if not result.get("valid"):
        raise SystemExit(result.get("error") or "score_model reported invalid model")


if __name__ == "__main__":
    main()
