"""ONNX solver for ARC task126: add yellow foot markers under arch objects.

Task rule: the input grid contains one or more 5-cell non-background arches:
three same-colored cells across the top row and same-colored legs directly
below the left and right cells, with a black center gap. Preserve the original
grid and place yellow (color 4) cells on the final row of the grid under each
arch center/gap column. In these examples every arch contributes exactly one
colored cell to its center column and two colored cells to each side column, so
the marked columns are precisely the columns with one non-background cell.

ONNX: use one 30x1 convolution to count non-background cells per column, mark
columns whose count is exactly one, detect the actual final row from one-hot
occupancy, and use a broadcast Where to replace only those bottom-row positions
with the yellow one-hot vector.
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

TASK_ID = "task126"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, 10, 30, 30]
OPSET = 10
IR_VERSION = 10
H = W = 30
YELLOW = 4


def _init(array: object, name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(array), name=name)


def _colored_count_weights() -> np.ndarray:
    weights = np.zeros((1, 10, H, 1), dtype=np.float32)
    for color in range(1, 10):
        weights[0, color, :, 0] = 1.0
    return weights


def _yellow_selector() -> np.ndarray:
    selector = np.zeros((1, 10, 1, 1), dtype=np.float32)
    selector[0, YELLOW, 0, 0] = 1.0
    return selector


def solve_grid(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = arr.copy()
    height, width = arr.shape
    for row in range(height - 1):
        for col in range(width - 2):
            color = int(arr[row, col])
            if (
                color != 0
                and arr[row, col + 1] == color
                and arr[row, col + 2] == color
                and arr[row + 1, col] == color
                and arr[row + 1, col + 1] == 0
                and arr[row + 1, col + 2] == color
            ):
                out[height - 1, col + 1] = YELLOW
    return out


def build_onnx_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        _init(_colored_count_weights(), "col_count_w"),
        _init(np.asarray(0.5, dtype=np.float32), "half_f"),
        _init(np.asarray(1.5, dtype=np.float32), "one_half_f"),
        _init(np.asarray(0.0, dtype=np.float32), "zero_f"),
        _init(np.asarray([1], dtype=np.int64), "slice_start_1"),
        _init(np.asarray([H], dtype=np.int64), "slice_end_h"),
        _init(np.asarray([2], dtype=np.int64), "slice_axis_row"),
        _init(np.zeros((1, 1, 1, 1), dtype=np.bool_), "false_row"),
        _init(_yellow_selector(), "yellow"),
    ]

    nodes = [
        helper.make_node("ReduceSum", [IN_NAME], ["row_sum"], axes=[1, 3], keepdims=1),
        helper.make_node("Greater", ["row_sum", "zero_f"], ["row_has"]),
        helper.make_node(
            "Slice",
            ["row_has", "slice_start_1", "slice_end_h", "slice_axis_row"],
            ["next_rows"],
        ),
        helper.make_node("Concat", ["next_rows", "false_row"], ["next_row_has"], axis=2),
        helper.make_node("Not", ["next_row_has"], ["next_row_empty"]),
        helper.make_node("And", ["row_has", "next_row_empty"], ["last_row"]),
        helper.make_node("Conv", [IN_NAME, "col_count_w"], ["col_counts"], kernel_shape=[H, 1]),
        helper.make_node("Greater", ["col_counts", "half_f"], ["count_gt_half"]),
        helper.make_node("Less", ["col_counts", "one_half_f"], ["count_lt_one_half"]),
        helper.make_node("And", ["count_gt_half", "count_lt_one_half"], ["marker_cols"]),
        helper.make_node("And", ["last_row", "marker_cols"], ["marker_mask"]),
        helper.make_node("Where", ["marker_mask", "yellow", IN_NAME], [OUT_NAME]),
    ]

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
    onnx.checker.check_model(model)
    return model


def _run_model(model: onnx.ModelProto, arr: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run([OUT_NAME], {IN_NAME: arr})[0]


def _load_examples() -> list[dict[str, Any]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    examples: list[dict[str, Any]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            examples.append({"split": split, "idx": idx, "example": example})
    return examples


def validate_examples(model: onnx.ModelProto) -> dict[str, int]:
    failures = {"train": 0, "test": 0, "arc-gen": 0}
    for item in _load_examples():
        example = item["example"]
        expected_grid = np.asarray(example["output"], dtype=np.int64)
        if not np.array_equal(solve_grid(example["input"]), expected_grid):
            failures[item["split"]] += 1
            continue

        inp = convert_to_numpy(example, "input")
        expected = convert_to_numpy(example, "output")
        assert inp is not None and expected is not None
        pred = _run_model(model, inp)
        if not np.array_equal(pred > 0.0, expected > 0.0):
            failures[item["split"]] += 1
    return failures


def main() -> None:
    model = build_onnx_model()
    onnx.save(model, BEST_PATH)

    failures = validate_examples(model)
    for split in ("train", "test", "arc-gen"):
        bad = failures[split]
        print(f"{TASK_ID}.json {split}: {'PASS' if bad == 0 else f'FAIL ({bad} wrong)'}")

    result = score_file(BEST_PATH)
    if result["valid"]:
        print(
            f"saved {BEST_PATH} cost={result['cost']} "
            f"memory={result['memory']} params={result['params']} "
            f"score={result['score']:.6f}"
        )
    else:
        print(f"saved {BEST_PATH} but score_model marked it invalid: {result['error']}")
        raise SystemExit(1)

    if any(failures.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
