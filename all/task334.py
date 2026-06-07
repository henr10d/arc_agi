"""Compact ONNX solver for ARC task334.

Task rule: each input is a 5x5 black grid containing exactly one foreground
color. The foreground color alone determines a fixed 3x3 gray glyph:
blue (1) -> plus, red (2) -> T, green (3) -> bottom-right L/corner. The
foreground geometry is ignored. Cells outside the 3x3 output grid remain
all-zero padding in NeuroGolf's [1, 10, 30, 30] tensor.

ONNX approach: count channels to identify the foreground color, convert that
color into the glyph's horizontal row and vertical column, build the 3x3 glyph
as a bool mask, then cast only the compact 1x6x3x3 tensor before final padding.
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

TASK_ID = "task334"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, 10, 30, 30]
OPSET = 10
IR_VERSION = 10
OUT_SIZE = 3
PAD = 30 - OUT_SIZE

BLUE = 1
RED = 2
GREEN = 3
GRAY = 5


def _init(array: np.ndarray | int | float, name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(array), name=name)


def solve_grid(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    colors = set(int(v) for v in arr.ravel() if int(v) != 0)
    if RED in colors:
        return np.asarray([[5, 5, 5], [0, 5, 0], [0, 5, 0]], dtype=np.int64)
    if BLUE in colors:
        return np.asarray([[0, 5, 0], [5, 5, 5], [0, 5, 0]], dtype=np.int64)
    if GREEN in colors:
        return np.asarray([[0, 0, 5], [0, 0, 5], [5, 5, 5]], dtype=np.int64)
    raise ValueError("task334 expects one foreground color from {1, 2, 3}")


def build_onnx_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        _init(np.asarray(0.0, dtype=np.float32), "zero_f"),
        _init(np.asarray(0, dtype=np.int32), "zero_i"),
        _init(np.asarray(1, dtype=np.int32), "one_i"),
        _init(np.asarray(2, dtype=np.int32), "two_i"),
        _init(np.asarray([[[[0], [1], [2]]]], dtype=np.int32), "row_coord"),
        _init(np.asarray([[[[0, 1, 2]]]], dtype=np.int32), "col_coord"),
    ]

    nodes = [
        helper.make_node("ReduceSum", [IN_NAME], ["channel_counts"], axes=[0, 2, 3], keepdims=0),
        helper.make_node("Gather", ["channel_counts", "two_i"], ["red_count"], axis=0),
        helper.make_node("Gather", ["channel_counts", "one_i"], ["blue_count"], axis=0),
        helper.make_node("Greater", ["red_count", "zero_f"], ["has_red"]),
        helper.make_node("Greater", ["blue_count", "zero_f"], ["has_blue"]),
        helper.make_node("Where", ["has_blue", "one_i", "two_i"], ["row_not_red"]),
        helper.make_node("Where", ["has_red", "zero_i", "row_not_red"], ["row_idx"]),
        helper.make_node("Or", ["has_red", "has_blue"], ["red_or_blue"]),
        helper.make_node("Where", ["red_or_blue", "one_i", "two_i"], ["col_idx"]),
        helper.make_node("Equal", ["row_coord", "row_idx"], ["row_mask"]),
        helper.make_node("Equal", ["col_coord", "col_idx"], ["col_mask"]),
        helper.make_node("Or", ["row_mask", "col_mask"], ["gray_bool"]),
        helper.make_node("Not", ["gray_bool"], ["black_bool"]),
        helper.make_node("And", ["gray_bool", "black_bool"], ["zero_bool"]),
        helper.make_node(
            "Concat",
            ["black_bool", "zero_bool", "zero_bool", "zero_bool", "zero_bool", "gray_bool"],
            ["out6_bool"],
            axis=1,
        ),
        helper.make_node("Cast", ["out6_bool"], ["out6"], to=TensorProto.FLOAT),
        helper.make_node(
            "Pad",
            ["out6"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 4, PAD, PAD],
        ),
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
    failures: dict[str, int] = {"train": 0, "test": 0, "arc-gen": 0}
    for item in _load_examples():
        example = item["example"]
        expected_grid = solve_grid(example["input"])
        json_grid = np.asarray(example["output"], dtype=np.int64)
        if not np.array_equal(expected_grid, json_grid):
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
