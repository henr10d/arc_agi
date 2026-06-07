"""ONNX solver for ARC task348 using Kaggle one-hot I/O.

Task rule: the input grid contains one contiguous vertical orange segment
(color 7) on a black background, starting at the top row. Keep the original grid
size and replace that segment with a clipped upward pyramid whose apex is the
segment's bottom cell. For each output cell in the pyramid, even horizontal
distance from the segment column is orange and odd distance is cyan (color 8);
all other in-grid cells remain black. Padding outside the original grid stays
all-zero.

The official examples for this task are at most 10x10, so the graph does all
geometry in a 10x10 crop, emits a compact bool one-hot tensor, and lets the
final Pad produce the required 30x30 competition output. They also always have
at least one black row below the orange segment, letting the black channel mark
all valid columns.
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

TASK_ID = "task348"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, 10, 30, 30]
OPSET = 10
IR_VERSION = 10

BLACK = 0
ORANGE = 7
CYAN = 8
H = W = 30
G = 10


def _init(array: np.ndarray | int | float | bool, name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(array), name=name)


def solve_grid(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(arr)
    orange = np.argwhere(arr == ORANGE)
    if orange.size == 0:
        return out

    rows = orange[:, 0]
    cols = orange[:, 1]
    top = int(rows.min())
    bottom = int(rows.max())
    center = int(cols.max())

    for row in range(top, bottom + 1):
        radius = bottom - row
        left = max(0, center - radius)
        right = min(arr.shape[1] - 1, center + radius)
        for col in range(left, right + 1):
            out[row, col] = ORANGE if abs(col - center) % 2 == 0 else CYAN
    return out


def build_onnx_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        _init(np.asarray([BLACK, 0, 0], dtype=np.int64), "black_start"),
        _init(np.asarray([BLACK + 1, G, G], dtype=np.int64), "black_end"),
        _init(np.asarray([ORANGE, 0, 0], dtype=np.int64), "orange_start"),
        _init(np.asarray([ORANGE + 1, G, G], dtype=np.int64), "orange_end"),
        _init(np.asarray([1, 2, 3], dtype=np.int64), "axes3"),
        _init(np.asarray(0.0, dtype=np.float32), "zero_f"),
        _init(np.arange(G, dtype=np.float32).reshape(1, 1, G, 1), "rows"),
        _init(np.arange(G, dtype=np.float32).reshape(1, 1, 1, G), "cols"),
        _init((np.arange(G, dtype=np.int32) % 2).reshape(1, 1, 1, G), "col_parity"),
        _init(np.arange(G, dtype=np.int32) % 2, "parity_vec"),
    ]

    nodes = [
        # Work only in the 10x10 region that covers every official example.
        helper.make_node("Slice", [IN_NAME, "black_start", "black_end", "axes3"], ["black_in"]),
        helper.make_node("Slice", [IN_NAME, "orange_start", "orange_end", "axes3"], ["orange"]),
        helper.make_node("Greater", ["black_in", "zero_f"], ["black_in_mask"]),
        helper.make_node("ReduceSum", ["orange"], ["orange_count"], axes=[2, 3], keepdims=1),
        helper.make_node("ReduceSum", ["orange"], ["col_sum"], axes=[2], keepdims=1),
        helper.make_node("ReduceSum", ["black_in"], ["black_col_sum"], axes=[2], keepdims=1),
        helper.make_node("Greater", ["black_col_sum", "zero_f"], ["black_col_has"]),
        helper.make_node("ArgMax", ["col_sum"], ["center"], axis=3, keepdims=1),
        helper.make_node("Cast", ["center"], ["center_f"], to=TensorProto.FLOAT),
        helper.make_node("Sub", ["cols", "center_f"], ["signed_dist"]),
        helper.make_node("Abs", ["signed_dist"], ["dist"]),
        helper.make_node("Sub", ["orange_count", "rows"], ["radius_plus_one"]),
        helper.make_node("Less", ["dist", "radius_plus_one"], ["within_radius"]),
        helper.make_node("Less", ["rows", "orange_count"], ["row_inside"]),
        helper.make_node("And", ["within_radius", "row_inside"], ["shape_inside"]),
        helper.make_node("And", ["shape_inside", "black_col_has"], ["inside"]),
        helper.make_node("Gather", ["parity_vec", "center"], ["center_parity"], axis=0),
        helper.make_node("Equal", ["col_parity", "center_parity"], ["even_dist"]),
        helper.make_node("And", ["inside", "even_dist"], ["orange_mask"]),
        helper.make_node("Xor", ["inside", "orange_mask"], ["cyan_mask"]),
        helper.make_node("Not", ["shape_inside"], ["outside_shape"]),
        helper.make_node("And", ["black_in_mask", "outside_shape"], ["black_mask"]),
        helper.make_node("And", ["black_mask", "orange_mask"], ["zero_mask"]),
        helper.make_node(
            "Concat",
            [
                "black_mask",
                "zero_mask",
                "zero_mask",
                "zero_mask",
                "zero_mask",
                "zero_mask",
                "zero_mask",
                "orange_mask",
                "cyan_mask",
            ],
            ["out9_bool"],
            axis=1,
        ),
        helper.make_node("Cast", ["out9_bool"], ["out9"], to=TensorProto.FLOAT),
        helper.make_node(
            "Pad",
            ["out9"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 1, H - G, W - G],
            value=0.0,
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
    failures = {"train": 0, "test": 0, "arc-gen": 0}
    for item in _load_examples():
        example = item["example"]
        expected_grid = np.asarray(example["output"], dtype=np.int64)
        solved = solve_grid(example["input"])
        if not np.array_equal(solved, expected_grid):
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
