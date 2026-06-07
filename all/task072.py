"""Compact ONNX for task072: XOR the two red masks around a separator row.

Task rule: the input's active area is 13x5.  A full non-background separator
row splits two 6x5 red/black patterns.  Compare the pattern above and below the
separator cell-by-cell; output green (3) where exactly one side is red and
black (0) where both sides match.  Local train/test/arc-gen data fixes the
separator at row 6 and the red color at channel 2, so the exported ONNX uses
those proven constants for a smaller graph.
"""

from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, sanitize_model, score_file  # noqa: E402

TASK_ID = "task072"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, 10, 30, 30]
OPSET = 10
IR_VERSION = 10


def _init(inits: list[onnx.TensorProto], vals: Any, dtype: np.dtype[Any], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=dtype), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, vals, np.int64, name)


def solve_grid(grid: list[list[int]]) -> list[list[int]]:
    """Reference Python solver with robust separator/color detection."""
    height = len(grid)
    width = len(grid[0])
    counts = Counter(color for row in grid for color in row if color != 0)

    candidates: list[tuple[int, int, int]] = []
    for row_idx, row in enumerate(grid):
        color = row[0]
        if color == 0 or any(cell != color for cell in row):
            continue
        if row_idx != height - row_idx - 1:
            continue
        if color not in (0, 2):
            candidates.append((0, counts[color], row_idx))
        else:
            candidates.append((1, counts[color], row_idx))
    if not candidates:
        raise ValueError("no separator row found")

    _, _, sep_row = min(candidates)
    sep_color = grid[sep_row][0]
    top = grid[:sep_row]
    bottom = grid[sep_row + 1 :]
    if len(top) != len(bottom):
        raise ValueError(f"separator at row {sep_row} does not split equal halves")

    color_counts = Counter(
        cell
        for row in top + bottom
        for cell in row
        if cell not in (0, sep_color)
    )
    red_color = color_counts.most_common(1)[0][0] if color_counts else 2
    return [
        [3 if ((top[r][c] == red_color) ^ (bottom[r][c] == red_color)) else 0 for c in range(width)]
        for r in range(len(top))
    ]


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    axes = _i64(inits, [1, 2, 3], "slice_axes")
    top_starts = _i64(inits, [2, 0, 0], "top_starts")
    top_ends = _i64(inits, [3, 6, 5], "top_ends")
    bottom_starts = _i64(inits, [2, 7, 0], "bottom_starts")
    bottom_ends = _i64(inits, [3, 13, 5], "bottom_ends")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, top_starts, top_ends, axes], ["top_red"]),
            helper.make_node("Slice", [IN_NAME, bottom_starts, bottom_ends, axes], ["bottom_red"]),
            helper.make_node("Cast", ["top_red"], ["top_red_bool"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["bottom_red"], ["bottom_red_bool"], to=TensorProto.BOOL),
            helper.make_node("Xor", ["top_red_bool", "bottom_red_bool"], ["xor_red"]),
            helper.make_node("Not", ["xor_red"], ["same_red_state"]),
            helper.make_node("And", ["same_red_state", "xor_red"], ["false_plane"]),
            helper.make_node(
                "Concat",
                [
                    "same_red_state",
                    "false_plane",
                    "false_plane",
                    "xor_red",
                ],
                ["out4_bool"],
                axis=1,
            ),
            helper.make_node("Cast", ["out4_bool"], ["out4_float"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out4_float"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 6, 24, 25],
            ),
        ]
    )

    graph = helper.make_graph(
        nodes,
        f"{TASK_ID}_red_xor",
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


def _load_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _check_reference_solver(data: dict[str, list[dict[str, list[list[int]]]]]) -> dict[str, tuple[int, int]]:
    counts: dict[str, tuple[int, int]] = {}
    for split in ("train", "test", "arc-gen"):
        passed = 0
        examples = data.get(split, [])
        for example in examples:
            if solve_grid(example["input"]) == example["output"]:
                passed += 1
        counts[split] = (passed, len(examples))
    return counts


def _check_onnx(model: onnx.ModelProto, data: dict[str, list[dict[str, list[list[int]]]]]) -> dict[str, tuple[int, int]]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise ValueError("model failed sanitizer")

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        sanitized.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    counts: dict[str, tuple[int, int]] = {}
    for split in ("train", "test", "arc-gen"):
        passed = 0
        total = 0
        for example in data.get(split, []):
            inp = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if inp is None or expected is None:
                continue
            total += 1
            pred = session.run([OUT_NAME], {IN_NAME: inp})[0]
            if np.array_equal((pred > 0.0).astype(np.float32), expected):
                passed += 1
        counts[split] = (passed, total)
    return counts


def _all_passed(counts: dict[str, tuple[int, int]]) -> bool:
    return all(passed == total for passed, total in counts.values())


def main() -> None:
    data = _load_data()

    reference_counts = _check_reference_solver(data)
    print(f"reference solver: {reference_counts}")
    if not _all_passed(reference_counts):
        raise SystemExit("reference solver mismatch")

    model = build_model()
    onnx_counts = _check_onnx(model, data)
    print(f"onnx solver:      {onnx_counts}")
    if not _all_passed(onnx_counts):
        raise SystemExit("ONNX solver mismatch")

    onnx.save(model, BEST_PATH)
    result = score_file(BEST_PATH)
    if not result["valid"]:
        raise SystemExit(f"scoring failed: {result['error']}")
    print(f"saved {BEST_PATH}")
    print(
        f"memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']:.6f}"
    )


if __name__ == "__main__":
    main()
