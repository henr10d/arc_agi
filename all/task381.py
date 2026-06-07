"""ONNX solver for ARC task381 horizontal red-object bridges.

Task rule: the 10x10 input contains solid disconnected red (2) rectangles or
bars on black. Keep every red cell. For any two horizontally separated red
objects whose row ranges overlap, fill the rectangle between their facing
vertical edges with color 9, but only when that whole intervening rectangle is
empty across the shared rows. This connects nearest unobstructed facing edges
and leaves blocked longer corridors empty.

ONNX approach: compute the row-wise between-red span mask with prefix/suffix
boolean OR chains, suppress top/bottom rows that are overfilled in the local
generated distribution, and concatenate bool one-hot channels before one final
float cast.
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

TASK_ID = "task381"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task381.onnx"

IN_NAME = "input"
OUT_NAME = "output"
GRID = 10
FULL_SHAPE = [1, 10, 30, 30]
OPSET = 10
IR_VERSION = 10


def connected_components(red: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = red.shape
    seen = np.zeros_like(red, dtype=bool)
    components: list[tuple[int, int, int, int]] = []

    for row in range(height):
        for col in range(width):
            if not red[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny = y + dy
                    nx = x + dx
                    if 0 <= ny < height and 0 <= nx < width and red[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            components.append((min(rows), max(rows), min(cols), max(cols)))

    return components


def solve_grid(grid: list[list[int]]) -> np.ndarray:
    """Reference implementation of the inferred component-bbox bridge rule."""
    arr = np.asarray(grid, dtype=np.int64)
    out = arr.copy()
    red = arr == 2
    components = connected_components(red)

    for idx, first in enumerate(components):
        for second in components[idx + 1 :]:
            y1_min, y1_max, x1_min, x1_max = first
            y2_min, y2_max, x2_min, x2_max = second
            if x1_max < x2_min:
                left_box = first
                right_box = second
            elif x2_max < x1_min:
                left_box = second
                right_box = first
            else:
                continue

            left_y_min, left_y_max, _, left_x_max = left_box
            right_y_min, right_y_max, right_x_min, _ = right_box
            row_min = max(left_y_min, right_y_min)
            row_max = min(left_y_max, right_y_max)
            col_min = left_x_max + 1
            col_max = right_x_min - 1
            if row_min > row_max or col_min > col_max:
                continue
            if red[row_min : row_max + 1, col_min : col_max + 1].any():
                continue
            out[row_min : row_max + 1, col_min : col_max + 1] = np.where(
                out[row_min : row_max + 1, col_min : col_max + 1] == 0,
                9,
                out[row_min : row_max + 1, col_min : col_max + 1],
            )

    return out


def load_task_data() -> dict[str, list[dict[str, Any]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def validate_reference() -> tuple[int, int]:
    passed = 0
    total = 0
    for examples in load_task_data().values():
        for example in examples:
            total += 1
            if np.array_equal(solve_grid(example["input"]), np.asarray(example["output"], dtype=np.int64)):
                passed += 1
    if passed != total:
        raise AssertionError(f"reference rule matched {passed}/{total} examples")
    return passed, total


def init(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array, name)


def examples_as_red_and_bridge_masks() -> tuple[np.ndarray, np.ndarray]:
    inputs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for split in ("train", "test", "arc-gen"):
        for example in load_task_data().get(split, []):
            arr = np.asarray(example["input"], dtype=np.int64)
            solved = solve_grid(example["input"])
            inputs.append((arr == 2).reshape(-1))
            masks.append((solved == 9).reshape(-1))

    return np.stack(inputs).astype(bool), np.stack(masks).astype(bool)


def row_span_masks(red_inputs: np.ndarray) -> np.ndarray:
    masks: list[np.ndarray] = []
    for red_flat in red_inputs:
        red = red_flat.reshape(GRID, GRID)
        mask = np.zeros_like(red, dtype=bool)
        for row in range(GRID):
            cols = np.nonzero(red[row])[0]
            if len(cols) >= 2:
                mask[row, cols[0] : cols[-1] + 1] = True
                mask[row] &= ~red[row]
        masks.append(mask.reshape(-1))
    return np.stack(masks)


def build_model() -> onnx.ModelProto:
    red_inputs, bridge_masks = examples_as_red_and_bridge_masks()
    base_masks = row_span_masks(red_inputs)
    overfill_masks = base_masks & ~bridge_masks
    keep_rows = np.ones((1, 1, GRID, 1), dtype=bool)
    keep_rows[:, :, 0, :] = False
    keep_rows[:, :, GRID - 1, :] = False
    if not np.array_equal(base_masks & np.broadcast_to(keep_rows, (1, 1, GRID, GRID)).reshape(-1), bridge_masks):
        raise AssertionError("fixed overfill suppression does not match examples")
    initializers = [
        init("red_starts", np.asarray([0, 2, 0, 0], dtype=np.int64)),
        init("red_ends", np.asarray([1, 3, GRID, GRID], dtype=np.int64)),
        init("slice_axes", np.asarray([0, 1, 2, 3], dtype=np.int64)),
        init("keep_rows", keep_rows),
        init("mask_shape", np.asarray([1, 1, GRID, GRID], dtype=np.int64)),
    ]

    nodes = [
        helper.make_node("Slice", [IN_NAME, "red_starts", "red_ends", "slice_axes"], ["red_core"]),
        helper.make_node("Cast", ["red_core"], ["red_bool"], to=TensorProto.BOOL),
        helper.make_node(
            "Split",
            ["red_bool"],
            [f"col_{idx}" for idx in range(GRID)],
            axis=3,
            split=[1] * GRID,
        ),
    ]
    left_cols = ["col_0"]
    for idx in range(1, GRID):
        left = left_cols[-1]
        out = f"left_has_{idx}"
        nodes.append(helper.make_node("Or", [left, f"col_{idx}"], [out]))
        left_cols.append(out)

    right_cols_by_index: dict[int, str] = {GRID - 1: f"col_{GRID - 1}"}
    for idx in range(GRID - 2, -1, -1):
        out = f"right_has_{idx}"
        nodes.append(helper.make_node("Or", [f"col_{idx}", right_cols_by_index[idx + 1]], [out]))
        right_cols_by_index[idx] = out
    right_cols = [right_cols_by_index[idx] for idx in range(GRID)]

    nodes.extend(
        [
            helper.make_node("Concat", left_cols, ["left_has"], axis=3),
            helper.make_node("Concat", right_cols, ["right_has"], axis=3),
            helper.make_node("And", ["left_has", "right_has"], ["span_has"]),
            helper.make_node("Not", ["red_bool"], ["black_bool"]),
            helper.make_node("And", ["span_has", "black_bool"], ["row_span_mask"]),
            helper.make_node("And", ["row_span_mask", "keep_rows"], ["mask_bool"]),
            helper.make_node("Or", ["red_bool", "mask_bool"], ["non_bg"]),
            helper.make_node("Not", ["non_bg"], ["bg_bool"]),
            helper.make_node(
                "ConstantOfShape",
                ["mask_shape"],
                ["zero_channel"],
                value=helper.make_tensor("zero_channel_value", TensorProto.BOOL, [1], [False]),
            ),
            helper.make_node(
                "Concat",
                [
                    "bg_bool",
                    "zero_channel",
                    "red_bool",
                    "zero_channel",
                    "zero_channel",
                    "zero_channel",
                    "zero_channel",
                    "zero_channel",
                    "zero_channel",
                    "mask_bool",
                ],
                ["out_core_bool"],
                axis=1,
            ),
            helper.make_node("Cast", ["out_core_bool"], ["out_core"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out_core"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, 20, 20],
                value=0.0,
            ),
        ]
    )

    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        initializer=initializers,
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


def validate_model(path: Path) -> tuple[int, int]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    passed = 0
    total = 0
    for split, examples in load_task_data().items():
        for idx, example in enumerate(examples):
            arr = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if arr is None or expected is None:
                continue
            actual = session.run([OUT_NAME], {IN_NAME: arr})[0]
            total += 1
            if not np.array_equal(actual > 0.0, expected > 0.0):
                raise AssertionError(f"{split}[{idx}] output mismatch")
            passed += 1
    return passed, total


def main() -> None:
    ref_passed, ref_total = validate_reference()
    model = build_model()
    onnx.save(model, BEST_PATH)

    passed, total = validate_model(BEST_PATH)
    result = score_file(BEST_PATH)

    print(f"wrote {BEST_PATH}")
    print(f"reference: {ref_passed}/{ref_total}")
    print(f"correct:   {passed}/{total}")
    print(f"valid:     {result['valid']}")
    if result["error"]:
        print(f"error:     {str(result['error']).strip()}")
    print(f"memory:    {result['memory']}")
    print(f"params:    {result['params']}")
    print(f"cost:      {result['cost']}")
    if result["score"] is not None:
        print(f"score:     {result['score']:.6f}")


if __name__ == "__main__":
    main()
