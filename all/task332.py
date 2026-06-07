"""Build ONNX for ARC task332: recolor gray cells by column parity.

Task rule: each input is a 3xN grid containing only black (0) and gray (5).
The output preserves the grid size and black cells. Gray cells whose column has
the same parity as the grid width remain gray; gray cells in the opposite
parity columns become green (3). The provided task data uses widths 10..20, so
the ONNX computes width parity from columns 10..19, emits the first 20 columns,
and pads the final output to 30x30.
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
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task332"
OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / "task332.onnx"

IN_NAME = "input"
OUT_NAME = "output"
C = 10
N = 30
MIN_ACTIVE_WIDTH = 10
MAX_ACTIVE_WIDTH = 20
SHAPE = [1, C, N, N]
BLACK = 0
GREEN = 3
GRAY = 5
OPSET = 10
IR_VERSION = 10


def solve(grid: list[list[int]]) -> list[list[int]]:
    """Reference grid transform."""
    width = len(grid[0])
    keep_phase = width % 2
    out = [row[:] for row in grid]
    for row in out:
        for col, color in enumerate(row):
            if color == GRAY and col % 2 != keep_phase:
                row[col] = GREEN
    return out


def _load_task() -> dict[str, Any]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _examples(data: dict[str, Any]) -> list[tuple[str, int, dict[str, Any]]]:
    return [
        (split, index, example)
        for split in ("train", "test", "arc-gen")
        for index, example in enumerate(data.get(split, []))
    ]


def _validate_rule(data: dict[str, Any]) -> None:
    for split, index, example in _examples(data):
        grid = example["input"]
        if len(grid) != 3:
            raise ValueError(f"{split}[{index}] has height {len(grid)}, expected 3")
        width = len(grid[0])
        if width < MIN_ACTIVE_WIDTH or width > MAX_ACTIVE_WIDTH:
            raise ValueError(f"{split}[{index}] has invalid width {width}")
        if any(len(row) != width for row in grid):
            raise ValueError(f"{split}[{index}] is ragged")
        colors = {cell for row in grid for cell in row}
        if not colors <= {BLACK, GRAY}:
            raise ValueError(f"{split}[{index}] has unexpected input colors {sorted(colors)}")
        if solve(grid) != example["output"]:
            raise ValueError(f"{split}[{index}] does not match the parity recolor rule")


def _init(inits: list[onnx.TensorProto], name: str, arr: np.ndarray) -> str:
    inits.append(numpy_helper.from_array(arr, name=name))
    return name


def _i64(inits: list[onnx.TensorProto], name: str, values: list[int]) -> str:
    return _init(inits, name, np.asarray(values, dtype=np.int64))


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes_chw = _i64(inits, "axes_chw", [1, 2, 3])
    axes_hw = _i64(inits, "axes_hw", [2, 3])
    st_c0 = _i64(inits, "st_c0", [BLACK, 0, 0])
    en_c0 = _i64(inits, "en_c0", [BLACK + 1, 3, MAX_ACTIVE_WIDTH])
    st_c5 = _i64(inits, "st_c5", [GRAY, 0, 0])
    en_c5 = _i64(inits, "en_c5", [GRAY + 1, 3, MAX_ACTIVE_WIDTH])
    st_tail = _i64(inits, "st_tail", [0, MIN_ACTIVE_WIDTH])
    en_tail = _i64(inits, "en_tail", [1, MAX_ACTIVE_WIDTH])

    alt = np.asarray(
        [[[[1.0 if col % 2 == 0 else -1.0 for col in range(MIN_ACTIVE_WIDTH, MAX_ACTIVE_WIDTH)]]]],
        dtype=np.float32,
    )
    even_cols = np.asarray(
        [[[[col % 2 == 0 for col in range(MAX_ACTIVE_WIDTH)]]]],
        dtype=np.bool_,
    )
    _init(inits, "alt", alt)
    _init(inits, "even_cols", even_cols)
    _init(inits, "zero_f", np.asarray([0.0], dtype=np.float32))
    _init(inits, "zero", np.zeros((1, 1, 3, MAX_ACTIVE_WIDTH), dtype=np.float32))

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st_c0, en_c0, axes_chw], ["black"]),
            helper.make_node("Slice", [IN_NAME, st_c5, en_c5, axes_chw], ["gray_f"]),
            helper.make_node("Slice", ["black", st_tail, en_tail, axes_hw], ["black_tail"]),
            helper.make_node("Slice", ["gray_f", st_tail, en_tail, axes_hw], ["gray_tail"]),
            helper.make_node("Add", ["black_tail", "gray_tail"], ["active_cols"]),
            helper.make_node("Mul", ["active_cols", "alt"], ["alt_active"]),
            helper.make_node("ReduceSum", ["alt_active"], ["width_parity_f"], axes=[1, 2, 3], keepdims=0),
            helper.make_node("Greater", ["width_parity_f", "zero_f"], ["width_odd"]),
            helper.make_node("Xor", ["width_odd", "even_cols"], ["keep_cols"]),
            helper.make_node("Where", ["keep_cols", "gray_f", "zero"], ["gray"]),
            helper.make_node("Sub", ["gray_f", "gray"], ["green"]),
            helper.make_node(
                "Concat",
                ["black", "zero", "zero", "green", "zero", "gray"],
                ["top6"],
                axis=1,
            ),
            helper.make_node("Pad", ["top6"], [OUT_NAME], pads=[0, 0, 0, 0, 0, C - 6, N - 3, N - MAX_ACTIVE_WIDTH]),
        ]
    )

    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task332",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _mismatch_coords(actual: np.ndarray, expected: np.ndarray) -> list[tuple[int, int, int]]:
    diff = (actual > 0.0) != (expected > 0.0)
    return [tuple(map(int, coord)) for coord in np.argwhere(diff[0])]


def validate_model(model: onnx.ModelProto, data: dict[str, Any]) -> int:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    failures = 0
    for split, index, example in _examples(data):
        expected_grid = solve(example["input"])
        if expected_grid != example["output"]:
            raise AssertionError(f"reference mismatch on {split}[{index}]")
        input_tensor = convert_to_numpy(example, "input")
        expected_tensor = convert_to_numpy(example, "output")
        if input_tensor is None or expected_tensor is None:
            continue
        actual = session.run([OUT_NAME], {IN_NAME: input_tensor})[0]
        if not np.array_equal(actual > 0.0, expected_tensor > 0.0):
            failures += 1
            coords = _mismatch_coords(actual, expected_tensor)[:40]
            print(f"{split}[{index}] mismatch coords (channel,row,col): {coords}")
    return failures


def main() -> None:
    data = _load_task()
    _validate_rule(data)
    model = build_model()
    failures = validate_model(model, data)
    if failures:
        raise SystemExit(f"{failures} examples failed")
    onnx.save(model, BEST_PATH)
    result = score_file(BEST_PATH)
    print(
        f"{BEST_PATH.name}: valid={result['valid']} memory={result['memory']} "
        f"params={result['params']} cost={result['cost']} score={result['score']}"
    )


if __name__ == "__main__":
    main()
