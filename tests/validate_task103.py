"""Validate task103 ONNX against symmetry examples and bundled task data.

Task rule: return blue for 3x3 red/black patterns that are symmetric across
both vertical and horizontal axes; otherwise return orange.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SHAPE = (1, 10, 30, 30)


def grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def expected_onehot(color: int) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    out[0, int(color), 0, 0] = 1.0
    return out


def solve(grid: list[list[int]]) -> int:
    arr = np.asarray(grid, dtype=np.int64)
    return 1 if np.array_equal(arr, arr[:, ::-1]) and np.array_equal(arr, arr[::-1, :]) else 7


def load_cases() -> list[tuple[str, list[list[int]], int]]:
    cases: list[tuple[str, list[list[int]], int]] = [
        ("shown_x", [[2, 0, 2], [0, 2, 0], [2, 0, 2]], 1),
        ("corners", [[2, 0, 2], [0, 0, 0], [2, 0, 2]], 1),
        ("diamond", [[0, 2, 0], [2, 0, 2], [0, 2, 0]], 1),
        ("diagonal_only", [[2, 0, 0], [0, 2, 0], [0, 0, 2]], 7),
        ("asymmetric", [[0, 0, 2], [2, 0, 0], [0, 2, 0]], 7),
    ]
    data_path = ROOT / "data" / "task103.json"
    if data_path.is_file():
        data = json.loads(data_path.read_text(encoding="utf-8"))
        for split in ("train", "test", "arc-gen"):
            for index, example in enumerate(data.get(split, [])):
                cases.append((f"{split}_{index}", example["input"], int(example["output"][0][0])))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate task103 ONNX output.")
    parser.add_argument("model", nargs="?", type=Path, default=ROOT / "all" / "task103.onnx")
    args = parser.parse_args()

    model = onnx.load(str(args.model))
    onnx.checker.check_model(model)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])

    for name, grid, expected_color in load_cases():
        reference_color = solve(grid)
        if reference_color != expected_color:
            raise AssertionError(f"{name}: data label {expected_color} disagrees with symmetry rule {reference_color}")
        pred = session.run(["output"], {"input": grid_to_onehot(grid)})[0]
        expected = expected_onehot(expected_color)
        if not np.array_equal(pred > 0.0, expected > 0.0):
            active = np.argwhere(pred > 0.0).tolist()
            raise AssertionError(f"{name}: expected color {expected_color}, got active entries {active}")

    print(f"passed {len(load_cases())} task103 cases")


if __name__ == "__main__":
    main()
