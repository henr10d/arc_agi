"""Validate task068 ONNX on prompt examples and random singleton cases."""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import GRID_SHAPE, sanitize_model  # noqa: E402


def encode(grid: list[list[int]]) -> np.ndarray:
    arr = np.zeros(GRID_SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            arr[0, color, r, c] = 1.0
    return arr


def expected(grid: list[list[int]]) -> np.ndarray:
    counts = {color: sum(row.count(color) for row in grid) for color in range(1, 10)}
    singleton_colors = [color for color, count in counts.items() if count == 1]
    assert len(singleton_colors) == 1, singleton_colors
    color = singleton_colors[0]
    [(rr, cc)] = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == color]

    out = [[0 for _ in range(10)] for _ in range(10)]
    for r in range(max(0, rr - 1), min(10, rr + 2)):
        for c in range(max(0, cc - 1), min(10, cc + 2)):
            out[r][c] = 2
    out[rr][cc] = color
    return encode(out)


def make_case(color: int, row: int, col: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    grid = [[0 for _ in range(10)] for _ in range(10)]
    grid[row][col] = color
    for other in range(1, 10):
        if other == color:
            continue
        for _ in range(rng.randint(0, 4)):
            placed = 0
            while placed < 2:
                r = rng.randrange(10)
                c = rng.randrange(10)
                if grid[r][c] == 0:
                    grid[r][c] = other
                    placed += 1
    return grid


def main() -> None:
    model_path = ROOT / "all" / "task068.onnx"
    model = sanitize_model(copy.deepcopy(onnx.load(model_path)))
    assert model is not None
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])

    cases = [
        make_case(4, 6, 1, 1),
        make_case(6, 2, 7, 2),
    ]
    for seed in range(100):
        color = 1 + seed % 9
        row = (seed * 7) % 10
        col = (seed * 3) % 10
        cases.append(make_case(color, row, col, seed + 100))

    for idx, grid in enumerate(cases):
        pred = sess.run(["output"], {"input": encode(grid)})[0]
        want = expected(grid)
        if not np.array_equal(pred > 0.0, want > 0.0):
            raise AssertionError(f"case {idx} failed")

    print(f"validated {len(cases)} task068 cases")


if __name__ == "__main__":
    main()
