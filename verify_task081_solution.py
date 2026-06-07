"""Verify the task081 ONNX model on official and random 7x7 examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from score_model import convert_to_numpy, score_file

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "all" / "task081.onnx"
TASK_PATH = ROOT / "data" / "task081.json"


def solve(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for r in range(6):
        for c in range(6):
            block = grid[r : r + 2, c : c + 2]
            if np.count_nonzero(block == 8) == 3 and np.count_nonzero(block == 0) == 1:
                rr, cc = np.argwhere(block == 0)[0]
                out[r + rr, c + cc] = 1
    return out


def encode(grid: np.ndarray) -> np.ndarray:
    arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            arr[0, int(grid[r, c]), r, c] = 1.0
    return arr


def decode(pred: np.ndarray) -> np.ndarray:
    active = pred[0, :, :7, :7] > 0.0
    if not np.all(active.sum(axis=0) == 1):
        raise AssertionError("invalid one-hot output inside 7x7 crop")
    if np.any(pred[0, :, 7:, :] > 0.0) or np.any(pred[0, :, :, 7:] > 0.0):
        raise AssertionError("model wrote non-zero values outside the 7x7 crop")
    return active.argmax(axis=0).astype(np.int64)


def run_one(session: ort.InferenceSession, grid: np.ndarray) -> None:
    pred = session.run(["output"], {"input": encode(grid)})[0]
    got = decode(pred)
    expected = solve(grid)
    if not np.array_equal(got, expected):
        raise AssertionError(f"mismatch\ninput:\n{grid}\nexpected:\n{expected}\ngot:\n{got}")


def official_examples() -> list[np.ndarray]:
    data = json.loads(TASK_PATH.read_text())
    grids: list[np.ndarray] = []
    for split in ("train", "test", "arc-gen"):
        for example in data.get(split, []):
            grids.append(np.asarray(example["input"], dtype=np.int64))
    return grids


def orientation_examples() -> list[np.ndarray]:
    examples: list[np.ndarray] = []
    missing = [(0, 0), (0, 1), (1, 0), (1, 1)]
    starts = [(0, 0), (0, 5), (5, 0), (5, 5)]
    for miss_r, miss_c in missing:
        for r, c in starts:
            grid = np.zeros((7, 7), dtype=np.int64)
            grid[r : r + 2, c : c + 2] = 8
            grid[r + miss_r, c + miss_c] = 0
            examples.append(grid)
    return examples


def random_examples(seed: int, count: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    examples: list[np.ndarray] = []
    missing = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for _ in range(count):
        grid = np.zeros((7, 7), dtype=np.int64)
        for _ in range(rng.integers(1, 9)):
            r = int(rng.integers(0, 6))
            c = int(rng.integers(0, 6))
            miss_r, miss_c = missing[int(rng.integers(0, 4))]
            grid[r : r + 2, c : c + 2] = 8
            grid[r + miss_r, c + miss_c] = 0
        examples.append(grid)
    return examples


def verify_against_json_outputs(session: ort.InferenceSession) -> int:
    data = json.loads(TASK_PATH.read_text())
    checked = 0
    for split in ("train", "test", "arc-gen"):
        for example in data.get(split, []):
            inp = convert_to_numpy(example, "input")
            expected = np.asarray(example["output"], dtype=np.int64)
            pred = session.run(["output"], {"input": inp})[0]
            got = decode(pred)
            if not np.array_equal(got, expected):
                raise AssertionError(f"{split} example mismatch")
            checked += 1
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--random", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=81)
    args = parser.parse_args()

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    official_count = verify_against_json_outputs(session)

    generated = official_examples() + orientation_examples() + random_examples(args.seed, args.random)
    for grid in generated:
        run_one(session, grid)

    result = score_file(args.model)
    print(f"official examples: {official_count}")
    print(f"generated checks:  {len(generated)}")
    print(f"memory:            {result['memory']}")
    print(f"params:            {result['params']}")
    print(f"cost:              {result['cost']}")
    print(f"score:             {result['score']:.6f}")


if __name__ == "__main__":
    main()
