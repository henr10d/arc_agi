"""Verify task112.onnx against all local task112 examples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "task112.onnx"
DATA_PATH = ROOT / "data" / "task112.json"
SHAPE = (1, 10, 30, 30)


def one_hot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def main() -> None:
    data = json.loads(DATA_PATH.read_text())
    session = ort.InferenceSession(MODEL_PATH.read_bytes(), providers=["CPUExecutionProvider"])

    total = 0
    passed = 0
    first_fail: tuple[str, int] | None = None
    for split in ("train", "test", "arc-gen"):
        split_total = 0
        split_passed = 0
        for idx, example in enumerate(data.get(split, [])):
            pred = session.run(["output"], {"input": one_hot(example["input"])})[0]
            expected = one_hot(example["output"])
            ok = np.array_equal(pred > 0.0, expected > 0.0)
            split_total += 1
            total += 1
            if ok:
                split_passed += 1
                passed += 1
            elif first_fail is None:
                first_fail = (split, idx)
        print(f"{split}: {split_passed}/{split_total}")

    print(f"exact-match accuracy: {passed}/{total}")
    if first_fail is not None:
        split, idx = first_fail
        raise SystemExit(f"first failure: {split}[{idx}]")


if __name__ == "__main__":
    main()
