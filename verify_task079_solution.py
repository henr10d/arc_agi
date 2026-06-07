"""Verify task079 solution.onnx against the local task examples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "solution.onnx"
TASK_JSON = ROOT / "data" / "task079.json"


def encode(grid: list[list[int]]) -> np.ndarray:
    arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            arr[0, int(color), r, c] = 1.0
    return arr


def main() -> None:
    task = json.loads(TASK_JSON.read_text(encoding="utf-8"))
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    checked = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(task.get(split, [])):
            pred = session.run(["output"], {"input": encode(ex["input"])})[0]
            rows = len(ex["output"])
            cols = len(ex["output"][0])
            pred_grid = np.argmax(pred[0, :, :rows, :cols], axis=0)
            expected = np.asarray(ex["output"], dtype=np.int64)
            if not np.array_equal(pred_grid, expected):
                raise SystemExit(f"mismatch on {split} {idx}: {pred_grid.tolist()} != {expected.tolist()}")
            checked += 1
    print(f"task079 solution.onnx passed {checked} examples")


if __name__ == "__main__":
    main()
