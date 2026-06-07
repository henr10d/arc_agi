"""Minimal ONNX for ARC task337: swap gray and azure cells.

Task rule: the grid size and all non-{5, 8} colors are unchanged. Every gray
cell (color 5) becomes azure/teal (color 8), and every azure/teal cell becomes
gray. The examples use small 3x3, 4x4, and 5x5 grids, but the same color swap
is valid over the full padded 30x30 competition tensor. Padding outside the JSON
grid remains all-zero in NeuroGolf one-hot I/O.

ONNX: because inputs are already one-hot encoded as [1, 10, 30, 30], the whole
transformation is a channel permutation that swaps channels 5 and 8. A one-node
Gather keeps memory at zero; zero-parameter Split/Concat and cropped variants
realize intermediate tensors and score worse.
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

TASK_ID = "task337"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, 10, 30, 30]
OPSET = 10
IR_VERSION = 10
GRAY = 5
AZURE = 8


def solve_grid(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    return np.where(arr == GRAY, AZURE, np.where(arr == AZURE, GRAY, arr))


def build_model() -> onnx.ModelProto:
    perm = np.asarray([0, 1, 2, 3, 4, 8, 6, 7, 5, 9], dtype=np.int64)
    inits = [numpy_helper.from_array(perm, name="perm")]
    nodes = [helper.make_node("Gather", [IN_NAME, "perm"], [OUT_NAME], axis=1)]

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
    onnx.checker.check_model(model, full_check=True)
    return model


def _load_examples() -> list[dict[str, Any]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [
        {"split": split, "idx": idx, "example": example}
        for split in ("train", "test", "arc-gen")
        for idx, example in enumerate(data.get(split, []))
    ]


def _run_model(model: onnx.ModelProto, arr: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run([OUT_NAME], {IN_NAME: arr})[0]


def validate_examples(model: onnx.ModelProto) -> dict[str, int]:
    failures = {"train": 0, "test": 0, "arc-gen": 0}
    for item in _load_examples():
        example = item["example"]
        if not np.array_equal(solve_grid(example["input"]), np.asarray(example["output"], dtype=np.int64)):
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
    model = build_model()
    onnx.save(model, BEST_PATH)

    failures = validate_examples(model)
    for split in ("train", "test", "arc-gen"):
        bad = failures[split]
        print(f"{TASK_ID}.json {split}: {'PASS' if bad == 0 else f'FAIL ({bad} wrong)'}")

    result = score_file(BEST_PATH)
    if not result["valid"]:
        print(f"saved {BEST_PATH} but score_model marked it invalid: {result['error']}")
        raise SystemExit(1)

    print(
        f"saved {BEST_PATH} cost={result['cost']} "
        f"memory={result['memory']} params={result['params']} "
        f"score={result['score']:.6f}"
    )

    if any(failures.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
