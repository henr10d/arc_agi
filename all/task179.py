"""Minimal ONNX for ARC task179 using Kaggle one-hot I/O.

Task rule: transpose the 3x3 input grid across its main diagonal. All local
train/test/arc-gen examples are 3x3. The competition tensor is already padded
to 30x30, so a single Transpose over the spatial axes produces the required
visible output and leaves padding all-zero. Because the Transpose writes
directly to graph output, official scoring excludes that tensor and the model
has zero memory and zero parameters.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "task179"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task179.onnx"
DATA_PATH = ROOT / "data" / "task179.json"

C = 10
H = W = 30
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
IR_VERSION = 10
OPSET = 10


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference implementation: transpose the visible 3x3 grid."""
    return np.asarray(grid, dtype=np.int64).T


def grid_to_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def build_model() -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    node = helper.make_node("Transpose", [IN_NAME], [OUT_NAME], perm=[0, 1, 3, 2])
    graph = helper.make_graph([node], "task179_transpose", [x_info], [y_info])
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def validate_model(model: onnx.ModelProto) -> None:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(data.get(split, [])):
            expected = grid_to_onehot(example["output"])
            actual = session.run([OUT_NAME], {IN_NAME: grid_to_onehot(example["input"])})[0]
            if not np.array_equal(actual > 0.0, expected > 0.0):
                raise AssertionError(f"{TASK_ID} failed {split}[{index}]")


def main() -> None:
    model = build_model()
    validate_model(model)
    onnx.save(model, BEST_PATH)
    print(f"wrote {BEST_PATH}")


if __name__ == "__main__":
    main()
