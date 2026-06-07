"""ONNX solution for NeuroGolf task372 using Kaggle one-hot I/O.

Task rule: the input is always an 11x11 grid with a full gray divider row at
row 5. Ignore the divider and vertically compress the grid by overlaying each
top row with the corresponding row below the divider:

    output row 0 = input rows 0 and 6
    output row 1 = input rows 1 and 7
    output row 2 = input rows 2 and 8
    output row 3 = input rows 3 and 9
    output row 4 = input rows 4 and 10

Columns and colors are preserved. The local examples have no conflicting
non-black colors at the same compressed location, so matching color channels can
be added directly. The ONNX graph crops the 11x11 input with a negative Pad,
then uses a depthwise dilated convolution to add row r and row r+6 for every
channel. Background channel 0 gets a -1 bias, making it positive only when both
source cells are background; padding outside the 5x11 output remains all-zero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TASK_ID = "task372"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task372.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
IN_H = IN_W = 11
OUT_H = 5
OUT_W = 11
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
IR_VERSION = 10
OPSET = 10


def solve(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    """Reference transform on raw integer grids."""
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros((OUT_H, OUT_W), dtype=np.int64)
    for r in range(OUT_H):
        top = arr[r, :OUT_W]
        bottom = arr[r + 6, :OUT_W]
        out[r] = np.where(top != 0, top, bottom)
    return out


def build_onnx_model() -> onnx.ModelProto:
    """Build the compact opset-10 graph used for submission."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    weights = np.zeros((C, 1, 2, 1), dtype=np.float32)
    weights[:, 0, 0, 0] = 1.0
    weights[:, 0, 1, 0] = 1.0
    bias = np.zeros((C,), dtype=np.float32)
    bias[0] = -1.0
    inits.append(numpy_helper.from_array(weights, name="weights"))
    inits.append(numpy_helper.from_array(bias, name="bias"))

    nodes.extend(
        [
            helper.make_node(
                "Pad",
                [IN_NAME],
                ["crop"],
                pads=[0, 0, 0, 0, 0, 0, IN_H - H, OUT_W - W],
            ),
            helper.make_node(
                "Conv",
                ["crop", "weights", "bias"],
                ["out5"],
                group=C,
                dilations=[OUT_H + 1, 1],
            ),
            helper.make_node(
                "Pad",
                ["out5"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - OUT_H, W - OUT_W],
            ),
        ]
    )

    graph = helper.make_graph(nodes, "task372", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _grid_to_onehot(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            color = int(arr[r, c])
            if 0 <= color < C:
                out[0, color, r, c] = 1.0
    return out


def validate_model(model: onnx.ModelProto) -> tuple[bool, str]:
    try:
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        return False, f"ORT load failed: {exc}"

    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            inp = _grid_to_onehot(example["input"])
            expected = _grid_to_onehot(example["output"])
            pred = sess.run([OUT_NAME], {IN_NAME: inp})[0]
            if not np.array_equal(pred > 0.0, expected > 0.0):
                return False, f"{split}[{idx}] mismatch"
    return True, "ok"


def main() -> None:
    model = build_onnx_model()
    ok, message = validate_model(model)
    if not ok:
        raise SystemExit(message)
    onnx.save(model, BEST_PATH)
    print(f"wrote {BEST_PATH}")


if __name__ == "__main__":
    main()
