"""ONNX for ARC task352: paint blue 3x3 halos around red cells.

Task rule: preserve the input grid shape.  Every red cell (2) expands into a
solid blue (1) square with Chebyshev radius 1, clipped by the grid border.  The
original non-black cells are copied back after painting, so red centers and all
other colored marker cells keep their input colors.  Padding outside the task
grid remains all-zero in the 30x30 NeuroGolf one-hot tensor.

ONNX approach: one grouped 3x3 Conv writes the final one-hot output directly.
The black channel is active only for valid black cells with no neighboring red;
the blue channel is active for original blue cells or valid black cells within
the 3x3 red neighborhood.  Channels 2..9 are copied from the input center tap.
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

from score_model import score_file  # noqa: E402

TASK_ID = "task352"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task352.onnx"
DATA_PATH = ROOT / "data" / "task352.json"

C = 10
H = W = 30
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10
GROUPS = 2
GROUP_CHANNELS = C // GROUPS


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference solver for local JSON validation."""
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(arr)
    rows, cols = arr.shape
    for r, c in np.argwhere(arr == 2):
        r0, r1 = max(0, int(r) - 1), min(rows, int(r) + 2)
        c0, c1 = max(0, int(c) - 1), min(cols, int(c) + 2)
        out[r0:r1, c0:c1] = 1
    out[arr != 0] = arr[arr != 0]
    return out


def _f32(arr: Any, name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name)


def _conv_kernel() -> np.ndarray:
    """Grouped Conv weights for the full task rule."""
    weight = np.zeros((C, GROUP_CHANNELS, 3, 3), dtype=np.float32)

    # Output channels 0..4 read input channels 0..4.  Red is local channel 2.
    weight[0, 0, 1, 1] = 1.0
    weight[0, 2, :, :] = -1.0

    weight[1, 2, :, :] = 1.0
    weight[1, 1, 1, 1] = 10.0
    weight[1, 0, 1, 1] = 9.0

    for color in range(2, 5):
        weight[color, color, 1, 1] = 1.0

    # Output channels 5..9 read input channels 5..9 in the second group.
    for color in range(5, C):
        weight[color, color - GROUP_CHANNELS, 1, 1] = 1.0

    return weight


def build_model() -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    bias = np.zeros((C,), dtype=np.float32)
    bias[1] = -9.0
    nodes = [
        helper.make_node(
            "Conv",
            [IN_NAME, "W", "B"],
            [OUT_NAME],
            group=GROUPS,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
    ]
    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [x_info],
        [y_info],
        initializer=[_f32(_conv_kernel(), "W"), _f32(bias, "B")],
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


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            ref = solve(ex["input"])
            expected_grid = np.asarray(ex["output"], dtype=np.int64)
            if not np.array_equal(ref, expected_grid):
                print(f"reference mismatch: {split}[{idx}]")
                bad += 1
                continue

            expected = _grid_to_onehot(ex["output"]) > 0.0
            pred = _run_onnx(model, _grid_to_onehot(ex["input"])) > 0.0
            if not np.array_equal(pred, expected):
                print(f"onnx mismatch: {split}[{idx}]")
                bad += 1
    return bad


def main() -> None:
    model = build_model()
    bad = validate_json(model)
    assert bad == 0, f"{bad} JSON examples failed"
    onnx.save(model, BEST_PATH)
    print(score_file(BEST_PATH))


if __name__ == "__main__":
    main()
