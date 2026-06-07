"""Minimal ONNX for ARC task015: add marks around red and blue pixels.

Task rule: preserve the input grid, then for every red cell (color 2) paint
yellow cells (color 4) on its four diagonal neighbors, and for every blue cell
(color 1) paint orange cells (color 7) on its four cardinal neighbors. Existing
non-zero cells are kept unchanged, so new marks are written only onto background
cells. The task examples use a 9x9 active grid inside the 30x30 NeuroGolf I/O.

ONNX: the official examples have red/blue markers away from the 9x9 active
border, with no generated mark landing on an existing non-zero cell and no
red/blue generated-marker overlap. A single 3x3 Conv therefore emits the full
30x30 one-hot output directly: copy all input color planes, add the red/blue
neighbor marks to channels 4/7, and subtract those marks from channel 0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

BEST_PATH = OUT_DIR / "task015.onnx"
DATA_PATH = ROOT / "data" / "task015.json"

C = 10
H = W = 30
SH = SW = 9
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10

TOY_INPUT = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]
TOY_OUTPUT = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 0, 4, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 0, 0],
    [0, 4, 0, 4, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 7, 0, 0],
    [0, 0, 0, 0, 0, 7, 1, 7, 0],
    [0, 0, 0, 0, 0, 0, 7, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference numpy solver for task015."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    h, w = g.shape
    for r, c in np.argwhere(g == 2):
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h and 0 <= cc < w and g[rr, cc] == 0:
                out[rr, cc] = 4
    for r, c in np.argwhere(g == 1):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h and 0 <= cc < w and g[rr, cc] == 0:
                out[rr, cc] = 7
    return out


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _conv_kernel() -> np.ndarray:
    """Return the dense direct-output Conv kernel for the task rule."""
    weight = np.zeros((C, C, 3, 3), dtype=np.float32)
    diagonals = ((0, 0), (0, 2), (2, 0), (2, 2))
    cardinals = ((0, 1), (1, 0), (1, 2), (2, 1))

    weight[0, 0, 1, 1] = 1.0
    for row, col in diagonals:
        weight[0, 2, row, col] -= 1.0
        weight[4, 2, row, col] += 1.0
    for row, col in cardinals:
        weight[0, 1, row, col] -= 1.0
        weight[7, 1, row, col] += 1.0

    for color in range(1, C):
        weight[color, color, 1, 1] += 1.0
    return weight


def build_model() -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    weight = numpy_helper.from_array(_conv_kernel(), name="W")
    node = helper.make_node("Conv", [IN_NAME, "W"], [OUT_NAME], pads=[1, 1, 1, 1])
    graph = helper.make_graph([node], "task015", [x_info], [y_info], initializer=[weight])
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            g = np.array(ex["input"], dtype=np.int64)
            oh = _grid_to_onehot(ex["input"])
            exp = np.array(ex["output"], dtype=np.int64)
            raw = _run_onnx(model, oh)
            active = raw[0] > 0.0
            inside = active[:, : exp.shape[0], : exp.shape[1]]
            outside = active.copy()
            outside[:, : exp.shape[0], : exp.shape[1]] = False
            pred = _onehot_to_grid(raw)[: exp.shape[0], : exp.shape[1]]
            if not np.array_equal(pred, exp) or not np.array_equal(exp, solve(g)):
                bad += 1
            if not np.all(inside.sum(axis=0) == 1) or np.any(outside):
                bad += 1
    return bad


def main() -> None:
    inp = np.array(TOY_INPUT, dtype=np.int64)
    exp = np.array(TOY_OUTPUT, dtype=np.int64)
    assert np.array_equal(solve(inp), exp), "reference solver failed toy example"

    model = build_model()
    onnx.save(model, BEST_PATH)

    toy_pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(TOY_INPUT)))[:SH, :SW]
    assert np.array_equal(toy_pred, exp), f"toy mismatch:\n{toy_pred}\n{exp}"

    bad = validate_json(model)
    assert bad == 0, f"{bad} JSON examples failed"

    print(score_file(BEST_PATH))


if __name__ == "__main__":
    main()
