"""Minimal ONNX for ARC task176: row-wise periodic yellow fill.

Task rule: the grid is three rows high and variable-width, with the provided
examples spanning widths 5 through 25. Red cells stay fixed. In each row,
black cells on a row-specific periodic lattice become yellow:
row 0 marks columns 5,6,7 modulo 12; row 1 marks columns 0 modulo 6;
row 2 marks columns 0,1,11 modulo 12. Marks are clipped to the actual grid
width by multiplying with the black input channel, and padding remains all-zero.

ONNX approach: work only on the first 3 rows and first 25 columns, concatenate
channels 0..4, then use the final unscored Pad to add rows, trailing columns,
and always-empty channels 5..9.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task176"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
RH = 3
CW = 25
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def yellow_pattern() -> np.ndarray:
    pat = np.zeros((1, 1, RH, CW), dtype=np.float32)
    for col in range(CW):
        if col % 12 in (5, 6, 7):
            pat[0, 0, 0, col] = 1.0
        if col % 6 == 0:
            pat[0, 0, 1, col] = 1.0
        if col % 12 in (0, 1, 11):
            pat[0, 0, 2, col] = 1.0
    return pat


def solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    h, w = g.shape
    assert h == RH
    for col in range(w):
        if col % 12 in (5, 6, 7) and out[0, col] == 0:
            out[0, col] = 4
        if col % 6 == 0 and out[1, col] == 0:
            out[1, col] = 4
        if col % 12 in (0, 1, 11) and out[2, col] == 0:
            out[2, col] = 4
    return out


def _grid_to_onehot(grid: List[List[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes_chw = _i64(inits, [1, 2, 3], "axes_chw")
    black_st = _i64(inits, [0, 0, 0], "black_st")
    black_en = _i64(inits, [1, RH, CW], "black_en")
    red_st = _i64(inits, [2, 0, 0], "red_st")
    red_en = _i64(inits, [3, RH, CW], "red_en")
    ypat = _f32(inits, yellow_pattern(), "ypat")
    zero = _f32(inits, np.zeros((1, 1, RH, CW), dtype=np.float32), "zero")
    pads = [0, 0, 0, 0, 0, C - 5, H - RH, W - CW]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, black_st, black_en, axes_chw], ["black_in"]),
            helper.make_node("Slice", [IN_NAME, red_st, red_en, axes_chw], ["red"]),
            helper.make_node("Mul", ["black_in", ypat], ["yellow"]),
            helper.make_node("Sub", ["black_in", "yellow"], ["black"]),
            helper.make_node(
                "Concat",
                ["black", zero, "red", zero, "yellow"],
                ["out5"],
                axis=1,
            ),
            helper.make_node("Pad", ["out5"], [OUT_NAME], pads=pads),
        ]
    )

    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits)
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
            if g.shape[0] > H or g.shape[1] > W:
                continue
            pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(ex["input"])))[: g.shape[0], : g.shape[1]]
            exp = np.array(ex["output"], dtype=np.int64)
            if not np.array_equal(pred, exp) or not np.array_equal(solve(g), exp):
                bad += 1
    return bad


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    bad = validate_json(model)
    assert bad == 0, f"{bad} examples failed validation"

    result = score_file(BEST_PATH)
    assert result["valid"], result["error"]
    print(
        f"{BEST_PATH.name}: memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']:.6f}"
    )


if __name__ == "__main__":
    main()
