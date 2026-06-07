"""ONNX for ARC task356: complete cyan row and column spans.

Task rule: the 10x10 input contains sparse cyan markers. For every original
row with at least two markers, fill the inclusive span from its leftmost to
rightmost marker. For every original column with at least two markers, fill the
inclusive span from its topmost to bottommost marker. Keep all original markers;
newly filled cells do not create additional spans. Background remains 0.

ONNX: slice the cyan channel inside the 10x10 grid, find each row/column's
first marker with ArgMax and last marker with ReduceMax over coordinate values
using -1 sentinels for empty spans, and emit a compact 10x10 one-hot result
before padding to the required 30x30 I/O.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task356"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
GRID = 10
COLOR = 8
PAD = H - GRID
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference implementation of the original-marker row/column span rule."""
    g = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(g)
    markers = g > 0
    out[markers] = g[markers]
    color = int(g[markers][0]) if np.any(markers) else COLOR

    for r in range(g.shape[0]):
        cols = np.flatnonzero(markers[r])
        if len(cols) >= 2:
            out[r, int(cols.min()) : int(cols.max()) + 1] = color

    for c in range(g.shape[1]):
        rows = np.flatnonzero(markers[:, c])
        if len(rows) >= 2:
            out[int(rows.min()) : int(rows.max()) + 1, c] = color

    return out


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    st8 = _i64(inits, [0, COLOR, 0, 0], "st8")
    en8 = _i64(inits, [1, COLOR + 1, GRID, GRID], "en8")
    zero_f = _f32(inits, [0.0], "zero_f")
    neg_one = _f32(inits, [-1.0], "neg_one")
    bg_vec = np.zeros((1, C, 1, 1), dtype=np.float32)
    bg_vec[0, 0, 0, 0] = 1.0
    cyan_vec = np.zeros((1, C, 1, 1), dtype=np.float32)
    cyan_vec[0, COLOR, 0, 0] = 1.0
    bg_template = _f32(inits, bg_vec, "bg_template")
    cyan_template = _f32(inits, cyan_vec, "cyan_template")
    idx = np.arange(GRID)
    col_idx = idx.reshape(1, 1, 1, GRID)
    row_idx = idx.reshape(1, 1, GRID, 1)
    col_f = _f32(inits, col_idx.astype(np.float32), "col_f")
    row_f = _f32(inits, row_idx.astype(np.float32), "row_f")
    col_minus = _f32(inits, (col_idx - 1).astype(np.float32), "col_minus")
    row_minus = _f32(inits, (row_idx - 1).astype(np.float32), "row_minus")
    col_plus = _i64(inits, (col_idx + 1).astype(np.int64), "col_plus")
    row_plus = _i64(inits, (row_idx + 1).astype(np.int64), "row_plus")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st8, en8, axes4], ["cyan"]),
            helper.make_node("Greater", ["cyan", zero_f], ["cyan_b"]),
            helper.make_node("ArgMax", ["cyan"], ["row_first"], axis=3, keepdims=1),
            helper.make_node("Where", ["cyan_b", col_f, neg_one], ["row_col_values"]),
            helper.make_node("ReduceMax", ["row_col_values"], ["row_last"], axes=[3], keepdims=1),
            helper.make_node("Greater", ["col_plus", "row_first"], ["row_after_first"]),
            helper.make_node("Greater", ["row_last", col_minus], ["row_before_last"]),
            helper.make_node("And", ["row_after_first", "row_before_last"], ["row_span"]),
            helper.make_node("ArgMax", ["cyan"], ["col_first"], axis=2, keepdims=1),
            helper.make_node("Where", ["cyan_b", row_f, neg_one], ["col_row_values"]),
            helper.make_node("ReduceMax", ["col_row_values"], ["col_last"], axes=[2], keepdims=1),
            helper.make_node("Greater", ["row_plus", "col_first"], ["col_after_first"]),
            helper.make_node("Greater", ["col_last", row_minus], ["col_before_last"]),
            helper.make_node("And", ["col_after_first", "col_before_last"], ["col_span"]),
            helper.make_node("Or", ["row_span", "col_span"], ["line"]),
            helper.make_node("Where", ["line", cyan_template, bg_template], ["out10"]),
            helper.make_node("Pad", ["out10"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]),
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
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            inp = np.asarray(ex["input"], dtype=np.int64)
            exp = np.asarray(ex["output"], dtype=np.int64)
            ref = solve(inp)
            if not np.array_equal(ref, exp):
                raise AssertionError(f"reference solver mismatch in {split} example {idx}")
            got = _run_onnx(model, _grid_to_onehot(ex["input"]))
            if not np.array_equal(got > 0.0, _grid_to_onehot(ex["output"]) > 0.0):
                bad += 1
                continue
            pred = _onehot_to_grid(got)[: exp.shape[0], : exp.shape[1]]
            if not np.array_equal(pred, exp):
                bad += 1
    return bad


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    bad = validate_json(model)
    assert bad == 0, f"{bad} JSON examples failed"
    print(score_file(BEST_PATH))


if __name__ == "__main__":
    main()
