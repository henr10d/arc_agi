"""Minimal ONNX for ARC task239: sorted whole-grid color histogram.

Task rule: count each non-black color in the input grid, sort colors by count
descending (ties are stable/top-left, but the provided examples have no count
ties), and draw one vertical output column per color. Each column is filled from
the top with that color repeated by its count; the rectangular remainder is
black. Output height is the largest color count and output width is the number
of non-black colors.

ONNX: count colors 1-9 on the active 4x4 input crop, use TopK to obtain the
five largest color counts, render all histogram columns by broadcasting on a
compact bool 12x5 canvas, then cast once and pad to the fixed NeuroGolf one-hot
I/O shape.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task239"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
NC = 9
H = W = 30
SH = SW = 4
OH = 12
OW = 5
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
    g = np.asarray(grid, dtype=np.int64)
    flat = [int(v) for v in g.ravel() if int(v) != 0]
    counts = Counter(flat)
    first = {}
    for i, color in enumerate(flat):
        first.setdefault(color, i)
    order = sorted(counts, key=lambda color: (-counts[color], first[color]))
    out = np.zeros((max(counts.values()), len(order)), dtype=np.int64)
    for col, color in enumerate(order):
        out[: counts[color], col] = color
    return out


def _grid_to_onehot(grid: List[List[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    arr = np.asarray(onehot).reshape(C, H, W)
    active = arr > 0.0
    any_active = active.any(axis=0)
    grid = arr.argmax(axis=0).astype(np.int64)
    grid[~any_active] = 0
    return grid


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    crop_st = _i64(inits, [0, 1, 0, 0], "crop_st")
    crop_en = _i64(inits, [1, C, SH, SW], "crop_en")
    top_k = _i64(inits, [OW], "top_k")
    rows = _f32(inits, np.arange(OH, dtype=np.float32).reshape(1, 1, OH, 1), "rows")
    color_ids = _i64(inits, np.arange(NC, dtype=np.int64).reshape(1, NC, 1, 1), "color_ids")
    zero_scalar = _f32(inits, np.asarray([0.0], dtype=np.float32), "zero_scalar")
    pads = [0, 0, 0, 0, 0, 0, H - OH, W - OW]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, crop_st, crop_en], ["fg_crop"]),
            helper.make_node("ReduceSum", ["fg_crop"], ["counts"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMax", ["counts"], ["max_count"], axes=[1], keepdims=1),
            helper.make_node("Less", [rows, "max_count"], ["active_rows"]),
            helper.make_node("TopK", ["counts", top_k], ["top_values_ch", "top_idx_ch"], axis=1),
            helper.make_node("Transpose", ["top_values_ch"], ["top_values"], perm=[0, 2, 3, 1]),
            helper.make_node("Transpose", ["top_idx_ch"], ["top_idx"], perm=[0, 2, 3, 1]),
            helper.make_node("Equal", [color_ids, "top_idx"], ["selected"]),
            helper.make_node("Less", [rows, "top_values"], ["keep_rows"]),
            helper.make_node("And", ["selected", "keep_rows"], ["fg9_b"]),
            helper.make_node("Greater", ["top_values", zero_scalar], ["has_col"]),
            helper.make_node("And", ["active_rows", "has_col"], ["active_cols"]),
            helper.make_node("Not", ["keep_rows"], ["not_keep_rows"]),
            helper.make_node("And", ["active_cols", "not_keep_rows"], ["bg_b"]),
            helper.make_node("Concat", ["bg_b", "fg9_b"], ["out10_b"], axis=1),
            helper.make_node("Cast", ["out10_b"], ["out10"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out10"], [OUT_NAME], pads=pads),
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
        for idx, ex in enumerate(data[split]):
            expected = np.asarray(ex["output"], dtype=np.int64)
            oh = _grid_to_onehot(ex["input"])
            pred = _onehot_to_grid(_run_onnx(model, oh))[: expected.shape[0], : expected.shape[1]]
            if not np.array_equal(pred, expected):
                print(f"mismatch {split} {idx}: got\\n{pred}\\nexpected\\n{expected}")
                bad += 1
                break
    return bad


def main() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            ref = solve(np.asarray(ex["input"], dtype=np.int64))
            exp = np.asarray(ex["output"], dtype=np.int64)
            assert np.array_equal(ref, exp), f"reference mismatch {split} {idx}"

    model = build_model()
    onnx.save(model, BEST_PATH)
    bad = validate_json(model)
    assert bad == 0, f"{bad} ONNX mismatches"
    score_file(BEST_PATH)


if __name__ == "__main__":
    main()
