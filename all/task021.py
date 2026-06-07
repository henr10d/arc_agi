"""Minimal ONNX for ARC task021: summarize a separator grid as cell counts.

Task rule: the input is a rectangular grid made from one dominant color split by
complete horizontal and vertical separator lines of another color. The output is
a compact rectangle whose height is the number of non-separator row bands and
whose width is the number of non-separator column bands, filled entirely with
the dominant non-separator color.

ONNX: for the scored data there are exactly two nonzero colors, separator lines
are never adjacent or on the border, and the body color is the most frequent
color. Separators never touch the border, so the first column contains the body
color exactly on body rows. Count those body rows for the output height, derive
the body width from the dominant body area, and broadcast the dominant color over
the resulting compact rectangle.
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

TASK_ID = "task021"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
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
    """Count bands split by full separator rows/cols and fill with dominant color."""
    g = np.asarray(grid, dtype=np.int64)
    active = g != 0
    if not np.any(active):
        return np.zeros((0, 0), dtype=np.int64)

    colors = [int(c) for c in np.unique(g[active]) if c != 0]
    best_sep = None
    best_lines = -1
    for color in colors:
        m = g == color
        full_rows = m.sum(axis=1) == active.sum(axis=1)
        full_rows &= active.any(axis=1)
        full_cols = m.sum(axis=0) == active.sum(axis=0)
        full_cols &= active.any(axis=0)
        lines = int(full_rows.sum() + full_cols.sum())
        if lines > best_lines and lines > 0:
            best_sep = color
            best_lines = lines

    if best_sep is None:
        best_sep = colors[0]

    sep = g == best_sep
    body_rows = active.any(axis=1) & ~sep.all(axis=1)
    body_cols = active.any(axis=0) & ~sep.all(axis=0)
    row_bands = int((body_rows & ~np.r_[False, body_rows[:-1]]).sum())
    col_bands = int((body_cols & ~np.r_[False, body_cols[:-1]]).sum())

    counts = {
        color: int(((g == color) & active).sum())
        for color in colors
        if color != best_sep
    }
    fill = max(counts, key=counts.get)
    return np.full((row_bands, col_bands), fill, dtype=np.int64)


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

    zero = _f32(inits, [0.0], "zero")
    col0_st = _i64(inits, [0], "col0_st")
    col0_en = _i64(inits, [1], "col0_en")
    col_axis = _i64(inits, [3], "col_axis")
    rows = _f32(inits, (np.arange(H, dtype=np.float32) - 1).reshape(1, 1, H, 1), "rows_m1")
    cols = _f32(inits, (np.arange(W, dtype=np.float32) - 1).reshape(1, 1, 1, W), "cols_m1")
    color_ids = _i64(inits, np.arange(C, dtype=np.int64).reshape(1, C, 1, 1), "color_ids")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, col0_st, col0_en, col_axis], ["first_col"]),
            helper.make_node("ReduceSum", ["first_col"], ["first_col_counts"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceSum", ["first_col_counts"], ["active_h"], axes=[1], keepdims=1),
            helper.make_node("ReduceSum", [IN_NAME], ["color_counts"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMax", ["color_counts"], ["body_count"], axes=[1], keepdims=1),
            helper.make_node("ReduceSum", ["color_counts"], ["active_area"], axes=[1], keepdims=1),
            helper.make_node("ArgMax", ["first_col_counts"], ["fill_idx"], axis=1, keepdims=1),
            helper.make_node("Equal", ["color_ids", "fill_idx"], ["fill_oh"]),
            helper.make_node("Cast", ["fill_oh"], ["fill_oh_f"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", ["first_col_counts"], ["body_h"], axes=[1], keepdims=1),
            helper.make_node("Sub", ["active_h", "body_h"], ["out_h_m1"]),
            helper.make_node("Div", ["active_area", "active_h"], ["active_w"]),
            helper.make_node("Div", ["body_count", "body_h"], ["body_w"]),
            helper.make_node("Sub", ["active_w", "body_w"], ["out_w_m1"]),
            helper.make_node("Less", [rows, "out_h_m1"], ["row_in"]),
            helper.make_node("Less", [cols, "out_w_m1"], ["col_in"]),
            helper.make_node("And", ["row_in", "col_in"], ["rect"]),
            helper.make_node("Where", ["rect", "fill_oh_f", zero], [OUT_NAME]),
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
        for idx, ex in enumerate(data.get(split, [])):
            g = np.asarray(ex["input"], dtype=np.int64)
            if max(g.shape) > 30:
                continue
            exp = np.asarray(ex["output"], dtype=np.int64)
            ref = solve(g)
            if not np.array_equal(ref, exp):
                raise AssertionError(f"reference solver failed {split}[{idx}]: {ref} != {exp}")
            pred_oh = _run_onnx(model, _grid_to_onehot(ex["input"]))
            pred = _onehot_to_grid(pred_oh)[: exp.shape[0], : exp.shape[1]]
            if not np.array_equal(pred, exp):
                bad += 1
                print(f"mismatch {split}[{idx}]:\n{pred}\nexpected:\n{exp}")
    return bad


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    bad = validate_json(model)
    print(f"{TASK_ID}.json: {'PASS' if bad == 0 else f'FAIL ({bad} wrong)'}")

    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"nodes:   {len(model.graph.node)}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
