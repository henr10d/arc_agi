"""Minimal ONNX for ARC task013: extend two colored seeds into periodic stripes.

Task rule: the input grid contains exactly two colored seed cells on black. If
the seeds mark a horizontal period (same row, or top-to-bottom placement), draw
full-height vertical stripes through the seed columns and keep repeating the
two-color motif every seed-column distance. If the seeds mark a vertical period
(same column, or left-to-right placement), draw full-width horizontal stripes
through the seed rows and repeat every seed-row distance. The output grid keeps
the input size; padded cells outside the ARC grid remain empty.

ONNX: detect the two non-black seed coordinates from one-hot input, compute row
and column extrema, select stripe orientation, build a scalar color grid from
the two repeated seed axes for k=0..7, then one-hot encode at the output.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import sanitize_model, score_file  # noqa: E402

TASK_ID = "task013"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task013.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver for the two-seed periodic row/column stripe rule."""
    g = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(g)
    pts = np.argwhere(g != 0)
    if len(pts) != 2:
        return out
    rows = pts[:, 0]
    cols = pts[:, 1]
    row_mode = not (rows[0] == rows[1] or (cols[0] != cols[1] and rows.min() == 0))
    if row_mode:
        step = int(rows.max() - rows.min())
        for r, c in pts:
            color = int(g[r, c])
            rr = int(r)
            while rr < g.shape[0]:
                out[rr, :] = color
                rr += 2 * step
    else:
        step = int(cols.max() - cols.min())
        for r, c in pts:
            color = int(g[r, c])
            cc = int(c)
            while cc < g.shape[1]:
                out[:, cc] = color
                cc += 2 * step
    return out


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _near(nodes: List[onnx.NodeProto], a: str, b: str, half: str, tag: str) -> str:
    nodes.extend(
        [
            helper.make_node("Sub", [a, b], [f"{tag}_d"]),
            helper.make_node("Abs", [f"{tag}_d"], [f"{tag}_ad"]),
            helper.make_node("Less", [f"{tag}_ad", half], [f"{tag}_eq"]),
        ]
    )
    return f"{tag}_eq"


def _or_many(nodes: List[onnx.NodeProto], names: list[str], tag: str) -> str:
    cur = names[0]
    for idx, name in enumerate(names[1:], start=1):
        out = f"{tag}_or{idx}"
        nodes.append(helper.make_node("Or", [cur, name], [out]))
        cur = out
    return cur


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    bg_st = _i64(inits, [0, 0, 0, 0], "bg_st")
    bg_en = _i64(inits, [1, 1, H, W], "bg_en")
    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    half = _f32(inits, [0.5], "half")
    zero = _f32(inits, [0.0], "zero")
    two = _f32(inits, [2.0], "two")
    big = _f32(inits, np.full((1, 1, H, W), 100.0, dtype=np.float32), "big")
    rows = _f32(inits, np.arange(H, dtype=np.float32).reshape(1, 1, H, 1), "rows")
    cols = _f32(inits, np.arange(W, dtype=np.float32).reshape(1, 1, 1, W), "cols")
    channel_ids = _f32(inits, np.arange(C, dtype=np.float32).reshape(1, C, 1, 1), "channel_ids")
    channel_ids_i = _i64(inits, np.arange(C, dtype=np.int64).reshape(1, C, 1, 1), "channel_ids_i")

    nodes.extend(
        [
            helper.make_node("ReduceMax", [IN_NAME], ["active_f"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["active_f", half], ["active"]),
            helper.make_node("Slice", [IN_NAME, bg_st, bg_en, axes4], ["bg_in"]),
            helper.make_node("Less", ["bg_in", half], ["not_bg"]),
            helper.make_node("And", ["active", "not_bg"], ["fg_any"]),
            helper.make_node("Cast", ["fg_any"], ["fg_any_f2"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["fg_any_f2", rows], ["seed_rows"]),
            helper.make_node("Mul", ["fg_any_f2", cols], ["seed_cols"]),
            helper.make_node("ReduceMax", ["seed_rows"], ["row_max"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMax", ["seed_cols"], ["col_max"], axes=[2, 3], keepdims=1),
            helper.make_node("Where", ["fg_any", "seed_rows", big], ["seed_rows_big"]),
            helper.make_node("Where", ["fg_any", "seed_cols", big], ["seed_cols_big"]),
            helper.make_node("ReduceMin", ["seed_rows_big"], ["row_min"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMin", ["seed_cols_big"], ["col_min"], axes=[2, 3], keepdims=1),
            helper.make_node("Sub", ["row_max", "row_min"], ["row_step"]),
            helper.make_node("Sub", ["col_max", "col_min"], ["col_step"]),
        ]
    )

    same_row = _near(nodes, "row_step", zero, half, "fast_same_row")
    same_col = _near(nodes, "col_step", zero, half, "fast_same_col")
    top_seed = _near(nodes, "row_min", zero, half, "fast_top_seed")
    nodes.extend(
        [
            helper.make_node("Not", [same_col], ["not_same_col"]),
            helper.make_node("And", ["not_same_col", top_seed], ["top_noncol"]),
            helper.make_node("Or", [same_row, "top_noncol"], ["col_mode"]),
            helper.make_node("Mul", [IN_NAME, channel_ids], ["weighted_input"]),
            helper.make_node("ReduceSum", ["weighted_input"], ["color_plane"], axes=[1], keepdims=1),
        ]
    )

    eq_col_min = _near(nodes, cols, "col_min", half, "eq_col_min")
    eq_col_max = _near(nodes, cols, "col_max", half, "eq_col_max")
    eq_row_min = _near(nodes, rows, "row_min", half, "eq_row_min")
    eq_row_max = _near(nodes, rows, "row_max", half, "eq_row_max")
    for tag, eq in (
        ("col_min", eq_col_min),
        ("col_max", eq_col_max),
        ("row_min", eq_row_min),
        ("row_max", eq_row_max),
    ):
        nodes.extend(
            [
                helper.make_node("And", ["fg_any", eq], [f"{tag}_seed"]),
                helper.make_node("Where", [f"{tag}_seed", "color_plane", zero], [f"{tag}_colors"]),
                helper.make_node("ReduceMax", [f"{tag}_colors"], [f"{tag}_color"], axes=[2, 3], keepdims=1),
            ]
        )

    nodes.extend(
        [
            helper.make_node("Mul", ["col_step", two], ["col_period"]),
            helper.make_node("Mul", ["row_step", two], ["row_period"]),
        ]
    )
    col_min_hits = []
    col_max_hits = []
    row_min_hits = []
    row_max_hits = []
    for k in range(8):
        k_name = _f32(inits, [float(k)], f"fast_k{k}")
        nodes.extend(
            [
                helper.make_node("Mul", ["col_period", k_name], [f"col_add{k}"]),
                helper.make_node("Add", ["col_min", f"col_add{k}"], [f"col_min_pos{k}"]),
                helper.make_node("Add", ["col_max", f"col_add{k}"], [f"col_max_pos{k}"]),
                helper.make_node("Mul", ["row_period", k_name], [f"row_add{k}"]),
                helper.make_node("Add", ["row_min", f"row_add{k}"], [f"row_min_pos{k}"]),
                helper.make_node("Add", ["row_max", f"row_add{k}"], [f"row_max_pos{k}"]),
            ]
        )
        col_min_hits.append(_near(nodes, cols, f"col_min_pos{k}", half, f"col_min_hit{k}"))
        col_max_hits.append(_near(nodes, cols, f"col_max_pos{k}", half, f"col_max_hit{k}"))
        row_min_hits.append(_near(nodes, rows, f"row_min_pos{k}", half, f"row_min_hit{k}"))
        row_max_hits.append(_near(nodes, rows, f"row_max_pos{k}", half, f"row_max_hit{k}"))

    col_min_mask = _or_many(nodes, col_min_hits, "col_min_mask")
    col_max_mask = _or_many(nodes, col_max_hits, "col_max_mask")
    row_min_mask = _or_many(nodes, row_min_hits, "row_min_mask")
    row_max_mask = _or_many(nodes, row_max_hits, "row_max_mask")
    nodes.extend(
        [
            helper.make_node("Where", [col_min_mask, "col_min_color", zero], ["col_value_a"]),
            helper.make_node("Where", [col_max_mask, "col_max_color", "col_value_a"], ["col_value"]),
            helper.make_node("Where", [row_min_mask, "row_min_color", zero], ["row_value_a"]),
            helper.make_node("Where", [row_max_mask, "row_max_color", "row_value_a"], ["row_value"]),
            helper.make_node("Where", ["col_mode", "col_value", "row_value"], ["color_grid"]),
            helper.make_node("Cast", ["color_grid"], ["color_grid_i"], to=TensorProto.INT64),
            helper.make_node("Equal", [channel_ids_i, "color_grid_i"], ["onehot_b"]),
            helper.make_node("Cast", ["onehot_b"], ["onehot_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["active"], ["active_out"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["onehot_f", "active_out"], [OUT_NAME]),
        ]
    )

    graph = helper.make_graph(nodes, f"{TASK_ID}_scalar_grid", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _load_examples() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    bad = 0
    for split, examples in _load_examples().items():
        for idx, ex in enumerate(examples):
            g = np.asarray(ex["input"], dtype=np.int64)
            if g.shape[0] > H or g.shape[1] > W:
                continue
            expected = np.asarray(ex["output"], dtype=np.int64)
            assert np.array_equal(solve(g), expected), f"solver mismatch {split} {idx}"
            pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(ex["input"])))[: g.shape[0], : g.shape[1]]
            if not np.array_equal(pred, expected):
                bad += 1
                print(f"Mismatch {split} {idx}")
                print(pred)
                print(expected)
                break
    return bad


def main() -> None:
    model = build_model()
    sanitized = sanitize_model(copy.deepcopy(model))
    assert sanitized is not None, "sanitize_model failed"
    bad = validate_json(sanitized)
    assert bad == 0, f"{bad} examples failed"
    onnx.save(sanitized, BEST_PATH)
    print(f"saved {BEST_PATH}")
    print(score_file(BEST_PATH))


if __name__ == "__main__":
    main()
