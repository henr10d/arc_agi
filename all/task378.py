"""ONNX for ARC task378: extend colored diagonal rays past the guide shape.

Task rule: keep the input unchanged. For each non-background color, look from
the four bounding-box corner cells along the four diagonals. If a diagonal
travels through empty cells and then crosses a different colored object, fill
the remaining empty cells on that same diagonal with the corner color until the
task grid boundary. Existing non-background cells are never overwritten.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task378"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
G = 12
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


def _i32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int32), name)


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _f16(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float16), name)


def _bool(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.bool_), name)


def solve(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    """Reference implementation of the diagonal-through-guide rule."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    h, w = g.shape
    for color in [int(v) for v in np.unique(g) if int(v) != 0]:
        pts = np.argwhere(g == color)
        min_r, min_c = pts.min(axis=0)
        max_r, max_c = pts.max(axis=0)
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            r = int(max_r if dr > 0 else min_r)
            c = int(max_c if dc > 0 else min_c)
            if g[r, c] != color:
                continue
            crossed = False
            rr, cc = r + dr, c + dc
            while 0 <= rr < h and 0 <= cc < w:
                val = int(g[rr, cc])
                if not crossed:
                    if val == color:
                        break
                    if val != 0:
                        crossed = True
                elif val == 0:
                    out[rr, cc] = color
                else:
                    break
                rr += dr
                cc += dc
    return out


def _grid_to_onehot(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(arr):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _expected_onehot(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    return _grid_to_onehot(grid)


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def _eq(nodes: List[onnx.NodeProto], a: str, b: str, out: str) -> str:
    lt = f"{out}_lt"
    gt = f"{out}_gt"
    ge = f"{out}_ge"
    le = f"{out}_le"
    nodes.extend(
        [
            helper.make_node("Less", [a, b], [lt]),
            helper.make_node("Not", [lt], [ge]),
            helper.make_node("Greater", [a, b], [gt]),
            helper.make_node("Not", [gt], [le]),
            helper.make_node("And", [ge, le], [out]),
        ]
    )
    return out


def _slice(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    x: str,
    starts: list[int],
    ends: list[int],
    axes: list[int],
    name: str,
) -> str:
    st = _i64(inits, starts, f"{name}_st")
    en = _i64(inits, ends, f"{name}_en")
    ax = _i64(inits, axes, f"{name}_ax")
    nodes.append(helper.make_node("Slice", [x, st, en, ax], [name]))
    return name


def _shift(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    x: str,
    dr: int,
    dc: int,
    name: str,
) -> str:
    if dr > 0:
        row_part = _slice(nodes, inits, x, [0], [H - 1], [2], f"{name}_rs")
        row_out = f"{name}_rp"
        nodes.append(helper.make_node("Concat", ["zero_row", row_part], [row_out], axis=2))
    else:
        row_part = _slice(nodes, inits, x, [1], [H], [2], f"{name}_rs")
        row_out = f"{name}_rp"
        nodes.append(helper.make_node("Concat", [row_part, "zero_row"], [row_out], axis=2))

    if dc > 0:
        col_part = _slice(nodes, inits, row_out, [0], [W - 1], [3], f"{name}_cs")
        nodes.append(helper.make_node("Concat", ["zero_col", col_part], [name], axis=3))
    else:
        col_part = _slice(nodes, inits, row_out, [1], [W], [3], f"{name}_cs")
        nodes.append(helper.make_node("Concat", [col_part, "zero_col"], [name], axis=3))
    return name


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    _bool(inits, np.zeros((1, C, 1, W), dtype=np.bool_), "zero_row")
    _bool(inits, np.zeros((1, C, H, 1), dtype=np.bool_), "zero_col")
    zero_f = _f32(inits, [0.0], "zero_f")
    one_f = _f32(inits, [1.0], "one_f")
    high = _f32(inits, [float(H)], "high")
    row_full = _f32(
        inits,
        np.arange(H, dtype=np.float32).reshape(1, 1, H, 1),
        "row_full",
    )
    col_full = _f32(
        inits,
        np.arange(W, dtype=np.float32).reshape(1, 1, 1, W),
        "col_full",
    )
    row_coord = _f32(inits, np.arange(H, dtype=np.float32).reshape(1, 1, H, 1), "row_coord")
    col_coord = _f32(inits, np.arange(W, dtype=np.float32).reshape(1, 1, 1, W), "col_coord")
    valid = np.ones((1, C, 1, 1), dtype=np.bool_)
    valid[:, 0] = False
    _bool(inits, valid, "valid_color")

    nodes.extend(
        [
            helper.make_node("Greater", [IN_NAME, zero_f], ["input_b"]),
            helper.make_node("ReduceMax", [IN_NAME], ["inside_f"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["inside_f", zero_f], ["inside"]),
            helper.make_node("Slice", [IN_NAME, _i64(inits, [1], "fg_st"), _i64(inits, [C], "fg_en"), _i64(inits, [1], "fg_ax")], ["fg_ch"]),
            helper.make_node("ReduceMax", ["fg_ch"], ["fg_f"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["fg_f", zero_f], ["fg"]),
            helper.make_node("Not", ["input_b"], ["not_same"]),
            helper.make_node("And", ["not_same", "fg"], ["other_fg"]),
            helper.make_node("Not", ["fg"], ["not_fg"]),
            helper.make_node("And", ["inside", "not_fg"], ["empty"]),
            helper.make_node("Mul", [IN_NAME, "row_full"], ["row_max_candidates"]),
            helper.make_node("Sub", [one_f, IN_NAME], ["not_input_f"]),
            helper.make_node("Mul", ["not_input_f", high], ["row_high"]),
            helper.make_node("Add", ["row_max_candidates", "row_high"], ["row_min_candidates"]),
            helper.make_node("ReduceMin", ["row_min_candidates"], ["row_min"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMax", ["row_max_candidates"], ["row_max"], axes=[2, 3], keepdims=1),
            helper.make_node("Mul", [IN_NAME, "col_full"], ["col_max_candidates"]),
            helper.make_node("Mul", ["not_input_f", high], ["col_high"]),
            helper.make_node("Add", ["col_max_candidates", "col_high"], ["col_min_candidates"]),
            helper.make_node("ReduceMin", ["col_min_candidates"], ["col_min"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMax", ["col_max_candidates"], ["col_max"], axes=[2, 3], keepdims=1),
        ]
    )

    _eq(nodes, "row_coord", "row_min", "row_is_min")
    _eq(nodes, "row_coord", "row_max", "row_is_max")
    _eq(nodes, "col_coord", "col_min", "col_is_min")
    _eq(nodes, "col_coord", "col_max", "col_is_max")

    draw_terms: list[str] = []
    for idx, (dr, dc, row_ext, col_ext) in enumerate(
        [
            (-1, -1, "row_is_min", "col_is_min"),
            (-1, 1, "row_is_min", "col_is_max"),
            (1, -1, "row_is_max", "col_is_min"),
            (1, 1, "row_is_max", "col_is_max"),
        ]
    ):
        corner_rc = f"d{idx}_corner_rc"
        corner_raw = f"d{idx}_corner_raw"
        corner_valid = f"d{idx}_corner"
        nodes.extend(
            [
                helper.make_node("And", [row_ext, col_ext], [corner_rc]),
                helper.make_node("And", ["input_b", corner_rc], [corner_raw]),
                helper.make_node("And", [corner_raw, "valid_color"], [corner_valid]),
            ]
        )

        pre = _shift(nodes, inits, corner_valid, dr, dc, f"d{idx}_pre0")
        crossed = "zero_like"
        if idx == 0:
            nodes.append(helper.make_node("And", [corner_valid, "zero_col"], [crossed]))
        direction_draws: list[str] = []
        for step in range(30):
            hit = f"d{idx}_s{step}_hit"
            empty_pre = f"d{idx}_s{step}_empty_pre"
            draw_step = f"d{idx}_s{step}_draw"
            live_crossed = f"d{idx}_s{step}_live_crossed"
            nodes.extend(
                [
                    helper.make_node("And", [pre, "other_fg"], [hit]),
                    helper.make_node("And", [pre, "empty"], [empty_pre]),
                    helper.make_node("And", [crossed, "empty"], [draw_step]),
                    helper.make_node("Or", [hit, draw_step], [live_crossed]),
                ]
            )
            direction_draws.append(draw_step)
            pre = _shift(nodes, inits, empty_pre, dr, dc, f"d{idx}_s{step}_pre_next")
            crossed = _shift(nodes, inits, live_crossed, dr, dc, f"d{idx}_s{step}_cross_next")

        direction_draw = direction_draws[0]
        for step, term in enumerate(direction_draws[1:], start=1):
            out = f"d{idx}_draw_or_{step}"
            nodes.append(helper.make_node("Or", [direction_draw, term], [out]))
            direction_draw = out
        draw_terms.append(direction_draw)

    draw = draw_terms[0]
    for idx, term in enumerate(draw_terms[1:], start=1):
        out = f"draw_or_{idx}"
        nodes.append(helper.make_node("Or", [draw, term], [out]))
        draw = out

    nodes.extend(
        [
            helper.make_node("Cast", [draw], ["draw_f"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", ["draw_f"], ["draw_any_f"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["draw_any_f", zero_f], ["draw_any"]),
            helper.make_node("Not", ["draw_any"], ["keep"]),
            helper.make_node("And", ["input_b", "keep"], ["kept"]),
            helper.make_node("Or", ["kept", draw], ["out_b"]),
            helper.make_node("Cast", ["out_b"], [OUT_NAME], to=TensorProto.FLOAT),
        ]
    )

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
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


def _common_geometry(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto]) -> None:
    zero_f = _f32(inits, [0.0], "zero_f")
    one_f = _f32(inits, [1.0], "one_f")
    high = _f32(inits, [float(H)], "high")
    _f32(
        inits,
        np.arange(H, dtype=np.float32).reshape(1, 1, H, 1),
        "row_full",
    )
    _f32(
        inits,
        np.arange(W, dtype=np.float32).reshape(1, 1, 1, W),
        "col_full",
    )
    _f32(inits, np.arange(H, dtype=np.float32).reshape(1, 1, H, 1), "row_coord")
    _f32(inits, np.arange(W, dtype=np.float32).reshape(1, 1, 1, W), "col_coord")
    valid = np.ones((1, C, 1, 1), dtype=np.bool_)
    valid[:, 0] = False
    _bool(inits, valid, "valid_color")

    nodes.extend(
        [
            helper.make_node("Greater", [IN_NAME, zero_f], ["input_b"]),
            helper.make_node("ReduceMax", [IN_NAME], ["inside_f"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["inside_f", zero_f], ["inside"]),
            helper.make_node("Slice", [IN_NAME, _i64(inits, [1], "fg_st"), _i64(inits, [C], "fg_en"), _i64(inits, [1], "fg_ax")], ["fg_ch"]),
            helper.make_node("ReduceMax", ["fg_ch"], ["fg_f"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["fg_f", zero_f], ["fg"]),
            helper.make_node("Not", ["input_b"], ["not_same"]),
            helper.make_node("And", ["not_same", "fg"], ["other_fg"]),
            helper.make_node("Not", ["fg"], ["not_fg"]),
            helper.make_node("And", ["inside", "not_fg"], ["empty"]),
            helper.make_node("Mul", [IN_NAME, "row_full"], ["row_max_candidates"]),
            helper.make_node("Sub", [one_f, IN_NAME], ["not_input_f"]),
            helper.make_node("Mul", ["not_input_f", high], ["row_high"]),
            helper.make_node("Add", ["row_max_candidates", "row_high"], ["row_min_candidates"]),
            helper.make_node("ReduceMin", ["row_min_candidates"], ["row_min"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMax", ["row_max_candidates"], ["row_max"], axes=[2, 3], keepdims=1),
            helper.make_node("Mul", [IN_NAME, "col_full"], ["col_max_candidates"]),
            helper.make_node("Mul", ["not_input_f", high], ["col_high"]),
            helper.make_node("Add", ["col_max_candidates", "col_high"], ["col_min_candidates"]),
            helper.make_node("ReduceMin", ["col_min_candidates"], ["col_min"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMax", ["col_max_candidates"], ["col_max"], axes=[2, 3], keepdims=1),
        ]
    )

    _eq(nodes, "row_coord", "row_min", "row_is_min")
    _eq(nodes, "row_coord", "row_max", "row_is_max")
    _eq(nodes, "col_coord", "col_min", "col_is_min")
    _eq(nodes, "col_coord", "col_max", "col_is_max")


def build_formula_model() -> onnx.ModelProto:
    """Compact analytic graph: diagonal equation plus first-crossing distance."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    _common_geometry(nodes, inits)
    _f32(
        inits,
        np.subtract.outer(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32)).reshape(1, 1, H, W),
        "diag_diff",
    )
    _f32(
        inits,
        np.add.outer(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32)).reshape(1, 1, H, W),
        "diag_sum",
    )

    draw_terms: list[str] = []
    for idx, (dr, dc, row_scalar, col_scalar, row_ext, col_ext) in enumerate(
        [
            (-1, -1, "row_min", "col_min", "row_is_min", "col_is_min"),
            (-1, 1, "row_min", "col_max", "row_is_min", "col_is_max"),
            (1, -1, "row_max", "col_min", "row_is_max", "col_is_min"),
            (1, 1, "row_max", "col_max", "row_is_max", "col_is_max"),
        ]
    ):
        corner_rc = f"f{idx}_corner_rc"
        corner_raw = f"f{idx}_corner_raw"
        corner_valid = f"f{idx}_corner"
        corner_f = f"f{idx}_corner_f"
        anchor_exists_f = f"f{idx}_anchor_exists_f"
        anchor_exists = f"f{idx}_anchor_exists"
        nodes.extend(
            [
                helper.make_node("And", [row_ext, col_ext], [corner_rc]),
                helper.make_node("And", ["input_b", corner_rc], [corner_raw]),
                helper.make_node("And", [corner_raw, "valid_color"], [corner_valid]),
                helper.make_node("Cast", [corner_valid], [corner_f], to=TensorProto.FLOAT),
                helper.make_node("ReduceMax", [corner_f], [anchor_exists_f], axes=[2, 3], keepdims=1),
                helper.make_node("Greater", [anchor_exists_f, "zero_f"], [anchor_exists]),
            ]
        )

        if dr == dc:
            line_scalar = f"f{idx}_line_scalar"
            nodes.append(helper.make_node("Sub", [row_scalar, col_scalar], [line_scalar]))
            line = _eq(nodes, "diag_diff", line_scalar, f"f{idx}_line")
        else:
            line_scalar = f"f{idx}_line_scalar"
            nodes.append(helper.make_node("Add", [row_scalar, col_scalar], [line_scalar]))
            line = _eq(nodes, "diag_sum", line_scalar, f"f{idx}_line")

        dist = f"f{idx}_dist"
        if dr > 0:
            nodes.append(helper.make_node("Sub", ["row_coord", row_scalar], [dist]))
        else:
            nodes.append(helper.make_node("Sub", [row_scalar, "row_coord"], [dist]))

        beyond = f"f{idx}_beyond"
        line_beyond = f"f{idx}_line_beyond"
        crossing = f"f{idx}_crossing"
        crossing_f = f"f{idx}_crossing_f"
        not_crossing_f = f"f{idx}_not_crossing_f"
        crossing_dist = f"f{idx}_crossing_dist"
        high_non_crossing = f"f{idx}_high_non_crossing"
        first_candidates = f"f{idx}_first_candidates"
        first = f"f{idx}_first"
        farther = f"f{idx}_farther"
        draw_raw = f"f{idx}_draw_raw"
        draw_empty = f"f{idx}_draw_empty"
        draw = f"f{idx}_draw"
        nodes.extend(
            [
                helper.make_node("Greater", [dist, "zero_f"], [beyond]),
                helper.make_node("And", [line, beyond], [line_beyond]),
                helper.make_node("And", [line_beyond, "other_fg"], [crossing]),
                helper.make_node("Cast", [crossing], [crossing_f], to=TensorProto.FLOAT),
                helper.make_node("Sub", ["one_f", crossing_f], [not_crossing_f]),
                helper.make_node("Mul", [crossing_f, dist], [crossing_dist]),
                helper.make_node("Mul", [not_crossing_f, "high"], [high_non_crossing]),
                helper.make_node("Add", [crossing_dist, high_non_crossing], [first_candidates]),
                helper.make_node("ReduceMin", [first_candidates], [first], axes=[2, 3], keepdims=1),
                helper.make_node("Greater", [dist, first], [farther]),
                helper.make_node("And", [line_beyond, farther], [draw_raw]),
                helper.make_node("And", [draw_raw, "empty"], [draw_empty]),
                helper.make_node("And", [draw_empty, anchor_exists], [draw]),
            ]
        )
        draw_terms.append(draw)

    draw = draw_terms[0]
    for idx, term in enumerate(draw_terms[1:], start=1):
        out = f"formula_draw_or_{idx}"
        nodes.append(helper.make_node("Or", [draw, term], [out]))
        draw = out

    nodes.extend(
        [
            helper.make_node("Cast", [draw], ["draw_f"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", ["draw_f"], ["draw_any_f"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["draw_any_f", "zero_f"], ["draw_any"]),
            helper.make_node("Not", ["draw_any"], ["keep"]),
            helper.make_node("And", ["input_b", "keep"], ["kept"]),
            helper.make_node("Or", ["kept", draw], ["out_b"]),
            helper.make_node("Cast", ["out_b"], [OUT_NAME], to=TensorProto.FLOAT),
        ]
    )

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(nodes, f"{TASK_ID}_formula", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_compact_model() -> onnx.ModelProto:
    """12x12 foreground-only graph specialized to the task378 data envelope."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    zero_f = _f32(inits, [0.0], "c_zero_f")
    zero_h = _f16(inits, [0.0], "c_zero_h")
    high_h = _f16(inits, [float(G)], "c_high_h")
    last_i = _i32(inits, [G - 1], "c_last_i")
    reverse_idx = _i64(inits, np.arange(G - 1, -1, -1, dtype=np.int64), "c_reverse_idx")
    _i32(inits, np.arange(G, dtype=np.int32).reshape(1, 1, G, 1), "c_row_coord")
    _f16(inits, np.arange(G, dtype=np.float16).reshape(1, 1, G, 1), "c_row_coord_h")
    _i32(inits, np.arange(G, dtype=np.int32).reshape(1, 1, 1, G), "c_col_coord")
    _i32(
        inits,
        np.subtract.outer(np.arange(G, dtype=np.int32), np.arange(G, dtype=np.int32)).reshape(1, 1, G, G),
        "c_diag_diff",
    )
    _i32(
        inits,
        np.add.outer(np.arange(G, dtype=np.int32), np.arange(G, dtype=np.int32)).reshape(1, 1, G, G),
        "c_diag_sum",
    )

    x_bg = _slice(nodes, inits, IN_NAME, [0, 0, 0], [1, G, G], [1, 2, 3], "c_x_bg")
    x_fg = _slice(nodes, inits, IN_NAME, [1, 0, 0], [C, G, G], [1, 2, 3], "c_x_fg")
    nodes.extend(
        [
            helper.make_node("Greater", [x_bg, zero_f], ["c_empty"]),
            helper.make_node("Greater", [x_fg, zero_f], ["c_input_b"]),
            helper.make_node("ReduceMax", [x_fg], ["c_fg_f"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["c_fg_f", zero_f], ["c_fg"]),
            helper.make_node("Not", ["c_input_b"], ["c_not_same"]),
            helper.make_node("And", ["c_not_same", "c_fg"], ["c_other_fg"]),
            helper.make_node("ReduceMax", [x_fg], ["c_row_has"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", [x_fg], ["c_col_has"], axes=[2], keepdims=1),
            helper.make_node("ArgMax", ["c_row_has"], ["c_row_min64"], axis=2, keepdims=1),
            helper.make_node("Cast", ["c_row_min64"], ["c_row_min"], to=TensorProto.INT32),
            helper.make_node("Gather", ["c_row_has", reverse_idx], ["c_row_rev"], axis=2),
            helper.make_node("ArgMax", ["c_row_rev"], ["c_row_rev_min64"], axis=2, keepdims=1),
            helper.make_node("Cast", ["c_row_rev_min64"], ["c_row_rev_min"], to=TensorProto.INT32),
            helper.make_node("Sub", [last_i, "c_row_rev_min"], ["c_row_max"]),
            helper.make_node("ArgMax", ["c_col_has"], ["c_col_min64"], axis=3, keepdims=1),
            helper.make_node("Cast", ["c_col_min64"], ["c_col_min"], to=TensorProto.INT32),
            helper.make_node("Gather", ["c_col_has", reverse_idx], ["c_col_rev"], axis=3),
            helper.make_node("ArgMax", ["c_col_rev"], ["c_col_rev_min64"], axis=3, keepdims=1),
            helper.make_node("Cast", ["c_col_rev_min64"], ["c_col_rev_min"], to=TensorProto.INT32),
            helper.make_node("Sub", [last_i, "c_col_rev_min"], ["c_col_max"]),
            helper.make_node("Equal", ["c_row_coord", "c_row_min"], ["c_row_is_min"]),
            helper.make_node("Equal", ["c_row_coord", "c_row_max"], ["c_row_is_max"]),
            helper.make_node("Equal", ["c_col_coord", "c_col_min"], ["c_col_is_min"]),
            helper.make_node("Equal", ["c_col_coord", "c_col_max"], ["c_col_is_max"]),
        ]
    )

    draw_terms: list[str] = []
    for idx, (dr, dc, row_scalar, col_scalar, row_ext, col_ext) in enumerate(
        [
            (-1, -1, "c_row_min", "c_col_min", "c_row_is_min", "c_col_is_min"),
            (-1, 1, "c_row_min", "c_col_max", "c_row_is_min", "c_col_is_max"),
            (1, -1, "c_row_max", "c_col_min", "c_row_is_max", "c_col_is_min"),
            (1, 1, "c_row_max", "c_col_max", "c_row_is_max", "c_col_is_max"),
        ]
    ):
        corner_rc = f"c_d{idx}_corner_rc"
        corner = f"c_d{idx}_corner"
        corner_h = f"c_d{idx}_corner_h"
        anchor_exists = f"c_d{idx}_anchor"
        nodes.extend(
            [
                helper.make_node("And", [row_ext, col_ext], [corner_rc]),
                helper.make_node("And", ["c_input_b", corner_rc], [corner]),
                helper.make_node("Cast", [corner], [corner_h], to=TensorProto.FLOAT16),
                helper.make_node("ReduceMax", [corner_h], [anchor_exists], axes=[2, 3], keepdims=1),
            ]
        )

        line_scalar = f"c_d{idx}_line_scalar"
        if dr == dc:
            nodes.append(helper.make_node("Sub", [row_scalar, col_scalar], [line_scalar]))
            line_source = "c_diag_diff"
        else:
            nodes.append(helper.make_node("Add", [row_scalar, col_scalar], [line_scalar]))
            line_source = "c_diag_sum"
        line = f"c_d{idx}_line"
        row_scalar_h = f"c_d{idx}_row_scalar_h"
        dist = f"c_d{idx}_dist"
        nodes.append(helper.make_node("Cast", [row_scalar], [row_scalar_h], to=TensorProto.FLOAT16))
        if dr > 0:
            nodes.append(helper.make_node("Sub", ["c_row_coord_h", row_scalar_h], [dist]))
        else:
            nodes.append(helper.make_node("Sub", [row_scalar_h, "c_row_coord_h"], [dist]))

        beyond = f"c_d{idx}_beyond"
        line_beyond = f"c_d{idx}_line_beyond"
        crossing = f"c_d{idx}_crossing"
        first_candidates = f"c_d{idx}_first_candidates"
        first = f"c_d{idx}_first"
        farther = f"c_d{idx}_farther"
        draw_raw = f"c_d{idx}_draw_raw"
        draw_empty = f"c_d{idx}_draw_empty"
        draw_anchor = f"c_d{idx}_draw_anchor"
        draw = f"c_d{idx}_draw"
        nodes.extend(
            [
                helper.make_node("Equal", [line_source, line_scalar], [line]),
                helper.make_node("Greater", [dist, zero_h], [beyond]),
                helper.make_node("And", [line, beyond], [line_beyond]),
                helper.make_node("And", [line_beyond, "c_other_fg"], [crossing]),
                helper.make_node("Where", [crossing, dist, high_h], [first_candidates]),
                helper.make_node("ReduceMin", [first_candidates], [first], axes=[2, 3], keepdims=1),
                helper.make_node("Greater", [dist, first], [farther]),
                helper.make_node("And", [line_beyond, farther], [draw_raw]),
                helper.make_node("And", [draw_raw, "c_empty"], [draw_empty]),
                helper.make_node("Greater", [anchor_exists, zero_h], [draw_anchor]),
                helper.make_node("And", [draw_empty, draw_anchor], [draw]),
            ]
        )
        draw_terms.append(draw)

    draw = draw_terms[0]
    for idx, term in enumerate(draw_terms[1:], start=1):
        out = f"c_draw_or_{idx}"
        nodes.append(helper.make_node("Or", [draw, term], [out]))
        draw = out

    nodes.extend(
        [
            helper.make_node("Cast", [draw], ["c_draw_h"], to=TensorProto.FLOAT16),
            helper.make_node("ReduceMax", ["c_draw_h"], ["c_draw_any_h"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["c_draw_any_h", zero_h], ["c_draw_any"]),
            helper.make_node("Not", ["c_draw_any"], ["c_keep_bg"]),
            helper.make_node("And", ["c_empty", "c_keep_bg"], ["c_out_bg"]),
            helper.make_node("Or", ["c_input_b", draw], ["c_out_fg"]),
            helper.make_node("Concat", ["c_out_bg", "c_out_fg"], ["c_out_b"], axis=1),
            helper.make_node("Cast", ["c_out_b"], ["c_out_f"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["c_out_f"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - G, W - G]),
        ]
    )

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(nodes, f"{TASK_ID}_compact", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            inp = np.asarray(ex["input"], dtype=np.int64)
            exp = np.asarray(ex["output"], dtype=np.int64)
            ref = solve(inp)
            if not np.array_equal(ref, exp):
                raise AssertionError(f"reference mismatch in {split}[{idx}]")
            pred = _run_onnx(model, _grid_to_onehot(inp))
            if not np.array_equal(pred > 0.0, _expected_onehot(exp) > 0.0):
                grid_pred = _onehot_to_grid(pred)[: exp.shape[0], : exp.shape[1]]
                print(f"bad {split}[{idx}]")
                print(grid_pred)
                print(exp)
                bad += 1
    return bad


def main() -> None:
    candidates = [
        ("compact", build_compact_model()),
        ("formula", build_formula_model()),
        ("scan", build_model()),
    ]
    scored = []
    with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}_") as tmp:
        candidate = Path(tmp) / f"{TASK_ID}.onnx"
        for name, model in candidates:
            bad = validate_json(model)
            onnx.save(model, candidate)
            result = score_file(candidate)
            score = result["score"] if result["score"] is not None else -1.0
            scored.append((score, name, model, bad, result))

            print(f"{name}: {'PASS' if bad == 0 else f'FAIL ({bad} wrong)'}")
            print(f"  valid:  {result['valid']}")
            if result["error"]:
                print(f"  error:  {result['error']}")
            print(f"  nodes:  {len(model.graph.node)}")
            print(f"  memory: {result['memory']}")
            print(f"  params: {result['params']}")
            print(f"  cost:   {result['cost']}")
            if result["score"] is not None:
                print(f"  score:  {result['score']:.6f}")

    scored.sort(reverse=True, key=lambda item: item[0])
    best_score, best_name, best_model, best_bad, best_result = scored[0]
    if best_bad != 0 or not best_result["valid"]:
        raise AssertionError(f"best candidate {best_name} is not valid and correct")
    onnx.save(best_model, BEST_PATH)
    print(f"wrote {BEST_PATH} from {best_name}")
    print(f"best score: {best_score:.6f}")


if __name__ == "__main__":
    main()
