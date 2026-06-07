"""ONNX for ARC task197: complete colored rows from the exemplar row pattern.

Task rule: each grid contains one already-complete colored row, located at row
1 in the examples. Its left-to-right two-color pattern is the template. Every
other non-empty row shows only the prefix up to the first occurrence of the
second template class; fill that row horizontally by copying the exemplar's
class pattern and substituting the row's own two colors. Empty rows remain
background, and the grid size is unchanged.

ONNX: work on the observed 14x10 task envelope, slice foreground channels
separately from the row-0 background width marker, build template A/B column
masks from row 1, derive each row's two colors from its visible prefix, then
pad the compact 14x10 result to 30x30 output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

BEST_PATH = OUT_DIR / "task197.onnx"
DATA_PATH = ROOT / "data" / "task197.json"

C = 10
H = W = 30
MAX_H = 14
MAX_W = 10
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
    """Reference implementation for JSON validation."""
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    out = np.zeros_like(g)
    template = g[1]
    a = int(template[0])
    b_cols = np.flatnonzero(template != a)
    if len(b_cols) == 0:
        return g.copy()
    first_b = int(b_cols[0])
    a_mask = template == a

    for r in range(h):
        row = g[r]
        if not np.any(row):
            continue
        color_a = int(row[0])
        color_b = int(row[first_b]) if row[first_b] != 0 else color_a
        out[r, a_mask[:w]] = color_a
        out[r, ~a_mask[:w]] = color_b
    return out


def _grid_to_onehot(grid: List[List[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def build_model(max_h: int = MAX_H, max_w: int = MAX_W) -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, [0, 1, 2, 3], "rr_axes")
    tmpl_st = _i64(inits, [0, 1, 1, 0], "rr_tmpl_st")
    tmpl_en = _i64(inits, [1, C, 2, max_w], "rr_tmpl_en")
    local_st = _i64(inits, [0, 0, 0, 0], "rr_local_st")
    tmpl_col0_en = _i64(inits, [1, C - 1, 1, 1], "rr_tmpl_col0_en")
    row0_bg_st = _i64(inits, [0, 0, 0, 0], "rr_row0_bg_st")
    row0_bg_en = _i64(inits, [1, 1, 1, max_w], "rr_row0_bg_en")
    row_a_st = _i64(inits, [0, 1, 0, 0], "rr_row_a_st")
    row_a_en = _i64(inits, [1, C, max_h, 1], "rr_row_a_en")
    row_colors_en = _i64(inits, [1, C, max_h, 1], "rr_row_colors_en")
    zero = _f32(inits, [0.0], "rr_zero")
    pads = [0, 0, 0, 0, 0, 0, H - max_h, W - max_w]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, tmpl_st, tmpl_en, axes], ["rr_tmpl"]),
            helper.make_node("Slice", ["rr_tmpl", local_st, tmpl_col0_en, axes], ["rr_tmpl0"]),
            helper.make_node("Mul", ["rr_tmpl", "rr_tmpl0"], ["rr_tmpl_a_hits"]),
            helper.make_node("ReduceMax", ["rr_tmpl_a_hits"], ["rr_mask_a_f"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["rr_mask_a_f", zero], ["rr_mask_a"]),
            helper.make_node("Slice", [IN_NAME, row0_bg_st, row0_bg_en, axes], ["rr_width_bg_f"]),
            helper.make_node("Greater", ["rr_width_bg_f", zero], ["rr_active_col"]),
            helper.make_node("Slice", [IN_NAME, row_a_st, row_a_en, axes], ["rr_row_a_f"]),
            helper.make_node("Greater", ["rr_row_a_f", zero], ["rr_row_a"]),
            helper.make_node("ReduceMax", [IN_NAME], ["rr_all_row_colors_f"], axes=[3], keepdims=1),
            helper.make_node("Slice", ["rr_all_row_colors_f", row_a_st, row_colors_en, axes], ["rr_row_colors_f"]),
            helper.make_node("Greater", ["rr_row_colors_f", zero], ["rr_row_colors"]),
            helper.make_node("Not", ["rr_row_a"], ["rr_not_row_a"]),
            helper.make_node("And", ["rr_row_colors", "rr_not_row_a"], ["rr_row_b"]),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Not", ["rr_mask_a"], ["rr_not_mask_a"]),
            helper.make_node("And", ["rr_not_mask_a", "rr_active_col"], ["rr_mask_b"]),
            helper.make_node("And", ["rr_row_a", "rr_mask_a"], ["rr_out_a"]),
            helper.make_node("And", ["rr_row_b", "rr_mask_b"], ["rr_out_b"]),
            helper.make_node("Or", ["rr_out_a", "rr_out_b"], ["rr_fg_out"]),
        ]
    )
    nodes.extend(
        [
            helper.make_node("ReduceMax", ["rr_row_colors_f"], ["rr_active_row_f"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["rr_active_row_f", zero], ["rr_active_row"]),
            helper.make_node("Not", ["rr_active_row"], ["rr_empty_row"]),
            helper.make_node("And", ["rr_empty_row", "rr_active_col"], ["rr_bg"]),
            helper.make_node("Concat", ["rr_bg", "rr_fg_out"], ["rr_out_bool"], axis=1),
            helper.make_node("Cast", ["rr_out_bool"], ["rr_out_crop"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["rr_out_crop"], [OUT_NAME], pads=pads),
        ]
    )

    graph = helper.make_graph(
        nodes,
        f"task197_row_reduce_{max_h}x{max_w}",
        [x_info],
        [y_info],
        initializer=inits,
    )
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
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            g = np.asarray(ex["input"], dtype=np.int64)
            pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(ex["input"])))[: g.shape[0], : g.shape[1]]
            expected = np.asarray(ex["output"], dtype=np.int64)
            ref = solve(g)
            if not np.array_equal(ref, expected):
                raise AssertionError(f"reference mismatch on {split}[{idx}]")
            if not np.array_equal(pred, expected):
                bad += 1
    return bad


def main() -> None:
    model = build_model(MAX_H, MAX_W)
    bad = validate_json(model)
    if bad:
        raise AssertionError(f"model failed {bad} examples")
    onnx.save(model, BEST_PATH)
    final = score_file(BEST_PATH)
    if not final["valid"]:
        raise AssertionError(final["error"])

    print("task197.json: PASS")
    print(f"wrote {BEST_PATH}")
    print(
        f"best: row_reduce_14x10 nodes={len(model.graph.node)} memory={final['memory']} "
        f"params={final['params']} cost={final['cost']} score={final['score']:.6f}"
    )


if __name__ == "__main__":
    main()
