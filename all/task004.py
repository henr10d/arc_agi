"""Minimal ONNX for ARC task004: per-shape right shift with anchored lower-right hook.

Task rule: each disconnected colored shape shifts one cell to the right, except
pixels on the shape bottom row and the rightmost pixel on the row above the
bottom (the lower-right corner hook), which stay fixed. Colors are independent;
on this task each color occupies contiguous rows and columns, so per-color row
and column occupancy identifies the bottom row and rightmost column. Grid size
is unchanged; background stays black.

ONNX: process colors 1-9 on a 16x16 crop, build bottom/right occupancy masks
with compact reductions and boolean shifts, gate foreground by the valid input
grid before casting to float, then pad to 30x30 I/O.
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

BEST_PATH = OUT_DIR / "task004.onnx"
DATA_PATH = ROOT / "data" / "task004.json"

C = 10
NC = 9
H = W = 30
SH = 16
SW = 16
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10

TOY_INPUT = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 2, 2, 0, 0, 3, 0],
    [0, 2, 0, 0, 0, 3, 0],
    [0, 0, 0, 0, 0, 0, 0],
]
TOY_OUTPUT = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 3, 0],
    [0, 2, 0, 0, 0, 3, 0],
    [0, 0, 0, 0, 0, 0, 0],
]


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _bool(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.bool_), name)


def solve(grid: np.ndarray) -> np.ndarray:
    """Per-color bbox right shift; bottom row and penultimate-rightmost stay."""
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    out = np.zeros_like(g)
    rows = np.arange(h, dtype=np.int64)[:, None]
    cols = np.arange(w, dtype=np.int64)[None, :]
    for color in np.unique(g[g != 0]):
        m = g == color
        max_y = int((rows[:, 0] * m.any(axis=1)).max())
        max_x = int((cols[0] * m.any(axis=0)).max())
        stay = m & ((rows == max_y) | ((rows == max_y - 1) & (cols == max_x)))
        move = m & ~stay
        shifted = np.zeros_like(m)
        shifted[:, 1:] = move[:, :-1]
        out[stay] = color
        out[shifted] = color
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
    axis_row = _i64(inits, [2], "axis_row")
    axis_col = _i64(inits, [3], "axis_col")
    fg_st = _i64(inits, [1, 0, 0], "fg_st")
    fg_en = _i64(inits, [C, SH, SW], "fg_en")
    bg_st = _i64(inits, [0, 0, 0], "bg_st")
    bg_en = _i64(inits, [1, SH, SW], "bg_en")
    s0 = _i64(inits, [0], "s0")
    s1 = _i64(inits, [1], "s1")
    e15 = _i64(inits, [SW - 1], "e15")
    e16 = _i64(inits, [SW], "e16")
    out_pads = [0, 0, 0, 0, 0, 0, H - SH, W - SW]
    false1 = _bool(inits, np.zeros((1, NC, 1, 1), dtype=np.bool_), "false1")
    zcol = _bool(inits, np.zeros((1, NC, SH, 1), dtype=np.bool_), "zcol")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, fg_st, fg_en, axes_chw], ["fg"]),
            helper.make_node("Cast", ["fg"], ["nz"], to=TensorProto.BOOL),
            helper.make_node("ReduceMax", ["fg"], ["rowf"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["fg"], ["colf"], axes=[2], keepdims=1),
            helper.make_node("Cast", ["rowf"], ["row_occ"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["colf"], ["col_occ"], to=TensorProto.BOOL),
            helper.make_node("Slice", ["row_occ", s1, e16, axis_row], ["row_after_15"]),
            helper.make_node("Concat", ["row_after_15", false1], ["row_after"], axis=2),
            helper.make_node("Not", ["row_after"], ["not_row_after"]),
            helper.make_node("And", ["row_occ", "not_row_after"], ["bottom_row"]),
            helper.make_node("Slice", ["bottom_row", s1, e16, axis_row], ["bottom_after_15"]),
            helper.make_node("Concat", ["bottom_after_15", false1], ["pen_row"], axis=2),
            helper.make_node("Slice", ["col_occ", s1, e16, axis_col], ["col_after_15"]),
            helper.make_node("Concat", ["col_after_15", false1], ["col_after"], axis=3),
            helper.make_node("Not", ["col_after"], ["not_col_after"]),
            helper.make_node("And", ["col_occ", "not_col_after"], ["right_col"]),
            helper.make_node("And", ["pen_row", "right_col"], ["hook_line"]),
            helper.make_node("Or", ["bottom_row", "hook_line"], ["stay_lines"]),
            helper.make_node("And", ["nz", "stay_lines"], ["stay"]),
            helper.make_node("Not", ["stay"], ["nst"]),
            helper.make_node("And", ["nz", "nst"], ["move"]),
            helper.make_node("Slice", ["move", s0, e15, axis_col], ["mvt"]),
            helper.make_node("Concat", ["zcol", "mvt"], ["sh"], axis=3),
            helper.make_node("Or", ["stay", "sh"], ["out9"]),
            helper.make_node("Slice", [IN_NAME, bg_st, bg_en, axes_chw], ["bg_in"]),
            helper.make_node("ReduceMax", ["fg"], ["fg_any_in"], axes=[1], keepdims=1),
            helper.make_node("Add", ["bg_in", "fg_any_in"], ["activef"]),
            helper.make_node("Cast", ["activef"], ["active"], to=TensorProto.BOOL),
            helper.make_node("And", ["out9", "active"], ["out9m_b"]),
            helper.make_node("Cast", ["out9m_b"], ["out9m"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", ["out9m"], ["any_out"], axes=[1], keepdims=1),
            helper.make_node("Sub", ["activef", "any_out"], ["bgf"]),
            helper.make_node("Concat", ["bgf", "out9m"], ["out10"], axis=1),
            helper.make_node("Pad", ["out10"], [OUT_NAME], pads=out_pads),
        ]
    )

    graph = helper.make_graph(nodes, "task004", [x_info], [y_info], initializer=inits)
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
    if not DATA_PATH.is_file():
        return 0
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            g = np.array(ex["input"], dtype=np.int64)
            if g.shape[0] > H or g.shape[1] > W:
                continue
            oh = _grid_to_onehot(ex["input"])
            pred = _onehot_to_grid(_run_onnx(model, oh))[: g.shape[0], : g.shape[1]]
            if not np.array_equal(pred, solve(g)):
                bad += 1
    return bad


def main() -> None:
    inp = np.array(TOY_INPUT, dtype=np.int64)
    exp = np.array(TOY_OUTPUT, dtype=np.int64)
    ref = solve(inp)
    assert np.array_equal(ref, exp), "reference solver failed toy example"

    model = build_model()
    onnx.save(model, BEST_PATH)

    toy_oh = _grid_to_onehot(TOY_INPUT)
    toy_pred = _onehot_to_grid(_run_onnx(model, toy_oh))[:4, :7]
    assert np.array_equal(toy_pred, exp), f"toy mismatch:\n{toy_pred}\n{exp}"

    bad = validate_json(model)
    print("toy: PASS")
    print(f"task004.json: {'PASS' if bad == 0 else f'FAIL ({bad} wrong)'}")

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
