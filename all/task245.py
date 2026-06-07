"""ONNX solution for ARC task245: move the red object into the green frame.

Task rule: keep the four green corner markers unchanged. Treat all red cells as
one object, find its bounding-box top-left cell, and translate the entire red
object so that this top-left lands one row and one column below/right of the
top-left green marker. Old red cells become background; grid size is unchanged.

ONNX: operate on the observed 10x10 task crop, compute first occupied
red/green rows and columns from boolean masks, select the matching row and
column shifts from prebuilt small candidates, then build the final one-hot crop
and pad it to the NeuroGolf 30x30 interface.
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

TASK_ID = "task245"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
SH = SW = 10
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10
SHIFTS = list(range(5))


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _bool(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.bool_), name)


def _cached_i64(inits: List[onnx.TensorProto], cache: dict[tuple[int, ...], str], vals, name: str) -> str:
    key = tuple(int(v) for v in vals)
    cached = cache.get(key)
    if cached is not None:
        return cached
    cache[key] = _i64(inits, key, name)
    return cache[key]


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference implementation for task245."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    red = np.argwhere(g == 2)
    green = np.argwhere(g == 3)
    if len(red) == 0 or len(green) == 0:
        return out
    gr, gc = green.min(axis=0)
    rmin, cmin = red.min(axis=0)
    dr = int(gr + 1 - rmin)
    dc = int(gc + 1 - cmin)
    out[g == 2] = 0
    for r, c in red:
        nr = int(r + dr)
        nc = int(c + dc)
        if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
            out[nr, nc] = 2
    return out


def _grid_to_onehot(grid: List[List[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def _shift_candidate(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    const_cache: dict[tuple[int, ...], str],
    src: str,
    shift: int,
    axis: int,
    axis_const: str,
    zero_name: str,
    prefix: str,
) -> str:
    if shift == 0:
        return src
    if abs(shift) >= SH:
        return zero_name
    if shift > 0:
        start = _cached_i64(inits, const_cache, [0], "slice_0")
        end = _cached_i64(inits, const_cache, [SH - shift], f"slice_{SH - shift}")
        nodes.append(helper.make_node("Slice", [src, start, end, axis_const], [f"{prefix}_body"]))
        pad_shape = (1, 1, shift, SW) if axis == 2 else (1, 1, SH, shift)
        pad_name = _bool(inits, np.zeros(pad_shape, dtype=np.bool_), f"{prefix}_pad")
        nodes.append(helper.make_node("Concat", [pad_name, f"{prefix}_body"], [f"{prefix}_out"], axis=axis))
        return f"{prefix}_out"

    amount = -shift
    start = _cached_i64(inits, const_cache, [amount], f"slice_{amount}")
    end = _cached_i64(inits, const_cache, [SH], f"slice_{SH}")
    nodes.append(helper.make_node("Slice", [src, start, end, axis_const], [f"{prefix}_body"]))
    pad_shape = (1, 1, amount, SW) if axis == 2 else (1, 1, SH, amount)
    pad_name = _bool(inits, np.zeros(pad_shape, dtype=np.bool_), f"{prefix}_pad")
    nodes.append(helper.make_node("Concat", [f"{prefix}_body", pad_name], [f"{prefix}_out"], axis=axis))
    return f"{prefix}_out"


def _gather_shift(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    const_cache: dict[tuple[int, ...], str],
    src: str,
    shift_value: str,
    axis: int,
    axis_const: str,
    zero_grid: str,
    prefix: str,
) -> str:
    candidates: List[str] = []
    for shift in SHIFTS:
        token = f"{shift:+d}".replace("+", "p").replace("-", "m")
        cand = _shift_candidate(nodes, inits, const_cache, src, shift, axis, axis_const, zero_grid, f"{prefix}_{token}")
        candidates.append(cand)
    nodes.append(helper.make_node("Concat", candidates, [f"{prefix}_stack"], axis=1))
    idx_shape = _cached_i64(inits, const_cache, [1], "shape_1")
    nodes.append(helper.make_node("Reshape", [shift_value, idx_shape], [f"{prefix}_idx"]))
    nodes.append(helper.make_node("Gather", [f"{prefix}_stack", f"{prefix}_idx"], [f"{prefix}_out"], axis=1))
    return f"{prefix}_out"


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    const_cache: dict[tuple[int, ...], str] = {}

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes_chw = _i64(inits, [1, 2, 3], "axes_chw")
    axis_row = _i64(inits, [2], "axis_row")
    axis_col = _i64(inits, [3], "axis_col")
    red_st = _i64(inits, [2, 0, 0], "red_st")
    red_en = _i64(inits, [3, SH, SW], "red_en")
    green_st = _i64(inits, [3, 0, 0], "green_st")
    green_en = _i64(inits, [4, SH, SW], "green_en")
    bg_st = _i64(inits, [0, 0, 0], "bg_st")
    bg_en = _i64(inits, [1, SH, SW], "bg_en")
    one_i = _cached_i64(inits, const_cache, [1], "one_i")
    zero_float = _init(inits, np.zeros((1, 1, SH, SW), dtype=np.float32), "zero_float")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, red_st, red_en, axes_chw], ["red_f"]),
            helper.make_node("Slice", [IN_NAME, green_st, green_en, axes_chw], ["green_f"]),
            helper.make_node("Slice", [IN_NAME, bg_st, bg_en, axes_chw], ["bg_in_f"]),
            helper.make_node("Cast", ["red_f"], ["red"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["green_f"], ["green"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["bg_in_f"], ["bg_in"], to=TensorProto.BOOL),
            helper.make_node("Or", ["red", "green"], ["input_fg"]),
            helper.make_node("Or", ["bg_in", "input_fg"], ["active"]),
            helper.make_node("ReduceMax", ["red_f"], ["red_rows_f"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["red_f"], ["red_cols_f"], axes=[2], keepdims=1),
            helper.make_node("ReduceMax", ["green_f"], ["green_rows_f"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["green_f"], ["green_cols_f"], axes=[2], keepdims=1),
            helper.make_node("ArgMax", ["red_rows_f"], ["red_row_min"], axis=2, keepdims=1),
            helper.make_node("ArgMax", ["red_cols_f"], ["red_col_min"], axis=3, keepdims=1),
            helper.make_node("ArgMax", ["green_rows_f"], ["green_row_min"], axis=2, keepdims=1),
            helper.make_node("ArgMax", ["green_cols_f"], ["green_col_min"], axis=3, keepdims=1),
            helper.make_node("Add", ["green_row_min", one_i], ["green_row_inner"]),
            helper.make_node("Add", ["green_col_min", one_i], ["green_col_inner"]),
            helper.make_node("Sub", ["green_row_inner", "red_row_min"], ["row_shift"]),
            helper.make_node("Sub", ["green_col_inner", "red_col_min"], ["col_shift"]),
        ]
    )

    row_shifted = _gather_shift(nodes, inits, const_cache, "red", "row_shift", 2, axis_row, "", "rowshift")
    moved_red = _gather_shift(nodes, inits, const_cache, row_shifted, "col_shift", 3, axis_col, "", "colshift")

    out_channels = ["bg_f", zero_float, "moved_red_f", "green_f"] + [zero_float] * 6
    nodes.extend(
        [
            helper.make_node("Or", [moved_red, "green"], ["fg_any"]),
            helper.make_node("Not", ["fg_any"], ["not_fg"]),
            helper.make_node("And", ["active", "not_fg"], ["bg"]),
            helper.make_node("Cast", ["bg"], ["bg_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", [moved_red], ["moved_red_f"], to=TensorProto.FLOAT),
            helper.make_node("Concat", out_channels, ["out10"], axis=1),
            helper.make_node("Pad", ["out10"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - SH, W - SW]),
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


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data.get(split, []):
            g = np.array(ex["input"], dtype=np.int64)
            if g.shape[0] > H or g.shape[1] > W:
                continue
            total += 1
            pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(ex["input"])))[: g.shape[0], : g.shape[1]]
            exp = np.array(ex["output"], dtype=np.int64)
            ref = solve(g)
            if not np.array_equal(ref, exp):
                raise AssertionError(f"reference solver mismatch on {split} example {total}")
            if not np.array_equal(pred, exp):
                bad += 1
                print(f"mismatch {split} #{total}:\ninput:\n{g}\npred:\n{pred}\nexp:\n{exp}")
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
