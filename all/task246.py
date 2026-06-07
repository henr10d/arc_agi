"""Minimal ONNX for ARC task246: draw a red-to-green right-angle connector.

Task rule: each grid contains exactly one red cell (2) and one green cell (3).
Preserve both endpoints, then draw cyan (8) on the L-shaped orthogonal path
whose corner is at (red_row, green_col): a horizontal segment across the red
row and a vertical segment down/up the green column. Grid size is unchanged.

ONNX: work on the observed 20x20 task region, infer red/green row and column
one-hot masks with reductions, use a shared triangular matrix to build inclusive
row/column intervals, assemble bool output channels, cast once, then pad to the
30x30 NeuroGolf I/O shape.
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

TASK_ID = "task246"
BEST_PATH = OUT_DIR / "task246.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
SH = SW = 20
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


def _bool(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.bool_), name)


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver for the red-row/green-column L connector."""
    g = np.asarray(grid, dtype=np.int64)
    red = np.argwhere(g == 2)
    green = np.argwhere(g == 3)
    assert len(red) == 1, f"expected exactly one red cell, got {len(red)}"
    assert len(green) == 1, f"expected exactly one green cell, got {len(green)}"

    rr, rc = map(int, red[0])
    gr, gc = map(int, green[0])
    out = g.copy()
    out[rr, min(rc, gc) : max(rc, gc) + 1] = 8
    out[min(rr, gr) : max(rr, gr) + 1, gc] = 8
    out[rr, rc] = 2
    out[gr, gc] = 3

    assert out[rr, rc] == 2
    assert out[gr, gc] == 3
    row_cells = {(rr, c) for c in range(min(rc, gc), max(rc, gc) + 1)}
    col_cells = {(r, gc) for r in range(min(rr, gr), max(rr, gr) + 1)}
    connector = (row_cells | col_cells) - {(rr, rc), (gr, gc)}
    assert (rr, gc) in row_cells | col_cells
    if (rr, gc) in connector:
        assert out[rr, gc] == 8
    assert all(out[r, c] == 8 for r, c in connector)
    return out


def _grid_to_onehot(grid: List[List[int]] | np.ndarray) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    arr = np.asarray(grid, dtype=np.int64)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def _interval_from_positions(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    first: str,
    second: str,
    shape: List[int],
    out_name: str,
) -> str:
    """Inclusive interval mask between two 1D one-hot position vectors."""
    mat = "tri"
    zero = "zero"
    if not any(init.name == mat for init in inits):
        tri = np.tri(SH, SH, dtype=np.float32).T
        _f32(inits, tri, mat)
    if not any(init.name == zero for init in inits):
        _f32(inits, [0.0], zero)

    first_flat = f"{out_name}_first_flat"
    second_flat = f"{out_name}_second_flat"
    first_prefix = f"{out_name}_first_prefix"
    second_prefix = f"{out_name}_second_prefix"
    first_pos = f"{out_name}_first_pos"
    second_pos = f"{out_name}_second_pos"
    first_le = f"{out_name}_first_le"
    second_le = f"{out_name}_second_le"
    left_only = f"{out_name}_left_only"
    right_only = f"{out_name}_right_only"
    open_interval = f"{out_name}_open"
    endpoints = f"{out_name}_ends"

    flat_shape = _i64(inits, [1, SH], f"{out_name}_flat_shape")
    out_shape = _i64(inits, shape, f"{out_name}_shape")
    nodes.extend(
        [
            helper.make_node("Reshape", [first, flat_shape], [first_flat]),
            helper.make_node("Reshape", [second, flat_shape], [second_flat]),
            helper.make_node("MatMul", [first_flat, mat], [first_prefix]),
            helper.make_node("MatMul", [second_flat, mat], [second_prefix]),
            helper.make_node("Greater", [first_flat, zero], [first_pos]),
            helper.make_node("Greater", [second_flat, zero], [second_pos]),
            helper.make_node("Greater", [first_prefix, zero], [first_le]),
            helper.make_node("Greater", [second_prefix, zero], [second_le]),
            helper.make_node("And", [first_le, second_le], [left_only]),
            helper.make_node("Or", [first_le, second_le], [right_only]),
            helper.make_node("Not", [left_only], [f"{out_name}_not_both"]),
            helper.make_node("And", [right_only, f"{out_name}_not_both"], [open_interval]),
            helper.make_node("Or", [first_pos, second_pos], [endpoints]),
            helper.make_node("Or", [open_interval, endpoints], [f"{out_name}_flat"]),
            helper.make_node("Reshape", [f"{out_name}_flat", out_shape], [out_name]),
        ]
    )
    return out_name


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes_chw = _i64(inits, [1, 2, 3], "axes_chw")
    red_st = _i64(inits, [2, 0, 0], "red_st")
    red_en = _i64(inits, [3, SH, SW], "red_en")
    green_st = _i64(inits, [3, 0, 0], "green_st")
    green_en = _i64(inits, [4, SH, SW], "green_en")
    bg_st = _i64(inits, [0, 0, 0], "bg_st")
    bg_en = _i64(inits, [1, SH, SW], "bg_en")
    zero_f = _f32(inits, [0.0], "zero_f")
    zero_full = _f32(inits, np.zeros((1, 1, H, W), dtype=np.float32), "zero_full")
    pads = [0, 0, 0, 0, 0, 0, H - SH, W - SW]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, red_st, red_en, axes_chw], ["redf"]),
            helper.make_node("Slice", [IN_NAME, green_st, green_en, axes_chw], ["greenf"]),
            helper.make_node("Slice", [IN_NAME, bg_st, bg_en, axes_chw], ["bgf"]),
            helper.make_node("Cast", ["redf"], ["red"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["greenf"], ["green"], to=TensorProto.BOOL),
            helper.make_node("ReduceMax", ["redf"], ["red_row_f"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["redf"], ["red_col_f"], axes=[2], keepdims=1),
            helper.make_node("ReduceMax", ["greenf"], ["green_row_f"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["greenf"], ["green_col_f"], axes=[2], keepdims=1),
            helper.make_node("Cast", ["red_row_f"], ["red_row"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["green_col_f"], ["green_col"], to=TensorProto.BOOL),
        ]
    )

    _interval_from_positions(nodes, inits, "red_col_f", "green_col_f", [1, 1, 1, SW], "col_between")
    _interval_from_positions(nodes, inits, "red_row_f", "green_row_f", [1, 1, SH, 1], "row_between")

    nodes.extend(
        [
            helper.make_node("And", ["red_row", "col_between"], ["hseg"]),
            helper.make_node("And", ["green_col", "row_between"], ["vseg"]),
            helper.make_node("Or", ["hseg", "vseg"], ["connector"]),
            helper.make_node("Or", ["red", "green"], ["endpoints"]),
            helper.make_node("Not", ["endpoints"], ["not_endpoints"]),
            helper.make_node("And", ["connector", "not_endpoints"], ["cyan"]),
            helper.make_node("Where", ["connector", "zero_f", "bgf"], ["out0f"]),
            helper.make_node("Cast", ["cyan"], ["cyanf"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out0f"], ["out0p"], pads=pads),
            helper.make_node("Pad", ["redf"], ["redp"], pads=pads),
            helper.make_node("Pad", ["greenf"], ["greenp"], pads=pads),
            helper.make_node("Pad", ["cyanf"], ["cyanp"], pads=pads),
            helper.make_node(
                "Concat",
                [
                    "out0p",
                    zero_full,
                    "redp",
                    "greenp",
                    zero_full,
                    zero_full,
                    zero_full,
                    zero_full,
                    "cyanp",
                    zero_full,
                ],
                [OUT_NAME],
                axis=1,
            ),
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


def validate_hypotheses() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    def alt(grid: np.ndarray, corner: str) -> np.ndarray:
        red = np.argwhere(grid == 2)
        green = np.argwhere(grid == 3)
        assert len(red) == 1 and len(green) == 1
        rr, rc = map(int, red[0])
        gr, gc = map(int, green[0])
        out = grid.copy()
        if corner == "red_row_green_col":
            row, col = rr, gc
        elif corner == "green_row_red_col":
            row, col = gr, rc
        else:
            raise ValueError(corner)
        out[row, min(rc, gc) : max(rc, gc) + 1] = 8
        out[min(rr, gr) : max(rr, gr) + 1, col] = 8
        out[rr, rc] = 2
        out[gr, gc] = 3
        return out

    failures = {"red_row_green_col": 0, "green_row_red_col": 0}
    total = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            inp = np.asarray(ex["input"], dtype=np.int64)
            exp = np.asarray(ex["output"], dtype=np.int64)
            total += 1
            red = np.argwhere(inp == 2)
            green = np.argwhere(inp == 3)
            assert len(red) == 1, f"{split}: expected one red, got {len(red)}"
            assert len(green) == 1, f"{split}: expected one green, got {len(green)}"
            assert np.array_equal(solve(inp), exp), f"{split}: reference solver mismatch"
            for name in failures:
                failures[name] += int(not np.array_equal(alt(inp, name), exp))

    assert failures["red_row_green_col"] == 0
    assert failures["green_row_red_col"] == total
    print(f"hypothesis red_row_green_col: PASS ({total}/{total})")
    print(f"alternative green_row_red_col / vertical-first: FAIL ({failures['green_row_red_col']}/{total})")


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            inp = np.asarray(ex["input"], dtype=np.int64)
            exp = np.asarray(ex["output"], dtype=np.int64)
            oh = _grid_to_onehot(inp)
            pred = _onehot_to_grid(_run_onnx(model, oh))[: inp.shape[0], : inp.shape[1]]
            if not np.array_equal(pred, exp):
                print(f"{split}[{idx}] mismatch")
                bad += 1
    return bad


def main() -> None:
    validate_hypotheses()
    model = build_model()
    onnx.save(model, BEST_PATH)

    bad = validate_json(model)
    print(f"task246.json: {'PASS' if bad == 0 else f'FAIL ({bad} wrong)'}")

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
