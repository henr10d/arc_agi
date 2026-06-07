"""ONNX for ARC task228: move rectangle interior corner markers outside.

Task rule: the input is a 10x10 grid containing one hollow rectangular frame of
one non-black color and four single-cell markers at the frame's inner corners.
The output preserves only the frame, clears its interior, and moves each marker
to the diagonally opposite outside corner: inner top-left moves to outer
bottom-right, inner top-right to outer bottom-left, inner bottom-left to outer
top-right, and inner bottom-right to outer top-left.

ONNX: work in the 10x10 active region, identify the frame as the most frequent
foreground color, derive top/bottom/left/right masks from that frame, build one
compact int32 color grid with the frame plus four moved marker colors, then
one-hot encode once just before the final float output pad.
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

TASK_ID = "task228"
BEST_PATH = OUT_DIR / "task228.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
NC = 9
H = W = 30
GH = GW = 10
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


def _i32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int32), name)


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver matching the JSON examples."""
    g = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(g)
    nz = g[g != 0]
    frame_color = int(np.bincount(nz).argmax())
    rr, cc = np.where(g == frame_color)
    top, bottom = int(rr.min()), int(rr.max())
    left, right = int(cc.min()), int(cc.max())

    out[top, left : right + 1] = frame_color
    out[bottom, left : right + 1] = frame_color
    out[top : bottom + 1, left] = frame_color
    out[top : bottom + 1, right] = frame_color

    markers = [
        ((top + 1, left + 1), (bottom + 1, right + 1)),
        ((top + 1, right - 1), (bottom + 1, left - 1)),
        ((bottom - 1, left + 1), (top - 1, right + 1)),
        ((bottom - 1, right - 1), (top - 1, left - 1)),
    ]
    for (sr, sc), (dr, dc) in markers:
        color = int(g[sr, sc])
        if color != 0 and color != frame_color:
            out[dr, dc] = color
    return out


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _frame_mask(top: int, bottom: int, left: int, right: int) -> np.ndarray:
    mask = np.zeros((1, 1, GH, GW), dtype=np.float32)
    mask[:, :, top, left : right + 1] = 1.0
    mask[:, :, bottom, left : right + 1] = 1.0
    mask[:, :, top : bottom + 1, left] = 1.0
    mask[:, :, top : bottom + 1, right] = 1.0
    return mask


def _frame_mask_bool(top: int, bottom: int, left: int, right: int) -> np.ndarray:
    return _frame_mask(top, bottom, left, right).astype(np.bool_)


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes_chw = _i64(inits, [1, 2, 3], "axes_chw")
    fg_st = _i64(inits, [1, 0, 0], "fg_st")
    fg_en = _i64(inits, [C, GH, GW], "fg_en")
    axis_row = _i64(inits, [2], "axis_row")
    axis_col = _i64(inits, [3], "axis_col")
    s0 = _i64(inits, [0], "s0")
    s1 = _i64(inits, [1], "s1")
    e9 = _i64(inits, [GH - 1], "e9")
    e10 = _i64(inits, [GH], "e10")
    zero = _f32(inits, [0.0], "zero")
    one_i32 = _i32(inits, [1], "one_i32")
    zero_i32 = _i32(inits, [0], "zero_i32")
    colors_i32 = _i32(inits, np.arange(C, dtype=np.int32).reshape(1, C, 1, 1), "colors_i32")
    false_row = _bool(inits, np.zeros((1, 1, 1, 1), dtype=np.bool_), "false_row")
    false_col = _bool(inits, np.zeros((1, 1, 1, 1), dtype=np.bool_), "false_col")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, fg_st, fg_en, axes_chw], ["fg"]),
            helper.make_node("ArgMax", ["fg"], ["fg_arg"], axis=1, keepdims=1),
            helper.make_node("Cast", ["fg_arg"], ["fg_arg_i32"], to=TensorProto.INT32),
            helper.make_node("ReduceSum", ["fg"], ["counts"], axes=[2, 3], keepdims=0),
            helper.make_node("ArgMax", ["counts"], ["frame_idx"], axis=1, keepdims=0),
            helper.make_node("Cast", ["frame_idx"], ["frame_idx_i32"], to=TensorProto.INT32),
            helper.make_node("Add", ["frame_idx_i32", one_i32], ["frame_color"]),
            helper.make_node("Gather", ["fg", "frame_idx"], ["framef"], axis=1),
            helper.make_node("Greater", ["framef", zero], ["frame"]),
            helper.make_node("ReduceMax", ["framef"], ["rowf"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["framef"], ["colf"], axes=[2], keepdims=1),
            helper.make_node("Greater", ["rowf", zero], ["row_occ"]),
            helper.make_node("Greater", ["colf", zero], ["col_occ"]),
            helper.make_node("Slice", ["row_occ", s0, e9, axis_row], ["row_0_8"]),
            helper.make_node("Slice", ["row_occ", s1, e10, axis_row], ["row_1_9"]),
            helper.make_node("Concat", ["false_row", "row_0_8"], ["row_prev"], axis=2),
            helper.make_node("Concat", ["row_1_9", "false_row"], ["row_next"], axis=2),
            helper.make_node("Not", ["row_prev"], ["not_row_prev"]),
            helper.make_node("Not", ["row_next"], ["not_row_next"]),
            helper.make_node("And", ["row_occ", "not_row_prev"], ["top"]),
            helper.make_node("And", ["row_occ", "not_row_next"], ["bottom"]),
            helper.make_node("Slice", ["top", s0, e9, axis_row], ["top_0_8"]),
            helper.make_node("Slice", ["top", s1, e10, axis_row], ["top_1_9"]),
            helper.make_node("Slice", ["bottom", s0, e9, axis_row], ["bottom_0_8"]),
            helper.make_node("Slice", ["bottom", s1, e10, axis_row], ["bottom_1_9"]),
            helper.make_node("Concat", ["false_row", "top_0_8"], ["top_in"], axis=2),
            helper.make_node("Concat", ["top_1_9", "false_row"], ["top_out"], axis=2),
            helper.make_node("Concat", ["bottom_1_9", "false_row"], ["bottom_in"], axis=2),
            helper.make_node("Concat", ["false_row", "bottom_0_8"], ["bottom_out"], axis=2),
            helper.make_node("Slice", ["col_occ", s0, e9, axis_col], ["col_0_8"]),
            helper.make_node("Slice", ["col_occ", s1, e10, axis_col], ["col_1_9"]),
            helper.make_node("Concat", ["false_col", "col_0_8"], ["col_prev"], axis=3),
            helper.make_node("Concat", ["col_1_9", "false_col"], ["col_next"], axis=3),
            helper.make_node("Not", ["col_prev"], ["not_col_prev"]),
            helper.make_node("Not", ["col_next"], ["not_col_next"]),
            helper.make_node("And", ["col_occ", "not_col_prev"], ["left"]),
            helper.make_node("And", ["col_occ", "not_col_next"], ["right"]),
            helper.make_node("Slice", ["left", s0, e9, axis_col], ["left_0_8"]),
            helper.make_node("Slice", ["left", s1, e10, axis_col], ["left_1_9"]),
            helper.make_node("Slice", ["right", s0, e9, axis_col], ["right_0_8"]),
            helper.make_node("Slice", ["right", s1, e10, axis_col], ["right_1_9"]),
            helper.make_node("Concat", ["false_col", "left_0_8"], ["left_in"], axis=3),
            helper.make_node("Concat", ["left_1_9", "false_col"], ["left_out"], axis=3),
            helper.make_node("Concat", ["right_1_9", "false_col"], ["right_in"], axis=3),
            helper.make_node("Concat", ["false_col", "right_0_8"], ["right_out"], axis=3),
        ]
    )

    corner_pairs = (
        ("br_to_tl", "bottom_in", "right_in", "top_out", "left_out"),
        ("bl_to_tr", "bottom_in", "left_in", "top_out", "right_out"),
        ("tr_to_bl", "top_in", "right_in", "bottom_out", "left_out"),
        ("tl_to_br", "top_in", "left_in", "bottom_out", "right_out"),
    )
    accum = "frame_grid"
    nodes.append(helper.make_node("Where", ["frame", "frame_color", zero_i32], [accum]))
    for name, src_row, src_col, dst_row, dst_col in corner_pairs:
        nodes.extend(
            [
                helper.make_node("And", [src_row, src_col], [f"{name}_src"]),
                helper.make_node("And", [dst_row, dst_col], [f"{name}_dst"]),
                helper.make_node("Where", [f"{name}_src", "fg_arg_i32", zero_i32], [f"{name}_src_grid"]),
                helper.make_node(
                    "ReduceMax",
                    [f"{name}_src_grid"],
                    [f"{name}_color0"],
                    axes=[2, 3],
                    keepdims=1,
                ),
                helper.make_node("Add", [f"{name}_color0", one_i32], [f"{name}_color"]),
                helper.make_node("Where", [f"{name}_dst", f"{name}_color", accum], [f"{name}_grid"]),
            ]
        )
        accum = f"{name}_grid"

    nodes.extend(
        [
            helper.make_node("Equal", [accum, colors_i32], ["out10b"]),
            helper.make_node("Cast", ["out10b"], ["out10"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out10"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - GH, W - GW]),
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
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            inp = np.asarray(ex["input"], dtype=np.int64)
            expected = np.asarray(ex["output"], dtype=np.int64)
            ref = solve(inp)
            pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(ex["input"])))[: expected.shape[0], : expected.shape[1]]
            if not np.array_equal(ref, expected):
                print(f"reference mismatch {split} {idx}")
                bad += 1
            elif not np.array_equal(pred, expected):
                print(f"onnx mismatch {split} {idx}")
                print(pred)
                print(expected)
                bad += 1
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
