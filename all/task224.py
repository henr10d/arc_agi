"""ONNX for ARC task224: draw an outer square through gray guide markers.

Task rule: keep the input grid unchanged, find the four gray marker pixels
(color 5), and draw a larger hollow square in the same non-background,
non-gray color as the existing inner ring. The new square perimeter is one
cell inside the min/max gray rows and columns. Gray marker pixels remain gray.
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

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

BEST_PATH = OUT_DIR / "task224.onnx"
DATA_PATH = ROOT / "data" / "task224.json"

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


def _bool(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.bool_), name)


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference implementation for the marker-defined outer ring."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    gray = np.argwhere(g == 5)
    top = int(gray[:, 0].min()) + 1
    bottom = int(gray[:, 0].max()) - 1
    left = int(gray[:, 1].min()) + 1
    right = int(gray[:, 1].max()) - 1
    colors = [int(v) for v in np.unique(g) if int(v) not in (0, 5)]
    color = colors[0]

    out[top, left : right + 1] = color
    out[bottom, left : right + 1] = color
    out[top : bottom + 1, left] = color
    out[top : bottom + 1, right] = color
    out[g == 5] = 5
    return out


def _grid_to_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(arr):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _common_extrema(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto]) -> None:
    axes_chw = _i64(inits, [1, 2, 3], "axes_chw")
    ch5_st = _i64(inits, [5, 0, 0], "ch5_st")
    ch5_en = _i64(inits, [6, H, W], "ch5_en")
    row_coord = _f32(inits, np.arange(H, dtype=np.float32).reshape(1, 1, H, 1), "row_coord")
    col_coord = _f32(inits, np.arange(W, dtype=np.float32).reshape(1, 1, 1, W), "col_coord")
    row_full = _f32(inits, np.broadcast_to(np.arange(H, dtype=np.float32).reshape(1, 1, H, 1), (1, 1, H, W)), "row_full")
    col_full = _f32(inits, np.broadcast_to(np.arange(W, dtype=np.float32).reshape(1, 1, 1, W), (1, 1, H, W)), "col_full")
    high = _f32(inits, np.full((1, 1, H, W), H, dtype=np.float32), "high")
    one_f = _f32(inits, [1.0], "one_f")
    zero_f = _f32(inits, [0.0], "zero_f")
    valid_ch = np.ones((1, C, 1, 1), dtype=np.bool_)
    valid_ch[:, 0] = False
    valid_ch[:, 5] = False
    _bool(inits, valid_ch, "valid_ch")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, ch5_st, ch5_en, axes_chw], ["gray_f"]),
            helper.make_node("Greater", ["gray_f", zero_f], ["gray"]),
            helper.make_node("Mul", ["gray_f", "row_full"], ["row_max_candidates"]),
            helper.make_node("Sub", ["one_f", "gray_f"], ["not_gray_f"]),
            helper.make_node("Mul", ["not_gray_f", "high"], ["row_high"]),
            helper.make_node("Add", ["row_max_candidates", "row_high"], ["row_min_candidates"]),
            helper.make_node("ReduceMin", ["row_min_candidates"], ["top_marker"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMax", ["row_max_candidates"], ["bottom_marker"], axes=[2, 3], keepdims=1),
            helper.make_node("Mul", ["gray_f", "col_full"], ["col_max_candidates"]),
            helper.make_node("Mul", ["not_gray_f", "high"], ["col_high"]),
            helper.make_node("Add", ["col_max_candidates", "col_high"], ["col_min_candidates"]),
            helper.make_node("ReduceMin", ["col_min_candidates"], ["left_marker"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMax", ["col_max_candidates"], ["right_marker"], axes=[2, 3], keepdims=1),
            helper.make_node("Add", ["top_marker", "one_f"], ["top"]),
            helper.make_node("Sub", ["bottom_marker", "one_f"], ["bottom"]),
            helper.make_node("Add", ["left_marker", "one_f"], ["left"]),
            helper.make_node("Sub", ["right_marker", "one_f"], ["right"]),
            helper.make_node("Less", ["row_coord", "top"], ["row_lt_top_eq"]),
            helper.make_node("Not", ["row_lt_top_eq"], ["row_ge_top_eq"]),
            helper.make_node("Greater", ["row_coord", "top"], ["row_gt_top_eq"]),
            helper.make_node("Not", ["row_gt_top_eq"], ["row_le_top_eq"]),
            helper.make_node("And", ["row_ge_top_eq", "row_le_top_eq"], ["top_row"]),
            helper.make_node("Less", ["row_coord", "bottom"], ["row_lt_bottom_eq"]),
            helper.make_node("Not", ["row_lt_bottom_eq"], ["row_ge_bottom_eq"]),
            helper.make_node("Greater", ["row_coord", "bottom"], ["row_gt_bottom_eq"]),
            helper.make_node("Not", ["row_gt_bottom_eq"], ["row_le_bottom_eq"]),
            helper.make_node("And", ["row_ge_bottom_eq", "row_le_bottom_eq"], ["bottom_row"]),
            helper.make_node("Less", ["col_coord", "left"], ["col_lt_left_eq"]),
            helper.make_node("Not", ["col_lt_left_eq"], ["col_ge_left_eq"]),
            helper.make_node("Greater", ["col_coord", "left"], ["col_gt_left_eq"]),
            helper.make_node("Not", ["col_gt_left_eq"], ["col_le_left_eq"]),
            helper.make_node("And", ["col_ge_left_eq", "col_le_left_eq"], ["left_col"]),
            helper.make_node("Less", ["col_coord", "right"], ["col_lt_right_eq"]),
            helper.make_node("Not", ["col_lt_right_eq"], ["col_ge_right_eq"]),
            helper.make_node("Greater", ["col_coord", "right"], ["col_gt_right_eq"]),
            helper.make_node("Not", ["col_gt_right_eq"], ["col_le_right_eq"]),
            helper.make_node("And", ["col_ge_right_eq", "col_le_right_eq"], ["right_col"]),
            helper.make_node("Less", ["col_coord", "left"], ["col_lt_left"]),
            helper.make_node("Not", ["col_lt_left"], ["col_ge_left"]),
            helper.make_node("Greater", ["col_coord", "right"], ["col_gt_right"]),
            helper.make_node("Not", ["col_gt_right"], ["col_le_right"]),
            helper.make_node("And", ["col_ge_left", "col_le_right"], ["col_in"]),
            helper.make_node("Less", ["row_coord", "top"], ["row_lt_top"]),
            helper.make_node("Not", ["row_lt_top"], ["row_ge_top"]),
            helper.make_node("Greater", ["row_coord", "bottom"], ["row_gt_bottom"]),
            helper.make_node("Not", ["row_gt_bottom"], ["row_le_bottom"]),
            helper.make_node("And", ["row_ge_top", "row_le_bottom"], ["row_in"]),
            helper.make_node("Greater", [IN_NAME, zero_f], ["input_b"]),
            helper.make_node("ReduceMax", [IN_NAME], ["color_max"], axes=[2, 3], keepdims=1),
            helper.make_node("Greater", ["color_max", zero_f], ["has_color"]),
            helper.make_node("And", ["has_color", "valid_ch"], ["draw_color"]),
        ]
    )


def _finish(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], graph_name: str) -> onnx.ModelProto:
    nodes.extend(
        [
            helper.make_node("Not", ["gray"], ["not_gray"]),
            helper.make_node("And", ["perimeter", "not_gray"], ["draw_here"]),
            helper.make_node("And", ["draw_color", "perimeter"], ["perimeter_color"]),
            helper.make_node("And", ["draw_here", "perimeter_color"], ["drawn_b"]),
            helper.make_node("Not", ["draw_here"], ["keep_here"]),
            helper.make_node("And", ["keep_here", "input_b"], ["kept_b"]),
            helper.make_node("Or", ["drawn_b", "kept_b"], ["out_b"]),
            helper.make_node("Cast", ["out_b"], [OUT_NAME], to=TensorProto.FLOAT),
        ]
    )

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(nodes, graph_name, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_direct_model() -> onnx.ModelProto:
    """Variant A: construct the four bounded sides directly."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    _common_extrema(nodes, inits)
    nodes.extend(
        [
            helper.make_node("Or", ["top_row", "bottom_row"], ["edge_row"]),
            helper.make_node("And", ["edge_row", "col_in"], ["h_sides"]),
            helper.make_node("Or", ["left_col", "right_col"], ["edge_col"]),
            helper.make_node("And", ["row_in", "edge_col"], ["v_sides"]),
            helper.make_node("Or", ["h_sides", "v_sides"], ["perimeter"]),
        ]
    )
    return _finish(nodes, inits, "task224_direct")


def build_mask_model() -> onnx.ModelProto:
    """Variant B: make a filled box and remove the strict interior."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    _common_extrema(nodes, inits)
    nodes.extend(
        [
            helper.make_node("And", ["row_in", "col_in"], ["filled_box"]),
            helper.make_node("Greater", ["row_coord", "top"], ["row_gt_top_strict"]),
            helper.make_node("Less", ["row_coord", "bottom"], ["row_lt_bottom_strict"]),
            helper.make_node("And", ["row_gt_top_strict", "row_lt_bottom_strict"], ["row_inner"]),
            helper.make_node("Greater", ["col_coord", "left"], ["col_gt_left_strict"]),
            helper.make_node("Less", ["col_coord", "right"], ["col_lt_right_strict"]),
            helper.make_node("And", ["col_gt_left_strict", "col_lt_right_strict"], ["col_inner"]),
            helper.make_node("And", ["row_inner", "col_inner"], ["inner_box"]),
            helper.make_node("Not", ["inner_box"], ["not_inner"]),
            helper.make_node("And", ["filled_box", "not_inner"], ["perimeter"]),
        ]
    )
    return _finish(nodes, inits, "task224_mask")


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
            exp = np.asarray(ex["output"], dtype=np.int64)
            if inp.shape[0] > H or inp.shape[1] > W:
                continue
            ref = solve(inp)
            if not np.array_equal(ref, exp):
                raise AssertionError(f"reference mismatch in {split}[{idx}]")
            pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(inp)))[: exp.shape[0], : exp.shape[1]]
            if not np.array_equal(pred, exp):
                bad += 1
    return bad


def main() -> None:
    candidates = [
        ("direct", build_direct_model()),
        ("mask", build_mask_model()),
    ]
    scored = []
    for name, model in candidates:
        bad = validate_json(model)
        path = OUT_DIR / f"task224_{name}.onnx"
        onnx.save(model, path)
        result = score_file(path)
        scored.append((result["score"] if result["score"] is not None else -1.0, name, model, path, bad, result))
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
    best_score, best_name, best_model, _, best_bad, best_result = scored[0]
    if best_bad != 0 or not best_result["valid"]:
        raise AssertionError(f"best candidate {best_name} is not valid and correct")
    onnx.save(best_model, BEST_PATH)
    print(f"wrote {BEST_PATH} from {best_name}")
    print(f"best score: {best_score:.6f}")


if __name__ == "__main__":
    main()
