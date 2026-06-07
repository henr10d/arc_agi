"""ONNX generator for ARC task268: fill an open box in its opening direction.

Task rule: the input grid contains one non-black axis-aligned hollow shape.
It is a rectangular box with one side opened by a contiguous gap.  Preserve the
original colored outline, fill every background cell inside the box with yellow
(color 4), and project that open-side gap outward to the grid edge.  The
projection keeps the gap columns/rows filled and adds the two diagonal boundary
rays from the gap corners, clipped by the visible grid.

ONNX: infer the object bounding box and the missing side gap from one-hot input,
build all four possible directional masks with integer row/column coordinate
comparisons, select the active direction, then change only black cells covered
by the fill mask to channel 4.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task268"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task268.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
G = 10
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


def _unary(nodes: List[onnx.NodeProto], op: str, x: str, y: str, **attrs: Any) -> str:
    nodes.append(helper.make_node(op, [x], [y], **attrs))
    return y


def _binary(nodes: List[onnx.NodeProto], op: str, a: str, b: str, y: str, **attrs: Any) -> str:
    nodes.append(helper.make_node(op, [a, b], [y], **attrs))
    return y


def _not(nodes: List[onnx.NodeProto], x: str, y: str) -> str:
    return _unary(nodes, "Not", x, y)


def _and(nodes: List[onnx.NodeProto], a: str, b: str, y: str) -> str:
    return _binary(nodes, "And", a, b, y)


def _or(nodes: List[onnx.NodeProto], a: str, b: str, y: str) -> str:
    return _binary(nodes, "Or", a, b, y)


def _where(nodes: List[onnx.NodeProto], cond: str, a: str, b: str, y: str) -> str:
    nodes.append(helper.make_node("Where", [cond, a, b], [y]))
    return y


def _between(nodes: List[onnx.NodeProto], x: str, lo: str, hi: str, name: str) -> str:
    lt = _binary(nodes, "Less", x, lo, f"{name}_lt")
    gt = _binary(nodes, "Greater", x, hi, f"{name}_gt")
    ge = _not(nodes, lt, f"{name}_ge")
    le = _not(nodes, gt, f"{name}_le")
    return _and(nodes, ge, le, name)


def _slice_channel_range(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    source: str,
    start: int,
    end: int,
    name: str,
) -> str:
    starts = _i64(inits, [0, start, 0, 0], f"{name}_starts")
    ends = _i64(inits, [1, end, G, G], f"{name}_ends")
    nodes.append(helper.make_node("Slice", [source, starts, ends], [name]))
    return name


def _minmax_last_axis(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    mask: str,
    axis: int,
    rev: str,
    name: str,
) -> tuple[str, str, str]:
    f = _unary(nodes, "Cast", mask, f"{name}_f", to=TensorProto.FLOAT)
    other_axis = 2 if axis == 3 else 3
    nodes.append(helper.make_node("ReduceMax", [f], [f"{name}_has"], axes=[other_axis], keepdims=1))
    nodes.append(helper.make_node("ArgMax", [f"{name}_has"], [f"{name}_min"], axis=axis, keepdims=1))
    nodes.append(helper.make_node("Gather", [f"{name}_has", rev], [f"{name}_rev"], axis=axis))
    nodes.append(helper.make_node("ArgMax", [f"{name}_rev"], [f"{name}_rev_arg"], axis=axis, keepdims=1))
    last = _i64(inits, G - 1, f"{name}_last")
    _binary(nodes, "Sub", last, f"{name}_rev_arg", f"{name}_max")
    return f"{name}_min", f"{name}_max", f"{name}_has"


def _projection_has_gap(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    projection: str,
    name: str,
) -> str:
    nodes.append(helper.make_node("ReduceMax", [projection], [f"{name}_max"], axes=[2, 3], keepdims=1))
    zero = _f32(inits, 0.0, f"{name}_zero")
    return _binary(nodes, "Greater", f"{name}_max", zero, name)


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    row_coords = _i64(inits, np.arange(G, dtype=np.int64).reshape(1, 1, G, 1), "row_coords")
    col_coords = _i64(inits, np.arange(G, dtype=np.int64).reshape(1, 1, 1, G), "col_coords")
    rev = _i64(inits, np.arange(G - 1, -1, -1, dtype=np.int64), "rev")

    ch1_st = _i64(inits, [0, 1, 0, 0], "ch1_st")
    ch1_en = _i64(inits, [1, C, G, G], "ch1_en")
    nodes.append(helper.make_node("Slice", [IN_NAME, ch1_st, ch1_en], ["nonblack"]))
    nodes.append(helper.make_node("ReduceSum", ["nonblack"], ["obj_f"], axes=[1], keepdims=1))
    nodes.append(helper.make_node("Cast", ["obj_f"], ["obj"], to=TensorProto.BOOL))
    _not(nodes, "obj", "not_obj")

    nodes.append(helper.make_node("ReduceMax", ["obj_f"], ["row_has_f"], axes=[3], keepdims=1))
    nodes.append(helper.make_node("ReduceMax", ["obj_f"], ["col_has_f"], axes=[2], keepdims=1))
    nodes.append(helper.make_node("ArgMax", ["row_has_f"], ["top"], axis=2, keepdims=1))
    nodes.append(helper.make_node("ArgMax", ["col_has_f"], ["left"], axis=3, keepdims=1))
    nodes.append(helper.make_node("Gather", ["row_has_f", rev], ["row_has_rev"], axis=2))
    nodes.append(helper.make_node("Gather", ["col_has_f", rev], ["col_has_rev"], axis=3))
    nodes.append(helper.make_node("ArgMax", ["row_has_rev"], ["bottom_rev"], axis=2, keepdims=1))
    nodes.append(helper.make_node("ArgMax", ["col_has_rev"], ["right_rev"], axis=3, keepdims=1))
    last = _i64(inits, G - 1, "last")
    _binary(nodes, "Sub", last, "bottom_rev", "bottom")
    _binary(nodes, "Sub", last, "right_rev", "right")

    row_in = _between(nodes, row_coords, "top", "bottom", "row_in")
    col_in = _between(nodes, col_coords, "left", "right", "col_in")
    bbox = _and(nodes, row_in, col_in, "bbox")
    row_eq_top = _binary(nodes, "Equal", row_coords, "top", "row_eq_top")
    row_eq_bottom = _binary(nodes, "Equal", row_coords, "bottom", "row_eq_bottom")
    col_eq_left = _binary(nodes, "Equal", col_coords, "left", "col_eq_left")
    col_eq_right = _binary(nodes, "Equal", col_coords, "right", "col_eq_right")
    row_lt_top = _binary(nodes, "Less", row_coords, "top", "row_lt_top")
    row_gt_bottom = _binary(nodes, "Greater", row_coords, "bottom", "row_gt_bottom")
    col_lt_left = _binary(nodes, "Less", col_coords, "left", "col_lt_left")
    col_gt_right = _binary(nodes, "Greater", col_coords, "right", "col_gt_right")

    top_side = _and(nodes, row_eq_top, col_in, "top_side")
    bottom_side = _and(nodes, row_eq_bottom, col_in, "bottom_side")
    left_side = _and(nodes, row_in, col_eq_left, "left_side")
    right_side = _and(nodes, row_in, col_eq_right, "right_side")
    top_gap = _and(nodes, top_side, "not_obj", "top_gap")
    bottom_gap = _and(nodes, bottom_side, "not_obj", "bottom_gap")
    left_gap = _and(nodes, left_side, "not_obj", "left_gap")
    right_gap = _and(nodes, right_side, "not_obj", "right_gap")

    up_min, up_max, up_has = _minmax_last_axis(nodes, inits, top_gap, 3, rev, "up_gap")
    down_min, down_max, down_has = _minmax_last_axis(nodes, inits, bottom_gap, 3, rev, "down_gap")
    left_min, left_max, left_has = _minmax_last_axis(nodes, inits, left_gap, 2, rev, "left_gap_mm")
    right_min, right_max, right_has = _minmax_last_axis(nodes, inits, right_gap, 2, rev, "right_gap_mm")

    open_up = _projection_has_gap(nodes, inits, up_has, "open_up")
    open_down = _projection_has_gap(nodes, inits, down_has, "open_down")
    open_left = _projection_has_gap(nodes, inits, left_has, "open_left")
    open_right = _projection_has_gap(nodes, inits, right_has, "open_right")

    open_vertical = _or(nodes, open_up, open_down, "open_vertical")
    open_horizontal = _or(nodes, open_left, open_right, "open_horizontal")

    v_min = _where(nodes, open_up, up_min, down_min, "v_min")
    v_max = _where(nodes, open_up, up_max, down_max, "v_max")
    v_center = _between(nodes, col_coords, v_min, v_max, "v_center")
    up_d = _binary(nodes, "Sub", "top", row_coords, "up_d")
    down_d = _binary(nodes, "Sub", row_coords, "bottom", "down_d")
    v_d = _where(nodes, open_up, up_d, down_d, "v_d")
    v_lcol = _binary(nodes, "Sub", v_min, v_d, "v_lcol")
    v_rcol = _binary(nodes, "Add", v_max, v_d, "v_rcol")
    v_diag_l = _binary(nodes, "Equal", col_coords, v_lcol, "v_diag_l")
    v_diag_r = _binary(nodes, "Equal", col_coords, v_rcol, "v_diag_r")
    v_diag = _or(nodes, v_diag_l, v_diag_r, "v_diag")
    v_out_base = _or(nodes, v_center, v_diag, "v_out_base")
    up_out_rows = _and(nodes, open_up, row_lt_top, "up_out_rows")
    down_out_rows = _and(nodes, open_down, row_gt_bottom, "down_out_rows")
    v_out_rows = _or(nodes, up_out_rows, down_out_rows, "v_out_rows")
    v_out = _and(nodes, v_out_rows, v_out_base, "v_out")
    v_mask = _or(nodes, bbox, v_out, "v_mask")
    v_selected = _and(nodes, open_vertical, v_mask, "v_selected")

    h_min = _where(nodes, open_left, left_min, right_min, "h_min")
    h_max = _where(nodes, open_left, left_max, right_max, "h_max")
    h_center = _between(nodes, row_coords, h_min, h_max, "h_center")
    left_d = _binary(nodes, "Sub", "left", col_coords, "left_d")
    right_d = _binary(nodes, "Sub", col_coords, "right", "right_d")
    h_d = _where(nodes, open_left, left_d, right_d, "h_d")
    h_trow = _binary(nodes, "Sub", h_min, h_d, "h_trow")
    h_brow = _binary(nodes, "Add", h_max, h_d, "h_brow")
    h_diag_t = _binary(nodes, "Equal", row_coords, h_trow, "h_diag_t")
    h_diag_b = _binary(nodes, "Equal", row_coords, h_brow, "h_diag_b")
    h_diag = _or(nodes, h_diag_t, h_diag_b, "h_diag")
    h_out_base = _or(nodes, h_center, h_diag, "h_out_base")
    left_out_cols = _and(nodes, open_left, col_lt_left, "left_out_cols")
    right_out_cols = _and(nodes, open_right, col_gt_right, "right_out_cols")
    h_out_cols = _or(nodes, left_out_cols, right_out_cols, "h_out_cols")
    h_out = _and(nodes, h_out_cols, h_out_base, "h_out")
    h_mask = _or(nodes, bbox, h_out, "h_mask")
    h_selected = _and(nodes, open_horizontal, h_mask, "h_selected")

    fill = _or(nodes, v_selected, h_selected, "fill")

    ch0 = _slice_channel_range(nodes, inits, IN_NAME, 0, 1, "ch0")
    ch1_3 = _slice_channel_range(nodes, inits, IN_NAME, 1, 4, "ch1_3")
    ch4 = _slice_channel_range(nodes, inits, IN_NAME, 4, 5, "ch4")
    ch5_9 = _slice_channel_range(nodes, inits, IN_NAME, 5, 10, "ch5_9")
    nodes.append(helper.make_node("Cast", [ch0], ["ch0_b"], to=TensorProto.BOOL))
    fill_black = _and(nodes, fill, "ch0_b", "fill_black")
    zero = _f32(inits, 0.0, "zero")
    one = _f32(inits, 1.0, "one")
    nodes.append(helper.make_node("Where", [fill, zero, ch0], ["ch0_out"]))
    nodes.append(helper.make_node("Where", [fill_black, one, ch4], ["ch4_out"]))
    nodes.append(helper.make_node("Concat", ["ch0_out", ch1_3, "ch4_out", ch5_9], ["out10"], axis=1))
    nodes.append(
        helper.make_node(
            "Pad",
            ["out10"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - G, W - G],
        )
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


def validate_examples(path: Path) -> tuple[int, int, dict[str, int]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    split_passed: dict[str, int] = {}
    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        split_passed[split] = 0
        for idx, example in enumerate(data.get(split, [])):
            arr = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if arr is None or expected is None:
                continue
            actual = session.run([OUT_NAME], {IN_NAME: arr})[0]
            total += 1
            if not np.array_equal(actual > 0.0, expected > 0.0):
                raise AssertionError(f"{split}[{idx}] output mismatch")
            passed += 1
            split_passed[split] += 1
    return passed, total, split_passed


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    passed, total, split_passed = validate_examples(BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"correct: {passed}/{total}")
    print(
        "splits:  "
        f"train {split_passed['train']}, "
        f"test {split_passed['test']}, "
        f"arc-gen {split_passed['arc-gen']}"
    )
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
