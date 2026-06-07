"""ONNX solution for ARC task308: centered per-color landmark rasterization.

Task rule: the input has one dominant background color and several sparse
non-background colors.  For each non-background color, take all of its pixels as
one landmark pattern, compute that color's bounding box, and translate the
pattern into a shared square output lattice.  The shared side length is one more
than the largest row/column span of any non-background pattern.  Patterns whose
box is shorter in one axis are centered on that axis; all remaining cells in the
square are filled with the background color and everything outside the square is
padding.

The graph detects the background by channel population, computes per-color
bounding boxes from small row/column reductions, and rasterizes each color into
a compact 7x7 lattice because every task output is 3x3, 5x5, or 7x7.  All known
examples are at most 20x20 and use colors 1-9 only, so the graph first crops
the padded competition input to that spatial working area and drops the unused
color-0 channel.  It keeps intermediates boolean/int32 where possible, casts only
the compact one-hot tensor to float, and pads that tensor directly to the
required 30x30 output.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task308"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
IH = IW = 20
OH = OW = 7
LANDMARK_COLORS = range(1, C)
IN_NAME = "input"
WORK_NAME = "input20"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference solver for the centered landmark rule."""
    arr = np.asarray(grid, dtype=np.int64)
    bg = Counter(int(v) for v in arr.ravel()).most_common(1)[0][0]
    colors = sorted(int(v) for v in set(arr.ravel()) if int(v) != bg)

    meta: dict[int, tuple[np.ndarray, np.ndarray, int, int, int, int]] = {}
    side_minus_one = 0
    for color in colors:
        rows, cols = np.where(arr == color)
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        row_span = r1 - r0
        col_span = c1 - c0
        side_minus_one = max(side_minus_one, row_span, col_span)
        meta[color] = (rows, cols, r0, c0, row_span, col_span)

    out = np.full((side_minus_one + 1, side_minus_one + 1), bg, dtype=np.int64)
    for color, (rows, cols, r0, c0, row_span, col_span) in meta.items():
        row_offset = (side_minus_one - row_span) // 2
        col_offset = (side_minus_one - col_span) // 2
        out[rows - r0 + row_offset, cols - c0 + col_offset] = color
    return out


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray, rows: int, cols: int) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)[:rows, :cols]


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _i32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int32), name)


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    zero_f = _f32(inits, [0.0], "zero_f")
    zero_i = _i32(inits, [0], "zero_i")
    one_i = _i32(inits, [1], "one_i")
    two_i = _i32(inits, [2], "two_i")
    neg_one_i = _i32(inits, [-1], "neg_one_i")
    input_limit_i = _i32(inits, [IH], "input_limit_i")
    row_vec = _i32(inits, np.arange(IH).reshape(1, 1, IH), "row_vec")
    col_vec = _i32(inits, np.arange(IW).reshape(1, 1, IW), "col_vec")
    idx_1d = _i32(inits, np.arange(OH), "idx_1d")
    out_rows = _i32(inits, np.arange(OH).reshape(OH, 1), "out_rows")
    out_cols = _i32(inits, np.arange(OW).reshape(1, OW), "out_cols")
    crop_starts = _i64(inits, [1, 0, 0], "crop_starts")
    crop_ends = _i64(inits, [C, IH, IW], "crop_ends")
    crop_axes = _i64(inits, [1, 2, 3], "crop_axes")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, crop_starts, crop_ends, crop_axes], [WORK_NAME]),
            helper.make_node("ReduceSum", [WORK_NAME], ["channel_counts"], axes=[2, 3], keepdims=0),
            helper.make_node("ArgMax", ["channel_counts"], ["bg_idx"], axis=1, keepdims=1),
        ]
    )

    spans: list[str] = []
    landmark_masks: list[str] = []
    landmark_by_color: dict[int, str] = {}
    is_bg_names: list[str] = [""] * C
    is_non_bg_names: list[str] = [""] * C
    channel_slices: dict[int, str] = {}

    for color in LANDMARK_COLORS:
        c_name = _i64(inits, [[color - 1]], f"color_{color}")
        is_bg = f"is_bg_{color}"
        is_non_bg = f"is_non_bg_{color}"
        nodes.extend(
            [
                helper.make_node("Equal", ["bg_idx", c_name], [is_bg]),
                helper.make_node("Not", [is_bg], [is_non_bg]),
            ]
        )
        is_bg_names[color] = is_bg
        is_non_bg_names[color] = is_non_bg

    for color in LANDMARK_COLORS:
        channel = f"ch_{color}"
        row_sum = f"row_sum_{color}"
        col_sum = f"col_sum_{color}"
        row_has = f"row_has_{color}"
        col_has = f"col_has_{color}"
        valid = is_non_bg_names[color]
        min_row_candidates = f"min_row_candidates_{color}"
        max_row_candidates = f"max_row_candidates_{color}"
        min_col_candidates = f"min_col_candidates_{color}"
        max_col_candidates = f"max_col_candidates_{color}"
        min_row = f"min_row_{color}"
        max_row = f"max_row_{color}"
        min_col = f"min_col_{color}"
        max_col = f"max_col_{color}"
        row_span = f"row_span_{color}"
        col_span = f"col_span_{color}"
        row_span_gt = f"row_span_gt_{color}"
        span_max = f"span_max_{color}"
        span_valid = f"span_valid_{color}"
        gather_channel = _i64(inits, [color - 1], f"gather_channel_{color}")

        nodes.extend(
            [
                helper.make_node("Gather", [WORK_NAME, gather_channel], [channel], axis=1),
                helper.make_node("ReduceSum", [channel], [row_sum], axes=[3], keepdims=0),
                helper.make_node("ReduceSum", [channel], [col_sum], axes=[2], keepdims=0),
                helper.make_node("Greater", [row_sum, zero_f], [row_has]),
                helper.make_node("Greater", [col_sum, zero_f], [col_has]),
                helper.make_node("Where", [row_has, row_vec, input_limit_i], [min_row_candidates]),
                helper.make_node("Where", [row_has, row_vec, neg_one_i], [max_row_candidates]),
                helper.make_node("Where", [col_has, col_vec, input_limit_i], [min_col_candidates]),
                helper.make_node("Where", [col_has, col_vec, neg_one_i], [max_col_candidates]),
                helper.make_node("ReduceMin", [min_row_candidates], [min_row], axes=[2], keepdims=0),
                helper.make_node("ReduceMax", [max_row_candidates], [max_row], axes=[2], keepdims=0),
                helper.make_node("ReduceMin", [min_col_candidates], [min_col], axes=[2], keepdims=0),
                helper.make_node("ReduceMax", [max_col_candidates], [max_col], axes=[2], keepdims=0),
                helper.make_node("Sub", [max_row, min_row], [row_span]),
                helper.make_node("Sub", [max_col, min_col], [col_span]),
                helper.make_node("Greater", [row_span, col_span], [row_span_gt]),
                helper.make_node("Where", [row_span_gt, row_span, col_span], [span_max]),
                helper.make_node("Where", [valid, span_max, zero_i], [span_valid]),
            ]
        )
        spans.append(span_valid)
        channel_slices[color] = channel

    nodes.extend(
        [
            helper.make_node("Concat", spans, ["all_spans"], axis=1),
            helper.make_node("ReduceMax", ["all_spans"], ["side_minus_one"], axes=[1], keepdims=1),
            helper.make_node("Add", ["side_minus_one", one_i], ["side_len"]),
            helper.make_node("Less", [out_rows, "side_len"], ["valid_rows"]),
            helper.make_node("Less", [out_cols, "side_len"], ["valid_cols"]),
            helper.make_node("And", ["valid_rows", "valid_cols"], ["valid_out"]),
        ]
    )

    for color in LANDMARK_COLORS:
        valid = is_non_bg_names[color]
        min_row = f"min_row_{color}"
        min_col = f"min_col_{color}"
        row_span = f"row_span_{color}"
        col_span = f"col_span_{color}"
        row_missing = f"row_missing_{color}"
        col_missing = f"col_missing_{color}"
        row_offset = f"row_offset_{color}"
        col_offset = f"col_offset_{color}"
        row_offset_s = f"row_offset_s_{color}"
        col_offset_s = f"col_offset_s_{color}"
        min_row_s = f"min_row_s_{color}"
        min_col_s = f"min_col_s_{color}"
        input_rows_shifted = f"input_rows_shifted_{color}"
        input_cols_shifted = f"input_cols_shifted_{color}"
        input_rows = f"input_rows_{color}"
        input_cols = f"input_cols_{color}"
        rows_negative = f"rows_negative_{color}"
        cols_negative = f"cols_negative_{color}"
        rows_non_negative = f"rows_non_negative_{color}"
        cols_non_negative = f"cols_non_negative_{color}"
        rows_below_30 = f"rows_below_30_{color}"
        cols_below_30 = f"cols_below_30_{color}"
        rows_valid = f"rows_valid_idx_{color}"
        cols_valid = f"cols_valid_idx_{color}"
        safe_rows = f"safe_rows_{color}"
        safe_cols = f"safe_cols_{color}"
        row_projected = f"row_gathered_{color}"
        projected = f"projected_{color}"
        projected_bool = f"projected_bool_{color}"
        landmark = f"landmark_{color}"

        nodes.extend(
            [
                helper.make_node("Sub", ["side_minus_one", row_span], [row_missing]),
                helper.make_node("Sub", ["side_minus_one", col_span], [col_missing]),
                helper.make_node("Div", [row_missing, two_i], [row_offset]),
                helper.make_node("Div", [col_missing, two_i], [col_offset]),
                helper.make_node("Squeeze", [row_offset], [row_offset_s], axes=[0, 1]),
                helper.make_node("Squeeze", [col_offset], [col_offset_s], axes=[0, 1]),
                helper.make_node("Squeeze", [min_row], [min_row_s], axes=[0, 1]),
                helper.make_node("Squeeze", [min_col], [min_col_s], axes=[0, 1]),
                helper.make_node("Sub", [idx_1d, row_offset_s], [input_rows_shifted]),
                helper.make_node("Sub", [idx_1d, col_offset_s], [input_cols_shifted]),
                helper.make_node("Add", [input_rows_shifted, min_row_s], [input_rows]),
                helper.make_node("Add", [input_cols_shifted, min_col_s], [input_cols]),
                helper.make_node("Less", [input_rows, zero_i], [rows_negative]),
                helper.make_node("Less", [input_cols, zero_i], [cols_negative]),
                helper.make_node("Not", [rows_negative], [rows_non_negative]),
                helper.make_node("Not", [cols_negative], [cols_non_negative]),
                helper.make_node("Less", [input_rows, input_limit_i], [rows_below_30]),
                helper.make_node("Less", [input_cols, input_limit_i], [cols_below_30]),
                helper.make_node("And", [rows_non_negative, rows_below_30], [rows_valid]),
                helper.make_node("And", [cols_non_negative, cols_below_30], [cols_valid]),
                helper.make_node("Where", [rows_valid, input_rows, zero_i], [safe_rows]),
                helper.make_node("Where", [cols_valid, input_cols, zero_i], [safe_cols]),
                helper.make_node("Gather", [channel_slices[color], safe_rows], [row_projected], axis=2),
                helper.make_node("Gather", [row_projected, safe_cols], [projected], axis=3),
                helper.make_node("Greater", [projected, zero_f], [projected_bool]),
                helper.make_node("And", [projected_bool, valid], [landmark]),
            ]
        )
        landmark_masks.append(landmark)
        landmark_by_color[color] = landmark

    any_landmark = landmark_masks[0]
    for idx, landmark in enumerate(landmark_masks[1:], start=2):
        out_name = f"any_landmark_{idx}"
        nodes.append(helper.make_node("Or", [any_landmark, landmark], [out_name]))
        any_landmark = out_name

    nodes.extend(
        [
            helper.make_node("Unsqueeze", ["valid_out"], ["valid_out4"], axes=[0, 1]),
            helper.make_node("Not", [any_landmark], ["no_landmark"]),
            helper.make_node("And", ["valid_out4", "no_landmark"], ["bg_cells"]),
        ]
    )

    channel_masks: list[str] = []
    for color in LANDMARK_COLORS:
        bg_for_channel = f"bg_for_channel_{color}"
        channel_mask = f"out_mask_{color}"
        nodes.append(helper.make_node("And", ["bg_cells", is_bg_names[color]], [bg_for_channel]))
        if color in landmark_by_color:
            nodes.append(helper.make_node("Or", [bg_for_channel, landmark_by_color[color]], [channel_mask]))
        else:
            channel_mask = bg_for_channel
        channel_masks.append(channel_mask)

    nodes.extend(
        [
            helper.make_node("Concat", channel_masks, ["out_nchw_bool"], axis=1),
            helper.make_node("Cast", ["out_nchw_bool"], ["out7_float"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out7_float"],
                [OUT_NAME],
                pads=[0, 1, 0, 0, 0, 0, H - OH, W - OW],
                value=0.0,
            ),
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
    onnx.checker.check_model(model, full_check=True)
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
            expected = np.asarray(ex["output"], dtype=np.int64)
            ref = solve(ex["input"])
            pred_oh = _run_onnx(model, _grid_to_onehot(ex["input"]))
            pred = _onehot_to_grid(pred_oh, *expected.shape)
            if not np.array_equal(ref, expected) or not np.array_equal(pred, expected):
                bad += 1
                print(f"bad {split} {idx}: ref={np.array_equal(ref, expected)} pred={np.array_equal(pred, expected)}")
                if bad >= 5:
                    return bad
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
