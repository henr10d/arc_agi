"""ONNX generator for ARC task396: crop and recolor the selected frame.

Task rule: find the largest hollow rectangular frame in the input, crop to its
bounding box, recolor the completed frame border with the scattered marker
color, and keep only marker-color pixels that fall inside that frame. Every
other cell in the cropped output is black. The selected frame is the strongest
large rectangle by perimeter evidence; smaller rectangles and marker pixels
outside the selected frame are ignored.

The ONNX graph uses the task-specific fact that every example has exactly two
non-background colors: the more frequent one is the frame color and the other
is the marker color. It scores every possible 4x4 through 8x8 frame location
with compact signed Conv kernels, gathers the marker-color pixels into a
dynamic 8x8 crop, draws the selected border, and pads to the NeuroGolf
[1, 10, 30, 30] one-hot output.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task396"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task396.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
FG = 9
H = W = 30
SEARCH_H = SEARCH_W = 18
MAX_OUT = 8
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


def _i32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int32), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _bool(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.bool_), name)


def _slice(
    nodes: List[onnx.NodeProto],
    data: str,
    out: str,
    starts: str,
    ends: str,
    axes: str,
) -> str:
    nodes.append(helper.make_node("Slice", [data, starts, ends, axes], [out]))
    return out


def _padded_grid(grid: list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros((H, W), dtype=np.int64)
    out[: arr.shape[0], : arr.shape[1]] = arr
    return out


def _components(arr: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    seen = np.zeros(arr.shape, dtype=np.bool_)
    comps: list[list[tuple[int, int]]] = []
    rows, cols = arr.shape
    for r in range(rows):
        for c in range(cols):
            if seen[r, c] or arr[r, c] != color:
                continue
            queue = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            for rr, cc in queue:
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and not seen[nr, nc]
                        and arr[nr, nc] == color
                    ):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(cells)
    return comps


def detect_frame(grid: list[list[int]]) -> tuple[int, tuple[int, int, int, int]]:
    """Reference frame detector used to derive compact ONNX candidates."""
    arr = _padded_grid(grid)
    best: tuple[tuple[int, int, int, int], int, tuple[int, int, int, int]] | None = None
    for color in sorted(set(arr.ravel()) - {0}):
        for cells in _components(arr, int(color)):
            rs = [cell[0] for cell in cells]
            cs = [cell[1] for cell in cells]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            h, w = r1 - r0 + 1, c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            perimeter = (
                {(r0, c) for c in range(c0, c1 + 1)}
                | {(r1, c) for c in range(c0, c1 + 1)}
                | {(r, c0) for r in range(r0 + 1, r1)}
                | {(r, c1) for r in range(r0 + 1, r1)}
            )
            coverage = sum(arr[r, c] == color for r, c in perimeter)
            interior = sum(
                arr[r, c] == color
                for r in range(r0 + 1, r1)
                for c in range(c0 + 1, c1)
            )
            if coverage < max(4, int(0.7 * len(perimeter))):
                continue
            if interior > max(1, ((h - 2) * (w - 2)) // 6):
                continue
            score = (coverage - 3 * interior, h * w, len(cells), coverage)
            if best is None or score > best[0]:
                best = (score, int(color), (r0, c0, h, w))
    if best is None:
        raise ValueError("no frame candidate found")
    return best[1], best[2]


def solve(grid: list[list[int]]) -> np.ndarray:
    """Reference solver for validation."""
    arr = _padded_grid(grid)
    frame_color, (r0, c0, h, w) = detect_frame(grid)
    counts = Counter(int(v) for v in arr.ravel())
    counts.pop(0, None)
    counts.pop(frame_color, None)
    marker_color = counts.most_common(1)[0][0]

    out = np.zeros((h, w), dtype=np.int64)
    out[0, :] = marker_color
    out[-1, :] = marker_color
    out[:, 0] = marker_color
    out[:, -1] = marker_color
    crop = arr[r0 : r0 + h, c0 : c0 + w]
    out[crop == marker_color] = marker_color
    return out


def make_frame_score_tensors() -> tuple[
    list[tuple[int, int, int, int, str, str, str]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return Conv metadata for all possible 4x4..8x8 frame candidates."""
    groups: list[tuple[int, int, int, int, str, str, str]] = []
    heights: list[int] = []
    widths: list[int] = []
    out_widths: list[int] = []

    for h in range(4, MAX_OUT + 1):
        for w in range(4, MAX_OUT + 1):
            out_h = SEARCH_H - h + 1
            out_w = SEARCH_W - w + 1
            groups.append((h, w, out_h, out_w, f"score_w_{h}_{w}", f"score_b_{h}_{w}", f"score_shape_{h}_{w}"))
            heights.append(h)
            widths.append(w)
            out_widths.append(out_w)

    return (
        groups,
        np.asarray(heights, dtype=np.int64),
        np.asarray(widths, dtype=np.int64),
        np.asarray(out_widths, dtype=np.int64),
    )


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    groups, heights, widths, out_widths = make_frame_score_tensors()

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    channel_ids = _i64(inits, np.arange(C, dtype=np.int64).reshape(1, C), "channel_ids")
    color_ids = _i64(inits, np.arange(C, dtype=np.int64).reshape(1, C, 1), "color_ids")
    zero_color = _i64(inits, [0], "zero_color")
    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    frame_st = _i64(inits, [0, 0, 0, 0], "frame_st")
    frame_en = _i64(inits, [1, 1, SEARCH_H, SEARCH_W], "frame_en")
    flat_marker = _i64(inits, [H * W], "flat_marker")
    crop64 = _i64(inits, [1, 1, MAX_OUT * MAX_OUT], "crop64_shape")
    out64 = _i64(inits, [1, C, MAX_OUT, MAX_OUT], "out64_shape")
    zero = _f32(inits, [0.0], "zero")
    neg = _f32(inits, [-1000000.0], "neg")
    one_i = _i64(inits, [1], "one_i")
    thirty_i = _i64(inits, [W], "thirty_i")
    row_offsets = _i64(
        inits,
        np.repeat(np.arange(MAX_OUT, dtype=np.int64), MAX_OUT),
        "row_offsets",
    )
    col_offsets = _i64(
        inits,
        np.tile(np.arange(MAX_OUT, dtype=np.int64), MAX_OUT),
        "col_offsets",
    )
    _i64(inits, heights, "group_heights")
    _i64(inits, widths, "group_widths")
    _i64(inits, out_widths, "group_out_widths")

    nodes.append(helper.make_node("ReduceSum", [IN_NAME], ["color_counts"], axes=[2, 3], keepdims=0))
    nodes.append(helper.make_node("Equal", ["channel_ids", "zero_color"], ["is_bg_color"]))
    nodes.append(helper.make_node("Where", ["is_bg_color", "neg", "color_counts"], ["nonbg_scores"]))
    nodes.append(helper.make_node("ArgMax", ["nonbg_scores"], ["frame_color"], axis=1, keepdims=0))
    nodes.append(helper.make_node("Equal", ["channel_ids", "frame_color"], ["is_frame_color"]))
    nodes.append(helper.make_node("Or", ["is_bg_color", "is_frame_color"], ["blocked_colors"]))
    nodes.append(helper.make_node("Where", ["blocked_colors", "neg", "color_counts"], ["marker_scores"]))
    nodes.append(helper.make_node("ArgMax", ["marker_scores"], ["marker_color"], axis=1, keepdims=0))

    nodes.append(helper.make_node("Gather", [IN_NAME, "frame_color"], ["frame_ch"], axis=1))
    _slice(nodes, "frame_ch", "frame_roi", frame_st, frame_en, axes4)

    group_scores: list[str] = []
    local_choices: list[str] = []
    for h, w, out_h, out_w, weight_name, bias_name, shape_name in groups:
        kernel = np.full((1, 1, h, w), -3000.0, dtype=np.float32)
        kernel[:, :, 0, :] = 1000.0
        kernel[:, :, h - 1, :] = 1000.0
        kernel[:, :, :, 0] = 1000.0
        kernel[:, :, :, w - 1] = 1000.0
        _f32(inits, kernel, weight_name)
        _f32(inits, [float(h * w)], bias_name)
        _i64(inits, [out_h], shape_name)
        local_out_w = _i64(inits, [out_w], f"local_out_w_{h}_{w}")
        score_name = f"score_{h}_{w}"
        max_name = f"group_score_{h}_{w}"
        row_scores_name = f"row_scores_{h}_{w}"
        row_idx_name = f"row_idx_{h}_{w}"
        row_idx_flat_name = f"row_idx_flat_{h}_{w}"
        col_by_row_name = f"col_by_row_{h}_{w}"
        col_by_row_flat_name = f"col_by_row_flat_{h}_{w}"
        col_choice_name = f"col_choice_{h}_{w}"
        row_start_name = f"row_start_{h}_{w}"
        choice_name = f"local_choice_{h}_{w}"
        nodes.append(helper.make_node("Conv", ["frame_roi", weight_name, bias_name], [score_name]))
        nodes.append(helper.make_node("ReduceMax", [score_name], [max_name], axes=[2, 3], keepdims=0))
        nodes.append(helper.make_node("ReduceMax", [score_name], [row_scores_name], axes=[3], keepdims=0))
        nodes.append(helper.make_node("ArgMax", [row_scores_name], [row_idx_name], axis=2, keepdims=0))
        nodes.append(helper.make_node("Reshape", [row_idx_name, "one_i"], [row_idx_flat_name]))
        nodes.append(helper.make_node("ArgMax", [score_name], [col_by_row_name], axis=3, keepdims=0))
        nodes.append(helper.make_node("Reshape", [col_by_row_name, shape_name], [col_by_row_flat_name]))
        nodes.append(helper.make_node("Gather", [col_by_row_flat_name, row_idx_flat_name], [col_choice_name], axis=0))
        nodes.append(helper.make_node("Mul", [row_idx_flat_name, local_out_w], [row_start_name]))
        nodes.append(helper.make_node("Add", [row_start_name, col_choice_name], [choice_name]))
        group_scores.append(max_name)
        local_choices.append(choice_name)

    nodes.append(helper.make_node("Concat", group_scores, ["group_scores"], axis=1))
    nodes.append(helper.make_node("ArgMax", ["group_scores"], ["group_idx"], axis=1, keepdims=0))
    nodes.append(helper.make_node("Concat", local_choices, ["local_choices"], axis=0))
    nodes.append(helper.make_node("Gather", ["local_choices", "group_idx"], ["local_choice"], axis=0))
    nodes.append(helper.make_node("Gather", ["group_heights", "group_idx"], ["box_h"], axis=0))
    nodes.append(helper.make_node("Gather", ["group_widths", "group_idx"], ["box_w"], axis=0))
    nodes.append(helper.make_node("Gather", ["group_out_widths", "group_idx"], ["group_out_w"], axis=0))
    nodes.append(helper.make_node("Div", ["local_choice", "group_out_w"], ["box_r0"]))
    nodes.append(helper.make_node("Mod", ["local_choice", "group_out_w"], ["box_c0"]))

    nodes.append(helper.make_node("Gather", [IN_NAME, "marker_color"], ["marker_ch"], axis=1))
    nodes.append(helper.make_node("Reshape", ["marker_ch", flat_marker], ["marker_flat"]))
    nodes.append(helper.make_node("Add", ["box_r0", "row_offsets"], ["crop_rows"]))
    nodes.append(helper.make_node("Add", ["box_c0", "col_offsets"], ["crop_cols"]))
    nodes.append(helper.make_node("Mul", ["crop_rows", "thirty_i"], ["crop_row_flats"]))
    nodes.append(helper.make_node("Add", ["crop_row_flats", "crop_cols"], ["crop_indices"]))
    nodes.append(helper.make_node("Gather", ["marker_flat", "crop_indices"], ["marker_crop_f"], axis=0))
    nodes.append(helper.make_node("Greater", ["marker_crop_f", "zero"], ["marker_crop"]))

    nodes.append(helper.make_node("Less", ["row_offsets", "box_h"], ["row_active"]))
    nodes.append(helper.make_node("Less", ["col_offsets", "box_w"], ["col_active"]))
    nodes.append(helper.make_node("And", ["row_active", "col_active"], ["active_crop"]))
    nodes.append(helper.make_node("Sub", ["box_h", "one_i"], ["last_row"]))
    nodes.append(helper.make_node("Sub", ["box_w", "one_i"], ["last_col"]))
    nodes.append(helper.make_node("Equal", ["row_offsets", "zero_color"], ["is_top"]))
    nodes.append(helper.make_node("Equal", ["row_offsets", "last_row"], ["is_bottom"]))
    nodes.append(helper.make_node("Equal", ["col_offsets", "zero_color"], ["is_left"]))
    nodes.append(helper.make_node("Equal", ["col_offsets", "last_col"], ["is_right"]))
    nodes.append(helper.make_node("Or", ["is_top", "is_bottom"], ["row_border"]))
    nodes.append(helper.make_node("Or", ["is_left", "is_right"], ["col_border"]))
    nodes.append(helper.make_node("Or", ["row_border", "col_border"], ["border_any"]))
    nodes.append(helper.make_node("And", ["active_crop", "border_any"], ["border_crop"]))
    nodes.append(helper.make_node("And", ["marker_crop", "active_crop"], ["marker_inside"]))
    nodes.append(helper.make_node("Or", ["marker_inside", "border_crop"], ["fg_crop"]))
    nodes.append(helper.make_node("Not", ["fg_crop"], ["not_fg_crop"]))
    nodes.append(helper.make_node("And", ["active_crop", "not_fg_crop"], ["bg_crop"]))

    nodes.append(helper.make_node("Reshape", ["fg_crop", crop64], ["fg_crop64"]))
    nodes.append(helper.make_node("Reshape", ["bg_crop", crop64], ["bg_crop64"]))
    nodes.append(helper.make_node("Equal", ["color_ids", "marker_color"], ["paint_color"]))
    nodes.append(helper.make_node("Equal", ["color_ids", "zero_color"], ["paint_bg_color"]))
    nodes.append(helper.make_node("And", ["paint_color", "fg_crop64"], ["paint_fg"]))
    nodes.append(helper.make_node("And", ["paint_bg_color", "bg_crop64"], ["paint_bg"]))
    nodes.append(helper.make_node("Or", ["paint_fg", "paint_bg"], ["out8b"]))
    nodes.append(helper.make_node("Reshape", ["out8b", out64], ["out8b4"]))
    nodes.append(helper.make_node("Cast", ["out8b4"], ["out8"], to=TensorProto.FLOAT))
    nodes.append(
        helper.make_node(
            "Pad",
            ["out8"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - MAX_OUT, W - MAX_OUT],
        )
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


def validate_reference() -> tuple[int, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            got = solve(example["input"])
            expected = np.asarray(example["output"], dtype=np.int64)
            if got.shape != expected.shape or not np.array_equal(got, expected):
                raise AssertionError(f"reference mismatch on {split}[{idx}]")
            total += 1
    return total, total


def validate_model(path: Path) -> tuple[int, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            arr = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if arr is None or expected is None:
                continue
            got = session.run([OUT_NAME], {IN_NAME: arr})[0]
            total += 1
            if not np.array_equal(got > 0.0, expected > 0.0):
                raise AssertionError(f"model mismatch on {split}[{idx}]")
            passed += 1
    return passed, total


def main() -> None:
    ref_passed, ref_total = validate_reference()
    groups, *_ = make_frame_score_tensors()
    model = build_model()
    onnx.save(model, BEST_PATH)
    passed, total = validate_model(BEST_PATH)
    result = score_file(BEST_PATH)

    print(f"wrote {BEST_PATH}")
    print(f"reference:  {ref_passed}/{ref_total}")
    print(f"candidate groups: {len(groups)}")
    print(f"correct:    {passed}/{total}")
    print(f"valid:      {result['valid']}")
    if result["error"]:
        print(f"error:      {result['error']}")
    print(f"filesize:   {result['filesize']}")
    print(f"memory:     {result['memory']}")
    print(f"params:     {result['params']}")
    print(f"cost:       {result['cost']}")
    if result["score"] is not None:
        print(f"score:      {result['score']:.6f}")


if __name__ == "__main__":
    main()
