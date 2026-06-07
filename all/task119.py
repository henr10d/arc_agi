"""Compact ONNX for NeuroGolf task119.

Task rule: the 12x12 visible grid contains an edge red barrier and a short cyan
diagonal seed.  The output preserves the input, then adds green cells along the
diagonal path implied by the seed after it reflects off the red barrier.  The
green path never enters red cells; cyan and red cells remain unchanged.

ONNX approach: identify which diagonal family contains the cyan seed, compute
the reflected diagonal directly from the red edge side/thickness, materialize
only compact 12x12 bool masks, build a proper one-hot 12x12 output, and pad it
to the required 30x30 tensor.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402


TASK_ID = "task119"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, 10, 30, 30]
VISIBLE = 12
CELLS = VISIBLE * VISIBLE
IR_VERSION = 10
OPSET = 10

Grid = list[list[int]]
CellSet = frozenset[tuple[int, int]]


def _f32(name: str, values: np.ndarray | list[float]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.float32), name)


def _i64(name: str, values: list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name)


def cells(grid: Grid, color: int) -> CellSet:
    return frozenset((r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == color)


def load_task_data() -> dict[str, list[dict[str, Grid]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def edge_side(red: CellSet) -> str:
    rows = {r for r, _ in red}
    cols = {c for _, c in red}
    if rows == set(range(VISIBLE)):
        return "left" if min(cols) == 0 else "right"
    if cols == set(range(VISIBLE)):
        return "top" if min(rows) == 0 else "bottom"
    raise ValueError(f"red cells are not a full edge barrier: {sorted(red)}")


def direction_from(points: CellSet) -> tuple[int, int]:
    ordered = sorted(points)
    (r0, c0), (r1, c1) = ordered[0], ordered[-1]
    dr, dc = r1 - r0, c1 - c0
    step = math.gcd(abs(dr), abs(dc))
    return dr // step, dc // step


def line_stop_no_reflect(grid: Grid) -> Grid:
    """Extend the first/last cyan line toward the red edge, stopping at red."""
    out = [row[:] for row in grid]
    red = cells(grid, 2)
    cyan = cells(grid, 8)
    side = edge_side(red)
    d = direction_from(cyan)
    endpoints = [(d, max(cyan)), ((-d[0], -d[1]), min(cyan))]

    def toward(step: tuple[int, int]) -> bool:
        dr, dc = step
        return (
            (side == "left" and dc < 0)
            or (side == "right" and dc > 0)
            or (side == "top" and dr < 0)
            or (side == "bottom" and dr > 0)
        )

    step, (r, c) = next(((s, p) for s, p in endpoints if toward(s)), endpoints[0])
    while True:
        r += step[0]
        c += step[1]
        if not (0 <= r < VISIBLE and 0 <= c < VISIBLE) or (r, c) in red:
            return out
        if out[r][c] == 0:
            out[r][c] = 3


def ordered_polyline_no_reflect(grid: Grid) -> Grid:
    """Continue the row-major last cyan segment without reflecting."""
    out = [row[:] for row in grid]
    red = cells(grid, 2)
    ordered = sorted(cells(grid, 8))
    (r0, c0), (r1, c1) = ordered[-2], ordered[-1]
    dr, dc = r1 - r0, c1 - c0
    step = math.gcd(abs(dr), abs(dc))
    dr, dc = dr // step, dc // step
    r, c = r1, c1
    while True:
        r += dr
        c += dc
        if not (0 <= r < VISIBLE and 0 <= c < VISIBLE) or (r, c) in red:
            return out
        if out[r][c] == 0:
            out[r][c] = 3


def reflected_line(grid: Grid) -> Grid:
    """Geometric reflected-line hypothesis used for the training examples."""
    out = [row[:] for row in grid]
    red = cells(grid, 2)
    cyan = cells(grid, 8)
    side = edge_side(red)
    d = direction_from(cyan)
    endpoints = [(d, max(cyan)), ((-d[0], -d[1]), min(cyan))]

    def toward(step: tuple[int, int]) -> bool:
        dr, dc = step
        return (
            (side == "left" and dc < 0)
            or (side == "right" and dc > 0)
            or (side == "top" and dr < 0)
            or (side == "bottom" and dr > 0)
        )

    (dr, dc), (r, c) = next(((s, p) for s, p in endpoints if toward(s)), endpoints[0])
    reflected = False
    for _ in range(CELLS * 2):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < VISIBLE and 0 <= nc < VISIBLE):
            return out
        if (nr, nc) in red:
            if reflected:
                return out
            reflected = True
            if side in {"left", "right"}:
                dc = -dc
            else:
                dr = -dr
            continue
        r, c = nr, nc
        if out[r][c] == 0:
            out[r][c] = 3
    return out


def nearest_boundary_extension(grid: Grid) -> Grid:
    """Try the cyan endpoint/direction that reaches red in the fewest steps."""
    red = cells(grid, 2)
    cyan = cells(grid, 8)
    d = direction_from(cyan)
    candidates = [(d, max(cyan)), ((-d[0], -d[1]), min(cyan))]
    best: tuple[int, tuple[int, int], tuple[int, int]] | None = None
    for step, start in candidates:
        r, c = start
        for dist in range(1, CELLS):
            r += step[0]
            c += step[1]
            if not (0 <= r < VISIBLE and 0 <= c < VISIBLE):
                break
            if (r, c) in red:
                best = min(best, (dist, step, start)) if best else (dist, step, start)
                break
    if best is None:
        return [row[:] for row in grid]
    out = [row[:] for row in grid]
    _, (dr, dc), (r, c) = best
    while True:
        r += dr
        c += dc
        if not (0 <= r < VISIBLE and 0 <= c < VISIBLE) or (r, c) in red:
            return out
        if out[r][c] == 0:
            out[r][c] = 3


def collect_candidates(data: dict[str, list[dict[str, Grid]]]) -> list[tuple[CellSet, CellSet]]:
    candidates = {
        (cells(example["input"], 2), cells(example["output"], 8) | cells(example["output"], 3))
        for examples in data.values()
        for example in examples
    }
    return sorted(candidates, key=lambda item: (sorted(item[0]), sorted(item[1])))


def red_id(red: CellSet) -> float:
    rows = {r for r, _ in red}
    cols = {c for _, c in red}
    if rows == set(range(VISIBLE)):
        side = "left" if min(cols) == 0 else "right"
        thickness = len(cols)
    elif cols == set(range(VISIBLE)):
        side = "top" if min(rows) == 0 else "bottom"
        thickness = len(rows)
    else:
        raise ValueError(f"red cells are not a full edge barrier: {sorted(red)}")
    base = {"top": 0, "bottom": 5, "left": 10, "right": 15}[side]
    return float(base + thickness - 1)


def candidate_selector(grid: Grid, candidates: list[tuple[CellSet, CellSet]]) -> Grid:
    """Reference for the ONNX selector: exact red mask and cyan subset of path."""
    red = cells(grid, 2)
    cyan = cells(grid, 8)
    selected = [path for red_mask, path in candidates if red_mask == red and cyan <= path]
    if len(selected) != 1:
        raise ValueError(f"expected one candidate, found {len(selected)}")
    path = selected[0]
    out = [row[:] for row in grid]
    for r, c in path - red - cyan:
        if out[r][c] == 0:
            out[r][c] = 3
    return out


def hypothesis_table(data: dict[str, list[dict[str, Grid]]], candidates: list[tuple[CellSet, CellSet]]) -> list[tuple[str, int, int]]:
    hypotheses: list[tuple[str, Callable[[Grid], Grid]]] = [
        ("single_line_stop", line_stop_no_reflect),
        ("line_through_first_last", line_stop_no_reflect),
        ("ordered_polyline", ordered_polyline_no_reflect),
        ("reflection_completion", reflected_line),
        ("nearest_boundary", nearest_boundary_extension),
        ("candidate_reflection_selector", lambda grid: candidate_selector(grid, candidates)),
    ]
    rows: list[tuple[str, int, int]] = []
    examples = [example for split in ("train", "test", "arc-gen") for example in data[split]]
    for name, solve in hypotheses:
        passed = 0
        for example in examples:
            try:
                passed += solve(example["input"]) == example["output"]
            except Exception:
                pass
        rows.append((name, passed, len(examples)))
    return rows


def build_model(candidates: list[tuple[CellSet, CellSet]]) -> onnx.ModelProto:
    del candidates

    sum_diag_indices: list[list[int]] = [[] for _ in range(2 * VISIBLE - 1)]
    diff_diag_indices: list[list[int]] = [[] for _ in range(2 * VISIBLE - 1)]
    cell_sum_indices = np.zeros((CELLS,), dtype=np.int64)
    cell_diff_indices = np.zeros((CELLS,), dtype=np.int64)
    for r in range(VISIBLE):
        for c in range(VISIBLE):
            idx = r * VISIBLE + c
            sum_value = r + c
            diff_value = r - c + VISIBLE - 1
            sum_diag_indices[sum_value].append(idx)
            diff_diag_indices[diff_value].append(idx)
            cell_sum_indices[idx] = sum_value
            cell_diff_indices[idx] = diff_value

    initializers = [
        _i64("axes", [0, 1, 2, 3]),
        _i64("red_st", [0, 2, 0, 0]),
        _i64("red_en", [1, 3, VISIBLE, VISIBLE]),
        _i64("cyan_st", [0, 8, 0, 0]),
        _i64("cyan_en", [1, 9, VISIBLE, VISIBLE]),
        _i64("edge_st0", [0, 0, 0, 0]),
        _i64("top_en", [1, 1, 1, VISIBLE]),
        _i64("bottom_st", [0, 0, VISIBLE - 1, 0]),
        _i64("full_en", [1, 1, VISIBLE, VISIBLE]),
        _i64("left_en", [1, 1, VISIBLE, 1]),
        _i64("right_st", [0, 0, 0, VISIBLE - 1]),
        _i64("cell_sum_idx", cell_sum_indices.tolist()),
        _i64("cell_diff_idx", cell_diff_indices.tolist()),
        _i64("flat_shape", [1, CELLS]),
        _i64("mask_shape", [1, 1, VISIBLE, VISIBLE]),
        _i64("edge_shape", [1, 4]),
        _f32("half", [0.5]),
        _f32("twelve", [12.0]),
        _f32("two", [2.0]),
        _f32("twenty_two", [22.0]),
        _f32("edge_threshold", [11.5]),
        _f32("sum_values", np.arange(2 * VISIBLE - 1, dtype=np.float32)),
        _f32("diff_values", np.arange(-(VISIBLE - 1), VISIBLE, dtype=np.float32)),
        _f32("zero_plane", np.zeros((1, 1, VISIBLE, VISIBLE), dtype=np.float32)),
    ]
    initializers.extend(_i64(f"sum_diag_idx_{idx:02d}", values) for idx, values in enumerate(sum_diag_indices))
    initializers.extend(_i64(f"diff_diag_idx_{idx:02d}", values) for idx, values in enumerate(diff_diag_indices))

    def diagonal_count_nodes(flat_name: str, index_prefix: str, output_prefix: str, output_name: str) -> list[onnx.NodeProto]:
        count_nodes: list[onnx.NodeProto] = []
        count_names: list[str] = []
        for idx in range(2 * VISIBLE - 1):
            gathered = f"{output_prefix}_gather_{idx:02d}"
            counted = f"{output_prefix}_count_{idx:02d}"
            count_nodes.append(helper.make_node("Gather", [flat_name, f"{index_prefix}_diag_idx_{idx:02d}"], [gathered], axis=1))
            count_nodes.append(helper.make_node("ReduceSum", [gathered], [counted], axes=[1], keepdims=1))
            count_names.append(counted)
        count_nodes.append(helper.make_node("Concat", count_names, [output_name], axis=1))
        return count_nodes

    nodes = [
        helper.make_node("Slice", [IN_NAME, "red_st", "red_en", "axes"], ["red"]),
        helper.make_node("Slice", [IN_NAME, "cyan_st", "cyan_en", "axes"], ["cyan"]),
        helper.make_node("Reshape", ["cyan", "flat_shape"], ["cyan_flat"]),
        helper.make_node("ReduceSum", ["red"], ["red_sum"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Slice", ["red", "edge_st0", "top_en", "axes"], ["top_edge"]),
        helper.make_node("Slice", ["red", "bottom_st", "full_en", "axes"], ["bottom_edge"]),
        helper.make_node("Slice", ["red", "edge_st0", "left_en", "axes"], ["left_edge"]),
        helper.make_node("Slice", ["red", "right_st", "full_en", "axes"], ["right_edge"]),
        helper.make_node("ReduceSum", ["top_edge"], ["top_count"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("ReduceSum", ["bottom_edge"], ["bottom_count"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("ReduceSum", ["left_edge"], ["left_count"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("ReduceSum", ["right_edge"], ["right_count"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Concat", ["top_count", "bottom_count", "left_count", "right_count"], ["edge_counts_1d"], axis=0),
        helper.make_node("Reshape", ["edge_counts_1d", "edge_shape"], ["edge_counts"]),
        helper.make_node("Greater", ["edge_counts", "edge_threshold"], ["side_b"]),
        helper.make_node("Cast", ["side_b"], ["side_f"], to=TensorProto.FLOAT),
        helper.make_node("Div", ["red_sum", "twelve"], ["thickness"]),
        *diagonal_count_nodes("cyan_flat", "sum", "sum", "sum_counts"),
        *diagonal_count_nodes("cyan_flat", "diff", "diff", "diff_counts"),
        helper.make_node("ReduceMax", ["sum_counts"], ["sum_max"], axes=[1], keepdims=1),
        helper.make_node("ReduceMax", ["diff_counts"], ["diff_max"], axes=[1], keepdims=1),
        helper.make_node("Greater", ["sum_max", "diff_max"], ["sum_chosen"]),
        helper.make_node("Not", ["sum_chosen"], ["diff_chosen"]),
        helper.make_node("ArgMax", ["sum_counts"], ["sum_idx"], axis=1, keepdims=1),
        helper.make_node("ArgMax", ["diff_counts"], ["diff_idx"], axis=1, keepdims=1),
        helper.make_node("Gather", ["sum_values", "sum_idx"], ["sum_value_raw"], axis=0),
        helper.make_node("Gather", ["diff_values", "diff_idx"], ["diff_value_raw"], axis=0),
        helper.make_node("Mul", ["thickness", "two"], ["two_t"]),
        helper.make_node("Sub", ["twenty_two", "two_t"], ["edge2_right"]),
        helper.make_node("Sub", ["sum_value_raw", "edge2_right"], ["sum_ref_right"]),
        helper.make_node("Sub", ["sum_value_raw", "two_t"], ["sum_ref_left"]),
        helper.make_node("Sub", ["edge2_right", "sum_value_raw"], ["sum_ref_bottom"]),
        helper.make_node("Sub", ["two_t", "sum_value_raw"], ["sum_ref_top"]),
        helper.make_node(
            "Concat",
            ["sum_ref_top", "sum_ref_bottom", "sum_ref_left", "sum_ref_right"],
            ["sum_ref_candidates"],
            axis=1,
        ),
        helper.make_node("Mul", ["sum_ref_candidates", "side_f"], ["sum_ref_weighted"]),
        helper.make_node("ReduceSum", ["sum_ref_weighted"], ["sum_ref_raw"], axes=[1], keepdims=1),
        helper.make_node("Sub", ["twenty_two", "two_t"], ["edge2_diff"]),
        helper.make_node("Add", ["diff_value_raw", "edge2_diff"], ["diff_ref_right"]),
        helper.make_node("Add", ["diff_value_raw", "two_t"], ["diff_ref_left"]),
        helper.make_node("Sub", ["edge2_diff", "diff_value_raw"], ["diff_ref_bottom"]),
        helper.make_node("Sub", ["two_t", "diff_value_raw"], ["diff_ref_top"]),
        helper.make_node(
            "Concat",
            ["diff_ref_top", "diff_ref_bottom", "diff_ref_left", "diff_ref_right"],
            ["diff_ref_candidates"],
            axis=1,
        ),
        helper.make_node("Mul", ["diff_ref_candidates", "side_f"], ["diff_ref_weighted"]),
        helper.make_node("ReduceSum", ["diff_ref_weighted"], ["diff_ref_raw"], axes=[1], keepdims=1),
        helper.make_node("Sub", ["sum_counts", "sum_max"], ["sum_primary_delta"]),
        helper.make_node("Abs", ["sum_primary_delta"], ["sum_primary_abs"]),
        helper.make_node("Less", ["sum_primary_abs", "half"], ["sum_primary"]),
        helper.make_node("Sub", ["diff_values", "sum_ref_raw"], ["sum_reflected_delta"]),
        helper.make_node("Abs", ["sum_reflected_delta"], ["sum_reflected_abs"]),
        helper.make_node("Less", ["sum_reflected_abs", "half"], ["sum_reflected"]),
        helper.make_node("And", ["sum_primary", "sum_chosen"], ["sum_primary_selected"]),
        helper.make_node("And", ["sum_reflected", "sum_chosen"], ["sum_reflected_selected"]),
        helper.make_node("Sub", ["diff_counts", "diff_max"], ["diff_primary_delta"]),
        helper.make_node("Abs", ["diff_primary_delta"], ["diff_primary_abs"]),
        helper.make_node("Less", ["diff_primary_abs", "half"], ["diff_primary"]),
        helper.make_node("Sub", ["sum_values", "diff_ref_raw"], ["diff_reflected_delta"]),
        helper.make_node("Abs", ["diff_reflected_delta"], ["diff_reflected_abs"]),
        helper.make_node("Less", ["diff_reflected_abs", "half"], ["diff_reflected"]),
        helper.make_node("And", ["diff_primary", "diff_chosen"], ["diff_primary_selected"]),
        helper.make_node("And", ["diff_reflected", "diff_chosen"], ["diff_reflected_selected"]),
        helper.make_node("Or", ["sum_primary_selected", "diff_reflected_selected"], ["sum_diag_b"]),
        helper.make_node("Or", ["diff_primary_selected", "sum_reflected_selected"], ["diff_diag_b"]),
        helper.make_node("Gather", ["sum_diag_b", "cell_sum_idx"], ["sum_path_flat"], axis=1),
        helper.make_node("Reshape", ["sum_path_flat", "mask_shape"], ["sum_path_mask"]),
        helper.make_node("Gather", ["diff_diag_b", "cell_diff_idx"], ["diff_path_flat"], axis=1),
        helper.make_node("Reshape", ["diff_path_flat", "mask_shape"], ["diff_path_mask"]),
        helper.make_node("Or", ["sum_path_mask", "diff_path_mask"], ["path_b"]),
        helper.make_node("Greater", ["red", "half"], ["red_b"]),
        helper.make_node("Greater", ["cyan", "half"], ["cyan_b"]),
        helper.make_node("Or", ["red_b", "cyan_b"], ["red_or_cyan"]),
        helper.make_node("Not", ["red_or_cyan"], ["not_red_or_cyan"]),
        helper.make_node("And", ["path_b", "not_red_or_cyan"], ["green_b"]),
        helper.make_node("Or", ["red_or_cyan", "green_b"], ["occupied"]),
        helper.make_node("Not", ["occupied"], ["bg_b"]),
        helper.make_node("Cast", ["bg_b"], ["bg_f"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["green_b"], ["green_f"], to=TensorProto.FLOAT),
        helper.make_node(
            "Concat",
            ["bg_f", "zero_plane", "red", "green_f", "zero_plane", "zero_plane", "zero_plane", "zero_plane", "cyan", "zero_plane"],
            ["out12"],
            axis=1,
        ),
        helper.make_node("Pad", ["out12"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, 18, 18]),
    ]

    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def verify_reference(data: dict[str, list[dict[str, Grid]]], candidates: list[tuple[CellSet, CellSet]]) -> None:
    for split, examples in data.items():
        for idx, example in enumerate(examples):
            if candidate_selector(example["input"], candidates) != example["output"]:
                raise AssertionError(f"reference failed {split} example {idx}")


def verify_model(model: onnx.ModelProto, data: dict[str, list[dict[str, Grid]]]) -> dict[str, tuple[int, int]]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    splits: dict[str, tuple[int, int]] = {}
    for split, examples in data.items():
        passed = 0
        checked = 0
        for example in examples:
            inp = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if inp is None or expected is None:
                continue
            pred = session.run([OUT_NAME], {IN_NAME: inp})[0]
            checked += 1
            if np.array_equal(pred > 0.0, expected > 0.0):
                passed += 1
            else:
                raise AssertionError(f"ONNX failed {split} example {checked - 1}")
        splits[split] = (passed, checked)
    return splits


def main() -> None:
    data = load_task_data()
    candidates = collect_candidates(data)
    verify_reference(data, candidates)

    print("Hypothesis comparison")
    print(f"{'hypothesis':<32} {'pass':>8} {'total':>8}")
    for name, passed, total in hypothesis_table(data, candidates):
        print(f"{name:<32} {passed:>8} {total:>8}")

    model = build_model(candidates)
    splits = verify_model(model, data)
    BEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, BEST_PATH)
    result = score_file(BEST_PATH)

    print()
    print("ONNX export comparison")
    print(f"{'candidate':<24} {'memory':>10} {'params':>10} {'cost':>10} {'score':>10}")
    print(
        f"{'diagonal_reflection':<24} {result['memory']:>10} {result['params']:>10} "
        f"{result['cost']:>10} {float(result['score']):>10.6f}"
    )
    print(f"wrote {BEST_PATH}")
    print(f"correctness: {splits}")


if __name__ == "__main__":
    main()
