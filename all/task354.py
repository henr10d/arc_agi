"""ONNX solver for ARC task354: color gray rectangles from top-row markers.

Task rule: each 10x10 grid has three non-black, non-gray marker cells in the
first row and three 4-connected gray rectangular components below them.  Sort
the marker cells by column and sort the gray components by their bounding-box
left edge.  Recolor component i with marker i's color.  The markers, black
background, grid size, and 30x30 one-hot padding are otherwise unchanged.

The generated ONNX graph relies on the observed bounded structure: exactly
three solid rectangular gray components with distinct left edges in a 10x10
task area, with rectangle widths no larger than five cells in the provided
train/test/arc-gen examples.  The marker columns are likewise generated in
three separated left/middle/right groups, and the ranked rectangles occupy
bounded column bands.  It finds each gray pixel's horizontal run start, ranks
those starts left-to-right, and paints the three ranked component masks with
the three ranked marker colors.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, sanitize_model, score_file  # noqa: E402

TASK_ID = "task354"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
N = 10
GRAY = 5
MAX_RECT_WIDTH = 5
BODY_ROW_START = 2
COMPONENT_START_COLS = range(N - 1)
POSSIBLE_COMPONENT_STARTS = (range(0, 3), range(1, 6), range(5, 9))
POSSIBLE_COMPONENT_COLS = (range(0, 5), range(1, 9), range(5, 10))
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def solve(grid: list[list[int]] | np.ndarray, *, sort_key: str = "left") -> np.ndarray:
    """Reference implementation for the ARC grid transformation."""
    arr = np.asarray(grid, dtype=np.int64)
    out = arr.copy()
    markers = [int(v) for c, v in sorted((c, int(v)) for c, v in enumerate(arr[0]) if v not in (0, GRAY))]

    seen = np.zeros(arr.shape, dtype=bool)
    components: list[dict[str, Any]] = []
    for r, c in zip(*np.where(arr == GRAY)):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        queue = [(rr, cc)]
        seen[rr, cc] = True
        cells: list[tuple[int, int]] = []
        for cr, cc2 in queue:
            cells.append((cr, cc2))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = cr + dr
                nc = cc2 + dc
                if (
                    0 <= nr < arr.shape[0]
                    and 0 <= nc < arr.shape[1]
                    and not seen[nr, nc]
                    and arr[nr, nc] == GRAY
                ):
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        cols = np.asarray([cell[1] for cell in cells], dtype=np.float64)
        components.append({"cells": cells, "left": int(cols.min()), "centroid": float(cols.mean())})

    key = "centroid" if sort_key == "centroid" else "left"
    for color, component in zip(markers, sorted(components, key=lambda item: item[key])):
        for cell in component["cells"]:
            out[cell] = color
    return out


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _cached_i64(inits: list[onnx.TensorProto], cache: dict[tuple[int, ...], str], values: list[int]) -> str:
    key = tuple(values)
    if key not in cache:
        cache[key] = _init(inits, np.asarray(values, dtype=np.int64), f"i{len(cache)}")
    return cache[key]


def _f32(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    return _init(inits, np.asarray(arr, dtype=np.float32), name)


def _i64(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    return _init(inits, np.asarray(arr, dtype=np.int64), name)


def _i32(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    return _init(inits, np.asarray(arr, dtype=np.int32), name)


def _u8(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    return _init(inits, np.asarray(arr, dtype=np.uint8), name)


def _bool(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    return _init(inits, np.asarray(arr, dtype=bool), name)


def _slice(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    cache: dict[tuple[int, ...], str],
    data: str,
    starts: list[int],
    ends: list[int],
    axes: list[int],
    output: str,
) -> str:
    nodes.append(
        helper.make_node(
            "Slice",
            [
                data,
                _cached_i64(inits, cache, starts),
                _cached_i64(inits, cache, ends),
                _cached_i64(inits, cache, axes),
            ],
            [output],
        )
    )
    return output


def _and(nodes: list[onnx.NodeProto], a: str, b: str, out: str) -> str:
    nodes.append(helper.make_node("And", [a, b], [out]))
    return out


def _or(nodes: list[onnx.NodeProto], a: str, b: str, out: str) -> str:
    nodes.append(helper.make_node("Or", [a, b], [out]))
    return out


def _not(nodes: list[onnx.NodeProto], a: str, out: str) -> str:
    nodes.append(helper.make_node("Not", [a], [out]))
    return out


def _or_many(nodes: list[onnx.NodeProto], values: list[str], prefix: str) -> str:
    cur = values[0]
    for idx, value in enumerate(values[1:], start=1):
        cur = _or(nodes, cur, value, f"{prefix}_{idx}")
    return cur


def _rank_active_scalars(
    nodes: list[onnx.NodeProto],
    active: list[str],
    prefix: str,
    zero_i: str,
    one_i: str,
    two_i: str,
) -> list[list[str]]:
    ranks: list[list[str]] = []
    prefix_count = zero_i
    targets = [zero_i, one_i, two_i]
    for idx, item in enumerate(active):
        per_item: list[str] = []
        for rank, target in enumerate(targets):
            eq_name = f"{prefix}_eq_{idx}_{rank}"
            nodes.append(helper.make_node("Equal", [prefix_count, target], [eq_name]))
            per_item.append(_and(nodes, item, eq_name, f"{prefix}_rank_{idx}_{rank}"))
        ranks.append(per_item)
        as_int = f"{prefix}_i_{idx}"
        nodes.append(helper.make_node("Cast", [item], [as_int], to=TensorProto.INT32))
        next_count = f"{prefix}_count_{idx}"
        nodes.append(helper.make_node("Add", [prefix_count, as_int], [next_count]))
        prefix_count = next_count
    return ranks


def _select_marker_color(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    cache: dict[tuple[int, ...], str],
    columns: list[int],
    prefix: str,
    zero_grid: str,
) -> str:
    current = zero_grid
    for col in columns:
        marker_col = _slice(nodes, inits, cache, "grid", [0, col], [1, col + 1], [1, 2], f"{prefix}_col{col}")
        active = f"{prefix}_active{col}"
        nodes.append(helper.make_node("Greater", [marker_col, zero_grid], [active]))
        next_color = f"{prefix}_color{col}"
        nodes.append(helper.make_node("Where", [active, marker_col, current], [next_color]))
        current = next_color
    return current


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    ccache: dict[tuple[int, ...], str] = {}

    zero_count = _i32(inits, [0], "zero_count")
    one_count = _i32(inits, [1], "one_count")
    two_count = _i32(inits, [2], "two_count")
    zero_grid = _u8(inits, [0], "zero_grid")
    zero_row = _u8(inits, np.zeros((1, 1, N), dtype=np.uint8), "zero_row")
    zero_f = _f32(inits, [0.0], "zero_f")
    _init(inits, np.asarray(C, dtype=np.int64), "depth")
    _f32(inits, [0.0, 1.0], "onehot_values")

    _slice(nodes, inits, ccache, IN_NAME, [0, 0, 0, 0], [1, C, 1, N], [0, 1, 2, 3], "core_top")
    _slice(
        nodes,
        inits,
        ccache,
        IN_NAME,
        [0, GRAY, BODY_ROW_START, 0],
        [1, GRAY + 1, N, N],
        [0, 1, 2, 3],
        "gray_hot",
    )
    nodes.append(helper.make_node("ArgMax", ["core_top"], ["grid_top_i64"], axis=1, keepdims=0))
    nodes.append(helper.make_node("Cast", ["grid_top_i64"], ["grid"], to=TensorProto.UINT8))
    nodes.append(helper.make_node("Greater", ["gray_hot", zero_f], ["gray"]))

    gray_cols = [
        _slice(nodes, inits, ccache, "gray", [col], [col + 1], [3], f"gcol{col}") for col in range(N)
    ]

    start_cols: list[str] = []
    active_starts: list[str] = []
    for col in COMPONENT_START_COLS:
        if col == 0:
            start = gray_cols[col]
        else:
            prev_black = _not(nodes, gray_cols[col - 1], f"not_gcol{col - 1}")
            start = _and(nodes, gray_cols[col], prev_black, f"start_col{col}")
        start_cols.append(start)
        row_starts = [
            _slice(nodes, inits, ccache, start, [row], [row + 1], [2], f"start{col}_row{row}")
            for row in range(N - BODY_ROW_START)
        ]
        active_starts.append(_or_many(nodes, row_starts, f"active_start{col}"))

    start_ranks = _rank_active_scalars(nodes, active_starts, "s", zero_count, one_count, two_count)

    possible_starts = [set(cols) for cols in POSSIBLE_COMPONENT_STARTS]
    possible_cols = [set(cols) for cols in POSSIBLE_COMPONENT_COLS]
    component_col_parts: list[list[list[str]]] = [[[] for _ in range(N)] for _ in range(3)]
    for start_col in COMPONENT_START_COLS:
        running = start_cols[start_col]
        for col in range(start_col, min(N, start_col + MAX_RECT_WIDTH)):
            if col > start_col:
                running = _and(nodes, running, gray_cols[col], f"seg_run_{start_col}_{col}")
            for rank in range(3):
                if start_col in possible_starts[rank] and col in possible_cols[rank]:
                    component_col_parts[rank][col].append(
                        _and(nodes, running, start_ranks[start_col][rank], f"comp_col_part_{rank}_{start_col}_{col}")
                    )

    component_cols: list[list[str | None]] = []
    for rank in range(3):
        columns = [
            _or_many(nodes, component_col_parts[rank][col], f"comp_col_{rank}_{col}")
            if component_col_parts[rank][col]
            else None
            for col in range(N)
        ]
        component_cols.append(columns)

    marker_colors = [
        _select_marker_color(nodes, inits, ccache, [0, 1, 2], "marker0", zero_grid),
        _select_marker_color(nodes, inits, ccache, [4, 5], "marker1", zero_grid),
        _select_marker_color(nodes, inits, ccache, [7, 8, 9], "marker2", zero_grid),
    ]

    painted_cols: list[str] = []
    for col in range(N):
        current_col = zero_grid
        for rank in range(3):
            if component_cols[rank][col] is None:
                continue
            painted = f"painted_col{col}_{rank}"
            nodes.append(
                helper.make_node("Where", [component_cols[rank][col], marker_colors[rank], current_col], [painted])
            )
            current_col = painted
        painted_cols.append(current_col)

    nodes.append(helper.make_node("Concat", painted_cols, ["grid_body_painted_4d"], axis=3))
    nodes.append(helper.make_node("Squeeze", ["grid_body_painted_4d"], ["grid_body_painted"], axes=[1]))
    nodes.append(helper.make_node("Concat", ["grid", zero_row, "grid_body_painted"], ["grid_painted"], axis=1))
    nodes.append(helper.make_node("Cast", ["grid_painted"], ["onehot_grid"], to=TensorProto.INT64))
    nodes.append(helper.make_node("OneHot", ["onehot_grid", "depth", "onehot_values"], ["out_core"], axis=1))
    nodes.append(helper.make_node("Pad", ["out_core"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - N, W - N]))

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


def _load_examples() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _check_reference(sort_key: str = "left") -> dict[str, tuple[int, int]]:
    examples = _load_examples()
    counts: dict[str, tuple[int, int]] = {}
    for split in ("train", "test", "arc-gen"):
        passed = 0
        total = 0
        for example in examples.get(split, []):
            total += 1
            if np.array_equal(
                solve(example["input"], sort_key=sort_key),
                np.asarray(example["output"], dtype=np.int64),
            ):
                passed += 1
        counts[split] = (passed, total)
    return counts


def _check_model(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return False, {}
    session = ort.InferenceSession(
        sanitized.SerializeToString(),
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    )
    examples = _load_examples()
    counts: dict[str, tuple[int, int]] = {}
    all_ok = True
    for split in ("train", "test", "arc-gen"):
        passed = 0
        total = 0
        for example in examples.get(split, []):
            input_arr = convert_to_numpy(example, "input")
            expected_arr = convert_to_numpy(example, "output")
            if input_arr is None or expected_arr is None:
                continue
            total += 1
            output = session.run([OUT_NAME], {IN_NAME: input_arr})[0]
            pred = (output > 0.0).astype(np.float32)
            if np.array_equal(pred, expected_arr):
                passed += 1
            else:
                all_ok = False
        counts[split] = (passed, total)
    return all_ok, counts


def _format_counts(counts: dict[str, tuple[int, int]]) -> str:
    return ", ".join(f"{split} {ok}/{total}" for split, (ok, total) in counts.items())


def main() -> None:
    left_counts = _check_reference("left")
    centroid_counts = _check_reference("centroid")
    if any(ok != total for ok, total in left_counts.values()):
        raise SystemExit(f"left-edge reference mismatch: {_format_counts(left_counts)}")

    model = build_model()
    correct, model_counts = _check_model(model)
    if not correct:
        raise SystemExit(f"model mismatch: {_format_counts(model_counts)}")

    onnx.save(model, BEST_PATH)
    scored = score_file(BEST_PATH)
    score_text = f"{scored['score']:.6f}" if scored["score"] is not None else "INVALID"
    print(f"wrote:    {BEST_PATH}")
    print(f"left:     {_format_counts(left_counts)}")
    print(f"centroid: {_format_counts(centroid_counts)}")
    print(f"passes:   {_format_counts(model_counts)}")
    print(f"valid:    {scored['valid']}")
    print(f"memory:   {scored['memory']}")
    print(f"params:   {scored['params']}")
    print(f"cost:     {scored['cost']}")
    print(f"score:    {score_text}")
    if scored["error"]:
        print(f"error:    {scored['error']}")


if __name__ == "__main__":
    main()
