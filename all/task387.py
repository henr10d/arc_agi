"""ONNX for ARC task387: expand rectangle-corner seeds and add gray links.

Task rule: four isolated non-black seeds mark the corners of an axis-aligned
rectangle. In the task data, opposite corners share colors; each seed becomes
the center of a 3x3 block whose surrounding eight cells use the other corner
color, while the center keeps the original seed color. Gray connector pixels
are then drawn symmetrically inward from both ends of each rectangle side,
starting just outside the 3x3 corner blocks. The output keeps the input grid
size and leaves padded cells empty for NeuroGolf I/O.

ONNX approach: work inside the 18x18 envelope used by all task examples, find
foreground row/column occupancy, build a float16 scalar color grid by
MaxPooling the opposite-color seed values into 3x3 shells, synthesize dashed
side connector masks from the dynamic side lengths, pad scalar invalid cells
with -1, then one-hot encode once at full 30x30 size.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import print_report, score_file  # noqa: E402

TASK_ID = "task387"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task387.onnx"

CHANNELS = 10
HEIGHT = WIDTH = 30
CORE = 18
SHAPE = [1, CHANNELS, HEIGHT, WIDTH]
IR_VERSION = 10
OPSET = 10


def load_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def solve_grid(grid: list[list[int]]) -> list[list[int]]:
    h, w = len(grid), len(grid[0])
    out = [[0 for _ in range(w)] for _ in range(h)]
    seeds = [(r, c, value) for r, row in enumerate(grid) for c, value in enumerate(row) if value != 0]
    rows = sorted({r for r, _, _ in seeds})
    cols = sorted({c for _, c, _ in seeds})
    if len(seeds) != 4 or len(rows) != 2 or len(cols) != 2:
        return out

    colors = sorted({value for _, _, value in seeds})
    if len(colors) != 2:
        return out

    top, bottom = rows
    left, right = cols
    other_color = {colors[0]: colors[1], colors[1]: colors[0]}

    for r, c, color in seeds:
        fill = other_color[color]
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr][cc] = fill
        out[r][c] = color

    def connector_offsets(distance: int) -> list[int]:
        offsets: set[int] = set()
        step = 2
        while step * 2 <= distance:
            offsets.add(step)
            offsets.add(distance - step)
            step += 2
        return sorted(offsets)

    for offset in connector_offsets(right - left):
        out[top][left + offset] = 5
        out[bottom][left + offset] = 5
    for offset in connector_offsets(bottom - top):
        out[top + offset][left] = 5
        out[top + offset][right] = 5

    return out


def one_hot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def _init(inits: list[onnx.TensorProto], name: str, value: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(value), name=name))
    return name


def _eq(nodes: list[onnx.NodeProto], a: str, b: str, tag: str) -> str:
    nodes.append(helper.make_node("Equal", [a, b], [f"{tag}_eq"]))
    return f"{tag}_eq"


def _or_many(nodes: list[onnx.NodeProto], names: list[str], tag: str) -> str:
    cur = names[0]
    for idx, name in enumerate(names[1:], start=1):
        out = f"{tag}_or{idx}"
        nodes.append(helper.make_node("Or", [cur, name], [out]))
        cur = out
    return cur


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)

    half = _init(inits, "half", np.array([0.5], dtype=np.float32))
    pair_half = _init(inits, "pair_half", np.array([0.5], dtype=np.float16))
    zero = _init(inits, "zero", np.array([0.0], dtype=np.float16))
    neg_one = _init(inits, "neg_one", np.array([-1.0], dtype=np.float16))
    five = _init(inits, "five", np.array([5.0], dtype=np.float16))
    zero_i64 = _init(inits, "zero_i64", np.array([0], dtype=np.int64))
    zero_i32 = _init(inits, "zero_i32", np.array([0], dtype=np.int32))
    big_i32 = _init(inits, "big_i32", np.array([100], dtype=np.int32))
    rows_i = _init(inits, "rows_i", np.arange(CORE, dtype=np.int32).reshape(1, 1, CORE, 1))
    cols_i = _init(inits, "cols_i", np.arange(CORE, dtype=np.int32).reshape(1, 1, 1, CORE))
    bg_st = _init(inits, "bg_st", np.array([0, 0, 0, 0], dtype=np.int64))
    bg_en = _init(inits, "bg_en", np.array([1, 1, CORE, CORE], dtype=np.int64))
    core_st = _init(inits, "core_st", np.array([0, 0, 0, 0], dtype=np.int64))
    core_en = _init(inits, "core_en", np.array([1, 1, CORE, CORE], dtype=np.int64))
    axes4 = _init(inits, "axes4", np.array([0, 1, 2, 3], dtype=np.int64))
    color_ids_i = _init(inits, "color_ids_i", np.arange(CHANNELS, dtype=np.int32).reshape(1, CHANNELS, 1, 1))

    nodes.extend(
        [
            helper.make_node("ArgMax", ["input"], ["color_plane_i30"], axis=1, keepdims=1),
            helper.make_node("Slice", ["color_plane_i30", core_st, core_en, axes4], ["color_plane_i"]),
            helper.make_node("Greater", ["color_plane_i", zero_i64], ["fg_any"]),
            helper.make_node("Slice", ["input", bg_st, bg_en, axes4], ["bg_input"]),
            helper.make_node("Greater", ["bg_input", half], ["bg_active"]),
            helper.make_node("Cast", ["color_plane_i"], ["color_plane"], to=TensorProto.FLOAT16),
            helper.make_node("Or", ["bg_active", "fg_any"], ["valid"]),
            helper.make_node("Cast", ["fg_any"], ["fg_i"], to=TensorProto.INT32),
            helper.make_node("ReduceMax", ["fg_i"], ["row_any_i"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["fg_i"], ["col_any_i"], axes=[2], keepdims=1),
            helper.make_node("Greater", ["row_any_i", zero_i32], ["row_any"]),
            helper.make_node("Greater", ["col_any_i", zero_i32], ["col_any"]),
            helper.make_node("Mul", ["row_any_i", rows_i], ["fg_rows"]),
            helper.make_node("Mul", ["col_any_i", cols_i], ["fg_cols"]),
            helper.make_node("ReduceMax", ["fg_rows"], ["row_max"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMax", ["fg_cols"], ["col_max"], axes=[2, 3], keepdims=1),
            helper.make_node("Where", ["row_any", rows_i, big_i32], ["fg_rows_big"]),
            helper.make_node("Where", ["col_any", cols_i, big_i32], ["fg_cols_big"]),
            helper.make_node("ReduceMin", ["fg_rows_big"], ["row_min"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMin", ["fg_cols_big"], ["col_min"], axes=[2, 3], keepdims=1),
            helper.make_node("Sub", ["row_max", "row_min"], ["row_dist"]),
            helper.make_node("Sub", ["col_max", "col_min"], ["col_dist"]),
            helper.make_node("ReduceSum", ["color_plane"], ["color_sum"], axes=[2, 3], keepdims=1),
            helper.make_node("Mul", ["color_sum", pair_half], ["color_pair_sum"]),
            helper.make_node("Sub", ["color_pair_sum", "color_plane"], ["other_color_value"]),
            helper.make_node("Where", ["fg_any", "other_color_value", zero], ["other_seed_value"]),
            helper.make_node(
                "MaxPool",
                ["other_seed_value"],
                ["block_shell_value"],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
            ),
            helper.make_node("Where", ["fg_any", "color_plane", "block_shell_value"], ["block_value"]),
        ]
    )

    row_min_mask = _eq(nodes, rows_i, "row_min", "row_min")
    row_max_mask = _eq(nodes, rows_i, "row_max", "row_max")
    col_min_mask = _eq(nodes, cols_i, "col_min", "col_min")
    col_max_mask = _eq(nodes, cols_i, "col_max", "col_max")

    left_hits: list[str] = []
    right_hits: list[str] = []
    top_hits: list[str] = []
    bottom_hits: list[str] = []
    for offset in range(2, 6, 2):
        off = _init(inits, f"off{offset}", np.array([offset], dtype=np.int32))
        need = _init(inits, f"need{offset}", np.array([offset * 2 - 1], dtype=np.int32))
        nodes.extend(
            [
                helper.make_node("Greater", ["col_dist", need], [f"h_allow{offset}"]),
                helper.make_node("Add", ["col_min", off], [f"left_pos{offset}"]),
                helper.make_node("Sub", ["col_max", off], [f"right_pos{offset}"]),
                helper.make_node("Greater", ["row_dist", need], [f"v_allow{offset}"]),
                helper.make_node("Add", ["row_min", off], [f"top_pos{offset}"]),
                helper.make_node("Sub", ["row_max", off], [f"bottom_pos{offset}"]),
            ]
        )
        left_eq = _eq(nodes, cols_i, f"left_pos{offset}", f"left_eq{offset}")
        right_eq = _eq(nodes, cols_i, f"right_pos{offset}", f"right_eq{offset}")
        top_eq = _eq(nodes, rows_i, f"top_pos{offset}", f"top_eq{offset}")
        bottom_eq = _eq(nodes, rows_i, f"bottom_pos{offset}", f"bottom_eq{offset}")
        nodes.extend(
            [
                helper.make_node("And", [left_eq, f"h_allow{offset}"], [f"left_hit{offset}"]),
                helper.make_node("And", [right_eq, f"h_allow{offset}"], [f"right_hit{offset}"]),
                helper.make_node("And", [top_eq, f"v_allow{offset}"], [f"top_hit{offset}"]),
                helper.make_node("And", [bottom_eq, f"v_allow{offset}"], [f"bottom_hit{offset}"]),
            ]
        )
        left_hits.append(f"left_hit{offset}")
        right_hits.append(f"right_hit{offset}")
        top_hits.append(f"top_hit{offset}")
        bottom_hits.append(f"bottom_hit{offset}")

    h_col_mask = _or_many(nodes, [*left_hits, *right_hits], "h_col")
    v_row_mask = _or_many(nodes, [*top_hits, *bottom_hits], "v_row")
    top_bottom_mask = _or_many(nodes, [row_min_mask, row_max_mask], "top_bottom")
    left_right_mask = _or_many(nodes, [col_min_mask, col_max_mask], "left_right")
    nodes.extend(
        [
            helper.make_node("And", [top_bottom_mask, h_col_mask], ["h_conn"]),
            helper.make_node("And", [left_right_mask, v_row_mask], ["v_conn"]),
            helper.make_node("Or", ["h_conn", "v_conn"], ["connector"]),
            helper.make_node("Where", ["valid", "block_value", neg_one], ["base_value"]),
            helper.make_node("Where", ["connector", five, "base_value"], ["color_grid18"]),
            helper.make_node(
                "Pad",
                ["color_grid18"],
                ["color_grid"],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, HEIGHT - CORE, WIDTH - CORE],
                value=-1.0,
            ),
            helper.make_node("Cast", ["color_grid"], ["color_grid_i"], to=TensorProto.INT32),
            helper.make_node("Equal", ["color_ids_i", "color_grid_i"], ["out_onehot"]),
            helper.make_node("Cast", ["out_onehot"], ["output"], to=TensorProto.FLOAT),
        ]
    )

    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], inits)
    model = helper.make_model(
        graph,
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def validate_reference(data: dict[str, list[dict[str, list[list[int]]]]]) -> None:
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            actual = solve_grid(example["input"])
            if actual != example["output"]:
                raise AssertionError(f"reference solver failed {split}[{idx}]")


def validate_model(path: Path, data: dict[str, list[dict[str, list[list[int]]]]]) -> None:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            pred = session.run(["output"], {"input": one_hot(example["input"])})[0]
            expected = one_hot(example["output"])
            if not np.array_equal(pred > 0.0, expected > 0.0):
                diff = np.argwhere((pred > 0.0) != (expected > 0.0))
                raise AssertionError(f"model failed {split}[{idx}] at {diff[:8].tolist()}")


def main() -> None:
    data = load_data()
    validate_reference(data)
    model = build_model()
    onnx.save(model, BEST_PATH)
    validate_model(BEST_PATH, data)
    print_report(score_file(BEST_PATH))


if __name__ == "__main__":
    main()
