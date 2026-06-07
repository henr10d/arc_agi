"""ONNX generator for ARC task301 using row-bar length stacking.

Task rule from the JSON examples: every non-black row contains one horizontal
monochrome bar, and the bar lengths are exactly 1..W for the task width W.
The output keeps each bar's color and length, right-aligns every bar, and
stacks the bars in the bottom W rows by increasing length from top to bottom.
Rows above the stack are black, and padded cells outside the task rectangle
remain all-zero in the NeuroGolf tensor.

The prompt's bottom-up gravity variants are tested below for comparison, but
the JSON train pairs select the right-aligned length-stack rule.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task301"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
ACTIVE_H = 12
ACTIVE_W = 9
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


@dataclass(frozen=True)
class VariantResult:
    name: str
    train_ok: int
    train_total: int
    all_ok: int
    all_total: int


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _i32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int32), name)


def _u8(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.uint8), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


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


def _runs(grid: list[list[int]]) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for row, values in enumerate(grid):
        col = 0
        while col < len(values):
            if values[col] == 0:
                col += 1
                continue
            color = values[col]
            start = col
            while col < len(values) and values[col] == color:
                col += 1
            objects.append(
                {
                    "row": row,
                    "cols": tuple(range(start, col)),
                    "color": color,
                    "cells": [(0, c) for c in range(start, col)],
                    "height": 1,
                    "width": col - start,
                }
            )
    return objects


def _components(grid: list[list[int]]) -> list[dict[str, Any]]:
    height = len(grid)
    width = len(grid[0])
    seen: set[tuple[int, int]] = set()
    objects: list[dict[str, Any]] = []
    for row in range(height):
        for col in range(width):
            color = grid[row][col]
            if color == 0 or (row, col) in seen:
                continue
            queue = [(row, col)]
            seen.add((row, col))
            cells: list[tuple[int, int]] = []
            for rr, cc in queue:
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and (nr, nc) not in seen
                        and grid[nr][nc] == color
                    ):
                        seen.add((nr, nc))
                        queue.append((nr, nc))
            top = min(rr for rr, _ in cells)
            bottom = max(rr for rr, _ in cells)
            cols = tuple(sorted({cc for _, cc in cells}))
            objects.append(
                {
                    "row": top,
                    "cols": cols,
                    "color": color,
                    "cells": [(rr - top, cc) for rr, cc in cells],
                    "height": bottom - top + 1,
                    "width": len(cols),
                }
            )
    return objects


def solve_gravity(grid: list[list[int]], use_components: bool) -> list[list[int]] | None:
    height = len(grid)
    width = len(grid[0])
    out = [[0 for _ in range(width)] for _ in range(height)]
    stack_heights = [0 for _ in range(width)]
    objects = _components(grid) if use_components else _runs(grid)
    for obj in sorted(objects, key=lambda item: (-item["row"], min(item["cols"]))):
        support = max(stack_heights[col] for col in obj["cols"])
        target = height - obj["height"] - support
        if target < 0:
            return None
        for rr, cc in obj["cells"]:
            out[target + rr][cc] = obj["color"]
        for col in obj["cols"]:
            stack_heights[col] += sum(1 for _, cell_col in obj["cells"] if cell_col == col)
    return out


def solve_length_stack(grid: list[list[int]]) -> list[list[int]]:
    height = len(grid)
    width = len(grid[0])
    out = [[0 for _ in range(width)] for _ in range(height)]
    for obj in sorted(_runs(grid), key=lambda item: item["width"]):
        row = height - width + obj["width"] - 1
        for col in range(width - obj["width"], width):
            out[row][col] = obj["color"]
    return out


def _examples() -> list[dict[str, Any]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [ex for split in ("train", "test", "arc-gen") for ex in data.get(split, [])]


def _data_by_split() -> dict[str, list[dict[str, Any]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def evaluate_reference_variants() -> list[VariantResult]:
    data = _data_by_split()
    variants = [
        ("same_color_runs_bottom_gravity", lambda grid: solve_gravity(grid, False)),
        ("connected_components_bottom_gravity", lambda grid: solve_gravity(grid, True)),
        ("right_aligned_length_stack", solve_length_stack),
    ]
    results: list[VariantResult] = []
    all_examples = [ex for split in ("train", "test", "arc-gen") for ex in data.get(split, [])]
    for name, solver in variants:
        train_ok = sum(1 for ex in data["train"] if solver(ex["input"]) == ex["output"])
        all_ok = sum(1 for ex in all_examples if solver(ex["input"]) == ex["output"])
        results.append(VariantResult(name, train_ok, len(data["train"]), all_ok, len(all_examples)))
    return results


def _and(nodes: List[onnx.NodeProto], left: str, right: str, out: str) -> str:
    nodes.append(helper.make_node("And", [left, right], [out]))
    return out


def _where(nodes: List[onnx.NodeProto], cond: str, yes: str, no: str, out: str) -> str:
    nodes.append(helper.make_node("Where", [cond, yes, no], [out]))
    return out


def _add(nodes: List[onnx.NodeProto], left: str, right: str, out: str) -> str:
    nodes.append(helper.make_node("Add", [left, right], [out]))
    return out


def _sub(nodes: List[onnx.NodeProto], left: str, right: str, out: str) -> str:
    nodes.append(helper.make_node("Sub", [left, right], [out]))
    return out


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    fg_st = _i64(inits, [0, 1, 0, 0], "fg_st")
    fg_en = _i64(inits, [1, C, ACTIVE_H, ACTIVE_W], "fg_en")
    max_h = _i64(inits, np.asarray([[[[ACTIVE_H]]]], dtype=np.int64), "max_h")
    one_i = _i64(inits, np.asarray([[[[1]]]], dtype=np.int64), "one_i")
    one_i32 = _i32(inits, np.asarray([[[[1]]]], dtype=np.int32), "one_i32")
    zero_i32 = _i32(inits, np.asarray([[[[0]]]], dtype=np.int32), "zero_i32")
    zero_u8 = _u8(inits, np.asarray([[[[0]]]], dtype=np.uint8), "zero_u8")
    ten_u8 = _u8(inits, np.asarray([[[[10]]]], dtype=np.uint8), "ten_u8")
    rows_i = _i64(inits, np.arange(ACTIVE_H, dtype=np.int64).reshape(1, 1, ACTIVE_H, 1), "rows_i")
    cols_i = _i64(inits, np.arange(ACTIVE_W, dtype=np.int64).reshape(1, 1, 1, ACTIVE_W), "cols_i")
    channels_i = _i32(inits, np.arange(C, dtype=np.int32).reshape(1, C, 1, 1), "channels_i")
    rev_rows = _i64(inits, np.arange(ACTIVE_H - 1, -1, -1, dtype=np.int64), "rev_rows")

    _slice(nodes, IN_NAME, "fg_ch", fg_st, fg_en, axes4)
    nodes.append(helper.make_node("ArgMax", ["fg_ch"], ["color_idx0"], axis=1, keepdims=1))
    nodes.append(helper.make_node("Cast", ["color_idx0"], ["color_idx0_i32"], to=TensorProto.INT32))
    nodes.append(
        helper.make_node(
            "ReduceMax",
            ["color_idx0_i32"],
            ["row_color_idx0_i32"],
            axes=[3],
            keepdims=1,
        )
    )

    nodes.append(helper.make_node("ReduceSum", ["fg_ch"], ["row_counts"], axes=[1, 3], keepdims=1))
    nodes.append(helper.make_node("Gather", ["row_counts", rev_rows], ["row_rev"], axis=2))
    nodes.append(helper.make_node("ArgMax", ["row_rev"], ["row_pad"], axis=2, keepdims=1))
    _sub(nodes, max_h, "row_pad", "actual_h")
    nodes.append(helper.make_node("ReduceMax", ["row_counts"], ["actual_w_f"], axes=[2], keepdims=1))
    nodes.append(helper.make_node("Cast", ["actual_w_f"], ["actual_w"], to=TensorProto.INT64))
    nodes.append(helper.make_node("Cast", ["row_counts"], ["row_counts_i32"], to=TensorProto.INT32))
    _sub(nodes, "actual_h", "actual_w", "stack_top")

    color_grid = zero_u8

    for length in range(1, ACTIVE_W + 1):
        length_count_i32 = _i32(inits, np.asarray([[[[length]]]], dtype=np.int32), f"len{length}_count_i32")
        length_i = _i64(inits, np.asarray([[[[length]]]], dtype=np.int64), f"len{length}_i")
        length_m1_i = _i64(inits, np.asarray([[[[length - 1]]]], dtype=np.int64), f"len{length}_m1_i")
        nodes.append(helper.make_node("Equal", ["row_counts_i32", length_count_i32], [f"row_is_len{length}"]))
        _where(nodes, f"row_is_len{length}", "row_color_idx0_i32", zero_i32, f"color_masked_l{length}")
        nodes.append(
            helper.make_node(
                "ReduceMax",
                [f"color_masked_l{length}"],
                [f"color0_l{length}"],
                axes=[2],
                keepdims=1,
            )
        )
        _add(nodes, f"color0_l{length}", one_i32, f"color_l{length}_i32")
        nodes.append(helper.make_node("Cast", [f"color_l{length}_i32"], [f"color_l{length}_u8"], to=TensorProto.UINT8))

        _add(nodes, "stack_top", length_m1_i, f"target_row_l{length}")
        _sub(nodes, "actual_w", length_i, f"start_col_l{length}")
        _sub(nodes, f"start_col_l{length}", one_i, f"start_col_m1_l{length}")
        nodes.append(helper.make_node("Equal", [rows_i, f"target_row_l{length}"], [f"row_mask_l{length}"]))
        nodes.append(helper.make_node("Less", [f"start_col_m1_l{length}", cols_i], [f"col_ge_l{length}"]))
        _and(nodes, f"row_mask_l{length}", f"col_ge_l{length}", f"bar_mask_l{length}")
        color_grid = _where(nodes, f"bar_mask_l{length}", f"color_l{length}_u8", color_grid, f"grid_l{length}")

    nodes.append(helper.make_node("Less", [rows_i, "actual_h"], ["inside_rows"]))
    nodes.append(helper.make_node("Less", [cols_i, "actual_w"], ["inside_cols"]))
    _and(nodes, "inside_rows", "inside_cols", "inside")
    _where(nodes, "inside", color_grid, ten_u8, "color_grid_sentinel")
    nodes.append(helper.make_node("Cast", ["color_grid_sentinel"], ["color_grid_i32"], to=TensorProto.INT32))
    nodes.append(helper.make_node("Equal", [channels_i, "color_grid_i32"], ["onehot"]))
    nodes.append(helper.make_node("Cast", ["onehot"], ["out_small"], to=TensorProto.FLOAT))
    nodes.append(
        helper.make_node(
            "Pad",
            ["out_small"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - ACTIVE_H, W - ACTIVE_W],
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


def verify_correct(model_path: Path, examples: Iterable[dict[str, Any]] | None = None) -> tuple[bool, str]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    checked = 0
    for idx, example in enumerate(examples or _examples()):
        inp = convert_to_numpy(example, "input")
        expected = convert_to_numpy(example, "output")
        if inp is None or expected is None:
            continue
        actual = session.run([OUT_NAME], {IN_NAME: inp})[0]
        if not np.array_equal(actual > 0.0, expected > 0.0):
            return False, f"example {idx} mismatch"
        checked += 1
    return True, f"{checked} examples matched"


def main() -> None:
    print("reference variant comparison")
    for result in evaluate_reference_variants():
        print(
            f"{result.name:<34} train={result.train_ok}/{result.train_total} "
            f"all={result.all_ok}/{result.all_total}"
        )

    model = build_model()
    onnx.save(model, BEST_PATH)
    ok, note = verify_correct(BEST_PATH)
    result = score_file(BEST_PATH)

    print(f"wrote {BEST_PATH}")
    print(f"correct: {ok} ({note})")
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
