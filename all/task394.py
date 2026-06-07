"""Minimal ONNX for ARC task394 using periodic hole reconstruction.

Task rule: the active input is a small square grid containing a repeated
non-zero color pattern with one rectangular block replaced by color 0. Emit
the missing block itself in the top-left of the output. The task instances use
4x4 grids with a 1x1 hole, 5x5/6x6 grids with a 2x2 hole, and 7x7 grids with
either a 2x2 or 3x3 hole; the underlying period is 2 for sizes 4-6 and 3 for
size 7.

The ONNX graph detects the zero rectangle, recovers the hidden colors by
counting non-zero cells in each row/column residue class, and gates the
matching candidate rectangle into a compact 3x3 output before padding to the
competition [1, 10, 30, 30] tensor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task394"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task394.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
ACTIVE = 7
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


def _candidate_specs() -> Iterable[tuple[int, int, int, int, int]]:
    # (active grid size, hole size, period, hole row, hole col)
    for grid_size, hole_size, period in ((4, 1, 2), (5, 2, 2), (6, 2, 2), (7, 2, 3), (7, 3, 3)):
        for row0 in range(grid_size - hole_size + 1):
            for col0 in range(grid_size - hole_size + 1):
                yield grid_size, hole_size, period, row0, col0


def _candidate_groups() -> Iterable[tuple[int, int, int, int, tuple[int, ...]]]:
    grouped: dict[tuple[int, int, int, int], set[int]] = {}
    for grid_size, hole_size, period, row0, col0 in _candidate_specs():
        grouped.setdefault((hole_size, period, row0, col0), set()).add(grid_size * grid_size)
    for (hole_size, period, row0, col0), grid_areas in sorted(grouped.items()):
        yield hole_size, period, row0, col0, tuple(sorted(grid_areas))


def _or_many(nodes: List[onnx.NodeProto], inputs: list[str], output: str) -> str:
    if len(inputs) == 1:
        return inputs[0]
    current = inputs[0]
    for idx, name in enumerate(inputs[1:], start=1):
        out = output if idx == len(inputs) - 1 else f"{output}_{idx}"
        nodes.append(helper.make_node("Or", [current, name], [out]))
        current = out
    return current


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    slice_axes = _i64(inits, [0, 1, 2, 3], "slice_axes")
    zero_ch_start = _i64(inits, [0, 0, 0, 0], "zero_ch_start")
    zero_ch_end = _i64(inits, [1, 1, ACTIVE, ACTIVE], "zero_ch_end")
    line_axes = _i64(inits, [2, 3], "line_axes")
    residue_end = _i64(inits, [1, C, ACTIVE, ACTIVE], "residue_end")
    residue_steps = {period: _i64(inits, [1, 1, period, period], f"residue_steps_{period}") for period in (2, 3)}
    channels = _i64(inits, np.arange(C - 1, dtype=np.int64).reshape(1, C - 1, 1, 1), "channels")
    false_plane = _init(inits, np.zeros((1, 1, 3, 3), dtype=bool), "false_plane")

    nodes.append(helper.make_node("Slice", [IN_NAME, zero_ch_start, zero_ch_end, slice_axes], ["zero_ch"]))
    nodes.append(helper.make_node("ReduceSum", [IN_NAME], ["active_total"], axes=[0, 1, 2, 3], keepdims=1))
    nodes.append(helper.make_node("ReduceSum", ["zero_ch"], ["zero_total"], axes=[0, 1, 2, 3], keepdims=1))
    nodes.append(helper.make_node("Cast", ["active_total"], ["active_total_i"], to=TensorProto.INT64))
    nodes.append(helper.make_node("Cast", ["zero_total"], ["zero_total_i"], to=TensorProto.INT64))

    hole_eq: dict[int, str] = {}
    for hole_area in (1, 4, 9):
        hole_area_name = _i64(inits, [[hole_area]], f"hole_area_{hole_area}")
        hole_eq[hole_area] = f"hole_eq_{hole_area}"
        nodes.append(helper.make_node("Equal", ["zero_total_i", hole_area_name], [hole_eq[hole_area]]))

    grid_eq: dict[int, str] = {}
    for grid_area in (16, 25, 36, 49):
        grid_area_name = _i64(inits, [[grid_area]], f"grid_area_{grid_area}")
        grid_eq[grid_area] = f"grid_eq_{grid_area}"
        nodes.append(helper.make_node("Equal", ["active_total_i", grid_area_name], [grid_eq[grid_area]]))

    zero_count = _i64(inits, [[0]], "zero_count")
    line_count: dict[int, str] = {1: "hole_area_1"}
    for count in (2, 3):
        line_count[count] = _i64(inits, [[count]], f"line_count_{count}")

    row_zero: dict[int, str] = {}
    row_eq: dict[tuple[int, int], str] = {}
    col_zero: dict[int, str] = {}
    col_eq: dict[tuple[int, int], str] = {}

    for row in range(ACTIVE):
        start = _i64(inits, [row, 0], f"row_start_idx_{row}")
        end = _i64(inits, [row + 1, ACTIVE], f"row_end_idx_{row}")
        nodes.append(helper.make_node("Slice", ["zero_ch", start, end, line_axes], [f"row_g_{row}"]))
        nodes.append(helper.make_node("ReduceSum", [f"row_g_{row}"], [f"row_sum_{row}"], axes=[0, 1, 2, 3], keepdims=1))
        nodes.append(helper.make_node("Cast", [f"row_sum_{row}"], [f"row_sum_i_{row}"], to=TensorProto.INT64))
        row_zero[row] = f"row_zero_{row}"
        nodes.append(helper.make_node("Equal", [f"row_sum_i_{row}", zero_count], [row_zero[row]]))
        for count in (1, 2, 3):
            row_eq[(count, row)] = f"row_eq_{count}_{row}"
            nodes.append(helper.make_node("Equal", [f"row_sum_i_{row}", line_count[count]], [row_eq[(count, row)]]))

    for col in range(ACTIVE):
        start = _i64(inits, [0, col], f"col_start_idx_{col}")
        end = _i64(inits, [ACTIVE, col + 1], f"col_end_idx_{col}")
        nodes.append(helper.make_node("Slice", ["zero_ch", start, end, line_axes], [f"col_g_{col}"]))
        nodes.append(helper.make_node("ReduceSum", [f"col_g_{col}"], [f"col_sum_{col}"], axes=[0, 1, 2, 3], keepdims=1))
        nodes.append(helper.make_node("Cast", [f"col_sum_{col}"], [f"col_sum_i_{col}"], to=TensorProto.INT64))
        col_zero[col] = f"col_zero_{col}"
        nodes.append(helper.make_node("Equal", [f"col_sum_i_{col}", zero_count], [col_zero[col]]))
        for count in (1, 2, 3):
            col_eq[(count, col)] = f"col_eq_{count}_{col}"
            nodes.append(helper.make_node("Equal", [f"col_sum_i_{col}", line_count[count]], [col_eq[(count, col)]]))

    row_start: dict[tuple[int, int], str] = {}
    col_start: dict[tuple[int, int], str] = {}
    for count in (1, 2, 3):
        for row in range(ACTIVE - count + 1):
            if row == 0:
                row_start[(count, row)] = row_eq[(count, row)]
            else:
                name = f"row_start_{count}_{row}"
                nodes.append(helper.make_node("And", [row_eq[(count, row)], row_zero[row - 1]], [name]))
                row_start[(count, row)] = name
        for col in range(ACTIVE - count + 1):
            if col == 0:
                col_start[(count, col)] = col_eq[(count, col)]
            else:
                name = f"col_start_{count}_{col}"
                nodes.append(helper.make_node("And", [col_eq[(count, col)], col_zero[col - 1]], [name]))
                col_start[(count, col)] = name

    class_cells: dict[tuple[int, int, int], str] = {}
    for period in (2, 3):
        for row_residue in range(period):
            for col_residue in range(period):
                suffix = f"p{period}_{row_residue}_{col_residue}"
                start = _i64(inits, [0, 1, row_residue, col_residue], f"residue_start_{suffix}")
                nodes.append(
                    helper.make_node(
                        "Slice",
                        [IN_NAME, start, residue_end, slice_axes, residue_steps[period]],
                        [f"residue_{suffix}"],
                    )
                )
                nodes.append(
                    helper.make_node(
                        "ReduceSum",
                        [f"residue_{suffix}"],
                        [f"counts_{suffix}"],
                        axes=[0, 2, 3],
                        keepdims=1,
                    )
                )
                nodes.append(helper.make_node("ArgMax", [f"counts_{suffix}"], [f"arg_{suffix}"], axis=1, keepdims=1))
                nodes.append(helper.make_node("Equal", [channels, f"arg_{suffix}"], [f"hot9_{suffix}"]))
                class_cells[(period, row_residue, col_residue)] = f"hot9_{suffix}"

    cell_class_conds: dict[tuple[int, int], dict[tuple[int, int, int], list[str]]] = {
        (row, col): {} for row in range(3) for col in range(3)
    }

    for cand_idx, (hole_size, period, row0, col0, grid_areas) in enumerate(_candidate_groups()):
        suffix = f"c{cand_idx}"
        hole_area = hole_size * hole_size

        grid_name = _or_many(nodes, [grid_eq[area] for area in grid_areas], f"grid_or_{suffix}")
        nodes.append(helper.make_node("And", [row_start[(hole_size, row0)], col_start[(hole_size, col0)]], [f"pos_and_{suffix}"]))
        nodes.append(helper.make_node("And", [f"pos_and_{suffix}", hole_eq[hole_area]], [f"hole_and_{suffix}"]))
        nodes.append(helper.make_node("And", [f"hole_and_{suffix}", grid_name], [f"condb_{suffix}"]))

        for out_row in range(hole_size):
            for out_col in range(hole_size):
                class_key = (period, (row0 + out_row) % period, (col0 + out_col) % period)
                cell_class_conds[(out_row, out_col)].setdefault(class_key, []).append(f"condb_{suffix}")

    compact_cells: list[list[str]] = []
    for out_row in range(3):
        row_cells: list[str] = []
        for out_col in range(3):
            terms: list[str] = []
            for class_idx, (class_key, conds) in enumerate(sorted(cell_class_conds[(out_row, out_col)].items())):
                cond_name = _or_many(nodes, conds, f"cond_or_{out_row}_{out_col}_{class_idx}")
                term_name = f"term_{out_row}_{out_col}_{class_idx}"
                nodes.append(helper.make_node("And", [cond_name, class_cells[class_key]], [term_name]))
                terms.append(term_name)
            cell_name = _or_many(nodes, terms, f"cell_{out_row}_{out_col}")
            row_cells.append(cell_name)
        compact_cells.append(row_cells)

    row_names: list[str] = []
    for out_row, row_cells in enumerate(compact_cells):
        row_name = f"row_{out_row}"
        nodes.append(helper.make_node("Concat", row_cells, [row_name], axis=3))
        row_names.append(row_name)
    nodes.append(helper.make_node("Concat", row_names, ["out9"], axis=2))
    nodes.append(helper.make_node("Concat", [false_plane, "out9"], ["out3"], axis=1))
    nodes.append(helper.make_node("Cast", ["out3"], ["out3f"], to=TensorProto.FLOAT))
    nodes.append(
        helper.make_node(
            "Pad",
            ["out3f"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - 3, W - 3],
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


def _check_examples(path: Path) -> tuple[int, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    passed = 0
    failed = 0
    for split in ("train", "test", "arc-gen"):
        for example in data.get(split, []):
            expected = convert_to_numpy(example, "output")
            actual = session.run([OUT_NAME], {IN_NAME: convert_to_numpy(example, "input")})[0]
            if np.array_equal(actual > 0.0, expected > 0.0):
                passed += 1
            else:
                failed += 1
    return passed, failed


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    passed, failed = _check_examples(BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"examples: {passed} pass, {failed} fail")
    print(f"valid:    {result['valid']}")
    if result["error"]:
        print(f"error:    {result['error']}")
    print(f"memory:   {result['memory']}")
    print(f"params:   {result['params']}")
    print(f"cost:     {result['cost']}")
    if result["score"] is not None:
        print(f"score:    {result['score']:.6f}")


if __name__ == "__main__":
    main()
