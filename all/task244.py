"""Collapse separator-partitioned square cells, then mirror the coarse grid.

Task rule: the input is divided by full horizontal and vertical separator
lines into equal square blocks.  Each non-separator block is either black or a
solid colored object.  Collapse each block to one pixel using the block color,
and output the resulting coarse grid mirrored horizontally.  Separator colors
are ignored; the observed layouts are 3x3 or 4x4 blocks with cell sizes 2-5.

ONNX: detect the active layout from the true input extent, sample one interior
pixel per block in mirrored column order, gate each candidate layout, add the
selected candidate, and pad the compact one-hot result to the 30x30 NeuroGolf
output.  The only ambiguous extent is 11x11, where a single separator-line
check distinguishes 3x3 cells of size 3 from 4x4 cells of size 2.
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

from score_model import score_file  # noqa: E402

TASK_ID = "task244"
OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10

# (coarse grid size, cell size).  Total input side is n * k + (n - 1).
LAYOUTS = [(3, 2), (3, 3), (3, 4), (3, 5), (4, 2), (4, 3), (4, 4), (4, 5)]


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _make_model(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto]) -> onnx.ModelProto:
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


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver for local JSON validation."""
    g = np.asarray(grid, dtype=np.int64)
    sep_rows = [r for r, row in enumerate(g) if row[0] != 0 and np.all(row == row[0])]
    if not sep_rows:
        raise ValueError("no separator rows")
    n = len(sep_rows) + 1
    k = sep_rows[0]
    out = np.zeros((n, n), dtype=np.int64)
    for r in range(n):
        for c in range(n):
            out[r, c] = g[r * (k + 1), (n - 1 - c) * (k + 1)]
    return out


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _line_color_condition(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    *,
    n: int,
    k: int,
    total: int,
    axes4: str,
) -> str:
    """Return [1,9,1,1] bool for the separator color if this layout matches."""
    line_positions = [k + i * (k + 1) for i in range(n - 1)]
    full_threshold = _f32(inits, [float(total) - 0.5], f"full_threshold_{n}_{k}")
    half = _f32(inits, [0.5], f"half_{n}_{k}")
    current = ""

    for idx, pos in enumerate(line_positions):
        row_st = _i64(inits, [0, 1, pos, 0], f"r_st_{n}_{k}_{idx}")
        row_en = _i64(inits, [1, C, pos + 1, total], f"r_en_{n}_{k}_{idx}")
        col_st = _i64(inits, [0, 1, 0, pos], f"c_st_{n}_{k}_{idx}")
        col_en = _i64(inits, [1, C, total, pos + 1], f"c_en_{n}_{k}_{idx}")
        row = f"row_{n}_{k}_{idx}"
        col = f"col_{n}_{k}_{idx}"
        row_sum = f"row_sum_{n}_{k}_{idx}"
        col_sum = f"col_sum_{n}_{k}_{idx}"
        row_ok = f"row_ok_{n}_{k}_{idx}"
        col_ok = f"col_ok_{n}_{k}_{idx}"
        both = f"line_ok_{n}_{k}_{idx}"
        nodes.extend(
            [
                helper.make_node("Slice", [IN_NAME, row_st, row_en, axes4], [row]),
                helper.make_node("Slice", [IN_NAME, col_st, col_en, axes4], [col]),
                helper.make_node("ReduceSum", [row], [row_sum], axes=[3], keepdims=1),
                helper.make_node("ReduceSum", [col], [col_sum], axes=[2], keepdims=1),
                helper.make_node("Greater", [row_sum, full_threshold], [row_ok]),
                helper.make_node("Greater", [col_sum, full_threshold], [col_ok]),
                helper.make_node("And", [row_ok, col_ok], [both]),
            ]
        )
        if current:
            out = f"lines_ok_{n}_{k}_{idx}"
            nodes.append(helper.make_node("And", [current, both], [out]))
            current = out
        else:
            current = both

    # The first row and column after the true grid are all-zero padded.  This
    # prevents a 3x3 prefix of a larger 4x4 grid from selecting the 3x3 layout.
    b_row_st = _i64(inits, [0, 0, total, 0], f"br_st_{n}_{k}")
    b_row_en = _i64(inits, [1, C, total + 1, 1], f"br_en_{n}_{k}")
    b_col_st = _i64(inits, [0, 0, 0, total], f"bc_st_{n}_{k}")
    b_col_en = _i64(inits, [1, C, 1, total + 1], f"bc_en_{n}_{k}")
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, b_row_st, b_row_en, axes4], [f"brow_{n}_{k}"]),
            helper.make_node("Slice", [IN_NAME, b_col_st, b_col_en, axes4], [f"bcol_{n}_{k}"]),
            helper.make_node("ReduceSum", [f"brow_{n}_{k}"], [f"brow_sum_{n}_{k}"], axes=[1, 2, 3], keepdims=1),
            helper.make_node("ReduceSum", [f"bcol_{n}_{k}"], [f"bcol_sum_{n}_{k}"], axes=[1, 2, 3], keepdims=1),
            helper.make_node("Less", [f"brow_sum_{n}_{k}", half], [f"brow_ok_{n}_{k}"]),
            helper.make_node("Less", [f"bcol_sum_{n}_{k}", half], [f"bcol_ok_{n}_{k}"]),
            helper.make_node("And", [f"brow_ok_{n}_{k}", f"bcol_ok_{n}_{k}"], [f"bound_ok_{n}_{k}"]),
            helper.make_node("And", [current, f"bound_ok_{n}_{k}"], [f"layout_ok_line_{n}_{k}"]),
        ]
    )
    return f"layout_ok_line_{n}_{k}"


def _after_total_zero_gate(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    *,
    total: int,
    axes123: str,
) -> str:
    """Return scalar bool true when row total is padded all-zero."""
    zero_threshold = _f32(inits, [0.5], f"zero_threshold_{total}")
    # Every supported cell size has its first vertical separator in columns 2-5.
    # Looking only at that band is enough to distinguish in-grid rows from
    # padding, while avoiding full-width row tensors for 19x19 and 23x23 inputs.
    after_st = _i64(inits, [1, total, 2], f"after_st_{total}")
    after_en = _i64(inits, [C, total + 1, 6], f"after_en_{total}")
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, after_st, after_en, axes123], [f"after_row_{total}"]),
            helper.make_node("ReduceSum", [f"after_row_{total}"], [f"after_sum_{total}"], axes=[1, 2, 3], keepdims=1),
            helper.make_node("Less", [f"after_sum_{total}", zero_threshold], [f"after_zero_{total}"]),
        ]
    )
    return f"after_zero_{total}"


def _separator_gate(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    *,
    total: int,
    pos: int,
    name: str,
    axes123: str,
) -> str:
    """Return scalar bool true when row pos is a full non-black line."""
    full_threshold = _f32(inits, [3.5], f"{name}_full_threshold")
    row_st = _i64(inits, [1, pos, 0], f"{name}_row_st")
    row_en = _i64(inits, [C, pos + 1, 4], f"{name}_row_en")
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, row_st, row_en, axes123], [f"{name}_row"]),
            helper.make_node("ReduceSum", [f"{name}_row"], [f"{name}_row_sum"], axes=[3], keepdims=1),
            helper.make_node("Greater", [f"{name}_row_sum", full_threshold], [f"{name}_row_ok"]),
            helper.make_node("Cast", [f"{name}_row_ok"], [f"{name}_color_f"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", [f"{name}_color_f"], [f"{name}_gate"], axes=[1], keepdims=1),
            helper.make_node("Greater", [f"{name}_gate", _f32(inits, [0.5], f"{name}_half")], [f"{name}_ok"]),
        ]
    )
    return f"{name}_ok"


def _sample_candidate(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    *,
    n: int,
    k: int,
    axes4: str,
) -> str:
    row_names: list[str] = []
    for r in range(n):
        cell_names: list[str] = []
        for c in range(n):
            rr = r * (k + 1)
            cc = (n - 1 - c) * (k + 1)
            st = _i64(inits, [0, 0, rr, cc], f"cell_st_{n}_{k}_{r}_{c}")
            en = _i64(inits, [1, C, rr + 1, cc + 1], f"cell_en_{n}_{k}_{r}_{c}")
            name = f"cell_{n}_{k}_{r}_{c}"
            nodes.append(helper.make_node("Slice", [IN_NAME, st, en, axes4], [name]))
            cell_names.append(name)
        row_name = f"cand_row_{n}_{k}_{r}"
        nodes.append(helper.make_node("Concat", cell_names, [row_name], axis=3))
        row_names.append(row_name)

    cand = f"cand_{n}_{k}"
    nodes.append(helper.make_node("Concat", row_names, [cand], axis=2))
    return cand


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    axes23 = _i64(inits, [2, 3], "axes23")
    axes123 = _i64(inits, [1, 2, 3], "axes123")
    zero_after: dict[int, str] = {}
    total_gates: dict[int, str] = {}
    totals = sorted({n * k + (n - 1) for n, k in LAYOUTS})
    for idx, total in enumerate(totals):
        zero_after[total] = _after_total_zero_gate(nodes, inits, total=total, axes123=axes123)
        if idx == 0:
            total_gates[total] = zero_after[total]
        else:
            prev_total = totals[idx - 1]
            nodes.append(helper.make_node("Not", [zero_after[prev_total]], [f"after_nonzero_{prev_total}"]))
            nodes.append(
                helper.make_node("And", [zero_after[total], f"after_nonzero_{prev_total}"], [f"total_ok_{total}"])
            )
            total_gates[total] = f"total_ok_{total}"
    layout_gates: dict[tuple[int, int], str] = {}
    for n, k in LAYOUTS:
        total = n * k + (n - 1)
        layout_ok = total_gates[total]
        if total == 11:
            sep_pos = 3 if n == 3 else 2
            sep_ok = _separator_gate(
                nodes,
                inits,
                total=total,
                pos=sep_pos,
                name=f"sep_{n}_{k}",
                axes123=axes123,
            )
            nodes.append(helper.make_node("And", [layout_ok, sep_ok], [f"layout_ok_{n}_{k}"]))
            layout_ok = f"layout_ok_{n}_{k}"
        nodes.append(helper.make_node("Cast", [layout_ok], [f"layout_gate_{n}_{k}"], to=TensorProto.FLOAT))
        layout_gates[(n, k)] = f"layout_gate_{n}_{k}"

    cell_cache: dict[tuple[int, int], str] = {}
    row_names: list[str] = []
    for out_r in range(4):
        cell_names: list[str] = []
        for out_c in range(4):
            parts: list[str] = []
            for n, k in LAYOUTS:
                if out_r >= n or out_c >= n:
                    continue
                rr = out_r * (k + 1)
                cc = (n - 1 - out_c) * (k + 1)
                cell = cell_cache.get((rr, cc))
                if cell is None:
                    st = _i64(inits, [rr, cc], f"cell_st_{rr}_{cc}")
                    en = _i64(inits, [rr + 1, cc + 1], f"cell_en_{rr}_{cc}")
                    cell = f"cell_{rr}_{cc}"
                    nodes.append(helper.make_node("Slice", [IN_NAME, st, en, axes23], [cell]))
                    cell_cache[(rr, cc)] = cell
                gated = f"gcell_{n}_{k}_{out_r}_{out_c}"
                nodes.append(helper.make_node("Mul", [cell, layout_gates[(n, k)]], [gated]))
                parts.append(gated)
            merged_cell = f"out_cell_{out_r}_{out_c}"
            nodes.append(helper.make_node("Sum", parts, [merged_cell]))
            cell_names.append(merged_cell)
        row_name = f"out_row_{out_r}"
        nodes.append(helper.make_node("Concat", cell_names, [row_name], axis=3))
        row_names.append(row_name)

    nodes.append(helper.make_node("Concat", row_names, ["merged"], axis=2))
    nodes.append(helper.make_node("Pad", ["merged"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - 4, W - 4]))
    return _make_model(nodes, inits)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            grid = np.asarray(ex["input"], dtype=np.int64)
            expected = np.asarray(ex["output"], dtype=np.int64)
            ref = solve(grid)
            if not np.array_equal(ref, expected):
                raise AssertionError(f"reference mismatch {split} {idx}")
            pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(ex["input"])))[: expected.shape[0], : expected.shape[1]]
            if not np.array_equal(pred, expected):
                print(f"mismatch {split} {idx}\n{pred}\n{expected}")
                bad += 1
    return bad


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    bad = validate_json(model)
    if bad:
        raise SystemExit(f"{bad} validation mismatches")
    result = score_file(BEST_PATH)
    print(f"saved {BEST_PATH}")
    print(f"valid={result['valid']} memory={result['memory']} params={result['params']} cost={result['cost']} score={result['score']}")
    if not result["valid"]:
        raise SystemExit(result["error"])


if __name__ == "__main__":
    main()
