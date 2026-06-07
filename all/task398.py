"""Compact ONNX generator for ARC task398 diagonal row expansion.

Task rule: the input is a single 1x5 row. Let k be the number of non-zero
cells and N = 5 * k. The output is the top-left N x N square, padded to the
competition 30x30 canvas. Each non-zero input cell at original column p and
color c is copied along the full up-right diagonal row + col = N - 1 + p,
clipped to the N x N square; all other cells inside the square are color 0.

The real task JSON includes k=1..5 examples, so the ONNX graph selects one
of five precomputed 30x30 index maps, then gathers directly from a compact
float table containing zero, background, and the five input cells.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, sanitize_model, score_file  # noqa: E402

TASK_ID = "task398"
TASK_NUM = 398
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
G = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10

Grid = list[list[int]]
Hypothesis = Callable[[Grid], Grid]


def _init(inits: list[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _i32(inits: list[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int32), name)


def _f32(inits: list[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _slice(
    nodes: list[onnx.NodeProto],
    data: str,
    out: str,
    starts: str,
    ends: str,
    axes: str,
) -> str:
    nodes.append(helper.make_node("Slice", [data, starts, ends, axes], [out]))
    return out


def _load_task() -> dict[str, list[dict[str, Grid]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _all_examples() -> Iterable[tuple[str, int, dict[str, Grid]]]:
    data = _load_task()
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            yield split, idx, example


def _active(row: list[int]) -> list[tuple[int, int]]:
    return [(idx, color) for idx, color in enumerate(row) if color != 0]


def _draw(
    row: list[int],
    *,
    n_mode: str,
    full_diagonal: bool,
    reverse_rows: bool,
    compact_columns: bool,
) -> Grid:
    active = _active(row)
    k = len(active)
    if n_mode == "count":
        n = 5 if k == 1 else 5 * k
    elif n_mode == "max_count":
        n = 5 * max(1, k)
    elif n_mode == "span":
        if not active:
            n = 5
        else:
            n = 5 * max(1, active[-1][0] - active[0][0] + 1)
    else:
        raise ValueError(n_mode)

    out = [[0 for _ in range(n)] for _ in range(n)]
    steps = range(n if full_diagonal else 5)
    for compact_idx, (col0, color) in enumerate(active):
        start_col = compact_idx if compact_columns else col0
        for step in steps:
            row_idx = step if reverse_rows else n - 1 - step
            col_idx = start_col + step
            if 0 <= row_idx < n and 0 <= col_idx < n:
                out[row_idx][col_idx] = color
    return out


def _hypotheses() -> list[tuple[str, Hypothesis]]:
    return [
        (
            "given length-5 diagonals, original columns, N=count",
            lambda grid: _draw(
                grid[0],
                n_mode="count",
                full_diagonal=False,
                reverse_rows=False,
                compact_columns=False,
            ),
        ),
        (
            "full diagonals, original columns, N=5*max(1,count)",
            lambda grid: _draw(
                grid[0],
                n_mode="max_count",
                full_diagonal=True,
                reverse_rows=False,
                compact_columns=False,
            ),
        ),
        (
            "full diagonals, span-based N",
            lambda grid: _draw(
                grid[0],
                n_mode="span",
                full_diagonal=True,
                reverse_rows=False,
                compact_columns=False,
            ),
        ),
        (
            "full diagonals, reversed row direction",
            lambda grid: _draw(
                grid[0],
                n_mode="max_count",
                full_diagonal=True,
                reverse_rows=True,
                compact_columns=False,
            ),
        ),
        (
            "full diagonals, compacted nonzero columns",
            lambda grid: _draw(
                grid[0],
                n_mode="max_count",
                full_diagonal=True,
                reverse_rows=False,
                compact_columns=True,
            ),
        ),
    ]


def evaluate_hypotheses() -> list[tuple[str, int, int]]:
    data = _load_task()
    rows: list[tuple[str, int, int]] = []
    for name, fn in _hypotheses():
        ok = 0
        total = len(data["train"])
        for example in data["train"]:
            ok += int(fn(example["input"]) == example["output"])
        rows.append((name, ok, total))
    return rows


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    top_st = _i64(inits, [0, 0, 0, 0], "top_st")
    top_en = _i64(inits, [1, C, 1, 5], "top_en")
    ch0_en = _i64(inits, [1, 1, 1, 5], "ch0_en")

    four_f = _f32(inits, [4.0], "four")
    rows = np.arange(G, dtype=np.int32).reshape(G, 1)
    cols = np.arange(G, dtype=np.int32).reshape(1, G)
    index_cases = []
    for k_value in range(1, 6):
        n_value = 5 * k_value
        diagonal_offset = rows + cols - (n_value - 1)
        inside_square = (rows < n_value) & (cols < n_value)
        input_column = (0 <= diagonal_offset) & (diagonal_offset < 5)
        idx = np.where(inside_square, np.where(input_column, diagonal_offset + 2, 1), 0)
        index_cases.append(idx.astype(np.int32))
    index_table = _i32(inits, np.stack(index_cases, axis=0), "index_table")
    zero_vec = _f32(inits, np.zeros((1, C, 1), dtype=np.float32), "zero_vec")
    bg_vec_arr = np.zeros((1, C, 1), dtype=np.float32)
    bg_vec_arr[0, 0, 0] = 1.0
    bg_vec = _f32(inits, bg_vec_arr, "bg_vec")

    _slice(nodes, IN_NAME, "top", top_st, top_en, axes4)
    _slice(nodes, "top", "top0", top_st, ch0_en, axes4)
    nodes.append(helper.make_node("ReduceSum", ["top0"], ["bg_count"], axes=[3], keepdims=1))
    nodes.append(helper.make_node("Sub", [four_f, "bg_count"], ["k0f"]))
    nodes.append(helper.make_node("Cast", ["k0f"], ["k0"], to=TensorProto.INT32))
    nodes.append(helper.make_node("Squeeze", ["k0"], ["k0_s"], axes=[0, 1, 2, 3]))
    nodes.append(helper.make_node("Gather", [index_table, "k0_s"], ["pidx_safe"], axis=0))

    nodes.append(helper.make_node("Squeeze", ["top"], ["top2"], axes=[2]))
    nodes.append(helper.make_node("Concat", [zero_vec, bg_vec, "top2"], ["table"], axis=2))
    nodes.append(helper.make_node("Gather", ["table", "pidx_safe"], [OUT_NAME], axis=2))

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


def _verify_onnx(path: Path) -> dict[str, tuple[int, int]]:
    model = sanitize_model(onnx.load(path))
    if model is None:
        raise RuntimeError("model failed NeuroGolf sanitization")
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    counts = {"train": [0, 0], "test": [0, 0], "arc-gen": [0, 0]}
    for split, _idx, example in _all_examples():
        inp = convert_to_numpy(example, "input")
        expected = convert_to_numpy(example, "output")
        if inp is None or expected is None:
            continue
        pred = session.run([OUT_NAME], {IN_NAME: inp})[0] > 0.0
        counts[split][1] += 1
        counts[split][0] += int(np.array_equal(pred, expected > 0.0))
    return {split: (vals[0], vals[1]) for split, vals in counts.items()}


def realized_tensor_count(model: onnx.ModelProto) -> int:
    sanitized = sanitize_model(model)
    if sanitized is None:
        return -1
    graph = onnx.shape_inference.infer_shapes(sanitized, strict_mode=True).graph
    return sum(1 for node in graph.node for out in node.output if out and out != OUT_NAME)


def main() -> None:
    hypothesis_scores = evaluate_hypotheses()
    for name, ok, total in hypothesis_scores:
        print(f"hypothesis: {name}: train {ok}/{total}")

    selected = "full diagonals, original columns, N=5*max(1,count)"
    model = build_model()
    onnx.save(model, BEST_PATH)

    counts = _verify_onnx(BEST_PATH)
    result = score_file(BEST_PATH)

    print(f"wrote {BEST_PATH}")
    print(f"selected hypothesis: {selected}")
    print("output shape logic: N = 5 * number_of_nonzero_input_cells, for k=1..5")
    print(f"train correctness: {counts['train'][0]}/{counts['train'][1]}")
    print(f"test correctness: {counts['test'][0]}/{counts['test'][1]}")
    print(f"arc-gen correctness: {counts['arc-gen'][0]}/{counts['arc-gen'][1]}")
    print(f"node count: {len(model.graph.node)}")
    print(f"realized tensor count: {realized_tensor_count(model)}")
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
