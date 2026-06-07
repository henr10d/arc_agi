"""ONNX solver for task335: draw a yellow L path between cyan and red cells.

Task rule: each grid contains one cyan cell (color 8) and one red cell
(color 2) on a black background.  The output keeps the same grid size, preserves
the two endpoints, and fills color 4 along the orthogonal L whose vertical leg
uses the cyan column and whose horizontal leg uses the red row.  The corner is
at (red row, cyan column); padded cells outside the input grid remain all-zero
in the NeuroGolf tensor.  The ONNX graph searches the 17x17 interior region
containing all visible endpoint examples, then uses the original one-hot input
as the default output and overwrites only interior path cells with yellow.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "task335"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task335.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

sys.path.insert(0, str(ROOT))
from score_model import score_file  # noqa: E402

C = 10
H = W = 30
N = 17
OFFSET = 1
RED_ROWS = 15
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10

BG = 0
RED = 2
PATH = 4
CYAN = 8


def _init(inits: List[onnx.TensorProto], name: str, arr: np.ndarray) -> str:
    inits.append(numpy_helper.from_array(arr, name=name))
    return name


def _i64(inits: List[onnx.TensorProto], name: str, vals: list[int]) -> str:
    return _init(inits, name, np.asarray(vals, dtype=np.int64))


def _f32(inits: List[onnx.TensorProto], name: str, arr) -> str:
    return _init(inits, name, np.asarray(arr, dtype=np.float32))


def _bool(inits: List[onnx.TensorProto], name: str, arr) -> str:
    return _init(inits, name, np.asarray(arr, dtype=bool))


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference NumPy implementation of the L-shaped connector rule."""
    arr = np.asarray(grid, dtype=np.int64)
    out = np.full(arr.shape, BG, dtype=np.int64)
    cyan_pos = np.argwhere(arr == CYAN)
    red_pos = np.argwhere(arr == RED)
    if cyan_pos.shape != (1, 2) or red_pos.shape != (1, 2):
        raise ValueError("task335 expects exactly one cyan cell and one red cell")

    rc, cc = map(int, cyan_pos[0])
    rr, cr = map(int, red_pos[0])
    out[min(rc, rr) : max(rc, rr) + 1, cc] = PATH
    out[rr, min(cc, cr) : max(cc, cr) + 1] = PATH
    out[rc, cc] = CYAN
    out[rr, cr] = RED
    return out


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, "a4", [0, 1, 2, 3])
    cyan_st = _i64(inits, "s8", [0, CYAN, OFFSET, OFFSET])
    cyan_en = _i64(inits, "e8", [1, CYAN + 1, OFFSET + N, OFFSET + N])
    red_st = _i64(inits, "s2", [0, RED, OFFSET, OFFSET])
    red_en = _i64(inits, "e2", [1, RED + 1, OFFSET + RED_ROWS, OFFSET + N])

    _init(inits, "rii", np.arange(N, dtype=np.int64).reshape(1, 1, N, 1))
    _init(inits, "cii", np.arange(N, dtype=np.int64).reshape(1, 1, 1, N))
    false_left = _bool(inits, "false_left", np.zeros((1, 1, N, OFFSET), dtype=bool))
    false_right = _bool(
        inits,
        "false_right",
        np.zeros((1, 1, N, W - OFFSET - N), dtype=bool),
    )
    false_top = _bool(inits, "false_top", np.zeros((1, 1, OFFSET, W), dtype=bool))
    false_bottom = _bool(
        inits,
        "false_bottom",
        np.zeros((1, 1, H - OFFSET - N, W), dtype=bool),
    )
    yellow = _f32(
        inits,
        "yellow",
        np.eye(1, C, PATH, dtype=np.float32).reshape(1, C, 1, 1),
    )

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, cyan_st, cyan_en, axes4], ["cyan_f"]),
            helper.make_node("Slice", [IN_NAME, red_st, red_en, axes4], ["red_f"]),
            helper.make_node("ReduceMax", ["cyan_f"], ["cyan_row_f"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["cyan_f"], ["cyan_col_f"], axes=[2], keepdims=1),
            helper.make_node("ReduceMax", ["red_f"], ["red_row_f"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["red_f"], ["red_col_f"], axes=[2], keepdims=1),
            helper.make_node("ArgMax", ["cyan_row_f"], ["cyan_row_i"], axis=2, keepdims=1),
            helper.make_node("ArgMax", ["red_row_f"], ["red_row_i"], axis=2, keepdims=1),
            helper.make_node("ArgMax", ["cyan_col_f"], ["cyan_col_i"], axis=3, keepdims=1),
            helper.make_node("ArgMax", ["red_col_f"], ["red_col_i"], axis=3, keepdims=1),
            helper.make_node("Equal", ["rii", "cyan_row_i"], ["cyan_row_m"]),
            helper.make_node("Equal", ["cii", "cyan_col_i"], ["cyan_col_m"]),
            helper.make_node("Equal", ["rii", "red_row_i"], ["red_row_m"]),
            helper.make_node("Equal", ["cii", "red_col_i"], ["red_col_m"]),
            helper.make_node("Less", ["cyan_row_i", "red_row_i"], ["cyan_above"]),
            helper.make_node("Where", ["cyan_above", "cyan_row_i", "red_row_i"], ["row_min"]),
            helper.make_node("Where", ["cyan_above", "red_row_i", "cyan_row_i"], ["row_max"]),
            helper.make_node("Less", ["cyan_col_i", "red_col_i"], ["cyan_left"]),
            helper.make_node("Where", ["cyan_left", "cyan_col_i", "red_col_i"], ["col_min"]),
            helper.make_node("Where", ["cyan_left", "red_col_i", "cyan_col_i"], ["col_max"]),
            helper.make_node("Less", ["rii", "row_min"], ["row_before"]),
            helper.make_node("Greater", ["rii", "row_max"], ["row_after"]),
            helper.make_node("Or", ["row_before", "row_after"], ["row_out"]),
            helper.make_node("Not", ["row_out"], ["row_range"]),
            helper.make_node("Less", ["cii", "col_min"], ["col_before"]),
            helper.make_node("Greater", ["cii", "col_max"], ["col_after"]),
            helper.make_node("Or", ["col_before", "col_after"], ["col_out"]),
            helper.make_node("Not", ["col_out"], ["col_range"]),
            helper.make_node("Not", ["cyan_row_m"], ["not_cyan_row"]),
            helper.make_node("Not", ["red_col_m"], ["not_red_col"]),
            helper.make_node("And", ["row_range", "not_cyan_row"], ["row_range_inner"]),
            helper.make_node("And", ["col_range", "not_red_col"], ["col_range_inner"]),
            helper.make_node("And", ["row_range_inner", "cyan_col_m"], ["vertical_inner"]),
            helper.make_node("And", ["red_row_m", "col_range_inner"], ["horizontal_inner"]),
            helper.make_node("Or", ["vertical_inner", "horizontal_inner"], ["path_inner"]),
            helper.make_node(
                "Concat",
                ["false_left", "path_inner", "false_right"],
                ["path17x30"],
                axis=3,
            ),
            helper.make_node(
                "Concat",
                ["false_top", "path17x30", "false_bottom"],
                ["path30"],
                axis=2,
            ),
            helper.make_node("Where", ["path30", yellow, IN_NAME], [OUT_NAME]),
        ]
    )

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


def _onehot(grid: list[list[int]]) -> np.ndarray:
    arr = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            arr[0, int(color), r, c] = 1.0
    return arr


def validate(path: Path) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            expected_grid = np.asarray(example["output"], dtype=np.int64)
            ref = solve(np.asarray(example["input"], dtype=np.int64))
            if not np.array_equal(ref, expected_grid):
                raise AssertionError(f"reference mismatch on {split}[{idx}]")
            pred = session.run([OUT_NAME], {IN_NAME: _onehot(example["input"])})[0]
            expected = _onehot(example["output"])
            if not np.array_equal(pred > 0.0, expected > 0.0):
                raise AssertionError(f"ONNX mismatch on {split}[{idx}]")
            total += 1
    return total


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    total = validate(BEST_PATH)
    result = score_file(BEST_PATH)

    print(f"verified {total} examples")
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
