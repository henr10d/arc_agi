"""ONNX solution for NeuroGolf task388: tiled columns with cyan fill.

Task rule: the input is an NxN grid, with N between 2 and 6 in the local task
data. The output is 2N by 2N. Tile the input twice vertically and horizontally.
For every original input column that contains at least one non-black cell,
replace tiled black cells in the corresponding output columns with cyan
(color 8); tiled non-black cells keep their original color. Columns that were
empty in the input remain black except for any tiled foreground already there.
All official local examples use a single non-black foreground color per grid,
and no input foreground uses color 8.

ONNX approach: infer N from the valid one-hot rows, gather N-specific row and
column modulo maps for the 12x12 maximum output, tile compact foreground and
valid masks, broadcast the single foreground color vector only where needed,
then cast the compact 12x12 bool output to float before padding to 30x30.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task388"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task388.onnx"

C = 10
H = W = 30
MAX_N = 6
MAX_OUT = MAX_N * 2
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def solve(grid: Sequence[Sequence[int]]) -> np.ndarray:
    """Reference implementation on raw integer grids."""
    arr = np.asarray(grid, dtype=np.int64)
    n = arr.shape[0]
    active_cols = np.any(arr != 0, axis=0)
    out = np.zeros((2 * n, 2 * n), dtype=np.int64)
    for r in range(2 * n):
        for c in range(2 * n):
            base = arr[r % n, c % n]
            if base != 0:
                out[r, c] = base
            elif active_cols[c % n]:
                out[r, c] = 8
    return out


def _index_maps() -> np.ndarray:
    maps = np.zeros((MAX_N - 1, MAX_OUT), dtype=np.int32)
    for size in range(2, MAX_N + 1):
        layer = size - 2
        for r in range(MAX_OUT):
            if r < 2 * size:
                maps[layer, r] = r % size
            else:
                maps[layer, r] = 0 if size == MAX_N else size
    return maps


def build_model() -> onnx.ModelProto:
    """Build the compact opset-10 ONNX graph."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    st0 = _init(inits, np.asarray([0, 0, 0, 0], dtype=np.int64), "st0")
    end6 = _init(inits, np.asarray([1, C, MAX_N, MAX_N], dtype=np.int64), "end6")
    end_ch0_6 = _init(inits, np.asarray([1, 1, MAX_N, MAX_N], dtype=np.int64), "end_ch0_6")
    present_ch1_start = _init(inits, np.asarray([0, 1, 0, 0], dtype=np.int64), "present_ch1_start")
    present_ch1_7_end = _init(inits, np.asarray([1, 8, 1, 1], dtype=np.int64), "present_ch1_7_end")
    present_ch9_start = _init(inits, np.asarray([0, 9, 0, 0], dtype=np.int64), "present_ch9_start")
    present_end = _init(inits, np.asarray([1, C, 1, 1], dtype=np.int64), "present_end")
    two_i = _init(inits, np.asarray(2, dtype=np.int64), "two_i")
    zero_f = _init(inits, np.asarray(0.0, dtype=np.float32), "zero_f")

    _init(inits, _index_maps(), "idx_maps")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st0, end6], ["x6"]),
            helper.make_node("ReduceMax", ["x6"], ["valid6"], axes=[1], keepdims=1),
            helper.make_node("ReduceMax", ["valid6"], ["row_valid"], axes=[3], keepdims=1),
            helper.make_node("ReduceSum", ["row_valid"], ["n_f"], axes=[0, 1, 2, 3], keepdims=0),
            helper.make_node("Cast", ["n_f"], ["n_i"], to=TensorProto.INT64),
            helper.make_node("Sub", ["n_i", two_i], ["size_idx"]),
            helper.make_node("Gather", ["idx_maps", "size_idx"], ["idx"], axis=0),
            helper.make_node("Slice", ["x6", st0, end_ch0_6], ["ch0_6"]),
            helper.make_node("Less", ["ch0_6", "valid6"], ["fg6b"]),
            helper.make_node("Cast", ["fg6b"], ["fg6"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", ["fg6"], ["col_fg"], axes=[2], keepdims=1),
            helper.make_node("Gather", ["col_fg", "idx"], ["active_cols"], axis=3),
            helper.make_node("Greater", ["active_cols", zero_f], ["active_col_b"]),
            helper.make_node("Gather", ["fg6b", "idx"], ["base_fg_rows"], axis=2),
            helper.make_node("Gather", ["base_fg_rows", "idx"], ["base_fg"], axis=3),
            helper.make_node("Greater", ["row_valid", zero_f], ["row_valid_b"]),
            helper.make_node("Gather", ["row_valid_b", "idx"], ["valid_rows"], axis=2),
            helper.make_node("ReduceMax", ["valid6"], ["col_valid"], axes=[2], keepdims=1),
            helper.make_node("Greater", ["col_valid", zero_f], ["col_valid_b"]),
            helper.make_node("Gather", ["col_valid_b", "idx"], ["valid_cols"], axis=3),
            helper.make_node("And", ["valid_rows", "valid_cols"], ["base_valid"]),
            helper.make_node("Not", ["base_fg"], ["no_fg"]),
            helper.make_node("And", ["no_fg", "base_valid"], ["base_black"]),
            helper.make_node("And", ["active_col_b", "base_black"], ["fill"]),
            helper.make_node("Not", ["active_col_b"], ["inactive_col_b"]),
            helper.make_node("And", ["base_black", "inactive_col_b"], ["out0"]),
            helper.make_node("ReduceMax", ["x6"], ["present_f"], axes=[2, 3], keepdims=1),
            helper.make_node("Greater", ["present_f", zero_f], ["present_b"]),
            helper.make_node("Slice", ["present_b", present_ch1_start, present_ch1_7_end], ["color1_7"]),
            helper.make_node("And", ["color1_7", "base_fg"], ["out1_7"]),
            helper.make_node("Slice", ["present_b", present_ch9_start, present_end], ["color9"]),
            helper.make_node("And", ["color9", "base_fg"], ["out9"]),
            helper.make_node("Concat", ["out0", "out1_7", "fill", "out9"], ["out12b"], axis=1),
            helper.make_node("Cast", ["out12b"], ["out12"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out12"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - MAX_OUT, W - MAX_OUT],
            ),
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
    onnx.checker.check_model(model)
    return model


def _grid_to_onehot(grid: Sequence[Sequence[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def _load_task() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def validate_model(path: Path) -> None:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    task = _load_task()
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(task.get(split, [])):
            expected_grid = solve(example["input"])
            json_expected = np.asarray(example["output"], dtype=np.int64)
            if not np.array_equal(expected_grid, json_expected):
                raise AssertionError(f"reference mismatch on {split}[{idx}]")

            expected = _grid_to_onehot(example["output"])
            actual = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(example["input"])})[0]
            if not np.array_equal(actual > 0.0, expected > 0.0):
                raise AssertionError(f"ONNX mismatch on {split}[{idx}]")


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    validate_model(BEST_PATH)

    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"{TASK_ID}.json: PASS")
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
