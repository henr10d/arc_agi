"""ONNX solution for ARC task371: add a green plus between two blue cells.

Task rule: each grid contains exactly two blue cells (color 1) aligned on the
same row or the same column. Preserve those blue cells and draw a green
5-cell plus (color 3) centered at the integer midpoint between them. In the
local task data the endpoints are at least 6 cells apart, so the plus never
overlaps a blue endpoint or clips against the grid bounds. The output grid has
the same dimensions as the input.

ONNX approach: all examples fit within a 14x14 crop. The graph computes the
midpoint from weighted row/column sums of the blue mask, creates one-hot masks
for the center row/column and neighboring rows/columns, then uses a final
opset-10 Scatter into the original input tensor to update only channel 0 and
channel 3 over the crop. Because the Scatter output is the graph output, the
full 30x30 result is excluded from NeuroGolf memory scoring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "task371"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task371.onnx"

C = 10
OH = OW = 30
N = 14
BLUE = 1
GREEN = 3
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, OH, OW]
OPSET = 10
IR_VERSION = 10


def solve(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    """Reference solver for JSON grids."""
    x = np.asarray(grid, dtype=np.int64)
    out = x.copy()
    blue = np.argwhere(x == BLUE)
    if len(blue) != 2:
        raise ValueError("expected exactly two blue cells")
    mr, mc = ((blue[0] + blue[1]) // 2).astype(int)
    for r, c in ((mr, mc), (mr - 1, mc), (mr + 1, mc), (mr, mc - 1), (mr, mc + 1)):
        if 0 <= r < x.shape[0] and 0 <= c < x.shape[1] and x[r, c] != BLUE:
            out[r, c] = GREEN
    return out


def _arr(inits: list[onnx.TensorProto], values: Any, name: str, dtype: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(values, dtype=dtype), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], values: Any, name: str) -> str:
    return _arr(inits, values, name, np.int64)


def _i32(inits: list[onnx.TensorProto], values: Any, name: str) -> str:
    return _arr(inits, values, name, np.int32)


def _f32(inits: list[onnx.TensorProto], values: Any, name: str) -> str:
    return _arr(inits, values, name, np.float32)


def _grid_to_onehot(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(arr):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def _decode(onehot: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return (onehot[0, :, :h, :w] > 0.0).argmax(axis=0).astype(np.int64)


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    black_start = _i64(inits, [0, 0, 0, 0], "black_start")
    black_end = _i64(inits, [1, 1, N, N], "black_end")
    blue_start = _i64(inits, [0, BLUE, 0, 0], "blue_start")
    blue_end = _i64(inits, [1, BLUE + 1, N, N], "blue_end")
    half = _f32(inits, [2.0], "two")
    one_i = _i32(inits, [1], "one_i")
    row_ids = _f32(inits, np.arange(N, dtype=np.float32).reshape(1, 1, N, 1), "row_ids")
    col_ids = _f32(inits, np.arange(N, dtype=np.float32).reshape(1, 1, 1, N), "col_ids")
    row_ids_i = _i32(inits, np.arange(N, dtype=np.int32).reshape(1, 1, N, 1), "row_ids_i")
    col_ids_i = _i32(inits, np.arange(N, dtype=np.int32).reshape(1, 1, 1, N), "col_ids_i")
    scatter_idx = np.zeros((1, 2, N, N), dtype=np.int64)
    scatter_idx[:, 1, :, :] = GREEN
    scatter_idx_name = _i64(inits, scatter_idx, "scatter_idx")

    def add(op: str, ins: list[str], outs: list[str], **kwargs: Any) -> None:
        nodes.append(helper.make_node(op, ins, outs, **kwargs))

    add("Slice", [IN_NAME, black_start, black_end], ["black_f"])
    add("Slice", [IN_NAME, blue_start, blue_end], ["blue_f"])

    add("ReduceSum", ["blue_f"], ["blue_rows"], axes=[3], keepdims=1)
    add("ReduceSum", ["blue_f"], ["blue_cols"], axes=[2], keepdims=1)
    add("Mul", ["blue_rows", row_ids], ["blue_rows_weighted"])
    add("Mul", ["blue_cols", col_ids], ["blue_cols_weighted"])
    add("ReduceSum", ["blue_rows_weighted"], ["row_sum"], axes=[2, 3], keepdims=0)
    add("ReduceSum", ["blue_cols_weighted"], ["col_sum"], axes=[2, 3], keepdims=0)
    add("Div", ["row_sum", half], ["mid_row_f"])
    add("Div", ["col_sum", half], ["mid_col_f"])
    add("Cast", ["mid_row_f"], ["mid_row_i"], to=TensorProto.INT32)
    add("Cast", ["mid_col_f"], ["mid_col_i"], to=TensorProto.INT32)

    add("Unsqueeze", ["mid_row_i"], ["mid_row_idx"], axes=[2, 3])
    add("Unsqueeze", ["mid_col_i"], ["mid_col_idx"], axes=[2, 3])
    add("Sub", ["mid_row_idx", one_i], ["mid_row_prev_idx"])
    add("Add", ["mid_row_idx", one_i], ["mid_row_next_idx"])
    add("Sub", ["mid_col_idx", one_i], ["mid_col_prev_idx"])
    add("Add", ["mid_col_idx", one_i], ["mid_col_next_idx"])

    add("Equal", [row_ids_i, "mid_row_idx"], ["row_mid"])
    add("Equal", [row_ids_i, "mid_row_prev_idx"], ["row_prev"])
    add("Equal", [row_ids_i, "mid_row_next_idx"], ["row_next"])
    add("Equal", [col_ids_i, "mid_col_idx"], ["col_mid"])
    add("Equal", [col_ids_i, "mid_col_prev_idx"], ["col_prev"])
    add("Equal", [col_ids_i, "mid_col_next_idx"], ["col_next"])

    add("Or", ["row_prev", "row_mid"], ["rows_pm"])
    add("Or", ["rows_pm", "row_next"], ["rows3"])
    add("Or", ["col_prev", "col_mid"], ["cols_pm"])
    add("Or", ["cols_pm", "col_next"], ["cols3"])
    add("And", ["rows3", "col_mid"], ["vertical_arm"])
    add("And", ["row_mid", "cols3"], ["horizontal_arm"])
    add("Or", ["vertical_arm", "horizontal_arm"], ["plus"])
    add("Cast", ["plus"], ["plus_f"], to=TensorProto.FLOAT)
    add("Sub", ["black_f", "plus_f"], ["bg_update"])
    add("Concat", ["bg_update", "plus_f"], ["scatter_updates"], axis=1)
    add("Scatter", [IN_NAME, scatter_idx_name, "scatter_updates"], [OUT_NAME], axis=1)

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


def save_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    model = build_model()
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return model


def validate_rule() -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    checked = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            pred = solve(ex["input"])
            expected = np.asarray(ex["output"], dtype=np.int64)
            if not np.array_equal(pred, expected):
                raise AssertionError(f"rule mismatch for {split}#{idx}")
            checked += 1
    return checked


def validate_model(model: onnx.ModelProto, examples: Iterable[dict[str, Any]]) -> int:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    checked = 0
    for ex in examples:
        inp = np.asarray(ex["input"], dtype=np.int64)
        expected = _grid_to_onehot(ex["output"])
        pred = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(inp)})[0]
        if not np.array_equal(pred > 0.0, expected > 0.0):
            decoded = _decode(pred, np.asarray(ex["output"]).shape)
            raise AssertionError(
                f"prediction mismatch after {checked} examples:\n"
                f"pred={decoded.tolist()}\nexpected={ex['output']}"
            )
        checked += 1
    return checked


def main() -> None:
    validate_rule()
    model = save_model()
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    examples = [ex for split in ("train", "test", "arc-gen") for ex in data[split]]
    checked = validate_model(model, examples)
    print(f"saved {BEST_PATH} ({checked} examples validated)")


if __name__ == "__main__":
    main()
