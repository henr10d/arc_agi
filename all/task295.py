"""ONNX for ARC task295: grow a one-row colored prefix downward.

Task rule: the input is a single row with a contiguous non-black prefix of
length k at column 0, followed by black cells up to width W.  The output keeps
that width and has W/2 rows in the local JSON data; row r is filled with the
same non-black color through columns 0..k+r-1, with the rest black.  Padding
outside the output rectangle remains all-zero for the competition I/O.

ONNX approach: infer W from active cells in input row 0 and k from non-black
cells in that row, build compact coordinate comparisons for the growing prefix
mask, then broadcast only the selected input color into the non-black channels.
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

TASK_ID = "task295"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task295.onnx"
DATA_PATH = ROOT / "data" / "task295.json"

C = 10
H = W = 30
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


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


def load_examples() -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    examples: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            examples.append(
                (
                    split,
                    idx,
                    np.asarray(ex["input"], dtype=np.int64),
                    np.asarray(ex["output"], dtype=np.int64),
                )
            )
    return examples


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver matching the JSON examples."""
    g = np.asarray(grid, dtype=np.int64)
    assert g.shape[0] == 1
    width = g.shape[1]
    nonblack = np.flatnonzero(g[0] != 0)
    assert nonblack.size > 0
    k = int(nonblack.size)
    color = int(g[0, 0])
    out = np.zeros((width // 2, width), dtype=np.int64)
    for r in range(out.shape[0]):
        out[r, : k + r] = color
    return out


def _grid_to_onehot(grid: np.ndarray) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            out[0, int(grid[r, c]), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    active = onehot > 0.0
    return active.argmax(axis=1)[0, : shape[0], : shape[1]].astype(np.int64)


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    ch_start = _i64(inits, [1], "ch_start")
    ch_end = _i64(inits, [10], "ch_end")
    ch_axis = _i64(inits, [1], "ch_axis")
    row0_start = _i64(inits, [0], "row0_start")
    row1_end = _i64(inits, [1], "row1_end")
    row_axis = _i64(inits, [2], "row_axis")
    col0_start = _i64(inits, [0], "col0_start")
    col1_end = _i64(inits, [1], "col1_end")
    col_axis = _i64(inits, [3], "col_axis")
    two = _f32(inits, [2.0], "two")
    rows = _f32(inits, np.arange(H, dtype=np.float32).reshape(1, 1, H, 1), "rows")
    cols = _f32(inits, np.arange(W, dtype=np.float32).reshape(1, 1, 1, W), "cols")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, row0_start, row1_end, row_axis], ["row0_in"]),
            helper.make_node("ReduceSum", ["row0_in"], ["row0_any"], axes=[1], keepdims=1),
            helper.make_node("ReduceSum", ["row0_any"], ["width"], axes=[3], keepdims=1),
            helper.make_node("Slice", ["row0_in", ch_start, ch_end, ch_axis], ["nonblack_row0"]),
            helper.make_node("ReduceSum", ["nonblack_row0"], ["prefix_len"], axes=[1, 3], keepdims=1),
            helper.make_node("Div", ["width", two], ["out_height"]),
            helper.make_node("Less", [rows, "out_height"], ["row_in_out"]),
            helper.make_node("Less", [cols, "width"], ["col_in_out"]),
            helper.make_node("Add", ["prefix_len", rows], ["limit"]),
            helper.make_node("Less", [cols, "limit"], ["col_in_color"]),
            helper.make_node("And", ["row_in_out", "col_in_out"], ["out_area"]),
            helper.make_node("And", ["out_area", "col_in_color"], ["color_mask_b"]),
            helper.make_node("Not", ["color_mask_b"], ["not_color_b"]),
            helper.make_node("And", ["out_area", "not_color_b"], ["black_mask_b"]),
            helper.make_node("Cast", ["black_mask_b"], ["black"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["color_mask_b"], ["color_mask"], to=TensorProto.FLOAT),
            helper.make_node("Slice", ["nonblack_row0", col0_start, col1_end, col_axis], ["color_vec"]),
            helper.make_node("Mul", ["color_vec", "color_mask"], ["colored"]),
            helper.make_node("Concat", ["black", "colored"], [OUT_NAME], axis=1),
        ]
    )
    return _make_model(nodes, inits)


def validate_model(model: onnx.ModelProto, examples: list[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    for split, idx, inp, expected in examples:
        solved = solve(inp)
        assert np.array_equal(solved, expected), (split, idx, solved, expected)
        pred = sess.run([OUT_NAME], {IN_NAME: _grid_to_onehot(inp)})[0]
        got = _onehot_to_grid(pred, expected.shape)
        if not np.array_equal(got, expected):
            raise AssertionError((split, idx, got, expected))


def main() -> None:
    examples = load_examples()
    model = build_model()
    validate_model(model, examples)
    onnx.save(model, BEST_PATH)
    result = score_file(BEST_PATH)
    print(
        f"{BEST_PATH.name}: valid={result['valid']} memory={result['memory']} "
        f"params={result['params']} cost={result['cost']} score={result['score']}"
    )
    if not result["valid"]:
        raise SystemExit(result["error"])


if __name__ == "__main__":
    main()
