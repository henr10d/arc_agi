"""ONNX for ARC task199: drop the lone colored cell one row under a striped canopy.

Task rule: each square input grid contains exactly one non-black, non-yellow
cell.  Move that cell down by one row, preserving its color.  Fill rows 0
through the cell's original row with color 4 in every other column whose parity
matches the cell's column, leaving all other in-grid cells black.  All task
grids are square and no larger than 15x15, so the graph computes only that
top-left region and pads the competition output to 30x30.
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
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task199"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
N = 15
PAD = H - N
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def solve(grid: np.ndarray | List[List[int]]) -> np.ndarray:
    """Reference implementation for the ARC transformation."""
    g = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(g)
    cells = np.argwhere(g != 0)
    if len(cells) != 1:
        return out
    r, c = map(int, cells[0])
    out[: r + 1, c % 2 :: 2] = 4
    if r + 1 < g.shape[0]:
        out[r + 1, c] = g[r, c]
    return out


def _arr(inits: List[onnx.TensorProto], value, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(value), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], value: List[int], name: str) -> str:
    return _arr(inits, np.asarray(value, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], value, name: str) -> str:
    return _arr(inits, np.asarray(value, dtype=np.float32), name)


def _bool(inits: List[onnx.TensorProto], value, name: str) -> str:
    return _arr(inits, np.asarray(value, dtype=bool), name)


def build_onnx_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    ax4 = _i64(inits, [0, 1, 2, 3], "ax4")
    ax2 = _i64(inits, [0, 1], "ax2")
    s0 = _i64(inits, [0, 0, 0, 0], "s0")
    e_ch0 = _i64(inits, [1, 1, N, N], "ech0")
    s_present = _i64(inits, [0, 1], "spresent")
    e_present = _i64(inits, [1, 10], "epresent")
    e_col0 = _i64(inits, [1, 1, N, 1], "ecol0")
    s_col1 = _i64(inits, [0, 0, 0, 1], "scol1")
    e_col1 = _i64(inits, [1, 1, N, 2], "ecol1")
    e_row0 = _i64(inits, [1, 1, 1, N], "erow0")
    s_row1 = _i64(inits, [0, 0, 1, 0], "srow1")
    e_row1 = _i64(inits, [1, 1, 2, N], "erow1")
    ax_row = _i64(inits, [2], "axr")
    s_row = _i64(inits, [0], "sr")
    e_row = _i64(inits, [N - 1], "er")
    one_i = _i64(inits, [1], "one_i")
    two_i = _i64(inits, [2], "two_i")
    _i64(inits, list(range(N)), "row_idx")
    _i64(inits, (np.arange(N) % 2).reshape(1, 1, 1, N).tolist(), "col_parity")
    _i64(inits, np.array([1, 2, 3], dtype=np.int64).reshape(1, 3, 1, 1), "ids123")
    _i64(inits, np.array([5, 6, 7, 8, 9], dtype=np.int64).reshape(1, 5, 1, 1), "ids56789")
    _bool(inits, np.zeros((1, 1, 1, N), dtype=bool), "false_pos_row")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, s0, e_ch0, ax4], ["ch0"]),
            helper.make_node("Cast", ["ch0"], ["bg_in"], to=TensorProto.BOOL),
            helper.make_node("Slice", ["bg_in", s0, e_col0, ax4], ["valid_rows_0"]),
            helper.make_node("Slice", ["bg_in", s_col1, e_col1, ax4], ["valid_rows_1"]),
            helper.make_node("Or", ["valid_rows_0", "valid_rows_1"], ["valid_rows"]),
            helper.make_node("Slice", ["bg_in", s0, e_row0, ax4], ["valid_cols_0"]),
            helper.make_node("Slice", ["bg_in", s_row1, e_row1, ax4], ["valid_cols_1"]),
            helper.make_node("Or", ["valid_cols_0", "valid_cols_1"], ["valid_cols"]),
            helper.make_node("And", ["valid_rows", "valid_cols"], ["valid"]),
            helper.make_node("Not", ["bg_in"], ["not_bg_in"]),
            helper.make_node("And", ["valid", "not_bg_in"], ["marker_pos_b"]),
            helper.make_node("Cast", ["marker_pos_b"], ["marker_pos_u8"], to=TensorProto.UINT8),
            helper.make_node("ArgMax", ["marker_pos_u8"], ["row_per_col"], axis=2, keepdims=1),
            helper.make_node("ReduceMax", ["row_per_col"], ["marker_row"], axes=[3], keepdims=0),
            helper.make_node("Add", ["marker_row", one_i], ["marker_row_plus"]),
            helper.make_node("Less", ["row_idx", "marker_row_plus"], ["stripe_rows_sq"]),
            helper.make_node("Unsqueeze", ["stripe_rows_sq"], ["stripe_rows"], axes=[3]),
            helper.make_node("ArgMax", ["marker_pos_u8"], ["col_per_row"], axis=3, keepdims=1),
            helper.make_node("ReduceMax", ["col_per_row"], ["marker_col"], axes=[2], keepdims=1),
            helper.make_node("Mod", ["marker_col", two_i], ["marker_parity"]),
            helper.make_node("Equal", ["marker_parity", "col_parity"], ["stripe_cols"]),
            helper.make_node("And", ["stripe_rows", "stripe_cols"], ["stripe_rc"]),
            helper.make_node("And", ["stripe_rc", "valid"], ["stripe"]),
            helper.make_node("Slice", ["marker_pos_b", s_row, e_row, ax_row], ["pos_top"]),
            helper.make_node("Concat", ["false_pos_row", "pos_top"], ["moved_pos"], axis=2),
            helper.make_node("ReduceMax", [IN_NAME], ["present10"], axes=[2, 3], keepdims=0),
            helper.make_node("Slice", ["present10", s_present, e_present, ax2], ["present9"]),
            helper.make_node("ArgMax", ["present9"], ["marker_color0"], axis=1, keepdims=1),
            helper.make_node("Add", ["marker_color0", one_i], ["marker_color2"]),
            helper.make_node("Unsqueeze", ["marker_color2"], ["marker_color"], axes=[2, 3]),
            helper.make_node("Equal", ["marker_color", "ids123"], ["is123"]),
            helper.make_node("And", ["moved_pos", "is123"], ["mv123"]),
            helper.make_node("Equal", ["marker_color", "ids56789"], ["is56789"]),
            helper.make_node("And", ["moved_pos", "is56789"], ["mv56789"]),
            helper.make_node("Or", ["stripe", "moved_pos"], ["occupied"]),
            helper.make_node("Not", ["occupied"], ["empty"]),
            helper.make_node("And", ["valid", "empty"], ["bg"]),
            helper.make_node("Concat", ["bg", "mv123", "stripe", "mv56789"], ["out15b"], axis=1),
            helper.make_node("Cast", ["out15b"], ["out15"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out15"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]),
        ]
    )

    graph = helper.make_graph(nodes, "task199", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _grid_to_onehot(grid: List[List[int]]) -> np.ndarray:
    out = np.zeros((1, C, H, W), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def _verify(path: Path) -> None:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            pred = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(ex["input"])})[0]
            expected = _grid_to_onehot(ex["output"])
            if not np.array_equal(pred > 0.0, expected > 0.0):
                raise AssertionError(f"{split}[{idx}] failed")


def main() -> None:
    model = build_onnx_model()
    onnx.save(model, BEST_PATH)
    _verify(BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"saved {BEST_PATH}")
    print(f"memory={result['memory']} params={result['params']} cost={result['cost']} score={result['score']:.6f}")


if __name__ == "__main__":
    main()
