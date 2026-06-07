"""ONNX for ARC task342: move quadrant marker colors into the cyan block.

Task rule: each 10x10 input contains one cyan 2x2 block (color 8) and
exactly four isolated non-black, non-cyan marker cells. The output is black
except at the former cyan block. The marker above-left of the block is copied
to the block's top-left cell, above-right to top-right, below-left to
bottom-left, and below-right to bottom-right.

ONNX: slice the 10x10 non-black planes, cast them to float16, and use row-major
ArgMax on the cyan plane to get the cyan block's top-left cell. Gather tiny
row/column tables for the four marker quadrants and the four destination cells,
reduce the masked marker colors to four one-hot color vectors, place them with
a compact MatMul, add the black background channel, then cast/pad to the
required 30x30 NeuroGolf output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task342"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
G = 10
LOC = G - 1
PAD = H - G
CYAN = 8
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def solve(grid: Sequence[Sequence[int]]) -> np.ndarray:
    """Reference NumPy solver for task342."""
    x = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(x)
    cyan_cells = np.argwhere(x == CYAN)
    if cyan_cells.size == 0:
        return out
    r = int(cyan_cells[:, 0].min())
    c = int(cyan_cells[:, 1].min())

    for rr, cc in np.argwhere((x != 0) & (x != CYAN)):
        rr_i, cc_i = int(rr), int(cc)
        color = int(x[rr_i, cc_i])
        if rr_i < r and cc_i < c:
            out[r, c] = color
        elif rr_i < r and cc_i > c + 1:
            out[r, c + 1] = color
        elif rr_i > r + 1 and cc_i < c:
            out[r + 1, c] = color
        else:
            out[r + 1, c + 1] = color
    return out


def _i64(vals: Sequence[int], name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name)


def _i32(vals: Sequence[int], name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(vals, dtype=np.int32), name=name)


def _f32(arr: np.ndarray | Sequence[float], name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name)


def _f16(arr: np.ndarray | Sequence[float], name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr, dtype=np.float16), name=name)


def _row_col_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return float16 row/column masks indexed by cyan top-left row/column."""
    row_quad = np.zeros((G, G, 2), dtype=np.float16)
    col_quad = np.zeros((G, G, 2), dtype=np.float16)
    row_cell = np.zeros((G, G, 2), dtype=np.float16)
    col_cell = np.zeros((G, 2, G), dtype=np.float16)
    for start in range(G):
        for idx in range(G):
            row_quad[start, idx, 0] = 1.0 if idx < start else 0.0
            row_quad[start, idx, 1] = 1.0 if idx > start + 1 else 0.0
            col_quad[start, idx, 0] = 1.0 if idx < start else 0.0
            col_quad[start, idx, 1] = 1.0 if idx > start + 1 else 0.0
            row_cell[start, idx, 0] = 1.0 if idx == start else 0.0
            row_cell[start, idx, 1] = 1.0 if idx == start + 1 else 0.0
            col_cell[start, 0, idx] = 1.0 if idx == start else 0.0
            col_cell[start, 1, idx] = 1.0 if idx == start + 1 else 0.0
    return row_quad, col_quad, row_cell, col_cell


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    inits.extend(
        [
            _i64([0, 1, 2, 3], "axes4"),
            _i64([0, 0, 0, 0], "core_st"),
            _i64([1, C, G, G], "core_en"),
            _i64([1, G * G], "shape_1x100"),
            _i64([1, 4], "shape_color_ids"),
            _i64([1, C - 1, 2, 2], "shape_colors22"),
            _i64([1], "one_i"),
            _i64([C - 1], "depth9"),
            _f32(np.arange(C, dtype=np.float32).reshape(1, C, 1, 1), "color_w"),
            _f32([CYAN - 0.5], "cyan_lo"),
            _f32([CYAN + 0.5], "cyan_hi"),
            _f16([1.0], "one_h"),
            _f16([0.0, 1.0], "onehot_values_h"),
            _i32([r for r in range(G) for _ in range(G)], "row_for_flat"),
            _i32([c for _ in range(G) for c in range(G)], "col_for_flat"),
        ]
    )
    row_quad, col_quad, row_cell, col_cell = _row_col_tables()
    inits.extend(
        [
            _f16(row_quad, "row_quad"),
            _f16(col_quad, "col_quad"),
            _f16(row_cell, "row_cell"),
            _f16(col_cell, "col_cell"),
        ]
    )

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, "core_st", "core_en", "axes4"], ["core"]),
            helper.make_node("Conv", ["core", "color_w"], ["color_f"]),
            helper.make_node("Greater", ["color_f", "cyan_lo"], ["cyan_ge"]),
            helper.make_node("Less", ["color_f", "cyan_hi"], ["cyan_le"]),
            helper.make_node("And", ["cyan_ge", "cyan_le"], ["cyan_b"]),
            helper.make_node("Cast", ["cyan_b"], ["cyan_h"], to=TensorProto.FLOAT16),
            helper.make_node("Reshape", ["cyan_h", "shape_1x100"], ["cyan_flat"]),
            helper.make_node("ArgMax", ["cyan_flat"], ["flat_idx"], axis=1, keepdims=0),
            helper.make_node("Gather", ["row_for_flat", "flat_idx"], ["row_idx"], axis=0),
            helper.make_node("Gather", ["col_for_flat", "flat_idx"], ["col_idx"], axis=0),
            helper.make_node("Gather", ["row_quad", "row_idx"], ["row_q"], axis=0),
            helper.make_node("Gather", ["col_quad", "col_idx"], ["col_q"], axis=0),
            helper.make_node("Unsqueeze", ["col_q"], ["col_q4"], axes=[1]),
            helper.make_node("Cast", ["color_f"], ["ids_h"], to=TensorProto.FLOAT16),
            helper.make_node("MatMul", ["ids_h", "col_q4"], ["col_sum"]),
            helper.make_node("Transpose", ["col_sum"], ["col_sum_t"], perm=[0, 1, 3, 2]),
            helper.make_node("Unsqueeze", ["row_q"], ["row_q4"], axes=[1]),
            helper.make_node("MatMul", ["col_sum_t", "row_q4"], ["colors_cr"]),
            helper.make_node("Transpose", ["colors_cr"], ["colors_rc"], perm=[0, 1, 3, 2]),
            helper.make_node("Reshape", ["colors_rc", "shape_color_ids"], ["color_ids_h"]),
            helper.make_node("Cast", ["color_ids_h"], ["color_ids_i"], to=TensorProto.INT64),
            helper.make_node("Sub", ["color_ids_i", "one_i"], ["color_ids0"]),
            helper.make_node(
                "OneHot",
                ["color_ids0", "depth9", "onehot_values_h"],
                ["colors"],
                axis=1,
            ),
            helper.make_node("Reshape", ["colors", "shape_colors22"], ["colors22"]),
            helper.make_node("Gather", ["row_cell", "row_idx"], ["row_p"], axis=0),
            helper.make_node("Gather", ["col_cell", "col_idx"], ["col_p"], axis=0),
            helper.make_node("Unsqueeze", ["row_p"], ["row_p4"], axes=[1]),
            helper.make_node("Unsqueeze", ["col_p"], ["col_p4"], axes=[1]),
            helper.make_node("MatMul", ["row_p4", "colors22"], ["placed_cols"]),
            helper.make_node("MatMul", ["placed_cols", "col_p4"], ["fg10_h"]),
            helper.make_node("Sub", ["one_h", "cyan_h"], ["bg_h"]),
            helper.make_node("Concat", ["bg_h", "fg10_h"], ["out10_h"], axis=1),
            helper.make_node("Cast", ["out10_h"], ["out10"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out10"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]),
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
        for c, value in enumerate(row):
            out[0, int(value), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    bad = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            inp = np.asarray(ex["input"], dtype=np.int64)
            expected = np.asarray(ex["output"], dtype=np.int64)
            raw = _run_onnx(model, _grid_to_onehot(ex["input"]))
            active = raw[0] > 0.0
            inside = active[:, : expected.shape[0], : expected.shape[1]]
            outside = active.copy()
            outside[:, : expected.shape[0], : expected.shape[1]] = False
            pred = _onehot_to_grid(raw)[: expected.shape[0], : expected.shape[1]]
            if not np.array_equal(solve(inp), expected):
                bad += 1
            if not np.array_equal(pred, expected):
                bad += 1
            if not np.all(inside.sum(axis=0) == 1) or np.any(outside):
                bad += 1
    return bad


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    bad = validate_json(model)
    assert bad == 0, f"{bad} JSON checks failed"

    print(score_file(BEST_PATH))


if __name__ == "__main__":
    main()
