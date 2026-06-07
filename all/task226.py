"""Minimal ONNX for ARC task226: color three cells in a gray partition grid.

Task rule: gray divider lines split the 10x10 grid into rectangular cells.
Keep all gray divider pixels unchanged. Fill the top-left cell blue (1), the
bottom-right cell green (3), and the central cell red (2). Divider bands may be
thicker than one pixel, so cells are derived from gaps between contiguous gray
row/column bands.

ONNX: on the 10x10 crop, detect full gray rows/columns, find band starts, use
small prefix-count masks to identify the first, middle, and last row/column
cells, build boolean rectangle masks, then cast once for the final one-hot pad.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task226"
BEST_PATH = OUT_DIR / "task226.onnx"
DATA_PATH = ROOT / "data" / "task226.json"

C = 10
H = W = 30
SH = SW = 10
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


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _bool(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.bool_), name)


def _bands(mask: np.ndarray) -> List[Tuple[int, int]]:
    bands: List[Tuple[int, int]] = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < len(mask) and mask[j]:
            j += 1
        bands.append((i, j))
        i = j
    return bands


def _cell_masks(line_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bands = _bands(line_mask)
    cells: List[Tuple[int, int]] = []
    start = 0
    for b0, b1 in bands:
        cells.append((start, b0))
        start = b1
    cells.append((start, len(line_mask)))

    n = len(line_mask)
    first = np.zeros(n, dtype=bool)
    middle = np.zeros(n, dtype=bool)
    last = np.zeros(n, dtype=bool)
    for mask, (lo, hi) in (
        (first, cells[0]),
        (middle, cells[len(cells) // 2]),
        (last, cells[-1]),
    ):
        mask[lo:hi] = True
    return first, middle, last


def solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    row_div = np.all(g == 5, axis=1)
    col_div = np.all(g == 5, axis=0)
    top, mid_r, bot = _cell_masks(row_div)
    left, mid_c, right = _cell_masks(col_div)

    out[np.outer(top, left)] = 1
    out[np.outer(mid_r, mid_c)] = 2
    out[np.outer(bot, right)] = 3
    out[g == 5] = 5
    return out


def _grid_to_onehot(grid: Sequence[Sequence[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _prefix_lt(size: int) -> np.ndarray:
    idx = np.arange(size)
    return (idx[:, None] > idx[None, :]).astype(np.float32)


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes_chw = _i64(inits, [1, 2, 3], "axes_chw")
    gray_st = _i64(inits, [5, 0, 0], "gray_st")
    gray_en = _i64(inits, [6, SH, SW], "gray_en")
    s0 = _i64(inits, [0], "s0")
    s9 = _i64(inits, [SH - 1], "s9")
    zero_i = _init(inits, np.asarray([0], dtype=np.int32), "zero_i")
    axis_row = _i64(inits, [2], "axis_row")
    axis_col = _i64(inits, [3], "axis_col")
    false1 = _bool(inits, np.zeros((1, 1, 1, 1), dtype=np.bool_), "false1")
    false10 = _bool(inits, np.zeros((1, 1, SH, SW), dtype=np.bool_), "false10")
    prefix = _f32(inits, _prefix_lt(SH).T, "prefix")

    pads = [0, 0, 0, 0, 0, C - 6, H - SH, W - SW]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, gray_st, gray_en, axes_chw], ["gray"]),
            helper.make_node("ReduceMin", ["gray"], ["row_div_f"], axes=[3], keepdims=1),
            helper.make_node("ReduceMin", ["gray"], ["col_div_f"], axes=[2], keepdims=1),
            helper.make_node("Cast", ["row_div_f"], ["row_div"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["col_div_f"], ["col_div"], to=TensorProto.BOOL),
            helper.make_node("Slice", ["row_div", s0, s9, axis_row], ["row_prev_tail"]),
            helper.make_node("Concat", [false1, "row_prev_tail"], ["row_prev"], axis=2),
            helper.make_node("Not", ["row_prev"], ["not_row_prev"]),
            helper.make_node("And", ["row_div", "not_row_prev"], ["row_start"]),
            helper.make_node("Slice", ["col_div", s0, s9, axis_col], ["col_prev_tail"]),
            helper.make_node("Concat", [false1, "col_prev_tail"], ["col_prev"], axis=3),
            helper.make_node("Not", ["col_prev"], ["not_col_prev"]),
            helper.make_node("And", ["col_div", "not_col_prev"], ["col_start"]),
            helper.make_node("Cast", ["row_start"], ["row_start_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["col_start"], ["col_start_f"], to=TensorProto.FLOAT),
            helper.make_node("ReduceSum", ["row_start_f"], ["row_bands"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceSum", ["col_start_f"], ["col_bands"], axes=[2, 3], keepdims=1),
            helper.make_node("Transpose", ["row_start_f"], ["row_start_t"], perm=[0, 1, 3, 2]),
            helper.make_node("MatMul", ["row_start_t", prefix], ["row_idx_t"]),
            helper.make_node("Transpose", ["row_idx_t"], ["row_idx"], perm=[0, 1, 3, 2]),
            helper.make_node("MatMul", ["col_start_f", prefix], ["col_idx_t"]),
            helper.make_node("Cast", ["row_bands"], ["row_bands_i"], to=TensorProto.INT32),
            helper.make_node("Cast", ["col_bands"], ["col_bands_i"], to=TensorProto.INT32),
            helper.make_node("Cast", ["row_idx"], ["row_idx_i"], to=TensorProto.INT32),
            helper.make_node("Cast", ["col_idx_t"], ["col_idx_i"], to=TensorProto.INT32),
            helper.make_node("Equal", ["row_idx_i", zero_i], ["row_top_raw"]),
            helper.make_node("Equal", ["col_idx_i", zero_i], ["col_left_raw"]),
            helper.make_node("Equal", ["row_idx_i", "row_bands_i"], ["row_bot_raw"]),
            helper.make_node("Equal", ["col_idx_i", "col_bands_i"], ["col_right_raw"]),
            helper.make_node("Add", ["row_idx_i", "row_idx_i"], ["row_idx2"]),
            helper.make_node("Add", ["col_idx_i", "col_idx_i"], ["col_idx2"]),
            helper.make_node("Equal", ["row_idx2", "row_bands_i"], ["row_mid_raw"]),
            helper.make_node("Equal", ["col_idx2", "col_bands_i"], ["col_mid_raw"]),
            helper.make_node("Not", ["row_div"], ["not_row_div"]),
            helper.make_node("Not", ["col_div"], ["not_col_div"]),
            helper.make_node("And", ["row_top_raw", "not_row_div"], ["row_top"]),
            helper.make_node("And", ["row_mid_raw", "not_row_div"], ["row_mid"]),
            helper.make_node("And", ["row_bot_raw", "not_row_div"], ["row_bot"]),
            helper.make_node("And", ["col_left_raw", "not_col_div"], ["col_left"]),
            helper.make_node("And", ["col_mid_raw", "not_col_div"], ["col_mid"]),
            helper.make_node("And", ["col_right_raw", "not_col_div"], ["col_right"]),
            helper.make_node("And", ["row_top", "col_left"], ["blue_b"]),
            helper.make_node("And", ["row_mid", "col_mid"], ["red_b"]),
            helper.make_node("And", ["row_bot", "col_right"], ["green_b"]),
            helper.make_node("Or", ["blue_b", "red_b"], ["br_b"]),
            helper.make_node("Or", ["br_b", "green_b"], ["color_b"]),
            helper.make_node("Cast", ["gray"], ["gray_b"], to=TensorProto.BOOL),
            helper.make_node("Or", ["color_b", "gray_b"], ["non_bg_b"]),
            helper.make_node("Not", ["non_bg_b"], ["bg_b"]),
            helper.make_node(
                "Concat",
                ["bg_b", "blue_b", "red_b", "green_b", false10, "gray_b"],
                ["out6_b"],
                axis=1,
            ),
            helper.make_node("Cast", ["out6_b"], ["out6"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out6"], [OUT_NAME], pads=pads),
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


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            g = np.array(ex["input"], dtype=np.int64)
            expected = np.array(ex["output"], dtype=np.int64)
            ref = solve(g)
            if not np.array_equal(ref, expected):
                print(f"reference mismatch {split}[{idx}]")
                bad += 1
                continue
            pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(ex["input"])))[: g.shape[0], : g.shape[1]]
            if not np.array_equal(pred, expected):
                print(f"onnx mismatch {split}[{idx}]")
                bad += 1
    return bad


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    bad = validate_json(model)
    print(f"task226.json: {'PASS' if bad == 0 else f'FAIL ({bad} wrong)'}")

    result = score_file(BEST_PATH)
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
