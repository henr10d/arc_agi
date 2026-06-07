"""ONNX for ARC task215: extend a three-row periodic foreground band.

Task rule: the input grid has one non-black color and exactly three consecutive
non-empty rows starting at row 3 or row 4.  Those three rows are one period of a
horizontal lattice.  Repeat that row band upward and downward with period 3,
using the band position's absolute phase, and keep the original grid size.
Cells outside the input grid remain all-zero padding for NeuroGolf I/O.
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

TASK_ID = "task215"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task215.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
MAX_SH = MAX_SW = 20
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver: repeat the detected three-row band with period 3."""
    g = np.asarray(grid, dtype=np.int64)
    rows = np.flatnonzero((g != 0).any(axis=1))
    if len(rows) == 0:
        return np.zeros_like(g)
    top = int(rows[0])
    band = g[top : top + 3]
    out = np.zeros_like(g)
    for r in range(g.shape[0]):
        out[r] = band[(r - top) % 3]
    return out


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes_chw = _i64(inits, [1, 2, 3], "axes_chw")
    axis_ch = _i64(inits, [1], "axis_ch")
    axis_row = _i64(inits, [2], "axis_row")
    axis_col = _i64(inits, [3], "axis_col")
    fg3_st = _i64(inits, [1, 3, 0], "fg3_st")
    fg3_en = _i64(inits, [C, 4, MAX_SW], "fg3_en")
    bg4_st = _i64(inits, [0, 3, 0], "bg4_st")
    bg4_en = _i64(inits, [1, 7, MAX_SW], "bg4_en")
    color_st = _i64(inits, [1], "color_st")
    color_en = _i64(inits, [C], "color_en")
    s0 = _i64(inits, [0], "s0")
    e20 = _i64(inits, [MAX_SH], "e20")
    zero_f = _init(inits, np.asarray(0.0, dtype=np.float32), "zero_f")

    # If row 3 contains foreground, use rows [3,4,5].  Otherwise use [4,5,6].
    # The gathered indices preserve absolute period-3 phase over the 20x20 crop.
    idx_top3 = _i64(inits, ([0, 1, 2] * 7)[:MAX_SH], "idx_top3")
    idx_top4 = _i64(inits, ([3, 1, 2] * 7)[:MAX_SH], "idx_top4")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, fg3_st, fg3_en, axes_chw], ["fg3f"]),
            helper.make_node("ReduceMax", ["fg3f"], ["row3max"], axes=[0, 1, 2, 3], keepdims=0),
            helper.make_node("Greater", ["row3max", zero_f], ["has_row3"]),
            helper.make_node("Where", ["has_row3", idx_top3, idx_top4], ["row_idx"]),
            helper.make_node("Slice", [IN_NAME, bg4_st, bg4_en, axes_chw], ["bg4f"]),
            helper.make_node("Cast", ["bg4f"], ["bg4"], to=TensorProto.BOOL),
            helper.make_node("Not", ["bg4"], ["pat4"]),
            helper.make_node("Gather", ["pat4", "row_idx"], ["pat_rep"], axis=2),
            helper.make_node("ReduceMax", [IN_NAME], ["color_f"], axes=[2, 3], keepdims=1),
            helper.make_node("Slice", ["color_f", color_st, color_en, axis_ch], ["fg_color_f"]),
            helper.make_node("Greater", ["fg_color_f", zero_f], ["fg_color"]),
            helper.make_node("ReduceMax", [IN_NAME], ["row_f"], axes=[1, 3], keepdims=1),
            helper.make_node("ReduceMax", [IN_NAME], ["col_f"], axes=[1, 2], keepdims=1),
            helper.make_node("Slice", ["row_f", s0, e20, axis_row], ["row20_f"]),
            helper.make_node("Slice", ["col_f", s0, e20, axis_col], ["col20_f"]),
            helper.make_node("Greater", ["row20_f", zero_f], ["row_on"]),
            helper.make_node("Greater", ["col20_f", zero_f], ["col_on"]),
            helper.make_node("And", ["row_on", "col_on"], ["grid_on"]),
            helper.make_node("And", ["pat_rep", "grid_on"], ["pat_grid"]),
            helper.make_node("And", ["pat_grid", "fg_color"], ["fg_out"]),
            helper.make_node("Not", ["pat_grid"], ["no_fg"]),
            helper.make_node("And", ["grid_on", "no_fg"], ["bg_out"]),
            helper.make_node("Concat", ["bg_out", "fg_out"], ["out_bool"], axis=1),
            helper.make_node("Cast", ["out_bool"], ["out20f"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out20f"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - MAX_SH, W - MAX_SW],
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


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def verify_model(model: onnx.ModelProto) -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            g = np.asarray(ex["input"], dtype=np.int64)
            expected = np.asarray(ex["output"], dtype=np.int64)
            assert np.array_equal(solve(g), expected), f"reference mismatch {split}[{idx}]"

            pred_oh = _run_onnx(model, _grid_to_onehot(ex["input"]))
            pred = _onehot_to_grid(pred_oh)[: g.shape[0], : g.shape[1]]
            if not np.array_equal(pred, expected):
                raise AssertionError(f"ONNX grid mismatch {split}[{idx}]")

            expected_oh = _grid_to_onehot(ex["output"])
            if not np.array_equal(pred_oh > 0.0, expected_oh > 0.0):
                raise AssertionError(f"ONNX one-hot mismatch {split}[{idx}]")


def main() -> None:
    model = build_model()
    verify_model(model)
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
