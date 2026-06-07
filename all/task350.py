"""ONNX solution for ARC task350: connect aligned blue anchors.

Task rule: input grids contain blue anchor cells (color 1) on background 0.
For each row, every non-anchor cell between blue anchors becomes cyan
(color 8); equivalently it has at least one blue anchor to its left and one to
its right. The same rule is applied independently to columns using anchors
above and below. Original blue anchors remain blue, and cells outside the
visible padded grid stay all-zero in NeuroGolf one-hot I/O.

ONNX approach: crop the active blue channel to the largest observed task
extent (26x24), use ArgMax to find the first anchor in each row/column, and
use ReduceMax over coordinate-weighted blue cells to find the last anchor.
The compact bool fill mask is padded back to 30x30 and used in one final
broadcasted Where against the original input, so the full 10-channel tensor is
only the graph output rather than a scored intermediate.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_all_onnx import verify_correctness  # noqa: E402
from score_model import print_report, score_file  # noqa: E402

TASK_ID = "task350"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
ROOT_PATH = ROOT / f"{TASK_ID}.onnx"

C = 10
H = W = 30
CORE_H = 26
CORE_W = 24
BLUE = 1
CYAN = 8
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _slice(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    source: str,
    out: str,
    starts: list[int],
    ends: list[int],
    axes: list[int],
) -> str:
    s = _init(inits, f"{out}_s", np.asarray(starts, dtype=np.int64))
    e = _init(inits, f"{out}_e", np.asarray(ends, dtype=np.int64))
    a = _init(inits, f"{out}_a", np.asarray(axes, dtype=np.int64))
    nodes.append(helper.make_node("Slice", [source, s, e, a], [out]))
    return out


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    half_f = _init(inits, "half_f", np.asarray([0.5], dtype=np.float32))
    cyan_value = np.zeros((1, C, 1, 1), dtype=np.float32)
    cyan_value[0, CYAN, 0, 0] = 1.0
    cyan_const = _init(inits, "cyan", cyan_value)
    bottom_false = _init(inits, "bottom_false", np.zeros((1, 1, H - CORE_H, CORE_W), dtype=bool))
    right_false = _init(inits, "right_false", np.zeros((1, 1, H, W - CORE_W), dtype=bool))
    col_idx_i = _init(inits, "col_idx_i", np.arange(CORE_W, dtype=np.int64).reshape(1, 1, 1, CORE_W))
    row_idx_i = _init(inits, "row_idx_i", np.arange(CORE_H, dtype=np.int64).reshape(1, 1, CORE_H, 1))
    col_idx_f = _init(inits, "col_idx_f", np.arange(CORE_W, dtype=np.float32).reshape(1, 1, 1, CORE_W))
    row_idx_f = _init(inits, "row_idx_f", np.arange(CORE_H, dtype=np.float32).reshape(1, 1, CORE_H, 1))

    blue_f = _slice(nodes, inits, IN_NAME, "blue_f", [BLUE, 0, 0], [BLUE + 1, CORE_H, CORE_W], [1, 2, 3])
    nodes.extend(
        [
            helper.make_node("Less", [blue_f, half_f], ["not_blue"]),
            helper.make_node("ArgMax", [blue_f], ["first_col"], axis=3, keepdims=1),
            helper.make_node("Mul", [blue_f, col_idx_f], ["blue_cols"]),
            helper.make_node("ReduceMax", ["blue_cols"], ["last_col"], axes=[3], keepdims=1),
            helper.make_node("Greater", [col_idx_i, "first_col"], ["after_first_col"]),
            helper.make_node("Less", [col_idx_f, "last_col"], ["before_last_col"]),
            helper.make_node("And", ["after_first_col", "before_last_col"], ["h_line"]),
            helper.make_node("ArgMax", [blue_f], ["first_row"], axis=2, keepdims=1),
            helper.make_node("Mul", [blue_f, row_idx_f], ["blue_rows"]),
            helper.make_node("ReduceMax", ["blue_rows"], ["last_row"], axes=[2], keepdims=1),
            helper.make_node("Greater", [row_idx_i, "first_row"], ["after_first_row"]),
            helper.make_node("Less", [row_idx_f, "last_row"], ["before_last_row"]),
            helper.make_node("And", ["after_first_row", "before_last_row"], ["v_line"]),
            helper.make_node("Or", ["h_line", "v_line"], ["line"]),
            helper.make_node("And", ["line", "not_blue"], ["fill_small"]),
            helper.make_node("Concat", ["fill_small", bottom_false], ["fill_cols"], axis=2),
            helper.make_node("Concat", ["fill_cols", right_false], ["fill"], axis=3),
            helper.make_node("Where", ["fill", cyan_const, IN_NAME], [OUT_NAME]),
        ]
    )

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


def solve_grid(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.uint8)
    out = arr.copy()
    blue = arr == BLUE

    for r in range(arr.shape[0]):
        cols = np.flatnonzero(blue[r])
        for c1, c2 in zip(cols, cols[1:]):
            out[r, c1 + 1 : c2] = CYAN

    for c in range(arr.shape[1]):
        rows = np.flatnonzero(blue[:, c])
        for r1, r2 in zip(rows, rows[1:]):
            out[r1 + 1 : r2, c] = CYAN

    out[blue] = BLUE
    return out


def onehot(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(H, arr.shape[0])):
        for c in range(min(W, arr.shape[1])):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def validate_reference() -> tuple[bool, str]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data.get(split, []):
            total += 1
            expected = np.asarray(ex["output"], dtype=np.uint8)
            actual = solve_grid(ex["input"])
            if np.array_equal(actual, expected):
                passed += 1
    return passed == total, f"{passed}/{total}"


def validate_model(model: onnx.ModelProto) -> tuple[bool, str]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        session = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        return False, f"load failed: {exc}"

    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data.get(split, []):
            grid = np.asarray(ex["input"], dtype=np.uint8)
            if max(grid.shape) > 30:
                continue
            total += 1
            expected = onehot(ex["output"]) > 0.0
            pred = session.run([OUT_NAME], {IN_NAME: onehot(grid)})[0] > 0.0
            if np.array_equal(pred, expected):
                passed += 1
            else:
                return False, f"{split} case {total} failed ({passed}/{total})"
    return passed == total, f"{passed}/{total}"


def main() -> None:
    ref_ok, ref_summary = validate_reference()
    if not ref_ok:
        raise SystemExit(f"reference solver mismatch: {ref_summary}")

    model = build_model()
    model_ok, model_summary = validate_model(model)
    if not model_ok:
        raise SystemExit(f"ONNX validation failed: {model_summary}")

    onnx.save(model, BEST_PATH)
    shutil.copy2(BEST_PATH, ROOT_PATH)

    correctness_ok, correctness, _passed, _total = verify_correctness(BEST_PATH)
    result = score_file(BEST_PATH)

    print(f"reference:   {ref_summary}")
    print(f"onnx local:  {model_summary}")
    print(f"correctness: {correctness} ({correctness_ok})")
    print_report(result)
    print(f"copied root model: {ROOT_PATH}")

    if not correctness_ok or not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
