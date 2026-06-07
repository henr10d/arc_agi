"""Compact ONNX for ARC task329: keep only the grid's middle vertical line.

Task rule: preserve the original grid size, copy the center column exactly, and
turn every off-column cell black. Nonzero distractors outside the center column
are removed; black gaps in the center column remain black.

ONNX: all examples are top-left odd square grids of size 3, 5, 7, or 9. Infer
the width from the occupied first row, dynamically gather the center input
column, build a compact bool 9x9 one-hot result with black off-center cells,
cast once to float, and pad that compact result to the 30x30 competition I/O.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task329"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task329.onnx"
DATA_PATH = ROOT / "data" / "task329.json"

C = 10
H = W = 30
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    arr = onehot.reshape(C, H, W)
    active = arr > 0.0
    bad = active.sum(axis=0) != 1
    decoded = arr.argmax(axis=0).astype(np.int64)
    decoded[bad] = -1
    return decoded


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver: keep only the center column of the visible grid."""
    out = np.zeros_like(grid)
    out[:, grid.shape[1] // 2] = grid[:, grid.shape[1] // 2]
    return out


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    one = _f32(inits, 1.0, "one")
    two = _f32(inits, 2.0, "two")
    rows9 = _f32(inits, np.arange(9, dtype=np.float32).reshape(1, 1, 9, 1), "rows9")
    cols9 = _f32(inits, np.arange(9, dtype=np.float32).reshape(1, 1, 1, 9), "cols9")
    cols9_i = _i64(inits, np.arange(9, dtype=np.int64).reshape(1, 1, 1, 9), "cols9_i")
    top_st = _i64(inits, [0, 0, 0, 0], "top_st")
    top_en = _i64(inits, [1, C, 1, 9], "top_en")
    slice_axes = _i64(inits, [0, 1, 2, 3], "slice_axes")
    slice_steps = _i64(inits, [1, 1, 1, 1], "slice_steps")
    row_st = _i64(inits, [0], "row_st")
    one_step = _i64(inits, [1], "one_step")
    one_i = _i64(inits, 1, "one_i")
    center_st_prefix = _i64(inits, [0, 0, 0], "center_st_prefix")
    center_en_prefix = _i64(inits, [1, C, 9], "center_en_prefix")
    ch_end = _i64(inits, [C], "ch_end")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, top_st, top_en, slice_axes, slice_steps], ["top"]),
            helper.make_node("ReduceSum", ["top"], ["top_occ"], axes=[0, 1, 2], keepdims=0),
            helper.make_node("ReduceSum", ["top_occ"], ["width"], axes=[0], keepdims=0),
            helper.make_node("Sub", ["width", one], ["width_minus_one"]),
            helper.make_node("Div", ["width_minus_one", two], ["center"]),
            helper.make_node("Cast", ["center"], ["center_i"], to=TensorProto.INT64),
            helper.make_node("Equal", [cols9_i, "center_i"], ["center_col"]),
            helper.make_node("Not", ["center_col"], ["not_center_col"]),
            helper.make_node("Less", [rows9, "width"], ["valid_rows"]),
            helper.make_node("Less", [cols9, "width"], ["valid_cols"]),
            helper.make_node("And", ["valid_rows", "valid_cols"], ["visible"]),
            helper.make_node("And", ["visible", "not_center_col"], ["visible_off_center"]),
            helper.make_node("Unsqueeze", ["center_i"], ["center_idx"], axes=[0]),
            helper.make_node("Concat", [center_st_prefix, "center_idx"], ["center_st"], axis=0),
            helper.make_node("Add", ["center_i", one_i], ["center_end_i"]),
            helper.make_node("Unsqueeze", ["center_end_i"], ["center_end_idx"], axes=[0]),
            helper.make_node("Concat", [center_en_prefix, "center_end_idx"], ["center_en"], axis=0),
            helper.make_node("Slice", [IN_NAME, "center_st", "center_en", slice_axes, slice_steps], ["center9"]),
            helper.make_node("Cast", ["center9"], ["center9b"], to=TensorProto.BOOL),
            helper.make_node("Slice", ["center9b", row_st, one_step, one_step, one_step], ["center0"]),
            helper.make_node("Slice", ["center9b", one_step, ch_end, one_step, one_step], ["center_rest"]),
            helper.make_node("And", ["center_col", "center0"], ["center0_part"]),
            helper.make_node("Or", ["center0_part", "visible_off_center"], ["out0"]),
            helper.make_node("And", ["center_col", "center_rest"], ["out_rest"]),
            helper.make_node("Concat", ["out0", "out_rest"], ["out9b"], axis=1),
            helper.make_node("Cast", ["out9b"], ["out9"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out9"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - 9, W - 9]),
        ]
    )

    center9_info = helper.make_tensor_value_info("center9", TensorProto.FLOAT, [1, C, 9, 1])
    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits, value_info=[center9_info])
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


def validate_json(model: onnx.ModelProto) -> tuple[int, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    bad = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            grid = np.asarray(ex["input"], dtype=np.int64)
            if grid.shape[0] > H or grid.shape[1] > W:
                continue
            total += 1
            pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(ex["input"])))
            pred = pred[: grid.shape[0], : grid.shape[1]]
            expected = np.asarray(ex["output"], dtype=np.int64)
            if not np.array_equal(pred, expected):
                bad += 1
                print(f"mismatch {split} #{idx}")
                print("predicted:")
                print(pred)
                print("expected:")
                print(expected)
                return bad, total
    return bad, total


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    bad, total = validate_json(model)
    assert bad == 0, f"{bad} mismatches across {total} checked examples"
    print(f"verified {total} task329 examples")

    try:
        result = score_file(BEST_PATH)
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"score_model failed: {exc}")
    else:
        print(result)


if __name__ == "__main__":
    main()
