"""Minimal ONNX for ARC task259: crop the object bbox and blacken gaps.

Task rule: inputs use color 1 as the dominant blue background, and object
colors are 2-9.  Find the minimal bounding box containing every object cell,
copy that box to the top-left of the output, and turn any background cells
inside the box into black 0.  The observed bbox is at most 3x3, so cells outside
the bbox extent in the fixed 3x3 ONNX crop are left all-zero padding.

ONNX: work on only foreground channels 2-9 in the top-left 7x7 area, compute
min/max occupied row and column with ReduceMax/ArgMax, gather a 3x3 foreground
crop by dynamic row/column indices, derive black holes as valid cells with no
foreground, then assemble bool output channels before one final cast to float
and pad to the required 30x30 tensor.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task259"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task259.onnx"
DATA_PATH = ROOT / "data" / "task259.json"

C = 10
H = W = 30
G = 7
OUT = 3
PAD = H - OUT
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


def _make_model(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto]) -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(nodes, "task259_bbox_crop", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def solve(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    """Reference solver for the color-1 background bbox crop."""
    arr = np.asarray(grid, dtype=np.int64)
    mask = arr != 1
    rows, cols = np.where(mask)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    out = arr[r0 : r1 + 1, c0 : c1 + 1].copy()
    out[out == 1] = 0
    return out


def _grid_to_onehot(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            color = int(arr[r, c])
            if 0 <= color < C:
                out[0, color, r, c] = 1.0
    return out


def _expected_onehot(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    return _grid_to_onehot(grid)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def build_onnx_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    one_i = _i64(inits, [1], "one_i")
    max_index = _i64(inits, [G - 1], "max_index")
    grid_len = _i64(inits, [G], "grid_len")
    offsets = _i64(inits, [0, 1, 2], "offsets")
    rev = _i64(inits, list(range(G - 1, -1, -1)), "reverse7")
    valid_row_shape = _i64(inits, [1, 1, OUT, 1], "valid_row_shape")
    valid_col_shape = _i64(inits, [1, 1, 1, OUT], "valid_col_shape")
    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    starts_fg = _i64(inits, [0, 2, 0, 0], "starts_fg")
    ends_fg = _i64(inits, [1, C, G, G], "ends_fg")
    zero_ch = _init(inits, np.zeros((1, 1, OUT, OUT), dtype=np.bool_), "zero_ch")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts_fg, ends_fg, axes4], ["fg_channels"]),
            helper.make_node("ReduceSum", ["fg_channels"], ["fg_float"], axes=[1], keepdims=1),
            helper.make_node("ReduceMax", ["fg_float"], ["row_has"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["fg_float"], ["col_has"], axes=[2], keepdims=1),
            helper.make_node("ArgMax", ["row_has"], ["row_min"], axis=2, keepdims=0),
            helper.make_node("ArgMax", ["col_has"], ["col_min"], axis=3, keepdims=0),
            helper.make_node("Gather", ["row_has", rev], ["row_rev"], axis=2),
            helper.make_node("Gather", ["col_has", rev], ["col_rev"], axis=3),
            helper.make_node("ArgMax", ["row_rev"], ["row_rev_first"], axis=2, keepdims=0),
            helper.make_node("ArgMax", ["col_rev"], ["col_rev_first"], axis=3, keepdims=0),
            helper.make_node("Sub", [max_index, "row_rev_first"], ["row_max"]),
            helper.make_node("Sub", [max_index, "col_rev_first"], ["col_max"]),
            helper.make_node("Squeeze", ["row_min"], ["row_min_1"], axes=[0, 1]),
            helper.make_node("Squeeze", ["col_min"], ["col_min_1"], axes=[0, 1]),
            helper.make_node("Squeeze", ["row_max"], ["row_max_1"], axes=[0, 1]),
            helper.make_node("Squeeze", ["col_max"], ["col_max_1"], axes=[0, 1]),
            helper.make_node("Sub", ["row_max_1", "row_min_1"], ["row_span"]),
            helper.make_node("Sub", ["col_max_1", "col_min_1"], ["col_span"]),
            helper.make_node("Add", ["row_min_1", offsets], ["row_indices_raw"]),
            helper.make_node("Add", ["col_min_1", offsets], ["col_indices_raw"]),
            helper.make_node("Less", ["row_indices_raw", grid_len], ["row_in_range"]),
            helper.make_node("Less", ["col_indices_raw", grid_len], ["col_in_range"]),
            helper.make_node("Where", ["row_in_range", "row_indices_raw", max_index], ["row_indices"]),
            helper.make_node("Where", ["col_in_range", "col_indices_raw", max_index], ["col_indices"]),
            helper.make_node("Add", ["row_span", one_i], ["row_len"]),
            helper.make_node("Add", ["col_span", one_i], ["col_len"]),
            helper.make_node("Less", [offsets, "row_len"], ["valid_rows_1d"]),
            helper.make_node("Less", [offsets, "col_len"], ["valid_cols_1d"]),
            helper.make_node("Reshape", ["valid_rows_1d", valid_row_shape], ["valid_rows"]),
            helper.make_node("Reshape", ["valid_cols_1d", valid_col_shape], ["valid_cols"]),
            helper.make_node("And", ["valid_rows", "valid_cols"], ["valid"]),
            helper.make_node("Gather", ["fg_channels", "row_indices"], ["crop_rows"], axis=2),
            helper.make_node("Gather", ["crop_rows", "col_indices"], ["crop_float"], axis=3),
            helper.make_node("Cast", ["crop_float"], ["crop_fg"], to=TensorProto.BOOL),
            helper.make_node("ReduceMax", ["crop_float"], ["crop_any_float"], axes=[1], keepdims=1),
            helper.make_node("Cast", ["crop_any_float"], ["crop_any"], to=TensorProto.BOOL),
            helper.make_node("Not", ["crop_any"], ["crop_empty"]),
            helper.make_node("And", ["crop_empty", "valid"], ["out0"]),
            helper.make_node("And", ["crop_fg", "valid"], ["out2_9"]),
            helper.make_node("Concat", ["out0", zero_ch, "out2_9"], ["out_bool"], axis=1),
            helper.make_node("Cast", ["out_bool"], ["out_float"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out_float"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]),
        ]
    )

    return _make_model(nodes, inits)


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            expected = _expected_onehot(ex["output"])
            pred = _run_onnx(model, _grid_to_onehot(ex["input"]))
            if not np.array_equal(pred > 0.0, expected > 0.0):
                bad += 1
                if bad <= 5:
                    print(f"failed {split} example {idx}")
    return bad


def main() -> None:
    model = build_onnx_model()
    bad = validate_json(model)
    if bad:
        raise AssertionError(f"{TASK_ID} failed {bad} examples")

    with tempfile.NamedTemporaryFile(suffix=f"_{TASK_ID}.onnx", dir=OUT_DIR, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        onnx.save(model, tmp_path)
        result = score_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not result["valid"]:
        raise AssertionError(f"{TASK_ID} invalid: {result['error']}")

    onnx.save(model, BEST_PATH)
    final = score_file(BEST_PATH)
    print(
        f"wrote {BEST_PATH} nodes={len(model.graph.node)} memory={final['memory']} "
        f"params={final['params']} cost={final['cost']} score={final['score']:.6f}"
    )


if __name__ == "__main__":
    main()
