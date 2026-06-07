"""ONNX solution for NeuroGolf task383.

Task rule: the input has a colored rectangular frame around a second colored
content rectangle, with one-cell content-color protrusions poking into the
frame. Rows or columns whose content cells protrude outside the core content
rectangle become symmetry axes. The output extends those axes through visible
background in the content color and changes content cells on the axes back to
the frame color.

The ONNX graph detects the frame color from the first colored bounding-box
corner, treats the other non-background color as content, defines core
rows/columns as those with more than three content cells, and marks protruding
core rows/columns as axes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "task383"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task383.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, 10, 30, 30]
OPSET = 10
IR_VERSION = 10


def _init(inits: list[onnx.TensorProto], arr: Any, name: str, dtype: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=dtype), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    return _init(inits, arr, name, np.int64)


def _f32(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    return _init(inits, arr, name, np.float32)


def _bool(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    return _init(inits, arr, name, bool)


def build_onnx_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    ch0_st = _i64(inits, [0], "ch0_st")
    ch1_en = _i64(inits, [1], "ch1_en")
    ch_axis = _i64(inits, [1], "ch_axis")
    zero = _f32(inits, [0.0], "zero")
    three = _f32(inits, [3.0], "three")
    nonbg_vec = _bool(inits, np.array([[[[False]], [[True]], [[True]], [[True]], [[True]], [[True]], [[True]], [[True]], [[True]], [[True]]]]), "nonbg_vec")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, ch0_st, ch1_en, ch_axis], ["bg_f"]),
            helper.make_node("ReduceMax", [IN_NAME], ["active_cell_f"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["active_cell_f", "bg_f"], ["fg_b"]),
            helper.make_node("Greater", ["bg_f", zero], ["bg_b"]),
            helper.make_node("Cast", ["fg_b"], ["fg_h"], to=TensorProto.FLOAT16),
            helper.make_node("ReduceMax", ["fg_h"], ["row_has"], axes=[1, 3], keepdims=0),
            helper.make_node("ReduceMax", ["fg_h"], ["col_has"], axes=[1, 2], keepdims=0),
            helper.make_node("ArgMax", ["row_has"], ["r0"], axis=1, keepdims=0),
            helper.make_node("ArgMax", ["col_has"], ["c0"], axis=1, keepdims=0),
            helper.make_node("Gather", [IN_NAME, "r0"], ["frame_row"], axis=2),
            helper.make_node("Gather", ["frame_row", "c0"], ["frame_vec_f"], axis=3),
            helper.make_node("ReduceMax", [IN_NAME], ["active_f"], axes=[2, 3], keepdims=1),
            helper.make_node("Greater", ["active_f", zero], ["active_b"]),
            helper.make_node("Greater", ["frame_vec_f", zero], ["frame_vec_b"]),
            helper.make_node("Not", ["frame_vec_b"], ["not_frame_vec"]),
            helper.make_node("And", ["active_b", nonbg_vec], ["active_nonbg"]),
            helper.make_node("And", ["active_nonbg", "not_frame_vec"], ["content_vec_b"]),
            helper.make_node("Cast", ["content_vec_b"], ["content_vec_h"], to=TensorProto.FLOAT16),
            helper.make_node("ArgMax", ["content_vec_h"], ["content_idx_raw"], axis=1, keepdims=0),
            helper.make_node("Squeeze", ["content_idx_raw"], ["content_idx"], axes=[1, 2]),
            helper.make_node("Gather", [IN_NAME, "content_idx"], ["content_cell_f"], axis=1),
            helper.make_node("Greater", ["content_cell_f", zero], ["content_cell_b"]),
            helper.make_node("ReduceSum", ["content_cell_f"], ["row_count"], axes=[3], keepdims=1),
            helper.make_node("ReduceSum", ["content_cell_f"], ["col_count"], axes=[2], keepdims=1),
            helper.make_node("Greater", ["row_count", three], ["core_row"]),
            helper.make_node("Greater", ["col_count", three], ["core_col"]),
            helper.make_node("Cast", ["core_col"], ["core_col_h"], to=TensorProto.FLOAT16),
            helper.make_node("Cast", ["core_row"], ["core_row_h"], to=TensorProto.FLOAT16),
            helper.make_node("ReduceSum", ["core_col_h"], ["core_width_h"], axes=[3], keepdims=1),
            helper.make_node("ReduceSum", ["core_row_h"], ["core_height_h"], axes=[2], keepdims=1),
            helper.make_node("Cast", ["core_width_h"], ["core_width"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["core_height_h"], ["core_height"], to=TensorProto.FLOAT),
            helper.make_node("Greater", ["row_count", "core_width"], ["row_protrudes"]),
            helper.make_node("Greater", ["col_count", "core_height"], ["col_protrudes"]),
            helper.make_node("And", ["core_row", "row_protrudes"], ["h_axis"]),
            helper.make_node("And", ["core_col", "col_protrudes"], ["v_axis"]),
            helper.make_node("Or", ["h_axis", "v_axis"], ["axis_b"]),
            helper.make_node("And", ["axis_b", "content_cell_b"], ["axis_content"]),
            helper.make_node("And", ["axis_b", "bg_b"], ["axis_bg"]),
            helper.make_node("Cast", [IN_NAME], ["input_u8"], to=TensorProto.UINT8),
            helper.make_node("Cast", ["frame_vec_b"], ["frame_vec_u8"], to=TensorProto.UINT8),
            helper.make_node("Cast", ["content_vec_b"], ["content_vec_u8"], to=TensorProto.UINT8),
            helper.make_node("Where", ["axis_content", "frame_vec_u8", "input_u8"], ["content_to_frame_u8"]),
            helper.make_node("Where", ["axis_bg", "content_vec_u8", "content_to_frame_u8"], ["output_u8"]),
            helper.make_node("Cast", ["output_u8"], [OUT_NAME], to=TensorProto.FLOAT),
        ]
    )

    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name=TASK_ID,
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return model


def _one_hot(grid: list[list[int]]) -> np.ndarray:
    arr = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            arr[0, int(color), r, c] = 1.0
    return arr


def _decode(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    active = arr[0, :, :h, :w] > 0.0
    return np.argmax(active, axis=0)


def validate_model(model: onnx.ModelProto) -> None:
    data = json.loads(DATA_PATH.read_text())
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data[split]):
            h = len(example["input"])
            w = len(example["input"][0])
            expected = np.asarray(example["output"], dtype=np.int64)
            pred = session.run([OUT_NAME], {IN_NAME: _one_hot(example["input"])})[0]
            expected_hot = _one_hot(example["output"])[0, :, :h, :w] > 0.0
            pred_hot = pred[0, :, :h, :w] > 0.0
            if not np.array_equal(pred_hot, expected_hot):
                bad = np.argwhere(pred_hot != expected_hot)[0]
                ch, r, c = map(int, bad)
                raise AssertionError(
                    f"{split}[{idx}] one-hot mismatch at channel/row/col {(ch, r, c)}: "
                    f"got {pred_hot[ch, r, c]}, expected {expected_hot[ch, r, c]}"
                )
            got = _decode(pred, h, w)
            if not np.array_equal(got, expected):
                diff = np.argwhere(got != expected)[0]
                r, c = map(int, diff)
                raise AssertionError(
                    f"{split}[{idx}] mismatch at {(r, c)}: got {got[r, c]}, expected {expected[r, c]}"
                )
            total += 1
    print(f"validated {total} examples")


def save_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    model = build_onnx_model()
    onnx.save(model, str(path))
    return model


def main() -> None:
    model = save_model()
    validate_model(model)
    print(BEST_PATH)


if __name__ == "__main__":
    main()
