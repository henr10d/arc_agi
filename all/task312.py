"""Build ONNX for task312: recolor gray cells from the left row template.

Task rule: the first column is a row-by-row color key. Every gray cell
(ARC color 5) is replaced by the color that appears in column 0 of the same
row. All other cells, including black background and the template strip, stay
unchanged. All examples are 12x12, but the graph applies the rule across the
full 30x30 competition tensor so padding remains all-zero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy  # noqa: E402

TASK_ID = "task312"
OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
C = 10
N = 30
SHAPE = [1, C, N, N]
GRAY = 5
OPSET = 10
IR_VERSION = 10


def solve(grid: list[list[int]]) -> list[list[int]]:
    """Reference grid transform."""
    return [
        [row[0] if color == GRAY else color for color in row]
        for row in grid
    ]


def _i64(inits: list[onnx.TensorProto], name: str, vals: list[int]) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))
    return name


def _load_task() -> dict[str, Any]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _examples(data: dict[str, Any]) -> list[dict[str, list[list[int]]]]:
    return [
        example
        for split in ("train", "test", "arc-gen")
        for example in data.get(split, [])
    ]


def _task_dims(data: dict[str, Any]) -> tuple[int, int]:
    dims = {
        (len(example["input"]), len(example["input"][0]))
        for example in _examples(data)
        if example.get("input")
    }
    if len(dims) != 1:
        raise ValueError(f"{TASK_ID} needs one static HxW, got {sorted(dims)}")
    h, w = next(iter(dims))
    if h <= 0 or w <= 0 or h > N or w > N:
        raise ValueError(f"invalid {TASK_ID} dimensions: {h}x{w}")
    return h, w


def _validate_rule(data: dict[str, Any]) -> None:
    gray_seen = False
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(data.get(split, [])):
            grid = example["input"]
            gray_seen = gray_seen or any(GRAY in row for row in grid)
            expected = solve(grid)
            if example["output"] != expected:
                raise ValueError(f"{split}[{index}] does not match row-template recolor rule")
    if not gray_seen:
        raise ValueError(f"{TASK_ID} has no color {GRAY} cells to recolor")


def build_onnx_model(h: int, w: int) -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    gray_st = _i64(inits, "gray_st", [GRAY, 0, 0])
    gray_en = _i64(inits, "gray_en", [GRAY + 1, h, w])
    gray_axes = _i64(inits, "gray_axes", [1, 2, 3])
    col_st = _i64(inits, "col_st", [0])
    col_en = _i64(inits, "col_en", [1])
    col_axis = _i64(inits, "col_axis", [3])
    right_shape = _i64(inits, "right_shape", [1, 1, h, N - w])
    bottom_shape = _i64(inits, "bottom_shape", [1, 1, N - h, N])
    false_value = helper.make_tensor("false_value", TensorProto.BOOL, [1], [False])

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, gray_st, gray_en, gray_axes], ["gray_plane"]),
            helper.make_node("Cast", ["gray_plane"], ["gray_mask_core"], to=TensorProto.BOOL),
            helper.make_node("ConstantOfShape", [right_shape], ["right_false"], value=false_value),
            helper.make_node("Concat", ["gray_mask_core", "right_false"], ["gray_mask_rows"], axis=3),
            helper.make_node("ConstantOfShape", [bottom_shape], ["bottom_false"], value=false_value),
            helper.make_node("Concat", ["gray_mask_rows", "bottom_false"], ["gray_mask"], axis=2),
            helper.make_node("Slice", [IN_NAME, col_st, col_en, col_axis], ["row_colors"]),
            helper.make_node("Where", ["gray_mask", "row_colors", IN_NAME], [OUT_NAME]),
        ]
    )

    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task312",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _verify_examples(model: onnx.ModelProto, data: dict[str, Any]) -> None:
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    )
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(data.get(split, [])):
            input_tensor = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if input_tensor is None or expected is None:
                continue
            actual = session.run([OUT_NAME], {IN_NAME: input_tensor})[0]
            if not np.array_equal(actual > 0.0, expected > 0.0):
                raise AssertionError(f"{split}[{index}] failed")


def main() -> None:
    data = _load_task()
    _validate_rule(data)
    h, w = _task_dims(data)
    model = build_onnx_model(h, w)
    _verify_examples(model, data)
    onnx.save(model, BEST_PATH)
    print(f"wrote {BEST_PATH} for {h}x{w} gray recolor")


if __name__ == "__main__":
    main()
