"""Minimal ONNX for ARC task389 using the local task-data rule.

Task rule from data/task389.json: each input contains gray 5 and one non-black
foreground color. Preserve the visible 3x3, 4x4, or 5x5 shape, move the
foreground color onto every gray cell, and turn the original foreground cells
black 0. Padding outside the visible grid stays all-zero for NeuroGolf one-hot
I/O.

The graph specializes to the local data: visible grids are top-left 3x3, 4x4,
or 5x5 squares, and no input foreground is black. It detects the square size
from cells (3,3) and (4,4), works in the compact 5x5 region, then pads that
result to 30x30.
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

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task389"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task389.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
K = 5
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    def row_mask(n: int) -> np.ndarray:
        mask = np.zeros((1, 1, K, 1), dtype=np.uint8)
        mask[:, :, :n, :] = 1
        return mask

    axes_chsp = _i64(inits, [1, 2, 3], "axes_chsp")
    starts_gray = _i64(inits, [5, 0, 0], "starts_gray")
    ends_gray = _i64(inits, [6, K, K], "ends_gray")
    starts_33 = _i64(inits, [1, 3, 3], "starts_33")
    ends_33 = _i64(inits, [C, 4, 4], "ends_33")
    starts_44 = _i64(inits, [1, 4, 4], "starts_44")
    ends_44 = _i64(inits, [C, K, K], "ends_44")
    axis_channel = _i64(inits, [1], "axis_channel")
    row3 = _init(inits, row_mask(3), "row3")
    row4 = _init(inits, row_mask(4), "row4")
    row5 = _init(inits, row_mask(5), "row5")
    ch1 = _i64(inits, [1], "ch1")
    ch5 = _i64(inits, [5], "ch5")
    ch6 = _i64(inits, [6], "ch6")
    ch10 = _i64(inits, [10], "ch10")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts_gray, ends_gray, axes_chsp], ["gray"]),
            helper.make_node("Cast", ["gray"], ["gray_b"], to=TensorProto.BOOL),
            helper.make_node("Slice", [IN_NAME, starts_33, ends_33, axes_chsp], ["cell33"]),
            helper.make_node("ReduceMax", ["cell33"], ["has4_f"], axes=[1], keepdims=1),
            helper.make_node("Cast", ["has4_f"], ["has4"], to=TensorProto.BOOL),
            helper.make_node("Slice", [IN_NAME, starts_44, ends_44, axes_chsp], ["cell44"]),
            helper.make_node("ReduceMax", ["cell44"], ["has5_f"], axes=[1], keepdims=1),
            helper.make_node("Cast", ["has5_f"], ["has5"], to=TensorProto.BOOL),
            helper.make_node("Where", ["has4", row4, row3], ["row4_u8"]),
            helper.make_node("Where", ["has5", row5, "row4_u8"], ["row_u8"]),
            helper.make_node("Cast", ["row_u8"], ["row_b"], to=TensorProto.BOOL),
            helper.make_node("Transpose", ["row_b"], ["col_b"], perm=[0, 1, 3, 2]),
            helper.make_node("And", ["row_b", "col_b"], ["visible"]),
            helper.make_node("Xor", ["visible", "gray_b"], ["fg_cells"]),
            helper.make_node("ReduceMax", [IN_NAME], ["present"], axes=[2, 3], keepdims=1),
            helper.make_node("Cast", ["present"], ["present_b"], to=TensorProto.BOOL),
            helper.make_node("Slice", ["present_b", ch1, ch5, axis_channel], ["present_ch1_4"]),
            helper.make_node("Slice", ["present_b", ch6, ch10, axis_channel], ["present_ch6_9"]),
            helper.make_node("And", ["gray_b", "present_ch1_4"], ["out_ch1_4"]),
            helper.make_node("And", ["gray_b", "present_ch6_9"], ["out_ch6_9"]),
            helper.make_node("And", ["fg_cells", "gray_b"], ["out_ch5"]),
            helper.make_node("Concat", ["fg_cells", "out_ch1_4", "out_ch5", "out_ch6_9"], ["out_core"], axis=1),
            helper.make_node("Cast", ["out_core"], ["out_core_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out_core_f"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - K, W - K],
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


def validate_examples(path: Path) -> tuple[int, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            arr = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if arr is None or expected is None:
                continue
            actual = session.run([OUT_NAME], {IN_NAME: arr})[0]
            total += 1
            if not np.array_equal(actual > 0.0, expected > 0.0):
                raise AssertionError(f"{split}[{idx}] output mismatch")
            passed += 1
    return passed, total


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    passed, total = validate_examples(BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"correct: {passed}/{total}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
