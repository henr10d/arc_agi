"""Minimal ONNX for ARC task373 alternating two horizontal bar colors.

Task rule: the active grid is 2x6. Row 0 is a full-width bar of color A and
row 1 is a full-width bar of color B. Recolor the occupied cells by column
parity: row 0 becomes A, B, A, B, A, B while row 1 becomes B, A, B, A, B, A.
Padding outside the 2x6 grid stays all-zero.

ONNX approach: slice the one-hot color vectors from the first cell of each row,
then use one broadcasted checkerboard mask to choose between those two vectors
for every cell of the compact 2x6 output. The final Pad expands only that
compact float tensor to the required [1, 10, 30, 30] output tensor.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task373"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task373.onnx"

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


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes_hw = _i64(inits, [2, 3], "axes_hw")
    top_starts = _i64(inits, [0, 0], "top_starts")
    top_ends = _i64(inits, [1, 1], "top_ends")
    bottom_starts = _i64(inits, [1, 0], "bottom_starts")
    bottom_ends = _i64(inits, [2, 1], "bottom_ends")
    checker = _init(
        inits,
        np.asarray(
            [
                [False, True, False, True, False, True],
                [True, False, True, False, True, False],
            ],
            dtype=np.bool_,
        ).reshape(1, 1, 2, 6),
        "checker",
    )

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, top_starts, top_ends, axes_hw], ["top_f"]),
            helper.make_node("Slice", [IN_NAME, bottom_starts, bottom_ends, axes_hw], ["bottom_f"]),
            helper.make_node("Where", [checker, "bottom_f", "top_f"], ["out_float"]),
            helper.make_node(
                "Pad",
                ["out_float"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, 28, 24],
                value=0.0,
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
