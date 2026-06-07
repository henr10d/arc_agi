"""Compact ONNX for task321: overlay three 4-column row layers.

Task rule: each 4x14 input row contains three 4-cell masks separated by red
divider cells: columns 0:4 are yellow/gold (color 4), columns 5:9 are maroon
(color 9), and columns 10:14 are blue (color 1).  For each row and output
column, overlay the aligned cells with priority 4 > 9 > 1 > 0, producing a
4x4 output grid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "task321"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task321.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
OUT = 4
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
IR_VERSION = 10
OPSET = 10


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference NumPy solver for the ARC grid, not the one-hot tensor."""
    arr = np.asarray(grid, dtype=np.int64)
    gold = arr[:OUT, 0:4] == 4
    maroon = arr[:OUT, 5:9] == 9
    blue = arr[:OUT, 10:14] == 1

    out = np.zeros((OUT, OUT), dtype=np.int64)
    out = np.where(blue, 1, out)
    out = np.where(maroon, 9, out)
    out = np.where(gold, 4, out)
    return out


def _i64(inits: list[onnx.TensorProto], vals: Iterable[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(list(vals), dtype=np.int64), name=name))
    return name


def _grid_to_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _load_examples() -> list[dict[str, list[list[int]]]]:
    data = json.loads(DATA_PATH.read_text())
    return [ex for split in ("train", "test", "arc-gen") for ex in data.get(split, [])]


def build_onnx_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, [1, 2, 3], "axes")
    s4 = _i64(inits, [4, 0, 0], "s4")
    e4 = _i64(inits, [5, OUT, OUT], "e4")
    s9 = _i64(inits, [9, 0, 5], "s9")
    e9 = _i64(inits, [10, OUT, 9], "e9")
    s1 = _i64(inits, [1, 0, 10], "s1")
    e1 = _i64(inits, [2, OUT, 14], "e1")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, s4, e4, axes], ["f4"]),
            helper.make_node("Slice", [IN_NAME, s9, e9, axes], ["f9"]),
            helper.make_node("Slice", [IN_NAME, s1, e1, axes], ["f1"]),
            helper.make_node("Cast", ["f4"], ["c4"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["f9"], ["r9"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["f1"], ["r1"], to=TensorProto.BOOL),
            helper.make_node("Not", ["c4"], ["n4"]),
            helper.make_node("And", ["r9", "n4"], ["c9"]),
            helper.make_node("Not", ["r9"], ["n9"]),
            helper.make_node("And", ["n4", "n9"], ["no49"]),
            helper.make_node("And", ["no49", "r1"], ["c1"]),
            helper.make_node("Not", ["r1"], ["n1"]),
            helper.make_node("And", ["no49", "n1"], ["c0"]),
            helper.make_node("And", ["c4", "n4"], ["z"]),
            helper.make_node(
                "Concat",
                ["c0", "c1", "z", "z", "c4", "z", "z", "z", "z", "c9"],
                ["out4b"],
                axis=1,
            ),
            helper.make_node("Cast", ["out4b"], ["out4"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out4"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - OUT, W - OUT]),
        ]
    )

    graph = helper.make_graph(nodes, "task321", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def verify_model(model: onnx.ModelProto) -> None:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    for idx, ex in enumerate(_load_examples()):
        actual = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(ex["input"])})[0]
        expected = _grid_to_onehot(ex["output"])
        if not np.array_equal(actual > 0.0, expected > 0.0):
            pred = (actual[0, :, :OUT, :OUT] > 0.0).argmax(axis=0)
            raise AssertionError(f"example {idx} failed: predicted {pred.tolist()}, expected {ex['output']}")


def main() -> None:
    model = build_onnx_model()
    verify_model(model)
    onnx.save(model, BEST_PATH)
    print(f"wrote {BEST_PATH}")


if __name__ == "__main__":
    main()
