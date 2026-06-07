"""ONNX for ARC task114 using a compact woven expansion.

Task rule: the input is a 2x2, 2x3, 3x2, or 3x3 grid in the top-left corner.
Each row is expanded into a woven strip: for width 2, [a, a, b, b]; for width
3, [a, a, b, c, c]. The first output row is black padding, then the first input
row, then black padding; the last output row does the same for the final input
row. All meaningful work is kept in a 5x5 bool one-hot tensor and padded to the
competition 30x30 float output only at the final node.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "task114"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task114.onnx"

C = 10
H = W = 30
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
IR_VERSION = 10
OPSET = 10


def _init(inits: List[onnx.TensorProto], name: str, arr: np.ndarray) -> str:
    inits.append(numpy_helper.from_array(arr, name=name))
    return name


def _i64(inits: List[onnx.TensorProto], name: str, vals: list[int]) -> str:
    return _init(inits, name, np.asarray(vals, dtype=np.int64))


def _bool_cell(inits: List[onnx.TensorProto], name: str, channel0: bool) -> str:
    arr = np.zeros((1, C, 1, 1), dtype=bool)
    if channel0:
        arr[0, 0, 0, 0] = True
    return _init(inits, name, arr)


def build_bool_concat_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    def sel(cond: str, neg_cond: str, a: str, b: str, out: str) -> None:
        nodes.extend(
            [
                helper.make_node("And", [cond, a], [out + "t"]),
                helper.make_node("And", [neg_cond, b], [out + "f"]),
                helper.make_node("Or", [out + "t", out + "f"], [out]),
            ]
        )

    def sel_a_zero_when_false(neg_cond: str, a: str, b: str, out: str) -> None:
        nodes.extend(
            [
                helper.make_node("And", [neg_cond, b], [out + "f"]),
                helper.make_node("Or", [a, out + "f"], [out]),
            ]
        )

    def sel_zero(cond: str, a: str, out: str) -> None:
        nodes.append(helper.make_node("And", [cond, a], [out]))

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes_hw = _i64(inits, "ahw", [2, 3])
    for r in range(3):
        for c in range(3):
            _i64(inits, f"s{r}{c}", [r, c])
            _i64(inits, f"e{r}{c}", [r + 1, c + 1])
    zero_f = _init(inits, "zf", np.asarray([0.0], dtype=np.float32))
    black = _bool_cell(inits, "b", True)

    for r in range(3):
        for c in range(3):
            nodes.append(helper.make_node("Slice", [IN_NAME, f"s{r}{c}", f"e{r}{c}", axes_hw], [f"f{r}{c}"]))
            nodes.append(helper.make_node("Greater", [f"f{r}{c}", zero_f], [f"c{r}{c}"]))

    nodes.extend(
        [
            helper.make_node("ReduceSum", ["f02"], ["c2s"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["c2s", zero_f], ["has_c2"]),
            helper.make_node("Not", ["has_c2"], ["no_c2"]),
            helper.make_node("ReduceSum", ["f20"], ["r2s"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["r2s", zero_f], ["has_r2"]),
            helper.make_node("Not", ["has_r2"], ["no_r2"]),
        ]
    )

    sel_a_zero_when_false("no_c2", "c02", black, "top3")
    sel_zero("has_c2", black, "top4")
    nodes.extend(
        [
            helper.make_node("Concat", [black, "c00", "c01", "top3", "top4"], ["top"], axis=3),
        ]
    )
    sel_a_zero_when_false("no_c2", "c02", "c01", "r0x3")
    nodes.extend(
        [
            helper.make_node("Concat", ["c00", "c00", "c01", "r0x3", "c02"], ["er0"], axis=3),
        ]
    )
    sel_a_zero_when_false("no_c2", "c12", "c11", "r1x3")
    nodes.extend(
        [
            helper.make_node("Concat", ["c10", "c10", "c11", "r1x3", "c12"], ["er1"], axis=3),
        ]
    )
    sel_a_zero_when_false("no_c2", "c22", "c21", "r2x3")
    nodes.append(helper.make_node("And", ["no_r2", black], ["bot1x0"]))
    nodes.append(helper.make_node("And", ["no_r2", "c10"], ["bot1x1"]))
    nodes.append(helper.make_node("And", ["no_r2", "c11"], ["bot1x2"]))
    nodes.append(helper.make_node("And", ["no_r2", "has_c2"], ["no_r2_has_c2"]))
    nodes.append(helper.make_node("And", ["no_r2_has_c2", "c12"], ["bot1x3t"]))
    nodes.append(helper.make_node("And", ["no_r2", "no_c2"], ["no_r2_no_c2"]))
    nodes.append(helper.make_node("And", ["no_r2_no_c2", black], ["bot1x3f"]))
    nodes.append(helper.make_node("Or", ["bot1x3t", "bot1x3f"], ["bot1x3"]))
    nodes.append(helper.make_node("And", ["no_r2_has_c2", black], ["bot1x4"]))
    nodes.append(helper.make_node("Or", ["c20", "bot1x0"], ["row3x0"]))
    nodes.append(helper.make_node("Or", ["c20", "bot1x1"], ["row3x1"]))
    nodes.append(helper.make_node("Or", ["c21", "bot1x2"], ["row3x2"]))
    nodes.append(helper.make_node("Or", ["r2x3", "bot1x3"], ["row3x3"]))
    nodes.append(helper.make_node("Or", ["c22", "bot1x4"], ["row3x4"]))
    nodes.extend(
        [
            helper.make_node("Concat", ["row3x0", "row3x1", "row3x2", "row3x3", "row3x4"], ["row3"], axis=3),
        ]
    )
    nodes.append(helper.make_node("And", ["has_r2", black], ["bot2x0"]))
    nodes.append(helper.make_node("And", ["has_r2", "no_c2"], ["has_r2_no_c2"]))
    nodes.append(helper.make_node("And", ["has_r2_no_c2", black], ["bot2x3f"]))
    nodes.append(helper.make_node("Or", ["c22", "bot2x3f"], ["bot2x3"]))
    nodes.append(helper.make_node("And", ["has_r2", "has_c2"], ["has_r2_c2"]))
    nodes.append(helper.make_node("And", ["has_r2_c2", black], ["bot2x4"]))
    nodes.extend(
        [
            helper.make_node("Concat", ["bot2x0", "c20", "c21", "bot2x3", "bot2x4"], ["bot2"], axis=3),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Concat", ["top", "er0", "er1", "row3", "bot2"], ["smallb"], axis=2),
            helper.make_node("Cast", ["smallb"], ["smallf"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["smallf"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, 25, 25]),
        ]
    )

    graph = helper.make_graph(nodes, "task114_bool_concat", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def main() -> None:
    model = build_bool_concat_model()
    onnx.save(model, BEST_PATH)
    print(BEST_PATH)


if __name__ == "__main__":
    main()
