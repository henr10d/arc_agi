"""Dynamic ONNX generator for NeuroGolf task156.

Task rule: the 10x10 input contains exactly two solid color-4 rectangles on a
color-0 background. The rectangles are separated by at least one blank row.
Keep every original rectangle border unchanged, recolor only the strict
interior of the smaller rectangle to blue (1), and recolor only the strict
interior of the larger rectangle to red (2). The competition tensor is padded
to the required 30x30 one-hot output.

The graph detects the top occupied row run, uses row sums to compare top and
bottom rectangle areas, detects strict interiors with a 3x3 AveragePool, then
builds compact 10x10 boolean output channels before a final cast and pad.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402


TASK_NUM = "156"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
IR_VERSION = 10


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []

    def init(self, name: str, values: Any, dtype: np.dtype[Any]) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=dtype), name))
        return name

    def init_i64(self, name: str, values: Any) -> str:
        return self.init(name, values, np.int64)

    def init_f32(self, name: str, values: Any) -> str:
        return self.init(name, values, np.float32)

    def node(self, op_type: str, inputs: list[str], output: str, **attrs: Any) -> str:
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        return output


def make_model(nodes: list[onnx.NodeProto], initializers: list[onnx.TensorProto]) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 10)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def add_end_of_first_run(b: Builder, x_bool: str, prefix: str) -> str:
    curr = b.node("Slice", [x_bool, "s0", "s9", "ax2"], f"{prefix}_curr")
    nxt = b.node("Slice", [x_bool, "s1", "s10", "ax2"], f"{prefix}_next")
    end_b = b.node("And", [curr, b.node("Not", [nxt], f"{prefix}_not_next")], f"{prefix}_end_b")
    end_u8 = b.node("Cast", [end_b], f"{prefix}_end_u8", to=TensorProto.UINT8)
    return b.node("ArgMax", [end_u8], f"{prefix}_end", axis=2, keepdims=1)


def build_model() -> onnx.ModelProto:
    b = Builder()

    b.init_i64("starts4", [0, 4, 0, 0])
    b.init_i64("ends4", [1, 5, 10, 10])
    b.init_i64("axes4", [0, 1, 2, 3])
    b.init_i64("s0", [0])
    b.init_i64("s1", [1])
    b.init_i64("s9", [9])
    b.init_i64("s10", [10])
    b.init_i64("ax2", [2])
    b.init_i64("one_i64", np.array([[[1]]]))
    b.init_i64("row_coords", np.arange(10).reshape(1, 1, 10, 1))
    b.init_f32("zero_f", np.array(0.0))
    b.init_f32("almost_one_f", np.array(0.99))

    color4 = b.node("Slice", [IN_NAME, "starts4", "ends4", "axes4"], "color4")
    obj_b = b.node("Greater", [color4, "zero_f"], "obj_b")
    row_sum = b.node("ReduceSum", [color4], "row_sum", axes=[3], keepdims=0)
    row_has = b.node("Greater", [row_sum, "zero_f"], "row_has")
    row_u8 = b.node("Cast", [row_has], "row_u8", to=TensorProto.UINT8)

    top_min = b.node("ArgMax", [row_u8], "top_min", axis=2, keepdims=1)
    top_max = add_end_of_first_run(b, row_has, "top")

    top_min_prev = b.node("Sub", [top_min, "one_i64"], "top_min_prev")
    top_max_next = b.node("Add", [top_max, "one_i64"], "top_max_next")
    top_rows_lo = b.node("Greater", ["row_coords", top_min_prev], "top_rows_lo")
    top_rows_hi = b.node("Less", ["row_coords", top_max_next], "top_rows_hi")
    top_rows = b.node("And", [top_rows_lo, top_rows_hi], "top_rows")
    top_rows_flat = b.node("Squeeze", [top_rows], "top_rows_flat", axes=[3])
    top_rows_f = b.node("Cast", [top_rows_flat], "top_rows_f", to=TensorProto.FLOAT)
    top_row_sum = b.node("Mul", [row_sum, top_rows_f], "top_row_sum")
    top_area = b.node("ReduceSum", [top_row_sum], "top_area", axes=[2], keepdims=1)
    total_area = b.node("ReduceSum", [row_sum], "total_area", axes=[2], keepdims=1)
    bottom_area = b.node("Sub", [total_area, top_area], "bottom_area")
    top_smaller = b.node("Less", [top_area, bottom_area], "top_smaller")

    interior_avg = b.node(
        "AveragePool",
        [color4],
        "interior_avg",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        count_include_pad=1,
    )
    interior = b.node("Greater", [interior_avg, "almost_one_f"], "interior")
    bg = b.node("Not", [obj_b], "bg")
    false_ch = b.node("And", [bg, obj_b], "false_ch")
    rect_border = b.node("Xor", [obj_b, interior], "rect_border")
    blue_rows = b.node("Not", [b.node("Xor", [top_rows, top_smaller], "blue_rows_xor")], "blue_rows")
    blue = b.node("And", [interior, blue_rows], "blue")
    red = b.node("Xor", [interior, blue], "red")
    small_bool = b.node(
        "Concat",
        [bg, blue, red, false_ch, rect_border],
        "small_bool",
        axis=1,
    )
    small_float = b.node("Cast", [small_bool], "small_float", to=TensorProto.FLOAT)
    b.nodes.append(
        helper.make_node(
            "Pad",
            [small_float],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 5, 20, 20],
            value=0.0,
        )
    )
    return make_model(b.nodes, b.initializers)


def load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def verify_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    all_ok = True
    splits: dict[str, tuple[int, int]] = {}
    for split, examples in load_task_data().items():
        passed = 0
        checked = 0
        for example in examples:
            inp = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if inp is None or expected is None:
                continue
            pred = session.run([OUT_NAME], {IN_NAME: inp})[0]
            checked += 1
            if np.array_equal(pred > 0.0, expected > 0.0):
                passed += 1
            else:
                all_ok = False
        splits[split] = (passed, checked)
    return all_ok, splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    model = build_model()
    ok, splits = verify_correct(model)
    if not ok:
        raise SystemExit(f"verification failed: {splits}")

    with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}_") as tmp:
        tmp_path = Path(tmp) / f"{TASK_ID}.onnx"
        onnx.save(model, str(tmp_path))
        result = score_file(tmp_path)

    if not result["valid"]:
        raise SystemExit(f"scoring failed: {result}")

    if not args.check_only:
        onnx.save(model, str(BEST_PATH))

    print(f"verified: {splits}")
    print(
        "score: "
        f"memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} points={result['score']:.6f}"
    )
    if not args.check_only:
        print(f"wrote {BEST_PATH}")


if __name__ == "__main__":
    main()
