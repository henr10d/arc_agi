"""Compact ONNX for NeuroGolf task298.

Task rule: the input square is a stack of nested square layers. Let A be the
color at the outer border cell (0, 0), B the color one layer in at (1, 1), and
C the color at (2, 2). Recolor every in-grid cell by the cycle A -> C,
B -> A, and C -> B. Black can be one of these rotating colors; only padded
cells outside the ARC grid remain all-zero.

The ONNX graph extracts the three role colors dynamically, detects whether the
grid is 6x6 or 8x8 from the one-hot padding, selects compact fixed spatial ring
masks, fills an 8x8 uint8 one-hot result, casts only that compact result to
float, then pads to the required 30x30 output.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_NUM = "298"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
CHANNELS = 10
FULL_SIZE = 30
LIVE_SIZE = 8
FULL_SHAPE = [1, CHANNELS, FULL_SIZE, FULL_SIZE]
IR_VERSION = 10
OPSET = 10


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self._seen: set[str] = set()

    def init_i64(self, name: str, values: list[int]) -> str:
        if name not in self._seen:
            self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name))
            self._seen.add(name)
        return name

    def init_u8_scalar(self, name: str, value: int) -> str:
        if name not in self._seen:
            self.initializers.append(numpy_helper.from_array(np.asarray(value, dtype=np.uint8), name))
            self._seen.add(name)
        return name

    def init_f32_scalar(self, name: str, value: float) -> str:
        if name not in self._seen:
            self.initializers.append(numpy_helper.from_array(np.asarray(value, dtype=np.float32), name))
            self._seen.add(name)
        return name

    def init_u8_mask(self, name: str, values: np.ndarray) -> str:
        if name not in self._seen:
            self.initializers.append(
                numpy_helper.from_array(values.astype(np.uint8).reshape(1, 1, LIVE_SIZE, LIVE_SIZE), name)
            )
            self._seen.add(name)
        return name

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
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def ring_masks(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return output masks for colors C, A, B in the recolored square."""
    yy, xx = np.indices((LIVE_SIZE, LIVE_SIZE))
    in_grid = (yy < size) & (xx < size)
    dist = np.minimum.reduce([yy, xx, size - 1 - yy, size - 1 - xx])

    # Input color A appears on the outer ring and, for 8x8 grids, the center 2x2.
    a_source = in_grid & ((dist == 0) | ((size == 8) & (dist == 3)))
    b_source = in_grid & (dist == 1)
    c_source = in_grid & (dist == 2)
    return a_source, b_source, c_source


def role_color(b: Builder, name: str, starts: str, ends: str) -> str:
    """Return one_hot_uint8_4d for one role position."""
    pos = b.node("Slice", [IN_NAME, starts, ends, "spatial_axes"], f"{name}_pos")
    return b.node("Cast", [pos], f"{name}_one_hot", to=TensorProto.UINT8)


def build_model() -> onnx.ModelProto:
    b = Builder()
    b.init_i64("s00", [0, 0])
    b.init_i64("s11", [1, 1])
    b.init_i64("s22", [2, 2])
    b.init_i64("s33", [3, 3])
    b.init_i64("s66", [6, 6])
    b.init_i64("s77", [7, 7])
    b.init_i64("spatial_axes", [2, 3])
    b.init_u8_scalar("zero_u8", 0)
    b.init_f32_scalar("zero_f32", 0.0)

    for suffix, size in (("6", 6), ("8", 8)):
        c_mask, a_mask, b_mask = ring_masks(size)
        b.init_u8_mask(f"c_mask{suffix}", c_mask)
        b.init_u8_mask(f"a_mask{suffix}", a_mask)
        b.init_u8_mask(f"b_mask{suffix}", b_mask)

    a_one_hot = role_color(b, "a", "s00", "s11")
    b_one_hot = role_color(b, "b", "s11", "s22")
    c_one_hot = role_color(b, "c", "s22", "s33")

    # Padding at (6, 6) is empty for 6x6 grids and one-hot for 8x8 grids.
    size_pos = b.node("Slice", [IN_NAME, "s66", "s77", "spatial_axes"], "size_pos")
    size_sum = b.node("ReduceSum", [size_pos], "size_sum", axes=[1], keepdims=1)
    is8 = b.node("Greater", [size_sum, "zero_f32"], "is8")

    c_mask_u8 = b.node("Where", [is8, "c_mask8", "c_mask6"], "c_mask_u8")
    a_mask_u8 = b.node("Where", [is8, "a_mask8", "a_mask6"], "a_mask_u8")
    b_mask_u8 = b.node("Where", [is8, "b_mask8", "b_mask6"], "b_mask_u8")
    c_mask = b.node("Cast", [c_mask_u8], "c_mask", to=TensorProto.BOOL)
    a_mask = b.node("Cast", [a_mask_u8], "a_mask", to=TensorProto.BOOL)
    b_mask = b.node("Cast", [b_mask_u8], "b_mask", to=TensorProto.BOOL)

    # Recolor source A->C, B->A, C->B over the selected ring masks.
    c_or_zero = b.node("Where", [c_mask, c_one_hot, "zero_u8"], "c_or_zero")
    a_or_c = b.node("Where", [a_mask, a_one_hot, c_or_zero], "a_or_c")
    y8u = b.node("Where", [b_mask, b_one_hot, a_or_c], "y8u")
    y8 = b.node("Cast", [y8u], "y8", to=TensorProto.FLOAT)

    pad = FULL_SIZE - LIVE_SIZE
    b.nodes.append(
        helper.make_node(
            "Pad",
            [y8],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, pad, pad],
            value=0.0,
        )
    )
    return make_model(b.nodes, b.initializers)


def load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def validate_rule_on_json() -> dict[str, tuple[int, int]]:
    results: dict[str, tuple[int, int]] = {}
    for split, examples in load_task_data().items():
        passed = 0
        for example in examples:
            grid = example["input"]
            a = grid[0][0]
            b = grid[1][1]
            c = grid[2][2]
            mapping = {a: c, b: a, c: b}
            pred = [[mapping[color] for color in row] for row in grid]
            passed += int(pred == example["output"])
        results[split] = (passed, len(examples))
    return results


def verify_model(model: onnx.ModelProto) -> dict[str, tuple[int, int]]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    results: dict[str, tuple[int, int]] = {}
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
            passed += int(np.array_equal(pred > 0.0, expected > 0.0))
        results[split] = (passed, checked)
    return results


def main() -> None:
    rule_results = validate_rule_on_json()
    if any(passed != total for passed, total in rule_results.values()):
        raise SystemExit(f"rule validation failed: {rule_results}")

    model = build_model()
    model_results = verify_model(model)
    if any(passed != total for passed, total in model_results.values()):
        raise SystemExit(f"model validation failed: {model_results}")

    onnx.save(model, BEST_PATH)
    result = score_file(BEST_PATH)

    print(f"wrote {BEST_PATH}")
    print(f"rule:  {rule_results}")
    print(f"model: {model_results}")
    print(
        "score: "
        f"memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']:.6f}"
    )


if __name__ == "__main__":
    main()
