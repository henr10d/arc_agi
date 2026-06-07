"""Compact ONNX generator for NeuroGolf task147.

Task rule: every green cell (color 3) with at least one orthogonal green
neighbor becomes light blue (color 8); isolated green cells remain green and
black background remains black. Diagonal contact does not count.

The task examples fit within the top-left 6x6 area of the NeuroGolf 30x30
one-hot tensor. The best variant computes only that compact green crop, then
uses a final Scatter on the original full input to overwrite channels 3 and 8;
the full-grid Scatter output is the graph output and is excluded from scored
activation memory.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402


TASK_NUM = "147"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
CROP = 6


@dataclass(frozen=True)
class Variant:
    name: str
    build: Callable[[], onnx.ModelProto]


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []

    def init_i64(self, name: str, values: list[int]) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name))
        return name

    def init_f32(self, name: str, values: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(values.astype(np.float32), name))
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
        opset_imports=[helper.make_opsetid("", 10)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def crop_channel(b: Builder, channel: int, name: str) -> str:
    b.init_i64(f"{name}_starts", [0, channel, 0, 0])
    b.init_i64(f"{name}_ends", [1, channel + 1, CROP, CROP])
    return b.node("Slice", [IN_NAME, f"{name}_starts", f"{name}_ends"], name)


def slice_axis(b: Builder, x: str, name: str, axis: int, start: int, end: int) -> str:
    b.init_i64(f"{name}_starts", [start])
    b.init_i64(f"{name}_ends", [end])
    b.init_i64(f"{name}_axes", [axis])
    return b.node("Slice", [x, f"{name}_starts", f"{name}_ends", f"{name}_axes"], name)


def pad_to_output(b: Builder, compact_float: str) -> None:
    b.node(
        "Pad",
        [compact_float],
        OUT_NAME,
        mode="constant",
        pads=[0, 0, 0, 0, 0, 0, 30 - CROP, 30 - CROP],
        value=0.0,
    )


def emit_output_from_masks(b: Builder, bg: str, green: str, adjacent: str) -> None:
    isolated = b.node("And", [green, b.node("Not", [adjacent], "not_adjacent")], "isolated")
    zero = b.node("And", [green, b.node("Not", [green], "not_green")], "zero")
    small_bool = b.node(
        "Concat",
        [bg, zero, zero, isolated, zero, zero, zero, zero, adjacent, zero],
        "small_bool",
        axis=1,
    )
    small_float = b.node("Cast", [small_bool], "small_float", to=TensorProto.FLOAT)
    pad_to_output(b, small_float)


def build_bool_shift() -> onnx.ModelProto:
    b = Builder()
    bg = b.node("Cast", [crop_channel(b, 0, "bg_f")], "bg", to=TensorProto.BOOL)
    green = b.node("Cast", [crop_channel(b, 3, "green_f")], "green", to=TensorProto.BOOL)
    not_green = b.node("Not", [green], "shift_not_green")
    zero = b.node("And", [green, not_green], "shift_zero")
    zero_row = slice_axis(b, zero, "zero_row", 2, 0, 1)
    zero_col = slice_axis(b, zero, "zero_col", 3, 0, 1)

    up = b.node("Concat", [zero_row, slice_axis(b, green, "up_core", 2, 0, CROP - 1)], "up", axis=2)
    down = b.node("Concat", [slice_axis(b, green, "down_core", 2, 1, CROP), zero_row], "down", axis=2)
    left = b.node("Concat", [zero_col, slice_axis(b, green, "left_core", 3, 0, CROP - 1)], "left", axis=3)
    right = b.node("Concat", [slice_axis(b, green, "right_core", 3, 1, CROP), zero_col], "right", axis=3)

    v = b.node("Or", [up, down], "v")
    h = b.node("Or", [left, right], "h")
    neighbor = b.node("Or", [v, h], "neighbor")
    adjacent = b.node("And", [green, neighbor], "adjacent")
    emit_output_from_masks(b, bg, green, adjacent)
    return make_model(b.nodes, b.initializers)


def build_conv() -> onnx.ModelProto:
    b = Builder()
    bg = b.node("Cast", [crop_channel(b, 0, "bg_f")], "bg", to=TensorProto.BOOL)
    green_f = crop_channel(b, 3, "green_f")
    green = b.node("Cast", [green_f], "green", to=TensorProto.BOOL)
    kernel = np.asarray([[[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]], dtype=np.float32)
    b.init_f32("kernel", kernel)
    b.init_f32("zero_f", np.asarray([0.0], dtype=np.float32))
    count = b.node("Conv", [green_f, "kernel"], "count", pads=[1, 1, 1, 1])
    neighbor = b.node("Greater", [count, "zero_f"], "neighbor")
    adjacent = b.node("And", [green, neighbor], "adjacent")
    emit_output_from_masks(b, bg, green, adjacent)
    return make_model(b.nodes, b.initializers)


def build_conv_float_out() -> onnx.ModelProto:
    b = Builder()
    bg = crop_channel(b, 0, "bg_f")
    green_f = crop_channel(b, 3, "green_f")
    green = b.node("Cast", [green_f], "green", to=TensorProto.BOOL)
    kernel = np.asarray([[[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]], dtype=np.float32)
    b.init_f32("kernel", kernel)
    b.init_f32("zero_f", np.asarray([0.0], dtype=np.float32))
    count = b.node("Conv", [green_f, "kernel"], "count", pads=[1, 1, 1, 1])
    neighbor = b.node("Greater", [count, "zero_f"], "neighbor")
    adjacent = b.node("And", [green, neighbor], "adjacent")
    adjacent_f = b.node("Cast", [adjacent], "adjacent_f", to=TensorProto.FLOAT)
    isolated = b.node("Sub", [green_f, adjacent_f], "isolated")
    zero = b.node("Mul", [green_f, "zero_f"], "zero")
    small_float = b.node(
        "Concat",
        [bg, zero, zero, isolated, zero, zero, zero, zero, adjacent_f, zero],
        "small_float",
        axis=1,
    )
    pad_to_output(b, small_float)
    return make_model(b.nodes, b.initializers)


def build_conv_center_threshold() -> onnx.ModelProto:
    b = Builder()
    bg = crop_channel(b, 0, "bg_f")
    green_f = crop_channel(b, 3, "green_f")
    kernel = np.asarray([[[[0, 1, 0], [1, 4, 1], [0, 1, 0]]]], dtype=np.float32)
    b.init_f32("kernel", kernel)
    b.init_f32("four_f", np.asarray([4.0], dtype=np.float32))
    score = b.node("Conv", [green_f, "kernel"], "score", pads=[1, 1, 1, 1])
    adjacent = b.node("Greater", [score, "four_f"], "adjacent")
    adjacent_f = b.node("Cast", [adjacent], "adjacent_f", to=TensorProto.FLOAT)
    isolated = b.node("Sub", [green_f, adjacent_f], "isolated")
    zero = b.node("Sub", [green_f, green_f], "zero")
    small_float = b.node(
        "Concat",
        [bg, zero, zero, isolated, zero, zero, zero, zero, adjacent_f, zero],
        "small_float",
        axis=1,
    )
    pad_to_output(b, small_float)
    return make_model(b.nodes, b.initializers)


def build_scatter_update() -> onnx.ModelProto:
    b = Builder()
    green_f = crop_channel(b, 3, "green_f")
    kernel = np.asarray([[[[0, 1, 0], [1, 4, 1], [0, 1, 0]]]], dtype=np.float32)
    b.init_f32("kernel", kernel)
    b.init_f32("four_f", np.asarray([4.0], dtype=np.float32))
    score = b.node("Conv", [green_f, "kernel"], "score", pads=[1, 1, 1, 1])
    adjacent = b.node("Greater", [score, "four_f"], "adjacent")
    adjacent_f = b.node("Cast", [adjacent], "adjacent_f", to=TensorProto.FLOAT)
    isolated = b.node("Sub", [green_f, adjacent_f], "isolated")
    updates = b.node("Concat", [isolated, adjacent_f], "updates", axis=1)
    indices = np.empty((1, 2, CROP, CROP), dtype=np.int32)
    indices[:, 0, :, :] = 3
    indices[:, 1, :, :] = 8
    b.initializers.append(numpy_helper.from_array(indices, "scatter_indices"))
    b.node("Scatter", [IN_NAME, "scatter_indices", updates], OUT_NAME, axis=1)
    return make_model(b.nodes, b.initializers)


def build_float_shift() -> onnx.ModelProto:
    b = Builder()
    bg = b.node("Cast", [crop_channel(b, 0, "bg_f")], "bg", to=TensorProto.BOOL)
    green_f = crop_channel(b, 3, "green_f")
    green = b.node("Cast", [green_f], "green", to=TensorProto.BOOL)
    b.init_f32("zero_f", np.asarray([0.0], dtype=np.float32))

    up = b.node("Pad", [slice_axis(b, green_f, "up_core", 2, 0, CROP - 1)], "up", pads=[0, 0, 1, 0, 0, 0, 0, 0])
    down = b.node("Pad", [slice_axis(b, green_f, "down_core", 2, 1, CROP)], "down", pads=[0, 0, 0, 0, 0, 0, 1, 0])
    left = b.node("Pad", [slice_axis(b, green_f, "left_core", 3, 0, CROP - 1)], "left", pads=[0, 0, 0, 1, 0, 0, 0, 0])
    right = b.node("Pad", [slice_axis(b, green_f, "right_core", 3, 1, CROP)], "right", pads=[0, 0, 0, 0, 0, 0, 0, 1])

    v = b.node("Add", [up, down], "v")
    h = b.node("Add", [left, right], "h")
    count = b.node("Add", [v, h], "count")
    neighbor = b.node("Greater", [count, "zero_f"], "neighbor")
    adjacent = b.node("And", [green, neighbor], "adjacent")
    emit_output_from_masks(b, bg, green, adjacent)
    return make_model(b.nodes, b.initializers)


def variants() -> list[Variant]:
    return [
        Variant("bool_shift", build_bool_shift),
        Variant("conv", build_conv),
        Variant("conv_center_threshold", build_conv_center_threshold),
        Variant("conv_float_out", build_conv_float_out),
        Variant("scatter_update", build_scatter_update),
        Variant("float_shift", build_float_shift),
    ]


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


def write_model(model: onnx.ModelProto, path: Path) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(path))


def benchmark_variants() -> tuple[dict[str, dict[str, Any]], str, dict[str, onnx.ModelProto]]:
    built = {variant.name: variant.build() for variant in variants()}
    results: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}_") as tmp:
        path = Path(tmp) / f"{TASK_ID}.onnx"
        for name, model in built.items():
            ok, splits = verify_correct(model)
            write_model(model, path)
            result = score_file(path)
            result["correct"] = ok
            result["splits"] = splits
            results[name] = result

    def sort_key(item: tuple[str, dict[str, Any]]) -> int:
        result = item[1]
        if not result["valid"] or not result["correct"]:
            return 10**18
        return int(result["cost"])

    best_name, best_result = min(results.items(), key=sort_key)
    if sort_key((best_name, best_result)) >= 10**18:
        raise RuntimeError(f"no valid correct variant: {results}")
    return results, best_name, built


def print_benchmark(results: dict[str, dict[str, Any]], best_name: str) -> None:
    print(f"{'variant':<16} {'ok':<5} {'valid':<6} {'memory':>8} {'params':>7} {'cost':>8} {'score':>10}")
    for name, result in sorted(results.items(), key=lambda item: (item[0] != best_name, item[0])):
        score = result["score"]
        score_text = f"{score:.6f}" if isinstance(score, float) else "None"
        print(
            f"{name:<16} {str(result['correct']):<5} {str(result['valid']):<6} "
            f"{str(result['memory']):>8} {str(result['params']):>7} "
            f"{str(result['cost']):>8} {score_text:>10}"
        )
        if not result["correct"]:
            print(f"  splits: {result['splits']}")
        if result["error"]:
            print(f"  error: {str(result['error']).strip()}")
    print(f"best: {best_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and benchmark task147 ONNX variants.")
    parser.add_argument("--benchmark-only", action="store_true", help="print variant scores without writing best model")
    args = parser.parse_args()

    results, best_name, built = benchmark_variants()
    print_benchmark(results, best_name)
    if not args.benchmark_only:
        OUT_DIR.mkdir(exist_ok=True)
        write_model(built[best_name], BEST_PATH)
        print(f"wrote: {BEST_PATH}")


if __name__ == "__main__":
    main()
