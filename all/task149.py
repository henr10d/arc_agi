"""Compact ONNX generator for NeuroGolf task149.

Task rule: the 11x11 input is split by cyan divider rows/columns at indices
3 and 7 into a 3x3 layout of 3x3 regions. For each region, output blue if
that region contains at least two magenta pixels, otherwise output black. The
produced 3x3 ARC output is padded to the required NeuroGolf 30x30 one-hot
tensor.

The best graph uses one sparse-in-value Conv over the full input to count
magenta pixels in each 3x3 stride-4 window, crops the needed top-left 3x3
counts, shifts the count so one magenta is negative and two magentas are
positive, negates that signed score for the black channel, then pads the
2-channel 3x3 result directly into graph output. The decoder only checks
whether each output value is greater than zero, so negative inactive channels
are valid and avoid a separate boolean threshold/Cast path.
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


TASK_NUM = "149"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
IR_VERSION = 10


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

    def init_f32(self, name: str, values: list[float]) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.float32), name))
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


def pad_2ch_3x3_to_output(b: Builder, x: str) -> None:
    b.nodes.append(
        helper.make_node(
            "Pad",
            [x],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 8, 27, 27],
            value=0.0,
        )
    )


def conv_kernel() -> np.ndarray:
    weight = np.zeros((1, 10, 3, 3), dtype=np.float32)
    weight[0, 6, :, :] = 1.0
    return weight


def build_avgpool_threshold() -> onnx.ModelProto:
    """Slice magenta 11x11, AveragePool regions, threshold count >= 2, pad."""
    b = Builder()
    b.init_i64("starts", [6, 0, 0])
    b.init_i64("ends", [7, 11, 11])
    b.init_i64("axes", [1, 2, 3])
    b.init_f32("one_pixel", [0.12])

    magenta = b.node("Slice", [IN_NAME, "starts", "ends", "axes"], "magenta")
    density = b.node("AveragePool", [magenta], "density", kernel_shape=[3, 3], strides=[4, 4])
    blue_b = b.node("Greater", [density, "one_pixel"], "blue_b")
    black_b = b.node("Not", [blue_b], "black_b")
    two_b = b.node("Concat", [black_b, blue_b], "two_b", axis=1)
    two_channels = b.node("Cast", [two_b], "two_channels", to=TensorProto.FLOAT)
    pad_2ch_3x3_to_output(b, two_channels)
    return make_model(b.nodes, b.initializers)


def build_full_conv_threshold() -> onnx.ModelProto:
    """Use a sparse-valued 10-channel Conv and signed logits for the two colors."""
    b = Builder()
    b.initializers.append(numpy_helper.from_array(conv_kernel(), "weight"))
    b.init_f32("bias", [-1.5])
    b.init_i64("starts", [0, 0])
    b.init_i64("ends", [3, 3])
    b.init_i64("axes", [2, 3])

    blue7 = b.node("Conv", [IN_NAME, "weight", "bias"], "blue7", kernel_shape=[3, 3], strides=[4, 4])
    blue = b.node("Slice", [blue7, "starts", "ends", "axes"], "blue")
    black = b.node("Neg", [blue], "black")
    two_channels = b.node("Concat", [black, blue], "two_channels", axis=1)
    pad_2ch_3x3_to_output(b, two_channels)
    return make_model(b.nodes, b.initializers)


def build_maxpool_any_magenta() -> onnx.ModelProto:
    """User-prompt rule variant: blue for any magenta; expected to fail data."""
    b = Builder()
    b.init_i64("starts", [6, 0, 0])
    b.init_i64("ends", [7, 11, 11])
    b.init_i64("axes", [1, 2, 3])

    magenta = b.node("Slice", [IN_NAME, "starts", "ends", "axes"], "magenta")
    blue_f = b.node("MaxPool", [magenta], "blue_f", kernel_shape=[3, 3], strides=[4, 4])
    blue_b = b.node("Cast", [blue_f], "blue_b", to=TensorProto.BOOL)
    black_b = b.node("Not", [blue_b], "black_b")
    two_b = b.node("Concat", [black_b, blue_b], "two_b", axis=1)
    two_channels = b.node("Cast", [two_b], "two_channels", to=TensorProto.FLOAT)
    pad_2ch_3x3_to_output(b, two_channels)
    return make_model(b.nodes, b.initializers)


def variants() -> list[Variant]:
    return [
        Variant("full_conv_threshold", build_full_conv_threshold),
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
        tmp_path = Path(tmp)
        for name, model in built.items():
            ok, splits = verify_correct(model)
            path = tmp_path / f"{TASK_ID}.onnx"
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
    print(f"{'variant':<26} {'ok':<5} {'valid':<6} {'memory':>8} {'params':>7} {'cost':>8} {'score':>10}")
    for name, result in sorted(results.items(), key=lambda item: (item[0] != best_name, item[0])):
        score = result["score"]
        score_text = f"{score:.6f}" if isinstance(score, float) else "None"
        print(
            f"{name:<26} {str(result['correct']):<5} {str(result['valid']):<6} "
            f"{str(result['memory']):>8} {str(result['params']):>7} "
            f"{str(result['cost']):>8} {score_text:>10}"
        )
        if not result["correct"]:
            print(f"  splits: {result['splits']}")
        if result["error"]:
            print(f"  error: {str(result['error']).strip()}")
    print(f"best: {best_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and benchmark task149 ONNX variants.")
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
