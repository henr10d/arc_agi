"""Compact ONNX generator for NeuroGolf task150.

Task rule: the ARC grid is an NxN square in the top-left of the competition
one-hot tensor, with N varying across examples. The output is the same grid
mirrored horizontally: each row is reversed left-to-right, and all padding
outside the NxN grid remains zero.

The best graph reduces the input to a 30-column occupancy vector and uses
ArgMin to find the first zero padding column as the active width N. It then
builds a 30-element int32 gather index vector ``[N-1, ..., 0, -1, -2, ...]``;
the negative indices still point into the zero-padded region, so no explicit
padding mask is needed. The final Gather writes directly to the graph output,
so the large flipped tensor is excluded from NeuroGolf activation memory.
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


TASK_NUM = "150"
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

    def init(self, name: str, array: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(array, name))
        return name

    def init_i64(self, name: str, values: list[int] | np.ndarray) -> str:
        return self.init(name, np.asarray(values, dtype=np.int64))

    def init_f32(self, name: str, values: list[float] | np.ndarray) -> str:
        return self.init(name, np.asarray(values, dtype=np.float32))

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


def build_dynamic_width_gather() -> onnx.ModelProto:
    """One final Gather, with the reversal indices computed from active width."""
    b = Builder()
    b.init_f32("zero_f", [0.0])
    b.init_i64("cols", list(range(30)))
    b.init_i64("one_i", [1])

    col_counts = b.node("ReduceSum", [IN_NAME], "col_counts", axes=[0, 1, 2], keepdims=0)
    active_cols = b.node("Greater", [col_counts, "zero_f"], "active_cols")
    active_cols_f = b.node("Cast", [active_cols], "active_cols_f", to=TensorProto.FLOAT)
    width_f = b.node("ReduceSum", [active_cols_f], "width_f", axes=[0], keepdims=0)
    width_i = b.node("Cast", [width_f], "width_i", to=TensorProto.INT64)
    in_range = b.node("Less", ["cols", width_i], "in_range")
    width_minus_one = b.node("Sub", [width_i, "one_i"], "width_minus_one")
    reversed_cols = b.node("Sub", [width_minus_one, "cols"], "reversed_cols")
    gather_cols = b.node("Where", [in_range, reversed_cols, width_i], "gather_cols")
    b.node("Gather", [IN_NAME, gather_cols], OUT_NAME, axis=3)
    return make_model(b.nodes, b.initializers)


def build_argmax_width_gather() -> onnx.ModelProto:
    """Use the first padding column as N, then gather directly to output."""
    b = Builder()
    b.init_f32("one_f", [1.0])
    b.init_i64("cols", list(range(30)))
    b.init_i64("one_i", [1])

    col_counts = b.node("ReduceSum", [IN_NAME], "col_counts", axes=[0, 1, 2], keepdims=0)
    padding_cols = b.node("Less", [col_counts, "one_f"], "padding_cols")
    padding_cols_f = b.node("Cast", [padding_cols], "padding_cols_f", to=TensorProto.FLOAT)
    width_i = b.node("ArgMax", [padding_cols_f], "width_i", axis=0, keepdims=0)
    in_range = b.node("Less", ["cols", width_i], "in_range")
    width_minus_one = b.node("Sub", [width_i, "one_i"], "width_minus_one")
    reversed_cols = b.node("Sub", [width_minus_one, "cols"], "reversed_cols")
    gather_cols = b.node("Where", [in_range, reversed_cols, width_i], "gather_cols")
    b.node("Gather", [IN_NAME, gather_cols], OUT_NAME, axis=3)
    return make_model(b.nodes, b.initializers)


def build_argmax_u8_i32_width_gather() -> onnx.ModelProto:
    """Same direct gather as argmax_width_gather, with compact mask/index tensors."""
    b = Builder()
    b.init_f32("one_f", [1.0])
    b.init("cols", np.arange(30, dtype=np.int32))
    b.init("one_i", np.asarray([1], dtype=np.int32))

    col_counts = b.node("ReduceSum", [IN_NAME], "col_counts", axes=[0, 1, 2], keepdims=0)
    padding_cols = b.node("Less", [col_counts, "one_f"], "padding_cols")
    padding_cols_u8 = b.node("Cast", [padding_cols], "padding_cols_u8", to=TensorProto.UINT8)
    width_i64 = b.node("ArgMax", [padding_cols_u8], "width_i64", axis=0, keepdims=0)
    width_i = b.node("Cast", [width_i64], "width_i", to=TensorProto.INT32)
    in_range = b.node("Less", ["cols", width_i], "in_range")
    width_minus_one = b.node("Sub", [width_i, "one_i"], "width_minus_one")
    reversed_cols = b.node("Sub", [width_minus_one, "cols"], "reversed_cols")
    gather_cols = b.node("Where", [in_range, reversed_cols, width_i], "gather_cols")
    b.node("Gather", [IN_NAME, gather_cols], OUT_NAME, axis=3)
    return make_model(b.nodes, b.initializers)


def build_argmin_negative_width_gather() -> onnx.ModelProto:
    """Use first zero column as N and negative Gather indices for padded outputs."""
    b = Builder()
    b.init("cols_plus_one", np.arange(1, 31, dtype=np.int32))

    col_counts = b.node("ReduceSum", [IN_NAME], "col_counts", axes=[0, 1, 2], keepdims=0)
    width_i64 = b.node("ArgMin", [col_counts], "width_i64", axis=0, keepdims=0)
    width_i = b.node("Cast", [width_i64], "width_i", to=TensorProto.INT32)
    gather_cols = b.node("Sub", [width_i, "cols_plus_one"], "gather_cols")
    b.node("Gather", [IN_NAME, gather_cols], OUT_NAME, axis=3)
    return make_model(b.nodes, b.initializers)


def build_fixed_4_full_gather() -> onnx.ModelProto:
    """Fast 4x4-only candidate; kept in the benchmark to catch prompt drift."""
    b = Builder()
    b.init_i64("cols4", [3, 2, 1, 0] + [4] * 26)
    b.node("Gather", [IN_NAME, "cols4"], OUT_NAME, axis=3)
    return make_model(b.nodes, b.initializers)


def build_crop4_gather_pad() -> onnx.ModelProto:
    """The literal crop/gather/pad 4x4 approach from the prompt."""
    b = Builder()
    b.init_i64("starts", [0, 0])
    b.init_i64("ends", [4, 4])
    b.init_i64("axes", [2, 3])
    b.init_i64("cols4", [3, 2, 1, 0])
    crop = b.node("Slice", [IN_NAME, "starts", "ends", "axes"], "crop")
    flipped = b.node("Gather", [crop, "cols4"], "flipped", axis=3)
    b.nodes.append(
        helper.make_node(
            "Pad",
            [flipped],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 26, 26],
            value=0.0,
        )
    )
    return make_model(b.nodes, b.initializers)


def variants() -> list[Variant]:
    return [
        Variant("argmin_negative_width_gather", build_argmin_negative_width_gather),
        Variant("argmax_u8_i32_width_gather", build_argmax_u8_i32_width_gather),
        Variant("argmax_width_gather", build_argmax_width_gather),
        Variant("dynamic_width_gather", build_dynamic_width_gather),
        Variant("fixed_4_full_gather", build_fixed_4_full_gather),
        Variant("crop4_gather_pad", build_crop4_gather_pad),
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
    print(f"{'variant':<24} {'ok':<5} {'valid':<6} {'memory':>8} {'params':>7} {'cost':>8} {'score':>10}")
    for name, result in sorted(results.items(), key=lambda item: (item[0] != best_name, item[0])):
        score = result["score"]
        score_text = f"{score:.6f}" if isinstance(score, float) else "None"
        print(
            f"{name:<24} {str(result['correct']):<5} {str(result['valid']):<6} "
            f"{str(result['memory']):>8} {str(result['params']):>7} "
            f"{str(result['cost']):>8} {score_text:>10}"
        )
        if not result["correct"]:
            print(f"  splits: {result['splits']}")
        if result["error"]:
            print(f"  error: {str(result['error']).strip()}")
    print(f"best: {best_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and benchmark task150 ONNX variants.")
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
