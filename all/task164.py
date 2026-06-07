"""Minimal ONNX generator for NeuroGolf task164.

Task rule: the meaningful input is always the top-left 3x3 ARC grid. The
output is a 3x6 grid made by concatenating the input pattern with its
left-right mirror, so each row [a, b, c] becomes [a, b, c, c, b, a], then the
result is zero-padded to the required 30x30 competition tensor.

The best graph gathers the full 30-column input directly into the graph output:
columns 0..5 select input columns [0, 1, 2, 2, 1, 0], and columns 6..29 select
padded zero column 3. Rows 3..29 are already all-zero padding, so this single
Gather produces the full output with no counted activation memory; only the 30
column-index parameters are charged.
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


TASK_NUM = "164"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
I64_MIN = -9223372036854775808


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
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def pad_to_output(b: Builder, x: str) -> None:
    b.nodes.append(
        helper.make_node(
            "Pad",
            [x],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 27, 24],
            value=0.0,
        )
    )


def build_full_col_gather() -> onnx.ModelProto:
    """Best candidate: one full-width Gather writes directly to graph output."""
    b = Builder()
    b.init_i64("col_indices", [0, 1, 2, 2, 1, 0] + [3] * 24)
    b.nodes.append(helper.make_node("Gather", [IN_NAME, "col_indices"], [OUT_NAME], axis=3))
    return make_model(b.nodes, b.initializers)


def build_gather_flip() -> onnx.ModelProto:
    """Requested variant: crop, Gather [2,1,0] for the flipped half, concat, pad."""
    b = Builder()
    b.init_i64("crop_starts", [0, 0])
    b.init_i64("crop_ends", [3, 3])
    b.init_i64("crop_axes", [2, 3])
    b.init_i64("flip_indices", [2, 1, 0])

    x = b.node("Slice", [IN_NAME, "crop_starts", "crop_ends", "crop_axes"], "x")
    x_flip = b.node("Gather", [x, "flip_indices"], "x_flip", axis=3)
    y_small = b.node("Concat", [x, x_flip], "y_small", axis=3)
    pad_to_output(b, y_small)
    return make_model(b.nodes, b.initializers)


def build_slice_negative_flip() -> onnx.ModelProto:
    """Requested variant: crop, negative-step Slice for the flipped half, concat, pad."""
    b = Builder()
    b.init_i64("crop_starts", [0, 0])
    b.init_i64("crop_ends", [3, 3])
    b.init_i64("crop_axes", [2, 3])
    b.init_i64("flip_starts", [2])
    b.init_i64("flip_ends", [I64_MIN])
    b.init_i64("flip_axes", [3])
    b.init_i64("flip_steps", [-1])

    x = b.node("Slice", [IN_NAME, "crop_starts", "crop_ends", "crop_axes"], "x")
    x_flip = b.node("Slice", [x, "flip_starts", "flip_ends", "flip_axes", "flip_steps"], "x_flip")
    y_small = b.node("Concat", [x, x_flip], "y_small", axis=3)
    pad_to_output(b, y_small)
    return make_model(b.nodes, b.initializers)


def build_direct_small_gather() -> onnx.ModelProto:
    """Requested variant: crop to 3x3, Gather [0,1,2,2,1,0], then pad."""
    b = Builder()
    b.init_i64("crop_starts", [0, 0])
    b.init_i64("crop_ends", [3, 3])
    b.init_i64("crop_axes", [2, 3])
    b.init_i64("six_cols", [0, 1, 2, 2, 1, 0])

    x = b.node("Slice", [IN_NAME, "crop_starts", "crop_ends", "crop_axes"], "x")
    y_small = b.node("Gather", [x, "six_cols"], "y_small", axis=3)
    pad_to_output(b, y_small)
    return make_model(b.nodes, b.initializers)


def variants() -> list[Variant]:
    return [
        Variant("full_col_gather", build_full_col_gather),
        Variant("gather_flip", build_gather_flip),
        Variant("slice_negative_flip", build_slice_negative_flip),
        Variant("direct_small_gather", build_direct_small_gather),
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
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
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
    print(f"{'variant':<22} {'ok':<5} {'valid':<6} {'memory':>8} {'params':>7} {'cost':>8} {'score':>10}")
    for name, result in sorted(results.items(), key=lambda item: (item[0] != best_name, item[0])):
        score = result["score"]
        score_text = f"{score:.6f}" if isinstance(score, float) else "None"
        print(
            f"{name:<22} {str(result['correct']):<5} {str(result['valid']):<6} "
            f"{str(result['memory']):>8} {str(result['params']):>7} "
            f"{str(result['cost']):>8} {score_text:>10}"
        )
        if not result["correct"]:
            print(f"  splits: {result['splits']}")
        if result["error"]:
            print(f"  error: {str(result['error']).strip()}")
    print(f"best: {best_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and benchmark task164 ONNX variants.")
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
