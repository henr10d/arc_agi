"""Compact ONNX generator for NeuroGolf task129.

Task rule: the logical input is always a 3x3 ARC grid in the task data. Count
the occurrences of each color 0..9, choose the unique most frequent color, and
fill the whole 3x3 output with that majority color. Padding outside the 3x3
grid remains all-zero for the competition one-hot [1,10,30,30] tensor contract.

ONNX: the best variant reduces the one-hot input directly over the full H/W
axes. Padded cells are all-zero, so they do not affect the counts. ArgMax finds
the majority channel, OneHot reconstructs a single color vector, Expand tiles it
to 3x3, and the final Pad writes the required full-size output tensor.
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


TASK_NUM = "129"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
OPSET = 10


@dataclass(frozen=True)
class Variant:
    name: str
    build: Callable[[], onnx.ModelProto]


def _init(array: object, name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(array), name=name)


def _make_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        initializer=inits,
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


def _final_pad(nodes: list[onnx.NodeProto], small: str) -> None:
    nodes.append(
        helper.make_node(
            "Pad",
            [small],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 27, 27],
        )
    )


def build_direct_reduce_onehot() -> onnx.ModelProto:
    """Best candidate: full-grid ReduceSum, tiny OneHot, Expand to 3x3."""
    inits = [
        _init(np.asarray(10, dtype=np.int64), "depth"),
        _init(np.asarray([0.0, 1.0], dtype=np.float32), "values"),
        _init(np.asarray([1, 10, 3, 3], dtype=np.int64), "shape3"),
    ]
    nodes = [
        helper.make_node("ReduceSum", [IN_NAME], ["counts"], axes=[2, 3], keepdims=0),
        helper.make_node("ArgMax", ["counts"], ["color"], axis=1, keepdims=0),
        helper.make_node("OneHot", ["color", "depth", "values"], ["vec"], axis=1),
        helper.make_node("Unsqueeze", ["vec"], ["vec4"], axes=[2, 3]),
        helper.make_node("Expand", ["vec4", "shape3"], ["small"]),
    ]
    _final_pad(nodes, "small")
    return _make_model(nodes, inits, "task129_direct_reduce_onehot")


def build_direct_reduce_keepdims_onehot() -> onnx.ModelProto:
    """Best candidate: keep H/W singleton dims and avoid an Unsqueeze tensor."""
    inits = [
        _init(np.asarray(10, dtype=np.int64), "depth"),
        _init(np.asarray([0.0, 1.0], dtype=np.float32), "values"),
        _init(np.asarray([1, 10, 3, 3], dtype=np.int64), "shape3"),
    ]
    nodes = [
        helper.make_node("ReduceSum", [IN_NAME], ["counts"], axes=[2, 3], keepdims=1),
        helper.make_node("ArgMax", ["counts"], ["color"], axis=1, keepdims=0),
        helper.make_node("OneHot", ["color", "depth", "values"], ["vec4"], axis=1),
        helper.make_node("Expand", ["vec4", "shape3"], ["small"]),
    ]
    _final_pad(nodes, "small")
    return _make_model(nodes, inits, "task129_direct_reduce_keepdims_onehot")


def build_direct_reduce_equal_bool() -> onnx.ModelProto:
    """Bool reconstruction variant: lower tile memory, but needs final Cast."""
    inits = [
        _init(np.arange(10, dtype=np.int64).reshape(1, 10), "colors"),
        _init(np.asarray([1, 10, 3, 3], dtype=np.int64), "shape3"),
    ]
    nodes = [
        helper.make_node("ReduceSum", [IN_NAME], ["counts"], axes=[2, 3], keepdims=0),
        helper.make_node("ArgMax", ["counts"], ["color"], axis=1, keepdims=1),
        helper.make_node("Equal", ["color", "colors"], ["vec"]),
        helper.make_node("Unsqueeze", ["vec"], ["vec4"], axes=[2, 3]),
        helper.make_node("Expand", ["vec4", "shape3"], ["small_b"]),
    ]
    nodes.append(helper.make_node("Cast", ["small_b"], ["small"], to=TensorProto.FLOAT))
    _final_pad(nodes, "small")
    return _make_model(nodes, inits, "task129_direct_reduce_equal_bool")


def build_crop_reduce_onehot() -> onnx.ModelProto:
    """Requested crop variant: explicitly counts only the top-left 3x3 region."""
    inits = [
        _init(np.asarray([0, 0], dtype=np.int64), "starts"),
        _init(np.asarray([3, 3], dtype=np.int64), "ends"),
        _init(np.asarray([2, 3], dtype=np.int64), "axes"),
        _init(np.asarray(10, dtype=np.int64), "depth"),
        _init(np.asarray([0.0, 1.0], dtype=np.float32), "values"),
        _init(np.asarray([1, 10, 3, 3], dtype=np.int64), "shape3"),
    ]
    nodes = [
        helper.make_node("Slice", [IN_NAME, "starts", "ends", "axes"], ["crop"]),
        helper.make_node("ReduceSum", ["crop"], ["counts"], axes=[2, 3], keepdims=0),
        helper.make_node("ArgMax", ["counts"], ["color"], axis=1, keepdims=0),
        helper.make_node("OneHot", ["color", "depth", "values"], ["vec"], axis=1),
        helper.make_node("Unsqueeze", ["vec"], ["vec4"], axes=[2, 3]),
        helper.make_node("Expand", ["vec4", "shape3"], ["small"]),
    ]
    _final_pad(nodes, "small")
    return _make_model(nodes, inits, "task129_crop_reduce_onehot")


def variants() -> list[Variant]:
    return [
        Variant("direct_reduce_keepdims_onehot", build_direct_reduce_keepdims_onehot),
        Variant("direct_reduce_onehot", build_direct_reduce_onehot),
        Variant("direct_reduce_equal_bool", build_direct_reduce_equal_bool),
        Variant("crop_reduce_onehot", build_crop_reduce_onehot),
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
    parser = argparse.ArgumentParser(description="Build and benchmark task129 ONNX variants.")
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
