"""Compact ONNX generator for NeuroGolf task152.

Task rule: the input is always a 3x3 ARC grid in the top-left of the
competition one-hot tensor. The 6x6 output mirrors that pattern into four
quadrants: original, horizontal flip, vertical flip, and both-axis flip.

The best graph crops the live 3x3 region, gathers mirrored row indices
[0, 1, 2, 2, 1, 0], gathers mirrored column indices [0, 1, 2, 2, 1, 0],
then pads the compact 6x6 tensor to the required 30x30 graph output.
"""

from __future__ import annotations

import argparse
import json
import math
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

from score_model import calculate_params, convert_to_numpy, score, score_file  # noqa: E402


TASK_NUM = "152"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
ROOT_BEST_PATH = ROOT / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
OPSET = 10
I64_MIN = -9223372036854775808


@dataclass(frozen=True)
class Variant:
    name: str
    build: Callable[[], onnx.ModelProto]


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self._seen_initializers: set[str] = set()

    def init_i64(self, name: str, values: list[int]) -> str:
        if name not in self._seen_initializers:
            self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name))
            self._seen_initializers.add(name)
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


def pad_to_output(b: Builder, x: str) -> None:
    b.nodes.append(
        helper.make_node(
            "Pad",
            [x],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 24, 24],
            value=0.0,
        )
    )


def crop_3x3(b: Builder) -> str:
    b.init_i64("crop_starts", [0, 0])
    b.init_i64("crop_ends", [3, 3])
    b.init_i64("crop_axes", [2, 3])
    return b.node("Slice", [IN_NAME, "crop_starts", "crop_ends", "crop_axes"], "crop")


def build_gather_compact() -> onnx.ModelProto:
    """Crop 3x3, gather mirrored rows and columns, then pad."""
    b = Builder()
    crop = crop_3x3(b)
    b.init_i64("mirror_idx", [0, 1, 2, 2, 1, 0])
    rows = b.node("Gather", [crop, "mirror_idx"], "rows", axis=2)
    out6 = b.node("Gather", [rows, "mirror_idx"], "out6", axis=3)
    pad_to_output(b, out6)
    return make_model(b.nodes, b.initializers)


def build_gather_compact_bool() -> onnx.ModelProto:
    """Bool compact gathers reduce counted activation bytes before final float pad."""
    b = Builder()
    crop_f = crop_3x3(b)
    crop = b.node("Cast", [crop_f], "crop_b", to=TensorProto.BOOL)
    b.init_i64("mirror_idx", [0, 1, 2, 2, 1, 0])
    rows = b.node("Gather", [crop, "mirror_idx"], "rows_b", axis=2)
    out6_b = b.node("Gather", [rows, "mirror_idx"], "out6_b", axis=3)
    out6 = b.node("Cast", [out6_b], "out6", to=TensorProto.FLOAT)
    pad_to_output(b, out6)
    return make_model(b.nodes, b.initializers)


def build_slice_concat_quadrants() -> onnx.ModelProto:
    """Expected construction: crop, negative-step Slice flips, Concat quadrants."""
    b = Builder()
    crop = crop_3x3(b)
    b.init_i64("h_starts", [2])
    b.init_i64("h_ends", [I64_MIN])
    b.init_i64("h_axes", [3])
    b.init_i64("rev_steps", [-1])
    b.init_i64("v_starts", [2])
    b.init_i64("v_ends", [I64_MIN])
    b.init_i64("v_axes", [2])

    ah = b.node("Slice", [crop, "h_starts", "h_ends", "h_axes", "rev_steps"], "ah")
    av = b.node("Slice", [crop, "v_starts", "v_ends", "v_axes", "rev_steps"], "av")
    avh = b.node("Slice", [av, "h_starts", "h_ends", "h_axes", "rev_steps"], "avh")
    top = b.node("Concat", [crop, ah], "top", axis=3)
    bottom = b.node("Concat", [av, avh], "bottom", axis=3)
    out6 = b.node("Concat", [top, bottom], "out6", axis=2)
    pad_to_output(b, out6)
    return make_model(b.nodes, b.initializers)


def build_row_slice_concat() -> onnx.ModelProto:
    """Slice three compact rows, mirror each row horizontally, reuse rows vertically."""
    b = Builder()
    b.init_i64("row_axes", [2, 3])
    b.init_i64("flip_starts", [2])
    b.init_i64("flip_ends", [I64_MIN])
    b.init_i64("flip_axes", [3])
    b.init_i64("flip_steps", [-1])

    row_outputs: list[str] = []
    for row in range(3):
        starts = b.init_i64(f"r{row}_starts", [row, 0])
        ends = b.init_i64(f"r{row}_ends", [row + 1, 3])
        part = b.node("Slice", [IN_NAME, starts, ends, "row_axes"], f"r{row}")
        rev = b.node("Slice", [part, "flip_starts", "flip_ends", "flip_axes", "flip_steps"], f"r{row}h")
        row_outputs.append(b.node("Concat", [part, rev], f"row{row}", axis=3))

    out6 = b.node("Concat", [row_outputs[0], row_outputs[1], row_outputs[2], row_outputs[2], row_outputs[1], row_outputs[0]], "out6", axis=2)
    pad_to_output(b, out6)
    return make_model(b.nodes, b.initializers)


def build_gather_full_rows_then_cols() -> onnx.ModelProto:
    """No crop before row gather; included to measure the full-row intermediate penalty."""
    b = Builder()
    b.init_i64("row_idx", [0, 1, 2, 2, 1, 0] + [3] * 24)
    b.init_i64("col_idx", [0, 1, 2, 2, 1, 0] + [3] * 24)
    rows = b.node("Gather", [IN_NAME, "row_idx"], "rows", axis=2)
    b.node("Gather", [rows, "col_idx"], OUT_NAME, axis=3)
    return make_model(b.nodes, b.initializers)


def variants() -> list[Variant]:
    return [
        Variant("gather_compact_bool", build_gather_compact_bool),
        Variant("gather_compact", build_gather_compact),
        Variant("slice_concat_quadrants", build_slice_concat_quadrants),
        Variant("row_slice_concat", build_row_slice_concat),
        Variant("gather_full_rows_then_cols", build_gather_full_rows_then_cols),
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


def inferred_internal_shapes(model: onnx.ModelProto) -> dict[str, tuple[str, list[int]]]:
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    tensor_map = {value.name: value for value in graph.value_info}
    tensor_map.update({value.name: value for value in graph.output})
    shapes: dict[str, tuple[str, list[int]]] = {}
    for node in graph.node:
        for output_name in node.output:
            if output_name == OUT_NAME:
                continue
            value = tensor_map.get(output_name)
            if value is None or not value.type.HasField("tensor_type"):
                continue
            tensor_type = value.type.tensor_type
            dtype = TensorProto.DataType.Name(tensor_type.elem_type)
            dims = [int(dim.dim_value) for dim in tensor_type.shape.dim]
            shapes[output_name] = (dtype, dims)
    return shapes


def static_memory_estimate(model: onnx.ModelProto) -> int:
    shapes = inferred_internal_shapes(model)
    total = 0
    for dtype_name, dims in shapes.values():
        elem_type = getattr(TensorProto, dtype_name)
        itemsize = np.dtype(helper.tensor_dtype_to_np_dtype(elem_type)).itemsize
        total += int(math.prod(dims) * itemsize)
    return total


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
            result["params_static"] = calculate_params(model)
            result["memory_static"] = static_memory_estimate(model)
            result["shapes"] = inferred_internal_shapes(model)
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
    print(f"{'variant':<28} {'ok':<5} {'valid':<6} {'memory':>8} {'params':>7} {'cost':>8} {'score':>10}")
    for name, result in sorted(results.items(), key=lambda item: (item[0] != best_name, item[0])):
        score_value = result["score"]
        score_text = f"{score_value:.6f}" if isinstance(score_value, float) else "None"
        print(
            f"{name:<28} {str(result['correct']):<5} {str(result['valid']):<6} "
            f"{str(result['memory']):>8} {str(result['params']):>7} "
            f"{str(result['cost']):>8} {score_text:>10}"
        )
        print(f"  static estimate: memory={result['memory_static']} params={result['params_static']} score={score(int(result['memory_static']) + int(result['params_static'])):.6f}")
        shape_text = ", ".join(f"{key}:{dtype}{dims}" for key, (dtype, dims) in result["shapes"].items())
        print(f"  internal shapes: {shape_text}")
        if not result["correct"]:
            print(f"  splits: {result['splits']}")
        if result["error"]:
            print(f"  error: {str(result['error']).strip()}")
    print(f"best: {best_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and benchmark task152 ONNX variants.")
    parser.add_argument("--benchmark-only", action="store_true", help="print variant scores without writing best model")
    parser.add_argument("--root-output", action="store_true", help=f"also write {ROOT_BEST_PATH}")
    args = parser.parse_args()

    results, best_name, built = benchmark_variants()
    print_benchmark(results, best_name)
    if not args.benchmark_only:
        OUT_DIR.mkdir(exist_ok=True)
        write_model(built[best_name], BEST_PATH)
        print(f"wrote: {BEST_PATH}")
        if args.root_output:
            write_model(built[best_name], ROOT_BEST_PATH)
            print(f"wrote: {ROOT_BEST_PATH}")


if __name__ == "__main__":
    main()
