"""Compact ONNX generator for NeuroGolf task083.

Task rule: the input is a 3x4 pattern. The output is a 6x8 mirror mosaic:
top-left is the original pattern, top-right is its horizontal mirror,
bottom-left is its vertical mirror, and bottom-right is its 180-degree
rotation. Everything outside the logical 6x8 output remains unactivated
padding on the 30x30 competition canvas.

The best graph slices the three 1x4 input rows independently, casts each row
to bool for compact internal tensors, mirrors each row horizontally, then
concatenates the three mirrored rows in vertical mirror order. Only the final
6x8 bool result is cast back to float before padding directly into the graph
output.
"""

from __future__ import annotations

import argparse
import json
import shutil
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


TASK_NUM = "083"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.counter = 0

    def init_i64(self, name: str, values: list[int]) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name))
        return name

    def name(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def node(self, op_type: str, inputs: list[str], prefix: str, **attrs: Any) -> str:
        out = self.name(prefix)
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out


def make_model(nodes: list[onnx.NodeProto], initializers: list[onnx.TensorProto], opset: int) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=10)
    onnx.checker.check_model(model, full_check=True)
    return model


def add_pad_to_output(b: Builder, x: str, *, opset: int) -> None:
    pads = [0, 0, 0, 0, 0, 0, 24, 22]
    if opset >= 11:
        b.init_i64("pad_to_30", pads)
        b.nodes.append(helper.make_node("Pad", [x, "pad_to_30"], [OUT_NAME], mode="constant"))
    else:
        b.nodes.append(helper.make_node("Pad", [x], [OUT_NAME], mode="constant", pads=pads, value=0.0))


def build_bool_slice_model(*, opset: int = 10) -> onnx.ModelProto:
    """Bool-small variant using negative-step Slice for horizontal/vertical mirrors."""
    b = Builder()
    b.init_i64("crop_starts", [0, 0, 0, 0])
    b.init_i64("crop_ends", [1, 10, 3, 4])
    b.init_i64("h_starts", [3])
    b.init_i64("h_ends", [-5])
    b.init_i64("h_axis", [3])
    b.init_i64("neg_step", [-1])
    b.init_i64("v_starts", [2])
    b.init_i64("v_ends", [-4])
    b.init_i64("v_axis", [2])

    a_float = b.node("Slice", [IN_NAME, "crop_starts", "crop_ends"], "a_float")
    a = b.node("Cast", [a_float], "a_bool", to=TensorProto.BOOL)
    h = b.node("Slice", [a, "h_starts", "h_ends", "h_axis", "neg_step"], "h_bool")
    row0 = b.node("Concat", [a, h], "row0", axis=3)
    row1 = b.node("Slice", [row0, "v_starts", "v_ends", "v_axis", "neg_step"], "row1",)
    small = b.node("Concat", [row0, row1], "small_bool", axis=2)
    small_float = b.node("Cast", [small], "small_float", to=TensorProto.FLOAT)
    add_pad_to_output(b, small_float, opset=opset)
    return make_model(b.nodes, b.initializers, opset)


def build_float_slice_model(*, opset: int = 10) -> onnx.ModelProto:
    """Pure-float variant with fewer casts but larger internal tensors."""
    b = Builder()
    b.init_i64("crop_starts", [0, 0, 0, 0])
    b.init_i64("crop_ends", [1, 10, 3, 4])
    b.init_i64("h_starts", [3])
    b.init_i64("h_ends", [-5])
    b.init_i64("h_axis", [3])
    b.init_i64("neg_step", [-1])
    b.init_i64("v_starts", [2])
    b.init_i64("v_ends", [-4])
    b.init_i64("v_axis", [2])

    a = b.node("Slice", [IN_NAME, "crop_starts", "crop_ends"], "a_float")
    h = b.node("Slice", [a, "h_starts", "h_ends", "h_axis", "neg_step"], "h_float")
    row0 = b.node("Concat", [a, h], "row0", axis=3)
    row1 = b.node("Slice", [row0, "v_starts", "v_ends", "v_axis", "neg_step"], "row1")
    small = b.node("Concat", [row0, row1], "small_float", axis=2)
    add_pad_to_output(b, small, opset=opset)
    return make_model(b.nodes, b.initializers, opset)


def build_bool_gather_model(*, opset: int = 10) -> onnx.ModelProto:
    """Bool-small fallback using Gather indices instead of negative-step Slice."""
    b = Builder()
    b.init_i64("crop_starts", [0, 0, 0, 0])
    b.init_i64("crop_ends", [1, 10, 3, 4])
    b.init_i64("rows_rev", [2, 1, 0])
    b.init_i64("cols_rev", [3, 2, 1, 0])

    a_float = b.node("Slice", [IN_NAME, "crop_starts", "crop_ends"], "a_float")
    a = b.node("Cast", [a_float], "a_bool", to=TensorProto.BOOL)
    h = b.node("Gather", [a, "cols_rev"], "h_bool", axis=3)
    row0 = b.node("Concat", [a, h], "row0", axis=3)
    row1 = b.node("Gather", [row0, "rows_rev"], "row1", axis=2)
    small = b.node("Concat", [row0, row1], "small_bool", axis=2)
    small_float = b.node("Cast", [small], "small_float", to=TensorProto.FLOAT)
    add_pad_to_output(b, small_float, opset=opset)
    return make_model(b.nodes, b.initializers, opset)


def build_rowwise_bool_model(*, opset: int = 10) -> onnx.ModelProto:
    """Lower-memory variant that avoids materializing the whole 3x4 bool crop."""
    b = Builder()
    b.init_i64("row_axes", [2, 3])
    for row in range(3):
        b.init_i64(f"row{row}_starts", [row, 0])
        b.init_i64(f"row{row}_ends", [row + 1, 4])
    b.init_i64("h_starts", [3])
    b.init_i64("h_ends", [-5])
    b.init_i64("h_axis", [3])
    b.init_i64("neg_step", [-1])

    rows: list[str] = []
    for row in range(3):
        row_float = b.node("Slice", [IN_NAME, f"row{row}_starts", f"row{row}_ends", "row_axes"], "row_float")
        row_bool = b.node("Cast", [row_float], "row_bool", to=TensorProto.BOOL)
        row_h = b.node("Slice", [row_bool, "h_starts", "h_ends", "h_axis", "neg_step"], "row_h")
        rows.append(b.node("Concat", [row_bool, row_h], "row_mirror", axis=3))

    small = b.node("Concat", [rows[0], rows[1], rows[2], rows[2], rows[1], rows[0]], "small_bool", axis=2)
    small_float = b.node("Cast", [small], "small_float", to=TensorProto.FLOAT)
    add_pad_to_output(b, small_float, opset=opset)
    return make_model(b.nodes, b.initializers, opset)


def build_variants() -> dict[str, onnx.ModelProto]:
    return {
        "rowwise_bool_opset10": build_rowwise_bool_model(opset=10),
        "bool_slice_opset10": build_bool_slice_model(opset=10),
        "float_slice_opset10": build_float_slice_model(opset=10),
        "bool_gather_opset10": build_bool_gather_model(opset=10),
        "bool_slice_opset11": build_bool_slice_model(opset=11),
    }


def load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def verify_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    split_counts: dict[str, tuple[int, int]] = {}
    all_ok = True
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
        split_counts[split] = (passed, checked)
    return all_ok, split_counts


def write_model(model: onnx.ModelProto, path: Path) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(path))


def benchmark_variants() -> tuple[dict[str, dict[str, Any]], str, dict[str, onnx.ModelProto]]:
    variants = build_variants()
    results: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}_") as tmp:
        tmp_path = Path(tmp)
        for name, model in variants.items():
            ok, splits = verify_correct(model)
            path = tmp_path / f"{TASK_ID}.onnx"
            write_model(model, path)
            result = score_file(path)
            result["correct"] = ok
            result["splits"] = splits
            results[name] = result

    def sort_key(name: str) -> int:
        result = results[name]
        if not result["valid"] or not result["correct"]:
            return 10**18
        return int(result["cost"])

    best_name = min(results, key=sort_key)
    if sort_key(best_name) >= 10**18:
        raise RuntimeError(f"no valid correct variant: {results}")
    return results, best_name, variants


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
    parser = argparse.ArgumentParser(description="Build and benchmark task083 ONNX variants.")
    parser.add_argument("--benchmark-only", action="store_true", help="print variant scores without writing best model")
    args = parser.parse_args()

    results, best_name, variants = benchmark_variants()
    print_benchmark(results, best_name)
    if not args.benchmark_only:
        OUT_DIR.mkdir(exist_ok=True)
        write_model(variants[best_name], BEST_PATH)
        print(f"wrote: {BEST_PATH}")


if __name__ == "__main__":
    main()
