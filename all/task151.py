"""ONNX generator for NeuroGolf task151.

Task rule: the input contains one full horizontal colored line and one full
vertical colored line on a black background. Fill the 3x3 square centered on
their intersection with yellow (color 4), but leave the intersection cell in
its original input color. All line cells outside that 3x3 square, and all
background cells, are copied from the input. Examples have variable grid sizes
inside the standard 30x30 one-hot competition tensor.

The best graph keeps the position logic compact: count non-black cells with
full-line convolutions, expand the one-dimensional line masks by one cell, form
the 3x3 square and center masks by broadcasting, XOR them to get the yellow
cells, and write the final float tensor with a single Where whose output is the
official graph output and therefore not counted as activation memory.
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


TASK_NUM = "151"
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
        self.sparse_initializers: list[onnx.SparseTensorProto] = []

    def init_array(self, name: str, array: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(array, name))
        return name

    def init_i64(self, name: str, values: list[int]) -> str:
        return self.init_array(name, np.asarray(values, dtype=np.int64))

    def init_f32(self, name: str, values: list[float], shape: list[int]) -> str:
        return self.init_array(name, np.asarray(values, dtype=np.float32).reshape(shape))

    def init_bool(self, name: str, values: list[bool], shape: list[int]) -> str:
        return self.init_array(name, np.asarray(values, dtype=np.bool_).reshape(shape))

    def init_sparse_f32(self, name: str, dense: np.ndarray) -> str:
        flat = dense.reshape(-1)
        indices = np.nonzero(flat)[0].astype(np.int64)
        values = flat[indices].astype(np.float32)
        value_tensor = numpy_helper.from_array(values, name)
        index_tensor = numpy_helper.from_array(indices, f"{name}_idx")
        self.sparse_initializers.append(helper.make_sparse_tensor(value_tensor, index_tensor, dense.shape))
        return name

    def node(self, op_type: str, inputs: list[str], output: str, **attrs: Any) -> str:
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        return output


def make_model(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    sparse_initializers: list[onnx.SparseTensorProto] | None = None,
) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        initializers,
        sparse_initializer=sparse_initializers,
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


def add_common_line_masks(b: Builder) -> tuple[str, str, str, str]:
    b.init_f32("one", [1.0], [1])
    b.init_i64("ch_s1", [1])
    b.init_i64("ch_e10", [10])
    b.init_i64("ch_axis", [1])

    row_sum_ch = b.node("ReduceSum", [IN_NAME], "row_sum_ch", axes=[3], keepdims=1)
    col_sum_ch = b.node("ReduceSum", [IN_NAME], "col_sum_ch", axes=[2], keepdims=1)
    row_nb = b.node("Slice", [row_sum_ch, "ch_s1", "ch_e10", "ch_axis"], "row_nb")
    col_nb = b.node("Slice", [col_sum_ch, "ch_s1", "ch_e10", "ch_axis"], "col_nb")
    row_max = b.node("ReduceMax", [row_nb], "row_max", axes=[1], keepdims=1)
    col_max = b.node("ReduceMax", [col_nb], "col_max", axes=[1], keepdims=1)
    row_has = b.node("Greater", [row_max, "one"], "row_has")
    col_has = b.node("Greater", [col_max, "one"], "col_has")
    return row_has, col_has, "row_max", "col_max"


def add_conv_line_masks(b: Builder) -> tuple[str, str]:
    b.init_f32("one", [1.0], [1])
    weights = np.ones((1, 10, 1, 1), dtype=np.float32)
    weights[0, 0, 0, 0] = 0.0
    b.init_array("nonblack_w", weights)
    occ = b.node("Conv", [IN_NAME, "nonblack_w"], "occ")
    row_sum = b.node("ReduceSum", [occ], "row_sum", axes=[3], keepdims=1)
    col_sum = b.node("ReduceSum", [occ], "col_sum", axes=[2], keepdims=1)
    row_has = b.node("Greater", [row_sum, "one"], "row_has")
    col_has = b.node("Greater", [col_sum, "one"], "col_has")
    return row_has, col_has


def add_full_kernel_conv_line_masks(b: Builder) -> tuple[str, str]:
    b.init_f32("one", [1.0], [1])
    row_w = np.ones((1, 10, 1, 30), dtype=np.float32)
    col_w = np.ones((1, 10, 30, 1), dtype=np.float32)
    row_w[:, 0, :, :] = 0.0
    col_w[:, 0, :, :] = 0.0
    b.init_array("row_w", row_w)
    b.init_array("col_w", col_w)
    row_sum = b.node("Conv", [IN_NAME, "row_w"], "row_sum")
    col_sum = b.node("Conv", [IN_NAME, "col_w"], "col_sum")
    row_has = b.node("Greater", [row_sum, "one"], "row_has")
    col_has = b.node("Greater", [col_sum, "one"], "col_has")
    return row_has, col_has


def add_slice_concat_output(b: Builder, row_has: str, col_has: str) -> None:
    b.init_i64("s0", [0])
    b.init_i64("s1", [1])
    b.init_i64("e29", [29])
    b.init_i64("e30", [30])
    b.init_i64("row_axis", [2])
    b.init_i64("col_axis", [3])
    b.init_bool("false_cell", [False], [1, 1, 1, 1])

    row_tail = b.node("Slice", [row_has, "s1", "e30", "row_axis"], "row_tail")
    row_head = b.node("Slice", [row_has, "s0", "e29", "row_axis"], "row_head")
    row_prev = b.node("Concat", [row_tail, "false_cell"], "row_prev", axis=2)
    row_next = b.node("Concat", ["false_cell", row_head], "row_next", axis=2)
    row_near_a = b.node("Or", [row_has, row_prev], "row_near_a")
    row_near = b.node("Or", [row_near_a, row_next], "row_near")

    col_tail = b.node("Slice", [col_has, "s1", "e30", "col_axis"], "col_tail")
    col_head = b.node("Slice", [col_has, "s0", "e29", "col_axis"], "col_head")
    col_prev = b.node("Concat", [col_tail, "false_cell"], "col_prev", axis=3)
    col_next = b.node("Concat", ["false_cell", col_head], "col_next", axis=3)
    col_near_a = b.node("Or", [col_has, col_prev], "col_near_a")
    col_near = b.node("Or", [col_near_a, col_next], "col_near")

    square = b.node("And", [row_near, col_near], "square")
    center = b.node("And", [row_has, col_has], "center")
    yellow_mask = b.node("Xor", [square, center], "yellow_mask")

    yellow = [0.0] * 10
    yellow[4] = 1.0
    b.init_f32("yellow", yellow, [1, 10, 1, 1])
    b.node("Where", [yellow_mask, "yellow", IN_NAME], OUT_NAME)


def build_slice_concat_dynamic() -> onnx.ModelProto:
    """Compact bool Slice/Concat shifts with few params."""
    b = Builder()
    row_has, col_has, _row_sum, _col_sum = add_common_line_masks(b)
    add_slice_concat_output(b, row_has, col_has)
    return make_model(b.nodes, b.initializers)


def add_gather_output(b: Builder, row_has: str, col_has: str) -> None:
    b.init_i64("prev_idx", list(range(1, 30)) + [29])
    b.init_i64("next_idx", [0] + list(range(0, 29)))

    row_prev = b.node("Gather", [row_has, "prev_idx"], "row_prev", axis=2)
    row_next = b.node("Gather", [row_has, "next_idx"], "row_next", axis=2)
    row_near_a = b.node("Or", [row_has, row_prev], "row_near_a")
    row_near = b.node("Or", [row_near_a, row_next], "row_near")
    col_prev = b.node("Gather", [col_has, "prev_idx"], "col_prev", axis=3)
    col_next = b.node("Gather", [col_has, "next_idx"], "col_next", axis=3)
    col_near_a = b.node("Or", [col_has, col_prev], "col_near_a")
    col_near = b.node("Or", [col_near_a, col_next], "col_near")

    square = b.node("And", [row_near, col_near], "square")
    center = b.node("And", [row_has, col_has], "center")
    yellow_mask = b.node("Xor", [square, center], "yellow_mask")

    yellow = [0.0] * 10
    yellow[4] = 1.0
    b.init_f32("yellow", yellow, [1, 10, 1, 1])
    b.node("Where", [yellow_mask, "yellow", IN_NAME], OUT_NAME)


def build_gather_dynamic() -> onnx.ModelProto:
    """Shift one-dimensional masks with Gather index vectors."""
    b = Builder()
    row_has, col_has, _row_sum, _col_sum = add_common_line_masks(b)
    add_gather_output(b, row_has, col_has)
    return make_model(b.nodes, b.initializers)


def build_conv_gather_dynamic() -> onnx.ModelProto:
    """Best candidate: non-black occupancy from one 1x1 Conv, then compact masks."""
    b = Builder()
    row_has, col_has = add_conv_line_masks(b)
    add_gather_output(b, row_has, col_has)
    return make_model(b.nodes, b.initializers)


def build_full_kernel_conv_gather_dynamic() -> onnx.ModelProto:
    """Use two direct line-count Conv kernels; higher params, much lower memory."""
    b = Builder()
    row_has, col_has = add_full_kernel_conv_line_masks(b)
    add_gather_output(b, row_has, col_has)
    return make_model(b.nodes, b.initializers, b.sparse_initializers)


def build_sparse_full_kernel_conv_gather_dynamic() -> onnx.ModelProto:
    """Sparse form of the full-kernel Conv weights, if runtime accepts it."""
    b = Builder()
    b.init_f32("one", [1.0], [1])
    row_w = np.ones((1, 10, 1, 30), dtype=np.float32)
    col_w = np.ones((1, 10, 30, 1), dtype=np.float32)
    row_w[:, 0, :, :] = 0.0
    col_w[:, 0, :, :] = 0.0
    b.init_sparse_f32("row_w", row_w)
    b.init_sparse_f32("col_w", col_w)
    row_sum = b.node("Conv", [IN_NAME, "row_w"], "row_sum")
    col_sum = b.node("Conv", [IN_NAME, "col_w"], "col_sum")
    row_has = b.node("Greater", [row_sum, "one"], "row_has")
    col_has = b.node("Greater", [col_sum, "one"], "col_has")
    add_gather_output(b, row_has, col_has)
    return make_model(b.nodes, b.initializers, b.sparse_initializers)


def build_first_example_constant() -> onnx.ModelProto:
    """Rejected baseline for the prompt's fixed 4x4 orientation only."""
    b = Builder()
    out = np.zeros(FULL_SHAPE, dtype=np.float32)
    grid = np.zeros((30, 30), dtype=np.int64)
    grid[:4, :4] = np.asarray(
        [
            [4, 4, 4, 0],
            [4, 2, 4, 2],
            [4, 4, 4, 0],
            [0, 3, 0, 0],
        ],
        dtype=np.int64,
    )
    for row in range(30):
        for col in range(30):
            out[0, grid[row, col], row, col] = 1.0
    b.nodes.append(
        helper.make_node(
            "Constant",
            [],
            [OUT_NAME],
            value=numpy_helper.from_array(out, "value"),
        )
    )
    return make_model(b.nodes, b.initializers)


def variants() -> list[Variant]:
    return [
        Variant("full_kernel_conv_gather", build_full_kernel_conv_gather_dynamic),
        Variant("conv_gather_dynamic", build_conv_gather_dynamic),
        Variant("slice_concat_dynamic", build_slice_concat_dynamic),
        Variant("gather_dynamic", build_gather_dynamic),
        Variant("first_example_constant", build_first_example_constant),
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
    parser = argparse.ArgumentParser(description="Build and benchmark task151 ONNX variants.")
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
