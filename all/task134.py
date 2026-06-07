"""ONNX generator for NeuroGolf task134.

Task rule: the input contains a movable 3x3 arrangement of solid square
blocks in one color plus scattered single-pixel noise in a second color. The
output is the 3x3 block-occupancy mask recolored to the noise color, with
background zero.

The graph identifies the block color by local adjacency, finds the block
bounding box from row/column projections, tests each of the nine logical cells
with compact broadcast masks, and emits the 3x3 one-hot output padded to the
required 30x30 tensor.
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


TASK_NUM = "134"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
I64_MAX = 9223372036854775807


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


def make_model(nodes: list[onnx.NodeProto], initializers: list[onnx.TensorProto], *, opset: int) -> onnx.ModelProto:
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
        opset_imports=[helper.make_opsetid("", opset)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def add_scalar_i64(b: Builder, name: str, value: int) -> str:
    return b.init_i64(name, np.asarray([value], dtype=np.int64))


def reshape_scalar4(b: Builder, x: str, name: str) -> str:
    return b.node("Reshape", [x, "shape1111"], name)


def add_i64(b: Builder, a: str, c: str, name: str) -> str:
    return b.node("Add", [a, c], name)


def mul_i64(b: Builder, a: str, c: str, name: str) -> str:
    return b.node("Mul", [a, c], name)


def div_i64(b: Builder, a: str, c: str, name: str) -> str:
    return b.node("Div", [a, c], name)


def build_adjacency_bbox_op10() -> onnx.ModelProto:
    b = Builder()
    b.init_i64("ch_starts", [0, 1, 0, 0])
    b.init_i64("ch_ends", [1, 10, 30, 30])
    b.init_i64("q00_starts", [0, 0, 0, 0])
    b.init_i64("q00_ends", [1, 9, 29, 29])
    b.init_i64("q01_starts", [0, 0, 0, 1])
    b.init_i64("q01_ends", [1, 9, 29, 30])
    b.init_i64("q10_starts", [0, 0, 1, 0])
    b.init_i64("q10_ends", [1, 9, 30, 29])
    b.init_i64("q11_starts", [0, 0, 1, 1])
    b.init_i64("q11_ends", [1, 9, 30, 30])
    b.init_i64("shape1111", [1, 1, 1, 1])
    b.init_i64("shape_noise", [1, 1, 1, 1])
    b.init_i64("color_ids", list(range(10)))
    b.init_i64("color_ids_ch", np.arange(10, dtype=np.int64).reshape(1, 10, 1, 1))
    b.init_i64("row_ids", np.arange(30, dtype=np.int64).reshape(1, 1, 30, 1))
    b.init_i64("col_ids", np.arange(30, dtype=np.int64).reshape(1, 1, 1, 30))
    b.init_i64("rev30", list(range(29, -1, -1)))
    b.init_f32("zero_f", [0.0])
    b.init_i64("zero_i", [0])
    b.init_i64("one_i", [1])
    b.init_i64("three_i", [3])
    b.init_i64("twentynine_i", [29])

    nz_f = b.node("Slice", [IN_NAME, "ch_starts", "ch_ends"], "nz_f")
    nz = b.node("Greater", [nz_f, "zero_f"], "nz")

    q00 = b.node("Slice", [nz, "q00_starts", "q00_ends"], "q00")
    q01 = b.node("Slice", [nz, "q01_starts", "q01_ends"], "q01")
    q10 = b.node("Slice", [nz, "q10_starts", "q10_ends"], "q10")
    q11 = b.node("Slice", [nz, "q11_starts", "q11_ends"], "q11")
    q0 = b.node("And", [q00, q01], "q0")
    q1 = b.node("And", [q10, q11], "q1")
    q = b.node("And", [q0, q1], "q")
    q_f = b.node("Cast", [q], "q_f", to=TensorProto.FLOAT)
    square_counts = b.node("ReduceSum", [q_f], "square_counts", axes=[2, 3], keepdims=0)
    obj0 = b.node("ArgMax", [square_counts], "obj0", axis=1, keepdims=0)
    obj = b.node("Add", [obj0, "one_i"], "obj")
    obj_mask = b.node("Gather", [nz, "obj0"], "obj_mask", axis=1)
    obj_f = b.node("Cast", [obj_mask], "obj_f", to=TensorProto.FLOAT)

    row_sum = b.node("ReduceSum", [obj_f], "row_sum", axes=[1, 3], keepdims=0)
    row_has = b.node("Greater", [row_sum, "zero_f"], "row_has")
    row_has_i = b.node("Cast", [row_has], "row_has_i", to=TensorProto.INT64)
    top = b.node("ArgMax", [row_has_i], "top", axis=1, keepdims=0)
    row_rev = b.node("Gather", [row_has_i, "rev30"], "row_rev", axis=1)
    rev_top = b.node("ArgMax", [row_rev], "rev_top", axis=1, keepdims=0)
    bottom = b.node("Sub", ["twentynine_i", rev_top], "bottom")
    height = b.node("Add", [b.node("Sub", [bottom, top], "height_m1"), "one_i"], "height")
    size = div_i64(b, height, "three_i", "size")

    col_sum = b.node("ReduceSum", [obj_f], "col_sum", axes=[1, 2], keepdims=0)
    col_has = b.node("Greater", [col_sum, "zero_f"], "col_has")
    col_has_i = b.node("Cast", [col_has], "col_has_i", to=TensorProto.INT64)
    left = b.node("ArgMax", [col_has_i], "left", axis=1, keepdims=0)

    full_counts = b.node("ReduceSum", [IN_NAME], "full_counts", axes=[2, 3], keepdims=0)
    exists = b.node("Greater", [full_counts, "zero_f"], "exists")
    exists_i = b.node("Cast", [exists], "exists_i", to=TensorProto.INT64)
    present_ids = b.node("Mul", [exists_i, "color_ids"], "present_ids")
    id_sum = b.node("ReduceSum", [present_ids], "id_sum", axes=[1], keepdims=0)
    noise = b.node("Sub", [id_sum, obj], "noise")
    noise4 = b.node("Reshape", [noise, "shape_noise"], "noise4")
    channel_match = b.node("Equal", ["color_ids_ch", noise4], "channel_match")

    col_masks: list[str] = []
    for c in range(3):
        c_off = add_scalar_i64(b, f"c{c}_off", c)
        c_delta = mul_i64(b, size, c_off, f"c{c}_delta")
        c_start = add_i64(b, left, c_delta, f"c{c}_start")
        c_end = add_i64(b, c_start, size, f"c{c}_end")
        cs4 = reshape_scalar4(b, c_start, f"c{c}_start4")
        ce4 = reshape_scalar4(b, c_end, f"c{c}_end4")
        col_ge = b.node("Not", [b.node("Less", ["col_ids", cs4], f"c{c}_lt_start")], f"c{c}_ge")
        col_lt = b.node("Less", ["col_ids", ce4], f"c{c}_lt_end")
        col_masks.append(b.node("And", [col_ge, col_lt], f"c{c}_in"))

    rows: list[str] = []
    for r in range(3):
        cells: list[str] = []
        r_off = add_scalar_i64(b, f"r{r}_off", r)
        r_delta = mul_i64(b, size, r_off, f"r{r}_delta")
        r_start = add_i64(b, top, r_delta, f"r{r}_start")
        r_end = add_i64(b, r_start, size, f"r{r}_end")
        rs4 = reshape_scalar4(b, r_start, f"r{r}_start4")
        re4 = reshape_scalar4(b, r_end, f"r{r}_end4")
        row_ge = b.node("Not", [b.node("Less", ["row_ids", rs4], f"r{r}_lt_start")], f"r{r}_ge")
        row_lt = b.node("Less", ["row_ids", re4], f"r{r}_lt_end")
        row_in = b.node("And", [row_ge, row_lt], f"r{r}_in")
        for c in range(3):
            region = b.node("And", [row_in, col_masks[c]], f"r{r}c{c}_region")
            hit_mask = b.node("And", [obj_mask, region], f"r{r}c{c}_hit_mask")
            hit_f = b.node("Cast", [hit_mask], f"r{r}c{c}_hit_f", to=TensorProto.FLOAT)
            hit_sum = b.node("ReduceSum", [hit_f], f"r{r}c{c}_hit_sum", axes=[2, 3], keepdims=1)
            cells.append(b.node("Greater", [hit_sum, "zero_f"], f"r{r}c{c}_active"))
        rows.append(b.node("Concat", cells, f"row{r}", axis=3))

    active = b.node("Concat", rows, "active", axis=2)
    fg_bool = b.node("And", [active, channel_match], "fg_bool")
    inactive = b.node("Not", [active], "inactive")
    zero_channel = b.node("Equal", ["color_ids_ch", "zero_i"], "zero_channel")
    bg_bool = b.node("And", [inactive, zero_channel], "bg_bool")
    small_bool = b.node("Or", [fg_bool, bg_bool], "small_bool")
    small = b.node("Cast", [small_bool], "small", to=TensorProto.FLOAT)
    b.node("Pad", [small], OUT_NAME, mode="constant", pads=[0, 0, 0, 0, 0, 0, 27, 27], value=0.0)
    return make_model(b.nodes, b.initializers, opset=10)


def build_conv_sample_op10() -> onnx.ModelProto:
    b = Builder()
    b.init_i64("shape_noise", [1, 1, 1, 1])
    b.init_i64("color_ids", list(range(10)))
    b.init_i64("color_ids_ch", np.arange(10, dtype=np.int64).reshape(1, 10, 1, 1))
    b.init_i64("rev30", list(range(29, -1, -1)))
    b.init_f32("zero_f", [0.0])
    b.init_i64("zero_i", [0])
    b.init_i64("one_i", [1])
    b.init_i64("three_i", [3])
    b.init_i64("twentynine_i", [29])

    kernels = np.ones((10, 1, 2, 2), dtype=np.float32)
    kernels[0, :, :, :] = 0.0
    b.init_f32("solid2x2_k", kernels)

    solid_score = b.node("Conv", [IN_NAME, "solid2x2_k"], "solid_score", group=10)
    color_support = b.node("ReduceMax", [solid_score], "color_support", axes=[2, 3], keepdims=0)
    obj = b.node("ArgMax", [color_support], "obj", axis=1, keepdims=0)
    obj_f = b.node("Gather", [IN_NAME, obj], "obj_f", axis=1)

    row_sum = b.node("ReduceSum", [obj_f], "row_sum", axes=[1, 3], keepdims=0)
    row_has = b.node("Greater", [row_sum, "zero_f"], "row_has")
    row_has_u8 = b.node("Cast", [row_has], "row_has_u8", to=TensorProto.UINT8)
    top = b.node("ArgMax", [row_has_u8], "top", axis=1, keepdims=0)
    row_rev = b.node("Gather", [row_has_u8, "rev30"], "row_rev", axis=1)
    rev_top = b.node("ArgMax", [row_rev], "rev_top", axis=1, keepdims=0)
    bottom = b.node("Sub", ["twentynine_i", rev_top], "bottom")
    height = b.node("Add", [b.node("Sub", [bottom, top], "height_m1"), "one_i"], "height")
    size = div_i64(b, height, "three_i", "size")

    col_sum = b.node("ReduceSum", [obj_f], "col_sum", axes=[1, 2], keepdims=0)
    col_has = b.node("Greater", [col_sum, "zero_f"], "col_has")
    col_has_u8 = b.node("Cast", [col_has], "col_has_u8", to=TensorProto.UINT8)
    left = b.node("ArgMax", [col_has_u8], "left", axis=1, keepdims=0)

    row_indices: list[str] = []
    col_indices: list[str] = []
    for i in range(3):
        off = add_scalar_i64(b, f"idx{i}_off", i)
        delta = mul_i64(b, size, off, f"idx{i}_delta")
        row_indices.append(add_i64(b, top, delta, f"row{i}_idx"))
        col_indices.append(add_i64(b, left, delta, f"col{i}_idx"))
    rows_idx = b.node("Concat", row_indices, "rows_idx", axis=0)
    cols_idx = b.node("Concat", col_indices, "cols_idx", axis=0)
    sampled_rows = b.node("Gather", [obj_f, rows_idx], "sampled_rows", axis=2)
    sampled = b.node("Gather", [sampled_rows, cols_idx], "sampled", axis=3)
    active = b.node("Greater", [sampled, "zero_f"], "active")

    full_counts = b.node("ReduceSum", [IN_NAME], "full_counts", axes=[2, 3], keepdims=0)
    exists = b.node("Greater", [full_counts, "zero_f"], "exists")
    exists_i = b.node("Cast", [exists], "exists_i", to=TensorProto.INT64)
    present_ids = b.node("Mul", [exists_i, "color_ids"], "present_ids")
    id_sum = b.node("ReduceSum", [present_ids], "id_sum", axes=[1], keepdims=0)
    noise = b.node("Sub", [id_sum, obj], "noise")
    noise4 = b.node("Reshape", [noise, "shape_noise"], "noise4")
    channel_match = b.node("Equal", ["color_ids_ch", noise4], "channel_match")

    fg_bool = b.node("And", [active, channel_match], "fg_bool")
    inactive = b.node("Not", [active], "inactive")
    zero_channel = b.node("Equal", ["color_ids_ch", "zero_i"], "zero_channel")
    bg_bool = b.node("And", [inactive, zero_channel], "bg_bool")
    small_bool = b.node("Or", [fg_bool, bg_bool], "small_bool")
    small = b.node("Cast", [small_bool], "small", to=TensorProto.FLOAT)
    b.node("Pad", [small], OUT_NAME, mode="constant", pads=[0, 0, 0, 0, 0, 0, 27, 27], value=0.0)
    return make_model(b.nodes, b.initializers, opset=10)


def variants() -> list[Variant]:
    return [Variant("conv_sample_op10", build_conv_sample_op10)]




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
    parser = argparse.ArgumentParser(description="Build and benchmark task134 ONNX variants.")
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
