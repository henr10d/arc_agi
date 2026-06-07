"""ONNX generator for NeuroGolf task162.

Task rule: the active grid is always 20x20. Find every 3x3 window whose
cells are all black (color 0), paint exactly the cells covered by those
windows blue (color 1), and leave every other cell unchanged. The output is
the same 20x20 grid padded back to the competition 30x30 one-hot tensor.

The best graph keeps most tensors in compact bool/uint8 planes: cast the
one-hot input to bool, detect 3x3 black windows with ConvInteger on the black
channel, suppress northwest-overlapping detections with bool Slice/Concat,
expand selected windows with ConvTranspose, build a bool one-hot 20x20 output,
then cast/pad once to satisfy the competition float32 output contract.
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


TASK_NUM = "162"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
SOLUTION_PATH = ROOT / "solution.onnx"

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
        self._i64_cache: dict[tuple[int, ...], str] = {}

    def init_i64(self, values: list[int]) -> str:
        key = tuple(values)
        if key not in self._i64_cache:
            name = f"i64_{len(self._i64_cache)}"
            self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name))
            self._i64_cache[key] = name
        return self._i64_cache[key]

    def init_float(self, name: str, values: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(values.astype(np.float32), name))
        return name

    def init_bool(self, name: str, values: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(values.astype(np.bool_), name))
        return name

    def init_uint8(self, name: str, values: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(values.astype(np.uint8), name))
        return name

    def node(self, op_type: str, inputs: list[str], output: str, **attrs: Any) -> str:
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        return output

    def slice(self, x: str, output: str, starts: list[int], ends: list[int], axes: list[int]) -> str:
        return self.node("Slice", [x, self.init_i64(starts), self.init_i64(ends), self.init_i64(axes)], output)


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


def and_chain(b: Builder, tensors: list[str], prefix: str) -> str:
    acc = tensors[0]
    for idx, tensor in enumerate(tensors[1:], start=1):
        acc = b.node("And", [acc, tensor], f"{prefix}_{idx}")
    return acc


def or_chain(b: Builder, tensors: list[str], prefix: str) -> str:
    acc = tensors[0]
    for idx, tensor in enumerate(tensors[1:], start=1):
        acc = b.node("Or", [acc, tensor], f"{prefix}_{idx}")
    return acc


def shifted18_float(b: Builder, x: str, output: str, d_row: int, d_col: int) -> str:
    row_start = 0
    row_end = 18 - d_row
    top = d_row
    bottom = 0
    if d_col >= 0:
        col_start = 0
        col_end = 18 - d_col
        left = d_col
        right = 0
    else:
        col_start = -d_col
        col_end = 18
        left = 0
        right = -d_col
    sliced = b.slice(x, f"{output}_slice", [row_start, col_start], [row_end, col_end], [2, 3])
    return b.node(
        "Pad",
        [sliced],
        output,
        mode="constant",
        pads=[0, 0, top, left, 0, 0, bottom, right],
        value=0.0,
    )


def suppress_overlaps(b: Builder, valid: str) -> str:
    valid_f = b.node("Cast", [valid], "valid_float_raw", to=TensorProto.FLOAT)
    shifts = [
        shifted18_float(b, valid_f, f"prior_{idx}", d_row, d_col)
        for idx, (d_row, d_col) in enumerate([(1, 1)])
    ]
    prior = shifts[0]
    for idx, tensor in enumerate(shifts[1:], start=1):
        prior = b.node("Max", [prior, tensor], f"prior_any_{idx}")
    prior_bool = b.node("Cast", [prior], "prior_bool", to=TensorProto.BOOL)
    not_prior = b.node("Not", [prior_bool], "not_prior")
    return b.node("And", [valid, not_prior], "valid_filtered")


def suppress_overlaps_bool_concat(b: Builder, valid: str) -> str:
    prior_core = b.slice(valid, "prior_core", [0, 0], [17, 17], [2, 3])
    b.init_bool("false_col", np.zeros((1, 1, 17, 1), dtype=np.bool_))
    b.init_bool("false_row", np.zeros((1, 1, 1, 18), dtype=np.bool_))
    prior_body = b.node("Concat", ["false_col", prior_core], "prior_body", axis=3)
    prior = b.node("Concat", ["false_row", prior_body], "prior_bool", axis=2)
    not_prior = b.node("Not", [prior], "not_prior")
    return b.node("And", [valid, not_prior], "valid_filtered")


def finish_from_cover_float(b: Builder, input_bool: str, cover20_f: str, ch0: str) -> None:
    cover20 = b.node("Cast", [cover20_f], "cover_bool", to=TensorProto.BOOL)
    ch1 = b.slice(input_bool, "ch1", [1, 0, 0], [2, 20, 20], [1, 2, 3])
    rest = b.slice(input_bool, "rest", [2, 0, 0], [10, 20, 20], [1, 2, 3])
    not_cover = b.node("Not", [cover20], "not_cover")
    out0 = b.node("And", [ch0, not_cover], "out0")
    out1 = b.node("Or", [ch1, cover20], "out1")
    small = b.node("Concat", [out0, out1, rest], "small_bool", axis=1)
    small_f = b.node("Cast", [small], "small_float", to=TensorProto.FLOAT)
    b.node(
        "Pad",
        [small_f],
        OUT_NAME,
        mode="constant",
        pads=[0, 0, 0, 0, 0, 0, 10, 10],
        value=0.0,
    )


def finish_float_arithmetic(b: Builder, cover20_f: str, ch0_f: str) -> None:
    ch1_f = b.slice(IN_NAME, "ch1_float", [1, 0, 0], [2, 20, 20], [1, 2, 3])
    rest_f = b.slice(IN_NAME, "rest_float", [2, 0, 0], [10, 20, 20], [1, 2, 3])
    b.init_float("one_scalar", np.asarray([1.0], dtype=np.float32))
    not_cover_f = b.node("Sub", ["one_scalar", cover20_f], "not_cover_float")
    out0_f = b.node("Mul", [ch0_f, not_cover_f], "out0_float")
    out1_f = b.node("Max", [ch1_f, cover20_f], "out1_float")
    small_f = b.node("Concat", [out0_f, out1_f, rest_f], "small_float", axis=1)
    b.node(
        "Pad",
        [small_f],
        OUT_NAME,
        mode="constant",
        pads=[0, 0, 0, 0, 0, 0, 10, 10],
        value=0.0,
    )


def build_shifted_and() -> onnx.ModelProto:
    b = Builder()
    input_bool = b.node("Cast", [IN_NAME], "input_bool", to=TensorProto.BOOL)
    black20 = b.slice(input_bool, "black20", [0, 0, 0], [1, 20, 20], [1, 2, 3])

    windows = [
        b.slice(black20, f"w{r}{c}", [r, c], [r + 18, c + 18], [2, 3])
        for r in range(3)
        for c in range(3)
    ]
    valid = suppress_overlaps(b, and_chain(b, windows, "valid"))
    valid_f = b.node("Cast", [valid], "valid_float", to=TensorProto.FLOAT)
    placements = [
        b.node(
            "Pad",
            [valid_f],
            f"p{r}{c}",
            mode="constant",
            pads=[0, 0, r, c, 0, 0, 2 - r, 2 - c],
            value=0.0,
        )
        for r in range(3)
        for c in range(3)
    ]
    cover = placements[0]
    for idx, tensor in enumerate(placements[1:], start=1):
        cover = b.node("Max", [cover, tensor], f"cover_{idx}")
    finish_from_cover_float(b, input_bool, cover, black20)
    return make_model(b.nodes, b.initializers)


def build_conv_sum() -> onnx.ModelProto:
    b = Builder()
    input_bool = b.node("Cast", [IN_NAME], "input_bool", to=TensorProto.BOOL)
    black_bool = b.slice(input_bool, "black_bool", [0, 0, 0], [1, 20, 20], [1, 2, 3])
    black = b.node("Cast", [black_bool], "black_float", to=TensorProto.FLOAT)
    b.init_float("kernel", np.ones((1, 1, 3, 3), dtype=np.float32))
    summed = b.node("Conv", [black, "kernel"], "sum9")
    b.init_float("eight_half", np.asarray([8.5], dtype=np.float32))
    valid = suppress_overlaps(b, b.node("Greater", [summed, "eight_half"], "valid"))
    valid_f = b.node("Cast", [valid], "valid_float", to=TensorProto.FLOAT)
    placements = [
        b.node(
            "Pad",
            [valid_f],
            f"p{r}{c}",
            mode="constant",
            pads=[0, 0, r, c, 0, 0, 2 - r, 2 - c],
            value=0.0,
        )
        for r in range(3)
        for c in range(3)
    ]
    cover = placements[0]
    for idx, tensor in enumerate(placements[1:], start=1):
        cover = b.node("Max", [cover, tensor], f"cover_{idx}")
    finish_from_cover_float(b, input_bool, cover, black_bool)
    return make_model(b.nodes, b.initializers)


def build_conv_transpose_expand() -> onnx.ModelProto:
    b = Builder()
    input_bool = b.node("Cast", [IN_NAME], "input_bool", to=TensorProto.BOOL)
    black_bool = b.slice(input_bool, "black_bool", [0, 0, 0], [1, 20, 20], [1, 2, 3])
    black = b.node("Cast", [black_bool], "black_float", to=TensorProto.FLOAT)
    b.init_float("kernel", np.ones((1, 1, 3, 3), dtype=np.float32))
    summed = b.node("Conv", [black, "kernel"], "sum9")
    b.init_float("eight_half", np.asarray([8.5], dtype=np.float32))
    valid = suppress_overlaps(b, b.node("Greater", [summed, "eight_half"], "valid"))
    valid_f = b.node("Cast", [valid], "valid_float", to=TensorProto.FLOAT)
    cover = b.node("ConvTranspose", [valid_f, "kernel"], "cover")
    finish_from_cover_float(b, input_bool, cover, black_bool)
    return make_model(b.nodes, b.initializers)


def build_conv_transpose_float_output() -> onnx.ModelProto:
    b = Builder()
    black = b.slice(IN_NAME, "black_float", [0, 0, 0], [1, 20, 20], [1, 2, 3])
    b.init_float("kernel", np.ones((1, 1, 3, 3), dtype=np.float32))
    summed = b.node("Conv", [black, "kernel"], "sum9")
    b.init_float("eight_half", np.asarray([8.5], dtype=np.float32))
    valid = suppress_overlaps(b, b.node("Greater", [summed, "eight_half"], "valid"))
    valid_f = b.node("Cast", [valid], "valid_float", to=TensorProto.FLOAT)
    cover = b.node("ConvTranspose", [valid_f, "kernel"], "cover")
    finish_float_arithmetic(b, cover, black)
    return make_model(b.nodes, b.initializers)


def build_conv_integer_transpose() -> onnx.ModelProto:
    b = Builder()
    input_bool = b.node("Cast", [IN_NAME], "input_bool", to=TensorProto.BOOL)
    black_bool = b.slice(input_bool, "black_bool", [0, 0, 0], [1, 20, 20], [1, 2, 3])
    black_u8 = b.node("Cast", [black_bool], "black_u8", to=TensorProto.UINT8)
    b.init_uint8("kernel_u8", np.ones((1, 1, 3, 3), dtype=np.uint8))
    summed = b.node("ConvInteger", [black_u8, "kernel_u8"], "sum9_i32")
    b.initializers.append(numpy_helper.from_array(np.asarray([8], dtype=np.int32), "eight_i32"))
    valid = suppress_overlaps_bool_concat(b, b.node("Greater", [summed, "eight_i32"], "valid"))
    valid_f = b.node("Cast", [valid], "valid_float", to=TensorProto.FLOAT)
    b.init_float("kernel", np.ones((1, 1, 3, 3), dtype=np.float32))
    cover = b.node("ConvTranspose", [valid_f, "kernel"], "cover")
    finish_from_cover_float(b, input_bool, cover, black_bool)
    return make_model(b.nodes, b.initializers)


def build_reduce_min_stack() -> onnx.ModelProto:
    b = Builder()
    input_bool = b.node("Cast", [IN_NAME], "input_bool", to=TensorProto.BOOL)
    black_bool = b.slice(input_bool, "black_bool", [0, 0, 0], [1, 20, 20], [1, 2, 3])
    black = b.node("Cast", [black_bool], "black_float", to=TensorProto.FLOAT)
    windows = [
        b.slice(black, f"w{r}{c}", [r, c], [r + 18, c + 18], [2, 3])
        for r in range(3)
        for c in range(3)
    ]
    stack = b.node("Concat", windows, "stack", axis=1)
    valid_f = b.node("ReduceMin", [stack], "valid_f", axes=[1], keepdims=1)
    b.init_float("half", np.asarray([0.5], dtype=np.float32))
    valid = suppress_overlaps(b, b.node("Greater", [valid_f, "half"], "valid"))
    valid_float = b.node("Cast", [valid], "valid_float", to=TensorProto.FLOAT)
    placements = [
        b.node(
            "Pad",
            [valid_float],
            f"p{r}{c}",
            mode="constant",
            pads=[0, 0, r, c, 0, 0, 2 - r, 2 - c],
            value=0.0,
        )
        for r in range(3)
        for c in range(3)
    ]
    cover = placements[0]
    for idx, tensor in enumerate(placements[1:], start=1):
        cover = b.node("Max", [cover, tensor], f"cover_{idx}")
    finish_from_cover_float(b, input_bool, cover, black_bool)
    return make_model(b.nodes, b.initializers)


def variants() -> list[Variant]:
    return [
        Variant("conv_integer_transpose", build_conv_integer_transpose),
        Variant("conv_transpose_float", build_conv_transpose_float_output),
        Variant("conv_transpose", build_conv_transpose_expand),
        Variant("shifted_and", build_shifted_and),
        Variant("conv_sum", build_conv_sum),
        Variant("reduce_min_stack", build_reduce_min_stack),
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
    print(f"{'variant':<20} {'ok':<5} {'valid':<6} {'memory':>8} {'params':>7} {'cost':>8} {'score':>10}")
    for name, result in sorted(results.items(), key=lambda item: (item[0] != best_name, item[0])):
        score = result["score"]
        score_text = f"{score:.6f}" if isinstance(score, float) else "None"
        print(
            f"{name:<20} {str(result['correct']):<5} {str(result['valid']):<6} "
            f"{str(result['memory']):>8} {str(result['params']):>7} "
            f"{str(result['cost']):>8} {score_text:>10}"
        )
        if not result["correct"]:
            print(f"  splits: {result['splits']}")
        if result["error"]:
            print(f"  error: {str(result['error']).strip()}")
    print(f"best: {best_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and benchmark task162 ONNX variants.")
    parser.add_argument("--benchmark-only", action="store_true", help="print variant scores without writing models")
    args = parser.parse_args()

    results, best_name, built = benchmark_variants()
    print_benchmark(results, best_name)
    if not args.benchmark_only:
        OUT_DIR.mkdir(exist_ok=True)
        write_model(built[best_name], BEST_PATH)
        write_model(built[best_name], SOLUTION_PATH)
        print(f"wrote: {BEST_PATH}")
        print(f"wrote: {SOLUTION_PATH}")


if __name__ == "__main__":
    main()
