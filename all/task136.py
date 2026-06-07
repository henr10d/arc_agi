"""Compact ONNX generator for NeuroGolf task136.

Task rule: the 10x10 input contains one blue 2x2 square and one red 2x2
square on black. The output keeps both squares, extends a 1-cell blue diagonal
up-left from the blue square's top-left corner, and extends a 1-cell red
diagonal down-right from the red square's bottom-right corner.

The best graph works directly on the blue/red one-hot channels. It uses a
shared 2x2 convolution to detect the blue top-left and red bottom-right cells,
expands those cells along their diagonals with compact 10x10 ConvTranspose
kernels, assembles the final 3 active color channels as bool masks, casts once,
and pads once to the required 30x30 interface.
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


TASK_NUM = "136"
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
    def __init__(self, opset: int, mask_dtype: int = TensorProto.FLOAT) -> None:
        self.opset = opset
        self.mask_dtype = mask_dtype
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self._i64_cache: dict[tuple[int, ...], str] = {}
        self.zero_test_scalar = "one_scalar_u8" if mask_dtype == TensorProto.UINT8 else "half_scalar"

    def i64(self, values: list[int]) -> str:
        key = tuple(values)
        if key not in self._i64_cache:
            name = f"i64_{len(self._i64_cache)}"
            self._i64_cache[key] = name
            self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name))
        return self._i64_cache[key]

    def node(self, op_type: str, inputs: list[str], output: str, **attrs: Any) -> str:
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        return output

    def slice(self, x: str, output: str, starts: list[int], ends: list[int], axes: list[int]) -> str:
        if self.opset < 10:
            return self.node("Slice", [x], output, starts=starts, ends=ends, axes=axes)
        return self.node("Slice", [x, self.i64(starts), self.i64(ends), self.i64(axes)], output)

    def pad(self, x: str, output: str, pads: list[int]) -> str:
        return self.node("Pad", [x], output, mode="constant", pads=pads, value=0.0)


def make_model(b: Builder) -> onnx.ModelProto:
    graph = helper.make_graph(
        b.nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        b.initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", b.opset)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def add_chain(b: Builder, xs: list[str], prefix: str) -> str:
    out = xs[0]
    for index, x in enumerate(xs[1:], 1):
        out = b.node("Add", [out, x], f"{prefix}_{index}")
    return out


def as_float(b: Builder, x: str, output: str) -> str:
    return b.node("Cast", [x], output, to=TensorProto.FLOAT)


def as_mask(b: Builder, x: str, output: str) -> str:
    return b.node("Cast", [x], output, to=b.mask_dtype)


def is_zero(b: Builder, x: str, output: str) -> str:
    return b.node("Less", [x, b.zero_test_scalar], output)


def mask_not(b: Builder, x: str, output: str) -> str:
    return as_mask(b, is_zero(b, x, f"{output}_is_zero"), output)


def where_open(b: Builder, x: str, value: str, output: str) -> str:
    return b.node("Where", [is_zero(b, x, f"{output}_open"), value, "zero_scalar"], output)


def shift_ul(b: Builder, x: str, k: int, output: str) -> str:
    if k == 0:
        return x
    sliced = b.slice(x, f"{output}_slice", [k, k], [10, 10], [2, 3])
    return b.pad(sliced, output, [0, 0, 0, 0, 0, 0, k, k])


def shift_dr(b: Builder, x: str, k: int, output: str) -> str:
    if k == 0:
        return x
    sliced = b.slice(x, f"{output}_slice", [0, 0], [10 - k, 10 - k], [2, 3])
    return b.pad(sliced, output, [0, 0, k, k, 0, 0, 0, 0])


def build_mask_graph(opset: int, mask_dtype: int = TensorProto.FLOAT) -> onnx.ModelProto:
    b = Builder(opset, mask_dtype)
    if mask_dtype == TensorProto.UINT8:
        b.initializers.append(numpy_helper.from_array(np.asarray(1, dtype=np.uint8), "one_scalar_u8"))
    else:
        b.initializers.append(numpy_helper.from_array(np.asarray(0.5, dtype=np.float32), "half_scalar"))

    black_f = b.slice(IN_NAME, "black_f", [0, 0, 0], [1, 10, 10], [1, 2, 3])
    blue_f = b.slice(IN_NAME, "blue_f", [1, 0, 0], [2, 10, 10], [1, 2, 3])
    red_f = b.slice(IN_NAME, "red_f", [2, 0, 0], [3, 10, 10], [1, 2, 3])
    if mask_dtype != TensorProto.FLOAT:
        black_f = b.node("Cast", [black_f], "black_m", to=mask_dtype)
        blue_f = b.node("Cast", [blue_f], "blue_m", to=mask_dtype)
        red_f = b.node("Cast", [red_f], "red_m", to=mask_dtype)

    blue_above = b.pad(b.slice(blue_f, "blue_above_core", [0], [9], [2]), "blue_above", [0, 0, 1, 0, 0, 0, 0, 0])
    blue_left = b.pad(b.slice(blue_f, "blue_left_core", [0], [9], [3]), "blue_left", [0, 0, 0, 1, 0, 0, 0, 0])
    blue_not_tl = add_chain(b, [blue_above, blue_left], "blue_not_tl")
    blue_tl_open = as_mask(b, is_zero(b, blue_not_tl, "blue_tl_open_b"), "blue_tl_open")
    blue_tl = b.node("Mul", [blue_f, blue_tl_open], "blue_tl")

    red_below = b.pad(b.slice(red_f, "red_below_core", [1], [10], [2]), "red_below", [0, 0, 0, 0, 0, 0, 1, 0])
    red_right = b.pad(b.slice(red_f, "red_right_core", [1], [10], [3]), "red_right", [0, 0, 0, 0, 0, 0, 0, 1])
    red_not_br = add_chain(b, [red_below, red_right], "red_not_br")
    red_br_open = as_mask(b, is_zero(b, red_not_br, "red_br_open_b"), "red_br_open")
    red_br = b.node("Mul", [red_f, red_br_open], "red_br")

    blue_diag_sum = add_chain(b, [shift_ul(b, blue_tl, k, f"blue_ul_{k}") for k in range(10)], "blue_diag_sum")
    red_diag_sum = add_chain(b, [shift_dr(b, red_br, k, f"red_dr_{k}") for k in range(10)], "red_diag_sum")
    blue_line = b.node("Mul", [blue_diag_sum, mask_not(b, red_f, "red_not")], "blue_line")
    blue_out_sum = add_chain(b, [blue_f, blue_line], "blue_out_sum")
    blue_out = blue_out_sum
    red_line = b.node("Mul", [red_diag_sum, mask_not(b, blue_out, "blue_out_not")], "red_line")
    red_out_sum = add_chain(b, [red_f, red_line], "red_out_sum")
    red_out = red_out_sum
    colored_sum = add_chain(b, [blue_out, red_out], "colored_sum")
    colored = colored_sum
    black_out = b.node("Mul", [black_f, mask_not(b, colored, "colored_not")], "black_out")

    onehot = b.node("Concat", [black_out, blue_out, red_out], "onehot", axis=1)
    if mask_dtype != TensorProto.FLOAT:
        onehot = b.node("Cast", [onehot], "onehot_f", to=TensorProto.FLOAT)
    b.pad(onehot, OUT_NAME, [0, 0, 0, 0, 0, 7, 20, 20])
    return make_model(b)


def build_convtranspose_graph(opset: int = 10) -> onnx.ModelProto:
    b = Builder(opset)
    b.initializers.append(numpy_helper.from_array(np.asarray(0.5, dtype=np.float32), "half_scalar"))
    kernel = np.eye(10, dtype=np.float32).reshape(1, 1, 10, 10)
    b.initializers.append(numpy_helper.from_array(kernel, "diag_kernel"))

    black_f = b.slice(IN_NAME, "black_f", [0, 0, 0], [1, 10, 10], [1, 2, 3])
    blue_f = b.slice(IN_NAME, "blue_f", [1, 0, 0], [2, 10, 10], [1, 2, 3])
    red_f = b.slice(IN_NAME, "red_f", [2, 0, 0], [3, 10, 10], [1, 2, 3])

    blue_above = b.pad(b.slice(blue_f, "blue_above_core", [0], [9], [2]), "blue_above", [0, 0, 1, 0, 0, 0, 0, 0])
    blue_left = b.pad(b.slice(blue_f, "blue_left_core", [0], [9], [3]), "blue_left", [0, 0, 0, 1, 0, 0, 0, 0])
    blue_not_tl = add_chain(b, [blue_above, blue_left], "blue_not_tl")
    blue_tl_open = as_float(b, is_zero(b, blue_not_tl, "blue_tl_open_b"), "blue_tl_open")
    blue_tl = b.node("Mul", [blue_f, blue_tl_open], "blue_tl")

    red_below = b.pad(b.slice(red_f, "red_below_core", [1], [10], [2]), "red_below", [0, 0, 0, 0, 0, 0, 1, 0])
    red_right = b.pad(b.slice(red_f, "red_right_core", [1], [10], [3]), "red_right", [0, 0, 0, 0, 0, 0, 0, 1])
    red_not_br = add_chain(b, [red_below, red_right], "red_not_br")
    red_br_open = as_float(b, is_zero(b, red_not_br, "red_br_open_b"), "red_br_open")
    red_br = b.node("Mul", [red_f, red_br_open], "red_br")

    blue_diag_sum = b.node(
        "ConvTranspose",
        [blue_tl, "diag_kernel"],
        "blue_diag_sum",
        pads=[9, 9, 0, 0],
    )
    red_diag_sum = b.node(
        "ConvTranspose",
        [red_br, "diag_kernel"],
        "red_diag_sum",
        pads=[0, 0, 9, 9],
    )
    blue_line = b.node("Mul", [blue_diag_sum, mask_not(b, red_f, "red_not")], "blue_line")
    blue_out_sum = add_chain(b, [blue_f, blue_line], "blue_out_sum")
    blue_out = blue_out_sum
    red_line = b.node("Mul", [red_diag_sum, mask_not(b, blue_out, "blue_out_not")], "red_line")
    red_out_sum = add_chain(b, [red_f, red_line], "red_out_sum")
    red_out = red_out_sum
    colored_sum = add_chain(b, [blue_out, red_out], "colored_sum")
    colored = colored_sum
    black_out = b.node("Mul", [black_f, mask_not(b, colored, "colored_not")], "black_out")

    onehot = b.node("Concat", [black_out, blue_out, red_out], "onehot", axis=1)
    b.pad(onehot, OUT_NAME, [0, 0, 0, 0, 0, 7, 20, 20])
    return make_model(b)


def build_conv_detect_graph(opset: int = 9) -> onnx.ModelProto:
    b = Builder(opset)
    b.initializers.append(numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), "zero_scalar"))
    b.initializers.append(numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), "one_scalar"))
    b.initializers.append(numpy_helper.from_array(np.asarray(0.5, dtype=np.float32), "half_scalar"))
    b.initializers.append(numpy_helper.from_array(np.asarray(3.5, dtype=np.float32), "three_half_scalar"))
    square_kernel = np.ones((1, 1, 2, 2), dtype=np.float32)
    b.initializers.append(numpy_helper.from_array(square_kernel, "square_kernel"))
    diag_kernel = np.eye(10, dtype=np.float32).reshape(1, 1, 10, 10)
    b.initializers.append(numpy_helper.from_array(diag_kernel, "diag_kernel"))

    blue_f = b.slice(IN_NAME, "blue_f", [1, 0, 0], [2, 10, 10], [1, 2, 3])
    red_f = b.slice(IN_NAME, "red_f", [2, 0, 0], [3, 10, 10], [1, 2, 3])

    blue_hits = b.node("Conv", [blue_f, "square_kernel"], "blue_hits", pads=[0, 0, 1, 1])
    blue_tl = as_float(b, b.node("Greater", [blue_hits, "three_half_scalar"], "blue_tl_b"), "blue_tl")
    red_hits = b.node("Conv", [red_f, "square_kernel"], "red_hits", pads=[1, 1, 0, 0])
    red_br = as_float(b, b.node("Greater", [red_hits, "three_half_scalar"], "red_br_b"), "red_br")

    blue_diag_sum = b.node(
        "ConvTranspose",
        [blue_tl, "diag_kernel"],
        "blue_diag_sum",
        pads=[9, 9, 0, 0],
    )
    red_diag_sum = b.node(
        "ConvTranspose",
        [red_br, "diag_kernel"],
        "red_diag_sum",
        pads=[0, 0, 9, 9],
    )
    blue_line = where_open(b, red_f, blue_diag_sum, "blue_line")
    blue_out_sum = add_chain(b, [blue_f, blue_line], "blue_out_sum")
    blue_out = blue_out_sum
    red_line = where_open(b, blue_out, red_diag_sum, "red_line")
    red_out_sum = add_chain(b, [red_f, red_line], "red_out_sum")
    red_out = red_out_sum
    colored_sum = add_chain(b, [blue_out, red_out], "colored_sum")
    colored = colored_sum
    black_out = where_open(b, colored, "one_scalar", "black_out")

    onehot = b.node("Concat", [black_out, blue_out, red_out], "onehot", axis=1)
    b.pad(onehot, OUT_NAME, [0, 0, 0, 0, 0, 7, 20, 20])
    return make_model(b)


def build_conv_detect_bool_output_graph(opset: int = 9) -> onnx.ModelProto:
    b = Builder(opset)
    b.initializers.append(numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), "zero_scalar"))
    b.initializers.append(numpy_helper.from_array(np.asarray(0.5, dtype=np.float32), "half_scalar"))
    b.initializers.append(numpy_helper.from_array(np.asarray(3.5, dtype=np.float32), "three_half_scalar"))
    square_kernel = np.ones((1, 1, 2, 2), dtype=np.float32)
    b.initializers.append(numpy_helper.from_array(square_kernel, "square_kernel"))
    diag_kernel = np.eye(10, dtype=np.float32).reshape(1, 1, 10, 10)
    b.initializers.append(numpy_helper.from_array(diag_kernel, "diag_kernel"))

    blue_f = b.slice(IN_NAME, "blue_f", [1, 0, 0], [2, 10, 10], [1, 2, 3])
    red_f = b.slice(IN_NAME, "red_f", [2, 0, 0], [3, 10, 10], [1, 2, 3])

    blue_hits = b.node("Conv", [blue_f, "square_kernel"], "blue_hits", pads=[0, 0, 1, 1])
    blue_tl = as_float(b, b.node("Greater", [blue_hits, "three_half_scalar"], "blue_tl_b"), "blue_tl")
    red_hits = b.node("Conv", [red_f, "square_kernel"], "red_hits", pads=[1, 1, 0, 0])
    red_br = as_float(b, b.node("Greater", [red_hits, "three_half_scalar"], "red_br_b"), "red_br")

    blue_diag_sum = b.node(
        "ConvTranspose",
        [blue_tl, "diag_kernel"],
        "blue_diag_sum",
        pads=[9, 9, 0, 0],
    )
    red_diag_sum = b.node(
        "ConvTranspose",
        [red_br, "diag_kernel"],
        "red_diag_sum",
        pads=[0, 0, 9, 9],
    )
    blue_line = where_open(b, red_f, blue_diag_sum, "blue_line")
    blue_out_sum = add_chain(b, [blue_f, blue_line], "blue_out_sum")
    blue_open = is_zero(b, blue_out_sum, "blue_open")
    red_line = b.node("Where", [blue_open, red_diag_sum, "zero_scalar"], "red_line")
    red_out_sum = add_chain(b, [red_f, red_line], "red_out_sum")
    red_open = is_zero(b, red_out_sum, "red_open")

    black_out_b = b.node("And", [blue_open, red_open], "black_out_b")
    blue_out_b = b.node("Not", [blue_open], "blue_out_b")
    red_out_b = b.node("Not", [red_open], "red_out_b")
    onehot_b = b.node("Concat", [black_out_b, blue_out_b, red_out_b], "onehot_b", axis=1)
    onehot = b.node("Cast", [onehot_b], "onehot", to=TensorProto.FLOAT)
    b.pad(onehot, OUT_NAME, [0, 0, 0, 0, 0, 7, 20, 20])
    return make_model(b)


def variants() -> list[Variant]:
    return [
        Variant("conv_bool_op9", build_conv_detect_bool_output_graph),
        Variant("conv_bool_op10", lambda: build_conv_detect_bool_output_graph(10)),
        Variant("conv_detect_op9", build_conv_detect_graph),
        Variant("conv_detect_op10", lambda: build_conv_detect_graph(10)),
        Variant("convtranspose_op9", lambda: build_convtranspose_graph(9)),
        Variant("convtranspose_op10", build_convtranspose_graph),
        Variant("mask_float_op9", lambda: build_mask_graph(9)),
        Variant("mask_float_op10", lambda: build_mask_graph(10)),
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


def make_case(blue_top: tuple[int, int], red_top: tuple[int, int]) -> dict[str, list[list[int]]]:
    inp = [[0 for _ in range(10)] for _ in range(10)]
    br, bc = blue_top
    rr, rc = red_top
    for r in (br, br + 1):
        for c in (bc, bc + 1):
            inp[r][c] = 1
    for r in (rr, rr + 1):
        for c in (rc, rc + 1):
            inp[r][c] = 2

    out = [row[:] for row in inp]
    for k in range(1, min(br, bc) + 1):
        out[br - k][bc - k] = 1
    red_r1, red_c1 = rr + 1, rc + 1
    for k in range(1, min(9 - red_r1, 9 - red_c1) + 1):
        out[red_r1 + k][red_c1 + k] = 2
    return {"input": inp, "output": out}


def verify_custom(model: onnx.ModelProto) -> bool:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    cases = [
        make_case((4, 4), (0, 6)),
        make_case((0, 0), (5, 5)),
        make_case((2, 7), (8, 0)),
        make_case((7, 6), (0, 2)),
    ]
    for case in cases:
        inp = convert_to_numpy(case, "input")
        expected = convert_to_numpy(case, "output")
        assert inp is not None and expected is not None
        pred = session.run([OUT_NAME], {IN_NAME: inp})[0]
        if not np.array_equal(pred > 0.0, expected > 0.0):
            return False
    return True


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
            custom_ok = verify_custom(model)
            path = tmp_path / f"{TASK_ID}.onnx"
            write_model(model, path)
            result = score_file(path)
            result["correct"] = ok
            result["custom"] = custom_ok
            result["splits"] = splits
            results[name] = result

    def sort_key(item: tuple[str, dict[str, Any]]) -> int:
        result = item[1]
        if not result["valid"] or not result["correct"] or not result["custom"]:
            return 10**18
        return int(result["cost"])

    best_name, best_result = min(results.items(), key=sort_key)
    if sort_key((best_name, best_result)) >= 10**18:
        raise RuntimeError(f"no valid correct variant: {results}")
    return results, best_name, built


def print_benchmark(results: dict[str, dict[str, Any]], best_name: str) -> None:
    print(f"{'variant':<18} {'ok':<5} {'custom':<7} {'valid':<6} {'memory':>8} {'params':>7} {'cost':>8} {'score':>10}")
    for name, result in sorted(results.items(), key=lambda item: (item[0] != best_name, item[0])):
        score = result["score"]
        score_text = f"{score:.6f}" if isinstance(score, float) else "None"
        print(
            f"{name:<18} {str(result['correct']):<5} {str(result['custom']):<7} {str(result['valid']):<6} "
            f"{str(result['memory']):>8} {str(result['params']):>7} {str(result['cost']):>8} {score_text:>10}"
        )
        if not result["correct"]:
            print(f"  splits: {result['splits']}")
        if result["error"]:
            print(f"  error: {str(result['error']).strip()}")
    print(f"best: {best_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and benchmark task136 ONNX variants.")
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
