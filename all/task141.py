"""Compact ONNX generator for NeuroGolf task141.

Task rule: each input grid is a square black field with exactly one nonzero
colored cell. The output keeps that cell's color and draws both complete
diagonals through it, clipped to the original grid size; all other in-grid
cells are black and padding outside the ARC grid remains all-zero.

The best graph uses a 1x1 Conv to isolate channels 1..9 and locate the single
colored cell, gathers that pixel's 10-channel one-hot color vector, builds the
two diagonal masks with int32 row/col ranges, clips them by the black channel,
and overlays the color onto the original input with the free graph output.
"""

from __future__ import annotations

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


TASK_NUM = "141"
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

    def init(self, name: str, values: Any, dtype: np.dtype[Any] | type[np.generic]) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=dtype), name))
        return name

    def init_i64(self, name: str, values: Any) -> str:
        return self.init(name, values, np.int64)

    def init_f32(self, name: str, values: Any) -> str:
        return self.init(name, values, np.float32)

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


def build_conv_direct_u8_i32_diag4() -> onnx.ModelProto:
    """Best candidate: detect the colored pixel by Conv and overlay on black cells."""
    b = Builder()
    b.init("ch0_idx", np.array([0], dtype=np.int32), np.int32)
    b.init("thirty", np.array([30], dtype=np.int32), np.int32)
    b.init("rows", np.arange(30, dtype=np.int32).reshape(1, 1, 30, 1), np.int32)
    b.init("cols", np.arange(30, dtype=np.int32).reshape(1, 1, 1, 30), np.int32)

    color_w = np.ones((1, 10, 1, 1), dtype=np.float32)
    color_w[0, 0, 0, 0] = 0.0
    b.init_f32("color_w", color_w)

    color_sum = b.node("Conv", [IN_NAME, "color_w"], "color_sum")
    colored_u8 = b.node("Cast", [color_sum], "colored_u8", to=TensorProto.UINT8)
    flat_pos = b.node("Flatten", [colored_u8], "flat_pos", axis=1)
    idx64 = b.node("ArgMax", [flat_pos], "idx64", axis=1, keepdims=0)
    idx = b.node("Cast", [idx64], "idx", to=TensorProto.INT32)
    row = b.node("Div", [idx, "thirty"], "row")
    col = b.node("Mod", [idx, "thirty"], "col")

    row_pick = b.node("Gather", [IN_NAME, row], "row_pick", axis=2)
    pixel = b.node("Gather", [row_pick, col], "pixel", axis=3)
    ch0 = b.node("Gather", [IN_NAME, "ch0_idx"], "ch0", axis=1)
    black = b.node("Cast", [ch0], "black", to=TensorProto.BOOL)

    row_minus_col = b.node("Sub", [row, col], "row_minus_col")
    row_plus_col = b.node("Add", [row, col], "row_plus_col")
    diag1_x = b.node("Sub", ["rows", row_minus_col], "diag1_x")
    diag1 = b.node("Equal", [diag1_x, "cols"], "diag1")
    diag2_x = b.node("Sub", [row_plus_col, "rows"], "diag2_x")
    diag2 = b.node("Equal", [diag2_x, "cols"], "diag2")
    diag = b.node("Or", [diag1, diag2], "diag")
    diag_black = b.node("And", [diag, black], "diag_black")
    b.node("Where", [diag_black, pixel, IN_NAME], OUT_NAME)
    return make_model(b.nodes, b.initializers)


def variants() -> list[Variant]:
    return [
        Variant("conv_direct_u8_i32_diag4", build_conv_direct_u8_i32_diag4),
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
    results, best_name, built = benchmark_variants()
    print_benchmark(results, best_name)
    OUT_DIR.mkdir(exist_ok=True)
    write_model(built[best_name], BEST_PATH)
    write_model(built[best_name], SOLUTION_PATH)
    print(f"wrote: {BEST_PATH}")
    print(f"wrote: {SOLUTION_PATH}")


if __name__ == "__main__":
    main()
