"""Tiny ONNX generator for NeuroGolf task144.

Task rule: the active input grid is 9x4. Rows 0..3 are block A, row 4 is an
ignored separator, and rows 5..8 are block B. For each of the 4x4 cell
positions, output green (color 3) exactly when both A and B are black at that
position; otherwise output black (color 0). The separator color is irrelevant,
and any non-black color in either block counts as occupied.

The graph uses only channel 0 from the two compact 4x4 blocks. In ARC one-hot
encoding that channel is true exactly for black cells, so the green mask is the
boolean AND of the two background slices. A broadcasted Where creates only the
needed leading output channels [black, 0, 0, green] for the compact 4x4 result;
the final Pad adds the remaining six color channels and the 30x30 spatial
padding directly into the graph output, so that large padded tensor is unscored.
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


TASK_NUM = "144"
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

    def init_i64(self, name: str, values: list[int]) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name))
        return name

    def init_float(self, name: str, values: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(values.astype(np.float32), name))
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
    return model


def add_final_pad_4ch(b: Builder, small: str) -> None:
    b.nodes.append(
        helper.make_node(
            "Pad",
            [small],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 6, 26, 26],
            value=0.0,
        )
    )


def add_background_slices(b: Builder) -> tuple[str, str]:
    b.init_i64("top_starts", [0, 0, 0])
    b.init_i64("top_ends", [1, 4, 4])
    b.init_i64("bottom_starts", [0, 5, 0])
    b.init_i64("bottom_ends", [1, 9, 4])
    b.init_i64("slice_axes", [1, 2, 3])
    top = b.node("Slice", [IN_NAME, "top_starts", "top_ends", "slice_axes"], "top_bg_f")
    bottom = b.node("Slice", [IN_NAME, "bottom_starts", "bottom_ends", "slice_axes"], "bottom_bg_f")
    return top, bottom


def build_where_broadcast() -> onnx.ModelProto:
    """Compact bool mask plus four-channel broadcasted color templates."""
    b = Builder()
    top_f, bottom_f = add_background_slices(b)

    top = b.node("Cast", [top_f], "top_bg_b", to=TensorProto.BOOL)
    bottom = b.node("Cast", [bottom_f], "bottom_bg_b", to=TensorProto.BOOL)
    mask = b.node("And", [top, bottom], "green_mask")

    black = np.zeros((1, 4, 1, 1), dtype=np.float32)
    black[0, 0, 0, 0] = 1.0
    green = np.zeros((1, 4, 1, 1), dtype=np.float32)
    green[0, 3, 0, 0] = 1.0
    b.init_float("black_template", black)
    b.init_float("green_template", green)

    small = b.node("Where", [mask, "green_template", "black_template"], "small")
    add_final_pad_4ch(b, small)
    return make_model(b.nodes, b.initializers)


def variants() -> list[Variant]:
    return [Variant("where_broadcast", build_where_broadcast)]


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


def make_random_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
    out = np.zeros((1, 10, 30, 30), dtype=np.float32)
    top = rng.integers(0, 10, size=(4, 4), endpoint=False)
    bottom = rng.integers(0, 10, size=(4, 4), endpoint=False)
    sep = rng.integers(0, 10, size=(4,), endpoint=False)
    grid = np.vstack([top, sep.reshape(1, 4), bottom])
    for r in range(9):
        for c in range(4):
            inp[0, int(grid[r, c]), r, c] = 1.0
    mask = (top == 0) & (bottom == 0)
    out[0, 0, :4, :4] = (~mask).astype(np.float32)
    out[0, 3, :4, :4] = mask.astype(np.float32)
    return inp, out


def verify_random(model: onnx.ModelProto, count: int = 200) -> bool:
    rng = np.random.default_rng(144)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    for _ in range(count):
        inp, expected = make_random_case(rng)
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
            random_ok = verify_random(model)
            path = tmp_path / f"{TASK_ID}.onnx"
            write_model(model, path)
            result = score_file(path)
            result["correct"] = ok
            result["random_correct"] = random_ok
            result["splits"] = splits
            results[name] = result

    def sort_key(item: tuple[str, dict[str, Any]]) -> int:
        result = item[1]
        if not result["valid"] or not result["correct"] or not result["random_correct"]:
            return 10**18
        return int(result["cost"])

    best_name, best_result = min(results.items(), key=sort_key)
    if sort_key((best_name, best_result)) >= 10**18:
        raise RuntimeError(f"no valid correct variant: {results}")
    return results, best_name, built


def print_benchmark(results: dict[str, dict[str, Any]], best_name: str) -> None:
    print(f"{'variant':<18} {'ok':<5} {'rnd':<5} {'valid':<6} {'memory':>8} {'params':>7} {'cost':>8} {'score':>10}")
    for name, result in sorted(results.items(), key=lambda item: (item[0] != best_name, item[0])):
        score = result["score"]
        score_text = f"{score:.6f}" if isinstance(score, float) else "None"
        print(
            f"{name:<18} {str(result['correct']):<5} {str(result['random_correct']):<5} "
            f"{str(result['valid']):<6} {str(result['memory']):>8} {str(result['params']):>7} "
            f"{str(result['cost']):>8} {score_text:>10}"
        )
        if not result["correct"]:
            print(f"  splits: {result['splits']}")
        if result["error"]:
            print(f"  error: {str(result['error']).strip()}")
    print(f"best: {best_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and benchmark task144 ONNX variants.")
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
