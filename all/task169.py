"""Compact ONNX generator for NeuroGolf task169.

Task rule: every input is a 10x10 grid containing disconnected gray
polyominoes on black. The output preserves every object position and recolors
each whole gray component by its size: tetrominoes become blue (1), triominoes
become red (2), and dominoes become green (3). Rotations and reflections do not
matter; only 4-connected component size matters.

ONNX: crop the 10x10 gray channel, use one shared 4-neighbor Conv kernel to
compute local foreground degrees and degree-neighbor sums, then classify size-2
and size-3 components with factored one-sided thresholds. All remaining gray
cells are size 4. The graph builds only a compact 4-channel 10x10 bool result,
casts it once, and pads directly to the required [1, 10, 30, 30] output.
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

TASK_NUM = "169"
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


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []

    def init_i64(self, name: str, values: list[int]) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name))
        return name

    def init_f32(self, name: str, values: Any) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.float32), name))
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
        initializer=initializers,
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


def four_neighbor_kernel() -> np.ndarray:
    weight = np.zeros((1, 1, 3, 3), dtype=np.float32)
    weight[0, 0, 0, 1] = 1.0
    weight[0, 0, 1, 0] = 1.0
    weight[0, 0, 1, 2] = 1.0
    weight[0, 0, 2, 1] = 1.0
    return weight


def add_common_degree_graph(b: Builder) -> tuple[str, str, str, str, str]:
    b.init_i64("st_gray", [5, 0, 0])
    b.init_i64("en_gray", [6, 10, 10])
    b.init_i64("axes_chw", [1, 2, 3])
    b.init_f32("k4", four_neighbor_kernel())

    fg = b.node("Slice", [IN_NAME, "st_gray", "en_gray", "axes_chw"], "fg")
    fg_b = b.node("Cast", [fg], "fg_b", to=TensorProto.BOOL)
    bg = b.node("Not", [fg_b], "bg")
    deg = b.node("Conv", [fg, "k4"], "deg", pads=[1, 1, 1, 1])
    deg_fg = b.node("Mul", [deg, "fg"], "deg_fg")
    s1 = b.node("Conv", [deg_fg, "k4"], "s1", pads=[1, 1, 1, 1])
    return bg, fg_b, fg, deg, s1


def finish_output(b: Builder, bg: str, fg_b: str, is_domino: str, is_triomino: str) -> None:
    is_small = b.node("Or", [is_domino, is_triomino], "is_small")
    tetromino = b.node("Xor", [fg_b, is_small], "tetromino")
    out4_b = b.node("Concat", [bg, tetromino, is_triomino, is_domino], "out4_b", axis=1)
    out4 = b.node("Cast", [out4_b], "out4", to=TensorProto.FLOAT)
    b.node(
        "Pad",
        [out4],
        OUT_NAME,
        mode="constant",
        pads=[0, 0, 0, 0, 0, 6, 20, 20],
        value=0.0,
    )


def add_eq1_eq2_constants(b: Builder) -> None:
    b.init_f32("half", [0.5])
    b.init_f32("one_half", [1.5])
    b.init_f32("two_half", [2.5])


def eq_one(b: Builder, x: str, prefix: str) -> str:
    lo = b.node("Greater", [x, "half"], f"{prefix}_gt_half")
    hi = b.node("Less", [x, "one_half"], f"{prefix}_lt_one_half")
    return b.node("And", [lo, hi], prefix)


def eq_two(b: Builder, x: str, prefix: str) -> str:
    lo = b.node("Greater", [x, "one_half"], f"{prefix}_gt_one_half")
    hi = b.node("Less", [x, "two_half"], f"{prefix}_lt_two_half")
    return b.node("And", [lo, hi], prefix)


def build_threshold_descriptor() -> onnx.ModelProto:
    """Selected candidate: use one-sided degree thresholds instead of exact tests."""
    b = Builder()
    bg, fg_b, fg, deg, s1 = add_common_degree_graph(b)
    b.init_f32("one_half", [1.5])
    b.init_f32("two_half", [2.5])

    s1_fg = b.node("Mul", [s1, fg], "s1_fg")
    s1n = b.node("Conv", [s1_fg, "k4"], "s1n", pads=[1, 1, 1, 1])

    deg_lt2 = b.node("Less", [deg, "one_half"], "deg_lt2")
    deg_gt1 = b.node("Greater", [deg, "one_half"], "deg_gt1")
    s1_lt2 = b.node("Less", [s1, "one_half"], "s1_lt2")
    s1_gt1 = b.node("Greater", [s1, "one_half"], "s1_gt1")
    s1_lt3 = b.node("Less", [s1, "two_half"], "s1_lt3")
    s1n_lt3 = b.node("Less", [s1n, "two_half"], "s1n_lt3")

    dom_a = b.node("And", [deg_lt2, s1_lt2], "dom_a")
    is_domino = b.node("And", [fg_b, dom_a], "is_domino")

    tri_center = b.node("And", [deg_gt1, s1_lt3], "tri_center")
    tri_end_a = b.node("And", [deg_lt2, s1_gt1], "tri_end_a")
    tri_end = b.node("And", [tri_end_a, s1n_lt3], "tri_end")
    tri_raw = b.node("Or", [tri_center, tri_end], "tri_raw")
    is_triomino = b.node("And", [fg_b, tri_raw], "is_triomino")

    finish_output(b, bg, fg_b, is_domino, is_triomino)
    return make_model(b.nodes, b.initializers)


def build_factored_threshold_descriptor() -> onnx.ModelProto:
    """Factor triomino center/end tests to save one 10x10 bool mask."""
    b = Builder()
    bg, fg_b, fg, deg, s1 = add_common_degree_graph(b)
    b.init_f32("one_half", [1.5])
    b.init_f32("two_half", [2.5])

    s1_fg = b.node("Mul", [s1, fg], "s1_fg")
    s1n = b.node("Conv", [s1_fg, "k4"], "s1n", pads=[1, 1, 1, 1])

    deg_lt2 = b.node("Less", [deg, "one_half"], "deg_lt2")
    deg_gt1 = b.node("Greater", [deg, "one_half"], "deg_gt1")
    s1_lt2 = b.node("Less", [s1, "one_half"], "s1_lt2")
    s1_gt1 = b.node("Greater", [s1, "one_half"], "s1_gt1")
    s1_lt3 = b.node("Less", [s1, "two_half"], "s1_lt3")
    s1n_lt3 = b.node("Less", [s1n, "two_half"], "s1n_lt3")

    dom_a = b.node("And", [deg_lt2, s1_lt2], "dom_a")
    is_domino = b.node("And", [fg_b, dom_a], "is_domino")

    tri_s1 = b.node("And", [s1_gt1, s1_lt3], "tri_s1")
    tri_kind = b.node("Or", [deg_gt1, s1n_lt3], "tri_kind")
    tri_raw = b.node("And", [tri_s1, tri_kind], "tri_raw")
    is_triomino = b.node("And", [fg_b, tri_raw], "is_triomino")

    finish_output(b, bg, fg_b, is_domino, is_triomino)
    return make_model(b.nodes, b.initializers)


def build_degree_descriptor() -> onnx.ModelProto:
    """Selected variant: resolve domino/triomino/tetromino by degree sums."""
    b = Builder()
    bg, fg_b, fg, deg, s1 = add_common_degree_graph(b)
    add_eq1_eq2_constants(b)

    s1_fg = b.node("Mul", [s1, fg], "s1_fg")
    s1n = b.node("Conv", [s1_fg, "k4"], "s1n", pads=[1, 1, 1, 1])

    deg1 = eq_one(b, deg, "deg1")
    deg2 = eq_two(b, deg, "deg2")
    s1_1 = eq_one(b, s1, "s1_1")
    s1_2 = eq_two(b, s1, "s1_2")
    s1n_2 = eq_two(b, s1n, "s1n_2")

    dom_a = b.node("And", [deg1, s1_1], "dom_a")
    is_domino = b.node("And", [fg_b, dom_a], "is_domino")

    tri_center = b.node("And", [deg2, s1_2], "tri_center")
    tri_end_a = b.node("And", [deg1, s1_2], "tri_end_a")
    tri_end = b.node("And", [tri_end_a, s1n_2], "tri_end")
    tri_raw = b.node("Or", [tri_center, tri_end], "tri_raw")
    is_triomino = b.node("And", [fg_b, tri_raw], "is_triomino")

    finish_output(b, bg, fg_b, is_domino, is_triomino)
    return make_model(b.nodes, b.initializers)


def build_immediate_degree_only() -> onnx.ModelProto:
    """Ablation: cheaper but confuses endpoints of length-4 tetrominoes."""
    b = Builder()
    bg, fg_b, _fg, deg, s1 = add_common_degree_graph(b)
    add_eq1_eq2_constants(b)

    deg1 = eq_one(b, deg, "deg1")
    deg2 = eq_two(b, deg, "deg2")
    s1_1 = eq_one(b, s1, "s1_1")
    s1_2 = eq_two(b, s1, "s1_2")

    dom_a = b.node("And", [deg1, s1_1], "dom_a")
    is_domino = b.node("And", [fg_b, dom_a], "is_domino")

    deg12 = b.node("Or", [deg1, deg2], "deg12")
    tri_raw = b.node("And", [deg12, s1_2], "tri_raw")
    is_triomino = b.node("And", [fg_b, tri_raw], "is_triomino")

    finish_output(b, bg, fg_b, is_domino, is_triomino)
    return make_model(b.nodes, b.initializers)


def variants() -> list[Variant]:
    return [
        Variant("factored_threshold_descriptor", build_factored_threshold_descriptor),
        Variant("threshold_descriptor", build_threshold_descriptor),
        Variant("degree_descriptor", build_degree_descriptor),
        Variant("immediate_degree_only", build_immediate_degree_only),
    ]


def load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def solve(grid: np.ndarray) -> np.ndarray:
    fg = np.asarray(grid) == 5
    out = np.zeros_like(grid, dtype=np.int64)
    seen = np.zeros(fg.shape, dtype=np.bool_)
    h, w = fg.shape
    for row in range(h):
        for col in range(w):
            if not fg[row, col] or seen[row, col]:
                continue
            comp = [(row, col)]
            seen[row, col] = True
            for rr, cc in comp:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and fg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        comp.append((nr, nc))
            color = {4: 1, 3: 2, 2: 3}[len(comp)]
            for rr, cc in comp:
                out[rr, cc] = color
    return out


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


def verify_reference() -> dict[str, tuple[int, int]]:
    splits: dict[str, tuple[int, int]] = {}
    for split, examples in load_task_data().items():
        passed = 0
        for example in examples:
            actual = solve(np.asarray(example["input"], dtype=np.int64))
            expected = np.asarray(example["output"], dtype=np.int64)
            passed += int(np.array_equal(actual, expected))
        splits[split] = (passed, len(examples))
    return splits


def write_model(model: onnx.ModelProto, path: Path) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(path))


def graph_stats(model: onnx.ModelProto) -> dict[str, int | None]:
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception:
        inferred = model
    producers = {out: node for node in inferred.graph.node for out in node.output}
    depth_cache: dict[str, int] = {}

    def depth(name: str) -> int:
        if name in depth_cache:
            return depth_cache[name]
        node = producers.get(name)
        if node is None:
            depth_cache[name] = 0
            return 0
        value = 1 + max((depth(inp) for inp in node.input), default=0)
        depth_cache[name] = value
        return value

    tensor_names = {out for node in inferred.graph.node for out in node.output if out}
    tensor_names.update(info.name for info in inferred.graph.value_info)
    return {
        "nodes": len(inferred.graph.node),
        "internal_tensors": len(tensor_names - {OUT_NAME}),
        "initializers": len(inferred.graph.initializer),
        "depth": depth(OUT_NAME),
    }


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
            result["stats"] = graph_stats(model)
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
    print(f"reference: {verify_reference()}")
    print(f"{'variant':<24} {'ok':<5} {'valid':<6} {'memory':>8} {'params':>7} {'cost':>8} {'score':>10}")
    for name, result in sorted(results.items(), key=lambda item: (item[0] != best_name, item[0])):
        score = result["score"]
        score_text = f"{score:.6f}" if isinstance(score, float) else "None"
        print(
            f"{name:<24} {str(result['correct']):<5} {str(result['valid']):<6} "
            f"{str(result['memory']):>8} {str(result['params']):>7} "
            f"{str(result['cost']):>8} {score_text:>10}"
        )
        print(f"  splits: {result['splits']} stats: {result['stats']}")
        if result["error"]:
            print(f"  error: {str(result['error']).strip()}")
    print(f"best: {best_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and benchmark task169 ONNX variants.")
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
