"""Compact ONNX generator for NeuroGolf task139.

Task rule: the 9x9 active grid contains two incomplete 3x3 yellow (color 4)
square-like objects on a black background. The upper object is either in box
(rows 1..3, cols 0..2) or (rows 1..3, cols 2..4); the lower object is paired
respectively with box (rows 4..6, cols 5..7) or (rows 6..8, cols 5..7).
Complete both selected 3x3 boxes by changing only their missing background
cells to orange (color 7), while preserving all original color-4 cells.

The best graph detects the layout from whether any color-4 cell appears in
upper column 0, then scatters over the static union of the four possible 3x3
boxes. Only the update values are dynamic: selected box cells replace color 0
with color 7 when the input cell is missing color 4; unselected candidate cells
are left unchanged.
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


TASK_NUM = "139"
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

    def init_i64(self, name: str, values: np.ndarray | list[int]) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name))
        return name

    def init_f32(self, name: str, values: list[float]) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.float32), name))
        return name

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


def box_cells(boxes: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in boxes for dr in range(3) for dc in range(3)]


def nd_indices(channel: int, cells: list[tuple[int, int]]) -> np.ndarray:
    return np.asarray([[0, channel, r, c] for r, c in cells], dtype=np.int64)


def build_scatternd_op11() -> onnx.ModelProto:
    """Smallest local-score graph: dynamic 4-D ScatterND into the output."""
    b = Builder()

    layout_a_cells = box_cells([(1, 0), (4, 5)])
    layout_b_cells = box_cells([(1, 2), (6, 5)])

    # If any upper color-4 cell is in column 0, choose layout A; otherwise B.
    b.init_i64("probe_idx", np.asarray([[0, 4, r, 0] for r in range(1, 4)], dtype=np.int64))
    b.init_f32("zero", [0.0])
    b.init_f32("one", [1.0])
    b.init_f32("zeros18", [0.0] * 18)
    b.init_i64(
        "scatter_a",
        np.concatenate([nd_indices(0, layout_a_cells), nd_indices(7, layout_a_cells)], axis=0),
    )
    b.init_i64(
        "scatter_b",
        np.concatenate([nd_indices(0, layout_b_cells), nd_indices(7, layout_b_cells)], axis=0),
    )
    b.init_i64("gather4_a", nd_indices(4, layout_a_cells))
    b.init_i64("gather4_b", nd_indices(4, layout_b_cells))

    probed = b.node("GatherND", [IN_NAME, "probe_idx"], "probed")
    total = b.node("ReduceSum", [probed], "probe_sum", keepdims=0)
    layout_a = b.node("Greater", [total, "zero"], "layout_a")
    scatter_idx = b.node("Where", [layout_a, "scatter_a", "scatter_b"], "scatter_idx")
    existing4_a = b.node("GatherND", [IN_NAME, "gather4_a"], "existing4_a")
    existing4_b = b.node("GatherND", [IN_NAME, "gather4_b"], "existing4_b")
    existing4 = b.node("Where", [layout_a, existing4_a, existing4_b], "existing4")
    fill7 = b.node("Sub", ["one", existing4], "fill7")
    updates = b.node("Concat", ["zeros18", fill7], "updates", axis=0)
    b.node("ScatterND", [IN_NAME, scatter_idx, updates], OUT_NAME)
    return make_model(b.nodes, b.initializers, opset=11)


def build_static_union_scatternd_op11() -> onnx.ModelProto:
    """Scatter static candidate cells and gate updates by the detected layout."""
    b = Builder()

    layout_a_cells = set(box_cells([(1, 0), (4, 5)]))
    layout_b_cells = set(box_cells([(1, 2), (6, 5)]))
    union_cells = sorted(layout_a_cells | layout_b_cells)

    selected_a = [1.0 if cell in layout_a_cells else 0.0 for cell in union_cells]
    selected_b = [1.0 if cell in layout_b_cells else 0.0 for cell in union_cells]

    # If any upper color-4 cell is in column 0, choose layout A; otherwise B.
    b.init_i64("probe_idx", np.asarray([[0, 4, r, 0] for r in range(1, 4)], dtype=np.int64))
    b.init_f32("zero", [0.0])
    b.init_f32("one", [1.0])
    b.init_f32("selected_a", selected_a)
    b.init_f32("selected_b", selected_b)
    b.init_i64(
        "scatter_union",
        np.concatenate([nd_indices(0, union_cells), nd_indices(7, union_cells)], axis=0),
    )
    b.init_i64("gather4_union", nd_indices(4, union_cells))

    probed = b.node("GatherND", [IN_NAME, "probe_idx"], "probed")
    total = b.node("ReduceSum", [probed], "probe_sum", keepdims=0)
    layout_a = b.node("Greater", [total, "zero"], "layout_a")
    selected = b.node("Where", [layout_a, "selected_a", "selected_b"], "selected")
    existing4 = b.node("GatherND", [IN_NAME, "gather4_union"], "existing4")
    fill7 = b.node("Sub", [selected, existing4], "fill7")
    clear0 = b.node("Sub", ["one", selected], "clear0")
    updates = b.node("Concat", [clear0, fill7], "updates", axis=0)
    b.node("ScatterND", [IN_NAME, "scatter_union", updates], OUT_NAME)
    return make_model(b.nodes, b.initializers, opset=11)


def build_partitioned_static_scatternd_op11() -> onnx.ModelProto:
    """Use static overlap/A-only/B-only regions to avoid a selected-mask tensor."""
    b = Builder()

    layout_a_cells = set(box_cells([(1, 0), (4, 5)]))
    layout_b_cells = set(box_cells([(1, 2), (6, 5)]))
    overlap_cells = sorted(layout_a_cells & layout_b_cells)
    a_only_cells = sorted(layout_a_cells - layout_b_cells)
    b_only_cells = sorted(layout_b_cells - layout_a_cells)
    ordered_cells = overlap_cells + a_only_cells + b_only_cells

    # If any upper color-4 cell is in column 0, choose layout A; otherwise B.
    b.init_i64("probe_idx", np.asarray([[0, 4, r, 0] for r in range(1, 4)], dtype=np.int64))
    b.init_f32("zero", [0.0])
    b.init_f32("one", [1.0])
    b.init_f32("zeros6", [0.0] * len(overlap_cells))
    b.init_f32("zeros12", [0.0] * len(a_only_cells))
    b.init_f32("ones12", [1.0] * len(a_only_cells))
    b.init_i64(
        "scatter_partitioned",
        np.concatenate([nd_indices(0, ordered_cells), nd_indices(7, ordered_cells)], axis=0),
    )
    b.init_i64("gather4_overlap", nd_indices(4, overlap_cells))
    b.init_i64("gather4_a_only", nd_indices(4, a_only_cells))
    b.init_i64("gather4_b_only", nd_indices(4, b_only_cells))

    probed = b.node("GatherND", [IN_NAME, "probe_idx"], "probed")
    total = b.node("ReduceSum", [probed], "probe_sum", keepdims=0)
    layout_a = b.node("Greater", [total, "zero"], "layout_a")

    existing_overlap = b.node("GatherND", [IN_NAME, "gather4_overlap"], "existing_overlap")
    existing_a_only = b.node("GatherND", [IN_NAME, "gather4_a_only"], "existing_a_only")
    existing_b_only = b.node("GatherND", [IN_NAME, "gather4_b_only"], "existing_b_only")

    fill_overlap = b.node("Sub", ["one", existing_overlap], "fill_overlap")

    clear_a = b.node("Where", [layout_a, "zeros12", "ones12"], "clear_a")
    clear_b = b.node("Where", [layout_a, "ones12", "zeros12"], "clear_b")
    fill_a = b.node("Sub", [clear_b, existing_a_only], "fill_a")
    fill_b = b.node("Sub", [clear_a, existing_b_only], "fill_b")
    updates = b.node(
        "Concat",
        ["zeros6", clear_a, clear_b, fill_overlap, fill_a, fill_b],
        "updates",
        axis=0,
    )
    b.node("ScatterND", [IN_NAME, "scatter_partitioned", updates], OUT_NAME)
    return make_model(b.nodes, b.initializers, opset=11)


def variants() -> list[Variant]:
    return [
        Variant("partitioned_op11", build_partitioned_static_scatternd_op11),
        Variant("static_union_op11", build_static_union_scatternd_op11),
        Variant("scatternd_op11", build_scatternd_op11),
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
    print(f"{'variant':<18} {'ok':<5} {'valid':<6} {'memory':>8} {'params':>7} {'cost':>8} {'score':>10}")
    for name, result in sorted(results.items(), key=lambda item: (item[0] != best_name, item[0])):
        score = result["score"]
        score_text = f"{score:.6f}" if isinstance(score, float) else "None"
        print(
            f"{name:<18} {str(result['correct']):<5} {str(result['valid']):<6} "
            f"{str(result['memory']):>8} {str(result['params']):>7} "
            f"{str(result['cost']):>8} {score_text:>10}"
        )
        if not result["correct"]:
            print(f"  splits: {result['splits']}")
        if result["error"]:
            print(f"  error: {str(result['error']).strip()}")
    print(f"best: {best_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and benchmark task139 ONNX variants.")
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
