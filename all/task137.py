"""Optimized ONNX generator for NeuroGolf task137.

Task rule: the input grid contains exactly three same-colored non-background
cells on a diagonal. Their average row/column is the center, and half the row
range is the spacing. The output keeps the same grid size and color, drawing
all clipped square-ring border cells whose Chebyshev distance from the center
is a multiple of that spacing; all other in-grid cells are black and padding
outside the ARC grid remains all-zero.
"""

from __future__ import annotations

import argparse
import json
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


TASK_NUM = "137"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = Path(__file__).resolve().parent / f"{TASK_ID}.onnx"
SOLUTION_PATH = ROOT / "solution.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
OPSET = 10


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []

    def init(self, name: str, array: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(array, name))
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


def build_model() -> onnx.ModelProto:
    b = Builder()
    rows = np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)
    rows_rev = (30.0 - np.arange(30, dtype=np.float32)).reshape(1, 1, 30, 1)
    cols = np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)
    non_bg_weight = np.array([False] + [True] * 9, dtype=np.bool_).reshape(1, 10, 1, 1)

    b.init("rows", rows)
    b.init("rows_rev", rows_rev)
    b.init("cols", cols)
    b.init("non_bg_weight", non_bg_weight)
    b.init("c0_starts", np.array([0], dtype=np.int64))
    b.init("c0_ends", np.array([1], dtype=np.int64))
    b.init("c_axis", np.array([1], dtype=np.int64))
    b.init("three", np.array(3.0, dtype=np.float32))
    b.init("two", np.array(2.0, dtype=np.float32))
    b.init("eps", np.array(0.00001, dtype=np.float32))
    b.init("zero", np.array(0.0, dtype=np.float32))
    b.init("thirty", np.array(30.0, dtype=np.float32))

    bg_f = b.node("Slice", [IN_NAME, "c0_starts", "c0_ends", "c_axis"], "bg_f")
    bg_cell = b.node("Greater", [bg_f, "zero"], "bg_cell")

    color_counts = b.node("ReduceSum", [IN_NAME], "color_counts", axes=[2, 3], keepdims=1)
    color_sel_all = b.node("Greater", [color_counts, "zero"], "color_sel_all")
    color_sel = b.node("And", [color_sel_all, "non_bg_weight"], "color_sel")

    row_valid = b.node("ReduceSum", [IN_NAME], "row_valid", axes=[1, 3], keepdims=1)
    col_valid = b.node("ReduceSum", [IN_NAME], "col_valid", axes=[1, 2], keepdims=1)
    row_bg = b.node("ReduceSum", [bg_f], "row_bg", axes=[3], keepdims=1)
    col_bg = b.node("ReduceSum", [bg_f], "col_bg", axes=[2], keepdims=1)
    row_count = b.node("Sub", [row_valid, row_bg], "row_count")
    col_count = b.node("Sub", [col_valid, col_bg], "col_count")
    row_sum_grid = b.node("Mul", [row_count, "rows"], "row_sum_grid")
    col_sum_grid = b.node("Mul", [col_count, "cols"], "col_sum_grid")
    row_sum = b.node("ReduceSum", [row_sum_grid], "row_sum", axes=[2, 3], keepdims=1)
    col_sum = b.node("ReduceSum", [col_sum_grid], "col_sum", axes=[2, 3], keepdims=1)
    cy = b.node("Div", [row_sum, "three"], "cy")
    cx = b.node("Div", [col_sum, "three"], "cx")

    row_max = b.node("ReduceMax", [row_sum_grid], "row_max", axes=[2, 3], keepdims=1)
    row_rev_grid = b.node("Mul", [row_count, "rows_rev"], "row_rev_grid")
    row_rev_max = b.node("ReduceMax", [row_rev_grid], "row_rev_max", axes=[2, 3], keepdims=1)
    row_min = b.node("Sub", ["thirty", row_rev_max], "row_min")
    row_span = b.node("Sub", [row_max, row_min], "row_span")
    spacing = b.node("Div", [row_span, "two"], "spacing")

    dy = b.node("Abs", [b.node("Sub", ["rows", cy], "dy_signed")], "dy")
    dx = b.node("Abs", [b.node("Sub", ["cols", cx], "dx_signed")], "dx")
    dy_rem = b.node("Mod", [dy, spacing], "dy_rem", fmod=1)
    dx_rem = b.node("Mod", [dx, spacing], "dx_rem", fmod=1)
    dy_ring = b.node("Less", [dy_rem, "eps"], "dy_ring")
    dx_ring = b.node("Less", [dx_rem, "eps"], "dx_ring")
    dy_lt_dx = b.node("Less", [dy, dx], "dy_lt_dx")
    dy_ge_dx = b.node("Not", [dy_lt_dx], "dy_ge_dx")
    y_part = b.node("And", [dy_ge_dx, dy_ring], "y_part")
    x_part = b.node("And", [dy_lt_dx, dx_ring], "x_part")
    ring_raw = b.node("Or", [y_part, x_part], "ring_raw")
    ring = b.node("And", [ring_raw, bg_cell], "ring")

    color_sel_f = b.node("Cast", [color_sel], "color_sel_f", to=TensorProto.FLOAT)
    b.node("Where", [ring, color_sel_f, IN_NAME], OUT_NAME)
    return make_model(b.nodes, b.initializers)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build task137 NeuroGolf ONNX solver.")
    parser.add_argument("--output", type=Path, default=SOLUTION_PATH, help="primary output ONNX path")
    parser.add_argument("--repo-copy", action="store_true", help=f"also write {BEST_PATH}")
    args = parser.parse_args()

    model = build_model()
    ok, splits = verify_correct(model)
    if not ok:
        raise SystemExit(f"verification failed: {splits}")

    with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}_") as tmp:
        tmp_path = Path(tmp) / f"{TASK_ID}.onnx"
        write_model(model, tmp_path)
        result = score_file(tmp_path)

    write_model(model, args.output)
    if args.repo_copy or args.output.resolve() != BEST_PATH.resolve():
        write_model(model, BEST_PATH)

    score_text = f"{result['score']:.6f}" if isinstance(result["score"], float) else "INVALID"
    print(f"verified: {splits}")
    print(
        f"score: memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={score_text}"
    )
    print(f"wrote: {args.output}")
    if args.output.resolve() != BEST_PATH.resolve():
        print(f"wrote: {BEST_PATH}")


if __name__ == "__main__":
    main()
