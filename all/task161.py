"""Compact ONNX generator for NeuroGolf task161.

Task rule: the input is a rectangular ARC grid padded inside the standard
1x10x30x30 one-hot tensor. Among the non-background colors, exactly one color
places four pixels on opposite borders: either matching left/right rows,
matching top/bottom columns, or one pair of each. The output keeps only that
color, drawing a full line across every matched border row/column inside the
original rectangle; all other valid cells become background, and padding stays
all-zero.

The ONNX graph infers the valid rectangle from one-hot occupancy, detects
opposite-border pairs for every non-background channel, broadcasts those compact
row/column pair masks into line masks, and casts to float only at the graph
output.
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


TASK_NUM = "161"
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

    def init_f32(self, name: str, values: list[float]) -> str:
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


def build_dynamic_border_pairs_bool() -> onnx.ModelProto:
    """General solution: dynamic rectangle, bool line masks, final float cast."""
    b = Builder()
    zero = b.init_f32("zero", [0.0])
    one = b.init_f32("one", [1.0])
    one_half = b.init_f32("one_half", [1.5])
    three_half = b.init_f32("three_half", [3.5])
    four_half = b.init_f32("four_half", [4.5])
    ch_col_starts = b.init_i64("ch_col_starts", [1, 0])
    ch_col_ends = b.init_i64("ch_col_ends", [10, 1])
    ch_col_axes = b.init_i64("ch_col_axes", [1, 3])
    ch_row_starts = b.init_i64("ch_row_starts", [1, 0])
    ch_row_ends = b.init_i64("ch_row_ends", [10, 1])
    ch_row_axes = b.init_i64("ch_row_axes", [1, 2])
    ch_starts = b.init_i64("ch_starts", [1])
    ch_ends = b.init_i64("ch_ends", [10])
    ch_axis = b.init_i64("ch_axis", [1])
    col0_starts = b.init_i64("col0_starts", [0])
    col0_ends = b.init_i64("col0_ends", [1])
    col_axis = b.init_i64("col_axis", [3])
    row0_starts = b.init_i64("row0_starts", [0])
    row0_ends = b.init_i64("row0_ends", [1])
    row_axis = b.init_i64("row_axis", [2])

    top_all = b.node("Slice", [IN_NAME, row0_starts, row0_ends, row_axis], "top_all")
    top_occ = b.node("ReduceSum", [top_all], "top_occ", axes=[1], keepdims=1)
    width_count = b.node("ReduceSum", [top_occ], "width_count", axes=[3], keepdims=0)
    right_idx_f = b.node("Sub", [width_count, one], "right_idx_f")
    right_idx_keep = b.node("Cast", [right_idx_f], "right_idx_keep", to=TensorProto.INT64)
    right_idx = b.node("Squeeze", [right_idx_keep], "right_idx", axes=[0, 1, 2])

    left_all = b.node("Slice", [IN_NAME, col0_starts, col0_ends, col_axis], "left_all")
    left_occ = b.node("ReduceSum", [left_all], "left_occ", axes=[1], keepdims=1)
    height_count = b.node("ReduceSum", [left_occ], "height_count", axes=[2], keepdims=0)
    bottom_idx_f = b.node("Sub", [height_count, one], "bottom_idx_f")
    bottom_idx_keep = b.node("Cast", [bottom_idx_f], "bottom_idx_keep", to=TensorProto.INT64)
    bottom_idx = b.node("Squeeze", [bottom_idx_keep], "bottom_idx", axes=[0, 1, 2])
    valid_cols = b.node("Greater", [top_occ, zero], "valid_cols")
    valid_rows = b.node("Greater", [left_occ, zero], "valid_rows")
    valid = b.node("And", [valid_rows, valid_cols], "valid")

    left_px = b.node("Slice", [IN_NAME, ch_col_starts, ch_col_ends, ch_col_axes], "left_px")
    left_sq = b.node("Squeeze", [left_px], "left_sq", axes=[3])
    left = b.node("Greater", [left_sq, zero], "left")
    right_all = b.node("Gather", [IN_NAME, right_idx], "right_all", axis=3)
    right_count = b.node("Slice", [right_all, ch_starts, ch_ends, ch_axis], "right_count")
    right = b.node("Greater", [right_count, zero], "right")
    h_pairs = b.node("And", [left, right], "h_pairs")

    top_px = b.node("Slice", [IN_NAME, ch_row_starts, ch_row_ends, ch_row_axes], "top_px")
    top_sq = b.node("Squeeze", [top_px], "top_sq", axes=[2])
    top = b.node("Greater", [top_sq, zero], "top")
    bottom_all = b.node("Gather", [IN_NAME, bottom_idx], "bottom_all", axis=2)
    bottom_count = b.node("Slice", [bottom_all, ch_starts, ch_ends, ch_axis], "bottom_count")
    bottom = b.node("Greater", [bottom_count, zero], "bottom")
    v_pairs = b.node("And", [top, bottom], "v_pairs")

    h_pairs_f = b.node("Cast", [h_pairs], "h_pairs_f", to=TensorProto.FLOAT)
    h_pair_count = b.node("ReduceSum", [h_pairs_f], "h_pair_count", axes=[2], keepdims=0)
    v_pairs_f = b.node("Cast", [v_pairs], "v_pairs_f", to=TensorProto.FLOAT)
    v_pair_count = b.node("ReduceSum", [v_pairs_f], "v_pair_count", axes=[2], keepdims=0)
    pair_count = b.node("Add", [h_pair_count, v_pair_count], "pair_count")
    at_least_two = b.node("Greater", [pair_count, one_half], "at_least_two")
    left_f = b.node("Cast", [left], "left_f", to=TensorProto.FLOAT)
    left_count = b.node("ReduceSum", [left_f], "left_count", axes=[2], keepdims=0)
    right_total = b.node("ReduceSum", [right_count], "right_total", axes=[2], keepdims=0)
    top_f = b.node("Cast", [top], "top_f", to=TensorProto.FLOAT)
    top_count = b.node("ReduceSum", [top_f], "top_count", axes=[2], keepdims=0)
    bottom_total = b.node("ReduceSum", [bottom_count], "bottom_total", axes=[2], keepdims=0)
    horizontal_border_count = b.node("Add", [left_count, right_total], "horizontal_border_count")
    vertical_border_count = b.node("Add", [top_count, bottom_total], "vertical_border_count")
    border_count = b.node("Add", [horizontal_border_count, vertical_border_count], "border_count")
    at_least_four = b.node("Greater", [border_count, three_half], "at_least_four")
    less_than_five = b.node("Less", [border_count, four_half], "less_than_five")
    four_border = b.node("And", [at_least_four, less_than_five], "four_border")
    target_color = b.node("And", [at_least_two, four_border], "target_color")
    target_color_u = b.node("Unsqueeze", [target_color], "target_color_u", axes=[2])
    h_target_pairs = b.node("And", [h_pairs, target_color_u], "h_target_pairs")
    v_target_pairs = b.node("And", [v_pairs, target_color_u], "v_target_pairs")

    h_target_pairs_f = b.node("Cast", [h_target_pairs], "h_target_pairs_f", to=TensorProto.FLOAT)
    h_any_count = b.node("ReduceSum", [h_target_pairs_f], "h_any_count", axes=[1], keepdims=1)
    h_any = b.node("Greater", [h_any_count, zero], "h_any")
    h_any_line = b.node("Unsqueeze", [h_any], "h_any_line", axes=[3])
    v_target_pairs_f = b.node("Cast", [v_target_pairs], "v_target_pairs_f", to=TensorProto.FLOAT)
    v_any_count = b.node("ReduceSum", [v_target_pairs_f], "v_any_count", axes=[1], keepdims=1)
    v_any = b.node("Greater", [v_any_count, zero], "v_any")
    v_any_line = b.node("Unsqueeze", [v_any], "v_any_line", axes=[2])
    line_any = b.node("Or", [h_any_line, v_any_line], "line_any")

    h_lines = b.node("Unsqueeze", [h_target_pairs], "h_lines", axes=[3])
    v_lines = b.node("Unsqueeze", [v_target_pairs], "v_lines", axes=[2])
    fg_lines = b.node("Or", [h_lines, v_lines], "fg_lines")
    fg = b.node("And", [fg_lines, valid], "fg")

    not_line_any = b.node("Not", [line_any], "not_line_any")
    bg = b.node("And", [valid, not_line_any], "bg")
    out_bool = b.node("Concat", [bg, fg], "out_bool", axis=1)
    b.node("Cast", [out_bool], OUT_NAME, to=TensorProto.FLOAT)
    return make_model(b.nodes, b.initializers)

def variants() -> list[Variant]:
    return [Variant("dynamic_border_pairs_bool", build_dynamic_border_pairs_bool)]


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
    with tempfile.TemporaryDirectory(prefix="task161_") as tmp:
        tmpdir = Path(tmp)
        for name, model in built.items():
            ok, splits = verify_correct(model)
            path = tmpdir / f"{TASK_ID}.onnx"
            write_model(model, path)
            score = score_file(path)
            results[name] = {"ok": ok, "splits": splits, **score}

    valid = [
        (name, result)
        for name, result in results.items()
        if result["ok"] and result["valid"] and result["cost"] is not None
    ]
    if not valid:
        raise RuntimeError(f"no valid correct variants: {results}")
    best_name, _ = min(valid, key=lambda item: int(item[1]["cost"]))
    return results, best_name, built


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Build and score {TASK_ID}.")
    parser.add_argument("--out", type=Path, default=BEST_PATH, help="destination ONNX path")
    args = parser.parse_args()

    results, best_name, built = benchmark_variants()
    write_model(built[best_name], args.out)
    final = score_file(args.out)
    ok, splits = verify_correct(built[best_name])

    print(f"best variant: {best_name}")
    print(f"wrote: {args.out}")
    print(f"score: {final['score']:.6f}")
    print(f"cost: {final['cost']}")
    print(f"memory: {final['memory']}")
    print(f"params: {final['params']}")
    print(f"correct: {ok}")
    for split, (passed, checked) in splits.items():
        print(f"{split}: {passed}/{checked}")

    if len(results) > 1:
        print("variants:")
        for name, result in sorted(results.items(), key=lambda item: (item[1]["cost"] is None, item[1]["cost"] or 10**18)):
            print(
                f"  {name}: ok={result['ok']} valid={result['valid']} "
                f"cost={result['cost']} memory={result['memory']} params={result['params']} "
                f"score={result['score']}"
            )


if __name__ == "__main__":
    main()
