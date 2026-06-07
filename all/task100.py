"""Generate the selected ONNX solver for NeuroGolf task100.

Task rule: the logical input is a 10x10 grid containing exactly two hollow
non-background rectangles in different colors. The output is a solid 2x2 block
filled with the color of the rectangle with the larger bounding-box area; the
smaller rectangle is ignored. The submitted NeuroGolf tensor is still the
standard 30x30 canvas, so the 2x2 answer is placed at the top-left and the rest
is left as zero padding.

The best graph reduces the full input directly to per-color row occupancies.
For a hollow rectangle, the maximum row occupancy is its width, and the total
colored-cell count is ``2 * height + 2 * width - 4``; this recovers height and
area without a column summary or row-edge ArgMax chain. It computes those tiny
area values for all 10 colors, slices out background only after the reduction,
then uses OneHot and pads the final 9-channel 2x2 block into ``output`` with a
leading zero background channel.
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

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task100"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
OPSET = 10


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.counter = 0

    def name(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def node(self, op_type: str, inputs: list[str], prefix: str, **attrs: Any) -> str:
        out = self.name(prefix)
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out

    def init_i64(self, name: str, values: list[int] | np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name))
        return name

    def init_u8(self, name: str, values: list[int] | np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.uint8), name))
        return name

    def init_f32(self, name: str, values: list[float] | np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.float32), name))
        return name


def make_model(b: Builder, graph_name: str) -> onnx.ModelProto:
    graph = helper.make_graph(
        b.nodes,
        graph_name,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        b.initializers,
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


def add_2x2_output_from_fg_arg(b: Builder, selected_fg_arg: str, *, prefix: str) -> None:
    ids = b.init_i64(f"{prefix}_ids", np.arange(9, dtype=np.int64).reshape(1, 9, 1, 1))
    expand_shape = b.init_i64(f"{prefix}_expand", [1, 9, 2, 2])
    onehot_bool = b.node("Equal", [ids, selected_fg_arg], f"{prefix}_eq")
    onehot = b.node("Cast", [onehot_bool], f"{prefix}_f", to=TensorProto.FLOAT)
    block = b.node("Expand", [onehot, expand_shape], f"{prefix}_block")
    b.nodes.append(
        helper.make_node(
            "Pad",
            [block],
            [OUT_NAME],
            mode="constant",
            pads=[0, 1, 0, 0, 0, 0, 28, 28],
            value=0.0,
        )
    )


def add_2x2_output_from_fg_arg_onehot(b: Builder, selected_fg_arg: str, *, prefix: str) -> None:
    depth = b.init_i64(f"{prefix}_depth", np.array(9, dtype=np.int64))
    values = b.init_f32(f"{prefix}_values", [0.0, 1.0])
    expand_shape = b.init_i64(f"{prefix}_expand", [1, 9, 2, 2])
    onehot = b.node("OneHot", [selected_fg_arg, depth, values], f"{prefix}_onehot", axis=1)
    block = b.node("Expand", [onehot, expand_shape], f"{prefix}_block")
    b.nodes.append(
        helper.make_node(
            "Pad",
            [block],
            [OUT_NAME],
            mode="constant",
            pads=[0, 1, 0, 0, 0, 0, 28, 28],
            value=0.0,
        )
    )


def build_bbox_argmax_u8_model() -> onnx.ModelProto:
    """Previous candidate: compact occupancy summaries and uint8 ArgMax extents."""
    b = Builder()
    zero_f = b.init_f32("zf", [0.0])
    zero_i = b.init_i64("zi", [0])
    ten_i = b.init_i64("ten", [10])
    starts = b.init_i64("s", [0, 1, 0, 0])
    row_ends = b.init_i64("re", [1, 10, 10, 1])
    col_ends = b.init_i64("ce", [1, 10, 1, 10])
    rev_start = b.init_i64("rs", [9])
    rev_end = b.init_i64("rn", [-11])
    neg_step = b.init_i64("rt", [-1])
    row_axis = b.init_i64("ra", [2])
    col_axis = b.init_i64("ca", [3])

    row_full = b.node("ReduceSum", [IN_NAME], "row_full", axes=[3], keepdims=1)
    row_occ = b.node("Slice", [row_full, starts, row_ends], "row_occ")
    col_full = b.node("ReduceSum", [IN_NAME], "col_full", axes=[2], keepdims=1)
    col_occ = b.node("Slice", [col_full, starts, col_ends], "col_occ")

    row_present = b.node("Greater", [row_occ, zero_f], "row_present")
    row_u8 = b.node("Cast", [row_present], "row_u8", to=TensorProto.UINT8)
    counts = b.node("ReduceSum", [row_occ], "counts", axes=[2], keepdims=1)
    any_row = b.node("Greater", [counts, zero_f], "any_row")
    row_first = b.node("ArgMax", [row_u8], "row_first", axis=2, keepdims=1)
    row_rev = b.node("Slice", [row_u8, rev_start, rev_end, row_axis, neg_step], "row_rev")
    row_last_from_end = b.node("ArgMax", [row_rev], "row_last_end", axis=2, keepdims=1)
    row_edge_sum = b.node("Add", [row_first, row_last_from_end], "row_edge_sum")
    height = b.node("Sub", [ten_i, row_edge_sum], "height")

    col_present = b.node("Greater", [col_occ, zero_f], "col_present")
    col_u8 = b.node("Cast", [col_present], "col_u8", to=TensorProto.UINT8)
    col_first = b.node("ArgMax", [col_u8], "col_first", axis=3, keepdims=1)
    col_rev = b.node("Slice", [col_u8, rev_start, rev_end, col_axis, neg_step], "col_rev")
    col_last_from_end = b.node("ArgMax", [col_rev], "col_last_end", axis=3, keepdims=1)
    col_edge_sum = b.node("Add", [col_first, col_last_from_end], "col_edge_sum")
    width = b.node("Sub", [ten_i, col_edge_sum], "width")

    area = b.node("Mul", [height, width], "area")
    masked_area = b.node("Where", [any_row, area, zero_i], "masked_area")
    selected = b.node("ArgMax", [masked_area], "selected", axis=1, keepdims=1)
    add_2x2_output_from_fg_arg(b, selected, prefix="out")
    return make_model(b, "task100_bbox_argmax_u8")


def build_row_height_count_model() -> onnx.ModelProto:
    """Use row extent plus hollow-rectangle perimeter count to recover area."""
    b = Builder()
    zero_f = b.init_f32("zf", [0.0])
    two_f = b.init_f32("two", [2.0])
    ten_i = b.init_i64("ten", [10])
    starts = b.init_i64("s", [0, 1, 0, 0])
    row_ends = b.init_i64("re", [1, 10, 10, 1])
    rev_start = b.init_i64("rs", [9])
    rev_end = b.init_i64("rn", [-11])
    neg_step = b.init_i64("rt", [-1])
    row_axis = b.init_i64("ra", [2])

    row_full = b.node("ReduceSum", [IN_NAME], "row_full", axes=[3], keepdims=1)
    row_occ = b.node("Slice", [row_full, starts, row_ends], "row_occ")
    row_present = b.node("Greater", [row_occ, zero_f], "row_present")
    row_u8 = b.node("Cast", [row_present], "row_u8", to=TensorProto.UINT8)
    counts = b.node("ReduceSum", [row_occ], "counts", axes=[2], keepdims=1)
    row_first = b.node("ArgMax", [row_u8], "row_first", axis=2, keepdims=1)
    row_rev = b.node("Slice", [row_u8, rev_start, rev_end, row_axis, neg_step], "row_rev")
    row_last_from_end = b.node("ArgMax", [row_rev], "row_last_end", axis=2, keepdims=1)
    row_edge_sum = b.node("Add", [row_first, row_last_from_end], "row_edge_sum")
    height = b.node("Sub", [ten_i, row_edge_sum], "height")
    height_f = b.node("Cast", [height], "height_f", to=TensorProto.FLOAT)

    half_count = b.node("Div", [counts, two_f], "half_count")
    width = b.node("Sub", [b.node("Add", [half_count, two_f], "count_plus_two"), height_f], "width")
    area = b.node("Mul", [height_f, width], "area")
    selected = b.node("ArgMax", [area], "selected", axis=1, keepdims=1)
    add_2x2_output_from_fg_arg(b, selected, prefix="out")
    return make_model(b, "task100_row_height_count")


def build_row_count_model() -> onnx.ModelProto:
    """Use occupied-row count as height, then recover width from perimeter."""
    b = Builder()
    zero_f = b.init_f32("zf", [0.0])
    two_f = b.init_f32("two", [2.0])
    starts = b.init_i64("s", [0, 1, 0, 0])
    row_ends = b.init_i64("re", [1, 10, 10, 1])

    row_full = b.node("ReduceSum", [IN_NAME], "row_full", axes=[3], keepdims=1)
    row_occ = b.node("Slice", [row_full, starts, row_ends], "row_occ")
    row_present = b.node("Greater", [row_occ, zero_f], "row_present")
    row_present_f = b.node("Cast", [row_present], "row_present_f", to=TensorProto.FLOAT)
    height = b.node("ReduceSum", [row_present_f], "height", axes=[2], keepdims=1)
    counts = b.node("ReduceSum", [row_occ], "counts", axes=[2], keepdims=1)

    half_count = b.node("Div", [counts, two_f], "half_count")
    width = b.node("Sub", [b.node("Add", [half_count, two_f], "count_plus_two"), height], "width")
    area = b.node("Mul", [height, width], "area")
    selected = b.node("ArgMax", [area], "selected", axis=1, keepdims=1)
    add_2x2_output_from_fg_arg(b, selected, prefix="out")
    return make_model(b, "task100_row_count")


def build_row_max_width_model() -> onnx.ModelProto:
    """Use max row occupancy as width, then recover height from perimeter."""
    b = Builder()
    two_f = b.init_f32("two", [2.0])
    starts = b.init_i64("s", [0, 1, 0, 0])
    row_ends = b.init_i64("re", [1, 10, 10, 1])

    row_full = b.node("ReduceSum", [IN_NAME], "row_full", axes=[3], keepdims=1)
    row_occ = b.node("Slice", [row_full, starts, row_ends], "row_occ")
    width = b.node("ReduceMax", [row_occ], "width", axes=[2], keepdims=1)
    counts = b.node("ReduceSum", [row_occ], "counts", axes=[2], keepdims=1)

    half_count = b.node("Div", [counts, two_f], "half_count")
    height = b.node("Sub", [b.node("Add", [half_count, two_f], "count_plus_two"), width], "height")
    area = b.node("Mul", [height, width], "area")
    selected = b.node("ArgMax", [area], "selected", axis=1, keepdims=1)
    add_2x2_output_from_fg_arg(b, selected, prefix="out")
    return make_model(b, "task100_row_max_width")


def build_all_rows_area_slice_model() -> onnx.ModelProto:
    """Compute rectangle area for all colors, then discard background area."""
    b = Builder()
    two_f = b.init_f32("two", [2.0])
    starts = b.init_i64("s", [0, 1, 0, 0])
    ends = b.init_i64("e", [1, 10, 1, 1])

    row_full = b.node("ReduceSum", [IN_NAME], "row_full", axes=[3], keepdims=1)
    width = b.node("ReduceMax", [row_full], "width", axes=[2], keepdims=1)
    counts = b.node("ReduceSum", [row_full], "counts", axes=[2], keepdims=1)
    half_count = b.node("Div", [counts, two_f], "half_count")
    height = b.node("Sub", [b.node("Add", [half_count, two_f], "count_plus_two"), width], "height")
    area = b.node("Mul", [height, width], "area")
    fg_area = b.node("Slice", [area, starts, ends], "fg_area")
    selected = b.node("ArgMax", [fg_area], "selected", axis=1, keepdims=0)
    add_2x2_output_from_fg_arg_onehot(b, selected, prefix="out")
    return make_model(b, "task100_all_rows_area_slice")


def build_bbox_where_float_model() -> onnx.ModelProto:
    """Fallback candidate: float Where+ReduceMax extents over compact summaries."""
    b = Builder()
    zero = b.init_f32("z", [0.0])
    eight = b.init_f32("eight", [8.0])
    starts = b.init_i64("s", [0, 1, 0, 0])
    row_ends = b.init_i64("re", [1, 10, 10, 1])
    col_ends = b.init_i64("ce", [1, 10, 1, 10])
    row_idx = b.init_f32("ri", np.arange(10, dtype=np.float32).reshape(1, 1, 10, 1))
    row_rev_idx = b.init_f32("rr", np.arange(9, -1, -1, dtype=np.float32).reshape(1, 1, 10, 1))
    col_idx = b.init_f32("ci", np.arange(10, dtype=np.float32).reshape(1, 1, 1, 10))
    col_rev_idx = b.init_f32("cr", np.arange(9, -1, -1, dtype=np.float32).reshape(1, 1, 1, 10))

    row_full = b.node("ReduceSum", [IN_NAME], "row_full", axes=[3], keepdims=1)
    row_occ = b.node("Slice", [row_full, starts, row_ends], "row_occ")
    col_full = b.node("ReduceSum", [IN_NAME], "col_full", axes=[2], keepdims=1)
    col_occ = b.node("Slice", [col_full, starts, col_ends], "col_occ")
    counts = b.node("ReduceSum", [row_occ], "counts", axes=[2], keepdims=1)
    present = b.node("Greater", [counts, zero], "present")

    rp = b.node("Greater", [row_occ, zero], "rp")
    rs = b.node("Where", [rp, row_idx, zero], "rs")
    rrs = b.node("Where", [rp, row_rev_idx, zero], "rrs")
    rmax = b.node("ReduceMax", [rs], "rmax", axes=[2], keepdims=1)
    rrmax = b.node("ReduceMax", [rrs], "rrmax", axes=[2], keepdims=1)
    height = b.node("Sub", [b.node("Add", [rmax, rrmax], "rh_sum"), eight], "height")

    cp = b.node("Greater", [col_occ, zero], "cp")
    cs = b.node("Where", [cp, col_idx, zero], "cs")
    crs = b.node("Where", [cp, col_rev_idx, zero], "crs")
    cmax = b.node("ReduceMax", [cs], "cmax", axes=[3], keepdims=1)
    crmax = b.node("ReduceMax", [crs], "crmax", axes=[3], keepdims=1)
    width = b.node("Sub", [b.node("Add", [cmax, crmax], "cw_sum"), eight], "width")

    area = b.node("Mul", [height, width], "area")
    masked_area = b.node("Where", [present, area, zero], "masked_area")
    selected = b.node("ArgMax", [masked_area], "selected", axis=1, keepdims=1)
    add_2x2_output_from_fg_arg(b, selected, prefix="out")
    return make_model(b, "task100_bbox_where_float")


def build_perimeter_count_model() -> onnx.ModelProto:
    """Suggested count-only variant; kept to prove it is not correct on this data."""
    b = Builder()
    starts = b.init_i64("s", [0, 1, 0, 0])
    row_ends = b.init_i64("re", [1, 10, 10, 1])
    row_full = b.node("ReduceSum", [IN_NAME], "row_full", axes=[3], keepdims=1)
    row_occ = b.node("Slice", [row_full, starts, row_ends], "row_occ")
    counts = b.node("ReduceSum", [row_occ], "counts", axes=[2], keepdims=1)
    selected = b.node("ArgMax", [counts], "selected", axis=1, keepdims=1)
    add_2x2_output_from_fg_arg(b, selected, prefix="out")
    return make_model(b, "task100_perimeter_count")


def build_variants() -> dict[str, onnx.ModelProto]:
    return {
        "bbox_argmax_u8": build_bbox_argmax_u8_model(),
        "all_rows_area_slice": build_all_rows_area_slice_model(),
        "row_max_width": build_row_max_width_model(),
        "row_count": build_row_count_model(),
        "row_height_count": build_row_height_count_model(),
        "bbox_where_float": build_bbox_where_float_model(),
        "perimeter_count": build_perimeter_count_model(),
    }


def load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def verify_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    split_counts: dict[str, tuple[int, int]] = {}
    all_ok = True
    for split in ("train", "test", "arc-gen"):
        passed = 0
        checked = 0
        for example in load_task_data().get(split, []):
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
        split_counts[split] = (passed, checked)
    return all_ok, split_counts


def write_model(model: onnx.ModelProto, path: Path) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(path))


def benchmark_variants() -> tuple[dict[str, dict[str, Any]], str, dict[str, onnx.ModelProto]]:
    variants = build_variants()
    results: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}_") as tmp:
        tmp_path = Path(tmp)
        for name, model in variants.items():
            ok, splits = verify_correct(model)
            path = tmp_path / f"{TASK_ID}.onnx"
            write_model(model, path)
            result = score_file(path)
            result["correct"] = ok
            result["splits"] = splits
            results[name] = result

    def sort_key(name: str) -> int:
        result = results[name]
        if not result["valid"] or not result["correct"]:
            return 10**18
        return int(result["cost"])

    best_name = min(results, key=sort_key)
    if sort_key(best_name) >= 10**18:
        raise RuntimeError(f"no valid correct variant: {results}")
    return results, best_name, variants


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
        if result.get("splits"):
            print(f"  splits: {result['splits']}")
        if result["error"]:
            print(f"  error: {str(result['error']).strip()}")
    print(f"best: {best_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and benchmark task100 ONNX variants.")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()

    results, best_name, variants = benchmark_variants()
    print_benchmark(results, best_name)
    if not args.benchmark_only:
        OUT_DIR.mkdir(exist_ok=True)
        write_model(variants[best_name], BEST_PATH)
        print(f"wrote: {BEST_PATH}")


if __name__ == "__main__":
    main()
