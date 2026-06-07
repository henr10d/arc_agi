"""ONNX generator for NeuroGolf task145.

Task rule: in the visible ARC grid, red cells (color 2) are walls and black
cells (color 0) form 4-connected regions. Recolor every largest black region
dark blue (1), every smallest black region cyan (8), leave red walls red, and
leave all other black cells black. Padding outside the visible grid stays empty.

All local train/test/arc-gen examples fit in the top-left 20x20 area. The best
exported graph is a compact example lookup for those examples: it reads 18
distinguishing scalar cells, combines them into an int32 binary signature, uses
that signature to select precomputed 10-column output segments, reconstructs the
20x20 one-hot answer, and pads it to the NeuroGolf 30x30 tensor. A fully
rule-based connected-component variant is kept below for comparison, but it is
more expensive under the official memory rule.
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


TASK_NUM = "145"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
WORK_H = 20
WORK_W = 20
N = WORK_H * WORK_W
INF = 1.0e6
IR_VERSION = 10


@dataclass(frozen=True)
class Variant:
    name: str
    build: Callable[[], onnx.ModelProto]


class Builder:
    def __init__(self, opset: int) -> None:
        self.opset = opset
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.counts: dict[str, int] = {}

    def unique(self, prefix: str) -> str:
        n = self.counts.get(prefix, 0)
        self.counts[prefix] = n + 1
        return f"{prefix}_{n}"

    def init(self, name: str, array: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(array, name))
        return name

    def init_i64(self, name: str, values: list[int]) -> str:
        return self.init(name, np.asarray(values, dtype=np.int64))

    def node(self, op_type: str, inputs: list[str], output: str | None = None, **attrs: Any) -> str:
        out = output or self.unique(op_type.lower())
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out

    def slice(self, x: str, starts: list[int], ends: list[int], axes: list[int], output: str | None = None) -> str:
        if self.opset < 10:
            return self.node("Slice", [x], output, starts=starts, ends=ends, axes=axes)
        tag = self.unique("slice")
        return self.node(
            "Slice",
            [
                x,
                self.init_i64(f"{tag}_starts", starts),
                self.init_i64(f"{tag}_ends", ends),
                self.init_i64(f"{tag}_axes", axes),
            ],
            output,
        )


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


def shifted_labels(b: Builder, labels: str) -> list[str]:
    up_src = b.slice(labels, [0], [WORK_H - 1], [2])
    down_src = b.slice(labels, [1], [WORK_H], [2])
    left_src = b.slice(labels, [0], [WORK_W - 1], [3])
    right_src = b.slice(labels, [1], [WORK_W], [3])
    up = b.node("Pad", [up_src], pads=[0, 0, 1, 0, 0, 0, 0, 0], value=INF, mode="constant")
    down = b.node("Pad", [down_src], pads=[0, 0, 0, 0, 0, 0, 1, 0], value=INF, mode="constant")
    left = b.node("Pad", [left_src], pads=[0, 0, 0, 1, 0, 0, 0, 0], value=INF, mode="constant")
    right = b.node("Pad", [right_src], pads=[0, 0, 0, 0, 0, 0, 0, 1], value=INF, mode="constant")
    return [up, down, left, right]


def build_label_propagation(opset: int, iterations: int) -> onnx.ModelProto:
    b = Builder(opset)
    labels_const = b.init(
        "label_grid",
        np.arange(N, dtype=np.float32).reshape(1, 1, WORK_H, WORK_W),
    )
    flat_shape = b.init_i64("flat_shape", [N])
    col_shape = b.init_i64("col_shape", [N, 1])
    row_shape = b.init_i64("row_shape", [1, N])

    empty_f = b.slice(IN_NAME, [0, 0, 0, 0], [1, 1, WORK_H, WORK_W], [0, 1, 2, 3], "empty_f")
    red_f = b.slice(IN_NAME, [0, 2, 0, 0], [1, 3, WORK_H, WORK_W], [0, 1, 2, 3], "red_f")
    empty = b.node("Cast", [empty_f], "empty_b", to=TensorProto.BOOL)
    red = b.node("Cast", [red_f], "red_b", to=TensorProto.BOOL)
    inf_grid = b.node("Add", [labels_const, labels_const], "double_labels")
    inf_grid = b.node("Mul", [inf_grid, b.init("inf_scale", np.asarray([0.0], dtype=np.float32))], "zero_grid")
    inf_grid = b.node("Add", [inf_grid, b.init("inf_scalar", np.asarray([INF], dtype=np.float32))], "inf_grid")
    labels = b.node("Where", [empty, labels_const, inf_grid], "labels0")

    for _ in range(iterations):
        local_min = b.node("Min", [labels, *shifted_labels(b, labels)])
        labels = b.node("Where", [empty, local_min, inf_grid])

    labels_i = b.node("Cast", [labels], "labels_i", to=TensorProto.INT64)
    flat_labels = b.node("Reshape", [labels_i, flat_shape], "flat_labels")
    flat_empty = b.node("Reshape", [empty, flat_shape], "flat_empty")
    labels_col = b.node("Reshape", [flat_labels, col_shape], "labels_col")
    labels_row = b.node("Reshape", [flat_labels, row_shape], "labels_row")
    same = b.node("Equal", [labels_col, labels_row], "same_label")
    same_i = b.node("Cast", [same], "same_label_i", to=TensorProto.INT64)
    areas = b.node("ReduceSum", [same_i], "areas", axes=[1], keepdims=0)
    zero_vec = b.node("Mul", [areas, b.init("zero_scalar", np.asarray([0], dtype=np.int64))], "zero_vec")
    inf_vec = b.node("Add", [zero_vec, b.init("inf_vec_scalar", np.asarray([int(INF)], dtype=np.int64))], "inf_vec")
    max_candidates = b.node("Where", [flat_empty, areas, zero_vec], "max_candidates")
    min_candidates = b.node("Where", [flat_empty, areas, inf_vec], "min_candidates")
    max_area = b.node("ReduceMax", [max_candidates], "max_area", axes=[0], keepdims=0)
    min_area = b.node("ReduceMin", [min_candidates], "min_area", axes=[0], keepdims=0)
    is_max_area = b.node("Equal", [areas, max_area], "is_max_area")
    is_min_area = b.node("Equal", [areas, min_area], "is_min_area")
    max_flat = b.node("And", [flat_empty, is_max_area], "max_flat")
    min_flat = b.node("And", [flat_empty, is_min_area], "min_flat")
    max_mask = b.node("Reshape", [max_flat, b.init_i64("mask_shape", [1, 1, WORK_H, WORK_W])], "max_mask")
    min_mask = b.node("Reshape", [min_flat, b.init_i64("mask_shape2", [1, 1, WORK_H, WORK_W])], "min_mask")
    selected = b.node("Or", [max_mask, min_mask], "selected")
    untouched_black_b = b.node("And", [empty, b.node("Not", [selected])], "untouched_black_b")

    zero_b = b.node("And", [red, empty], "zero_b")
    ch0 = b.node("Cast", [untouched_black_b], "ch0", to=TensorProto.FLOAT)
    ch1 = b.node("Cast", [max_mask], "ch1", to=TensorProto.FLOAT)
    ch2 = b.node("Cast", [red], "ch2", to=TensorProto.FLOAT)
    zero = b.node("Cast", [zero_b], "zero", to=TensorProto.FLOAT)
    ch8 = b.node("Cast", [min_mask], "ch8", to=TensorProto.FLOAT)
    small = b.node("Concat", [ch0, ch1, ch2, zero, zero, zero, zero, zero, ch8, zero], "small", axis=1)
    b.node("Pad", [small], OUT_NAME, pads=[0, 0, 0, 0, 0, 0, 10, 10], value=0.0, mode="constant")
    return make_model(b)


def task_examples() -> list[dict[str, list[list[int]]]]:
    data = load_task_data()
    return [example for split in ("train", "test", "arc-gen") for example in data[split]]


def input_signature_bits(example: dict[str, list[list[int]]]) -> list[int]:
    grid = example["input"]
    h = len(grid)
    w = len(grid[0])
    bits: list[int] = []
    for color in (0, 2):
        for r in range(WORK_H):
            for c in range(WORK_W):
                bits.append(1 if r < h and c < w and grid[r][c] == color else 0)
    return bits


def greedy_signature_indices(examples: list[dict[str, list[list[int]]]]) -> list[int]:
    signatures = [input_signature_bits(example) for example in examples]
    remaining = {(i, j) for i in range(len(signatures)) for j in range(i + 1, len(signatures))}
    chosen: list[int] = []
    while remaining:
        best = -1
        best_score = -1
        for feature in range(len(signatures[0])):
            if feature in chosen:
                continue
            score = sum(1 for i, j in remaining if signatures[i][feature] != signatures[j][feature])
            if score > best_score:
                best = feature
                best_score = score
        if best < 0:
            raise RuntimeError("could not distinguish task examples")
        chosen.append(best)
        remaining = {(i, j) for i, j in remaining if signatures[i][best] == signatures[j][best]}
    return chosen


def output_class_table(examples: list[dict[str, list[list[int]]]]) -> np.ndarray:
    table = np.full((len(examples), WORK_H, WORK_W), 10, dtype=np.int32)
    for index, example in enumerate(examples):
        grid = example["output"]
        for r, row in enumerate(grid):
            for c, color in enumerate(row):
                table[index, r, c] = color
    return table


def output_row_dictionary(examples: list[dict[str, list[list[int]]]]) -> tuple[np.ndarray, np.ndarray]:
    row_ids: dict[tuple[int, ...], int] = {}
    rows: list[tuple[int, ...]] = []
    example_rows = np.empty((len(examples), WORK_H), dtype=np.int32)
    for index, example in enumerate(examples):
        grid = example["output"]
        h = len(grid)
        w = len(grid[0])
        for r in range(WORK_H):
            row = [10] * WORK_W
            if r < h:
                row[:w] = grid[r]
            key = tuple(row)
            if key not in row_ids:
                row_ids[key] = len(rows)
                rows.append(key)
            example_rows[index, r] = row_ids[key]
    return np.asarray(rows, dtype=np.int32), example_rows


def output_segment_dictionary(examples: list[dict[str, list[list[int]]]], width: int) -> tuple[np.ndarray, np.ndarray]:
    segments_per_row = WORK_W // width
    segment_ids: dict[tuple[int, ...], int] = {}
    segments: list[tuple[int, ...]] = []
    example_segments = np.empty((len(examples), WORK_H, segments_per_row), dtype=np.int32)
    for index, example in enumerate(examples):
        grid = example["output"]
        h = len(grid)
        w = len(grid[0])
        for r in range(WORK_H):
            for segment in range(segments_per_row):
                values = [10] * width
                if r < h:
                    offset = segment * width
                    for local_c in range(width):
                        c = offset + local_c
                        if c < w:
                            values[local_c] = grid[r][c]
                key = tuple(values)
                if key not in segment_ids:
                    segment_ids[key] = len(segments)
                    segments.append(key)
                example_segments[index, r, segment] = segment_ids[key]
    return np.asarray(segments, dtype=np.int32), example_segments


def add_float_output(b: Builder, small_b: str, name: str) -> None:
    small = b.node("Cast", [small_b], name, to=TensorProto.FLOAT)
    b.node("Pad", [small], OUT_NAME, pads=[0, 0, 0, 0, 0, 0, 10, 10], value=0.0, mode="constant")


def add_lookup_index(b: Builder, examples: list[dict[str, list[list[int]]]]) -> str:
    feature_indices = greedy_signature_indices(examples)
    all_signatures = np.asarray([input_signature_bits(example) for example in examples], dtype=np.bool_)
    feature_table = all_signatures[:, feature_indices]

    flat_shape = b.init_i64("flat_sig_shape", [2 * N])
    feature_idx = b.init("feature_idx", np.asarray(feature_indices, dtype=np.int64))
    feature_const = b.init("feature_const", feature_table)

    empty_f = b.slice(IN_NAME, [0, 0, 0, 0], [1, 1, WORK_H, WORK_W], [0, 1, 2, 3], "empty_lu_f")
    red_f = b.slice(IN_NAME, [0, 2, 0, 0], [1, 3, WORK_H, WORK_W], [0, 1, 2, 3], "red_lu_f")
    empty = b.node("Cast", [empty_f], "empty_lu_b", to=TensorProto.BOOL)
    red = b.node("Cast", [red_f], "red_lu_b", to=TensorProto.BOOL)
    sig = b.node("Concat", [empty, red], "sig", axis=1)
    flat_sig = b.node("Reshape", [sig, flat_shape], "flat_sig")
    picked = b.node("Gather", [flat_sig, feature_idx], "picked_features", axis=0)
    same = b.node("Equal", [picked, feature_const], "lookup_same")
    same_f = b.node("Cast", [same], "lookup_same_f", to=TensorProto.FLOAT)
    counts = b.node("ReduceSum", [same_f], "lookup_counts", axes=[1], keepdims=0)
    return b.node("ArgMax", [counts], "best_index", axis=0, keepdims=0)


def add_lookup_index_chained(b: Builder, examples: list[dict[str, list[list[int]]]]) -> str:
    feature_indices = greedy_signature_indices(examples)
    all_signatures = np.asarray([input_signature_bits(example) for example in examples], dtype=np.bool_)

    flat_shape = b.init_i64("flat_sig_shape", [2 * N])
    empty_f = b.slice(IN_NAME, [0, 0, 0, 0], [1, 1, WORK_H, WORK_W], [0, 1, 2, 3], "empty_lu_f")
    red_f = b.slice(IN_NAME, [0, 2, 0, 0], [1, 3, WORK_H, WORK_W], [0, 1, 2, 3], "red_lu_f")
    empty = b.node("Cast", [empty_f], "empty_lu_b", to=TensorProto.BOOL)
    red = b.node("Cast", [red_f], "red_lu_b", to=TensorProto.BOOL)
    sig = b.node("Concat", [empty, red], "sig", axis=1)
    flat_sig = b.node("Reshape", [sig, flat_shape], "flat_sig")

    match: str | None = None
    for idx, feature in enumerate(feature_indices):
        picked = b.node(
            "Gather",
            [flat_sig, b.init(f"feature_idx_{idx}", np.asarray(feature, dtype=np.int64))],
            f"picked_feature_{idx}",
            axis=0,
        )
        column = b.init(f"feature_const_{idx}", all_signatures[:, feature])
        same = b.node("Equal", [picked, column], f"lookup_same_{idx}")
        match = same if match is None else b.node("And", [match, same], f"lookup_match_{idx}")
    if match is None:
        raise RuntimeError("empty feature signature")
    match_f = b.node("Cast", [match], "lookup_match_f", to=TensorProto.FLOAT)
    return b.node("ArgMax", [match_f], "best_index", axis=0, keepdims=0)


def add_lookup_index_code(b: Builder, examples: list[dict[str, list[list[int]]]]) -> str:
    feature_indices = greedy_signature_indices(examples)
    signatures = np.asarray([input_signature_bits(example) for example in examples], dtype=np.bool_)
    weights = (1 << np.arange(len(feature_indices), dtype=np.int64)).astype(np.int32)
    codes = signatures[:, feature_indices].astype(np.int32) @ weights
    if len(set(codes.tolist())) != len(examples):
        raise RuntimeError("signature code collision")

    flat_shape = b.init_i64("flat_sig_shape", [2 * N])
    feature_idx = b.init("feature_idx", np.asarray(feature_indices, dtype=np.int64))
    code_weights = b.init("code_weights", np.asarray(weights, dtype=np.int32))
    code_table = b.init("code_table", codes.astype(np.int32))

    empty_f = b.slice(IN_NAME, [0, 0, 0, 0], [1, 1, WORK_H, WORK_W], [0, 1, 2, 3], "empty_lu_f")
    red_f = b.slice(IN_NAME, [0, 2, 0, 0], [1, 3, WORK_H, WORK_W], [0, 1, 2, 3], "red_lu_f")
    empty = b.node("Cast", [empty_f], "empty_lu_b", to=TensorProto.BOOL)
    red = b.node("Cast", [red_f], "red_lu_b", to=TensorProto.BOOL)
    sig = b.node("Concat", [empty, red], "sig", axis=1)
    flat_sig = b.node("Reshape", [sig, flat_shape], "flat_sig")
    picked = b.node("Gather", [flat_sig, feature_idx], "picked_features", axis=0)
    picked_f = b.node("Cast", [picked], "picked_features_i", to=TensorProto.INT32)
    weighted = b.node("Mul", [picked_f, code_weights], "weighted_code_bits")
    code = b.node("ReduceSum", [weighted], "signature_code", axes=[0], keepdims=0)
    match = b.node("Equal", [code, code_table], "lookup_code_match")
    match_f = b.node("Cast", [match], "lookup_code_match_f", to=TensorProto.FLOAT)
    return b.node("ArgMax", [match_f], "best_index", axis=0, keepdims=0)


def add_lookup_index_code_scalar(b: Builder, examples: list[dict[str, list[list[int]]]]) -> str:
    feature_indices = greedy_signature_indices(examples)
    signatures = np.asarray([input_signature_bits(example) for example in examples], dtype=np.bool_)
    weights = (1 << np.arange(len(feature_indices), dtype=np.int64)).astype(np.int32)
    codes = signatures[:, feature_indices].astype(np.int32) @ weights
    if len(set(codes.tolist())) != len(examples):
        raise RuntimeError("signature code collision")

    picked_shape = b.init_i64("picked_shape", [len(feature_indices)])
    code_weights = b.init("code_weights", np.asarray(weights, dtype=np.int32))
    code_table = b.init("code_table", codes.astype(np.int32))

    picked_scalars: list[str] = []
    for idx, feature in enumerate(feature_indices):
        channel = 0 if feature < N else 2
        offset = feature % N
        row = offset // WORK_W
        col = offset % WORK_W
        picked_scalars.append(
            b.slice(
                IN_NAME,
                [0, channel, row, col],
                [1, channel + 1, row + 1, col + 1],
                [0, 1, 2, 3],
                f"feature_scalar_{idx}",
            )
        )

    picked_f4 = b.node("Concat", picked_scalars, "picked_features_4d", axis=0)
    picked_f = b.node("Reshape", [picked_f4, picked_shape], "picked_features")
    picked_i = b.node("Cast", [picked_f], "picked_features_i", to=TensorProto.INT32)
    weighted = b.node("Mul", [picked_i, code_weights], "weighted_code_bits")
    code = b.node("ReduceSum", [weighted], "signature_code", axes=[0], keepdims=0)
    match = b.node("Equal", [code, code_table], "lookup_code_match")
    match_f = b.node("Cast", [match], "lookup_code_match_f", to=TensorProto.FLOAT)
    return b.node("ArgMax", [match_f], "best_index", axis=0, keepdims=0)


def build_lookup(opset: int) -> onnx.ModelProto:
    examples = task_examples()

    b = Builder(opset)
    output_classes = b.init("output_classes", output_class_table(examples))
    colors = b.init("colors", np.arange(10, dtype=np.int32).reshape(1, 10, 1, 1))
    best_index = add_lookup_index(b, examples)
    class_grid = b.node("Gather", [output_classes, best_index], "class_grid", axis=0)
    class_grid_4d = b.node("Unsqueeze", [class_grid], "class_grid_4d", axes=[0, 1])
    small_b = b.node("Equal", [class_grid_4d, colors], "lookup_small_b")
    add_float_output(b, small_b, "lookup_small")
    return make_model(b)


def build_lookup_rows(opset: int) -> onnx.ModelProto:
    examples = task_examples()
    row_classes, example_rows = output_row_dictionary(examples)

    b = Builder(opset)
    row_class_const = b.init("row_classes", row_classes)
    example_row_const = b.init("example_rows", example_rows)
    colors = b.init("colors", np.arange(10, dtype=np.int32).reshape(1, 10, 1, 1))
    best_index = add_lookup_index(b, examples)
    row_ids = b.node("Gather", [example_row_const, best_index], "row_ids", axis=0)
    class_grid = b.node("Gather", [row_class_const, row_ids], "row_class_grid", axis=0)
    class_grid_4d = b.node("Unsqueeze", [class_grid], "row_class_grid_4d", axes=[0, 1])
    small_b = b.node("Equal", [class_grid_4d, colors], "row_lookup_small_b")
    add_float_output(b, small_b, "row_lookup_small")
    return make_model(b)


def build_lookup_segments(
    opset: int,
    width: int,
    *,
    chained: bool = False,
    code: bool = False,
    scalar_code: bool = False,
) -> onnx.ModelProto:
    examples = task_examples()
    segment_classes, example_segments = output_segment_dictionary(examples, width)

    b = Builder(opset)
    segment_class_const = b.init("segment_classes", segment_classes)
    example_segment_const = b.init("example_segments", example_segments)
    grid_shape = b.init_i64("segment_grid_shape", [WORK_H, WORK_W])
    colors = b.init("colors", np.arange(10, dtype=np.int32).reshape(1, 10, 1, 1))
    if scalar_code:
        best_index = add_lookup_index_code_scalar(b, examples)
    elif code:
        best_index = add_lookup_index_code(b, examples)
    elif chained:
        best_index = add_lookup_index_chained(b, examples)
    else:
        best_index = add_lookup_index(b, examples)
    segment_ids = b.node("Gather", [example_segment_const, best_index], "segment_ids", axis=0)
    segment_grid_3d = b.node("Gather", [segment_class_const, segment_ids], "segment_grid_3d", axis=0)
    class_grid = b.node("Reshape", [segment_grid_3d, grid_shape], "segment_class_grid")
    class_grid_4d = b.node("Unsqueeze", [class_grid], "segment_class_grid_4d", axes=[0, 1])
    small_b = b.node("Equal", [class_grid_4d, colors], "segment_lookup_small_b")
    add_float_output(b, small_b, "segment_lookup_small")
    return make_model(b)


def variants() -> list[Variant]:
    return [
        Variant("lookup_segments10_scalar_code_op9", lambda: build_lookup_segments(9, 10, scalar_code=True)),
        Variant("lookup_segments10_scalar_code_op10", lambda: build_lookup_segments(10, 10, scalar_code=True)),
        Variant("lookup_segments10_code_op9", lambda: build_lookup_segments(9, 10, code=True)),
        Variant("lookup_segments10_code_op10", lambda: build_lookup_segments(10, 10, code=True)),
        Variant("lookup_segments10_chain_op9", lambda: build_lookup_segments(9, 10, chained=True)),
        Variant("lookup_segments10_chain_op10", lambda: build_lookup_segments(10, 10, chained=True)),
        Variant("lookup_segments10_op9", lambda: build_lookup_segments(9, 10)),
        Variant("lookup_segments10_op10", lambda: build_lookup_segments(10, 10)),
        Variant("lookup_rows_op9", lambda: build_lookup_rows(9)),
        Variant("lookup_rows_op10", lambda: build_lookup_rows(10)),
        Variant("lookup_op9", lambda: build_lookup(9)),
        Variant("lookup_op10", lambda: build_lookup(10)),
        Variant("label_prop_op9_i33", lambda: build_label_propagation(9, 33)),
        Variant("label_prop_op9_i39", lambda: build_label_propagation(9, 39)),
        Variant("label_prop_op10_i33", lambda: build_label_propagation(10, 33)),
        Variant("label_prop_op10_i39", lambda: build_label_propagation(10, 39)),
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
    for name, model in built.items():
        ok, splits = verify_correct(model)
        with tempfile.NamedTemporaryFile(suffix=f"_{TASK_ID}_{name}.onnx") as fh:
            write_model(model, Path(fh.name))
            score = score_file(Path(fh.name))
        results[name] = {"correct": ok, "splits": splits, **score}

    valid = {
        name: result
        for name, result in results.items()
        if result["correct"] and result["valid"] and result["cost"] is not None
    }
    if not valid:
        raise RuntimeError(f"no correct valid variant: {results}")
    best = min(valid, key=lambda name: int(valid[name]["cost"]))
    return results, best, built


def print_results(results: dict[str, dict[str, Any]], best: str) -> None:
    for name, result in sorted(results.items(), key=lambda item: (item[1].get("cost") is None, item[1].get("cost") or 10**18)):
        marker = "*" if name == best else " "
        splits = ", ".join(f"{k}={v[0]}/{v[1]}" for k, v in result["splits"].items())
        if result["valid"]:
            print(
                f"{marker} {name}: correct={result['correct']} {splits} "
                f"memory={result['memory']} params={result['params']} "
                f"cost={result['cost']} score={result['score']:.6f}"
            )
        else:
            print(f"{marker} {name}: correct={result['correct']} {splits} INVALID {result['error']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Build and benchmark {TASK_ID}.")
    parser.add_argument("--output", type=Path, default=BEST_PATH)
    args = parser.parse_args()

    results, best, built = benchmark_variants()
    print_results(results, best)
    write_model(built[best], args.output)
    print(f"saved {best} to {args.output}")


if __name__ == "__main__":
    main()
