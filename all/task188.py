"""Minimal ONNX for ARC task188: remove one duplicated half of a small grid.

Task rule: the input is two identical copies of the output placed either
side-by-side or stacked vertically. Infer the repeated direction from the grid
itself, keep the first copy, and discard the duplicate. Training examples show
horizontal repeats of widths 2/3/4 and a vertical repeat of height 3; generated
examples include both directions.

ONNX: test the six possible repeats (horizontal/vertical with copy size 2, 3,
or 4), build a compact row/column mask for the winning repeat, and multiply the
original one-hot input by that mask. This keeps output padding zero without
materializing a full 30x30 candidate for every hypothesis.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task188"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10
COPY_SIZES = (2, 3, 4)
MAX_COPY_H = 4
MAX_COPY_W = 4
MAX_INPUT_H = 8
MAX_INPUT_W = 8


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def load_data() -> dict:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def infer_train_factors() -> List[Tuple[int, int, int, int]]:
    factors: List[Tuple[int, int, int, int]] = []
    for ex in load_data()["train"]:
        inp = np.asarray(ex["input"], dtype=np.int64)
        out = np.asarray(ex["output"], dtype=np.int64)
        if inp.shape[0] % out.shape[0] or inp.shape[1] % out.shape[1]:
            raise ValueError(f"non-integral factor {inp.shape} -> {out.shape}")
        factors.append((inp.shape[0], inp.shape[1], inp.shape[0] // out.shape[0], inp.shape[1] // out.shape[1]))
    return factors


def solve_repeated_half(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    if w % 2 == 0 and np.array_equal(g[:, : w // 2], g[:, w // 2 :]):
        return g[:, : w // 2].copy()
    if h % 2 == 0 and np.array_equal(g[: h // 2, :], g[h // 2 :, :]):
        return g[: h // 2, :].copy()
    raise ValueError(f"no repeated half found for shape {g.shape}")


def _mode(values: np.ndarray) -> int:
    counts = Counter(int(v) for v in values.ravel())
    return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]


def _non_background_mode(values: np.ndarray) -> int:
    nz = [int(v) for v in values.ravel() if int(v) != 0]
    return _mode(np.asarray(nz if nz else [0], dtype=np.int64))


def _block_reduce(grid: np.ndarray, out_shape: Tuple[int, int], fn: Callable[[np.ndarray], int]) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    oh, ow = out_shape
    rh, cw = g.shape[0] // oh, g.shape[1] // ow
    out = np.zeros((oh, ow), dtype=np.int64)
    for r in range(oh):
        for c in range(ow):
            out[r, c] = fn(g[r * rh : (r + 1) * rh, c * cw : (c + 1) * cw])
    return out


def hypothesis_report() -> Dict[str, int]:
    failures = {
        "block_mode": 0,
        "block_majority": 0,
        "first_cell_in_block": 0,
        "last_cell_in_block": 0,
        "most_frequent_non_background": 0,
        "repeated_first_half": 0,
    }
    for ex in load_data()["train"]:
        inp = np.asarray(ex["input"], dtype=np.int64)
        exp = np.asarray(ex["output"], dtype=np.int64)
        out_shape = exp.shape
        candidates = {
            "block_mode": _block_reduce(inp, out_shape, _mode),
            "block_majority": _block_reduce(inp, out_shape, _mode),
            "first_cell_in_block": _block_reduce(inp, out_shape, lambda block: int(block.ravel()[0])),
            "last_cell_in_block": _block_reduce(inp, out_shape, lambda block: int(block.ravel()[-1])),
            "most_frequent_non_background": _block_reduce(inp, out_shape, _non_background_mode),
            "repeated_first_half": solve_repeated_half(inp),
        }
        for name, pred in candidates.items():
            failures[name] += int(not np.array_equal(pred, exp))
    return failures


def grid_to_onehot(grid: List[List[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def run_model(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def _make_active(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    orient: str,
    k: int,
) -> str:
    axes = _i64(inits, [2, 3], f"{orient}{k}_axes")
    if orient == "h":
        a_starts = _i64(inits, [0, 0], f"{orient}{k}_a_starts")
        a_ends = _i64(inits, [MAX_COPY_H, k], f"{orient}{k}_a_ends")
        b_starts = _i64(inits, [0, k], f"{orient}{k}_b_starts")
        b_ends = _i64(inits, [MAX_COPY_H, 2 * k], f"{orient}{k}_b_ends")
    else:
        a_starts = _i64(inits, [0, 0], f"{orient}{k}_a_starts")
        a_ends = _i64(inits, [k, MAX_COPY_W], f"{orient}{k}_a_ends")
        b_starts = _i64(inits, [k, 0], f"{orient}{k}_b_starts")
        b_ends = _i64(inits, [2 * k, MAX_COPY_W], f"{orient}{k}_b_ends")
    zero = _f32(inits, [0.0], f"{orient}{k}_zero")
    left = f"{orient}{k}_a"
    right = f"{orient}{k}_b"
    diff = f"{orient}{k}_diff"
    abs_diff = f"{orient}{k}_abs"
    total_diff = f"{orient}{k}_total_diff"
    has_diff = f"{orient}{k}_has_diff"
    active = f"{orient}{k}_active"
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, a_starts, a_ends, axes], [left]),
            helper.make_node("Slice", [IN_NAME, b_starts, b_ends, axes], [right]),
            helper.make_node("Sub", [left, right], [diff]),
            helper.make_node("Abs", [diff], [abs_diff]),
            helper.make_node("ReduceSum", [abs_diff], [total_diff], keepdims=1),
            helper.make_node("Greater", [total_diff, zero], [has_diff]),
            helper.make_node("Not", [has_diff], [active]),
        ]
    )
    return active


def _make_index_active(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    source: str,
    orient: str,
    k: int,
    axes: str,
    half: str,
) -> str:
    if orient == "h":
        a_starts = _i64(inits, [0, 0], f"{orient}{k}_idx_a_starts")
        a_ends = _i64(inits, [MAX_COPY_H, k], f"{orient}{k}_idx_a_ends")
        b_starts = _i64(inits, [0, k], f"{orient}{k}_idx_b_starts")
        b_ends = _i64(inits, [MAX_COPY_H, 2 * k], f"{orient}{k}_idx_b_ends")
    else:
        a_starts = _i64(inits, [0, 0], f"{orient}{k}_idx_a_starts")
        a_ends = _i64(inits, [k, MAX_COPY_W], f"{orient}{k}_idx_a_ends")
        b_starts = _i64(inits, [k, 0], f"{orient}{k}_idx_b_starts")
        b_ends = _i64(inits, [2 * k, MAX_COPY_W], f"{orient}{k}_idx_b_ends")
    first = f"{orient}{k}_idx_a"
    second = f"{orient}{k}_idx_b"
    same = f"{orient}{k}_idx_same"
    diff = f"{orient}{k}_idx_diff"
    diff_f = f"{orient}{k}_idx_difff"
    total = f"{orient}{k}_idx_total"
    active = f"{orient}{k}_idx_active"
    nodes.extend(
        [
            helper.make_node("Slice", [source, a_starts, a_ends, axes], [first]),
            helper.make_node("Slice", [source, b_starts, b_ends, axes], [second]),
            helper.make_node("Equal", [first, second], [same]),
            helper.make_node("Not", [same], [diff]),
            helper.make_node("Cast", [diff], [diff_f], to=TensorProto.FLOAT),
            helper.make_node("ReduceSum", [diff_f], [total], keepdims=1),
            helper.make_node("Less", [total, half], [active]),
        ]
    )
    return active


def _make_cell_present(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    row: int,
    col: int,
    axes: str,
    zero: str,
) -> str:
    starts = _i64(inits, [0, row, col], f"probe_{row}_{col}_starts")
    ends = _i64(inits, [C, row + 1, col + 1], f"probe_{row}_{col}_ends")
    cell = f"probe_{row}_{col}_cell"
    total = f"probe_{row}_{col}_total"
    present = f"probe_{row}_{col}_present"
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts, ends, axes], [cell]),
            helper.make_node("ReduceSum", [cell], [total], keepdims=1),
            helper.make_node("Greater", [total, zero], [present]),
        ]
    )
    return present


def _make_color_pair_equal(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    source: str,
    a: Tuple[int, int],
    b: Tuple[int, int],
    axes: str,
    name: str,
) -> str:
    a_starts = _i64(inits, [a[0], a[1]], f"{name}_a_starts")
    a_ends = _i64(inits, [a[0] + 1, a[1] + 1], f"{name}_a_ends")
    b_starts = _i64(inits, [b[0], b[1]], f"{name}_b_starts")
    b_ends = _i64(inits, [b[0] + 1, b[1] + 1], f"{name}_b_ends")
    a_cell = f"{name}_a"
    b_cell = f"{name}_b"
    equal = f"{name}_equal"
    nodes.extend(
        [
            helper.make_node("Slice", [source, a_starts, a_ends, axes], [a_cell]),
            helper.make_node("Slice", [source, b_starts, b_ends, axes], [b_cell]),
            helper.make_node("Equal", [a_cell, b_cell], [equal]),
        ]
    )
    return equal


def build_mask_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    mask_terms: Dict[str, List[str]] = {"h": [], "v": []}

    for orient in ("h", "v"):
        for k in COPY_SIZES:
            active = _make_active(nodes, inits, orient, k)
            active_f = f"{orient}{k}_activef"
            nodes.append(helper.make_node("Cast", [active], [active_f], to=TensorProto.FLOAT))
            if orient == "h":
                mask = np.zeros((1, 1, 1, W), dtype=np.float32)
                mask[:, :, :, :k] = 1.0
            else:
                mask = np.zeros((1, 1, H, 1), dtype=np.float32)
                mask[:, :, :k, :] = 1.0
            mask_name = _f32(inits, mask, f"{orient}{k}_mask")
            term = f"{orient}{k}_mask_term"
            nodes.append(helper.make_node("Mul", [active_f, mask_name], [term]))
            mask_terms[orient].append(term)

    compact_masks: List[str] = []
    for orient in ("h", "v"):
        terms = mask_terms[orient]
        acc = terms[0]
        for idx, term in enumerate(terms[1:], start=1):
            out = f"{orient}_mask" if idx == len(terms) - 1 else f"{orient}_mask_acc{idx}"
            nodes.append(helper.make_node("Add", [acc, term], [out]))
            acc = out
        compact_masks.append(acc)
    nodes.append(helper.make_node("Add", compact_masks, ["mask"]))
    nodes.append(helper.make_node("Mul", [IN_NAME, "mask"], [OUT_NAME]))

    graph = helper.make_graph(
        nodes,
        f"{TASK_ID}_mask",
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        initializer=inits,
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


def build_dimension_compact_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    axes23 = _i64(inits, [2, 3], "dim_axes23")
    axes123 = _i64(inits, [1, 2, 3], "dim_axes123")
    zero = _f32(inits, [0.0], "dim_zero")

    starts4 = _i64(inits, [0, 0], "dim_starts4")
    ends4 = _i64(inits, [MAX_COPY_H, MAX_COPY_W], "dim_ends4")
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts4, ends4, axes23], ["dim_input4"]),
            helper.make_node("ArgMax", ["dim_input4"], ["dim_colors4"], axis=1, keepdims=1),
        ]
    )

    col3 = _make_cell_present(nodes, inits, 0, 3, axes123, zero)
    col4 = _make_cell_present(nodes, inits, 0, 4, axes123, zero)
    col6 = _make_cell_present(nodes, inits, 0, 6, axes123, zero)
    row3 = _make_cell_present(nodes, inits, 3, 0, axes123, zero)
    row4 = _make_cell_present(nodes, inits, 4, 0, axes123, zero)
    row6 = _make_cell_present(nodes, inits, 6, 0, axes123, zero)

    nodes.extend(
        [
            helper.make_node("Not", [col4], ["dim_not_col4"]),
            helper.make_node("And", [col3, "dim_not_col4"], ["dim_width4"]),
            helper.make_node("Not", [col6], ["dim_not_col6"]),
            helper.make_node("And", [col4, "dim_not_col6"], ["dim_width6"]),
            helper.make_node("Not", [row4], ["dim_not_row4"]),
            helper.make_node("And", [row3, "dim_not_row4"], ["dim_height4"]),
            helper.make_node("Not", [row6], ["dim_not_row6"]),
            helper.make_node("And", [row4, "dim_not_row6"], ["dim_height6"]),
        ]
    )

    h2_pair0 = _make_color_pair_equal(nodes, inits, "dim_colors4", (0, 0), (0, 2), axes23, "dim_h2_pair0")
    h2_pair1 = _make_color_pair_equal(nodes, inits, "dim_colors4", (1, 0), (1, 2), axes23, "dim_h2_pair1")
    nodes.extend(
        [
            helper.make_node("And", [h2_pair0, h2_pair1], ["dim_h2_equal"]),
            helper.make_node("Not", ["dim_h2_equal"], ["dim_v2_equal"]),
            helper.make_node("Not", ["dim_height4"], ["dim_not_height4"]),
            helper.make_node("Or", ["dim_not_height4", "dim_h2_equal"], ["dim_h2_ok"]),
            helper.make_node("And", ["dim_not_row4", "dim_h2_ok"], ["dim_h2_height_ok"]),
            helper.make_node("And", ["dim_width4", "dim_h2_height_ok"], ["dim_h2_active"]),
            helper.make_node("Not", ["dim_width4"], ["dim_not_width4"]),
            helper.make_node("Or", ["dim_not_width4", "dim_v2_equal"], ["dim_v2_ok"]),
            helper.make_node("And", ["dim_not_col4", "dim_v2_ok"], ["dim_v2_width_ok"]),
            helper.make_node("And", ["dim_height4", "dim_v2_width_ok"], ["dim_v2_active"]),
        ]
    )

    active_by_key = {
        ("h", 2): "dim_h2_active",
        ("h", 3): "dim_width6",
        ("h", 4): col6,
        ("v", 2): "dim_v2_active",
        ("v", 3): "dim_height6",
        ("v", 4): row6,
    }

    row_terms: List[str] = []
    col_terms: List[str] = []
    for orient in ("h", "v"):
        for k in COPY_SIZES:
            active_f = f"{orient}{k}_dim_activef"
            nodes.append(helper.make_node("Cast", [active_by_key[(orient, k)]], [active_f], to=TensorProto.FLOAT))

            row_keep = np.zeros((1, 1, MAX_COPY_H, 1), dtype=np.float32)
            col_keep = np.zeros((1, 1, 1, MAX_COPY_W), dtype=np.float32)
            if orient == "h":
                row_keep[:, :, :, :] = 1.0
                col_keep[:, :, :, :k] = 1.0
            else:
                row_keep[:, :, :k, :] = 1.0
                col_keep[:, :, :, :] = 1.0
            row_keep_name = _f32(inits, row_keep, f"{orient}{k}_dim_row_keep")
            col_keep_name = _f32(inits, col_keep, f"{orient}{k}_dim_col_keep")
            row_term = f"{orient}{k}_dim_row_term"
            col_term = f"{orient}{k}_dim_col_term"
            nodes.extend(
                [
                    helper.make_node("Mul", [active_f, row_keep_name], [row_term]),
                    helper.make_node("Mul", [active_f, col_keep_name], [col_term]),
                ]
            )
            row_terms.append(row_term)
            col_terms.append(col_term)

    acc = row_terms[0]
    for idx, term in enumerate(row_terms[1:], start=1):
        out = "dim_row_mask" if idx == len(row_terms) - 1 else f"dim_row_mask_acc{idx}"
        nodes.append(helper.make_node("Add", [acc, term], [out]))
        acc = out

    acc = col_terms[0]
    for idx, term in enumerate(col_terms[1:], start=1):
        out = "dim_col_mask" if idx == len(col_terms) - 1 else f"dim_col_mask_acc{idx}"
        nodes.append(helper.make_node("Add", [acc, term], [out]))
        acc = out

    nodes.extend(
        [
            helper.make_node("Mul", ["dim_row_mask", "dim_col_mask"], ["dim_mask"]),
            helper.make_node("Mul", ["dim_input4", "dim_mask"], ["dim_output4"]),
            helper.make_node(
                "Pad",
                ["dim_output4"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - MAX_COPY_H, W - MAX_COPY_W],
            ),
        ]
    )

    graph = helper.make_graph(
        nodes,
        f"{TASK_ID}_dimension_compact",
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        initializer=inits,
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


def build_index_compact_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    axes = _i64(inits, [2, 3], "idx_axes")
    starts8 = _i64(inits, [0, 0], "idx_starts8")
    ends8 = _i64(inits, [MAX_INPUT_H, MAX_INPUT_W], "idx_ends8")
    half = _f32(inits, [0.5], "idx_half")
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts8, ends8, axes], ["input8"]),
            helper.make_node("ArgMax", ["input8"], ["colors8"], axis=1, keepdims=1),
        ]
    )

    terms: List[str] = []
    for orient in ("h", "v"):
        for k in COPY_SIZES:
            active = _make_index_active(nodes, inits, "colors8", orient, k, axes, half)
            active_f = f"{orient}{k}_idx_activef"
            nodes.append(helper.make_node("Cast", [active], [active_f], to=TensorProto.FLOAT))

            keep = np.zeros((1, 1, MAX_COPY_H, MAX_COPY_W), dtype=np.float32)
            if orient == "h":
                keep[:, :, :, :k] = 1.0
            else:
                keep[:, :, :k, :] = 1.0
            keep_name = _f32(inits, keep, f"{orient}{k}_idx_keep")
            term = f"{orient}{k}_idx_term"
            nodes.append(helper.make_node("Mul", [active_f, keep_name], [term]))
            terms.append(term)

    acc = terms[0]
    for idx, term in enumerate(terms[1:], start=1):
        out = "idx_mask" if idx == len(terms) - 1 else f"idx_mask_acc{idx}"
        nodes.append(helper.make_node("Add", [acc, term], [out]))
        acc = out

    starts4 = _i64(inits, [0, 0], "idx_starts4")
    ends4 = _i64(inits, [MAX_COPY_H, MAX_COPY_W], "idx_ends4")
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts4, ends4, axes], ["idx_input4"]),
            helper.make_node("Mul", ["idx_input4", "idx_mask"], ["idx_output4"]),
            helper.make_node(
                "Pad",
                ["idx_output4"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - MAX_COPY_H, W - MAX_COPY_W],
            ),
        ]
    )

    graph = helper.make_graph(
        nodes,
        f"{TASK_ID}_index_compact",
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        initializer=inits,
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


def build_compact_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    terms: List[str] = []

    for orient in ("h", "v"):
        for k in COPY_SIZES:
            active = _make_active(nodes, inits, orient, k)
            active_f = f"{orient}{k}_compact_activef"
            nodes.append(helper.make_node("Cast", [active], [active_f], to=TensorProto.FLOAT))

            keep = np.zeros((1, 1, MAX_COPY_H, MAX_COPY_W), dtype=np.float32)
            if orient == "h":
                keep[:, :, :, :k] = 1.0
            else:
                keep[:, :, :k, :] = 1.0
            keep_name = _f32(inits, keep, f"{orient}{k}_compact_keep")
            term = f"{orient}{k}_compact_term"
            nodes.append(helper.make_node("Mul", [active_f, keep_name], [term]))
            terms.append(term)

    acc = terms[0]
    for idx, term in enumerate(terms[1:], start=1):
        out = "compact_mask" if idx == len(terms) - 1 else f"compact_mask_acc{idx}"
        nodes.append(helper.make_node("Add", [acc, term], [out]))
        acc = out

    axes = _i64(inits, [2, 3], "compact_axes")
    starts = _i64(inits, [0, 0], "compact_starts")
    ends = _i64(inits, [MAX_COPY_H, MAX_COPY_W], "compact_ends")
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts, ends, axes], ["compact_input"]),
            helper.make_node("Mul", ["compact_input", "compact_mask"], ["compact_output"]),
            helper.make_node(
                "Pad",
                ["compact_output"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - MAX_COPY_H, W - MAX_COPY_W],
            ),
        ]
    )

    graph = helper.make_graph(
        nodes,
        f"{TASK_ID}_compact",
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        initializer=inits,
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


def build_candidate_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    terms: List[str] = []

    for orient in ("h", "v"):
        for k in COPY_SIZES:
            active = _make_active(nodes, inits, orient, k)
            active_f = f"{orient}{k}_candidate_activef"
            nodes.append(helper.make_node("Cast", [active], [active_f], to=TensorProto.FLOAT))
            axis = _i64(inits, [3 if orient == "h" else 2], f"{orient}{k}_candidate_axis")
            s0 = _i64(inits, [0], f"{orient}{k}_candidate_s0")
            ek = _i64(inits, [k], f"{orient}{k}_candidate_ek")
            compact = f"{orient}{k}_candidate_compact"
            padded = f"{orient}{k}_candidate_padded"
            term = f"{orient}{k}_candidate_term"
            pads = [0, 0, 0, 0, 0, 0, 0 if orient == "h" else H - k, W - k if orient == "h" else 0]
            nodes.extend(
                [
                    helper.make_node("Slice", [IN_NAME, s0, ek, axis], [compact]),
                    helper.make_node("Pad", [compact], [padded], pads=pads),
                    helper.make_node("Mul", [padded, active_f], [term]),
                ]
            )
            terms.append(term)

    acc = terms[0]
    for idx, term in enumerate(terms[1:], start=1):
        out = OUT_NAME if idx == len(terms) - 1 else f"candidate_acc{idx}"
        nodes.append(helper.make_node("Add", [acc, term], [out]))
        acc = out

    graph = helper.make_graph(
        nodes,
        f"{TASK_ID}_candidate",
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        initializer=inits,
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


def validate_model(model: onnx.ModelProto) -> int:
    bad = 0
    data = load_data()
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            inp = np.asarray(ex["input"], dtype=np.int64)
            exp = np.asarray(ex["output"], dtype=np.int64)
            ref = solve_repeated_half(inp)
            if not np.array_equal(ref, exp):
                print(f"reference mismatch {split} {idx}")
                bad += 1
                continue
            raw = run_model(model, grid_to_onehot(ex["input"]))
            pred = onehot_to_grid(raw)
            cropped = pred[: exp.shape[0], : exp.shape[1]]
            outside = raw[0, :, exp.shape[0] :, :].sum()
            outside += raw[0, :, :, exp.shape[1] :].sum()
            if not np.array_equal(cropped, exp) or outside != 0.0:
                print(f"onnx mismatch {split} {idx}: {inp.shape} -> {exp.shape}")
                bad += 1
    return bad


def _score_variant(name: str, model: onnx.ModelProto) -> dict:
    path = OUT_DIR / f"{TASK_ID}_{name}.onnx"
    onnx.save(model, path)
    bad = validate_model(model)
    result = score_file(path)
    result["bad"] = bad
    result["nodes"] = len(model.graph.node)
    result["tensors"] = len(model.graph.node)
    return result


def main() -> None:
    factors = infer_train_factors()
    print("train input factors (h,w,rh,cw):", factors)
    print("hypothesis train failures:", hypothesis_report())

    variants = {
        "candidate": build_candidate_model(),
        "mask": build_mask_model(),
        "compact": build_compact_model(),
        "index_compact": build_index_compact_model(),
        "dimension_compact": build_dimension_compact_model(),
    }
    scored = {name: _score_variant(name, model) for name, model in variants.items()}
    valid = {name: res for name, res in scored.items() if res["valid"] and res["bad"] == 0}
    if not valid:
        raise SystemExit(f"no valid variants: {scored}")

    best_name = min(valid, key=lambda name: int(valid[name]["cost"]))
    onnx.save(variants[best_name], BEST_PATH)

    for name, result in scored.items():
        print(
            f"{name}: bad={result['bad']} valid={result['valid']} nodes={result['nodes']} "
            f"memory={result['memory']} params={result['params']} cost={result['cost']} "
            f"score={result['score']}"
        )
        if result["error"]:
            print(f"{name} error: {result['error']}")
    for name in variants:
        (OUT_DIR / f"{TASK_ID}_{name}.onnx").unlink(missing_ok=True)
    print(f"selected: {best_name} -> {BEST_PATH}")


if __name__ == "__main__":
    main()
