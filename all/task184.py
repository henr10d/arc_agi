"""ONNX for ARC task184: summarize noisy colored block grids.

Task rule: the input contains a 2x2, 2x3, 3x2, or 3x3 arrangement of
rectangular noisy non-black blocks separated by fully black horizontal gaps and
vertical gaps. Each block contains a single non-black color with black noise.
Replace each original block by one output pixel of that non-black color,
preserving the block row/column order. The compact summary is written in the
top-left of the 30x30 one-hot output; unused cells are left as all-zero padding.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task184"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
NC = 9
H = W = 30
MAX_GROUPS = 3
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


class Builder:
    def __init__(self) -> None:
        self.nodes: List[onnx.NodeProto] = []
        self.inits: List[onnx.TensorProto] = []
        self.n = 0

    def name(self, prefix: str) -> str:
        self.n += 1
        return f"{prefix}{self.n}"

    def init(self, arr: Any, name: str | None = None) -> str:
        out = name or self.name("c")
        self.inits.append(numpy_helper.from_array(np.asarray(arr), name=out))
        return out

    def i64(self, vals: Any, name: str | None = None) -> str:
        return self.init(np.asarray(vals, dtype=np.int64), name)

    def f32(self, vals: Any, name: str | None = None) -> str:
        return self.init(np.asarray(vals, dtype=np.float32), name)

    def add(self, op: str, ins: list[str], outs: list[str], **kwargs: Any) -> str:
        self.nodes.append(helper.make_node(op, ins, outs, **kwargs))
        return outs[0]


def _shift_right_bool(b: Builder, x: str, false_like: str) -> str:
    """Previous-position mask for a [1,30] bool vector."""
    tail = b.name("tail")
    out = b.name("prev")
    b.add("Slice", [x, "idx0", "idx29", "axis1"], [tail])
    b.add("Concat", [false_like, tail], [out], axis=1)
    return out


def _prefix_count(b: Builder, starts_bool: str, mat_name: str) -> str:
    starts_f = b.name("starts_f")
    prefix = b.name("prefix")
    b.add("Cast", [starts_bool], [starts_f], to=TensorProto.FLOAT)
    b.add("MatMul", [starts_f, mat_name], [prefix])
    return prefix


def _band_mask(b: Builder, occ_bool: str, prefix: str, group_index: int, prefix_name: str) -> str:
    gt = b.name(f"{prefix_name}_gt")
    lt = b.name(f"{prefix_name}_lt")
    eq = b.name(f"{prefix_name}_eq")
    both = b.name(f"{prefix_name}_b")
    cast = b.name(f"{prefix_name}_f")
    reshaped = b.name(f"{prefix_name}_m")
    b.add("Greater", [prefix, f"k{group_index + 1}_lo"], [gt])
    b.add("Less", [prefix, f"k{group_index + 1}_hi"], [lt])
    b.add("And", [gt, lt], [eq])
    b.add("And", [eq, occ_bool], [both])
    b.add("Cast", [both], [cast], to=TensorProto.FLOAT)
    b.add("Reshape", [cast, "row_mask_shape"], [reshaped])
    return reshaped


def _col_vec(b: Builder, col_occ: str, col_prefix: str, col_index: int, name: str) -> str:
    cgt = b.name(f"{name}_cgt")
    clt = b.name(f"{name}_clt")
    ceq = b.name(f"{name}_ceq")
    cb = b.name(f"{name}_cb")
    cf = b.name(f"{name}_cf")
    vec = b.name(f"{name}_vec")
    b.add("Greater", [col_prefix, f"k{col_index + 1}_lo"], [cgt])
    b.add("Less", [col_prefix, f"k{col_index + 1}_hi"], [clt])
    b.add("And", [cgt, clt], [ceq])
    b.add("And", [ceq, col_occ], [cb])
    b.add("Cast", [cb], [cf], to=TensorProto.FLOAT)
    b.add("Reshape", [cf, "col_vec_shape"], [vec])
    return vec


def _three_group_vectors(b: Builder, occ_bool: str, starts_bool: str, prefix: str) -> list[str]:
    tail = b.name(f"{prefix}_tail")
    tail_f = b.name(f"{prefix}_tail_f")
    second0 = b.name(f"{prefix}_second0")
    second = b.name(f"{prefix}_second")
    second_f = b.name(f"{prefix}_second_f")
    second_lo = b.name(f"{prefix}_second_lo")
    lt2 = b.name(f"{prefix}_lt2")
    ge2 = b.name(f"{prefix}_ge2")

    b.add("Slice", [starts_bool, "idx1", "idx30", "axis1"], [tail])
    b.add("Cast", [tail], [tail_f], to=TensorProto.FLOAT)
    b.add("ArgMax", [tail_f], [second0], axis=1, keepdims=1)
    b.add("Add", [second0, "one_i64"], [second])
    b.add("Cast", [second], [second_f], to=TensorProto.FLOAT)
    b.add("Sub", [second_f, "half"], [second_lo])
    b.add("Less", ["coord", second_lo], [lt2])
    b.add("Greater", ["coord", second_lo], [ge2])

    after2 = b.name(f"{prefix}_after2")
    third_candidates = b.name(f"{prefix}_third_cand")
    third_candidates_f = b.name(f"{prefix}_third_cand_f")
    has_third_f = b.name(f"{prefix}_has3_f")
    has_third = b.name(f"{prefix}_has3")
    not_has_third = b.name(f"{prefix}_not_has3")
    third = b.name(f"{prefix}_third")
    third_f = b.name(f"{prefix}_third_f")
    third_lo = b.name(f"{prefix}_third_lo")
    lt3 = b.name(f"{prefix}_lt3")
    ge3 = b.name(f"{prefix}_ge3")
    mid_limit = b.name(f"{prefix}_mid_limit")

    b.add("Greater", ["coord", second_f], [after2])
    b.add("And", [starts_bool, after2], [third_candidates])
    b.add("Cast", [third_candidates], [third_candidates_f], to=TensorProto.FLOAT)
    b.add("ReduceMax", [third_candidates_f], [has_third_f], axes=[1], keepdims=1)
    b.add("Greater", [has_third_f, "half"], [has_third])
    b.add("Not", [has_third], [not_has_third])
    b.add("ArgMax", [third_candidates_f], [third], axis=1, keepdims=1)
    b.add("Cast", [third], [third_f], to=TensorProto.FLOAT)
    b.add("Sub", [third_f, "half"], [third_lo])
    b.add("Less", ["coord", third_lo], [lt3])
    b.add("Greater", ["coord", third_lo], [ge3])
    b.add("Or", [not_has_third, lt3], [mid_limit])

    g0 = b.name(f"{prefix}_g0")
    g1a = b.name(f"{prefix}_g1a")
    g1 = b.name(f"{prefix}_g1")
    g2a = b.name(f"{prefix}_g2a")
    g2 = b.name(f"{prefix}_g2")
    b.add("And", [occ_bool, lt2], [g0])
    b.add("And", [occ_bool, ge2], [g1a])
    b.add("And", [g1a, mid_limit], [g1])
    b.add("And", [occ_bool, ge3], [g2a])
    b.add("And", [g2a, has_third], [g2])
    return [g0, g1, g2]


def build_model() -> onnx.ModelProto:
    b = Builder()
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    b.f32([0.5], "half")
    b.f32(np.arange(H, dtype=np.float32).reshape(1, H), "coord")
    b.f32(np.zeros((1, 1, 1, MAX_GROUPS), dtype=np.float32), "zero_row")
    b.init(np.zeros((1, 1), dtype=np.bool_), "false_vec")
    b.i64([0], "idx0")
    b.i64([1], "idx1")
    b.i64([29], "idx29")
    b.i64([30], "idx30")
    b.i64([[1]], "one_i64")
    b.i64([1], "axis1")
    b.i64([1, 1, H, 1], "row_mask_shape")
    b.i64([H, 1], "row_vec_shape")
    b.i64([1, W], "col_vec_shape")
    b.i64([W, 1], "col_start_vec_shape")

    b.i64([0], "ch0_st")
    b.i64([1], "ch0_en")
    b.i64([1], "ch1_st")
    b.i64([C], "ch1_en")
    b.i64([1], "axis_ch")
    pads = [0, 0, 0, 0, 0, 0, H - MAX_GROUPS, W - MAX_GROUPS]

    b.add("Slice", [IN_NAME, "ch0_st", "ch0_en", "axis_ch"], ["black"])
    b.add("ReduceMax", [IN_NAME], ["valid"], axes=[1], keepdims=1)
    b.add("Sub", ["valid", "black"], ["nonblack"])
    b.add("ReduceMax", ["nonblack"], ["row_any_f"], axes=[1, 3], keepdims=0)
    b.add("Greater", ["row_any_f", "half"], ["row_occ"])
    row_prev = _shift_right_bool(b, "row_occ", "false_vec")
    b.add("Not", [row_prev], ["not_row_prev"])
    b.add("And", ["row_occ", "not_row_prev"], ["row_starts"])
    b.add("ReduceMax", ["nonblack"], ["col_any_f"], axes=[1, 2], keepdims=0)
    b.add("Greater", ["col_any_f", "half"], ["col_occ"])
    col_occ = "col_occ"
    col_prev = _shift_right_bool(b, col_occ, "false_vec")
    b.add("Not", [col_prev], ["not_col_prev"])
    b.add("And", ["col_occ", "not_col_prev"], ["col_starts"])

    row_groups = _three_group_vectors(b, "row_occ", "row_starts", "row")
    col_groups = _three_group_vectors(b, "col_starts", "col_starts", "col")
    col_start_vecs = []
    for c, vec_bool in enumerate(col_groups):
        cf = b.name(f"col{c}_f")
        vec = b.name(f"col{c}_vec")
        mat_vec = b.name(f"col{c}_start_col")
        b.add("Cast", [vec_bool], [cf], to=TensorProto.FLOAT)
        b.add("Reshape", [cf, "col_vec_shape"], [vec])
        b.add("Reshape", [vec, "col_start_vec_shape"], [mat_vec])
        col_start_vecs.append(mat_vec)

    col_start_mat = b.name("col_start_mat")
    first_cols = b.name("first_cols")
    first_cols_t = b.name("first_cols_t")
    b.add("Concat", col_start_vecs, [col_start_mat], axis=1)
    b.add("MatMul", [IN_NAME, col_start_mat], [first_cols])
    b.add("Transpose", [first_cols], [first_cols_t], perm=[0, 1, 3, 2])

    rows = []
    for r, row_bool in enumerate(row_groups):
        row_f = b.name(f"row{r}_f")
        row_mask = b.name(f"row{r}_m")
        row_vec = b.name(f"row{r}_vec")
        b.add("Cast", [row_bool], [row_f], to=TensorProto.FLOAT)
        b.add("Reshape", [row_f, "row_mask_shape"], [row_mask])
        b.add("Reshape", [row_mask, "row_vec_shape"], [row_vec])
        counts = b.name(f"row{r}_counts")
        counts_t = b.name(f"row{r}_counts_t")
        counts_fg = b.name(f"row{r}_counts_fg")
        present = b.name(f"row{r}_present")
        present_f = b.name(f"row{r}_present_f")
        row = b.name(f"row{r}_out")
        b.add("MatMul", [first_cols_t, row_vec], [counts])
        b.add("Transpose", [counts], [counts_t], perm=[0, 1, 3, 2])
        b.add("Slice", [counts_t, "ch1_st", "ch1_en", "axis_ch"], [counts_fg])
        b.add("Greater", [counts_fg, "half"], [present])
        b.add("Cast", [present], [present_f], to=TensorProto.FLOAT)
        b.add("Concat", ["zero_row", present_f], [row], axis=1)
        rows.append(row)

    compact = b.name("compact")
    b.add("Concat", rows, [compact], axis=2)
    b.add("Pad", [compact], [OUT_NAME], pads=pads)

    graph = helper.make_graph(b.nodes, TASK_ID, [x_info], [y_info], initializer=b.inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def solve_reference(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    nonblack = g != 0
    row_occ = nonblack.any(axis=1)
    row_starts = np.flatnonzero(row_occ & ~np.r_[False, row_occ[:-1]])
    row_ends = np.r_[row_starts[1:], len(row_occ)]
    col_occ = nonblack.any(axis=0)
    col_starts = np.flatnonzero(col_occ & ~np.r_[False, col_occ[:-1]])
    col_ends = np.r_[col_starts[1:], len(col_occ)]
    out: list[list[int]] = []
    for rs, re_limit in zip(row_starts, row_ends):
        while re_limit > rs and not row_occ[re_limit - 1]:
            re_limit -= 1
        out_row: list[int] = []
        for cs, ce_limit in zip(col_starts, col_ends):
            while ce_limit > cs and not col_occ[ce_limit - 1]:
                ce_limit -= 1
            vals = g[rs:re_limit, cs:ce_limit]
            colors, counts = np.unique(vals[vals != 0], return_counts=True)
            out_row.append(int(colors[np.argmax(counts)]))
        out.append(out_row)
    return np.asarray(out, dtype=np.int64)


def grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def onehot_to_grid(arr: np.ndarray) -> np.ndarray:
    active = arr.reshape(C, H, W) > 0.0
    return active.argmax(axis=0).astype(np.int64)


def run_model(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_model(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            inp = np.asarray(ex["input"], dtype=np.int64)
            exp = np.asarray(ex["output"], dtype=np.int64)
            if max(inp.shape) > H:
                continue
            ref = solve_reference(inp)
            if not np.array_equal(ref, exp):
                print(f"reference mismatch {split} {idx}: got {ref.tolist()} exp {exp.tolist()}")
                bad += 1
                continue
            pred = onehot_to_grid(run_model(model, grid_to_onehot(ex["input"])))[: exp.shape[0], : exp.shape[1]]
            if not np.array_equal(pred, exp):
                print(f"onnx mismatch {split} {idx}: got {pred.tolist()} exp {exp.tolist()}")
                bad += 1
    return bad


def hypothesis_report() -> dict[str, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    failures = {"one_pixel_per_block": 0, "enlarged_cells": 0, "with_separators": 0}
    for ex in data["train"]:
        exp = np.asarray(ex["output"], dtype=np.int64)
        compact = solve_reference(np.asarray(ex["input"], dtype=np.int64))
        failures["one_pixel_per_block"] += int(not np.array_equal(compact, exp))
        failures["enlarged_cells"] += int(compact.shape == exp.shape)
        failures["with_separators"] += int(compact.shape == exp.shape)
    return failures


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    bad = validate_model(model)
    assert bad == 0, f"{bad} validation failures"
    result = score_file(BEST_PATH)
    print("hypotheses train failures:", hypothesis_report())
    print(
        f"{BEST_PATH.name}: valid={result['valid']} memory={result['memory']} "
        f"params={result['params']} cost={result['cost']} score={result['score']}"
    )


if __name__ == "__main__":
    main()
