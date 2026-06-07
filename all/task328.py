"""ONNX generator for ARC task328 corner-comb Voronoi fills.

Task rule: each non-black input cell is a colored seed at one of the square
grid's corners.  For every cell, first assign it to a seed only if that seed is
the unique closest present corner by Manhattan distance; distance ties remain
black.  In the local coordinates of the assigned corner, color cells satisfying
``(u is even and v <= u) or (v is even and v >= u)``.  Background cells inside
the original grid remain black, and padding outside the grid stays all-zero.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "task328"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
WORK = 18
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def solve_grid(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference numpy implementation used to validate the ONNX graph."""
    arr = np.asarray(grid, dtype=np.int64)
    h, w = arr.shape
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    seeds: list[tuple[int, int, int]] = [
        (int(r), int(c), int(arr[r, c])) for r, c in zip(*np.nonzero(arr))
    ]
    out = np.zeros_like(arr)
    if not seeds:
        return out

    dists = np.stack([np.abs(rows - r) + np.abs(cols - c) for r, c, _ in seeds])
    nearest = dists.min(axis=0)
    unique = (dists == nearest).sum(axis=0) == 1

    for idx, (r, c, color) in enumerate(seeds):
        u = rows if r == 0 else (h - 1 - rows)
        v = cols if c == 0 else (w - 1 - cols)
        comb = (((u % 2) == 0) & (v <= u)) | (((v % 2) == 0) & (v >= u))
        out[(dists[idx] == nearest) & unique & comb] = color
    return out


def _init_f32(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name))
    return name


def _init_i64(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.int64), name=name))
    return name


def _node(nodes: list[onnx.NodeProto], op: str, inputs: list[str], output: str, **attrs: Any) -> str:
    nodes.append(helper.make_node(op, inputs, [output], **attrs))
    return output


def _and(nodes: list[onnx.NodeProto], left: str, right: str, out: str) -> str:
    return _node(nodes, "And", [left, right], out)


def _or(nodes: list[onnx.NodeProto], left: str, right: str, out: str) -> str:
    return _node(nodes, "Or", [left, right], out)


def _not(nodes: list[onnx.NodeProto], value: str, out: str) -> str:
    return _node(nodes, "Not", [value], out)


def _strict_region(
    nodes: list[onnx.NodeProto],
    name: str,
    active: str,
    inside: str,
    own_dist: str,
    other_dists: list[str],
) -> str:
    region = _and(nodes, active, inside, f"{name}_act_inside")
    for idx, other in enumerate(other_dists):
        lt = _node(nodes, "Less", [own_dist, other], f"{name}_lt_{idx}")
        region = _and(nodes, region, lt, f"{name}_reg_{idx}")
    return region


def _comb(
    nodes: list[onnx.NodeProto],
    name: str,
    u: str,
    v: str,
    zero_i: str,
    two_i: str,
) -> str:
    u_mod = _node(nodes, "Mod", [u, two_i], f"{name}_umod", fmod=0)
    v_mod = _node(nodes, "Mod", [v, two_i], f"{name}_vmod", fmod=0)
    u_even = _node(nodes, "Equal", [u_mod, zero_i], f"{name}_ueven")
    v_even = _node(nodes, "Equal", [v_mod, zero_i], f"{name}_veven")
    v_lt_u = _node(nodes, "Less", [v, u], f"{name}_vltu")
    u_lt_v = _node(nodes, "Less", [u, v], f"{name}_ultv")
    u_eq_v = _node(nodes, "Equal", [u, v], f"{name}_ueqv")
    v_le_u = _or(nodes, v_lt_u, u_eq_v, f"{name}_vleu")
    v_ge_u = _or(nodes, u_lt_v, u_eq_v, f"{name}_vgeu")
    even_backbone = _and(nodes, u_even, v_le_u, f"{name}_back")
    even_teeth = _and(nodes, v_even, v_ge_u, f"{name}_teeth")
    return _or(nodes, even_backbone, even_teeth, f"{name}_comb")


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    full_rows = np.arange(H, dtype=np.float32).reshape(1, 1, H, 1)
    rows = np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1)
    cols = np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK)
    row_plus_col = rows + cols
    rows_i = np.arange(WORK, dtype=np.int64).reshape(1, 1, WORK, 1)
    cols_i = np.arange(WORK, dtype=np.int64).reshape(1, 1, 1, WORK)
    row_even_const = (rows_i % 2) == 0
    col_even_const = (cols_i % 2) == 0
    channel_ids = np.arange(C, dtype=np.int32).reshape(1, C, 1, 1)

    _init_f32(inits, "full_row_f", full_rows)
    _init_f32(inits, "row_f", rows)
    _init_f32(inits, "col_f", cols)
    _init_f32(inits, "row_plus_col_f", row_plus_col)
    _init_f32(inits, "one_f", np.array([1.0], dtype=np.float32))
    _init_i64(inits, "row_i", rows_i)
    _init_i64(inits, "col_i", cols_i)
    inits.append(numpy_helper.from_array(row_even_const, name="row_even"))
    inits.append(numpy_helper.from_array(col_even_const, name="col_even"))
    _init_i64(inits, "zero_i", np.array([0], dtype=np.int64))
    _init_i64(inits, "two_i", np.array([2], dtype=np.int64))
    _init_i64(inits, "shape1", np.array([1], dtype=np.int64))
    _init_i64(inits, "axes4", np.array([0, 1, 2, 3], dtype=np.int64))
    _init_i64(inits, "z1", np.array([0], dtype=np.int64))
    _init_i64(inits, "o1", np.array([1], dtype=np.int64))
    _init_i64(inits, "c10", np.array([C], dtype=np.int64))
    _init_i64(inits, "tl_s", np.array([0, 0, 0, 0], dtype=np.int64))
    _init_i64(inits, "tl_e", np.array([1, C, 1, 1], dtype=np.int64))
    inits.append(numpy_helper.from_array(np.asarray([0], dtype=np.int32), name="zero_i32"))
    inits.append(numpy_helper.from_array(np.asarray([C], dtype=np.int32).reshape(1, 1, 1, 1), name="ten_i32"))
    inits.append(numpy_helper.from_array(channel_ids, name="channel_ids"))

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    # Only the size finder touches the full 30x30 input.  All rule masks below
    # are evaluated on the known 18x18 maximum task extent, then padded once.
    row_any = _node(nodes, "ReduceMax", [IN_NAME], "row_any", axes=[1, 3], keepdims=1)
    weighted_extent = _node(nodes, "Mul", [row_any, "full_row_f"], "weighted_extent")
    m_f = _node(nodes, "ReduceMax", [weighted_extent], "m_f", axes=[2, 3], keepdims=1)
    m_i = _node(nodes, "Cast", [m_f], "m_i", to=TensorProto.INT64)
    m1_f = _node(nodes, "Add", [m_f, "one_f"], "m1_f")

    row_inside = _node(nodes, "Less", ["row_f", m1_f], "row_inside")
    col_inside = _node(nodes, "Less", ["col_f", m1_f], "col_inside")
    inside = _and(nodes, row_inside, col_inside, "inside")

    m1 = _node(nodes, "Reshape", [m_i, "shape1"], "m1")
    m1_next = _node(nodes, "Add", [m1, "o1"], "m1_next")
    tr_s = _node(nodes, "Concat", ["z1", "z1", "z1", m1], "tr_s", axis=0)
    tr_e = _node(nodes, "Concat", ["o1", "c10", "o1", m1_next], "tr_e", axis=0)
    bl_s = _node(nodes, "Concat", ["z1", "z1", m1, "z1"], "bl_s", axis=0)
    bl_e = _node(nodes, "Concat", ["o1", "c10", m1_next, "o1"], "bl_e", axis=0)
    br_s = _node(nodes, "Concat", ["z1", "z1", m1, m1], "br_s", axis=0)
    br_e = _node(nodes, "Concat", ["o1", "c10", m1_next, m1_next], "br_e", axis=0)

    slice_defs = {
        "tl": ("tl_s", "tl_e"),
        "tr": (tr_s, tr_e),
        "bl": (bl_s, bl_e),
        "br": (br_s, br_e),
    }
    seed_vecs: dict[str, str] = {}
    active: dict[str, str] = {}
    seed_colors: dict[str, str] = {}
    seed_colors_i32: dict[str, str] = {}
    for name, (starts, ends) in slice_defs.items():
        seed_vecs[name] = _node(nodes, "Slice", [IN_NAME, starts, ends, "axes4"], f"{name}_vec")
        seed_colors[name] = _node(nodes, "ArgMax", [seed_vecs[name]], f"{name}_color", axis=1, keepdims=1)
        active[name] = _node(nodes, "Greater", [seed_colors[name], "zero_i"], f"{name}_active")
        seed_colors_i32[name] = _node(
            nodes, "Cast", [seed_colors[name]], f"{name}_color_i32", to=TensorProto.INT32
        )

    m_minus_row_f = _node(nodes, "Sub", [m_f, "row_f"], "m_minus_row_f")
    m_minus_col_f = _node(nodes, "Sub", [m_f, "col_f"], "m_minus_col_f")
    row_lt_mrow = _node(nodes, "Less", ["row_f", m_minus_row_f], "row_lt_mrow")
    mrow_lt_row = _node(nodes, "Less", [m_minus_row_f, "row_f"], "mrow_lt_row")
    col_lt_mcol = _node(nodes, "Less", ["col_f", m_minus_col_f], "col_lt_mcol")
    mcol_lt_col = _node(nodes, "Less", [m_minus_col_f, "col_f"], "mcol_lt_col")
    sum_lt_m = _node(nodes, "Less", ["row_plus_col_f", m_f], "sum_lt_m")
    m_lt_sum = _node(nodes, "Less", [m_f, "row_plus_col_f"], "m_lt_sum")
    row_lt_col = _node(nodes, "Less", ["row_f", "col_f"], "row_lt_col")
    col_lt_row = _node(nodes, "Less", ["col_f", "row_f"], "col_lt_row")
    row_lt_mcol = _node(nodes, "Less", ["row_f", m_minus_col_f], "row_lt_mcol")
    mcol_lt_row = _node(nodes, "Less", [m_minus_col_f, "row_f"], "mcol_lt_row")
    mrow_lt_col = _node(nodes, "Less", [m_minus_row_f, "col_f"], "mrow_lt_col")
    col_lt_mrow = _node(nodes, "Less", ["col_f", m_minus_row_f], "col_lt_mrow")

    not_active = {name: _not(nodes, active[name], f"{name}_inactive") for name in slice_defs}
    pair_less = {
        ("tl", "tr"): col_lt_mcol,
        ("tl", "bl"): row_lt_mrow,
        ("tl", "br"): sum_lt_m,
        ("tr", "tl"): mcol_lt_col,
        ("tr", "bl"): row_lt_col,
        ("tr", "br"): row_lt_mrow,
        ("bl", "tl"): mrow_lt_row,
        ("bl", "tr"): col_lt_row,
        ("bl", "br"): col_lt_mcol,
        ("br", "tl"): m_lt_sum,
        ("br", "tr"): mrow_lt_row,
        ("br", "bl"): mcol_lt_col,
    }

    m_minus_row_i = _node(nodes, "Sub", [m_i, "row_i"], "m_minus_row_i")
    m_minus_col_i = _node(nodes, "Sub", [m_i, "col_i"], "m_minus_col_i")
    mrow_mod = _node(nodes, "Mod", [m_minus_row_i, "two_i"], "mrow_mod", fmod=0)
    mcol_mod = _node(nodes, "Mod", [m_minus_col_i, "two_i"], "mcol_mod", fmod=0)
    mrow_even = _node(nodes, "Equal", [mrow_mod, "zero_i"], "mrow_even")
    mcol_even = _node(nodes, "Equal", [mcol_mod, "zero_i"], "mcol_even")
    local_even = {
        "tl": ("row_even", "col_even"),
        "tr": ("row_even", mcol_even),
        "bl": (mrow_even, "col_even"),
        "br": (mrow_even, mcol_even),
    }

    not_row_lt_col = _not(nodes, row_lt_col, "not_row_lt_col")
    not_col_lt_row = _not(nodes, col_lt_row, "not_col_lt_row")
    not_row_lt_mcol = _not(nodes, row_lt_mcol, "not_row_lt_mcol")
    not_mcol_lt_row = _not(nodes, mcol_lt_row, "not_mcol_lt_row")
    not_mrow_lt_col = _not(nodes, mrow_lt_col, "not_mrow_lt_col")
    not_col_lt_mrow = _not(nodes, col_lt_mrow, "not_col_lt_mrow")
    comb_orders = {
        "tl": (not_row_lt_col, not_col_lt_row),
        "tr": (not_row_lt_mcol, not_mcol_lt_row),
        "bl": (not_mrow_lt_col, not_col_lt_mrow),
        "br": (not_col_lt_row, not_row_lt_col),
    }

    masks: dict[str, str] = {}
    names = ["tl", "tr", "bl", "br"]
    for name in names:
        region = _and(nodes, active[name], inside, f"{name}_act_inside")
        for other in names:
            if other == name:
                continue
            ok = _or(nodes, not_active[other], pair_less[(name, other)], f"{name}_ok_vs_{other}")
            region = _and(nodes, region, ok, f"{name}_reg_vs_{other}")
        vleu, vgeu = comb_orders[name]
        u_even, v_even = local_even[name]
        back = _and(nodes, u_even, vleu, f"{name}_back")
        teeth = _and(nodes, v_even, vgeu, f"{name}_teeth")
        comb = _or(nodes, back, teeth, f"{name}_comb")
        masks[name] = _and(nodes, region, comb, f"{name}_mask")

    color_idx = _node(nodes, "Where", [masks["tl"], seed_colors_i32["tl"], "zero_i32"], "color_tl")
    color_idx = _node(nodes, "Where", [masks["tr"], seed_colors_i32["tr"], color_idx], "color_tr")
    color_idx = _node(nodes, "Where", [masks["bl"], seed_colors_i32["bl"], color_idx], "color_bl")
    color_idx = _node(nodes, "Where", [masks["br"], seed_colors_i32["br"], color_idx], "color_idx")
    valid_color = _node(nodes, "Where", [inside, color_idx, "ten_i32"], "valid_color")
    onehot = _node(nodes, "Equal", [valid_color, "channel_ids"], "onehot")
    onehot_f = _node(nodes, "Cast", [onehot], "onehot_f", to=TensorProto.FLOAT)
    _node(
        nodes,
        "Pad",
        [onehot_f],
        OUT_NAME,
        pads=[0, 0, 0, 0, 0, 0, H - WORK, W - WORK],
        value=0.0,
    )

    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits)

    def vi(name: str, elem_type: int, shape: list[int]) -> None:
        graph.value_info.append(helper.make_tensor_value_info(name, elem_type, shape))

    scalar4 = [1, 1, 1, 1]
    full1 = [1, 1, WORK, WORK]
    row1 = [1, 1, WORK, 1]
    col1 = [1, 1, 1, WORK]
    for corner in ("tr", "bl", "br"):
        vi(f"{corner}_vec", TensorProto.FLOAT, [1, C, 1, 1])
        vi(f"{corner}_color", TensorProto.INT64, scalar4)
        vi(f"{corner}_active", TensorProto.BOOL, scalar4)
        vi(f"{corner}_color_i32", TensorProto.INT32, scalar4)
        vi(f"{corner}_inactive", TensorProto.BOOL, scalar4)
    for name, shape in {
        "tl_ok_vs_tr": col1,
        "tl_ok_vs_bl": row1,
        "tl_ok_vs_br": full1,
        "tr_ok_vs_bl": full1,
        "tr_ok_vs_br": row1,
        "bl_ok_vs_tr": full1,
        "bl_ok_vs_br": col1,
        "br_ok_vs_tr": row1,
        "br_ok_vs_bl": col1,
    }.items():
        vi(name, TensorProto.BOOL, shape)
    for corner in ("tl", "tr", "bl", "br"):
        for suffix in ("act_inside", "reg_vs_tl", "reg_vs_tr", "reg_vs_bl", "reg_vs_br", "mask"):
            name = f"{corner}_{suffix}"
            if name in {output for node in nodes for output in node.output}:
                vi(name, TensorProto.BOOL, full1)
    for name in ("color_tl", "color_tr", "color_bl", "color_idx", "valid_color"):
        vi(name, TensorProto.INT32, full1)
    vi("onehot", TensorProto.BOOL, [1, C, WORK, WORK])
    vi("onehot_f", TensorProto.FLOAT, [1, C, WORK, WORK])

    model = helper.make_model(
        graph,
        producer_name="neurogolf-task328",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _validate_reference(data: dict[str, list[dict[str, Any]]]) -> None:
    for split, examples in data.items():
        for idx, example in enumerate(examples):
            expected = np.asarray(example["output"], dtype=np.int64)
            got = solve_grid(example["input"])
            if not np.array_equal(got, expected):
                raise AssertionError(f"reference mismatch on {split} {idx}")


def _validate_onnx(model: onnx.ModelProto, data: dict[str, list[dict[str, Any]]]) -> None:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    for split, examples in data.items():
        for idx, example in enumerate(examples):
            expected = _grid_to_onehot(example["output"])
            got = sess.run([OUT_NAME], {IN_NAME: _grid_to_onehot(example["input"])})[0]
            if not np.array_equal(got > 0.0, expected > 0.0):
                raise AssertionError(f"ONNX mismatch on {split} {idx}")


def main() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    _validate_reference(data)
    model = build_model()
    _validate_onnx(model, data)
    onnx.save(model, BEST_PATH)
    print(f"saved {BEST_PATH}")


if __name__ == "__main__":
    main()
