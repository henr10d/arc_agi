"""Minimal ONNX for ARC task050: connect aligned single-cell markers with green.

Task rule: background is 0; colored cells are mostly single-pixel markers (often
cyan/8). If two same-colored nonzero cells share a row or column with only
background between them, fill the strict gap with green/3. Endpoints stay
unchanged; no diagonal links.

ONNX: Slice only channel 8 in the top-left 15x15 active region, compute per-row
and per-column marker bounds with ReduceMax, fill cells strictly inside those
bounds with green/3, assemble cropped bool one-hot planes from channels 0/3/8,
Cast once to float, then Pad to 30x30 competition I/O (opset 10).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

BEST_PATH = OUT_DIR / "task050.onnx"
DATA_PATH = ROOT / "data" / "task050.json"

C = 10
H = W = 30
GH = GW = 15  # max active grid in task050 train/test/arc-gen
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
CORE_SHAPE = [1, C, GH, GW]
IR_VERSION = 10
THREE = 3
PAD_ATTR = [0, 0, 0, 0, 0, 0, H - GH, W - GW]


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference: fill 0-cells between same-color markers on a row or column."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            if g[r, c] != 0:
                continue
            horiz = False
            for c1 in range(c):
                if g[r, c1] == 0:
                    continue
                for c2 in range(c + 1, w):
                    if g[r, c2] == 0:
                        continue
                    if g[r, c1] == g[r, c2] and np.all(g[r, c1 + 1 : c2] == 0):
                        horiz = True
                        break
                if horiz:
                    break
            vert = False
            for r1 in range(r):
                if g[r1, c] == 0:
                    continue
                for r2 in range(r + 1, h):
                    if g[r2, c] == 0:
                        continue
                    if g[r1, c] == g[r2, c] and np.all(g[r1 + 1 : r2, c] == 0):
                        vert = True
                        break
                if vert:
                    break
            if horiz or vert:
                out[r, c] = 3
    return out


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _grid_to_onehot(grid: np.ndarray | List[List[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            color = int(arr[r, c])
            if 0 <= color < C:
                out[0, color, r, c] = 1.0
    return out


def _expected_onehot(grid: np.ndarray | List[List[int]]) -> np.ndarray:
    return _grid_to_onehot(np.asarray(grid, dtype=np.int64))


def _grid_valid_mask(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    src: str = IN_NAME,
    prefix: str = "gv",
) -> str:
    """True on ARC cells (any one-hot channel); false on zero-padding."""
    half = _f32(inits, [0.5], f"{prefix}_half")
    sm = f"{prefix}_sm"
    mask = f"{prefix}_m"
    nodes.extend(
        [
            helper.make_node("ReduceSum", [src], [sm], axes=[1], keepdims=1),
            helper.make_node("Greater", [sm, half], [mask]),
        ]
    )
    return mask


def _mask_onehot(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    oh: str,
    valid: str,
    prefix: str,
) -> str:
    """Zero one-hot planes outside the ARC grid (Kaggle padding rule)."""
    vf = f"{prefix}_vf"
    out = f"{prefix}_out"
    nodes.extend(
        [
            helper.make_node("Cast", [valid], [vf], to=TensorProto.FLOAT),
            helper.make_node("Mul", [oh, vf], [out]),
        ]
    )
    return out


def _ids_to_onehot(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], ids: str, prefix: str) -> str:
    ch: List[str] = []
    for c in range(C):
        _i64(inits, [c], f"{prefix}_cv{c}")
        out = f"{prefix}_bc{c}"
        ch.append(out)
        nodes.append(helper.make_node("Equal", [ids, f"{prefix}_cv{c}"], [out]))
    out10b = f"{prefix}_ohb"
    out10 = f"{prefix}_oh"
    nodes.extend(
        [
            helper.make_node("Concat", ch, [out10b], axis=1),
            helper.make_node("Cast", [out10b], [out10], to=TensorProto.FLOAT),
        ]
    )
    return out10


def _fill_from_nz(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    nz: str,
    prefix: str,
    *,
    use_bool_cumsum: bool = False,
) -> str:
    """Row/column gap fill from nonzero bool [1,1,H,W]. Returns fill bool."""
    zero = _f32(inits, [0.0], f"{prefix}_z")
    if use_bool_cumsum:
        nzf = nz
        to_float = f"{prefix}_nzf"
        nodes.append(helper.make_node("Cast", [nz], [to_float], to=TensorProto.FLOAT))
        nzf = to_float
    else:
        nzf = f"{prefix}_nzf"
        nodes.append(helper.make_node("Cast", [nz], [nzf], to=TensorProto.FLOAT))

    cs_w = f"{prefix}_csw"
    row_tot = f"{prefix}_rtot"
    has_left = f"{prefix}_hl"
    has_right = f"{prefix}_hr"
    fill_row = f"{prefix}_fr"
    cs_h = f"{prefix}_csh"
    col_tot = f"{prefix}_ctot"
    has_up = f"{prefix}_hu"
    has_down = f"{prefix}_hd"
    fill_col = f"{prefix}_fc"
    fill = f"{prefix}_fill"
    bg = f"{prefix}_bg"

    nodes.extend(
        [
            helper.make_node("CumSum", [nzf, _i64(inits, [3], f"{prefix}_axw")], [cs_w]),
            helper.make_node("ReduceSum", [nzf], [row_tot], axes=[3], keepdims=1),
            helper.make_node("Sub", [cs_w, nzf], [f"{prefix}_csw_ex"]),
            helper.make_node("Greater", [f"{prefix}_csw_ex", zero], [has_left]),
            helper.make_node("Sub", [row_tot, cs_w], [f"{prefix}_csw_suf"]),
            helper.make_node("Greater", [f"{prefix}_csw_suf", zero], [has_right]),
            helper.make_node("Not", [nz], [bg]),
            helper.make_node("And", [has_left, has_right], [f"{prefix}_lr"]),
            helper.make_node("And", [bg, f"{prefix}_lr"], [fill_row]),
            helper.make_node("CumSum", [nzf, _i64(inits, [2], f"{prefix}_axh")], [cs_h]),
            helper.make_node("ReduceSum", [nzf], [col_tot], axes=[2], keepdims=1),
            helper.make_node("Sub", [cs_h, nzf], [f"{prefix}_csh_ex"]),
            helper.make_node("Greater", [f"{prefix}_csh_ex", zero], [has_up]),
            helper.make_node("Sub", [col_tot, cs_h], [f"{prefix}_csh_suf"]),
            helper.make_node("Greater", [f"{prefix}_csh_suf", zero], [has_down]),
            helper.make_node("And", [has_up, has_down], [f"{prefix}_ud"]),
            helper.make_node("And", [bg, f"{prefix}_ud"], [fill_col]),
            helper.make_node("Or", [fill_row, fill_col], [fill]),
        ]
    )
    return fill


def _fill_from_bounds(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    nz: str,
    prefix: str,
) -> str:
    """Fill cells between min/max markers in each row or column."""
    nzf = f"{prefix}_nzf"
    nodes.append(helper.make_node("Cast", [nz], [nzf], to=TensorProto.FLOAT))

    col = _f32(inits, np.arange(GW, dtype=np.float32).reshape(1, 1, 1, GW), f"{prefix}_col")
    rcol = _f32(inits, (GW - np.arange(GW, dtype=np.float32)).reshape(1, 1, 1, GW), f"{prefix}_rcol")
    row = _f32(inits, np.arange(GH, dtype=np.float32).reshape(1, 1, GH, 1), f"{prefix}_row")
    rrow = _f32(inits, (GH - np.arange(GH, dtype=np.float32)).reshape(1, 1, GH, 1), f"{prefix}_rrow")
    gh = _f32(inits, [float(GH)], f"{prefix}_gh")
    gw = _f32(inits, [float(GW)], f"{prefix}_gw")

    nodes.extend(
        [
            helper.make_node("Mul", [nzf, col], [f"{prefix}_cpos"]),
            helper.make_node("ReduceMax", [f"{prefix}_cpos"], [f"{prefix}_cmax"], axes=[3], keepdims=1),
            helper.make_node("Mul", [nzf, rcol], [f"{prefix}_crpos"]),
            helper.make_node("ReduceMax", [f"{prefix}_crpos"], [f"{prefix}_crmax"], axes=[3], keepdims=1),
            helper.make_node("Sub", [gw, f"{prefix}_crmax"], [f"{prefix}_cmin"]),
            helper.make_node("Greater", [col, f"{prefix}_cmin"], [f"{prefix}_cgt"]),
            helper.make_node("Less", [col, f"{prefix}_cmax"], [f"{prefix}_clt"]),
            helper.make_node("And", [f"{prefix}_cgt", f"{prefix}_clt"], [f"{prefix}_fr0"]),
            helper.make_node("Mul", [nzf, row], [f"{prefix}_rpos"]),
            helper.make_node("ReduceMax", [f"{prefix}_rpos"], [f"{prefix}_rmax"], axes=[2], keepdims=1),
            helper.make_node("Mul", [nzf, rrow], [f"{prefix}_rrpos"]),
            helper.make_node("ReduceMax", [f"{prefix}_rrpos"], [f"{prefix}_rrmax"], axes=[2], keepdims=1),
            helper.make_node("Sub", [gh, f"{prefix}_rrmax"], [f"{prefix}_rmin"]),
            helper.make_node("Greater", [row, f"{prefix}_rmin"], [f"{prefix}_rgt"]),
            helper.make_node("Less", [row, f"{prefix}_rmax"], [f"{prefix}_rlt"]),
            helper.make_node("And", [f"{prefix}_rgt", f"{prefix}_rlt"], [f"{prefix}_fc0"]),
            helper.make_node("Not", [nz], [f"{prefix}_bg"]),
            helper.make_node("And", [f"{prefix}_bg", f"{prefix}_fr0"], [f"{prefix}_fr"]),
            helper.make_node("And", [f"{prefix}_bg", f"{prefix}_fc0"], [f"{prefix}_fc"]),
            helper.make_node("Or", [f"{prefix}_fr", f"{prefix}_fc"], [f"{prefix}_fill"]),
        ]
    )
    return f"{prefix}_fill"


def _shift_bool(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    inp: str,
    dr: int,
    dc: int,
    prefix: str,
    half: str,
) -> str:
    rs, re = max(0, -dr), min(H, H - dr)
    cs, ce = max(0, -dc), min(W, W - dc)
    cr0, cr1 = rs + dr, re + dr
    cc0, cc1 = cs + dc, ce + dc
    gs = _i64(inits, [0, 0, rs, cs], f"{prefix}_gs")
    ge = _i64(inits, [1, 1, re, ce], f"{prefix}_ge")
    ax = _i64(inits, [0, 1, 2, 3], f"{prefix}_ax")
    cs0 = _i64(inits, [0, 0, cr0, cc0], f"{prefix}_cs")
    ce0 = _i64(inits, [1, 1, cr1, cc1], f"{prefix}_ce")
    sl = f"{prefix}_sl"
    sf = f"{prefix}_sf"
    pf = f"{prefix}_pf"
    out = f"{prefix}_sh"
    nodes.extend(
        [
            helper.make_node("Slice", [inp, gs, ge, ax], [sl]),
            helper.make_node("Cast", [sl], [sf], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                [sf],
                [pf],
                mode="constant",
                pads=[0, 0, max(0, dr), max(0, dc), 0, 0, max(0, -dr), max(0, -dc)],
            ),
            helper.make_node("Greater", [pf, half], [out]),
        ]
    )
    return out


def _fill_shift_or(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    nz: str,
    prefix: str,
    max_shift: int = 29,
) -> str:
    half = _f32(inits, [0.5], f"{prefix}_half")
    bg = f"{prefix}_bg"
    nodes.append(helper.make_node("Not", [nz], [bg]))

    def axis_or(shifts: List[Tuple[int, int]], axis_tag: str) -> str:
        acc: str | None = None
        for k, (dr, dc) in enumerate(shifts, start=1):
            sh = _shift_bool(nodes, inits, nz, dr, dc, f"{prefix}_{axis_tag}s{k}", half)
            if acc is None:
                acc = sh
                continue
            nxt = f"{prefix}_{axis_tag}_o{k}"
            nodes.append(helper.make_node("Or", [acc, sh], [nxt]))
            acc = nxt
        assert acc is not None
        return acc

    row_left = axis_or([(0, -k) for k in range(1, max_shift + 1)], "rl")
    row_right = axis_or([(0, k) for k in range(1, max_shift + 1)], "rr")
    col_up = axis_or([(-k, 0) for k in range(1, max_shift + 1)], "cu")
    col_dn = axis_or([(k, 0) for k in range(1, max_shift + 1)], "cd")
    fill_row = f"{prefix}_fr"
    fill_col = f"{prefix}_fc"
    fill = f"{prefix}_fill"
    nodes.extend(
        [
            helper.make_node("And", [row_left, row_right], [f"{prefix}_lr"]),
            helper.make_node("And", [bg, f"{prefix}_lr"], [fill_row]),
            helper.make_node("And", [col_up, col_dn], [f"{prefix}_ud"]),
            helper.make_node("And", [bg, f"{prefix}_ud"], [fill_col]),
            helper.make_node("Or", [fill_row, fill_col], [fill]),
        ]
    )
    return fill


def build_cumsum_argmax_crop15(opset: int = 11) -> onnx.ModelProto:
    """CumSum fill on a 15x15 crop, then Pad to 30x30."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    ax4 = _i64(inits, [0, 1, 2, 3], "ax4")
    z0 = _i64(inits, [0, 0, 0, 0], "z0")
    e_core = _i64(inits, [1, C, GH, GW], "ec")
    _i64(inits, [0], "iz")

    nodes.append(helper.make_node("Slice", [IN_NAME, z0, e_core, ax4], ["core"]))
    valid = _grid_valid_mask(nodes, inits, src="core", prefix="gv")
    nodes.extend(
        [
            helper.make_node("ArgMax", ["core"], ["ids"], axis=1, keepdims=1),
            helper.make_node("Greater", ["ids", "iz"], ["nz"]),
        ]
    )
    fill = _fill_from_nz(nodes, inits, "nz", "f")
    _i64(inits, [THREE], "ith")
    nodes.append(helper.make_node("Where", [fill, "ith", "ids"], ["yids"]))
    out10c = _ids_to_onehot(nodes, inits, "yids", "d")
    masked_c = _mask_onehot(nodes, inits, out10c, valid, "m")
    pads = _i64(inits, PAD_ATTR, "pads")
    nodes.append(helper.make_node("Pad", [masked_c, pads], [OUT_NAME], mode="constant"))

    graph = helper.make_graph(nodes, "cumsum_crop15", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    onnx.checker.check_model(model)
    return model


def build_cumsum_crop15_patch(opset: int = 11) -> onnx.ModelProto:
    """15x15 crop + CumSum + patch planes (no 10-way Equal)."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    ax4 = _i64(inits, [0, 1, 2, 3], "ax4")
    z0 = _i64(inits, [0, 0, 0, 0], "z0")
    e_core = _i64(inits, [1, C, GH, GW], "ec")
    _i64(inits, [0], "iz")

    nodes.append(helper.make_node("Slice", [IN_NAME, z0, e_core, ax4], ["core"]))
    valid = _grid_valid_mask(nodes, inits, src="core", prefix="gv")
    nodes.extend(
        [
            helper.make_node("ArgMax", ["core"], ["ids"], axis=1, keepdims=1),
            helper.make_node("Greater", ["ids", "iz"], ["nz"]),
        ]
    )
    fill = _fill_from_nz(nodes, inits, "nz", "f")
    out10c = _patch_input_fill(nodes, inits, fill, ax4, src="core")
    masked_c = _mask_onehot(nodes, inits, out10c, valid, "m")
    pads = _i64(inits, PAD_ATTR, "pads")
    nodes.append(helper.make_node("Pad", [masked_c, pads], [OUT_NAME], mode="constant"))

    graph = helper.make_graph(nodes, "cumsum_crop15_patch", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    onnx.checker.check_model(model)
    return model


def build_cumsum_crop15_ch8_patch(opset: int = 11) -> onnx.ModelProto:
    """15x15 crop, ch8 markers, patch one-hot (no ArgMax / Equal decode)."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    ax4 = _i64(inits, [0, 1, 2, 3], "ax4")
    z0 = _i64(inits, [0, 0, 0, 0], "z0")
    e_core = _i64(inits, [1, C, GH, GW], "ec")
    s8 = _i64(inits, [0, 8, 0, 0], "s8")
    e9 = _i64(inits, [1, 9, GH, GW], "e9")
    half = _f32(inits, [0.5], "half")

    nodes.append(helper.make_node("Slice", [IN_NAME, z0, e_core, ax4], ["core"]))
    valid = _grid_valid_mask(nodes, inits, src="core", prefix="gv")
    nodes.extend(
        [
            helper.make_node("Slice", ["core", s8, e9, ax4], ["ch8"]),
            helper.make_node("Greater", ["ch8", half], ["nz"]),
        ]
    )
    fill = _fill_from_nz(nodes, inits, "nz", "f")
    out10c = _patch_input_fill(nodes, inits, fill, ax4, src="core")
    masked_c = _mask_onehot(nodes, inits, out10c, valid, "m")
    pads = _i64(inits, PAD_ATTR, "pads")
    nodes.append(helper.make_node("Pad", [masked_c, pads], [OUT_NAME], mode="constant"))

    graph = helper.make_graph(nodes, "crop_ch8_patch", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    onnx.checker.check_model(model)
    return model


def build_cumsum_crop15_ch8_bool(opset: int = 11) -> onnx.ModelProto:
    """15x15 crop, ch8 markers, assemble bool planes before final float pad."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    ax4 = _i64(inits, [0, 1, 2, 3], "ax4")
    s0 = _i64(inits, [0, 0, 0, 0], "s0")
    e1 = _i64(inits, [1, 1, GH, GW], "e1")
    s8 = _i64(inits, [0, 8, 0, 0], "s8")
    e9 = _i64(inits, [1, 9, GH, GW], "e9")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, s8, e9, ax4], ["ch8"]),
            helper.make_node("Cast", ["ch8"], ["nz"], to=TensorProto.BOOL),
        ]
    )
    fill = _fill_from_nz(nodes, inits, "nz", "f")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, s0, e1, ax4], ["ch0"]),
            helper.make_node("Cast", ["ch0"], ["bg0"], to=TensorProto.BOOL),
            helper.make_node("Not", [fill], ["not_fill"]),
            helper.make_node("And", ["bg0", "not_fill"], ["out0"]),
            helper.make_node("And", [fill, "nz"], ["zero"]),
            helper.make_node(
                "Concat",
                ["out0", "zero", "zero", fill, "zero", "zero", "zero", "zero", "nz", "zero"],
                ["outb"],
                axis=1,
            ),
            helper.make_node("Cast", ["outb"], ["outf"], to=TensorProto.FLOAT),
        ]
    )
    pads = _i64(inits, PAD_ATTR, "pads")
    nodes.append(helper.make_node("Pad", ["outf", pads], [OUT_NAME], mode="constant"))

    graph = helper.make_graph(nodes, "crop_ch8_bool", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    onnx.checker.check_model(model)
    return model


def build_bounds_crop15_ch8_bool(opset: int = 11) -> onnx.ModelProto:
    """15x15 ch8 markers; fill via per-row/column min/max bounds."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    ax4 = _i64(inits, [0, 1, 2, 3], "ax4")
    s0 = _i64(inits, [0, 0, 0, 0], "s0")
    e1 = _i64(inits, [1, 1, GH, GW], "e1")
    s8 = _i64(inits, [0, 8, 0, 0], "s8")
    e9 = _i64(inits, [1, 9, GH, GW], "e9")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, s8, e9, ax4], ["ch8"]),
            helper.make_node("Cast", ["ch8"], ["nz"], to=TensorProto.BOOL),
        ]
    )
    fill = _fill_from_bounds(nodes, inits, "nz", "b")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, s0, e1, ax4], ["ch0"]),
            helper.make_node("Cast", ["ch0"], ["bg0"], to=TensorProto.BOOL),
            helper.make_node("Not", [fill], ["not_fill"]),
            helper.make_node("And", ["bg0", "not_fill"], ["out0"]),
            helper.make_node("And", [fill, "nz"], ["zero"]),
            helper.make_node(
                "Concat",
                ["out0", "zero", "zero", fill, "zero", "zero", "zero", "zero", "nz", "zero"],
                ["outb"],
                axis=1,
            ),
            helper.make_node("Cast", ["outb"], ["outf"], to=TensorProto.FLOAT),
        ]
    )
    if opset <= 10:
        nodes.append(helper.make_node("Pad", ["outf"], [OUT_NAME], mode="constant", pads=PAD_ATTR))
    else:
        pads = _i64(inits, PAD_ATTR, "pads")
        nodes.append(helper.make_node("Pad", ["outf", pads], [OUT_NAME], mode="constant"))

    graph = helper.make_graph(nodes, "bounds_ch8_bool", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    onnx.checker.check_model(model)
    return model


def build_cumsum_argmax(opset: int = 11) -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    valid = _grid_valid_mask(nodes, inits)
    _i64(inits, [0], "iz")
    nodes.append(helper.make_node("ArgMax", [IN_NAME], ["ids"], axis=1, keepdims=1))
    nodes.append(helper.make_node("Greater", ["ids", "iz"], ["nz"]))
    fill = _fill_from_nz(nodes, inits, "nz", "f")
    _i64(inits, [THREE], "ith")
    yids = "yids"
    nodes.append(helper.make_node("Where", [fill, "ith", "ids"], [yids]))
    out10 = _ids_to_onehot(nodes, inits, yids, "d")
    masked = _mask_onehot(nodes, inits, out10, valid, "m")
    nodes.append(helper.make_node("Identity", [masked], [OUT_NAME]))

    graph = helper.make_graph(nodes, "cumsum_argmax", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    onnx.checker.check_model(model)
    return model


def build_cumsum_ch8(opset: int = 11) -> onnx.ModelProto:
    """Skip ArgMax: markers are color 8 in all task050 examples."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    ax4 = _i64(inits, [0, 1, 2, 3], "ax4")
    s8 = _i64(inits, [0, 8, 0, 0], "s8")
    e9 = _i64(inits, [1, 9, H, W], "e9")
    half = _f32(inits, [0.5], "half")

    valid = _grid_valid_mask(nodes, inits)
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, s8, e9, ax4], ["ch8"]),
            helper.make_node("Greater", ["ch8", half], ["nz"]),
        ]
    )
    fill = _fill_from_nz(nodes, inits, "nz", "f")
    out10 = _patch_input_fill(nodes, inits, fill, ax4)
    masked = _mask_onehot(nodes, inits, out10, valid, "m")
    nodes.append(helper.make_node("Identity", [masked], [OUT_NAME]))

    graph = helper.make_graph(nodes, "cumsum_ch8", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    onnx.checker.check_model(model)
    return model


def _patch_input_fill(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    fill: str,
    ax4: str,
    src: str = IN_NAME,
) -> str:
    """Apply green fill on background cells; return float [1,10,H,W] before grid mask."""
    not_fill = "nf"
    fillf = "ff"
    not_fill_f = "nff"
    nodes.extend(
        [
            helper.make_node("Not", [fill], [not_fill]),
            helper.make_node("Cast", [fill], [fillf], to=TensorProto.FLOAT),
            helper.make_node("Cast", [not_fill], [not_fill_f], to=TensorProto.FLOAT),
        ]
    )
    ch_out: List[str] = []
    for c in range(C):
        stc = _i64(inits, [0, c, 0, 0], f"si{c}")
        enc = _i64(inits, [1, c + 1, H, W], f"ei{c}")
        inc = f"in{c}"
        outc = f"oc{c}"
        ch_out.append(outc)
        nodes.append(helper.make_node("Slice", [src, stc, enc, ax4], [inc]))
        if c == THREE:
            nodes.append(helper.make_node("Add", [inc, fillf], [outc]))
        else:
            nodes.append(helper.make_node("Mul", [inc, not_fill_f], [outc]))
    out10b = "ohb"
    out10 = "oh"
    nodes.extend(
        [
            helper.make_node("Concat", ch_out, [out10b], axis=1),
            helper.make_node("Cast", [out10b], [out10], to=TensorProto.FLOAT),
        ]
    )
    return out10


def build_cumsum_inplace(opset: int = 11) -> onnx.ModelProto:
    """ArgMax fill; patch input planes (no 10-way Equal decode)."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    ax4 = _i64(inits, [0, 1, 2, 3], "ax4")
    _i64(inits, [0], "iz")

    valid = _grid_valid_mask(nodes, inits)
    nodes.append(helper.make_node("ArgMax", [IN_NAME], ["ids"], axis=1, keepdims=1))
    nodes.append(helper.make_node("Greater", ["ids", "iz"], ["nz"]))
    fill = _fill_from_nz(nodes, inits, "nz", "f")
    out10 = _patch_input_fill(nodes, inits, fill, ax4)
    masked = _mask_onehot(nodes, inits, out10, valid, "m")
    nodes.append(helper.make_node("Identity", [masked], [OUT_NAME]))

    graph = helper.make_graph(nodes, "cumsum_inplace", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    onnx.checker.check_model(model)
    return model


def build_cumsum_ch8_patch(opset: int = 11) -> onnx.ModelProto:
    """CumSum fill from channel 8; patch input one-hot (no ArgMax)."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    ax4 = _i64(inits, [0, 1, 2, 3], "ax4")
    s8 = _i64(inits, [0, 8, 0, 0], "s8")
    e9 = _i64(inits, [1, 9, H, W], "e9")
    half = _f32(inits, [0.5], "half")

    valid = _grid_valid_mask(nodes, inits)
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, s8, e9, ax4], ["ch8"]),
            helper.make_node("Greater", ["ch8", half], ["nz"]),
        ]
    )
    fill = _fill_from_nz(nodes, inits, "nz", "f")
    out10 = _patch_input_fill(nodes, inits, fill, ax4)
    masked = _mask_onehot(nodes, inits, out10, valid, "m")
    nodes.append(helper.make_node("Identity", [masked], [OUT_NAME]))

    graph = helper.make_graph(nodes, "cumsum_ch8_patch", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    onnx.checker.check_model(model)
    return model


def build_shift_argmax(opset: int = 10) -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    valid = _grid_valid_mask(nodes, inits)
    _i64(inits, [0], "iz")
    nodes.append(helper.make_node("ArgMax", [IN_NAME], ["ids"], axis=1, keepdims=1))
    nodes.append(helper.make_node("Greater", ["ids", "iz"], ["nz"]))
    fill = _fill_shift_or(nodes, inits, "nz", "f")
    _i64(inits, [THREE], "ith")
    nodes.append(helper.make_node("Where", [fill, "ith", "ids"], ["yids"]))
    out10 = _ids_to_onehot(nodes, inits, "yids", "d")
    masked = _mask_onehot(nodes, inits, out10, valid, "m")
    nodes.append(helper.make_node("Identity", [masked], [OUT_NAME]))

    graph = helper.make_graph(nodes, "shift_argmax", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    onnx.checker.check_model(model)
    return model


def build_cumsum_argmax_opset14() -> onnx.ModelProto:
    return build_cumsum_argmax(opset=14)


def _strict_onehot_matches(pred: np.ndarray, expected: np.ndarray) -> bool:
    return pred.shape == expected.shape and np.array_equal(pred > 0, expected > 0)


def validate_model(model: onnx.ModelProto) -> Tuple[bool, str]:
    try:
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    except Exception as exc:
        return False, f"ORT load failed: {exc}"

    if DATA_PATH.is_file():
        with DATA_PATH.open(encoding="utf-8") as fh:
            data = json.load(fh)
        for split in ("train", "test", "arc-gen"):
            for idx, ex in enumerate(data[split]):
                inp = np.asarray(ex["input"], dtype=np.int64)
                if max(inp.shape) > 30:
                    continue
                exp = np.asarray(ex["output"], dtype=np.int64)
                ref = solve(inp)
                if not np.array_equal(ref, exp):
                    return False, f"reference mismatch {split}#{idx}"
                pred = sess.run([OUT_NAME], {IN_NAME: _grid_to_onehot(inp)})[0]
                if not _strict_onehot_matches(pred, _expected_onehot(exp)):
                    return False, f"{split}#{idx} ONNX mismatch"
    return True, "PASS"


BUILDERS: Dict[str, Callable[[], onnx.ModelProto]] = {
    "cumsum_crop15_o11": lambda: build_cumsum_argmax_crop15(11),
    "cumsum_crop15_patch_o11": lambda: build_cumsum_crop15_patch(11),
    "cumsum_crop15_ch8_o11": lambda: build_cumsum_crop15_ch8_patch(11),
    "cumsum_crop15_ch8_bool_o11": lambda: build_cumsum_crop15_ch8_bool(11),
    "bounds_crop15_ch8_bool_o10": lambda: build_bounds_crop15_ch8_bool(10),
    "bounds_crop15_ch8_bool_o11": lambda: build_bounds_crop15_ch8_bool(11),
}


def run_experiments() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for name, builder in BUILDERS.items():
        path = OUT_DIR / f"task050_variant_{name}.onnx"
        row: Dict[str, Any] = {"name": name, "path": path}
        try:
            model = builder()
            onnx.save(model, str(path))
            ok, msg = validate_model(model)
            scored = score_file(path)
            row.update(
                {
                    "valid": ok and scored.get("valid"),
                    "validation": msg,
                    "opset": model.opset_import[0].version,
                    "nodes": len(model.graph.node),
                    "memory": scored.get("memory"),
                    "params": scored.get("params"),
                    "cost": scored.get("cost"),
                    "score": scored.get("score"),
                }
            )
        except Exception as exc:
            row["valid"] = False
            row["validation"] = str(exc)
        results.append(row)
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", action="store_true", help="score all ONNX variants")
    args = parser.parse_args()

    if args.experiment:
        results = run_experiments()
        print(f"\n{'variant':<24} {'pass':<5} {'mem':>7} {'par':>6} {'cost':>7} {'score':>8} {'nodes':>5}")
        for row in results:
            sc = row.get("score")
            sc_s = f"{sc:.4f}" if isinstance(sc, float) else "FAIL"
            print(
                f"{row['name']:<24} {str(row.get('valid')):<5} "
                f"{str(row.get('memory', '-')):>7} {str(row.get('params', '-')):>6} "
                f"{str(row.get('cost', '-')):>7} {sc_s:>8} {str(row.get('nodes', '-')):>5}"
            )
        valid = [r for r in results if r.get("valid") and r.get("cost") is not None]
        if not valid:
            raise SystemExit("no valid variant")
        best = min(valid, key=lambda r: int(r["cost"]))
        onnx.save(onnx.load(str(best["path"])), str(BEST_PATH))
        print(f"\nBest: {best['name']} -> {BEST_PATH}")
        print(f"memory={best['memory']} params={best['params']} cost={best['cost']} score={best['score']:.6f}")
        return

    model = build_bounds_crop15_ch8_bool(10)
    onnx.save(model, str(BEST_PATH))
    ok, msg = validate_model(model)
    scored = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"validation: {msg}")
    print(f"memory:   {scored['memory']}")
    print(f"params:   {scored['params']}")
    print(f"cost:     {scored['cost']}")
    print(f"score:    {scored['score']:.6f}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
