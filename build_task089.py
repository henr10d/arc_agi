#!/usr/bin/env python3
"""Build NeuroGolf ONNX for ARC task089: paste 2-color patterns at isolated markers.

Task rule (13x13): find complete 2-color patterns whose rare color is the marker (count 1
in the pattern). Copy the pattern onto every isolated cell of that marker color, aligning
marker-to-marker. Horizontal flip is applied when the pattern is asymmetric, marker != 3,
and target offset from source has dc != 0 or dr != 0.

ONNX: ArgMax locates the source anchor per marker color; Gather reads pattern colors at
fixed offsets; shifted target masks paint onto a uint8 grid; final Equal one-hot + Pad.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task089"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_PATH = ROOT / "solution_task089.onnx"

C = 10
N = 13
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
CORE_SHAPE = [1, C, N, N]
OPSET = 11
IR_VERSION = 10

# Pattern stencil radius (patterns fit in about 4x4)
OFFS = list(range(-2, 3))
OFFS_PAIRS = [(pdy, pdc) for pdy in OFFS for pdc in OFFS]
OFF_NBR = {
    (pdy, pdc): [
        (pdy2, pdc2)
        for pdy2 in OFFS
        for pdc2 in OFFS
        if abs(pdy2 - pdy) <= 1 and abs(pdc2 - pdc) <= 1 and (pdy2, pdc2) != (pdy, pdc)
    ]
    for pdy, pdc in OFFS_PAIRS
}
REACH_ITERS = 8

# Marker colors that host a 2-color pattern (singleton inside component)
MARKERS = [2, 3]

# Horizontal-flip patterns (from task JSON); marker 3 never flips in reference
ASYM = {2: True, 3: True}


def _init(inits: list[onnx.TensorProto], arr: np.ndarray, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _u8(inits: list[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.uint8), name)


def _f32(inits: list[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _or_chain(nodes: list[onnx.NodeProto], parts: list[str], name: str) -> str:
    cur = parts[0]
    for i, part in enumerate(parts[1:], start=1):
        nxt = f"{name}_or{i}"
        nodes.append(helper.make_node("Or", [cur, part], [nxt]))
        cur = nxt
    return cur


def _shift_n(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    tensor: str,
    dr: int,
    dc: int,
    name: str,
    *,
    pad_as_input: bool,
) -> str:
    r0 = max(0, -dr)
    r1 = N - max(0, dr)
    c0 = max(0, -dc)
    c1 = N - max(0, dc)
    top = max(0, dr)
    bottom = max(0, -dr)
    left = max(0, dc)
    right = max(0, -dc)
    starts = _i64(inits, [0, 0, r0, c0], f"{name}_st")
    ends = _i64(inits, [1, 1, r1, c1], f"{name}_en")
    axes = _i64(inits, [0, 1, 2, 3], f"{name}_ax")
    cropped = f"{name}_cr"
    nodes.append(helper.make_node("Slice", [tensor, starts, ends, axes], [cropped]))
    out = f"{name}_sh"
    if pad_as_input:
        pads = _i64(inits, [0, 0, top, left, 0, 0, bottom, right], f"{name}_pd")
        nodes.append(helper.make_node("Pad", [cropped, pads], [out], mode="constant"))
    else:
        nodes.append(
            helper.make_node(
                "Pad",
                [cropped],
                [out],
                mode="constant",
                pads=[0, 0, top, left, 0, 0, bottom, right],
            )
        )
    return out


def solve_reference(grid: np.ndarray) -> np.ndarray:
    """Reference solver on 13x13 integer grid."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()

    def has_partner8(r: int, c: int) -> bool:
        m = int(g[r, c])
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N:
                    v = int(g[nr, nc])
                    if v and v != m:
                        return True
        return False

    uniq: dict[tuple[int, frozenset], tuple] = {}
    for ar in range(N):
        for ac in range(N):
            if not g[ar, ac] or not has_partner8(ar, ac):
                continue
            marker = int(g[ar, ac])
            partners: set[int] = set()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = ar + dr, ac + dc
                    if 0 <= nr < N and 0 <= nc < N:
                        v = int(g[nr, nc])
                        if v and v != marker:
                            partners.add(v)
            allowed = {marker} | partners
            stack = [(ar, ac)]
            seen = {(ar, ac)}
            cells: list[tuple[int, int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c, int(g[r, c])))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < N and 0 <= nc < N and (nr, nc) not in seen:
                            v = int(g[nr, nc])
                            if v in allowed:
                                seen.add((nr, nc))
                                stack.append((nr, nc))
            counts: dict[int, int] = {}
            for _, _, col in cells:
                counts[col] = counts.get(col, 0) + 1
            if len(counts) != 2:
                continue
            mk = [col for col, n in counts.items() if n == 1]
            if len(mk) != 1 or mk[0] != marker:
                continue
            sr, sc = [(a, b) for a, b, col in cells if col == marker][0]
            comp_set = frozenset((a, b) for a, b, _ in cells)
            pid = (marker, comp_set)
            if pid not in uniq:
                uniq[pid] = (cells, sr, sc, marker, comp_set)

    done: set[tuple[int, frozenset]] = set()
    for cells, sr, sc, marker, comp_set in uniq.values():
        pid = (marker, comp_set)
        if pid in done:
            continue
        done.add(pid)
        rel = {(a - sr, b - sc) for a, b, _ in cells}
        asym = any((dr, dc) in rel and (dr, -dc) not in rel for dr, dc in rel)
        for tr in range(N):
            for tc in range(N):
                if g[tr, tc] != marker or (tr, tc) in comp_set or has_partner8(tr, tc):
                    continue
                dr, dc = tr - sr, tc - sc
                flip = asym and marker != 3 and (dc != 0 or dr != 0)
                for cr, cc, col in cells:
                    pdr, pdc = cr - sr, cc - sc
                    if flip:
                        pdc = -pdc
                    nr, nc = tr + pdr, tc + pdc
                    if 0 <= nr < N and 0 <= nc < N:
                        out[nr, nc] = col
    return out


def build_model(opset: int = OPSET) -> onnx.ModelProto:
    pad_as_input = opset >= 11
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    core_st = _i64(inits, [0, 0, 0, 0], "core_st")
    core_en = _i64(inits, [1, C, N, N], "core_en")
    zero_f = _f32(inits, [0.0], "zero_f")
    one_f = _f32(inits, [1.0], "one_f")
    n_sq = _i64(inits, [N * N], "n_sq")
    n_sq_f = _f32(inits, [float(N * N - 1)], "n_sq_f")
    flat_shape = _i64(inits, [N * N], "flat169")
    n_lin = _i64(inits, [N], "n_lin")
    chans = _i64(inits, np.arange(C, dtype=np.int64).reshape(1, C, 1, 1), "chans")

    rows = np.arange(N, dtype=np.int64).reshape(N, 1)
    cols = np.arange(N, dtype=np.int64).reshape(1, N)
    row_w = _f32(inits, (rows * N).reshape(1, 1, N, 1).astype(np.float32), "row_w")
    col_w = _f32(inits, cols.reshape(1, 1, 1, N).astype(np.float32), "col_w")
    row_grid = _i64(inits, np.broadcast_to(rows, (1, 1, N, N)), "row_grid")
    col_grid = _i64(inits, np.broadcast_to(cols, (1, 1, N, N)), "col_grid")

    ar_lut = _i64(inits, (np.arange(N * N, dtype=np.int64) // N), "ar_lut")
    ac_lut = _i64(inits, (np.arange(N * N, dtype=np.int64) % N), "ac_lut")

    zero_i = _i64(inits, [0], "zero_i")
    n13_i = _i64(inits, [N], "n13")
    sh1 = _i64(inits, [1, 1, 1, 1], "sh1111")
    sh13 = _i64(inits, [1, 1, N, N], "sh13")
    tile13 = _i64(inits, [1, 1, N, N], "tile13")
    ones13 = _f32(inits, np.ones((1, 1, N, N), dtype=np.float32), "ones13")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, core_st, core_en, axes4], ["core"]),
            helper.make_node("ArgMax", ["core"], ["ids64"], axis=1, keepdims=1),
            helper.make_node("Cast", ["ids64"], ["idsf"], to=TensorProto.FLOAT),
            helper.make_node("Reshape", ["ids64", "flat169"], ["ids_flat"]),
            helper.make_node("Greater", ["ids64", zero_i], ["nz_b"]),
        ]
    )

    combined = "idsf"

    for m in MARKERS:
        m_i = _i64(inits, [m], f"m{m}")
        nodes.append(helper.make_node("Equal", ["ids64", m_i], [f"occ{m}"]))
        nodes.append(helper.make_node("Cast", [f"occ{m}"], [f"occ{m}b"], to=TensorProto.BOOL))

        diff_parts: list[str] = []
        for di, (dr, dc) in enumerate(
            [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
        ):
            shifted = _shift_n(nodes, inits, "idsf", dr, dc, f"nb{m}_{di}", pad_as_input=pad_as_input)
            sh_i = f"nbi{m}_{di}"
            nodes.append(helper.make_node("Cast", [shifted], [sh_i], to=TensorProto.INT64))
            nbr_nz = f"nnz{m}_{di}"
            nbr_eq = f"neq{m}_{di}"
            nbr_diff = f"ndf{m}_{di}"
            nodes.append(helper.make_node("Greater", [sh_i, zero_i], [nbr_nz]))
            nodes.append(helper.make_node("Equal", [sh_i, m_i], [nbr_eq]))
            nodes.append(helper.make_node("Not", [nbr_eq], [f"nneq{m}_{di}"]))
            nodes.append(helper.make_node("And", [nbr_nz, f"nneq{m}_{di}"], [nbr_diff]))
            diff_parts.append(nbr_diff)
        partner = _or_chain(nodes, diff_parts, f"part{m}")
        nodes.append(helper.make_node("And", [f"occ{m}b", partner], [f"src{m}"]))
        nodes.append(helper.make_node("Not", [partner], [f"nop{m}"]))
        nodes.append(helper.make_node("And", [f"occ{m}b", f"nop{m}"], [f"tgt{m}"]))
        nodes.append(helper.make_node("Cast", [f"tgt{m}"], [f"tgtf{m}"], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Cast", [f"src{m}"], [f"srcf{m}"], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Add", [row_w, col_w], [f"rcw{m}"]))
        nodes.append(helper.make_node("Mul", [f"srcf{m}", f"rcw{m}"], [f"sw{m}"]))
        nodes.append(helper.make_node("Reshape", [f"sw{m}", "flat169"], [f"swf{m}"]))
        nodes.append(helper.make_node("ArgMax", [f"swf{m}"], [f"aidx{m}"], axis=0, keepdims=0))
        nodes.append(helper.make_node("Reshape", [f"aidx{m}", sh1], [f"aidx1{m}"]))
        nodes.append(helper.make_node("Gather", [ar_lut, f"aidx{m}"], [f"ar{m}"]))
        nodes.append(helper.make_node("Gather", [ac_lut, f"aidx{m}"], [f"ac{m}"]))
        nodes.append(helper.make_node("Reshape", [f"ar{m}", sh1], [f"ar1{m}"]))
        nodes.append(helper.make_node("Reshape", [f"ac{m}", sh1], [f"ac1{m}"]))
        nodes.append(helper.make_node("Sub", ["row_grid", f"ar1{m}"], [f"drow{m}"]))
        nodes.append(helper.make_node("Sub", ["col_grid", f"ac1{m}"], [f"dcol{m}"]))
        nodes.append(helper.make_node("Equal", [f"drow{m}", zero_i], [f"drq{m}"]))
        nodes.append(helper.make_node("Equal", [f"dcol{m}", zero_i], [f"dcq{m}"]))
        nodes.append(helper.make_node("Not", [f"drq{m}"], [f"drnz{m}"]))
        nodes.append(helper.make_node("Not", [f"dcq{m}"], [f"dcnz{m}"]))
        nodes.append(helper.make_node("Or", [f"drnz{m}", f"dcnz{m}"], [f"flb{m}"]))
        nodes.append(helper.make_node("And", [f"tgt{m}", f"flb{m}"], [f"fltgt{m}"]))
        nodes.append(helper.make_node("Cast", [f"fltgt{m}"], [f"fltgtf{m}"], to=TensorProto.FLOAT))

        anch0 = f"anc0{m}"
        anch_b = f"ancb{m}"
        nodes.append(helper.make_node("Equal", ["row_grid", f"ar1{m}"], [f"arq{m}"]))
        nodes.append(helper.make_node("Equal", ["col_grid", f"ac1{m}"], [f"acq{m}"]))
        nodes.append(helper.make_node("And", [f"arq{m}", f"acq{m}"], [anch_b]))
        nodes.append(helper.make_node("Cast", [anch_b], [anch0], to=TensorProto.FLOAT))
        partner_cols: list[str] = []
        for di, (dr, dc) in enumerate(
            [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
        ):
            noff = _i64(inits, [dr * N + dc], f"pnoff{m}_{di}")
            nidx = f"pnix{m}_{di}"
            nodes.append(helper.make_node("Add", [f"aidx{m}", noff], [nidx]))
            pnf2 = f"pnf2{m}_{di}"
            c0 = f"pc0{m}_{di}"
            cl = f"pcl{m}_{di}"
            nodes.append(helper.make_node("Cast", [nidx], [pnf2], to=TensorProto.FLOAT))
            nodes.append(helper.make_node("Max", [pnf2, zero_f], [c0]))
            nodes.append(helper.make_node("Min", [c0, n_sq_f], [cl]))
            cli = f"pcli{m}_{di}"
            nodes.append(helper.make_node("Cast", [cl], [cli], to=TensorProto.INT64))
            ncol = f"pncol{m}_{di}"
            nodes.append(helper.make_node("Gather", ["ids_flat", cli], [ncol]))
            partner_cols.append(ncol)
        comp = anch0
        for it in range(REACH_ITERS):
            cur = comp
            for di, (dr, dc) in enumerate(
                [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
            ):
                prop = _shift_n(nodes, inits, cur, dr, dc, f"cp{m}_{it}_{di}", pad_as_input=pad_as_input)
                sid = _shift_n(
                    nodes, inits, "ids64", dr, dc, f"cid{m}_{it}_{di}", pad_as_input=pad_as_input
                )
                snz_b = f"snzb{m}_{it}_{di}"
                nodes.append(helper.make_node("Greater", [sid, zero_i], [snz_b]))
                allow = f"alw{m}_{it}_{di}"
                nodes.append(helper.make_node("Equal", [sid, m_i], [allow]))
                cur_allow = allow
                for pi, pcol in enumerate(partner_cols):
                    pe = f"pe{m}_{it}_{di}_{pi}"
                    nodes.append(helper.make_node("Equal", [sid, pcol], [pe]))
                    nxt_allow = f"aln{m}_{it}_{di}_{pi}"
                    nodes.append(helper.make_node("Or", [cur_allow, pe], [nxt_allow]))
                    cur_allow = nxt_allow
                allow_f = f"alf{m}_{it}_{di}"
                nodes.append(helper.make_node("Cast", [snz_b], [f"snzf{m}_{it}_{di}"], to=TensorProto.FLOAT))
                nodes.append(helper.make_node("Cast", [cur_allow], [allow_f], to=TensorProto.FLOAT))
                prop1 = f"pp1{m}_{it}_{di}"
                prop2 = f"pp2{m}_{it}_{di}"
                nodes.append(helper.make_node("Mul", [prop, f"snzf{m}_{it}_{di}"], [prop1]))
                nodes.append(helper.make_node("Mul", [prop1, allow_f], [prop2]))
                nxt = f"gr{m}_{it}_{di}"
                nodes.append(helper.make_node("Max", [cur, prop2], [nxt]))
                cur = nxt
            comp = f"comp{m}_{it}"
            nodes.append(helper.make_node("Identity", [cur], [comp]))
        compf = comp
        comp_flat = f"cmpf{m}"
        nodes.append(helper.make_node("Reshape", [compf, "flat169"], [comp_flat]))

        hz: dict[tuple[int, int], str] = {}
        coln: dict[tuple[int, int], str] = {}
        for pdy in OFFS:
            for pdc in OFFS:
                tag = f"{m}_{pdy}_{pdc}"
                pdy_i = _i64(inits, [pdy], f"pdy{tag}")
                pdc_i = _i64(inits, [pdc], f"pdc{tag}")
                arp = f"arp{tag}"
                acp = f"acp{tag}"
                nodes.append(helper.make_node("Add", [f"ar{m}", pdy_i], [arp]))
                nodes.append(helper.make_node("Add", [f"ac{m}", pdc_i], [acp]))
                ok_r0 = f"okr0{tag}"
                ok_r1 = f"okr1{tag}"
                ok_c0 = f"okc0{tag}"
                ok_c1 = f"okc1{tag}"
                nodes.append(helper.make_node("Less", [arp, zero_i], [f"okr0l{tag}"]))
                nodes.append(helper.make_node("Not", [f"okr0l{tag}"], [ok_r0]))
                nodes.append(helper.make_node("Less", [arp, n13_i], [ok_r1]))
                nodes.append(helper.make_node("Less", [acp, zero_i], [f"okc0l{tag}"]))
                nodes.append(helper.make_node("Not", [f"okc0l{tag}"], [ok_c0]))
                nodes.append(helper.make_node("Less", [acp, n13_i], [ok_c1]))
                ok_rc = f"okrc{tag}"
                ok_all = f"ok{tag}"
                nodes.append(helper.make_node("And", [ok_r0, ok_r1], [ok_rc]))
                nodes.append(helper.make_node("And", [ok_rc, ok_c0], [f"okrc2{tag}"]))
                nodes.append(helper.make_node("And", [f"okrc2{tag}", ok_c1], [ok_all]))
                ok_f = f"okf{tag}"
                nodes.append(helper.make_node("Cast", [ok_all], [ok_f], to=TensorProto.FLOAT))

                off_n = _i64(inits, [pdy * N + pdc], f"offn{tag}")
                pidx_n = f"ixn{tag}"
                nodes.append(helper.make_node("Add", [f"aidx{m}", off_n], [pidx_n]))
                pnf = f"pnf{tag}"
                clip0_n = f"c0n{tag}"
                clip_n = f"cln{tag}"
                nodes.append(helper.make_node("Cast", [pidx_n], [pnf], to=TensorProto.FLOAT))
                nodes.append(helper.make_node("Max", [pnf, zero_f], [clip0_n]))
                nodes.append(helper.make_node("Min", [clip0_n, n_sq_f], [clip_n]))
                clip_ni = f"clni{tag}"
                nodes.append(helper.make_node("Cast", [clip_n], [clip_ni], to=TensorProto.INT64))
                cn = f"coln{tag}"
                hz_b = f"hzb{tag}"
                hz_name = f"hzn{tag}"
                nodes.append(helper.make_node("Gather", ["ids_flat", clip_ni], [cn]))
                nodes.append(helper.make_node("Greater", [cn, zero_i], [hz_b]))
                nodes.append(helper.make_node("Cast", [hz_b], [hz_name], to=TensorProto.FLOAT))
                mem = f"mem{tag}"
                nodes.append(helper.make_node("Gather", [comp_flat, clip_ni], [mem]))
                memf = f"memf{tag}"
                nodes.append(helper.make_node("Cast", [mem], [memf], to=TensorProto.FLOAT))
                hz_m = f"hzm{tag}"
                hz1 = f"hz1{tag}"
                nodes.append(helper.make_node("Mul", [hz_name, memf], [hz1]))
                nodes.append(helper.make_node("Mul", [hz1, ok_f], [hz_m]))
                coln[(pdy, pdc)] = cn
                hz[(pdy, pdc)] = hz_m

        reach: dict[tuple[int, int], str] = {(0, 0): hz[(0, 0)]}
        for it in range(REACH_ITERS):
            nxt: dict[tuple[int, int], str] = {(0, 0): hz[(0, 0)]}
            for pdy, pdc in OFFS_PAIRS:
                if (pdy, pdc) == (0, 0):
                    continue
                tag = f"{m}_{pdy}_{pdc}"
                nbrs = [reach.get(nb, zero_f) for nb in OFF_NBR[(pdy, pdc)]]
                nr = nbrs[0]
                for ni, nb in enumerate(nbrs[1:], start=1):
                    nm = f"rn{tag}_i{it}_{ni}"
                    nodes.append(helper.make_node("Max", [nr, nb], [nm]))
                    nr = nm
                rk = f"rk{tag}_i{it}"
                nodes.append(helper.make_node("Mul", [hz[(pdy, pdc)], nr], [rk]))
                nxt[(pdy, pdc)] = rk
            reach = nxt

        for pdy in OFFS:
            for pdc in OFFS:
                tag = f"{m}_{pdy}_{pdc}"
                hzn = reach[(pdy, pdc)]

                tgt_n = _shift_n(nodes, inits, f"tgtf{m}", pdy, pdc, f"tn{tag}", pad_as_input=pad_as_input)
                tgt_f = _shift_n(nodes, inits, f"tgtf{m}", pdy, -pdc, f"tf{tag}", pad_as_input=pad_as_input)

                if m == 3 or not ASYM.get(m, True):
                    flip_n = f"fln{tag}"
                    flip_f = f"flf{tag}"
                    nodes.append(helper.make_node("Cast", [zero_f], [f"ufz{tag}"], to=TensorProto.FLOAT))
                    nodes.append(helper.make_node("Reshape", [f"ufz{tag}", sh1], [f"uf1{tag}"]))
                    nodes.append(helper.make_node("Tile", [f"uf1{tag}", tile13], [flip_n]))
                    nodes.append(helper.make_node("Tile", [f"uf1{tag}", tile13], [flip_f]))
                else:
                    flip_n = _shift_n(
                        nodes, inits, f"fltgtf{m}", pdy, pdc, f"fln{tag}", pad_as_input=pad_as_input
                    )
                    flip_f = _shift_n(
                        nodes, inits, f"fltgtf{m}", pdy, -pdc, f"flf{tag}", pad_as_input=pad_as_input
                    )

                hznf = f"hznf{tag}"
                nodes.append(helper.make_node("Cast", [hzn], [f"hznc{tag}"], to=TensorProto.FLOAT))
                nodes.append(helper.make_node("Reshape", [f"hznc{tag}", sh1], [f"hz1n{tag}"]))
                nodes.append(helper.make_node("Tile", [f"hz1n{tag}", tile13], [hznf]))

                gate_n = f"gn{tag}"
                gate_f = f"gf{tag}"
                nodes.append(helper.make_node("Mul", [tgt_n, hznf], [gate_n]))
                nodes.append(helper.make_node("Mul", [tgt_f, hznf], [gate_f]))
                inv_flip_n = f"invn{tag}"
                nodes.append(helper.make_node("Sub", [ones13, flip_n], [inv_flip_n]))
                gate = f"gt{tag}"
                g1 = f"g1{tag}"
                g2 = f"g2{tag}"
                nodes.append(helper.make_node("Mul", [gate_n, inv_flip_n], [g1]))
                nodes.append(helper.make_node("Mul", [gate_f, flip_f], [g2]))
                nodes.append(helper.make_node("Add", [g1, g2], [gate]))

                col_nf = f"cnf{tag}"
                nodes.append(helper.make_node("Cast", [coln[(pdy, pdc)]], [col_nf], to=TensorProto.FLOAT))
                nodes.append(helper.make_node("Reshape", [col_nf, sh1], [f"cn1{tag}"]))
                col_g = f"cg{tag}"
                nodes.append(helper.make_node("Tile", [f"cn1{tag}", tile13], [col_g]))
                paint = f"pnt{tag}"
                nodes.append(helper.make_node("Mul", [gate, col_g], [paint]))
                mx = f"mx{tag}"
                nodes.append(helper.make_node("Max", [combined, paint], [mx]))
                combined = mx

    nodes.append(helper.make_node("Cast", [combined], ["ids_out"], to=TensorProto.INT64))
    nodes.append(helper.make_node("Equal", [chans, "ids_out"], ["oh_b"]))
    nodes.append(helper.make_node("Cast", ["oh_b"], ["oh_f"], to=TensorProto.FLOAT))
    pad = _i64(inits, [0, 0, 0, 0, 0, 0, H - N, W - N], "pad30")
    if pad_as_input:
        nodes.append(helper.make_node("Pad", ["oh_f", pad], [OUT_NAME], mode="constant"))
    else:
        nodes.append(
            helper.make_node(
                "Pad",
                ["oh_f"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - N, W - N],
            )
        )

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    return model


def validate(model: onnx.ModelProto, data: dict) -> tuple[bool, dict[str, tuple[int, int]]]:
    sess = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    counts: dict[str, tuple[int, int]] = {}
    ok = True
    for split in ("train", "test", "arc-gen"):
        passed = 0
        total = 0
        for ex in data.get(split, []):
            inp = convert_to_numpy(ex, "input")
            exp = convert_to_numpy(ex, "output")
            if inp is None or exp is None:
                continue
            total += 1
            out = sess.run([OUT_NAME], {IN_NAME: inp})[0]
            if np.array_equal((out > 0).astype(np.float32), exp):
                passed += 1
            else:
                ok = False
        counts[split] = (passed, total)
    return ok, counts


def main() -> None:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))

    # Verify reference
    ref_ok = True
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            g = np.array(ex["input"], dtype=np.int64)
            pred = solve_reference(g)
            exp = np.array(ex["output"], dtype=np.int64)
            if not np.array_equal(pred, exp):
                ref_ok = False
                break
    print(f"reference_solver_ok={ref_ok}")

    best: tuple[int, float, int, onnx.ModelProto] | None = None
    for opset in (10, 11, 13, 18):
        try:
            model = build_model(opset)
        except Exception as exc:
            print(f"opset {opset}: build failed: {exc}")
            continue
        try:
            ok, counts = validate(model, data)
        except Exception as exc:
            print(f"opset {opset}: validate failed: {exc}")
            continue
        print(f"opset {opset}: correct={ok} counts={counts}")
        if not ok:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / OUT_PATH.name
            onnx.save(model, p)
            stats = score_file(p)
        if not stats["valid"]:
            print(f"  score invalid: {stats.get('error')}")
            continue
        cost = int(stats["cost"])
        sc = float(stats["score"])
        print(f"  memory={stats['memory']} params={stats['params']} cost={cost} score={sc:.6f}")
        if best is None or cost < best[0]:
            best = (cost, sc, opset, model)

    if best is None:
        raise SystemExit("no valid ONNX candidate")

    cost, sc, opset, model = best
    onnx.save(model, OUT_PATH)
    print(f"\nWrote {OUT_PATH} (opset={opset}, cost={cost}, score={sc:.6f})")

    stats = score_file(OUT_PATH)
    print("\n=== Final score ===")
    print(f"memory: {stats['memory']}")
    print(f"params: {stats['params']}")
    print(f"cost:   {stats['cost']}")
    print(f"score:  {stats['score']:.6f}")


if __name__ == "__main__":
    main()
