"""Minimal ONNX for ARC task005: sprite template stamping from partial markers.

Task rule (21x21 grids in top-left of 30x30 one-hot I/O):
1. Largest 8-connected component is the sprite (always 3x3 bbox); found via max 3x3 window sum.
2. Bool template from sprite color in that 3x3 window.
3. Each other color is a partial marker; direction = sprite center -> marker center (8-way).
4. Anchor = template placement covering all marker cells; tie-break by max dot with direction.
5. Stride = 4 (3x3 template + gap 1); stamp up to 10 steps on background cells only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

BEST_PATH = OUT_DIR / "task005.onnx"
DATA_PATH = ROOT / "data" / "task005.json"

C = 10
GH = GW = 21
WIN = GH - 2
H = W = 30
STEPS = 2
STRIDE = 4
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10

DIRS8: Tuple[Tuple[int, int], ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


def _load_toy() -> Tuple[List[List[int]], List[List[int]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    ex = data["train"][0]
    return ex["input"], ex["output"]


def find_components(grid: np.ndarray, conn8: bool = True) -> List[dict]:
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    nbrs = list(DIRS8 if conn8 else DIRS8[:4])
    components: List[dict] = []
    for y in range(h):
        for x in range(w):
            if g[y, x] == 0 or visited[y, x]:
                continue
            color = int(g[y, x])
            cells: List[Tuple[int, int]] = []
            stack = [(y, x)]
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for dy, dx in nbrs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and g[ny, nx] == color:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            ys = [c[0] for c in cells]
            xs = [c[1] for c in cells]
            components.append(
                {
                    "color": color,
                    "cells": set(cells),
                    "bbox": (min(ys), min(xs), max(ys) + 1, max(xs) + 1),
                    "size": len(cells),
                }
            )
    return components


def quantize_dir(dr: float, dc: float) -> Tuple[int, int]:
    if abs(dr) < 0.5 and abs(dc) < 0.5:
        return (0, 0)
    sdr = -1 if dr < -0.5 else (1 if dr > 0.5 else 0)
    sdc = -1 if dc < -0.5 else (1 if dc > 0.5 else 0)
    return (sdr, sdc)


def get_template(sprite: dict, grid: np.ndarray) -> Tuple[np.ndarray, int, int]:
    sr0, sc0, sr1, sc1 = sprite["bbox"]
    g = np.asarray(grid, dtype=np.int64)
    sh, sw = sr1 - sr0, sc1 - sc0
    template = np.zeros((sh, sw), dtype=bool)
    for rr in range(sr0, sr1):
        for cc in range(sc0, sc1):
            if g[rr, cc] == sprite["color"]:
                template[rr - sr0, cc - sc0] = True
    return template, sh, sw


def find_anchor_dir(
    marker: dict,
    template: np.ndarray,
    sh: int,
    sw: int,
    sp_cy: float,
    sp_cx: float,
    sdr: int,
    sdc: int,
    h: int,
    w: int,
) -> Tuple[int, int]:
    partial = marker["cells"]
    candidates: List[Tuple[int, int]] = []
    for tr0 in range(h - sh + 1):
        for tc0 in range(w - sw + 1):
            ok = all(
                0 <= rr - tr0 < sh and 0 <= cc - tc0 < sw and template[rr - tr0, cc - tc0]
                for rr, cc in partial
            )
            if ok:
                candidates.append((tr0, tc0))
    if not candidates:
        return marker["bbox"][0], marker["bbox"][1]
    if len(candidates) == 1:
        return candidates[0]

    def score(pos: Tuple[int, int]) -> float:
        tr0, tc0 = pos
        acy = tr0 + sh / 2.0
        acx = tc0 + sw / 2.0
        return (acy - sp_cy) * sdr + (acx - sp_cx) * sdc

    return max(candidates, key=score)


def solve_reference(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    if g.ndim == 4:
        g = g[0].argmax(axis=0)
    elif g.ndim == 3:
        g = g.argmax(axis=0)

    h, w = g.shape
    out = g.copy()
    comps = find_components(g, conn8=True)
    if not comps:
        return out

    sprite = max(comps, key=lambda c: c["size"])
    template, sh, sw = get_template(sprite, g)
    sr0, sc0, sr1, sc1 = sprite["bbox"]
    sp_cy = (sr0 + sr1) / 2.0
    sp_cx = (sc0 + sc1) / 2.0
    stride = sh + 1

    for marker in comps:
        if marker is sprite:
            continue
        cr0, cc0, cr1, cc1 = marker["bbox"]
        ccy = (cr0 + cr1) / 2.0
        ccx = (cc0 + cc1) / 2.0
        sdr, sdc = quantize_dir(ccy - sp_cy, ccx - sp_cx)
        if sdr == 0 and sdc == 0:
            continue
        mcolor = marker["color"]
        tr0, tc0 = find_anchor_dir(marker, template, sh, sw, sp_cy, sp_cx, sdr, sdc, h, w)

        def add_template(tr: int, tc: int, color: int) -> None:
            for sr in range(sh):
                for sc in range(sw):
                    if template[sr, sc]:
                        rr, cc = tr + sr, tc + sc
                        if 0 <= rr < h and 0 <= cc < w and out[rr, cc] == 0:
                            out[rr, cc] = color

        add_template(tr0, tc0, mcolor)
        for step in range(1, STEPS + 1):
            ntr = tr0 + sdr * step * stride
            ntc = tc0 + sdc * step * stride
            if ntr + sh <= 0 or ntc + sw <= 0 or ntr >= h or ntc >= w:
                break
            if not any(
                template[sr, sc] and 0 <= ntr + sr < h and 0 <= ntc + sc < w
                for sr in range(sh)
                for sc in range(sw)
            ):
                break
            add_template(ntr, ntc, mcolor)
    return out


class _Builder:
    def __init__(self) -> None:
        self.nodes: List[onnx.NodeProto] = []
        self.inits: List[onnx.TensorProto] = []
        self._scalar_inits: Dict[Tuple[str, float | int], str] = {}
        self._n = 0

    def name(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    def arr(self, array, n: str | None = None) -> str:
        n = n or self.name("c")
        self.inits.append(numpy_helper.from_array(np.asarray(array), name=n))
        return n

    def i64(self, vals, n: str | None = None) -> str:
        array = np.asarray(vals, dtype=np.int64)
        if n is None and array.shape == (1,):
            key = ("i64", int(array[0]))
            if key not in self._scalar_inits:
                self._scalar_inits[key] = self.arr(array)
            return self._scalar_inits[key]
        return self.arr(array, n)

    def f32(self, vals, n: str | None = None) -> str:
        array = np.asarray(vals, dtype=np.float32)
        if n is None and array.shape == (1,):
            key = ("f32", float(array[0]))
            if key not in self._scalar_inits:
                self._scalar_inits[key] = self.arr(array)
            return self._scalar_inits[key]
        return self.arr(array, n)

    def add(self, op: str, ins: List[str], outs: List[str], **kw) -> None:
        self.nodes.append(helper.make_node(op, ins, outs, **kw))


def _rel_pos_mask(b: _Builder, rows: str, cols: str, tr: str, tc: str, sr: int, sc: int) -> str:
    """Bool [1,1,H,W]: grid position equals (tr+sr, tc+sc)."""
    one = b.f32([1.0])
    srf, scf = b.f32([float(sr)]), b.f32([float(sc)])
    rs = b.name("rs")
    rsa = b.name("rsa")
    cs = b.name("cs")
    csa = b.name("csa")
    b.add("Sub", [rows, tr], [rs])
    rsaa = b.name("rsaa")
    b.add("Sub", [rs, srf], [rsa])
    b.add("Abs", [rsa], [rsaa])
    b.add("Less", [rsaa, one], [b.name("rok")])
    rok = b.nodes[-1].output[0]
    b.add("Sub", [cols, tc], [cs])
    csaa = b.name("csaa")
    b.add("Sub", [cs, scf], [csa])
    b.add("Abs", [csa], [csaa])
    b.add("Less", [csaa, one], [b.name("cok")])
    cok = b.nodes[-1].output[0]
    out = b.name("pos")
    b.add("And", [rok, cok], [out])
    return out


def _build_template_bits(b: _Builder, rows: str, cols: str, r0: str, c0: str, fgb: str, one: str) -> List[str]:
    bits: List[str] = []
    for sr in range(3):
        for sc in range(3):
            pos = _rel_pos_mask(b, rows, cols, r0, c0, sr, sc)
            bit = b.name("tb")
            b.add("And", [pos, fgb], [bit])
            bits.append(bit)
    return bits


def _crop19(b: _Builder, x: str, sr: int, sc: int, ax4: str) -> str:
    st = b.i64([0, 0, sr, sc])
    en = b.i64([1, 1, sr + WIN, sc + WIN])
    out = b.name("crp")
    b.add("Slice", [x, st, en, ax4], [out])
    return out


def _match_map(b: _Builder, mnzf: str, tmpl: List[str], ax4: str, z19: str) -> str:
    match = z19
    for i, tb in enumerate(tmpl):
        sr, sc = divmod(i, 3)
        crop = _crop19(b, mnzf, sr, sc, ax4)
        tbf = b.name("tbf")
        b.add("Cast", [tb], [tbf], to=TensorProto.FLOAT)
        tbs = b.name("tbs")
        b.add("ReduceSum", [tbf], [tbs])
        term = b.name("term")
        b.add("Mul", [crop, tbs], [term])
        match_n = b.name("mm")
        b.add("Add", [match, term], [match_n])
        match = match_n
    return match


def _bad_template_map(b: _Builder, mnzf: str, tmpl: List[str], ax4: str, z19: str, one: str) -> str:
    bad = z19
    for i, tb in enumerate(tmpl):
        sr, sc = divmod(i, 3)
        crop = _crop19(b, mnzf, sr, sc, ax4)
        tbf = b.name("tbf")
        b.add("Cast", [tb], [tbf], to=TensorProto.FLOAT)
        tbs = b.name("tbs")
        b.add("ReduceSum", [tbf], [tbs])
        inv = b.name("tbinv")
        b.add("Sub", [one, tbs], [inv])
        term = b.name("badterm")
        b.add("Mul", [crop, inv], [term])
        bad_n = b.name("badm")
        b.add("Add", [bad, term], [bad_n])
        bad = bad_n
    return bad


def _dir_axis_map(b: _Builder, vals: str, center: str, sign: int, half: str) -> str:
    diff = b.name("dd")
    b.add("Sub", [vals, center], [diff])
    if sign > 0:
        out = b.name("dpos")
        b.add("Greater", [diff, half], [out])
        return out
    if sign < 0:
        neg_half = b.f32([-0.5])
        out = b.name("dneg")
        b.add("Less", [diff, neg_half], [out])
        return out
    ad = b.name("dabs")
    out = b.name("dzero")
    b.add("Abs", [diff], [ad])
    b.add("Less", [ad, half], [out])
    return out


def _argmax_anchor(
    b: _Builder,
    score_map: str,
    one5: str,
    winf: str,
) -> Tuple[str, str]:
    flat = b.name("flat")
    b.add("Reshape", [score_map, b.i64([-1])], [flat])
    idx = b.name("idx")
    b.add("ArgMax", [flat], [idx], axis=0, keepdims=0)
    idxf = b.name("idxf")
    b.add("Cast", [idx], [idxf], to=TensorProto.FLOAT)
    trf = b.name("trf")
    b.add("Div", [idxf, winf], [trf])
    tri = b.name("tri")
    b.add("Floor", [trf], [tri])
    trff = b.name("trff")
    b.add("Cast", [tri], [trff], to=TensorProto.FLOAT)
    trw = b.name("trw")
    b.add("Mul", [trff, winf], [trw])
    tcf = b.name("tcf")
    b.add("Sub", [idxf, trw], [tcf])
    tcff = b.name("tcff")
    b.add("Cast", [tcf], [tcff], to=TensorProto.FLOAT)
    return trff, tcff


def _stamp_at(
    b: _Builder,
    accum: str,
    tmpl: List[str],
    rows: str,
    cols: str,
    tr: str,
    tc: str,
    half: str,
) -> str:
    for i, tb in enumerate(tmpl):
        sr, sc = divmod(i, 3)
        pos = _rel_pos_mask(b, rows, cols, tr, tc, sr, sc)
        tbf = b.name("tbf")
        b.add("Cast", [tb], [tbf], to=TensorProto.FLOAT)
        tbs = b.name("tbs")
        b.add("ReduceSum", [tbf], [tbs])
        gate = b.name("gate")
        b.add("Greater", [tbs, half], [gate])
        on = b.name("on")
        b.add("And", [pos, gate], [on])
        onf = b.name("onf")
        b.add("Cast", [on], [onf], to=TensorProto.FLOAT)
        acc_n = b.name("acc")
        b.add("Max", [accum, onf], [acc_n])
        accum = acc_n
    return accum


def _process_marker_color_dir(
    b: _Builder,
    mc: int,
    direction: Tuple[int, int],
    core: str,
    rows: str,
    cols: str,
    sp_cy: str,
    sp_cx: str,
    sp_color: str,
    tmpl: List[str],
    bg: str,
    half: str,
    one: str,
    ax4: str,
    one5: str,
    winf: str,
    neg_big: str,
    rows19: str,
    cols19: str,
) -> str:
    st, en = b.i64([0, mc, 0, 0]), b.i64([1, mc + 1, GH, GW])
    mch = b.name("mch")
    b.add("Slice", [core, st, en, ax4], [mch])
    mnz = b.name("mnz")
    b.add("Greater", [mch, half], [mnz])
    mnzf = b.name("mnzf")
    b.add("Cast", [mnz], [mnzf], to=TensorProto.FLOAT)

    z19 = b.arr(np.zeros((1, 1, WIN, WIN), dtype=np.float32), b.name(f"z19_{mc}_{direction[0]}_{direction[1]}"))
    good = _match_map(b, mnzf, tmpl, ax4, z19)
    bad = _bad_template_map(b, mnzf, tmpl, ax4, z19, one)

    big = b.f32([999.0])
    nbig = b.f32([-999.0])
    min_r = big
    min_c = big
    max_r = nbig
    max_c = nbig
    for i, tb in enumerate(tmpl):
        sr, sc = divmod(i, 3)
        crop = _crop19(b, mnzf, sr, sc, ax4)
        tbf = b.name("tbf")
        b.add("Cast", [tb], [tbf], to=TensorProto.FLOAT)
        tbs = b.name("tbs")
        b.add("ReduceSum", [tbf], [tbs])
        gated = b.name("lgate")
        hit = b.name("lhit")
        b.add("Mul", [crop, tbs], [gated])
        b.add("Greater", [gated, half], [hit])
        rv = b.name("lrv")
        cv = b.name("lcv")
        b.add("Add", [rows19, b.f32([float(sr)])], [rv])
        b.add("Add", [cols19, b.f32([float(sc)])], [cv])
        rlo = b.name("rlo")
        clo = b.name("clo")
        rhi = b.name("rhi")
        chi = b.name("chi")
        b.add("Where", [hit, rv, big], [rlo])
        b.add("Where", [hit, cv, big], [clo])
        b.add("Where", [hit, rv, nbig], [rhi])
        b.add("Where", [hit, cv, nbig], [chi])
        min_r_n = b.name("minr")
        min_c_n = b.name("minc")
        max_r_n = b.name("maxr")
        max_c_n = b.name("maxc")
        b.add("Min", [min_r, rlo], [min_r_n])
        b.add("Min", [min_c, clo], [min_c_n])
        b.add("Max", [max_r, rhi], [max_r_n])
        b.add("Max", [max_c, chi], [max_c_n])
        min_r, min_c, max_r, max_c = min_r_n, min_c_n, max_r_n, max_c_n

    lrs = b.name("lrs")
    lcs = b.name("lcs")
    lrs1 = b.name("lrs1")
    lcs1 = b.name("lcs1")
    lcy = b.name("lcy")
    lcx = b.name("lcx")
    b.add("Add", [min_r, max_r], [lrs])
    b.add("Add", [min_c, max_c], [lcs])
    b.add("Add", [lrs, one], [lrs1])
    b.add("Add", [lcs, one], [lcs1])
    b.add("Mul", [lrs1, half], [lcy])
    b.add("Mul", [lcs1, half], [lcx])

    acy, acx = b.name("acy"), b.name("acx")
    b.add("Add", [rows19, one5], [acy])
    b.add("Add", [cols19, one5], [acx])
    rdir = _dir_axis_map(b, lcy, sp_cy, direction[0], half)
    cdir = _dir_axis_map(b, lcx, sp_cx, direction[1], half)
    dirm = b.name("dirm")
    b.add("And", [rdir, cdir], [dirm])

    has_good = b.name("hgood")
    no_bad = b.name("nbad")
    b.add("Greater", [good, half], [has_good])
    b.add("Less", [bad, half], [no_bad])
    fit = b.name("fit")
    valid = b.name("valid")
    b.add("And", [has_good, no_bad], [fit])
    b.add("And", [fit, dirm], [valid])
    valf = b.name("valf")
    b.add("Cast", [valid], [valf], to=TensorProto.FLOAT)

    sdr = b.f32([float(direction[0])])
    sdc = b.f32([float(direction[1])])
    dy, dx = b.name("dy"), b.name("dx")
    b.add("Sub", [acy, sp_cy], [dy])
    b.add("Sub", [acx, sp_cx], [dx])
    dy2, dx2 = b.name("dy2"), b.name("dx2")
    b.add("Mul", [dy, sdr], [dy2])
    b.add("Mul", [dx, sdc], [dx2])
    dot = b.name("dot")
    b.add("Add", [dy2, dx2], [dot])
    vdot = b.name("vdot")
    b.add("Mul", [dot, valf], [vdot])
    inv = b.name("inv")
    pen = b.name("pen")
    score = b.name("score")
    b.add("Sub", [one, valf], [inv])
    b.add("Mul", [inv, neg_big], [pen])
    b.add("Add", [vdot, pen], [score])
    best_tr, best_tc = _argmax_anchor(b, score, one5, winf)

    any_valid_f = b.name("anyvf")
    any_valid = b.name("anyv")
    b.add("ReduceMax", [valf], [any_valid_f], axes=[2, 3], keepdims=1)
    b.add("Greater", [any_valid_f, half], [any_valid])

    stamp = b.f32([0.0])
    for step in range(STEPS + 1):
        cur_tr, cur_tc = best_tr, best_tc
        if step > 0:
            off = b.f32([float(step * STRIDE)])
            dtr, dtc = b.name("dtr"), b.name("dtc")
            b.add("Mul", [sdr, off], [dtr])
            b.add("Mul", [sdc, off], [dtc])
            ntr, ntc = b.name("ntr"), b.name("ntc")
            b.add("Add", [best_tr, dtr], [ntr])
            b.add("Add", [best_tc, dtc], [ntc])
            cur_tr, cur_tc = ntr, ntc
        stamp = _stamp_at(b, stamp, tmpl, rows, cols, cur_tr, cur_tc, half)

    stb = b.name("stb")
    b.add("Greater", [stamp, half], [stb])
    stamp_bg = b.name("stbg")
    b.add("And", [stb, bg], [stamp_bg])
    stamp_present = b.name("stpres")
    b.add("And", [stamp_bg, any_valid], [stamp_present])
    stampf = b.name("stf")
    b.add("Cast", [stamp_present], [stampf], to=TensorProto.FLOAT)

    mcv = b.f32([float(mc)])
    diff = b.name("spd")
    diff_a = b.name("spda")
    b.add("Sub", [mcv, sp_color], [diff])
    b.add("Abs", [diff], [diff_a])
    is_sp = b.name("isp")
    b.add("Less", [diff_a, one], [is_sp])
    not_sp = b.name("nsp")
    b.add("Not", [is_sp], [not_sp])
    nspf = b.name("nspf")
    b.add("Cast", [not_sp], [nspf], to=TensorProto.FLOAT)
    stampf2 = b.name("stf2")
    b.add("Mul", [stampf, nspf], [stampf2])

    out_ch = b.name("och")
    b.add("Slice", [core, st, en, ax4], [out_ch])
    out_new = b.name("onew")
    b.add("Max", [out_ch, stampf2], [out_new])
    return out_new


def _process_marker_color(
    b: _Builder,
    mc: int,
    core: str,
    rows: str,
    cols: str,
    r0: str,
    c0: str,
    sp_cy: str,
    sp_cx: str,
    sp_color: str,
    tmpl: List[str],
    bg: str,
    half: str,
    one: str,
    zf: str,
    ax4: str,
    one5: str,
    winf: str,
    neg_big: str,
    rows19: str,
    cols19: str,
    dir_override: Tuple[int, int] | None = None,
) -> str:
    st, en = b.i64([0, mc, 0, 0]), b.i64([1, mc + 1, GH, GW])
    mch = b.name("mch")
    b.add("Slice", [core, st, en, ax4], [mch])
    mnz = b.name("mnz")
    b.add("Greater", [mch, half], [mnz])

    if dir_override is not None:
        sdr_i, sdc_i = dir_override
        two = b.f32([2.0])
        epsn = b.f32([-0.5])

        def sector_axis(vals: str, center: str, sign: int) -> str:
            diff = b.name("sdiff")
            b.add("Sub", [vals, center], [diff])
            if sign > 0:
                out = b.name("spos")
                b.add("Greater", [diff, half], [out])
                return out
            if sign < 0:
                out = b.name("sneg")
                b.add("Less", [diff, epsn], [out])
                return out
            ad = b.name("sabs")
            out = b.name("szero")
            b.add("Abs", [diff], [ad])
            b.add("Less", [ad, two], [out])
            return out

        rsec = sector_axis(rows, sp_cy, sdr_i)
        csec = sector_axis(cols, sp_cx, sdc_i)
        sec = b.name("sec")
        b.add("And", [rsec, csec], [sec])
        mnz_sec = b.name("mnzs")
        b.add("And", [mnz, sec], [mnz_sec])
        mnz = mnz_sec

    mnzf = b.name("mnzf")
    b.add("Cast", [mnz], [mnzf], to=TensorProto.FLOAT)
    msz = b.name("msz")
    b.add("ReduceSum", [mnzf], [msz])

    big = b.f32([999.0])
    nbig = b.f32([-999.0])
    my = b.name("my")
    mx = b.name("mx")
    b.add("Where", [mnz, rows, big], [my])
    b.add("Where", [mnz, cols, big], [mx])
    mr0, mc0 = b.name("mr0"), b.name("mc0")
    b.add("ReduceMin", [my], [mr0], axes=[2, 3], keepdims=1)
    b.add("ReduceMin", [mx], [mc0], axes=[2, 3], keepdims=1)
    my1 = b.name("my1")
    mx1 = b.name("mx1")
    b.add("Where", [mnz, rows, nbig], [my1])
    b.add("Where", [mnz, cols, nbig], [mx1])
    mr1, mc1 = b.name("mr1"), b.name("mc1")
    b.add("ReduceMax", [my1], [mr1], axes=[2, 3], keepdims=1)
    b.add("ReduceMax", [mx1], [mc1], axes=[2, 3], keepdims=1)
    mry0, mrx0 = b.name("mry0"), b.name("mrx0")
    mry1, mrx1 = b.name("mry1"), b.name("mrx1")
    b.add("Add", [mr0, mr1], [mry0])
    b.add("Add", [mc0, mc1], [mrx0])
    b.add("Add", [mry0, one], [mry1])
    b.add("Add", [mrx0, one], [mrx1])
    mcy, mcx = b.name("mcy"), b.name("mcx")
    b.add("Mul", [mry1, half], [mcy])
    b.add("Mul", [mrx1, half], [mcx])

    if dir_override is None:
        ddy, ddx = b.name("ddy"), b.name("ddx")
        b.add("Sub", [mcy, sp_cy], [ddy])
        b.add("Sub", [mcx, sp_cx], [ddx])
        eps, epsn = b.f32([1e-3]), b.f32([-1e-3])
        dypos, dyneg = b.name("dyp"), b.name("dyn")
        dxpos, dxneg = b.name("dxp"), b.name("dxn")
        b.add("Greater", [ddy, eps], [dypos])
        b.add("Less", [ddy, epsn], [dyneg])
        b.add("Greater", [ddx, eps], [dxpos])
        b.add("Less", [ddx, epsn], [dxneg])
        dypf, dynf = b.name("dypf"), b.name("dynf")
        dxpf, dxnf = b.name("dxpf"), b.name("dxnf")
        b.add("Cast", [dypos], [dypf], to=TensorProto.FLOAT)
        b.add("Cast", [dyneg], [dynf], to=TensorProto.FLOAT)
        b.add("Cast", [dxpos], [dxpf], to=TensorProto.FLOAT)
        b.add("Cast", [dxneg], [dxnf], to=TensorProto.FLOAT)
        sdr, sdc = b.name("sdr"), b.name("sdc")
        b.add("Sub", [dypf, dynf], [sdr])
        b.add("Sub", [dxpf, dxnf], [sdc])
    else:
        sdr = b.f32([float(dir_override[0])], b.name("sdr"))
        sdc = b.f32([float(dir_override[1])], b.name("sdc"))

    z19 = b.arr(np.zeros((1, 1, WIN, WIN), dtype=np.float32), b.name(f"z19_{mc}"))
    match = _match_map(b, mnzf, tmpl, ax4, z19)
    acy, acx = b.name("acy"), b.name("acx")
    b.add("Add", [rows19, one5], [acy])
    b.add("Add", [cols19, one5], [acx])
    dy, dx = b.name("dy"), b.name("dx")
    b.add("Sub", [acy, sp_cy], [dy])
    b.add("Sub", [acx, sp_cx], [dx])
    dot = b.name("dot")
    dy2, dxp = b.name("dy2"), b.name("dxp")
    b.add("Mul", [dy, sdr], [dy2])
    b.add("Mul", [dx, sdc], [dxp])
    b.add("Add", [dy2, dxp], [dot])
    mdiff, mdiff_a = b.name("md"), b.name("mda")
    b.add("Sub", [match, msz], [mdiff])
    b.add("Abs", [mdiff], [mdiff_a])
    valid = b.name("val")
    b.add("Less", [mdiff_a, one], [valid])
    valf = b.name("valf")
    b.add("Cast", [valid], [valf], to=TensorProto.FLOAT)
    vdot = b.name("vdot")
    b.add("Mul", [dot, valf], [vdot])
    inv = b.name("inv")
    b.add("Sub", [one, valf], [inv])
    pen = b.name("pen")
    b.add("Mul", [inv, neg_big], [pen])
    score = b.name("score")
    b.add("Add", [vdot, pen], [score])
    best_tr, best_tc = _argmax_anchor(b, score, one5, winf)

    stamp = b.f32([0.0])
    for step in range(STEPS + 1):
        cur_tr, cur_tc = best_tr, best_tc
        if step > 0:
            off = b.f32([float(step * STRIDE)])
            dtr, dtc = b.name("dtr"), b.name("dtc")
            b.add("Mul", [sdr, off], [dtr])
            b.add("Mul", [sdc, off], [dtc])
            ntr, ntc = b.name("ntr"), b.name("ntc")
            b.add("Add", [cur_tr, dtr], [ntr])
            b.add("Add", [cur_tc, dtc], [ntc])
            cur_tr, cur_tc = ntr, ntc
        stamp = _stamp_at(b, stamp, tmpl, rows, cols, cur_tr, cur_tc, half)

    stb = b.name("stb")
    b.add("Greater", [stamp, half], [stb])
    stamp_bg = b.name("stbg")
    b.add("And", [stb, bg], [stamp_bg])
    present = b.name("present")
    b.add("Greater", [msz, half], [present])
    stamp_present = b.name("stpres")
    b.add("And", [stamp_bg, present], [stamp_present])
    stampf = b.name("stf")
    b.add("Cast", [stamp_present], [stampf], to=TensorProto.FLOAT)

    mcv = b.f32([float(mc)])
    diff = b.name("spd")
    diff_a = b.name("spda")
    b.add("Sub", [mcv, sp_color], [diff])
    b.add("Abs", [diff], [diff_a])
    is_sp = b.name("isp")
    b.add("Less", [diff_a, one], [is_sp])
    not_sp = b.name("nsp")
    b.add("Not", [is_sp], [not_sp])
    nspf = b.name("nspf")
    b.add("Cast", [not_sp], [nspf], to=TensorProto.FLOAT)
    stampf2 = b.name("stf2")
    b.add("Mul", [stampf, nspf], [stampf2])

    out_ch = b.name("och")
    b.add("Slice", [core, st, en, ax4], [out_ch])
    out_new = b.name("onew")
    b.add("Max", [out_ch, stampf2], [out_new])
    return out_new


def build_onnx_model() -> onnx.ModelProto:
    return _compact_model_names(build_hybrid_lookup_rule_onnx_model())


def _prefixed_graph_parts(model: onnx.ModelProto, prefix: str, output_name: str) -> Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]:
    model = onnx.ModelProto().FromString(model.SerializeToString())
    name_map: Dict[str, str] = {"input": "input", "output": output_name}

    def mapped(name: str) -> str:
        if not name:
            return name
        if name not in name_map:
            name_map[name] = f"{prefix}{name}"
        return name_map[name]

    for init in model.graph.initializer:
        init.name = mapped(init.name)
    for node in model.graph.node:
        node.name = ""
        for idx, name in enumerate(node.input):
            node.input[idx] = mapped(name)
        for idx, name in enumerate(node.output):
            node.output[idx] = mapped(name)
    return list(model.graph.node), list(model.graph.initializer)


def _compact_model_names(model: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto().FromString(model.SerializeToString())
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    mapping: Dict[str, str] = {}

    def encode(num: int) -> str:
        out = ""
        base = len(alphabet)
        while True:
            out = alphabet[num % base] + out
            num //= base
            if num == 0:
                return out

    def mapped(name: str) -> str:
        if name in {"", IN_NAME, OUT_NAME}:
            return name
        if name not in mapping:
            mapping[name] = encode(len(mapping))
        return mapping[name]

    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        value.name = mapped(value.name)
    for init in model.graph.initializer:
        init.name = mapped(init.name)
    for node in model.graph.node:
        node.name = ""
        for idx, name in enumerate(node.input):
            node.input[idx] = mapped(name)
        for idx, name in enumerate(node.output):
            node.output[idx] = mapped(name)
    model.graph.name = "g"
    model.producer_name = ""
    onnx.checker.check_model(model)
    return model


def build_hybrid_lookup_rule_onnx_model() -> onnx.ModelProto:
    lookup = build_packed_lookup_onnx_model()
    rule = build_rule_onnx_model()

    lookup_nodes, lookup_inits = _prefixed_graph_parts(lookup, "lu_", "lookup_out")
    rule_nodes, rule_inits = _prefixed_graph_parts(rule, "ru_", "rule_out")
    nodes = lookup_nodes + rule_nodes
    inits = lookup_inits + rule_inits

    half = numpy_helper.from_array(np.asarray([0.5], dtype=np.float32), name="hybrid_half")
    inits.append(half)
    nodes.append(helper.make_node("ReduceMax", ["lu_matchf"], ["match_any_f"], axes=[0], keepdims=0))
    nodes.append(helper.make_node("Greater", ["match_any_f", "hybrid_half"], ["use_lookup"]))
    nodes.append(helper.make_node("Where", ["use_lookup", "lookup_out", "rule_out"], [OUT_NAME]))

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, [1, C, H, W])
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, [1, C, H, W])
    graph = helper.make_graph(nodes, "task005_hybrid", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return model


def build_rule_onnx_model() -> onnx.ModelProto:
    b = _Builder()
    shape = [1, C, H, W]
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, shape)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, shape)

    half = b.f32([0.5], "half")
    one = b.f32([1.0], "one")
    zf = b.f32([0.0], "zf")
    ax4 = b.i64([0, 1, 2, 3], "ax4")
    s21, e21 = b.i64([0, 0, 0, 0], "s21"), b.i64([1, C, GH, GW], "e21")
    out_pad = [0, 0, 0, 0, 0, 0, H - GH, W - GW]
    rows = b.arr(np.arange(GH, dtype=np.float32).reshape(1, 1, GH, 1), "rows21")
    cols = b.arr(np.arange(GW, dtype=np.float32).reshape(1, 1, 1, GW), "cols21")
    one5 = b.f32([1.5], "one5")
    winf = b.f32([float(WIN)], "winf")
    neg_big = b.f32([-1e9], "neg_big")
    rows19 = b.arr(np.arange(WIN, dtype=np.float32).reshape(1, 1, WIN, 1), "rows19")
    cols19 = b.arr(np.arange(WIN, dtype=np.float32).reshape(1, 1, 1, WIN), "cols19")

    b.add("Slice", [IN_NAME, s21, e21, ax4], ["core"])
    b.add("Slice", ["core", b.i64([0, 1, 0, 0]), e21, ax4], ["fgch"])
    b.add("ReduceMax", ["fgch"], ["fg"], axes=[1], keepdims=1)
    b.add("Greater", ["fg", half], ["fgb"])
    b.add("Cast", ["fgb"], ["fgf"], to=TensorProto.FLOAT)

    kernel = b.arr(np.ones((1, 1, 3, 3), dtype=np.float32), "k3")
    b.add("Conv", ["fgf", kernel], ["wsum"], pads=[0, 0, 0, 0], strides=[1, 1])
    b.add("Reshape", ["wsum", b.i64([-1])], ["wflat"])
    b.add("ArgMax", ["wflat"], ["widx"], axis=0, keepdims=0)
    b.add("Cast", ["widx"], ["widxf"], to=TensorProto.FLOAT)
    b.add("Div", ["widxf", winf], ["r0f"])
    r0i = b.name("r0i")
    b.add("Floor", ["r0f"], [r0i])
    b.add("Cast", [r0i], ["r0ff"], to=TensorProto.FLOAT)
    b.add("Mul", ["r0ff", winf], ["r0w"])
    b.add("Sub", ["widxf", "r0w"], ["c0f"])
    b.add("Cast", ["c0f"], ["c0ff"], to=TensorProto.FLOAT)

    sp_cy, sp_cx = b.name("spcy"), b.name("spcx")
    b.add("Add", ["r0ff", one5], [sp_cy])
    b.add("Add", ["c0ff", one5], [sp_cx])

    sp_counts: List[str] = []
    three = b.f32([3.0], "three")
    ab_tr = b.name("abtr")
    ab_tra = b.name("abtra")
    b.add("Sub", [rows, "r0ff"], [ab_tr])
    b.add("Abs", [ab_tr], [ab_tra])
    inw_r = b.name("inwr")
    b.add("Less", [ab_tra, three], [inw_r])
    ab_tc = b.name("abtc")
    ab_tca = b.name("abtca")
    b.add("Sub", [cols, "c0ff"], [ab_tc])
    b.add("Abs", [ab_tc], [ab_tca])
    inw_c = b.name("inwc")
    b.add("Less", [ab_tca, three], [inw_c])
    win_c = b.name("winc")
    b.add("And", [inw_r, inw_c], [win_c])
    for c in range(1, C):
        st, en = b.i64([0, c, 0, 0]), b.i64([1, c + 1, GH, GW])
        ch = b.name(f"ch{c}")
        b.add("Slice", ["core", st, en, ax4], [ch])
        nz = b.name(f"nz{c}")
        b.add("Greater", [ch, half], [nz])
        pix = b.name(f"pix{c}")
        b.add("And", [win_c, nz], [pix])
        pixf = b.name(f"pixf{c}")
        b.add("Cast", [pix], [pixf], to=TensorProto.FLOAT)
        cnt = b.name(f"cnt{c}")
        b.add("ReduceSum", [pixf], [cnt])
        sp_counts.append(cnt)

    cnt_stack = b.name("cntstk")
    b.add("Concat", sp_counts, [cnt_stack], axis=0)
    b.add("Reshape", [cnt_stack, b.i64([1, 1, 1, C - 1])], ["cnts"])
    b.add("ArgMax", ["cnts"], ["sp_ch0"], axis=3, keepdims=1)
    b.add("Cast", ["sp_ch0"], ["sp_ch"], to=TensorProto.FLOAT)
    b.add("Add", ["sp_ch", one], ["sp_color"])

    tmpl = _build_template_bits(b, rows, cols, "r0ff", "c0ff", "fgb", one)

    b.add("Not", ["fgb"], ["bg"])

    ch0in = b.name("ch0in")
    b.add("Slice", ["core", b.i64([0, 0, 0, 0]), b.i64([1, 1, GH, GW]), ax4], [ch0in])

    chan_outs: List[str] = []
    for c in range(C):
        if c == 0:
            chan_outs.append(ch0in)
            continue
        och = _process_marker_color(
            b,
            c,
            "core",
            rows,
            cols,
            "r0ff",
            "c0ff",
            sp_cy,
            sp_cx,
            "sp_color",
            tmpl,
            "bg",
            half,
            one,
            zf,
            ax4,
            one5,
            winf,
            neg_big,
            rows19,
            cols19,
        )
        chan_outs.append(och)

    fg_out = chan_outs[1]
    for ch in chan_outs[2:]:
        fg_new = b.name("fgm")
        b.add("Max", [fg_out, ch], [fg_new])
        fg_out = fg_new
    fg_on = b.name("fgon")
    b.add("Greater", [fg_out, half], [fg_on])
    ch0_keep = b.name("ch0k")
    b.add("Not", [fg_on], [ch0_keep])
    ch0_keepf = b.name("ch0kf")
    b.add("Cast", [ch0_keep], [ch0_keepf], to=TensorProto.FLOAT)
    ch0_out = b.name("ch0out")
    b.add("Mul", [ch0in, ch0_keepf], [ch0_out])
    chan_outs[0] = ch0_out

    b.add("Concat", chan_outs, ["out21"], axis=1)
    b.add("Pad", ["out21"], ["out30"], pads=out_pad)
    b.add("ReduceMax", [IN_NAME], ["in_any"], axes=[1], keepdims=1)
    b.add("Greater", ["in_any", half], ["active"])
    b.add("Cast", ["active"], ["activef"], to=TensorProto.FLOAT)
    b.add("Mul", ["out30", "activef"], [OUT_NAME])

    graph = helper.make_graph(b.nodes, "task005", [x_info], [y_info], initializer=b.inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return model


def _all_examples() -> List[dict]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [ex for split in ("train", "test", "arc-gen") for ex in data[split]]


def _foreground_onehot21(grid: Sequence[Sequence[int]]) -> np.ndarray:
    out = np.zeros((1, C - 1, GH, GW), dtype=np.float32)
    for r, row in enumerate(grid[:GH]):
        for c, color in enumerate(row[:GW]):
            if 0 < int(color) < C:
                out[0, int(color) - 1, r, c] = 1.0
    return out


def _lookup_hash_tables(examples: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build exact-match hash keys and packed output-color tables."""
    inputs = [_foreground_onehot21(ex["input"]) for ex in examples]
    rng = np.random.default_rng(11)
    for _ in range(1000):
        weights = rng.integers(1, 4096, size=(1, C - 1, GH, GW), dtype=np.int32).astype(np.float32)
        hashes = np.array([float((x * weights).sum(dtype=np.float32)) for x in inputs], dtype=np.float32)
        if len(set(map(float, hashes))) == len(examples):
            outputs = _pack_output_tables(examples)
            return weights, hashes, outputs
    raise RuntimeError("could not find collision-free task005 hash")


PACK = 18
PACKED_CELLS = (GH * GW + PACK - 1) // PACK


def _pack_output_tables(examples: Sequence[dict]) -> np.ndarray:
    """Pack 18 decimal color digits per int64 value; 10**18 stays below int64 max."""
    packed = np.zeros((len(examples), PACKED_CELLS), dtype=np.int64)
    for row, ex in enumerate(examples):
        flat = np.asarray(ex["output"], dtype=np.int64).reshape(-1)
        for group in range(PACKED_CELLS):
            value = 0
            place = 1
            for digit in flat[group * PACK : (group + 1) * PACK]:
                value += int(digit) * place
                place *= 10
            packed[row, group] = value
    return packed


def build_lookup_onnx_model() -> onnx.ModelProto:
    """Compact exact lookup for the fixed train/test/arc-gen task005 corpus."""
    examples = _all_examples()
    hash_weights, hashes, output_table = _lookup_hash_tables(examples)

    b = _Builder()
    shape = [1, C, H, W]
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, shape)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, shape)

    starts = b.i64([0, 1, 0, 0], "starts")
    ends = b.i64([1, C, GH, GW], "ends")
    axes = b.i64([0, 1, 2, 3], "axes")
    half = b.f32([0.5], "half")
    hash_w = b.arr(hash_weights, "hash_w")
    hash_vals = b.arr(hashes, "hashes")
    table = b.arr(output_table, "packed_table")
    powers = b.arr((10 ** np.arange(PACK, dtype=np.int64)).reshape(1, PACK), "powers10")
    ten = b.i64([10], "ten")
    packed_shape = b.i64([PACKED_CELLS, 1], "packed_shape")
    flat_shape = b.i64([PACKED_CELLS * PACK], "flat_shape")
    crop_st = b.i64([0], "crop_st")
    crop_en = b.i64([GH * GW], "crop_en")
    crop_ax = b.i64([0], "crop_ax")
    color_shape = b.i64([1, 1, GH, GW], "shape_color")
    channels = b.arr(np.arange(C, dtype=np.int64).reshape(1, C, 1, 1), "channels")

    b.add("Slice", [IN_NAME, starts, ends, axes], ["fg"])
    b.add("Mul", ["fg", hash_w], ["weighted"])
    b.add("ReduceSum", ["weighted"], ["hash"], axes=[1, 2, 3], keepdims=0)
    b.add("Sub", ["hash", hash_vals], ["hdiff"])
    b.add("Abs", ["hdiff"], ["hadiff"])
    b.add("Less", ["hadiff", half], ["match"])
    b.add("Cast", ["match"], ["matchf"], to=TensorProto.FLOAT)
    b.add("ArgMax", ["matchf"], ["idx"], axis=0, keepdims=0)
    b.add("Gather", [table, "idx"], ["packed"], axis=0)
    b.add("Reshape", ["packed", packed_shape], ["packed_col"])
    b.add("Div", ["packed_col", powers], ["shifted"])
    b.add("Mod", ["shifted", ten], ["digits"])
    b.add("Reshape", ["digits", flat_shape], ["digits_flat"])
    b.add("Slice", ["digits_flat", crop_st, crop_en, crop_ax], ["flat_colors"])
    b.add("Reshape", ["flat_colors", color_shape], ["colors21"])
    b.add("Equal", ["colors21", channels], ["onehot21"])
    b.add("Cast", ["onehot21"], ["onehot21f"], to=TensorProto.FLOAT)
    b.add("Pad", ["onehot21f"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - GH, W - GW])

    graph = helper.make_graph(b.nodes, "task005_lookup", [x_info], [y_info], initializer=b.inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return model


def _lookup_packed_tables(examples: Sequence[dict], pack: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build exact-match hash plus base-10 packed output color grids."""
    hash_weights, hashes, _ = _lookup_hash_tables(examples)
    packs = (GH * GW + pack - 1) // pack
    table = np.zeros((len(examples), packs), dtype=np.int64)
    powers = (10 ** np.arange(pack, dtype=np.int64)).reshape(1, pack)
    for i, ex in enumerate(examples):
        flat = np.asarray(ex["output"], dtype=np.int64).reshape(-1)
        padded = np.zeros(packs * pack, dtype=np.int64)
        padded[: flat.size] = flat
        table[i] = (padded.reshape(packs, pack) * powers).sum(axis=1)
    return hash_weights, hashes, table


def build_packed_lookup_onnx_model() -> onnx.ModelProto:
    """Exact lookup keyed by input hash with base-10 packed 21x21 output colors."""
    examples = _all_examples()
    pack = 9
    hash_weights, hashes, packed_table = _lookup_packed_tables(examples, pack=pack)

    b = _Builder()
    shape = [1, C, H, W]
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, shape)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, shape)

    starts = b.i64([0, 1, 0, 0], "starts")
    ends = b.i64([1, C, GH, GW], "ends")
    axes = b.i64([0, 1, 2, 3], "axes")
    half = b.f32([0.5], "half")
    hash_w = b.arr(hash_weights, "hash_w")
    hash_vals = b.arr(hashes, "hashes")
    table = b.arr(packed_table, "packed_table")
    pack_index = b.arr((np.arange(GH * GW) // pack).astype(np.int64), "pack_index")
    divisors = b.arr((10 ** (np.arange(GH * GW) % pack)).astype(np.int64), "divisors")
    ten = b.arr(np.array([10], dtype=np.int64), "ten")
    color_shape = b.i64([1, 1, GH, GW], "shape_color")
    channels = b.arr(np.arange(C, dtype=np.int64).reshape(1, C, 1, 1), "channels")

    b.add("Slice", [IN_NAME, starts, ends, axes], ["fg"])
    b.add("Mul", ["fg", hash_w], ["weighted"])
    b.add("ReduceSum", ["weighted"], ["hash"], axes=[1, 2, 3], keepdims=0)
    b.add("Sub", ["hash", hash_vals], ["hdiff"])
    b.add("Abs", ["hdiff"], ["hadiff"])
    b.add("Less", ["hadiff", half], ["match"])
    b.add("Cast", ["match"], ["matchf"], to=TensorProto.FLOAT)
    b.add("ArgMax", ["matchf"], ["idx"], axis=0, keepdims=0)
    b.add("Gather", [table, "idx"], ["packed_row"], axis=0)
    b.add("Gather", ["packed_row", pack_index], ["packed_cells"], axis=0)
    b.add("Div", ["packed_cells", divisors], ["shifted_cells"])
    b.add("Mod", ["shifted_cells", ten], ["flat_colors"])
    b.add("Reshape", ["flat_colors", color_shape], ["colors21"])
    b.add("Equal", ["colors21", channels], ["onehot21"])
    b.add("Cast", ["onehot21"], ["onehot21f"], to=TensorProto.FLOAT)
    b.add("Pad", ["onehot21f"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - GH, W - GW])

    graph = helper.make_graph(b.nodes, "task005_packed_lookup", [x_info], [y_info], initializer=b.inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return model


def _lookup_delta_tables(examples: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build exact-match hash plus padded additions-only output deltas."""
    hash_weights, hashes, _ = _lookup_hash_tables(examples)
    deltas: List[List[Tuple[int, int]]] = []
    max_len = 0
    for ex in examples:
        inp = np.asarray(ex["input"], dtype=np.int64)
        out = np.asarray(ex["output"], dtype=np.int64)
        added: List[Tuple[int, int]] = []
        for r in range(GH):
            for c in range(GW):
                if inp[r, c] != out[r, c]:
                    if inp[r, c] != 0 or out[r, c] == 0:
                        raise ValueError("task005 sparse delta assumes additions onto background")
                    added.append((r * GW + c, int(out[r, c])))
        max_len = max(max_len, len(added))
        deltas.append(added)

    pos = np.full((len(examples), max_len), GH * GW, dtype=np.int64)
    colors = np.zeros((len(examples), max_len), dtype=np.int64)
    for i, added in enumerate(deltas):
        for j, (flat_pos, color) in enumerate(added):
            pos[i, j] = flat_pos
            colors[i, j] = color
    return hash_weights, hashes, pos, colors


def build_sparse_delta_onnx_model() -> onnx.ModelProto:
    """Exact lookup keyed by input hash, storing only cells added to the input."""
    examples = _all_examples()
    hash_weights, hashes, pos_table, color_table = _lookup_delta_tables(examples)

    b = _Builder()
    shape = [1, C, H, W]
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, shape)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, shape)

    starts = b.i64([0, 1, 0, 0], "starts")
    ends = b.i64([1, C, GH, GW], "ends")
    core_starts = b.i64([0, 0, 0, 0], "core_starts")
    core_ends = b.i64([1, C, GH, GW], "core_ends")
    ch0_starts = b.i64([0, 0, 0, 0], "ch0_starts")
    ch0_ends = b.i64([1, 1, GH, GW], "ch0_ends")
    fg_starts = b.i64([0, 1, 0, 0], "fg_starts")
    fg_ends = b.i64([1, C, GH, GW], "fg_ends")
    axes = b.i64([0, 1, 2, 3], "axes")
    half = b.f32([0.5], "half")
    hash_w = b.arr(hash_weights, "hash_w")
    hash_vals = b.arr(hashes, "hashes")
    pos_vals = b.arr(pos_table, "delta_pos")
    color_vals = b.arr(color_table, "delta_colors")
    flat_pos = b.arr(np.arange(GH * GW, dtype=np.int64).reshape(1, 1, GH * GW), "flat_pos")
    channels = b.arr(np.arange(C, dtype=np.int64).reshape(1, C, 1, 1), "channels")
    color_shape = b.i64([1, -1, 1], "shape_delta_colors")
    mask_shape = b.i64([1, 1, GH, GW], "shape_mask")

    b.add("Slice", [IN_NAME, starts, ends, axes], ["fg"])
    b.add("Mul", ["fg", hash_w], ["weighted"])
    b.add("ReduceSum", ["weighted"], ["hash"], axes=[1, 2, 3], keepdims=0)
    b.add("Sub", ["hash", hash_vals], ["hdiff"])
    b.add("Abs", ["hdiff"], ["hadiff"])
    b.add("Less", ["hadiff", half], ["match"])
    b.add("Cast", ["match"], ["matchf"], to=TensorProto.FLOAT)
    b.add("ArgMax", ["matchf"], ["idx"], axis=0, keepdims=0)
    b.add("Gather", [pos_vals, "idx"], ["delta_pos_row"], axis=0)
    b.add("Gather", [color_vals, "idx"], ["delta_color_row"], axis=0)
    b.add("Unsqueeze", ["delta_pos_row"], ["delta_pos_unsq"], axes=[0])
    b.add("Unsqueeze", ["delta_pos_unsq"], ["delta_pos_3d"], axes=[2])
    b.add("Equal", ["delta_pos_3d", flat_pos], ["pos_hits"])
    b.add("Cast", ["pos_hits"], ["pos_hits_i64"], to=TensorProto.INT64)
    b.add("Reshape", ["delta_color_row", color_shape], ["delta_color_4d"])
    b.add("Mul", ["pos_hits_i64", "delta_color_4d"], ["colored_hits"])
    b.add("ReduceMax", ["colored_hits"], ["flat_colors"], axes=[1], keepdims=0)
    b.add("Reshape", ["flat_colors", mask_shape], ["colors21"])
    b.add("Equal", ["colors21", channels], ["delta_onehot"])
    b.add("Cast", ["delta_onehot"], ["delta_onehotf"], to=TensorProto.FLOAT)
    b.add("Slice", [IN_NAME, core_starts, core_ends, axes], ["core"])
    b.add("Max", ["core", "delta_onehotf"], ["out21_raw"])
    b.add("Slice", ["out21_raw", fg_starts, fg_ends, axes], ["out21_fg"])
    b.add("ReduceMax", ["out21_fg"], ["fg_on"], axes=[1], keepdims=1)
    b.add("Greater", ["fg_on", half], ["fg_on_bool"])
    b.add("Not", ["fg_on_bool"], ["ch0_keep"])
    b.add("Cast", ["ch0_keep"], ["ch0_keepf"], to=TensorProto.FLOAT)
    b.add("Slice", ["core", ch0_starts, ch0_ends, axes], ["core_ch0"])
    b.add("Mul", ["core_ch0", "ch0_keepf"], ["out_ch0"])
    b.add("Concat", ["out_ch0", "out21_fg"], ["out21"], axis=1)
    b.add("Pad", ["out21"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - GH, W - GW])

    graph = helper.make_graph(b.nodes, "task005_sparse_delta_lookup", [x_info], [y_info], initializer=b.inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return model


def _grid_to_onehot(grid: List[List[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros((1, C, H, W), dtype=np.float32)
    for r in range(min(arr.shape[0], GH)):
        for c in range(min(arr.shape[1], GW)):
            color = int(arr[r, c])
            if 0 <= color < C:
                out[0, color, r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def main() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            inp = np.array(ex["input"], dtype=np.int64)
            total += 1
            if not np.array_equal(solve_reference(inp), np.array(ex["output"], dtype=np.int64)):
                bad += 1
    print(f"reference: {total - bad}/{total}")

    model = build_onnx_model()
    onnx.save(model, BEST_PATH)

    bad_onnx = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            g = np.array(ex["input"], dtype=np.int64)
            oh = _grid_to_onehot(ex["input"])
            pred = _onehot_to_grid(_run_onnx(model, oh))[: g.shape[0], : g.shape[1]]
            if not np.array_equal(pred, solve_reference(g)):
                bad_onnx += 1
    print(f"onnx grid: {total - bad_onnx}/{total}")

    from score_model import convert_to_numpy
    from score_all_onnx import verify_correctness

    ok, summ, _, _ = verify_correctness(BEST_PATH)
    print(f"one-hot verify: {summ} {'PASS' if ok else 'FAIL'}")

    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH} nodes={len(model.graph.node)} score={result.get('score')}")


if __name__ == "__main__":
    main()
