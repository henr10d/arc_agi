"""Build and validate compact ONNX for NeuroGolf task044.

Task rule: on a 10x10 grid, gray (5) forms rectangular frames with interior holes
(zeros inside the gray bounding box). Each non-gray color that appears as exactly
one connected component is a movable piece; copy it into the hole with the same
binary shape, erase the source, and leave gray plus unmatched colors unchanged.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "task044.json"
OUT_PATH = ROOT / "task044.onnx"
ALL_OUT_PATH = ROOT / "all" / "task044.onnx"

C = 10
N = 10
H = W = 30
SHAPE = [1, C, H, W]
GRAY = 5
GRAY_DILATE_STEPS = 9
COLOR_DILATE_STEPS = {1: 4, 2: 3, 3: 4, 4: 4, 6: 3, 7: 3, 8: 4, 9: 4}
MIN_GRAY_FRAME = 8
IR_VERSION = 10
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


@dataclass(frozen=True)
class Variant:
    name: str
    build: Callable[[], onnx.ModelProto]


def find_components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: set[tuple[int, int]] = set()
            while q:
                cr, cc = q.popleft()
                cells.add((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def normalize_shape(cells: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    if not cells:
        return frozenset()
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, c0 = min(rs), min(cs)
    return frozenset((r - r0, c - c0) for r, c in cells)


def bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def solve_grid(grid: list[list[int]]) -> list[list[int]]:
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    holes: list[tuple[frozenset[tuple[int, int]], set[tuple[int, int]]]] = []
    for comp in find_components(g == GRAY):
        r0, c0, r1, c1 = bbox(comp)
        hole_cells = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if g[r, c] == 0
        }
        if hole_cells:
            holes.append((normalize_shape(hole_cells), hole_cells))

    free_pieces: list[tuple[int, frozenset[tuple[int, int]], set[tuple[int, int]]]] = []
    for color in range(10):
        if color in (0, GRAY):
            continue
        comps = find_components(g == color)
        if len(comps) == 1:
            free_pieces.append((color, normalize_shape(comps[0]), comps[0]))

    for color, shape, cells in free_pieces:
        for hi, (hshape, hcells) in enumerate(holes):
            if shape == hshape:
                for r, c in cells:
                    out[r, c] = 0
                for r, c in hcells:
                    out[r, c] = color
                break
    return out.tolist()


def grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def onehot_to_grid(tensor: np.ndarray) -> list[list[int]]:
    return tensor[0, :, :N, :N].argmax(axis=0).astype(int).tolist()


class GraphBuilder:
    def __init__(self, opset: int) -> None:
        self.opset = opset
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self._counter = 0
        self.rows: str | None = None
        self.cols: str | None = None
        self.ten: str | None = None
        self.nine: str | None = None
        self.zero_i32: str | None = None
        self.half: str | None = None
        self.revr: str | None = None
        self.revc: str | None = None
        self.flat_shape: str | None = None
        self.conv3: str | None = None
        self.big_i: str | None = None
        self.score_grid: str | None = None

    def _name(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def _const(self, arr: np.ndarray, name: str | None = None) -> str:
        name = name or self._name("c")
        self.inits.append(numpy_helper.from_array(arr, name=name))
        return name

    def _scalar_i64(self, value: int, name: str | None = None) -> str:
        return self._const(np.asarray([value], dtype=np.int64), name)

    def _node(self, op: str, inputs: list[str], outputs: list[str], **attrs: Any) -> None:
        self.nodes.append(helper.make_node(op, inputs, outputs, **attrs))

    def _init_shared(self) -> None:
        if self.rows is not None:
            return
        self.rows = self._const(np.arange(N, dtype=np.int64).reshape(1, 1, N, 1), "rows")
        self.cols = self._const(np.arange(N, dtype=np.int64).reshape(1, 1, 1, N), "cols")
        self.ten = self._scalar_i64(10, "ten")
        self.nine = self._scalar_i64(9, "nine")
        self.zero_i32 = self._const(np.asarray([0], dtype=np.int32), "zero_i32")
        self.big_i = self._scalar_i64(99, "big_i")
        self.half = self._const(np.asarray([0.5], dtype=np.float32), "half")
        self.revr = self._const(np.arange(N - 1, -1, -1, dtype=np.int64), "revr")
        self.revc = self._const(np.arange(N - 1, -1, -1, dtype=np.int64), "revc")
        self.flat_shape = self._const(np.asarray([N * N], dtype=np.int64), "flat_shape")
        self.conv3 = self._const(np.ones((1, 1, 3, 3), dtype=np.float32), "conv3")
        self.score_grid = self._const(np.arange(N * N, dtype=np.int64).reshape(1, 1, N, N), "score_grid")

    def _shift_bool(self, inp: str, off_r: int, off_c: int, prefix: str) -> str:
        out = self._name(prefix)
        rs, re = max(0, off_r), min(N, N + off_r)
        cs, ce = max(0, off_c), min(N, N + off_c)
        pb_r, pa_r = max(0, -off_r), max(0, off_r)
        pb_c, pa_c = max(0, -off_c), max(0, off_c)
        sliced = self._name(f"{prefix}_s")
        padded_f = self._name(f"{prefix}_pf")
        if self.opset >= 10:
            starts = self._const(np.asarray([0, 0, rs, cs], dtype=np.int64), f"{prefix}_st")
            ends = self._const(np.asarray([1, 1, re, ce], dtype=np.int64), f"{prefix}_en")
            axes = self._const(np.asarray([0, 1, 2, 3], dtype=np.int64), f"{prefix}_ax")
            self._node("Slice", [inp, starts, ends, axes], [sliced])
        else:
            self._node(
                "Slice",
                [inp],
                [sliced],
                axes=[0, 1, 2, 3],
                starts=[0, 0, rs, cs],
                ends=[1, 1, re, ce],
            )
        self._node("Cast", [sliced], [f"{prefix}_sf"], to=TensorProto.FLOAT)
        self._node(
            "Pad",
            [f"{prefix}_sf"],
            [padded_f],
            mode="constant",
            pads=[0, 0, pb_r, pb_c, 0, 0, pa_r, pa_c],
        )
        assert self.half is not None
        self._node("Greater", [padded_f, self.half], [out])
        return out

    def _top_left(self, mask: str, prefix: str) -> str:
        up = self._shift_bool(mask, -1, 0, f"{prefix}_up")
        left = self._shift_bool(mask, 0, -1, f"{prefix}_lf")
        not_up = self._name(f"{prefix}_nu")
        not_left = self._name(f"{prefix}_nl")
        tl = self._name(f"{prefix}_tl")
        self._node("Not", [up], [not_up])
        self._node("Not", [left], [not_left])
        self._node("And", [mask, not_up], [f"{prefix}_t0"])
        self._node("And", [f"{prefix}_t0", not_left], [tl])
        return tl

    def _min_seed(self, tl: str, prefix: str) -> str:
        assert self.score_grid and self.big_i
        min_s = self._name(f"{prefix}_mins")
        eq = self._name(f"{prefix}_eq")
        seed = self._name(f"{prefix}_seed")
        self._node("Where", [tl, self.score_grid, self.big_i], [f"{prefix}_masked"])
        self._node("ReduceMin", [f"{prefix}_masked"], [min_s], keepdims=0)
        self._node("Equal", [f"{prefix}_masked", min_s], [eq])
        self._node("And", [tl, eq], [seed])
        return seed

    def _dilate(self, seed: str, mask: str, prefix: str, steps: int) -> str:
        assert self.conv3 and self.half
        reach = seed
        for step in range(steps):
            reach_f = self._name(f"{prefix}_rf{step}")
            nbr_f = self._name(f"{prefix}_nf{step}")
            nbr_b = self._name(f"{prefix}_nb{step}")
            nbr_m = self._name(f"{prefix}_nm{step}")
            nxt = self._name(f"{prefix}_d{step}")
            self._node("Cast", [reach], [reach_f], to=TensorProto.FLOAT)
            self._node("Conv", [reach_f, self.conv3], [nbr_f], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
            self._node("Greater", [nbr_f, self.half], [nbr_b])
            self._node("And", [nbr_b, mask], [nbr_m])
            self._node("Or", [reach, nbr_m], [nxt])
            reach = nxt
        return reach

    def _bbox_scalars(self, mask: str, prefix: str) -> tuple[str, str, str, str, str, str]:
        assert self.revr and self.revc and self.nine
        rev_rows = self.revr
        rev_cols = self.revc
        nine = self.nine
        row_any = self._name(f"{prefix}_rowa")
        col_any = self._name(f"{prefix}_cola")
        top4 = self._name(f"{prefix}_top4")
        left4 = self._name(f"{prefix}_left4")
        bot4 = self._name(f"{prefix}_bot4")
        right4 = self._name(f"{prefix}_right4")
        top = self._name(f"{prefix}_top")
        left = self._name(f"{prefix}_left")
        bottom = self._name(f"{prefix}_bottom")
        right = self._name(f"{prefix}_right")
        height = self._name(f"{prefix}_h")
        width = self._name(f"{prefix}_w")
        one = self._scalar_i64(1, f"{prefix}_one")
        self._node("Cast", [mask], [f"{prefix}_mf"], to=TensorProto.FLOAT)
        self._node("ReduceMax", [f"{prefix}_mf"], [row_any], axes=[3], keepdims=1)
        self._node("ReduceMax", [f"{prefix}_mf"], [col_any], axes=[2], keepdims=1)
        self._node("ArgMax", [row_any], [top4], axis=2, keepdims=1)
        self._node("ArgMax", [col_any], [left4], axis=3, keepdims=1)
        self._node("Gather", [row_any, rev_rows], [f"{prefix}_rowrev"], axis=2)
        self._node("Gather", [col_any, rev_cols], [f"{prefix}_colrev"], axis=3)
        self._node("ArgMax", [f"{prefix}_rowrev"], [bot4], axis=2, keepdims=1)
        self._node("ArgMax", [f"{prefix}_colrev"], [right4], axis=3, keepdims=1)
        self._node("Cast", [top4], [top], to=TensorProto.INT64)
        self._node("Cast", [left4], [left], to=TensorProto.INT64)
        self._node("Cast", [bot4], [f"{prefix}_botr"], to=TensorProto.INT64)
        self._node("Cast", [right4], [f"{prefix}_ri"], to=TensorProto.INT64)
        self._node("Sub", [nine, f"{prefix}_botr"], [bottom])
        self._node("Sub", [nine, f"{prefix}_ri"], [right])
        h0 = self._name(f"{prefix}_h0")
        w0 = self._name(f"{prefix}_w0")
        self._node("Sub", [bottom, top], [h0])
        self._node("Add", [h0, one], [height])
        self._node("Sub", [right, left], [w0])
        self._node("Add", [w0, one], [width])
        return top, bottom, left, right, height, width

    def _append_hole(
        self,
        comp: str,
        bg: str,
        holes: list[tuple[str, str, str, str, str, str]],
        tag: str,
    ) -> None:
        gtop, gbot, gleft, gright, _, _ = self._bbox_scalars(comp, f"{tag}g")
        hole = self._hole_mask(bg, gtop, gbot, gleft, gright, tag)
        htop, hbot, hleft, hright, hh, hw = self._bbox_scalars(hole, tag)
        hole_norm = self._gather_grid(hole, htop, hleft, f"{tag}_norm")
        holes.append((hole, hole_norm, hleft, hh, hw, tag))

    def _hole_mask(self, bg: str, top: str, bottom: str, left: str, right: str, prefix: str) -> str:
        assert self.rows and self.cols
        rows = self.rows
        cols = self.cols
        ge_top = self._name(f"{prefix}_get")
        le_bot = self._name(f"{prefix}_leb")
        ge_left = self._name(f"{prefix}_gel")
        le_right = self._name(f"{prefix}_ler")
        inside = self._name(f"{prefix}_in")
        hole = self._name(f"{prefix}_hole")
        if self.opset >= 12:
            self._node("GreaterOrEqual", [rows, top], [ge_top])
            self._node("LessOrEqual", [rows, bottom], [le_bot])
            self._node("GreaterOrEqual", [cols, left], [ge_left])
            self._node("LessOrEqual", [cols, right], [le_right])
        else:
            self._node("Greater", [rows, top], [f"{prefix}_gt0"])
            self._node("Equal", [rows, top], [f"{prefix}_eqt"])
            self._node("Or", [f"{prefix}_gt0", f"{prefix}_eqt"], [ge_top])
            self._node("Less", [rows, bottom], [f"{prefix}_ltb"])
            self._node("Equal", [rows, bottom], [f"{prefix}_eqb"])
            self._node("Or", [f"{prefix}_ltb", f"{prefix}_eqb"], [le_bot])
            self._node("Greater", [cols, left], [f"{prefix}_gl0"])
            self._node("Equal", [cols, left], [f"{prefix}_eql"])
            self._node("Or", [f"{prefix}_gl0", f"{prefix}_eql"], [ge_left])
            self._node("Less", [cols, right], [f"{prefix}_ltr"])
            self._node("Equal", [cols, right], [f"{prefix}_eqr"])
            self._node("Or", [f"{prefix}_ltr", f"{prefix}_eqr"], [le_right])
        self._node("And", [ge_top, le_bot], [f"{prefix}_r0"])
        self._node("And", [ge_left, le_right], [f"{prefix}_c0"])
        self._node("And", [f"{prefix}_r0", f"{prefix}_c0"], [inside])
        self._node("And", [inside, bg], [hole])
        return hole

    def _gather_grid(self, data: str, top: str, left: str, prefix: str) -> str:
        assert self.rows and self.cols and self.ten and self.nine and self.flat_shape
        rows = self.rows
        cols = self.cols
        flat = self._name(f"{prefix}_flat")
        abs_r = self._name(f"{prefix}_ar")
        abs_c = self._name(f"{prefix}_ac")
        abs_rc = self._name(f"{prefix}_arc")
        abs_cc = self._name(f"{prefix}_acc")
        idx = self._name(f"{prefix}_idx")
        gathered = self._name(f"{prefix}_g")
        self._node("Reshape", [data, self.flat_shape], [flat])
        self._node("Add", [top, rows], [abs_r])
        self._node("Add", [left, cols], [abs_c])
        self._node("Greater", [abs_r, self.nine], [f"{prefix}_rgt"])
        self._node("Where", [f"{prefix}_rgt", self.nine, abs_r], [abs_rc])
        self._node("Greater", [abs_c, self.nine], [f"{prefix}_cgt"])
        self._node("Where", [f"{prefix}_cgt", self.nine, abs_c], [abs_cc])
        self._node("Mul", [abs_rc, self.ten], [f"{prefix}_r10"])
        self._node("Add", [f"{prefix}_r10", abs_cc], [idx])
        self._node("Gather", [flat, idx], [gathered])
        return gathered

    def _shape_match(
        self,
        hole_norm: str,
        hole_h: str,
        hole_w: str,
        piece_top: str,
        piece_left: str,
        piece_h: str,
        piece_w: str,
        piece_norm: str,
        prefix: str,
    ) -> str:
        assert self.rows and self.cols and self.zero_i32
        rows = self.rows
        cols = self.cols
        in_h0 = self._name(f"{prefix}_inh0")
        in_h = self._name(f"{prefix}_inh")
        in_p0 = self._name(f"{prefix}_inp0")
        in_p = self._name(f"{prefix}_inp")
        hole_bit = hole_norm
        piece_bit = piece_norm
        xor = self._name(f"{prefix}_xor")
        diff = self._name(f"{prefix}_diff")
        self._node("Less", [rows, hole_h], [in_h0])
        self._node("Less", [cols, hole_w], [f"{prefix}_inhw"])
        self._node("And", [in_h0, f"{prefix}_inhw"], [in_h])
        self._node("Less", [rows, piece_h], [in_p0])
        self._node("Less", [cols, piece_w], [f"{prefix}_inpw"])
        self._node("And", [in_p0, f"{prefix}_inpw"], [in_p])
        self._node("And", [in_h, hole_bit], [f"{prefix}_hp"])
        self._node("And", [in_p, piece_bit], [f"{prefix}_pp"])
        self._node("Xor", [f"{prefix}_hp", f"{prefix}_pp"], [xor])
        self._node("Cast", [xor], [f"{prefix}_xori"], to=TensorProto.INT32)
        self._node("ReduceSum", [f"{prefix}_xori"], [diff], keepdims=0)
        self._node("Equal", [diff, self.zero_i32], [f"{prefix}_nod"])
        return f"{prefix}_nod"

    def build(self) -> onnx.ModelProto:
        self._init_shared()
        axes4 = self._const(np.asarray([0, 1, 2, 3], dtype=np.int64), "axes4")

        assert self.half is not None
        channels: dict[int, str] = {}
        for ch in range(C):
            c_st = self._const(np.asarray([0, ch, 0, 0], dtype=np.int64), f"st_c{ch}")
            c_en = self._const(np.asarray([1, ch + 1, N, N], dtype=np.int64), f"en_c{ch}")
            c_float = self._name(f"cf{ch}")
            c_bool = self._name(f"cb{ch}")
            self._node("Slice", ["input", c_st, c_en, axes4], [c_float])
            self._node("Greater", [c_float, self.half], [c_bool])
            channels[ch] = c_bool
        out_bg = channels[0]
        gray = channels[GRAY]
        out_colors: dict[int, str] = {}
        for color in COLORS:
            out_colors[color] = channels[color]

        seed1 = self._min_seed(gray, "g1")
        comp1 = self._dilate(seed1, gray, "gc1", GRAY_DILATE_STEPS)
        rem = self._name("grem")
        self._node("Not", [comp1], [rem])
        self._node("And", [gray, rem], [f"{rem}_g"])
        seed2 = self._min_seed(f"{rem}_g", "g2")
        comp2 = self._dilate(seed2, f"{rem}_g", "gc2", GRAY_DILATE_STEPS - 1)

        holes: list[tuple[str, str, str, str, str, str]] = []
        for idx, comp in enumerate((comp1, comp2), start=1):
            self._append_hole(comp, out_bg, holes, f"h{idx}")

        for color in COLORS:
            color_mask = out_colors[color]
            pix_sum = self._name(f"ps{color}")
            assert self.zero_i32 is not None
            zero = self.zero_i32
            movable = self._name(f"mov{color}")
            reach_sum = self._name(f"rs{color}")
            seed = self._min_seed(color_mask, f"cs{color}")
            piece = self._dilate(seed, color_mask, f"cp{color}", COLOR_DILATE_STEPS[color])
            self._node("Cast", [color_mask], [f"cmc{color}"], to=TensorProto.INT32)
            self._node("ReduceSum", [f"cmc{color}"], [pix_sum], keepdims=0)
            self._node("Cast", [piece], [f"pc{color}"], to=TensorProto.INT32)
            self._node("ReduceSum", [f"pc{color}"], [reach_sum], keepdims=0)
            self._node("Greater", [pix_sum, zero], [f"has{color}"])
            self._node("Equal", [pix_sum, reach_sum], [f"onecc{color}"])
            self._node("And", [f"onecc{color}", f"has{color}"], [movable])
            ptop, pbot, pleft, pright, ph, pw = self._bbox_scalars(piece, f"pb{color}")
            piece_norm = self._gather_grid(piece, ptop, pleft, f"pn{color}")

            match_terms: list[str] = []
            apply_terms: list[str] = []
            for hole, hole_norm, _hleft, hh, hw, htag in holes:
                match = self._shape_match(
                    hole_norm,
                    hh,
                    hw,
                    ptop,
                    pleft,
                    ph,
                    pw,
                    piece_norm,
                    f"m_{htag}_c{color}",
                )
                match_terms.append(match)
                apply = self._name(f"app_{htag}_c{color}")
                self._node("And", [match, movable], [apply])
                apply_terms.append(apply)

            for apply, (hole, _htop, _hleft, _hh, _hw, htag) in zip(apply_terms, holes):
                apply_b = self._name(f"appb_{htag}_c{color}")
                self._node("Cast", [apply], [apply_b], to=TensorProto.BOOL)
                erase = self._name(f"er_{htag}_c{color}")
                fill = self._name(f"fi_{htag}_c{color}")
                not_erase = self._name(f"ner_{htag}_c{color}")
                cleared = self._name(f"clr_{htag}_c{color}")
                filled = self._name(f"fil_{htag}_c{color}")
                not_fill = self._name(f"nfi_{htag}_c{color}")
                self._node("And", [piece, apply_b], [erase])
                self._node("And", [hole, apply_b], [fill])
                self._node("Not", [erase], [not_erase])
                self._node("And", [out_colors[color], not_erase], [cleared])
                self._node("Or", [cleared, fill], [filled])
                out_colors[color] = filled
                bg_tmp = self._name(f"bgt_{htag}_c{color}")
                bg_erased = self._name(f"bge_{htag}_c{color}")
                bg_filled = self._name(f"bgf_{htag}_c{color}")
                self._node("And", [out_bg, not_erase], [bg_tmp])
                self._node("Or", [bg_tmp, erase], [bg_erased])
                self._node("Not", [fill], [not_fill])
                self._node("And", [bg_erased, not_fill], [bg_filled])
                out_bg = bg_filled

        out_chs: list[str | None] = [None] * C
        out_chs[0] = out_bg
        for color, name in out_colors.items():
            out_chs[color] = name
        for ch in range(C):
            if out_chs[ch] is not None:
                continue
            out_chs[ch] = channels[ch]
        out10b = self._name("out10b")
        self._node("Concat", [out_chs[0], out_chs[1], out_chs[2], out_chs[3], out_chs[4],
                              out_chs[5], out_chs[6], out_chs[7], out_chs[8], out_chs[9]], [out10b], axis=1)
        self._node("Cast", [out10b], ["out10"], to=TensorProto.FLOAT)
        self._node(
            "Pad",
            ["out10"],
            ["output"],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - N, W - N],
        )

        graph = helper.make_graph(
            self.nodes,
            "task044_generic",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)],
            initializer=self.inits,
        )
        model = helper.make_model(
            graph,
            producer_name="neurogolf-task044",
            ir_version=IR_VERSION,
            opset_imports=[helper.make_opsetid("", self.opset)],
        )
        onnx.checker.check_model(model)
        return model


def build_generic_model(opset: int = 10) -> onnx.ModelProto:
    return GraphBuilder(opset).build()


def validate_model(path: Path, examples: dict[str, list[dict[str, Any]]], verbose: bool) -> int:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    failures = 0
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(examples.get(split, [])):
            inp = grid_to_onehot(example["input"])
            got = session.run(["output"], {"input": inp})[0]
            exp = grid_to_onehot(example["output"])
            ok = np.array_equal(got > 0.0, exp > 0.0)
            failures += int(not ok)
            if verbose and split == "train":
                print(f"{split}[{idx}]: {'OK' if ok else 'FAIL'}")
                if not ok:
                    print("got:\n", "\n".join(" ".join(map(str, row)) for row in onehot_to_grid(got)))
                    print("exp:\n", "\n".join(" ".join(map(str, row)) for row in example["output"]))
    return failures


def score_candidate(path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from score_model import score_file

    with tempfile.TemporaryDirectory(prefix="task044_score_") as tmp:
        tmp_path = Path(tmp) / "task044.onnx"
        shutil.copy2(path, tmp_path)
        return score_file(tmp_path)


def shape_summary(model: onnx.ModelProto) -> None:
    inferred = onnx.shape_inference.infer_shapes(model)
    print(f"ONNX node count: {len(model.graph.node)}")
    print(f"shape inference tensors: {len(inferred.graph.value_info)}")


def main() -> None:
    examples = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    for split, split_examples in examples.items():
        for idx, example in enumerate(split_examples):
            solved = solve_grid(example["input"])
            if solved != example["output"]:
                raise SystemExit(f"Python rule mismatch {split}[{idx}]")

    candidates = [
        Variant("generic_opset10", lambda: build_generic_model(10)),
    ]

    results: list[tuple[int, Variant, Path, dict[str, Any]]] = []
    for candidate in candidates:
        path = ROOT / f"task044_{candidate.name}.onnx"
        try:
            model = candidate.build()
        except Exception as exc:
            print(f"{candidate.name}: build failed: {exc}")
            continue
        onnx.save(model, path)
        failures = validate_model(path, examples, verbose=False)
        if failures:
            print(f"{candidate.name}: {failures} validation failures")
            continue
        result = score_candidate(path)
        if not result["valid"]:
            print(f"{candidate.name}: invalid score: {result['error']}")
            continue
        cost = int(result["cost"])
        results.append((cost, candidate, path, result))
        print(
            f"{candidate.name:20s} nodes={len(model.graph.node)} memory={result['memory']} "
            f"params={result['params']} cost={result['cost']} score={result['score']:.6f}"
        )

    if not results:
        raise SystemExit("no valid candidate")

    results.sort(key=lambda item: item[0])
    _, best_candidate, best_path, best_result = results[0]
    shutil.copy2(best_path, OUT_PATH)
    shutil.copy2(best_path, ALL_OUT_PATH)
    print(f"\nBest: {best_candidate.name} -> {OUT_PATH}")

    print("\nTrain correctness:")
    validate_model(OUT_PATH, examples, verbose=True)

    best_model = onnx.load(str(OUT_PATH))
    print(f"\nFinal model: {len(best_model.graph.node)} nodes, opset {best_model.opset_import[0].version}")
    shape_summary(best_model)

    sys.path.insert(0, str(ROOT))
    from score_model import print_report, score_file

    print()
    print_report(score_file(OUT_PATH))


if __name__ == "__main__":
    main()
