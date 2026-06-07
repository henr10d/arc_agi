"""Minimal ONNX for ARC task009: connect aligned 2x2 seed blocks by color.

Task rule: the grid has a checker/lattice background (non-zero stripe colors and
0 holes). Monochrome 2x2 seed blocks of each foreground color appear on the
lattice. For each color, connect every pair of seed top-left corners that share
the same row (horizontal, 2-row-tall strip) or the same column (vertical,
2-col-wide strip) by painting that color only into input cells that are 0.
Stripe/structure colors are never overwritten. Colors with fewer than two seeds
on a line are unchanged.

ONNX: per channel 1..9 detect 2x2 top-left seeds, row/column min-max gaps,
build horizontal and vertical fill masks (only where input channel 0 is set),
OR with the original channel, then clear background where any color was added.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

BEST_PATH = OUT_DIR / "task009.onnx"
DATA_PATH = ROOT / "data" / "task009.json"

C = 10
NC = 9
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10

TOY_INPUT = [
    [8, 0, 0, 8, 0, 0, 8, 0, 0, 8],
    [8, 0, 0, 8, 0, 0, 8, 0, 0, 8],
    [8, 2, 2, 8, 0, 0, 8, 2, 2, 8],
    [8, 2, 2, 8, 0, 0, 8, 2, 2, 8],
    [8, 0, 0, 8, 0, 0, 8, 0, 0, 8],
]
TOY_OUTPUT = [
    [8, 0, 0, 8, 0, 0, 8, 0, 0, 8],
    [8, 0, 0, 8, 0, 0, 8, 0, 0, 8],
    [8, 2, 2, 8, 2, 2, 8, 2, 2, 8],
    [8, 2, 2, 8, 2, 2, 8, 2, 2, 8],
    [8, 0, 0, 8, 0, 0, 8, 0, 0, 8],
]


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], arr, name: str) -> str:
    return _init(inits, np.asarray(arr, dtype=np.float32), name)


def solve(grid: np.ndarray) -> np.ndarray:
    """Connect same-color 2x2 seeds along shared rows/columns; fill only 0 cells."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    zero = g == 0
    h, w = g.shape
    for color in range(1, 10):
        if not np.any(g == color):
            continue
        blocks: list[tuple[int, int]] = []
        for r in range(h - 1):
            for c in range(w - 1):
                if not np.all(g[r : r + 2, c : c + 2] == color):
                    continue
                if c > 0 and g[r, c - 1] == color and g[r + 1, c - 1] == color:
                    continue
                if r > 0 and g[r - 1, c] == color and g[r - 1, c + 1] == color:
                    continue
                blocks.append((r, c))
        by_row: dict[int, list[int]] = {}
        by_col: dict[int, list[int]] = {}
        for r, c in blocks:
            by_row.setdefault(r, []).append(c)
            by_col.setdefault(c, []).append(r)
        for r, cols in by_row.items():
            if len(cols) < 2:
                continue
            c0, c1 = min(cols), max(cols)
            if c1 <= c0 + 2:
                continue
            sl = out[r : r + 2, c0 + 2 : c1]
            out[r : r + 2, c0 + 2 : c1] = np.where(zero[r : r + 2, c0 + 2 : c1], color, sl)
        for c, rows in by_col.items():
            if len(rows) < 2:
                continue
            r0, r1 = min(rows), max(rows)
            if r1 <= r0 + 2:
                continue
            sl = out[r0 + 2 : r1, c : c + 2]
            out[r0 + 2 : r1, c : c + 2] = np.where(zero[r0 + 2 : r1, c : c + 2], color, sl)
    return out


def _grid_to_onehot(grid: List[List[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _detect_tl(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    nz: str,
    tag: str,
    axes4: str,
    sh_half: str,
) -> str:
    """Top-left corner of each 2x2 block (non-overlapping) in channel bool nz."""
    st29 = _i64(inits, [0, 0, 0, 0], f"{tag}_st29")
    en29 = _i64(inits, [1, 1, H - 1, W - 1], f"{tag}_en29")
    st01 = _i64(inits, [0, 0, 0, 1], f"{tag}_st01")
    en01 = _i64(inits, [1, 1, H - 1, W], f"{tag}_en01")
    st10 = _i64(inits, [0, 0, 1, 0], f"{tag}_st10")
    en10 = _i64(inits, [1, 1, H, W - 1], f"{tag}_en10")
    st_pn = _i64(inits, [0, 0, 1, 0], f"{tag}_stpn")
    en_pn = _i64(inits, [1, 1, H, W - 2], f"{tag}_enpn")
    st11 = _i64(inits, [0, 0, 1, 1], f"{tag}_st11")
    en11 = _i64(inits, [1, 1, H, W], f"{tag}_en11")
    st28 = _i64(inits, [0, 0, 0, 0], f"{tag}_st28")
    en28 = _i64(inits, [1, 1, H - 1, W - 2], f"{tag}_en28")
    st_pn2 = _i64(inits, [0, 0, 0, 1], f"{tag}_stpn2")
    en_pn2 = _i64(inits, [1, 1, H - 2, W], f"{tag}_enpn2")
    st_pr = _i64(inits, [0, 0, 0, 0], f"{tag}_stpr")
    en_pr = _i64(inits, [1, 1, H - 2, W - 1], f"{tag}_enpr")

    zcol = _init(inits, np.zeros((1, 1, H - 1, 1), dtype=np.bool_), f"{tag}_zcol")
    zrow = _init(inits, np.zeros((1, 1, 1, W - 1), dtype=np.bool_), f"{tag}_zrow")

    c00, c01, c10, c11 = f"{tag}_00", f"{tag}_01", f"{tag}_10", f"{tag}_11"
    left, top = f"{tag}_l", f"{tag}_t"
    f2a, f2b, f2 = f"{tag}_f2a", f"{tag}_f2b", f"{tag}_f2"
    tl = f"{tag}_tl"
    nodes.extend(
        [
            helper.make_node("Slice", [nz, st29, en29, axes4], [c00]),
            helper.make_node("Slice", [nz, st01, en01, axes4], [c01]),
            helper.make_node("Slice", [nz, st10, en10, axes4], [c10]),
            helper.make_node("Slice", [nz, st11, en11, axes4], [c11]),
            helper.make_node("Slice", [nz, st28, en28, axes4], [f"{tag}_pc"]),
            helper.make_node("Slice", [nz, st_pn, en_pn, axes4], [f"{tag}_pn"]),
            helper.make_node("Concat", [zcol, f"{tag}_pc"], [f"{tag}_xm"], axis=3),
            helper.make_node("Concat", [zcol, f"{tag}_pn"], [f"{tag}_xmn"], axis=3),
            helper.make_node("And", [f"{tag}_xm", f"{tag}_xmn"], [left]),
            helper.make_node("Slice", [nz, st_pr, en_pr, axes4], [f"{tag}_pr"]),
            helper.make_node("Slice", [nz, st_pn2, en_pn2, axes4], [f"{tag}_pn2"]),
            helper.make_node("Concat", [zrow, f"{tag}_pr"], [f"{tag}_ym"], axis=2),
            helper.make_node("Concat", [zrow, f"{tag}_pn2"], [f"{tag}_ymn"], axis=2),
            helper.make_node("And", [f"{tag}_ym", f"{tag}_ymn"], [top]),
            helper.make_node("And", [c00, c01], [f2a]),
            helper.make_node("And", [c10, c11], [f2b]),
            helper.make_node("And", [f2a, f2b], [f2]),
            helper.make_node("Not", [left], [f"{tag}_nl"]),
            helper.make_node("Not", [top], [f"{tag}_nt"]),
            helper.make_node("And", [f2, f"{tag}_nl"], [f"{tag}_f2l"]),
            helper.make_node("And", [f"{tag}_f2l", f"{tag}_nt"], [tl]),
        ]
    )
    return tl


def _horiz_fill(
    nodes: List[onnx.NodeProto],
    tl: str,
    tag: str,
    sh: Dict[str, str],
) -> str:
    col29, big, neg, two = sh["col29"], sh["big"], sh["neg"], sh["two"]
    r29, r32 = sh["r29"], sh["r32"]
    y_coord, x_coord = sh["y_coord"], sh["x_coord"]

    cx, cp, cn, cn2 = f"{tag}_cx", f"{tag}_cp", f"{tag}_cn", f"{tag}_cn2"
    gap = f"{tag}_gap"
    yge, yle, yr, rhb = f"{tag}_yge", f"{tag}_yle", f"{tag}_yr", f"{tag}_rhb"
    pl, pr, ly, ry = f"{tag}_pl", f"{tag}_pr", f"{tag}_ly", f"{tag}_ry"
    gx0, gx1, hz = f"{tag}_gx0", f"{tag}_gx1", f"{tag}_hz"

    nodes.extend(
        [
            helper.make_node("Where", [tl, col29, sh["zero_f"]], [f"{tag}_cpmax"]),
            helper.make_node("ReduceMax", [f"{tag}_cpmax"], [cx], axes=[3], keepdims=1),
            helper.make_node("Where", [tl, col29, big], [cp]),
            helper.make_node("ReduceMin", [cp], [cn], axes=[3], keepdims=1),
            helper.make_node("Add", [cn, two], [cn2]),
            helper.make_node("Greater", [cx, cn2], [gap]),
            helper.make_node("Reshape", [gap, sh["sh_gr"]], [f"{tag}_gapb"]),
            helper.make_node("Less", [y_coord, r29], [f"{tag}_ylt"]),
            helper.make_node("Not", [f"{tag}_ylt"], [yge]),
            helper.make_node("Less", [y_coord, r32], [yle]),
            helper.make_node("And", [yge, yle], [yr]),
            helper.make_node("And", [yr, f"{tag}_gapb"], [rhb]),
            helper.make_node("Reshape", [cn2, sh["sh_gr"]], [f"{tag}_cn2b"]),
            helper.make_node("Reshape", [cx, sh["sh_gr"]], [f"{tag}_cxb"]),
            helper.make_node("Where", [rhb, f"{tag}_cn2b", big], [pl]),
            helper.make_node("ReduceMin", [pl], [ly], axes=[3], keepdims=1),
            helper.make_node("Where", [rhb, f"{tag}_cxb", neg], [pr]),
            helper.make_node("ReduceMax", [pr], [ry], axes=[3], keepdims=1),
            helper.make_node("Cast", [rhb], [f"{tag}_rhbf"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", [f"{tag}_rhbf"], [f"{tag}_rowf"], axes=[3], keepdims=1),
            helper.make_node("Greater", [f"{tag}_rowf", sh["half"]], [f"{tag}_rowm"]),
            helper.make_node("Less", [x_coord, ly], [f"{tag}_xl0"]),
            helper.make_node("Not", [f"{tag}_xl0"], [gx0]),
            helper.make_node("Less", [x_coord, ry], [gx1]),
            helper.make_node("And", [f"{tag}_rowm", gx0], [f"{tag}_hz0"]),
            helper.make_node("And", [f"{tag}_hz0", gx1], [hz]),
        ]
    )
    return hz


def _vert_fill(
    nodes: List[onnx.NodeProto],
    tl: str,
    tag: str,
    sh: Dict[str, str],
) -> str:
    """2-wide vertical strips between same-column seed pairs (transpose of horiz)."""
    row29, big, neg, two = sh["row29"], sh["big"], sh["neg"], sh["two"]
    c29, c32 = sh["c29"], sh["c32"]
    y_coord, x_coord = sh["y_coord"], sh["x_coord"]

    rx, rp, rn, rn2 = f"{tag}_rx", f"{tag}_rp", f"{tag}_rn", f"{tag}_rn2"
    gap = f"{tag}_vg"
    xge, xle, xr, chb = f"{tag}_xge", f"{tag}_xle", f"{tag}_xr", f"{tag}_chb"
    pu, pl, ty, by = f"{tag}_pu", f"{tag}_pl", f"{tag}_ty", f"{tag}_by"
    gy0, gy1, vt = f"{tag}_gy0", f"{tag}_gy1", f"{tag}_vt"

    nodes.extend(
        [
            helper.make_node("Where", [tl, row29, sh["zero_f"]], [f"{tag}_rpmax"]),
            helper.make_node("ReduceMax", [f"{tag}_rpmax"], [rx], axes=[2], keepdims=1),
            helper.make_node("Where", [tl, row29, big], [rp]),
            helper.make_node("ReduceMin", [rp], [rn], axes=[2], keepdims=1),
            helper.make_node("Add", [rn, two], [rn2]),
            helper.make_node("Greater", [rx, rn2], [gap]),
            helper.make_node("Reshape", [gap, sh["sh_gv"]], [f"{tag}_gapv"]),
            helper.make_node("Less", [x_coord, c29], [f"{tag}_xlt"]),
            helper.make_node("Not", [f"{tag}_xlt"], [xge]),
            helper.make_node("Less", [x_coord, c32], [xle]),
            helper.make_node("And", [xge, xle], [xr]),
            helper.make_node("And", [xr, f"{tag}_gapv"], [chb]),
            helper.make_node("Reshape", [rn2, sh["sh_gv"]], [f"{tag}_rn2v"]),
            helper.make_node("Reshape", [rx, sh["sh_gv"]], [f"{tag}_rxv"]),
            helper.make_node("Where", [chb, f"{tag}_rn2v", big], [pu]),
            helper.make_node("ReduceMin", [pu], [ty], axes=[2], keepdims=1),
            helper.make_node("Where", [chb, f"{tag}_rxv", neg], [pl]),
            helper.make_node("ReduceMax", [pl], [by], axes=[2], keepdims=1),
            helper.make_node("Cast", [chb], [f"{tag}_chbf"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", [f"{tag}_chbf"], [f"{tag}_colf"], axes=[2], keepdims=1),
            helper.make_node("Greater", [f"{tag}_colf", sh["half"]], [f"{tag}_colm"]),
            helper.make_node("Less", [y_coord, ty], [f"{tag}_yl0"]),
            helper.make_node("Not", [f"{tag}_yl0"], [gy0]),
            helper.make_node("Less", [y_coord, by], [gy1]),
            helper.make_node("And", [f"{tag}_colm", gy0], [f"{tag}_vt0"]),
            helper.make_node("And", [f"{tag}_vt0", gy1], [vt]),
        ]
    )
    return vt


def _process_color(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    ch: str,
    tag: str,
    sh: Dict[str, str],
) -> tuple[str, str]:
    nz = f"{tag}_nz"
    nodes.append(helper.make_node("Greater", [ch, sh["half"]], [nz]))
    tl = _detect_tl(nodes, inits, nz, f"{tag}_d", sh["axes4"], sh["half"])
    hz = _horiz_fill(nodes, tl, f"{tag}_h", sh)
    vt = _vert_fill(nodes, tl, f"{tag}_v", sh)
    fill = f"{tag}_fill"
    fillz = f"{tag}_fz"
    out = f"{tag}_out"
    nodes.extend(
        [
            helper.make_node("Or", [hz, vt], [fill]),
            helper.make_node("And", [fill, sh["zero"]], [fillz]),
            helper.make_node("Or", [nz, fillz], [f"{tag}_outb"]),
            helper.make_node("Cast", [f"{tag}_outb"], [out], to=TensorProto.FLOAT),
        ]
    )
    return out, fillz


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    half = _f32(inits, 0.5, "half")
    two = _f32(inits, 2.0, "two")
    big = _f32(inits, 999.0, "big")
    neg = _f32(inits, -1.0, "neg")
    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    row29 = _f32(inits, np.arange(H - 1, dtype=np.float32).reshape(1, 1, H - 1, 1), "row29")
    col29 = _f32(inits, np.arange(W - 1, dtype=np.float32).reshape(1, 1, 1, W - 1), "col29")
    r29 = _f32(inits, np.arange(H - 1, dtype=np.float32).reshape(1, 1, 1, H - 1), "r29")
    r32 = _f32(inits, (np.arange(H - 1, dtype=np.float32) + 2.0).reshape(1, 1, 1, H - 1), "r32")
    c29 = _f32(inits, np.arange(W - 1, dtype=np.float32).reshape(1, 1, W - 1, 1), "c29")
    c32 = _f32(inits, (np.arange(W - 1, dtype=np.float32) + 2.0).reshape(1, 1, W - 1, 1), "c32")
    y_coord = _f32(inits, np.arange(H, dtype=np.float32).reshape(1, 1, H, 1), "y_coord")
    x_coord = _f32(inits, np.arange(W, dtype=np.float32).reshape(1, 1, 1, W), "x_coord")
    zero = _f32(inits, 0.0, "zero_f")
    ch0_st = _i64(inits, [0, 0, 0, 0], "ch0_st")
    ch0_en = _i64(inits, [1, 1, H, W], "ch0_en")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, ch0_st, ch0_en, axes4], ["ch0"]),
            helper.make_node("Greater", ["ch0", half], ["zero"]),
        ]
    )

    sh_gr = _i64(inits, [1, 1, 1, H - 1], "sh_gr")
    sh_gv = _i64(inits, [1, 1, H - 1, 1], "sh_gv")
    sh: Dict[str, str] = {
        "half": half,
        "two": two,
        "big": big,
        "neg": neg,
        "axes4": axes4,
        "col29": col29,
        "row29": row29,
        "r29": r29,
        "r32": r32,
        "c29": c29,
        "c32": c32,
        "y_coord": y_coord,
        "x_coord": x_coord,
        "zero": "zero",
        "zero_f": zero,
        "sh_gr": sh_gr,
        "sh_gv": sh_gv,
    }

    out_planes: List[str] = []
    fill_terms: List[str] = []
    for color in range(1, C):
        st = _i64(inits, [0, color, 0, 0], f"s{color}")
        en = _i64(inits, [1, color + 1, H, W], f"e{color}")
        ch = f"ch{color}"
        nodes.append(helper.make_node("Slice", [IN_NAME, st, en, axes4], [ch]))
        out, fillz = _process_color(nodes, inits, ch, f"c{color}", sh)
        out_planes.append(out)
        fill_terms.append(fillz)

    fg = fill_terms[0]
    for i, nxt in enumerate(fill_terms[1:], start=1):
        fg_new = f"fg{i}"
        nodes.append(helper.make_node("Or", [fg, nxt], [fg_new]))
        fg = fg_new

    ch0_out = "ch0_out"
    nodes.extend(
        [
            helper.make_node("Not", [fg], ["fg_not"]),
            helper.make_node("And", [sh["zero"], "fg_not"], ["ch0_outb"]),
            helper.make_node("Cast", ["ch0_outb"], [ch0_out], to=TensorProto.FLOAT),
        ]
    )

    all_ch = [ch0_out] + out_planes
    nodes.append(helper.make_node("Concat", all_ch, [OUT_NAME], axis=1))

    graph = helper.make_graph(nodes, "task009", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    if not DATA_PATH.is_file():
        return 0
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            g = np.array(ex["input"], dtype=np.int64)
            if g.shape[0] > H or g.shape[1] > W:
                continue
            oh = _grid_to_onehot(ex["input"])
            pred = _onehot_to_grid(_run_onnx(model, oh))[: g.shape[0], : g.shape[1]]
            if not np.array_equal(pred, solve(g)):
                bad += 1
    return bad


def main() -> None:
    inp = np.array(TOY_INPUT, dtype=np.int64)
    exp = np.array(TOY_OUTPUT, dtype=np.int64)
    assert np.array_equal(solve(inp), exp), "reference solver failed toy example"

    model = build_model()
    onnx.save(model, BEST_PATH)

    toy_oh = _grid_to_onehot(TOY_INPUT)
    toy_pred = _onehot_to_grid(_run_onnx(model, toy_oh))[:5, :10]
    assert np.array_equal(toy_pred, exp), f"toy mismatch:\n{toy_pred}\n{exp}"

    bad = validate_json(model)
    print("toy: PASS")
    print(f"task009.json: {'PASS' if bad == 0 else f'FAIL ({bad} wrong)'}")

    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"nodes:   {len(model.graph.node)}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
