"""Minimal ONNX for ARC task012: enlarge plus shapes with arm/center colors.

Task rule: each plus has a center pixel (C_center) and four orthogonal arm pixels
(C_arm) on black background. Output keeps the original plus, extends each arm by
one cell (orthogonal length 5: arm-arm-center-arm-arm), and adds diagonal pixels
at Manhattan (1,1) and (2,2) from center, all diagonals using C_center.

Pattern (A=arm, C=center):
  ..C..
  .ACA.
  AACAA
  .ACA.
  ..C..

ONNX: ArgMax label grid on a 12x12 crop (all task grids are 12x12); bool 4-neighbor center
test; value-weighted shifts for diagonals then cardinals; pad once per output channel.
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

BEST_PATH = OUT_DIR / "task012.onnx"
DATA_PATH = ROOT / "data" / "task012.json"

C = 10
H = W = 30
S = 12  # all examples are 12x12; padded output cells must remain all-zero
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10

TOY_INPUT = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 7, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 7, 2, 7, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]
TOY_OUTPUT = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 7, 0, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 7, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [7, 7, 2, 7, 7, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 7, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 7, 0, 2, 0, 2, 0, 7, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 2, 7, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 7, 7, 2, 7, 7, 0],
    [0, 0, 0, 0, 0, 0, 0, 2, 7, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 2, 0, 7, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

_DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1), (-2, -2), (-2, 2), (2, -2), (2, 2)]
_CARD = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _shift_spec(dy: int, dx: int) -> Tuple[List[int], List[int], List[int]]:
    ty, tx = max(0, -dy), max(0, -dx)
    by, bx = max(0, dy), max(0, dx)
    sy, sx = max(0, dy), max(0, dx)
    return (
        [0, ty, tx, 0, by, bx],
        [0, sy, sx],
        [1, S + sy, S + sx],
    )


def _i64(inits: List[onnx.TensorProto], vals: Sequence[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))
    return name


def _f32(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name))
    return name


def _np_shift(g: np.ndarray, dy: int, dx: int) -> np.ndarray:
    h, w = g.shape
    out = np.zeros_like(g)
    sy, sx = max(0, -dy), max(0, -dx)
    ey, ex = h - max(0, dy), w - max(0, dx)
    out[sy:ey, sx:ex] = g[max(0, dy) : max(0, dy) + ey - sy, max(0, dx) : max(0, dx) + ex - sx]
    return out


def solve(grid: np.ndarray) -> np.ndarray:
    """Expand each plus: extend arms +1, add center-colored diagonals."""
    g = np.asarray(grid, dtype=np.int64)
    if g.ndim == 4:
        g = g[0].argmax(axis=0).astype(np.int64)
    elif g.ndim == 3:
        g = g.argmax(axis=0).astype(np.int64)

    p = g > 0
    nbr = _np_shift(p, -1, 0) & _np_shift(p, 1, 0) & _np_shift(p, 0, -1) & _np_shift(p, 0, 1)
    cntr = p & nbr
    arm = p & ~nbr

    out = g.copy()
    nc = np.zeros_like(p)
    cv = g.astype(np.float32) * cntr
    for dy, dx in _DIAG:
        m = _np_shift(cntr, dy, dx) & ~p
        out = np.where(m, _np_shift(cv, dy, dx).astype(np.int64), out)
        nc |= m

    av = g.astype(np.float32) * arm
    for dy, dx in _CARD:
        m = _np_shift(arm, dy, dx) & ~p & ~nc
        out = np.where(m, _np_shift(av, dy, dx).astype(np.int64), out)
    return out


def _grid_to_onehot(grid: Sequence[Sequence[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            color = int(val)
            if 0 <= color < C:
                out[0, color, r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def build_onnx_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    ax3 = _i64(inits, [0, 1, 2], "ax3")
    z = _f32(inits, 0.0, "z")
    o = _f32(inits, 1.0, "o")
    s0 = _i64(inits, [0, 0, 0], "s0")
    eS = _i64(inits, [1, S, S], "eS")
    slice_keys: Dict[Tuple[int, ...], str] = {(0, 0, 0): s0}
    for dy, dx in _DIAG + _CARD:
        for part in _shift_spec(dy, dx)[1:]:
            t = tuple(part)
            if t not in slice_keys:
                slice_keys[t] = _i64(inits, list(part), f"k{t[1]}{t[2]}")

    def _shift(src: str, tag: str, dy: int, dx: int) -> str:
        pads, ss, ee = _shift_spec(dy, dx)
        pd = f"{tag}_pd"
        sl = f"{tag}_sl"
        nodes.extend(
            [
                helper.make_node("Pad", [src], [pd], pads=pads),
                helper.make_node("Slice", [pd, slice_keys[tuple(ss)], slice_keys[tuple(ee)], ax3], [sl]),
            ]
        )
        return sl

    def _and4(a: str, b: str, c: str, d: str, out: str) -> None:
        ab = f"{out}_ab"
        abc = f"{out}_abc"
        nodes.extend(
            [
                helper.make_node("And", [a, b], [ab]),
                helper.make_node("And", [ab, c], [abc]),
                helper.make_node("And", [abc, d], [out]),
            ]
        )

    def _max2(a: str, b: str, out: str) -> None:
        nodes.append(helper.make_node("Max", [a, b], [out]))

    nodes.extend(
        [
            helper.make_node("ArgMax", [IN_NAME], ["g2d"], axis=1, keepdims=0),
            helper.make_node("Slice", ["g2d", s0, eS, ax3], ["gi"]),
            helper.make_node("Cast", ["gi"], ["gf"], to=TensorProto.FLOAT),
            helper.make_node("Greater", ["gf", z], ["pb"]),
            helper.make_node("Cast", ["pb"], ["p"], to=TensorProto.FLOAT),
        ]
    )

    pb = [
        _shift("p", "pu", -1, 0),
        _shift("p", "pd", 1, 0),
        _shift("p", "pl", 0, -1),
        _shift("p", "pr", 0, 1),
    ]
    for i, s in enumerate(pb):
        nodes.append(helper.make_node("Cast", [s], [f"pz{i}"], to=TensorProto.BOOL))
    _and4("pz0", "pz1", "pz2", "pz3", "nbrb")
    nodes.append(helper.make_node("Cast", ["nbrb"], ["nbr"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Mul", ["p", "nbr"], ["cntr"]))
    nodes.append(helper.make_node("Sub", [o, "cntr"], ["notc"]))
    nodes.append(helper.make_node("Mul", ["p", "notc"], ["armf"]))
    nodes.append(helper.make_node("Mul", ["gf", "cntr"], ["cval"]))

    g_out = "gf"
    nc = None
    inv_p = "invP"
    nodes.append(helper.make_node("Sub", [o, "p"], [inv_p]))
    for i, (dy, dx) in enumerate(_DIAG):
        sm = _shift("cntr", f"ds{i}", dy, dx)
        mask = f"dm{i}"
        nodes.append(helper.make_node("Mul", [sm, inv_p], [mask]))
        maskb = f"dmb{i}"
        nodes.append(helper.make_node("Greater", [mask, z], [maskb]))
        val = _shift("cval", f"dv{i}", dy, dx)
        nxt = f"g{i}"
        nodes.append(helper.make_node("Where", [maskb, val, g_out], [nxt]))
        g_out = nxt
        if nc is None:
            nc = mask
        else:
            merged = f"nc{i}"
            _max2(nc, mask, merged)
            nc = merged

    inv_nc = "invNC"
    nodes.append(helper.make_node("Sub", [o, nc], [inv_nc]))
    free = "free"
    nodes.append(helper.make_node("Mul", [inv_p, inv_nc], [free]))
    nodes.append(helper.make_node("Mul", ["gf", "armf"], ["aval"]))
    for i, (dy, dx) in enumerate(_CARD):
        sm = _shift("armf", f"as{i}", dy, dx)
        mask = f"am{i}"
        nodes.append(helper.make_node("Mul", [sm, free], [mask]))
        maskb = f"amb{i}"
        nodes.append(helper.make_node("Greater", [mask, z], [maskb]))
        val = _shift("aval", f"av{i}", dy, dx)
        nxt = f"ga{i}"
        nodes.append(helper.make_node("Where", [maskb, val, g_out], [nxt]))
        g_out = nxt

    nodes.append(helper.make_node("Cast", [g_out], ["gout"], to=TensorProto.INT64))
    cv0 = _i64(inits, [0], "cv0")
    nodes.append(helper.make_node("Equal", ["gout", cv0], ["eq0"]))
    nodes.append(helper.make_node("Cast", ["eq0"], ["ch0"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Unsqueeze", ["ch0"], ["chu0"], axes=[1]))
    nodes.append(helper.make_node("Pad", ["chu0"], ["bg"], pads=[0, 0, 0, 0, 0, 0, H - S, W - S]))

    chans: List[str] = []
    for color in range(1, C):
        cv = _i64(inits, [color], f"cv{color}")
        eq = f"eq{color}"
        nodes.append(helper.make_node("Equal", ["gout", cv], [eq]))
        ch = f"ch{color}"
        nodes.append(helper.make_node("Cast", [eq], [ch], to=TensorProto.FLOAT))
        chu = f"chu{color}"
        nodes.append(helper.make_node("Unsqueeze", [ch], [chu], axes=[1]))
        chp = f"chp{color}"
        nodes.append(helper.make_node("Pad", [chu], [chp], pads=[0, 0, 0, 0, 0, 0, H - S, W - S]))
        chans.append(chp)

    nodes.append(helper.make_node("Concat", ["bg", *chans], [OUT_NAME], axis=1))

    graph = helper.make_graph(nodes, "task012", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="task012",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return model


def save_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    model = build_onnx_model()
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return model


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def verify_all(model: onnx.ModelProto) -> int:
    bad = 0
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            g = np.array(ex["input"], dtype=np.int64)
            if g.shape[0] > H or g.shape[1] > W:
                continue
            oh = _grid_to_onehot(g.tolist())
            pred = _run_onnx(model, oh)
            ref = _grid_to_onehot(ex["output"])
            if not np.array_equal(pred > 0.0, ref > 0.0):
                bad += 1
    return bad


def main() -> None:
    model = save_model()
    bad = verify_all(model)
    print(f"verify: {'PASS' if bad == 0 else f'FAIL ({bad})'}")
    stats = score_file(BEST_PATH)
    print(
        f"memory={stats.get('memory')} params={stats.get('params')} "
        f"cost={stats.get('cost')} score={stats.get('score', 0):.4f}"
    )


if __name__ == "__main__":
    main()
