"""Minimal ONNX for ARC task008: red (2) slides toward cyan (8) until contact.

Task rule: the red connected component translates rigidly one cell at a time toward the
cyan component (sign of centroid offset). When |dy| and |dx| differ, move on the dominant
axis only; when tied, compare vertical and horizontal and take the fewest steps (vertical
wins equal-step ties). Stop before overlap or out-of-bounds clipping. Cyan fixed.

ONNX: 16x16 masks for colors 2 and 8, centroid direction, 9 unrolled shift steps for the
vertical and horizontal paths, Pad/Slice shifts, min-step tie-break, pad to 30x30 one-hot
output.
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

BEST_PATH = OUT_DIR / "task008.onnx"
DATA_PATH = ROOT / "data" / "task008.json"

C = 10
H = W = 30
SH = SW = 16
RED = 2
CYAN = 8
MAX_STEPS = 9
EPS = 1e-3
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10

TOY_INPUT = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 2, 2, 0, 0, 0, 0, 0],
    [2, 2, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 0, 0, 0, 0],
    [0, 0, 0, 8, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]

_SHIFT_TABLE: Dict[Tuple[int, int], Tuple[List[int], Tuple[int, int, int, int], Tuple[int, int, int, int]]] = {
    (-1, 0): ([0, 0, 0, 0, 0, 0, 1, 0], (0, 0, 1, 0), (1, 1, SH + 1, SW)),
    (1, 0): ([0, 0, 1, 0, 0, 0, 0, 0], (0, 0, 0, 0), (1, 1, SH, SW)),
    (0, -1): ([0, 0, 0, 0, 0, 0, 0, 1], (0, 0, 0, 1), (1, 1, SH, SW + 1)),
    (0, 1): ([0, 0, 0, 1, 0, 0, 0, 0], (0, 0, 0, 0), (1, 1, SH, SW)),
    (-1, -1): ([0, 0, 0, 0, 0, 0, 1, 1], (0, 0, 1, 1), (1, 1, SH + 1, SW + 1)),
    (-1, 1): ([0, 0, 0, 1, 0, 0, 1, 0], (0, 0, 1, 0), (1, 1, SH + 1, SW)),
    (1, -1): ([0, 0, 1, 0, 0, 0, 0, 1], (0, 0, 0, 1), (1, 1, SH, SW + 1)),
    (1, 1): ([0, 0, 1, 1, 0, 0, 0, 0], (0, 0, 0, 0), (1, 1, SH, SW)),
}


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Sequence[int], name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _sign(x: float) -> int:
    return int((x > 0) - (x < 0))


def _centers(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0.0, 0.0
    return float(ys.mean()), float(xs.mean())


def _candidates(cy: float, cx: float, ry: float, rx: float) -> List[Tuple[int, int]]:
    dy, dx = _sign(cy - ry), _sign(cx - rx)
    ady, adx = abs(cy - ry), abs(cx - rx)
    if ady > adx + EPS:
        return [(dy, 0)]
    if adx > ady + EPS:
        return [(0, dx)]
    cands: List[Tuple[int, int]] = []
    if dy:
        cands.append((dy, 0))
    if dx:
        cands.append((0, dx))
    return cands


def _shift_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(mask)
    h, w = mask.shape
    for y, x in zip(*np.where(mask)):
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            out[ny, nx] = True
    return out


def _count_steps(red: np.ndarray, cyan: np.ndarray, dy: int, dx: int) -> int:
    cur = red.copy()
    steps = 0
    if dy == 0 and dx == 0:
        return 0
    for _ in range(MAX_STEPS):
        shifted = _shift_mask(cur, dy, dx)
        if shifted.sum() != cur.sum() or np.any(shifted & cyan):
            break
        cur = shifted
        steps += 1
    return steps


def _simulate(red: np.ndarray, cyan: np.ndarray, dy: int, dx: int) -> np.ndarray:
    cur = red.copy()
    if dy == 0 and dx == 0:
        return cur
    for _ in range(MAX_STEPS):
        shifted = _shift_mask(cur, dy, dx)
        if shifted.sum() != cur.sum() or np.any(shifted & cyan):
            break
        cur = shifted
    return cur


def solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    red = g == RED
    cyan = g == CYAN
    if not red.any() or not cyan.any():
        return g.copy()
    cy, cx = _centers(cyan)
    ry, rx = _centers(red)
    pri = {(_sign(cy - ry), 0): 0, (0, _sign(cx - rx)): 1}
    dy0, dx0 = _sign(cy - ry), _sign(cx - rx)
    if dy0 and dx0:
        pri[(dy0, dx0)] = 2
    best_key = (999, 9)
    best_cur = red
    for dy, dx in _candidates(cy, cx, ry, rx):
        key = (_count_steps(red, cyan, dy, dx), pri.get((dy, dx), 9))
        if key < best_key:
            best_key, best_cur = key, _simulate(red, cyan, dy, dx)
    out = np.zeros_like(g)
    out[cyan] = CYAN
    out[best_cur] = RED
    return out


def _grid_to_onehot(grid: Sequence[Sequence[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _shift_once(
    nodes: List[onnx.NodeProto],
    src: str,
    tag: str,
    dy: int,
    dx: int,
    ax4: str,
    slice_starts: Dict[Tuple[int, int, int, int], str],
    slice_ends: Dict[Tuple[int, int, int, int], str],
) -> str:
    pads, st, en = _SHIFT_TABLE[(dy, dx)]
    pd = f"{tag}_pd"
    out = f"{tag}_sl"
    nodes.extend(
        [
            helper.make_node("Pad", [src], [pd], pads=pads),
            helper.make_node("Slice", [pd, slice_starts[st], slice_ends[en], ax4], [out]),
        ]
    )
    return out


def _shift_dy(
    nodes: List[onnx.NodeProto],
    cur: str,
    tag: str,
    dys: str,
    eps: str,
    eps_neg: str,
    ax4: str,
    slice_starts: Dict[Tuple[int, int, int, int], str],
    slice_ends: Dict[Tuple[int, int, int, int], str],
) -> str:
    up = _shift_once(nodes, cur, f"{tag}_u", -1, 0, ax4, slice_starts, slice_ends)
    dn = _shift_once(nodes, cur, f"{tag}_d", 1, 0, ax4, slice_starts, slice_ends)
    return _pick_by_sign(nodes, dn, up, cur, dys, eps, eps_neg, tag)


def _shift_dx(
    nodes: List[onnx.NodeProto],
    cur: str,
    tag: str,
    dxs: str,
    eps: str,
    eps_neg: str,
    ax4: str,
    slice_starts: Dict[Tuple[int, int, int, int], str],
    slice_ends: Dict[Tuple[int, int, int, int], str],
) -> str:
    rt = _shift_once(nodes, cur, f"{tag}_r", 0, 1, ax4, slice_starts, slice_ends)
    lt = _shift_once(nodes, cur, f"{tag}_l", 0, -1, ax4, slice_starts, slice_ends)
    return _pick_by_sign(nodes, rt, lt, cur, dxs, eps, eps_neg, tag)


def _advance(
    nodes: List[onnx.NodeProto],
    cur: str,
    shifted: str,
    cyan_bool: str,
    half: str,
    zf: str,
    tag: str,
) -> Tuple[str, str]:
    gb = f"{tag}b"
    sf = f"{tag}f"
    sc = f"{tag}sc"
    ss = f"{tag}ss"
    eq = f"{tag}eq"
    ov = f"{tag}ov"
    ovf = f"{tag}ovf"
    os_ = f"{tag}os"
    hit = f"{tag}hit"
    oob = f"{tag}oob"
    blk = f"{tag}blk"
    ok = f"{tag}ok"
    okf = f"{tag}okf"
    nxt = f"{tag}nxt"
    nodes.extend(
        [
            helper.make_node("Greater", [shifted, half], [gb]),
            helper.make_node("Cast", [gb], [sf], to=TensorProto.FLOAT),
            helper.make_node("ReduceSum", [cur], [sc], keepdims=0),
            helper.make_node("ReduceSum", [sf], [ss], keepdims=0),
            helper.make_node("Cast", [sc], [f"{tag}sci"], to=TensorProto.INT64),
            helper.make_node("Cast", [ss], [f"{tag}ssi"], to=TensorProto.INT64),
            helper.make_node("Equal", [f"{tag}sci", f"{tag}ssi"], [eq]),
            helper.make_node("And", [gb, cyan_bool], [ov]),
            helper.make_node("Cast", [ov], [ovf], to=TensorProto.FLOAT),
            helper.make_node("ReduceSum", [ovf], [os_]),
            helper.make_node("Greater", [os_, zf], [hit]),
            helper.make_node("Not", [eq], [oob]),
            helper.make_node("Or", [hit, oob], [blk]),
            helper.make_node("Not", [blk], [ok]),
            helper.make_node("Cast", [ok], [okf], to=TensorProto.FLOAT),
            helper.make_node("Where", [blk, cur, sf], [nxt]),
        ]
    )
    return nxt, okf


def _simulate_axis(
    nodes: List[onnx.NodeProto],
    red: str,
    cyan_bool: str,
    axis: str,
    dys: str,
    dxs: str,
    tag: str,
    half: str,
    zf: str,
    eps: str,
    eps_neg: str,
    ax4: str,
    slice_starts: Dict[Tuple[int, int, int, int], str],
    slice_ends: Dict[Tuple[int, int, int, int], str],
) -> Tuple[str, str]:
    cur = red
    steps = zf
    for i in range(MAX_STEPS):
        if axis == "v":
            sh = _shift_dy(nodes, cur, f"{tag}{i}", dys, eps, eps_neg, ax4, slice_starts, slice_ends)
        else:
            sh = _shift_dx(nodes, cur, f"{tag}{i}", dxs, eps, eps_neg, ax4, slice_starts, slice_ends)
        nxt, okf = _advance(nodes, cur, sh, cyan_bool, half, zf, f"{tag}{i}")
        stp = f"{tag}{i}stp"
        nodes.append(helper.make_node("Add", [steps, okf], [stp]))
        cur = nxt
        steps = stp
    return cur, steps


def _pick_by_sign(
    nodes: List[onnx.NodeProto],
    pos: str,
    neg: str,
    idle: str,
    sign_scalar: str,
    eps: str,
    eps_neg: str,
    tag: str,
) -> str:
    out = f"{tag}_sel"
    sp = f"{tag}_sp"
    sn = f"{tag}_sn"
    mid = f"{tag}_mid"
    nodes.extend(
        [
            helper.make_node("Greater", [sign_scalar, eps], [sp]),
            helper.make_node("Less", [sign_scalar, eps_neg], [sn]),
            helper.make_node("Where", [sp, pos, idle], [mid]),
            helper.make_node("Where", [sn, neg, mid], [out]),
        ]
    )
    return out


def _prefer_a(
    nodes: List[onnx.NodeProto],
    steps_a: str,
    pri_a: str,
    steps_b: str,
    pri_b: str,
    mask_a: str,
    mask_b: str,
    tag: str,
) -> Tuple[str, str, str]:
    sa = f"{tag}_sa"
    sb = f"{tag}_sb"
    pa = f"{tag}_pa_i"
    pb = f"{tag}_pb_i"
    lt = f"{tag}_lt"
    eq = f"{tag}_eq"
    ple = f"{tag}_ple"
    bp = f"{tag}_bp"
    pick_a = f"{tag}_pa"
    mask_out = f"{tag}_m"
    steps_out = f"{tag}_s"
    pri_out = f"{tag}_p"
    nodes.extend(
        [
            helper.make_node("Cast", [steps_a], [sa], to=TensorProto.INT64),
            helper.make_node("Cast", [steps_b], [sb], to=TensorProto.INT64),
            helper.make_node("Cast", [pri_a], [pa], to=TensorProto.INT64),
            helper.make_node("Cast", [pri_b], [pb], to=TensorProto.INT64),
            helper.make_node("Less", [sa, sb], [lt]),
            helper.make_node("Equal", [sa, sb], [eq]),
            helper.make_node("Less", [pa, pb], [ple]),
            helper.make_node("And", [eq, ple], [bp]),
            helper.make_node("Or", [lt, bp], [pick_a]),
            helper.make_node("Where", [pick_a, mask_a, mask_b], [mask_out]),
            helper.make_node("Where", [pick_a, steps_a, steps_b], [steps_out]),
            helper.make_node("Where", [pick_a, pa, pb], [pri_out]),
        ]
    )
    return mask_out, steps_out, pri_out


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    half = _f32(inits, [0.5], "half")
    zf = _f32(inits, [0.0], "zf")
    eps = _f32(inits, [EPS], "eps")
    eps_neg = _f32(inits, [-EPS], "eps_neg")
    pri_v = _i64(inits, [0], "pri_v")
    pri_h = _i64(inits, [1], "pri_h")

    ax4 = _i64(inits, [0, 1, 2, 3], "ax4")
    slice_starts = {
        (0, 0, 0, 0): _i64(inits, [0, 0, 0, 0], "s00"),
        (0, 0, 1, 0): _i64(inits, [0, 0, 1, 0], "s10"),
        (0, 0, 0, 1): _i64(inits, [0, 0, 0, 1], "s01"),
    }
    slice_ends = {
        (1, 1, SH, SW): _i64(inits, [1, 1, SH, SW], "eHW"),
        (1, 1, SH + 1, SW): _i64(inits, [1, 1, SH + 1, SW], "eH1W"),
        (1, 1, SH, SW + 1): _i64(inits, [1, 1, SH, SW + 1], "eHW1"),
    }

    rows = _f32(inits, np.arange(SH, dtype=np.float32).reshape(1, 1, SH, 1), "rows")
    cols = _f32(inits, np.arange(SW, dtype=np.float32).reshape(1, 1, 1, SW), "cols")
    sc2 = _i64(inits, [0, RED, 0, 0], "sc2")
    ec2 = _i64(inits, [1, RED + 1, SH, SW], "ec2")
    sc8 = _i64(inits, [0, CYAN, 0, 0], "sc8")
    ec8 = _i64(inits, [1, CYAN + 1, SH, SW], "ec8")
    z30 = _f32(inits, np.zeros((1, 1, H, W), dtype=np.float32), "z30")
    out_pad = [0, 0, 0, 0, 0, 0, H - SH, W - SW]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, sc2, ec2, ax4], ["rsl"]),
            helper.make_node("Slice", [IN_NAME, sc8, ec8, ax4], ["csl"]),
            helper.make_node("Greater", ["rsl", half], ["rb"]),
            helper.make_node("Greater", ["csl", half], ["cb"]),
            helper.make_node("Cast", ["rb"], ["rf"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["cb"], ["cf"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["rf"], ["rff"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["cf"], ["cff"], to=TensorProto.FLOAT),
            helper.make_node("ReduceSum", ["rff"], ["rsum"]),
            helper.make_node("ReduceSum", ["cff"], ["csum"]),
            helper.make_node("Mul", ["rff", rows], ["rry"]),
            helper.make_node("Mul", ["rff", cols], ["rrx"]),
            helper.make_node("Mul", ["cff", rows], ["cry"]),
            helper.make_node("Mul", ["cff", cols], ["crx"]),
            helper.make_node("ReduceSum", ["rry"], ["rsy"]),
            helper.make_node("ReduceSum", ["rrx"], ["rsx"]),
            helper.make_node("ReduceSum", ["cry"], ["csy"]),
            helper.make_node("ReduceSum", ["crx"], ["csx"]),
            helper.make_node("Div", ["rsy", "rsum"], ["ry"]),
            helper.make_node("Div", ["rsx", "rsum"], ["rx"]),
            helper.make_node("Div", ["csy", "csum"], ["cy"]),
            helper.make_node("Div", ["csx", "csum"], ["cx"]),
            helper.make_node("Sub", ["cy", "ry"], ["ddy"]),
            helper.make_node("Sub", ["cx", "rx"], ["ddx"]),
            helper.make_node("Abs", ["ddy"], ["ady"]),
            helper.make_node("Abs", ["ddx"], ["adx"]),
            helper.make_node("Add", ["adx", eps], ["adxe"]),
            helper.make_node("Add", ["ady", eps], ["adye"]),
            helper.make_node("Greater", ["ady", "adxe"], ["vdom"]),
            helper.make_node("Greater", ["adx", "adye"], ["hdom"]),
            helper.make_node("Not", ["vdom"], ["nvdom"]),
            helper.make_node("Not", ["hdom"], ["nhdom"]),
            helper.make_node("And", ["nvdom", "nhdom"], ["tie"]),
            helper.make_node("Greater", ["ddy", eps], ["dypos"]),
            helper.make_node("Less", ["ddy", eps_neg], ["dyneg"]),
            helper.make_node("Greater", ["ddx", eps], ["dxpos"]),
            helper.make_node("Less", ["ddx", eps_neg], ["dxneg"]),
            helper.make_node("Cast", ["dypos"], ["dypf"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["dyneg"], ["dynf"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["dxpos"], ["dxpf"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["dxneg"], ["dxnf"], to=TensorProto.FLOAT),
            helper.make_node("Sub", ["dypf", "dynf"], ["dys"]),
            helper.make_node("Sub", ["dxpf", "dxnf"], ["dxs"]),
        ]
    )

    vmask, sv = _simulate_axis(
        nodes, "rff", "cb", "v", "dys", "dxs", "vp", half, zf, eps, eps_neg, ax4, slice_starts, slice_ends
    )
    hmask, sh = _simulate_axis(
        nodes, "rff", "cb", "h", "dys", "dxs", "hp", half, zf, eps, eps_neg, ax4, slice_starts, slice_ends
    )
    tie_vh, tie_vh_s, tie_vh_p = _prefer_a(nodes, sv, pri_v, sh, pri_h, vmask, hmask, "tvh")
    nodes.extend(
        [
            helper.make_node("Where", ["vdom", vmask, hmask], ["dom_pick"]),
            helper.make_node("Where", ["tie", tie_vh, "dom_pick"], ["rfin"]),
        ]
    )
    nodes.append(helper.make_node("Cast", ["cf"], ["cffout"], to=TensorProto.FLOAT))
    nodes.extend(
        [
            helper.make_node("Greater", ["rfin", half], ["r_on"]),
            helper.make_node("Greater", ["cffout", half], ["c_on"]),
            helper.make_node("Or", ["r_on", "c_on"], ["fg_on"]),
            helper.make_node("Not", ["fg_on"], ["bg_ok"]),
            helper.make_node("Cast", ["bg_ok"], ["ch0f"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", [IN_NAME], ["in_any"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["in_any", half], ["active"]),
        ]
    )
    nodes.append(helper.make_node("Pad", ["rfin"], ["rpad"], pads=out_pad))
    nodes.append(helper.make_node("Pad", ["cffout"], ["cpad"], pads=out_pad))
    nodes.append(helper.make_node("Pad", ["ch0f"], ["ch0out"], pads=out_pad))
    nodes.append(helper.make_node("Where", ["active", "ch0out", z30], ["ch0masked"]))

    chans: List[str] = []
    for c in range(C):
        if c == RED:
            chans.append("rpad")
        elif c == CYAN:
            chans.append("cpad")
        elif c == 0:
            chans.append("ch0masked")
        else:
            chans.append(z30)
    nodes.append(helper.make_node("Concat", chans, [OUT_NAME], axis=1))

    graph = helper.make_graph(nodes, "task008", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = "task008: red slides toward cyan until contact"
    onnx.checker.check_model(model)
    return model


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> Tuple[int, int]:
    if not DATA_PATH.is_file():
        return 0, 0
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            expected = np.array(ex["output"], dtype=np.int64)
            if expected.shape[0] > H or expected.shape[1] > W:
                continue
            total += 1
            oh = _grid_to_onehot(ex["input"])
            pred = _onehot_to_grid(_run_onnx(model, oh))[: expected.shape[0], : expected.shape[1]]
            if not np.array_equal(pred, expected):
                bad += 1
    return bad, total


def main() -> None:
    ref = solve(np.array(TOY_INPUT, dtype=np.int64))
    model = build_model()
    onnx.save(model, BEST_PATH)

    toy_oh = _grid_to_onehot(TOY_INPUT)
    toy_pred = _onehot_to_grid(_run_onnx(model, toy_oh))[: len(TOY_INPUT), : len(TOY_INPUT[0])]
    assert np.array_equal(toy_pred, ref), f"toy mismatch\npred:\n{toy_pred}\nref:\n{ref}"

    bad, total = validate_json(model)
    print("toy: PASS")
    print(f"task008.json: {'PASS' if bad == 0 else f'FAIL ({bad}/{total})'}")

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
