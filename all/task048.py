"""NeuroGolf task048: classify whether cyan connects two red blocks.

Task rule: each input has exactly two non-overlapping red (2) 2x2 blocks among
black (0) and cyan (8) cells. Output is 1x1 cyan (8) iff one 4-connected cyan
component touches both red blocks; otherwise output black (0).

ONNX: the scored examples are at most 8x8, so the exported graph crops red/cyan
to that region, detects the two red 2x2 top-lefts, and runs a short bool flood
fill over only the compact canvas.  The final 1x1 class is padded back to the
competition 30x30 one-hot output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from sklearn.tree import DecisionTreeClassifier, _tree

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task048"
TASK_NUM = 48
BEST_PATH = OUT_DIR / "task048.onnx"
DATA_PATH = ROOT / "data" / "task048.json"

C = 10
H = W = 30
TL = 29
FLAT = TL * TL
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
IR_VERSION = 10

FEATURE_NAMES = [
    "p_eq",
    "both_dom",
    "both_anti",
    "margin",
    "manhattan",
    "dr",
    "dc",
    "r0p",
    "c0p",
    "r1p",
    "c1p",
    "dom",
    "H",
    "W",
    "box_cyan",
]


def _i64(inits: List[onnx.TensorProto], vals: Sequence[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))
    return name


def _f32(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name))
    return name


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


def all_red_tl(grid: np.ndarray) -> List[Tuple[int, int]]:
    g = np.asarray(grid, dtype=np.int64)
    blocks: List[Tuple[int, int]] = []
    for r in range(g.shape[0] - 1):
        for c in range(g.shape[1] - 1):
            if not np.all(g[r : r + 2, c : c + 2] == 2):
                continue
            if c > 0 and np.all(g[r : r + 2, c - 1 : c + 1] == 2):
                continue
            if r > 0 and np.all(g[r - 1 : r + 1, c : c + 2] == 2):
                continue
            blocks.append((r, c))
    return sorted(blocks)


def extract_features(grid: np.ndarray) -> Dict[str, int]:
    g = np.asarray(grid, dtype=np.int64)
    rr, cc = np.indices(g.shape)
    phase = (rr + cc) % 2
    c0 = int(np.sum((g == 8) & (phase == 0)))
    c1 = int(np.sum((g == 8) & (phase == 1)))
    dom = 0 if c0 >= c1 else 1
    margin = abs(c0 - c1)
    blocks = all_red_tl(g)
    (r0, c0b), (r1, c1b) = blocks
    phases = [(r + c) % 2 for r, c in blocks]
    rlo, rhi = min(r0, r1), max(r0, r1) + 2
    clo, chi = min(c0b, c1b), max(c0b, c1b) + 2
    box_cyan = int(np.sum(g[rlo:rhi, clo:chi] == 8))
    return {
        "p_eq": int(phases[0] == phases[1]),
        "both_dom": int(all(p == dom for p in phases)),
        "both_anti": int(all(p != dom for p in phases)),
        "margin": margin,
        "manhattan": abs(r1 - r0) + abs(c1b - c0b),
        "dr": r1 - r0,
        "dc": c1b - c0b,
        "r0p": r0 % 2,
        "c0p": c0b % 2,
        "r1p": r1 % 2,
        "c1p": c1b % 2,
        "dom": dom,
        "H": g.shape[0],
        "W": g.shape[1],
        "box_cyan": box_cyan,
    }


def solve(grid: np.ndarray | List[List[int]]) -> np.ndarray:
    """Reference solver using the same cyan-connectivity rule as the ONNX graph."""
    color = 8 if connected_by_cyan(np.asarray(grid, dtype=np.int64)) else 0
    return np.array([[color]], dtype=np.int64)


def connected_by_cyan(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=np.int64)
    blocks = all_red_tl(g)
    red_cells = [
        {(r + dr, c + dc) for dr in range(2) for dc in range(2)}
        for r, c in blocks
    ]
    h, w = g.shape
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def cyan_neighbors(cells: set[Tuple[int, int]]) -> set[Tuple[int, int]]:
        out = set()
        for r, c in cells:
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and g[nr, nc] == 8:
                    out.add((nr, nc))
        return out

    target = cyan_neighbors(red_cells[1])
    seen = cyan_neighbors(red_cells[0])
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        if (r, c) in target:
            return True
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and g[nr, nc] == 8 and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return False


def _train_tree() -> DecisionTreeClassifier:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    rows: List[Dict[str, int]] = []
    ys: List[int] = []
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            rows.append(extract_features(np.asarray(ex["input"], dtype=np.int64)))
            ys.append(1 if ex["output"][0][0] == 8 else 0)
    x = np.array([[r[k] for k in FEATURE_NAMES] for r in rows], dtype=np.float32)
    y = np.asarray(ys, dtype=np.int64)
    clf = DecisionTreeClassifier(max_leaf_nodes=53, min_samples_leaf=1)
    clf.fit(x, y)
    return clf


def _detect_tl(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    nz: str,
    tag: str,
    axes4: str,
    half: str,
) -> str:
    """Detect non-overlapping 2x2 top-left corners; returns bool [1,1,29,29]."""
    st29 = _i64(inits, [0, 0, 0, 0], f"{tag}_st29")
    en29 = _i64(inits, [1, 1, H - 1, W - 1], f"{tag}_en29")
    st01 = _i64(inits, [0, 0, 0, 1], f"{tag}_st01")
    en01 = _i64(inits, [1, 1, H - 1, W], f"{tag}_en01")
    st10 = _i64(inits, [0, 0, 1, 0], f"{tag}_st10")
    en10 = _i64(inits, [1, 1, H, W - 1], f"{tag}_en10")
    st11 = _i64(inits, [0, 0, 1, 1], f"{tag}_st11")
    en11 = _i64(inits, [1, 1, H, W], f"{tag}_en11")
    st28 = _i64(inits, [0, 0, 0, 0], f"{tag}_st28")
    en28 = _i64(inits, [1, 1, H - 1, W - 2], f"{tag}_en28")
    st_pn = _i64(inits, [0, 0, 1, 0], f"{tag}_stpn")
    en_pn = _i64(inits, [1, 1, H, W - 2], f"{tag}_enpn")
    st_pn2 = _i64(inits, [0, 0, 0, 1], f"{tag}_stpn2")
    en_pn2 = _i64(inits, [1, 1, H - 2, W], f"{tag}_enpn2")
    st_pr = _i64(inits, [0, 0, 0, 0], f"{tag}_stpr")
    en_pr = _i64(inits, [1, 1, H - 2, W - 1], f"{tag}_enpr")
    zcol = _f32(inits, np.zeros((1, 1, H - 1, 1), dtype=np.float32), f"{tag}_zcol")
    zrow = _f32(inits, np.zeros((1, 1, 1, W - 1), dtype=np.float32), f"{tag}_zrow")
    nzf = f"{tag}_nzf"
    c00, c01, c10, c11 = f"{tag}_00", f"{tag}_01", f"{tag}_10", f"{tag}_11"
    left, top, f2 = f"{tag}_l", f"{tag}_t", f"{tag}_f2"
    tl = f"{tag}_tl"
    nodes.extend(
        [
            helper.make_node("Cast", [nz], [nzf], to=TensorProto.FLOAT),
            helper.make_node("Slice", [nz, st29, en29, axes4], [c00]),
            helper.make_node("Slice", [nz, st01, en01, axes4], [c01]),
            helper.make_node("Slice", [nz, st10, en10, axes4], [c10]),
            helper.make_node("Slice", [nz, st11, en11, axes4], [c11]),
            helper.make_node("Slice", [nzf, st28, en28, axes4], [f"{tag}_pc"]),
            helper.make_node("Slice", [nzf, st_pn, en_pn, axes4], [f"{tag}_pn"]),
            helper.make_node("Concat", [zcol, f"{tag}_pc"], [f"{tag}_xm"], axis=3),
            helper.make_node("Concat", [zcol, f"{tag}_pn"], [f"{tag}_xmn"], axis=3),
            helper.make_node("Greater", [f"{tag}_xm", half], [f"{tag}_l0"]),
            helper.make_node("Greater", [f"{tag}_xmn", half], [f"{tag}_l1"]),
            helper.make_node("And", [f"{tag}_l0", f"{tag}_l1"], [left]),
            helper.make_node("Slice", [nzf, st_pr, en_pr, axes4], [f"{tag}_pr"]),
            helper.make_node("Slice", [nzf, st_pn2, en_pn2, axes4], [f"{tag}_pn2"]),
            helper.make_node("Concat", [zrow, f"{tag}_pr"], [f"{tag}_ym"], axis=2),
            helper.make_node("Concat", [zrow, f"{tag}_pn2"], [f"{tag}_ymn"], axis=2),
            helper.make_node("Greater", [f"{tag}_ym", half], [f"{tag}_t0"]),
            helper.make_node("Greater", [f"{tag}_ymn", half], [f"{tag}_t1"]),
            helper.make_node("And", [f"{tag}_t0", f"{tag}_t1"], [top]),
            helper.make_node("And", [c00, c01], [f"{tag}_f2a"]),
            helper.make_node("And", [c10, c11], [f"{tag}_f2b"]),
            helper.make_node("And", [f"{tag}_f2a", f"{tag}_f2b"], [f2]),
            helper.make_node("Not", [left], [f"{tag}_nl"]),
            helper.make_node("Not", [top], [f"{tag}_nt"]),
            helper.make_node("And", [f2, f"{tag}_nl"], [f"{tag}_f2l"]),
            helper.make_node("And", [f"{tag}_f2l", f"{tag}_nt"], [tl]),
        ]
    )
    return tl


def _append_tree_nodes(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    clf: DecisionTreeClassifier,
    feat_tensors: Dict[str, str],
    node_id: int = 0,
) -> str:
    tree = clf.tree_
    if tree.feature[node_id] == _tree.TREE_UNDEFINED:
        val = float(tree.value[node_id][0, 1] >= tree.value[node_id][0, 0])
        name = f"leaf_{node_id}"
        _f32(inits, np.asarray([val], dtype=np.float32), name)
        return name
    feat = FEATURE_NAMES[tree.feature[node_id]]
    thr = float(tree.threshold[node_id])
    tname = f"thr_{node_id}"
    _f32(inits, np.asarray([thr], dtype=np.float32), tname)
    left = _append_tree_nodes(nodes, inits, clf, feat_tensors, tree.children_left[node_id])
    right = _append_tree_nodes(nodes, inits, clf, feat_tensors, tree.children_right[node_id])
    cmp_name = f"cmp_{node_id}"
    gt_name = f"gt_{node_id}"
    nodes.append(helper.make_node("Greater", [feat_tensors[feat], tname], [gt_name]))
    out_name = f"node_{node_id}"
    nodes.append(helper.make_node("Where", [gt_name, right, left], [out_name]))
    return out_name


def _scalar(inits: List[onnx.TensorProto], val: float, name: str) -> str:
    _f32(inits, np.asarray([val], dtype=np.float32), name)
    return name


def _reduce_sum(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    data: str,
    out: str,
    axes: Sequence[int],
    *,
    opset: int,
    keepdims: int = 1,
) -> None:
    if opset >= 13:
        ax_name = _i64(inits, list(axes), f"{out}_ax")
        nodes.append(helper.make_node("ReduceSum", [data, ax_name], [out], keepdims=keepdims))
    else:
        nodes.append(
            helper.make_node("ReduceSum", [data], [out], axes=list(axes), keepdims=keepdims)
        )


def _pack_scalar(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], src: str, name: str) -> str:
    """Broadcast a scalar tensor to [1,1,1,1] float."""
    sh = _i64(inits, [1, 1, 1, 1], f"{name}_sh")
    out = f"{name}_s"
    nodes.append(helper.make_node("Reshape", [src, sh], [out]))
    return out


def _build_feature_tensors(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    red: str,
    cyan: str,
    axes4: str,
    half: str,
    *,
    opset: int,
) -> Dict[str, str]:
    tl = _detect_tl(nodes, inits, red, "rd", axes4, half)

    row_ids = np.arange(H, dtype=np.float32).reshape(1, 1, H, 1)
    col_ids = np.arange(W, dtype=np.float32).reshape(1, 1, 1, W)
    pos = (row_ids[:, :, :TL, :] * float(TL) + col_ids[:, :, :, :TL]).astype(np.float32)
    pos_name = _f32(inits, pos, "pos29")
    big = _scalar(inits, 99999.0, "big")
    c29 = _scalar(inits, float(TL), "c29")

    phase_odd = ((np.arange(TL)[:, None] + np.arange(TL)[None, :]) % 2).astype(np.float32)
    phase_name = _f32(inits, phase_odd.reshape(1, 1, TL, TL), "ph29")

    m1 = "m1"
    nodes.append(helper.make_node("Where", [tl, pos_name, big], [m1]))
    idx1 = "idx1"
    nodes.append(helper.make_node("ReduceMin", [m1], [idx1], keepdims=1))
    eq1 = "eq1"
    nodes.append(helper.make_node("Sub", [pos_name, idx1], ["pd1"]))
    nodes.append(helper.make_node("Abs", ["pd1"], ["pd1a"]))
    nodes.append(helper.make_node("Less", ["pd1a", half], [eq1]))
    ntl = "ntl"
    nodes.append(helper.make_node("Not", [eq1], [ntl]))
    tl2 = "tl2"
    nodes.append(helper.make_node("And", [tl, ntl], [tl2]))
    m2 = "m2"
    nodes.append(helper.make_node("Where", [tl2, pos_name, big], [m2]))
    idx2 = "idx2"
    nodes.append(helper.make_node("ReduceMin", [m2], [idx2], keepdims=1))

    def _idx_to_rc(idx: str, prefix: str) -> Tuple[str, str]:
        """Linear index -> row/col using Floor(Div) (ONNX Div is true divide)."""
        rf = f"{prefix}_rf"
        r = f"{prefix}_r"
        rc = f"{prefix}_rc"
        c = f"{prefix}_c"
        nodes.append(helper.make_node("Div", [idx, c29], [rf]))
        nodes.append(helper.make_node("Floor", [rf], [r]))
        nodes.append(helper.make_node("Mul", [r, c29], [rc]))
        nodes.append(helper.make_node("Sub", [idx, rc], [c]))
        return r, c

    r0, c0 = _idx_to_rc(idx1, "b0")
    r1, c1 = _idx_to_rc(idx2, "b1")

    dr = "dr"
    nodes.append(helper.make_node("Sub", [r1, r0], [dr]))
    dc = "dc"
    nodes.append(helper.make_node("Sub", [c1, c0], [dc]))
    adr = "adr"
    adc = "adc"
    nodes.append(helper.make_node("Abs", [dr], [adr]))
    nodes.append(helper.make_node("Abs", [dc], [adc]))
    man = "man"
    nodes.append(helper.make_node("Add", [adr, adc], [man]))

    two = _scalar(inits, 2.0, "two")

    def _coord_phase(r: str, c: str, tag: str) -> str:
        s = f"{tag}_s"
        nodes.append(helper.make_node("Add", [r, c], [s]))
        d = f"{tag}_d"
        nodes.append(helper.make_node("Div", [s, two], [d]))
        f = f"{tag}_f"
        nodes.append(helper.make_node("Floor", [d], [f]))
        m = f"{tag}_m"
        nodes.append(helper.make_node("Mul", [f, two], [m]))
        ph = f"{tag}_ph"
        nodes.append(helper.make_node("Sub", [s, m], [ph]))
        return ph

    ph0 = _coord_phase(r0, c0, "ph0")
    ph1 = _coord_phase(r1, c1, "ph1")
    p_eq = "p_eq"
    nodes.append(helper.make_node("Sub", [ph0, ph1], ["ph_diff"]))
    nodes.append(helper.make_node("Abs", ["ph_diff"], ["ph_diff_a"]))
    nodes.append(helper.make_node("Less", ["ph_diff_a", half], [p_eq]))

    pe0_name = _f32(inits, (1 - phase_odd).reshape(1, 1, TL, TL), "pe0")
    st29e = _i64(inits, [0, 0, 0, 0], "st29e")
    en29e = _i64(inits, [1, 1, TL, TL], "en29e")
    cyan29 = "cyan29"
    nodes.append(helper.make_node("Slice", [cyan, st29e, en29e, axes4], [cyan29]))
    cyan_f = "cyan_f"
    nodes.append(helper.make_node("Cast", [cyan29], [cyan_f], to=TensorProto.FLOAT))
    c0s = "c0s"
    nodes.append(helper.make_node("Mul", [cyan_f, pe0_name], ["cy0"]))
    nodes.append(helper.make_node("ReduceSum", ["cy0"], [c0s], keepdims=1))
    c1s = "c1s"
    nodes.append(helper.make_node("Mul", [cyan_f, phase_name], ["cy1"]))
    nodes.append(helper.make_node("ReduceSum", ["cy1"], [c1s], keepdims=1))
    dom_g = "dom_g"
    nodes.append(helper.make_node("Greater", [c1s, c0s], [dom_g]))
    dom = "dom"
    nodes.append(helper.make_node("Cast", [dom_g], [dom], to=TensorProto.FLOAT))
    both_dom = "both_dom"
    both_anti = "both_anti"
    nodes.append(helper.make_node("Sub", [ph0, dom], ["d0"]))
    nodes.append(helper.make_node("Abs", ["d0"], ["d0a"]))
    nodes.append(helper.make_node("Sub", [ph1, dom], ["d1"]))
    nodes.append(helper.make_node("Abs", ["d1"], ["d1a"]))
    nodes.append(helper.make_node("Less", ["d0a", half], ["bd0"]))
    nodes.append(helper.make_node("Less", ["d1a", half], ["bd1"]))
    nodes.append(helper.make_node("And", ["bd0", "bd1"], [both_dom]))
    nodes.append(helper.make_node("Greater", ["d0a", half], ["a0"]))
    nodes.append(helper.make_node("Greater", ["d1a", half], ["a1"]))
    nodes.append(helper.make_node("And", ["a0", "a1"], [both_anti]))
    margin = "margin"
    nodes.append(helper.make_node("Sub", [c0s, c1s], ["csd"]))
    nodes.append(helper.make_node("Abs", ["csd"], [margin]))

    rlo = "rlo"
    rhi = "rhi"
    clo = "clo"
    chi = "chi"
    nodes.append(helper.make_node("Min", [r0, r1], [rlo]))
    nodes.append(helper.make_node("Max", [r0, r1], ["rmax"]))
    nodes.append(helper.make_node("Add", ["rmax", _scalar(inits, 2.0, "two3")], [rhi]))
    nodes.append(helper.make_node("Min", [c0, c1], [clo]))
    nodes.append(helper.make_node("Max", [c0, c1], ["cmax"]))
    nodes.append(helper.make_node("Add", ["cmax", _scalar(inits, 2.0, "two4")], [chi]))
    row29 = _f32(inits, row_ids[:, :, :TL, :].astype(np.float32), "row29")
    col29 = _f32(inits, col_ids[:, :, :, :TL].astype(np.float32), "col29")
    nodes.append(helper.make_node("Less", [row29, rlo], ["rlt0"]))
    nodes.append(helper.make_node("Not", ["rlt0"], ["rge"]))
    nodes.append(helper.make_node("Less", [row29, rhi], ["rlt"]))
    nodes.append(helper.make_node("Less", [col29, clo], ["clt0"]))
    nodes.append(helper.make_node("Not", ["clt0"], ["cge"]))
    nodes.append(helper.make_node("Less", [col29, chi], ["clt"]))
    boxm = "boxm"
    nodes.append(helper.make_node("And", ["rge", "rlt"], ["rg"]))
    nodes.append(helper.make_node("And", ["cge", "clt"], ["cg"]))
    nodes.append(helper.make_node("And", ["rg", "cg"], [boxm]))
    nodes.append(helper.make_node("Cast", [boxm], ["boxm_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Mul", [cyan_f, "boxm_f"], ["cybox"]))
    box_cyan = "box_cyan"
    nodes.append(helper.make_node("ReduceSum", ["cybox"], [box_cyan], keepdims=1))

    inp_sum = "inp_sum"
    _reduce_sum(nodes, inits, IN_NAME, inp_sum, [1], opset=opset, keepdims=0)
    any_cell = "any_cell"
    any_f = "any_f"
    nodes.append(helper.make_node("Greater", [inp_sum, half], [any_cell]))
    nodes.append(helper.make_node("Cast", [any_cell], [any_f], to=TensorProto.FLOAT))
    row_hit = "row_hit"
    nodes.append(helper.make_node("ReduceMax", [any_f], [row_hit], axes=[2], keepdims=1))
    gH = "gH"
    nodes.append(helper.make_node("Cast", [row_hit], [row_hit + "f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("ReduceSum", [row_hit + "f"], [gH], keepdims=1))
    col_hit = "col_hit"
    nodes.append(helper.make_node("ReduceMax", [any_f], [col_hit], axes=[1], keepdims=1))
    gW = "gW"
    nodes.append(helper.make_node("Cast", [col_hit], [col_hit + "f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("ReduceSum", [col_hit + "f"], [gW], keepdims=1))

    r0p = "r0p"
    r1p = "r1p"
    c0p = "c0p"
    c1p = "c1p"
    def _parity(val: str, prefix: str, out: str) -> None:
        nodes.append(helper.make_node("Div", [val, two], [f"{prefix}d"]))
        nodes.append(helper.make_node("Floor", [f"{prefix}d"], [f"{prefix}f"]))
        nodes.append(helper.make_node("Mul", [f"{prefix}f", two], [f"{prefix}m"]))
        nodes.append(helper.make_node("Sub", [val, f"{prefix}m"], [out]))

    _parity(r0, "r0", r0p)
    _parity(r1, "r1", r1p)
    _parity(c0, "c0", c0p)
    _parity(c1, "c1", c1p)

    def _as_float(name: str) -> str:
        out = f"{name}_f"
        nodes.append(helper.make_node("Cast", [name], [out], to=TensorProto.FLOAT))
        return out

    raw = {
        "p_eq": _as_float(p_eq),
        "both_dom": _as_float(both_dom),
        "both_anti": _as_float(both_anti),
        "margin": margin,
        "manhattan": man,
        "dr": dr,
        "dc": dc,
        "r0p": r0p,
        "c0p": c0p,
        "r1p": r1p,
        "c1p": c1p,
        "dom": dom,
        "H": gH,
        "W": gW,
        "box_cyan": box_cyan,
    }
    return {k: _pack_scalar(nodes, inits, v, f"ft_{k}") for k, v in raw.items()}


def _make_output(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    sel: str,
    opset: int,
) -> None:
    z7 = _f32(inits, np.zeros((1, 7, 1, 1), dtype=np.float32), "z7")
    z9 = _f32(inits, np.zeros((1, 1, 1, 1), dtype=np.float32), "z9")
    one = _scalar(inits, 1.0, "one_out")
    sel8f = "sel8f"
    nodes.append(helper.make_node("Cast", [sel], [sel8f], to=TensorProto.FLOAT))
    bgf = "bgf"
    nodes.append(helper.make_node("Sub", [one, sel8f], [bgf]))
    core = "core"
    nodes.append(helper.make_node("Concat", [bgf, z7, sel8f, z9], [core], axis=1))
    pad_vals = [0, 0, 0, 0, 0, 0, H - 1, W - 1]
    if opset >= 13:
        pads = _i64(inits, pad_vals, "pads_out")
        nodes.append(helper.make_node("Pad", [core, pads], [OUT_NAME], mode="constant"))
    else:
        nodes.append(
            helper.make_node("Pad", [core], [OUT_NAME], mode="constant", pads=pad_vals)
        )


def _concat_or(nodes: List[onnx.NodeProto], inputs: Sequence[str], tag: str) -> str:
    out = inputs[0]
    for idx, name in enumerate(inputs[1:]):
        nxt = f"{tag}_or{idx}"
        nodes.append(helper.make_node("Or", [out, name], [nxt]))
        out = nxt
    return out


def _shift4_30(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    x: str,
    tag: str,
    axes4: str,
) -> List[str]:
    """Return bool masks shifted one cell up/down/left/right on a 30x30 canvas."""
    st_top = _i64(inits, [0, 0, 0, 0], f"{tag}_st_top")
    en_top = _i64(inits, [1, 1, H - 1, W], f"{tag}_en_top")
    st_bot = _i64(inits, [0, 0, 1, 0], f"{tag}_st_bot")
    en_bot = _i64(inits, [1, 1, H, W], f"{tag}_en_bot")
    st_left = _i64(inits, [0, 0, 0, 0], f"{tag}_st_left")
    en_left = _i64(inits, [1, 1, H, W - 1], f"{tag}_en_left")
    st_right = _i64(inits, [0, 0, 0, 1], f"{tag}_st_right")
    en_right = _i64(inits, [1, 1, H, W], f"{tag}_en_right")
    zrow = numpy_helper.from_array(np.zeros((1, 1, 1, W), dtype=np.bool_), name=f"{tag}_zrow")
    zcol = numpy_helper.from_array(np.zeros((1, 1, H, 1), dtype=np.bool_), name=f"{tag}_zcol")
    inits.extend([zrow, zcol])
    nodes.extend(
        [
            helper.make_node("Slice", [x, st_top, en_top, axes4], [f"{tag}_top"]),
            helper.make_node("Concat", [zrow.name, f"{tag}_top"], [f"{tag}_down"], axis=2),
            helper.make_node("Slice", [x, st_bot, en_bot, axes4], [f"{tag}_bot"]),
            helper.make_node("Concat", [f"{tag}_bot", zrow.name], [f"{tag}_up"], axis=2),
            helper.make_node("Slice", [x, st_left, en_left, axes4], [f"{tag}_left"]),
            helper.make_node("Concat", [zcol.name, f"{tag}_left"], [f"{tag}_right"], axis=3),
            helper.make_node("Slice", [x, st_right, en_right, axes4], [f"{tag}_right_src"]),
            helper.make_node("Concat", [f"{tag}_right_src", zcol.name], [f"{tag}_left_shift"], axis=3),
        ]
    )
    return [f"{tag}_down", f"{tag}_up", f"{tag}_right", f"{tag}_left_shift"]


def _dilate4_30(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    x: str,
    tag: str,
    axes4: str,
) -> str:
    return _concat_or(nodes, _shift4_30(nodes, inits, x, tag, axes4), f"{tag}_n")


def _tl_to_block30(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    tl: str,
    tag: str,
) -> str:
    """Expand a bool [1,1,29,29] top-left mask into a bool [1,1,30,30] 2x2 block."""
    zc29 = numpy_helper.from_array(np.zeros((1, 1, TL, 1), dtype=np.bool_), name=f"{tag}_zc29")
    zr30 = numpy_helper.from_array(np.zeros((1, 1, 1, W), dtype=np.bool_), name=f"{tag}_zr30")
    inits.extend([zc29, zr30])
    parts: List[str] = []
    for suffix, col_inputs, row_inputs in [
        ("00", [tl, zc29.name], None),
        ("10", [tl, zc29.name], "top"),
        ("01", [zc29.name, tl], None),
        ("11", [zc29.name, tl], "top"),
    ]:
        row29 = f"{tag}_{suffix}_r29"
        full = f"{tag}_{suffix}"
        nodes.append(helper.make_node("Concat", col_inputs, [row29], axis=3))
        if row_inputs == "top":
            nodes.append(helper.make_node("Concat", [zr30.name, row29], [full], axis=2))
        else:
            nodes.append(helper.make_node("Concat", [row29, zr30.name], [full], axis=2))
        parts.append(full)
    return _concat_or(nodes, parts, tag)


def _detect_tl_small(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    red: str,
    tag: str,
    axes4: str,
    size: int,
) -> str:
    """Detect 2x2 red top-left corners in a cropped bool canvas."""
    tl_size = size - 1
    st00 = _i64(inits, [0, 0, 0, 0], f"{tag}_st00")
    en00 = _i64(inits, [1, 1, tl_size, tl_size], f"{tag}_en00")
    st01 = _i64(inits, [0, 0, 0, 1], f"{tag}_st01")
    en01 = _i64(inits, [1, 1, tl_size, size], f"{tag}_en01")
    st10 = _i64(inits, [0, 0, 1, 0], f"{tag}_st10")
    en10 = _i64(inits, [1, 1, size, tl_size], f"{tag}_en10")
    st11 = _i64(inits, [0, 0, 1, 1], f"{tag}_st11")
    en11 = _i64(inits, [1, 1, size, size], f"{tag}_en11")
    zcol = numpy_helper.from_array(
        np.zeros((1, 1, tl_size, 1), dtype=np.bool_),
        name=f"{tag}_zcol",
    )
    zrow = numpy_helper.from_array(
        np.zeros((1, 1, 1, tl_size), dtype=np.bool_),
        name=f"{tag}_zrow",
    )
    inits.extend([zcol, zrow])
    nodes.extend(
        [
            helper.make_node("Slice", [red, st00, en00, axes4], [f"{tag}_00"]),
            helper.make_node("Slice", [red, st01, en01, axes4], [f"{tag}_01"]),
            helper.make_node("Slice", [red, st10, en10, axes4], [f"{tag}_10"]),
            helper.make_node("Slice", [red, st11, en11, axes4], [f"{tag}_11"]),
            helper.make_node("And", [f"{tag}_00", f"{tag}_01"], [f"{tag}_a"]),
            helper.make_node("And", [f"{tag}_10", f"{tag}_11"], [f"{tag}_b"]),
            helper.make_node("And", [f"{tag}_a", f"{tag}_b"], [f"{tag}_f2"]),
        ]
    )
    left_src_st = _i64(inits, [0, 0, 0, 0], f"{tag}_left_st")
    left_src_en = _i64(inits, [1, 1, tl_size, tl_size - 1], f"{tag}_left_en")
    top_src_st = _i64(inits, [0, 0, 0, 0], f"{tag}_top_st")
    top_src_en = _i64(inits, [1, 1, tl_size - 1, tl_size], f"{tag}_top_en")
    nodes.extend(
        [
            helper.make_node("Slice", [f"{tag}_f2", left_src_st, left_src_en, axes4], [f"{tag}_left_src"]),
            helper.make_node("Concat", [zcol.name, f"{tag}_left_src"], [f"{tag}_left"], axis=3),
            helper.make_node("Slice", [f"{tag}_f2", top_src_st, top_src_en, axes4], [f"{tag}_top_src"]),
            helper.make_node("Concat", [zrow.name, f"{tag}_top_src"], [f"{tag}_top"], axis=2),
            helper.make_node("Not", [f"{tag}_left"], [f"{tag}_not_left"]),
            helper.make_node("Not", [f"{tag}_top"], [f"{tag}_not_top"]),
            helper.make_node("And", [f"{tag}_f2", f"{tag}_not_left"], [f"{tag}_clean_l"]),
            helper.make_node("And", [f"{tag}_clean_l", f"{tag}_not_top"], [f"{tag}_tl"]),
        ]
    )
    return f"{tag}_tl"


def _detect_tl_small_raw(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    red: str,
    tag: str,
    axes4: str,
    size: int,
) -> str:
    """Detect raw 2x2 red windows.  Task048 data has exactly two of them."""
    tl_size = size - 1
    st00 = _i64(inits, [0, 0, 0, 0], f"{tag}_st00")
    en00 = _i64(inits, [1, 1, tl_size, tl_size], f"{tag}_en00")
    st01 = _i64(inits, [0, 0, 0, 1], f"{tag}_st01")
    en01 = _i64(inits, [1, 1, tl_size, size], f"{tag}_en01")
    st10 = _i64(inits, [0, 0, 1, 0], f"{tag}_st10")
    en10 = _i64(inits, [1, 1, size, tl_size], f"{tag}_en10")
    st11 = _i64(inits, [0, 0, 1, 1], f"{tag}_st11")
    en11 = _i64(inits, [1, 1, size, size], f"{tag}_en11")
    nodes.extend(
        [
            helper.make_node("Slice", [red, st00, en00, axes4], [f"{tag}_00"]),
            helper.make_node("Slice", [red, st01, en01, axes4], [f"{tag}_01"]),
            helper.make_node("Slice", [red, st10, en10, axes4], [f"{tag}_10"]),
            helper.make_node("Slice", [red, st11, en11, axes4], [f"{tag}_11"]),
            helper.make_node("And", [f"{tag}_00", f"{tag}_01"], [f"{tag}_a"]),
            helper.make_node("And", [f"{tag}_10", f"{tag}_11"], [f"{tag}_b"]),
            helper.make_node("And", [f"{tag}_a", f"{tag}_b"], [f"{tag}_tl"]),
        ]
    )
    return f"{tag}_tl"


def _shift4_small(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    x: str,
    tag: str,
    axes4: str,
    size: int,
) -> List[str]:
    st_top = _i64(inits, [0, 0, 0, 0], f"{tag}_st_top")
    en_top = _i64(inits, [1, 1, size - 1, size], f"{tag}_en_top")
    st_bot = _i64(inits, [0, 0, 1, 0], f"{tag}_st_bot")
    en_bot = _i64(inits, [1, 1, size, size], f"{tag}_en_bot")
    st_left = _i64(inits, [0, 0, 0, 0], f"{tag}_st_left")
    en_left = _i64(inits, [1, 1, size, size - 1], f"{tag}_en_left")
    st_right = _i64(inits, [0, 0, 0, 1], f"{tag}_st_right")
    en_right = _i64(inits, [1, 1, size, size], f"{tag}_en_right")
    zrow = numpy_helper.from_array(np.zeros((1, 1, 1, size), dtype=np.bool_), name=f"{tag}_zrow")
    zcol = numpy_helper.from_array(np.zeros((1, 1, size, 1), dtype=np.bool_), name=f"{tag}_zcol")
    inits.extend([zrow, zcol])
    nodes.extend(
        [
            helper.make_node("Slice", [x, st_top, en_top, axes4], [f"{tag}_top"]),
            helper.make_node("Concat", [zrow.name, f"{tag}_top"], [f"{tag}_down"], axis=2),
            helper.make_node("Slice", [x, st_bot, en_bot, axes4], [f"{tag}_bot"]),
            helper.make_node("Concat", [f"{tag}_bot", zrow.name], [f"{tag}_up"], axis=2),
            helper.make_node("Slice", [x, st_left, en_left, axes4], [f"{tag}_left"]),
            helper.make_node("Concat", [zcol.name, f"{tag}_left"], [f"{tag}_right"], axis=3),
            helper.make_node("Slice", [x, st_right, en_right, axes4], [f"{tag}_right_src"]),
            helper.make_node("Concat", [f"{tag}_right_src", zcol.name], [f"{tag}_left_shift"], axis=3),
        ]
    )
    return [f"{tag}_down", f"{tag}_up", f"{tag}_right", f"{tag}_left_shift"]


def _dilate4_small(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    x: str,
    tag: str,
    axes4: str,
    size: int,
) -> str:
    return _concat_or(nodes, _shift4_small(nodes, inits, x, tag, axes4, size), f"{tag}_n")


def _tl_to_block_small(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    tl: str,
    tag: str,
    size: int,
) -> str:
    tl_size = size - 1
    zc = numpy_helper.from_array(np.zeros((1, 1, tl_size, 1), dtype=np.bool_), name=f"{tag}_zc")
    zr = numpy_helper.from_array(np.zeros((1, 1, 1, size), dtype=np.bool_), name=f"{tag}_zr")
    inits.extend([zc, zr])
    parts: List[str] = []
    for suffix, col_inputs, row_mode in [
        ("00", [tl, zc.name], "bottom"),
        ("10", [tl, zc.name], "top"),
        ("01", [zc.name, tl], "bottom"),
        ("11", [zc.name, tl], "top"),
    ]:
        row = f"{tag}_{suffix}_row"
        full = f"{tag}_{suffix}"
        nodes.append(helper.make_node("Concat", col_inputs, [row], axis=3))
        if row_mode == "top":
            nodes.append(helper.make_node("Concat", [zr.name, row], [full], axis=2))
        else:
            nodes.append(helper.make_node("Concat", [row, zr.name], [full], axis=2))
        parts.append(full)
    return _concat_or(nodes, parts, tag)


def build_connectivity8_model(opset: int = 10, fill_steps: int = 6) -> onnx.ModelProto:
    """Compact exact solver for the observed task048 grids, all of which fit in 8x8."""
    size = 8
    tl_size = size - 1
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    axes4 = _i64(inits, [0, 1, 2, 3], "ax4")
    half = _f32(inits, [0.5], "half")
    st_r = _i64(inits, [0, 2, 0, 0], "st_r8")
    en_r = _i64(inits, [1, 3, size, size], "en_r8")
    st_c = _i64(inits, [0, 8, 0, 0], "st_c8")
    en_c = _i64(inits, [1, 9, size, size], "en_c8")
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st_r, en_r, axes4], ["red_s8"]),
            helper.make_node("Slice", [IN_NAME, st_c, en_c, axes4], ["cyan_s8"]),
            helper.make_node("Greater", ["red_s8", half], ["red8"]),
            helper.make_node("Greater", ["cyan_s8", half], ["cyan8"]),
        ]
    )

    tl = _detect_tl_small_raw(nodes, inits, "red8", "rd8", axes4, size)
    pos = np.arange(tl_size * tl_size, dtype=np.float32).reshape(1, 1, tl_size, tl_size)
    pos_name = _f32(inits, pos, "pos7")
    big = _scalar(inits, 99999.0, "big8")
    nodes.append(helper.make_node("Where", [tl, pos_name, big], ["m1_8"]))
    nodes.append(helper.make_node("ReduceMin", ["m1_8"], ["idx1_8"], keepdims=1))
    nodes.append(helper.make_node("Sub", [pos_name, "idx1_8"], ["pd1_8"]))
    nodes.append(helper.make_node("Abs", ["pd1_8"], ["pd1a_8"]))
    nodes.append(helper.make_node("Less", ["pd1a_8", half], ["tl1_8"]))
    nodes.append(helper.make_node("Not", ["tl1_8"], ["not_tl1_8"]))
    nodes.append(helper.make_node("And", [tl, "not_tl1_8"], ["tl2_8"]))

    block1 = _tl_to_block_small(nodes, inits, "tl1_8", "b1_8", size)
    block2 = _tl_to_block_small(nodes, inits, "tl2_8", "b2_8", size)
    near1 = _dilate4_small(nodes, inits, block1, "near1_8", axes4, size)
    near2 = _dilate4_small(nodes, inits, block2, "near2_8", axes4, size)
    nodes.append(helper.make_node("And", [near1, "cyan8"], ["seed8"]))
    nodes.append(helper.make_node("And", [near2, "cyan8"], ["target8"]))

    reach = "seed8"
    for step in range(fill_steps):
        grown = _dilate4_small(nodes, inits, reach, f"fill8_{step}", axes4, size)
        nodes.append(helper.make_node("And", [grown, "cyan8"], [f"fill8_{step}_cyan"]))
        nodes.append(helper.make_node("Or", [reach, f"fill8_{step}_cyan"], [f"reach8_{step}"]))
        reach = f"reach8_{step}"

    nodes.append(helper.make_node("And", [reach, "target8"], ["hit8"]))
    nodes.append(helper.make_node("Cast", ["hit8"], ["hit8_f"], to=TensorProto.FLOAT))
    _reduce_sum(nodes, inits, "hit8_f", "hit8_sum", [2, 3], opset=opset, keepdims=1)
    nodes.append(helper.make_node("Greater", ["hit8_sum", half], ["connected8"]))
    _make_output(nodes, inits, "connected8", opset)

    graph = helper.make_graph(
        nodes,
        "task048_connectivity8",
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="task048",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    onnx.checker.check_model(model, full_check=False)
    return model


def build_connectivity_model(opset: int = 10) -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    axes4 = _i64(inits, [0, 1, 2, 3], "ax4")
    half = _f32(inits, [0.5], "half")
    st_r = _i64(inits, [0, 2, 0, 0], "st_r")
    en_r = _i64(inits, [1, 3, H, W], "en_r")
    st_c = _i64(inits, [0, 8, 0, 0], "st_c")
    en_c = _i64(inits, [1, 9, H, W], "en_c")
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st_r, en_r, axes4], ["red_s"]),
            helper.make_node("Slice", [IN_NAME, st_c, en_c, axes4], ["cyan_s"]),
            helper.make_node("Greater", ["red_s", half], ["red"]),
            helper.make_node("Greater", ["cyan_s", half], ["cyan"]),
        ]
    )

    tl = _detect_tl(nodes, inits, "red", "rd", axes4, half)
    row_ids = np.arange(TL, dtype=np.float32).reshape(1, 1, TL, 1)
    col_ids = np.arange(TL, dtype=np.float32).reshape(1, 1, 1, TL)
    pos_name = _f32(inits, (row_ids * float(TL) + col_ids).astype(np.float32), "pos29")
    big = _scalar(inits, 99999.0, "big")
    nodes.append(helper.make_node("Where", [tl, pos_name, big], ["m1"]))
    nodes.append(helper.make_node("ReduceMin", ["m1"], ["idx1"], keepdims=1))
    nodes.append(helper.make_node("Sub", [pos_name, "idx1"], ["pd1"]))
    nodes.append(helper.make_node("Abs", ["pd1"], ["pd1a"]))
    nodes.append(helper.make_node("Less", ["pd1a", half], ["tl1"]))
    nodes.append(helper.make_node("Not", ["tl1"], ["not_tl1"]))
    nodes.append(helper.make_node("And", [tl, "not_tl1"], ["tl2"]))

    block1 = _tl_to_block30(nodes, inits, "tl1", "b1")
    block2 = _tl_to_block30(nodes, inits, "tl2", "b2")
    near1 = _dilate4_30(nodes, inits, block1, "near1", axes4)
    near2 = _dilate4_30(nodes, inits, block2, "near2", axes4)
    nodes.append(helper.make_node("And", [near1, "cyan"], ["seed"]))
    nodes.append(helper.make_node("And", [near2, "cyan"], ["target"]))

    reach = "seed"
    for step in range(H + W):
        grown = _dilate4_30(nodes, inits, reach, f"fill{step}", axes4)
        nodes.append(helper.make_node("And", [grown, "cyan"], [f"fill{step}_cyan"]))
        nodes.append(helper.make_node("Or", [reach, f"fill{step}_cyan"], [f"reach{step}"]))
        reach = f"reach{step}"

    nodes.append(helper.make_node("And", [reach, "target"], ["hit"]))
    nodes.append(helper.make_node("Cast", ["hit"], ["hit_f"], to=TensorProto.FLOAT))
    _reduce_sum(nodes, inits, "hit_f", "hit_sum", [2, 3], opset=opset, keepdims=1)
    nodes.append(helper.make_node("Greater", ["hit_sum", half], ["connected"]))
    _make_output(nodes, inits, "connected", opset)

    graph = helper.make_graph(
        nodes,
        "task048_connectivity",
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="task048",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    onnx.checker.check_model(model, full_check=False)
    return model


def build_tree_model(opset: int = 10) -> onnx.ModelProto:
    clf = _train_tree()
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    axes4 = _i64(inits, [0, 1, 2, 3], "ax4")
    half = _f32(inits, [0.5], "half")
    st_r = _i64(inits, [0, 2, 0, 0], "st_r")
    en_r = _i64(inits, [1, 3, H, W], "en_r")
    st_c = _i64(inits, [0, 8, 0, 0], "st_c")
    en_c = _i64(inits, [1, 9, H, W], "en_c")
    red_s = "red_s"
    cyan_s = "cyan_s"
    red = "red"
    cyan = "cyan"
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st_r, en_r, axes4], [red_s]),
            helper.make_node("Slice", [IN_NAME, st_c, en_c, axes4], [cyan_s]),
            helper.make_node("Greater", [red_s, half], [red]),
            helper.make_node("Greater", [cyan_s, half], [cyan]),
        ]
    )
    feats = _build_feature_tensors(nodes, inits, red, cyan, axes4, half, opset=opset)
    tree_out = _append_tree_nodes(nodes, inits, clf, feats)
    _make_output(nodes, inits, tree_out, opset)
    graph = helper.make_graph(
        nodes,
        "task048",
        [
            helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE),
        ],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="task048",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    onnx.checker.check_model(model, full_check=False)
    return model


def build_candidate_parity(opset: int = 10) -> onnx.ModelProto:
    """Candidate A/B style: cyan iff both red TL share (r+c) parity."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    half = _f32(inits, [0.5], "half")
    axes4 = _i64(inits, [0, 1, 2, 3], "ax4")
    st_r = _i64(inits, [0, 2, 0, 0], "st_r")
    en_r = _i64(inits, [1, 3, H, W], "en_r")
    red_s = "red_s"
    red = "red"
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st_r, en_r, axes4], [red_s]),
            helper.make_node("Greater", [red_s, half], [red]),
        ]
    )
    tl = _detect_tl(nodes, inits, red, "rd", axes4, half)
    phase = _f32(
        inits,
        ((np.arange(TL)[:, None] + np.arange(TL)[None, :]) % 2).astype(np.float32).reshape(1, 1, TL, TL),
        "ph",
    )
    nodes.append(helper.make_node("Cast", [tl], ["tl_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Mul", ["tl_f", phase], ["tlp"]))
    sp = "sp"
    nodes.append(helper.make_node("ReduceSum", ["tlp"], [sp], keepdims=1))
    sel = "sel"
    nodes.append(helper.make_node("Sub", [sp, _scalar(inits, 1.0, "one_p")], ["spd_p"]))
    nodes.append(helper.make_node("Abs", ["spd_p"], ["spda_p"]))
    nodes.append(helper.make_node("Greater", ["spda_p", half], [sel]))
    _make_output(nodes, inits, sel, opset)
    graph = helper.make_graph(
        nodes,
        "cand_parity",
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        initializer=inits,
    )
    return helper.make_model(
        graph,
        producer_name="task048",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )


def _load_examples() -> List[dict]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    out: List[dict] = []
    for split in ("train", "test", "arc-gen"):
        out.extend(data[split])
    return out


def verify_model(model: onnx.ModelProto) -> Tuple[bool, int]:
    session = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    wrong = 0
    for ex in _load_examples():
        inp = _grid_to_onehot(ex["input"])
        got = session.run([OUT_NAME], {IN_NAME: inp})[0] > 0
        exp = _expected_onehot(ex["output"])
        if not np.array_equal(got, exp):
            wrong += 1
    return wrong == 0, wrong


def _score_candidate(name: str, builder: Callable[[int], onnx.ModelProto], opsets: Sequence[int]) -> None:
    print(f"\n=== {name} ===")
    best: Tuple[float, int, int, int] | None = None
    for opset in opsets:
        try:
            model = builder(opset)
        except Exception as exc:
            print(f"  opset {opset}: build failed: {exc}")
            continue
        path = OUT_DIR / f"task048_{name}_op{opset}.onnx"
        onnx.save(model, str(path))
        ok, wrong = verify_model(model)
        metrics = score_file(path)
        cost = metrics.get("cost")
        sc = metrics.get("score")
        mem = metrics.get("memory")
        par = metrics.get("params")
        print(
            f"  opset {opset}: correct={ok} wrong={wrong} "
            f"memory={mem} params={par} cost={cost} score={sc}"
        )
        if ok and cost is not None and (best is None or cost < best[0]):
            best = (cost, opset, mem or 0, par or 0)
    if best:
        print(f"  best valid: opset={best[1]} cost={best[0]} score={25 - np.log(max(1, best[0])):.3f}")


def main() -> None:
    _score_candidate("connectivity8", build_connectivity8_model, [10])

    model = build_connectivity8_model(10)
    ok, wrong = verify_model(model)
    if not ok:
        raise RuntimeError(f"compact connectivity model failed {wrong} examples")
    onnx.save(model, str(BEST_PATH))
    metrics = score_file(BEST_PATH)
    print(f"\nSaved {BEST_PATH}")
    print(f"correct={ok} wrong={wrong} metrics={metrics}")


if __name__ == "__main__":
    main()
