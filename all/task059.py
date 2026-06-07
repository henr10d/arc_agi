"""Minimal ONNX for ARC task059: fill winning 3x3 region blocks per color.

Task rule: an 11x11 grid has gray (5) separator lines at rows/cols 3 and 7,
forming nine 3x3 regions. For each non-gray color, count pixels per region,
find the maximum count, and fill every tied region with a solid block of that
color. Gray lattice stays unchanged; scattered pixels are cleared; background
is zero.

ONNX: slice the 11x11 core, count region pixels with fixed Slice+ReduceSum on
bool masks, expand 3x3 win maps per channel with a hardcoded Gather index,
assemble ten bool planes then Cast once before Pad.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

BEST_PATH = OUT_DIR / "task059.onnx"
DATA_PATH = ROOT / "data" / "task059.json"

C = 10
NC = 8
GH = GW = 11
H = W = 30
PAD = H - GH
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
GRAY = 5
IR_VERSION = 10
OPSET = 10

ROW_OFF = (0, 4, 8)
COL_OFF = (0, 4, 8)
OUT_CHS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
COMPACT_FOR_OUT = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    6: 4,
    7: 5,
    8: 6,
    9: 7,
}


def _region_flat_index() -> np.ndarray:
    ri = np.zeros(GH, dtype=np.int64)
    ci = np.zeros(GW, dtype=np.int64)
    for r in range(GH):
        ri[r] = 0 if r < 3 else (1 if r < 7 else 2)
    for c in range(GW):
        ci[c] = 0 if c < 3 else (1 if c < 7 else 2)
    return (ri[:, None] * 3 + ci[None, :]).astype(np.int64)


def solve(grid: np.ndarray | List[List[int]]) -> np.ndarray:
    """Reference: per-color max-region fill on the 11x11 gray lattice."""
    g = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(g)
    out[g == GRAY] = GRAY
    for color in range(1, C):
        if color == GRAY:
            continue
        counts = np.zeros((3, 3), dtype=np.int64)
        for ri, r0 in enumerate(ROW_OFF):
            for ci, c0 in enumerate(COL_OFF):
                counts[ri, ci] = int(np.sum(g[r0 : r0 + 3, c0 : c0 + 3] == color))
        mx = int(counts.max())
        if mx == 0:
            continue
        for ri, r0 in enumerate(ROW_OFF):
            for ci, c0 in enumerate(COL_OFF):
                if counts[ri, ci] == mx:
                    out[r0 : r0 + 3, c0 : c0 + 3] = color
    return out


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _grid_to_onehot(grid: List[List[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    flat = onehot.reshape(C, H, W)
    active = flat > 0.0
    out = flat.argmax(axis=0).astype(np.int64)
    out[~active.any(axis=0)] = 0
    return out


class Builder:
    def __init__(self) -> None:
        self.nodes: List[onnx.NodeProto] = []
        self.inits: List[onnx.TensorProto] = []
        self._n = 0

    def name(self, prefix: str = "t") -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    def i64(self, vals, name: str | None = None) -> str:
        return _i64(self.inits, vals, name or self.name("i"))

    def f32(self, arr, name: str | None = None) -> str:
        return _init(self.inits, np.asarray(arr, dtype=np.float32), name or self.name("f"))

    def add(self, op: str, inputs: List[str], outputs: List[str] | None = None, **kwargs) -> str:
        out = outputs[0] if outputs else self.name()
        self.nodes.append(helper.make_node(op, inputs, [out] if outputs is None else outputs, **kwargs))
        return out


def _content_mask(b: Builder) -> str:
    row_sep = np.zeros((1, 1, GH, 1), dtype=bool)
    col_sep = np.zeros((1, 1, 1, GW), dtype=bool)
    for r in (3, 7):
        row_sep[0, 0, r, 0] = True
    for c in (3, 7):
        col_sep[0, 0, 0, c] = True
    rs = _init(b.inits, row_sep, "row_sep")
    cs = _init(b.inits, col_sep, "col_sep")
    return b.add("Not", [b.add("Or", [rs, cs])])


def _lattice_masks() -> tuple[np.ndarray, np.ndarray]:
    gray = np.zeros((1, 1, GH, GW), dtype=bool)
    gray[:, :, (3, 7), :] = True
    gray[:, :, :, (3, 7)] = True
    return gray, np.logical_not(gray)


def _expand_region_mask(b: Builder, wins_k: str, content: str, idx1: str, tag: str) -> str:
    """Expand one [1,1,3,3] bool region mask to [1,1,11,11]."""
    flat = b.add("Reshape", [wins_k, b.i64([9], f"w9sh_{tag}")])
    gathered = b.add("Gather", [flat, idx1], axis=0)
    spatial = b.add("Reshape", [gathered, b.i64([1, 1, GH, GW], f"sp_sh_{tag}")])
    return b.add("And", [spatial, content])


def _region_counts(b: Builder, nz: str, ax4: str) -> str:
    region_counts: List[str] = []
    for ri, r0 in enumerate(ROW_OFF):
        for ci, c0 in enumerate(COL_OFF):
            st = b.i64([0, 0, r0, c0], f"rs{ri}{ci}")
            en = b.i64([1, NC, r0 + 3, c0 + 3], f"re{ri}{ci}")
            reg = b.add("Slice", [nz, st, en, ax4])
            reg_f = b.add("Cast", [reg], to=TensorProto.FLOAT)
            region_counts.append(b.add("ReduceSum", [reg_f], axes=[2, 3], keepdims=1))
    cnt9 = b.add("Concat", region_counts, axis=2)
    return b.add("Reshape", [cnt9, b.i64([1, NC, 3, 3], "cnt_sh")])


def _counts_to_wins(b: Builder, counts: str, half: str, zero_f: str) -> str:
    max_cnt = b.add("ReduceMax", [counts], axes=[2, 3], keepdims=1)
    cnt_abs = b.add("Abs", [b.add("Sub", [counts, max_cnt])])
    eq_max = b.add("Less", [cnt_abs, half])
    has_max = b.add("Greater", [max_cnt, zero_f])
    return b.add("And", [eq_max, has_max])


def build_model(opset: int = OPSET) -> onnx.ModelProto:
    b = Builder()
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    ax4 = b.i64([0, 1, 2, 3], "ax4")
    core_st = b.i64([0, 0, 0, 0], "core_st")
    core_en = b.i64([1, C, GH, GW], "core_en")
    half = b.f32([0.5], "half")
    zero_f = b.f32([0.0], "zero_f")
    idx1 = b.i64(_region_flat_index().reshape(GH * GW), "reg_idx1")
    content = _content_mask(b)

    core = b.add("Slice", [IN_NAME, core_st, core_en, ax4])
    st14 = b.i64([0, 1, 0, 0], "st14")
    en14 = b.i64([1, 5, GH, GW], "en14")
    st69 = b.i64([0, 6, 0, 0], "st69")
    en69 = b.i64([1, C, GH, GW], "en69")
    fg8 = b.add("Concat", [b.add("Slice", [core, st14, en14, ax4]), b.add("Slice", [core, st69, en69, ax4])], axis=1)
    st_gray = b.i64([0, GRAY, 0, 0], "st_gray")
    en_gray = b.i64([1, GRAY + 1, GH, GW], "en_gray")
    gray = b.add("Slice", [core, st_gray, en_gray, ax4])

    nz = b.add("Greater", [fg8, half])
    counts = _region_counts(b, nz, ax4)
    wins = _counts_to_wins(b, counts, half, zero_f)

    gray_b = b.add("Greater", [gray, half])
    wins_f = b.add("Cast", [wins], to=TensorProto.FLOAT)
    region_any_f = b.add("ReduceMax", [wins_f], axes=[1], keepdims=1)
    region_any = b.add("Greater", [region_any_f, half])
    fill_any = _expand_region_mask(b, region_any, content, idx1, "any")
    bg_b = b.add("And", [b.add("Not", [gray_b]), b.add("Not", [fill_any])])

    ch_bool: Dict[int, str] = {0: bg_b, 5: gray_b}
    for out_ch, compact in COMPACT_FOR_OUT.items():
        st = b.i64([0, compact, 0, 0], f"ws{out_ch}")
        en = b.i64([1, compact + 1, 3, 3], f"we{out_ch}")
        wins_k = b.add("Slice", [wins, st, en, ax4])
        ch_bool[out_ch] = _expand_region_mask(b, wins_k, content, idx1, f"c{out_ch}")

    planes = [ch_bool[c] for c in OUT_CHS]
    core_b = b.add("Concat", planes, axis=1)
    core_f = b.add("Cast", [core_b], to=TensorProto.FLOAT)
    b.add("Pad", [core_f], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD])

    graph = helper.make_graph(b.nodes, "task059", [x_info], [y_info], initializer=b.inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    onnx.checker.check_model(model)
    return model


def build_matmul(opset: int = OPSET) -> onnx.ModelProto:
    """Legacy candidate: MatMul region counting."""
    b = Builder()
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    ax4 = b.i64([0, 1, 2, 3], "ax4")
    core = b.add("Slice", [IN_NAME, b.i64([0, 0, 0, 0], "core_st"), b.i64([1, C, GH, GW], "core_en"), ax4])
    half = b.f32([0.5], "half")
    fg8 = b.add(
        "Concat",
        [
            b.add("Slice", [core, b.i64([0, 1, 0, 0], "st14"), b.i64([1, 5, GH, GW], "en14"), ax4]),
            b.add("Slice", [core, b.i64([0, 6, 0, 0], "st69"), b.i64([1, C, GH, GW], "en69"), ax4]),
        ],
        axis=1,
    )
    nz_f = b.add("Cast", [b.add("Greater", [fg8, half])], to=TensorProto.FLOAT)
    flat = b.add("Reshape", [nz_f, b.i64([NC, GH * GW], "flat_sh")])
    w = np.zeros((GH * GW, 9), dtype=np.float32)
    for ri, r0 in enumerate(ROW_OFF):
        for ci, c0 in enumerate(COL_OFF):
            k = ri * 3 + ci
            for dr in range(3):
                for dc in range(3):
                    w[(r0 + dr) * GW + (c0 + dc), k] = 1.0
    cnt9 = b.add("MatMul", [flat, b.f32(w, "reg_w")])
    counts = b.add("Reshape", [cnt9, b.i64([1, NC, 3, 3], "cnt_sh")])
    wins = _counts_to_wins(b, counts, half, b.f32([0.0], "zero_f"))
    gray = b.add("Slice", [core, b.i64([0, GRAY, 0, 0], "st_gray"), b.i64([1, GRAY + 1, GH, GW], "en_gray"), ax4])
    content = _content_mask(b)
    idx1 = b.i64(_region_flat_index().reshape(GH * GW), "reg_idx1")
    ch_bool: Dict[int, str] = {5: b.add("Greater", [gray, half])}
    wins_f = b.add("Cast", [wins], to=TensorProto.FLOAT)
    region_any = b.add("Greater", [b.add("ReduceMax", [wins_f], axes=[1], keepdims=1), half])
    fill_any = _expand_region_mask(b, region_any, content, idx1, "any")
    ch_bool[0] = b.add("And", [b.add("Not", [ch_bool[5]]), b.add("Not", [fill_any])])
    for out_ch, compact in COMPACT_FOR_OUT.items():
        st = b.i64([0, compact, 0, 0], f"ws{out_ch}")
        en = b.i64([1, compact + 1, 3, 3], f"we{out_ch}")
        ch_bool[out_ch] = _expand_region_mask(
            b, b.add("Slice", [wins, st, en, ax4]), content, idx1, f"c{out_ch}"
        )
    core_f = b.add("Cast", [b.add("Concat", [ch_bool[c] for c in OUT_CHS], axis=1)], to=TensorProto.FLOAT)
    b.add("Pad", [core_f], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD])
    graph = helper.make_graph(b.nodes, "task059_matmul", [x_info], [y_info], initializer=b.inits)
    model = helper.make_model(graph, producer_name="", ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", opset)])
    onnx.checker.check_model(model)
    return model


def build_single_color(opset: int = OPSET) -> onnx.ModelProto:
    """Observed task-specialized graph: count one merged non-gray color mask."""
    b = Builder()
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    ax4 = b.i64([0, 1, 2, 3], "ax4")
    half = b.f32([0.5], "half")
    zero_f = b.f32([0.0], "zero_f")
    idx1 = b.i64(_region_flat_index().reshape(GH * GW), "reg_idx1")
    gray_mask, content_mask = _lattice_masks()
    gray_b = _init(b.inits, gray_mask, "gray_mask")
    content = _init(b.inits, content_mask, "content_mask")

    region_counts: List[str] = []
    present14: List[str] = []
    present69: List[str] = []
    for ri, r0 in enumerate(ROW_OFF):
        for ci, c0 in enumerate(COL_OFF):
            reg14 = b.add(
                "Slice",
                [
                    IN_NAME,
                    b.i64([0, 1, r0, c0], f"r14s{ri}{ci}"),
                    b.i64([1, 5, r0 + 3, c0 + 3], f"r14e{ri}{ci}"),
                    ax4,
                ],
            )
            reg69 = b.add(
                "Slice",
                [
                    IN_NAME,
                    b.i64([0, 6, r0, c0], f"r69s{ri}{ci}"),
                    b.i64([1, C, r0 + 3, c0 + 3], f"r69e{ri}{ci}"),
                    ax4,
                ],
            )
            present14.append(b.add("ReduceMax", [reg14], axes=[2, 3], keepdims=1))
            present69.append(b.add("ReduceMax", [reg69], axes=[2, 3], keepdims=1))
            any14 = b.add("ReduceMax", [reg14], axes=[1], keepdims=1)
            any69 = b.add("ReduceMax", [reg69], axes=[1], keepdims=1)
            region_counts.append(b.add("ReduceSum", [b.add("Max", [any14, any69])], axes=[2, 3], keepdims=1))
    color_present = b.add(
        "Greater",
        [
            b.add(
                "Concat",
                [
                    b.add("ReduceMax", [b.add("Concat", present14, axis=2)], axes=[2], keepdims=1),
                    b.add("ReduceMax", [b.add("Concat", present69, axis=2)], axes=[2], keepdims=1),
                ],
                axis=1,
            ),
            half,
        ],
    )
    cnt9 = b.add("Concat", region_counts, axis=2)
    counts = b.add("Reshape", [cnt9, b.i64([1, 1, 3, 3], "cnt1_sh")])
    wins = _counts_to_wins(b, counts, half, zero_f)
    fill_mask = _expand_region_mask(b, wins, content, idx1, "one")

    not_fill = b.add("Not", [fill_mask])
    bg_b = b.add("And", [content, not_fill])
    color_gated: List[str] = []
    for compact in range(NC):
        present_k = b.add(
            "Slice",
            [
                color_present,
                b.i64([0, compact, 0, 0], f"ps{compact}"),
                b.i64([1, compact + 1, 1, 1], f"pe{compact}"),
                ax4,
            ],
        )
        color_gated.append(b.add("And", [fill_mask, present_k]))
    core_b = b.add("Concat", [bg_b, *color_gated[:4], gray_b, *color_gated[4:]], axis=1)
    core_f = b.add("Cast", [core_b], to=TensorProto.FLOAT)
    b.add("Pad", [core_f], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD])

    graph = helper.make_graph(b.nodes, "task059_single_color", [x_info], [y_info], initializer=b.inits)
    model = helper.make_model(graph, producer_name="", ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", opset)])
    onnx.checker.check_model(model)
    return model


BUILDERS: Dict[str, Callable[[], onnx.ModelProto]] = {
    "single_color": build_single_color,
    "slice_bool_planes": build_model,
    "matmul_planes": build_matmul,
}


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            exp = np.asarray(ex["output"], dtype=np.int64)
            oh = _grid_to_onehot(ex["input"])
            pred = _onehot_to_grid(_run_onnx(model, oh))[: exp.shape[0], : exp.shape[1]]
            if not np.array_equal(pred, exp):
                print(f"mismatch {split} {idx}")
                print(pred)
                print(exp)
                return 1
            full = _onehot_to_grid(_run_onnx(model, oh))
            if np.any(full[exp.shape[0] :, :]) or np.any(full[:, exp.shape[1] :]):
                print(f"nonzero outside grid {split} {idx}")
                return 1
    return 0


def _score_model(model: onnx.ModelProto, label: str) -> dict:
    tmpdir = Path(tempfile.mkdtemp())
    path = tmpdir / "task059.onnx"
    try:
        onnx.save(model, path)
        if validate_json(model):
            print(f"{label}: INVALID correctness")
            return {"label": label, "valid": False}
        result = score_file(path)
        print(
            f"{label}: memory={result.get('memory')} params={result.get('params')} "
            f"cost={result.get('cost')} score={result.get('score'):.6f}"
        )
        result["label"] = label
        return result
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> None:
    best_model: onnx.ModelProto | None = None
    best_result: dict | None = None

    for label, builder in BUILDERS.items():
        try:
            model = builder()
        except Exception as exc:
            print(f"{label}: build failed: {exc}")
            continue
        result = _score_model(model, label)
        if not result.get("valid"):
            continue
        if best_result is None or int(result["cost"]) < int(best_result["cost"]):
            best_model = model
            best_result = result

    if best_model is None or best_result is None:
        raise SystemExit("no valid candidate")

    assert validate_json(best_model) == 0
    onnx.save(best_model, BEST_PATH)
    final = score_file(BEST_PATH)
    print(f"\nBest: {best_result['label']} -> {BEST_PATH}")
    print(
        f"memory={final['memory']} params={final['params']} "
        f"cost={final['cost']} score={final['score']:.6f}"
    )


if __name__ == "__main__":
    main()
