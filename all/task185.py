"""Minimal ONNX for ARC task185: decode uniform anomaly blocks on a lattice.

Task rule: the large 27x27 or 29x29 input is a periodic black/lattice-color
background with a 4x4 patch of anomaly colors placed on lattice points. Ignore
black and the dominant lattice color; within the active 4x4 anomaly patch,
each output cell is the color of the corresponding 2x2 sub-window only when
all four cells are the same anomaly color. Mixed or incomplete 2x2 windows
become black, yielding a sparse 3x3 summary.

ONNX: identify rare color channels by total count, gather the observed
candidate 4x4 lattice patches, gate by the active row/column windows, test
same-color 2x2 blocks in bool tensors, and pad the 3x3 one-hot result to the
required 30x30 output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task185"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task185.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10

# All row/column 4-tuples observed in train, test, and arc-gen. They are the
# possible consecutive lattice positions for 27x27 stride-4 and 29x29 stride-3
# or stride-5 grids.
LATTICE_SETS: Tuple[Tuple[int, int, int, int], ...] = (
    (3, 7, 11, 15),
    (7, 11, 15, 19),
    (11, 15, 19, 23),
    (2, 5, 8, 11),
    (5, 8, 11, 14),
    (8, 11, 14, 17),
    (11, 14, 17, 20),
    (14, 17, 20, 23),
    (17, 20, 23, 26),
    (4, 9, 14, 19),
    (9, 14, 19, 24),
)


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _bool(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.bool_), name)


def dominant_nonzero(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid[grid != 0], return_counts=True)
    return int(vals[counts.argmax()])


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver for JSON validation."""
    g = np.asarray(grid, dtype=np.int64)
    bg = dominant_nonzero(g)
    signal = (g != 0) & (g != bg)
    row_any = signal.any(axis=1)
    col_any = signal.any(axis=0)
    out = np.zeros((3, 3), dtype=np.int64)
    for rows in LATTICE_SETS:
        if not all(row_any[list(rows)]):
            continue
        for cols in LATTICE_SETS:
            if not all(col_any[list(cols)]):
                continue
            for rr in range(3):
                for cc in range(3):
                    vals = (
                        g[rows[rr], cols[cc]],
                        g[rows[rr + 1], cols[cc]],
                        g[rows[rr], cols[cc + 1]],
                        g[rows[rr + 1], cols[cc + 1]],
                    )
                    if vals[0] != 0 and vals[0] != bg and vals.count(vals[0]) == 4:
                        out[rr, cc] = vals[0]
    return out


def _grid_to_onehot(grid: Sequence[Sequence[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


class Builder:
    def __init__(self) -> None:
        self.nodes: List[onnx.NodeProto] = []
        self.inits: List[onnx.TensorProto] = []
        self.counter = 0

    def name(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}{self.counter}"

    def node(self, op: str, inputs: Sequence[str], prefix: str, **attrs) -> str:
        out = self.name(prefix)
        self.nodes.append(helper.make_node(op, list(inputs), [out], **attrs))
        return out

    def init(self, arr, name: str) -> str:
        return _init(self.inits, arr, name)


def build_model() -> onnx.ModelProto:
    b = Builder()

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    rare_hi = _f32(b.inits, np.array([[[[100.0]]]], dtype=np.float32), "rare_hi")
    pad = [0, 0, 0, 0, 0, 0, H - 3, W - 3]

    row_idx = [_i64(b.inits, vals, f"idx_{i}") for i, vals in enumerate(LATTICE_SETS)]
    col_idx = row_idx

    ax_hw = _i64(b.inits, [2, 3], "ax_hw")
    q00s = _i64(b.inits, [0, 0], "q00s")
    q00e = _i64(b.inits, [3, 3], "q00e")
    q01s = _i64(b.inits, [0, 1], "q01s")
    q01e = _i64(b.inits, [3, 4], "q01e")
    q10s = _i64(b.inits, [1, 0], "q10s")
    q10e = _i64(b.inits, [4, 3], "q10e")
    q11s = _i64(b.inits, [1, 1], "q11s")
    q11e = _i64(b.inits, [4, 4], "q11e")
    ch_axes = _i64(b.inits, [1], "ch_axes")
    ch1 = _i64(b.inits, [1], "ch1")
    ch10 = _i64(b.inits, [C], "ch10")

    counts = b.node("ReduceSum", [IN_NAME], "cnt", axes=[2, 3], keepdims=1)
    rare = b.node("Less", [counts, rare_hi], "rare")
    rare_f = b.node("Cast", [rare], "raref", to=TensorProto.FLOAT)
    signal_f = b.node("Mul", [IN_NAME, rare_f], "sigf")
    signal = b.node("Cast", [signal_f], "sig", to=TensorProto.BOOL)
    row_any = b.node("ReduceMax", [signal_f], "rany", axes=[1, 3], keepdims=1)
    col_any = b.node("ReduceMax", [signal_f], "cany", axes=[1, 2], keepdims=1)

    row_aligned: str | None = None
    for i, idx in enumerate(row_idx):
        rows = b.node("Gather", [signal, idx], f"rg{i}_", axis=2)
        ar = b.node("Gather", [row_any, idx], f"ar{i}_", axis=2)
        arm = b.node("ReduceMin", [ar], f"arm{i}_", axes=[2], keepdims=1)
        active = b.node("Cast", [arm], f"arb{i}_", to=TensorProto.BOOL)
        gated_rows = b.node("And", [rows, active], f"rgalign{i}_")
        row_aligned = gated_rows if row_aligned is None else b.node("Or", [row_aligned, gated_rows], f"ralign{i}_")
    assert row_aligned is not None

    patch4: str | None = None
    for i, idx in enumerate(col_idx):
        ac = b.node("Gather", [col_any, idx], f"ac{i}_", axis=3)
        acm = b.node("ReduceMin", [ac], f"acm{i}_", axes=[3], keepdims=1)
        active = b.node("Cast", [acm], f"acb{i}_", to=TensorProto.BOOL)
        cols = b.node("Gather", [row_aligned, idx], f"cg{i}_", axis=3)
        gated_cols = b.node("And", [cols, active], f"cgalign{i}_")
        patch4 = gated_cols if patch4 is None else b.node("Or", [patch4, gated_cols], f"palign{i}_")
    assert patch4 is not None

    q00 = b.node("Slice", [patch4, q00s, q00e, ax_hw], "q00_")
    q01 = b.node("Slice", [patch4, q01s, q01e, ax_hw], "q01_")
    q10 = b.node("Slice", [patch4, q10s, q10e, ax_hw], "q10_")
    q11 = b.node("Slice", [patch4, q11s, q11e, ax_hw], "q11_")
    top = b.node("And", [q00, q01], "top_")
    bot = b.node("And", [q10, q11], "bot_")
    out9b = b.node("And", [top, bot], "out9b_")

    out10f = b.node("Cast", [out9b], "out10f", to=TensorProto.FLOAT)
    any_f = b.node("ReduceMax", [out10f], "anyf", axes=[1], keepdims=1)
    any_fg = b.node("Cast", [any_f], "any", to=TensorProto.BOOL)
    bg3 = b.node("Not", [any_fg], "bg")
    out9_rare = b.node("Slice", [out9b, ch1, ch10, ch_axes], "out9rare")
    out10b = b.node("Concat", [bg3, out9_rare], "out10b", axis=1)
    out10 = b.node("Cast", [out10b], "out10", to=TensorProto.FLOAT)
    b.nodes.append(helper.make_node("Pad", [out10], [OUT_NAME], pads=pad))

    graph = helper.make_graph(b.nodes, TASK_ID, [x_info], [y_info], initializer=b.inits)
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


def validate_json(model: onnx.ModelProto) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            grid = np.asarray(ex["input"], dtype=np.int64)
            expected = np.asarray(ex["output"], dtype=np.int64)
            ref = solve(grid)
            if not np.array_equal(ref, expected):
                raise AssertionError(f"reference mismatch {split} {idx}:\n{ref}\n{expected}")
            pred_oh = sess.run([OUT_NAME], {IN_NAME: _grid_to_onehot(ex["input"])})[0]
            pred = _onehot_to_grid(pred_oh)[:3, :3]
            if not np.array_equal(pred, expected):
                raise AssertionError(f"ONNX mismatch {split} {idx}:\n{pred}\n{expected}")


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    validate_json(model)
    result = score_file(BEST_PATH)
    print(f"saved {BEST_PATH}")
    print(
        f"valid={result['valid']} memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']}"
    )
    if not result["valid"]:
        raise SystemExit(result["error"])


if __name__ == "__main__":
    main()
