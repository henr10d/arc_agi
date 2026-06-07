"""Minimal ONNX for ARC task250 using the red block as a gray-marker compass.

Task rule: each 10x10 input contains a red 2x2 square and several gray cells.
Keep the red square unchanged, erase the original gray cells, and move each
gray cell to the nearest cell on the 4x4 perimeter ring around the red square.
Equivalently, clamp each gray cell's row/column to one step outside-or-on the
red 2x2 bounding box.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task250"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
GH = GW = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self._n = 0

    def name(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    def arr(self, array: Any, name: str | None = None) -> str:
        name = name or self.name("c")
        self.inits.append(numpy_helper.from_array(np.asarray(array), name=name))
        return name

    def f32(self, value: Any, name: str | None = None) -> str:
        return self.arr(np.asarray(value, dtype=np.float32), name)

    def i64(self, value: Any, name: str | None = None) -> str:
        return self.arr(np.asarray(value, dtype=np.int64), name)

    def add(self, op: str, ins: Sequence[str], outs: Sequence[str], **attrs: Any) -> str:
        self.nodes.append(helper.make_node(op, list(ins), list(outs), **attrs))
        return outs[0]


def _eqf(b: Builder, a: str, c: str, half: str) -> str:
    del half
    out = b.name("eq")
    b.add("Equal", [a, c], [out])
    return out


def _row_has_gray_by_col(b: Builder, grayf: str, rcat: str, zero: str) -> str:
    masked = b.name("rowmasked")
    by_col = b.name("rowbycol")
    b.add("Where", [rcat, grayf, zero], [masked])
    b.add("ReduceMax", [masked], [by_col], axes=[2], keepdims=1)
    return by_col


def _region_has_gray(b: Builder, row_has_cols: str, ccat: str, zero_col: str) -> str:
    masked = b.name("colmasked")
    has = b.name("has")
    b.add("Where", [ccat, row_has_cols, zero_col], [masked])
    b.add("ReduceMax", [masked], [has], axes=[3], keepdims=1)
    return has


def build_rule_onnx_model() -> onnx.ModelProto:
    b = Builder()
    shape = [1, C, H, W]
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, shape)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, shape)

    axes4 = b.i64([0, 1, 2, 3], "axes4")
    half = b.f32([0.5], "half")
    one = b.i64([1], "one_i")
    rows = b.i64(np.arange(GH, dtype=np.int64).reshape(1, 1, GH, 1), "rows")
    cols = b.i64(np.arange(GW, dtype=np.int64).reshape(1, 1, 1, GW), "cols")

    redf = b.add("Slice", [IN_NAME, b.i64([0, 2, 0, 0]), b.i64([1, 3, GH, GW]), axes4], ["redf"])
    grayf_in = b.add("Slice", [IN_NAME, b.i64([0, 5, 0, 0]), b.i64([1, 6, GH, GW]), axes4], ["grayf_in"])
    red = b.add("Greater", [redf, half], ["red"])

    red_row_presence = b.add("ReduceMax", [redf], ["red_row_presence"], axes=[3], keepdims=1)
    red_col_presence = b.add("ReduceMax", [redf], ["red_col_presence"], axes=[2], keepdims=1)
    r0 = b.add("ArgMax", [red_row_presence], ["red_r0"], axis=2, keepdims=1)
    c0 = b.add("ArgMax", [red_col_presence], ["red_c0"], axis=3, keepdims=1)
    r1 = b.add("Add", [r0, one], ["red_r1"])
    c1 = b.add("Add", [c0, one], ["red_c1"])
    rm1 = b.add("Sub", [r0, one], ["red_rm1"])
    cm1 = b.add("Sub", [c0, one], ["red_cm1"])
    rp2 = b.add("Add", [r1, one], ["red_rp2"])
    cp2 = b.add("Add", [c1, one], ["red_cp2"])

    r_top = b.add("Less", [rows, r0], ["rtop"])
    r_eq0 = _eqf(b, rows, r0, half)
    r_eq1 = _eqf(b, rows, r1, half)
    r_bot = b.add("Greater", [rows, r1], ["rbot"])
    c_left = b.add("Less", [cols, c0], ["cleft"])
    c_eq0 = _eqf(b, cols, c0, half)
    c_eq1 = _eqf(b, cols, c1, half)
    c_right = b.add("Greater", [cols, c1], ["cright"])

    row_cats = [r_top, r_eq0, r_eq1, r_bot]
    col_cats = [c_left, c_eq0, c_eq1, c_right]
    out_rows = [rm1, r0, r1, rp2]
    out_col_eq = [_eqf(b, cols, cm1, half), c_eq0, c_eq1, _eqf(b, cols, cp2, half)]

    zero = b.f32(np.zeros((1, 1, GH, GW), dtype=np.float32), "zero")
    zero_col = b.f32(np.zeros((1, 1, 1, GW), dtype=np.float32), "zero_col")
    zero_i_col = b.i64(np.zeros((1, 1, 1, GW), dtype=np.int64), "zero_i_col")
    row_has_cols = [_row_has_gray_by_col(b, grayf_in, rcat, zero) for rcat in row_cats]
    gray_rows: list[str] = []
    gray_row_indices: list[str] = []
    for ri in range(4):
        row_acc: str | None = None
        for ci in range(4):
            if ri in (1, 2) and ci in (1, 2):
                continue
            has = _region_has_gray(b, row_has_cols[ri], col_cats[ci], zero_col)
            mark = b.add("Where", [out_col_eq[ci], has, zero_col], [b.name("grayrowmark")])
            row_acc = mark if row_acc is None else b.add("Add", [row_acc, mark], [b.name("grayrowacc")])
        assert row_acc is not None
        gray_rows.append(row_acc)
        gray_row_indices.append(b.add("Add", [out_rows[ri], zero_i_col], [b.name("grayrowidx")]))

    gray_updates = b.add("Concat", gray_rows, ["gray_updates"], axis=2)
    gray_indices = b.add("Concat", gray_row_indices, ["gray_indices"], axis=2)
    gray_out = b.add("Scatter", [zero, gray_indices, gray_updates], ["gray_out"], axis=2)
    gray_mask = b.add("Greater", [gray_out, half], ["gray_mask"])
    occupied = b.add("Or", [red, gray_mask], ["occupied"])
    bg_keep = b.add("Not", [occupied], ["bg_keep"])
    bg = b.add("Cast", [bg_keep], ["bg"], to=TensorProto.FLOAT)
    red_out = b.add("Cast", [red], ["red_out"], to=TensorProto.FLOAT)

    chans = [bg, zero, red_out, zero, zero, gray_out, zero, zero, zero, zero]
    out10 = b.add("Concat", chans, ["out10"], axis=1)
    b.add("Pad", [out10], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - GH, W - GW])

    graph = helper.make_graph(b.nodes, f"{TASK_ID}_rule", [x_info], [y_info], initializer=b.inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return model


def _all_examples() -> list[dict[str, Any]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [ex for split in ("train", "test", "arc-gen") for ex in data[split]]


def _grid_to_onehot(grid: Sequence[Sequence[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros((1, C, H, W), dtype=np.float32)
    for r in range(min(arr.shape[0], GH)):
        for c in range(min(arr.shape[1], GW)):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _solve_reference(grid: Sequence[Sequence[int]]) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(g)
    red = np.argwhere(g == 2)
    r0, c0 = red.min(axis=0)
    r1, c1 = red.max(axis=0)
    out[r0 : r1 + 1, c0 : c1 + 1] = 2
    for r, c in np.argwhere(g == 5):
        rr = min(max(int(r), int(r0) - 1), int(r1) + 1)
        cc = min(max(int(c), int(c0) - 1), int(c1) + 1)
        out[rr, cc] = 5
    return out


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def _validate_model(model: onnx.ModelProto) -> None:
    examples = _all_examples()
    for idx, ex in enumerate(examples):
        pred = _run_onnx(model, _grid_to_onehot(ex["input"]))
        got = pred[0, :, :GH, :GW].argmax(axis=0).astype(np.int64)
        exp = np.asarray(ex["output"], dtype=np.int64)
        expected = _grid_to_onehot(ex["output"])
        if not np.array_equal(pred > 0.0, expected > 0.0) or not np.array_equal(got, exp):
            raise RuntimeError(f"ONNX mismatch on example {idx}")


def main() -> None:
    examples = _all_examples()
    bad = [
        idx
        for idx, ex in enumerate(examples)
        if not np.array_equal(_solve_reference(ex["input"]), np.asarray(ex["output"], dtype=np.int64))
    ]
    if bad:
        raise RuntimeError(f"reference rule mismatches examples: {bad[:10]}")

    model = build_rule_onnx_model()
    _validate_model(model)
    onnx.save(model, BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"saved {BEST_PATH}")
    print(f"valid={result['valid']} cost={result['cost']} score={result['score']}")


if __name__ == "__main__":
    main()
