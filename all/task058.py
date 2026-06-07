"""Compact ONNX for ARC task058: draw the green square spiral.

Task rule: the input is an N x N all-black grid. The output keeps the same
N x N extent, with background color 0 and ARC green color 3 forming a
one-cell-wide clockwise spiral. The path starts across the top row, continues
down the right edge, left across the bottom, then back up the left side, and
continues inward with one-cell gaps. The observed examples include connector
cells at (2, 1), (4, 3), ... so the inner top segment begins one cell left of
the new rectangular band.

ONNX: infer N from the active black cells in row 0, evaluate the spiral as
boolean predicates over compact 20x20 row/column coordinate tensors (the
official examples use sizes 5..20), concatenate a compact boolean four-channel
[black, 0, 0, green] tensor, cast it once, and let the final Pad node produce
the required 10x30x30 output.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task058"
BEST_PATH = OUT_DIR / "task058.onnx"
DATA_PATH = ROOT / "data" / "task058.json"

C = 10
H = W = 30
CORE = 20
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
IR_VERSION = 10
OPSET = 10
GREEN = 3


def solve(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    """Reference implementation matching the JSON examples."""
    n = len(grid)
    out = np.zeros((n, n), dtype=np.int64)
    k = 0
    while True:
        hi = n - 1 - k
        start = max(0, k - 1)
        if start > hi:
            break
        out[k, start : hi + 1] = GREEN
        out[k + 1 : hi + 1, hi] = GREEN
        out[hi, k:hi] = GREEN
        out[k + 2 : hi, k] = GREEN
        k += 2
    return out


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    return _init(inits, np.asarray(arr, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    return _init(inits, np.asarray(arr, dtype=np.float32), name)


def _bool(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    return _init(inits, np.asarray(arr, dtype=bool), name)


class Builder:
    def __init__(self) -> None:
        self.nodes: List[onnx.NodeProto] = []
        self.inits: List[onnx.TensorProto] = []
        self._counter = 0

    def name(self, prefix: str = "t") -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    def i64(self, arr: Any, name: str) -> str:
        return _i64(self.inits, arr, name)

    def f32(self, arr: Any, name: str) -> str:
        return _f32(self.inits, arr, name)

    def bool(self, arr: Any, name: str) -> str:
        return _bool(self.inits, arr, name)

    def add(
        self,
        op: str,
        inputs: list[str],
        outputs: list[str] | None = None,
        **attrs: Any,
    ) -> str:
        out = outputs[0] if outputs else self.name()
        self.nodes.append(helper.make_node(op, inputs, [out] if outputs is None else outputs, **attrs))
        return out

    def ge(self, lhs: str, rhs: str) -> str:
        return self.add("Not", [self.add("Less", [lhs, rhs])])

    def le(self, lhs: str, rhs: str) -> str:
        return self.add("Not", [self.add("Greater", [lhs, rhs])])

    def and_many(self, names: list[str]) -> str:
        cur = names[0]
        for name in names[1:]:
            cur = self.add("And", [cur, name])
        return cur

    def or_many(self, names: list[str]) -> str:
        cur = names[0]
        for name in names[1:]:
            cur = self.add("Or", [cur, name])
        return cur


def build_model() -> onnx.ModelProto:
    b = Builder()
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    rows = np.arange(CORE, dtype=np.int64).reshape(1, 1, CORE, 1)
    cols = np.arange(CORE, dtype=np.int64).reshape(1, 1, 1, CORE)
    top_start = np.maximum(np.arange(CORE, dtype=np.int64) - 1, 0).reshape(1, 1, CORE, 1)
    top_start_minus = top_start - 1
    rows_plus = (np.arange(CORE, dtype=np.int64) + 1).reshape(1, 1, CORE, 1)
    cols_plus = (np.arange(CORE, dtype=np.int64) + 1).reshape(1, 1, 1, CORE)

    r = b.i64(rows, "r")
    c = b.i64(cols, "c")
    top_lo_minus = b.i64(top_start_minus, "top_lo_minus")
    r_plus = b.i64(rows_plus, "r_plus")
    c_plus = b.i64(cols_plus, "c_plus")
    r_even = b.bool((rows % 2) == 0, "r_even")
    c_even = b.bool((cols % 2) == 0, "c_even")
    ax4 = b.i64([0, 1, 2, 3], "ax4")
    ch0_st = b.i64([0, 0, 0, 0], "ch0_st")
    row0_en = b.i64([1, 1, 1, W], "row0_en")
    one_i = b.i64([1], "one_i")
    two_i = b.i64([2], "two_i")

    row0 = b.add("Slice", [IN_NAME, ch0_st, row0_en, ax4])
    n_float = b.add("ReduceSum", [row0], axes=[3], keepdims=1)
    n = b.add("Cast", [n_float], to=TensorProto.INT64)
    in_grid = b.add("And", [b.add("Less", [r, n]), b.add("Less", [c, n])])

    n_minus_r = b.add("Sub", [n, r])
    n_minus_c = b.add("Sub", [n, c])

    top = b.and_many(
        [
            r_even,
            b.add("Greater", [c, top_lo_minus]),
            b.add("Less", [c, n_minus_r]),
        ]
    )

    right_even = b.add("Equal", [b.add("Mod", [n_minus_c, two_i]), one_i])
    right = b.and_many(
        [
            right_even,
            b.add("Greater", [r, b.add("Sub", [n_minus_c, two_i])]),
            b.add("Less", [r, c_plus]),
        ]
    )

    bottom_even = b.add("Equal", [b.add("Mod", [n_minus_r, two_i]), one_i])
    bottom = b.and_many(
        [
            bottom_even,
            b.add("Greater", [c, b.add("Sub", [n_minus_r, two_i])]),
            b.add("Less", [c, r_plus]),
        ]
    )

    left = b.and_many(
        [
            c_even,
            b.add("Greater", [r, c_plus]),
            b.add("Less", [r, n_minus_c]),
        ]
    )

    geom = b.or_many([top, right, bottom, left])
    spiral = b.add("And", [geom, in_grid])
    background = b.add("Xor", [in_grid, spiral])
    zero_core = b.bool(np.zeros((1, 1, CORE, CORE), dtype=bool), "zero_core")
    compact_bool = b.add("Concat", [background, zero_core, zero_core, spiral], axis=1)
    compact = b.add("Cast", [compact_bool], to=TensorProto.FLOAT)
    pads = [0, 0, 0, 0, 0, C - 4, H - CORE, W - CORE]
    b.add("Pad", [compact], [OUT_NAME], pads=pads, mode="constant", value=0.0)

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


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            out[0, int(color), row, col] = 1.0
    return out


def _expected_onehot(grid: list[list[int]]) -> np.ndarray:
    return _grid_to_onehot(grid)


def validate_model(model: onnx.ModelProto) -> tuple[bool, str]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            ref = solve(ex["input"])
            target = np.asarray(ex["output"], dtype=np.int64)
            if not np.array_equal(ref, target):
                return False, f"reference mismatch {split} {idx}"
            pred = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(ex["input"])})[0]
            expected = _expected_onehot(ex["output"])
            if not np.array_equal(pred > 0.0, expected > 0.0):
                return False, f"ONNX mismatch {split} {idx}"
    return True, "ok"


def main() -> None:
    model = build_model()
    ok, msg = validate_model(model)
    if not ok:
        raise SystemExit(msg)

    tmpdir = Path(tempfile.mkdtemp())
    try:
        tmp_path = tmpdir / "task058.onnx"
        onnx.save(model, tmp_path)
        tmp_score = score_file(tmp_path)
        if not tmp_score.get("valid"):
            raise SystemExit(tmp_score.get("error"))
        onnx.save(model, BEST_PATH)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    final = score_file(BEST_PATH)
    print(
        f"saved {BEST_PATH} | memory={final['memory']} params={final['params']} "
        f"cost={final['cost']} score={final['score']:.6f}"
    )


if __name__ == "__main__":
    main()
