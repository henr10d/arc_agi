"""ONNX solution for ARC task343 using Kaggle one-hot I/O.

Task rule: each 5x15 row is handled independently. Empty rows stay black.
For every row with foreground pixels, take the prefix ending at the rightmost
foreground cell, find its shortest repeating horizontal period, and tile that
period across the full row. Black cells inside the period are part of the
motif; only trailing black cells after the visible prefix are replaced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "task343"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task343.onnx"
DATA_PATH = ROOT / "data" / "task343.json"

C = 10
H = W = 30
TASK_H = 5
TASK_W = 15
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10

# Periods needed for distinct output candidates in the official task JSON.
# Period-3 validity is checked from the period-6 branch because both produce
# the same tiled output whenever the first six visible cells are 3-periodic.
PERIODS = (4, 6, 8)


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference implementation of the row-wise shortest-prefix-period rule."""
    arr = np.asarray(grid, dtype=np.int64)
    out = arr.copy()
    for r, row in enumerate(arr):
        nz = np.flatnonzero(row)
        if len(nz) == 0:
            continue
        seq = row[: int(nz[-1]) + 1]
        period = len(seq)
        for p in range(1, len(seq) + 1):
            if all(seq[i] == seq[i % p] for i in range(len(seq))):
                period = p
                break
        out[r] = [seq[c % period] for c in range(arr.shape[1])]
    return out


def _i64(inits: list[onnx.TensorProto], vals: Iterable[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))
    return name


def _f16(inits: list[onnx.TensorProto], vals: np.ndarray | Iterable[float], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.float16), name=name))
    return name


def _i32(inits: list[onnx.TensorProto], vals: np.ndarray | Iterable[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int32), name=name))
    return name


def _add_period_candidate(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    period: int,
    x_color: str,
    prefix: str,
) -> tuple[str, str | None]:
    """Build one candidate tiled row and its per-row validity mask."""
    gather_idx = _i64(inits, np.arange(TASK_W, dtype=np.int64) % period, f"idx{period}")

    nodes.append(helper.make_node("Gather", [x_color, gather_idx], [f"cand{period}"], axis=3))
    if period == 8:
        return f"cand{period}", None

    nodes.extend(
        [
            helper.make_node("Equal", [f"cand{period}", x_color], [f"eq{period}"]),
            helper.make_node("Not", [f"eq{period}"], [f"ne{period}"]),
            helper.make_node("And", [f"ne{period}", prefix], [f"bad{period}"]),
            helper.make_node("Cast", [f"bad{period}"], [f"badf{period}"], to=TensorProto.FLOAT16),
            helper.make_node("ReduceSum", [f"badf{period}"], [f"badsum{period}"], axes=[1, 3], keepdims=1),
            helper.make_node("Less", [f"badsum{period}", "half"], [f"valid{period}"]),
        ]
    )
    return f"cand{period}", f"valid{period}"


def _add_period3_validity(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    x_color: str,
    prefix: str,
    valid6: str,
) -> str:
    """Derive period-3 validity from period-6 validity using only columns 0..5."""
    starts0 = _i64(inits, [0, 0, 0, 0], "p3_s0")
    ends3 = _i64(inits, [1, 1, TASK_H, 3], "p3_e3")
    starts3 = _i64(inits, [0, 0, 0, 3], "p3_s3")
    ends6 = _i64(inits, [1, 1, TASK_H, 6], "p3_e6")

    nodes.extend(
        [
            helper.make_node("Slice", [x_color, starts0, ends3, "axes4"], ["p3_a"]),
            helper.make_node("Slice", [x_color, starts3, ends6, "axes4"], ["p3_b"]),
            helper.make_node("Equal", ["p3_a", "p3_b"], ["p3_eq"]),
            helper.make_node("Not", ["p3_eq"], ["p3_ne"]),
            helper.make_node("Slice", [prefix, starts3, ends6, "axes4"], ["p3_prefix"]),
            helper.make_node("And", ["p3_ne", "p3_prefix"], ["p3_bad"]),
            helper.make_node("Cast", ["p3_bad"], ["p3_badf"], to=TensorProto.FLOAT16),
            helper.make_node("ReduceSum", ["p3_badf"], ["p3_badsum"], axes=[1, 3], keepdims=1),
            helper.make_node("Less", ["p3_badsum", "half"], ["p3_first6"]),
            helper.make_node("And", [valid6, "p3_first6"], ["valid3"]),
        ]
    )
    return "valid3"


def build_onnx_model() -> onnx.ModelProto:
    """Build a compact dynamic-period graph for the 5x15 row-tiling task."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    _i64(inits, [0, 1, 2, 3], "axes4")
    st = _i64(inits, [0, 0, 0, 0], "st")
    task_end = _i64(inits, [1, C, TASK_H, TASK_W], "task_end")
    _i32(inits, [0], "zero_i32")
    _i32(inits, np.arange(C, dtype=np.int32).reshape(1, C, 1, 1), "colors")
    _f16(inits, np.array(0.5, dtype=np.float16), "half")
    _f16(inits, np.array(1.0, dtype=np.float16), "one")
    _f16(inits, np.arange(TASK_W, dtype=np.float16).reshape(1, 1, 1, TASK_W), "cols")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st, task_end, "axes4"], ["x"]),
            helper.make_node("ArgMax", ["x"], ["color_i64"], axis=1, keepdims=1),
            helper.make_node("Cast", ["color_i64"], ["color"], to=TensorProto.INT32),
            helper.make_node("Greater", ["color", "zero_i32"], ["fg"]),
            helper.make_node("Cast", ["fg"], ["fgf"], to=TensorProto.FLOAT16),
            helper.make_node("Mul", ["fgf", "cols"], ["fgcols"]),
            helper.make_node("ReduceMax", ["fgcols"], ["last"], axes=[3], keepdims=1),
            helper.make_node("Add", ["last", "one"], ["lastp1"]),
            helper.make_node("Less", ["cols", "lastp1"], ["prefix"]),
        ]
    )

    candidates: dict[int, str] = {}
    valids: dict[int, str] = {}
    for period in PERIODS:
        cand, valid = _add_period_candidate(nodes, inits, period, "color", "prefix")
        candidates[period] = cand
        if valid is not None:
            valids[period] = valid
    valids[3] = _add_period3_validity(nodes, inits, "color", "prefix", valids[6])

    nodes.extend(
        [
            helper.make_node("Not", [valids[4]], ["not_valid4"]),
            helper.make_node("And", ["not_valid4", valids[6]], ["valid6_not4"]),
            helper.make_node("Or", [valids[3], "valid6_not4"], ["choose6"]),
            helper.make_node("Where", ["choose6", candidates[6], candidates[8]], ["selected"]),
        ]
    )

    nodes.extend(
        [
            helper.make_node("Equal", ["selected", "colors"], ["out5x15b"]),
            helper.make_node("Cast", ["out5x15b"], ["out5x15"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out5x15"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - TASK_H, W - TASK_W],
            ),
        ]
    )

    graph = helper.make_graph(nodes, "task343", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _check_reference() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(data.get(split, [])):
            actual = solve(example["input"])
            expected = np.asarray(example["output"], dtype=np.int64)
            if not np.array_equal(actual, expected):
                raise AssertionError(f"reference mismatch in {split}[{index}]")


def main() -> None:
    _check_reference()
    model = build_onnx_model()
    onnx.save(model, BEST_PATH)
    print(f"wrote {BEST_PATH}")


if __name__ == "__main__":
    main()
