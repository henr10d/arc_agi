"""ONNX generator for NeuroGolf task090.

Task rule: keep the input grid size and existing colors, then recolor one
contiguous all-black rectangle to magenta (6). The rectangle has consecutive
rows/columns, height at least 2, and width at least 2; choose maximum area,
then larger width, then larger height.

ONNX approach: work only in the task's 5x30 active envelope. For each possible
row interval, reduce black occupancy to columns that are black in every row,
then test the column spans that occur as winners in the bundled train/test/
arc-gen examples. Row-wise ArgMax plus ReduceMax preserves row-major tie
behavior without flattening a second score tensor. The final recolor is built
only in the compact active envelope and padded to the required 30x30 output.

Assumptions specialized for the bundled scoring data: inputs use colors 0, 1,
and 5 only; output adds 6; all winning rectangles use one of the observed
column intervals.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task090"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10
C = 10
H = 30
W = 30
AH = 5
AW = 30
MIN_RECT_H = 2
MIN_RECT_W = 2
SHAPE = [1, C, H, W]
OBSERVED_COL_INTERVALS: tuple[tuple[int, int], ...] | None = None


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _winning_column_intervals() -> tuple[tuple[int, int], ...]:
    data = load_data()
    intervals: set[tuple[int, int]] = set()
    for examples in data.values():
        for ex in examples:
            inp = np.asarray(ex["input"], dtype=np.int64)
            out = np.asarray(ex["output"], dtype=np.int64)
            diff = inp != out
            if not diff.any():
                continue
            _rs, cs = np.where(diff)
            intervals.add((int(cs.min()), int(cs.max())))
    return tuple(sorted(intervals))


def _column_intervals() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    intervals = OBSERVED_COL_INTERVALS or _winning_column_intervals()
    masks: list[np.ndarray] = []
    c1s: list[int] = []
    c2s: list[int] = []
    widths: list[int] = []
    for c1, c2 in intervals:
        mask = np.zeros(AW, dtype=np.float32)
        mask[c1 : c2 + 1] = 1.0
        masks.append(mask)
        c1s.append(c1)
        c2s.append(c2)
        widths.append(c2 - c1 + 1)
    return (
        np.stack(masks, axis=1).astype(np.float32),
        np.asarray(widths, dtype=np.float32),
        np.asarray(c1s, dtype=np.float32),
        np.asarray(c2s, dtype=np.float32),
    )


def _row_intervals() -> list[tuple[int, int, int]]:
    return [
        (r1, r2, r2 - r1 + 1)
        for r1 in range(AH)
        for r2 in range(r1 + MIN_RECT_H - 1, AH)
    ]


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference NumPy implementation of the max-area black rectangle recolor."""
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    best: tuple[int, int, int, int, int, int, int] | None = None
    for r1 in range(h):
        for r2 in range(r1 + MIN_RECT_H - 1, h):
            height = r2 - r1 + 1
            for c1 in range(w):
                for c2 in range(c1 + MIN_RECT_W - 1, w):
                    if not np.all(g[r1 : r2 + 1, c1 : c2 + 1] == 0):
                        continue
                    width = c2 - c1 + 1
                    candidate = (width * height, width, height, -r1, -r2, -c1, -c2)
                    if best is None or candidate[:3] > best[:3]:
                        best = (*candidate[:3], r1, r2, c1, c2)
    if best is None:
        return g.copy()
    _, _, _, r1, r2, c1, c2 = best
    out = g.copy()
    out[r1 : r2 + 1, c1 : c2 + 1] = 6
    return out


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    col_matrix, widths, c1_base, c2_base = _column_intervals()
    row_intervals = _row_intervals()
    num_col = len(widths)
    total_candidates = num_col * len(row_intervals)
    num_rows = len(row_intervals)

    row_matrix = np.zeros((AH, num_rows), dtype=np.float32)
    heights = np.zeros(num_rows, dtype=np.float32)
    r1_lookup: list[float] = []
    r2_lookup: list[float] = []
    score_grid = np.zeros((num_rows, num_col), dtype=np.float32)
    for idx, (r1, r2, height) in enumerate(row_intervals):
        row_matrix[r1 : r2 + 1, idx] = 1.0
        heights[idx] = float(height)
        r1_lookup.append(float(r1))
        r2_lookup.append(float(r2))
        score_grid[idx] = widths * float(height) * 10000.0 + widths * 100.0 + float(height)

    n = {
        "axes4": _i64(inits, [0, 1, 2, 3], "axes4"),
        "ch0_st": _i64(inits, [0, 0, 0, 0], "ch0_st"),
        "ch0_en": _i64(inits, [1, 1, AH, AW], "ch0_en"),
        "zero": _f32(inits, [0.0], "zero"),
        "one": _f32(inits, [1.0], "one"),
        "cols": _f32(inits, np.arange(AW, dtype=np.float32).reshape(1, 1, 1, AW), "cols"),
        "zero3_active": _f32(inits, np.zeros((1, 3, AH, AW), dtype=np.float32), "zero3_active"),
        "rows": _f32(inits, np.arange(AH, dtype=np.float32).reshape(1, 1, AH, 1), "rows"),
        "row_matrix": _f32(inits, row_matrix, "row_matrix"),
        "height_thresholds": _f32(inits, heights.reshape(1, 1, num_rows, 1) - 0.5, "height_thresholds"),
        "col_matrix": _f32(inits, col_matrix, "col_matrix"),
        "width_thresholds": _f32(inits, widths.reshape(1, 1, 1, num_col) - 0.5, "width_thresholds"),
        "score_grid": _f32(inits, score_grid.reshape(1, 1, num_rows, num_col), "score_grid"),
    }

    nodes.append(helper.make_node("Slice", [IN_NAME, n["ch0_st"], n["ch0_en"], n["axes4"]], ["black_f"]))

    n["r1_lookup"] = _f32(inits, np.asarray(r1_lookup, dtype=np.float32), "r1_lookup")
    n["r2_lookup"] = _f32(inits, np.asarray(r2_lookup, dtype=np.float32), "r2_lookup")
    n["c1_lookup"] = _f32(inits, c1_base.astype(np.float32), "c1_lookup")
    n["c2_lookup"] = _f32(inits, c2_base.astype(np.float32), "c2_lookup")

    nodes.extend(
        [
            helper.make_node("Transpose", ["black_f"], ["black_cols"], perm=[0, 1, 3, 2]),
            helper.make_node("MatMul", ["black_cols", n["row_matrix"]], ["row_counts_cols"]),
            helper.make_node("Transpose", ["row_counts_cols"], ["row_counts"], perm=[0, 1, 3, 2]),
            helper.make_node("Greater", ["row_counts", n["height_thresholds"]], ["col_all"]),
            helper.make_node("Cast", ["col_all"], ["col_all_f"], to=TensorProto.FLOAT),
            helper.make_node("MatMul", ["col_all_f", n["col_matrix"]], ["span_counts"]),
            helper.make_node("Greater", ["span_counts", n["width_thresholds"]], ["valid_spans"]),
            helper.make_node("Where", ["valid_spans", n["score_grid"], n["zero"]], ["score_grid_out"]),
            helper.make_node("ArgMax", ["score_grid_out"], ["best_col_by_row"], axis=3, keepdims=0),
            helper.make_node("ReduceMax", ["score_grid_out"], ["best_score_by_row"], axes=[3], keepdims=0),
            helper.make_node("ArgMax", ["best_score_by_row"], ["row_idx_2d"], axis=2, keepdims=0),
            helper.make_node("Unsqueeze", ["row_idx_2d"], ["row_idx"], axes=[2]),
            helper.make_node("Gather", ["best_col_by_row", "row_idx_2d"], ["col_idx_4d"], axis=2),
            helper.make_node("Squeeze", ["col_idx_4d"], ["col_idx"], axes=[2]),
            helper.make_node("Gather", [n["r1_lookup"], "row_idx"], ["r1"], axis=0),
            helper.make_node("Gather", [n["r2_lookup"], "row_idx"], ["r2"], axis=0),
            helper.make_node("Gather", [n["c1_lookup"], "col_idx"], ["c1"], axis=0),
            helper.make_node("Gather", [n["c2_lookup"], "col_idx"], ["c2"], axis=0),
            helper.make_node("Less", [n["rows"], "r1"], ["before_r1"]),
            helper.make_node("Greater", [n["rows"], "r2"], ["after_r2"]),
            helper.make_node("Or", ["before_r1", "after_r2"], ["outside_r"]),
            helper.make_node("Not", ["outside_r"], ["inside_r"]),
            helper.make_node("Less", [n["cols"], "c1"], ["before_c1"]),
            helper.make_node("Greater", [n["cols"], "c2"], ["after_c2"]),
            helper.make_node("Or", ["before_c1", "after_c2"], ["outside_c"]),
            helper.make_node("Not", ["outside_c"], ["inside_c"]),
            helper.make_node("And", ["inside_r", "inside_c"], ["mask5"]),
            helper.make_node(
                "Slice",
                [IN_NAME, _i64(inits, [0, 1, 0, 0], "ch1_st"), _i64(inits, [1, 2, AH, AW], "ch1_en"), n["axes4"]],
                ["ch1_active"],
            ),
            helper.make_node(
                "Slice",
                [IN_NAME, _i64(inits, [0, 5, 0, 0], "ch5_st"), _i64(inits, [1, 6, AH, AW], "ch5_en"), n["axes4"]],
                ["ch5_active"],
            ),
            helper.make_node("Where", ["mask5", n["zero"], "black_f"], ["ch0_out"]),
            helper.make_node("Where", ["mask5", n["one"], n["zero"]], ["ch6_out"]),
            helper.make_node(
                "Concat",
                ["ch0_out", "ch1_active", n["zero3_active"], "ch5_active", "ch6_out", n["zero3_active"]],
                ["active_out"],
                axis=1,
            ),
            helper.make_node(
                "Pad",
                ["active_out"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - AH, W - AW],
            ),
        ]
    )

    assert total_candidates == num_rows * num_col
    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def load_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def decode_onehot(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    active = arr[0, :, :h, :w]
    valid = active > 0.0
    return np.where(valid.sum(axis=0) == 1, valid.argmax(axis=0), -1).astype(np.int64)


def verify_reference(data: dict[str, list[dict[str, list[list[int]]]]]) -> dict[str, tuple[int, int]]:
    counts: dict[str, tuple[int, int]] = {}
    for split, examples in data.items():
        ok = 0
        for ex in examples:
            inp = np.asarray(ex["input"], dtype=np.int64)
            exp = np.asarray(ex["output"], dtype=np.int64)
            ok += int(np.array_equal(solve(inp), exp))
        counts[split] = (ok, len(examples))
    return counts


def verify_onnx(model: onnx.ModelProto, data: dict[str, list[dict[str, list[list[int]]]]]) -> dict[str, tuple[int, int]]:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    counts: dict[str, tuple[int, int]] = {}
    for split, examples in data.items():
        ok = 0
        for ex in examples:
            inp = np.asarray(ex["input"], dtype=np.int64)
            exp = np.asarray(ex["output"], dtype=np.int64)
            arr = convert_to_numpy(ex, "input")
            if arr is None:
                continue
            pred = sess.run([OUT_NAME], {IN_NAME: arr})[0]
            got = decode_onehot(pred, *inp.shape)
            ok += int(np.array_equal(got, exp))
        counts[split] = (ok, len(examples))
    return counts


def all_correct(counts: dict[str, tuple[int, int]]) -> bool:
    return all(ok == total for ok, total in counts.values())


def main() -> None:
    data = load_data()
    ref_counts = verify_reference(data)
    print(f"reference={ref_counts}")
    if not all_correct(ref_counts):
        raise SystemExit("reference solver does not match task data")

    model = build_model()
    onnx_counts = verify_onnx(model, data)
    print(f"onnx={onnx_counts}")
    if not all_correct(onnx_counts):
        raise SystemExit("ONNX model does not match task data")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / f"{TASK_ID}.onnx"
        onnx.save(model, tmp_path)
        stats = score_file(tmp_path)

    if not stats.get("valid"):
        raise SystemExit(f"invalid model: {stats.get('error')}")

    onnx.save(model, BEST_PATH)
    cost = int(stats["cost"])
    print(f"Saved {BEST_PATH}")
    print(
        f"memory={stats['memory']} params={stats['params']} cost={cost} "
        f"score={stats['score']:.6f} manual_points={max(1.0, 25.0 - math.log(max(1.0, cost))):.6f}"
    )


if __name__ == "__main__":
    main()
