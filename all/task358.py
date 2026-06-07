"""Minimal ONNX for ARC task358: extend a colored L motif into a cross.

Task rule: the input grid contains one small non-background L-shaped motif. The
row and column with the most colored cells intersect at the cross junction. The
contiguous colors on that row and column, in their visible order, are cyclic
patterns of length 3 or 4. The output keeps the same grid size, fills the whole
junction row and column by repeating those two patterns periodically, and leaves
all other cells as background.

ONNX: decode each cell to a scalar color id, use the task-data property that the
junction row/column also have the largest color sums, gather the motif row and
column, count only the gathered row to choose period 3 vs 4, spread colors by
residue slices plus Tile, then one-hot encode only the final scalar color grid.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task358"
BEST_PATH = OUT_DIR / "task358.onnx"
DATA_PATH = ROOT / "data" / "task358.json"

C = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _i32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int32), name)


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _f16(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float16), name)


def _period_matrix(period: int) -> np.ndarray:
    return np.asarray(
        [[1.0 if (dst - src) % period == 0 else 0.0 for dst in range(W)] for src in range(W)],
        dtype=np.float16,
    )


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver matching the ONNX modulo-spread construction."""
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    foreground = g != 0
    row_counts = foreground.sum(axis=1)
    col_counts = foreground.sum(axis=0)
    r0 = int(row_counts.argmax())
    c0 = int(col_counts.argmax())
    period = int(row_counts[r0])

    row_src = np.zeros((C, W), dtype=np.float32)
    col_src = np.zeros((C, H), dtype=np.float32)
    for c in range(w):
        if g[r0, c] != 0:
            row_src[int(g[r0, c]), c] = 1.0
    for r in range(h):
        if g[r, c0] != 0:
            col_src[int(g[r, c0]), r] = 1.0

    mat = _period_matrix(period).astype(np.float32)
    row_out = row_src @ mat
    col_out = col_src @ mat

    out = np.zeros((h, w), dtype=np.int64)
    for c in range(w):
        out[r0, c] = int(row_out[:, c].argmax())
    for r in range(h):
        out[r, c0] = int(col_out[:, r].argmax())
    return out


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    arr = onehot.reshape(C, H, W)
    active = arr > 0.0
    bad = active.sum(axis=0) != 1
    decoded = arr.argmax(axis=0).astype(np.int64)
    decoded[bad] = -1
    return decoded


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    half32 = _f32(inits, [0.5], "half32")
    three_half = _f16(inits, [3.5], "three_half")
    decode_w = _f32(inits, np.arange(C, dtype=np.float32).reshape(1, C, 1, 1), "decode_w")
    neg_one = _f16(inits, [-1.0], "neg_one")
    zero16 = _f16(inits, [0.0], "zero16")
    colors = _i32(inits, np.arange(C, dtype=np.int32).reshape(1, C, 1, 1), "colors")
    axis1 = _i64(inits, [1], "axis1")
    end30 = _i64(inits, [30], "end30")
    step3 = _i64(inits, [3], "step3")
    step4 = _i64(inits, [4], "step4")
    repeats3 = _i64(inits, [1, 10], "repeats3")
    repeats4 = _i64(inits, [1, 8], "repeats4")
    starts = {idx: _i64(inits, [idx], f"start{idx}") for idx in range(4)}
    coords = _i64(inits, np.arange(W, dtype=np.int64).reshape(1, W), "coords")

    periodic_nodes: List[onnx.NodeProto] = []

    def spread(src: str, prefix: str, period: int) -> str:
        pieces: list[str] = []
        step = step3 if period == 3 else step4
        repeats = repeats3 if period == 3 else repeats4
        for residue in range(period):
            sliced = f"{prefix}_{period}_slice{residue}"
            summed = f"{prefix}_{period}_sum{residue}"
            periodic_nodes.append(
                helper.make_node("Slice", [src, starts[residue], end30, axis1, step], [sliced])
            )
            periodic_nodes.append(helper.make_node("ReduceSum", [sliced], [summed], axes=[1], keepdims=1))
            pieces.append(summed)

        base = f"{prefix}_{period}_base"
        tiled = f"{prefix}_{period}_tile"
        periodic_nodes.append(helper.make_node("Concat", pieces, [base], axis=1))
        periodic_nodes.append(helper.make_node("Tile", [base, repeats], [tiled]))
        if period == 3:
            return tiled
        out = f"{prefix}_{period}_out"
        periodic_nodes.append(helper.make_node("Slice", [tiled, starts[0], end30, axis1], [out]))
        return out

    row3 = spread("row_src", "row", 3)
    row4 = spread("row_src", "row", 4)
    col3 = spread("col_src", "col", 3)
    col4 = spread("col_src", "col", 4)

    nodes.extend(
        [
            helper.make_node("Conv", [IN_NAME, decode_w], ["color32"]),
            helper.make_node("ReduceSum", [IN_NAME], ["valid_rows_sum"], axes=[1, 3], keepdims=0),
            helper.make_node("ReduceSum", [IN_NAME], ["valid_cols_sum"], axes=[1, 2], keepdims=0),
            helper.make_node("Greater", ["valid_rows_sum", half32], ["valid_rows"]),
            helper.make_node("Greater", ["valid_cols_sum", half32], ["valid_cols"]),
            helper.make_node("ReduceSum", ["color32"], ["row_sums"], axes=[1, 3], keepdims=0),
            helper.make_node("ReduceSum", ["color32"], ["col_sums"], axes=[1, 2], keepdims=0),
            helper.make_node("ArgMax", ["row_sums"], ["r0"], axis=1, keepdims=0),
            helper.make_node("ArgMax", ["col_sums"], ["c0"], axis=1, keepdims=0),
            helper.make_node("Equal", ["coords", "r0"], ["row_mask"]),
            helper.make_node("Equal", ["coords", "c0"], ["col_mask"]),
            helper.make_node("Gather", ["color32", "r0"], ["row_src4"], axis=2),
            helper.make_node("Gather", ["color32", "c0"], ["col_src4"], axis=3),
            helper.make_node("Squeeze", ["row_src4"], ["row_src32"], axes=[1, 2]),
            helper.make_node("Squeeze", ["col_src4"], ["col_src32"], axes=[1, 3]),
            helper.make_node("Greater", ["row_src32", half32], ["row_fg"]),
            helper.make_node("Cast", ["row_fg"], ["row_fg16"], to=TensorProto.FLOAT16),
            helper.make_node("ReduceSum", ["row_fg16"], ["row_len"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["row_len", three_half], ["use4"]),
            helper.make_node("Cast", ["row_src32"], ["row_src"], to=TensorProto.FLOAT16),
            helper.make_node("Cast", ["col_src32"], ["col_src"], to=TensorProto.FLOAT16),
            *periodic_nodes,
            helper.make_node("Where", ["use4", row4, row3], ["row_vec"]),
            helper.make_node("Where", ["use4", col4, col3], ["col_vec"]),
            helper.make_node("Unsqueeze", ["row_vec"], ["row_vec4"], axes=[1, 2]),
            helper.make_node("Unsqueeze", ["col_vec"], ["col_vec4"], axes=[1, 3]),
            helper.make_node("Unsqueeze", ["row_mask"], ["row_mask4"], axes=[1, 3]),
            helper.make_node("Unsqueeze", ["col_mask"], ["col_mask4"], axes=[1, 2]),
            helper.make_node("Unsqueeze", ["valid_cols"], ["valid_cols4"], axes=[1, 2]),
            helper.make_node("Unsqueeze", ["valid_rows"], ["valid_rows4"], axes=[1, 3]),
            helper.make_node("Or", ["row_mask4", "col_mask4"], ["axis_any"]),
            helper.make_node("And", ["valid_rows4", "valid_cols4"], ["grid_valid"]),
            helper.make_node("And", ["axis_any", "grid_valid"], ["write_any"]),
            helper.make_node("Where", ["row_mask4", "row_vec4", "col_vec4"], ["write_color"]),
            helper.make_node("Where", ["grid_valid", zero16, neg_one], ["default_color"]),
            helper.make_node("Where", ["write_any", "write_color", "default_color"], ["valid_color"]),
            helper.make_node("Cast", ["valid_color"], ["color_img32"], to=TensorProto.INT32),
            helper.make_node("Equal", ["color_img32", colors], ["onehot_raw"]),
            helper.make_node("Cast", ["onehot_raw"], [OUT_NAME], to=TensorProto.FLOAT),
        ]
    )

    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits)
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


def validate_json(model: onnx.ModelProto) -> tuple[int, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    bad = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            g = np.array(ex["input"], dtype=np.int64)
            expected = np.array(ex["output"], dtype=np.int64)
            assert np.array_equal(solve(g), expected), f"reference mismatch {split} #{idx}"

            raw = _run_onnx(model, _grid_to_onehot(ex["input"]))
            pred = _onehot_to_grid(raw)[: expected.shape[0], : expected.shape[1]]
            active = raw[0] > 0.0
            inside = active[:, : expected.shape[0], : expected.shape[1]]
            outside = active.copy()
            outside[:, : expected.shape[0], : expected.shape[1]] = False
            total += 1
            if not np.array_equal(pred, expected) or not np.all(inside.sum(axis=0) == 1) or np.any(outside):
                bad += 1
                print(f"mismatch {split} #{idx}")
                print(pred)
                print(expected)
                break
    return bad, total


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    bad, total = validate_json(model)
    assert bad == 0, f"{bad} mismatches across {total} examples"
    print(f"verified {total} {TASK_ID} examples")
    print(score_file(BEST_PATH))


if __name__ == "__main__":
    main()
