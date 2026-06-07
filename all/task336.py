"""ONNX generator for ARC task336: fill a gray U-shape and extend a cyan stem.

Task rule: the input is a 10x10 grid with black background and one gray
U-shaped frame (color 5), open on exactly one side.  Preserve every gray frame
cell, fill the rectangle inside the frame with cyan (color 8), and draw a cyan
ray from the midpoint of the opening outward to that side of the grid.  The
provided train/test/arc-gen examples include top, bottom, left, and right
openings.

ONNX approach: crop the fixed 10x10 task area, find the gray bounding box with
coordinate masks, identify the missing cell on each bbox side, then render the
cyan interior/stem with row and column comparisons before padding to 30x30.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task336"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
CORE = 10
PAD = H - CORE
GRAY = 5
CYAN = 8
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def solve_grid(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference implementation of the gray U-shape fill-and-stem rule."""
    arr = np.asarray(grid, dtype=np.int64)
    out = arr.copy()
    rows, cols = np.where(arr == GRAY)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())

    out[r0 + 1 : r1, c0 + 1 : c1] = CYAN

    top_missing = np.where(arr[r0, c0 : c1 + 1] != GRAY)[0]
    bottom_missing = np.where(arr[r1, c0 : c1 + 1] != GRAY)[0]
    left_missing = np.where(arr[r0 : r1 + 1, c0] != GRAY)[0]
    right_missing = np.where(arr[r0 : r1 + 1, c1] != GRAY)[0]

    if top_missing.size:
        col = c0 + int((int(top_missing[0]) + int(top_missing[-1])) // 2)
        out[: r0 + 1, col] = CYAN
    elif bottom_missing.size:
        col = c0 + int((int(bottom_missing[0]) + int(bottom_missing[-1])) // 2)
        out[r1:, col] = CYAN
    elif left_missing.size:
        row = r0 + int((int(left_missing[0]) + int(left_missing[-1])) // 2)
        out[row, : c0 + 1] = CYAN
    elif right_missing.size:
        row = r0 + int((int(right_missing[0]) + int(right_missing[-1])) // 2)
        out[row, c1:] = CYAN
    return out


def _init(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    return _init(inits, name, np.asarray(arr, dtype=np.int64))


def _f32(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    return _init(inits, name, np.asarray(arr, dtype=np.float32))


def _bool(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    return _init(inits, name, np.asarray(arr, dtype=np.bool_))


def _node(nodes: list[onnx.NodeProto], op: str, inputs: list[str], output: str, **attrs: Any) -> str:
    nodes.append(helper.make_node(op, inputs, [output], **attrs))
    return output


def _not(nodes: list[onnx.NodeProto], value: str, out: str) -> str:
    return _node(nodes, "Not", [value], out)


def _and(nodes: list[onnx.NodeProto], left: str, right: str, out: str) -> str:
    return _node(nodes, "And", [left, right], out)


def _or(nodes: list[onnx.NodeProto], left: str, right: str, out: str) -> str:
    return _node(nodes, "Or", [left, right], out)


def _ge(nodes: list[onnx.NodeProto], left: str, right: str, out: str) -> str:
    lt = _node(nodes, "Less", [left, right], f"{out}_lt")
    return _not(nodes, lt, out)


def _le(nodes: list[onnx.NodeProto], left: str, right: str, out: str) -> str:
    gt = _node(nodes, "Greater", [left, right], f"{out}_gt")
    return _not(nodes, gt, out)


def _eq(nodes: list[onnx.NodeProto], left: str, right: str, out: str) -> str:
    return _node(nodes, "Equal", [left, right], out)


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, "axes4", [0, 1, 2, 3])
    gray_st = _i64(inits, "gray_st", [0, GRAY, 0, 0])
    gray_en = _i64(inits, "gray_en", [1, GRAY + 1, CORE, CORE])

    zero = _f32(inits, "zero", [0.0])
    three = _i64(inits, "three", [3])
    nine_i = _i64(inits, "nine_i", [CORE - 1])
    vec_shape = _i64(inits, "vec_shape", [1])
    rev_idx = _i64(inits, "rev_idx", list(range(CORE - 1, -1, -1)))
    cyan_value = _f32(
        inits,
        "cyan_value",
        np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32).reshape(1, C, 1, 1),
    )
    false_right = _bool(inits, "false_right", np.zeros((1, 1, CORE, PAD), dtype=np.bool_))
    false_bottom = _bool(inits, "false_bottom", np.zeros((1, 1, PAD, W), dtype=np.bool_))

    rows = np.arange(CORE, dtype=np.int64).reshape(1, 1, CORE, 1)
    cols = np.arange(CORE, dtype=np.int64).reshape(1, 1, 1, CORE)
    _i64(inits, "rows", rows)
    _i64(inits, "cols", cols)

    gray_f = _node(nodes, "Slice", [IN_NAME, gray_st, gray_en, axes4], "gray_f")

    row_has = _node(nodes, "ReduceMax", [gray_f], "row_has", axes=[3], keepdims=1)
    col_has = _node(nodes, "ReduceMax", [gray_f], "col_has", axes=[2], keepdims=1)
    r0_i = _node(nodes, "ArgMax", [row_has], "r0_i", axis=2, keepdims=1)
    c0_i = _node(nodes, "ArgMax", [col_has], "c0_i", axis=3, keepdims=1)
    row_rev = _node(nodes, "Gather", [row_has, rev_idx], "row_rev", axis=2)
    col_rev = _node(nodes, "Gather", [col_has, rev_idx], "col_rev", axis=3)
    rrev_i = _node(nodes, "ArgMax", [row_rev], "rrev_i", axis=2, keepdims=1)
    crev_i = _node(nodes, "ArgMax", [col_rev], "crev_i", axis=3, keepdims=1)
    r1_i = _node(nodes, "Sub", [nine_i, rrev_i], "r1_i")
    c1_i = _node(nodes, "Sub", [nine_i, crev_i], "c1_i")
    r0 = r0_i
    r1 = r1_i
    c0 = c0_i
    c1 = c1_i

    row_gt_r0 = _node(nodes, "Greater", ["rows", r0], "row_gt_r0")
    row_lt_r1 = _node(nodes, "Less", ["rows", r1], "row_lt_r1")
    col_gt_c0 = _node(nodes, "Greater", ["cols", c0], "col_gt_c0")
    col_lt_c1 = _node(nodes, "Less", ["cols", c1], "col_lt_c1")
    inside_rows = _and(nodes, row_gt_r0, row_lt_r1, "inside_rows")
    inside_cols = _and(nodes, col_gt_c0, col_lt_c1, "inside_cols")
    interior = _and(nodes, inside_rows, inside_cols, "interior")

    stem_col = _node(nodes, "Add", [c0, three], "stem_col")
    stem_row = _node(nodes, "Add", [r0, three], "stem_row")

    r0_v = _node(nodes, "Reshape", [r0, vec_shape], "r0_v")
    r1_v = _node(nodes, "Reshape", [r1, vec_shape], "r1_v")
    c0_v = _node(nodes, "Reshape", [c0, vec_shape], "c0_v")
    c1_v = _node(nodes, "Reshape", [c1, vec_shape], "c1_v")
    stem_col_v = _node(nodes, "Reshape", [stem_col, vec_shape], "stem_col_v")
    stem_row_v = _node(nodes, "Reshape", [stem_row, vec_shape], "stem_row_v")

    row_r0 = _node(nodes, "Gather", [gray_f, r0_v], "row_r0", axis=2)
    row_r1 = _node(nodes, "Gather", [gray_f, r1_v], "row_r1", axis=2)
    row_stem = _node(nodes, "Gather", [gray_f, stem_row_v], "row_stem", axis=2)

    def open_present(row_plane: str, col_idx: str, prefix: str) -> str:
        cell = _node(nodes, "Gather", [row_plane, col_idx], f"{prefix}_cell", axis=3)
        cell_is_gray = _node(nodes, "Greater", [cell, zero], f"{prefix}_cell_is_gray")
        return _not(nodes, cell_is_gray, f"{prefix}_present")

    top_present = open_present(row_r0, stem_col_v, "top")
    bottom_present = open_present(row_r1, stem_col_v, "bottom")
    left_present = open_present(row_stem, c0_v, "left")
    right_present = open_present(row_stem, c1_v, "right")

    stem_col_eq = _eq(nodes, "cols", stem_col, "stem_col_eq")
    stem_row_eq = _eq(nodes, "rows", stem_row, "stem_row_eq")
    row_le_top = _le(nodes, "rows", r0, "row_le_top")
    row_ge_bottom = _ge(nodes, "rows", r1, "row_ge_bottom")
    col_le_left = _le(nodes, "cols", c0, "col_le_left")
    col_ge_right = _ge(nodes, "cols", c1, "col_ge_right")

    vertical_present = _or(nodes, top_present, bottom_present, "vertical_present")
    vertical_cols = _and(nodes, vertical_present, stem_col_eq, "vertical_cols")
    top_rows = _and(nodes, top_present, row_le_top, "top_rows")
    bottom_rows = _and(nodes, bottom_present, row_ge_bottom, "bottom_rows")
    vertical_rows = _or(nodes, top_rows, bottom_rows, "vertical_rows")
    vertical_stem = _and(nodes, vertical_cols, vertical_rows, "vertical_stem")

    horizontal_present = _or(nodes, left_present, right_present, "horizontal_present")
    horizontal_rows = _and(nodes, horizontal_present, stem_row_eq, "horizontal_rows")
    left_cols = _and(nodes, left_present, col_le_left, "left_cols")
    right_cols = _and(nodes, right_present, col_ge_right, "right_cols")
    horizontal_cols = _or(nodes, left_cols, right_cols, "horizontal_cols")
    horizontal_stem = _and(nodes, horizontal_rows, horizontal_cols, "horizontal_stem")
    stem = _or(nodes, vertical_stem, horizontal_stem, "stem")
    cyan = _or(nodes, interior, stem, "cyan")

    cyan_wide = _node(nodes, "Concat", [cyan, false_right], "cyan_wide", axis=3)
    cyan30 = _node(nodes, "Concat", [cyan_wide, false_bottom], "cyan30", axis=2)
    _node(nodes, "Where", [cyan30, cyan_value, IN_NAME], OUT_NAME)

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


def _expected_onehot(grid: list[list[int]]) -> np.ndarray:
    return convert_to_numpy({"input": grid}, "input")


def verify_model(path: Path) -> tuple[bool, str]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        return False, f"ORT load failed: {exc}"
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            inp = convert_to_numpy(example, "input")
            exp = _expected_onehot(example["output"])
            if inp is None or exp is None:
                continue
            got = session.run([OUT_NAME], {IN_NAME: inp})[0] > 0.0
            if not np.array_equal(got, exp.astype(bool)):
                ref = solve_grid(example["input"])
                if not np.array_equal(ref, np.asarray(example["output"], dtype=np.int64)):
                    return False, f"{split}[{idx}] reference solver failed"
                return False, f"{split}[{idx}] ONNX mismatch"
    return True, "ok"


def main() -> None:
    candidate = OUT_DIR / f"{TASK_ID}_candidate.onnx"
    model = build_model()
    onnx.save(model, candidate)

    valid, message = verify_model(candidate)
    if not valid:
        raise SystemExit(message)

    result = score_file(candidate)
    if not result["valid"]:
        raise SystemExit(f"score invalid: {result['error']}")

    shutil.copyfile(candidate, BEST_PATH)
    print(
        f"{BEST_PATH.name}: memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={float(result['score']):.6f}"
    )


if __name__ == "__main__":
    main()
