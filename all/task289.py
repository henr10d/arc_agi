"""ONNX for ARC task289: expand a 3x3 grid by its distinct color count.

Task rule: count the distinct non-black colors present in the 3x3 input grid.
Each input cell, including black cells, becomes an s by s block of the same
color where s is that count. The output occupies the top-left 3s by 3s area of
the 30x30 NeuroGolf canvas; cells outside that area remain padding with no
active one-hot channel.

The rule is learned from the train JSON by checking candidate features. Raw
non-black cell count is ambiguous; the simplest perfect train fit is
``k == number of distinct non-black colors``.

ONNX approach: derive s from the used non-zero one-hot channels of the 3x3
core, gather a precomputed coordinate vector for that scale, and apply it to
the float 3x3 one-hot core padded with an all-zero sentinel row and column.
The final Gather directly produces the required float output tensor; sentinel
coordinates leave the unused canvas empty.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task289"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task289.onnx"
DATA_PATH = ROOT / "data" / "task289.json"

C = 10
H = W = 30
CORE = 3
SENTINEL = CORE
MAX_SCALE = 9
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10
IN_NAME = "input"
OUT_NAME = "output"


def _init(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], name: str, vals: Any) -> str:
    return _init(inits, name, np.asarray(vals, dtype=np.int64))


def _f32(inits: list[onnx.TensorProto], name: str, vals: Any) -> str:
    return _init(inits, name, np.asarray(vals, dtype=np.float32))


def _scale_indices(scale: int) -> np.ndarray:
    idx = np.full((H,), SENTINEL, dtype=np.int64)
    for out_idx in range(CORE * scale):
        idx[out_idx] = out_idx // scale
    return idx


def _onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _connected_components(arr: np.ndarray) -> int:
    seen = np.zeros(arr.shape, dtype=bool)
    components = 0
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            if arr[r, c] == 0 or seen[r, c]:
                continue
            components += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    in_bounds = 0 <= nr < arr.shape[0] and 0 <= nc < arr.shape[1]
                    if in_bounds and arr[nr, nc] != 0 and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return components


def _features(grid: list[list[int]] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(grid, dtype=np.int64)
    nz = arr != 0
    rows = np.where(nz.any(axis=1))[0]
    cols = np.where(nz.any(axis=0))[0]
    colors, counts = np.unique(arr[nz], return_counts=True)
    bbox_h = int(rows[-1] - rows[0] + 1) if len(rows) else 0
    bbox_w = int(cols[-1] - cols[0] + 1) if len(cols) else 0
    return {
        "nonzero_cells": int(nz.sum()),
        "distinct_nonzero_colors": int(len(colors)),
        "connected_components": _connected_components(arr),
        "bbox_height": bbox_h,
        "bbox_width": bbox_w,
        "bbox_area": bbox_h * bbox_w,
        "occupied_rows": int(len(rows)),
        "occupied_cols": int(len(cols)),
        "color_multiplicities": tuple(sorted(int(v) for v in counts)),
    }


def _infer_scale(ex: dict[str, list[list[int]]]) -> int:
    out = np.asarray(ex["output"], dtype=np.int64)
    if out.shape[0] % CORE != 0 or out.shape[1] != out.shape[0]:
        raise AssertionError(f"unexpected output shape {out.shape}")
    scale = out.shape[0] // CORE
    expected = np.kron(np.asarray(ex["input"], dtype=np.int64), np.ones((scale, scale), dtype=np.int64))
    if not np.array_equal(out, expected):
        raise AssertionError("output is not a kron expansion of the input")
    return scale


def learn_scale_rule() -> str:
    train = [(ex, _infer_scale(ex)) for split, _, ex in _load_examples() if split == "train"]
    scalar_candidates = (
        "nonzero_cells",
        "distinct_nonzero_colors",
        "connected_components",
        "bbox_height",
        "bbox_width",
        "bbox_area",
        "occupied_rows",
        "occupied_cols",
    )
    for name in scalar_candidates:
        if all(_features(ex["input"])[name] == scale for ex, scale in train):
            return name

    lookup_candidates = (*scalar_candidates, "color_multiplicities")
    for name in lookup_candidates:
        mapping: dict[Any, int] = {}
        for ex, scale in train:
            value = _features(ex["input"])[name]
            if value in mapping and mapping[value] != scale:
                break
            mapping[value] = scale
        else:
            return f"{name}_lookup"

    raise AssertionError("no candidate scale rule fits all train examples")


def predict_scale(grid: list[list[int]] | np.ndarray) -> int:
    return int(_features(grid)["distinct_nonzero_colors"])


def solve_grid(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    scale = predict_scale(arr)
    return np.kron(arr, np.ones((scale, scale), dtype=np.int64))


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    zero = _f32(inits, "zero", np.array(0.0, dtype=np.float32))
    one_i = _i64(inits, "one_i", np.array(1, dtype=np.int64))
    crop_st = _i64(inits, "crop_st", [0, 0, 0, 0])
    crop_en = _i64(inits, "crop_en", [1, C, CORE, CORE])
    axes_all = _i64(inits, "axes_all", [0, 1, 2, 3])
    ch_st = _i64(inits, "ch_st", [1])
    ch_en = _i64(inits, "ch_en", [C])
    ch_axis = _i64(inits, "ch_axis", [1])
    row_idx_table = _i64(inits, "row_idx_table", np.stack([_scale_indices(s) for s in range(1, MAX_SCALE + 1)]))
    zero_col = _f32(inits, "zero_col", np.zeros((1, C, CORE, 1), dtype=np.float32))
    zero_row = _f32(inits, "zero_row", np.zeros((1, C, 1, CORE + 1), dtype=np.float32))

    nodes.append(helper.make_node("Slice", [IN_NAME, crop_st, crop_en, axes_all], ["core"]))
    nodes.append(helper.make_node("ReduceSum", ["core"], ["color_sum"], axes=[2, 3], keepdims=0))
    nodes.append(helper.make_node("Slice", ["color_sum", ch_st, ch_en, ch_axis], ["nonzero_sum"]))
    nodes.append(helper.make_node("Greater", ["nonzero_sum", zero], ["color_used"]))
    nodes.append(helper.make_node("Cast", ["color_used"], ["color_used_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("ReduceSum", ["color_used_f"], ["scale_f"], keepdims=0))
    nodes.append(helper.make_node("Cast", ["scale_f"], ["scale_i"], to=TensorProto.INT64))
    nodes.append(helper.make_node("Sub", ["scale_i", one_i], ["scale_idx"]))
    nodes.append(helper.make_node("Gather", ["row_idx_table", "scale_idx"], ["row_idx"], axis=0))

    nodes.append(helper.make_node("Concat", ["core", "zero_col"], ["core_wide"], axis=3))
    nodes.append(helper.make_node("Concat", ["core_wide", "zero_row"], ["core4"], axis=2))
    nodes.append(helper.make_node("Gather", ["core4", "row_idx"], ["rows"], axis=2))
    nodes.append(helper.make_node("Gather", ["rows", "row_idx"], [OUT_NAME], axis=3))

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


def _load_examples() -> list[tuple[str, int, dict[str, list[list[int]]]]]:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    examples: list[tuple[str, int, dict[str, list[list[int]]]]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            examples.append((split, idx, ex))
    return examples


def validate_python_solver(splits: tuple[str, ...] = ("train", "test", "arc-gen")) -> None:
    for split, idx, ex in _load_examples():
        if split not in splits:
            continue
        _infer_scale(ex)
        expected_grid = np.asarray(ex["output"], dtype=np.int64)
        solved = solve_grid(ex["input"])
        if not np.array_equal(solved, expected_grid):
            raise AssertionError(f"{split} {idx} Python solver failed")


def validate(path: Path, splits: tuple[str, ...] = ("train", "test", "arc-gen")) -> None:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    for split, idx, ex in _load_examples():
        if split not in splits:
            continue
        pred = session.run([OUT_NAME], {IN_NAME: _onehot(ex["input"])})[0]
        expected = _onehot(ex["output"])
        if not np.array_equal(pred > 0.0, expected > 0.0):
            got = np.argmax(pred[0], axis=0)
            exp = np.argmax(expected[0], axis=0)
            diff = np.argwhere((pred > 0.0) != (expected > 0.0))
            raise AssertionError(
                f"{split} {idx} failed at {diff[:5].tolist()}\n"
                f"got:\n{got[:15, :15]}\nexpected:\n{exp[:15, :15]}"
            )


def main() -> None:
    rule = learn_scale_rule()
    if rule != "distinct_nonzero_colors":
        raise AssertionError(f"unexpected learned rule: {rule}")
    validate_python_solver()
    onnx.save(build_model(), BEST_PATH)
    validate(BEST_PATH)
    result = score_file(BEST_PATH)
    if not result["valid"]:
        raise RuntimeError(result["error"])
    print(
        f"kept {BEST_PATH}: memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={float(result['score']):.6f}"
    )


if __name__ == "__main__":
    main()
