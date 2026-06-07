"""ONNX solution for task213: compact colored guide lines into a square.

Task rule: ignore the gray filled rectangle (color 5). The remaining colored
objects are either horizontal or vertical one-cell-wide guide lines, sometimes
split by the gray block. Read the guide-line colors in spatial order: top to
bottom for horizontal lines, left to right for vertical lines. The output is an
N by N square using the same orientation as the guides: horizontal source lines
become full-width color rows; vertical source lines become full-height color
columns.

The graph detects line rows/columns with per-color population counts, uses
float16 masks plus TopK to compact the first seven active coordinates, merges
the horizontal and vertical compact color vectors with their N-wide/N-tall
output masks, then casts the compact bool result to float before padding to the
competition I/O shape.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

TASK_ID = "task213"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
MAX_N = 7
PAD = H - MAX_N
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference implementation for direct grid validation."""
    arr = np.asarray(grid, dtype=np.int64)
    candidates: list[tuple[str, int, int]] = []

    for r in range(arr.shape[0]):
        for color in range(1, C):
            if color == 5:
                continue
            if int(np.count_nonzero(arr[r, :] == color)) > 3:
                candidates.append(("h", r, color))

    vertical: list[tuple[str, int, int]] = []
    for c in range(arr.shape[1]):
        for color in range(1, C):
            if color == 5:
                continue
            if int(np.count_nonzero(arr[:, c] == color)) > 3:
                vertical.append(("v", c, color))

    if len(vertical) > len(candidates):
        colors = [color for _, _, color in vertical]
        out = np.tile(np.asarray(colors, dtype=np.int64), (len(colors), 1))
    else:
        colors = [color for _, _, color in candidates]
        out = np.repeat(np.asarray(colors, dtype=np.int64)[:, None], len(colors), axis=1)
    return out


def _i64(inits: List[onnx.TensorProto], vals: Iterable[int] | np.ndarray, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))
    return name


def _f32(inits: List[onnx.TensorProto], vals: Iterable[float] | np.ndarray, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.float32), name=name))
    return name


def _f16(inits: List[onnx.TensorProto], vals: Iterable[float] | np.ndarray, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.float16), name=name))
    return name


def _bool(inits: List[onnx.TensorProto], vals: np.ndarray, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=bool), name=name))
    return name


def _grid_to_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            color = int(arr[r, c])
            if 0 <= color < C:
                out[0, color, r, c] = 1.0
    return out


def _expected_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    return _grid_to_onehot(grid)


def build_onnx_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    # Constants are kept as initializers so the graph remains opset-10 friendly.
    color_mask = np.zeros((1, C, 1), dtype=bool)
    color_mask[:, [1, 2, 3, 4, 6, 7, 8, 9], :] = True
    mask = _bool(inits, color_mask, "m")
    threshold = _f32(inits, [3.0], "t")
    zero = _f16(inits, [0.0], "z")
    topk = _i64(inits, [MAX_N], "k")

    # Detect candidate guide rows and guide columns. A real guide has more than
    # three cells of the same non-gray color on its row/column; incidental
    # perpendicular crossings stay below that threshold.
    nodes.extend(
        [
            helper.make_node("ReduceSum", [IN_NAME], ["row_counts"], axes=[3], keepdims=0),
            helper.make_node("Greater", ["row_counts", threshold], ["row_color_raw"]),
            helper.make_node("And", ["row_color_raw", mask], ["row_color"]),
            helper.make_node("Cast", ["row_color"], ["row_color_f"], to=TensorProto.FLOAT16),
            helper.make_node("ReduceSum", ["row_color_f"], ["row_active_f1"], axes=[1], keepdims=0),
            helper.make_node("Greater", ["row_active_f1", zero], ["row_active_1"]),
            helper.make_node("Squeeze", ["row_active_1"], ["row_active"], axes=[0]),
            helper.make_node("Cast", ["row_active"], ["row_active_f"], to=TensorProto.FLOAT16),
            helper.make_node("TopK", ["row_active_f", topk], ["row_topv", "row_idx"], axis=0),
            helper.make_node("Gather", ["row_color", "row_idx"], ["row_pick"], axis=2),
            helper.make_node("Gather", ["row_active", "row_idx"], ["row_pick_active"], axis=0),
            helper.make_node("Unsqueeze", ["row_pick"], ["h_colors"], axes=[3]),
            helper.make_node("Unsqueeze", ["row_pick_active"], ["h_mask1"], axes=[0, 1, 2]),
            helper.make_node("ReduceSum", [IN_NAME], ["col_counts"], axes=[2], keepdims=0),
            helper.make_node("Greater", ["col_counts", threshold], ["col_color_raw"]),
            helper.make_node("And", ["col_color_raw", mask], ["col_color"]),
            helper.make_node("Cast", ["col_color"], ["col_color_f"], to=TensorProto.FLOAT16),
            helper.make_node("ReduceSum", ["col_color_f"], ["col_active_f1"], axes=[1], keepdims=0),
            helper.make_node("Greater", ["col_active_f1", zero], ["col_active_1"]),
            helper.make_node("Squeeze", ["col_active_1"], ["col_active"], axes=[0]),
            helper.make_node("Cast", ["col_active"], ["col_active_f"], to=TensorProto.FLOAT16),
            helper.make_node("TopK", ["col_active_f", topk], ["col_topv", "col_idx"], axis=0),
            helper.make_node("Gather", ["col_color", "col_idx"], ["col_pick"], axis=2),
            helper.make_node("Gather", ["col_active", "col_idx"], ["col_pick_active"], axis=0),
            helper.make_node("Unsqueeze", ["col_pick"], ["v_colors"], axes=[2]),
            helper.make_node("Unsqueeze", ["col_pick_active"], ["v_mask1"], axes=[0, 1, 3]),
            helper.make_node("Or", ["h_colors", "v_colors"], ["hv_colors"]),
            helper.make_node("Or", ["h_mask1", "v_mask1"], ["hv_mask"]),
            helper.make_node("And", ["hv_colors", "hv_mask"], ["out7b"]),
            helper.make_node("Cast", ["out7b"], ["out7"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out7"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]),
        ]
    )

    graph = helper.make_graph(nodes, "task213", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def save_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    model = build_onnx_model()
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return model


def validate_model(model: onnx.ModelProto) -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)

    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    checked = 0
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            expected_grid = np.asarray(example["output"], dtype=np.int64)
            ref = solve(example["input"])
            if not np.array_equal(ref, expected_grid):
                raise AssertionError(f"reference mismatch {split} {idx}: {ref.tolist()} != {expected_grid.tolist()}")
            pred = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(example["input"])})[0]
            expected = _expected_onehot(expected_grid)
            if not np.array_equal(pred > 0.0, expected > 0.0):
                raise AssertionError(f"ONNX mismatch {split} {idx}")
            checked += 1
    print(f"validated {checked} examples")


def main() -> None:
    model = save_model()
    validate_model(model)
    print(BEST_PATH)


if __name__ == "__main__":
    main()
