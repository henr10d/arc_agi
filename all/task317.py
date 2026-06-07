"""Minimal ONNX for ARC task317 using Kaggle one-hot I/O.

Task rule: the 9x9 input contains isolated gray cells on a 3x3 lattice at
rows/columns 1, 4, and 7.  Replace every gray center with a solid blue 3x3
block in the corresponding lattice block, unioning adjacent blocks naturally,
and keep all other cells black.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task317"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task317.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
GRID = 9
LATTICE = 3
BLOCK = 3
GRAY = 5
BLUE = 1
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def solve(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    """Reference implementation: paint a blue 3x3 block around each gray cell."""
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(arr)
    rows, cols = arr.shape
    for r in range(rows):
        for c in range(cols):
            if arr[r, c] == GRAY:
                out[max(0, r - 1) : min(rows, r + 2), max(0, c - 1) : min(cols, c + 2)] = BLUE
    return out


def _grid_to_onehot(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            color = int(arr[r, c])
            if 0 <= color < C:
                out[0, color, r, c] = 1.0
    return out


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    starts = _i64(inits, [0, GRAY, 1, 1], "starts")
    ends = _i64(inits, [1, GRAY + 1, GRID - 1, GRID - 1], "ends")
    axes = _i64(inits, [0, 1, 2, 3], "axes")
    steps = _i64(inits, [1, 1, BLOCK, BLOCK], "steps")
    reps6 = _i64(inits, [1, 1, 1, BLOCK, 1, BLOCK], "reps6")
    shape9 = _i64(inits, [1, 2, GRID, GRID], "shape9")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts, ends, axes, steps], ["centers"]),
            helper.make_node("Cast", ["centers"], ["blue3"], to=TensorProto.BOOL),
            helper.make_node("Not", ["blue3"], ["black3"]),
            helper.make_node("Concat", ["black3", "blue3"], ["out2_3"], axis=1),
            helper.make_node("Unsqueeze", ["out2_3"], ["out2_6"], axes=[3, 5]),
            helper.make_node("Tile", ["out2_6", reps6], ["out2_6_tiled"]),
            helper.make_node("Reshape", ["out2_6_tiled", shape9], ["out2_bool"]),
            helper.make_node("Cast", ["out2_bool"], ["out2"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out2"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, C - 2, H - GRID, W - GRID],
            ),
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


def validate_model(model: onnx.ModelProto) -> tuple[bool, str]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(data[split]):
            expected_grid = np.asarray(example["output"], dtype=np.int64)
            solved_grid = solve(example["input"])
            if not np.array_equal(solved_grid, expected_grid):
                return False, f"{split}#{index}: reference rule mismatch"

            expected = _grid_to_onehot(expected_grid) > 0
            pred = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(example["input"])})[0]
            if pred.shape != tuple(SHAPE):
                return False, f"{split}#{index}: bad output shape {pred.shape}"
            if not np.array_equal(pred > 0, expected):
                return False, f"{split}#{index}: ONNX output mismatch"

    return True, "PASS"


def main() -> None:
    model = build_model()
    ok, message = validate_model(model)
    if not ok:
        raise SystemExit(message)

    onnx.save(model, BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"validation: {message}")
    print(f"path: {BEST_PATH}")
    print(f"memory: {result.get('memory')}")
    print(f"params: {result.get('params')}")
    print(f"cost: {result.get('cost')}")
    print(f"score: {result.get('score')}")
    if not result.get("valid"):
        raise SystemExit(result.get("error") or "score_model reported invalid model")


if __name__ == "__main__":
    main()
