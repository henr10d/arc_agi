"""ONNX generator for ARC task195.

Task rule: the gray input is a tight 9x9 arrangement of solid 3x3 blocks,
possibly translated inside the padded competition input. Collapse those blocks
to a 3x3 binary occupancy mask, then build the 9x9 output by replacing every
occupied coarse cell with a copy of that same 3x3 mask. Empty output cells are
black, occupied cells are gray.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task195"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
IN_H = 14
IN_W = 16
OUT = 9
GRAY = 5
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    ys, xs = np.where(g == GRAY)
    r0, c0 = int(ys.min()), int(xs.min())
    coarse = np.zeros((3, 3), dtype=np.int64)
    for br in range(3):
        for bc in range(3):
            block = g[r0 + br * 3 : r0 + br * 3 + 3, c0 + bc * 3 : c0 + bc * 3 + 3]
            coarse[br, bc] = int(np.any(block == GRAY))
    return (np.kron(coarse, coarse) * GRAY).astype(np.int64)


def _init(inits: List[onnx.TensorProto], arr: np.ndarray, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    starts = _init(inits, np.asarray([0, GRAY, 0, 0], dtype=np.int64), "starts")
    ends = _init(inits, np.asarray([1, GRAY + 1, IN_H, IN_W], dtype=np.int64), "ends")
    axes = _init(inits, np.asarray([0, 1, 2, 3], dtype=np.int64), "axes")
    _init(inits, np.asarray([0.5], dtype=np.float32), "half")
    offsets = _init(inits, np.asarray([0, 3, 6], dtype=np.int64), "offsets")
    sample_shape = _init(inits, np.asarray([1, 9], dtype=np.int64), "sample_shape")
    col_shape = _init(inits, np.asarray([1, 9, 1], dtype=np.int64), "col_shape")
    row_shape = _init(inits, np.asarray([1, 1, 9], dtype=np.int64), "row_shape")
    outer_shape = _init(inits, np.asarray([1, 3, 3, 3, 3], dtype=np.int64), "outer_shape")
    mask_shape = _init(inits, np.asarray([1, 1, OUT, OUT], dtype=np.int64), "mask_shape")
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts, ends, axes], ["gray"]),
            helper.make_node("ReduceSum", ["gray"], ["row_sum"], axes=[3], keepdims=0),
            helper.make_node("ReduceSum", ["gray"], ["col_sum"], axes=[2], keepdims=0),
            helper.make_node("Greater", ["row_sum", "half"], ["row_has"]),
            helper.make_node("Greater", ["col_sum", "half"], ["col_has"]),
            helper.make_node("Cast", ["row_has"], ["row_has_i"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["col_has"], ["col_has_i"], to=TensorProto.FLOAT),
            helper.make_node("ArgMax", ["row_has_i"], ["r0"], axis=2, keepdims=1),
            helper.make_node("ArgMax", ["col_has_i"], ["c0"], axis=2, keepdims=1),
            helper.make_node("Add", ["r0", offsets], ["rows"]),
            helper.make_node("Add", ["c0", offsets], ["cols"]),
            helper.make_node("Gather", ["gray", "rows"], ["sample_rows"], axis=2),
            helper.make_node("Gather", ["sample_rows", "cols"], ["sample_grid"], axis=5),
            helper.make_node("Reshape", ["sample_grid", sample_shape], ["coarse"]),
            helper.make_node("Reshape", ["coarse", col_shape], ["coarse_col"]),
            helper.make_node("Reshape", ["coarse", row_shape], ["coarse_row"]),
            helper.make_node("MatMul", ["coarse_col", "coarse_row"], ["outer"]),
            helper.make_node("Reshape", ["outer", outer_shape], ["outer5"]),
            helper.make_node("Transpose", ["outer5"], ["grid5"], perm=[0, 1, 3, 2, 4]),
            helper.make_node("Reshape", ["grid5", mask_shape], ["fg"]),
            helper.make_node("Sub", ["fg", "fg"], ["zero"]),
            helper.make_node("Sub", ["half", "fg"], ["bg_logits"]),
            helper.make_node(
                "Concat",
                ["bg_logits", "zero", "zero", "zero", "zero", "fg"],
                ["out6"],
                axis=1,
            ),
            helper.make_node("Pad", ["out6"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 4, H - OUT, W - OUT]),
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


def _grid_to_onehot(grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def validate(path: Path) -> None:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data[split]):
            inp = np.asarray(example["input"], dtype=np.int64)
            expected_grid = np.asarray(example["output"], dtype=np.int64)
            expected = _grid_to_onehot(expected_grid)
            pred = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(inp)})[0]
            if not np.array_equal(pred > 0.0, expected > 0.0):
                raise AssertionError(f"{split}#{idx} mismatch")
            ref = solve(inp)
            if not np.array_equal(ref, expected_grid):
                raise AssertionError(f"{split}#{idx} reference rule mismatch")


def main() -> None:
    model = build_model()
    onnx.save(model, str(BEST_PATH))
    validate(BEST_PATH)
    result = score_file(BEST_PATH)
    print(
        f"{BEST_PATH} memory={result.get('memory')} params={result.get('params')} "
        f"cost={result.get('cost')} score={result.get('score')}"
    )


if __name__ == "__main__":
    main()
