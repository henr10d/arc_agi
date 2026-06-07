"""Minimal ONNX for ARC task211 using row/column mirroring.

Task rule: the 3x2 input is expanded to 9x4.  Each input row [a, b]
becomes [b, a, a, b].  The three expanded rows are then stacked as
reversed rows, original rows, and reversed rows again.  Colors, including
black/background, are preserved exactly.

ONNX: slice the 3x2 one-hot input on the spatial axes only, gather columns
[1, 0, 0, 1], gather rows [2, 1, 0, 0, 1, 2, 2, 1, 0], then pad to the
required 30x30 output.
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

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task211"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
IH = 3
IW = 2
OH = 9
OW = 4
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


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference implementation for the row/column expansion."""
    g = np.asarray(grid, dtype=np.int64)
    expanded = np.stack([g[:, 1], g[:, 0], g[:, 0], g[:, 1]], axis=1)
    return np.concatenate([expanded[::-1], expanded, expanded[::-1]], axis=0)


def _grid_to_onehot(grid: List[List[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _expected_onehot(grid: List[List[int]]) -> np.ndarray:
    return _grid_to_onehot(grid)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, [2, 3], "axes")
    starts = _i64(inits, [0, 0], "starts")
    ends = _i64(inits, [IH, IW], "ends")
    col_idx = _i64(inits, [1, 0, 0, 1], "col_idx")
    row_idx = _i64(inits, [2, 1, 0, 0, 1, 2, 2, 1, 0], "row_idx")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts, ends, axes], ["core"]),
            helper.make_node("Gather", ["core", col_idx], ["wide"], axis=3),
            helper.make_node("Gather", ["wide", row_idx], ["out9"], axis=2),
            helper.make_node("Pad", ["out9"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - OH, W - OW]),
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


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            ref_grid = solve(np.array(ex["input"], dtype=np.int64))
            expected_grid = np.array(ex["output"], dtype=np.int64)
            if not np.array_equal(ref_grid, expected_grid):
                print(f"{split}[{idx}] reference mismatch")
                bad += 1
                continue

            pred = _run_onnx(model, _grid_to_onehot(ex["input"]))
            expected = _expected_onehot(ex["output"])
            if not np.array_equal(pred > 0.0, expected > 0.0):
                print(f"{split}[{idx}] ONNX mismatch")
                bad += 1
    return bad


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    bad = validate_json(model)
    print(f"{TASK_ID}.json: {'PASS' if bad == 0 else f'FAIL ({bad} wrong)'}")

    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"nodes:   {len(model.graph.node)}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
