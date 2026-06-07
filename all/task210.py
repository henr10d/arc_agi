"""Minimal ONNX for ARC task210: append a vertical mirror of the 3x3 input.

Task rule: the input is a 3x3 binary grid using colors 0 and 1. The output is
6x3: first copy the input unchanged, then append the same rows in reverse
vertical order. This is not vertical 2x nearest-neighbor scaling; every train
example is input followed by its top-to-bottom reflection.

ONNX approach: gather rows directly from the padded competition input. Rows
[0, 1, 2, 2, 1, 0] form the 6x3 answer, and the remaining output rows gather
input row 3, which is padding and therefore all-zero. Width padding is already
zero in the input rows, so the Gather node can write the graph output directly.
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

TASK_ID = "task210"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
CORE_H = 3
CORE_W = 3
OUT_H = 6
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def solve(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    """Reference implementation: concatenate the 3x3 input and its vertical flip."""
    g = np.asarray(grid, dtype=np.int64)[:CORE_H, :CORE_W]
    return np.concatenate([g, g[::-1]], axis=0)


def _i64(inits: List[onnx.TensorProto], vals: list[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))
    return name


def _grid_to_onehot(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def build_onnx_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    row_indices = _i64(inits, [0, 1, 2, 2, 1, 0] + [CORE_H] * (H - OUT_H), "row_indices")

    nodes.extend(
        [
            helper.make_node("Gather", [IN_NAME, row_indices], [OUT_NAME], axis=2),
        ]
    )

    graph = helper.make_graph(nodes, "g", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def validate_hypotheses() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    def scale2(grid: list[list[int]]) -> np.ndarray:
        g = np.asarray(grid, dtype=np.int64)
        out = np.zeros((g.shape[0] * 2, g.shape[1]), dtype=np.int64)
        for r in range(g.shape[0]):
            out[2 * r : 2 * r + 2] = g[r]
        return out

    hypotheses = {
        "H1 per-cell vertical duplication": scale2,
        "H2 nearest-neighbor vertical 2x scaling": scale2,
        "H3 per-column doubled segments and gaps": scale2,
        "mirror append": solve,
    }
    for name, fn in hypotheses.items():
        failures = []
        for idx, example in enumerate(data["train"]):
            pred = fn(example["input"])
            expected = np.asarray(example["output"], dtype=np.int64)
            if pred.shape != expected.shape or not np.array_equal(pred, expected):
                failures.append(idx)
        status = "OK" if not failures else f"FAIL train indices {failures}"
        print(f"{name}: {status}")


def validate_model(model: onnx.ModelProto) -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            got = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(example["input"])})[0]
            expected = _grid_to_onehot(example["output"])
            if not np.array_equal(got > 0.0, expected > 0.0):
                raise AssertionError(f"{split}[{idx}] failed")
    print("model correctness: OK on train, test, and arc-gen")


def main() -> None:
    validate_hypotheses()
    model = build_onnx_model()
    validate_model(model)
    onnx.save(model, BEST_PATH)
    result = score_file(BEST_PATH)
    print(
        f"saved {BEST_PATH} | memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']:.6f}"
    )


if __name__ == "__main__":
    main()
