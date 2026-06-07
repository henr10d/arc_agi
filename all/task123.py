"""ONNX solver for NeuroGolf task123 using Kaggle one-hot I/O.

Task rule: the input is a diagonal/max-index square pattern, either 5x5 or a
4x4 pattern padded with zero color in the last row and column. The first row is
the color cycle. The output extends the pattern to 10x10 with period p in
{4, 5}: output[r, c] = input[0, max(r, c) % p], then the result is padded as
all-zero outside the ARC output region for the fixed 30x30 NeuroGolf tensor
contract.

ONNX approach: slice the first row of the 5x5 visible region, detect whether
the fifth first-row entry is zero padding, select the corresponding 10x10 gather index
matrix, gather the compact 10x10 result, and pad directly to the required
30x30 output.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "task123"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task123.onnx"
DATA_PATH = ROOT / "data" / "task123.json"

C = 10
H = W = 30
CORE = 5
OUT = 10
PAD = H - OUT
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def _i64(values: list[int] | np.ndarray, name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)


def _i32(values: list[int] | np.ndarray, name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.int32), name=name)


def _f32(values: list[float] | np.ndarray, name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.float32), name=name)


def solve_grid(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    period = 4 if arr[4, 4] == 0 else 5
    return np.asarray([[arr[0, max(r, c) % period] for c in range(OUT)] for r in range(OUT)], dtype=np.int64)


def to_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def build_model() -> onnx.ModelProto:
    idx4 = np.fromfunction(lambda r, c: np.maximum(r, c) % 4, (OUT, OUT), dtype=int)
    idx5 = np.fromfunction(lambda r, c: np.maximum(r, c) % 5, (OUT, OUT), dtype=int)
    inits = [
        _i64([0, 0, 0, 0], "s0"),
        _i64([1, C, 1, CORE], "erow"),
        _i64([0, 0, 4], "s04"),
        _i64([1, 1, 5], "e04"),
        _f32([0.0], "zero"),
        _i32(idx4, "idx4"),
        _i32(idx5, "idx5"),
    ]

    nodes = [
        helper.make_node("Slice", [IN_NAME, "s0", "erow"], ["row0_4d"]),
        helper.make_node("Squeeze", ["row0_4d"], ["row0"], axes=[2]),
        helper.make_node("Slice", ["row0", "s04", "e04"], ["c04"]),
        helper.make_node("Squeeze", ["c04"], ["c04s"], axes=[0, 1, 2]),
        helper.make_node("Greater", ["c04s", "zero"], ["is_pad4"]),
        helper.make_node("Where", ["is_pad4", "idx4", "idx5"], ["idx"]),
        helper.make_node("Gather", ["row0", "idx"], ["tile10"], axis=2),
        helper.make_node("Pad", ["tile10"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]),
    ]

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


def validate_outputs(path: Path) -> None:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    checked = 0
    for split in ("train", "test", "arc-gen"):
        for example in data.get(split, []):
            expected = to_onehot(example["output"])
            reference = to_onehot(solve_grid(example["input"]))
            if not np.array_equal(reference, expected):
                raise AssertionError(f"reference rule mismatch in {split} example {checked}")
            actual = session.run([OUT_NAME], {IN_NAME: to_onehot(example["input"])})[0]
            if not np.array_equal(actual > 0.0, expected > 0.0):
                raise AssertionError(f"model output mismatch in {split} example {checked}")
            checked += 1


def print_score(path: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from score_model import score_file

    result = score_file(path)
    if not result["valid"]:
        raise RuntimeError(result["error"])

    print(f"saved: {path}")
    print(f"params count: {result['params']}")
    print(f"internal tensor memory: {result['memory']}")
    print(f"total cost: {result['cost']}")
    print(f"NeuroGolf score: {float(result['score']):.6f}")


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    validate_outputs(BEST_PATH)
    print_score(BEST_PATH)


if __name__ == "__main__":
    main()
