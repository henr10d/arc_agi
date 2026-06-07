"""Minimal ONNX for NeuroGolf task155.

Task rule: the visible input grid is an NxN square in the top-left of the
competition one-hot tensor, with N between 4 and 8 in the local data. The output
flips that visible square vertically: output row r copies input row N - 1 - r,
and padded rows/columns stay all-zero.

ONNX approach: count all active one-hot cells to get N^2, recover N by testing
that scalar count against the thresholds between 4^2, 5^2, ..., 8^2, then build
the 30 row indices as N - [1, 2, ..., 30]. Negative Gather indices wrap to the
bottom padded rows, which are all zero, so no explicit outside-square mask is
needed. The final Gather writes directly to ``output`` and is not charged as
internal memory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402


TASK_ID = "task155"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
OPSET = 10


def _init_i32(name: str, values: list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.int32), name)


def _init_f32(name: str, values: list[float]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.float32), name)


def build_model() -> onnx.ModelProto:
    """Build the selected total-count vertical flip graph."""
    initializers = [
        _init_i32("row_offsets", [i - 4 for i in range(1, 31)]),
        _init_f32("thresholds", [20.0, 30.0, 42.0, 56.0]),
    ]
    nodes = [
        helper.make_node("ReduceSum", [IN_NAME], ["cell_count"], axes=[0, 1, 2, 3], keepdims=0),
        helper.make_node("Greater", ["cell_count", "thresholds"], ["above_threshold"]),
        helper.make_node("Cast", ["above_threshold"], ["above_i"], to=TensorProto.INT32),
        helper.make_node("ReduceSum", ["above_i"], ["extra_rows"], axes=[0], keepdims=1),
        helper.make_node("Sub", ["extra_rows", "row_offsets"], ["src_rows"]),
        helper.make_node("Gather", [IN_NAME, "src_rows"], [OUT_NAME], axis=2),
    ]
    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def verify_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    all_ok = True
    splits: dict[str, tuple[int, int]] = {}
    for split, examples in load_task_data().items():
        passed = 0
        checked = 0
        for example in examples:
            inp = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if inp is None or expected is None:
                continue
            pred = session.run([OUT_NAME], {IN_NAME: inp})[0]
            checked += 1
            if np.array_equal(pred > 0.0, expected > 0.0):
                passed += 1
            else:
                all_ok = False
        splits[split] = (passed, checked)
    return all_ok, splits


def verify_random_squares(model: onnx.ModelProto, trials: int = 100) -> None:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(155)
    for n in range(4, 9):
        for _ in range(trials):
            colors = rng.integers(0, 10, size=(n, n))
            x = np.zeros(SHAPE, dtype=np.float32)
            for r in range(n):
                for c in range(n):
                    x[0, colors[r, c], r, c] = 1.0
            y = session.run([OUT_NAME], {IN_NAME: x})[0]
            expected = np.zeros_like(x)
            for r in range(n):
                expected[:, :, r, :n] = x[:, :, n - 1 - r, :n]
            if not np.array_equal(y > 0.0, expected > 0.0):
                raise AssertionError(f"random {n}x{n} vertical flip validation failed")


def main() -> None:
    model = build_model()
    ok, splits = verify_correct(model)
    if not ok:
        raise SystemExit(f"{TASK_ID} failed correctness: {splits}")
    verify_random_squares(model)
    onnx.save(model, BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"correctness: {splits}")
    print(result)


if __name__ == "__main__":
    main()
