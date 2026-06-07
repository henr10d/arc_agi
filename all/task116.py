"""One-node ONNX generator for NeuroGolf task116.

Task rule: every input is a 3x4 ARC grid. The 6x4 output stacks a vertical
mirror of the input over the original input, so output rows are:
input row2, input row1, input row0, input row0, input row1, input row2.

ONNX approach: gather the full 30-row competition tensor directly into the
graph output. Rows 0..5 select input rows [2, 1, 0, 0, 1, 2], and rows 6..29
select padded input row 3, which is all zero. Input columns 4..29 are already
zero padding, so one Gather produces the required full float output. Because
the Gather writes to graph output, activation memory is 0; only the 30 row
indices are counted as parameters.
"""

from __future__ import annotations

import argparse
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


TASK_NUM = "116"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_PATH = Path(__file__).resolve().parent / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
OPSET = 10


def build_model() -> onnx.ModelProto:
    row_indices = numpy_helper.from_array(
        np.asarray([2, 1, 0, 0, 1, 2] + [3] * 24, dtype=np.int64),
        "row_indices",
    )
    node = helper.make_node("Gather", [IN_NAME, "row_indices"], [OUT_NAME], axis=2)
    graph = helper.make_graph(
        [node],
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [row_indices],
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


def verify_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    with TASK_PATH.open(encoding="utf-8") as fh:
        task_data = json.load(fh)

    all_ok = True
    splits: dict[str, tuple[int, int]] = {}
    for split, examples in task_data.items():
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the task116 ONNX model.")
    parser.add_argument("--check-only", action="store_true", help="verify and score without writing")
    args = parser.parse_args()

    model = build_model()
    ok, splits = verify_correct(model)
    if not ok:
        raise SystemExit(f"verification failed: {splits}")

    if not args.check_only:
        onnx.save(model, str(OUT_PATH))

    score_path = OUT_PATH
    if args.check_only:
        import tempfile

        with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}_") as tmpdir:
            score_path = Path(tmpdir) / f"{TASK_ID}.onnx"
            onnx.save(model, str(score_path))
            result = score_file(score_path)
    else:
        result = score_file(score_path)

    if not result["valid"]:
        raise SystemExit(f"score_model invalid: {result['error']}")

    print(
        "score_model: "
        f"memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']:.6f}"
    )
    print(f"splits: {splits}")
    if not args.check_only:
        print(f"wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
