"""Minimal ONNX for NeuroGolf task135.

Task rule: the 9x9 input is partitioned into nine 3x3 blocks, and the output
is always the top-right block: rows 0..2 and columns 6..8. The selected block is
placed at the top-left of the competition's 30x30 one-hot output tensor; the
remaining cells stay zero-padded.

ONNX approach: one opset-10 Slice extracts input[:, :, 0:3, 6:9], then Pad
writes that compact crop into the required [1, 10, 30, 30] output. This realizes
only one scored internal tensor, the 3x3x10 float crop; Pad's attribute pads are
not counted as params by the NeuroGolf scorer.
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


TASK_NUM = "135"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
OPSET = 10


def init_i32(name: str, values: list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.int32), name)


def build_model() -> onnx.ModelProto:
    initializers = [
        init_i32("starts", [0, 6]),
        init_i32("ends", [3, 9]),
        init_i32("axes", [2, 3]),
    ]
    nodes = [
        helper.make_node("Slice", [IN_NAME, "starts", "ends", "axes"], ["crop"]),
        helper.make_node(
            "Pad",
            ["crop"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 27, 27],
            value=0.0,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, FULL_SHAPE)],
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


def write_model(model: onnx.ModelProto, path: Path) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and score task135 ONNX.")
    parser.add_argument("--check-only", action="store_true", help="verify and score without writing the model")
    args = parser.parse_args()

    model = build_model()
    ok, splits = verify_correct(model)
    if not ok:
        raise SystemExit(f"verification failed: {splits}")

    path = BEST_PATH
    if args.check_only:
        path = OUT_DIR / f".{TASK_ID}.check.onnx"

    write_model(model, path)
    result = score_file(path)
    print(
        f"correct={ok} splits={splits} valid={result['valid']} "
        f"memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']:.6f}"
    )
    if not args.check_only:
        print(f"wrote: {path}")
    elif path.exists():
        path.unlink()


if __name__ == "__main__":
    main()
