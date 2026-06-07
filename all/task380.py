"""Minimal ONNX for ARC task380 top-left 3x3 counter-clockwise rotation.

Task rule: every example is a 3x3 grid in the top-left corner, with one
foreground color plus black. Rotate that full 3x3 color pattern 90 degrees
counter-clockwise and leave the rest of the 30x30 competition canvas empty.

ONNX approach: rotate the one-hot tensor directly so the color is preserved:
slice the top-left 3x3 core while reversing columns, transpose height and
width, then pad the compact 3x3 result back to the required 30x30 tensor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task380"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task380.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
OPSET = 10
IR_VERSION = 10


MaskTransform = Callable[[np.ndarray], np.ndarray]


def _mask(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64) != 0


def _candidate_transforms() -> dict[str, MaskTransform]:
    return {
        "identity": lambda x: x,
        "rot90_ccw": lambda x: np.rot90(x, 1),
        "rot180": lambda x: np.rot90(x, 2),
        "rot90_cw": lambda x: np.rot90(x, 3),
        "flip_vertical": np.flipud,
        "flip_horizontal": np.fliplr,
        "transpose": lambda x: x.T,
        "anti_transpose": lambda x: np.rot90(x.T, 2),
    }


def infer_rule() -> str:
    """Enumerate simple 3x3 mask transforms and choose the train-perfect rule."""
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    matches: list[str] = []
    for name, transform in _candidate_transforms().items():
        if all(
            np.array_equal(transform(_mask(example["input"])), _mask(example["output"]))
            for example in data["train"]
        ):
            matches.append(name)

    if "rot90_ccw" not in matches:
        raise AssertionError(f"expected rot90_ccw to match train examples, got {matches}")
    return "rot90_ccw"


def _i64(initializers: list[onnx.TensorProto], name: str, values: list[int]) -> str:
    initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name))
    return name


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    _i64(initializers, "s", [0, 2])
    _i64(initializers, "e", [3, -31])
    _i64(initializers, "a", [2, 3])
    _i64(initializers, "t", [1, -1])

    nodes.extend(
        [
            helper.make_node(
                "Slice",
                [IN_NAME, "s", "e", "a", "t"],
                ["x"],
            ),
            helper.make_node("Transpose", ["x"], ["y"], perm=[0, 1, 3, 2]),
            helper.make_node(
                "Pad",
                ["y"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, 27, 27],
                value=0.0,
            ),
        ]
    )

    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        initializer=initializers,
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


def validate_examples(path: Path) -> tuple[int, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            arr = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if arr is None or expected is None:
                continue
            actual = session.run([OUT_NAME], {IN_NAME: arr})[0]
            total += 1
            if not np.array_equal(actual > 0.0, expected > 0.0):
                raise AssertionError(f"{split}[{idx}] output mismatch")
            passed += 1
    return passed, total


def main() -> None:
    selected = infer_rule()
    if selected != "rot90_ccw":
        raise AssertionError(f"unsupported inferred rule: {selected}")

    model = build_model()
    onnx.save(model, BEST_PATH)

    passed, total = validate_examples(BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"rule:    {selected}")
    print(f"correct: {passed}/{total}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
