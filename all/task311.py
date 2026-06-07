"""Build ONNX for task311: append each row's horizontal mirror.

Task rule: keep the input height and double the input width. Each output row is
the original row followed by that same row reversed left-to-right. The ONNX
graph applies the rule to the one-hot tensor without color-specific logic:
slice the active HxW region, slice it again with a negative column step, concat
the two compact tensors along the column axis, then pad to the required
30x30 competition output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy  # noqa: E402

TASK_ID = "task311"
OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
C = 10
N = 30
SHAPE = [1, C, N, N]
OPSET = 10
IR_VERSION = 10


def solve(grid: list[list[int]]) -> list[list[int]]:
    """Reference grid transform."""
    return [row + list(reversed(row)) for row in grid]


def _i64(inits: list[onnx.TensorProto], name: str, vals: list[int]) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))
    return name


def _load_task() -> dict[str, Any]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _examples(data: dict[str, Any]) -> list[dict[str, list[list[int]]]]:
    return [
        example
        for split in ("train", "test", "arc-gen")
        for example in data.get(split, [])
    ]


def _task_dims(data: dict[str, Any]) -> tuple[int, int]:
    dims = {
        (len(example["input"]), len(example["input"][0]))
        for example in _examples(data)
        if example.get("input")
    }
    if len(dims) != 1:
        raise ValueError(f"{TASK_ID} needs one static HxW for ONNX shape inference, got {sorted(dims)}")
    h, w = next(iter(dims))
    if h <= 0 or w <= 0 or h > N or 2 * w > N:
        raise ValueError(f"invalid {TASK_ID} dimensions: {h}x{w} -> {h}x{2 * w}")
    return h, w


def _validate_rule(data: dict[str, Any]) -> None:
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(data.get(split, [])):
            expected = solve(example["input"])
            if example["output"] != expected:
                raise ValueError(f"{split}[{index}] does not match mirror rule")


def build_onnx_model(h: int, w: int) -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes_hw = _i64(inits, "axes_hw", [2, 3])
    crop_st = _i64(inits, "crop_st", [0, 0])
    crop_en = _i64(inits, "crop_en", [h, w])
    rev_st = _i64(inits, "rev_st", [0, w - 1])
    rev_en = _i64(inits, "rev_en", [h, -(N + 1)])
    rev_steps = _i64(inits, "rev_steps", [1, -1])

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, crop_st, crop_en, axes_hw], ["crop"]),
            helper.make_node("Slice", [IN_NAME, rev_st, rev_en, axes_hw, rev_steps], ["rev"]),
            helper.make_node("Concat", ["crop", "rev"], ["mirrored"], axis=3),
            helper.make_node(
                "Pad",
                ["mirrored"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, N - h, N - 2 * w],
            ),
        ]
    )

    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task311",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _verify_examples(model: onnx.ModelProto, data: dict[str, Any]) -> None:
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    )
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(data.get(split, [])):
            input_tensor = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if input_tensor is None or expected is None:
                continue
            actual = session.run([OUT_NAME], {IN_NAME: input_tensor})[0]
            if not np.array_equal(actual > 0.0, expected > 0.0):
                raise AssertionError(f"{split}[{index}] failed")


def main() -> None:
    data = _load_task()
    _validate_rule(data)
    h, w = _task_dims(data)
    model = build_onnx_model(h, w)
    _verify_examples(model, data)
    onnx.save(model, BEST_PATH)
    print(f"wrote {BEST_PATH} for {h}x{w} -> {h}x{2 * w}")


if __name__ == "__main__":
    main()
