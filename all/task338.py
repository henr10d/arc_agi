"""ONNX for task338: turn enclosed red-rectangle holes green.

Task rule: inputs contain red (2) rectangular outlines and sometimes filled red
rectangles on a black background. The output preserves the input grid size,
changes every black cell enclosed by a red outline to green (3), and makes all
other in-grid cells black. Filled red rectangles do not create green output.

The graph works on the observed 25x25 maximum task area, flood-fills black cells
from the dynamic input-grid boundary, and treats unreached black cells as the
holes. The examples only use colors 0 and 2, so the valid in-grid mask is
black-or-red rather than a 10-channel reduction.
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

TASK_ID = "task338"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task338.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
CORE = 25
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10
FLOOD_STEPS = 12


def solve(grid: list[list[int]]) -> list[list[int]]:
    """Reference implementation using the same 8-neighbor exterior fill."""
    arr = np.asarray(grid, dtype=np.int64)
    h, w = arr.shape
    black = arr == 0
    exterior = np.zeros_like(black, dtype=bool)
    exterior[0, :] |= black[0, :]
    exterior[h - 1, :] |= black[h - 1, :]
    exterior[:, 0] |= black[:, 0]
    exterior[:, w - 1] |= black[:, w - 1]

    for _ in range(max(h, w)):
        padded = np.pad(exterior, 1)
        near = np.zeros_like(exterior)
        for dr in range(3):
            for dc in range(3):
                near |= padded[dr : dr + h, dc : dc + w]
        nxt = black & near
        if np.array_equal(nxt, exterior):
            break
        exterior = nxt

    out = np.zeros_like(arr)
    out[black & ~exterior] = 3
    return out.tolist()


def _i64(inits: list[onnx.TensorProto], vals: list[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))
    return name


def _f16(inits: list[onnx.TensorProto], vals: list[float], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.float16), name=name))
    return name


def _f32_arr(inits: list[onnx.TensorProto], arr: np.ndarray, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name))
    return name


def _f16_arr(inits: list[onnx.TensorProto], arr: np.ndarray, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float16), name=name))
    return name


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def _expected_onehot(grid: list[list[int]]) -> np.ndarray:
    return _grid_to_onehot(grid)


def _load_examples() -> list[dict[str, Any]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [
        example
        for split in ("train", "test", "arc-gen")
        for example in data.get(split, [])
        if max(len(example["input"]), len(example["input"][0])) <= 30
    ]


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, [0, 1, 2, 3], "axes")
    start_black = _i64(inits, [0, 0, 0, 0], "start_black")
    end_black = _i64(inits, [1, 1, CORE, CORE], "end_black")
    start_red = _i64(inits, [0, 2, 0, 0], "start_red")
    end_red = _i64(inits, [1, 3, CORE, CORE], "end_red")
    one = _f16(inits, [1.0], "one")
    boundary = np.zeros((1, 1, CORE, CORE), dtype=np.float16)
    boundary[:, :, 0, :] = 1.0
    boundary[:, :, -1, :] = 1.0
    boundary[:, :, :, 0] = 1.0
    boundary[:, :, :, -1] = 1.0
    _f16_arr(inits, boundary, "boundary")
    _f32_arr(inits, np.zeros((1, 1, H, W), dtype=np.float32), "zero30")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, start_black, end_black, axes], ["black_f"]),
            helper.make_node("Slice", [IN_NAME, start_red, end_red, axes], ["red_f"]),
            helper.make_node("Cast", ["black_f"], ["black"], to=TensorProto.FLOAT16),
            helper.make_node("Cast", ["red_f"], ["red"], to=TensorProto.FLOAT16),
            helper.make_node("Add", ["black", "red"], ["valid"]),
            helper.make_node("Sub", [one, "valid"], ["invalid"]),
            helper.make_node(
                "MaxPool",
                ["invalid"],
                ["invalid_near"],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                strides=[1, 1],
            ),
            helper.make_node("Max", ["invalid_near", "boundary"], ["seed_mask"]),
            helper.make_node("Mul", ["black", "seed_mask"], ["ext0"]),
        ]
    )

    prev = "ext0"
    for idx in range(FLOOD_STEPS):
        pooled = f"ext_pool{idx}"
        filled = f"ext{idx + 1}"
        nodes.append(
            helper.make_node(
                "MaxPool",
                [prev],
                [pooled],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                strides=[1, 1],
            )
        )
        nodes.append(helper.make_node("Mul", ["black", pooled], [filled]))
        prev = filled

    nodes.extend(
        [
            helper.make_node("Sub", ["black", prev], ["green_h"]),
            helper.make_node("Sub", ["valid", "green_h"], ["bg_h"]),
            helper.make_node(
                "Pad",
                ["green_h"],
                ["green30_h"],
                pads=[0, 0, 0, 0, 0, 0, H - CORE, W - CORE],
            ),
            helper.make_node(
                "Pad",
                ["bg_h"],
                ["bg30_h"],
                pads=[0, 0, 0, 0, 0, 0, H - CORE, W - CORE],
            ),
            helper.make_node("Cast", ["green30_h"], ["green30"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["bg30_h"], ["bg30"], to=TensorProto.FLOAT),
            helper.make_node(
                "Concat",
                ["bg30", "zero30", "zero30", "green30", "zero30", "zero30", "zero30", "zero30", "zero30", "zero30"],
                [OUT_NAME],
                axis=1,
            ),
        ]
    )

    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task338",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def validate_model(model: onnx.ModelProto) -> None:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    for index, example in enumerate(_load_examples()):
        predicted = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(example["input"])})[0]
        expected_grid = solve(example["input"])
        if expected_grid != example["output"]:
            raise AssertionError(f"reference rule mismatch on example {index}")
        expected = _expected_onehot(example["output"])
        if not np.array_equal(predicted > 0.0, expected > 0.0):
            raise AssertionError(f"ONNX mismatch on example {index}")


def main() -> None:
    model = build_model()
    validate_model(model)
    onnx.save(model, BEST_PATH)
    print(f"wrote {BEST_PATH}")


if __name__ == "__main__":
    main()
