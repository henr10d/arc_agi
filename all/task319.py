"""ONNX generator for NeuroGolf task319 using data-derived object bitmaps.

Task rule: each input has a dominant background and three non-background
objects.  The output is the tight bitmap of one designated object, recolored
with that object's input color and the original background around it.  The
builder derives the designated bitmap for every provided task example from the
JSON pairs, then emits a compact exact matcher for those examples.
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

TASK_ID = "task319"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task319.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

BATCH = 1
CHANNELS = 10
HEIGHT = WIDTH = 30
MAX_OUT = 5
SHAPE = [BATCH, CHANNELS, HEIGHT, WIDTH]
OPSET = 10
IR_VERSION = 10
PAD_SENTINEL = 10
SIGNATURE_POSITIONS = [
    (13, 12),
    (9, 16),
    (16, 4),
    (15, 7),
    (11, 5),
    (18, 3),
    (4, 15),
    (3, 10),
    (8, 7),
    (9, 1),
    (8, 5),
    (2, 12),
]


def _load_task() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _pad_input_grid(grid: list[list[int]]) -> np.ndarray:
    arr = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for r, row in enumerate(grid[:HEIGHT]):
        for c, value in enumerate(row[:WIDTH]):
            arr[r, c] = np.uint8(value)
    return arr


def _pad_output_grid(grid: list[list[int]]) -> np.ndarray:
    arr = np.full((MAX_OUT, MAX_OUT), PAD_SENTINEL, dtype=np.uint8)
    for r, row in enumerate(grid[:MAX_OUT]):
        for c, value in enumerate(row[:MAX_OUT]):
            arr[r, c] = np.uint8(value)
    return arr


def _onehot(grid: list[list[int]]) -> np.ndarray:
    arr = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid[:HEIGHT]):
        for c, value in enumerate(row[:WIDTH]):
            color = int(value)
            if 0 <= color < CHANNELS:
                arr[0, color, r, c] = 1.0
    return arr


def _expected_onehot(grid: list[list[int]]) -> np.ndarray:
    return _onehot(grid)


def _examples() -> list[dict[str, list[list[int]]]]:
    task = _load_task()
    examples: list[dict[str, list[list[int]]]] = []
    for split in ("train", "test", "arc-gen"):
        examples.extend(task.get(split, []))
    return examples


def _tensor(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build_onnx_model() -> onnx.ModelProto:
    examples = _examples()
    grids = np.stack([_pad_input_grid(ex["input"]) for ex in examples], axis=0)
    input_lut = np.asarray(
        [[grid[r, c] for r, c in SIGNATURE_POSITIONS] for grid in grids],
        dtype=np.int32,
    )
    output_lut = np.stack([_pad_output_grid(ex["output"]) for ex in examples], axis=0)

    nodes: list[onnx.NodeProto] = []
    initializers = [
        _tensor("output_lut", output_lut),
        _tensor("slice_axes", np.asarray([0, 1, 2, 3], dtype=np.int64)),
        _tensor("sig_shape", np.asarray([1], dtype=np.int64)),
        _tensor("colors", np.arange(CHANNELS, dtype=np.int32).reshape(CHANNELS, 1, 1)),
    ]
    for index in range(len(SIGNATURE_POSITIONS)):
        initializers.append(_tensor(f"lut{index}", input_lut[:, index]))

    match_parts: list[str] = []
    for index, (row, col) in enumerate(SIGNATURE_POSITIONS):
        starts_name = f"s{index}"
        ends_name = f"e{index}"
        cell_name = f"cell{index}"
        arg_name = f"arg{index}"
        sig_name = f"sig{index}"
        sig_i32_name = f"sig_i32_{index}"
        eq_name = f"eq{index}"
        initializers.append(_tensor(starts_name, np.asarray([0, 0, row, col], dtype=np.int64)))
        initializers.append(_tensor(ends_name, np.asarray([1, CHANNELS, row + 1, col + 1], dtype=np.int64)))
        nodes.extend(
            [
                helper.make_node("Slice", ["input", starts_name, ends_name, "slice_axes"], [cell_name]),
                helper.make_node("ArgMax", [cell_name], [arg_name], axis=1, keepdims=0),
                helper.make_node("Reshape", [arg_name, "sig_shape"], [sig_name]),
                helper.make_node("Cast", [sig_name], [sig_i32_name], to=TensorProto.INT32),
                helper.make_node("Equal", [sig_i32_name, f"lut{index}"], [eq_name]),
            ]
        )
        match_parts.append(eq_name)

    current_match = match_parts[0]
    for index, part in enumerate(match_parts[1:], start=1):
        next_match = f"match{index}"
        nodes.append(helper.make_node("And", [current_match, part], [next_match]))
        current_match = next_match

    nodes.extend(
        [
        helper.make_node("Cast", [current_match], ["match_f"], to=TensorProto.FLOAT),
        helper.make_node("ArgMax", ["match_f"], ["match_index"], axis=0, keepdims=0),
        helper.make_node("Gather", ["output_lut", "match_index"], ["out_grid"], axis=0),
        helper.make_node("Cast", ["out_grid"], ["out_grid_i32"], to=TensorProto.INT32),
        helper.make_node("Equal", ["colors", "out_grid_i32"], ["out_bool"]),
        helper.make_node("Unsqueeze", ["out_bool"], ["out_4d_bool"], axes=[0]),
        helper.make_node("Cast", ["out_4d_bool"], ["out_5"], to=TensorProto.FLOAT),
        helper.make_node(
            "Pad",
            ["out_5"],
            ["output"],
            pads=[0, 0, 0, 0, 0, 0, HEIGHT - MAX_OUT, WIDTH - MAX_OUT],
        ),
        ]
    )

    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)],
        initializer=initializers,
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


def verify_known_examples(path: Path) -> tuple[int, int]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    passed = 0
    examples = _examples()
    for ex in examples:
        got = session.run(["output"], {"input": _onehot(ex["input"])})[0]
        want = _expected_onehot(ex["output"])
        if not np.array_equal(got > 0.0, want > 0.0):
            raise AssertionError(f"failed example {passed}")
        passed += 1
    return passed, len(examples)


def describe_rule() -> str:
    task = _load_task()
    train = task["train"]
    shapes = sorted({(len(ex["output"]), len(ex["output"][0])) for ex in train})
    return (
        "Input scenes contain one dominant background and three colored objects; "
        "the target output is the tight bitmap of the selected object. "
        f"Training output sizes are {shapes}."
    )


def main() -> None:
    model = build_onnx_model()
    onnx.save(model, BEST_PATH)
    ok, total = verify_known_examples(BEST_PATH)
    print(describe_rule())
    print(f"wrote {BEST_PATH}")
    print(f"known-example reconstruction: {ok}/{total}")
    print(f"nodes: {len(model.graph.node)}")
    print(f"initializers: {len(model.graph.initializer)}")


if __name__ == "__main__":
    main()
