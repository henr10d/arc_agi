"""ONNX for ARC task360: fold the right half onto the left of a gray divider.

Task rule: each input is a 10x9 grid with a gray vertical separator in column 4.
Remove the separator and mirror the four columns to its right back onto the
four-column canvas to its left.  The output is the union of the original left
side and the mirrored right side; all observed overlaps have the same color, so
foreground one-hot channels can be added directly.  Black remains background.

ONNX approach: slice the left side and reverse-slice the right side, then use
the left black channel as a broadcast condition: black-left cells take the
mirrored right one-hot cell, and non-black-left cells keep the left one-hot
cell.  The compact 10x4 result is padded to the required NeuroGolf output
tensor.  Ties are harmless because the JSON has no conflicting nonzero overlaps.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task360"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
TASK_H = 10
HALF_W = 4
DIVIDER_COL = 4
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
IR_VERSION = 10
OPSET = 10


def load_examples() -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    examples: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            examples.append(
                (
                    split,
                    idx,
                    np.asarray(example["input"], dtype=np.int64),
                    np.asarray(example["output"], dtype=np.int64),
                )
            )
    return examples


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver: left side union mirrored right side."""
    g = np.asarray(grid, dtype=np.int64)
    left = g[:, :HALF_W]
    right_mirrored = np.flip(g[:, DIVIDER_COL + 1 : DIVIDER_COL + 1 + HALF_W], axis=1)
    out = np.zeros_like(left)
    for r in range(left.shape[0]):
        for c in range(left.shape[1]):
            lv = int(left[r, c])
            rv = int(right_mirrored[r, c])
            if lv not in (0, 5):
                out[r, c] = lv
            elif rv not in (0, 5):
                out[r, c] = rv
    return out


def one_hot(grid: np.ndarray) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            out[0, int(grid[r, c]), r, c] = 1.0
    return out


def decode(onehot: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return (onehot[0, :, :h, :w] > 0.0).argmax(axis=0).astype(np.int64)


def _init(inits: list[onnx.TensorProto], name: str, value: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(value, dtype=np.int64), name=name))
    return name


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _init(inits, "axes", [2, 3])
    reverse_steps = _init(inits, "reverse_steps", [1, -1])

    left_st = _init(inits, "left_st", [0, 0])
    folded_en = _init(inits, "folded_en", [TASK_H, HALF_W])
    right_st = _init(inits, "right_st", [0, DIVIDER_COL + HALF_W])
    bg_idx = _init(inits, "bg_idx", [0])
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, left_st, folded_en, axes], ["left"]),
            helper.make_node("Slice", [IN_NAME, right_st, folded_en, axes, reverse_steps], ["right"]),
            helper.make_node("Gather", ["left", bg_idx], ["left_bg"], axis=1),
            helper.make_node("Cast", ["left_bg"], ["take_right"], to=TensorProto.BOOL),
            helper.make_node("Where", ["take_right", "right", "left"], ["folded"]),
            helper.make_node(
                "Pad",
                ["folded"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - TASK_H, W - HALF_W],
                value=0.0,
            ),
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
    onnx.checker.check_model(model, full_check=True)
    return model


def run_model(model: onnx.ModelProto, grid: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run([OUT_NAME], {IN_NAME: one_hot(grid)})[0]


def validate_reference(examples: list[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    for split, idx, inp, expected in examples:
        got = solve(inp)
        if not np.array_equal(got, expected):
            raise AssertionError(f"reference failed {split}[{idx}]")


def validate_model(model: onnx.ModelProto, examples: list[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    for split, idx, inp, expected in examples:
        got = decode(run_model(model, inp), expected.shape)
        if not np.array_equal(got, expected):
            raise AssertionError(f"model failed {split}[{idx}]")


def main() -> None:
    examples = load_examples()
    validate_reference(examples)

    model = build_model()
    validate_model(model, examples)

    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
        onnx.save(model, tmp.name)
        metrics = score_file(Path(tmp.name))
    if not metrics["valid"]:
        raise AssertionError(f"scoring failed: {metrics['error']}")

    onnx.save(model, BEST_PATH)
    print(f"saved {BEST_PATH}")
    print(
        f"memory={metrics['memory']} params={metrics['params']} "
        f"cost={metrics['cost']} score={metrics['score']:.6f}"
    )


if __name__ == "__main__":
    main()
