"""ONNX solver for ARC task323: draw a gray staircase through the cyan cell.

The grid is always 13x13 and contains exactly one cyan cell. The output keeps
that cyan source and draws the repeating orthogonal staircase passing through
it: rows alternate between one-cell vertical connectors and three-cell
horizontal runs, with each horizontal run shifted two columns as it moves away
from cyan. Every staircase cell is gray except the cyan source; all other
in-grid cells are black.

ONNX: crop the 13x13 cyan channel, convolve it with a fixed 25x25 relative
staircase stencil to get the translated path mask, build compact float one-hot
channels, and pad only at the final output.
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
TASK_ID = "task323"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task323.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

sys.path.insert(0, str(ROOT))

IN_NAME = "input"
OUT_NAME = "output"
C = 10
H = W = 30
N = 13
SHAPE = [1, C, H, W]
IR_VERSION = 10
OPSET = 10


def _init(inits: list[onnx.TensorProto], name: str, arr: np.ndarray) -> str:
    inits.append(numpy_helper.from_array(arr, name=name))
    return name


def _i64(inits: list[onnx.TensorProto], name: str, vals: list[int]) -> str:
    return _init(inits, name, np.asarray(vals, dtype=np.int64))


def _f32(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    return _init(inits, name, np.asarray(arr, dtype=np.float32))


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference implementation for the translated length-3 staircase."""
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(arr)
    cr, cc = np.argwhere(arr == 8)[0]
    out[cr, cc] = 8

    for dr in range(-N + 1, N):
        if dr == 0:
            continue
        if dr < 0:
            if dr % 2 == 0:
                dcs = range(-dr - 2, -dr + 1)
            else:
                dcs = (-dr - 1,)
        else:
            if dr % 2 == 0:
                dcs = range(-dr, -dr + 3)
            else:
                dcs = (-dr + 1,)
        for dc in dcs:
            r, c = cr + dr, cc + dc
            if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
                out[r, c] = 5

    out[cr, cc] = 8
    return out


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, "a4", [0, 1, 2, 3])
    cyan_st = _i64(inits, "sc", [0, 8, 0, 0])
    cyan_en = _i64(inits, "ec", [1, 9, N, N])
    one_grid = _f32(inits, "one_grid", np.ones((1, 1, N, N), dtype=np.float32))
    blank = _f32(inits, "blank", np.zeros((1, 1, N, N), dtype=np.float32))

    kernel = np.zeros((1, 1, 2 * N - 1, 2 * N - 1), dtype=np.float32)
    for dr in range(-N + 1, N):
        if dr == 0:
            kernel[0, 0, N - 1, N - 1] = 1.0
            continue
        if dr < 0:
            if dr % 2 == 0:
                dcs = range(-dr - 2, -dr + 1)
            else:
                dcs = (-dr - 1,)
        else:
            if dr % 2 == 0:
                dcs = range(-dr, -dr + 3)
            else:
                dcs = (-dr + 1,)
        for dc in dcs:
            if -N < dc < N:
                # Conv performs cross-correlation; negate offsets so a source
                # cyan at (r, c) activates output cell (r + dr, c + dc).
                kernel[0, 0, N - 1 - dr, N - 1 - dc] = 1.0
    _f32(inits, "kernel", kernel)

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, cyan_st, cyan_en, axes4], ["cy"]),
            helper.make_node("Conv", ["cy", "kernel"], ["path"], pads=[N - 1, N - 1, N - 1, N - 1]),
            helper.make_node("Sub", ["one_grid", "path"], ["black"]),
            helper.make_node("Sub", ["path", "cy"], ["gray"]),
            helper.make_node(
                "Concat",
                [
                    "black",
                    "blank",
                    "blank",
                    "blank",
                    "blank",
                    "gray",
                    "blank",
                    "blank",
                    "cy",
                    "blank",
                ],
                ["out13"],
                axis=1,
            ),
            helper.make_node(
                "Pad",
                ["out13"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - N, W - N],
            ),
        ]
    )

    graph = helper.make_graph(nodes, "task323", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _onehot(grid: list[list[int]]) -> np.ndarray:
    arr = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            arr[0, int(color), r, c] = 1.0
    return arr


def verify_reference() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        task = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(task[split]):
            pred = solve(example["input"])
            expected = np.asarray(example["output"], dtype=np.int64)
            if not np.array_equal(pred, expected):
                raise AssertionError(f"reference failed {split}[{idx}]")


def verify_model(path: Path) -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        task = json.load(fh)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(task[split]):
            pred = session.run([OUT_NAME], {IN_NAME: _onehot(example["input"])})[0]
            expected = _onehot(example["output"])
            if not np.array_equal(pred > 0.0, expected > 0.0):
                raise AssertionError(f"model failed {split}[{idx}]")
            total += 1
    print(f"verified {total} examples")


def main() -> None:
    verify_reference()
    model = build_model()
    onnx.save(model, BEST_PATH)
    verify_model(BEST_PATH)
    print(BEST_PATH)


if __name__ == "__main__":
    main()
