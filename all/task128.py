"""ONNX for ARC task128: lift each bottom-touching colored rectangle upward.

Task rule: in the 15x15 active grid, every non-background rectangle of colors
1, 2, or 4 touches the bottom border.  Each column of a rectangle is moved up
by exactly the rectangle height, preserving color and column; the vacated cells
become background.  Generated examples only use heights 1 through 7.

The graph detects the exact bottom-run height once from foreground occupancy
(the color is irrelevant to height), combines those compact 1x15 row masks with
the bottom-row color, casts once, and pads to the required 30x30 NeuroGolf
output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TASK_ID = "task128"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
ROOT_PATH = ROOT / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
C = 10
H = W = 30
ACTIVE = 15
COLORS = (1, 2, 4)
MAX_HEIGHT = 7
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _i64(inits: list[onnx.TensorProto], vals: Iterable[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(list(vals), dtype=np.int64), name=name))
    return name


def _bool_init(inits: list[onnx.TensorProto], arr: np.ndarray, name: str) -> str:
    inits.append(numpy_helper.from_array(arr.astype(np.bool_), name=name))
    return name


def _f32_init(inits: list[onnx.TensorProto], val: float, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(val, dtype=np.float32), name=name))
    return name


def _node(nodes: list[onnx.NodeProto], op: str, inputs: list[str], output: str, **attrs) -> str:
    nodes.append(helper.make_node(op, inputs, [output], **attrs))
    return output


def solve_grid(grid: list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(arr)
    h, w = arr.shape
    for col in range(w):
        color = int(arr[h - 1, col])
        if color == 0:
            continue
        run = 0
        for row in range(h - 1, -1, -1):
            if int(arr[row, col]) != color:
                break
            run += 1
        out[h - 2 * run : h - run, col] = color
    return out


def _onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros((1, C, H, W), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def _grid_from_onehot(arr: np.ndarray) -> np.ndarray:
    return arr[0, :, :ACTIVE, :ACTIVE].argmax(axis=0).astype(np.int64)


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes3 = _i64(inits, [1, 2, 3], "axes3")
    half_f = _f32_init(inits, 0.5, "half_f")

    rows: dict[int, str] = {}
    for row in range(ACTIVE - MAX_HEIGHT - 1, ACTIVE):
        bg_f = _node(
            nodes,
            "Slice",
            [
                IN_NAME,
                _i64(inits, [0, row, 0], f"bg_rs{row}"),
                _i64(inits, [1, row + 1, ACTIVE], f"bg_re{row}"),
                axes3,
            ],
            f"bg_rf{row}",
        )
        rows[row] = _node(nodes, "Less", [bg_f, half_f], f"fg_r{row}")

    ge: dict[int, str] = {1: rows[14]}
    for height in range(2, MAX_HEIGHT + 1):
        ge[height] = _node(nodes, "And", [ge[height - 1], rows[ACTIVE - height]], f"ge{height}")

    zero_row = _bool_init(inits, np.zeros((1, 1, 1, ACTIVE), dtype=np.bool_), "zero_row")
    bg_row = np.zeros((1, C, 1, ACTIVE), dtype=np.bool_)
    bg_row[:, 0, :, :] = True
    bg_row_name = _bool_init(inits, bg_row, "bg_row")

    not_row7 = _node(nodes, "Not", [rows[7]], "not_row7")
    not_ge = {height: _node(nodes, "Not", [ge[height]], f"nge{height}") for height in range(2, MAX_HEIGHT + 1)}
    h7 = _node(nodes, "And", [ge[7], not_row7], "h7")
    h67 = _node(nodes, "And", [ge[6], not_row7], "h67")
    h567 = _node(nodes, "And", [ge[5], not_row7], "h567")
    h4567 = _node(nodes, "And", [ge[4], not_row7], "h4567")
    h456 = _node(nodes, "And", [ge[4], not_ge[7]], "h456")
    h345 = _node(nodes, "And", [ge[3], not_ge[6]], "h345")
    h34 = _node(nodes, "And", [ge[3], not_ge[5]], "h34")
    h23 = _node(nodes, "And", [ge[2], not_ge[4]], "h23")
    h2 = _node(nodes, "And", [ge[2], not_ge[3]], "h2")
    h1 = _node(nodes, "And", [ge[1], not_ge[2]], "h1")
    bottom: dict[int, str] = {}
    for color in COLORS:
        color_f = _node(
            nodes,
            "Slice",
            [
                IN_NAME,
                _i64(inits, [color, ACTIVE - 1, 0], f"c{color}_bs"),
                _i64(inits, [color + 1, ACTIVE, ACTIVE], f"c{color}_be"),
                axes3,
            ],
            f"c{color}_bf",
        )
        bottom[color] = _node(nodes, "Cast", [color_f], f"c{color}_b", to=TensorProto.BOOL)

    out_masks = [
        zero_row,
        h7,
        h7,
        h67,
        h67,
        h567,
        h567,
        h4567,
        h456,
        h345,
        h34,
        h23,
        h2,
        h1,
        zero_row,
    ]
    row_outputs: list[str] = []
    row_cache: dict[str, str] = {zero_row: bg_row_name}
    for row, mask in enumerate(out_masks):
        if mask not in row_cache:
            bg = _node(nodes, "Not", [mask], f"out{row}_bg")
            c1 = _node(nodes, "And", [bottom[1], mask], f"out{row}_c1")
            c2 = _node(nodes, "And", [bottom[2], mask], f"out{row}_c2")
            c4 = _node(nodes, "And", [bottom[4], mask], f"out{row}_c4")
            row_cache[mask] = _node(
                nodes,
                "Concat",
                [bg, c1, c2, zero_row, c4, zero_row, zero_row, zero_row, zero_row, zero_row],
                f"out{row}",
                axis=1,
            )
        row_outputs.append(row_cache[mask])

    out15b = _node(nodes, "Concat", row_outputs, "out15b", axis=2)
    out15 = _node(nodes, "Cast", [out15b], "out15", to=TensorProto.FLOAT)
    _node(nodes, "Pad", [out15], OUT_NAME, pads=[0, 0, 0, 0, 0, 0, H - ACTIVE, W - ACTIVE])

    graph = helper.make_graph(nodes, "task128", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def verify(path: Path) -> tuple[int, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data.get(split, []):
            pred = sess.run([OUT_NAME], {IN_NAME: _onehot(ex["input"])})[0]
            expected = _onehot(ex["output"])
            if not np.array_equal(pred > 0.0, expected > 0.0):
                raise AssertionError(f"failed {split} example {total}: got\n{_grid_from_onehot(pred)}")
            passed += 1
            total += 1
    return passed, total


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    onnx.save(model, ROOT_PATH)
    passed, total = verify(ROOT_PATH)
    print(f"saved {ROOT_PATH}")
    print(f"saved {BEST_PATH}")
    print(f"verified {passed}/{total}")


if __name__ == "__main__":
    main()
