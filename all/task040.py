"""ONNX solver for task040: recolor green markers by nearest active border.

The grid is always 10x10.  Two opposite sides are fully colored: either the
left/right columns or the top/bottom rows.  Every green (color 3) marker in the
10x10 area is recolored to the nearer active side's color using the midpoint
split, active border sides are preserved, and every other cell becomes
background color 0.  The graph works on the compact 10x10 crop and pads only at
the final output tensor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "task040"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task040.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

sys.path.insert(0, str(ROOT))

IN_NAME = "input"
OUT_NAME = "output"
C = 10
H = W = 30
N = 10
SHAPE = [1, C, H, W]
IR_VERSION = 10
OPSET = 10


def _init(inits: List[onnx.TensorProto], name: str, arr: np.ndarray) -> str:
    inits.append(numpy_helper.from_array(arr, name=name))
    return name


def _i64(inits: List[onnx.TensorProto], name: str, vals: list[int]) -> str:
    return _init(inits, name, np.asarray(vals, dtype=np.int64))


def _f32(inits: List[onnx.TensorProto], name: str, arr) -> str:
    return _init(inits, name, np.asarray(arr, dtype=np.float32))


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, "a", [0, 1, 2, 3])
    green_st = _i64(inits, "sg", [0, 3, 0, 0])
    green_en = _i64(inits, "eg", [1, 4, N, N])
    c00_st = _i64(inits, "s00", [0, 0, 0, 0])
    c00_en = _i64(inits, "e00", [1, C, 1, 1])
    c09_st = _i64(inits, "s09", [0, 0, 0, 9])
    c09_en = _i64(inits, "e09", [1, C, 1, 10])
    c90_st = _i64(inits, "s90", [0, 0, 9, 0])
    c90_en = _i64(inits, "e90", [1, C, 10, 1])
    zero_f = _f32(inits, "zf", [0.0])
    depth = _i64(inits, "dep", [C])
    values = _f32(inits, "vals", [0.0, 1.0])

    lr_left = np.zeros((1, 1, N), dtype=bool)
    lr_left[:, :, :5] = True
    tb_top = np.zeros((1, N, 1), dtype=bool)
    tb_top[:, :5, :] = True
    lr_sides = np.zeros((1, 1, N), dtype=bool)
    lr_sides[:, :, 0] = True
    lr_sides[:, :, 9] = True
    tb_sides = np.zeros((1, N, 1), dtype=bool)
    tb_sides[:, 0, :] = True
    tb_sides[:, 9, :] = True

    _init(inits, "mll", lr_left)
    _init(inits, "mtt", tb_top)
    _init(inits, "mls", lr_sides)
    _init(inits, "mts", tb_sides)

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, green_st, green_en, axes4], ["green4"]),
            helper.make_node("Cast", ["green4"], ["green4b"], to=TensorProto.BOOL),
            helper.make_node("Squeeze", ["green4b"], ["green"], axes=[1]),
            helper.make_node("Slice", [IN_NAME, c00_st, c00_en, axes4], ["c00oh"]),
            helper.make_node("Slice", [IN_NAME, c09_st, c09_en, axes4], ["c09oh"]),
            helper.make_node("Slice", [IN_NAME, c90_st, c90_en, axes4], ["c90oh"]),
            helper.make_node("ArgMax", ["c00oh"], ["c00i"], axis=1, keepdims=0),
            helper.make_node("ArgMax", ["c09oh"], ["c09i"], axis=1, keepdims=0),
            helper.make_node("ArgMax", ["c90oh"], ["c90i"], axis=1, keepdims=0),
            helper.make_node("Cast", ["c00i"], ["c00"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["c09i"], ["c09"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["c90i"], ["c90"], to=TensorProto.FLOAT),
            # Top/bottom examples have identical top-left and top-right colors.
            helper.make_node("Equal", ["c00i", "c09i"], ["tb"]),
            helper.make_node("Not", ["tb"], ["lr"]),
            helper.make_node("And", ["tb", "mtt"], ["tbtop"]),
            helper.make_node("And", ["lr", "mll"], ["lrleft"]),
            helper.make_node("Or", ["tbtop", "lrleft"], ["near1"]),
            helper.make_node("And", ["tb", "mts"], ["tbsides"]),
            helper.make_node("And", ["lr", "mls"], ["lrsides"]),
            helper.make_node("Or", ["tbsides", "lrsides"], ["sides"]),
            helper.make_node("Where", ["tb", "c90", "c09"], ["c2"]),
            helper.make_node("Where", ["near1", "c00", "c2"], ["paint"]),
            helper.make_node("Or", ["green", "sides"], ["active"]),
            helper.make_node("Where", ["active", "paint", zero_f], ["outgridf"]),
            helper.make_node("Cast", ["outgridf"], ["outgrid"], to=TensorProto.INT64),
            helper.make_node("OneHot", ["outgrid", "dep", "vals"], ["out10"], axis=1),
            helper.make_node(
                "Pad",
                ["out10"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - N, W - N],
            ),
        ]
    )

    graph = helper.make_graph(nodes, "task040", [x_info], [y_info], initializer=inits)
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


def verify(path: Path) -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        task = json.load(fh)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(task[split]):
            pred = session.run([OUT_NAME], {IN_NAME: _onehot(example["input"])})[0]
            expected = _onehot(example["output"])
            if not np.array_equal(pred > 0.0, expected > 0.0):
                raise AssertionError(f"{split}[{idx}] failed")
            total += 1
    print(f"verified {total} examples")


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    verify(BEST_PATH)
    print(BEST_PATH)


if __name__ == "__main__":
    main()
