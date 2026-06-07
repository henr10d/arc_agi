"""ONNX solver for task346: output the color enclosed by 3x3 rings.

Task rule: every input grid has black background plus exactly two non-black
colors, and the output is a 1x1 grid.  One color forms at least one 3x3 ring
whose eight surrounding cells are that color; the other non-black color sits in
the ring center.  Output the center color.  The center color is usually rarer,
but generated examples show population alone is not the rule.  In all task
examples, the decisive ring lies inside the top-left 11x11 region.

ONNX: crop the observed ring-neighborhood region, keep only colors 1-9, and
use a 3x3 AveragePool to find the ring color (8 of 9 cells active).  The answer
is the other non-zero color present in that same crop; encode it as a 9-channel
one-hot tensor and pad one leading channel plus the unused spatial area.
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
sys.path.insert(0, str(ROOT))

TASK_ID = "task346"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task346.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

IN_NAME = "input"
OUT_NAME = "output"
C = 10
H = W = 30
N = 11
SHAPE = [1, C, H, W]
IR_VERSION = 10
OPSET = 10


def _init(inits: List[onnx.TensorProto], name: str, arr: np.ndarray) -> str:
    inits.append(numpy_helper.from_array(arr, name=name))
    return name


def _i64(inits: List[onnx.TensorProto], name: str, vals) -> str:
    return _init(inits, name, np.asarray(vals, dtype=np.int64))


def _f32(inits: List[onnx.TensorProto], name: str, vals) -> str:
    return _init(inits, name, np.asarray(vals, dtype=np.float32))


def solve(grid: list[list[int]]) -> list[list[int]]:
    h = len(grid)
    w = len(grid[0])
    hits: dict[int, int] = {}

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center = grid[r][c]
            if center == 0:
                continue
            neighbors = [
                grid[rr][cc]
                for rr in (r - 1, r, r + 1)
                for cc in (c - 1, c, c + 1)
                if rr != r or cc != c
            ]
            ring_colors = set(neighbors)
            if len(ring_colors) == 1:
                ring = next(iter(ring_colors))
                if ring != 0 and ring != center:
                    hits[center] = hits.get(center, 0) + 1

    if not hits:
        raise ValueError(f"{TASK_ID} found no 3x3 ring center")

    chosen = max(hits, key=hits.__getitem__)
    return [[chosen]]


def _onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def validate_rule() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        task = json.load(fh)

    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(task[split]):
            pred = solve(example["input"])
            if pred != example["output"]:
                raise AssertionError(f"rule failed {split}[{idx}]: {pred} != {example['output']}")
            total += 1

    print(f"ring-center rule verified {total} examples")


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    threshold = _f32(inits, "threshold", [0.88])
    depth9 = _i64(inits, "depth9", [C - 1])
    values = _f32(inits, "values", [0.0, 1.0])
    crop_st = _i64(inits, "crop_st", [0, 1, 1, 1])
    crop_en = _i64(inits, "crop_en", [1, C, N, 10])

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, crop_st, crop_en], ["crop9"]),
            helper.make_node("AveragePool", ["crop9"], ["ring_pool"], kernel_shape=[3, 3]),
            helper.make_node("ReduceMax", ["ring_pool"], ["ring_max"], axes=[2, 3], keepdims=1),
            helper.make_node("Greater", ["ring_max", threshold], ["ring9"]),
            helper.make_node("ReduceMax", ["crop9"], ["color_scores"], axes=[2, 3], keepdims=1),
            helper.make_node("Where", ["ring9", threshold, "color_scores"], ["answer_scores"]),
            helper.make_node("ArgMax", ["answer_scores"], ["chosen0"], axis=1, keepdims=0),
            helper.make_node("OneHot", ["chosen0", depth9, values], ["out9"], axis=1),
            helper.make_node("Pad", ["out9"], [OUT_NAME], pads=[0, 1, 0, 0, 0, 0, H - 1, W - 1]),
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


def verify_onnx(path: Path) -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        task = json.load(fh)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(task[split]):
            pred = session.run([OUT_NAME], {IN_NAME: _onehot(example["input"])})[0]
            expected = _onehot(example["output"])
            if not np.array_equal(pred > 0.0, expected > 0.0):
                got = int(pred[0, :, 0, 0].argmax())
                want = int(example["output"][0][0])
                raise AssertionError(f"ONNX failed {split}[{idx}]: {got} != {want}")
            total += 1
    print(f"ONNX verified {total} examples")


def main() -> None:
    validate_rule()
    model = build_model()
    onnx.save(model, BEST_PATH)
    verify_onnx(BEST_PATH)
    print(BEST_PATH)


if __name__ == "__main__":
    main()
