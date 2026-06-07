"""Minimal ONNX for ARC task193: remove one-cell-thick noise around blocks.

Task rule: the input contains one non-background color. Keep exactly the
foreground cells that belong to at least one same-color 2x2 square, preserving
their original color and position. This keeps the solid rectangular/block-like
components and removes isolated cells plus single-cell protrusions attached to
otherwise valid blocks. All task grids are at most 20x20, then padded to the
30x30 NeuroGolf one-hot I/O tensor.

ONNX approach: build a compact 20x20 foreground occupancy mask from the full
one-hot active mask minus channel 0, find 2x2 foreground squares with pooling,
and broadcast the retained mask by a tiny detected-color vector only at the end.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task193"
BEST_PATH = OUT_DIR / "task193.onnx"
DATA_PATH = ROOT / "data" / "task193.json"

C = 10
H = W = 30
SH = SW = 20
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def solve(grid: np.ndarray) -> np.ndarray:
    """Keep foreground cells that are part of a monochrome 2x2 block."""
    g = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(g)
    keep = np.zeros_like(g, dtype=bool)
    for color in np.unique(g[g != 0]):
        m = g == color
        sq = m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]
        k = np.zeros_like(m)
        k[:-1, :-1] |= sq
        k[1:, :-1] |= sq
        k[:-1, 1:] |= sq
        k[1:, 1:] |= sq
        keep |= k
    out[keep] = g[keep]
    return out


def _grid_to_onehot(grid: List[List[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes_chw = _i64(inits, [1, 2, 3], "axes_chw")
    active_st = _i64(inits, [0, 0, 0], "active_st")
    active_en = _i64(inits, [1, SH, SW], "active_en")
    bg_st = _i64(inits, [0, 0, 0], "bg_st")
    bg_en = _i64(inits, [1, SH, SW], "bg_en")
    color_st = _i64(inits, [1, 0, 0], "color_st")
    color_en = _i64(inits, [C, 1, 1], "color_en")
    thresh = _f32(inits, [0.99], "thresh")
    out_pads = [0, 0, 0, 0, 0, 0, H - SH, W - SW]

    nodes.extend(
        [
            helper.make_node("ReduceMax", [IN_NAME], ["active30"], axes=[1], keepdims=1),
            helper.make_node("Slice", ["active30", active_st, active_en, axes_chw], ["active"]),
            helper.make_node("Slice", [IN_NAME, bg_st, bg_en, axes_chw], ["bg_in"]),
            helper.make_node("Sub", ["active", "bg_in"], ["fg_any"]),
            helper.make_node("AveragePool", ["fg_any"], ["avg"], kernel_shape=[2, 2], strides=[1, 1]),
            helper.make_node("MaxPool", ["avg"], ["keep_score"], kernel_shape=[2, 2], strides=[1, 1], pads=[1, 1, 1, 1]),
            helper.make_node("ReduceMax", [IN_NAME], ["colors"], axes=[2, 3], keepdims=1),
            helper.make_node("Greater", ["colors", thresh], ["colors_b"]),
            helper.make_node("Slice", ["colors_b", color_st, color_en, axes_chw], ["fg_color_b"]),
            helper.make_node("Greater", ["keep_score", thresh], ["keep_b"]),
            helper.make_node("And", ["keep_b", "fg_color_b"], ["out9_b"]),
            helper.make_node("Greater", ["active", thresh], ["active_b"]),
            helper.make_node("Xor", ["active_b", "keep_b"], ["bg_b"]),
            helper.make_node("Concat", ["bg_b", "out9_b"], ["out20_b"], axis=1),
            helper.make_node("Cast", ["out20_b"], ["out20"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out20"], [OUT_NAME], pads=out_pads),
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
    onnx.checker.check_model(model)
    return model


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> dict[str, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    correct: dict[str, int] = {}
    total: dict[str, int] = {}
    for split in ("train", "test", "arc-gen"):
        correct[split] = 0
        total[split] = 0
        for ex in data[split]:
            g = np.array(ex["input"], dtype=np.int64)
            if g.shape[0] > H or g.shape[1] > W:
                continue
            total[split] += 1
            oh = _grid_to_onehot(ex["input"])
            pred = _onehot_to_grid(_run_onnx(model, oh))[: g.shape[0], : g.shape[1]]
            if np.array_equal(pred, np.array(ex["output"], dtype=np.int64)) and np.array_equal(pred, solve(g)):
                correct[split] += 1
    return {split: correct[split] for split in correct} | {f"{split}_total": total[split] for split in total}


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    acc = validate_json(model)
    result = score_file(BEST_PATH)

    print(f"wrote {BEST_PATH}")
    for split in ("train", "test", "arc-gen"):
        print(f"{split}: {acc[split]}/{acc[f'{split}_total']}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"nodes:   {len(model.graph.node)}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
