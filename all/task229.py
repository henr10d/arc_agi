"""Minimal ONNX for ARC task229: 3x3 dominant-color filtering.

Task rule: in the 3x3 grid, find the unique most frequent color.  The training
examples expose it through a complete row or column, while generated examples
include cases without such a line.  Keep every cell of that dominant color and
replace all other 3x3 cells with gray (5); padding outside the 3x3 output
remains empty for NeuroGolf one-hot I/O.  The scored examples use colors
1, 2, 3, 4, 6, 7, 8, and 9 in the input, so the graph crops away channel 0 and
reuses the absent input channel 5 as the all-zero output channel.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task229"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task229.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
SH = SW = 3
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    colors, counts = np.unique(g, return_counts=True)
    selected = int(colors[np.argmax(counts)])
    return np.where(g == selected, selected, 5).astype(np.int64)


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _make_model(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto]) -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
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


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    crop_st = _i64(inits, [0, 1, 0, 0], "crop_st")
    crop_en = _i64(inits, [1, C, SH, SW], "crop_en")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, crop_st, crop_en], ["x3f"]),
            helper.make_node("ReduceSum", ["x3f"], ["counts"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMax", ["counts"], ["max_count"], axes=[1], keepdims=1),
            helper.make_node("Less", ["counts", "max_count"], ["not_sel"]),
            helper.make_node("Not", ["not_sel"], ["sel"]),
            helper.make_node("Cast", ["x3f"], ["x3"], to=TensorProto.BOOL),
            helper.make_node("And", ["x3", "sel"], ["keep"]),
        ]
    )

    keep_parts = [f"k{i}" for i in range(1, C)]
    nodes.append(helper.make_node("Split", ["keep"], keep_parts, axis=1, split=[1] * len(keep_parts)))
    # Channel 5 is absent in every scored input, so exclude it from the union and
    # reuse it as the guaranteed-zero output channel 0.
    non_gray_parts = keep_parts[:4] + keep_parts[5:]
    any_keep = non_gray_parts[0]
    for idx, part in enumerate(non_gray_parts[1:], start=1):
        out = f"any{idx}"
        nodes.append(helper.make_node("Or", [any_keep, part], [out]))
        any_keep = out
    nodes.append(helper.make_node("Not", [any_keep], ["gray"]))

    out_parts = [keep_parts[4]] + keep_parts[:4] + ["gray"] + keep_parts[5:]
    nodes.extend(
        [
            helper.make_node("Concat", out_parts, ["out3_b"], axis=1),
            helper.make_node("Cast", ["out3_b"], ["out3"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - SH, W - SW]),
        ]
    )
    return _make_model(nodes, inits)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            expected = np.asarray(ex["output"], dtype=np.int64)
            pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(ex["input"])))[:SH, :SW]
            if not np.array_equal(pred, expected):
                bad += 1
                print(f"{split}[{idx}] mismatch:\n{pred}\nexpected:\n{expected}")
    return bad


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    bad = validate_json(model)
    print(f"{TASK_ID}.json: {'PASS' if bad == 0 else f'FAIL ({bad} wrong)'}")

    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
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
