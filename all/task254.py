"""Minimal ONNX for ARC task254: keep only the tallest and shortest bars.

Task rule: the 9x9 input contains gray vertical bars rising from the bottom.
Measure each gray bar's height by column.  Recolor the unique tallest bar to
blue, recolor the unique shortest bar to red, and make every other cell black.
The chosen bars keep their original columns and heights.

ONNX: work on only gray channel 5 in the 9x9 crop, keep column heights as
float32, use ArgMax/ArgMin to find the selected columns, build compact bool
channels 0-2, cast that 3-channel crop to float, then final Pad supplies zero
channels 3-9 and the unused 30x30 area.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task254"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
G = 9
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


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _make_model(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], name: str) -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(nodes, name, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver for the 9x9 gray-bar rule."""
    g = np.asarray(grid, dtype=np.int64)
    heights = (g == 5).sum(axis=0)
    cols = np.flatnonzero(heights > 0)
    max_col = int(cols[np.argmax(heights[cols])])
    min_col = int(cols[np.argmin(heights[cols])])
    out = np.zeros_like(g)
    out[g[:, max_col] == 5, max_col] = 1
    out[g[:, min_col] == 5, min_col] = 2
    return out


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def _gray_crop(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto]) -> str:
    starts = _i64(inits, [0, 5, 0, 0], "starts")
    ends = _i64(inits, [1, 6, G, G], "ends")
    nodes.append(helper.make_node("Slice", [IN_NAME, starts, ends], ["gray"]))
    return "gray"


def _finish_from_gray_and_cols(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    gray_for_pixels: str,
    maxcol: str,
    mincol: str,
    name: str,
) -> onnx.ModelProto:
    nodes.extend(
        [
            helper.make_node("Cast", [gray_for_pixels], ["grayb"], to=TensorProto.BOOL),
            helper.make_node("And", ["grayb", maxcol], ["blue"]),
            helper.make_node("And", ["grayb", mincol], ["red"]),
            helper.make_node("Or", ["blue", "red"], ["painted"]),
            helper.make_node("Not", ["painted"], ["bg"]),
            helper.make_node("Concat", ["bg", "blue", "red"], ["out3b"], axis=1),
            helper.make_node("Cast", ["out3b"], ["out3"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=[0, 0, 0, 0, 0, C - 3, H - G, W - G]),
        ]
    )
    return _make_model(nodes, inits, name)


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    gray = _gray_crop(nodes, inits)
    zero = _f32(inits, 0, "zero")
    big = _f32(inits, 10, "big")
    cols = _i64(inits, np.arange(G, dtype=np.int64).reshape(1, 1, 1, G), "cols")
    nodes.extend(
        [
            helper.make_node("ReduceSum", [gray], ["heightf"], axes=[2], keepdims=1),
            helper.make_node("Greater", ["heightf", zero], ["hasbar"]),
            helper.make_node("Where", ["hasbar", "heightf", big], ["minsrc"]),
            helper.make_node("ArgMax", ["heightf"], ["maxidx"], axis=3, keepdims=1),
            helper.make_node("ArgMin", ["minsrc"], ["minidx"], axis=3, keepdims=1),
            helper.make_node("Equal", [cols, "maxidx"], ["maxcol"]),
            helper.make_node("Equal", [cols, "minidx"], ["mincol"]),
        ]
    )
    return _finish_from_gray_and_cols(nodes, inits, gray, "maxcol", "mincol", "task254_arg_indices")


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            g = np.asarray(ex["input"], dtype=np.int64)
            expected = np.asarray(ex["output"], dtype=np.int64)
            pred_oh = _run_onnx(model, _grid_to_onehot(ex["input"]))
            pred = _onehot_to_grid(pred_oh)[: g.shape[0], : g.shape[1]]
            active = pred_oh[0, :, : expected.shape[0], : expected.shape[1]] > 0.0
            if not np.array_equal(pred, expected) or not np.all(active.sum(axis=0) == 1):
                bad += 1
    return bad


def _score_model(model: onnx.ModelProto) -> dict[str, Any]:
    bad = validate_json(model)
    if bad:
        raise AssertionError(f"model failed {bad} examples")

    with tempfile.NamedTemporaryFile(suffix=f"_{TASK_ID}.onnx", dir=OUT_DIR, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        onnx.save(model, tmp_path)
        result = score_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not result["valid"]:
        raise AssertionError(f"model invalid: {result['error']}")
    assert result["cost"] is not None and result["score"] is not None
    return result


def main() -> None:
    model = build_model()
    result = _score_model(model)
    print(
        f"arg-indices: nodes={len(model.graph.node)} memory={result['memory']} "
        f"params={result['params']} cost={result['cost']} score={result['score']:.6f}"
    )
    onnx.save(model, BEST_PATH)
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
