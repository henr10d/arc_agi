"""Minimal ONNX for ARC task186: count blue cells and draw red count marker.

Task rule: the input and output are 3x3. Count blue cells (color 1) anywhere
in the input. The output is black except for red cells (color 2) in a fixed
count pattern: counts 1-3 fill the top row from left to right, and count 4
adds the center cell. Locations of the input blue cells do not matter.

ONNX: slice the 3x3 blue channel, reduce to a scalar count, compare it with a
3x3 threshold mask for the red channel, derive the black channel by Not, then
pad the compact 3-channel 3x3 tensor to the required competition output.
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

TASK_ID = "task186"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
SH = SW = 3
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


def _bool(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.bool_), name)


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver for JSON validation."""
    count = int((np.asarray(grid) == 1).sum())
    out = np.zeros((SH, SW), dtype=np.int64)
    for r, c in ((0, 0), (0, 1), (0, 2), (1, 1))[:count]:
        out[r, c] = 2
    return out


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
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

    starts = _i64(inits, [0, 1, 0, 0], "starts")
    ends = _i64(inits, [1, 2, SH, SW], "ends")
    false = _bool(inits, False, "false")
    thresholds = _f32(
        inits,
        [[[[0.5, 1.5, 2.5], [9.5, 3.5, 9.5], [9.5, 9.5, 9.5]]]],
        "thresholds",
    )

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts, ends], ["blue3"]),
            helper.make_node("ReduceSum", ["blue3"], ["count"], axes=[1, 2, 3], keepdims=1),
            helper.make_node("Greater", ["count", thresholds], ["redb"]),
            helper.make_node("Not", ["redb"], ["blackb"]),
            helper.make_node("And", ["redb", false], ["zerob"]),
            helper.make_node("Concat", ["blackb", "zerob", "redb"], ["out3b"], axis=1),
            helper.make_node("Cast", ["out3b"], ["out3"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=[0, 0, 0, 0, 0, C - 3, H - SH, W - SW]),
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


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            g = np.array(ex["input"], dtype=np.int64)
            expected = np.array(ex["output"], dtype=np.int64)
            ref = solve(g)
            if not np.array_equal(ref, expected):
                raise AssertionError(f"rule mismatch in {split}[{idx}]:\n{ref}\n{expected}")
            pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(ex["input"])))[:SH, :SW]
            if not np.array_equal(pred, expected):
                bad += 1
                print(f"ONNX mismatch in {split}[{idx}]:\n{pred}\n{expected}")
    return bad


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    bad = validate_json(model)
    if bad:
        raise SystemExit(f"{bad} JSON examples failed")

    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(
        f"valid={result['valid']} memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']}"
    )
    if not result["valid"]:
        raise SystemExit(result["error"])


if __name__ == "__main__":
    main()
