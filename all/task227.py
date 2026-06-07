"""Minimal ONNX for ARC task227: mark cells absent from both stacked patterns.

Task rule: the 8x4 input is two aligned 4x4 layers. The top layer contains
green pixels on black and the bottom layer contains blue pixels on black. The
4x4 output is red exactly where the corresponding top and bottom cells are both
black; every other output cell is black.

ONNX: slice the two 4x4 background masks from channel 0, compute their boolean
intersection as the red mask, build only the three needed channels
[background, zero, red], cast to float, then let the final unscored Pad add the
unused channels and spatial padding for the required 30x30 competition tensor.
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

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task227"
BEST_PATH = OUT_DIR / "task227.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
OH = OW = 4
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


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def solve(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    """Return red where aligned top and bottom cells are both background."""
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros((OH, OW), dtype=np.int64)
    out[(arr[:OH, :OW] == 0) & (arr[OH : 2 * OH, :OW] == 0)] = 2
    return out


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    top_st = _i64(inits, [0, 0, 0, 0], "top_st")
    top_en = _i64(inits, [1, 1, OH, OW], "top_en")
    bot_st = _i64(inits, [0, 0, OH, 0], "bot_st")
    bot_en = _i64(inits, [1, 1, 2 * OH, OW], "bot_en")
    pads = [0, 0, 0, 0, 0, C - 3, H - OH, W - OW]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, top_st, top_en], ["top0"]),
            helper.make_node("Slice", [IN_NAME, bot_st, bot_en], ["bot0"]),
            helper.make_node("Cast", ["top0"], ["topb"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["bot0"], ["botb"], to=TensorProto.BOOL),
            helper.make_node("And", ["topb", "botb"], ["redb"]),
            helper.make_node("Not", ["redb"], ["bg_b"]),
            helper.make_node("And", ["redb", "bg_b"], ["zero1_b"]),
            helper.make_node(
                "Concat",
                ["bg_b", "zero1_b", "redb"],
                ["out3_b"],
                axis=1,
            ),
            helper.make_node("Cast", ["out3_b"], ["out3"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=pads),
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
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            expected = np.asarray(ex["output"], dtype=np.int64)
            ref = solve(ex["input"])
            if not np.array_equal(ref, expected):
                raise AssertionError(f"reference mismatch on {split}[{idx}]")
            pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(ex["input"])))[:OH, :OW]
            if not np.array_equal(pred, expected):
                bad += 1
    return bad


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    bad = validate_json(model)
    if bad:
        raise AssertionError(f"{bad} JSON examples failed")

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
