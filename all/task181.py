"""Minimal ONNX for ARC task181: mirror the cyan core toward the yellow stem.

Task rule: the active grid is 6x9. Rows 0-2 contain a cyan shape inside the
middle 3 columns. Rows 3-5 contain a yellow/olive T shape. Preserve the yellow
shape exactly. If the T shape's upper stem cell is on the left, copy a
horizontally mirrored version of the 3x3 cyan core into columns 0-2; if the
stem is on the right, copy the mirrored cyan core into columns 6-8. The
original cyan core in columns 3-5 remains in place and all other cells stay
background.
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

TASK_ID = "task181"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
GH = 6
GW = 9
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


def solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    core = g[:3, 3:6] == 8
    mirrored = core[:, ::-1]
    cyan = np.zeros((3, 9), dtype=bool)
    cyan[:, 3:6] = core
    if g[3, 3] == 4:
        cyan[:, 0:3] = mirrored
    else:
        cyan[:, 6:9] = mirrored
    out[:3] = np.where(cyan, 8, 0)
    return out


def _grid_to_onehot(grid: List[List[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    arr = onehot.reshape(C, H, W)
    return arr.argmax(axis=0).astype(np.int64)


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes_chw = _i64(inits, [1, 2, 3], "axes_chw")
    core_st = _i64(inits, [8, 0, 3], "core_st")
    core_en = _i64(inits, [9, 3, 6], "core_en")
    stem_st = _i64(inits, [4, 3, 3], "stem_st")
    stem_en = _i64(inits, [5, 4, 4], "stem_en")
    gather_rev = _i64(inits, [2, 1, 0], "gather_rev")
    bottom_left_top = _init(inits, np.asarray([[[[0, 0, 0, 1, 0, 0, 0, 0, 0]]]], dtype=bool), "bottom_left_top")
    bottom_right_top = _init(inits, np.asarray([[[[0, 0, 0, 0, 0, 1, 0, 0, 0]]]], dtype=bool), "bottom_right_top")
    bottom_mid = _init(inits, np.asarray([[[[0, 0, 0, 1, 1, 1, 0, 0, 0]]]], dtype=bool), "bottom_mid")
    bottom_low = _init(inits, np.asarray([[[[0, 0, 0, 0, 1, 0, 0, 0, 0]]]], dtype=bool), "bottom_low")
    pads = [0, 0, 0, 0, 0, 0, H - GH, W - GW]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, core_st, core_en, axes_chw], ["core"]),
            helper.make_node("Cast", ["core"], ["core_b"], to=TensorProto.BOOL),
            helper.make_node("Gather", ["core_b", gather_rev], ["mir"], axis=3),
            helper.make_node("Slice", [IN_NAME, stem_st, stem_en, axes_chw], ["stem_f"]),
            helper.make_node("Cast", ["stem_f"], ["left_flag"], to=TensorProto.BOOL),
            helper.make_node("Not", ["left_flag"], ["right_flag"]),
            helper.make_node("And", ["left_flag", "mir"], ["left_mir"]),
            helper.make_node("And", ["right_flag", "mir"], ["right_mir"]),
            helper.make_node("Concat", ["left_mir", "core_b", "right_mir"], ["cyan9"], axis=3),
            helper.make_node("Not", ["cyan9"], ["bg_b"]),
            helper.make_node("And", ["cyan9", "bg_b"], ["zero9"]),
            helper.make_node("And", ["left_flag", bottom_left_top], ["bottom_left"]),
            helper.make_node("And", ["right_flag", bottom_right_top], ["bottom_right"]),
            helper.make_node("Or", ["bottom_left", "bottom_right"], ["bottom_top"]),
            helper.make_node("Concat", ["bottom_top", bottom_mid, bottom_low], ["bottom4"], axis=2),
            helper.make_node("Not", ["bottom4"], ["bottom_bg"]),
            helper.make_node("Concat", ["bg_b", "bottom_bg"], ["bg6"], axis=2),
            helper.make_node("Concat", ["zero9", "bottom4"], ["ch4"], axis=2),
            helper.make_node("Concat", ["cyan9", "zero9"], ["ch8"], axis=2),
            helper.make_node("Concat", ["zero9", "zero9"], ["zero6"], axis=2),
            helper.make_node(
                "Concat",
                ["bg6", "zero6", "zero6", "zero6", "ch4", "zero6", "zero6", "zero6", "ch8", "zero6"],
                ["out6_b"],
                axis=1,
            ),
            helper.make_node("Cast", ["out6_b"], ["out6"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out6"], [OUT_NAME], pads=pads),
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
        for idx, ex in enumerate(data[split]):
            inp = np.array(ex["input"], dtype=np.int64)
            expected = np.array(ex["output"], dtype=np.int64)
            ref = solve(inp)
            if not np.array_equal(ref, expected):
                raise AssertionError(f"reference mismatch on {split}[{idx}]")
            pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(ex["input"])))[: inp.shape[0], : inp.shape[1]]
            if not np.array_equal(pred, expected):
                print(f"ONNX mismatch on {split}[{idx}]")
                print(pred)
                print(expected)
                bad += 1
    return bad


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    bad = validate_json(model)
    assert bad == 0, f"{bad} JSON examples failed"

    result = score_file(BEST_PATH)
    print(
        f"{BEST_PATH.name}: valid={result['valid']} memory={result['memory']} "
        f"params={result['params']} cost={result['cost']} score={result['score']}"
    )


if __name__ == "__main__":
    main()
