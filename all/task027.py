"""Minimal ONNX for ARC task027: complete 180-degree rotational symmetry.

Task rule: the 10x10 input contains one blue shape on black background. The
output keeps all blue cells and paints red only in empty cells that are required
to complete the shape under 180-degree rotation. The latent center is either
(4.5, 4.5) or (5, 5); the correct center is the one that creates fewer missing
counterpart cells.

ONNX: crop the blue 10x10 mask, build two rotated masks with Gather, mask out
the invalid row/column for the (5, 5) center, count missing cells for both
centers, select the smaller mask, and emit 10x10 one-hot channels before one
final Pad to the fixed competition output.
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

BEST_PATH = OUT_DIR / "task027.onnx"
DATA_PATH = ROOT / "data" / "task027.json"

C = 10
H = W = 30
GH = GW = 10
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
    """Reference solver: add the smaller 180-rotation completion in red."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    preds = []
    for s in (9, 10):
        red = np.zeros_like(g, dtype=bool)
        for r, c in np.argwhere(g == 1):
            rr, cc = s - int(r), s - int(c)
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and g[rr, cc] == 0:
                red[rr, cc] = True
        preds.append(red)
    red = preds[0] if int(preds[0].sum()) < int(preds[1].sum()) else preds[1]
    out[red] = 2
    return out


def _grid_to_onehot(grid: List[List[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    arr = onehot.reshape(C, H, W)
    active = arr > 0.0
    bad = active.sum(axis=0) != 1
    decoded = arr.argmax(axis=0).astype(np.int64)
    decoded[bad] = -1
    return decoded


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    starts = _i64(inits, [0, 1, 0, 0], "st")
    ends = _i64(inits, [1, 2, GH, GW], "en")
    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    half = _f32(inits, [0.5], "half")
    half16 = _init(inits, np.asarray([0.5], dtype=np.float16), "half16")
    rev9 = _i64(inits, list(range(9, -1, -1)), "rev9")
    rev10 = _i64(inits, [0] + list(range(9, 0, -1)), "rev10")
    valid10 = np.zeros((1, 1, GH, GW), dtype=np.bool_)
    valid10[:, :, 1:, 1:] = True
    mask10 = _bool(inits, valid10, "mask10")
    pads = [0, 0, 0, 0, 0, C - 3, H - GH, W - GW]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts, ends, axes4], ["blue_f"]),
            helper.make_node("Greater", ["blue_f", half], ["blue"]),
            helper.make_node("Gather", ["blue", rev9], ["r9_y"], axis=2),
            helper.make_node("Gather", ["r9_y", rev9], ["rot9"], axis=3),
            helper.make_node("Gather", ["blue", rev10], ["r10_y"], axis=2),
            helper.make_node("Gather", ["r10_y", rev10], ["rot10_raw"], axis=3),
            helper.make_node("And", ["rot10_raw", mask10], ["rot10"]),
            helper.make_node("Not", ["blue"], ["not_blue"]),
            helper.make_node("And", ["rot9", "not_blue"], ["miss9"]),
            helper.make_node("And", ["rot10", "not_blue"], ["miss10"]),
            helper.make_node("Cast", ["miss9"], ["miss9_h"], to=TensorProto.FLOAT16),
            helper.make_node("Cast", ["miss10"], ["miss10_h"], to=TensorProto.FLOAT16),
            helper.make_node("ReduceSum", ["miss9_h"], ["sum9"], axes=[0, 1, 2, 3], keepdims=0),
            helper.make_node("ReduceSum", ["miss10_h"], ["sum10"], axes=[0, 1, 2, 3], keepdims=0),
            helper.make_node("Less", ["sum9", "sum10"], ["use9"]),
            helper.make_node("Where", ["use9", "miss9_h", "miss10_h"], ["red_h"]),
            helper.make_node("Greater", ["red_h", half16], ["red"]),
            helper.make_node("Or", ["blue", "red"], ["fg"]),
            helper.make_node("Not", ["fg"], ["bg"]),
            helper.make_node("Cast", ["bg"], ["bg_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["red_h"], ["red_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "Concat",
                ["bg_f", "blue_f", "red_f"],
                ["out3"],
                axis=1,
            ),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=pads),
        ]
    )

    graph = helper.make_graph(nodes, "task027", [x_info], [y_info], initializer=inits)
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


def validate_json(model: onnx.ModelProto) -> tuple[int, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            g = np.array(ex["input"], dtype=np.int64)
            if g.shape[0] > H or g.shape[1] > W:
                continue
            total += 1
            oh = _grid_to_onehot(ex["input"])
            pred = _onehot_to_grid(_run_onnx(model, oh))[: g.shape[0], : g.shape[1]]
            expected = np.array(ex["output"], dtype=np.int64)
            if not np.array_equal(pred, expected):
                bad += 1
                print(f"mismatch {split} #{total - 1}")
                print(pred)
                print(expected)
                break
    return bad, total


def print_cost_heuristic() -> None:
    print(
        "Heuristic cost: two 10x10 bool rotations, two 10x10 float16 count "
        "casts, one 10x10 red float32 cast, one 10x10 background float32 cast, "
        "and one 3-channel 10x10 pre-pad tensor. Dominant memory is about "
        "4 KiB; params are about 135 elements, mostly the valid-center mask."
    )
    print(
        "Near-optimal rationale: the graph tests only the two possible centers and "
        "uses the smaller missing mask; this avoids connected components, generic "
        "symmetry search, full 30x30 intermediates, and per-cell constants."
    )


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    bad, total = validate_json(model)
    assert bad == 0, f"{bad} mismatches across {total} examples"
    print(f"verified {total} task027 examples")
    print_cost_heuristic()

    try:
        result = score_file(BEST_PATH)
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"score_model failed: {exc}")
    else:
        print(result)


if __name__ == "__main__":
    main()
