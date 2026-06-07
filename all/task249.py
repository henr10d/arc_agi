"""Minimal ONNX for ARC task249: duplicate each row horizontally.

Task rule: for an HxW input grid, produce an Hx(2W) output grid where each
row is the original row followed by a second copy of the same row. The task
data only uses heights and widths 3..5, so the competition graph works on the
compact 5x5 crop, selects the correct duplicated width, and pads back to the
required 30x30 one-hot tensor.

ONNX: slice the one-hot input crop, concatenate it with itself along the
channel-first width axis, select the width-3/4/5 candidate from padding
presence, then final-pad to the NeuroGolf output contract.
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

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task249"
BEST_PATH = OUT_DIR / "task249.onnx"
DATA_PATH = ROOT / "data" / "task249.json"

C = 10
H = W = 30
MAX_HW = 5
MAX_OUT_W = MAX_HW * 2
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
    """Reference solver: concatenate the grid with itself on the column axis."""
    return np.concatenate([grid, grid], axis=1)


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

    axes = _i64(inits, [0, 1, 2, 3], "axes")
    st = _i64(inits, [0, 0, 0, 0], "st")
    e5 = _i64(inits, [1, C, MAX_HW, MAX_HW], "e5")
    e4 = _i64(inits, [1, C, MAX_HW, 4], "e4")
    e3 = _i64(inits, [1, C, MAX_HW, 3], "e3")
    col4_st = _i64(inits, [0, 0, 0, 4], "col4_st")
    col4_en = _i64(inits, [1, C, MAX_HW, 5], "col4_en")
    col3_st = _i64(inits, [0, 0, 0, 3], "col3_st")
    col3_en = _i64(inits, [1, C, MAX_HW, 4], "col3_en")
    zero = _f32(inits, [0.0], "zero")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st, e5, axes], ["x5"]),
            helper.make_node("Slice", ["x5", st, e4, axes], ["x4"]),
            helper.make_node("Slice", ["x4", st, e3, axes], ["x3"]),
            helper.make_node("Concat", ["x3", "x3"], ["y3"], axis=3),
            helper.make_node("Pad", ["y3"], ["p3"], pads=[0, 0, 0, 0, 0, 0, 0, MAX_OUT_W - 6]),
            helper.make_node("Concat", ["x4", "x4"], ["y4"], axis=3),
            helper.make_node("Pad", ["y4"], ["p4"], pads=[0, 0, 0, 0, 0, 0, 0, MAX_OUT_W - 8]),
            helper.make_node("Concat", ["x5", "x5"], ["p5"], axis=3),
            helper.make_node("Slice", [IN_NAME, col3_st, col3_en, axes], ["col3"]),
            helper.make_node("ReduceSum", ["col3"], ["sum3"], axes=[1, 2, 3], keepdims=1),
            helper.make_node("Greater", ["sum3", zero], ["is_w4_or_w5"]),
            helper.make_node("Slice", [IN_NAME, col4_st, col4_en, axes], ["col4"]),
            helper.make_node("ReduceSum", ["col4"], ["sum4"], axes=[1, 2, 3], keepdims=1),
            helper.make_node("Greater", ["sum4", zero], ["is_w5"]),
            helper.make_node("Where", ["is_w4_or_w5", "p4", "p3"], ["w34"]),
            helper.make_node("Where", ["is_w5", "p5", "w34"], ["out10"]),
            helper.make_node("Pad", ["out10"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - MAX_HW, W - MAX_OUT_W]),
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


def build_dynamic_concat_model() -> onnx.ModelProto:
    """Non-submission fallback: true dynamic H,W row duplication via one Concat."""
    x_info = helper.make_tensor_value_info(
        IN_NAME,
        TensorProto.FLOAT,
        [1, C, "height", "width"],
    )
    y_info = helper.make_tensor_value_info(
        OUT_NAME,
        TensorProto.FLOAT,
        [1, C, "height", "double_width"],
    )
    node = helper.make_node("Concat", [IN_NAME, IN_NAME], [OUT_NAME], axis=3)
    graph = helper.make_graph([node], "task249_dynamic", [x_info], [y_info])
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

    bad: dict[str, int] = {}
    for split in ("train", "test", "arc-gen"):
        bad[split] = 0
        for ex in data[split]:
            g = np.asarray(ex["input"], dtype=np.int64)
            expected = np.asarray(ex["output"], dtype=np.int64)
            pred = _run_onnx(model, _grid_to_onehot(ex["input"]))
            expected_onehot = convert_to_numpy(ex, "output")
            if expected_onehot is None:
                continue
            pred_crop = _onehot_to_grid(pred)[: expected.shape[0], : expected.shape[1]]
            if (
                not np.array_equal(pred > 0.0, expected_onehot > 0.0)
                or not np.array_equal(pred_crop, expected)
                or not np.array_equal(solve(g), expected)
            ):
                bad[split] += 1
    return bad


def validate_dynamic_model() -> bool:
    model = build_dynamic_concat_model()
    rng = np.random.default_rng(249)
    for height in (1, 2, 6):
        for width in (1, 3, 7):
            ids = rng.integers(0, C, size=(height, width), dtype=np.int64)
            x = np.zeros((1, C, height, width), dtype=np.float32)
            for r in range(height):
                for c in range(width):
                    x[0, ids[r, c], r, c] = 1.0
            y = _run_onnx(model, x)
            if y.shape != (1, C, height, width * 2):
                return False
            if not np.array_equal(y, np.concatenate([x, x], axis=3)):
                return False
    return True


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    bad = validate_json(model)
    train_status = "PASS" if bad["train"] == 0 else f"FAIL ({bad['train']} wrong)"
    test_status = "PASS" if bad["test"] == 0 else f"FAIL ({bad['test']} wrong)"
    arc_status = "PASS" if bad["arc-gen"] == 0 else f"FAIL ({bad['arc-gen']} wrong)"
    print(f"task249 train:   {train_status}")
    print(f"task249 test:    {test_status}")
    print(f"task249 arc-gen: {arc_status}")
    print(f"dynamic H,W concat fallback: {'PASS' if validate_dynamic_model() else 'FAIL'}")

    if any(bad.values()):
        raise SystemExit(1)

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
