"""Minimal ONNX for ARC task231 using horizontal periodic extension.

Task rule: preserve the 5-row input height and extend each row horizontally to
twice its original width by continuing the row's repeating color pattern.  The
task uses widths 6 through 10, so the active output width is exactly 12 through
20, with the remaining NeuroGolf 30x30 tensor left as all-zero padding.

The ONNX graph works on the compact 5x10 task area, detects whether the pattern
has period 2 or period 3 from the first columns, reads the active width from
the all-zero top row, gathers the corresponding 20-column continuation, masks
columns beyond 2*input_width, then pads once.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task231"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task231.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
TASK_H = 5
MAX_IN_W = 10
SRC_W = MAX_IN_W + 1
MAX_OUT_W = 20
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference solver: infer period 2 or 3, then continue to double width."""
    arr = np.asarray(grid, dtype=np.int64)
    period = 2 if np.array_equal(arr[:, :2], arr[:, 2:4]) else 3
    cols = np.arange(arr.shape[1] * 2) % period
    return arr[:, cols]


def build_onnx_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, [0, 1, 2, 3], "axes")
    st_core = _i64(inits, [0, 0, 0, 0], "st_core")
    en_core = _i64(inits, [1, C, TASK_H, SRC_W], "en_core")
    zero = _f32(inits, np.asarray(0.0, dtype=np.float32), "zero")
    p2_idx = _i64(inits, [0, 1] * 10, "p2_idx")
    p3_idx = _i64(inits, [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1], "p3_idx")
    pad_idx = _i64(inits, np.asarray(MAX_IN_W, dtype=np.int64), "pad_idx")
    active_src_idx = _i64(inits, [i // 2 for i in range(MAX_OUT_W)], "active_src_idx")
    zrow_start = _i64(inits, [0, 0], "zrow_start")
    zrow_end = _i64(inits, [1, 1], "zrow_end")
    zrow_axes = _i64(inits, [1, 2], "zrow_axes")
    p0_start = _i64(inits, [1, 0], "p0_start")
    p0_end = _i64(inits, [4, 2], "p0_end")
    p2_start = _i64(inits, [1, 2], "p2_start")
    p2_end = _i64(inits, [4, 4], "p2_end")
    p_axes = _i64(inits, [2, 3], "p_axes")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st_core, en_core, axes], ["core"]),
            helper.make_node("Greater", ["core", zero], ["core_b"]),
            helper.make_node("Slice", ["core_b", p0_start, p0_end, p_axes], ["left2"]),
            helper.make_node("Slice", ["core_b", p2_start, p2_end, p_axes], ["right2"]),
            helper.make_node("Equal", ["left2", "right2"], ["p2_eq"]),
            helper.make_node("Cast", ["p2_eq"], ["p2_f"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMin", ["p2_f"], ["p2_min"], axes=[0, 1, 2, 3], keepdims=0),
            helper.make_node("Greater", ["p2_min", zero], ["is_p2"]),
            helper.make_node("Where", ["is_p2", p2_idx, p3_idx], ["period_idx"]),
            helper.make_node("Slice", ["core_b", zrow_start, zrow_end, zrow_axes], ["col_active"]),
            helper.make_node("Gather", ["col_active", active_src_idx], ["out_active"], axis=3),
            helper.make_node("Squeeze", ["out_active"], ["out_active_1d"], axes=[0, 1, 2]),
            helper.make_node("Where", ["out_active_1d", "period_idx", pad_idx], ["gather_idx"]),
            helper.make_node("Gather", ["core", "gather_idx"], ["continued"], axis=3),
            helper.make_node(
                "Pad",
                ["continued"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - TASK_H, W - MAX_OUT_W],
            ),
        ]
    )

    graph = helper.make_graph(nodes, "task231_period_extend", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _examples() -> Iterable[tuple[str, int, dict[str, list[list[int]]]]]:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            yield split, idx, example


def validate_model(path: Path) -> None:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for split, idx, example in _examples():
        inp = convert_to_numpy(example, "input")
        expected = convert_to_numpy(example, "output")
        if inp is None or expected is None:
            continue
        got = session.run([OUT_NAME], {IN_NAME: inp})[0]
        if not np.array_equal(got > 0.0, expected > 0.0):
            raise AssertionError(f"{split}[{idx}] failed")


def main() -> None:
    model = build_onnx_model()
    onnx.save(model, BEST_PATH)
    validate_model(BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"saved: {BEST_PATH}")
    print(f"valid: {result['valid']}")
    print(f"memory: {result['memory']}")
    print(f"params: {result['params']}")
    print(f"cost: {result['cost']}")
    print(f"score: {result['score']:.6f}" if result["score"] is not None else "score: INVALID")


if __name__ == "__main__":
    main()
