"""Recover shifted periodic checker patterns for NeuroGolf task313.

Task rule: the input is a square grid whose top-left region contains a two- or
three-color repeating pattern. Rows alternate between two horizontal phases. A
uniform filler color replaces some suffix columns and/or suffix rows. The output
keeps the same grid size, removes the filler, and extends the pattern after
advancing the color cycle by one step:
output[r, c] = first_row_pattern[(row_parity + c + 1) % period].

ONNX approach: read the first three one-hot cells to get the dynamic palette,
detect period 2 vs 3 by comparing cells (0,0) and (0,2), construct a compact
color-index grid from precomputed masks for column plus row parity, mask it to
the input footprint, then convert indices to the required one-hot float output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task313"
OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / "task313.onnx"

C = 10
N = 30
M = 20
SHAPE = [1, C, N, N]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference transform using the first-row pattern prefix."""
    arr = np.asarray(grid, dtype=np.int64)
    period = 2 if int(arr[0, 0]) == int(arr[0, 2]) else 3
    palette = arr[0, :period]
    rows = np.arange(arr.shape[0], dtype=np.int64).reshape(-1, 1) % 2
    cols = np.arange(arr.shape[1], dtype=np.int64).reshape(1, -1)
    return palette[(rows + cols + 1) % period]


def _init(inits: list[onnx.TensorProto], arr: Any, name: str, dtype: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=dtype), name=name))
    return name


def _f32(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    return _init(inits, arr, name, np.float32)


def _i64(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    return _init(inits, arr, name, np.int64)


def _load_task() -> dict[str, Any]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _grid_to_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(arr.shape[0], N)):
        for c in range(min(arr.shape[1], N)):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _onehot_to_grid(tensor: np.ndarray, h: int, w: int) -> np.ndarray:
    return tensor[0, :, :h, :w].argmax(axis=0).astype(np.int64)


def _validate_rule(data: dict[str, Any]) -> None:
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(data.get(split, [])):
            expected = np.asarray(example["output"], dtype=np.int64)
            actual = solve(example["input"])
            if not np.array_equal(actual, expected):
                raise ValueError(f"{split}[{index}] does not match the task313 periodic rule")


def build_onnx_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes_hw = _i64(inits, [2, 3], "axes_hw")
    cell0_st = _i64(inits, [0, 0], "cell0_st")
    cell0_en = _i64(inits, [1, 1], "cell0_en")
    cell1_st = _i64(inits, [0, 1], "cell1_st")
    cell1_en = _i64(inits, [1, 2], "cell1_en")
    cell2_st = _i64(inits, [0, 2], "cell2_st")
    cell2_en = _i64(inits, [1, 3], "cell2_en")
    half = _f32(inits, [0.5], "half")
    zero = _f32(inits, [0.0], "zero")
    palette = _init(inits, np.arange(C, dtype=np.int32).reshape(1, C, 1, 1), "palette", np.int32)
    zero_mask = _f32(inits, [0.0], "zero_mask")

    rows = (np.arange(M, dtype=np.int64).reshape(M, 1) % 2)
    cols = np.arange(M, dtype=np.int64).reshape(1, M)
    shifted = rows + cols + 1
    m2_1 = _f32(inits, (shifted % 2 == 1).astype(np.float32).reshape(1, 1, M, M), "m2_1")
    m3_1 = _f32(inits, (shifted % 3 == 1).astype(np.float32).reshape(1, 1, M, M), "m3_1")
    m3_2 = _f32(inits, (shifted % 3 == 2).astype(np.float32).reshape(1, 1, M, M), "m3_2")
    active_st = _i64(inits, [0, 0], "active_st")
    active_en = _i64(inits, [M, M], "active_en")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, cell0_st, cell0_en, axes_hw], ["c0"]),
            helper.make_node("Slice", [IN_NAME, cell1_st, cell1_en, axes_hw], ["c1"]),
            helper.make_node("Slice", [IN_NAME, cell2_st, cell2_en, axes_hw], ["c2"]),
            helper.make_node("ArgMax", ["c0"], ["idx0"], axis=1, keepdims=1),
            helper.make_node("ArgMax", ["c1"], ["idx1"], axis=1, keepdims=1),
            helper.make_node("ArgMax", ["c2"], ["idx2"], axis=1, keepdims=1),
            helper.make_node("Cast", ["idx0"], ["idx0_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["idx1"], ["idx1_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["idx2"], ["idx2_f"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["c0", "c2"], ["c0c2"]),
            helper.make_node("ReduceSum", ["c0c2"], ["same02"], axes=[1], keepdims=1),
            helper.make_node("Less", ["same02", half], ["is_period3"]),
            helper.make_node("Where", ["is_period3", m3_1, m2_1], ["mask1_f"]),
            helper.make_node("Where", ["is_period3", m3_2, zero_mask], ["mask2_f"]),
            helper.make_node("Greater", ["mask1_f", zero], ["mask1"]),
            helper.make_node("Greater", ["mask2_f", zero], ["mask2"]),
            helper.make_node("Where", ["mask2", "idx2_f", "idx0_f"], ["base_idx"]),
            helper.make_node("Where", ["mask1", "idx1_f", "base_idx"], ["color_idx"]),
            helper.make_node("ReduceMax", [IN_NAME], ["valid_f30"], axes=[1], keepdims=1),
            helper.make_node("Slice", ["valid_f30", active_st, active_en, axes_hw], ["valid_f"]),
            helper.make_node("Greater", ["valid_f", zero], ["valid"]),
            helper.make_node("Cast", ["color_idx"], ["color_idx_i"], to=TensorProto.INT32),
            helper.make_node("Equal", ["color_idx_i", palette], ["out_unmasked"]),
            helper.make_node("And", ["out_unmasked", "valid"], ["out_bool"]),
            helper.make_node("Cast", ["out_bool"], ["out20"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out20"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, N - M, N - M],
            ),
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


def _verify_examples(model: onnx.ModelProto, data: dict[str, Any]) -> tuple[int, int]:
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    )
    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(data.get(split, [])):
            input_tensor = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if input_tensor is None or expected is None:
                continue
            actual = session.run([OUT_NAME], {IN_NAME: input_tensor})[0]
            total += 1
            if not np.array_equal(actual > 0.0, expected > 0.0):
                h = len(example["output"])
                w = len(example["output"][0])
                predicted = _onehot_to_grid(actual, h, w)
                wanted = np.asarray(example["output"], dtype=np.int64)
                raise AssertionError(
                    f"{split}[{index}] mismatch\npredicted:\n{predicted}\nexpected:\n{wanted}"
                )
            passed += 1
    return passed, total


def main() -> None:
    data = _load_task()
    _validate_rule(data)
    model = build_onnx_model()
    passed, total = _verify_examples(model, data)
    onnx.save(model, BEST_PATH)

    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"correct: {passed}/{total}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
