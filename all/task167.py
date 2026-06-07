"""Minimal ONNX for ARC task167: map color variety to a gray stencil.

Task rule: read the 3x3 input made from colors 2, 3, and 4. If the grid uses
one distinct color, output a gray top row; if it uses two distinct colors,
output a gray main diagonal; if it uses all three colors, output a gray
anti-diagonal. The rest of the 30x30 NeuroGolf output is black padding.

The ONNX graph counts which of channels 2/3/4 are present in the 3x3 core and
selects one of the three 3x3 foreground masks before padding to the required
one-hot output tensor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_all_onnx import verify_correctness  # noqa: E402
from score_model import score_file  # noqa: E402

TASK_ID = "task167"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
CORE = 3
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _f32(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name))
    return name


def _u8(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.uint8), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Iterable[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(list(vals), dtype=np.int64), name=name))
    return name


def _pattern(kind: str) -> np.ndarray:
    grid = np.zeros((CORE, CORE), dtype=np.uint8)
    if kind == "main":
        np.fill_diagonal(grid, 1)
    elif kind == "anti":
        np.fill_diagonal(np.fliplr(grid), 1)
    elif kind == "row":
        grid[0, :] = 1
    else:
        raise ValueError(kind)
    return grid.reshape(1, 1, CORE, CORE)


def solve_reference(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    distinct = len(set(np.asarray(grid, dtype=np.int64)[:CORE, :CORE].reshape(-1).tolist()))
    if distinct == 1:
        return np.array([[5, 5, 5], [0, 0, 0], [0, 0, 0]], dtype=np.int64)
    if distinct == 3:
        return np.array([[0, 0, 5], [0, 5, 0], [5, 0, 0]], dtype=np.int64)
    return np.eye(CORE, dtype=np.int64) * 5


def build_onnx_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, [0, 1, 2, 3], "axes")
    starts = _i64(inits, [0, 2, 0, 0], "starts")
    ends = _i64(inits, [1, 4, CORE, CORE], "ends")
    one = _f32(inits, [1.0], "one")
    zero = _f32(inits, [0.0], "zero")
    eight = _f32(inits, [float(CORE * CORE - 1)], "eight")
    nine = _f32(inits, [float(CORE * CORE)], "nine")
    idx0 = _i64(inits, [0], "idx0")
    idx1 = _i64(inits, [1], "idx1")
    zero4 = _f32(inits, np.zeros((1, 4, CORE, CORE), dtype=np.float32), "zero4")
    main_pat = _u8(inits, _pattern("main"), "main_pat")
    anti_pat = _u8(inits, _pattern("anti"), "anti_pat")
    row_pat = _u8(inits, _pattern("row"), "row_pat")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts, ends, axes], ["core23"]),
            helper.make_node("ReduceSum", ["core23"], ["counts23"], axes=[2, 3], keepdims=0),
            helper.make_node("ReduceSum", ["counts23"], ["cells23"], axes=[1], keepdims=0),
            helper.make_node("Greater", ["counts23", zero], ["present23"]),
            helper.make_node("Gather", ["present23", idx0], ["present2"], axis=1),
            helper.make_node("Gather", ["present23", idx1], ["present3"], axis=1),
            helper.make_node("And", ["present2", "present3"], ["present2_and_3"]),
            helper.make_node("Less", ["cells23", nine], ["present4"]),
            helper.make_node("And", ["present2_and_3", "present4"], ["is_anti"]),
            helper.make_node("Greater", ["counts23", eight], ["full23"]),
            helper.make_node("Gather", ["full23", idx0], ["all2"], axis=1),
            helper.make_node("Gather", ["full23", idx1], ["all3"], axis=1),
            helper.make_node("Or", ["all2", "all3"], ["all2_or_3"]),
            helper.make_node("Less", ["cells23", one], ["all4"]),
            helper.make_node("Or", ["all2_or_3", "all4"], ["is_row"]),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Where", ["is_anti", anti_pat, main_pat], ["diag_fg"]),
            helper.make_node("Where", ["is_row", row_pat, "diag_fg"], ["fg"]),
            helper.make_node("Cast", ["fg"], ["fg_f"], to=TensorProto.FLOAT),
            helper.make_node("Sub", [one, "fg_f"], ["bg"]),
            helper.make_node("Concat", ["bg", zero4, "fg_f"], ["out3"], axis=1),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 4, H - CORE, W - CORE]),
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


def _onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros((1, C, H, W), dtype=np.float32)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def main() -> None:
    model = build_onnx_model()
    onnx.save(model, str(BEST_PATH))

    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    session = ort.InferenceSession(str(BEST_PATH), providers=["CPUExecutionProvider"])
    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data.get(split, []):
            total += 1
            expected = _onehot(ex["output"])
            actual = session.run([OUT_NAME], {IN_NAME: _onehot(ex["input"])})[0]
            if np.array_equal(actual > 0.0, expected > 0.0):
                passed += 1

    ok, summary, _, _ = verify_correctness(BEST_PATH)
    result = score_file(BEST_PATH)

    print(f"wrote {BEST_PATH}")
    print(f"examples: {passed}/{total}")
    print(f"one-hot verify: {summary} {'PASS' if ok else 'FAIL'}")
    print(f"params: {result.get('params')}")
    print(f"memory: {result.get('memory')}")
    print(f"cost: {result.get('cost')}")
    score = result.get("score")
    print(f"score: {score:.6f}" if score is not None else "score: INVALID")


if __name__ == "__main__":
    main()
