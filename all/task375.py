"""ONNX for NeuroGolf task375: turn a centered odd square into diagonal black lines.

Task rule: each input is an odd N x N square in the top-left corner, with one
black center cell and every other in-grid cell sharing a single non-black color.
The output keeps the same N and foreground color, but makes every cell on either
main diagonal black; all remaining in-grid cells are the foreground color.

ONNX: infer N as the square root of the active one-hot cell count, build the
two diagonal masks inside a 15x15 crop, infer the foreground channel from the
top-left cell, then cast and pad the crop as the final output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import print_report, score_file  # noqa: E402

TASK_NUM = "375"
TASK_ID = f"task{TASK_NUM}"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
C = 10
H = W = 30
K = 15
SHAPE = [1, C, H, W]
IR_VERSION = 10
OPSET = 10


def _init(inits: list[onnx.TensorProto], values: Any, dtype: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(values, dtype=dtype), name))
    return name


def _i64(inits: list[onnx.TensorProto], values: Sequence[int], name: str) -> str:
    return _init(inits, values, np.int64, name)


def _i32(inits: list[onnx.TensorProto], values: Any, name: str) -> str:
    return _init(inits, values, np.int32, name)


def _f32(inits: list[onnx.TensorProto], values: Any, name: str) -> str:
    return _init(inits, values, np.float32, name)


def _bool(inits: list[onnx.TensorProto], values: Any, name: str) -> str:
    return _init(inits, values, np.bool_, name)


def solve(grid: Sequence[Sequence[int]]) -> np.ndarray:
    """Reference solver: blacken both diagonals of the odd square."""
    g = np.asarray(grid, dtype=np.int64)
    n = g.shape[0]
    if g.ndim != 2 or n != g.shape[1] or n % 2 != 1:
        raise ValueError(f"expected odd square, got {g.shape}")
    foreground = int(g[g != 0][0]) if np.any(g != 0) else 0
    out = np.full_like(g, foreground)
    rows, cols = np.indices(g.shape)
    out[(rows == cols) | (rows + cols == n - 1)] = 0
    return out


def _grid_to_onehot(grid: Sequence[Sequence[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _expected_onehot(grid: Sequence[Sequence[int]]) -> np.ndarray:
    return _grid_to_onehot(grid)


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    rows = np.arange(K, dtype=np.int32).reshape(1, 1, K, 1)
    cols = np.arange(K, dtype=np.int32).reshape(1, 1, 1, K)
    one_i = _i32(inits, 1, "one_i")
    row_idx = _i32(inits, rows, "row_idx")
    col_idx = _i32(inits, cols, "col_idx")
    main_diag = rows == cols
    main = _bool(inits, main_diag, "main")
    fg_start = _i64(inits, [0, 1, 0, 0], "fg_start")
    fg_end = _i64(inits, [1, C, 1, 1], "fg_end")

    nodes.extend(
        [
            helper.make_node("ReduceSum", [IN_NAME], ["area"], axes=[0, 1, 2, 3], keepdims=0),
            helper.make_node("Sqrt", ["area"], ["n_f"]),
            helper.make_node("Cast", ["n_f"], ["n_i"], to=TensorProto.INT32),
            helper.make_node("Sub", ["n_i", one_i], ["last_i"]),
            helper.make_node("Less", [row_idx, "n_i"], ["row_in"]),
            helper.make_node("Less", [col_idx, "n_i"], ["col_in"]),
            helper.make_node("And", ["row_in", "col_in"], ["inside"]),
            helper.make_node("And", [main, "row_in"], ["main_in"]),
            helper.make_node("Sub", ["last_i", row_idx], ["anti_col"]),
            helper.make_node("Equal", [col_idx, "anti_col"], ["anti"]),
            helper.make_node("Or", ["main_in", "anti"], ["diag"]),
            helper.make_node("Xor", ["inside", "diag"], ["fg_spatial"]),
            helper.make_node("Slice", [IN_NAME, fg_start, fg_end], ["fg_cell"]),
            helper.make_node("Cast", ["fg_cell"], ["fg_ch"], to=TensorProto.BOOL),
            helper.make_node("And", ["fg_ch", "fg_spatial"], ["fg_out"]),
            helper.make_node("Concat", ["diag", "fg_out"], ["crop_b"], axis=1),
            helper.make_node("Cast", ["crop_b"], ["crop_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["crop_f"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - K, W - K],
                value=0.0,
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
    onnx.checker.check_model(model, full_check=True)
    return model


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_reference() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(data.get(split, [])):
            expected = np.asarray(example["output"], dtype=np.int64)
            actual = solve(example["input"])
            if not np.array_equal(actual, expected):
                raise AssertionError(f"reference mismatch in {split}[{index}]")


def validate_model(model: onnx.ModelProto) -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(data.get(split, [])):
            inp = _grid_to_onehot(example["input"])
            expected = _expected_onehot(example["output"])
            actual = _run_onnx(model, inp)
            if not np.array_equal(actual > 0.0, expected > 0.0):
                raise AssertionError(f"ONNX mismatch in {split}[{index}]")


def main() -> None:
    validate_reference()
    model = build_model()
    validate_model(model)
    onnx.save(model, BEST_PATH)
    print_report(score_file(BEST_PATH))


if __name__ == "__main__":
    main()
