"""ONNX solution for ARC task112: reflect red shapes around a green 2x2 center.

Task rule: preserve the input grid size and keep the green 2x2 square.  The
red pattern on one side of that square is copied across both symmetry axes of
the square, yielding the original red cells plus their left-right, top-bottom,
and 180-degree reflected counterparts.  If the green block top-left is
``(gr, gc)``, the reflected coordinates use the center between the four green
cells: ``(r, 2*gc+1-c)``, ``(2*gr+1-r, c)``, and
``(2*gr+1-r, 2*gc+1-c)``.

ONNX approach: locate the first green row/column from compact color row/column
counts, build int32 reflected row/column gather indices, gather the red mask
across each axis, and broadcast a red one-hot value over the original input.
All examples keep the reflected cells inside the provided grid rectangle.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task112"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_PATH = Path(__file__).resolve().parent / f"{TASK_ID}.onnx"

C = 10
H = W = 30
N = H * W
SHAPE = [1, C, H, W]
CH_SHAPE = [1, 1, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = arr.copy()
    green = np.argwhere(arr == 3)
    if len(green) != 4:
        raise ValueError("expected one green 2x2 block")
    gr, gc = map(int, green.min(axis=0))
    for r, c in np.argwhere(arr == 2):
        r = int(r)
        c = int(c)
        for tr, tc in (
            (r, 2 * gc + 1 - c),
            (2 * gr + 1 - r, c),
            (2 * gr + 1 - r, 2 * gc + 1 - c),
        ):
            if 0 <= tr < arr.shape[0] and 0 <= tc < arr.shape[1]:
                out[tr, tc] = 2
    return out


def _init(inits: list[onnx.TensorProto], values: Any, name: str, dtype: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(values, dtype=dtype), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], values: Any, name: str) -> str:
    return _init(inits, values, name, np.int64)


def _f32(inits: list[onnx.TensorProto], values: Any, name: str) -> str:
    return _init(inits, values, name, np.float32)


def _bool(inits: list[onnx.TensorProto], values: Any, name: str) -> str:
    return _init(inits, values, name, np.bool_)


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    idx2 = _i64(inits, np.array(2, dtype=np.int64), "idx2")
    idx3 = _i64(inits, np.array(3, dtype=np.int64), "idx3")
    red_onehot = _f32(
        inits,
        np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32).reshape(1, 10, 1, 1),
        "red_onehot",
    )
    coord_m1 = _init(inits, np.arange(-1, W - 1, dtype=np.int32), "coord_m1", np.int32)
    neg_one_i = _init(inits, np.array(-1, dtype=np.int32), "neg_one_i", np.int32)
    thirty_i = _init(inits, np.array(W, dtype=np.int32), "thirty_i", np.int32)

    def add(op: str, ins: list[str], outs: list[str], **attrs: Any) -> None:
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    add("Gather", [IN_NAME, idx2], ["red3"], axis=1)

    add("ReduceSum", [IN_NAME], ["rows_by_color"], axes=[0, 3], keepdims=0)
    add("Gather", ["rows_by_color", idx3], ["green_rows"], axis=0)
    add("ArgMax", ["green_rows"], ["g_row64"], axis=0, keepdims=0)
    add("ReduceSum", [IN_NAME], ["cols_by_color"], axes=[0, 2], keepdims=0)
    add("Gather", ["cols_by_color", idx3], ["green_cols"], axis=0)
    add("ArgMax", ["green_cols"], ["g_col64"], axis=0, keepdims=0)
    add("Cast", ["g_row64"], ["g_row"], to=TensorProto.INT32)
    add("Cast", ["g_col64"], ["g_col"], to=TensorProto.INT32)

    for axis_name, marker_name in (("row", "g_row"), ("col", "g_col")):
        add("Add", [marker_name, marker_name], [f"{axis_name}_twice"])
        add("Sub", [f"{axis_name}_twice", coord_m1], [f"{axis_name}_src"])
        add("Greater", [f"{axis_name}_src", neg_one_i], [f"{axis_name}_ge0"])
        add("Less", [f"{axis_name}_src", thirty_i], [f"{axis_name}_lt30"])
        add("And", [f"{axis_name}_ge0", f"{axis_name}_lt30"], [f"{axis_name}_valid"])
        add("Where", [f"{axis_name}_valid", f"{axis_name}_src", marker_name], [f"{axis_name}_safe"])

    add("Cast", ["red3"], ["red_mask"], to=TensorProto.BOOL)
    add("Gather", ["red_mask", "col_safe"], ["h_raw"], axis=2)
    add("Gather", ["red_mask", "row_safe"], ["v_raw"], axis=1)
    add("Gather", ["h_raw", "row_safe"], ["hv_raw"], axis=1)
    add("Or", ["red_mask", "h_raw"], ["red_or_h"])
    add("Or", ["v_raw", "hv_raw"], ["v_or_hv"])
    add("Or", ["red_or_h", "v_or_hv"], ["all_red_mask"])
    add("Where", ["all_red_mask", red_onehot, IN_NAME], [OUT_NAME])

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


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            ref = solve(ex["input"])
            expected_grid = np.asarray(ex["output"], dtype=np.int64)
            if not np.array_equal(ref, expected_grid):
                print(f"reference mismatch: {split}[{idx}]")
                bad += 1
                continue

            expected = _grid_to_onehot(ex["output"]) > 0.0
            pred = _run_onnx(model, _grid_to_onehot(ex["input"])) > 0.0
            if not np.array_equal(pred, expected):
                print(f"onnx mismatch: {split}[{idx}]")
                bad += 1
    return bad


def main() -> None:
    model = build_model()
    bad = validate_json(model)
    assert bad == 0, f"{bad} JSON examples failed"
    onnx.save(model, OUT_PATH)
    print(score_file(OUT_PATH))


if __name__ == "__main__":
    main()
