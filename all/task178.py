"""ONNX solution for ARC task178: collapse adjacent equal color runs.

Task rule: the input grid is a one-dimensional sequence expanded by repeating
identical rows or identical columns. Compress consecutive duplicate rows and
columns, preserving the first cell of each run. The result is therefore either
a 1xN row from the top input row or an Nx1 column from the left input column.
The provided task data uses colors 1-9, at most 13 columns, at most 14 rows,
and at most five compressed runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TASK_ID = "task178"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task178.onnx"
DATA_PATH = ROOT / "data" / "task178.json"

C = 10
H = W = 30
MAX_RUNS = 5
MAX_COLS = 13
MAX_ROWS = 14
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = [row for i, row in enumerate(grid) if i == 0 or row != grid[i - 1]]
    cols = [
        j
        for j in range(len(rows[0]))
        if j == 0 or any(row[j] != row[j - 1] for row in rows)
    ]
    return [[row[j] for j in cols] for row in rows]


def _init(inits: list[onnx.TensorProto], name: str, arr: Iterable[int] | np.ndarray, dtype=np.int64) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=dtype), name=name))
    return name


def _slice(nodes: list[onnx.NodeProto], x: str, y: str, starts: str, ends: str, axes: str) -> None:
    nodes.append(helper.make_node("Slice", [x, starts, ends, axes], [y]))


def _build_axis_runs(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    prefix: str,
    line: str,
    length: int,
) -> str:
    """Return a [1,10,5] one-hot matrix of the first five run colors."""
    nodes.append(helper.make_node("ArgMax", [line], [f"{prefix}_color64"], axis=1, keepdims=0))
    nodes.append(helper.make_node("Cast", [f"{prefix}_color64"], [f"{prefix}_color"], to=TensorProto.INT32))
    nodes.append(helper.make_node("Greater", [f"{prefix}_color", "zero_i"], [f"{prefix}_active"]))

    end_last = _init(inits, f"{prefix}_end_last", [1, length - 1])
    end_full = _init(inits, f"{prefix}_end_full", [1, length])
    tri = _init(inits, f"{prefix}_tri", np.triu(np.ones((length, length), dtype=np.float32)), np.float32)

    _slice(nodes, f"{prefix}_active", f"{prefix}_active0", "start0", "end1", "ax12")
    _slice(nodes, f"{prefix}_color", f"{prefix}_prev", "start0", end_last, "ax12")
    _slice(nodes, f"{prefix}_color", f"{prefix}_curr", "start1", end_full, "ax12")
    _slice(nodes, f"{prefix}_active", f"{prefix}_active_tail", "start1", end_full, "ax12")
    nodes.append(helper.make_node("Equal", [f"{prefix}_prev", f"{prefix}_curr"], [f"{prefix}_same"]))
    nodes.append(helper.make_node("Not", [f"{prefix}_same"], [f"{prefix}_diff"]))
    nodes.append(helper.make_node("And", [f"{prefix}_active_tail", f"{prefix}_diff"], [f"{prefix}_tail_start"]))
    nodes.append(helper.make_node("Concat", [f"{prefix}_active0", f"{prefix}_tail_start"], [f"{prefix}_start"], axis=1))

    nodes.append(helper.make_node("Cast", [f"{prefix}_start"], [f"{prefix}_start_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("MatMul", [f"{prefix}_start_f", tri], [f"{prefix}_cum"]))

    nodes.append(helper.make_node("Unsqueeze", [f"{prefix}_cum"], [f"{prefix}_cum_u"], axes=[2]))
    nodes.append(helper.make_node("Greater", [f"{prefix}_cum_u", "rank_lo"], [f"{prefix}_gt"]))
    nodes.append(helper.make_node("Less", [f"{prefix}_cum_u", "rank_hi"], [f"{prefix}_lt"]))
    nodes.append(helper.make_node("And", [f"{prefix}_gt", f"{prefix}_lt"], [f"{prefix}_rank"]))
    nodes.append(helper.make_node("Unsqueeze", [f"{prefix}_start"], [f"{prefix}_start_u"], axes=[2]))
    nodes.append(helper.make_node("And", [f"{prefix}_start_u", f"{prefix}_rank"], [f"{prefix}_mask"]))
    nodes.append(helper.make_node("Cast", [f"{prefix}_mask"], [f"{prefix}_mask_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("MatMul", [line, f"{prefix}_mask_f"], [f"{prefix}_compact3"]))
    return f"{prefix}_compact3"


def _build_horizontal_runs_4d(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    line4: str,
    length: int,
) -> str:
    """Return a [1,10,1,5] one-hot row without materializing [1,10,13]."""
    prefix = "h"
    nodes.append(helper.make_node("ArgMax", [line4], [f"{prefix}_color64_4"], axis=1, keepdims=0))
    nodes.append(helper.make_node("Squeeze", [f"{prefix}_color64_4"], [f"{prefix}_color64"], axes=[1]))
    nodes.append(helper.make_node("Cast", [f"{prefix}_color64"], [f"{prefix}_color"], to=TensorProto.INT32))
    nodes.append(helper.make_node("Greater", [f"{prefix}_color", "zero_i"], [f"{prefix}_active"]))

    end_last = _init(inits, f"{prefix}_end_last", [1, length - 1])
    end_full = _init(inits, f"{prefix}_end_full", [1, length])
    tri = _init(inits, f"{prefix}_tri", np.triu(np.ones((length, length), dtype=np.float32)), np.float32)

    _slice(nodes, f"{prefix}_active", f"{prefix}_active0", "start0", "end1", "ax12")
    _slice(nodes, f"{prefix}_color", f"{prefix}_prev", "start0", end_last, "ax12")
    _slice(nodes, f"{prefix}_color", f"{prefix}_curr", "start1", end_full, "ax12")
    _slice(nodes, f"{prefix}_active", f"{prefix}_active_tail", "start1", end_full, "ax12")
    nodes.append(helper.make_node("Equal", [f"{prefix}_prev", f"{prefix}_curr"], [f"{prefix}_same"]))
    nodes.append(helper.make_node("Not", [f"{prefix}_same"], [f"{prefix}_diff"]))
    nodes.append(helper.make_node("And", [f"{prefix}_active_tail", f"{prefix}_diff"], [f"{prefix}_tail_start"]))
    nodes.append(helper.make_node("Concat", [f"{prefix}_active0", f"{prefix}_tail_start"], [f"{prefix}_start"], axis=1))

    nodes.append(helper.make_node("Cast", [f"{prefix}_start"], [f"{prefix}_start_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("MatMul", [f"{prefix}_start_f", tri], [f"{prefix}_cum"]))
    nodes.append(helper.make_node("Unsqueeze", [f"{prefix}_cum"], [f"{prefix}_cum_u"], axes=[2]))
    nodes.append(helper.make_node("Greater", [f"{prefix}_cum_u", "rank_lo"], [f"{prefix}_gt"]))
    nodes.append(helper.make_node("Less", [f"{prefix}_cum_u", "rank_hi"], [f"{prefix}_lt"]))
    nodes.append(helper.make_node("And", [f"{prefix}_gt", f"{prefix}_lt"], [f"{prefix}_rank"]))
    nodes.append(helper.make_node("Unsqueeze", [f"{prefix}_start"], [f"{prefix}_start_u"], axes=[2]))
    nodes.append(helper.make_node("And", [f"{prefix}_start_u", f"{prefix}_rank"], [f"{prefix}_mask"]))
    nodes.append(helper.make_node("Cast", [f"{prefix}_mask"], [f"{prefix}_mask_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Unsqueeze", [f"{prefix}_mask_f"], [f"{prefix}_mask4"], axes=[1]))
    nodes.append(helper.make_node("MatMul", [line4, f"{prefix}_mask4"], [f"{prefix}_compact"]))
    return f"{prefix}_compact"


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    _init(inits, "zero_i", np.asarray(0, dtype=np.int32), np.int32)
    _init(inits, "start0", [0, 0])
    _init(inits, "start1", [0, 1])
    _init(inits, "end1", [1, 1])
    _init(inits, "ax12", [0, 1])
    _init(inits, "top_left_start", [0, 0])
    _init(inits, "top_end", [1, MAX_COLS])
    _init(inits, "left_end", [MAX_ROWS, 1])
    _init(inits, "grid_axes", [2, 3])
    _init(inits, "rank_lo", [[[float(k) + 0.5 for k in range(MAX_RUNS)]]], np.float32)
    _init(inits, "rank_hi", [[[float(k) + 1.5 for k in range(MAX_RUNS)]]], np.float32)

    _slice(nodes, IN_NAME, "top4", "top_left_start", "top_end", "grid_axes")
    _slice(nodes, IN_NAME, "left4", "top_left_start", "left_end", "grid_axes")
    nodes.append(helper.make_node("Squeeze", ["left4"], ["left"], axes=[3]))

    h_compact = _build_horizontal_runs_4d(nodes, inits, "top4", MAX_COLS)
    v_compact3 = _build_axis_runs(nodes, inits, "v", "left", MAX_ROWS)

    nodes.append(helper.make_node("Unsqueeze", [v_compact3], ["v_compact"], axes=[3]))
    nodes.append(
        helper.make_node(
            "Pad",
            ["h_compact"],
            ["h_5x5"],
            pads=[0, 0, 0, 0, 0, 0, MAX_RUNS - 1, 0],
        )
    )
    nodes.append(
        helper.make_node(
            "Pad",
            ["v_compact"],
            ["v_5x5"],
            pads=[0, 0, 0, 0, 0, 0, 0, MAX_RUNS - 1],
        )
    )
    nodes.append(helper.make_node("Add", ["h_5x5", "v_5x5"], ["selected"]))
    nodes.append(
        helper.make_node(
            "Pad",
            ["selected"],
            [OUT_NAME],
            pads=[0, 0, 0, 0, 0, 0, H - MAX_RUNS, W - MAX_RUNS],
        )
    )

    graph = helper.make_graph(nodes, "task178", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, color, r, c] = 1.0
    return out


def verify(path: Path) -> None:
    data = json.loads(DATA_PATH.read_text())
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for split in ("train", "test", "arc-gen"):
        for i, ex in enumerate(data[split]):
            got = sess.run(None, {IN_NAME: _onehot(ex["input"])})[0] > 0
            want = _onehot(ex["output"]) > 0
            if not np.array_equal(got, want):
                raise AssertionError(f"{split} example {i} failed")


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    verify(BEST_PATH)
    print(f"wrote {BEST_PATH}")


if __name__ == "__main__":
    main()
