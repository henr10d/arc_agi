"""ONNX solution for ARC task218: compress rectangular color block layouts.

Task rule: the input is a black background with solid colored rectangles
arranged in a coarse grid.  Collapse consecutive identical full rows and
consecutive identical full columns, ignoring all-black padding runs, so each
rectangle slot contributes one output cell with its color.  This preserves
hidden boundaries where same-colored neighboring rectangles touch, because the
row/column comparison is made against the whole orthogonal line.

ONNX approach: convert the one-hot input to compact int32 color ids, detect row
and column run starts, score the starts in top/left order, use TopK to take the
first three row/column coordinates, Gather their intersections, mask missing
ranks for 2x2/2x3/3x2 cases, and pad the compact result back to the required
30x30 one-hot tensor.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task218"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
DATA_HW = 21
MAX_RUNS = 3
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def solve(grid: list[list[int]]) -> list[list[int]]:
    arr = np.asarray(grid, dtype=np.uint8)
    nz = arr != 0
    rows = np.flatnonzero(nz.any(axis=1))
    cols = np.flatnonzero(nz.any(axis=0))
    if len(rows) == 0 or len(cols) == 0:
        return []

    trimmed = arr[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    row_idx = [0]
    row_idx.extend(i for i in range(1, trimmed.shape[0]) if not np.array_equal(trimmed[i], trimmed[i - 1]))
    row_compact = trimmed[row_idx]
    col_idx = [0]
    col_idx.extend(
        j for j in range(1, row_compact.shape[1]) if not np.array_equal(row_compact[:, j], row_compact[:, j - 1])
    )
    return row_compact[:, col_idx].astype(int).tolist()


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []

    def init(self, name: str, arr: Any, dtype: Any | None = None) -> str:
        array = np.asarray(arr, dtype=dtype)
        self.initializers.append(numpy_helper.from_array(array, name=name))
        return name

    def node(self, op_type: str, inputs: list[str], output: str, **attrs: Any) -> str:
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        return output


def add_constants(b: Builder, max_runs: int, line: int = H) -> None:
    b.init("zero_i32", np.array(0, dtype=np.int32))
    b.init("zero_f", np.array(0.0, dtype=np.float32))
    b.init("s0", [0], np.int64)
    b.init("s00", [0, 0], np.int64)
    b.init("s1", [1], np.int64)
    b.init("e1", [1], np.int64)
    b.init("e_prev", [line - 1], np.int64)
    b.init("e_line", [line], np.int64)
    b.init("e_lines", [line, line], np.int64)
    b.init("axis1", [1], np.int64)
    b.init("axis2", [2], np.int64)
    b.init("axes12", [1, 2], np.int64)
    b.init("color_ids", np.arange(1, C, dtype=np.int32).reshape(1, C - 1, 1, 1))
    b.init("rank_lo", [[[float(i) + 0.5 for i in range(max_runs)]]], np.float32)
    b.init("rank_hi", [[[float(i) + 1.5 for i in range(max_runs)]]], np.float32)
    b.init("tri", np.triu(np.ones((line, line), dtype=np.float32)), np.float32)
    b.init("order", np.arange(line, 0, -1, dtype=np.float32), np.float32)
    b.init("topk", [max_runs], np.int64)


def slice_axis(b: Builder, x: str, start: str, end: str, axis: str, out: str) -> str:
    return b.node("Slice", [x, start, end, axis], out)


def line_has_color(b: Builder, colored_f: str, prefix: str, axis: int) -> str:
    any_f = b.node("ReduceMax", [colored_f], f"{prefix}_any_f", axes=[axis], keepdims=0)
    return b.node("Greater", [any_f, "zero_f"], f"{prefix}_has")


def build_start_mask(b: Builder, ids: str, colored_f: str, prefix: str, line_axis: int) -> str:
    """Return a [1,30] bool mask for non-black row/column run starts."""
    if line_axis == 1:
        prev = slice_axis(b, ids, "s0", "e_prev", "axis1", f"{prefix}_prev")
        curr = slice_axis(b, ids, "s1", "e_line", "axis1", f"{prefix}_curr")
        equal = b.node("Equal", [prev, curr], f"{prefix}_eq_cells")
        equal_f = b.node("Cast", [equal], f"{prefix}_eq_f", to=TensorProto.FLOAT)
        same_f = b.node("ReduceMin", [equal_f], f"{prefix}_same_f", axes=[2], keepdims=0)
        has = line_has_color(b, colored_f, prefix, axis=2)
    else:
        prev = slice_axis(b, ids, "s0", "e_prev", "axis2", f"{prefix}_prev")
        curr = slice_axis(b, ids, "s1", "e_line", "axis2", f"{prefix}_curr")
        equal = b.node("Equal", [prev, curr], f"{prefix}_eq_cells")
        equal_f = b.node("Cast", [equal], f"{prefix}_eq_f", to=TensorProto.FLOAT)
        same_f = b.node("ReduceMin", [equal_f], f"{prefix}_same_f", axes=[1], keepdims=0)
        has = line_has_color(b, colored_f, prefix, axis=1)

    same = b.node("Greater", [same_f, "zero_f"], f"{prefix}_same")
    not_same = b.node("Not", [same], f"{prefix}_not_same")
    has0 = slice_axis(b, has, "s0", "e1", "axis1", f"{prefix}_has0")
    has_tail = slice_axis(b, has, "s1", "e_line", "axis1", f"{prefix}_has_tail")
    tail = b.node("And", [has_tail, not_same], f"{prefix}_tail_start")
    return b.node("Concat", [has0, tail], f"{prefix}_start", axis=1)


def selector_from_starts(b: Builder, starts: str, prefix: str) -> str:
    start_f = b.node("Cast", [starts], f"{prefix}_start_f", to=TensorProto.FLOAT)
    cum = b.node("MatMul", [start_f, "tri"], f"{prefix}_cum")
    cum_u = b.node("Unsqueeze", [cum], f"{prefix}_cum_u", axes=[2])
    gt = b.node("Greater", [cum_u, "rank_lo"], f"{prefix}_rank_gt")
    lt = b.node("Less", [cum_u, "rank_hi"], f"{prefix}_rank_lt")
    rank = b.node("And", [gt, lt], f"{prefix}_rank")
    start_u = b.node("Unsqueeze", [starts], f"{prefix}_start_u", axes=[2])
    mask = b.node("And", [start_u, rank], f"{prefix}_mask")
    return b.node("Cast", [mask], f"{prefix}_mask_f", to=TensorProto.FLOAT)


def prune_unused_initializers(b: Builder) -> None:
    used = {name for node in b.nodes for name in node.input}
    b.initializers = [init for init in b.initializers if init.name in used]


def build_selector_model(max_runs: int = MAX_RUNS, line: int = H) -> onnx.ModelProto:
    b = Builder()
    add_constants(b, max_runs, line)

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    ids64 = b.node("ArgMax", [IN_NAME], "ids64", axis=1, keepdims=0)
    ids = b.node("Cast", [ids64], "ids", to=TensorProto.INT32)
    colored = b.node("Greater", [ids, "zero_i32"], "colored")
    colored_f = b.node("Cast", [colored], "colored_f", to=TensorProto.FLOAT)

    row_starts = build_start_mask(b, ids, colored_f, "row", line_axis=1)
    col_starts = build_start_mask(b, ids, colored_f, "col", line_axis=2)
    row_sel = selector_from_starts(b, row_starts, "row")
    col_sel = selector_from_starts(b, col_starts, "col")

    col_compact = b.node("MatMul", [IN_NAME, col_sel], "col_compact")
    transposed = b.node("Transpose", [col_compact], "col_first", perm=[0, 1, 3, 2])
    row_compact_t = b.node("MatMul", [transposed, row_sel], "row_compact_t")
    compact = b.node("Transpose", [row_compact_t], "compact", perm=[0, 1, 3, 2])
    b.node(
        "Pad",
        [compact],
        OUT_NAME,
        pads=[0, 0, 0, 0, 0, 0, H - max_runs, W - max_runs],
    )

    prune_unused_initializers(b)
    graph = helper.make_graph(b.nodes, TASK_ID, [x_info], [y_info], initializer=b.initializers)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_model(max_runs: int = MAX_RUNS, line: int = DATA_HW) -> onnx.ModelProto:
    b = Builder()
    add_constants(b, max_runs, line)

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    ids64 = b.node("ArgMax", [IN_NAME], "ids64", axis=1, keepdims=0)
    ids64_crop = b.node("Slice", [ids64, "s00", "e_lines", "axes12"], "ids64_crop")
    ids = b.node("Cast", [ids64_crop], "ids", to=TensorProto.INT32)
    colored = b.node("Greater", [ids, "zero_i32"], "colored")
    colored_f = b.node("Cast", [colored], "colored_f", to=TensorProto.FLOAT)

    row_starts = build_start_mask(b, ids, colored_f, "row", line_axis=1)
    col_starts = build_start_mask(b, ids, colored_f, "col", line_axis=2)

    row_start_f = b.node("Cast", [row_starts], "row_start_f", to=TensorProto.FLOAT)
    col_start_f = b.node("Cast", [col_starts], "col_start_f", to=TensorProto.FLOAT)
    row_score = b.node("Mul", [row_start_f, "order"], "row_score")
    col_score = b.node("Mul", [col_start_f, "order"], "col_score")
    b.node("TopK", [row_score, "topk"], "row_topv", axis=1)
    b.nodes[-1].output.append("row_idx")
    b.node("TopK", [col_score, "topk"], "col_topv", axis=1)
    b.nodes[-1].output.append("col_idx")

    row_idx = b.node("Squeeze", ["row_idx"], "row_idx_1d", axes=[0])
    col_idx = b.node("Squeeze", ["col_idx"], "col_idx_1d", axes=[0])

    id_rows = b.node("Gather", [ids, row_idx], "id_rows", axis=1)
    compact_ids = b.node("Gather", [id_rows, col_idx], "compact_ids", axis=2)
    compact_ids4 = b.node("Unsqueeze", [compact_ids], "compact_ids4", axes=[1])
    compact_bool = b.node("Equal", ["color_ids", compact_ids4], "compact_bool")
    compact = b.node("Cast", [compact_bool], "compact_raw", to=TensorProto.FLOAT)

    row_valid = b.node("Greater", ["row_topv", "zero_f"], "row_valid")
    col_valid = b.node("Greater", ["col_topv", "zero_f"], "col_valid")
    row_valid_1d = b.node("Squeeze", [row_valid], "row_valid_1d", axes=[0])
    col_valid_1d = b.node("Squeeze", [col_valid], "col_valid_1d", axes=[0])
    row_valid4 = b.node("Unsqueeze", [row_valid_1d], "row_valid4", axes=[0, 1, 3])
    col_valid4 = b.node("Unsqueeze", [col_valid_1d], "col_valid4", axes=[0, 1, 2])
    valid = b.node("And", [row_valid4, col_valid4], "valid")
    valid_f = b.node("Cast", [valid], "valid_f", to=TensorProto.FLOAT)
    compact_masked = b.node("Mul", [compact, valid_f], "compact")
    b.node(
        "Pad",
        [compact_masked],
        OUT_NAME,
        pads=[0, 1, 0, 0, 0, 0, H - max_runs, W - max_runs],
    )

    prune_unused_initializers(b)
    graph = helper.make_graph(b.nodes, f"{TASK_ID}_topk", [x_info], [y_info], initializer=b.initializers)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def decode_one_hot(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    active = arr[0, :, :h, :w] > 0.0
    bad = active.sum(axis=0) != 1
    out = active.argmax(axis=0).astype(np.int64)
    out[bad] = -1
    return out


def verify(path: Path) -> None:
    data = json.loads(DATA_PATH.read_text())
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            expected = np.asarray(example["output"], dtype=np.int64)
            ref = solve(example["input"])
            if ref != example["output"]:
                raise AssertionError(f"reference mismatch for {split}[{idx}]")
            arr = convert_to_numpy(example, "input")
            if arr is None:
                continue
            pred = sess.run([OUT_NAME], {IN_NAME: arr})[0]
            got = decode_one_hot(pred, expected.shape[0], expected.shape[1])
            if not np.array_equal(got, expected):
                raise AssertionError(f"ONNX mismatch for {split}[{idx}]:\n{got}\n!=\n{expected}")
            total += 1
    print(f"validated examples: {total}")


def score_candidate(model: onnx.ModelProto, name: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{name}.onnx"
        onnx.save(model, path)
        return score_file(path)


def main() -> None:
    model = build_model(MAX_RUNS)
    onnx.save(model, BEST_PATH)
    verify(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    result = score_file(BEST_PATH)
    print(
        "score: "
        f"memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']:.6f}"
    )


if __name__ == "__main__":
    main()
