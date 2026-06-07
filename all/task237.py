"""Minimal ONNX for NeuroGolf task237.

Task rule: the visible input grid is at most 9x9 and contains isolated
non-black seed pixels.  Each seed at (r, c) with color k draws a same-colored
path to the right edge: a horizontal arm from (r, c) through column W-1 and a
vertical arm down from (r, W-1) until the row immediately before the next seed,
or to the grid bottom for the final seed.  The background remains black inside
the visible grid and padded cells outside the visible grid remain all-zero.
All provided examples have at least one black cell in every visible row and
column, so the compact ONNX graph infers the visible bounds from color 0.

ONNX approach: slice the top-left 9x9 area, infer the visible height and width
from one-hot activity, use small coordinate broadcasts to form the horizontal
and right-edge vertical masks for colors 1..9, then cast and pad the compact
9x9 one-hot result directly to the competition output.
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

from score_model import convert_to_numpy, print_report, score_file  # noqa: E402


TASK_ID = "task237"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
C = 10
H = W = 30
N = 9
PAD = H - N
SHAPE = [1, C, H, W]
IR_VERSION = 10
OPSET = 10


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.value_infos: list[onnx.ValueInfoProto] = []
        self.counter = 0

    def init(self, name: str, value: Any, dtype: np.dtype[Any]) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def node(
        self,
        op_type: str,
        inputs: list[str],
        dtype: int,
        shape: tuple[int, ...],
        prefix: str,
        **attrs: Any,
    ) -> str:
        self.counter += 1
        out = f"{prefix}_{self.counter}"
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        self.value_infos.append(helper.make_tensor_value_info(out, dtype, list(shape)))
        return out


def _le(b: Builder, left: str, right: str, shape: tuple[int, ...], prefix: str) -> str:
    lt = b.node("Less", [left, right], TensorProto.BOOL, shape, f"{prefix}_lt")
    eq = b.node("Equal", [left, right], TensorProto.BOOL, shape, f"{prefix}_eq")
    return b.node("Or", [lt, eq], TensorProto.BOOL, shape, prefix)


def _positive_i32(b: Builder, name: str, shape: tuple[int, ...], prefix: str) -> str:
    return b.node("Greater", [name, "zero_i"], TensorProto.BOOL, shape, prefix)


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference implementation used to verify the inferred JSON rule."""
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(arr)
    height, width = arr.shape
    seeds = [(int(r), int(c), int(arr[r, c])) for r, c in zip(*np.nonzero(arr))]
    for idx, (r, c, color) in enumerate(seeds):
        next_row = seeds[idx + 1][0] if idx + 1 < len(seeds) else height
        out[r, c:width] = color
        out[r:next_row, width - 1] = color
    return out


def load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def verify_reference() -> None:
    failures: list[str] = []
    for split, examples in load_task_data().items():
        for idx, example in enumerate(examples):
            expected = np.asarray(example["output"], dtype=np.int64)
            actual = solve(example["input"])
            if not np.array_equal(actual, expected):
                failures.append(f"{split}[{idx}]")
                if len(failures) >= 5:
                    raise AssertionError("reference failed: " + ", ".join(failures))
    if failures:
        raise AssertionError("reference failed: " + ", ".join(failures))


def build_model() -> onnx.ModelProto:
    b = Builder()
    b.init("starts", [0, 0, 0, 0], np.int64)
    b.init("bg_end", [1, 1, N, N], np.int64)
    b.init("axes4", [0, 1, 2, 3], np.int64)
    b.init("fg_start", [0, 1, 0, 0], np.int64)
    b.init("fg_end", [1, C, N, N], np.int64)
    b.init("one_i", [1], np.int32)
    b.init("zero_i", [0], np.int32)
    b.init("zero_f", [0.0], np.float32)
    b.init("rows", np.arange(N, dtype=np.int32).reshape(1, 1, N, 1), np.int32)
    b.init("cols", np.arange(N, dtype=np.int32).reshape(1, 1, 1, N), np.int32)
    b.init("h_out_cols_plus1", (np.arange(N, dtype=np.int32) + 1).reshape(1, 1, 1, N), np.int32)
    b.init("false_color_col", np.zeros((1, C - 1, 1), dtype=np.bool_), np.bool_)
    b.init("false_seen_col", np.zeros((1, 1, 1), dtype=np.bool_), np.bool_)
    for idx in range(N):
        b.init(f"row_idx_{idx}", [idx], np.int64)

    inp = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    out = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    bg0 = b.node("Slice", [IN_NAME, "starts", "bg_end", "axes4"], TensorProto.FLOAT, (1, 1, N, N), "bg0")
    fg = b.node("Slice", [IN_NAME, "fg_start", "fg_end", "axes4"], TensorProto.FLOAT, (1, C - 1, N, N), "fg")

    row_sum = b.node("ReduceSum", [bg0], TensorProto.FLOAT, (N,), "row_sum", axes=[0, 1, 3], keepdims=0)
    col_sum = b.node("ReduceSum", [bg0], TensorProto.FLOAT, (N,), "col_sum", axes=[0, 1, 2], keepdims=0)
    row_has = b.node("Greater", [row_sum, "zero_f"], TensorProto.BOOL, (N,), "row_has")
    col_has = b.node("Greater", [col_sum, "zero_f"], TensorProto.BOOL, (N,), "col_has")
    row_i = b.node("Cast", [row_has], TensorProto.INT32, (N,), "row_i", to=TensorProto.INT32)
    col_i = b.node("Cast", [col_has], TensorProto.INT32, (N,), "col_i", to=TensorProto.INT32)
    height = b.node("ReduceSum", [row_i], TensorProto.INT32, (1,), "height", axes=[0], keepdims=1)
    width = b.node("ReduceSum", [col_i], TensorProto.INT32, (1,), "width", axes=[0], keepdims=1)
    w_last = b.node("Sub", [width, "one_i"], TensorProto.INT32, (1,), "w_last")

    row_inside = b.node("Less", ["rows", height], TensorProto.BOOL, (1, 1, N, 1), "row_inside")
    col_inside = b.node("Less", ["cols", width], TensorProto.BOOL, (1, 1, 1, N), "col_inside")
    inside = b.node("And", [row_inside, col_inside], TensorProto.BOOL, (1, 1, N, N), "inside")

    color_seed_sum = b.node("ReduceSum", [fg], TensorProto.FLOAT, (1, C - 1, N), "color_seed_sum", axes=[3], keepdims=0)
    color_seed_row = b.node("Greater", [color_seed_sum, "zero_f"], TensorProto.BOOL, (1, C - 1, N), "color_seed_row")
    h_col = b.node("ArgMax", [fg], TensorProto.INT64, (1, C - 1, N), "h_col", axis=3, keepdims=0)
    h_col_i = b.node("Cast", [h_col], TensorProto.INT32, (1, C - 1, N), "h_col_i", to=TensorProto.INT32)
    h_col4 = b.node("Unsqueeze", [h_col_i], TensorProto.INT32, (1, C - 1, N, 1), "h_col4", axes=[3])
    h_has4 = b.node("Unsqueeze", [color_seed_row], TensorProto.BOOL, (1, C - 1, N, 1), "h_has4", axes=[3])
    h_from_seed = b.node("Less", [h_col4, "h_out_cols_plus1"], TensorProto.BOOL, (1, C - 1, N, N), "h_from_seed")
    h_mask = b.node("And", [h_has4, h_from_seed], TensorProto.BOOL, (1, C - 1, N, N), "h_mask")

    color_seed_i = b.node("Cast", [color_seed_row], TensorProto.INT32, (1, C - 1, N), "color_seed_i", to=TensorProto.INT32)
    h_col_masked = b.node("Mul", [h_col_i, color_seed_i], TensorProto.INT32, (1, C - 1, N), "h_col_masked")
    h_col_any = b.node("ReduceSum", [h_col_masked], TensorProto.INT32, (1, N), "h_col_any", axes=[1], keepdims=0)
    h_col_any4 = b.node("Unsqueeze", [h_col_any], TensorProto.INT32, (1, 1, N, 1), "h_col_any4", axes=[1, 3])
    seed_row_count = b.node("ReduceSum", [color_seed_i], TensorProto.INT32, (1, 1, N), "seed_row_count", axes=[1], keepdims=1)
    seed_row_any = _positive_i32(b, seed_row_count, (1, 1, N), "seed_row_any")
    seed_row_any4 = b.node("Unsqueeze", [seed_row_any], TensorProto.BOOL, (1, 1, N, 1), "seed_row_any4", axes=[3])
    h_occupied_cols = b.node(
        "Less", [h_col_any4, "h_out_cols_plus1"], TensorProto.BOOL, (1, 1, N, N), "h_occupied_cols"
    )
    h_occupied = b.node("And", [seed_row_any4, h_occupied_cols], TensorProto.BOOL, (1, 1, N, N), "h_occupied")
    latest_parts: list[str] = []
    seen_parts: list[str] = []
    prev_color = "false_color_col"
    prev_seen = "false_seen_col"
    for idx in range(N):
        color_at_row = b.node(
            "Gather", [color_seed_row, f"row_idx_{idx}"], TensorProto.BOOL, (1, C - 1, 1), f"color_at_row_{idx}", axis=2
        )
        seed_at_row = b.node(
            "Gather", [seed_row_any, f"row_idx_{idx}"], TensorProto.BOOL, (1, 1, 1), f"seed_at_row_{idx}", axis=2
        )
        seed_not_at_row = b.node("Not", [seed_at_row], TensorProto.BOOL, (1, 1, 1), f"seed_not_at_row_{idx}")
        current_color_at_row = b.node(
            "And", [seed_at_row, color_at_row], TensorProto.BOOL, (1, C - 1, 1), f"current_color_at_row_{idx}"
        )
        previous_color_at_row = b.node(
            "And", [seed_not_at_row, prev_color], TensorProto.BOOL, (1, C - 1, 1), f"previous_color_at_row_{idx}"
        )
        latest_at_row = b.node(
            "Or", [current_color_at_row, previous_color_at_row], TensorProto.BOOL, (1, C - 1, 1), f"latest_at_row_{idx}"
        )
        seen_at_row = b.node("Or", [prev_seen, seed_at_row], TensorProto.BOOL, (1, 1, 1), f"seen_at_row_{idx}")
        latest_parts.append(latest_at_row)
        seen_parts.append(seen_at_row)
        prev_color = latest_at_row
        prev_seen = seen_at_row
    v_rows = b.node("Concat", latest_parts, TensorProto.BOOL, (1, C - 1, N), "v_rows", axis=2)
    v_occupied_rows = b.node("Concat", seen_parts, TensorProto.BOOL, (1, 1, N), "v_occupied_rows", axis=2)
    v_rows4 = b.node("Unsqueeze", [v_rows], TensorProto.BOOL, (1, C - 1, N, 1), "v_rows4", axes=[3])
    right_col = b.node("Equal", ["cols", w_last], TensorProto.BOOL, (1, 1, 1, N), "right_col")
    v_mask = b.node("And", [v_rows4, right_col], TensorProto.BOOL, (1, C - 1, N, N), "v_mask")
    v_occupied_rows4 = b.node(
        "Unsqueeze", [v_occupied_rows], TensorProto.BOOL, (1, 1, N, 1), "v_occupied_rows4", axes=[3]
    )
    v_occupied = b.node("And", [v_occupied_rows4, right_col], TensorProto.BOOL, (1, 1, N, N), "v_occupied")

    fg_any_kind = b.node("Or", [h_mask, v_mask], TensorProto.BOOL, (1, C - 1, N, N), "fg_any_kind")
    fg_mask = b.node("And", [fg_any_kind, inside], TensorProto.BOOL, (1, C - 1, N, N), "fg_mask")
    occupied_any = b.node("Or", [h_occupied, v_occupied], TensorProto.BOOL, (1, 1, N, N), "occupied_any")
    occupied = b.node("And", [occupied_any, inside], TensorProto.BOOL, (1, 1, N, N), "occupied")
    empty = b.node("Not", [occupied], TensorProto.BOOL, (1, 1, N, N), "empty")
    bg = b.node("And", [inside, empty], TensorProto.BOOL, (1, 1, N, N), "bg")
    out9b = b.node("Concat", [bg, fg_mask], TensorProto.BOOL, (1, C, N, N), "out9b", axis=1)
    out9 = b.node("Cast", [out9b], TensorProto.FLOAT, (1, C, N, N), "out9", to=TensorProto.FLOAT)
    b.nodes.append(helper.make_node("Pad", [out9], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]))

    graph = helper.make_graph(b.nodes, TASK_ID, [inp], [out], b.initializers, value_info=b.value_infos)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def verify_model(model: onnx.ModelProto) -> dict[str, tuple[int, int]]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    splits: dict[str, tuple[int, int]] = {}
    failures: list[str] = []
    for split, examples in load_task_data().items():
        passed = 0
        checked = 0
        for idx, example in enumerate(examples):
            x = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if x is None or expected is None:
                continue
            pred = session.run([OUT_NAME], {IN_NAME: x})[0]
            checked += 1
            if np.array_equal(pred > 0.0, expected > 0.0):
                passed += 1
            else:
                failures.append(f"{split}[{idx}]")
        splits[split] = (passed, checked)
    if failures:
        raise AssertionError("failed examples: " + ", ".join(failures[:10]))
    return splits


def main() -> None:
    verify_reference()
    model = build_model()
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    splits = verify_model(model)
    onnx.save(model, BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"correctness: {splits}")
    print_report(score_file(BEST_PATH))


if __name__ == "__main__":
    main()
