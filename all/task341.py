"""ONNX solver for ARC task341: connect two rectangles with cyan.

Task rule: each 10x10 input has exactly two solid non-black rectangles/bars on
black background.  Preserve the rectangles and add a cyan (8) connector across
the empty gap between them.  If the objects are vertically separated, fill the
empty rows between them over the interior of their horizontal overlap.  If they
are horizontally separated, fill the empty columns between them over the
interior of their vertical overlap.  The overlap span is inset by one cell at
both ends, matching the task JSON examples.

ONNX: crop only the 10x10 black channel, derive foreground row/column counts
from it, build the connector from compact 10-element summaries, pad that
single-channel connector mask to 30x30, and use one final Where to splice a
cyan one-hot value into the original full-size input tensor.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task341"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

IN_NAME = "input"
OUT_NAME = "output"
C = 10
G = 10
H = W = 30
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10
CYAN = 8


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.counter = 0

    def name(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def node(self, op_type: str, inputs: list[str], prefix: str, **attrs: Any) -> str:
        out = self.name(prefix)
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out

    def init(self, name: str, values: Any, dtype: np.dtype[Any] | type[np.generic]) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=dtype), name))
        return name

    def i64(self, name: str, values: Any) -> str:
        return self.init(name, values, np.int64)

    def f32(self, name: str, values: Any) -> str:
        return self.init(name, values, np.float32)


def make_model(b: Builder, graph_name: str) -> onnx.ModelProto:
    graph = helper.make_graph(
        b.nodes,
        graph_name,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        b.initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def _inner_run_mask(
    b: Builder,
    mask: str,
    *,
    axis: int,
    coords: str,
    rev: str,
    last: str,
    prefix: str,
    keep_mask: bool = True,
) -> str:
    """Return mask cells strictly between first and last true position."""
    as_u8 = b.node("Cast", [mask], f"{prefix}_u8", to=TensorProto.UINT8)
    first = b.node("ArgMax", [as_u8], f"{prefix}_first", axis=axis, keepdims=1)
    reversed_mask = b.node("Gather", [as_u8, rev], f"{prefix}_rev", axis=axis)
    from_end = b.node("ArgMax", [reversed_mask], f"{prefix}_from_end", axis=axis, keepdims=1)
    last_pos = b.node("Sub", [last, from_end], f"{prefix}_last")
    after_first = b.node("Greater", [coords, first], f"{prefix}_after_first")
    before_last = b.node("Less", [coords, last_pos], f"{prefix}_before_last")
    inside = b.node("And", [after_first, before_last], f"{prefix}_inside")
    if not keep_mask:
        return inside
    return b.node("And", [mask, inside], f"{prefix}_inner")


def _neighbor_inner_mask(
    b: Builder,
    mask: str,
    *,
    axis: int,
    zero_unit: str,
    slice0: str,
    slice1: str,
    slice9: str,
    slice10: str,
    axis_name: str,
    prefix: str,
) -> str:
    """Return true positions with true neighbors on both sides along axis."""
    left = b.node("Slice", [mask, slice0, slice9, axis_name], f"{prefix}_left_slice")
    has_left = b.node("Concat", [zero_unit, left], f"{prefix}_has_left", axis=axis)
    right = b.node("Slice", [mask, slice1, slice10, axis_name], f"{prefix}_right_slice")
    has_right = b.node("Concat", [right, zero_unit], f"{prefix}_has_right", axis=axis)
    both_neighbors = b.node("And", [has_left, has_right], f"{prefix}_both_neighbors")
    return b.node("And", [mask, both_neighbors], f"{prefix}_inner")


def build_model() -> onnx.ModelProto:
    b = Builder()

    half_f = b.f32("half_f", [0.5])
    ten_f = b.f32("ten_f", [float(G)])
    last = b.i64("last", [G - 1])
    rev = b.i64("rev", np.arange(G - 1, -1, -1))
    rows = b.i64("rows", np.arange(G).reshape(1, 1, G, 1))
    cols = b.i64("cols", np.arange(G).reshape(1, 1, 1, G))
    core_st = b.i64("core_st", [0, 0, 0, 0])
    black_en = b.i64("black_en", [1, 1, G, G])
    slice0 = b.i64("slice0", [0])
    slice1 = b.i64("slice1", [1])
    slice9 = b.i64("slice9", [G - 1])
    slice10 = b.i64("slice10", [G])
    axis_row = b.i64("axis_row", [2])
    axis_col = b.i64("axis_col", [3])
    cyan_onehot = np.zeros((1, C, 1, 1), dtype=np.float32)
    cyan_onehot[0, CYAN, 0, 0] = 1.0
    b.f32("cyan_onehot", cyan_onehot)
    b.init("zero_unit", np.zeros((1, 1, 1, 1), dtype=np.bool_), np.bool_)
    b.init("zero_right", np.zeros((1, 1, G, W - G), dtype=np.bool_), np.bool_)
    b.init("zero_bottom", np.zeros((1, 1, H - G, W), dtype=np.bool_), np.bool_)

    black = b.node("Slice", [IN_NAME, core_st, black_en], "black")
    bg_row_count = b.node("ReduceSum", [black], "bg_row_count", axes=[3], keepdims=1)
    bg_col_count = b.node("ReduceSum", [black], "bg_col_count", axes=[2], keepdims=1)
    row_count = b.node("Sub", [ten_f, bg_row_count], "row_count")
    col_count = b.node("Sub", [ten_f, bg_col_count], "col_count")

    row_has = b.node("Greater", [row_count, half_f], "row_has")
    col_has = b.node("Greater", [col_count, half_f], "col_has")

    row_inner_extent = _inner_run_mask(
        b,
        row_has,
        axis=2,
        coords=rows,
        rev=rev,
        last=last,
        prefix="row_extent",
        keep_mask=False,
    )
    col_inner_extent = _inner_run_mask(
        b,
        col_has,
        axis=3,
        coords=cols,
        rev=rev,
        last=last,
        prefix="col_extent",
        keep_mask=False,
    )
    row_gap = b.node("And", [row_inner_extent, b.node("Not", [row_has], "not_row_has")], "row_gap")
    col_gap = b.node("And", [col_inner_extent, b.node("Not", [col_has], "not_col_has")], "col_gap")

    max_row_count = b.node("ReduceMax", [row_count], "max_row_count", axes=[2], keepdims=1)
    max_col_count = b.node("ReduceMax", [col_count], "max_col_count", axes=[3], keepdims=1)
    row_overlap_threshold = b.node("Sub", [max_row_count, half_f], "row_overlap_threshold")
    col_overlap_threshold = b.node("Sub", [max_col_count, half_f], "col_overlap_threshold")
    row_overlap = b.node("Greater", [row_count, row_overlap_threshold], "row_overlap")
    col_overlap = b.node("Greater", [col_count, col_overlap_threshold], "col_overlap")

    row_overlap_inner = _neighbor_inner_mask(
        b,
        row_overlap,
        axis=2,
        zero_unit="zero_unit",
        slice0=slice0,
        slice1=slice1,
        slice9=slice9,
        slice10=slice10,
        axis_name=axis_row,
        prefix="row_overlap",
    )
    col_overlap_inner = _neighbor_inner_mask(
        b,
        col_overlap,
        axis=3,
        zero_unit="zero_unit",
        slice0=slice0,
        slice1=slice1,
        slice9=slice9,
        slice10=slice10,
        axis_name=axis_col,
        prefix="col_overlap",
    )

    vertical = b.node("And", [row_gap, col_overlap_inner], "vertical")
    horizontal = b.node("And", [row_overlap_inner, col_gap], "horizontal")
    fill = b.node("Or", [vertical, horizontal], "fill")
    fill_wide = b.node("Concat", [fill, "zero_right"], "fill_wide", axis=3)
    fill30 = b.node("Concat", [fill_wide, "zero_bottom"], "fill30", axis=2)
    b.nodes.append(
        helper.make_node(
            "Where",
            [fill30, "cyan_onehot", IN_NAME],
            [OUT_NAME],
        )
    )

    return make_model(b, "task341_connector")


def solve(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    fg = g > 0
    row_count = fg.sum(axis=1)
    col_count = fg.sum(axis=0)
    rows = np.arange(g.shape[0])
    cols = np.arange(g.shape[1])

    row_has = row_count > 0
    col_has = col_count > 0
    r0, r1 = int(row_has.argmax()), int(len(row_has) - 1 - row_has[::-1].argmax())
    c0, c1 = int(col_has.argmax()), int(len(col_has) - 1 - col_has[::-1].argmax())

    row_gap = (rows > r0) & (rows < r1) & ~row_has
    col_gap = (cols > c0) & (cols < c1) & ~col_has

    row_overlap = row_count == row_count.max()
    col_overlap = col_count == col_count.max()
    ro0, ro1 = int(row_overlap.argmax()), int(len(row_overlap) - 1 - row_overlap[::-1].argmax())
    co0, co1 = int(col_overlap.argmax()), int(len(col_overlap) - 1 - col_overlap[::-1].argmax())
    row_overlap_inner = row_overlap & (rows > ro0) & (rows < ro1)
    col_overlap_inner = col_overlap & (cols > co0) & (cols < co1)

    fill = (row_gap[:, None] & col_overlap_inner[None, :]) | (
        row_overlap_inner[:, None] & col_gap[None, :]
    )
    out = g.copy()
    out[fill] = CYAN
    return out


def load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def verify_reference() -> None:
    for split, examples in load_task_data().items():
        for idx, example in enumerate(examples):
            actual = solve(example["input"])
            expected = np.asarray(example["output"], dtype=np.int64)
            if not np.array_equal(actual, expected):
                raise AssertionError(f"reference solver failed on {split}[{idx}]")


def verify_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    split_counts: dict[str, tuple[int, int]] = {}
    all_ok = True
    for split in ("train", "test", "arc-gen"):
        passed = 0
        checked = 0
        for example in load_task_data().get(split, []):
            inp = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if inp is None or expected is None:
                continue
            pred = session.run([OUT_NAME], {IN_NAME: inp})[0]
            checked += 1
            if np.array_equal(pred > 0.0, expected > 0.0):
                passed += 1
            else:
                all_ok = False
        split_counts[split] = (passed, checked)
    return all_ok, split_counts


def write_model(model: onnx.ModelProto, path: Path) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(path))


def benchmark() -> dict[str, Any]:
    verify_reference()
    model = build_model()
    ok, splits = verify_correct(model)
    with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}_") as tmp:
        path = Path(tmp) / f"{TASK_ID}.onnx"
        write_model(model, path)
        result = score_file(path)
    result["correct"] = ok
    result["splits"] = splits
    result["model"] = model
    return result


def print_result(result: dict[str, Any]) -> None:
    score = result["score"]
    score_text = f"{score:.6f}" if isinstance(score, float) else "None"
    print(
        f"correct={result['correct']} valid={result['valid']} "
        f"memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={score_text}"
    )
    print(f"splits: {result['splits']}")
    if result["error"]:
        print(f"error: {str(result['error']).strip()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and benchmark task341 ONNX.")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()

    result = benchmark()
    print_result(result)
    if not result["correct"] or not result["valid"]:
        raise SystemExit("task341 model is not correct and valid")
    if not args.benchmark_only:
        OUT_DIR.mkdir(exist_ok=True)
        write_model(result["model"], BEST_PATH)
        print(f"wrote: {BEST_PATH}")


if __name__ == "__main__":
    main()
