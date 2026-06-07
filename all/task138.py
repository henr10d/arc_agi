"""ONNX solver for ARC task138: crop a colored frame and extend matching colors.

Task rule: each input contains four long colored line segments forming the
top, left, bottom, and right sides of a rectangle. The output is that rectangle
cropped to the top-left of the competition tensor. Interior cells whose color
matches a border side are extended in a straight line toward that same-colored
side: top-color cells fill upward, bottom-color cells fill downward, left-color
cells fill leftward, and right-color cells fill rightward. Other interior cells
remain black; border cells remain unchanged.

The graph detects the two horizontal and two vertical long lines, gathers the
dynamic crop into a fixed 30x30 workspace, performs four directional cumulative
fills on compact single-channel masks, then decodes a one-hot output.
"""

from __future__ import annotations

import copy
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

from score_model import convert_to_numpy, sanitize_model, score_file  # noqa: E402


TASK_ID = "task138"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
IR_VERSION = 10
OPSET = 10


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        _init(self.inits, np.asarray([0.0], dtype=np.float16), "zero_h")
        _i64(self.inits, [0], "zero_i")
        _i64(self.inits, [900], "flat_shape")
        _i64(self.inits, [29], "last_i")
        _init(self.inits, np.asarray([0], dtype=np.int32), "zero_i32")
        _init(self.inits, np.asarray([1], dtype=np.int32), "one_i32")
        _init(self.inits, np.asarray([30], dtype=np.int32), "thirty_i32")
        _i64(self.inits, [10], "depth_i")
        _f32(self.inits, [0.0, 1.0], "onehot_values")
        _init(self.inits, np.asarray([10], dtype=np.uint8), "ten_u8")
        _init(self.inits, np.arange(H, dtype=np.int32).reshape(1, 1, H, 1), "rows")
        _init(self.inits, np.arange(W, dtype=np.int32).reshape(1, 1, 1, W), "cols")
        _i64(self.inits, np.arange(H - 1, -1, -1), "rev30")
        lower = np.tril(np.ones((H, W), dtype=np.float16))
        _init(self.inits, lower, "tri_l")
        _init(self.inits, lower.T.copy(), "tri_u")

    def n(self, op_type: str, inputs: list[str], output: str, **attrs: Any) -> str:
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        return output

    def le(self, a: str, b: str, out: str) -> str:
        return self.n("Not", [self.n("Less", [b, a], f"{out}_gt")], out)

    def ge(self, a: str, b: str, out: str) -> str:
        return self.n("Not", [self.n("Less", [a, b], f"{out}_lt")], out)

    def eq(self, a: str, b: str, out: str) -> str:
        le_ab = self.le(a, b, f"{out}_le")
        ge_ab = self.ge(a, b, f"{out}_ge")
        return self.n("And", [le_ab, ge_ab], out)


def _scalar_from_grid(b: Builder, flat_grid: str, row: str, col: str, name: str) -> str:
    row_part = b.n("Mul", [row, "thirty_i32"], f"{name}_rp")
    idx = b.n("Add", [row_part, col], f"{name}_idx")
    return b.n("Gather", [flat_grid, idx], name, axis=0)


def _cum_positive(b: Builder, mask: str, axis: str, name: str, *, reverse: bool = False) -> str:
    mask_f = b.n("Cast", [mask], f"{name}_f", to=TensorProto.FLOAT16)
    if axis == "axis_h":
        summed = b.n("MatMul", ["tri_u" if reverse else "tri_l", mask_f], f"{name}_sum")
    elif axis == "axis_w":
        summed = b.n("MatMul", [mask_f, "tri_l" if reverse else "tri_u"], f"{name}_sum")
    else:
        raise ValueError(axis)
    return b.n("Greater", [summed, "zero_h"], name)


def build_model() -> onnx.ModelProto:
    b = Builder()

    ids64 = b.n("ArgMax", [IN_NAME], "ids64", axis=1, keepdims=1)
    ids = b.n("Cast", [ids64], "ids", to=TensorProto.UINT8)
    ids_flat = b.n("Reshape", [ids, "flat_shape"], "ids_flat")
    fg_any_b = b.n("Greater", [ids64, "zero_i"], "fg_any_b")
    fg_any = b.n("Cast", [fg_any_b], "fg_any", to=TensorProto.FLOAT16)
    row_counts = b.n("ReduceSum", [fg_any], "row_counts", axes=[3], keepdims=1)
    col_counts = b.n("ReduceSum", [fg_any], "col_counts", axes=[2], keepdims=1)

    top = b.n("ArgMax", [row_counts], "top", axis=2, keepdims=1)
    left = b.n("ArgMax", [col_counts], "left", axis=3, keepdims=1)
    row_rev = b.n("Gather", [row_counts, "rev30"], "row_rev", axis=2)
    col_rev = b.n("Gather", [col_counts, "rev30"], "col_rev", axis=3)
    bottom_rev = b.n("ArgMax", [row_rev], "bottom_rev", axis=2, keepdims=1)
    right_rev = b.n("ArgMax", [col_rev], "right_rev", axis=3, keepdims=1)
    bottom = b.n("Sub", ["last_i", bottom_rev], "bottom")
    right = b.n("Sub", ["last_i", right_rev], "right")
    height_m1 = b.n("Sub", [bottom, top], "height_m1")
    width_m1 = b.n("Sub", [right, left], "width_m1")
    top32 = b.n("Cast", [top], "top32", to=TensorProto.INT32)
    left32 = b.n("Cast", [left], "left32", to=TensorProto.INT32)
    bottom32 = b.n("Cast", [bottom], "bottom32", to=TensorProto.INT32)
    right32 = b.n("Cast", [right], "right32", to=TensorProto.INT32)
    height_m1_32 = b.n("Cast", [height_m1], "height_m1_32", to=TensorProto.INT32)
    width_m1_32 = b.n("Cast", [width_m1], "width_m1_32", to=TensorProto.INT32)

    r_ok = b.le("rows", height_m1_32, "r_ok")
    c_ok = b.le("cols", width_m1_32, "c_ok")
    in_bounds = b.n("And", [r_ok, c_ok], "in_bounds")

    safe_rows = b.n("Where", [r_ok, "rows", "zero_i32"], "safe_rows")
    safe_cols = b.n("Where", [c_ok, "cols", "zero_i32"], "safe_cols")
    src_r0 = b.n("Add", [safe_rows, top32], "src_r0")
    src_c0 = b.n("Add", [safe_cols, left32], "src_c0")
    src_r_part = b.n("Mul", [src_r0, "thirty_i32"], "src_r_part")
    src_idx = b.n("Add", [src_r_part, src_c0], "src_idx")
    crop_ids = b.n("Gather", [ids_flat, src_idx], "crop_ids", axis=0)

    left_plus = b.n("Add", [left32, "one_i32"], "left_plus")
    top_plus = b.n("Add", [top32, "one_i32"], "top_plus")
    top_color = _scalar_from_grid(b, ids_flat, top32, left_plus, "top_color")
    bottom_color = _scalar_from_grid(b, ids_flat, bottom32, left_plus, "bottom_color")
    left_color = _scalar_from_grid(b, ids_flat, top_plus, left32, "left_color")
    right_color = _scalar_from_grid(b, ids_flat, top_plus, right32, "right_color")
    crop_ids32_seed = b.n("Cast", [crop_ids], "crop_ids32_seed", to=TensorProto.INT32)
    top_color32 = b.n("Cast", [top_color], "top_color32", to=TensorProto.INT32)
    bottom_color32 = b.n("Cast", [bottom_color], "bottom_color32", to=TensorProto.INT32)
    left_color32 = b.n("Cast", [left_color], "left_color32", to=TensorProto.INT32)
    right_color32 = b.n("Cast", [right_color], "right_color32", to=TensorProto.INT32)

    r_gt0 = b.n("Less", ["zero_i32", "rows"], "r_gt0")
    c_gt0 = b.n("Less", ["zero_i32", "cols"], "c_gt0")
    r_lt_last = b.n("Less", ["rows", height_m1_32], "r_lt_last")
    c_lt_last = b.n("Less", ["cols", width_m1_32], "c_lt_last")
    interior_r = b.n("And", [r_gt0, r_lt_last], "interior_r")
    interior_c = b.n("And", [c_gt0, c_lt_last], "interior_c")
    interior = b.n("And", [b.n("And", [interior_r, interior_c], "interior0"), in_bounds], "interior")

    seed_top = b.n("And", [b.n("Equal", [crop_ids32_seed, top_color32], "seed_top0"), interior], "seed_top")
    seed_bottom = b.n("And", [b.n("Equal", [crop_ids32_seed, bottom_color32], "seed_bottom0"), interior], "seed_bottom")
    seed_left = b.n("And", [b.n("Equal", [crop_ids32_seed, left_color32], "seed_left0"), interior], "seed_left")
    seed_right = b.n("And", [b.n("Equal", [crop_ids32_seed, right_color32], "seed_right0"), interior], "seed_right")

    fill_top = b.n("And", [_cum_positive(b, seed_top, "axis_h", "fill_top_raw", reverse=True), interior], "fill_top")
    fill_bottom = b.n("And", [_cum_positive(b, seed_bottom, "axis_h", "fill_bottom_raw"), interior], "fill_bottom")
    fill_left = b.n("And", [_cum_positive(b, seed_left, "axis_w", "fill_left_raw", reverse=True), interior], "fill_left")
    fill_right = b.n("And", [_cum_positive(b, seed_right, "axis_w", "fill_right_raw"), interior], "fill_right")

    base_ids = b.n("Where", [in_bounds, crop_ids, "ten_u8"], "base_ids")
    out1 = b.n("Where", [fill_top, top_color, base_ids], "out1")
    out2 = b.n("Where", [fill_bottom, bottom_color, out1], "out2")
    out3 = b.n("Where", [fill_left, left_color, out2], "out3")
    out_ids = b.n("Where", [fill_right, right_color, out3], "out_ids")

    out_ids_sq = b.n("Squeeze", [out_ids], "out_ids_sq", axes=[1])
    out_ids_oh = b.n("Cast", [out_ids_sq], "out_ids_oh", to=TensorProto.INT64)
    b.n("OneHot", [out_ids_oh, "depth_i", "onehot_values"], OUT_NAME, axis=1)

    graph = helper.make_graph(
        b.nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        b.inits,
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


def _load_examples() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def verify_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]], str | None]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return False, {}, "sanitize failed"
    session = ort.InferenceSession(sanitized.SerializeToString(), providers=["CPUExecutionProvider"])
    counts: dict[str, tuple[int, int]] = {}
    all_ok = True
    first_error: str | None = None
    for split, examples in _load_examples().items():
        passed = 0
        total = 0
        for idx, example in enumerate(examples):
            x = convert_to_numpy(example, "input")
            y = convert_to_numpy(example, "output")
            if x is None or y is None:
                continue
            total += 1
            pred = session.run([OUT_NAME], {IN_NAME: x})[0]
            if np.array_equal(pred > 0.0, y > 0.0):
                passed += 1
            else:
                all_ok = False
                if first_error is None:
                    first_error = f"{split}[{idx}]"
        counts[split] = (passed, total)
    return all_ok, counts, first_error


def main() -> None:
    model = build_model()
    ok, counts, first_error = verify_correct(model)
    print("correct:", ok, counts, "first_error:", first_error)
    onnx.save(model, str(BEST_PATH))
    print("wrote", BEST_PATH)
    print(score_file(BEST_PATH))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
