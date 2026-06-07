"""ONNX solver for NeuroGolf task148 using dynamic wall/portal masks.

Task rule: each input grid contains two vertical red wall segments, one in the
upper region and one in the lower region, on opposite sides of the grid. Cyan
cells appear only in the upper wall's rows. Each cyan cell becomes yellow, the
cells between it and the upper red wall become cyan, and the same vertical
offset is emitted from the lower wall as a cyan horizontal ray toward the
opposite side. Red walls remain red, and all other valid-grid cells are
background. The JSON examples vary in height/width, but all fit in the 30x30
competition tensor; padding outside the encoded grid remains all-zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402


TASK_NUM = "148"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
OPSET = 10
CORE_H = 24
CORE_W = 12


@dataclass(frozen=True)
class Variant:
    name: str
    build: Callable[[], onnx.ModelProto]


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self._counter = 0

    def init(self, name: str, array: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(array, name))
        return name

    def i64(self, name: str, values: list[int] | np.ndarray) -> str:
        return self.init(name, np.asarray(values, dtype=np.int64))

    def f32(self, name: str, values: list[float] | np.ndarray) -> str:
        return self.init(name, np.asarray(values, dtype=np.float32))

    def node(self, op_type: str, inputs: list[str], output: str | None = None, **attrs: Any) -> str:
        if output is None:
            output = f"v{self._counter}"
            self._counter += 1
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        return output


def make_model(b: Builder, name: str) -> onnx.ModelProto:
    graph = helper.make_graph(
        b.nodes,
        name,
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


def slice_channel(b: Builder, color: int, out: str) -> str:
    return b.node(
        "Slice",
        [IN_NAME, f"s{color}", f"e{color}", "axis_chw"],
        out,
    )


def channel_4d(b: Builder, x: str, out: str) -> str:
    return b.node("Reshape", [x, "shape_1_1_h_w"], out)


def build_mask_solver() -> onnx.ModelProto:
    b = Builder()
    for color in (0, 2, 8):
        b.i64(f"s{color}", [color, 0, 0])
        b.i64(f"e{color}", [color + 1, CORE_H, CORE_W])
    b.i64("axis_chw", [1, 2, 3])
    b.f32("zero_f", [0.0])
    b.f32("half_f", [0.5])
    b.i64("row_idx", np.arange(CORE_H, dtype=np.int64))
    b.i64("col_idx", np.arange(CORE_W, dtype=np.int64))
    b.i64("col_rev_idx", np.arange(CORE_W - 1, -1, -1, dtype=np.int64))
    b.i64("one_i", [1])
    b.i64("last_col_i", [CORE_W - 1])
    b.i64("core_h_i", [CORE_H])
    b.i64("shape_h_1", [CORE_H, 1])
    b.i64("shape_1_w", [1, CORE_W])
    b.i64("shape_1_1_h_w", [1, 1, CORE_H, CORE_W])

    bg4 = slice_channel(b, 0, "bg4")
    red4 = slice_channel(b, 2, "red4")
    cyan4 = slice_channel(b, 8, "cyan4")
    bg_b4 = b.node("Greater", [bg4, "half_f"], "bg_b4")
    bg_b = b.node("Squeeze", [bg_b4], "bg_b", axes=[0, 1])
    red_b4 = b.node("Greater", [red4, "half_f"], "red_b4")
    red_b = b.node("Squeeze", [red_b4], "red_b", axes=[0, 1])
    cyan = b.node("Squeeze", [cyan4], "cyan", axes=[0, 1])
    cyan_b = b.node("Greater", [cyan, "half_f"], "cyan_b")

    col_sum = b.node("ReduceSum", [red4], "col_sum", axes=[0, 1, 2], keepdims=0)
    col_has = b.node("Greater", [col_sum, "zero_f"], "col_has")
    col_has_f = b.node("Cast", [col_has], "col_has_f", to=TensorProto.FLOAT)
    left_col = b.node("ArgMax", [col_has_f], "left_col", axis=0, keepdims=0)
    col_has_rev = b.node("Gather", [col_has_f, "col_rev_idx"], "col_has_rev", axis=0)
    right_from_end = b.node("ArgMax", [col_has_rev], "right_from_end", axis=0, keepdims=0)
    right_col = b.node("Sub", ["last_col_i", right_from_end], "right_col")

    left_wall4 = b.node("Gather", [red4, left_col], "left_wall4", axis=3)
    right_wall4 = b.node("Gather", [red4, right_col], "right_wall4", axis=3)
    left_wall = b.node("Squeeze", [left_wall4], "left_wall", axes=[0, 1])
    right_wall = b.node("Squeeze", [right_wall4], "right_wall", axes=[0, 1, 3])
    left_top = b.node("ArgMax", [left_wall], "left_top", axis=0, keepdims=0)
    right_top = b.node("ArgMax", [right_wall], "right_top", axis=0, keepdims=0)
    upper_is_left = b.node("Less", [left_top, right_top], "upper_is_left")
    upper_i = b.node("Cast", [upper_is_left], "upper_i", to=TensorProto.INT64)
    lower_i = b.node("Sub", ["one_i", upper_i], "lower_i")

    src_col_l = b.node("Mul", [upper_i, left_col], "src_col_l")
    src_col_r = b.node("Mul", [lower_i, right_col], "src_col_r")
    src_col = b.node("Add", [src_col_l, src_col_r], "src_col")
    dst_col_l = b.node("Mul", [upper_i, right_col], "dst_col_l")
    dst_col_r = b.node("Mul", [lower_i, left_col], "dst_col_r")
    dst_col = b.node("Add", [dst_col_l, dst_col_r], "dst_col")
    src_top_l = b.node("Mul", [upper_i, left_top], "src_top_l")
    src_top_r = b.node("Mul", [lower_i, right_top], "src_top_r")
    src_top = b.node("Add", [src_top_l, src_top_r], "src_top")
    dst_top_l = b.node("Mul", [upper_i, right_top], "dst_top_l")
    dst_top_r = b.node("Mul", [lower_i, left_top], "dst_top_r")
    dst_top = b.node("Add", [dst_top_l, dst_top_r], "dst_top")
    upper_not = b.node("Not", [upper_is_left], "upper_not")

    row_has_f = b.node("ReduceSum", [cyan], "row_has_f", axes=[1], keepdims=1)
    row_has = b.node("Greater", [row_has_f, "zero_f"], "row_has")
    cyan_col = b.node("ArgMax", [cyan], "cyan_col", axis=1, keepdims=1)
    col_grid = b.node("Reshape", ["col_idx", "shape_1_w"], "col_grid")

    gt_cyan = b.node("Greater", [col_grid, cyan_col], "gt_cyan")
    lt_cyan = b.node("Less", [col_grid, cyan_col], "lt_cyan")
    lt_src = b.node("Less", [col_grid, src_col], "lt_src")
    gt_src = b.node("Greater", [col_grid, src_col], "gt_src")
    between_right_a = b.node("And", [gt_cyan, lt_src], "between_right_a")
    between_left_a = b.node("And", [lt_cyan, gt_src], "between_left_a")
    between_left = b.node("And", [upper_is_left, between_left_a], "between_left")
    between_right = b.node("And", [upper_not, between_right_a], "between_right")
    source_between = b.node("Or", [between_left, between_right], "source_between")
    source_fill = b.node("And", [source_between, row_has], "source_fill")

    row_delta = b.node("Sub", ["row_idx", dst_top], "row_delta")
    source_rows = b.node("Add", [row_delta, src_top], "source_rows")
    src_ge_0 = b.node("Greater", [source_rows, b.i64("neg_one_i", [-1])], "src_ge_0")
    src_lt_30 = b.node("Less", [source_rows, "core_h_i"], "src_lt_30")
    src_valid = b.node("And", [src_ge_0, src_lt_30], "src_valid")
    src_valid_i = b.node("Cast", [src_valid], "src_valid_i", to=TensorProto.INT64)
    safe_source_rows = b.node("Mul", [src_valid_i, source_rows], "safe_source_rows")
    row_has_flat = b.node("Squeeze", [row_has], "row_has_flat", axes=[1])
    portal_rows_raw = b.node("Gather", [row_has_flat, safe_source_rows], "portal_rows_raw", axis=0)
    portal_rows = b.node("And", [portal_rows_raw, src_valid], "portal_rows")
    portal_rows_2d = b.node("Reshape", [portal_rows, "shape_h_1"], "portal_rows_2d")
    gt_dst = b.node("Greater", [col_grid, dst_col], "gt_dst")
    lt_dst = b.node("Less", [col_grid, dst_col], "lt_dst")
    portal_cols_l = b.node("And", [upper_is_left, lt_dst], "portal_cols_l")
    portal_cols_r = b.node("And", [upper_not, gt_dst], "portal_cols_r")
    portal_cols = b.node("Or", [portal_cols_l, portal_cols_r], "portal_cols")
    portal_fill = b.node("And", [portal_rows_2d, portal_cols], "portal_fill")

    cyan_any = b.node("Or", [source_fill, portal_fill], "cyan_any")
    cyan_out = b.node("And", [cyan_any, bg_b], "cyan_out")
    not_cyan_out = b.node("Not", [cyan_out], "not_cyan_out")
    bg = b.node("And", [bg_b, not_cyan_out], "bg")
    not_red = b.node("Not", [red_b], "not_red")
    false2 = b.node("And", [red_b, not_red], "false2")

    ch0 = channel_4d(b, bg, "ch0")
    chz = channel_4d(b, false2, "chz")
    ch2 = channel_4d(b, red_b, "ch2")
    ch4 = channel_4d(b, cyan_b, "ch4")
    ch8 = channel_4d(b, cyan_out, "ch8")
    out_b = b.node("Concat", [ch0, chz, ch2, chz, ch4, chz, chz, chz, ch8, chz], "out_b", axis=1)
    out_small = b.node("Cast", [out_b], "out_small", to=TensorProto.FLOAT)
    b.nodes.append(
        helper.make_node(
            "Pad",
            [out_small],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 30 - CORE_H, 30 - CORE_W],
            value=0.0,
        )
    )
    return make_model(b, "task148_mask_solver")


def build_mask_solver_4d() -> onnx.ModelProto:
    b = Builder()
    for color in (0, 2, 8):
        b.i64(f"s{color}", [color, 0, 0])
        b.i64(f"e{color}", [color + 1, CORE_H, CORE_W])
    b.i64("axis_chw", [1, 2, 3])
    b.f32("zero_f", [0.0])
    b.f32("half_f", [0.5])
    b.i64("row_idx", np.arange(CORE_H, dtype=np.int64))
    b.i64("col_idx", np.arange(CORE_W, dtype=np.int64))
    b.i64("col_rev_idx", np.arange(CORE_W - 1, -1, -1, dtype=np.int64))
    b.i64("zero_i", [0])
    b.i64("last_col_i", [CORE_W - 1])
    b.i64("core_h_i", [CORE_H])
    b.i64("neg_one_i", [-1])
    b.i64("shape_1_1_1_w", [1, 1, 1, CORE_W])
    b.i64("shape_1_1_h_1", [1, 1, CORE_H, 1])

    bg4 = slice_channel(b, 0, "bg4")
    red4 = slice_channel(b, 2, "red4")
    cyan4 = slice_channel(b, 8, "cyan4")
    bg_b4 = b.node("Greater", [bg4, "half_f"], "bg_b4")
    red_b4 = b.node("Greater", [red4, "half_f"], "red_b4")
    cyan_b4 = b.node("Greater", [cyan4, "half_f"], "cyan_b4")

    col_sum = b.node("ReduceSum", [red4], "col_sum", axes=[0, 1, 2], keepdims=0)
    col_has = b.node("Greater", [col_sum, "zero_f"], "col_has")
    col_has_f = b.node("Cast", [col_has], "col_has_f", to=TensorProto.FLOAT)
    col_has_rev = b.node("Gather", [col_has_f, "col_rev_idx"], "col_has_rev", axis=0)
    right_from_end = b.node("ArgMax", [col_has_rev], "right_from_end", axis=0, keepdims=0)
    right_col = b.node("Sub", ["last_col_i", right_from_end], "right_col")

    left_wall4 = b.node("Gather", [red4, "zero_i"], "left_wall4", axis=3)
    right_wall4 = b.node("Gather", [red4, right_col], "right_wall4", axis=3)
    left_wall = b.node("Squeeze", [left_wall4], "left_wall", axes=[0, 1, 3])
    right_wall = b.node("Squeeze", [right_wall4], "right_wall", axes=[0, 1, 3])
    left_top = b.node("ArgMax", [left_wall], "left_top", axis=0, keepdims=0)
    right_top = b.node("ArgMax", [right_wall], "right_top", axis=0, keepdims=0)
    upper_is_left = b.node("Less", [left_top, right_top], "upper_is_left")
    src_col = b.node("Where", [upper_is_left, "zero_i", right_col], "src_col")
    dst_col = b.node("Where", [upper_is_left, right_col, "zero_i"], "dst_col")
    src_top = b.node("Where", [upper_is_left, left_top, right_top], "src_top")
    dst_top = b.node("Where", [upper_is_left, right_top, left_top], "dst_top")

    row_has_f = b.node("ReduceSum", [cyan4], "row_has_f", axes=[0, 1, 3], keepdims=0)
    row_has_flat = b.node("Greater", [row_has_f, "zero_f"], "row_has_flat")
    row_has4 = b.node("Reshape", [row_has_flat, "shape_1_1_h_1"], "row_has4")
    cyan_col4 = b.node("ArgMax", [cyan4], "cyan_col4", axis=3, keepdims=1)
    col_grid4 = b.node("Reshape", ["col_idx", "shape_1_1_1_w"], "col_grid4")

    gt_cyan = b.node("Greater", [col_grid4, cyan_col4], "gt_cyan")
    lt_cyan = b.node("Less", [col_grid4, cyan_col4], "lt_cyan")
    lt_src = b.node("Less", [col_grid4, src_col], "lt_src")
    gt_src = b.node("Greater", [col_grid4, src_col], "gt_src")
    between_right_a = b.node("And", [gt_cyan, lt_src], "between_right_a")
    between_left_a = b.node("And", [lt_cyan, gt_src], "between_left_a")
    source_between = b.node("Or", [between_left_a, between_right_a], "source_between")
    source_fill = b.node("And", [source_between, row_has4], "source_fill")

    row_delta = b.node("Sub", ["row_idx", dst_top], "row_delta")
    source_rows = b.node("Add", [row_delta, src_top], "source_rows")
    src_ge_0 = b.node("Greater", [source_rows, "neg_one_i"], "src_ge_0")
    src_lt_30 = b.node("Less", [source_rows, "core_h_i"], "src_lt_30")
    src_valid = b.node("And", [src_ge_0, src_lt_30], "src_valid")
    safe_source_rows = b.node("Where", [src_valid, source_rows, "zero_i"], "safe_source_rows")
    portal_rows_raw = b.node("Gather", [row_has_flat, safe_source_rows], "portal_rows_raw", axis=0)
    portal_rows = b.node("And", [portal_rows_raw, src_valid], "portal_rows")
    portal_rows4 = b.node("Reshape", [portal_rows, "shape_1_1_h_1"], "portal_rows4")
    is_dst_col = b.node("Equal", [col_grid4, dst_col], "is_dst_col")
    portal_cols = b.node("Not", [is_dst_col], "portal_cols")
    portal_fill = b.node("And", [portal_rows4, portal_cols], "portal_fill")

    cyan_any = b.node("Or", [source_fill, portal_fill], "cyan_any")
    cyan_out = b.node("And", [cyan_any, bg_b4], "cyan_out")
    not_cyan_out = b.node("Not", [cyan_out], "not_cyan_out")
    bg = b.node("And", [bg_b4, not_cyan_out], "bg")
    zero4 = b.node("Less", [red4, "zero_f"], "zero4")

    out_b = b.node("Concat", [bg, zero4, red_b4, zero4, cyan_b4, zero4, zero4, zero4, cyan_out, zero4], "out_b", axis=1)
    out_small = b.node("Cast", [out_b], "out_small", to=TensorProto.FLOAT)
    b.nodes.append(
        helper.make_node(
            "Pad",
            [out_small],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 30 - CORE_H, 30 - CORE_W],
            value=0.0,
        )
    )
    return make_model(b, "task148_mask_solver_4d")


def variants() -> list[Variant]:
    return [
        Variant("mask_solver", build_mask_solver),
        Variant("mask_solver_4d", build_mask_solver_4d),
    ]



def load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def verify_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    ok = True
    splits: dict[str, tuple[int, int]] = {}
    for split, examples in load_task_data().items():
        passed = 0
        checked = 0
        for example in examples:
            inp = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if inp is None or expected is None:
                continue
            pred = session.run([OUT_NAME], {IN_NAME: inp})[0]
            checked += 1
            if np.array_equal(pred > 0.0, expected > 0.0):
                passed += 1
            else:
                ok = False
        splits[split] = (passed, checked)
    return ok, splits


def write_model(model: onnx.ModelProto, path: Path) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(path))


def benchmark_variants() -> tuple[dict[str, dict[str, Any]], str, dict[str, onnx.ModelProto]]:
    built = {variant.name: variant.build() for variant in variants()}
    results: dict[str, dict[str, Any]] = {}
    with __import__("tempfile").TemporaryDirectory(prefix="task148_") as tmp:
        tmpdir = Path(tmp)
        for name, model in built.items():
            path = tmpdir / f"{TASK_ID}_{name}.onnx"
            write_model(model, path)
            correct, splits = verify_correct(model)
            score = score_file(path) if correct else {"valid": False, "error": "incorrect"}
            results[name] = {"correct": correct, "splits": splits, **score}

    valid = [
        (name, result)
        for name, result in results.items()
        if result["correct"] and result.get("valid") and result.get("cost") is not None
    ]
    if not valid:
        raise RuntimeError(f"No valid correct {TASK_ID} variant: {results}")
    best_name = min(valid, key=lambda item: int(item[1]["cost"]))[0]
    return results, best_name, built


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Build and score {TASK_ID}.")
    parser.add_argument("--out", type=Path, default=BEST_PATH)
    args = parser.parse_args()

    results, best_name, built = benchmark_variants()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_model(built[best_name], args.out)

    for name, result in sorted(results.items(), key=lambda item: int(item[1].get("cost") or 10**18)):
        split_text = ", ".join(f"{k}={v[0]}/{v[1]}" for k, v in result["splits"].items())
        print(
            f"{name}: correct={result['correct']} valid={result.get('valid')} "
            f"cost={result.get('cost')} memory={result.get('memory')} "
            f"params={result.get('params')} score={result.get('score')} {split_text}"
        )
        if result.get("error"):
            print(f"  error: {str(result['error']).strip().splitlines()[-1]}")
    print(f"wrote {args.out} using {best_name}")


if __name__ == "__main__":
    main()
