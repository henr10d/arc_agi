"""ONNX generator for ARC task390 using red-boundary reflection.

Task rule: keep the red bracket/frame cells fixed. Gray cells inside the red
structure are reflected outward across the nearest long red boundary: for a
left/right bracket pair, reflect gray horizontally outside the closest side;
for a top/bottom frame, reflect gray vertically above or below the frame. The
active task grid is 15x15, and the remaining NeuroGolf output area is zero
padded.

The graph infers the orientation and red boundary rows/columns from the input
by counting red cells per row and column, builds compact row/column gather
indices for the reflection, and applies them to the gray mask.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task390"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task390.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
N = 15
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _i32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int32), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _slice(
    nodes: List[onnx.NodeProto],
    data: str,
    out: str,
    starts: str,
    ends: str,
    axes: str,
) -> str:
    nodes.append(helper.make_node("Slice", [data, starts, ends, axes], [out]))
    return out


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    red_st = _i64(inits, [0, 2, 0, 0], "red_st")
    red_en = _i64(inits, [1, 3, N, N], "red_en")
    gray_st = _i64(inits, [0, 5, 0, 0], "gray_st")
    gray_en = _i64(inits, [1, 6, N, N], "gray_en")
    half = _f32(inits, [0.5], "half")
    zero_i32 = _i32(inits, 0, "zero_i32")
    fourteen = _i32(inits, N - 1, "fourteen")
    two_i32 = _i32(inits, 2, "two_i32")
    reverse = _i64(inits, np.arange(N - 1, -1, -1, dtype=np.int64), "reverse")
    idx = _i32(inits, np.arange(N, dtype=np.int32), "idx")
    false15 = _init(inits, np.zeros((1, 1, N, N), dtype=np.bool_), "false15")

    _slice(nodes, IN_NAME, "red_f", red_st, red_en, axes4)
    _slice(nodes, IN_NAME, "gray_f4", gray_st, gray_en, axes4)
    nodes.append(helper.make_node("Greater", ["red_f", "half"], ["red"]))
    nodes.append(helper.make_node("Greater", ["gray_f4", "half"], ["gray_b4"]))

    nodes.append(helper.make_node("ReduceSum", ["red_f"], ["row_counts"], axes=[0, 1, 3], keepdims=0))
    nodes.append(helper.make_node("ReduceSum", ["red_f"], ["col_counts"], axes=[0, 1, 2], keepdims=0))
    nodes.append(helper.make_node("ReduceMax", ["row_counts"], ["row_max"], axes=[0], keepdims=0))
    nodes.append(helper.make_node("ReduceMax", ["col_counts"], ["col_max"], axes=[0], keepdims=0))
    nodes.append(helper.make_node("Less", ["row_max", "col_max"], ["horizontal"]))
    nodes.append(helper.make_node("Not", ["horizontal"], ["vertical"]))

    nodes.append(helper.make_node("ArgMax", ["row_counts"], ["top"], axis=0, keepdims=0))
    nodes.append(helper.make_node("Cast", ["top"], ["top_i"], to=TensorProto.INT32))
    nodes.append(helper.make_node("Gather", ["row_counts", "reverse"], ["row_counts_rev"], axis=0))
    nodes.append(helper.make_node("ArgMax", ["row_counts_rev"], ["bottom_rev"], axis=0, keepdims=0))
    nodes.append(helper.make_node("Cast", ["bottom_rev"], ["bottom_rev_i"], to=TensorProto.INT32))
    nodes.append(helper.make_node("Sub", ["fourteen", "bottom_rev_i"], ["bottom_i"]))

    nodes.append(helper.make_node("ArgMax", ["col_counts"], ["left"], axis=0, keepdims=0))
    nodes.append(helper.make_node("Cast", ["left"], ["left_i"], to=TensorProto.INT32))
    nodes.append(helper.make_node("Gather", ["col_counts", "reverse"], ["col_counts_rev"], axis=0))
    nodes.append(helper.make_node("ArgMax", ["col_counts_rev"], ["right_rev"], axis=0, keepdims=0))
    nodes.append(helper.make_node("Cast", ["right_rev"], ["right_rev_i"], to=TensorProto.INT32))
    nodes.append(helper.make_node("Sub", ["fourteen", "right_rev_i"], ["right_i"]))

    nodes.append(helper.make_node("Squeeze", ["gray_b4"], ["gray"], axes=[0, 1]))
    # For output rows above/below the red frame, gather the mirrored source row.
    nodes.append(helper.make_node("Less", ["idx", "top_i"], ["above_top"]))
    nodes.append(helper.make_node("Greater", ["idx", "bottom_i"], ["below_bottom"]))
    nodes.append(helper.make_node("Mul", ["top_i", "two_i32"], ["top2"]))
    nodes.append(helper.make_node("Mul", ["bottom_i", "two_i32"], ["bottom2"]))
    nodes.append(helper.make_node("Sub", ["top2", "idx"], ["src_top"]))
    nodes.append(helper.make_node("Sub", ["bottom2", "idx"], ["src_bottom"]))
    nodes.append(helper.make_node("Where", ["below_bottom", "src_bottom", "zero_i32"], ["row_src_below"]))
    nodes.append(helper.make_node("Where", ["above_top", "src_top", "row_src_below"], ["row_src_active"]))
    nodes.append(helper.make_node("Where", ["vertical", "row_src_active", "zero_i32"], ["row_src"]))
    nodes.append(helper.make_node("Add", ["top_i", "bottom_i"], ["row_sum"]))
    nodes.append(helper.make_node("Mul", ["row_src", "two_i32"], ["row_src2"]))
    nodes.append(helper.make_node("Less", ["row_sum", "row_src2"], ["row_lower_src"]))
    nodes.append(helper.make_node("Not", ["row_lower_src"], ["row_upper_src"]))
    nodes.append(helper.make_node("And", ["above_top", "row_upper_src"], ["above_valid"]))
    nodes.append(helper.make_node("And", ["below_bottom", "row_lower_src"], ["below_valid"]))
    nodes.append(helper.make_node("Or", ["above_valid", "below_valid"], ["outside_rows_raw"]))
    nodes.append(helper.make_node("And", ["vertical", "outside_rows_raw"], ["outside_rows"]))
    nodes.append(helper.make_node("Gather", ["gray", "row_src"], ["v_gray2"], axis=0))

    # For output columns outside the red brackets, gather the mirrored source column.
    nodes.append(helper.make_node("Less", ["idx", "left_i"], ["left_of_left"]))
    nodes.append(helper.make_node("Greater", ["idx", "right_i"], ["right_of_right"]))
    nodes.append(helper.make_node("Mul", ["left_i", "two_i32"], ["left2"]))
    nodes.append(helper.make_node("Mul", ["right_i", "two_i32"], ["right2"]))
    nodes.append(helper.make_node("Sub", ["left2", "idx"], ["src_left"]))
    nodes.append(helper.make_node("Sub", ["right2", "idx"], ["src_right"]))
    nodes.append(helper.make_node("Where", ["right_of_right", "src_right", "zero_i32"], ["col_src_right"]))
    nodes.append(helper.make_node("Where", ["left_of_left", "src_left", "col_src_right"], ["col_src_active"]))
    nodes.append(helper.make_node("Where", ["horizontal", "col_src_active", "zero_i32"], ["col_src"]))
    nodes.append(helper.make_node("Add", ["left_i", "right_i"], ["col_sum"]))
    nodes.append(helper.make_node("Mul", ["col_src", "two_i32"], ["col_src2"]))
    nodes.append(helper.make_node("Less", ["col_sum", "col_src2"], ["col_right_src"]))
    nodes.append(helper.make_node("Not", ["col_right_src"], ["col_left_src"]))
    nodes.append(helper.make_node("And", ["left_of_left", "col_left_src"], ["left_valid"]))
    nodes.append(helper.make_node("And", ["right_of_right", "col_right_src"], ["right_valid"]))
    nodes.append(helper.make_node("Or", ["left_valid", "right_valid"], ["outside_cols_raw"]))
    nodes.append(helper.make_node("And", ["horizontal", "outside_cols_raw"], ["outside_cols"]))
    nodes.append(helper.make_node("Gather", ["gray", "col_src"], ["h_gray2"], axis=1))
    nodes.append(helper.make_node("Unsqueeze", ["outside_rows"], ["outside_rows2"], axes=[1]))
    nodes.append(helper.make_node("Unsqueeze", ["outside_cols"], ["outside_cols2"], axes=[0]))
    nodes.append(helper.make_node("And", ["v_gray2", "outside_rows2"], ["v_gray2_masked"]))
    nodes.append(helper.make_node("And", ["h_gray2", "outside_cols2"], ["h_gray2_masked"]))
    nodes.append(helper.make_node("Unsqueeze", ["v_gray2_masked"], ["v_gray"], axes=[0, 1]))
    nodes.append(helper.make_node("Unsqueeze", ["h_gray2_masked"], ["h_gray"], axes=[0, 1]))
    nodes.append(helper.make_node("Or", ["v_gray", "h_gray"], ["gray_out"]))

    nodes.append(helper.make_node("Or", ["red", "gray_out"], ["fg"]))
    nodes.append(helper.make_node("Not", ["fg"], ["bg"]))
    chans = ["bg", "false15", "red", "false15", "false15", "gray_out"]
    nodes.append(helper.make_node("Concat", chans, ["out15b"], axis=1))
    nodes.append(helper.make_node("Cast", ["out15b"], ["out15"], to=TensorProto.FLOAT))
    nodes.append(
        helper.make_node(
            "Pad",
            ["out15"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, C - 6, H - N, W - N],
        )
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


def validate_examples(path: Path) -> tuple[int, int, dict[str, int]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    split_passed: dict[str, int] = {}
    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        split_passed[split] = 0
        for idx, example in enumerate(data.get(split, [])):
            arr = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if arr is None or expected is None:
                continue
            actual = session.run([OUT_NAME], {IN_NAME: arr})[0]
            total += 1
            if not np.array_equal(actual > 0.0, expected > 0.0):
                raise AssertionError(f"{split}[{idx}] output mismatch")
            passed += 1
            split_passed[split] += 1
    return passed, total, split_passed


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    passed, total, split_passed = validate_examples(BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"correct: {passed}/{total}")
    print(
        "splits:  "
        f"train {split_passed['train']}, "
        f"test {split_passed['test']}, "
        f"arc-gen {split_passed['arc-gen']}"
    )
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
