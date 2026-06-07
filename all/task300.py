"""Minimal ONNX for ARC task300 using largest-color footprint compression.

Task rule: the input contains several colored horizontal-run footprints on a
black background.  For every non-black color, compress its used rows and
columns to the tight bounding box of that color's footprint.  The target is the
unique foreground color with the largest occupied-cell count; emit only that
compressed footprint, preserving black holes inside the box.  All observed
outputs fit within 4x3, so the graph builds a fixed 4x3 crop and masks cells
outside the computed bbox before the final NeuroGolf padding.
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
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task300"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task300.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
RH = 9
RW = 15
OH = 4
OW = 3
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    sel_st = _i64(inits, [0, 0, 0, 0], "sel_st")
    roi_en = _i64(inits, [1, C, RH, RW], "roi_en")
    count_st = _i64(inits, [0, 1, 0, 0], "count_st")
    count_en = _i64(inits, [1, C, 1, 1], "count_en")
    one_shape = _i64(inits, [1], "one_shape")
    one_i = _i64(inits, [1], "one_i")
    zero_i = _i64(inits, [0], "zero_i")
    neg_one_i = _i64(inits, [-1], "neg_one_i")
    eight_i = _i64(inits, [RH - 1], "eight_i")
    nine_i = _i64(inits, [RH], "nine_i")
    fifteen_i = _i64(inits, [RW], "fifteen_i")
    half = _f32(inits, [0.5], "half")

    rev_rows = _i64(inits, np.arange(RH - 1, -1, -1), "rev_rows")
    rev_cols = _i64(inits, np.arange(RW - 1, -1, -1), "rev_cols")
    out_rows = _i64(inits, np.arange(OH), "out_rows")
    out_cols = _i64(inits, np.arange(OW), "out_cols")
    out_rows_4d = _i64(inits, np.arange(OH).reshape(1, 1, OH, 1), "out_rows_4d")
    out_cols_4d = _i64(inits, np.arange(OW).reshape(1, 1, 1, OW), "out_cols_4d")
    channels = _i64(inits, np.arange(C).reshape(1, C, 1, 1), "channels")

    nodes.extend(
        [
            helper.make_node("ReduceSum", [IN_NAME], ["counts10"], axes=[2, 3], keepdims=1),
            helper.make_node("Slice", ["counts10", count_st, count_en, axes4], ["counts"]),
            helper.make_node("ArgMax", ["counts"], ["best0"], axis=1, keepdims=1),
            helper.make_node("Reshape", ["best0", one_shape], ["best0v"]),
            helper.make_node("Add", ["best0v", one_i], ["best"]),
            helper.make_node("Gather", [IN_NAME, "best"], ["sel_full"], axis=1),
            helper.make_node("Slice", ["sel_full", sel_st, roi_en, axes4], ["sel"]),
            helper.make_node("ReduceMax", ["sel"], ["row_occ"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["sel"], ["col_occ"], axes=[2], keepdims=1),
            helper.make_node("ArgMax", ["row_occ"], ["top4"], axis=2, keepdims=1),
            helper.make_node("ArgMax", ["col_occ"], ["left4"], axis=3, keepdims=1),
            helper.make_node("Reshape", ["top4", one_shape], ["top"]),
            helper.make_node("Reshape", ["left4", one_shape], ["left"]),
            helper.make_node("Gather", ["row_occ", rev_rows], ["row_rev"], axis=2),
            helper.make_node("Gather", ["col_occ", rev_cols], ["col_rev"], axis=3),
            helper.make_node("ArgMax", ["row_rev"], ["rb4"], axis=2, keepdims=1),
            helper.make_node("ArgMax", ["col_rev"], ["cb4"], axis=3, keepdims=1),
            helper.make_node("Reshape", ["rb4", one_shape], ["rb"]),
            helper.make_node("Reshape", ["cb4", one_shape], ["cb"]),
            helper.make_node("Sub", [nine_i, "rb"], ["bottom"]),
            helper.make_node("Sub", [fifteen_i, "cb"], ["right"]),
            helper.make_node("Sub", ["bottom", "top"], ["height"]),
            helper.make_node("Sub", ["right", "left"], ["width"]),
            helper.make_node("Add", ["top", out_rows], ["row_idx"]),
            helper.make_node("Add", ["left", out_cols], ["col_idx"]),
            helper.make_node("Min", ["row_idx", eight_i], ["row_safe"]),
            helper.make_node("Gather", ["sel", "row_safe"], ["sel_rows"], axis=2),
            helper.make_node("Gather", ["sel_rows", "col_idx"], ["sel_crop"], axis=3),
            helper.make_node("Greater", ["sel_crop", half], ["pix"]),
            helper.make_node("Less", [out_rows_4d, "height"], ["row_in"]),
            helper.make_node("Less", [out_cols_4d, "width"], ["col_in"]),
            helper.make_node("And", ["row_in", "col_in"], ["inside"]),
            helper.make_node("Where", ["pix", "best", zero_i], ["raw_color_grid"]),
            helper.make_node("Where", ["inside", "raw_color_grid", neg_one_i], ["color_grid"]),
            helper.make_node("Equal", [channels, "color_grid"], ["outb"]),
            helper.make_node("Cast", ["outb"], ["out4"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out4"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - OH, W - OW],
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
    onnx.checker.check_model(model)
    return model


def _examples() -> list[tuple[str, int, dict[str, Any]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    examples: list[tuple[str, int, dict[str, Any]]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            examples.append((split, idx, example))
    return examples


def validate_task_structure() -> None:
    for split, idx, example in _examples():
        counts = np.bincount(np.ravel(example["input"]), minlength=C)
        target_colors = [color for row in example["output"] for color in row if color != 0]
        if not target_colors:
            raise AssertionError(f"{split}[{idx}] has no foreground output color")
        target = target_colors[0]
        if any(color not in {0, target} for row in example["output"] for color in row):
            raise AssertionError(f"{split}[{idx}] output has multiple foreground colors")

        foreground_counts = counts[1:]
        largest = int(np.argmax(foreground_counts) + 1)
        if int(foreground_counts[largest - 1]) != int(counts[target]):
            raise AssertionError(f"{split}[{idx}] target is not largest foreground color")
        if np.count_nonzero(foreground_counts == foreground_counts[largest - 1]) != 1:
            raise AssertionError(f"{split}[{idx}] largest foreground color is not unique")

        rows, cols = np.where(np.asarray(example["input"]) == target)
        cropped = np.asarray(example["input"])[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
        expected = np.asarray(example["output"])
        target_only = np.where(cropped == target, target, 0)
        if not np.array_equal(target_only, expected):
            raise AssertionError(f"{split}[{idx}] output is not the target bbox footprint")


def validate_examples(path: Path) -> tuple[int, int, dict[str, int]]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    split_passed = {"train": 0, "test": 0, "arc-gen": 0}
    passed = 0
    total = 0
    for split, idx, example in _examples():
        inp = convert_to_numpy(example, "input")
        expected = convert_to_numpy(example, "output")
        if inp is None or expected is None:
            continue
        actual = session.run([OUT_NAME], {IN_NAME: inp})[0]
        total += 1
        if not np.array_equal(actual > 0.0, expected > 0.0):
            raise AssertionError(f"{split}[{idx}] output mismatch")
        passed += 1
        split_passed[split] += 1
    return passed, total, split_passed


def main() -> None:
    validate_task_structure()
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
