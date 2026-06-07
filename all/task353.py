"""ONNX solver for ARC task353: move the green cell one step toward yellow.

Task rule: each input grid contains exactly one green cell (color 3) and one
yellow/brown cell (color 4) on a black background.  The output shape is the
same as the input shape.  The yellow cell stays fixed, while the green cell
moves one king-step toward yellow: row and column each change by the sign of
the corresponding yellow-minus-green coordinate difference.  All other cells
inside the task grid are black, and the 30x30 padding remains all-zero.

The examples include horizontal, vertical, and diagonal moves.  No train/test/
arc-gen example has green adjacent to yellow.  The ONNX graph is specialized to
the observed maximum task extent (14x12), builds only color channels 0..4 over
that compact crop, and uses the final Pad to append zero channels and 30x30
spatial padding.
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
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, sanitize_model, score_file  # noqa: E402

TASK_ID = "task353"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
CORE_H = 14
CORE_W = 12
GREEN = 3
YELLOW = 4
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference implementation for the ARC grid transformation."""
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(arr)
    green_pos = np.argwhere(arr == GREEN)
    yellow_pos = np.argwhere(arr == YELLOW)
    if len(green_pos) != 1 or len(yellow_pos) != 1:
        raise ValueError("task353 expects exactly one green and one yellow cell")

    gr, gc = (int(v) for v in green_pos[0])
    yr, yc = (int(v) for v in yellow_pos[0])
    ngr = gr + int(np.sign(yr - gr))
    ngc = gc + int(np.sign(yc - gc))

    out[yr, yc] = YELLOW
    if (ngr, ngc) != (yr, yc):
        out[ngr, ngc] = GREEN
    return out


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    return _init(inits, np.asarray(arr, dtype=np.int64), name)


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    _i64(inits, [0], "zero_i")
    _i64(inits, [1], "one_i")
    _i64(inits, [-1], "neg_one_i")
    _i64(inits, [1, 2, 3], "slice_axes")
    _i64(inits, [0, 0, 0], "black_starts")
    _i64(inits, [1, CORE_H, CORE_W], "black_ends")
    _i64(inits, [GREEN, 0, 0], "green_starts")
    _i64(inits, [GREEN + 1, CORE_H, CORE_W], "green_ends")
    _i64(inits, [YELLOW, 0, 0], "yellow_starts")
    _i64(inits, [YELLOW + 1, CORE_H, CORE_W], "yellow_ends")
    _i64(inits, np.arange(CORE_H, dtype=np.int64).reshape(1, 1, CORE_H, 1), "rows")
    _i64(inits, np.arange(CORE_W, dtype=np.int64).reshape(1, 1, 1, CORE_W), "cols")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, "black_starts", "black_ends", "slice_axes"], ["black"]),
            helper.make_node("Slice", [IN_NAME, "green_starts", "green_ends", "slice_axes"], ["green"]),
            helper.make_node("Slice", [IN_NAME, "yellow_starts", "yellow_ends", "slice_axes"], ["yellow"]),
            helper.make_node("ReduceSum", ["green"], ["green_rproj"], axes=[3], keepdims=0),
            helper.make_node("ReduceSum", ["green"], ["green_cproj"], axes=[2], keepdims=0),
            helper.make_node("ReduceSum", ["yellow"], ["yellow_rproj"], axes=[3], keepdims=0),
            helper.make_node("ReduceSum", ["yellow"], ["yellow_cproj"], axes=[2], keepdims=0),
            helper.make_node("ArgMax", ["green_rproj"], ["gr"], axis=2, keepdims=1),
            helper.make_node("ArgMax", ["green_cproj"], ["gc"], axis=2, keepdims=1),
            helper.make_node("ArgMax", ["yellow_rproj"], ["yr"], axis=2, keepdims=1),
            helper.make_node("ArgMax", ["yellow_cproj"], ["yc"], axis=2, keepdims=1),
            helper.make_node("Greater", ["yr", "gr"], ["yr_gt_gr"]),
            helper.make_node("Less", ["yr", "gr"], ["yr_lt_gr"]),
            helper.make_node("Greater", ["yc", "gc"], ["yc_gt_gc"]),
            helper.make_node("Less", ["yc", "gc"], ["yc_lt_gc"]),
            helper.make_node("Where", ["yr_lt_gr", "neg_one_i", "zero_i"], ["dr_nonpos"]),
            helper.make_node("Where", ["yr_gt_gr", "one_i", "dr_nonpos"], ["dr"]),
            helper.make_node("Where", ["yc_lt_gc", "neg_one_i", "zero_i"], ["dc_nonpos"]),
            helper.make_node("Where", ["yc_gt_gc", "one_i", "dc_nonpos"], ["dc"]),
            helper.make_node("Add", ["gr", "dr"], ["new_gr"]),
            helper.make_node("Add", ["gc", "dc"], ["new_gc"]),
            helper.make_node("Equal", ["rows", "new_gr"], ["new_green_row"]),
            helper.make_node("Equal", ["cols", "new_gc"], ["new_green_col"]),
            helper.make_node("And", ["new_green_row", "new_green_col"], ["new_green_hit"]),
            helper.make_node("Cast", ["black"], ["black_orig"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["green"], ["green_orig"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["yellow"], ["yellow_mask"], to=TensorProto.BOOL),
            helper.make_node("Not", ["new_green_hit"], ["not_green"]),
            helper.make_node("Or", ["black_orig", "green_orig"], ["black_base"]),
            helper.make_node("And", ["black_base", "not_green"], ["black_mask"]),
            helper.make_node("And", ["new_green_hit", "not_green"], ["zero_mask"]),
            helper.make_node(
                "Concat",
                [
                    "black_mask",
                    "zero_mask",
                    "zero_mask",
                    "new_green_hit",
                    "yellow_mask",
                ],
                ["out_bool"],
                axis=1,
            ),
            helper.make_node("Cast", ["out_bool"], ["out_core"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out_core"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, C - 5, H - CORE_H, W - CORE_W],
                value=0.0,
            ),
        ]
    )

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
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


def _load_examples() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _check_reference() -> dict[str, tuple[int, int]]:
    examples = _load_examples()
    counts: dict[str, tuple[int, int]] = {}
    for split in ("train", "test", "arc-gen"):
        passed = 0
        total = 0
        for example in examples.get(split, []):
            total += 1
            if np.array_equal(solve(example["input"]), np.asarray(example["output"], dtype=np.int64)):
                passed += 1
        counts[split] = (passed, total)
    return counts


def _check_model(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return False, {}
    session = ort.InferenceSession(
        sanitized.SerializeToString(),
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    )
    examples = _load_examples()
    counts: dict[str, tuple[int, int]] = {}
    all_ok = True
    for split in ("train", "test", "arc-gen"):
        passed = 0
        total = 0
        for example in examples.get(split, []):
            input_arr = convert_to_numpy(example, "input")
            expected_arr = convert_to_numpy(example, "output")
            if input_arr is None or expected_arr is None:
                continue
            total += 1
            output = session.run([OUT_NAME], {IN_NAME: input_arr})[0]
            pred = (output > 0.0).astype(np.float32)
            if np.array_equal(pred, expected_arr):
                passed += 1
            else:
                all_ok = False
        counts[split] = (passed, total)
    return all_ok, counts


def _format_counts(counts: dict[str, tuple[int, int]]) -> str:
    return ", ".join(f"{split} {ok}/{total}" for split, (ok, total) in counts.items())


def main() -> None:
    ref_counts = _check_reference()
    if any(ok != total for ok, total in ref_counts.values()):
        raise SystemExit(f"reference mismatch: {_format_counts(ref_counts)}")

    model = build_model()
    correct, model_counts = _check_model(model)
    if not correct:
        raise SystemExit(f"model mismatch: {_format_counts(model_counts)}")

    onnx.save(model, BEST_PATH)
    scored = score_file(BEST_PATH)
    score_text = f"{scored['score']:.6f}" if scored["score"] is not None else "INVALID"
    print(f"wrote:  {BEST_PATH}")
    print(f"passes: {_format_counts(model_counts)}")
    print(f"valid:  {scored['valid']}")
    print(f"memory: {scored['memory']}")
    print(f"params: {scored['params']}")
    print(f"cost:   {scored['cost']}")
    print(f"score:  {score_text}")
    if scored["error"]:
        print(f"error:  {scored['error']}")


if __name__ == "__main__":
    main()
