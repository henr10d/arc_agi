"""Compact ONNX for ARC task067 horizontal-period compression.

Task rule: each input is an H by 3H grid made by repeating a horizontal
motif three times across every row.  The output is the leftmost clean H by H
copy of that motif, preserving all colors including black.

The ONNX graph never converts out of one-hot form.  It slices the maximum
possible 5x5 output area, detects whether rows 2, 3, and 4 are real grid rows
from the padded one-hot input, masks columns beyond H, and pads directly to the
required 30x30 output tensor.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import (  # noqa: E402
    calculate_memory,
    calculate_params,
    convert_to_numpy,
    sanitize_model,
    score,
    score_file,
)

TASK_ID = "task067"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task067.onnx"
SOLUTION_PATH = ROOT / "solution.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
MAX_SIDE = 5
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
DEFAULT_OPSET = 10
IR_VERSION = 10


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _bool(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.bool_), name)


def _make_model(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    name: str,
    *,
    opset: int = DEFAULT_OPSET,
) -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(nodes, name, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_slice_mask_model() -> onnx.ModelProto:
    """Slice [1,10,5,5], mask columns >= active height, then final-pad."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    # A real row has one active channel in column 0.  Padded rows are all zero.
    detect_starts = _i64(inits, [0, 0, 2, 0], "detect_starts")
    detect_ends = _i64(inits, [1, C, MAX_SIDE, 1], "detect_ends")
    zero = _f32(inits, [0.0], "zero")
    one_bool = _bool(inits, np.ones((1, 1, 1, 1), dtype=np.bool_), "one_bool")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, detect_starts, detect_ends], ["row_probe"]),
            helper.make_node("ReduceSum", ["row_probe"], ["row_sum"], axes=[1, 3], keepdims=1),
            helper.make_node("Greater", ["row_sum", zero], ["row_exists"]),
            helper.make_node("Split", ["row_exists"], ["has_col2", "has_col3", "has_col4"], axis=2, split=[1, 1, 1]),
            helper.make_node(
                "Concat",
                [one_bool, one_bool, "has_col2", "has_col3", "has_col4"],
                ["col_mask_bool"],
                axis=3,
            ),
            helper.make_node("Cast", ["col_mask_bool"], ["col_mask"], to=TensorProto.FLOAT),
        ]
    )

    core_starts = _i64(inits, [0, 0, 0, 0], "core_starts")
    core_ends = _i64(inits, [1, C, MAX_SIDE, MAX_SIDE], "core_ends")
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, core_starts, core_ends], ["core5"]),
            helper.make_node("Mul", ["core5", "col_mask"], ["out5"]),
            helper.make_node(
                "Pad",
                ["out5"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - MAX_SIDE, W - MAX_SIDE],
            ),
        ]
    )

    return _make_model(nodes, inits, f"{TASK_ID}_slice_mask")


def build_slice_attr_model() -> onnx.ModelProto:
    """Same graph using opset-9 Slice attrs to avoid start/end initializers."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    zero = _f32(inits, [0.0], "zero")
    one_bool = _bool(inits, np.ones((1, 1, 1, 1), dtype=np.bool_), "one_bool")

    nodes.extend(
        [
            helper.make_node(
                "Slice",
                [IN_NAME],
                ["row_probe"],
                starts=[0, 0, 2, 0],
                ends=[1, C, MAX_SIDE, 1],
                axes=[0, 1, 2, 3],
            ),
            helper.make_node("ReduceSum", ["row_probe"], ["row_sum"], axes=[1, 3], keepdims=1),
            helper.make_node("Greater", ["row_sum", zero], ["row_exists"]),
            helper.make_node("Split", ["row_exists"], ["has_col2", "has_col3", "has_col4"], axis=2, split=[1, 1, 1]),
            helper.make_node(
                "Concat",
                [one_bool, one_bool, "has_col2", "has_col3", "has_col4"],
                ["col_mask_bool"],
                axis=3,
            ),
            helper.make_node("Cast", ["col_mask_bool"], ["col_mask"], to=TensorProto.FLOAT),
            helper.make_node(
                "Slice",
                [IN_NAME],
                ["core5"],
                starts=[0, 0, 0, 0],
                ends=[1, C, MAX_SIDE, MAX_SIDE],
                axes=[0, 1, 2, 3],
            ),
            helper.make_node("Mul", ["core5", "col_mask"], ["out5"]),
            helper.make_node(
                "Pad",
                ["out5"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - MAX_SIDE, W - MAX_SIDE],
            ),
        ]
    )

    return _make_model(nodes, inits, f"{TASK_ID}_slice_attr", opset=9)


def build_slice_where_model() -> onnx.ModelProto:
    """Use a bool width mask directly with Where, avoiding a float mask tensor."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    zero = _f32(inits, [0.0], "zero")
    one_bool = _bool(inits, np.ones((1, 1, 1, 1), dtype=np.bool_), "one_bool")

    nodes.extend(
        [
            helper.make_node(
                "Slice",
                [IN_NAME],
                ["row_probe"],
                starts=[0, 0, 2, 0],
                ends=[1, C, MAX_SIDE, 1],
                axes=[0, 1, 2, 3],
            ),
            helper.make_node("ReduceSum", ["row_probe"], ["row_sum"], axes=[1, 3], keepdims=1),
            helper.make_node("Greater", ["row_sum", zero], ["row_exists"]),
            helper.make_node("Split", ["row_exists"], ["has_col2", "has_col3", "has_col4"], axis=2, split=[1, 1, 1]),
            helper.make_node(
                "Concat",
                [one_bool, one_bool, "has_col2", "has_col3", "has_col4"],
                ["col_mask_bool"],
                axis=3,
            ),
            helper.make_node(
                "Slice",
                [IN_NAME],
                ["core5"],
                starts=[0, 0, 0, 0],
                ends=[1, C, MAX_SIDE, MAX_SIDE],
                axes=[0, 1, 2, 3],
            ),
            helper.make_node("Where", ["col_mask_bool", "core5", zero], ["out5"]),
            helper.make_node(
                "Pad",
                ["out5"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - MAX_SIDE, W - MAX_SIDE],
            ),
        ]
    )

    return _make_model(nodes, inits, f"{TASK_ID}_slice_where", opset=9)


def _load_task() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _check_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, int]]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return False, {}
    session = ort.InferenceSession(
        sanitized.SerializeToString(),
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    )
    counts: dict[str, int] = {}
    for split, examples in _load_task().items():
        counts[split] = 0
        for example in examples:
            inp = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if inp is None or expected is None:
                continue
            actual = session.run([OUT_NAME], {IN_NAME: inp})[0]
            if not np.array_equal(actual > 0.0, expected > 0.0):
                return False, counts
            counts[split] += 1
    return True, counts


def _profile_model(model: onnx.ModelProto) -> tuple[int | None, int | None]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return None, None
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(Path(tempfile.gettempdir()) / f"ng_{TASK_ID}")
    session = ort.InferenceSession(
        sanitized.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    inputs = [
        arr
        for split in _load_task().values()
        for example in split
        if (arr := convert_to_numpy(example, "input")) is not None
    ]
    for arr in inputs:
        session.run([OUT_NAME], {IN_NAME: arr})
    trace_path = session.end_profiling()
    return calculate_memory(sanitized, trace_path), calculate_params(sanitized)


def _internal_shapes(model: onnx.ModelProto) -> list[tuple[str, str, list[int]]]:
    graph = onnx.shape_inference.infer_shapes(sanitize_model(copy.deepcopy(model)), strict_mode=True).graph
    rows: list[tuple[str, str, list[int]]] = []
    for item in graph.value_info:
        tensor_type = item.type.tensor_type
        dtype = TensorProto.DataType.Name(tensor_type.elem_type)
        shape = [dim.dim_value for dim in tensor_type.shape.dim]
        rows.append((item.name, dtype, shape))
    return rows


def main() -> None:
    variants = {
        "slice_mask_opset10": build_slice_mask_model(),
        "slice_attr_opset9": build_slice_attr_model(),
        "slice_where_opset9": build_slice_where_model(),
    }
    results: list[tuple[int, str, onnx.ModelProto, int, int]] = []

    for name, model in variants.items():
        correct, counts = _check_correct(model)
        if not correct:
            print(f"{name}: incorrect after {counts}")
            continue
        memory, params = _profile_model(model)
        if memory is None or params is None:
            print(f"{name}: invalid score measurement")
            continue
        cost = memory + params
        results.append((cost, name, model, memory, params))
        print(f"{name}: memory={memory} params={params} cost={cost} score={score(cost):.6f}")

    if not results:
        raise SystemExit("no valid task067 variants")

    cost, name, model, memory, params = min(results, key=lambda item: item[0])
    onnx.save(model, BEST_PATH)
    onnx.save(model, SOLUTION_PATH)
    score_result = score_file(SOLUTION_PATH)

    print()
    print(f"kept: {name}")
    print(f"wrote: {BEST_PATH.relative_to(ROOT)}")
    print(f"wrote: {SOLUTION_PATH.relative_to(ROOT)}")
    print(f"verified examples: {_check_correct(model)[1]}")
    print(f"official-style memory={score_result['memory']} params={score_result['params']} cost={score_result['cost']} score={score_result['score']:.6f}")
    print("inferred internal tensor shapes:")
    for tensor_name, dtype, shape in _internal_shapes(model):
        elements = math.prod(shape)
        print(f"  {tensor_name}: {dtype} {shape} ({elements} elems)")


if __name__ == "__main__":
    main()
