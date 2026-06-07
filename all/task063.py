"""Minimal ONNX for ARC task063 using empty row/column corridor detection.

Task rule: grids use black background with red/cyan boundary structure. Preserve
all original non-black cells, and recolor black cells green when their row or
column has exactly two non-black boundary cells. Equivalently, fill the empty
interior rows and columns that are supported only by the two opposite walls.

The local data for this task has 10x10, 12x12, and 14x14 examples, so the graph
works on the top-left 14x14 window. It keeps the row/column logic compact and
pads the final fill mask as float16 before casting back to bool for the output
Where, avoiding a large float32 padded mask.
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
    load_task_examples,
    sanitize_model,
    score_file,
)

TASK_ID = "task063"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task063.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
K = 14
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
IR_VERSION = 10


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _make_model(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    name: str,
    opset: int,
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


def _slice_color(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], color: int, name: str, opset: int) -> None:
    if opset <= 9:
        nodes.append(
            helper.make_node(
                "Slice",
                [IN_NAME],
                [name],
                axes=[0, 1, 2, 3],
                starts=[0, color, 0, 0],
                ends=[1, color + 1, K, K],
            )
        )
        return
    starts = _i64(inits, [0, color, 0, 0], f"{name}_starts")
    ends = _i64(inits, [1, color + 1, K, K], f"{name}_ends")
    axes = _i64(inits, [0, 1, 2, 3], f"{name}_axes")
    nodes.append(helper.make_node("Slice", [IN_NAME, starts, ends, axes], [name]))


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    opset = 9

    _slice_color(nodes, inits, 0, "black", opset)
    _slice_color(nodes, inits, 2, "red", opset)
    _slice_color(nodes, inits, 8, "cyan", opset)

    green = _f32(inits, np.eye(1, C, 3, dtype=np.float32).reshape(1, C, 1, 1), "green")

    one = _f32(inits, [1.0], "one")
    three = _f32(inits, [3.0], "three")
    nodes.extend(
        [
            helper.make_node("ReduceSum", ["red"], ["red_row"], axes=[3], keepdims=1),
            helper.make_node("ReduceSum", ["cyan"], ["cyan_row"], axes=[3], keepdims=1),
            helper.make_node("Add", ["red_row", "cyan_row"], ["row_count"]),
            helper.make_node("ReduceSum", ["red"], ["red_col"], axes=[2], keepdims=1),
            helper.make_node("ReduceSum", ["cyan"], ["cyan_col"], axes=[2], keepdims=1),
            helper.make_node("Add", ["red_col", "cyan_col"], ["col_count"]),
            helper.make_node("Greater", ["row_count", one], ["row_gt"]),
            helper.make_node("Less", ["row_count", three], ["row_lt"]),
            helper.make_node("And", ["row_gt", "row_lt"], ["row_band"]),
            helper.make_node("Greater", ["col_count", one], ["col_gt"]),
            helper.make_node("Less", ["col_count", three], ["col_lt"]),
            helper.make_node("And", ["col_gt", "col_lt"], ["col_band"]),
        ]
    )

    nodes.extend(
        [
            helper.make_node("Or", ["row_band", "col_band"], ["band"]),
            helper.make_node("Cast", ["black"], ["empty"], to=TensorProto.BOOL),
            helper.make_node("And", ["empty", "band"], ["mask14"]),
            helper.make_node("Cast", ["mask14"], ["mask14h"], to=TensorProto.FLOAT16),
            helper.make_node(
                "Pad",
                ["mask14h"],
                ["maskh"],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - K, W - K],
            ),
            helper.make_node("Cast", ["maskh"], ["mask"], to=TensorProto.BOOL),
            helper.make_node("Where", ["mask", green, IN_NAME], [OUT_NAME]),
        ]
    )
    return _make_model(nodes, inits, f"{TASK_ID}_separate_counts_f16_pad", opset)


def _load_examples() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _check_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return False, {}
    try:
        session = ort.InferenceSession(
            sanitized.SerializeToString(),
            sess_options=ort.SessionOptions(),
            providers=["CPUExecutionProvider"],
        )
    except Exception:
        return False, {}
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
            out = session.run([OUT_NAME], {IN_NAME: input_arr})[0]
            pred = (out > 0.0).astype(np.float32)
            if np.array_equal(pred, expected_arr):
                passed += 1
            else:
                all_ok = False
        counts[split] = (passed, total)
    return all_ok, counts


def _profile_largest_internal(model: onnx.ModelProto) -> tuple[int | None, str | None, int | None]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return None, None, None

    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(Path(tempfile.gettempdir()) / f"ng_{TASK_ID}_largest")
    try:
        session = ort.InferenceSession(
            sanitized.SerializeToString(),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        for arr in load_task_examples(BEST_PATH):
            session.run([OUT_NAME], {IN_NAME: arr})
        trace_path = session.end_profiling()
    except Exception:
        return None, None, None

    try:
        graph = onnx.shape_inference.infer_shapes(sanitized, strict_mode=True).graph
    except Exception:
        return None, None, None
    outputs_by_node = {node.name: list(node.output) for node in graph.node}
    dtypes = {
        info.name: onnx.helper.tensor_dtype_to_np_dtype(info.type.tensor_type.elem_type)
        for info in list(graph.value_info) + list(graph.output)
        if info.type.HasField("tensor_type")
    }
    largest_name: str | None = None
    largest_bytes = -1
    with open(trace_path, encoding="utf-8") as fh:
        trace = json.load(fh)
    for event in trace:
        if event.get("cat") != "Node" or "output_type_shape" not in event.get("args", {}):
            continue
        node_name = event.get("name", "").replace("_kernel_time", "")
        for idx, shape_dict in enumerate(event["args"]["output_type_shape"]):
            outputs = outputs_by_node.get(node_name, [])
            if idx >= len(outputs):
                continue
            output_name = outputs[idx]
            if output_name == OUT_NAME or output_name not in dtypes:
                continue
            itemsize = np.dtype(dtypes[output_name]).itemsize
            size = itemsize * sum(math.prod(dims) for dims in shape_dict.values())
            if size > largest_bytes:
                largest_name = output_name
                largest_bytes = int(size)
    memory = calculate_memory(sanitized, trace_path)
    params = calculate_params(sanitized)
    return memory, largest_name, largest_bytes


def _format_counts(counts: dict[str, tuple[int, int]]) -> str:
    return ", ".join(f"{split} {ok}/{total}" for split, (ok, total) in counts.items())


def main() -> None:
    label = "op9_sep_f16_pad_range"
    model = build_model()
    onnx.save(model, BEST_PATH)
    correct, counts = _check_correct(model)
    scored = score_file(BEST_PATH)
    memory, largest_name, largest_bytes = _profile_largest_internal(model)
    if not correct or not scored["valid"]:
        raise SystemExit(scored.get("error") or "model failed correctness")

    print()
    print(f"kept:    {label}")
    print(f"wrote:   {BEST_PATH}")
    print(f"valid:   {scored['valid']}")
    print(f"passes:  {_format_counts(counts)}")
    print(f"memory:  {scored['memory']}")
    print(f"params:  {scored['params']}")
    print(f"cost:    {scored['cost']}")
    print(f"score:   {scored['score']:.6f}")
    print(f"largest: {largest_name} ({largest_bytes} bytes)")


if __name__ == "__main__":
    main()
