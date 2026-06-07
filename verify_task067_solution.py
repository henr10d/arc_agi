"""Verify solution.onnx for task067 against public train/test/arc-gen examples.

The task transforms an H by 3H horizontally repeated pattern into the leftmost
clean H by H period, preserving black cells when they are part of the motif.
"""

from __future__ import annotations

import copy
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from score_model import calculate_memory, calculate_params, convert_to_numpy, sanitize_model, score, score_file


ROOT = Path(__file__).resolve().parent
TASK_PATH = ROOT / "data" / "task067.json"
MODEL_PATH = ROOT / "solution.onnx"


def internal_shapes(model: onnx.ModelProto) -> list[tuple[str, str, list[int]]]:
    graph = onnx.shape_inference.infer_shapes(sanitize_model(copy.deepcopy(model)), strict_mode=True).graph
    rows: list[tuple[str, str, list[int]]] = []
    for item in graph.value_info:
        tensor_type = item.type.tensor_type
        dtype = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
        shape = [dim.dim_value for dim in tensor_type.shape.dim]
        rows.append((item.name, dtype, shape))
    return rows


def main() -> None:
    model = onnx.load(str(MODEL_PATH))
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise SystemExit("solution.onnx failed NeuroGolf sanitization")

    session = ort.InferenceSession(
        sanitized.SerializeToString(),
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    )
    with TASK_PATH.open(encoding="utf-8") as fh:
        task = json.load(fh)

    counts: dict[str, int] = {}
    inputs: list[np.ndarray] = []
    for split, examples in task.items():
        counts[split] = 0
        for example in examples:
            inp = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if inp is None or expected is None:
                continue
            actual = session.run(["output"], {"input": inp})[0]
            if not np.array_equal(actual > 0.0, expected > 0.0):
                raise SystemExit(f"mismatch in {split} example {counts[split]}")
            counts[split] += 1
            inputs.append(inp)

    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(Path(tempfile.gettempdir()) / "ng_verify_task067")
    profiled = ort.InferenceSession(
        sanitized.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    for inp in inputs:
        profiled.run(["output"], {"input": inp})
    trace_path = profiled.end_profiling()

    memory = calculate_memory(sanitized, trace_path)
    params = calculate_params(sanitized)
    if memory is None or params is None:
        raise SystemExit("could not measure memory/params")
    cost = memory + params
    official = score_file(MODEL_PATH)

    print(f"verified examples: {counts}")
    print(f"params: {params}")
    print(f"inferred memory: {memory}")
    print(f"cost: {cost}")
    print(f"score: {score(cost):.6f}")
    print(f"score_model valid: {official['valid']}")
    print("internal tensor shapes:")
    for name, dtype, shape in internal_shapes(model):
        print(f"  {name}: {dtype} {shape} ({math.prod(shape)} elems)")


if __name__ == "__main__":
    main()
