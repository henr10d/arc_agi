from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


BATCH_SIZE = 1
CHANNELS = 10
HEIGHT = WIDTH = 30
GRID_SHAPE = (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
EXCLUDED_OP_TYPES = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
FILESIZE_LIMIT_IN_BYTES = 1.44 * 1024 * 1024
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def score(cost: int | float) -> float:
    return max(1.0, 25.0 - math.log(max(1.0, float(cost))))


def sanitize_model(model: onnx.ModelProto) -> onnx.ModelProto | None:
    """Match the official NeuroGolf sanitizer before measurement."""
    model = copy.deepcopy(model)
    for node in model.graph.node:
        if not node.output:
            return None
        node.name = node.output[0]
        if "kernel_time" in node.output[0]:
            return None

    name_map: dict[str, str] = {}
    counter = 0

    def safe_name(old_name: str) -> str:
        nonlocal counter
        if not old_name or old_name in {"input", "output"}:
            return old_name
        if old_name not in name_map:
            name_map[old_name] = f"safe_name_{counter}"
            counter += 1
        return name_map[old_name]

    for inp in model.graph.input:
        inp.name = safe_name(inp.name)
    for init in model.graph.initializer:
        init.name = safe_name(init.name)
    for node in model.graph.node:
        for idx in range(len(node.input)):
            node.input[idx] = safe_name(node.input[idx])
        for idx in range(len(node.output)):
            node.output[idx] = safe_name(node.output[idx])
        if node.output and node.output[0]:
            node.name = node.output[0]
    for out in model.graph.output:
        out.name = safe_name(out.name)
    for value_info in model.graph.value_info:
        value_info.name = safe_name(value_info.name)
    for node in model.graph.node:
        node.name = node.output[0]
    return model


def calculate_params(model: onnx.ModelProto) -> int | None:
    params = 0
    for init in model.graph.initializer:
        if any(dim <= 0 for dim in init.dims):
            return None
        params += math.prod(init.dims)
    for sparse_init in model.graph.sparse_initializer:
        if any(dim <= 0 for dim in sparse_init.values.dims):
            return None
        params += math.prod(sparse_init.values.dims)
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                if any(dim <= 0 for dim in attr.t.dims):
                    return None
                params += math.prod(attr.t.dims)
            elif attr.name == "sparse_value":
                if any(dim <= 0 for dim in attr.sparse_tensor.values.dims):
                    return None
                params += math.prod(attr.sparse_tensor.values.dims)
            elif attr.name == "value_floats":
                params += len(attr.floats)
            elif attr.name == "value_ints":
                params += len(attr.ints)
            elif attr.name == "value_strings":
                params += len(attr.strings)
    return int(params)


def calculate_memory(model: onnx.ModelProto, trace_path: str) -> int | None:
    """Match the official NeuroGolf memory score: internal tensor bytes only."""
    try:
        onnx.checker.check_model(model, full_check=True)
        graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    except Exception:
        return None

    if len(graph.input) > 1 or len(graph.output) > 1:
        return None

    init_names = {init.name for init in graph.initializer}
    init_names.update(init.name for init in graph.sparse_initializer)
    io_names = {tensor.name for tensor in list(graph.input) + list(graph.output)}
    if io_names.intersection(init_names) or model.functions:
        return None
    for opset in model.opset_import:
        if opset.domain not in {"", "ai.onnx"}:
            return None

    node_outputs: dict[str, list[str]] = {}
    tensor_names: set[str] = set()
    for node in graph.node:
        for attr in node.attribute:
            if attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}:
                return None
        node_outputs[node.name] = list(node.output)
        for output_name in node.output:
            if output_name:
                tensor_names.add(output_name)

    tensor_memory: dict[str, int] = {}
    tensor_dtypes: dict[str, np.dtype[Any]] = {}
    tensor_map = {tensor.name: tensor for tensor in list(graph.input) + list(graph.value_info) + list(graph.output)}
    tensor_names.update(tensor_map.keys())
    for tensor_name in tensor_names:
        item = tensor_map.get(tensor_name)
        if not item:
            return None
        if item.type.HasField("sequence_type"):
            return None
        if not item.type.HasField("tensor_type"):
            continue
        tensor_type = item.type.tensor_type
        if not tensor_type.HasField("shape"):
            return None
        num_elements = 1
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0:
                return None
            num_elements *= dim.dim_value
        if tensor_name in {"input", "output"}:
            continue
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        tensor_memory[tensor_name] = int(num_elements * np.dtype(np_dtype).itemsize)
        tensor_dtypes[tensor_name] = np_dtype

    seen: set[str] = set()
    for item in list(graph.input) + list(graph.value_info) + list(graph.output):
        if item.name in seen:
            return None
        seen.add(item.name)
    for node in graph.node:
        for output_name in node.output:
            if output_name and output_name != "output":
                item = tensor_map.get(output_name)
                if item is None or not item.type.HasField("tensor_type"):
                    return None

    with open(trace_path, encoding="utf-8") as fh:
        trace_data = json.load(fh)
    for event in trace_data:
        if event.get("cat") != "Node" or "args" not in event:
            continue
        if "output_type_shape" not in event["args"]:
            continue
        node_name = event.get("name", "").replace("_kernel_time", "")
        if node_name not in node_outputs:
            continue
        for idx, shape_dict in enumerate(event["args"]["output_type_shape"]):
            if idx >= len(node_outputs[node_name]):
                continue
            output_name = node_outputs[node_name][idx]
            if output_name not in tensor_dtypes:
                continue
            itemsize = np.dtype(tensor_dtypes[output_name]).itemsize
            mem = itemsize * sum(math.prod(dims) for dims in shape_dict.values())
            tensor_memory[output_name] = max(tensor_memory[output_name], int(mem))

    return sum(tensor_memory.values())


def convert_to_numpy(example: dict[str, list[list[int]]], mode: str = "input") -> np.ndarray | None:
    grid = example[mode]
    if max(len(grid), len(grid[0])) > 30:
        return None
    out = np.zeros(GRID_SHAPE, dtype=np.float32)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            out[0, int(color), row, col] = 1.0
    return out


def load_task_examples(path: Path) -> list[np.ndarray]:
    task_path = DATA_DIR / f"{path.stem}.json"
    if not task_path.is_file():
        return [np.zeros(GRID_SHAPE, dtype=np.float32)]

    with task_path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    inputs: list[np.ndarray] = []
    for split in ("train", "test", "arc-gen"):
        for example in data.get(split, []):
            arr = convert_to_numpy(example, "input")
            if arr is not None:
                inputs.append(arr)
    return inputs or [np.zeros(GRID_SHAPE, dtype=np.float32)]


def run_profiled_session(model: onnx.ModelProto, inputs: list[np.ndarray], path: Path) -> tuple[str | None, str | None]:
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(Path(tempfile.gettempdir()) / f"ng_score_{path.stem}")
    try:
        session = ort.InferenceSession(
            model.SerializeToString(),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        for arr in inputs:
            session.run(["output"], {"input": arr})
        return session.end_profiling(), None
    except Exception:
        return None, traceback.format_exc()


def score_file(path: Path) -> dict[str, Any]:
    filesize = path.stat().st_size
    result: dict[str, Any] = {
        "path": path,
        "filesize": filesize,
        "memory": None,
        "params": None,
        "cost": None,
        "score": None,
        "valid": False,
        "error": None,
        "examples_profiled": 0,
    }
    if filesize > FILESIZE_LIMIT_IN_BYTES:
        result["error"] = f"filesize {filesize} exceeds {FILESIZE_LIMIT_IN_BYTES:.0f}"
        return result

    try:
        model = onnx.load(str(path))
    except Exception:
        result["error"] = traceback.format_exc()
        return result

    sanitized = sanitize_model(model)
    if sanitized is None:
        result["error"] = "model failed NeuroGolf name sanitization"
        return result

    for node in sanitized.graph.node:
        if node.op_type.upper() in EXCLUDED_OP_TYPES or "Sequence" in node.op_type:
            result["error"] = f"op type {node.op_type} is not permitted"
            return result

    inputs = load_task_examples(path)
    result["examples_profiled"] = len(inputs)
    trace_path, error = run_profiled_session(sanitized, inputs, path)
    if error or trace_path is None:
        result["error"] = error
        return result

    memory = calculate_memory(sanitized, trace_path)
    params = calculate_params(sanitized)
    result["memory"] = memory
    result["params"] = params
    if memory is None or params is None or memory < 0 or params < 0:
        result["error"] = "network performance could not be measured"
        return result

    cost = int(memory + params)
    result["cost"] = cost
    result["score"] = score(cost)
    result["valid"] = True
    return result


def print_report(result: dict[str, Any]) -> None:
    path = result["path"]
    assert isinstance(path, Path)
    print(path.name)
    print("----------------------------------------")
    print(f"filesize:          {result['filesize']}")
    print(f"profiled inputs:   {result['examples_profiled']}")
    if not result["valid"]:
        print("score:             INVALID")
        if result["error"]:
            print(f"error:             {str(result['error']).strip()}")
        return
    print(f"official memory:   {result['memory']}")
    print(f"official params:   {result['params']}")
    print(f"official cost:     {result['cost']}")
    print(f"official score:    {result['score']:.6f}")


def collect_onnx_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*.onnx") if p.is_file())
    raise FileNotFoundError(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate official NeuroGolf score for ONNX files.")
    parser.add_argument("path", type=Path, help="ONNX file or directory to scan recursively")
    args = parser.parse_args()

    paths = collect_onnx_paths(args.path)
    if not paths:
        raise SystemExit(f"no ONNX files found under {args.path}")

    results = [score_file(path) for path in paths]

    for index, result in enumerate(results):
        if index:
            print()
        print_report(result)

    if len(results) > 1:
        valid_results = [result for result in results if result["valid"]]
        total = sum(float(result["score"]) for result in valid_results)
        worst = sorted(valid_results, key=lambda result: int(result["cost"]), reverse=True)[:3]

        print()
        print(f"VALID MODELS: {len(valid_results)}/{len(results)}")
        print(f"TOTAL SCORE:  {total:.6f}")
        print()
        print("Worst tasks:")
        for result in worst:
            path = result["path"]
            assert isinstance(path, Path)
            print(f"{path.name:<15} cost={result['cost']} score={result['score']:.6f}")


if __name__ == "__main__":
    main()
