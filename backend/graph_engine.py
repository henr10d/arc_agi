from __future__ import annotations

from typing import Any

import numpy as np

DTYPES: dict[str, Any] = {
    "bool": np.bool_,
    "boolean": np.bool_,
    "float": np.float32,
    "float32": np.float32,
    "float64": np.float64,
    "int": np.int64,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
}


def as_dtype(name: str | None) -> Any:
    if not name:
        return np.int64
    key = str(name).lower()
    if key not in DTYPES:
        raise ValueError(f"Unsupported dtype: {name}")
    return DTYPES[key]


def constant_to_array(spec: dict[str, Any]) -> np.ndarray:
    value = spec.get("value")
    dtype = as_dtype(spec.get("dtype", "int64"))
    return np.asarray(value, dtype=dtype)


def _axes_from(value: Any) -> tuple[int, ...] | None:
    if value is None or value == "":
        return None
    if isinstance(value, np.ndarray):
        return tuple(int(x) for x in value.flatten().tolist())
    if isinstance(value, list):
        return tuple(int(x) for x in value)
    return (int(value),)


def _to_grid(value: np.ndarray) -> list[Any]:
    arr = np.asarray(value)
    arr = np.squeeze(arr)
    if arr.dtype == np.bool_:
        arr = arr.astype(np.int64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    return arr.tolist()


def execute_graph(graph: dict[str, Any], input_grid: list[list[int]]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    values: dict[str, np.ndarray] = {"input": np.asarray(input_grid, dtype=np.int64)}
    for name, spec in graph.get("constants", {}).items():
        values[name] = constant_to_array(spec)

    for node in graph.get("nodes", []):
        node_id = node["id"]
        op = node["op"]
        inputs = node.get("inputs", []) or []
        attrs = node.get("attrs", {}) or {}

        def val(index: int) -> np.ndarray:
            if index >= len(inputs):
                raise ValueError(f"{node_id} ({op}) missing input {index}")
            name = inputs[index]
            if name not in values:
                raise ValueError(f"{node_id} ({op}) references unknown input {name!r}")
            return values[name]

        if op == "Constant":
            values[node_id] = constant_to_array({
                "dtype": attrs.get("dtype", node.get("dtype", "int64")),
                "value": attrs.get("value", node.get("value", 0)),
            })
        elif op == "Cast":
            values[node_id] = val(0).astype(as_dtype(attrs.get("dtype") or attrs.get("to")))
        elif op == "Identity":
            values[node_id] = np.array(val(0), copy=True)
        elif op == "Equal":
            values[node_id] = np.equal(val(0), val(1))
        elif op == "Greater":
            values[node_id] = np.greater(val(0), val(1))
        elif op == "Less":
            values[node_id] = np.less(val(0), val(1))
        elif op == "Not":
            values[node_id] = np.logical_not(val(0))
        elif op == "And":
            values[node_id] = np.logical_and(val(0), val(1))
        elif op == "Or":
            values[node_id] = np.logical_or(val(0), val(1))
        elif op == "Where":
            values[node_id] = np.where(val(0), val(1), val(2))
        elif op == "ReduceSum":
            axes = _axes_from(attrs.get("axes"))
            keepdims = bool(int(attrs.get("keepdims", 1)))
            values[node_id] = np.sum(val(0), axis=axes, keepdims=keepdims)
        elif op == "ArgMax":
            axis = int(attrs.get("axis", 0))
            keepdims = bool(int(attrs.get("keepdims", 1)))
            out = np.argmax(val(0), axis=axis)
            values[node_id] = np.expand_dims(out, axis=axis) if keepdims else out
        elif op == "Slice":
            data = val(0)
            starts = val(1).astype(int).flatten() if len(inputs) > 1 else np.asarray(attrs.get("starts", []), dtype=int)
            ends = val(2).astype(int).flatten() if len(inputs) > 2 else np.asarray(attrs.get("ends", []), dtype=int)
            axes = val(3).astype(int).flatten() if len(inputs) > 3 else np.asarray(attrs.get("axes", range(len(starts))), dtype=int)
            steps = val(4).astype(int).flatten() if len(inputs) > 4 else np.asarray(attrs.get("steps", [1] * len(starts)), dtype=int)
            slices = [slice(None)] * data.ndim
            for start, end, axis, step in zip(starts, ends, axes, steps):
                slices[int(axis)] = slice(int(start), int(end), int(step))
            values[node_id] = data[tuple(slices)]
        elif op == "Pad":
            data = val(0)
            pads = val(1).astype(int).flatten() if len(inputs) > 1 else np.asarray(attrs.get("pads", []), dtype=int)
            if pads.size != data.ndim * 2:
                raise ValueError(f"{node_id} Pad requires {data.ndim * 2} pad values")
            pad_width = [(int(pads[i]), int(pads[i + data.ndim])) for i in range(data.ndim)]
            mode = attrs.get("mode", "constant")
            constant_value = float(attrs.get("constant_value", attrs.get("value", 0)))
            values[node_id] = np.pad(data, pad_width, mode=mode, constant_values=constant_value)
        elif op == "Reshape":
            shape = val(1).astype(int).flatten() if len(inputs) > 1 else np.asarray(attrs.get("shape", []), dtype=int)
            values[node_id] = np.reshape(val(0), tuple(int(x) for x in shape))
        elif op == "Concat":
            axis = int(attrs.get("axis", 0))
            values[node_id] = np.concatenate([values[name] for name in inputs], axis=axis)
        elif op == "Gather":
            axis = int(attrs.get("axis", 0))
            values[node_id] = np.take(val(0), val(1).astype(int), axis=axis)
        elif op == "Unsqueeze":
            axes = _axes_from(val(1) if len(inputs) > 1 else attrs.get("axes")) or ()
            out = val(0)
            for axis in sorted(axes):
                out = np.expand_dims(out, axis=int(axis))
            values[node_id] = out
        elif op == "Squeeze":
            axes = _axes_from(val(1) if len(inputs) > 1 else attrs.get("axes"))
            values[node_id] = np.squeeze(val(0), axis=axes)
        else:
            raise ValueError(f"Unsupported op: {op}")

    output_name = graph.get("output")
    if not output_name:
        raise ValueError("Graph is missing output")
    if output_name not in values:
        raise ValueError(f"Graph output {output_name!r} was not produced")
    return values[output_name], values


def run_graph(graph: dict[str, Any], input_grid: list[list[int]], expected_grid: list[list[int]] | None = None) -> dict[str, Any]:
    output, values = execute_graph(graph, input_grid)
    predicted = _to_grid(output)
    result: dict[str, Any] = {
        "predicted": predicted,
        "shape": list(np.asarray(predicted).shape),
        "tensors": {name: list(arr.shape) for name, arr in values.items()},
    }
    if expected_grid is not None:
        expected = np.asarray(expected_grid)
        pred = np.asarray(predicted)
        result["matches"] = bool(pred.shape == expected.shape and np.array_equal(pred, expected))
        result["mismatch_count"] = int(np.sum(pred != expected)) if pred.shape == expected.shape else None
    return result
