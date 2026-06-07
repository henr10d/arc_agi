from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from .graph_engine import as_dtype, constant_to_array, execute_graph

ONNX_DTYPE = {
    np.dtype(np.bool_): TensorProto.BOOL,
    np.dtype(np.float32): TensorProto.FLOAT,
    np.dtype(np.float64): TensorProto.DOUBLE,
    np.dtype(np.int32): TensorProto.INT32,
    np.dtype(np.int64): TensorProto.INT64,
    np.dtype(np.uint8): TensorProto.UINT8,
}

CAST_TO = {
    "bool": TensorProto.BOOL,
    "boolean": TensorProto.BOOL,
    "float": TensorProto.FLOAT,
    "float32": TensorProto.FLOAT,
    "float64": TensorProto.DOUBLE,
    "int": TensorProto.INT64,
    "int32": TensorProto.INT32,
    "int64": TensorProto.INT64,
    "uint8": TensorProto.UINT8,
}


def _tensor_type(arr: np.ndarray) -> int:
    return ONNX_DTYPE.get(np.asarray(arr).dtype, TensorProto.INT64)


def _attr_array(attrs: dict[str, Any], name: str, dtype: Any = np.int64) -> np.ndarray:
    if name not in attrs:
        raise ValueError(f"Missing required attribute {name!r}")
    return np.asarray(attrs[name], dtype=dtype)


def _initializer(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value), name=name)


def build_model(graph: dict[str, Any], sample_input: list[list[int]] | None = None) -> onnx.ModelProto:
    initializers: list[onnx.TensorProto] = []
    nodes: list[onnx.NodeProto] = []
    const_names: set[str] = set()

    for name, spec in graph.get("constants", {}).items():
        initializers.append(_initializer(name, constant_to_array(spec)))
        const_names.add(name)

    for node in graph.get("nodes", []):
        node_id = node["id"]
        op = node["op"]
        attrs = dict(node.get("attrs", {}) or {})
        inputs = list(node.get("inputs", []) or [])

        if op == "Constant":
            tensor = np.asarray(attrs.get("value", node.get("value", 0)), dtype=as_dtype(attrs.get("dtype", node.get("dtype", "int64"))))
            nodes.append(helper.make_node("Constant", [], [node_id], name=node_id, value=_initializer(f"{node_id}_value", tensor)))
            continue

        extra_inits: list[onnx.TensorProto] = []
        onnx_attrs: dict[str, Any] = {}
        onnx_inputs = inputs

        if op == "Cast":
            dtype_name = str(attrs.get("dtype") or attrs.get("to") or "int64").lower()
            onnx_attrs["to"] = CAST_TO[dtype_name]
        elif op in {"ReduceSum", "ArgMax", "Concat", "Gather"}:
            for key in ("axis", "axes", "keepdims"):
                if key in attrs:
                    onnx_attrs[key] = attrs[key]
        elif op == "Slice":
            onnx_inputs = [inputs[0]]
            for key in ("starts", "ends", "axes", "steps"):
                if len(inputs) > len(onnx_inputs):
                    onnx_inputs.append(inputs[len(onnx_inputs)])
                elif key in attrs:
                    init_name = f"{node_id}_{key}"
                    extra_inits.append(_initializer(init_name, _attr_array(attrs, key)))
                    onnx_inputs.append(init_name)
            if len(onnx_inputs) < 3:
                raise ValueError(f"{node_id} Slice needs starts and ends")
        elif op == "Pad":
            onnx_inputs = [inputs[0]]
            if len(inputs) > 1:
                onnx_inputs.append(inputs[1])
            else:
                init_name = f"{node_id}_pads"
                extra_inits.append(_initializer(init_name, _attr_array(attrs, "pads")))
                onnx_inputs.append(init_name)
            constant_value = attrs.get("constant_value", attrs.get("value"))
            if constant_value is not None:
                init_name = f"{node_id}_constant_value"
                extra_inits.append(_initializer(init_name, np.asarray(constant_value, dtype=np.float32)))
                onnx_inputs.append(init_name)
            onnx_attrs["mode"] = attrs.get("mode", "constant")
        elif op == "Reshape":
            if len(inputs) < 2:
                init_name = f"{node_id}_shape"
                extra_inits.append(_initializer(init_name, _attr_array(attrs, "shape")))
                onnx_inputs = [inputs[0], init_name]
        elif op in {"Unsqueeze", "Squeeze"}:
            if "axes" in attrs:
                onnx_attrs["axes"] = attrs["axes"]
            elif len(inputs) > 1:
                # Opset 11 uses axes as an attribute. Inline constant axes when provided as input.
                axis_name = inputs[1]
                const = graph.get("constants", {}).get(axis_name)
                if const is None:
                    raise ValueError(f"{node_id} {op} axes input must be a saved constant for ONNX opset 11")
                onnx_attrs["axes"] = constant_to_array(const).astype(int).flatten().tolist()
                onnx_inputs = [inputs[0]]
        elif op not in {"Identity", "Equal", "Greater", "Less", "Not", "And", "Or", "Where"}:
            raise ValueError(f"Unsupported op for ONNX export: {op}")

        initializers.extend(extra_inits)
        nodes.append(helper.make_node(op, onnx_inputs, [node_id], name=node_id, **onnx_attrs))

    output_name = graph.get("output")
    if not output_name:
        raise ValueError("Graph is missing output")

    input_shape: list[int | str | None] = [None, None]
    output_shape: list[int | str | None] | None = None
    output_type = TensorProto.INT64
    if sample_input is not None:
        sample = np.asarray(sample_input, dtype=np.int64)
        input_shape = list(sample.shape)
        try:
            output, _ = execute_graph(graph, sample_input)
            output_shape = list(np.asarray(output).shape)
            output_type = _tensor_type(np.asarray(output))
        except Exception:
            output_shape = None

    inputs = [helper.make_tensor_value_info("input", TensorProto.INT64, input_shape)]
    outputs = [helper.make_tensor_value_info(output_name, output_type, output_shape)]
    graph_proto = helper.make_graph(nodes, "NeuroGolfGraph", inputs, outputs, initializer=initializers)
    model = helper.make_model(graph_proto, producer_name="neurogolf-lab", opset_imports=[helper.make_operatorsetid("", 11)])
    model.ir_version = min(model.ir_version, 7)
    onnx.checker.check_model(model)
    return model


def compile_graph(graph: dict[str, Any], sample_input: list[list[int]] | None = None) -> dict[str, Any]:
    model = build_model(graph, sample_input)
    response: dict[str, Any] = {
        "ok": True,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "opset": 11,
        "output": graph.get("output"),
    }
    if sample_input is not None:
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        ort_output = session.run([graph["output"]], {"input": np.asarray(sample_input, dtype=np.int64)})[0]
        response["onnxRuntimeShape"] = list(ort_output.shape)
        response["onnxRuntimeOutput"] = np.squeeze(ort_output).astype(np.int64).tolist() if ort_output.dtype != np.bool_ else np.squeeze(ort_output).astype(np.int64).tolist()
    return response


def export_graph(path: Path, graph: dict[str, Any], sample_input: list[list[int]] | None = None) -> dict[str, Any]:
    model = build_model(graph, sample_input)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    response = compile_graph(graph, sample_input)
    response["path"] = str(path)
    return response
