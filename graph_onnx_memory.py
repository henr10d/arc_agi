"""Visualize ONNX graph nodes with official NeuroGolf memory per node.

Memory attribution follows the competition rules used in score_model.py:
  - each internal output tensor gets a byte count (max across profiled runs)
  - input/output graph tensors are excluded from the scored total
  - a node's label shows the sum of its scored output tensors

Usage:
  python graph_onnx_memory.py all/task001.onnx
  python graph_onnx_memory.py all/task001.onnx -o task001_memory.png
  python graph_onnx_memory.py all/task001.onnx --show
  python graph_onnx_memory.py all/task001.onnx --fast
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import networkx as nx
import numpy as np
import onnx
import onnxruntime as ort

from score_model import (
    GRID_SHAPE,
    convert_to_numpy,
    sanitize_model,
)

IN_NAME = "input"
OUT_NAME = "output"
SCORED_IO = {IN_NAME, OUT_NAME}


@dataclass
class TensorMemory:
    name: str
    bytes: int
    shape: tuple[int, ...]
    dtype: str
    scored: bool


@dataclass
class NodeMemory:
    key: str
    op_type: str
    flow_index: int = 0
    outputs: list[str] = field(default_factory=list)
    tensor_details: list[TensorMemory] = field(default_factory=list)

    @property
    def scored_bytes(self) -> int:
        return sum(t.bytes for t in self.tensor_details if t.scored)

    @property
    def total_output_bytes(self) -> int:
        return sum(t.bytes for t in self.tensor_details)


def _shape_from_value_info(item: onnx.ValueInfoProto) -> tuple[int, ...] | None:
    if not item.type.HasField("tensor_type") or not item.type.tensor_type.HasField("shape"):
        return None
    dims: list[int] = []
    for dim in item.type.tensor_type.shape.dim:
        if dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        dims.append(int(dim.dim_value))
    return tuple(dims)


def _dtype_name(item: onnx.ValueInfoProto) -> str:
    if not item.type.HasField("tensor_type"):
        return "?"
    np_dtype = onnx.helper.tensor_dtype_to_np_dtype(item.type.tensor_type.elem_type)
    return np.dtype(np_dtype).name


def load_profile_inputs(path: Path, *, fast: bool = False) -> list[np.ndarray]:
    if fast:
        return [np.zeros(GRID_SHAPE, dtype=np.float32)]

    task_path = Path(__file__).resolve().parent / "data" / f"{path.stem}.json"
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


def run_profile(model: onnx.ModelProto, inputs: list[np.ndarray], stem: str) -> str:
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(Path(tempfile.gettempdir()) / f"ng_graph_{stem}")
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    for arr in inputs:
        session.run([OUT_NAME], {IN_NAME: arr})
    trace_path = session.end_profiling()
    assert trace_path is not None
    return trace_path


def compute_tensor_memory(model: onnx.ModelProto, trace_path: str) -> dict[str, TensorMemory]:
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    tensor_map = {
        tensor.name: tensor
        for tensor in list(graph.input) + list(graph.value_info) + list(graph.output)
    }

    node_outputs: dict[str, list[str]] = {}
    for node in graph.node:
        node_outputs[node.name] = list(node.output)

    tensor_memory: dict[str, int] = {}
    tensor_meta: dict[str, tuple[tuple[int, ...], str]] = {}
    for tensor_name, item in tensor_map.items():
        if not item.type.HasField("tensor_type"):
            continue
        shape = _shape_from_value_info(item)
        if shape is None:
            continue
        dtype = _dtype_name(item)
        num_elements = math.prod(shape)
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(item.type.tensor_type.elem_type)
        bytes_ = int(num_elements * np.dtype(np_dtype).itemsize)
        if tensor_name not in SCORED_IO:
            tensor_memory[tensor_name] = bytes_
        tensor_meta[tensor_name] = (shape, dtype)

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
            if output_name in SCORED_IO or output_name not in tensor_meta:
                continue
            shape, dtype = tensor_meta[output_name]
            itemsize = np.dtype(dtype).itemsize
            mem = itemsize * sum(math.prod(dims) for dims in shape_dict.values())
            tensor_memory[output_name] = max(tensor_memory.get(output_name, 0), int(mem))

    out: dict[str, TensorMemory] = {}
    for tensor_name, (shape, dtype) in tensor_meta.items():
        bytes_ = tensor_memory.get(tensor_name, 0)
        if tensor_name in SCORED_IO:
            # Still show I/O footprint, but mark unscored.
            if bytes_ == 0:
                np_dtype = np.dtype(dtype)
                bytes_ = int(math.prod(shape) * np_dtype.itemsize)
        out[tensor_name] = TensorMemory(
            name=tensor_name,
            bytes=bytes_,
            shape=shape,
            dtype=dtype,
            scored=tensor_name not in SCORED_IO,
        )
    return out


def build_node_memory(model: onnx.ModelProto, tensor_memory: dict[str, TensorMemory]) -> list[NodeMemory]:
    nodes: list[NodeMemory] = []
    for index, node in enumerate(model.graph.node):
        key = node.name or node.output[0] or f"node_{index}"
        details = [tensor_memory[name] for name in node.output if name in tensor_memory]
        nodes.append(
            NodeMemory(
                key=key,
                op_type=node.op_type,
                flow_index=index,
                outputs=list(node.output),
                tensor_details=details,
            )
        )
    return nodes


def _flow_rank(model: onnx.ModelProto) -> dict[str, int]:
    """Stable execution order: input/initializers, ONNX nodes, output."""
    rank = {"__input__": -1, "__output__": len(model.graph.node) + len(model.graph.initializer)}
    for index, init in enumerate(model.graph.initializer):
        rank[f"init:{init.name}"] = index
    for index, node in enumerate(model.graph.node):
        key = node.name or node.output[0] or f"node_{index}"
        rank[key] = len(model.graph.initializer) + index
    return rank


def build_graph(model: onnx.ModelProto) -> nx.DiGraph:
    g = nx.DiGraph()
    producer: dict[str, str] = {IN_NAME: "__input__"}
    flow_rank = _flow_rank(model)

    for init in model.graph.initializer:
        init_key = f"init:{init.name}"
        g.add_node(init_key, kind="initializer", op_type="Initializer", label=init.name, flow_rank=flow_rank[init_key])
        producer[init.name] = init_key

    g.add_node("__input__", kind="io", op_type="Input", label=IN_NAME, flow_rank=flow_rank["__input__"])

    for index, node in enumerate(model.graph.node):
        key = node.name or node.output[0] or f"node_{index}"
        g.add_node(
            key,
            kind="node",
            op_type=node.op_type,
            outputs=list(node.output),
            flow_rank=flow_rank[key],
        )
        for out_name in node.output:
            if out_name:
                producer[out_name] = key
        for in_name in node.input:
            if not in_name:
                continue
            src = producer.get(in_name)
            if src is None:
                src = f"unknown:{in_name}"
                g.add_node(src, kind="unknown", op_type="?", label=in_name, flow_rank=10**9)
            g.add_edge(src, key, tensor=in_name)

    g.add_node("__output__", kind="io", op_type="Output", label=OUT_NAME, flow_rank=flow_rank["__output__"])
    if OUT_NAME in producer:
        g.add_edge(producer[OUT_NAME], "__output__", tensor=OUT_NAME)
    return g


def _layer_layout(g: nx.DiGraph) -> dict[str, tuple[float, float]]:
    def rank(node: str) -> int:
        return int(g.nodes[node].get("flow_rank", 10**9))

    layers: dict[int, list[str]] = defaultdict(list)
    indegree = {node: g.in_degree(node) for node in g.nodes}
    queue = sorted((node for node, deg in indegree.items() if deg == 0), key=rank)
    depth = {node: 0 for node in g.nodes}

    while queue:
        node = queue.pop(0)
        layers[depth[node]].append(node)
        ready: list[str] = []
        for child in g.successors(node):
            depth[child] = max(depth[child], depth[node] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        queue.extend(sorted(ready, key=rank))

    placed = {node for layer in layers.values() for node in layer}
    for leftover in sorted((node for node in g.nodes if node not in placed), key=rank):
        layers[max(layers.keys(), default=0) + 1].append(leftover)

    pos: dict[str, tuple[float, float]] = {}
    x_gap = 3.2
    y_gap = 1.8
    for layer_idx in sorted(layers):
        nodes = sorted(layers[layer_idx], key=rank)
        width = max(len(nodes) - 1, 0) * y_gap
        for row, node in enumerate(nodes):
            pos[node] = (layer_idx * x_gap, width / 2 - row * y_gap)
    return pos


def _memory_color(bytes_: int, max_bytes: int) -> tuple[float, float, float, float]:
    if max_bytes <= 0 or bytes_ <= 0:
        return (0.92, 0.94, 0.97, 1.0)
    t = min(1.0, bytes_ / max_bytes)
    return (1.0, 1.0 - 0.55 * t, 1.0 - 0.75 * t, 1.0)


def _node_label(g: nx.DiGraph, node: str, node_memory: dict[str, NodeMemory]) -> str:
    data = g.nodes[node]
    kind = data.get("kind")
    if kind == "io":
        return data.get("label", node)
    if kind == "initializer":
        return f"Const\n{data.get('label', node)}"
    if kind == "unknown":
        return data.get("label", node)

    info = node_memory.get(node)
    if info is None:
        return data.get("op_type", node)
    lines = [info.op_type, f"{info.scored_bytes} B scored"]
    if info.total_output_bytes != info.scored_bytes:
        lines.append(f"{info.total_output_bytes} B total out")
    for tensor in info.tensor_details:
        shape = "x".join(str(d) for d in tensor.shape)
        tag = "" if tensor.scored else " [io]"
        lines.append(f"{tensor.name}: {tensor.bytes} B{tag}")
        lines.append(f"  {tensor.dtype} [{shape}]")
    return "\n".join(lines)


def _configure_matplotlib(show: bool) -> bool:
    """Pick a backend. Returns True if an interactive display is available."""
    if show:
        for backend in ("TkAgg", "Qt5Agg", "GTK3Agg"):
            try:
                matplotlib.use(backend, force=True)
                import matplotlib.pyplot as plt

                fig = plt.figure()
                plt.close(fig)
                if "agg" not in matplotlib.get_backend().lower():
                    return True
            except Exception:
                continue
        print("Note: no interactive matplotlib backend; open the saved PNG instead.")
    matplotlib.use("Agg", force=True)
    return False


def render_graph(
    g: nx.DiGraph,
    node_memory: dict[str, NodeMemory],
    tensor_memory: dict[str, TensorMemory],
    *,
    title: str,
    output_path: Path | None,
    show: bool,
) -> None:
    interactive = _configure_matplotlib(show)
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    pos = _layer_layout(g)
    scored_total = sum(t.bytes for t in tensor_memory.values() if t.scored)
    max_node = max((info.scored_bytes for info in node_memory.values()), default=0)

    fig_w = max(12, len(set(round(x) for x, _ in pos.values())) * 2.8)
    fig_h = max(6, len(g.nodes) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_title(
        f"{title}\nofficial scored memory = {scored_total} B (input/output excluded)",
        fontsize=12,
        pad=12,
    )
    ax.axis("off")

    for src, dst, edge_data in g.edges(data=True):
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        ax.annotate(
            "",
            xy=(x2 - 0.55, y2),
            xytext=(x1 + 0.55, y1),
            arrowprops=dict(arrowstyle="-|>", color="#666666", lw=1.2, shrinkA=0, shrinkB=0),
        )
        tensor = edge_data.get("tensor")
        if tensor:
            ax.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2 + 0.15,
                tensor,
                fontsize=7,
                ha="center",
                va="bottom",
                color="#444444",
            )

    for node, (x, y) in pos.items():
        data = g.nodes[node]
        kind = data.get("kind", "node")
        info = node_memory.get(node)
        scored = info.scored_bytes if info else 0
        if kind == "node":
            face = _memory_color(scored, max_node)
            width, height = 2.5, 1.2 + 0.18 * max(len(info.tensor_details), 1) * 2 if info else 1.2
        elif kind == "io":
            face = (0.85, 0.92, 0.85, 1.0)
            width, height = 1.6, 0.8
        else:
            face = (0.95, 0.95, 0.90, 1.0)
            width, height = 1.8, 0.8

        box = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.0,
            edgecolor="#333333",
            facecolor=face,
        )
        ax.add_patch(box)
        ax.text(
            x,
            y,
            _node_label(g, node, node_memory),
            ha="center",
            va="center",
            fontsize=8,
            family="monospace",
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        print(f"Saved graph to {output_path}")
    if interactive:
        plt.show()
    plt.close(fig)


def print_text_report(node_memory: list[NodeMemory], tensor_memory: dict[str, TensorMemory]) -> None:
    scored_total = sum(t.bytes for t in tensor_memory.values() if t.scored)
    print(f"official scored memory: {scored_total} B")
    print()
    print("Per node in execution order (scored output bytes):")
    for info in node_memory:
        if info.scored_bytes == 0 and info.total_output_bytes == 0:
            continue
        print(f"  {info.op_type:12s} {info.scored_bytes:6d} B  outputs={info.outputs}")
        for tensor in info.tensor_details:
            shape = "x".join(str(d) for d in tensor.shape)
            tag = "scored" if tensor.scored else "io (excluded)"
            print(f"      {tensor.name:20s} {tensor.bytes:6d} B  {tensor.dtype:6s} [{shape}]  {tag}")


def analyze(
    path: Path,
    *,
    fast: bool = False,
) -> tuple[onnx.ModelProto, dict[str, TensorMemory], list[NodeMemory], list[np.ndarray]]:
    model = onnx.load(str(path))
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("model failed NeuroGolf name sanitization")

    inputs = load_profile_inputs(path, fast=fast)
    trace_path = run_profile(sanitized, inputs, path.stem)
    tensor_memory = compute_tensor_memory(sanitized, trace_path)
    node_memory = build_node_memory(sanitized, tensor_memory)
    return sanitized, tensor_memory, node_memory, inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph ONNX nodes with official memory bytes.")
    parser.add_argument("onnx_path", type=Path, help="path to .onnx file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="image output path (.png/.svg/.pdf). default: <model>_memory.png",
    )
    parser.add_argument("--show", action="store_true", help="open interactive window")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="profile one zero input instead of all task examples (faster, may differ)",
    )
    args = parser.parse_args()

    if not args.onnx_path.is_file():
        raise SystemExit(f"file not found: {args.onnx_path}")

    try:
        model, tensor_memory, node_memory, inputs = analyze(args.onnx_path, fast=args.fast)
    except Exception:
        raise SystemExit(traceback.format_exc())

    node_memory_by_key = {info.key: info for info in node_memory}
    g = build_graph(model)

    print(f"model: {args.onnx_path.name}")
    print(f"profiled inputs: {len(inputs)}")
    print_text_report(node_memory, tensor_memory)

    output_path = args.output
    if output_path is None:
        output_path = args.onnx_path.with_name(f"{args.onnx_path.stem}_memory.png")

    render_graph(
        g,
        node_memory_by_key,
        tensor_memory,
        title=args.onnx_path.name,
        output_path=output_path,
        show=args.show,
    )


if __name__ == "__main__":
    main()
