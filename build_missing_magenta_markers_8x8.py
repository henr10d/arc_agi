"""Build a compact ONNX model for the 8x8 magenta-defect marker task.

Task rule: preserve the 8x8 one-hot input grid, but for every defective
3x3-ish magenta shape, find the background cell inside the shape's implied
3x3 box. In that same column, set the bottom-row cell to yellow. The graph
detects candidate holes from the magenta channel using four-neighbor support:
background AND (left OR right magenta) AND (up OR down magenta).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


OUT_PATH = Path("missing_magenta_markers_8x8.onnx")
IR_VERSION = 10
OPSET = 10
C = 10
H = W = 8
SHAPE = [1, C, H, W]


def init(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array, name=name)


def build_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)

    yellow = np.zeros((1, C, 1, 1), dtype=np.float32)
    yellow[0, 4, 0, 0] = 1.0
    bottom = np.zeros((1, 1, H, W), dtype=bool)
    bottom[0, 0, 7, :] = True

    initializers = [
        init("idx6", np.array([6], dtype=np.int64)),
        init("zf", np.array([0.0], dtype=np.float32)),
        init("yellow", yellow),
        init("bottom", bottom),
    ]

    nodes = [
        helper.make_node("Gather", ["input", "idx6"], ["magf"], axis=1),
        helper.make_node("Greater", ["magf", "zf"], ["mag"]),
        helper.make_node("Not", ["mag"], ["bg"]),
        helper.make_node(
            "MaxPool",
            ["magf"],
            ["hmax"],
            kernel_shape=[1, 3],
            pads=[0, 1, 0, 1],
            strides=[1, 1],
        ),
        helper.make_node(
            "MaxPool",
            ["magf"],
            ["vmax"],
            kernel_shape=[3, 1],
            pads=[1, 0, 1, 0],
            strides=[1, 1],
        ),
        helper.make_node("Greater", ["hmax", "zf"], ["lr"]),
        helper.make_node("Greater", ["vmax", "zf"], ["ud"]),
        helper.make_node("And", ["bg", "lr"], ["bg_lr"]),
        helper.make_node("And", ["bg_lr", "ud"], ["hole"]),
        helper.make_node("Cast", ["hole"], ["holef"], to=TensorProto.FLOAT),
        helper.make_node("ReduceSum", ["holef"], ["colsum"], axes=[2], keepdims=1),
        helper.make_node("Greater", ["colsum", "zf"], ["cols"]),
        helper.make_node("And", ["cols", "bottom"], ["rowmask"]),
        helper.make_node("Where", ["rowmask", "yellow", "input"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "missing_magenta_markers_8x8", [x], [y], initializers)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def one_hot(grid: np.ndarray) -> np.ndarray:
    out = np.zeros((1, C, H, W), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            out[0, int(grid[r, c]), r, c] = 1.0
    return out


def decode(tensor: np.ndarray) -> np.ndarray:
    return tensor[0].argmax(axis=0).astype(np.int64)


def add_defect(grid: np.ndarray, top: int, left: int, missing: tuple[int, int]) -> None:
    for r in range(3):
        for c in range(3):
            if (r, c) != missing:
                grid[top + r, left + c] = 6


def expected(grid: np.ndarray, holes: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    for _, c in holes:
        out[7, c] = 4
    return out


def examples() -> list[tuple[np.ndarray, np.ndarray]]:
    pairs: list[tuple[np.ndarray, np.ndarray]] = []

    g = np.zeros((H, W), dtype=np.int64)
    add_defect(g, 1, 1, (1, 1))
    pairs.append((g, expected(g, [(2, 2)])))

    g = np.zeros((H, W), dtype=np.int64)
    add_defect(g, 0, 0, (0, 0))
    add_defect(g, 3, 5, (2, 1))
    pairs.append((g, expected(g, [(0, 0), (5, 6)])))

    g = np.zeros((H, W), dtype=np.int64)
    add_defect(g, 1, 4, (0, 2))
    add_defect(g, 4, 0, (1, 2))
    pairs.append((g, expected(g, [(1, 6), (5, 2)])))

    return pairs


def param_count(model: onnx.ModelProto) -> int:
    total = 0
    for tensor in model.graph.initializer:
        total += int(np.prod(tensor.dims))
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                total += int(np.prod(attr.t.dims))
            elif attr.name in {"value_floats", "value_ints", "value_strings"}:
                total += len(getattr(attr, attr.name.split("_", 1)[1]))
    return total


def inferred_internal_memory(model: onnx.ModelProto) -> int:
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    inits = {t.name for t in graph.initializer}
    io = {"input", "output"}
    value_info = {v.name: v for v in list(graph.value_info) + list(graph.output) + list(graph.input)}
    total = 0
    for node in graph.node:
        for name in node.output:
            if name in io or name in inits:
                continue
            info = value_info[name]
            tensor_type = info.type.tensor_type
            elem_type = tensor_type.elem_type
            itemsize = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(elem_type)).itemsize
            elems = 1
            for dim in tensor_type.shape.dim:
                elems *= dim.dim_value
            total += elems * itemsize
    return int(total)


def verify(path: Path) -> bool:
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), sess_options=sess_options, providers=["CPUExecutionProvider"])
    ok = True
    for idx, (inp, exp) in enumerate(examples()):
        pred = decode(session.run(["output"], {"input": one_hot(inp)})[0])
        same = np.array_equal(pred, exp)
        print(f"example {idx}: {'PASS' if same else 'FAIL'}")
        if not same:
            print("input:\n", inp)
            print("expected:\n", exp)
            print("pred:\n", pred)
        ok = ok and same
    return ok


def main() -> None:
    model = build_model()
    onnx.save(model, OUT_PATH)
    ok = verify(OUT_PATH)
    params = param_count(model)
    memory = inferred_internal_memory(model)
    print(f"wrote: {OUT_PATH}")
    print(f"initializer scalar count: {params}")
    print(f"inferred internal tensor memory estimate: {memory} bytes")
    print(f"output correctness: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
