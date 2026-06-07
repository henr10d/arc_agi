"""NeuroGolf task125: frame magenta rectangle objects and fill their holes.

Task rule: every input is a 15x15 cyan grid containing three magenta
rectangle-like objects. The output keeps magenta pixels, fills cyan cells that
are horizontally and vertically enclosed by magenta with yellow, and colors the
one-cell expanded rectangle span around each object green. Other cells remain
cyan; the competition tensor is padded back to 30x30.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = Path(__file__).resolve().parent / "task125.onnx"
DATA_PATH = ROOT / "data" / "task125.json"

sys.path.insert(0, str(ROOT))

C = 10
H = W = 30
N = 15
PAD = H - N
SHAPE = [1, C, H, W]


def init(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(arr, name=name)


def add(nodes: list[onnx.NodeProto], op: str, inputs: list[str], output: str, **attrs) -> str:
    nodes.append(helper.make_node(op, inputs, [output], **attrs))
    return output


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    inits.extend(
        [
            init("s6", np.array([0, 6, 0, 0], dtype=np.int64)),
            init("e7", np.array([1, 7, N, N], dtype=np.int64)),
            init("axes", np.array([0, 1, 2, 3], dtype=np.int64)),
            init("zero", np.array([0.0], dtype=np.float32)),
        ]
    )

    x_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)

    add(nodes, "Slice", ["input", "s6", "e7", "axes"], "mag4")
    add(nodes, "Greater", ["mag4", "zero"], "mag")

    add(nodes, "MaxPool", ["mag4"], "lf", kernel_shape=[1, 6], pads=[0, 5, 0, 0], strides=[1, 1])
    add(nodes, "MaxPool", ["mag4"], "rf", kernel_shape=[1, 6], pads=[0, 0, 0, 5], strides=[1, 1])
    add(nodes, "MaxPool", ["mag4"], "uf", kernel_shape=[6, 1], pads=[5, 0, 0, 0], strides=[1, 1])
    add(nodes, "MaxPool", ["mag4"], "df", kernel_shape=[6, 1], pads=[0, 0, 5, 0], strides=[1, 1])
    add(nodes, "Min", ["lf", "rf", "uf", "df"], "insidef")
    add(nodes, "Greater", ["insidef", "mag4"], "yellow")
    add(nodes, "And", ["mag", "yellow"], "empty")

    add(nodes, "MaxPool", ["mag4"], "dilf4", kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])
    add(nodes, "Greater", ["dilf4", "insidef"], "green")
    add(nodes, "Or", ["mag", "yellow"], "paint0")
    add(nodes, "Or", ["paint0", "green"], "paint")
    add(nodes, "Not", ["paint"], "bg")
    add(
        nodes,
        "Concat",
        ["empty", "empty", "empty", "green", "yellow", "empty", "mag", "empty", "bg", "empty"],
        "outb",
        axis=1,
    )
    add(nodes, "Cast", ["outb"], "out15", to=TensorProto.FLOAT)
    add(nodes, "Pad", ["out15"], "output", pads=[0, 0, 0, 0, 0, 0, PAD, PAD])

    graph = helper.make_graph(nodes, "task125", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 10)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, color, r, c] = 1.0
    return out


def verify_examples(path: Path) -> None:
    data = json.loads(DATA_PATH.read_text())
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    failed: list[str] = []
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            got = session.run(["output"], {"input": to_onehot(example["input"])})[0] > 0.0
            want = to_onehot(example["output"]) > 0.0
            if not np.array_equal(got, want):
                failed.append(f"{split}[{idx}]")
                if len(failed) >= 5:
                    raise AssertionError(f"failed examples: {failed}")
    if failed:
        raise AssertionError(f"failed examples: {failed}")


def main() -> None:
    model = build_model()
    onnx.save(model, OUT_PATH)
    verify_examples(OUT_PATH)
    try:
        from score_model import print_report, score_file

        print_report(score_file(OUT_PATH))
    except Exception as exc:
        print(f"saved {OUT_PATH}")
        print(f"scorer unavailable: {exc}")


if __name__ == "__main__":
    main()
