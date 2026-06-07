"""Build a compact ONNX model for NeuroGolf task078.

Task rule: preserve all blue obstacle cells (color 1). For each column,
count the red cells (color 2), then place that many red cells into the
highest non-blue positions in the same column. Background cells inside the
10x10 task canvas are color 0; padding outside that canvas remains all-zero.

ONNX approach: slice only the blue/red 10x10 masks, use per-column blue and
red counts with a tiny row-index constant to identify the compacted bands, and
reconstruct the blue top prefixes from their counts. Emit only channels 0..2
before padding the channel and spatial dimensions directly to the required
30x30 output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402


TASK_NUM = "078"
TASK_NAME = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_NAME}.json"
BEST_PATH = Path(__file__).resolve().parent / f"{TASK_NAME}.onnx"
SOLUTION_PATH = ROOT / "solution.onnx"


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.value_infos: list[onnx.ValueInfoProto] = []
        self.counter = 0

    def init(self, name: str, array: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(array, name=name))
        return name

    def out(self, prefix: str, dtype: int, shape: tuple[int, ...]) -> str:
        self.counter += 1
        name = f"{prefix}_{self.counter}"
        self.value_infos.append(helper.make_tensor_value_info(name, dtype, list(shape)))
        return name

    def node(
        self,
        op_type: str,
        inputs: list[str],
        prefix: str,
        dtype: int,
        shape: tuple[int, ...],
        **attrs: Any,
    ) -> str:
        output = self.out(prefix, dtype, shape)
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        return output


def build_model() -> onnx.ModelProto:
    b = Builder()
    b.init("axes_chw", np.array([1, 2, 3], dtype=np.int64))
    b.init("blue_starts", np.array([1, 0, 0], dtype=np.int64))
    b.init("blue_ends", np.array([2, 10, 10], dtype=np.int64))
    b.init("red_starts", np.array([2, 0, 0], dtype=np.int64))
    b.init("red_ends", np.array([3, 10, 10], dtype=np.int64))

    b.init("row_plus_half", (np.arange(10, dtype=np.float32) + 0.5).reshape(1, 1, 10, 1))

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])

    blue = b.node(
        "Slice",
        ["input", "blue_starts", "blue_ends", "axes_chw"],
        "blue",
        TensorProto.FLOAT,
        (1, 1, 10, 10),
    )
    red = b.node(
        "Slice",
        ["input", "red_starts", "red_ends", "axes_chw"],
        "red",
        TensorProto.FLOAT,
        (1, 1, 10, 10),
    )
    blue_count = b.node("ReduceSum", [blue], "blue_count", TensorProto.FLOAT, (1, 1, 1, 10), axes=[2], keepdims=1)
    red_count = b.node("ReduceSum", [red], "red_count", TensorProto.FLOAT, (1, 1, 1, 10), axes=[2], keepdims=1)
    stop_count = b.node("Add", [blue_count, red_count], "stop_count", TensorProto.FLOAT, (1, 1, 1, 10))
    below_blue = b.node("Greater", ["row_plus_half", blue_count], "below_blue", TensorProto.BOOL, (1, 1, 10, 10))
    above_stop = b.node("Greater", [stop_count, "row_plus_half"], "above_stop", TensorProto.BOOL, (1, 1, 10, 10))
    blue_b = b.node("Not", [below_blue], "blue_b", TensorProto.BOOL, (1, 1, 10, 10))
    placed_red_b = b.node("And", [below_blue, above_stop], "placed_red_b", TensorProto.BOOL, (1, 1, 10, 10))
    bg_b = b.node("Not", [above_stop], "bg_b", TensorProto.BOOL, (1, 1, 10, 10))
    crop3_b = b.node("Concat", [bg_b, blue_b, placed_red_b], "crop3_b", TensorProto.BOOL, (1, 3, 10, 10), axis=1)
    crop3 = b.node("Cast", [crop3_b], "crop3", TensorProto.FLOAT, (1, 3, 10, 10), to=TensorProto.FLOAT)

    b.nodes.append(
        helper.make_node(
            "Pad",
            [crop3],
            ["output"],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 7, 20, 20],
            value=0.0,
        )
    )

    graph = helper.make_graph(b.nodes, TASK_NAME, [inp], [out], b.initializers, value_info=b.value_infos)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 10)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


def grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            arr[0, int(value), r, c] = 1.0
    return arr


def decode(arr: np.ndarray, h: int = 10, w: int = 10) -> np.ndarray:
    active = arr[0, :, :h, :w] > 0.0
    if not np.all(active.sum(axis=0) == 1):
        raise AssertionError("invalid one-hot output inside the 10x10 canvas")
    return active.argmax(axis=0).astype(np.int64)


def verify(path: Path) -> None:
    with TASK_PATH.open(encoding="utf-8") as fh:
        task = json.load(fh)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    checked = 0
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(task.get(split, [])):
            expected = np.array(example["output"], dtype=np.int64)
            got = session.run(["output"], {"input": grid_to_onehot(example["input"])})[0]
            decoded = decode(got, expected.shape[0], expected.shape[1])
            if not np.array_equal(decoded, expected):
                raise AssertionError(f"{split}[{idx}] failed")
            checked += 1
    print(f"verified {checked} examples")


def print_score(path: Path) -> None:
    result = score_file(path)
    if not result["valid"]:
        raise RuntimeError(result["error"])
    print(
        "NeuroGolf score: "
        f"memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={float(result['score']):.6f}"
    )


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    onnx.save(model, SOLUTION_PATH)
    verify(BEST_PATH)
    print_score(BEST_PATH)
    print(f"wrote {BEST_PATH.relative_to(ROOT)}")
    print(f"wrote {SOLUTION_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
