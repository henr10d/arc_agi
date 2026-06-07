"""Compact ONNX for NeuroGolf task369 component-size recoloring.

Task rule: each 10x10 input has gray background (color 5) and disconnected
black 4-connected components (color 0). The output preserves all geometry and
positions, leaves gray cells gray, and recolors each black component by size:
singletons become green (3), dominoes become red (2), and triominoes become
blue (1). The task data contains no components larger than size 3.

ONNX: crop the black 10x10 channel, compute each black cell's 4-neighbor
degree with one uint8 QLinearConv, sum adjacent black-cell degrees with a
second uint8 QLinearConv, and classify singleton/red/blue masks directly. The
graph builds only a compact 5-channel 10x10 bool result before one float cast
and final pad to the competition output shape.
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

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_NUM = "369"
TASK_ID = f"task{TASK_NUM}"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
GRID = 10
IR_VERSION = 10
OPSET = 10


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []

    def init(self, name: str, values: Any, dtype: Any) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=dtype), name))
        return name

    def node(self, op_type: str, inputs: list[str], output: str, **attrs: Any) -> str:
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        return output


def four_neighbor_kernel() -> np.ndarray:
    weight = np.zeros((1, 1, 3, 3), dtype=np.uint8)
    weight[0, 0, 0, 1] = 1.0
    weight[0, 0, 1, 0] = 1.0
    weight[0, 0, 1, 2] = 1.0
    weight[0, 0, 2, 1] = 1.0
    return weight


def make_model(nodes: list[onnx.NodeProto], initializers: list[onnx.TensorProto]) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def build_model() -> onnx.ModelProto:
    b = Builder()
    b.init("starts", [0, 0, 0, 0], np.int64)
    b.init("ends", [1, 1, GRID, GRID], np.int64)
    b.init("axes4", [0, 1, 2, 3], np.int64)
    b.init("scale", [1.0], np.float32)
    b.init("zero_u8", [0], np.uint8)
    b.init("one_u8", [1], np.uint8)
    b.init("k4", four_neighbor_kernel(), np.uint8)

    black = b.node("Slice", [IN_NAME, "starts", "ends", "axes4"], "black")
    black_b = b.node("Cast", [black], "black_b", to=TensorProto.BOOL)
    black_u = b.node("Cast", [black], "black_u", to=TensorProto.UINT8)
    gray = b.node("Not", [black_b], "gray")
    zero = b.node("And", [black_b, "gray"], "zero")

    qconv_inputs = ["scale", "zero_u8", "k4", "scale", "zero_u8", "scale", "zero_u8"]
    deg = b.node("QLinearConv", [black_u, *qconv_inputs], "deg", pads=[1, 1, 1, 1])
    single_raw = b.node("Less", [deg, "one_u8"], "single_raw")
    single = b.node("And", [black_b, single_raw], "single")

    deg_black = b.node("Where", [black_b, deg, "zero_u8"], "deg_black")
    neigh_deg_sum = b.node("QLinearConv", [deg_black, *qconv_inputs], "neigh_deg_sum", pads=[1, 1, 1, 1])
    deg2_raw = b.node("Greater", [deg, "one_u8"], "deg2_raw")
    end_of_three = b.node("Greater", [neigh_deg_sum, "one_u8"], "end_of_three")
    tri_raw = b.node("Or", [deg2_raw, end_of_three], "tri_raw")
    tri = b.node("And", [black_b, tri_raw], "tri")

    known = b.node("Or", [single, tri], "known")
    unknown = b.node("Not", [known], "unknown")
    domino = b.node("And", [black_b, unknown], "domino")

    out5_b = b.node(
        "Concat",
        [tri, domino, single, zero, gray],
        "out5_b",
        axis=1,
    )
    out5 = b.node("Cast", [out5_b], "out5", to=TensorProto.FLOAT)
    b.node(
        "Pad",
        [out5],
        OUT_NAME,
        mode="constant",
        pads=[0, 1, 0, 0, 0, 4, 20, 20],
        value=0.0,
    )
    return make_model(b.nodes, b.initializers)


def solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    seen = np.zeros(g.shape, dtype=np.bool_)
    h, w = g.shape
    for row in range(h):
        for col in range(w):
            if g[row, col] != 0 or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and g[nr, nc] == 0 and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            color = {1: 3, 2: 2, 3: 1}[len(cells)]
            for r, c in cells:
                out[r, c] = color
    return out


def _onehot(grid: list[list[int]]) -> np.ndarray:
    return convert_to_numpy({"input": grid}, "input")


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> tuple[int, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    bad = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            expected = _onehot(ex["output"])
            if expected is None:
                continue
            total += 1
            pred = _run_onnx(model, _onehot(ex["input"]))
            if not np.array_equal(pred > 0.0, expected > 0.0):
                bad += 1
                got = (pred[0, :, :GRID, :GRID] > 0.0).argmax(axis=0)
                want = np.asarray(ex["output"], dtype=np.int64)
                print(f"mismatch {split} #{idx}")
                print(got)
                print(want)
                break
    return bad, total


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    bad, total = validate_json(model)
    assert bad == 0, f"{bad} mismatches across {total} examples"
    print(f"verified {total} {TASK_ID} examples")

    result = score_file(BEST_PATH)
    print(result)


if __name__ == "__main__":
    main()
