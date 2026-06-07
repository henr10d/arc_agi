"""ARC task061: restore a modular multiplication table with missing cells.

Task rule: every input is an 18x18 grid whose nonzero cells are samples from a
repeated multiplication table over colors 1..K, where K is the largest color
visible in the input. The output fills every cell with
``1 + ((row mod K) * (col mod K) mod K)`` and pads the ONNX output to 30x30.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

BEST_PATH = OUT_DIR / "task061.onnx"
DATA_PATH = ROOT / "data" / "task061.json"

C = 10
G = 18
H = W = 30
PAD = H - G
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
IR_VERSION = 10


def solve(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=np.int64)
    k = int(g.max())
    rows = np.arange(g.shape[0], dtype=np.int64)[:, None] % k
    cols = np.arange(g.shape[1], dtype=np.int64)[None, :] % k
    return 1 + ((rows * cols) % k)


def _grid_to_onehot(grid: Sequence[Sequence[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            color = int(val)
            if 0 <= color < C:
                out[0, color, r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _pattern_lut() -> np.ndarray:
    """Return color-id tables indexed by inferred max color K."""
    lut = np.zeros((6, 1, G, G), dtype=np.int32)
    rows = np.arange(G, dtype=np.int64)[:, None]
    cols = np.arange(G, dtype=np.int64)[None, :]
    for offset, k in enumerate(range(4, C)):
        lut[offset, 0] = 1 + (((rows % k) * (cols % k)) % k)
    return lut


class _G:
    """Small ONNX graph builder for modular multiplication tables."""

    def __init__(self, opset: int = 13) -> None:
        self.opset = opset
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self._n = 0
        self.i32(_pattern_lut(), "patterns")
        self.i64(np.int64(4), "four")
        self.f32(np.arange(C, dtype=np.float32).reshape(1, C, 1, 1), "ch_weight")
        self.b(np.eye(C, dtype=bool), "eye")

    def _name(self) -> str:
        self._n += 1
        return f"n{self._n}"

    def i64(self, arr, name: str) -> str:
        self.inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.int64), name=name))
        return name

    def i32(self, arr, name: str) -> str:
        self.inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.int32), name=name))
        return name

    def f32(self, arr, name: str) -> str:
        self.inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name))
        return name

    def b(self, arr, name: str) -> str:
        self.inits.append(numpy_helper.from_array(np.asarray(arr, dtype=bool), name=name))
        return name

    def add(self, op: str, inputs: Sequence[str], **kwargs) -> str:
        out = self._name()
        self.nodes.append(helper.make_node(op, list(inputs), [out], **kwargs))
        return out


def build_model(*, opset: int = 10) -> onnx.ModelProto:
    gb = _G(opset=opset)
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    present = gb.add("ReduceMax", [IN_NAME], keepdims=1, axes=[2, 3])
    weighted = gb.add("Mul", [present, "ch_weight"])
    max_color = gb.add("Cast", [gb.add("ReduceMax", [weighted], keepdims=0)], to=TensorProto.INT64)
    pattern_index = gb.add("Sub", [max_color, "four"])
    ids = gb.add("Gather", ["patterns", pattern_index], axis=0)
    oh_nhwc = gb.add("Gather", ["eye", ids], axis=0)
    oh = gb.add("Cast", [gb.add("Transpose", [oh_nhwc], perm=[0, 3, 1, 2])], to=TensorProto.FLOAT)
    if opset >= 13:
        pads = gb.i64([0, 0, 0, 0, 0, 0, PAD, PAD], "pads")
        gb.nodes.append(helper.make_node("Pad", [oh, pads], [OUT_NAME]))
    else:
        gb.nodes.append(
            helper.make_node("Pad", [oh], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD], mode="constant")
        )

    graph = helper.make_graph(gb.nodes, "task061", [x_info], [y_info], initializer=gb.inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    onnx.checker.check_model(model)
    return model


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            g = np.array(ex["input"], dtype=np.int64)
            oh = _grid_to_onehot(ex["input"])
            pred = _onehot_to_grid(sess.run([OUT_NAME], {IN_NAME: oh.astype(np.float32)})[0])[:G, :G]
            if not np.array_equal(pred, solve(g)):
                bad += 1
    return bad


def main() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            assert np.array_equal(solve(np.array(ex["input"])), ex["output"]), split

    print("building...")
    model = build_model(opset=10)
    bad = validate_json(model)
    onnx.save(model, BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"nodes={len(model.graph.node)} bytes={BEST_PATH.stat().st_size}")
    print(f"json={'PASS' if bad == 0 else f'FAIL({bad})'}")
    print(f"memory={result.get('memory')} params={result.get('params')} cost={result.get('cost')} score={result.get('score')}")
    print(f"valid={result.get('valid')} error={result.get('error')}")


if __name__ == "__main__":
    main()
