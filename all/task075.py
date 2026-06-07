"""Build an ONNX solution for NeuroGolf task075.

Task rule: keep the 9x13 input grid except for the 3x3 target slots to the
right of the gray separator. The 3x3 pattern in rows 0..2, cols 0..2 is
copied into each slot whose center cell is blue. Candidate slot centers are
rows 1, 4, 7 and columns 5, 8, 11; the blue marker is overwritten by the
copied pattern. Padding outside the 9x13 task grid remains all-zero.

ONNX approach: slice the source 3x3 pattern, reduce it to color indices, tile
those indices across the 9x9 target region, expand the 3x3 grid of blue marker
centers with ConvTranspose, then turn the selected color-index grid back into a
float one-hot update with OneHot before scattering it into the original input.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task075"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

OPSET = 10
IR_VERSION = 10
SHAPE = [1, 10, 30, 30]


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self.value_infos: list[onnx.ValueInfoProto] = []

    def init(self, name: str, values: np.ndarray) -> str:
        self.inits.append(numpy_helper.from_array(values, name))
        return name

    def vi(self, name: str, dtype: int, shape: list[int]) -> str:
        self.value_infos.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    def node(self, op_type: str, inputs: list[str], output: str, dtype: int, shape: list[int], **attrs: object) -> str:
        self.vi(output, dtype, shape)
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        return output


def build_model() -> onnx.ModelProto:
    b = Builder()
    b.init("axes23", np.array([2, 3], dtype=np.int64))
    b.init("start00", np.array([0, 0], dtype=np.int64))
    b.init("end33", np.array([3, 3], dtype=np.int64))
    b.init("center_st", np.array([1, 1, 5], dtype=np.int64))
    b.init("center_en", np.array([2, 8, 12], dtype=np.int64))
    b.init("axes123", np.array([1, 2, 3], dtype=np.int64))
    b.init("center_steps", np.array([1, 3, 3], dtype=np.int64))
    b.init("tile_repeats", np.array([1, 3, 3], dtype=np.int64))
    b.init("zero", np.array(0.0, dtype=np.float32))
    b.init("zero_i64", np.array(0, dtype=np.int64))
    b.init("expand_kernel", np.ones((1, 1, 3, 3), dtype=np.float32))
    b.init("onehot_depth", np.array(10, dtype=np.int64))
    b.init("onehot_values", np.array([0.0, 1.0], dtype=np.float32))

    scatter_idx = np.broadcast_to(np.arange(4, 13, dtype=np.int64), (1, 10, 9, 9)).copy()
    b.init("scatter_idx", scatter_idx)

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)

    patch = b.node("Slice", ["input", "start00", "end33", "axes23"], "patch", TensorProto.FLOAT, [1, 10, 3, 3])
    patch_color = b.node(
        "ArgMax",
        [patch],
        "patch_color",
        TensorProto.INT64,
        [1, 3, 3],
        axis=1,
        keepdims=0,
    )
    tiled = b.node("Tile", [patch_color, "tile_repeats"], "tiled", TensorProto.INT64, [1, 9, 9])
    centers = b.node(
        "Slice",
        ["input", "center_st", "center_en", "axes123", "center_steps"],
        "centers",
        TensorProto.FLOAT,
        [1, 1, 3, 3],
    )
    mask = b.node(
        "ConvTranspose",
        [centers, "expand_kernel"],
        "mask",
        TensorProto.FLOAT,
        [1, 1, 9, 9],
        kernel_shape=[3, 3],
        strides=[3, 3],
    )
    cond4 = b.node("Greater", [mask, "zero"], "cond4", TensorProto.BOOL, [1, 1, 9, 9])
    cond = b.node("Squeeze", [cond4], "cond", TensorProto.BOOL, [1, 9, 9], axes=[1])
    color = b.node("Where", [cond, tiled, "zero_i64"], "color", TensorProto.INT64, [1, 9, 9])
    updates = b.node(
        "OneHot",
        [color, "onehot_depth", "onehot_values"],
        "updates",
        TensorProto.FLOAT,
        [1, 10, 9, 9],
        axis=1,
    )
    b.nodes.append(helper.make_node("Scatter", ["input", "scatter_idx", updates], ["output"], axis=3))

    graph = helper.make_graph(
        b.nodes,
        TASK_ID,
        [inp],
        [out],
        initializer=b.inits,
        value_info=b.value_infos,
    )
    model = helper.make_model(
        graph,
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def _validate_examples(path: Path) -> tuple[int, int]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for example in data[split]:
            x = convert_to_numpy(example, "input")
            y = convert_to_numpy(example, "output")
            if x is None or y is None:
                continue
            pred = session.run(["output"], {"input": x})[0]
            total += 1
            if np.array_equal(pred > 0.0, y > 0.0):
                passed += 1
            else:
                raise AssertionError(f"{split} example {total} failed")
    return passed, total


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    passed, total = _validate_examples(BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"correct: {passed}/{total}")
    print(
        "score: "
        f"memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']:.6f}"
    )


if __name__ == "__main__":
    main()
