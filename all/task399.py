"""Count red 2x2 blocks and emit that count as a fixed blue 3x3 glyph.

Task rule: the input is a small square black grid containing one to five red
2x2 blocks.  The output is always a 3x3 grid.  If there are k red blocks, draw
the first k cells of this fixed blue pattern: top-left, top-right, center,
bottom-left, bottom-right.  All other output cells are black.

The ONNX graph sums the input by color, gathers the red-pixel count, compares
it to multiples of four, and builds only the two used 3x3 channels (black and
blue).  The final Pad adds the unused color channels and the 30x30 spatial
padding as the graph output, so those large tensors are not scored as internal
memory.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task399"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task399.onnx"

IN_NAME = "input"
OUT_NAME = "output"
C = 10
H = W = 30
IR_VERSION = 10
OPSET = 10


def _init(inits: list[onnx.TensorProto], vals: Any, dtype: np.dtype, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=dtype), name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, vals, np.dtype(np.int64), name)


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, vals, np.dtype(np.float32), name)


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, [1, C, H, W])
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, [1, C, H, W])

    red_index = _i64(inits, 2, "red_index")
    thresholds = _f32(
        inits,
        np.array([[[[0.0, 99.0, 4.0], [99.0, 8.0, 99.0], [12.0, 99.0, 16.0]]]], dtype=np.float32),
        "thresholds",
    )
    nodes.append(helper.make_node("ReduceSum", [IN_NAME], ["counts"], axes=[2, 3], keepdims=1))
    nodes.append(helper.make_node("Gather", ["counts", red_index], ["red_count"], axis=1))
    nodes.append(helper.make_node("Less", [thresholds, "red_count"], ["blue"]))
    nodes.append(helper.make_node("Not", ["blue"], ["black"]))
    nodes.append(helper.make_node("Concat", ["black", "blue"], ["out3_bool"], axis=1))
    nodes.append(helper.make_node("Cast", ["out3_bool"], ["out3"], to=TensorProto.FLOAT))
    nodes.append(
        helper.make_node(
            "Pad",
            ["out3"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, C - 2, H - 3, W - 3],
            value=0.0,
        )
    )

    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
