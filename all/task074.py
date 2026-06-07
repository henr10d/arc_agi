"""Generate an optimized ONNX model for NeuroGolf task074.

Task rule: the input is a 30x30 symmetric color pattern with occluding cells
marked as color 9. The completed output keeps the true symmetric pattern and
replaces each 9 by the color at an equivalent position under transpose and the
shifted row/column reflections `r -> 31-r` and `c -> 31-c`.

ONNX approach: decode the one-hot input to a compact uint8 class grid, build
the six symmetry candidates, take the per-cell minimum across the input and
candidates (valid colors are 0-8, occlusions are 9), then one-hot encode once
at the final output.
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


OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, print_report, score_file  # noqa: E402


TASK_ID = "task074"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
SOLUTION_PATH = ROOT / "solution.onnx"

H = W = 30
C = 10
IR_VERSION = 10
OPSET = 10


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.value_infos: list[onnx.ValueInfoProto] = []
        self.counter = 0

    def name(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def init(self, name: str, array: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(array, name))
        return name

    def node(
        self,
        op_type: str,
        inputs: list[str],
        dtype: int,
        shape: tuple[int, ...],
        prefix: str,
        **attrs: Any,
    ) -> str:
        out = self.name(prefix)
        self.value_infos.append(helper.make_tensor_value_info(out, dtype, list(shape)))
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out


def add_constants(b: Builder) -> None:
    b.init("colors", np.arange(C, dtype=np.int32).reshape(1, C, 1, 1))
    b.init("rev31", np.array([0, 1, *range(29, 1, -1)], dtype=np.int64))


def min_uint8(b: Builder, current: str, candidate: str, prefix: str) -> str:
    keep_current = b.node("Less", [current, candidate], TensorProto.BOOL, (1, 1, H, W), f"{prefix}_keep_current")
    return b.node("Where", [keep_current, current, candidate], TensorProto.UINT8, (1, 1, H, W), prefix)


def build_model() -> onnx.ModelProto:
    b = Builder()
    add_constants(b)

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, H, W])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, H, W])

    arg = b.node("ArgMax", ["input"], TensorProto.INT64, (1, 1, H, W), "arg", axis=1, keepdims=1)
    grid = b.node("Cast", [arg], TensorProto.UINT8, (1, 1, H, W), "grid", to=TensorProto.UINT8)

    d = b.node("Transpose", [grid], TensorProto.UINT8, (1, 1, H, W), "diag", perm=[0, 1, 3, 2])
    r = b.node("Gather", [grid, "rev31"], TensorProto.UINT8, (1, 1, H, W), "row31", axis=2)
    c = b.node("Gather", [grid, "rev31"], TensorProto.UINT8, (1, 1, H, W), "col31", axis=3)
    cd = b.node("Gather", [d, "rev31"], TensorProto.UINT8, (1, 1, H, W), "col31_diag", axis=3)
    rd = b.node("Gather", [d, "rev31"], TensorProto.UINT8, (1, 1, H, W), "row31_diag", axis=2)
    rc = b.node("Gather", [r, "rev31"], TensorProto.UINT8, (1, 1, H, W), "row_col31", axis=3)

    solved = grid
    solved = min_uint8(b, solved, d, "min_diag")
    solved = min_uint8(b, solved, r, "min_row31")
    solved = min_uint8(b, solved, c, "min_col31")
    solved = min_uint8(b, solved, cd, "min_col31_diag")
    solved = min_uint8(b, solved, rd, "min_row31_diag")
    solved = min_uint8(b, solved, rc, "solved")
    solved32 = b.node("Cast", [solved], TensorProto.INT32, (1, 1, H, W), "solved32", to=TensorProto.INT32)
    onehot = b.node("Equal", [solved32, "colors"], TensorProto.BOOL, (1, C, H, W), "onehot")
    b.nodes.append(helper.make_node("Cast", [onehot], ["output"], to=TensorProto.FLOAT))

    graph = helper.make_graph(b.nodes, TASK_ID, [inp], [out], b.initializers, value_info=b.value_infos)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], ir_version=IR_VERSION)
    return model


def verify_correctness(path: Path) -> None:
    with TASK_PATH.open(encoding="utf-8") as fh:
        task = json.load(fh)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    failures: list[str] = []
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(task.get(split, [])):
            x = convert_to_numpy(example, "input")
            y = convert_to_numpy(example, "output")
            if x is None or y is None:
                continue
            pred = session.run(["output"], {"input": x})[0]
            if not np.array_equal(pred > 0.0, y > 0.0):
                failures.append(f"{split}[{index}]")
                if len(failures) >= 5:
                    break
        if len(failures) >= 5:
            break
    if failures:
        raise AssertionError("failed examples: " + ", ".join(failures))


def main() -> None:
    model = build_model()
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, BEST_PATH)
    onnx.save(model, SOLUTION_PATH)
    verify_correctness(BEST_PATH)
    result = score_file(BEST_PATH)
    print_report(result)


if __name__ == "__main__":
    main()
