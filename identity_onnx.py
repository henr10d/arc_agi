"""Build a pass-through ONNX model for NeuroGolf I/O.

Input and output are both float32 one-hot tensors [1, 10, 30, 30].
The graph is a single Identity node: output = input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, 10, 30, 30]
OPSET = 10
IR_VERSION = 10

ROOT = Path(__file__).resolve().parent
DEFAULT_PATH = ROOT / "identity.onnx"


def build_identity_model() -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    node = helper.make_node("Identity", [IN_NAME], [OUT_NAME], name="identity")
    graph = helper.make_graph([node], "identity", [x_info], [y_info])
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def save_model(path: Path = DEFAULT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_identity_model(), str(path))
    return path


def smoke_test(path: Path) -> None:
    rng = np.random.default_rng(0)
    x = np.zeros(SHAPE, dtype=np.float32)
    for r in range(30):
        for c in range(30):
            x[0, int(rng.integers(0, 10)), r, c] = 1.0

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    y = sess.run([OUT_NAME], {IN_NAME: x})[0]
    if not np.array_equal(x, y):
        raise RuntimeError("Identity model output does not match input")


def main() -> None:
    path = save_model()
    smoke_test(path)
    print(f"Saved {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
