"""Compact ONNX for ARC task326: keep the top-left 2x2 color pattern.

The examples are generated from a repeating/noisy 2x2 pattern. Although the
larger grid can be viewed in halves or parity classes, the target is exactly
the first two rows and first two columns of the input, with all other output
cells omitted. The ONNX graph uses negative Pad attributes to crop the compact
2x2 one-hot block without slice initializers, then pads it back to the required
NeuroGolf 30x30 tensor only at the final output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "task326"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task326.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

C = 10
H = W = 30
OUT = 2
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
IR_VERSION = 10
OPSET = 10


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference solver on ARC integer grids."""
    arr = np.asarray(grid, dtype=np.int64)
    return arr[:OUT, :OUT].copy()


def _grid_to_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            out[0, int(arr[row, col]), row, col] = 1.0
    return out


def _examples() -> list[tuple[str, int, dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [
        (split, idx, example)
        for split in ("train", "test", "arc-gen")
        for idx, example in enumerate(data.get(split, []))
    ]


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    nodes.extend(
        [
            helper.make_node(
                "Pad",
                [IN_NAME],
                ["out2"],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, OUT - H, OUT - W],
            ),
            helper.make_node(
                "Pad",
                ["out2"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - OUT, W - OUT],
            ),
        ]
    )

    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info])
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def verify_reference() -> None:
    for split, idx, example in _examples():
        actual = solve(example["input"])
        expected = np.asarray(example["output"], dtype=np.int64)
        if not np.array_equal(actual, expected):
            raise AssertionError(
                f"reference failed {split}[{idx}]: "
                f"predicted {actual.tolist()}, expected {expected.tolist()}"
            )


def verify_model(model: onnx.ModelProto) -> None:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    for split, idx, example in _examples():
        actual = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(example["input"])})[0]
        expected = _grid_to_onehot(example["output"])
        if not np.array_equal(actual > 0.0, expected > 0.0):
            pred = (actual[0, :, :OUT, :OUT] > 0.0).argmax(axis=0)
            raise AssertionError(
                f"model failed {split}[{idx}]: "
                f"predicted {pred.tolist()}, expected {example['output']}"
            )


def main() -> None:
    verify_reference()
    model = build_model()
    verify_model(model)
    onnx.save(model, BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {str(result['error']).strip()}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
