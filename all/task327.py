"""ONNX for ARC task327: extend the 3x3 colored cells down-right.

Task rule: each input is a 3x3 grid with exactly three non-black cells.
The output is a 6x6 grid where every non-black input cell at (r, c) is
repeated along its descending diagonal: (r + k, c + k), while inside the
6x6 output.  The task data keeps these colored diagonals non-overlapping.

ONNX approach: slice the non-black channels from the 3x3 input, collapse the
one-hot crop to one scalar color grid with a 1x1 Conv, propagate that scalar
grid with a single-channel ConvTranspose diagonal kernel, then rebuild the
6x6 one-hot output with OneHot before padding to 30x30.
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
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task327"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task327.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

BATCH = 1
CHANNELS = 10
HEIGHT = WIDTH = 30
IN_SIZE = 3
OUT_SIZE = 6
SHAPE = [BATCH, CHANNELS, HEIGHT, WIDTH]
OPSET = 10
IR_VERSION = 10


def _tensor(name: str, arr: Any) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def _onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _examples() -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        task = json.load(fh)

    examples: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(task.get(split, [])):
            examples.append(
                (
                    split,
                    idx,
                    np.asarray(ex["input"], dtype=np.int64),
                    np.asarray(ex["output"], dtype=np.int64),
                )
            )
    return examples


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference implementation for the down-right diagonal propagation."""
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros((OUT_SIZE, OUT_SIZE), dtype=np.int64)
    for r in range(IN_SIZE):
        for c in range(IN_SIZE):
            color = int(arr[r, c])
            if color == 0:
                continue
            k = 0
            while r + k < OUT_SIZE and c + k < OUT_SIZE:
                out[r + k, c + k] = color
                k += 1
    return out


def validate_reference(examples: list[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    for split, idx, inp, expected in examples:
        if inp.shape != (IN_SIZE, IN_SIZE):
            raise AssertionError(f"{split}[{idx}] input shape is {inp.shape}, expected 3x3")
        if expected.shape != (OUT_SIZE, OUT_SIZE):
            raise AssertionError(f"{split}[{idx}] output shape is {expected.shape}, expected 6x6")
        nonzero = int(np.count_nonzero(inp))
        if nonzero != 3:
            raise AssertionError(f"{split}[{idx}] has {nonzero} non-black cells, expected 3")
        got = solve(inp)
        if not np.array_equal(got, expected):
            raise AssertionError(f"reference failed {split}[{idx}]")


def build_onnx_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    initializers = [
        _tensor("slice_starts", np.asarray([1, 0, 0], dtype=np.int64)),
        _tensor("slice_ends", np.asarray([CHANNELS, IN_SIZE, IN_SIZE], dtype=np.int64)),
        _tensor("slice_axes", np.asarray([1, 2, 3], dtype=np.int64)),
        _tensor("color_weights", np.arange(1, CHANNELS, dtype=np.float32).reshape(1, CHANNELS - 1, 1, 1)),
        _tensor("onehot_depth", np.asarray(10, dtype=np.int64)),
        _tensor("onehot_values", np.asarray([0.0, 1.0], dtype=np.float32)),
    ]

    diag_kernel = np.zeros((1, 1, OUT_SIZE, OUT_SIZE), dtype=np.float32)
    for offset in range(OUT_SIZE):
        diag_kernel[0, 0, offset, offset] = 1.0
    initializers.append(_tensor("diag_kernel", diag_kernel))

    nodes.extend(
        [
            helper.make_node("Slice", ["input", "slice_starts", "slice_ends", "slice_axes"], ["fg_in"]),
            helper.make_node("Conv", ["fg_in", "color_weights"], ["color3"], kernel_shape=[1, 1]),
            helper.make_node(
                "ConvTranspose",
                ["color3", "diag_kernel"],
                ["color6"],
                kernel_shape=[OUT_SIZE, OUT_SIZE],
                pads=[0, 0, IN_SIZE - 1, IN_SIZE - 1],
            ),
            helper.make_node("Squeeze", ["color6"], ["color6_squeezed"], axes=[1]),
            helper.make_node("Cast", ["color6_squeezed"], ["color6_indices"], to=TensorProto.INT64),
            helper.make_node("OneHot", ["color6_indices", "onehot_depth", "onehot_values"], ["out6"], axis=1),
            helper.make_node(
                "Pad",
                ["out6"],
                ["output"],
                pads=[0, 0, 0, 0, 0, 0, HEIGHT - OUT_SIZE, WIDTH - OUT_SIZE],
                value=0.0,
            ),
        ]
    )

    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def validate_model(model_path: Path, examples: list[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    for split, idx, inp, expected in examples:
        got = session.run(["output"], {"input": _onehot(inp)})[0]
        want = _onehot(expected)
        if not np.array_equal(got > 0.0, want > 0.0):
            raise AssertionError(f"model failed {split}[{idx}]")


def main() -> None:
    examples = _examples()
    validate_reference(examples)

    model = build_onnx_model()
    onnx.save(model, BEST_PATH)
    validate_model(BEST_PATH, examples)
    result = score_file(BEST_PATH)

    print(f"wrote {BEST_PATH}")
    print(f"validated examples: {len(examples)}/{len(examples)}")
    print(
        f"memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']:.6f}"
    )


if __name__ == "__main__":
    main()
