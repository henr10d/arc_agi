"""Generate an ONNX model for NeuroGolf task339.

Task rule: the input is a 3x3 grid on a black background with exactly one
non-black foreground color.  The output is a single row whose width equals the
number of foreground cells, filled with that foreground color.

The graph sums each non-black color channel over the input.  Since exactly one
non-black color is present, each channel count is either zero or the required
row length.  Subtracting column indices 0..8 from those counts gives positive
values exactly on the foreground channel and only before the output row ends;
the final Pad expands that compact 9x9 slab to the required 30x30 tensor.
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

from score_model import convert_to_numpy, print_report, score_file  # noqa: E402

TASK_ID = "task339"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
SOLUTION_PATH = ROOT / "solution.onnx"

C = 10
H = W = 30
CORE = 3
MAX_COUNT = CORE * CORE
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
IR_VERSION = 10
OPSET = 10


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference implementation for the count-to-row transformation."""
    arr = np.asarray(grid, dtype=np.int64)
    colors = sorted({int(v) for v in arr.reshape(-1) if int(v) != 0})
    if len(colors) != 1:
        raise ValueError(f"{TASK_ID} expects exactly one foreground color, got {colors}")
    count = int(np.count_nonzero(arr))
    return np.full((1, count), colors[0], dtype=np.int64)


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    starts = _init(inits, np.array([1], dtype=np.int64), "starts")
    ends = _init(inits, np.array([C], dtype=np.int64), "ends")
    axes = _init(inits, np.array([1], dtype=np.int64), "axes")
    cols = _init(inits, np.arange(MAX_COUNT, dtype=np.float32).reshape(1, 1, 1, MAX_COUNT), "cols")

    nodes.extend(
        [
            helper.make_node("ReduceSum", [IN_NAME], ["all_color_counts"], axes=[2, 3], keepdims=1),
            helper.make_node("Slice", ["all_color_counts", starts, ends, axes], ["color_counts"]),
            helper.make_node("Sub", ["color_counts", cols], ["row_float"]),
            helper.make_node(
                "Pad",
                ["row_float"],
                [OUT_NAME],
                pads=[0, 1, 0, 0, 0, 0, H - 1, W - MAX_COUNT],
            ),
        ]
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


def _examples() -> list[tuple[str, int, dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    out: list[tuple[str, int, dict[str, list[list[int]]]]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            out.append((split, idx, example))
    return out


def verify_reference() -> None:
    for split, idx, example in _examples():
        got = solve(example["input"])
        expected = np.asarray(example["output"], dtype=np.int64)
        if not np.array_equal(got, expected):
            raise AssertionError(f"reference mismatch on {split}[{idx}]: got {got}, expected {expected}")


def verify_model(path: Path) -> None:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for split, idx, example in _examples():
        x = convert_to_numpy(example, "input")
        expected = convert_to_numpy(example, "output")
        if x is None or expected is None:
            continue
        got = session.run([OUT_NAME], {IN_NAME: x})[0]
        if not np.array_equal(got > 0.0, expected > 0.0):
            raise AssertionError(f"model mismatch on {split}[{idx}]")


def main() -> None:
    verify_reference()
    model = build_model()
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, BEST_PATH)
    onnx.save(model, SOLUTION_PATH)
    verify_model(BEST_PATH)
    print_report(score_file(BEST_PATH))


if __name__ == "__main__":
    main()
