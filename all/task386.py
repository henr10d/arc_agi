"""Minimal ONNX for ARC task386: intersect empty cells across a blue divider.

Task rule: each input is a 4x7 grid with a blue/1 separator column at column 3.
For every row and for the three paired cells on either side of the separator,
write green/3 only when both paired cells are black/0; otherwise write black/0.
The output is the resulting 4x3 grid. The side colors themselves are ignored:
orange/7 on the left and gray/5 on the right only mean "occupied".

ONNX: after validating the fixed layout in the local examples, slice channel 0
from columns 0:3 and 4:7, compute the green mask with bool logic, use one Where
with two tiny float channel templates to make the 4-channel crop, and pad that
directly to the required [1,10,30,30] output.
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

from score_model import score_file  # noqa: E402

TASK_ID = "task386"
TASK_NUM = 386
OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
GH = 4
LEFT_W = 3
BLUE_COL = 3
GW = 3
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
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

    def vi(self, name: str, dtype: int, shape: tuple[int, ...]) -> str:
        self.value_infos.append(helper.make_tensor_value_info(name, dtype, list(shape)))
        return name

    def node(
        self,
        op_type: str,
        inputs: list[str],
        dtype: int,
        shape: tuple[int, ...],
        prefix: str,
        **attrs: object,
    ) -> str:
        out = self.vi(self.name(prefix), dtype, shape)
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference solution for task386."""
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros((GH, GW), dtype=np.int64)
    for r in range(GH):
        for c in range(GW):
            if arr[r, c] == 0 and arr[r, BLUE_COL + 1 + c] == 0:
                out[r, c] = 3
    return out


def load_task() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def iter_examples(data: dict[str, list[dict[str, list[list[int]]]]]):
    for split in ("train", "test", "arc-gen"):
        for example in data.get(split, []):
            yield split, example


def validate_fixed_layout(data: dict[str, list[dict[str, list[list[int]]]]]) -> None:
    """Fail fast if the optimized fixed-column model is not justified."""
    for split, example in iter_examples(data):
        grid = example["input"]
        if len(grid) != GH or any(len(row) != 7 for row in grid):
            raise ValueError(f"{split} example is not {GH}x7")
        blue_cols = [c for c in range(7) if all(row[c] == 1 for row in grid)]
        if blue_cols != [BLUE_COL]:
            raise ValueError(f"{split} example has separator columns {blue_cols}, expected [{BLUE_COL}]")
        expected = np.asarray(example["output"], dtype=np.int64)
        actual = solve(grid)
        if expected.shape != (GH, GW) or not np.array_equal(actual, expected):
            raise ValueError(f"{split} example does not match the documented rule")


def grid_to_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def build_model() -> onnx.ModelProto:
    """Build the compact fixed-column model for task386."""
    b = Builder()
    b.init("axes3", np.array([1, 2, 3], dtype=np.int64))
    b.init("left_starts", np.array([0, 0, 0], dtype=np.int64))
    b.init("left_ends", np.array([1, GH, LEFT_W], dtype=np.int64))
    b.init("right_starts", np.array([0, 0, BLUE_COL + 1], dtype=np.int64))
    b.init("right_ends", np.array([1, GH, BLUE_COL + 1 + GW], dtype=np.int64))
    b.init("green_vec", np.array([0, 0, 0, 1], dtype=np.float32).reshape(4, 1, 1))
    b.init("black_vec", np.array([1, 0, 0, 0], dtype=np.float32).reshape(4, 1, 1))

    inp = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    out = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    left_f = b.node(
        "Slice",
        [IN_NAME, "left_starts", "left_ends", "axes3"],
        TensorProto.FLOAT,
        (1, 1, GH, GW),
        "left_f",
    )
    right_f = b.node(
        "Slice",
        [IN_NAME, "right_starts", "right_ends", "axes3"],
        TensorProto.FLOAT,
        (1, 1, GH, GW),
        "right_f",
    )
    left = b.node("Cast", [left_f], TensorProto.BOOL, (1, 1, GH, GW), "left_b", to=TensorProto.BOOL)
    right = b.node("Cast", [right_f], TensorProto.BOOL, (1, 1, GH, GW), "right_b", to=TensorProto.BOOL)
    green = b.node("And", [left, right], TensorProto.BOOL, (1, 1, GH, GW), "green")
    out4 = b.node("Where", [green, "green_vec", "black_vec"], TensorProto.FLOAT, (1, 4, GH, GW), "out4")
    b.nodes.append(
        helper.make_node(
            "Pad",
            [out4],
            [OUT_NAME],
            pads=[0, 0, 0, 0, 0, C - 4, H - GH, W - GW],
        )
    )

    graph = helper.make_graph(
        b.nodes,
        TASK_ID,
        [inp],
        [out],
        initializer=b.initializers,
        value_info=b.value_infos,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task386",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model

def verify_model(model: onnx.ModelProto, data: dict[str, list[dict[str, list[list[int]]]]]) -> int:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    checked = 0
    for split, example in iter_examples(data):
        x = grid_to_onehot(example["input"])
        expected = grid_to_onehot(example["output"])
        got = session.run([OUT_NAME], {IN_NAME: x})[0]
        if not np.array_equal((got > 0.0).astype(np.float32), expected):
            raise AssertionError(f"{split} example failed")
        checked += 1
    return checked


def main() -> None:
    data = load_task()
    validate_fixed_layout(data)

    model = build_model()
    checked = verify_model(model, data)
    onnx.save(model, BEST_PATH)

    final = score_file(BEST_PATH)
    if not final["valid"]:
        raise RuntimeError(f"score_model failed: {final['error']}")
    print(f"validated fixed 4x7 layout with blue separator at column {BLUE_COL}")
    print(f"wrote {BEST_PATH} after verifying {checked} examples")
    print(
        f"final: cost={final['cost']} memory={final['memory']} "
        f"params={final['params']} score={final['score']:.6f}"
    )


if __name__ == "__main__":
    main()
