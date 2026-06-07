"""Compact ONNX for NeuroGolf task345: red rays detour around gray blockers.

Task rule: the 10x10 input contains black background, red seed cells on the
bottom row, and gray blockers on rows 2 through 6. Each red seed sends a
vertical red ray upward. When a ray reaches a gray cell, the gray cell is
preserved and the ray detours one cell to the right: the cells immediately
right of the blocker and below-right of the blocker become red, and the ray
continues upward in the shifted column. The output preserves all gray cells and
otherwise contains only black background and the traced red paths.
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

from score_model import score_file  # noqa: E402

TASK_ID = "task345"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
N = 10
H = W = 30
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.value_infos: list[onnx.ValueInfoProto] = []
        self.counter = 0

    def unique(self, prefix: str) -> str:
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
        **attrs: Any,
    ) -> str:
        out = self.vi(self.unique(prefix), dtype, shape)
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out


def shift_right(b: Builder, x: str, prefix: str) -> str:
    """Move a row mask one cell to the right, dropping column 9 overflow."""
    left = b.node("Slice", [x, "col0_st", "col9_en", "axes_w"], TensorProto.BOOL, (1, 1, 1, 9), f"{prefix}_left9")
    return b.node("Concat", ["zero_col", left], TensorProto.BOOL, (1, 1, 1, N), prefix, axis=3)


def build_model() -> onnx.ModelProto:
    b = Builder()

    b.init("axes_chw", np.array([1, 2, 3], dtype=np.int64))
    b.init("axes_h", np.array([2], dtype=np.int64))
    b.init("axes_w", np.array([3], dtype=np.int64))
    b.init("red_seed_st", np.array([2, N - 1, 0], dtype=np.int64))
    b.init("red_seed_en", np.array([3, N, N], dtype=np.int64))
    b.init("gray_st", np.array([5, 0, 0], dtype=np.int64))
    b.init("gray_en", np.array([6, N, N], dtype=np.int64))
    b.init("col0_st", np.array([0], dtype=np.int64))
    b.init("col9_en", np.array([N - 1], dtype=np.int64))
    b.init("zero_col", np.zeros((1, 1, 1, 1), dtype=np.bool_))
    b.init("zero_ch_f", np.zeros((1, 1, N, N), dtype=np.float32))
    for row in range(2, 7):
        b.init(f"row{row}_st", np.array([row], dtype=np.int64))
        b.init(f"row{row}_en", np.array([row + 1], dtype=np.int64))

    red_seed_crop = b.node("Slice", ["input", "red_seed_st", "red_seed_en", "axes_chw"], TensorProto.FLOAT, (1, 1, 1, N), "red_seed_crop")
    gray_crop = b.node("Slice", ["input", "gray_st", "gray_en", "axes_chw"], TensorProto.FLOAT, (1, 1, N, N), "gray_crop")
    red_seed = b.node("Cast", [red_seed_crop], TensorProto.BOOL, (1, 1, 1, N), "red_seed", to=TensorProto.BOOL)
    gray = b.node("Cast", [gray_crop], TensorProto.BOOL, (1, 1, N, N), "gray", to=TensorProto.BOOL)

    gray_rows = [""] * N
    for row in range(2, 7):
        gray_rows[row] = b.node(
            "Slice",
            [gray, f"row{row}_st", f"row{row}_en", "axes_h"],
            TensorProto.BOOL,
            (1, 1, 1, N),
            f"gray_r{row}",
        )

    out_rows: list[str] = [""] * N

    pos = red_seed
    out_rows[9] = pos
    out_rows[8] = pos

    hit_above = b.node("And", [pos, gray_rows[6]], TensorProto.BOOL, (1, 1, 1, N), "hit_above_r7")
    below_right = shift_right(b, hit_above, "below_right_r7")
    out_rows[7] = b.node("Or", [pos, below_right], TensorProto.BOOL, (1, 1, 1, N), "row_red_r7")

    for row in range(6, 1, -1):
        gray_row = gray_rows[row]
        hit = b.node("And", [pos, gray_row], TensorProto.BOOL, (1, 1, 1, N), f"hit_r{row}")
        straight = b.node("Xor", [pos, hit], TensorProto.BOOL, (1, 1, 1, N), f"straight_r{row}")
        detour = shift_right(b, hit, f"detour_r{row}")
        resolved = b.node("Or", [straight, detour], TensorProto.BOOL, (1, 1, 1, N), f"resolved_r{row}")

        row_red = resolved
        if row > 2:
            hit_above = b.node("And", [resolved, gray_rows[row - 1]], TensorProto.BOOL, (1, 1, 1, N), f"hit_above_r{row}")
            below_right = shift_right(b, hit_above, f"below_right_r{row}")
            row_red = b.node("Or", [row_red, below_right], TensorProto.BOOL, (1, 1, 1, N), f"row_red_plus_r{row}")
        out_rows[row] = row_red
        pos = resolved

    out_rows[1] = pos
    out_rows[0] = pos

    red10 = b.node("Concat", out_rows, TensorProto.BOOL, (1, 1, N, N), "red10", axis=2)
    occupied = b.node("Or", [red10, gray], TensorProto.BOOL, (1, 1, N, N), "occupied")
    out0 = b.node("Not", [occupied], TensorProto.BOOL, (1, 1, N, N), "out0")
    out0_f = b.node("Cast", [out0], TensorProto.FLOAT, (1, 1, N, N), "out0_f", to=TensorProto.FLOAT)
    red10_f = b.node("Cast", [red10], TensorProto.FLOAT, (1, 1, N, N), "red10_f", to=TensorProto.FLOAT)
    crop_float = b.node(
        "Concat",
        [
            out0_f,
            "zero_ch_f",
            red10_f,
            "zero_ch_f",
            "zero_ch_f",
            gray_crop,
            "zero_ch_f",
            "zero_ch_f",
            "zero_ch_f",
            "zero_ch_f",
        ],
        TensorProto.FLOAT,
        (1, C, N, N),
        "crop_float",
        axis=1,
    )
    b.nodes.append(
        helper.make_node(
            "Pad",
            [crop_float],
            ["output"],
            pads=[0, 0, 0, 0, 0, 0, H - N, W - N],
            mode="constant",
            value=0.0,
        )
    )

    graph = helper.make_graph(
        b.nodes,
        TASK_ID,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)],
        initializer=b.initializers,
        value_info=b.value_infos,
    )
    model = helper.make_model(graph, producer_name="", opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def solve(grid: list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    gray = arr == 5
    pos = arr[N - 1] == 2
    out = np.zeros_like(arr)

    def shift(mask: np.ndarray) -> np.ndarray:
        shifted = np.zeros_like(mask)
        shifted[1:] = mask[:-1]
        return shifted

    for row in range(N - 1, -1, -1):
        hit = pos & gray[row]
        resolved = (pos & ~gray[row]) | shift(hit)
        row_red = resolved.copy()
        if row > 0:
            row_red |= shift(resolved & gray[row - 1])
        row_red &= ~gray[row]
        out[row, row_red] = 2
        out[row, gray[row]] = 5
        pos = resolved
    return out


def grid_to_onehot(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            out[0, int(arr[row, col]), row, col] = 1.0
    return out


def verify_reference() -> None:
    data = json.loads(DATA_PATH.read_text())
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            got = solve(example["input"])
            expected = np.asarray(example["output"], dtype=np.int64)
            if not np.array_equal(got, expected):
                raise AssertionError(f"reference mismatch {split}[{idx}]")


def verify_onnx(path: Path) -> None:
    data = json.loads(DATA_PATH.read_text())
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            x = grid_to_onehot(example["input"])
            expected = grid_to_onehot(example["output"]) > 0.0
            got = session.run(["output"], {"input": x})[0] > 0.0
            if not np.array_equal(got, expected):
                raise AssertionError(f"ONNX mismatch {split}[{idx}]")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    verify_reference()
    model = build_model()
    onnx.save(model, str(BEST_PATH))
    verify_onnx(BEST_PATH)
    result = score_file(BEST_PATH)
    print(
        f"{TASK_ID}: valid={result['valid']} memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']}"
    )
    if result["error"]:
        print(f"error: {str(result['error']).strip()}")


if __name__ == "__main__":
    main()
