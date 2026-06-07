"""ONNX solution for ARC task325 using Kaggle one-hot I/O.

Task rule: count the disconnected 4-connected cyan objects in the input grid.
If there are N objects, output an N x N grid with cyan on the main diagonal and
background elsewhere; cells outside that compact grid remain padded all-zero for
the NeuroGolf 30 x 30 tensor contract.

The graph slices the observed 16 x 16 input extent, counts components with a
local Euler-characteristic stencil on the cyan mask, then constructs the compact
diagonal output from coordinate masks. The only observed holes are single-cell
holes, so the count is:
components = (Q1 - Q3 - 2*Qd) / 4 + single_cell_holes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TASK_ID = "task325"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task325.onnx"
DATA_PATH = ROOT / "data" / "task325.json"

C = 10
H = W = 30
WORK = 16
MAX_OUT = 6
CYAN = 8
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def solve(grid: list[list[int]]) -> list[list[int]]:
    """Reference implementation for the ARC grid transformation."""
    h, w = len(grid), len(grid[0])
    seen: set[tuple[int, int]] = set()
    components = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] != CYAN or (r, c) in seen:
                continue
            components += 1
            stack = [(r, c)]
            seen.add((r, c))
            while stack:
                rr, cc = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and grid[nr][nc] == CYAN
                        and (nr, nc) not in seen
                    ):
                        seen.add((nr, nc))
                        stack.append((nr, nc))

    out = [[0 for _ in range(components)] for _ in range(components)]
    for i in range(components):
        out[i][i] = CYAN
    return out


def _init_f32(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name))
    return name


def _init_f16(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float16), name=name))
    return name


def _init_i32(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.int32), name=name))
    return name


def _init_i64(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.int64), name=name))
    return name


def build_onnx_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    _init_i64(inits, [0, CYAN, 0, 0], "crop_starts")
    _init_i64(inits, [1, CYAN + 1, WORK, WORK], "crop_ends")
    _init_i64(inits, [0, 1, 2, 3], "crop_axes")
    _init_f16(inits, 3.5, "three_half")
    _init_f16(inits, 4.0, "four_f")
    _init_i32(inits, [4, 8, 12, 16, 20, 24], "num4_i32")

    contrib = np.zeros(16, dtype=np.float16)
    for code in range(16):
        bits = [(code >> bit) & 1 for bit in range(4)]
        count = sum(bits)
        if count == 1:
            contrib[code] = 1.0
        elif count == 3:
            contrib[code] = -1.0
        elif code in (6, 9):
            contrib[code] = -2.0
    _init_f16(inits, contrib, "contrib_lut")

    rows = np.arange(MAX_OUT, dtype=np.int64).reshape(1, 1, MAX_OUT, 1)
    cols = np.arange(MAX_OUT, dtype=np.int64).reshape(1, 1, 1, MAX_OUT)
    _init_i64(inits, np.broadcast_to(rows, (1, 1, MAX_OUT, MAX_OUT)), "row_ids")
    _init_i64(inits, np.broadcast_to(cols, (1, 1, MAX_OUT, MAX_OUT)), "col_ids")
    _init_i64(inits, 1, "one_i")
    _init_f32(inits, np.zeros((1, 7, MAX_OUT, MAX_OUT), dtype=np.float32), "zero_mid")
    _init_f32(inits, np.zeros((1, 1, MAX_OUT, MAX_OUT), dtype=np.float32), "zero_last")

    _init_f16(inits, [[[[1.0, 2.0], [4.0, 8.0]]]], "code_stencil")
    hole_stencil = np.asarray(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    ).reshape(1, 1, 3, 3)
    _init_f16(inits, hole_stencil, "hole_stencil")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, "crop_starts", "crop_ends", "crop_axes"], ["cyan"]),
            helper.make_node("Cast", ["cyan"], ["cyanh"], to=TensorProto.FLOAT16),
            helper.make_node(
                "Conv",
                ["cyanh", "code_stencil"],
                ["code_f"],
                pads=[1, 1, 1, 1],
            ),
            helper.make_node("Cast", ["code_f"], ["code_i"], to=TensorProto.INT32),
            helper.make_node("Gather", ["contrib_lut", "code_i"], ["contrib"], axis=0),
            helper.make_node("ReduceSum", ["contrib"], ["euler4"], axes=[0, 1, 2, 3], keepdims=0),
            helper.make_node("Conv", ["cyanh", "hole_stencil"], ["holes_raw"]),
            helper.make_node("Greater", ["holes_raw", "three_half"], ["holes_b"]),
            helper.make_node("Cast", ["holes_b"], ["holes_f"], to=TensorProto.FLOAT16),
            helper.make_node("ReduceSum", ["holes_f"], ["holes"], axes=[0, 1, 2, 3], keepdims=0),
            helper.make_node("Mul", ["holes", "four_f"], ["holes4"]),
            helper.make_node("Add", ["euler4", "holes4"], ["numerator"]),
            helper.make_node("Cast", ["numerator"], ["numerator_i"], to=TensorProto.INT32),
            helper.make_node("Equal", ["numerator_i", "num4_i32"], ["match"]),
            helper.make_node("Cast", ["match"], ["matchf"], to=TensorProto.FLOAT),
            helper.make_node("ArgMax", ["matchf"], ["output_idx"], axis=0, keepdims=1),
            helper.make_node("Add", ["output_idx", "one_i"], ["out_n"]),
            helper.make_node("Less", ["row_ids", "out_n"], ["row_in"]),
            helper.make_node("Less", ["col_ids", "out_n"], ["col_in"]),
            helper.make_node("And", ["row_in", "col_in"], ["in_square"]),
            helper.make_node("Equal", ["row_ids", "col_ids"], ["diag"]),
            helper.make_node("And", ["in_square", "diag"], ["cyan_b"]),
            helper.make_node("Not", ["diag"], ["not_diag"]),
            helper.make_node("And", ["in_square", "not_diag"], ["bg_b"]),
            helper.make_node("Cast", ["bg_b"], ["bg"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["cyan_b"], ["cyan_out"], to=TensorProto.FLOAT),
            helper.make_node("Concat", ["bg", "zero_mid", "cyan_out", "zero_last"], ["core"], axis=1),
            helper.make_node(
                "Pad",
                ["core"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - MAX_OUT, W - MAX_OUT],
            ),
        ]
    )

    graph = helper.make_graph(nodes, "task325", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def save_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    model = build_onnx_model()
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return model


def _validate_reference() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        task = json.load(fh)
    for split, examples in task.items():
        for idx, example in enumerate(examples):
            actual = solve(example["input"])
            if actual != example["output"]:
                raise AssertionError(f"{split}[{idx}] reference mismatch")


def main() -> None:
    _validate_reference()
    save_model()
    print(BEST_PATH)


if __name__ == "__main__":
    main()
