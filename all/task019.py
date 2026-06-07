"""ONNX for ARC task019: tile sparse markers and add diagonal cyan guides.

Task rule: the input grid is copied as a 2-by-2 tile, so each non-black cell
reappears at the same coordinate in all four tiled quadrants. Every remaining
black output cell that is diagonally adjacent to any copied non-black cell is
changed to cyan (8). Diagonal adjacency is bounded by the doubled output grid;
it does not wrap around the outer border.

ONNX: all task019 grids are at most 6x6, so infer H and W from the one-hot
valid-cell mask, build the dynamic 2H-by-2W tile inside a fixed 12x12 region
with Mod/Gather, use a small diagonal Conv to find cyan guide cells, then
one-hot encode and pad to the required 30x30 competition tensor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task019"
BEST_PATH = OUT_DIR / "task019.onnx"
DATA_PATH = ROOT / "data" / "task019.json"

C = 10
H = W = 30
CORE = 6
OUT_CORE = 12
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _i32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int32), name)


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference implementation for validation against the JSON examples."""
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    out = np.tile(g, (2, 2))
    cyan = np.zeros_like(out, dtype=bool)
    for r, c in np.argwhere(out > 0):
        for dr in (-1, 1):
            rr = int(r) + dr
            if not (0 <= rr < 2 * h):
                continue
            for dc in (-1, 1):
                cc = int(c) + dc
                if 0 <= cc < 2 * w:
                    cyan[rr, cc] = True
    out[(out == 0) & cyan] = 8
    return out


def _grid_to_onehot(grid: List[List[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    arr = onehot.reshape(C, H, W)
    active = arr > 0.0
    grid = arr.argmax(axis=0).astype(np.int64)
    grid[active.sum(axis=0) == 0] = 0
    return grid


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    st6 = _i64(inits, [0, 0, 0, 0], "st6")
    en6 = _i64(inits, [1, C, CORE, CORE], "en6")
    shape36 = _i64(inits, [CORE * CORE], "shape36")
    zero_f = _f32(inits, [0.0], "zero_f")
    half = _f32(inits, [0.5], "half")
    diag_kernel = np.array(
        [[[[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]]],
        dtype=np.float32,
    )
    _f32(inits, diag_kernel, "diag_kernel")

    rows12 = _i64(inits, np.arange(OUT_CORE).reshape(1, 1, OUT_CORE, 1), "rows12")
    cols12 = _i64(inits, np.arange(OUT_CORE).reshape(1, 1, 1, OUT_CORE), "cols12")
    six = _i64(inits, [CORE], "six")
    two = _i64(inits, [2], "two")
    minus_one = _i32(inits, [-1], "minus_one")
    zero = _i32(inits, [0], "zero")
    eight = _i32(inits, [8], "eight")
    color_vec = _i32(inits, np.arange(C).reshape(1, C, 1, 1), "color_vec")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st6, en6, axes4], ["core"]),
            helper.make_node("ReduceSum", ["core"], ["valid_sum"], axes=[1], keepdims=1),
            helper.make_node("ReduceMax", ["valid_sum"], ["row_sum"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["valid_sum"], ["col_sum"], axes=[2], keepdims=1),
            helper.make_node("Greater", ["row_sum", half], ["row_has"]),
            helper.make_node("Greater", ["col_sum", half], ["col_has"]),
            helper.make_node("Cast", ["row_has"], ["row_i"], to=TensorProto.INT64),
            helper.make_node("Cast", ["col_has"], ["col_i"], to=TensorProto.INT64),
            helper.make_node("ReduceSum", ["row_i"], ["grid_h"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceSum", ["col_i"], ["grid_w"], axes=[2, 3], keepdims=1),
            helper.make_node("ArgMax", ["core"], ["color64"], axis=1, keepdims=1),
            helper.make_node("Cast", ["color64"], ["color6"], to=TensorProto.INT32),
            helper.make_node("Reshape", ["color6", shape36], ["color_flat"]),
            helper.make_node("Mod", [rows12, "grid_h"], ["src_r"]),
            helper.make_node("Mod", [cols12, "grid_w"], ["src_c"]),
            helper.make_node("Mul", ["src_r", six], ["src_r6"]),
            helper.make_node("Add", ["src_r6", "src_c"], ["src_idx"]),
            helper.make_node("Gather", ["color_flat", "src_idx"], ["tiled"]),
            helper.make_node("Mul", ["grid_h", two], ["out_h"]),
            helper.make_node("Mul", ["grid_w", two], ["out_w"]),
            helper.make_node("Less", [rows12, "out_h"], ["row_active"]),
            helper.make_node("Less", [cols12, "out_w"], ["col_active"]),
            helper.make_node("And", ["row_active", "col_active"], ["active"]),
            helper.make_node("Where", ["active", "tiled", minus_one], ["tile_active"]),
            helper.make_node("Greater", ["tile_active", zero], ["nonblack"]),
            helper.make_node("Cast", ["nonblack"], ["nonblack_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "Conv",
                ["nonblack_f", "diag_kernel"],
                ["diag_count"],
                pads=[1, 1, 1, 1],
            ),
            helper.make_node("Greater", ["diag_count", zero_f], ["diag_hit"]),
            helper.make_node("Equal", ["tile_active", zero], ["is_zero"]),
            helper.make_node("And", ["diag_hit", "is_zero"], ["cyan0"]),
            helper.make_node("And", ["cyan0", "active"], ["cyan"]),
            helper.make_node("Where", ["cyan", eight, "tile_active"], ["final_color"]),
            helper.make_node("Equal", ["final_color", color_vec], ["out12b"]),
            helper.make_node("Cast", ["out12b"], ["out12"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out12"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - OUT_CORE, W - OUT_CORE],
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


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            g = np.array(ex["input"], dtype=np.int64)
            exp = np.array(ex["output"], dtype=np.int64)
            ref = solve(g)
            if not np.array_equal(ref, exp):
                raise AssertionError(f"reference solver mismatch in {split}[{idx}]")
            pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(ex["input"])))
            pred = pred[: exp.shape[0], : exp.shape[1]]
            if not np.array_equal(pred, exp):
                bad += 1
    return bad


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    bad = validate_json(model)
    assert bad == 0, f"{bad} JSON examples failed"
    result = score_file(BEST_PATH)
    print(result)


if __name__ == "__main__":
    main()
