"""ONNX solution for ARC task252: alternate colors along diagonal stripes.

Task rule: the grid shape is unchanged. Every nonzero cell belongs to a
continuous down-right diagonal stripe starting from the top row or left column.
Within each stripe, cells at even positions keep their original foreground
color and cells at odd positions are recolored to color 4. Background remains
color 0. All examples are square grids up to 15x15 using one foreground color
other than 4.

ONNX: use a 1x1 Conv to recover a compact full-grid foreground plane from the
one-hot input, then use a broadcast Where as the graph output to replace only
odd foreground cells with a float channel-4 selector. The full Where output is
declared as the model output, so it is excluded from scored activation memory.
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

TASK_ID = "task252"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
NC = 9
H = W = 30
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


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver for alternating odd positions in down-right runs."""
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    out = g.copy()
    for r in range(h):
        for c in range(w):
            if g[r, c] == 0:
                continue
            if r > 0 and c > 0 and g[r - 1, c - 1] != 0:
                continue
            i = 0
            while r + i < h and c + i < w and g[r + i, c + i] != 0:
                if i % 2 == 1:
                    out[r + i, c + i] = 4
                i += 1
    return out


def _grid_to_onehot(grid: List[List[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _expected_onehot(grid: List[List[int]]) -> np.ndarray:
    return _grid_to_onehot(grid)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    fg_weights = _f32(inits, np.array([0.0] + [1.0] * (C - 1), dtype=np.float32).reshape(1, C, 1, 1), "fg_weights")
    rows = np.arange(H).reshape(H, 1)
    cols = np.arange(W).reshape(1, W)
    odd_mask = (np.minimum(rows, cols) % 2 == 1).reshape(1, 1, H, W)
    _init(inits, odd_mask, "odd_mask")
    ch4_selector = np.zeros((1, C, 1, 1), dtype=np.float32)
    ch4_selector[0, 4, 0, 0] = 1.0
    _init(inits, ch4_selector, "ch4_selector")
    nodes.extend(
        [
            helper.make_node("Conv", [IN_NAME, fg_weights], ["fg_f"]),
            helper.make_node("Cast", ["fg_f"], ["is_fg"], to=TensorProto.BOOL),
            helper.make_node("And", ["is_fg", "odd_mask"], ["make_four"]),
            helper.make_node("Where", ["make_four", "ch4_selector", IN_NAME], [OUT_NAME]),
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


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            g = np.asarray(ex["input"], dtype=np.int64)
            if g.shape[0] > H or g.shape[1] > W:
                continue
            expected_grid = solve(g)
            if not np.array_equal(expected_grid, np.asarray(ex["output"], dtype=np.int64)):
                raise AssertionError(f"reference mismatch on {split}[{idx}]")
            pred = _run_onnx(model, _grid_to_onehot(ex["input"]))
            exp = _expected_onehot(ex["output"])
            if not np.array_equal(pred > 0.0, exp > 0.0):
                bad += 1
    return bad


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    bad = validate_json(model)
    print(f"{DATA_PATH.name}: {'PASS' if bad == 0 else f'FAIL ({bad} wrong)'}")

    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"nodes:   {len(model.graph.node)}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
