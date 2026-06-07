"""ONNX generator for ARC task240: unfold corner markers into nested frames.

The 19x19 grid uses only odd row/column positions.  Compress those cells to a
9x9 lattice.  Each non-zero marker describes one part of a rectangle ring:
diagonal markers (d, d) color that ring's four corners, markers at (d, d+1)
color the top/bottom sides, and markers at (d+1, d) color the left/right
sides.  The markers may appear near any outer corner, but the output is the
same nested-frame pattern: every lattice cell reads the marker belonging to its
nearest ring and side, then the lattice is expanded back to odd cells.

The ONNX graph collapses the odd input cells to integer color IDs, gathers the
four symmetric source candidates for each output lattice cell, expands directly
to a 30x30 integer grid, and one-hot encodes only at the final output size.
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

TASK_ID = "task240"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
GRID = 19
LATTICE = 9
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


def source_indices() -> np.ndarray:
    """Return four symmetric source candidates for each output-lattice cell."""
    idx = []
    for r in range(LATTICE):
        for c in range(LATTICE):
            dr = min(r, LATTICE - 1 - r)
            dc = min(c, LATTICE - 1 - c)
            if dr == dc:
                sr, sc = dr, dc
            elif dr < dc:
                sr, sc = dr, dr + 1
            else:
                sr, sc = dc + 1, dc
            idx.append(
                [
                    sr * LATTICE + sc,
                    sr * LATTICE + (LATTICE - 1 - sc),
                    (LATTICE - 1 - sr) * LATTICE + sc,
                    (LATTICE - 1 - sr) * LATTICE + (LATTICE - 1 - sc),
                ]
            )
    return np.asarray(idx, dtype=np.int64)


def expansion_indices() -> np.ndarray:
    """Map each 19x19 output cell to a lattice cell, or index 81 for zero."""
    idx = np.full((GRID, GRID), LATTICE * LATTICE, dtype=np.int64)
    zero_idx = LATTICE * LATTICE
    for r in range(GRID):
        for c in range(GRID):
            if r % 2 == 1 and c % 2 == 1:
                idx[r, c] = ((r - 1) // 2) * LATTICE + ((c - 1) // 2)
            else:
                idx[r, c] = zero_idx
    return idx


def expansion_indices_30() -> np.ndarray:
    """Map each 30x30 output cell to a lattice cell, or index 81 for zero/padding."""
    idx = np.full((H, W), LATTICE * LATTICE, dtype=np.int64)
    idx[:GRID, :GRID] = expansion_indices()
    return idx


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference implementation on the original integer grid."""
    g = np.asarray(grid, dtype=np.int64)
    odd = g[1:GRID:2, 1:GRID:2]
    flat = odd.reshape(-1)
    candidates = flat[source_indices()]
    out_lattice = np.where(candidates != 0, candidates, 0).max(axis=1).reshape(LATTICE, LATTICE)
    out = np.zeros((GRID, GRID), dtype=np.int64)
    out[1:GRID:2, 1:GRID:2] = out_lattice
    return out


def _grid_to_onehot(grid) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, [0, 1, 2, 3], "axes")
    starts = _i64(inits, [0, 0, 1, 1], "starts")
    ends = _i64(inits, [1, C, GRID - 1, GRID - 1], "ends")
    steps = _i64(inits, [1, 1, 2, 2], "steps")
    flat_shape = _i64(inits, [1, LATTICE * LATTICE], "flat_shape")
    src_idx = _i64(inits, source_indices(), "src_idx")
    exp_idx = _i64(inits, expansion_indices_30(), "exp_idx")
    zero_slot = _i64(inits, [[0]], "zero_slot")
    colors = _i64(inits, np.arange(C, dtype=np.int64).reshape(1, C, 1, 1), "colors")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts, ends, axes, steps], ["odd10"]),
            helper.make_node("ArgMax", ["odd10"], ["odd_color"], axis=1, keepdims=0),
            helper.make_node("Reshape", ["odd_color", flat_shape], ["color_flat"]),
            helper.make_node("Gather", ["color_flat", src_idx], ["candidates"], axis=1),
            helper.make_node("ReduceMax", ["candidates"], ["lat_color"], axes=[2], keepdims=0),
            helper.make_node("Concat", ["lat_color", zero_slot], ["lat_ext"], axis=1),
            helper.make_node("Gather", ["lat_ext", exp_idx], ["color_grid"], axis=1),
            helper.make_node("Equal", ["color_grid", colors], ["out_bool"]),
            helper.make_node("Cast", ["out_bool"], [OUT_NAME], to=TensorProto.FLOAT),
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


def validate_json(model: onnx.ModelProto) -> tuple[int, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    total = 0
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data.get(split, []):
            g = np.asarray(ex["input"], dtype=np.int64)
            expected = np.asarray(ex["output"], dtype=np.int64)
            if g.shape[0] > H or g.shape[1] > W:
                continue
            total += 1
            assert np.array_equal(solve(g), expected), f"reference mismatch in {split}"
            pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(ex["input"])))[: expected.shape[0], : expected.shape[1]]
            if not np.array_equal(pred, expected):
                bad += 1
    return total, bad


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    total, bad = validate_json(model)
    assert bad == 0, f"{bad}/{total} examples failed"
    result = score_file(BEST_PATH)
    print(f"validated {total} examples")
    print(
        f"{BEST_PATH.name}: memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']:.6f}"
    )


if __name__ == "__main__":
    main()
