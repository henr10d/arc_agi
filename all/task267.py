"""Minimal ONNX for ARC task267: recolor one object from a marker cell.

Task rule: the 7x7 input contains one large 4-connected nonzero object and one
isolated one-cell marker at the lower-left corner.  The output keeps the large
object's shape and position, recolors every object cell to the marker color,
and turns the marker cell itself black.

ONNX: use the verified fixed marker position (row 6, col 0).  The object never
uses only the 5x5 interior, so slice that background plane, threshold it to an
object mask, add the known false border, then use one Where to choose either
the marker's one-hot color or background before the final 30x30 pad.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task267"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task267.onnx"
DATA_PATH = ROOT / "data" / "task267.json"

C = 10
H = W = 30
G = 7
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


def _bool(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.bool_), name)


def _components4(grid: np.ndarray) -> List[dict]:
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[dict] = []
    for r in range(h):
        for c in range(w):
            if g[r, c] == 0 or seen[r, c]:
                continue
            cells: List[Tuple[int, int]] = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and g[ny, nx] != 0:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append({"cells": cells, "size": len(cells), "color": int(g[cells[0]])})
    return comps


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference component solver for the marker-recolor rule."""
    g = np.asarray(grid, dtype=np.int64)
    comps = _components4(g)
    if not comps:
        return np.zeros_like(g)

    singletons = [comp for comp in comps if comp["size"] == 1]
    larger = [comp for comp in comps if comp["size"] > 1]
    if len(singletons) > 1 and larger:
        object_colors = {int(g[r, c]) for comp in larger for r, c in comp["cells"]}
        filtered = [comp for comp in singletons if comp["color"] not in object_colors]
        if filtered:
            singletons = filtered

    marker = singletons[0]
    main = max(larger, key=lambda comp: comp["size"]) if larger else max(comps, key=lambda comp: comp["size"])
    marker_r, marker_c = marker["cells"][0]
    out = np.zeros_like(g)
    for r, c in main["cells"]:
        out[r, c] = int(g[marker_r, marker_c])
    return out


def _grid_to_onehot(grid: Sequence[Sequence[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, [0, 1, 2, 3], "axes")
    bg_st = _i64(inits, [0, 0, 1, 1], "bg_st")
    bg_en = _i64(inits, [1, 1, G - 1, G - 1], "bg_en")
    marker_st = _i64(inits, [0, 0, G - 1, 0], "marker_st")
    marker_en = _i64(inits, [1, C, G, 1], "marker_en")
    one = _f32(inits, [1.0], "one")
    background = np.zeros((1, C, 1, 1), dtype=np.float32)
    background[0, 0, 0, 0] = 1.0
    bg_color = _f32(inits, background, "bg_color")
    false_col = _bool(inits, np.zeros((1, 1, G - 2, 1), dtype=np.bool_), "false_col")
    false_row = _bool(inits, np.zeros((1, 1, 1, G), dtype=np.bool_), "false_row")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, bg_st, bg_en, axes], ["bg_in"]),
            helper.make_node("Less", ["bg_in", one], ["occupied_mid"]),
            helper.make_node("Concat", [false_col, "occupied_mid", false_col], ["occupied_rows"], axis=3),
            helper.make_node("Concat", [false_row, "occupied_rows", false_row], ["main"], axis=2),
            helper.make_node("Slice", [IN_NAME, marker_st, marker_en, axes], ["marker_f"]),
            helper.make_node("Where", ["main", "marker_f", bg_color], ["out10"]),
            helper.make_node("Pad", ["out10"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - G, W - G]),
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
    marker_positions = set()
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            g = np.asarray(ex["input"], dtype=np.int64)
            expected = np.asarray(ex["output"], dtype=np.int64)
            ref = solve(g)
            if not np.array_equal(ref, expected):
                raise AssertionError(f"reference solver mismatch on {split}[{idx}]")

            singles = [comp for comp in _components4(g) if comp["size"] == 1]
            marker_positions.update(comp["cells"][0] for comp in singles)

            pred_oh = _run_onnx(model, _grid_to_onehot(ex["input"]))
            pred = _onehot_to_grid(pred_oh)[: g.shape[0], : g.shape[1]]
            active = pred_oh[0, :, : expected.shape[0], : expected.shape[1]] > 0.0
            if not np.array_equal(pred, expected) or not np.all(active.sum(axis=0) == 1):
                bad += 1

    if marker_positions != {(G - 1, 0)}:
        raise AssertionError(f"fixed-marker ONNX assumption failed: {sorted(marker_positions)}")
    return bad


def main() -> None:
    model = build_model()
    bad = validate_json(model)
    if bad:
        raise AssertionError(f"{TASK_ID}.json validation failed on {bad} examples")

    onnx.save(model, BEST_PATH)
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
