"""Minimal ONNX for ARC task257: overlay four separated quadrants.

Task rule: the 9x9 input is split by a full blue middle row and full blue
middle column.  Ignore the separator and overlay the four 4x4 quadrants into a
4x4 output.  At each local cell, choose the first non-black cell by the
train-learned priority: top-left orange(7), top-right yellow(4),
bottom-left cyan(8), bottom-right magenta(6).  If all four are black, output
black.  The real task JSON contains only this 9x9-to-4x4 case.

ONNX: slice only the four relevant color channels as compact 1x1x4x4 float
masks, concatenate them, and use one 1x1 convolution with signed weights to
score the priority rule directly.  Because correctness is checked by
thresholding at >0, inactive channels may be zero or negative.  The compact
9-channel 4x4 score crop for colors 0..8 is then padded with a zero color-9
channel and spatial zeros to the required 30x30 output.
"""

from __future__ import annotations

import itertools
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, List, Sequence

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task257"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10

QUADRANTS = (
    ("tl", 7, (0, 0)),
    ("tr", 4, (0, 5)),
    ("bl", 8, (5, 0)),
    ("br", 6, (5, 5)),
)


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _make_model(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], name: str) -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(nodes, name, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _load_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _overlay_with_priority(grid: np.ndarray, priority: Sequence[int]) -> np.ndarray:
    out = np.zeros((4, 4), dtype=np.int64)
    occupied = np.zeros((4, 4), dtype=bool)
    for q_idx in priority:
        _name, color, (row0, col0) = QUADRANTS[q_idx]
        mask = grid[row0 : row0 + 4, col0 : col0 + 4] != 0
        take = mask & ~occupied
        out[take] = color
        occupied |= mask
    return out


def learn_priority(data: dict[str, list[dict[str, list[list[int]]]]]) -> tuple[int, ...]:
    matches: list[tuple[int, ...]] = []
    for priority in itertools.permutations(range(len(QUADRANTS))):
        ok = True
        for ex in data["train"]:
            grid = np.asarray(ex["input"], dtype=np.int64)
            expected = np.asarray(ex["output"], dtype=np.int64)
            if grid.shape != (9, 9):
                continue
            if not np.array_equal(_overlay_with_priority(grid, priority), expected):
                ok = False
                break
        if ok:
            matches.append(priority)
    if len(matches) != 1:
        raise AssertionError(f"expected one train priority, found {len(matches)}: {matches}")
    return matches[0]


def solve(grid: np.ndarray, priority: Sequence[int]) -> np.ndarray:
    """Reference solver for the real 9x9 quadrant-overlay examples."""
    g = np.asarray(grid, dtype=np.int64)
    if g.shape != (9, 9):
        raise ValueError(f"{TASK_ID} real JSON has no learned rule for shape {g.shape}")
    return _overlay_with_priority(g, priority)


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
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


def build_linear_conv_model(priority: Sequence[int]) -> onnx.ModelProto:
    if tuple(priority) != (0, 1, 2, 3):
        raise ValueError("linear conv coefficients are specialized to tl,tr,bl,br priority")

    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    float_masks = []
    for name, color, (row0, col0) in QUADRANTS:
        starts = _i64(inits, [0, color, row0, col0], f"{name}_starts")
        ends = _i64(inits, [1, color + 1, row0 + 4, col0 + 4], f"{name}_ends")
        nodes.append(helper.make_node("Slice", [IN_NAME, starts, ends], [f"{name}_f"]))
        float_masks.append(f"{name}_f")

    # Color 9 never appears in this task's outputs, so the compact Conv emits
    # channels 0..8 and the final Pad adds a zero channel 9 in the graph output.
    weights = np.zeros((9, 4, 1, 1), dtype=np.float32)
    bias = np.zeros((9,), dtype=np.float32)
    # Input channel order is tl, tr, bl, br.  Positive score means active.
    weights[0, :, 0, 0] = [-1.0, -1.0, -1.0, -1.0]
    bias[0] = 0.5
    weights[4, :, 0, 0] = [-1.0, 1.0, 0.0, 0.0]
    weights[6, :, 0, 0] = [-1.0, -1.0, -1.0, 1.0]
    weights[7, :, 0, 0] = [1.0, 0.0, 0.0, 0.0]
    weights[8, :, 0, 0] = [-1.0, -1.0, 1.0, 0.0]

    nodes.extend(
        [
            helper.make_node("Concat", float_masks, ["mask4"], axis=1),
            helper.make_node("Conv", ["mask4", _init(inits, weights, "w"), _init(inits, bias, "b")], ["out10"]),
            helper.make_node("Pad", ["out10"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 1, H - 4, W - 4]),
        ]
    )
    return _make_model(nodes, inits, "task257_linear_conv_priority")


def validate_reference(data: dict[str, list[dict[str, list[list[int]]]]], priority: Sequence[int]) -> None:
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            pred = solve(np.asarray(ex["input"], dtype=np.int64), priority)
            expected = np.asarray(ex["output"], dtype=np.int64)
            if not np.array_equal(pred, expected):
                raise AssertionError(f"reference failed {split} example {idx}")


def validate_json(model: onnx.ModelProto, data: dict[str, list[dict[str, list[list[int]]]]]) -> int:
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            grid = np.asarray(ex["input"], dtype=np.int64)
            expected = np.asarray(ex["output"], dtype=np.int64)
            pred_oh = _run_onnx(model, _grid_to_onehot(ex["input"]))
            pred = _onehot_to_grid(pred_oh)[: expected.shape[0], : expected.shape[1]]
            active = pred_oh[0, :, : expected.shape[0], : expected.shape[1]] > 0.0
            if grid.shape != (9, 9) or not np.array_equal(pred, expected) or not np.all(active.sum(axis=0) == 1):
                bad += 1
    return bad


def _score_candidate(
    label: str,
    build: Callable[[], onnx.ModelProto],
    data: dict[str, list[dict[str, list[list[int]]]]],
) -> tuple[int, float, onnx.ModelProto]:
    model = build()
    bad = validate_json(model, data)
    if bad:
        raise AssertionError(f"{label} failed {bad} examples")

    with tempfile.NamedTemporaryFile(suffix=f"_{TASK_ID}.onnx", dir=OUT_DIR, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        onnx.save(model, tmp_path)
        result = score_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not result["valid"]:
        raise AssertionError(f"{label} invalid: {result['error']}")
    assert result["cost"] is not None and result["score"] is not None
    print(
        f"{label}: nodes={len(model.graph.node)} memory={result['memory']} "
        f"params={result['params']} cost={result['cost']} score={result['score']:.6f}"
    )
    return int(result["cost"]), float(result["score"]), model


def main() -> None:
    data = _load_data()
    priority = learn_priority(data)
    priority_names = [QUADRANTS[idx][0] for idx in priority]
    print(f"learned priority: {priority_names}")
    validate_reference(data, priority)

    candidates = [
        _score_candidate("linear-conv", lambda: build_linear_conv_model(priority), data),
    ]
    _cost, _score, best = min(candidates, key=lambda item: item[0])
    onnx.save(best, BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"nodes:   {len(best.graph.node)}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
