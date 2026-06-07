"""ONNX for ARC task273: fill olive-corner rectangle interiors red.

Task rule: the 10x10 input contains black background and olive marker pixels
(color 4).  In the supplied train/test/arc-gen set, markers form one or two
axis-aligned rectangles with row and column spans from 2 to 5 cells; for each
rectangle, fill its non-empty interior red (color 2).  Preserve all olive
markers, including any marker that would otherwise fall inside another
rectangle.

ONNX approach: crop the 10x10 olive channel, detect complete row/column-pair
rectangles through compact pair tables, render the union of their interior
cells, then build one-hot channels 0, 2, and 4 before padding back to 30x30.
"""

from __future__ import annotations

import json
import sys
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task273"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task273.onnx"
DATA_PATH = ROOT / "data" / "task273.json"

C = 10
H = W = 30
G = 10
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _make_model(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto]) -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
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


def row_pairs() -> list[tuple[int, int]]:
    return [(a, b) for a, b in combinations(range(G), 2) if 1 < b - a <= 5]


def col_pairs() -> list[tuple[int, int]]:
    return row_pairs()


def complete_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    for r1, r2 in row_pairs():
        for c1, c2 in col_pairs():
            if grid[r1, c1] == grid[r1, c2] == grid[r2, c1] == grid[r2, c2] == 4:
                rects.append((r1, c1, r2, c2))
    return rects


def maximal_rectangles(rects: Sequence[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    maximal: list[tuple[int, int, int, int]] = []
    for a in rects:
        ar1, ac1, ar2, ac2 = a
        contained = False
        for b in rects:
            if a == b:
                continue
            br1, bc1, br2, bc2 = b
            if br1 <= ar1 and bc1 <= ac1 and br2 >= ar2 and bc2 >= ac2:
                contained = True
                break
        if not contained:
            maximal.append(a)
    return maximal


def render(grid: np.ndarray, rects: Iterable[tuple[int, int, int, int]]) -> np.ndarray:
    out = np.asarray(grid, dtype=np.int64).copy()
    for r1, c1, r2, c2 in rects:
        out[r1 + 1 : r2, c1 + 1 : c2] = 2
    out[grid == 4] = 4
    return out


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver for the complete-olive-rectangle rule."""
    rects = complete_rectangles(grid)
    rects.sort(key=lambda x: (x[2] - x[0] - 1) * (x[3] - x[1] - 1), reverse=True)
    return render(grid, rects)


def load_examples() -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    examples: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            inp = np.asarray(ex["input"], dtype=np.int64)
            out = np.asarray(ex["output"], dtype=np.int64)
            examples.append((split, idx, inp, out))
    return examples


def _grid_to_onehot(grid: np.ndarray) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            out[0, int(grid[r, c]), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def print_hypothesis_diagnostics(examples: Sequence[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    bad_all = 0
    bad_maximal = 0
    bad_area_first = 0
    rect_counts: dict[int, int] = {}
    for _split, _idx, inp, expected in examples:
        rects = complete_rectangles(inp)
        rect_counts[len(rects)] = rect_counts.get(len(rects), 0) + 1
        if not np.array_equal(render(inp, rects), expected):
            bad_all += 1
        if not np.array_equal(render(inp, maximal_rectangles(rects)), expected):
            bad_maximal += 1
        area_first = sorted(rects, key=lambda x: (x[2] - x[0] - 1) * (x[3] - x[1] - 1), reverse=True)
        if not np.array_equal(render(inp, area_first), expected):
            bad_area_first += 1
    print(f"hypothesis all-complete mismatches: {bad_all}")
    print(f"hypothesis maximal-only mismatches: {bad_maximal}")
    print(f"hypothesis area-first mismatches: {bad_area_first}")
    print(f"complete rectangle count histogram: {dict(sorted(rect_counts.items()))}")


def validate_reference(examples: Sequence[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    for split, idx, inp, expected in examples:
        if inp.shape != (G, G) or expected.shape != (G, G):
            raise AssertionError(f"{split}[{idx}] is not 10x10: {inp.shape} -> {expected.shape}")
        pred = solve(inp)
        if not np.array_equal(pred, expected):
            diff = int((pred != expected).sum())
            raise AssertionError(f"reference failed {split}[{idx}] with {diff} mismatches")


def validate_onnx(model: onnx.ModelProto, examples: Sequence[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    for split, idx, inp, expected in examples:
        pred_oh = _run_onnx(model, _grid_to_onehot(inp))
        active = pred_oh[0, :, :G, :G] > 0.0
        pred = _onehot_to_grid(pred_oh)[:G, :G]
        if not np.array_equal(pred, expected) or not np.all(active.sum(axis=0) == 1):
            diff = int((pred != expected).sum())
            invalid = int((active.sum(axis=0) != 1).sum())
            raise AssertionError(f"onnx failed {split}[{idx}] diff={diff} invalid_onehot={invalid}")


def build_pair_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = row_pairs()

    top_rows = np.asarray([r1 for r1, _r2 in rows], dtype=np.int64)
    bottom_rows = np.asarray([r2 for _r1, r2 in rows], dtype=np.int64)

    row_interior_t = np.zeros((G, len(rows)), dtype=np.float32)
    for i, (r1, r2) in enumerate(rows):
        row_interior_t[r1 + 1 : r2, i] = 1.0

    left_prefix = np.zeros((G, G), dtype=np.float32)
    right_suffix = np.zeros((G, G), dtype=np.float32)
    for marker_col in range(G):
        for out_col in range(G):
            if marker_col < out_col:
                left_prefix[marker_col, out_col] = 1.0
            if marker_col > out_col:
                right_suffix[marker_col, out_col] = 1.0

    return top_rows, bottom_rows, row_interior_t, left_prefix, right_suffix


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    top_rows, bottom_rows, row_interior_t, left_prefix, right_suffix = build_pair_tables()

    starts = _i64(inits, [0, 4, 0, 0], "starts")
    ends = _i64(inits, [1, 5, G, G], "ends")
    top_idx = _i64(inits, top_rows, "top_idx")
    bottom_idx = _i64(inits, bottom_rows, "bottom_idx")
    row_int_t = _f32(inits, row_interior_t, "row_int_t")
    left_pref = _f32(inits, left_prefix, "left_pref")
    right_suf = _f32(inits, right_suffix, "right_suf")
    zero_f = _f32(inits, 0.0, "zero_f")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts, ends], ["marker_f"]),
            helper.make_node("Cast", ["marker_f"], ["marker_b"], to=TensorProto.BOOL),
            helper.make_node("Gather", ["marker_b", top_idx], ["top_vals"], axis=2),
            helper.make_node("Gather", ["marker_b", bottom_idx], ["bottom_vals"], axis=2),
            helper.make_node("And", ["top_vals", "bottom_vals"], ["side_b"]),
            helper.make_node("Cast", ["side_b"], ["side_f"], to=TensorProto.FLOAT),
            helper.make_node("MatMul", ["side_f", left_pref], ["left_counts"]),
            helper.make_node("MatMul", ["side_f", right_suf], ["right_counts"]),
            helper.make_node("Greater", ["left_counts", zero_f], ["has_left"]),
            helper.make_node("Greater", ["right_counts", zero_f], ["has_right"]),
            helper.make_node("And", ["has_left", "has_right"], ["col_inside_b"]),
            helper.make_node("Cast", ["col_inside_b"], ["col_inside_f"], to=TensorProto.FLOAT),
            helper.make_node("MatMul", [row_int_t, "col_inside_f"], ["fill_sums"]),
            helper.make_node("Greater", ["fill_sums", zero_f], ["red"]),
            helper.make_node("Or", ["marker_b", "red"], ["occupied"]),
            helper.make_node("Not", ["occupied"], ["bg"]),
            helper.make_node("And", ["marker_b", "red"], ["zero_b"]),
            helper.make_node(
                "Concat",
                ["bg", "zero_b", "red", "zero_b", "marker_b"],
                ["out5b"],
                axis=1,
            ),
            helper.make_node("Cast", ["out5b"], ["out5"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out5"], [OUT_NAME], pads=[0, 0, 0, 0, 0, C - 5, H - G, W - G]),
        ]
    )
    return _make_model(nodes, inits)


def main() -> None:
    examples = load_examples()
    print_hypothesis_diagnostics(examples)
    validate_reference(examples)

    model = build_model()
    validate_onnx(model, examples)

    with tempfile.NamedTemporaryFile(suffix=f"_{TASK_ID}.onnx", dir=OUT_DIR, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        onnx.save(model, tmp_path)
        trial = score_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    if not trial["valid"]:
        raise AssertionError(f"candidate invalid: {trial['error']}")

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
