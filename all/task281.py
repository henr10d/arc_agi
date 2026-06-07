"""ONNX for ARC task281: stretch a framed rectangle toward the cyan marker.

Task rule: the input has one two-color framed rectangle and one isolated cyan
marker (8).  The marker shares either the rectangle's row span or its column
span.  Stretch the frame to the bounding-box union of the original rectangle
and marker.  The border and fill colors are preserved, the marker is removed,
and the rest of the original grid is background.

ONNX approach: work in the 13x13 active area used by this dataset.  Detect the
non-background, non-cyan object mask and cyan marker, compute original and
expanded bounding boxes from coordinate reductions, recover border/fill colors
from the original frame, render a boolean one-hot crop, cast once to float, and
pad to the required 30x30 output.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file, sanitize_model  # noqa: E402

TASK_ID = "task281"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task281.onnx"
DATA_PATH = ROOT / "data" / "task281.json"

C = 10
H = W = 30
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _bool(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=bool), name)


class GraphBuilder:
    def __init__(self) -> None:
        self.nodes: List[onnx.NodeProto] = []
        self.inits: List[onnx.TensorProto] = []
        self.counter = 0

    def name(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def node(self, op_type: str, inputs: Sequence[str], output: str, **attrs: Any) -> str:
        self.nodes.append(helper.make_node(op_type, list(inputs), [output], **attrs))
        return output

    def greater(self, a: str, b: str, out: str) -> str:
        return self.node("Greater", [a, b], out)

    def less(self, a: str, b: str, out: str) -> str:
        return self.node("Less", [a, b], out)

    def equal(self, a: str, b: str, out: str) -> str:
        return self.node("Equal", [a, b], out)

    def ge(self, a: str, b: str, out: str) -> str:
        gt = self.greater(a, b, self.name(f"{out}_gt"))
        eq = self.equal(a, b, self.name(f"{out}_eq"))
        return self.node("Or", [gt, eq], out)

    def le(self, a: str, b: str, out: str) -> str:
        lt = self.less(a, b, self.name(f"{out}_lt"))
        eq = self.equal(a, b, self.name(f"{out}_eq"))
        return self.node("Or", [lt, eq], out)

    def box_mask(
        self,
        rows: str,
        cols: str,
        top: str,
        bottom: str,
        left: str,
        right: str,
        one: str,
        prefix: str,
    ) -> tuple[str, str]:
        top_prev = self.node("Sub", [top, one], f"{prefix}_top_prev")
        bottom_next = self.node("Add", [bottom, one], f"{prefix}_bottom_next")
        left_prev = self.node("Sub", [left, one], f"{prefix}_left_prev")
        right_next = self.node("Add", [right, one], f"{prefix}_right_next")
        row_ge = self.greater(rows, top_prev, f"{prefix}_row_ge")
        row_le = self.less(rows, bottom_next, f"{prefix}_row_le")
        col_ge = self.greater(cols, left_prev, f"{prefix}_col_ge")
        col_le = self.less(cols, right_next, f"{prefix}_col_le")
        in_rows = self.node("And", [row_ge, row_le], f"{prefix}_in_rows")
        in_cols = self.node("And", [col_ge, col_le], f"{prefix}_in_cols")
        box = self.node("And", [in_rows, in_cols], f"{prefix}_box")

        top_eq = self.equal(rows, top, f"{prefix}_top_eq")
        bottom_eq = self.equal(rows, bottom, f"{prefix}_bottom_eq")
        left_eq = self.equal(cols, left, f"{prefix}_left_eq")
        right_eq = self.equal(cols, right, f"{prefix}_right_eq")
        row_edge = self.node("Or", [top_eq, bottom_eq], f"{prefix}_row_edge")
        col_edge = self.node("Or", [left_eq, right_eq], f"{prefix}_col_edge")
        edge = self.node("Or", [row_edge, col_edge], f"{prefix}_edge")
        border = self.node("And", [box, edge], f"{prefix}_border")
        return box, border


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


def connected_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = grid != 0
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def solve_grid(grid: np.ndarray) -> np.ndarray:
    comps = connected_components(grid)
    marker_comp = [comp for comp in comps if len(comp) == 1 and grid[comp[0]] == 8][0]
    object_comp = max((comp for comp in comps if comp != marker_comp), key=len)
    mr, mc = marker_comp[0]

    coords = np.asarray(object_comp, dtype=np.int64)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    border_color = int(grid[r0, c0])
    inner_colors = [
        int(grid[r, c])
        for r, c in object_comp
        if r0 < r < r1 and c0 < c < c1 and grid[r, c] != 0
    ]
    inner_color = inner_colors[0]

    nr0, nr1, nc0, nc1 = min(r0, mr), max(r1, mr), min(c0, mc), max(c1, mc)

    out = np.zeros_like(grid)
    out[nr0 : nr1 + 1, nc0 : nc1 + 1] = inner_color
    out[nr0, nc0 : nc1 + 1] = border_color
    out[nr1, nc0 : nc1 + 1] = border_color
    out[nr0 : nr1 + 1, nc0] = border_color
    out[nr0 : nr1 + 1, nc1] = border_color
    return out


def load_examples() -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    examples: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            examples.append(
                (
                    split,
                    idx,
                    np.asarray(ex["input"], dtype=np.int64),
                    np.asarray(ex["output"], dtype=np.int64),
                )
            )
    return examples


def _grid_to_onehot(grid: np.ndarray) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            out[0, int(grid[r, c]), r, c] = 1.0
    return out


def _expected_onehot(grid: np.ndarray) -> np.ndarray:
    return _grid_to_onehot(grid) > 0.0


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_reference(examples: Iterable[tuple[str, int, np.ndarray, np.ndarray]]) -> dict[str, int]:
    correct = {"train": 0, "test": 0, "arc-gen": 0}
    total = {"train": 0, "test": 0, "arc-gen": 0}
    for split, _idx, inp, expected in examples:
        total[split] += 1
        pred = solve_grid(inp)
        if np.array_equal(pred, expected):
            correct[split] += 1
    return {split: correct[split] for split in total}


def validate_model(model: onnx.ModelProto, examples: Iterable[tuple[str, int, np.ndarray, np.ndarray]]) -> dict[str, tuple[int, int]]:
    counts = {"train": [0, 0], "test": [0, 0], "arc-gen": [0, 0]}
    for split, idx, inp, expected in examples:
        counts[split][1] += 1
        pred = _run_onnx(model, _grid_to_onehot(inp)) > 0.0
        exp = _expected_onehot(expected)
        if np.array_equal(pred, exp):
            counts[split][0] += 1
        else:
            raise AssertionError(f"{split}[{idx}] failed")
    return {split: (vals[0], vals[1]) for split, vals in counts.items()}


def realized_tensor_count(model: onnx.ModelProto) -> int:
    sanitized = sanitize_model(model)
    if sanitized is None:
        return -1
    graph = onnx.shape_inference.infer_shapes(sanitized, strict_mode=True).graph
    return sum(1 for node in graph.node for out in node.output if out and out != OUT_NAME)


def build_model(g: int) -> onnx.ModelProto:
    b = GraphBuilder()
    starts_crop = _i64(b.inits, [0, 0, 0, 0], "starts_crop")
    ends_crop = _i64(b.inits, [1, C, g, g], "ends_crop")
    starts_bg = _i64(b.inits, [0, 0, 0, 0], "starts_bg")
    ends_bg = _i64(b.inits, [1, 1, g, g], "ends_bg")
    starts_cyan = _i64(b.inits, [0, 8, 0, 0], "starts_cyan")
    ends_cyan = _i64(b.inits, [1, 9, g, g], "ends_cyan")
    half = _f32(b.inits, [0.5], "half")
    big = _f32(b.inits, [[[[99.0]]]], "big")
    zero = _f32(b.inits, [[[[0.0]]]], "zero")
    one = _i64(b.inits, [[[[1]]]], "one")
    rows_f = _f32(b.inits, np.arange(g, dtype=np.float32).reshape(1, 1, g, 1), "rows_f")
    cols_f = _f32(b.inits, np.arange(g, dtype=np.float32).reshape(1, 1, 1, g), "cols_f")
    rows = _i64(b.inits, np.arange(g, dtype=np.int64).reshape(1, 1, g, 1), "rows")
    cols = _i64(b.inits, np.arange(g, dtype=np.int64).reshape(1, 1, 1, g), "cols")
    bg_colors = np.zeros((1, C, 1, 1), dtype=bool)
    bg_colors[0, 0, 0, 0] = True
    bg_color = _bool(b.inits, bg_colors, "bg_color")

    crop = b.node("Slice", [IN_NAME, starts_crop, ends_crop], "crop")
    bg0 = b.node("Slice", [IN_NAME, starts_bg, ends_bg], "bg0")
    cyan = b.node("Slice", [IN_NAME, starts_cyan, ends_cyan], "cyan")

    grid_any_f = b.node("ReduceMax", [crop], "grid_any_f", axes=[1], keepdims=1)
    grid = b.greater(grid_any_f, half, "grid")
    bg0_b = b.greater(bg0, half, "bg0_b")
    cyan_b = b.greater(cyan, half, "cyan_b")
    not_bg0 = b.node("Not", [bg0_b], "not_bg0")
    not_cyan = b.node("Not", [cyan_b], "not_cyan")
    colored = b.node("And", [grid, not_bg0], "colored")
    obj = b.node("And", [colored, not_cyan], "obj")
    obj_f = b.node("Cast", [obj], "obj_f", to=TensorProto.FLOAT)

    row_has_f = b.node("ReduceMax", [obj_f], "row_has_f", axes=[3], keepdims=1)
    col_has_f = b.node("ReduceMax", [obj_f], "col_has_f", axes=[2], keepdims=1)
    row_has = b.greater(row_has_f, half, "row_has")
    col_has = b.greater(col_has_f, half, "col_has")
    row_min_src = b.node("Where", [row_has, rows_f, big], "row_min_src")
    col_min_src = b.node("Where", [col_has, cols_f, big], "col_min_src")
    row_max_src = b.node("Where", [row_has, rows_f, zero], "row_max_src")
    col_max_src = b.node("Where", [col_has, cols_f, zero], "col_max_src")
    r0_f = b.node("ReduceMin", [row_min_src], "r0_f", axes=[2, 3], keepdims=1)
    c0_f = b.node("ReduceMin", [col_min_src], "c0_f", axes=[2, 3], keepdims=1)
    r1_f = b.node("ReduceMax", [row_max_src], "r1_f", axes=[2, 3], keepdims=1)
    c1_f = b.node("ReduceMax", [col_max_src], "c1_f", axes=[2, 3], keepdims=1)
    r0 = b.node("Cast", [r0_f], "r0", to=TensorProto.INT64)
    c0 = b.node("Cast", [c0_f], "c0", to=TensorProto.INT64)
    r1 = b.node("Cast", [r1_f], "r1", to=TensorProto.INT64)
    c1 = b.node("Cast", [c1_f], "c1", to=TensorProto.INT64)

    mark_rows = b.node("Mul", [cyan, rows_f], "mark_rows")
    mark_cols = b.node("Mul", [cyan, cols_f], "mark_cols")
    mr_f = b.node("ReduceSum", [mark_rows], "mr_f", axes=[2, 3], keepdims=1)
    mc_f = b.node("ReduceSum", [mark_cols], "mc_f", axes=[2, 3], keepdims=1)
    mr = b.node("Cast", [mr_f], "mr", to=TensorProto.INT64)
    mc = b.node("Cast", [mc_f], "mc", to=TensorProto.INT64)

    above = b.less(mr, r0, "above")
    below = b.greater(mr, r1, "below")
    left = b.less(mc, c0, "left")
    right = b.greater(mc, c1, "right")
    top = b.node("Where", [above, mr, r0], "top")
    bottom = b.node("Where", [below, mr, r1], "bottom")
    out_left = b.node("Where", [left, mc, c0], "out_left")
    out_right = b.node("Where", [right, mc, c1], "out_right")

    orig_box, orig_border = b.box_mask(rows, cols, r0, r1, c0, c1, one, "orig")
    new_box, new_border = b.box_mask(rows, cols, top, bottom, out_left, out_right, one, "new")
    not_new_border = b.node("Not", [new_border], "not_new_border")
    new_inner = b.node("And", [new_box, not_new_border], "new_inner")

    orig_border_f = b.node("Cast", [orig_border], "orig_border_f", to=TensorProto.FLOAT)
    not_orig_border = b.node("Not", [orig_border], "not_orig_border")
    orig_inner = b.node("And", [orig_box, not_orig_border], "orig_inner")
    orig_inner_f = b.node("Cast", [orig_inner], "orig_inner_f", to=TensorProto.FLOAT)
    border_pixels = b.node("Mul", [crop, orig_border_f], "border_pixels")
    inner_pixels = b.node("Mul", [crop, orig_inner_f], "inner_pixels")
    border_color_f = b.node("ReduceMax", [border_pixels], "border_color_f", axes=[2, 3], keepdims=1)
    inner_color_f = b.node("ReduceMax", [inner_pixels], "inner_color_f", axes=[2, 3], keepdims=1)
    border_color = b.greater(border_color_f, half, "border_color")
    inner_color = b.greater(inner_color_f, half, "inner_color")

    paint_border = b.node("And", [new_border, border_color], "paint_border")
    paint_inner = b.node("And", [new_inner, inner_color], "paint_inner")
    paint_obj = b.node("Or", [paint_border, paint_inner], "paint_obj")
    not_new_box = b.node("Not", [new_box], "not_new_box")
    bg = b.node("And", [grid, not_new_box], "bg")
    paint_bg = b.node("And", [bg, bg_color], "paint_bg")
    out_b = b.node("Or", [paint_obj, paint_bg], "out_b")
    out_crop = b.node("Cast", [out_b], "out_crop", to=TensorProto.FLOAT)
    b.node("Pad", [out_crop], OUT_NAME, pads=[0, 0, 0, 0, 0, 0, H - g, W - g])
    return _make_model(b.nodes, b.inits, f"{TASK_ID}_{g}x{g}")


def build_model_color_ids(g: int) -> onnx.ModelProto:
    b = GraphBuilder()
    starts_crop = _i64(b.inits, [0, 0, 0, 0], "ids_starts_crop")
    ends_crop = _i64(b.inits, [1, C, g, g], "ids_ends_crop")
    starts_bg = _i64(b.inits, [0, 0, 0, 0], "ids_starts_bg")
    ends_bg = _i64(b.inits, [1, 1, g, g], "ids_ends_bg")
    half = _f32(b.inits, [0.5], "ids_half")
    big_f = _f32(b.inits, [[[[99.0]]]], "ids_big_f")
    zero_f = _f32(b.inits, [[[[0.0]]]], "ids_zero_f")
    zero = _i64(b.inits, [[[[0]]]], "ids_zero")
    one = _i64(b.inits, [[[[1]]]], "ids_one")
    eight = _i64(b.inits, [[[[8]]]], "ids_eight")
    rows_f = _f32(b.inits, np.arange(g, dtype=np.float32).reshape(1, 1, g, 1), "ids_rows_f")
    cols_f = _f32(b.inits, np.arange(g, dtype=np.float32).reshape(1, 1, 1, g), "ids_cols_f")
    rows_i = _i64(b.inits, np.arange(g, dtype=np.int64).reshape(1, 1, g, 1), "ids_rows_i")
    cols_i = _i64(b.inits, np.arange(g, dtype=np.int64).reshape(1, 1, 1, g), "ids_cols_i")
    channel_i = _i64(b.inits, np.arange(C, dtype=np.int64).reshape(1, C, 1, 1), "ids_channel_i")

    crop = b.node("Slice", [IN_NAME, starts_crop, ends_crop], "ids_crop")
    bg0 = b.node("Slice", [IN_NAME, starts_bg, ends_bg], "ids_bg0")
    color_id = b.node("ArgMax", [crop], "ids_color_id", axis=1, keepdims=1)
    bg0_b = b.greater(bg0, half, "ids_bg0_b")

    nonzero = b.greater(color_id, zero, "ids_nonzero")
    cyan_b = b.equal(color_id, eight, "ids_cyan_b")
    not_cyan = b.node("Not", [cyan_b], "ids_not_cyan")
    obj = b.node("And", [nonzero, not_cyan], "ids_obj")

    obj_f = b.node("Cast", [obj], "ids_obj_f", to=TensorProto.FLOAT)
    row_has_f = b.node("ReduceMax", [obj_f], "ids_row_has_f", axes=[3], keepdims=1)
    col_has_f = b.node("ReduceMax", [obj_f], "ids_col_has_f", axes=[2], keepdims=1)
    row_has = b.greater(row_has_f, half, "ids_row_has")
    col_has = b.greater(col_has_f, half, "ids_col_has")
    row_min_src = b.node("Where", [row_has, rows_f, big_f], "ids_row_min_src")
    col_min_src = b.node("Where", [col_has, cols_f, big_f], "ids_col_min_src")
    row_max_src = b.node("Where", [row_has, rows_f, zero_f], "ids_row_max_src")
    col_max_src = b.node("Where", [col_has, cols_f, zero_f], "ids_col_max_src")
    r0_f = b.node("ReduceMin", [row_min_src], "ids_r0_f", axes=[2, 3], keepdims=1)
    c0_f = b.node("ReduceMin", [col_min_src], "ids_c0_f", axes=[2, 3], keepdims=1)
    r1_f = b.node("ReduceMax", [row_max_src], "ids_r1_f", axes=[2, 3], keepdims=1)
    c1_f = b.node("ReduceMax", [col_max_src], "ids_c1_f", axes=[2, 3], keepdims=1)
    r0 = b.node("Cast", [r0_f], "ids_r0", to=TensorProto.INT64)
    c0 = b.node("Cast", [c0_f], "ids_c0", to=TensorProto.INT64)
    r1 = b.node("Cast", [r1_f], "ids_r1", to=TensorProto.INT64)
    c1 = b.node("Cast", [c1_f], "ids_c1", to=TensorProto.INT64)

    cyan_f = b.node("Cast", [cyan_b], "ids_cyan_f", to=TensorProto.FLOAT)
    mark_rows = b.node("Mul", [cyan_f, rows_f], "ids_mark_rows")
    mark_cols = b.node("Mul", [cyan_f, cols_f], "ids_mark_cols")
    mr_f = b.node("ReduceSum", [mark_rows], "ids_mr_f", axes=[2, 3], keepdims=1)
    mc_f = b.node("ReduceSum", [mark_cols], "ids_mc_f", axes=[2, 3], keepdims=1)
    mr = b.node("Cast", [mr_f], "ids_mr", to=TensorProto.INT64)
    mc = b.node("Cast", [mc_f], "ids_mc", to=TensorProto.INT64)

    above = b.less(mr, r0, "ids_above")
    below = b.greater(mr, r1, "ids_below")
    left = b.less(mc, c0, "ids_left")
    right = b.greater(mc, c1, "ids_right")
    top = b.node("Where", [above, mr, r0], "ids_top")
    bottom = b.node("Where", [below, mr, r1], "ids_bottom")
    out_left = b.node("Where", [left, mc, c0], "ids_out_left")
    out_right = b.node("Where", [right, mc, c1], "ids_out_right")

    orig_box, orig_border = b.box_mask(rows_i, cols_i, r0, r1, c0, c1, one, "ids_orig")
    new_box, new_border = b.box_mask(rows_i, cols_i, top, bottom, out_left, out_right, one, "ids_new")
    not_new_border = b.node("Not", [new_border], "ids_not_new_border")
    new_inner = b.node("And", [new_box, not_new_border], "ids_new_inner")

    not_orig_border = b.node("Not", [orig_border], "ids_not_orig_border")
    orig_inner = b.node("And", [orig_box, not_orig_border], "ids_orig_inner")
    border_id_src = b.node("Where", [orig_border, color_id, zero], "ids_border_id_src")
    inner_id_src = b.node("Where", [orig_inner, color_id, zero], "ids_inner_id_src")
    border_id = b.node("ReduceMax", [border_id_src], "ids_border_id", axes=[2, 3], keepdims=1)
    inner_id = b.node("ReduceMax", [inner_id_src], "ids_inner_id", axes=[2, 3], keepdims=1)

    not_new_box = b.node("Not", [new_box], "ids_not_new_box")
    out_obj_id = b.node("Where", [new_border, border_id, inner_id], "ids_out_obj_id")
    out_id = b.node("Where", [new_box, out_obj_id, zero], "ids_out_id")
    bg = b.node("And", [bg0_b, not_new_box], "ids_bg")
    active = b.node("Or", [new_box, bg], "ids_active")
    out_color = b.equal(channel_i, out_id, "ids_out_color")
    out_b = b.node("And", [out_color, active], "ids_out_b")
    out_crop = b.node("Cast", [out_b], "ids_out_crop", to=TensorProto.FLOAT)
    b.node("Pad", [out_crop], OUT_NAME, pads=[0, 0, 0, 0, 0, 0, H - g, W - g])
    return _make_model(b.nodes, b.inits, f"{TASK_ID}_{g}x{g}_ids")


def print_train_diagnostics(examples: Sequence[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    for split, idx, inp, expected in examples:
        if split != "train":
            continue
        marker = tuple(map(int, np.argwhere(inp == 8)[0]))
        obj = np.argwhere((inp != 0) & (inp != 8))
        r0, c0 = obj.min(axis=0)
        r1, c1 = obj.max(axis=0)
        axis = "vertical" if c0 <= marker[1] <= c1 else "horizontal"
        solved = solve_grid(inp)
        print(
            f"train[{idx}]: marker={marker} box=({r0},{c0})-({r1},{c1}) "
            f"axis={axis} ok={np.array_equal(solved, expected)}"
        )


def score_candidate(label: str, model: onnx.ModelProto, examples: Sequence[tuple[str, int, np.ndarray, np.ndarray]]) -> tuple[int, onnx.ModelProto]:
    counts = validate_model(model, examples)
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
        f"{label}: train={counts['train'][0]}/{counts['train'][1]} "
        f"test={counts['test'][0]}/{counts['test'][1]} arc-gen={counts['arc-gen'][0]}/{counts['arc-gen'][1]} "
        f"nodes={len(model.graph.node)} tensors={realized_tensor_count(model)} "
        f"memory={result['memory']} params={result['params']} cost={result['cost']} "
        f"score={result['score']:.6f}"
    )
    return int(result["cost"]), model


def main() -> None:
    examples = load_examples()
    print_train_diagnostics(examples)
    ref_counts = validate_reference(examples)
    print(
        f"reference: train={ref_counts['train']}/3 test={ref_counts['test']}/1 "
        f"arc-gen={ref_counts['arc-gen']}/262"
    )

    candidates = [
        score_candidate("crop13", build_model(13), examples),
        score_candidate("ids13", build_model_color_ids(13), examples),
        score_candidate("crop30", build_model(30), examples),
    ]
    _cost, best = min(candidates, key=lambda item: item[0])
    onnx.save(best, BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"nodes:   {len(best.graph.node)}")
    print(f"tensors: {realized_tensor_count(best)}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
