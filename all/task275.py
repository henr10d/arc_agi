"""ONNX for ARC task275: recolor and tile the cyan stencil.

Task rule: the input is two equal square panels, either side-by-side or
stacked.  One panel is a cyan (8) stencil and the other is a color map using
colors 1-4 plus black.  For each non-black cell in the color map, copy the
entire cyan stencil into the corresponding output block and recolor it with
that cell's color.  The output side length is n*n for an n by n panel.

ONNX approach: build compact rule candidates for n=3 and n=4, both layouts,
and both possible stencil sides.  Shape/cyan-panel detectors select exactly
one candidate; compact 5-channel outputs are padded to the required 30x30
one-hot competition tensor only after candidate selection.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task275"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task275.onnx"
DATA_PATH = ROOT / "data" / "task275.json"

C = 10
H = W = 30
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10
CELL_STRIDE = H * W


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


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


def _flat_index(color: int, row: int, col: int) -> int:
    return color * CELL_STRIDE + row * W + col


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


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid.shape
    if w == 2 * h:
        n = h
        return grid[:, :n], grid[:, n:]
    if h == 2 * w:
        n = w
        return grid[:n, :], grid[n:, :]
    raise ValueError(f"unexpected input shape {grid.shape}")


def solve_tile(grid: np.ndarray) -> np.ndarray:
    """Reference implementation of the cyan-stencil tiling rule."""
    a, b = _split_panels(grid)
    a_is_mask = set(np.unique(a)).issubset({0, 8}) and bool(np.any(a == 8))
    b_is_mask = set(np.unique(b)).issubset({0, 8}) and bool(np.any(b == 8))
    if a_is_mask == b_is_mask:
        raise ValueError("expected exactly one cyan stencil panel")
    mask = a == 8 if a_is_mask else b == 8
    colors = b if a_is_mask else a
    n = mask.shape[0]
    out = np.zeros((n * n, n * n), dtype=np.int64)
    for row in range(n):
        for col in range(n):
            color = int(colors[row, col])
            if color:
                out[row * n : (row + 1) * n, col * n : (col + 1) * n][mask] = color
    return out


def _glyph_hypothesis(grid: np.ndarray) -> np.ndarray:
    """Rejected training diagnostic for the originally suspected glyph rule."""
    active_rows = np.flatnonzero(np.any((grid != 0) & (grid != 8), axis=1))
    if active_rows.size == 0:
        return np.zeros((1, 1), dtype=np.int64)
    compact = grid[active_rows]
    out_size = 4 * active_rows.size
    out = np.zeros((out_size, out_size), dtype=np.int64)
    for color in (1, 2, 3, 4):
        coords = np.argwhere(compact == color)
        for row, col in coords:
            y = int(round(row * (out_size - 1) / max(1, compact.shape[0] - 1)))
            x = int(round(col * (out_size - 1) / max(1, compact.shape[1] - 1)))
            if color == 1:
                points = ((0, 0), (1, 0), (1, 1))
            else:
                points = ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0))
            for dy, dx in points:
                yy, xx = y + dy, x + dx
                if 0 <= yy < out_size and 0 <= xx < out_size:
                    out[yy, xx] = color
    return out


def print_diagnostics(examples: Sequence[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    train = [(split, idx, inp, out) for split, idx, inp, out in examples if split == "train"]
    glyph_bad = sum(
        1
        for _split, _idx, inp, out in train
        if _glyph_hypothesis(inp).shape != out.shape or not np.array_equal(_glyph_hypothesis(inp), out)
    )
    tile_bad = sum(1 for _split, _idx, inp, out in examples if not np.array_equal(solve_tile(inp), out))
    print(f"candidate centroid-glyph rejected on {glyph_bad}/{len(train)} train examples")
    print(f"candidate square-panel tiling mismatches: {tile_bad}/{len(examples)}")
    if tile_bad:
        raise AssertionError("tile rule failed local data")


def grid_to_onehot(grid: np.ndarray) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            out[0, int(grid[row, col]), row, col] = 1.0
    return out


def run_onnx(model: onnx.ModelProto, grid: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    pred = sess.run([OUT_NAME], {IN_NAME: grid_to_onehot(grid)})[0]
    return pred[0]


def validate_json(model: onnx.ModelProto, examples: Iterable[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    for split, idx, _inp, expected in examples:
        pred = run_onnx(model, _inp)
        active = pred[:, : expected.shape[0], : expected.shape[1]] > 0.0
        decoded = active.argmax(axis=0)
        pad_active = np.any(pred[:, expected.shape[0] :, :] > 0.0) or np.any(
            pred[:, :, expected.shape[1] :] > 0.0
        )
        if not np.array_equal(decoded, expected) or not np.all(active.sum(axis=0) == 1) or pad_active:
            diff = int(np.sum(decoded != expected))
            raise AssertionError(f"{split}[{idx}] failed with {diff} mismatched cells")


def _sum_tensors(nodes: List[onnx.NodeProto], names: Sequence[str], prefix: str) -> str:
    if len(names) == 1:
        return names[0]
    current = names[0]
    for idx, name in enumerate(names[1:], start=1):
        out = f"{prefix}_sum_{idx}"
        nodes.append(helper.make_node("Add", [current, name], [out]))
        current = out
    return current


def _sum_bool_or(nodes: List[onnx.NodeProto], names: Sequence[str], prefix: str) -> str:
    if len(names) == 1:
        return names[0]
    current = names[0]
    for idx, name in enumerate(names[1:], start=1):
        out = f"{prefix}_or_{idx}"
        nodes.append(helper.make_node("Or", [current, name], [out]))
        current = out
    return current


def _any_cell(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], flat: str, row: int, col: int) -> str:
    indices = _i64(inits, [_flat_index(color, row, col) for color in range(C)], f"cell_idx_{row}_{col}")
    half = _f32(inits, [0.5], f"half_{row}_{col}")
    nodes.extend(
        [
            helper.make_node("Gather", [flat, indices], [f"cell_vals_{row}_{col}"], axis=0),
            helper.make_node("ReduceSum", [f"cell_vals_{row}_{col}"], [f"cell_sum_{row}_{col}"], axes=[0], keepdims=1),
            helper.make_node("Greater", [f"cell_sum_{row}_{col}", half], [f"cell_any_{row}_{col}"]),
        ]
    )
    return f"cell_any_{row}_{col}"


def _panel_cyan(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    flat: str,
    name: str,
    rows: range,
    cols: range,
) -> str:
    indices = [_flat_index(8, row, col) for row in rows for col in cols]
    idx_name = _i64(inits, indices, f"{name}_cyan_idx")
    half = _f32(inits, [0.5], f"{name}_cyan_half")
    nodes.extend(
        [
            helper.make_node("Gather", [flat, idx_name], [f"{name}_cyan_vals"], axis=0),
            helper.make_node("ReduceSum", [f"{name}_cyan_vals"], [f"{name}_cyan_sum"], axes=[0], keepdims=1),
            helper.make_node("Greater", [f"{name}_cyan_sum", half], [f"{name}_cyan"]),
        ]
    )
    return f"{name}_cyan"


def _candidate_indices(
    n: int,
    layout: str,
    mask_first: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    if layout == "h":
        left = (0, 0)
        right = (0, n)
        mask_origin = left if mask_first else right
        color_origin = right if mask_first else left
        side = "left" if mask_first else "right"
    elif layout == "v":
        top = (0, 0)
        bottom = (n, 0)
        mask_origin = top if mask_first else bottom
        color_origin = bottom if mask_first else top
        side = "top" if mask_first else "bottom"
    else:
        raise ValueError(layout)

    out_side = n * n
    color_indices = np.empty((4, out_side, out_side), dtype=np.int64)
    mask_indices = np.empty((out_side, out_side), dtype=np.int64)
    for out_r in range(out_side):
        for out_c in range(out_side):
            color_r = color_origin[0] + out_r // n
            color_c = color_origin[1] + out_c // n
            mask_r = mask_origin[0] + out_r % n
            mask_c = mask_origin[1] + out_c % n
            mask_indices[out_r, out_c] = _flat_index(8, mask_r, mask_c)
            for offset, color in enumerate((1, 2, 3, 4)):
                color_indices[offset, out_r, out_c] = _flat_index(color, color_r, color_c)
    return color_indices, mask_indices, f"n{n}_{layout}_mask_{side}"


def _panel_origins(n: int, layout: str, mask_first: bool) -> tuple[tuple[int, int], tuple[int, int], str]:
    if layout == "h":
        left = (0, 0)
        right = (0, n)
        side = "left" if mask_first else "right"
        return (left if mask_first else right), (right if mask_first else left), side
    if layout == "v":
        top = (0, 0)
        bottom = (n, 0)
        side = "top" if mask_first else "bottom"
        return (top if mask_first else bottom), (bottom if mask_first else top), side
    raise ValueError(layout)


def _panel_indices(
    n: int,
    layout: str,
    mask_first: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    mask_origin, color_origin, side = _panel_origins(n, layout, mask_first)
    color_indices = np.empty((4, n, n), dtype=np.int64)
    mask_indices = np.empty((n, n), dtype=np.int64)
    for row in range(n):
        for col in range(n):
            mask_indices[row, col] = _flat_index(8, mask_origin[0] + row, mask_origin[1] + col)
            for offset, color in enumerate((1, 2, 3, 4)):
                color_indices[offset, row, col] = _flat_index(
                    color,
                    color_origin[0] + row,
                    color_origin[1] + col,
                )
    return color_indices, mask_indices, f"n{n}_{layout}_mask_{side}"


def _build_compact_candidate(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    flat: str,
    n: int,
    layout: str,
    mask_first: bool,
    selector: str,
) -> str:
    color_indices, mask_indices, name = _candidate_indices(n, layout, mask_first)
    color_idx = _i64(inits, color_indices, f"{name}_color_idx")
    mask_idx = _i64(inits, mask_indices, f"{name}_mask_idx")
    shape_1111 = _i64(inits, [1, 1, 1, 1], f"{name}_sel_shape")
    out_side = n * n

    nodes.extend(
        [
            helper.make_node("Gather", [flat, color_idx], [f"{name}_color_f"], axis=0),
            helper.make_node("Cast", [f"{name}_color_f"], [f"{name}_color_b"], to=TensorProto.BOOL),
            helper.make_node("Gather", [flat, mask_idx], [f"{name}_mask_f"], axis=0),
            helper.make_node("Cast", [f"{name}_mask_f"], [f"{name}_mask_b"], to=TensorProto.BOOL),
            helper.make_node("And", [f"{name}_color_b", f"{name}_mask_b"], [f"{name}_paint4"]),
        ]
    )
    channels = [f"{name}_c{i}" for i in range(1, 5)]
    nodes.append(helper.make_node("Split", [f"{name}_paint4"], channels, axis=0, split=[1, 1, 1, 1]))
    nodes.extend(
        [
            helper.make_node("Or", [channels[0], channels[1]], [f"{name}_or12"]),
            helper.make_node("Or", [channels[2], channels[3]], [f"{name}_or34"]),
            helper.make_node("Or", [f"{name}_or12", f"{name}_or34"], [f"{name}_painted"]),
            helper.make_node("Not", [f"{name}_painted"], [f"{name}_bg"]),
            helper.make_node("Concat", [f"{name}_bg", *channels], [f"{name}_compact_b0"], axis=0),
            helper.make_node("Unsqueeze", [f"{name}_compact_b0"], [f"{name}_compact_b"], axes=[0]),
            helper.make_node("Cast", [f"{name}_compact_b"], [f"{name}_compact_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", [selector], [f"{name}_selector_f0"], to=TensorProto.FLOAT),
            helper.make_node("Reshape", [f"{name}_selector_f0", shape_1111], [f"{name}_selector_f"]),
            helper.make_node("Mul", [f"{name}_compact_f", f"{name}_selector_f"], [f"{name}_selected"]),
        ]
    )
    assert out_side in {9, 16}
    return f"{name}_selected"


def _build_selectors(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    flat: str,
) -> dict[tuple[int, str, bool], str]:
    any_0_3 = _any_cell(nodes, inits, flat, 0, 3)
    any_0_5 = _any_cell(nodes, inits, flat, 0, 5)
    any_0_7 = _any_cell(nodes, inits, flat, 0, 7)
    any_3_0 = _any_cell(nodes, inits, flat, 3, 0)
    any_5_0 = _any_cell(nodes, inits, flat, 5, 0)
    any_7_0 = _any_cell(nodes, inits, flat, 7, 0)
    nodes.extend(
        [
            helper.make_node("Not", [any_3_0], ["not_any_3_0"]),
            helper.make_node("Not", [any_0_3], ["not_any_0_3"]),
            helper.make_node("And", [any_0_5, "not_any_3_0"], ["shape_h3"]),
            helper.make_node("And", [any_0_7, any_3_0], ["shape_h4"]),
            helper.make_node("And", [any_5_0, "not_any_0_3"], ["shape_v3"]),
            helper.make_node("And", [any_7_0, any_0_3], ["shape_v4"]),
        ]
    )

    selectors: dict[tuple[int, str, bool], str] = {}
    for n in (3, 4):
        left = _panel_cyan(nodes, inits, flat, f"n{n}_left", range(n), range(n))
        right = _panel_cyan(nodes, inits, flat, f"n{n}_right", range(n), range(n, 2 * n))
        top = _panel_cyan(nodes, inits, flat, f"n{n}_top", range(n), range(n))
        bottom = _panel_cyan(nodes, inits, flat, f"n{n}_bottom", range(n, 2 * n), range(n))
        h_shape = f"shape_h{n}"
        v_shape = f"shape_v{n}"
        for layout, mask_first, panel_cyan, shape_name in (
            ("h", True, left, h_shape),
            ("h", False, right, h_shape),
            ("v", True, top, v_shape),
            ("v", False, bottom, v_shape),
        ):
            sel = f"sel_n{n}_{layout}_{'first' if mask_first else 'second'}"
            nodes.append(helper.make_node("And", [shape_name, panel_cyan], [sel]))
            selectors[(n, layout, mask_first)] = sel
    return selectors


def _build_rule_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    flat_shape = _i64(inits, [C * H * W], "flat_shape")
    nodes.append(helper.make_node("Reshape", [IN_NAME, flat_shape], ["flat"]))

    any_0_3 = _any_cell(nodes, inits, "flat", 0, 3)
    any_0_5 = _any_cell(nodes, inits, "flat", 0, 5)
    any_0_7 = _any_cell(nodes, inits, "flat", 0, 7)
    any_3_0 = _any_cell(nodes, inits, "flat", 3, 0)
    any_5_0 = _any_cell(nodes, inits, "flat", 5, 0)
    any_7_0 = _any_cell(nodes, inits, "flat", 7, 0)
    nodes.extend(
        [
            helper.make_node("Not", [any_3_0], ["not_any_3_0"]),
            helper.make_node("Not", [any_0_3], ["not_any_0_3"]),
            helper.make_node("And", [any_0_5, "not_any_3_0"], ["shape_h3"]),
            helper.make_node("And", [any_0_7, any_3_0], ["shape_h4"]),
            helper.make_node("And", [any_5_0, "not_any_0_3"], ["shape_v3"]),
            helper.make_node("And", [any_7_0, any_0_3], ["shape_v4"]),
        ]
    )

    selectors: dict[tuple[int, str, bool], str] = {}
    for n in (3, 4):
        left = _panel_cyan(nodes, inits, "flat", f"n{n}_left", range(n), range(n))
        right = _panel_cyan(nodes, inits, "flat", f"n{n}_right", range(n), range(n, 2 * n))
        top = _panel_cyan(nodes, inits, "flat", f"n{n}_top", range(n), range(n))
        bottom = _panel_cyan(nodes, inits, "flat", f"n{n}_bottom", range(n, 2 * n), range(n))
        h_shape = f"shape_h{n}"
        v_shape = f"shape_v{n}"
        for layout, mask_first, panel_cyan, shape_name in (
            ("h", True, left, h_shape),
            ("h", False, right, h_shape),
            ("v", True, top, v_shape),
            ("v", False, bottom, v_shape),
        ):
            sel = f"sel_n{n}_{layout}_{'first' if mask_first else 'second'}"
            nodes.append(helper.make_node("And", [shape_name, panel_cyan], [sel]))
            selectors[(n, layout, mask_first)] = sel

    selected_by_size: dict[int, list[str]] = {3: [], 4: []}
    for n in (3, 4):
        for layout, mask_first in (("h", True), ("h", False), ("v", True), ("v", False)):
            selected_by_size[n].append(
                _build_compact_candidate(nodes, inits, "flat", n, layout, mask_first, selectors[(n, layout, mask_first)])
            )

    padded: list[str] = []
    for n in (3, 4):
        out_side = n * n
        compact = _sum_tensors(nodes, selected_by_size[n], f"n{n}_compact")
        padded_name = f"n{n}_padded"
        nodes.append(
            helper.make_node(
                "Pad",
                [compact],
                [padded_name],
                pads=[0, 0, 0, 0, 0, C - 5, H - out_side, W - out_side],
            )
        )
        padded.append(padded_name)
    nodes.append(helper.make_node("Add", padded, [OUT_NAME]))
    return _make_model(nodes, inits, "task275_square_panel_tiling")


def _selector_float(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    selector: str,
    shape: Sequence[int],
    name: str,
) -> str:
    shape_name = _i64(inits, list(shape), f"{name}_shape")
    nodes.extend(
        [
            helper.make_node("Cast", [selector], [f"{name}_f0"], to=TensorProto.FLOAT),
            helper.make_node("Reshape", [f"{name}_f0", shape_name], [f"{name}_f"]),
        ]
    )
    return f"{name}_f"


def _build_merged_size(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    flat: str,
    selectors: dict[tuple[int, str, bool], str],
    n: int,
) -> str:
    out_side = n * n
    selected_colors: list[str] = []
    selected_masks: list[str] = []
    size_selectors: list[str] = []

    for layout, mask_first in (("h", True), ("h", False), ("v", True), ("v", False)):
        color_indices, mask_indices, cand_name = _candidate_indices(n, layout, mask_first)
        selector = selectors[(n, layout, mask_first)]
        size_selectors.append(selector)
        color_idx = _i64(inits, color_indices, f"merged_{cand_name}_color_idx")
        mask_idx = _i64(inits, mask_indices, f"merged_{cand_name}_mask_idx")
        sel_111 = _selector_float(nodes, inits, selector, [1, 1, 1], f"merged_{cand_name}_sel111")
        nodes.extend(
            [
                helper.make_node("Gather", [flat, color_idx], [f"merged_{cand_name}_color_f"], axis=0),
                helper.make_node("Mul", [f"merged_{cand_name}_color_f", sel_111], [f"merged_{cand_name}_color_sel"]),
                helper.make_node("Gather", [flat, mask_idx], [f"merged_{cand_name}_mask_f0"], axis=0),
                helper.make_node("Unsqueeze", [f"merged_{cand_name}_mask_f0"], [f"merged_{cand_name}_mask_f"], axes=[0]),
                helper.make_node("Mul", [f"merged_{cand_name}_mask_f", sel_111], [f"merged_{cand_name}_mask_sel"]),
            ]
        )
        selected_colors.append(f"merged_{cand_name}_color_sel")
        selected_masks.append(f"merged_{cand_name}_mask_sel")

    color_sum = _sum_tensors(nodes, selected_colors, f"merged_n{n}_color")
    mask_sum = _sum_tensors(nodes, selected_masks, f"merged_n{n}_mask")
    size_selector = _sum_bool_or(nodes, size_selectors, f"merged_n{n}_size")
    size_f = _selector_float(nodes, inits, size_selector, [1, 1, 1, 1], f"merged_n{n}_size")
    channels = [f"merged_n{n}_c{i}" for i in range(1, 5)]
    nodes.extend(
        [
            helper.make_node("Cast", [color_sum], [f"merged_n{n}_color_b"], to=TensorProto.BOOL),
            helper.make_node("Cast", [mask_sum], [f"merged_n{n}_mask_b"], to=TensorProto.BOOL),
            helper.make_node("And", [f"merged_n{n}_color_b", f"merged_n{n}_mask_b"], [f"merged_n{n}_paint4"]),
            helper.make_node("Split", [f"merged_n{n}_paint4"], channels, axis=0, split=[1, 1, 1, 1]),
            helper.make_node("Or", [channels[0], channels[1]], [f"merged_n{n}_or12"]),
            helper.make_node("Or", [channels[2], channels[3]], [f"merged_n{n}_or34"]),
            helper.make_node("Or", [f"merged_n{n}_or12", f"merged_n{n}_or34"], [f"merged_n{n}_painted"]),
            helper.make_node("Not", [f"merged_n{n}_painted"], [f"merged_n{n}_bg"]),
            helper.make_node("Concat", [f"merged_n{n}_bg", *channels], [f"merged_n{n}_compact_b0"], axis=0),
            helper.make_node("Unsqueeze", [f"merged_n{n}_compact_b0"], [f"merged_n{n}_compact_b"], axes=[0]),
            helper.make_node("Cast", [f"merged_n{n}_compact_b"], [f"merged_n{n}_compact_f"], to=TensorProto.FLOAT),
            helper.make_node("Mul", [f"merged_n{n}_compact_f", size_f], [f"merged_n{n}_selected"]),
        ]
    )
    padded = f"merged_n{n}_padded"
    nodes.append(
        helper.make_node(
            "Pad",
            [f"merged_n{n}_selected"],
            [padded],
            pads=[0, 0, 0, 0, 0, C - 5, H - out_side, W - out_side],
        )
    )
    return padded


def _build_merged_rule_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    flat_shape = _i64(inits, [C * H * W], "flat_shape")
    nodes.append(helper.make_node("Reshape", [IN_NAME, flat_shape], ["flat"]))

    selectors = _build_selectors(nodes, inits, "flat")
    n3 = _build_merged_size(nodes, inits, "flat", selectors, 3)
    n4 = _build_merged_size(nodes, inits, "flat", selectors, 4)
    nodes.append(helper.make_node("Add", [n3, n4], [OUT_NAME]))
    return _make_model(nodes, inits, "task275_merged_square_panel_tiling")


def _build_kron_size(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    flat: str,
    selectors: dict[tuple[int, str, bool], str],
    n: int,
) -> str:
    selected_colors: list[str] = []
    selected_masks: list[str] = []
    size_selectors: list[str] = []
    for layout, mask_first in (("h", True), ("h", False), ("v", True), ("v", False)):
        color_indices, mask_indices, cand_name = _panel_indices(n, layout, mask_first)
        selector = selectors[(n, layout, mask_first)]
        size_selectors.append(selector)
        color_idx = _i64(inits, color_indices, f"kron_{cand_name}_color_idx")
        mask_idx = _i64(inits, mask_indices, f"kron_{cand_name}_mask_idx")
        sel_111 = _selector_float(nodes, inits, selector, [1, 1, 1], f"kron_{cand_name}_sel111")
        nodes.extend(
            [
                helper.make_node("Gather", [flat, color_idx], [f"kron_{cand_name}_color_f"], axis=0),
                helper.make_node("Mul", [f"kron_{cand_name}_color_f", sel_111], [f"kron_{cand_name}_color_sel"]),
                helper.make_node("Gather", [flat, mask_idx], [f"kron_{cand_name}_mask_f0"], axis=0),
                helper.make_node("Unsqueeze", [f"kron_{cand_name}_mask_f0"], [f"kron_{cand_name}_mask_f"], axes=[0]),
                helper.make_node("Mul", [f"kron_{cand_name}_mask_f", sel_111], [f"kron_{cand_name}_mask_sel"]),
            ]
        )
        selected_colors.append(f"kron_{cand_name}_color_sel")
        selected_masks.append(f"kron_{cand_name}_mask_sel")

    color_sum = _sum_tensors(nodes, selected_colors, f"kron_n{n}_color")
    mask_sum = _sum_tensors(nodes, selected_masks, f"kron_n{n}_mask")
    size_selector = _sum_bool_or(nodes, size_selectors, f"kron_n{n}_size")
    size_f = _selector_float(nodes, inits, size_selector, [1, 1, 1, 1], f"kron_n{n}_size")
    tile_shape = _i64(inits, [4, n * n, n * n], f"kron_n{n}_tile_shape")

    nodes.extend(
        [
            helper.make_node("Unsqueeze", [color_sum], [f"kron_n{n}_color_e"], axes=[2, 4]),
            helper.make_node("Unsqueeze", [mask_sum], [f"kron_n{n}_mask_e"], axes=[1, 3]),
            helper.make_node("Mul", [f"kron_n{n}_color_e", f"kron_n{n}_mask_e"], [f"kron_n{n}_paint5d"]),
            helper.make_node("Reshape", [f"kron_n{n}_paint5d", tile_shape], [f"kron_n{n}_paint4"]),
            helper.make_node("ReduceSum", [f"kron_n{n}_paint4"], [f"kron_n{n}_painted_f"], axes=[0], keepdims=1),
            helper.make_node("Greater", [f"kron_n{n}_painted_f", _f32(inits, [0.5], f"kron_n{n}_half")], [f"kron_n{n}_painted_b"]),
            helper.make_node("Not", [f"kron_n{n}_painted_b"], [f"kron_n{n}_bg_b"]),
            helper.make_node("Cast", [f"kron_n{n}_bg_b"], [f"kron_n{n}_bg_f"], to=TensorProto.FLOAT),
            helper.make_node("Concat", [f"kron_n{n}_bg_f", f"kron_n{n}_paint4"], [f"kron_n{n}_compact0"], axis=0),
            helper.make_node("Unsqueeze", [f"kron_n{n}_compact0"], [f"kron_n{n}_compact"], axes=[0]),
            helper.make_node("Mul", [f"kron_n{n}_compact", size_f], [f"kron_n{n}_selected"]),
        ]
    )
    return f"kron_n{n}_selected"


def _build_kron_rule_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    flat_shape = _i64(inits, [C * H * W], "flat_shape")
    nodes.append(helper.make_node("Reshape", [IN_NAME, flat_shape], ["flat"]))

    selectors = _build_selectors(nodes, inits, "flat")
    n3 = _build_kron_size(nodes, inits, "flat", selectors, 3)
    n4 = _build_kron_size(nodes, inits, "flat", selectors, 4)
    nodes.extend(
        [
            helper.make_node("Pad", [n3], ["kron_n3_to_16"], pads=[0, 0, 0, 0, 0, 0, 7, 7]),
            helper.make_node("Add", ["kron_n3_to_16", n4], ["kron_combined_16"]),
            helper.make_node("Pad", ["kron_combined_16"], [OUT_NAME], pads=[0, 0, 0, 0, 0, C - 5, 14, 14]),
        ]
    )
    return _make_model(nodes, inits, "task275_kron_square_panel_tiling")


def _slice4(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    data: str,
    name: str,
    starts: Sequence[int],
    ends: Sequence[int],
) -> str:
    starts_name = _i64(inits, starts, f"{name}_starts")
    ends_name = _i64(inits, ends, f"{name}_ends")
    nodes.append(helper.make_node("Slice", [data, starts_name, ends_name], [name]))
    return name


def _any_cell_slice(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], row: int, col: int) -> str:
    vals = _slice4(nodes, inits, IN_NAME, f"slice_cell_{row}_{col}", [0, 0, row, col], [1, C, row + 1, col + 1])
    half = _f32(inits, [0.5], f"slice_half_{row}_{col}")
    nodes.extend(
        [
            helper.make_node("ReduceSum", [vals], [f"slice_cell_sum_{row}_{col}"], axes=[0, 1, 2, 3], keepdims=1),
            helper.make_node("Greater", [f"slice_cell_sum_{row}_{col}", half], [f"slice_cell_any_{row}_{col}"]),
        ]
    )
    return f"slice_cell_any_{row}_{col}"


def _panel_cyan_slice(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    name: str,
    row0: int,
    col0: int,
    n: int,
) -> str:
    vals = _slice4(nodes, inits, IN_NAME, f"slice_{name}_cyan_vals", [0, 8, row0, col0], [1, 9, row0 + n, col0 + n])
    half = _f32(inits, [0.5], f"slice_{name}_cyan_half")
    nodes.extend(
        [
            helper.make_node("ReduceSum", [vals], [f"slice_{name}_cyan_sum"], axes=[0, 1, 2, 3], keepdims=1),
            helper.make_node("Greater", [f"slice_{name}_cyan_sum", half], [f"slice_{name}_cyan"]),
        ]
    )
    return f"slice_{name}_cyan"


def _build_slice_selectors(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
) -> dict[tuple[int, str, bool], str]:
    any_0_3 = _any_cell_slice(nodes, inits, 0, 3)
    any_0_5 = _any_cell_slice(nodes, inits, 0, 5)
    any_0_7 = _any_cell_slice(nodes, inits, 0, 7)
    any_3_0 = _any_cell_slice(nodes, inits, 3, 0)
    any_5_0 = _any_cell_slice(nodes, inits, 5, 0)
    any_7_0 = _any_cell_slice(nodes, inits, 7, 0)
    nodes.extend(
        [
            helper.make_node("Not", [any_3_0], ["slice_not_any_3_0"]),
            helper.make_node("Not", [any_0_3], ["slice_not_any_0_3"]),
            helper.make_node("And", [any_0_5, "slice_not_any_3_0"], ["slice_shape_h3"]),
            helper.make_node("And", [any_0_7, any_3_0], ["slice_shape_h4"]),
            helper.make_node("And", [any_5_0, "slice_not_any_0_3"], ["slice_shape_v3"]),
            helper.make_node("And", [any_7_0, any_0_3], ["slice_shape_v4"]),
        ]
    )

    selectors: dict[tuple[int, str, bool], str] = {}
    for n in (3, 4):
        left = _panel_cyan_slice(nodes, inits, f"n{n}_left", 0, 0, n)
        right = _panel_cyan_slice(nodes, inits, f"n{n}_right", 0, n, n)
        top = _panel_cyan_slice(nodes, inits, f"n{n}_top", 0, 0, n)
        bottom = _panel_cyan_slice(nodes, inits, f"n{n}_bottom", n, 0, n)
        h_shape = f"slice_shape_h{n}"
        v_shape = f"slice_shape_v{n}"
        for layout, mask_first, panel_cyan, shape_name in (
            ("h", True, left, h_shape),
            ("h", False, right, h_shape),
            ("v", True, top, v_shape),
            ("v", False, bottom, v_shape),
        ):
            sel = f"slice_sel_n{n}_{layout}_{'first' if mask_first else 'second'}"
            nodes.append(helper.make_node("And", [shape_name, panel_cyan], [sel]))
            selectors[(n, layout, mask_first)] = sel
    return selectors


def _build_slice_kron_size(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    selectors: dict[tuple[int, str, bool], str],
    n: int,
) -> str:
    selected_colors: list[str] = []
    selected_masks: list[str] = []
    size_selectors: list[str] = []
    for layout, mask_first in (("h", True), ("h", False), ("v", True), ("v", False)):
        mask_origin, color_origin, side = _panel_origins(n, layout, mask_first)
        cand_name = f"slice_n{n}_{layout}_mask_{side}"
        selector = selectors[(n, layout, mask_first)]
        size_selectors.append(selector)
        sel_f = f"{cand_name}_sel_f"
        nodes.append(helper.make_node("Cast", [selector], [sel_f], to=TensorProto.FLOAT))
        color = _slice4(
            nodes,
            inits,
            IN_NAME,
            f"{cand_name}_color_f",
            [0, 1, color_origin[0], color_origin[1]],
            [1, 5, color_origin[0] + n, color_origin[1] + n],
        )
        mask = _slice4(
            nodes,
            inits,
            IN_NAME,
            f"{cand_name}_mask_f",
            [0, 8, mask_origin[0], mask_origin[1]],
            [1, 9, mask_origin[0] + n, mask_origin[1] + n],
        )
        nodes.extend(
            [
                helper.make_node("Mul", [color, sel_f], [f"{cand_name}_color_sel"]),
                helper.make_node("Mul", [mask, sel_f], [f"{cand_name}_mask_sel"]),
            ]
        )
        selected_colors.append(f"{cand_name}_color_sel")
        selected_masks.append(f"{cand_name}_mask_sel")

    color_sum = _sum_tensors(nodes, selected_colors, f"slice_kron_n{n}_color")
    mask_sum = _sum_tensors(nodes, selected_masks, f"slice_kron_n{n}_mask")
    size_selector = _sum_bool_or(nodes, size_selectors, f"slice_kron_n{n}_size")
    size_f = f"slice_kron_n{n}_size_f"
    tile_shape = _i64(inits, [1, 4, n * n, n * n], f"slice_kron_n{n}_tile_shape")
    nodes.extend(
        [
            helper.make_node("Cast", [size_selector], [size_f], to=TensorProto.FLOAT),
            helper.make_node("Unsqueeze", [color_sum], [f"slice_kron_n{n}_color_e"], axes=[3, 5]),
            helper.make_node("Unsqueeze", [mask_sum], [f"slice_kron_n{n}_mask_e"], axes=[2, 4]),
            helper.make_node("Mul", [f"slice_kron_n{n}_color_e", f"slice_kron_n{n}_mask_e"], [f"slice_kron_n{n}_paint6d"]),
            helper.make_node("Reshape", [f"slice_kron_n{n}_paint6d", tile_shape], [f"slice_kron_n{n}_paint4"]),
            helper.make_node("ReduceSum", [f"slice_kron_n{n}_paint4"], [f"slice_kron_n{n}_painted_f"], axes=[1], keepdims=1),
            helper.make_node(
                "Greater",
                [f"slice_kron_n{n}_painted_f", _f32(inits, [0.5], f"slice_kron_n{n}_half")],
                [f"slice_kron_n{n}_painted_b"],
            ),
            helper.make_node("Not", [f"slice_kron_n{n}_painted_b"], [f"slice_kron_n{n}_bg_b"]),
            helper.make_node("Cast", [f"slice_kron_n{n}_bg_b"], [f"slice_kron_n{n}_bg_f"], to=TensorProto.FLOAT),
            helper.make_node("Concat", [f"slice_kron_n{n}_bg_f", f"slice_kron_n{n}_paint4"], [f"slice_kron_n{n}_compact"], axis=1),
            helper.make_node("Mul", [f"slice_kron_n{n}_compact", size_f], [f"slice_kron_n{n}_selected"]),
        ]
    )
    return f"slice_kron_n{n}_selected"


def _build_slice_kron_rule_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    selectors = _build_slice_selectors(nodes, inits)
    n3 = _build_slice_kron_size(nodes, inits, selectors, 3)
    n4 = _build_slice_kron_size(nodes, inits, selectors, 4)
    nodes.extend(
        [
            helper.make_node("Pad", [n3], ["slice_kron_n3_to_16"], pads=[0, 0, 0, 0, 0, 0, 7, 7]),
            helper.make_node("Add", ["slice_kron_n3_to_16", n4], ["slice_kron_combined_16"]),
            helper.make_node("Pad", ["slice_kron_combined_16"], [OUT_NAME], pads=[0, 0, 0, 0, 0, C - 5, 14, 14]),
        ]
    )
    return _make_model(nodes, inits, "task275_slice_kron_square_panel_tiling")


def _build_slice_kron_color_size(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    selectors: dict[tuple[int, str, bool], str],
    n: int,
) -> str:
    selected_colors: list[str] = []
    selected_masks: list[str] = []
    for layout, mask_first in (("h", True), ("h", False), ("v", True), ("v", False)):
        mask_origin, color_origin, side = _panel_origins(n, layout, mask_first)
        cand_name = f"slice_color_n{n}_{layout}_mask_{side}"
        selector = selectors[(n, layout, mask_first)]
        sel_f = f"{cand_name}_sel_f"
        nodes.append(helper.make_node("Cast", [selector], [sel_f], to=TensorProto.FLOAT))
        color = _slice4(
            nodes,
            inits,
            IN_NAME,
            f"{cand_name}_color_f",
            [0, 1, color_origin[0], color_origin[1]],
            [1, 5, color_origin[0] + n, color_origin[1] + n],
        )
        mask = _slice4(
            nodes,
            inits,
            IN_NAME,
            f"{cand_name}_mask_f",
            [0, 8, mask_origin[0], mask_origin[1]],
            [1, 9, mask_origin[0] + n, mask_origin[1] + n],
        )
        nodes.extend(
            [
                helper.make_node("Mul", [color, sel_f], [f"{cand_name}_color_sel"]),
                helper.make_node("Mul", [mask, sel_f], [f"{cand_name}_mask_sel"]),
            ]
        )
        selected_colors.append(f"{cand_name}_color_sel")
        selected_masks.append(f"{cand_name}_mask_sel")

    color_sum = _sum_tensors(nodes, selected_colors, f"slice_color_n{n}_color")
    mask_sum = _sum_tensors(nodes, selected_masks, f"slice_color_n{n}_mask")
    tile_shape = _i64(inits, [1, 4, n * n, n * n], f"slice_color_n{n}_tile_shape")
    nodes.extend(
        [
            helper.make_node("Unsqueeze", [color_sum], [f"slice_color_n{n}_color_e"], axes=[3, 5]),
            helper.make_node("Unsqueeze", [mask_sum], [f"slice_color_n{n}_mask_e"], axes=[2, 4]),
            helper.make_node("Mul", [f"slice_color_n{n}_color_e", f"slice_color_n{n}_mask_e"], [f"slice_color_n{n}_paint6d"]),
            helper.make_node("Reshape", [f"slice_color_n{n}_paint6d", tile_shape], [f"slice_color_n{n}_paint4"]),
        ]
    )
    return f"slice_color_n{n}_paint4"


def _build_slice_color_rule_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    selectors = _build_slice_selectors(nodes, inits)
    n3 = _build_slice_kron_color_size(nodes, inits, selectors, 3)
    n4 = _build_slice_kron_color_size(nodes, inits, selectors, 4)
    n3_size = _sum_bool_or(nodes, [selectors[(3, layout, mask_first)] for layout, mask_first in (("h", True), ("h", False), ("v", True), ("v", False))], "slice_color_n3_size")
    n4_size = _sum_bool_or(nodes, [selectors[(4, layout, mask_first)] for layout, mask_first in (("h", True), ("h", False), ("v", True), ("v", False))], "slice_color_n4_size")
    area3 = np.zeros((1, 1, 16, 16), dtype=np.float32)
    area3[:, :, :9, :9] = 1.0
    area3_name = _f32(inits, area3, "slice_color_area3")
    nodes.extend(
        [
            helper.make_node("Pad", [n3], ["slice_color_n3_to_16"], pads=[0, 0, 0, 0, 0, 0, 7, 7]),
            helper.make_node("Add", ["slice_color_n3_to_16", n4], ["slice_color_paint16"]),
            helper.make_node("Cast", [n3_size], ["slice_color_n3_size_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", [n4_size], ["slice_color_n4_size_f"], to=TensorProto.FLOAT),
            helper.make_node("Mul", [area3_name, "slice_color_n3_size_f"], ["slice_color_area3_sel"]),
            helper.make_node("Add", ["slice_color_area3_sel", "slice_color_n4_size_f"], ["slice_color_area"]),
            helper.make_node("ReduceSum", ["slice_color_paint16"], ["slice_color_painted_f"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["slice_color_painted_f", _f32(inits, [0.5], "slice_color_half")], ["slice_color_painted_b"]),
            helper.make_node("Not", ["slice_color_painted_b"], ["slice_color_not_painted_b"]),
            helper.make_node("Cast", ["slice_color_not_painted_b"], ["slice_color_bg_all"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["slice_color_bg_all", "slice_color_area"], ["slice_color_bg"]),
            helper.make_node("Mul", ["slice_color_paint16", "slice_color_area"], ["slice_color_paint_area"]),
            helper.make_node("Concat", ["slice_color_bg", "slice_color_paint_area"], ["slice_color_compact"], axis=1),
            helper.make_node("Pad", ["slice_color_compact"], [OUT_NAME], pads=[0, 0, 0, 0, 0, C - 5, 14, 14]),
        ]
    )
    return _make_model(nodes, inits, "task275_slice_color_square_panel_tiling")


def _build_slice_bool_color_size(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    selectors: dict[tuple[int, str, bool], str],
    n: int,
) -> tuple[str, str]:
    selected_colors: list[str] = []
    selected_masks: list[str] = []
    size_selectors: list[str] = []
    for layout, mask_first in (("h", True), ("h", False), ("v", True), ("v", False)):
        mask_origin, color_origin, side = _panel_origins(n, layout, mask_first)
        cand_name = f"bool_color_n{n}_{layout}_mask_{side}"
        selector = selectors[(n, layout, mask_first)]
        size_selectors.append(selector)
        color = _slice4(
            nodes,
            inits,
            IN_NAME,
            f"{cand_name}_color_f",
            [0, 1, color_origin[0], color_origin[1]],
            [1, 5, color_origin[0] + n, color_origin[1] + n],
        )
        mask = _slice4(
            nodes,
            inits,
            IN_NAME,
            f"{cand_name}_mask_f",
            [0, 8, mask_origin[0], mask_origin[1]],
            [1, 9, mask_origin[0] + n, mask_origin[1] + n],
        )
        nodes.extend(
            [
                helper.make_node("Cast", [color], [f"{cand_name}_color_b"], to=TensorProto.BOOL),
                helper.make_node("Cast", [mask], [f"{cand_name}_mask_b"], to=TensorProto.BOOL),
                helper.make_node("And", [f"{cand_name}_color_b", selector], [f"{cand_name}_color_sel"]),
                helper.make_node("And", [f"{cand_name}_mask_b", selector], [f"{cand_name}_mask_sel"]),
            ]
        )
        selected_colors.append(f"{cand_name}_color_sel")
        selected_masks.append(f"{cand_name}_mask_sel")

    color_sum = _sum_bool_or(nodes, selected_colors, f"bool_color_n{n}_color")
    mask_sum = _sum_bool_or(nodes, selected_masks, f"bool_color_n{n}_mask")
    tile_shape = _i64(inits, [1, 4, n * n, n * n], f"bool_color_n{n}_tile_shape")
    nodes.extend(
        [
            helper.make_node("Unsqueeze", [color_sum], [f"bool_color_n{n}_color_e"], axes=[3, 5]),
            helper.make_node("Unsqueeze", [mask_sum], [f"bool_color_n{n}_mask_e"], axes=[2, 4]),
            helper.make_node("And", [f"bool_color_n{n}_color_e", f"bool_color_n{n}_mask_e"], [f"bool_color_n{n}_paint6d"]),
            helper.make_node("Reshape", [f"bool_color_n{n}_paint6d", tile_shape], [f"bool_color_n{n}_paint4"]),
        ]
    )
    return f"bool_color_n{n}_paint4", _sum_bool_or(nodes, size_selectors, f"bool_color_n{n}_size")


def _build_slice_bool_color_rule_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    selectors = _build_slice_selectors(nodes, inits)
    n3, n3_size = _build_slice_bool_color_size(nodes, inits, selectors, 3)
    n4, n4_size = _build_slice_bool_color_size(nodes, inits, selectors, 4)
    n3_channels = [f"bool_color_n3_c{i}" for i in range(1, 5)]
    n4_channels = [f"bool_color_n4_c{i}" for i in range(1, 5)]
    nodes.extend(
        [
            helper.make_node("Split", [n3], n3_channels, axis=1, split=[1, 1, 1, 1]),
            helper.make_node("Or", [n3_channels[0], n3_channels[1]], ["bool_color_n3_or12"]),
            helper.make_node("Or", [n3_channels[2], n3_channels[3]], ["bool_color_n3_or34"]),
            helper.make_node("Or", ["bool_color_n3_or12", "bool_color_n3_or34"], ["bool_color_n3_painted"]),
            helper.make_node("Not", ["bool_color_n3_painted"], ["bool_color_n3_not_painted"]),
            helper.make_node("And", ["bool_color_n3_not_painted", n3_size], ["bool_color_n3_bg"]),
            helper.make_node("Concat", ["bool_color_n3_bg", n3], ["bool_color_n3_compact"], axis=1),
            helper.make_node("Cast", ["bool_color_n3_compact"], ["bool_color_n3_compact_f"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["bool_color_n3_compact_f"], ["bool_color_n3_to_16"], pads=[0, 0, 0, 0, 0, 0, 7, 7]),
            helper.make_node("Split", [n4], n4_channels, axis=1, split=[1, 1, 1, 1]),
            helper.make_node("Or", [n4_channels[0], n4_channels[1]], ["bool_color_n4_or12"]),
            helper.make_node("Or", [n4_channels[2], n4_channels[3]], ["bool_color_n4_or34"]),
            helper.make_node("Or", ["bool_color_n4_or12", "bool_color_n4_or34"], ["bool_color_n4_painted"]),
            helper.make_node("Not", ["bool_color_n4_painted"], ["bool_color_n4_not_painted"]),
            helper.make_node("And", ["bool_color_n4_not_painted", n4_size], ["bool_color_n4_bg"]),
            helper.make_node("Concat", ["bool_color_n4_bg", n4], ["bool_color_n4_compact"], axis=1),
            helper.make_node("Cast", ["bool_color_n4_compact"], ["bool_color_n4_compact_f"], to=TensorProto.FLOAT),
            helper.make_node("Add", ["bool_color_n3_to_16", "bool_color_n4_compact_f"], ["bool_color_compact_f"]),
            helper.make_node("Pad", ["bool_color_compact_f"], [OUT_NAME], pads=[0, 0, 0, 0, 0, C - 5, 14, 14]),
        ]
    )
    return _make_model(nodes, inits, "task275_slice_bool_color_square_panel_tiling")


def _score_candidate(label: str, model: onnx.ModelProto, examples: Sequence[tuple[str, int, np.ndarray, np.ndarray]]) -> tuple[int, onnx.ModelProto]:
    validate_json(model, examples)
    with tempfile.NamedTemporaryFile(suffix=f"_{TASK_ID}.onnx", dir=OUT_DIR, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        onnx.save(model, tmp_path)
        result = score_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    if not result["valid"]:
        raise AssertionError(f"{label} invalid: {result['error']}")
    print(
        f"{label}: nodes={len(model.graph.node)} memory={result['memory']} "
        f"params={result['params']} cost={result['cost']} score={result['score']:.6f}"
    )
    assert result["cost"] is not None
    return int(result["cost"]), model


def main() -> None:
    examples = load_examples()
    print_diagnostics(examples)
    candidates = [
        _score_candidate("rule-candidates", _build_rule_model(), examples),
        _score_candidate("merged-rule", _build_merged_rule_model(), examples),
        _score_candidate("broadcast-rule", _build_kron_rule_model(), examples),
        _score_candidate("slice-broadcast-rule", _build_slice_kron_rule_model(), examples),
        _score_candidate("slice-color-rule", _build_slice_color_rule_model(), examples),
        _score_candidate("slice-bool-color-rule", _build_slice_bool_color_rule_model(), examples),
    ]
    _cost, best = min(candidates, key=lambda item: item[0])
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
