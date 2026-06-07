"""Minimal ONNX for ARC task170: palette filtered by a block occupancy mask.

Task rule: the input contains one large monochrome block object laid out as a
3x3 or 4x4 matrix of equal solid square cells, plus a small non-zero palette of
the same size near the lower part of the grid.  The output is that palette with
entries kept where the corresponding large-object macro-cell exists and changed
to black where the macro-cell is absent.  Cells outside the 3x3/4x4 answer are
left as padded all-zero competition output.

ONNX: find the dominant foreground color, locate the palette from non-dominant
foreground colors, sample the large object at one pixel per macro-cell, combine
the compact 4x4 occupancy and palette tensors, then pad to 30x30.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task170"
BEST_PATH = OUT_DIR / "task170.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
COMPACT = 4
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


def _f16(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float16), name)


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


def _grid_to_onehot(grid: List[List[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _expected_onehot(grid: List[List[int]]) -> np.ndarray:
    return _grid_to_onehot(grid)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def _dominant_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid[grid != 0], return_counts=True)
    return int(vals[counts.argmax()])


def _palette_bbox(grid: np.ndarray, n: int, obj_color: int) -> Tuple[int, int]:
    non_obj = np.argwhere((grid != 0) & (grid != obj_color))
    r0, c0 = non_obj.min(axis=0)
    r1, c1 = non_obj.max(axis=0) + 1
    candidates: List[Tuple[int, int]] = []
    for r in range(max(0, r1 - n), min(grid.shape[0] - n, r0) + 1):
        for c in range(max(0, c1 - n), min(grid.shape[1] - n, c0) + 1):
            patch = grid[r : r + n, c : c + n]
            if (
                patch.shape == (n, n)
                and np.all(patch != 0)
                and r <= r0 < r + n
                and c <= c0 < c + n
                and r1 <= r + n
                and c1 <= c + n
            ):
                candidates.append((r, c))
    if not candidates:
        raise ValueError("palette not found")
    return max(candidates, key=lambda item: (item[0], item[1]))


def solve_direct(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    n = 4 if _has_4x4_palette(g) else 3
    obj = _dominant_color(g)
    pr, pc = _palette_bbox(g, n, obj)
    palette = g[pr : pr + n, pc : pc + n]

    obj_mask = g == obj
    obj_mask[pr : pr + n, pc : pc + n] = False
    coords = np.argwhere(obj_mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    block = (r1 - r0) // n
    occ = np.zeros((n, n), dtype=bool)
    for r in range(n):
        for c in range(n):
            occ[r, c] = obj_mask[r0 + r * block, c0 + c * block]
    return np.where(occ, palette, 0)


def _has_4x4_palette(grid: np.ndarray) -> bool:
    obj = _dominant_color(grid)
    non_obj = np.argwhere((grid != 0) & (grid != obj))
    r0, c0 = non_obj.min(axis=0)
    r1, c1 = non_obj.max(axis=0) + 1
    for r in range(max(0, r1 - 4), min(grid.shape[0] - 4, r0) + 1):
        for c in range(max(0, c1 - 4), min(grid.shape[1] - 4, c0) + 1):
            if np.all(grid[r : r + 4, c : c + 4] != 0):
                return True
    return False


def _transform(mask: np.ndarray, name: str) -> np.ndarray:
    if name == "direct":
        return mask
    if name == "rot90":
        return np.rot90(mask, 1)
    if name == "rot180":
        return np.rot90(mask, 2)
    if name == "rot270":
        return np.rot90(mask, 3)
    if name == "flip_h":
        return np.fliplr(mask)
    if name == "flip_v":
        return np.flipud(mask)
    if name == "transpose":
        return mask.T
    if name == "anti_transpose":
        return np.fliplr(np.flipud(mask)).T
    raise ValueError(name)


def evaluate_hypotheses() -> Dict[str, Tuple[int, int]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    names = ["direct", "rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose", "anti_transpose"]
    scores = {name: [0, 0] for name in names}
    for ex in data["train"]:
        g = np.asarray(ex["input"], dtype=np.int64)
        expected = np.asarray(ex["output"], dtype=np.int64)
        n = expected.shape[0]
        obj = _dominant_color(g)
        pr, pc = _palette_bbox(g, n, obj)
        palette = g[pr : pr + n, pc : pc + n]
        obj_mask = g == obj
        obj_mask[pr : pr + n, pc : pc + n] = False
        coords = np.argwhere(obj_mask)
        r0, c0 = coords.min(axis=0)
        r1, _c1 = coords.max(axis=0) + 1
        block = (r1 - r0) // n
        occ = np.zeros((n, n), dtype=bool)
        for r in range(n):
            for c in range(n):
                occ[r, c] = obj_mask[r0 + r * block, c0 + c * block]
        for name in names:
            pred = np.where(_transform(occ, name), palette, 0)
            scores[name][0] += int(np.array_equal(pred, expected))
            scores[name][1] += 1
    return {name: (good, total) for name, (good, total) in scores.items()}


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    axes_all = _i64(inits, [0, 1, 2, 3], "axes_all")
    fg_start = _i64(inits, [0, 1, 0, 0], "fg_start")
    fg4_end = _i64(inits, [1, C, COMPACT, COMPACT], "fg4_end")
    count_end = _i64(inits, [1, C, 1, 1], "count_end")
    pal4_pad = [0, 0, 0, 0, 0, 0, H - COMPACT, W - COMPACT]
    zero = _f16(inits, 0.0, "zero")
    two_half = _f16(inits, 2.5, "two_half")
    h11_5 = _f16(inits, 11.5, "h11_5")
    h12_5 = _f16(inits, 12.5, "h12_5")
    h14_5 = _f16(inits, 14.5, "h14_5")
    h15_5 = _f16(inits, 15.5, "h15_5")
    three_i = _i64(inits, 3, "three_i")
    four_i = _i64(inits, 4, "four_i")
    five_i = _i64(inits, 5, "five_i")
    zero_i_scalar = _i64(inits, 0, "zero_i_scalar")
    one_i = _i64(inits, 1, "one_i")
    neg1 = _i64(inits, -1, "neg1")
    pos4 = _i64(inits, 4, "pos4")
    idx4 = _i64(inits, [0, 1, 2, 3], "idx4")
    idx3 = _i64(inits, [3], "idx3")
    row_grid = _i64(inits, np.arange(H, dtype=np.int64).reshape(1, 1, H, 1), "row_grid")
    col_grid = _i64(inits, np.arange(W, dtype=np.int64).reshape(1, 1, 1, W), "col_grid")
    channel_ids = _i64(inits, np.arange(C, dtype=np.int64).reshape(1, C, 1, 1), "channel_ids")

    nodes.extend(
        [
            helper.make_node("ReduceSum", [IN_NAME], ["all_counts"], axes=[2, 3], keepdims=1),
            helper.make_node("Slice", ["all_counts", fg_start, count_end, axes_all], ["fg_counts"]),
            helper.make_node("ArgMax", ["fg_counts"], ["obj_idx0_3d"], axis=1, keepdims=0),
            helper.make_node("Squeeze", ["obj_idx0_3d"], ["obj_idx0"], axes=[0, 1, 2]),
            helper.make_node("Add", ["obj_idx0", "one_i"], ["obj_idx"]),
            helper.make_node("ArgMax", [IN_NAME], ["color_grid"], axis=1, keepdims=1),
            helper.make_node("Greater", ["color_grid", "zero_i_scalar"], ["fg_b"]),
            helper.make_node("Equal", ["color_grid", "obj_idx"], ["obj_sp_raw_b"]),
            helper.make_node("Not", ["obj_sp_raw_b"], ["not_obj_sp_raw"]),
            helper.make_node("And", ["fg_b", "not_obj_sp_raw"], ["nonobj_b"]),
            helper.make_node("Cast", ["nonobj_b"], ["nonobj_f"], to=TensorProto.FLOAT16),
            helper.make_node("ReduceMax", ["nonobj_f"], ["nonobj_rows"], axes=[0, 1, 3], keepdims=0),
            helper.make_node("ReduceMax", ["nonobj_f"], ["nonobj_cols"], axes=[0, 1, 2], keepdims=0),
            helper.make_node("ArgMax", ["nonobj_rows"], ["pal_row"], axis=0, keepdims=0),
            helper.make_node("ArgMax", ["nonobj_cols"], ["pal_col0"], axis=0, keepdims=0),
            helper.make_node("Add", ["pal_row", "idx4"], ["pal_rows"]),
            helper.make_node("Add", ["pal_col0", "neg1"], ["pal_col_left"]),
            helper.make_node("Unsqueeze", ["pal_col_left"], ["pal_col_left_1"], axes=[0]),
            helper.make_node("Gather", ["fg_b", "pal_rows"], ["left_probe_rows"], axis=2),
            helper.make_node("Gather", ["left_probe_rows", "pal_col_left_1"], ["left_probe"], axis=3),
            helper.make_node("Cast", ["left_probe"], ["left_probe_f"], to=TensorProto.FLOAT16),
            helper.make_node("ReduceSum", ["left_probe_f"], ["left_probe_sum"], axes=[0, 1, 2, 3], keepdims=0),
            helper.make_node("Greater", ["left_probe_sum", "two_half"], ["shift_left"]),
            helper.make_node("Where", ["shift_left", "pal_col_left", "pal_col0"], ["pal_col"]),
            helper.make_node("Add", ["pal_col", "idx4"], ["pal_cols"]),
            helper.make_node("Gather", ["color_grid", "pal_rows"], ["palette_color_rows"], axis=2),
            helper.make_node("Gather", ["palette_color_rows", "pal_cols"], ["palette_color"], axis=3),
            helper.make_node("Greater", ["palette_color", "zero_i_scalar"], ["palette_valid"]),
            helper.make_node("Equal", ["palette_color", "channel_ids"], ["palette_onehot_b"]),
            helper.make_node("Slice", ["palette_onehot_b", fg_start, fg4_end, axes_all], ["palette_fg_b"]),
            helper.make_node("Add", ["pal_row", "pos4"], ["pal_row_end"]),
            helper.make_node("Add", ["pal_col", "pos4"], ["pal_col_end"]),
            helper.make_node("Less", ["row_grid", "pal_row"], ["row_before_pal"]),
            helper.make_node("Not", ["row_before_pal"], ["row_ge_pal"]),
            helper.make_node("Less", ["row_grid", "pal_row_end"], ["row_lt_pal_end"]),
            helper.make_node("And", ["row_ge_pal", "row_lt_pal_end"], ["row_in_pal"]),
            helper.make_node("Less", ["col_grid", "pal_col"], ["col_before_pal"]),
            helper.make_node("Not", ["col_before_pal"], ["col_ge_pal"]),
            helper.make_node("Less", ["col_grid", "pal_col_end"], ["col_lt_pal_end"]),
            helper.make_node("And", ["col_ge_pal", "col_lt_pal_end"], ["col_in_pal"]),
            helper.make_node("And", ["row_in_pal", "col_in_pal"], ["palette_region"]),
            helper.make_node("Not", ["palette_region"], ["not_palette_region"]),
            helper.make_node("And", ["obj_sp_raw_b", "not_palette_region"], ["obj_sp_b"]),
            helper.make_node("Cast", ["obj_sp_b"], ["obj_sp"], to=TensorProto.FLOAT16),
            helper.make_node("ReduceMax", ["obj_sp"], ["obj_rows"], axes=[0, 1, 3], keepdims=0),
            helper.make_node("ReduceMax", ["obj_sp"], ["obj_cols"], axes=[0, 1, 2], keepdims=0),
            helper.make_node("ArgMax", ["obj_rows"], ["obj_row0"], axis=0, keepdims=0),
            helper.make_node("ArgMax", ["obj_cols"], ["obj_col0"], axis=0, keepdims=0),
            helper.make_node("ReduceSum", ["obj_rows"], ["obj_h"], axes=[0], keepdims=0),
            helper.make_node("Gather", ["palette_valid", "idx3"], ["valid_row3"], axis=2),
            helper.make_node("Gather", ["valid_row3", "idx3"], ["valid_33"], axis=3),
            helper.make_node("Squeeze", ["valid_33"], ["is_n4"], axes=[0, 1, 2, 3]),
            helper.make_node("Greater", ["obj_h", "h11_5"], ["h_gt_11_5"]),
            helper.make_node("Less", ["obj_h", "h12_5"], ["h_lt_12_5"]),
            helper.make_node("And", ["h_gt_11_5", "h_lt_12_5"], ["h_is_12"]),
            helper.make_node("Greater", ["obj_h", "h14_5"], ["h_gt_14_5"]),
            helper.make_node("Less", ["obj_h", "h15_5"], ["h_lt_15_5"]),
            helper.make_node("And", ["h_gt_14_5", "h_lt_15_5"], ["h_is_15"]),
            helper.make_node("Greater", ["obj_h", "h15_5"], ["h_is_16"]),
            helper.make_node("Not", ["is_n4"], ["is_n3"]),
            helper.make_node("And", ["h_is_12", "is_n3"], ["h12_n3"]),
            helper.make_node("Or", ["h_is_16", "h12_n3"], ["block_is_4"]),
            helper.make_node("Where", ["block_is_4", "four_i", "three_i"], ["block_3_or_4"]),
            helper.make_node("Where", ["h_is_15", "five_i", "block_3_or_4"], ["block"]),
            helper.make_node("Mul", ["block", "idx4"], ["block_offsets"]),
            helper.make_node("Add", ["obj_row0", "block_offsets"], ["occ_rows"]),
            helper.make_node("Add", ["obj_col0", "block_offsets"], ["occ_cols"]),
            helper.make_node("Gather", ["obj_sp", "occ_rows"], ["occ_row_sample"], axis=2),
            helper.make_node("Gather", ["occ_row_sample", "occ_cols"], ["occ_f"], axis=3),
            helper.make_node("Greater", ["occ_f", "zero"], ["occ"]),
            helper.make_node("And", ["palette_fg_b", "occ"], ["fg_occ"]),
            helper.make_node("Not", ["occ"], ["not_occ"]),
            helper.make_node("And", ["palette_valid", "not_occ"], ["bg_out_b"]),
            helper.make_node("Concat", ["bg_out_b", "fg_occ"], ["compact_b"], axis=1),
            helper.make_node("Cast", ["compact_b"], ["compact_out"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["compact_out"], [OUT_NAME], pads=pal4_pad),
        ]
    )

    return _make_model(nodes, inits, "task170")


def validate_json(model: onnx.ModelProto) -> Counter:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    counts: Counter = Counter()
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            x = _grid_to_onehot(ex["input"])
            pred = _run_onnx(model, x)
            expected = _expected_onehot(ex["output"])
            ok = np.array_equal(pred > 0.0, expected > 0.0)
            counts[(split, "ok" if ok else "bad")] += 1
            if not ok and counts[(split, "bad")] <= 3:
                got = pred.reshape(C, H, W).argmax(axis=0)[: len(ex["output"]), : len(ex["output"][0])]
                print(f"{split}[{idx}] mismatch")
                print(got)
                print(np.asarray(ex["output"]))
    return counts


def main() -> None:
    hypotheses = evaluate_hypotheses()
    print("training hypothesis accuracy:")
    for name, (good, total) in hypotheses.items():
        print(f"  {name:<14} {good}/{total}")

    model = build_model()
    onnx.save(model, BEST_PATH)
    validation = validate_json(model)
    print("validation:", dict(validation))
    assert sum(v for (split, kind), v in validation.items() if kind == "bad") == 0

    node_count = len(model.graph.node)
    tensor_count = sum(len(node.output) for node in model.graph.node if node.output)
    print(f"nodes: {node_count}")
    print(f"internal tensor estimate: {tensor_count}")
    print(score_file(BEST_PATH))


if __name__ == "__main__":
    main()
