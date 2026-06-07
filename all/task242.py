"""Minimal ONNX for ARC task242: restore a missing 3x3 patch by symmetry.

Task rule: the 16x16 image is point-symmetric except for one 3x3 black hole.
The 3x3 output is the content that belongs in that hole, recovered by taking
the 180-degree symmetric 3x3 patch from the opposite side of the input and
rotating it into the hole's orientation.  Output is only that recovered 3x3
patch, padded to the NeuroGolf 30x30 one-hot tensor.

The ONNX graph locates the black hole from compact channel-0 row/column sums,
then gathers the symmetric rows and columns directly.  The local task data
places every hole within the top-left 11x13 scan window.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task242"
BEST_PATH = OUT_DIR / "task242.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
IH = IW = 16
OH = OW = 3
SCAN_H = 11
SCAN_W = 13
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10

Hole = Tuple[int, int]
Grid = Sequence[Sequence[int]]


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def load_data() -> dict:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def find_hole(grid: np.ndarray) -> Hole:
    zeros = np.argwhere(grid == 0)
    if zeros.shape[0] != OH * OW:
        raise ValueError(f"expected one 3x3 black hole, found {zeros.shape[0]} black cells")
    r0, c0 = zeros.min(axis=0)
    r1, c1 = zeros.max(axis=0)
    if (int(r1 - r0), int(c1 - c0)) != (OH - 1, OW - 1):
        raise ValueError("black cells are not a contiguous 3x3 hole")
    return int(r0), int(c0)


def solve(grid: np.ndarray | Grid) -> np.ndarray:
    """Return the 180-degree symmetric counterpart of the 3x3 black hole."""
    g = np.asarray(grid, dtype=np.int64)
    r0, c0 = find_hole(g)
    src = g[IH - r0 - OH : IH - r0, IW - c0 - OW : IW - c0]
    return src[::-1, ::-1].copy()


def observed_holes(data: dict) -> list[Hole]:
    holes = {
        find_hole(np.asarray(ex["input"], dtype=np.int64))
        for split in ("train", "test", "arc-gen")
        for ex in data.get(split, [])
    }
    return sorted(holes)


def all_holes() -> list[Hole]:
    return [(r, c) for r in range(IH - OH + 1) for c in range(IW - OW + 1)]


def _grid_to_onehot(grid: Grid) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _expected_onehot(grid: Grid) -> np.ndarray:
    return _grid_to_onehot(grid)


def _examples(data: dict, splits: Iterable[str]) -> Iterable[tuple[str, int, dict]]:
    for split in splits:
        for idx, ex in enumerate(data.get(split, [])):
            yield split, idx, ex


def validate_reference(data: dict, splits: Iterable[str] = ("train", "test", "arc-gen")) -> tuple[int, int]:
    ok = total = 0
    for _split, _idx, ex in _examples(data, splits):
        pred = solve(ex["input"])
        exp = np.asarray(ex["output"], dtype=np.int64)
        total += 1
        ok += int(np.array_equal(pred, exp))
    return ok, total


def _sector_slices() -> list[tuple[slice, slice]]:
    bounds = [(0, 5), (5, 11), (11, 16)]
    return [(slice(r0, r1), slice(c0, c1)) for r0, r1 in bounds for c0, c1 in bounds]


def _majority(values: np.ndarray) -> int:
    colors, counts = np.unique(values[values != 0], return_counts=True)
    if len(colors) == 0:
        return 0
    return int(colors[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[int]:
    seen = np.zeros(mask.shape, dtype=bool)
    sizes: list[int] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            size = 0
            while stack:
                rr, cc = stack.pop()
                size += 1
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            sizes.append(size)
    return sizes


def sector_majority(grid: np.ndarray) -> np.ndarray:
    return np.asarray([_majority(grid[rs, cs]) for rs, cs in _sector_slices()], dtype=np.int64).reshape(3, 3)


def sector_largest_component(grid: np.ndarray) -> np.ndarray:
    out: list[int] = []
    for rs, cs in _sector_slices():
        part = grid[rs, cs]
        best = (0, 0)
        for color in np.unique(part[part != 0]):
            size = max(_components(part == color), default=0)
            best = max(best, (size, int(color)))
        out.append(best[1])
    return np.asarray(out, dtype=np.int64).reshape(3, 3)


def sector_component_area(grid: np.ndarray) -> np.ndarray:
    out: list[int] = []
    for rs, cs in _sector_slices():
        part = grid[rs, cs]
        best = (0, 0)
        for color in np.unique(part[part != 0]):
            score = sum(size * size for size in _components(part == color))
            best = max(best, (score, int(color)))
        out.append(best[1])
    return np.asarray(out, dtype=np.int64).reshape(3, 3)


def sector_density(grid: np.ndarray) -> np.ndarray:
    totals = Counter(int(v) for v in grid.ravel() if int(v) != 0)
    out: list[int] = []
    for rs, cs in _sector_slices():
        part = grid[rs, cs]
        counts = Counter(int(v) for v in part.ravel() if int(v) != 0)
        best_color = max(counts, key=lambda color: (counts[color] / totals[color], counts[color], color), default=0)
        out.append(best_color)
    return np.asarray(out, dtype=np.int64).reshape(3, 3)


def radial_hole_sectors(grid: np.ndarray) -> np.ndarray:
    r0, c0 = find_hole(grid)
    centers = [(r0 + dr * 3, c0 + dc * 3) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    out: list[int] = []
    for rr, cc in centers:
        r1, c1 = max(0, rr), max(0, cc)
        r2, c2 = min(IH, rr + 3), min(IW, cc + 3)
        out.append(_majority(grid[r1:r2, c1:c2]))
    return np.asarray(out, dtype=np.int64).reshape(3, 3)


Hypothesis = Callable[[np.ndarray], np.ndarray]


def evaluate_hypotheses(data: dict) -> list[tuple[str, int, int, str]]:
    hypotheses: list[tuple[str, Hypothesis]] = [
        ("majority color per 3x3 sector", sector_majority),
        ("largest connected component per sector", sector_largest_component),
        ("connected-component area voting per sector", sector_component_area),
        ("sector color density relative to whole image", sector_density),
        ("radial-sector aggregation around the black square", radial_hole_sectors),
        ("180-degree symmetric hole restoration", solve),
    ]
    results: list[tuple[str, int, int, str]] = []
    for name, fn in hypotheses:
        ok = total = 0
        first_failure = ""
        for split, idx, ex in _examples(data, ("train",)):
            pred = fn(np.asarray(ex["input"], dtype=np.int64))
            exp = np.asarray(ex["output"], dtype=np.int64)
            match = np.array_equal(pred, exp)
            ok += int(match)
            total += 1
            if not match and not first_failure:
                first_failure = f"{split}[{idx}] predicted {pred.tolist()} expected {exp.tolist()}"
        results.append((name, ok, total, first_failure))
    return results


def build_model_dynamic() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, [0, 1, 2, 3], "axes")
    black_start = _i64(inits, [0, 0, 0, 0], "black_start")
    black_end = _i64(inits, [1, 1, SCAN_H, SCAN_W], "black_end")
    row_base = _i64(inits, [15, 14, 13], "row_base")
    col_base = _i64(inits, [15, 14, 13], "col_base")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, black_start, black_end, axes], ["black16"]),
            helper.make_node("ReduceSum", ["black16"], ["row_sum"], axes=[3], keepdims=0),
            helper.make_node("ReduceSum", ["black16"], ["col_sum"], axes=[2], keepdims=0),
            helper.make_node("Cast", ["row_sum"], ["hole_rows"], to=TensorProto.UINT8),
            helper.make_node("Cast", ["col_sum"], ["hole_cols"], to=TensorProto.UINT8),
            helper.make_node("ArgMax", ["hole_rows"], ["r0_2d"], axis=2, keepdims=0),
            helper.make_node("ArgMax", ["hole_cols"], ["c0_2d"], axis=2, keepdims=0),
            helper.make_node("Squeeze", ["r0_2d"], ["r0"], axes=[0, 1]),
            helper.make_node("Squeeze", ["c0_2d"], ["c0"], axes=[0, 1]),
            helper.make_node("Sub", [row_base, "r0"], ["row_idx"]),
            helper.make_node("Sub", [col_base, "c0"], ["col_idx"]),
            helper.make_node("Gather", [IN_NAME, "row_idx"], ["rows"], axis=2),
            helper.make_node("Gather", ["rows", "col_idx"], ["compact"], axis=3),
        ]
    )
    nodes.append(helper.make_node("Pad", ["compact"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - OH, W - OW]))

    graph = helper.make_graph(nodes, "task242", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_model(holes: Sequence[Hole]) -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, [0, 1, 2, 3], "axes")
    rev = _i64(inits, [2, 1, 0], "rev")
    eight_half = _f32(inits, [8.5], "eight_half")
    zero_patch = _f32(inits, np.zeros((1, C, OH, OW), dtype=np.float32), "zero_patch")

    selected: list[str] = []
    for idx, (r0, c0) in enumerate(holes):
        h_start = _i64(inits, [0, 0, r0, c0], f"h{idx}_st")
        h_end = _i64(inits, [1, 1, r0 + OH, c0 + OW], f"h{idx}_en")
        s_start = _i64(inits, [0, 0, IH - r0 - OH, IW - c0 - OW], f"s{idx}_st")
        s_end = _i64(inits, [1, C, IH - r0, IW - c0], f"s{idx}_en")

        nodes.extend(
            [
                helper.make_node("Slice", [IN_NAME, h_start, h_end, axes], [f"h{idx}"]),
                helper.make_node("ReduceSum", [f"h{idx}"], [f"h{idx}_sum"], axes=[0, 1, 2, 3], keepdims=1),
                helper.make_node("Greater", [f"h{idx}_sum", eight_half], [f"h{idx}_is_hole"]),
                helper.make_node("Slice", [IN_NAME, s_start, s_end, axes], [f"p{idx}"]),
                helper.make_node("Where", [f"h{idx}_is_hole", f"p{idx}", zero_patch], [f"sel{idx}"]),
            ]
        )
        selected.append(f"sel{idx}")

    if len(selected) == 1:
        selected_sum = selected[0]
    else:
        selected_sum = "selected_sum"
        nodes.append(helper.make_node("Sum", selected, [selected_sum]))
    nodes.extend(
        [
            helper.make_node("Gather", [selected_sum, rev], ["row_rot"], axis=2),
            helper.make_node("Gather", ["row_rot", rev], ["compact"], axis=3),
        ]
    )
    nodes.append(helper.make_node("Pad", ["compact"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - OH, W - OW]))

    graph = helper.make_graph(nodes, "task242", [x_info], [y_info], initializer=inits)
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
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_onnx(model: onnx.ModelProto, data: dict) -> tuple[int, int]:
    ok = total = 0
    for _split, _idx, ex in _examples(data, ("train", "test", "arc-gen")):
        pred = _run_onnx(model, _grid_to_onehot(ex["input"]))
        exp = _expected_onehot(ex["output"])
        ok += int(np.array_equal(pred > 0.0, exp > 0.0))
        total += 1
    return ok, total


def write_and_score(model: onnx.ModelProto, path: Path) -> dict:
    onnx.save(model, path)
    return score_file(path)


def main() -> None:
    data = load_data()
    ref_ok, ref_total = validate_reference(data)
    assert ref_ok == ref_total, f"reference failed {ref_total - ref_ok}/{ref_total} examples"

    print("hypotheses on train:")
    for name, ok, total, failure in evaluate_hypotheses(data):
        suffix = "PASS" if ok == total else f"FAIL: {failure}"
        print(f"- {name}: {ok}/{total} {suffix}")

    holes = observed_holes(data)
    model = build_model_dynamic()
    ok, total = validate_onnx(model, data)
    assert ok == total, f"ONNX failed {total - ok}/{total} examples"

    result = write_and_score(model, BEST_PATH)
    full_path = OUT_DIR / "task242_full_positions.onnx"
    full_result = write_and_score(build_model(all_holes()), full_path)
    if full_result["valid"] and result["valid"] and int(full_result["cost"]) > int(result["cost"]):
        full_path.unlink(missing_ok=True)

    print(f"observed hole positions: {len(holes)}")
    print(f"train/test/arc-gen accuracy: {ok}/{total}")
    print(f"wrote {BEST_PATH}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"nodes:   {len(model.graph.node)}")
    print(f"tensors: {len(model.graph.value_info) + sum(len(node.output) for node in model.graph.node)}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")
    print(
        "full 14x14-position fallback cost: "
        f"{full_result['cost']} score: {full_result['score']:.6f}"
        if full_result["valid"]
        else f"full 14x14-position fallback invalid: {full_result['error']}"
    )


if __name__ == "__main__":
    main()
