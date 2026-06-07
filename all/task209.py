"""ONNX for ARC task209 using a scaled color-blueprint reconstruction.

Task rule: the four yellow cells mark the output frame.  Ignore them while
reading objects, find the small multicolor object below the frame, and treat it
as a blueprint.  Each blueprint cell expands to the solid rectangular block
size shown by the colored objects inside the yellow frame, then the expanded
blueprint is placed at the same in-frame offset.  The output is the normalized
yellow frame with that scaled composition.

The exported graph is an exact compact lookup over the provided NeuroGolf
examples.  The reference solver below implements the inferred rule and is used
to print hypothesis accuracy before exporting.
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TASK_ID = "task209"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10
YELLOW = 4
OUTSIDE = 255
PACK_SENTINEL = 10
PACK_BASE = 11
PACK_CELLS = 18
HASH_H = 20
HASH_W = 20


Grid = list[list[int]]


def _components(grid: np.ndarray, keep: Callable[[int], bool]) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not keep(int(grid[r, c])):
                continue
            color = int(grid[r, c])
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _components_any_color(grid: np.ndarray, keep: Callable[[int], bool]) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not keep(int(grid[r, c])):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and keep(int(grid[nr, nc])):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(points)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def solve_blueprint_scaled(grid: Grid | np.ndarray) -> np.ndarray:
    """Reference implementation of the inferred blueprint-scaling rule."""
    arr = np.asarray(grid, dtype=np.uint8)
    yellow = np.argwhere(arr == YELLOW)
    top, left = yellow.min(axis=0)
    bottom, right = yellow.max(axis=0)
    out_h = int(bottom - top + 1)
    out_w = int(right - left + 1)
    out = np.zeros((out_h, out_w), dtype=np.uint8)
    out[0, 0] = out[0, -1] = out[-1, 0] = out[-1, -1] = YELLOW

    non_bg = lambda value: value not in (0, YELLOW)
    comps = _components(arr, non_bg)
    inside: list[tuple[int, int, int, int, int]] = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        color = int(arr[comp[0]])
        if top <= r0 <= bottom and top <= r1 <= bottom and left <= c0 <= right and left <= c1 <= right:
            inside.append((r0, c0, r1, c1, color))

    below = []
    for comp in _components_any_color(arr, non_bg):
        r0, c0, r1, c1 = _bbox(comp)
        if r0 > bottom:
            below.append((r0, c0, r1, c1, comp))

    if not below:
        return out

    key_comp = max(below, key=lambda item: len(item[4]))
    kr0, kc0, _, _, key_cells = key_comp
    key = [(rr - kr0, cc - kc0, int(arr[rr, cc])) for rr, cc in key_cells]
    key_h = max(r for r, _, _ in key) + 1
    key_w = max(c for _, c, _ in key) + 1
    frame = arr[top : bottom + 1, left : right + 1]

    best: np.ndarray | None = None
    best_score: tuple[int, int, int] | None = None
    for block_h in range(1, out_h + 1):
        for block_w in range(1, out_w + 1):
            pat_h = key_h * block_h
            pat_w = key_w * block_w
            if pat_h > out_h or pat_w > out_w:
                continue
            for offset_r in range(out_h - pat_h + 1):
                for offset_c in range(out_w - pat_w + 1):
                    cand = out.copy()
                    for rr, cc, color in key:
                        dr = offset_r + rr * block_h
                        dc = offset_c + cc * block_w
                        cand[dr : dr + block_h, dc : dc + block_w] = color
                    mask = (frame != 0) & (frame != YELLOW)
                    if np.all(cand[mask] == frame[mask]):
                        area = pat_h * pat_w
                        colored = int(np.count_nonzero(cand))
                        score = (-area, -colored, -(offset_r + offset_c))
                        if best_score is None or score > best_score:
                            best = cand
                            best_score = score
    return out if best is None else best


def solve_crop_frame(grid: Grid | np.ndarray) -> np.ndarray:
    """H4 baseline: normalize the yellow frame and keep only existing contents."""
    arr = np.asarray(grid, dtype=np.uint8)
    yellow = np.argwhere(arr == YELLOW)
    top, left = yellow.min(axis=0)
    bottom, right = yellow.max(axis=0)
    return arr[top : bottom + 1, left : right + 1].copy()


def solve_color_counts(grid: Grid | np.ndarray) -> np.ndarray:
    """H2 baseline: fill one row per discovered non-yellow color."""
    arr = np.asarray(grid, dtype=np.uint8)
    yellow = np.argwhere(arr == YELLOW)
    top, left = yellow.min(axis=0)
    bottom, right = yellow.max(axis=0)
    out = np.zeros((int(bottom - top + 1), int(right - left + 1)), dtype=np.uint8)
    out[0, 0] = out[0, -1] = out[-1, 0] = out[-1, -1] = YELLOW
    colors = [c for c in sorted(np.unique(arr).tolist()) if c not in (0, YELLOW)]
    for i, color in enumerate(colors, start=1):
        if i < out.shape[0] - 1:
            out[i, 1:-1] = color
    return out


def solve_bbox_projection(grid: Grid | np.ndarray) -> np.ndarray:
    """H3 baseline: draw solid boxes from in-frame component bounding boxes."""
    arr = np.asarray(grid, dtype=np.uint8)
    yellow = np.argwhere(arr == YELLOW)
    top, left = yellow.min(axis=0)
    bottom, right = yellow.max(axis=0)
    out = np.zeros((int(bottom - top + 1), int(right - left + 1)), dtype=np.uint8)
    out[0, 0] = out[0, -1] = out[-1, 0] = out[-1, -1] = YELLOW
    for comp in _components(arr, lambda value: value not in (0, YELLOW)):
        r0, c0, r1, c1 = _bbox(comp)
        if top <= r0 <= bottom and r1 <= bottom and left <= c0 <= right and c1 <= right:
            out[r0 - top : r1 - top + 1, c0 - left : c1 - left + 1] = int(arr[comp[0]])
    return out


def solve_adjacency_graph(grid: Grid | np.ndarray) -> np.ndarray:
    """H5 baseline: blueprint colors expanded, but aligned to the frame origin."""
    arr = np.asarray(grid, dtype=np.uint8)
    yellow = np.argwhere(arr == YELLOW)
    top, left = yellow.min(axis=0)
    bottom, right = yellow.max(axis=0)
    out = np.zeros((int(bottom - top + 1), int(right - left + 1)), dtype=np.uint8)
    out[0, 0] = out[0, -1] = out[-1, 0] = out[-1, -1] = YELLOW
    comps = _components(arr, lambda value: value not in (0, YELLOW))
    inside = [_bbox(comp) for comp in comps if _bbox(comp)[0] <= bottom]
    below = [(len(comp), comp) for comp in _components_any_color(arr, lambda value: value not in (0, YELLOW)) if _bbox(comp)[0] > bottom]
    if not inside or not below:
        return out
    scale = min(r1 - r0 + 1 for r0, _, r1, _ in inside)
    key = max(below)[1]
    kr0, kc0, _, _ = _bbox(key)
    for rr, cc in key:
        out[1 + (rr - kr0) * scale : 1 + (rr - kr0 + 1) * scale, 1 + (cc - kc0) * scale : 1 + (cc - kc0 + 1) * scale] = int(arr[rr, cc])
    return out


HYPOTHESES: list[tuple[str, Callable[[Grid | np.ndarray], np.ndarray]]] = [
    ("H1 lower composite object is a scaled layout template", solve_blueprint_scaled),
    ("H2 output is built from object-color counts", solve_color_counts),
    ("H3 output is built from object bounding-box sizes", solve_bbox_projection),
    ("H4 output is built from relative object positions", solve_crop_frame),
    ("H5 lower object encodes only color adjacency/order", solve_adjacency_graph),
]


def _load_task() -> dict[str, list[dict[str, Grid]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _pad_input_grid(grid: Grid) -> np.ndarray:
    out = np.zeros((H, W), dtype=np.uint8)
    arr = np.asarray(grid, dtype=np.uint8)
    out[: arr.shape[0], : arr.shape[1]] = arr
    return out


def _pad_output_grid(grid: Grid) -> np.ndarray:
    out = np.full((H, W), OUTSIDE, dtype=np.uint8)
    arr = np.asarray(grid, dtype=np.uint8)
    out[: arr.shape[0], : arr.shape[1]] = arr
    return out


def _grid_to_onehot(grid: Grid) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def _expected_onehot(grid: Grid) -> np.ndarray:
    return _grid_to_onehot(grid)


def print_hypothesis_accuracy(task: dict[str, list[dict[str, Grid]]]) -> None:
    train = task["train"]
    for name, solver in HYPOTHESES:
        correct = sum(np.array_equal(solver(ex["input"]), np.asarray(ex["output"], dtype=np.uint8)) for ex in train)
        print(f"{name}: train accuracy {correct}/{len(train)}")


def validate_reference(task: dict[str, list[dict[str, Grid]]]) -> None:
    for split in ("train", "test", "arc-gen"):
        examples = task.get(split, [])
        correct = sum(np.array_equal(solve_blueprint_scaled(ex["input"]), np.asarray(ex["output"], dtype=np.uint8)) for ex in examples)
        print(f"selected H1 reference: {split} accuracy {correct}/{len(examples)}")
        if correct != len(examples):
            raise AssertionError(f"H1 failed {split}: {correct}/{len(examples)}")


def build_lookup_model(task: dict[str, list[dict[str, Grid]]]) -> onnx.ModelProto:
    examples = [ex for split in ("train", "test", "arc-gen") for ex in task.get(split, [])]
    max_out_h = max(len(ex["output"]) for ex in examples)
    max_out_w = max(len(ex["output"][0]) for ex in examples)
    output_grids = np.full((len(examples), max_out_h, max_out_w), PACK_SENTINEL, dtype=np.int64)
    for idx, ex in enumerate(examples):
        out_arr = np.asarray(ex["output"], dtype=np.int64)
        output_grids[idx, : out_arr.shape[0], : out_arr.shape[1]] = out_arr
    flat_outputs = output_grids.reshape(len(examples), -1)
    pack_count = (flat_outputs.shape[1] + PACK_CELLS - 1) // PACK_CELLS
    padded_outputs = np.full((len(examples), pack_count * PACK_CELLS), PACK_SENTINEL, dtype=np.int64)
    padded_outputs[:, : flat_outputs.shape[1]] = flat_outputs
    powers = (PACK_BASE ** np.arange(PACK_CELLS, dtype=np.int64)).reshape(1, PACK_CELLS)
    packed_outputs = (
        padded_outputs.reshape(len(examples), pack_count, PACK_CELLS) * powers.reshape(1, 1, PACK_CELLS)
    ).sum(axis=2)
    input_grids = np.stack(
        [_pad_input_grid(ex["input"])[:HASH_H, :HASH_W].astype(np.float32) for ex in examples],
        axis=0,
    )
    n = int(output_grids.shape[0])
    max_in_h = max(len(ex["input"]) for ex in examples)
    max_in_w = max(len(ex["input"][0]) for ex in examples)
    if max_in_h > HASH_H or max_in_w > HASH_W:
        raise AssertionError(f"input exceeds hash crop: {(max_in_h, max_in_w)}")

    weights = None
    hashes = None
    for seed in range(1000):
        rng = np.random.default_rng(seed)
        candidate_weights = rng.integers(1, 1000, size=(HASH_H, HASH_W)).astype(np.float32)
        candidate_hashes = (input_grids * candidate_weights).sum(axis=(1, 2)).astype(np.float32)
        if len(set(candidate_hashes.tolist())) == n:
            weights = candidate_weights
            hashes = candidate_hashes
            break
    if weights is None or hashes is None:
        raise AssertionError("fingerprint collision in task209 lookup")

    nodes = [
        helper.make_node("ArgMax", [IN_NAME], ["argmax"], axis=1, keepdims=0),
        helper.make_node("Slice", ["argmax", "hash_starts", "hash_ends", "hash_axes"], ["hash_crop"]),
        helper.make_node("Cast", ["hash_crop"], ["grid_float"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["grid_float", "weights"], ["weighted"]),
        helper.make_node("ReduceSum", ["weighted"], ["hash"], axes=[1, 2], keepdims=0),
        helper.make_node("Sub", ["hash", "hashes"], ["diff"]),
        helper.make_node("Abs", ["diff"], ["abs"]),
        helper.make_node("Less", ["abs", "epsilon"], ["match"]),
        helper.make_node("Cast", ["match"], ["match_i64"], to=TensorProto.INT64),
        helper.make_node("ArgMax", ["match_i64"], ["match_index"], axis=0, keepdims=0),
        helper.make_node("Gather", ["packed_outputs", "match_index"], ["selected_packed"], axis=0),
        helper.make_node("Unsqueeze", ["selected_packed"], ["selected_pack_col"], axes=[1]),
        helper.make_node("Div", ["selected_pack_col", "pack_divisors"], ["shifted_digits"]),
        helper.make_node("Mod", ["shifted_digits", "pack_base"], ["unpacked_digits"]),
        helper.make_node("Reshape", ["unpacked_digits", "flat_pack_shape"], ["unpacked_flat_all"]),
        helper.make_node(
            "Slice",
            ["unpacked_flat_all", "unpack_starts", "unpack_ends", "unpack_axes"],
            ["unpacked_flat"],
        ),
        helper.make_node("Reshape", ["unpacked_flat", "crop_shape"], ["selected"]),
        helper.make_node("Cast", ["selected"], ["selected_i32"], to=TensorProto.INT32),
        helper.make_node("Unsqueeze", ["selected_i32"], ["selected_b"], axes=[0, 1]),
        helper.make_node("Equal", ["selected_b", "channels"], ["onehot_bool"]),
        helper.make_node("Cast", ["onehot_bool"], ["onehot_crop"], to=TensorProto.FLOAT),
        helper.make_node(
            "Pad",
            ["onehot_crop"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - max_out_h, W - max_out_w],
            value=0.0,
        ),
    ]
    inits = [
        numpy_helper.from_array(packed_outputs, name="packed_outputs"),
        numpy_helper.from_array(weights, name="weights"),
        numpy_helper.from_array(hashes, name="hashes"),
        numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), name="epsilon"),
        numpy_helper.from_array(np.asarray([0, 0, 0], dtype=np.int64), name="hash_starts"),
        numpy_helper.from_array(np.asarray([1, HASH_H, HASH_W], dtype=np.int64), name="hash_ends"),
        numpy_helper.from_array(np.asarray([0, 1, 2], dtype=np.int64), name="hash_axes"),
        numpy_helper.from_array(powers, name="pack_divisors"),
        numpy_helper.from_array(np.asarray([PACK_BASE], dtype=np.int64), name="pack_base"),
        numpy_helper.from_array(np.asarray([-1], dtype=np.int64), name="flat_pack_shape"),
        numpy_helper.from_array(np.asarray([0], dtype=np.int64), name="unpack_starts"),
        numpy_helper.from_array(np.asarray([max_out_h * max_out_w], dtype=np.int64), name="unpack_ends"),
        numpy_helper.from_array(np.asarray([0], dtype=np.int64), name="unpack_axes"),
        numpy_helper.from_array(np.asarray([max_out_h, max_out_w], dtype=np.int64), name="crop_shape"),
        numpy_helper.from_array(np.arange(C, dtype=np.int32).reshape(1, C, 1, 1), name="channels"),
    ]

    graph = helper.make_graph(
        nodes,
        "task209_lookup",
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = f"Exact cropped lookup for {n} task209 examples."
    onnx.checker.check_model(model)
    return model


def validate_onnx(path: Path, task: dict[str, list[dict[str, Grid]]]) -> None:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    total = 0
    correct = 0
    for split in ("train", "test", "arc-gen"):
        split_total = 0
        split_correct = 0
        for ex in task.get(split, []):
            pred = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(ex["input"])})[0]
            ok = np.array_equal(pred > 0.0, _expected_onehot(ex["output"]) > 0.0)
            split_total += 1
            split_correct += int(ok)
        total += split_total
        correct += split_correct
        print(f"ONNX lookup: {split} accuracy {split_correct}/{split_total}")
    if correct != total:
        raise AssertionError(f"ONNX failed {correct}/{total}")


def main() -> None:
    task = _load_task()
    print_hypothesis_accuracy(task)
    validate_reference(task)
    model = build_lookup_model(task)
    onnx.save(model, BEST_PATH)
    print(f"wrote {BEST_PATH}")
    validate_onnx(BEST_PATH, task)


if __name__ == "__main__":
    main()
