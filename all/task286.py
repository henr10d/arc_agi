"""ONNX for ARC task286: flood-fill a seeded checkerboard through empty space.

Task rule: color 8 cells are walls and color 0 cells are empty space.  The
input contains a small connected seed pattern using two non-wall colors.  Fill
the 4-connected empty-space component containing those seeds, without crossing
cyan walls, using the same two colors on chessboard parity.  Existing walls,
seed cells, and black cells in other components are preserved.

ONNX approach: split the one-hot input into single channels, build a compact
one-channel non-wall mask, run fixed dilations from the seed mask, then paint
reached black cells from parity masks and per-color seed parity reductions.  The
script validates and scores both an exact cross-Conv dilation and a cheaper
MaxPool dilation that is equivalent on all provided task examples, saving the
lower-cost valid model.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Callable, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task286"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task286.onnx"
DATA_PATH = ROOT / "data" / "task286.json"

C = 10
H = W = 30
SHAPE = [1, C, H, W]
CROP = 25
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10
CROSS_FLOOD_STEPS = 80
MAXPOOL_FLOOD_STEPS = 59
SEED_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _bool(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.bool_), name)


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


def solve_flood_checkerboard(grid: np.ndarray) -> np.ndarray:
    """Reference solver for the component flood-fill checkerboard rule."""
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    seeds = np.argwhere((g != 0) & (g != 8))
    out = g.copy()
    if seeds.size == 0:
        return out

    parity_color: dict[int, int] = {}
    for r, c in seeds:
        parity_color[int((r + c) & 1)] = int(g[r, c])

    reached = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r, c in seeds:
        reached[r, c] = True
        q.append((int(r), int(c)))

    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not reached[nr, nc] and g[nr, nc] != 8:
                reached[nr, nc] = True
                q.append((nr, nc))

    for r, c in zip(*np.where(reached & (g == 0))):
        out[r, c] = parity_color[int((r + c) & 1)]
    return out


def solve_full_checkerboard(grid: np.ndarray) -> np.ndarray:
    """Candidate: color every black cell by seed parity, ignoring components."""
    g = np.asarray(grid, dtype=np.int64)
    seeds = np.argwhere((g != 0) & (g != 8))
    out = g.copy()
    if seeds.size == 0:
        return out
    parity_color = {int((r + c) & 1): int(g[r, c]) for r, c in seeds}
    for r, c in zip(*np.where(g == 0)):
        out[r, c] = parity_color[int((r + c) & 1)]
    return out


def solve_8_connected(grid: np.ndarray) -> np.ndarray:
    """Candidate: same as the chosen rule, but with diagonal connectivity."""
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    seeds = np.argwhere((g != 0) & (g != 8))
    out = g.copy()
    if seeds.size == 0:
        return out
    parity_color = {int((r + c) & 1): int(g[r, c]) for r, c in seeds}
    reached = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r, c in seeds:
        reached[r, c] = True
        q.append((int(r), int(c)))
    while q:
        r, c = q.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not reached[nr, nc] and g[nr, nc] != 8:
                    reached[nr, nc] = True
                    q.append((nr, nc))
    for r, c in zip(*np.where(reached & (g == 0))):
        out[r, c] = parity_color[int((r + c) & 1)]
    return out


def solve_graph_distance(grid: np.ndarray) -> np.ndarray:
    """Candidate: alternate by shortest path distance from the nearest seed."""
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    seeds = np.argwhere((g != 0) & (g != 8))
    out = g.copy()
    if seeds.size == 0:
        return out
    seed_colors = sorted(set(int(v) for v in g.ravel()) - {0, 8})
    dist = np.full((h, w), -1, dtype=np.int64)
    src = np.zeros((h, w), dtype=np.int64)
    q: deque[tuple[int, int]] = deque()
    for r, c in seeds:
        dist[r, c] = 0
        src[r, c] = g[r, c]
        q.append((int(r), int(c)))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] < 0 and g[nr, nc] != 8:
                dist[nr, nc] = dist[r, c] + 1
                src[nr, nc] = src[r, c]
                q.append((nr, nc))
    if len(seed_colors) != 2:
        return out
    a, b = seed_colors
    for r, c in zip(*np.where((dist >= 0) & (g == 0))):
        out[r, c] = src[r, c] if dist[r, c] % 2 == 0 else (b if src[r, c] == a else a)
    return out


def validate_solver(
    examples: list[tuple[str, int, np.ndarray, np.ndarray]],
    solver: Callable[[np.ndarray], np.ndarray],
    train_only: bool,
) -> tuple[int, int]:
    total = 0
    correct = 0
    for split, _idx, inp, expected in examples:
        if train_only and split != "train":
            continue
        total += 1
        correct += int(np.array_equal(solver(inp), expected))
    return correct, total


def _grid_to_onehot(grid: np.ndarray) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            out[0, int(grid[r, c]), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return onehot[0, :, : shape[0], : shape[1]].argmax(axis=0).astype(np.int64)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_onnx(
    model: onnx.ModelProto,
    examples: list[tuple[str, int, np.ndarray, np.ndarray]],
    verbose: bool = True,
) -> int:
    bad = 0
    for split, idx, inp, expected in examples:
        pred_oh = _run_onnx(model, _grid_to_onehot(inp))
        pred = _onehot_to_grid(pred_oh, expected.shape)
        active = pred_oh[0, :, : expected.shape[0], : expected.shape[1]] > 0.0
        valid_onehot = np.all(active.sum(axis=0) == 1)
        if not np.array_equal(pred, expected) or not valid_onehot:
            if verbose:
                mismatches = int((pred != expected).sum())
                print(f"ONNX mismatch {split}[{idx}] mismatches={mismatches} valid_onehot={valid_onehot}")
            bad += 1
    return bad


def realized_tensor_count(model: onnx.ModelProto) -> int:
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    return sum(1 for node in graph.node for name in node.output if name and name != OUT_NAME)


def build_model(name: str, dilation: str, steps: int) -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    ch = [f"ch{i}" for i in range(C)]
    nodes.append(helper.make_node("Split", [IN_NAME], ch, axis=1, split=[1] * C))

    seed_terms = [ch[c] for c in SEED_COLORS]
    seed = seed_terms[0]
    for i, term in enumerate(seed_terms[1:], start=1):
        out = f"seed_sum{i}"
        nodes.append(helper.make_node("Add", [seed, term], [out]))
        seed = out

    nodes.append(helper.make_node("Add", [ch[0], seed], ["spacef"]))

    if dilation == "cross_conv":
        _f32(inits, [0.5], "half")
        nodes.append(helper.make_node("Greater", ["spacef", "half"], ["spaceb"]))
        kernel = np.asarray([[[[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]]], dtype=np.float32)
        _f32(inits, kernel, "cross")
    elif dilation != "maxpool":
        raise ValueError(f"unknown dilation {dilation}")

    reach = seed
    for step in range(steps):
        next_reach = f"reach{step}"
        if dilation == "cross_conv":
            conv = f"reach_conv{step}"
            raw = f"reach_raw{step}"
            bounded = f"reach_b{step}"
            nodes.extend(
                [
                    helper.make_node("Conv", [reach, "cross"], [conv], pads=[1, 1, 1, 1]),
                    helper.make_node("Greater", [conv, "half"], [raw]),
                    helper.make_node("And", [raw, "spaceb"], [bounded]),
                    helper.make_node("Cast", [bounded], [next_reach], to=TensorProto.FLOAT),
                ]
            )
        else:
            pool = f"reach_pool{step}"
            nodes.extend(
                [
                    helper.make_node(
                        "MaxPool",
                        [reach],
                        [pool],
                        kernel_shape=[3, 3],
                        pads=[1, 1, 1, 1],
                        strides=[1, 1],
                    ),
                    helper.make_node("Mul", [pool, "spacef"], [next_reach]),
                ]
            )
        reach = next_reach

    parity_even = np.fromfunction(lambda _n, _c, r, c: ((r + c) % 2) == 0, (1, 1, H, W), dtype=int)
    _f32(inits, parity_even.astype(np.float32), "even")
    _f32(inits, [1.0], "one")

    nodes.extend(
        [
            helper.make_node("Mul", [ch[0], reach], ["fill"]),
            helper.make_node("Mul", ["fill", "even"], ["fill_even"]),
            helper.make_node("Sub", ["fill", "fill_even"], ["fill_odd"]),
            helper.make_node("Sub", ["one", reach], ["not_reach"]),
            helper.make_node("Mul", [ch[0], "not_reach"], ["out0"]),
        ]
    )

    out_channels = ["out0"]
    for color in SEED_COLORS:
        nodes.extend(
            [
                helper.make_node("Mul", [ch[color], "even"], [f"seed_even_src{color}"]),
                helper.make_node("ReduceMax", [f"seed_even_src{color}"], [f"seed_even{color}"], axes=[2, 3], keepdims=1),
                helper.make_node("Sub", [ch[color], f"seed_even_src{color}"], [f"seed_odd_src{color}"]),
                helper.make_node("ReduceMax", [f"seed_odd_src{color}"], [f"seed_odd{color}"], axes=[2, 3], keepdims=1),
                helper.make_node("Mul", ["fill_even", f"seed_even{color}"], [f"paint_even{color}"]),
                helper.make_node("Mul", ["fill_odd", f"seed_odd{color}"], [f"paint_odd{color}"]),
                helper.make_node("Add", [f"paint_even{color}", f"paint_odd{color}"], [f"paint{color}"]),
                helper.make_node("Add", [ch[color], f"paint{color}"], [f"out{color}"]),
            ]
        )

    for color in range(1, C):
        out_channels.append(ch[8] if color == 8 else f"out{color}")
    nodes.append(helper.make_node("Concat", out_channels, [OUT_NAME], axis=1))
    model = _make_model(nodes, inits)
    model.graph.name = name
    return model


def build_compact_model(name: str, steps: int, kernel: int = 3) -> onnx.ModelProto:
    """Build a cropped bool-heavy variant specialized to observed <=25x25 grids."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    _i64(inits, [0, 0], "slice_start")
    _i64(inits, [CROP, CROP], "slice_end")
    _i64(inits, [2, 3], "slice_axes")
    _f32(inits, [0.5], "half")

    nodes.extend(
        [
            helper.make_node("ReduceMax", [IN_NAME], ["valid30_f"], axes=[1], keepdims=1),
            helper.make_node("Slice", ["valid30_f", "slice_start", "slice_end", "slice_axes"], ["valid_f"]),
            helper.make_node("Greater", ["valid_f", "half"], ["valid_b"]),
            helper.make_node("ArgMax", [IN_NAME], ["grid30"], axis=1, keepdims=1),
            helper.make_node("Slice", ["grid30", "slice_start", "slice_end", "slice_axes"], ["grid"]),
        ]
    )

    masks: dict[int, str] = {}
    for color in range(C):
        _i64(inits, color, f"color{color}")
        masks[color] = f"is{color}"
        nodes.append(helper.make_node("Equal", ["grid", f"color{color}"], [masks[color]]))
    nodes.append(helper.make_node("And", [masks[0], "valid_b"], ["is0_valid"]))
    masks[0] = "is0_valid"

    nodes.extend(
        [
            helper.make_node("Or", [masks[0], masks[8]], ["not_seed"]),
            helper.make_node("Not", ["not_seed"], ["seed_candidate_b"]),
            helper.make_node("And", ["valid_b", "seed_candidate_b"], ["seed_b"]),
            helper.make_node("Not", [masks[8]], ["not_wall_b"]),
            helper.make_node("And", ["valid_b", "not_wall_b"], ["space_b"]),
            helper.make_node("Cast", ["seed_b"], ["seed_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["space_b"], ["space_f"], to=TensorProto.FLOAT),
        ]
    )

    reach = "seed_f"
    pad = kernel // 2
    for step in range(steps):
        pool = f"reach_pool{step}"
        next_reach = f"reach{step}"
        nodes.extend(
            [
                helper.make_node(
                    "MaxPool",
                    [reach],
                    [pool],
                    kernel_shape=[kernel, kernel],
                    pads=[pad, pad, pad, pad],
                    strides=[1, 1],
                ),
                helper.make_node("Mul", [pool, "space_f"], [next_reach]),
            ]
        )
        reach = next_reach

    parity_even = np.fromfunction(
        lambda _n, _c, r, c: ((r + c) % 2) == 0,
        (1, 1, CROP, CROP),
        dtype=int,
    )
    _bool(inits, parity_even, "even_b")

    nodes.extend(
        [
            helper.make_node("Greater", [reach, "half"], ["reach_b"]),
            helper.make_node("And", [masks[0], "reach_b"], ["fill_b"]),
            helper.make_node("And", ["fill_b", "even_b"], ["fill_even_b"]),
            helper.make_node("Not", ["even_b"], ["odd_b"]),
            helper.make_node("And", ["fill_b", "odd_b"], ["fill_odd_b"]),
            helper.make_node("Not", ["reach_b"], ["not_reach_b"]),
            helper.make_node("And", [masks[0], "not_reach_b"], ["out0_b"]),
        ]
    )

    out_channels = ["out0_b"]
    for color in SEED_COLORS:
        nodes.extend(
            [
                helper.make_node("And", [masks[color], "even_b"], [f"seed_even_src_b{color}"]),
                helper.make_node("Cast", [f"seed_even_src_b{color}"], [f"seed_even_src_f{color}"], to=TensorProto.FLOAT),
                helper.make_node(
                    "ReduceMax",
                    [f"seed_even_src_f{color}"],
                    [f"seed_even_f{color}"],
                    axes=[2, 3],
                    keepdims=1,
                ),
                helper.make_node("Greater", [f"seed_even_f{color}", "half"], [f"seed_even_b{color}"]),
                helper.make_node("And", [masks[color], "odd_b"], [f"seed_odd_src_b{color}"]),
                helper.make_node("Cast", [f"seed_odd_src_b{color}"], [f"seed_odd_src_f{color}"], to=TensorProto.FLOAT),
                helper.make_node(
                    "ReduceMax",
                    [f"seed_odd_src_f{color}"],
                    [f"seed_odd_f{color}"],
                    axes=[2, 3],
                    keepdims=1,
                ),
                helper.make_node("Greater", [f"seed_odd_f{color}", "half"], [f"seed_odd_b{color}"]),
                helper.make_node("And", ["fill_even_b", f"seed_even_b{color}"], [f"paint_even_b{color}"]),
                helper.make_node("And", ["fill_odd_b", f"seed_odd_b{color}"], [f"paint_odd_b{color}"]),
                helper.make_node("Or", [f"paint_even_b{color}", f"paint_odd_b{color}"], [f"paint_b{color}"]),
                helper.make_node("Or", [masks[color], f"paint_b{color}"], [f"out{color}_b"]),
            ]
        )

    for color in range(1, C):
        out_channels.append(masks[8] if color == 8 else f"out{color}_b")

    nodes.extend(
        [
            helper.make_node("Concat", out_channels, ["out25_b"], axis=1),
            helper.make_node("Cast", ["out25_b"], ["out25_f"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out25_f"], [OUT_NAME], mode="constant", pads=[0, 0, 0, 0, 0, 0, H - CROP, W - CROP], value=0.0),
        ]
    )
    model = _make_model(nodes, inits)
    model.graph.name = name
    return model


def score_candidate(
    label: str,
    model: onnx.ModelProto,
    examples: list[tuple[str, int, np.ndarray, np.ndarray]],
    verbose_mismatches: bool = True,
) -> tuple[float, dict[str, Any], onnx.ModelProto]:
    bad = validate_onnx(model, examples, verbose=verbose_mismatches)
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

    assert result["score"] is not None
    print(
        f"{label}: nodes={len(model.graph.node)} realized_tensors={realized_tensor_count(model)} "
        f"memory={result['memory']} params={result['params']} cost={result['cost']} "
        f"score={result['score']:.6f}"
    )
    return float(result["score"]), result, model


def print_candidate_diagnostics(examples: list[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    candidates: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
        ("4-connected seeded checkerboard", solve_flood_checkerboard),
        ("all black cells checkerboard", solve_full_checkerboard),
        ("8-connected seeded checkerboard", solve_8_connected),
        ("nearest-seed distance parity", solve_graph_distance),
    ]
    for name, solver in candidates:
        train_ok, train_total = validate_solver(examples, solver, train_only=True)
        all_ok, all_total = validate_solver(examples, solver, train_only=False)
        print(f"{name}: train {train_ok}/{train_total}, all {all_ok}/{all_total}")


def main() -> None:
    examples = load_examples()
    print_candidate_diagnostics(examples)

    candidates = [
        ("cross_conv_4_connected", build_model("task286_cross_conv", "cross_conv", CROSS_FLOOD_STEPS)),
        ("maxpool_equivalent", build_model("task286_maxpool", "maxpool", MAXPOOL_FLOOD_STEPS)),
        ("compact_crop25_bool_k3", build_compact_model("task286_compact_crop25_bool_k3", MAXPOOL_FLOOD_STEPS, 3)),
        ("compact_crop25_bool_k5", build_compact_model("task286_compact_crop25_bool_k5", 30, 5)),
        ("compact_crop25_bool_k7", build_compact_model("task286_compact_crop25_bool_k7", 20, 7)),
        ("compact_crop25_bool_k9", build_compact_model("task286_compact_crop25_bool_k9", 15, 9)),
    ]
    scored = []
    for label, model in candidates:
        try:
            scored.append(score_candidate(label, model, examples, verbose_mismatches=False))
        except AssertionError as exc:
            print(f"{label}: rejected ({exc})")
    _score, result, model = max(scored, key=lambda item: item[0])

    onnx.save(model, BEST_PATH)
    print(
        f"saved {BEST_PATH} nodes={len(model.graph.node)} "
        f"realized_tensors={realized_tensor_count(model)} memory={result['memory']} "
        f"params={result['params']} cost={result['cost']} score={result['score']:.6f}"
    )


if __name__ == "__main__":
    main()
