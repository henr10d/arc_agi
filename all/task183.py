"""Compact ONNX for ARC task183: recolor the central cyan mask by corner quadrant.

Task rule: the input is a 6x6, 8x8, or 10x10 grid padded into the NeuroGolf
30x30 one-hot tensor. A blue rectangular frame surrounds a central square whose
cells are black or cyan (8), and each outer corner contains a non-blue marker.
The output is the central square (input rows/cols 2..N-3): cyan cells are
recolored with the marker from their corresponding quadrant, while non-cyan
cells become black. The output is placed at the top-left and padded with zeros.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task183"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    op_estimate: int
    solve: Callable[[np.ndarray], np.ndarray]


def load_task() -> dict:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def grid_array(example: dict, key: str) -> np.ndarray:
    return np.asarray(example[key], dtype=np.int64)


def all_examples(data: dict, splits: Iterable[str] = ("train", "test", "arc-gen")) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    rows: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    for split in splits:
        for idx, ex in enumerate(data.get(split, [])):
            rows.append((split, idx, grid_array(ex, "input"), grid_array(ex, "output")))
    return rows


def task_size(grid: np.ndarray) -> int:
    return int(grid.shape[0])


def corner_colors(grid: np.ndarray) -> tuple[int, int, int, int]:
    n = task_size(grid)
    return int(grid[0, 0]), int(grid[0, n - 1]), int(grid[n - 1, 0]), int(grid[n - 1, n - 1])


def central_mask(grid: np.ndarray) -> np.ndarray:
    n = task_size(grid)
    return grid[2 : n - 2, 2 : n - 2]


def solve_quadrant_cyan(grid: np.ndarray) -> np.ndarray:
    core = central_mask(grid)
    out_h, out_w = core.shape
    tl, tr, bl, br = corner_colors(grid)
    out = np.zeros_like(core)
    mid_r = out_h // 2
    mid_c = out_w // 2
    qcolors = np.empty_like(core)
    qcolors[:mid_r, :mid_c] = tl
    qcolors[:mid_r, mid_c:] = tr
    qcolors[mid_r:, :mid_c] = bl
    qcolors[mid_r:, mid_c:] = br
    out[core == 8] = qcolors[core == 8]
    return out


def solve_block_dominant(grid: np.ndarray, ignore: set[int]) -> np.ndarray:
    n = task_size(grid)
    out_n = n // 2
    out = np.zeros((out_n, out_n), dtype=np.int64)
    for r in range(out_n):
        for c in range(out_n):
            block = grid[2 * r : 2 * r + 2, 2 * c : 2 * c + 2].ravel()
            vals = [int(v) for v in block if int(v) not in ignore]
            if vals:
                counts = np.bincount(np.asarray(vals, dtype=np.int64), minlength=10)
                out[r, c] = int(np.argmax(counts))
    return out


def solve_region_corner(grid: np.ndarray, require_cyan: bool) -> np.ndarray:
    core = central_mask(grid)
    out_h, out_w = core.shape
    tl, tr, bl, br = corner_colors(grid)
    out = np.zeros_like(core)
    mid_r = out_h // 2
    mid_c = out_w // 2
    colors = ((tl, tr), (bl, br))
    for r in range(out_h):
        for c in range(out_w):
            if not require_cyan or core[r, c] != 0:
                out[r, c] = colors[int(r >= mid_r)][int(c >= mid_c)]
    return out


def solve_component_centroid(grid: np.ndarray) -> np.ndarray:
    core = central_mask(grid)
    out = np.zeros_like(core)
    ys, xs = np.where(core == 8)
    if len(ys) == 0:
        return out
    tl, tr, bl, br = corner_colors(grid)
    mid_r = core.shape[0] // 2
    mid_c = core.shape[1] // 2
    for y, x in zip(ys, xs):
        out[y, x] = ((tl, tr), (bl, br))[int(y >= mid_r)][int(x >= mid_c)]
    return out


def solve_nearest_corner(grid: np.ndarray) -> np.ndarray:
    core = central_mask(grid)
    out = np.zeros_like(core)
    colors = corner_colors(grid)
    points = np.asarray(
        [
            [0.0, 0.0],
            [0.0, core.shape[1] - 1.0],
            [core.shape[0] - 1.0, 0.0],
            [core.shape[0] - 1.0, core.shape[1] - 1.0],
        ]
    )
    for r, c in zip(*np.where(core == 8)):
        d2 = np.sum((points - np.asarray([r, c], dtype=np.float64)) ** 2, axis=1)
        out[r, c] = colors[int(np.argmin(d2))]
    return out


def candidate_solvers() -> list[Candidate]:
    return [
        Candidate(
            "block_2x2_ignore_blue_black",
            "A block-compression",
            16,
            lambda g: solve_block_dominant(g, {0, 1}),
        ),
        Candidate(
            "block_2x2_ignore_blue",
            "A block-compression",
            16,
            lambda g: solve_block_dominant(g, {1}),
        ),
        Candidate(
            "region_corner_nonblack",
            "B region-signature",
            14,
            lambda g: solve_region_corner(g, require_cyan=False) * (central_mask(g) != 0),
        ),
        Candidate(
            "region_corner_cyan",
            "B region-signature",
            12,
            solve_quadrant_cyan,
        ),
        Candidate(
            "component_centroid_quadrant",
            "C connected-component abstraction",
            18,
            solve_component_centroid,
        ),
        Candidate(
            "nearest_corner_cyan",
            "D quadrant propagation",
            20,
            solve_nearest_corner,
        ),
        Candidate(
            "vote_major_nonblue",
            "E color-voting",
            16,
            lambda g: solve_block_dominant(g, {1}),
        ),
    ]


def evaluate_candidate(candidate: Candidate, examples: list[tuple[str, int, np.ndarray, np.ndarray]]) -> dict:
    exact = 0
    cells_ok = 0
    cells_total = 0
    per_example: list[str] = []
    for split, idx, inp, expected in examples:
        pred = candidate.solve(inp)
        shape_ok = pred.shape == expected.shape
        if shape_ok:
            match = pred == expected
            ok = bool(np.all(match))
            good = int(np.sum(match))
            total = int(expected.size)
        else:
            ok = False
            good = 0
            total = int(expected.size)
        exact += int(ok)
        cells_ok += good
        cells_total += total
        per_example.append(f"{split}[{idx}]={'OK' if ok else f'{good}/{total}'}")
    return {
        "candidate": candidate,
        "exact": exact,
        "total": len(examples),
        "cells_ok": cells_ok,
        "cells_total": cells_total,
        "per_example": per_example,
    }


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self._counter = 0

    def name(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    def arr(self, value, name: str | None = None) -> str:
        out = name or self.name("c")
        self.inits.append(numpy_helper.from_array(np.asarray(value), out))
        return out

    def i64(self, value, name: str | None = None) -> str:
        return self.arr(np.asarray(value, dtype=np.int64), name)

    def f32(self, value, name: str | None = None) -> str:
        return self.arr(np.asarray(value, dtype=np.float32), name)

    def add(self, op: str, inputs: list[str], outputs: list[str], **kwargs) -> None:
        self.nodes.append(helper.make_node(op, inputs, outputs, **kwargs))


def _slice(b: Builder, x: str, starts: list[int], ends: list[int], axes: list[int] | None = None, prefix: str = "sl") -> str:
    out = b.name(prefix)
    inputs = [x, b.i64(starts), b.i64(ends)]
    if axes is not None:
        inputs.append(b.i64(axes))
    b.add("Slice", inputs, [out])
    return out


def _quadrant_bool_mask(out_n: int, quadrant: str, canvas: int = 6) -> np.ndarray:
    mask = np.zeros((1, 1, canvas, canvas), dtype=bool)
    mid = out_n // 2
    if quadrant == "tl":
        mask[:, :, :mid, :mid] = True
    elif quadrant == "tr":
        mask[:, :, :mid, mid:out_n] = True
    elif quadrant == "bl":
        mask[:, :, mid:out_n, :mid] = True
    elif quadrant == "br":
        mask[:, :, mid:out_n, mid:out_n] = True
    else:
        raise ValueError(quadrant)
    return mask


def _select_bool3(b: Builder, is6: str, is8: str, is10: str, v6: str, v8: str, v10: str, prefix: str) -> str:
    a6 = b.name(f"{prefix}_a6")
    a8 = b.name(f"{prefix}_a8")
    a10 = b.name(f"{prefix}_a10")
    first = b.name(f"{prefix}_first")
    out = b.name(f"{prefix}_sel")
    b.add("And", [is6, v6], [a6])
    b.add("And", [is8, v8], [a8])
    b.add("And", [is10, v10], [a10])
    b.add("Or", [a6, a8], [first])
    b.add("Or", [first, a10], [out])
    return out


def _selected_bool_mask_logic(b: Builder, q: str, is6: str, is8: str, is10: str) -> str:
    mask6 = b.arr(_quadrant_bool_mask(2, q), f"bmask6_{q}")
    mask8 = b.arr(_quadrant_bool_mask(4, q), f"bmask8_{q}")
    mask10 = b.arr(_quadrant_bool_mask(6, q), f"bmask10_{q}")
    return _select_bool3(b, is6, is8, is10, mask6, mask8, mask10, f"{q}_bmask")


def _corner_bool_vectors(b: Builder, is6: str, is8: str, is10: str) -> tuple[str, str, str, str]:
    half = b.f32([0.5])

    def bool_slice(name: str, starts: list[int], ends: list[int]) -> str:
        cell = _slice(b, IN_NAME, starts, ends, prefix=f"{name}_")
        out = b.name(f"{name}_bool")
        b.add("Greater", [cell, half], [out])
        return out

    tl = bool_slice("tl", [0, 0, 0, 0], [1, C, 1, 1])
    tr6 = bool_slice("tr6", [0, 0, 0, 5], [1, C, 1, 6])
    tr8 = bool_slice("tr8", [0, 0, 0, 7], [1, C, 1, 8])
    tr10 = bool_slice("tr10", [0, 0, 0, 9], [1, C, 1, 10])
    bl6 = bool_slice("bl6", [0, 0, 5, 0], [1, C, 6, 1])
    bl8 = bool_slice("bl8", [0, 0, 7, 0], [1, C, 8, 1])
    bl10 = bool_slice("bl10", [0, 0, 9, 0], [1, C, 10, 1])
    br6 = bool_slice("br6", [0, 0, 5, 5], [1, C, 6, 6])
    br8 = bool_slice("br8", [0, 0, 7, 7], [1, C, 8, 8])
    br10 = bool_slice("br10", [0, 0, 9, 9], [1, C, 10, 10])

    tr = _select_bool3(b, is6, is8, is10, tr6, tr8, tr10, "tr_corner")
    bl = _select_bool3(b, is6, is8, is10, bl6, bl8, bl10, "bl_corner")
    br = _select_bool3(b, is6, is8, is10, br6, br8, br10, "br_corner")
    return tl, tr, bl, br


def _bool_output(b: Builder, is8: str, is10: str) -> str:
    not8 = b.name("not8")
    not10 = b.name("not10")
    is8_only = b.name("is8_only")
    is6 = b.name("is6")
    b.add("Not", [is8], [not8])
    b.add("Not", [is10], [not10])
    b.add("And", [is8, not10], [is8_only])
    b.add("And", [not8, not10], [is6])

    cyan_f = _slice(b, IN_NAME, [0, 8, 2, 2], [1, 9, 8, 8], prefix="cyan_")
    cyan = b.name("cyan_bool")
    b.add("Greater", [cyan_f, b.f32([0.5])], [cyan])

    tl, tr, bl, br = _corner_bool_vectors(b, is6, is8_only, is10)
    tl_mask = _selected_bool_mask_logic(b, "tl", is6, is8_only, is10)
    tr_mask = _selected_bool_mask_logic(b, "tr", is6, is8_only, is10)
    bl_mask = _selected_bool_mask_logic(b, "bl", is6, is8_only, is10)
    br_mask = _selected_bool_mask_logic(b, "br", is6, is8_only, is10)
    valid_a = b.name("valid_a")
    valid_b = b.name("valid_b")
    valid = b.name("valid")
    b.add("Or", [tl_mask, tr_mask], [valid_a])
    b.add("Or", [bl_mask, br_mask], [valid_b])
    b.add("Or", [valid_a, valid_b], [valid])

    terms: list[str] = []
    for vec, mask, q in ((tl, tl_mask, "tl"), (tr, tr_mask, "tr"), (bl, bl_mask, "bl"), (br, br_mask, "br")):
        term = b.name(f"{q}_colored")
        b.add("And", [vec, mask], [term])
        terms.append(term)

    color_a = b.name("color_a")
    color_b = b.name("color_b")
    region = b.name("region")
    b.add("Or", [terms[0], terms[1]], [color_a])
    b.add("Or", [terms[2], terms[3]], [color_b])
    b.add("Or", [color_a, color_b], [region])

    cyan_color = b.name("cyan_color")
    b.add("And", [region, cyan], [cyan_color])
    not_cyan = b.name("not_cyan")
    black_cell = b.name("black_cell")
    b.add("Not", [cyan], [not_cyan])
    b.add("And", [valid, not_cyan], [black_cell])
    black_vec = np.zeros((1, C, 1, 1), dtype=bool)
    black_vec[:, 0, :, :] = True
    black = b.name("black")
    b.add("And", [b.arr(black_vec, "black_bool_vec"), black_cell], [black])
    content_bool = b.name("content_bool")
    content = b.name("content")
    b.add("Or", [cyan_color, black], [content_bool])
    b.add("Cast", [content_bool], [content], to=TensorProto.FLOAT)

    pads = [0, 0, 0, 0, 0, 0, H - 6, W - 6]
    b.add("Pad", [content], [OUT_NAME], mode="constant", pads=pads, value=0.0)
    return OUT_NAME


def _size_gate(b: Builder, n: int) -> str:
    cell = _slice(b, IN_NAME, [0, 0, n - 1, n - 1], [1, C, n, n], prefix=f"gate{n}_")
    mass = b.name(f"mass{n}")
    b.add("ReduceSum", [cell], [mass], axes=[1], keepdims=1)
    gate = b.name(f"is{n}")
    b.add("Greater", [mass, b.f32([0.5])], [gate])
    return gate


def build_onnx_model() -> onnx.ModelProto:
    b = Builder()
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, [1, C, H, W])
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, [1, C, H, W])

    is8 = _size_gate(b, 8)
    is10 = _size_gate(b, 10)
    _bool_output(b, is8, is10)

    graph = helper.make_graph(b.nodes, TASK_ID, [x_info], [y_info], b.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    return model


def one_hot_input(grid: np.ndarray) -> np.ndarray:
    arr = np.zeros((1, C, H, W), dtype=np.float32)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            arr[0, int(grid[r, c]), r, c] = 1.0
    return arr


def decode_output(arr: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    return np.argmax(arr[0, :, : out_shape[0], : out_shape[1]], axis=0).astype(np.int64)


def verify_onnx(path: Path, examples: list[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    failures: list[str] = []
    for split, idx, inp, expected in examples:
        pred = sess.run([OUT_NAME], {IN_NAME: one_hot_input(inp)})[0]
        decoded = decode_output(pred, expected.shape)
        valid_region = pred[0, :, : expected.shape[0], : expected.shape[1]] > 0.0
        active = np.sum(valid_region, axis=0)
        ok = np.array_equal(decoded, expected) and np.all(active == 1)
        if not ok:
            failures.append(f"{split}[{idx}]")
    if failures:
        raise AssertionError("ONNX verification failed: " + ", ".join(failures[:10]))


def print_candidate_report(results: list[dict]) -> None:
    print("Candidate agreement on train examples")
    for item in results:
        cand = item["candidate"]
        exact = item["exact"]
        total = item["total"]
        cells_ok = item["cells_ok"]
        cells_total = item["cells_total"]
        print(
            f"- {cand.name:<28} family={cand.family:<33} "
            f"exact={exact}/{total} cells={cells_ok}/{cells_total} op_est={cand.op_estimate}"
        )
        print("  " + ", ".join(item["per_example"]))


def main() -> None:
    data = load_task()
    train = all_examples(data, ("train",))
    all_rows = all_examples(data)
    candidates = candidate_solvers()
    results = [evaluate_candidate(c, train) for c in candidates]
    results.sort(
        key=lambda item: (
            -int(item["exact"]),
            -int(item["cells_ok"]),
            int(item["candidate"].op_estimate),
            item["candidate"].name,
        )
    )
    print_candidate_report(results)
    best = results[0]["candidate"]
    print(f"\nSelected rule: {best.name} ({best.family})")
    print(
        "Alternative outcome: 2x2 block voting returns cyan/block colors instead "
        "of marker colors. Several quadrant-based variants tie on train because "
        "the central non-black cells are exactly cyan; the selected rule is the "
        "smallest direct statement of that pattern."
    )

    all_eval = evaluate_candidate(best, all_rows)
    print(
        f"Best rule validation on all JSON examples: "
        f"exact={all_eval['exact']}/{all_eval['total']} "
        f"cells={all_eval['cells_ok']}/{all_eval['cells_total']}"
    )
    if all_eval["exact"] != all_eval["total"]:
        raise SystemExit("Best Python rule did not validate against all examples")

    model = build_onnx_model()
    onnx.checker.check_model(model, full_check=True)
    BEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, BEST_PATH)
    verify_onnx(BEST_PATH, all_rows)
    report = score_file(BEST_PATH)
    print("\nONNX score_model report")
    print(f"- path: {BEST_PATH}")
    print(f"- valid: {report['valid']}")
    print(f"- filesize: {report['filesize']}")
    print(f"- memory: {report['memory']}")
    print(f"- params: {report['params']}")
    print(f"- cost: {report['cost']}")
    print(f"- score: {report['score']}")
    if report.get("error"):
        print(f"- error: {str(report['error']).strip()}")


if __name__ == "__main__":
    main()
