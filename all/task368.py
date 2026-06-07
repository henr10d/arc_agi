"""ONNX solution for ARC task368: copy the colored prototype into gray blocks.

Task rule: the 10x10 input contains one non-background, non-gray colored
prototype rectangle and two or three solid gray rectangles of the same shape.
The output keeps the prototype and background unchanged, while every gray
rectangle is replaced by an identical, same-orientation copy of the prototype.
Color 0 is background and color 5 is the gray placeholder.

The graph detects exact 3x3, 3x4, and 4x3 foreground/gray rectangles, extracts
the unique colored prototype crop for the active shape, pads copies of that crop
to every gray target anchor, and uses those copies wherever the input is gray.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TASK_ID = "task368"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task368.onnx"
DATA_PATH = ROOT / "data" / "task368.json"

C = 10
H = W = 30
GRID = 10
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10
GRAY = 5
BACKGROUND = 0
RECT_SHAPES = ((3, 3), (3, 4), (4, 3))


def solve(grid: Sequence[Sequence[int]]) -> np.ndarray:
    """Reference implementation for local checking."""
    x = np.asarray(grid, dtype=np.int64)
    out = x.copy()
    proto_cells = np.argwhere((x != BACKGROUND) & (x != GRAY))
    pr0, pc0 = proto_cells.min(axis=0)
    pr1, pc1 = proto_cells.max(axis=0)
    pattern = x[pr0 : pr1 + 1, pc0 : pc1 + 1]

    seen = np.zeros_like(x, dtype=bool)
    for r, c in np.argwhere(x == GRAY):
        if seen[r, c]:
            continue
        stack = [(int(r), int(c))]
        seen[r, c] = True
        cells: List[Tuple[int, int]] = []
        for rr, cc in stack:
            cells.append((rr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < x.shape[0] and 0 <= nc < x.shape[1]:
                    if x[nr, nc] == GRAY and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
        rows = [rr for rr, _ in cells]
        cols = [cc for _, cc in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        out[r0 : r1 + 1, c0 : c1 + 1] = pattern
    return out


class GraphBuilder:
    def __init__(self) -> None:
        self.nodes: List[onnx.NodeProto] = []
        self.inits: List[onnx.TensorProto] = []
        self._counter = 0
        self._i64_cache: Dict[Tuple[int, ...], str] = {}
        self._f32_cache: Dict[Tuple[float, ...], str] = {}
        self.axes = self.i64([0, 1, 2, 3])
        self.zero = self.f32([0.0])

    def name(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def i64(self, vals: Iterable[int]) -> str:
        key = tuple(int(v) for v in vals)
        if key not in self._i64_cache:
            name = self.name("i")
            self.inits.append(numpy_helper.from_array(np.asarray(key, dtype=np.int64), name=name))
            self._i64_cache[key] = name
        return self._i64_cache[key]

    def f32(self, vals: Iterable[float]) -> str:
        key = tuple(float(v) for v in vals)
        if key not in self._f32_cache:
            name = self.name("f")
            self.inits.append(numpy_helper.from_array(np.asarray(key, dtype=np.float32), name=name))
            self._f32_cache[key] = name
        return self._f32_cache[key]

    def zero_tensor(self, shape: Sequence[int], name: str) -> str:
        self.inits.append(numpy_helper.from_array(np.zeros(shape, dtype=np.float32), name=name))
        return name

    def zero_bool_tensor(self, shape: Sequence[int], name: str) -> str:
        self.inits.append(numpy_helper.from_array(np.zeros(shape, dtype=bool), name=name))
        return name

    def slice4(self, x: str, starts: Sequence[int], ends: Sequence[int], prefix: str) -> str:
        y = self.name(prefix)
        self.nodes.append(helper.make_node("Slice", [x, self.i64(starts), self.i64(ends), self.axes], [y]))
        return y

    def reduce_hw(self, x: str, prefix: str) -> str:
        y = self.name(prefix)
        self.nodes.append(helper.make_node("ReduceSum", [x], [y], axes=[2, 3], keepdims=1))
        return y

    def add(self, a: str, b: str, prefix: str) -> str:
        y = self.name(prefix)
        self.nodes.append(helper.make_node("Add", [a, b], [y]))
        return y

    def mul(self, a: str, b: str, prefix: str) -> str:
        y = self.name(prefix)
        self.nodes.append(helper.make_node("Mul", [a, b], [y]))
        return y

    def greater_scalar(self, x: str, value: float, prefix: str) -> str:
        y = self.name(prefix)
        self.nodes.append(helper.make_node("Greater", [x, self.f32([value])], [y]))
        return y

    def less_scalar(self, x: str, value: float, prefix: str) -> str:
        y = self.name(prefix)
        self.nodes.append(helper.make_node("Less", [x, self.f32([value])], [y]))
        return y

    def and2(self, a: str, b: str, prefix: str) -> str:
        y = self.name(prefix)
        self.nodes.append(helper.make_node("And", [a, b], [y]))
        return y

    def or2(self, a: str, b: str, prefix: str) -> str:
        y = self.name(prefix)
        self.nodes.append(helper.make_node("Or", [a, b], [y]))
        return y

    def not1(self, x: str, prefix: str) -> str:
        y = self.name(prefix)
        self.nodes.append(helper.make_node("Not", [x], [y]))
        return y

    def cast_float(self, x: str, prefix: str) -> str:
        y = self.name(prefix)
        self.nodes.append(helper.make_node("Cast", [x], [y], to=TensorProto.FLOAT))
        return y

    def pad_to_full(self, x: str, r: int, c: int, h: int, w: int, prefix: str) -> str:
        y = self.name(prefix)
        pads = [0, 0, r, c, 0, 0, H - r - h, W - c - w]
        self.nodes.append(helper.make_node("Pad", [x], [y], pads=pads))
        return y


def _region_sum(g: GraphBuilder, mask: str, r: int, c: int, h: int, w: int, prefix: str) -> str:
    region = g.slice4(mask, [0, 0, r, c], [1, 1, r + h, c + w], prefix + "_sl")
    return g.reduce_hw(region, prefix + "_sum")


def _exact_rect_condition(g: GraphBuilder, mask: str, r: int, c: int, h: int, w: int, prefix: str) -> str:
    inside = _region_sum(g, mask, r, c, h, w, prefix + "_in")
    cond = g.greater_scalar(inside, float(h * w) - 0.5, prefix + "_full")

    boundaries: List[Tuple[int, int, int, int]] = []
    if r > 0:
        boundaries.append((r - 1, c, 1, w))
    if r + h < GRID:
        boundaries.append((r + h, c, 1, w))
    if c > 0:
        boundaries.append((r, c - 1, h, 1))
    if c + w < GRID:
        boundaries.append((r, c + w, h, 1))

    for idx, (br, bc, bh, bw) in enumerate(boundaries):
        edge_sum = _region_sum(g, mask, br, bc, bh, bw, f"{prefix}_b{idx}")
        edge_empty = g.less_scalar(edge_sum, 0.5, f"{prefix}_e{idx}")
        cond = g.and2(cond, edge_empty, f"{prefix}_and{idx}")
    return cond


def build_onnx_model() -> onnx.ModelProto:
    g = GraphBuilder()

    x10 = g.slice4(IN_NAME, [0, 0, 0, 0], [1, C, GRID, GRID], "x10")

    # Masks with shape [1, 1, 10, 10].
    active = g.name("active")
    g.nodes.append(helper.make_node("ReduceSum", [x10], [active], axes=[1], keepdims=1))
    bg = g.slice4(x10, [0, BACKGROUND, 0, 0], [1, BACKGROUND + 1, GRID, GRID], "bg")
    gray = g.slice4(x10, [0, GRAY, 0, 0], [1, GRAY + 1, GRID, GRID], "gray")
    bg_or_gray = g.add(bg, gray, "bg_gray")
    fg_score = g.name("fg_score")
    g.nodes.append(helper.make_node("Sub", [active, bg_or_gray], [fg_score]))
    fg_bool = g.name("fg_bool")
    g.nodes.append(helper.make_node("Greater", [fg_score, g.zero], [fg_bool]))
    gray_bool = g.name("gray_bool")
    g.nodes.append(helper.make_node("Greater", [gray, g.zero], [gray_bool]))
    x_bool = g.name("x_bool")
    g.nodes.append(helper.make_node("Greater", [x10, g.zero], [x_bool]))
    fg = g.cast_float(fg_bool, "fg")
    gray_mask = g.cast_float(gray_bool, "grayf")

    target_weights: Dict[Tuple[int, int, int, int], str] = {}
    pattern_cells: Dict[Tuple[int, int, int, int], str] = {}

    for h, w in RECT_SHAPES:
        pattern = g.zero_bool_tensor([1, C, h, w], f"pat0_{h}_{w}")

        for r in range(GRID - h + 1):
            for c in range(GRID - w + 1):
                proto_cond = _exact_rect_condition(g, fg, r, c, h, w, f"p_{h}_{w}_{r}_{c}")
                crop = g.slice4(x_bool, [0, 0, r, c], [1, C, r + h, c + w], f"pc_{h}_{w}_{r}_{c}")
                weighted_crop = g.and2(crop, proto_cond, f"pm_{h}_{w}_{r}_{c}")
                pattern = g.or2(pattern, weighted_crop, f"pa_{h}_{w}_{r}_{c}")

        for dr in range(h):
            for dc in range(w):
                pattern_cells[(h, w, dr, dc)] = g.slice4(
                    pattern, [0, 0, dr, dc], [1, C, dr + 1, dc + 1], f"ps_{h}_{w}_{dr}_{dc}"
                )

        for r in range(GRID - h + 1):
            for c in range(GRID - w + 1):
                target_cond = _exact_rect_condition(g, gray_mask, r, c, h, w, f"t_{h}_{w}_{r}_{c}")
                target_weights[(h, w, r, c)] = target_cond

    zero_cell = g.zero_bool_tensor([1, C, 1, 1], "zero_cell")
    rows: List[str] = []
    for r in range(GRID):
        cells: List[str] = []
        for c in range(GRID):
            replacement_cell = zero_cell
            for h, w in RECT_SHAPES:
                for dr in range(h):
                    ar = r - dr
                    if ar < 0 or ar > GRID - h:
                        continue
                    for dc in range(w):
                        ac = c - dc
                        if ac < 0 or ac > GRID - w:
                            continue
                        weighted = g.and2(
                            pattern_cells[(h, w, dr, dc)],
                            target_weights[(h, w, ar, ac)],
                            f"cm_{r}_{c}_{h}_{w}_{dr}_{dc}",
                        )
                        replacement_cell = g.or2(replacement_cell, weighted, f"ca_{r}_{c}_{h}_{w}_{dr}_{dc}")

            input_cell = g.slice4(x_bool, [0, 0, r, c], [1, C, r + 1, c + 1], f"ic_{r}_{c}")
            gray_cell = g.slice4(gray_bool, [0, 0, r, c], [1, 1, r + 1, c + 1], f"gc_{r}_{c}")
            kept_cell = g.and2(g.not1(gray_cell, f"ng_{r}_{c}"), input_cell, f"kc_{r}_{c}")
            copied_cell = g.and2(gray_cell, replacement_cell, f"rc_{r}_{c}")
            out_cell = g.or2(kept_cell, copied_cell, f"oc_{r}_{c}")
            cells.append(out_cell)

        row = g.name(f"row_{r}")
        g.nodes.append(helper.make_node("Concat", cells, [row], axis=3))
        rows.append(row)

    grid10 = g.name("grid10")
    g.nodes.append(helper.make_node("Concat", rows, [grid10], axis=2))
    grid10f = g.cast_float(grid10, "grid10f")
    g.nodes.append(helper.make_node("Pad", [grid10f], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - GRID, W - GRID]))

    graph = helper.make_graph(
        g.nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        initializer=g.inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf-task368",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _grid_to_onehot(grid: Sequence[Sequence[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _onehot_to_grid(y: np.ndarray) -> np.ndarray:
    return y.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def save_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    model = build_onnx_model()
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    return model


def test() -> None:
    model = save_model()
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    for split in ("train", "test", "arc-gen"):
        bad = 0
        for ex in data[split]:
            inp = np.asarray(ex["input"], dtype=np.int64)
            expected = np.asarray(ex["output"], dtype=np.int64)
            ref = solve(inp)
            if not np.array_equal(ref, expected):
                raise SystemExit(f"reference mismatch in {split}")
            pred = _onehot_to_grid(sess.run(None, {IN_NAME: _grid_to_onehot(inp)})[0])
            if not np.array_equal(pred[: expected.shape[0], : expected.shape[1]], expected):
                bad += 1
        print(f"{split}: {'PASS' if bad == 0 else f'FAIL ({bad})'}")
        if bad:
            raise SystemExit(1)


def main() -> None:
    save_model()
    print(f"saved {BEST_PATH}")


if __name__ == "__main__":
    main()
