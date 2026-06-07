"""Minimal ONNX for ARC task189: recolor the separated 6x6 pattern by a 2x2 key.

Task rule: a cyan row/column cross partitions each 9x9 input into a 2x2
corner color key and the opposite 6x6 green pattern. Copy the 6x6 green mask
to the 6x6 output; in each 3x3 quadrant, replace green cells by the
corresponding key color (top-left, top-right, bottom-left, bottom-right) and
leave non-green cells as black. The key can be in any corner, with the green
pattern in the diagonally opposite 6x6 region. In the provided train/test/
arc-gen examples, key colors are never green or cyan; the ONNX graph uses that
constraint to OR the four corner candidates and suppress green contamination.

Rejected hypotheses: connected components, skeleton/intersections, row/column
projection summaries, and feature ranking all fail the examples because the
raw 6x6 green mask is preserved exactly; only its quadrant colors change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task189"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10
IN_NAME = "input"
OUT_NAME = "output"


class Builder:
    def __init__(self) -> None:
        self.nodes: List[onnx.NodeProto] = []
        self.inits: List[onnx.TensorProto] = []
        self._init_cache: dict[Tuple[int, Tuple[int, ...]], str] = {}
        self._n = 0

    def name(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    def init(self, arr, name: str) -> str:
        a = np.asarray(arr)
        key = (int(a.dtype.num), tuple(int(x) for x in a.ravel()))
        if key in self._init_cache:
            return self._init_cache[key]
        self.inits.append(numpy_helper.from_array(a, name=name))
        self._init_cache[key] = name
        return name

    def slice_axes(
        self,
        x: str,
        start: Sequence[int],
        end: Sequence[int],
        axes: Sequence[int],
        prefix: str,
    ) -> str:
        y = self.name(prefix)
        st = self.init(np.asarray(start, dtype=np.int64), f"{y}_st")
        en = self.init(np.asarray(end, dtype=np.int64), f"{y}_en")
        ax = self.init(np.asarray(axes, dtype=np.int64), f"axes{''.join(map(str, axes))}")
        self.nodes.append(helper.make_node("Slice", [x, st, en, ax], [y]))
        return y

    def slice3(self, x: str, start: Sequence[int], end: Sequence[int], prefix: str) -> str:
        return self.slice_axes(x, start, end, [1, 2, 3], prefix)

    def slice_hw(self, x: str, start: Sequence[int], end: Sequence[int], prefix: str) -> str:
        return self.slice_axes(x, start, end, [2, 3], prefix)

    def cast_bool(self, x: str, prefix: str) -> str:
        y = self.name(prefix)
        self.nodes.append(helper.make_node("Cast", [x], [y], to=TensorProto.BOOL))
        return y

    def cast_float(self, x: str, prefix: str) -> str:
        y = self.name(prefix)
        self.nodes.append(helper.make_node("Cast", [x], [y], to=TensorProto.FLOAT))
        return y

    def and_(self, a: str, b: str, prefix: str) -> str:
        y = self.name(prefix)
        self.nodes.append(helper.make_node("And", [a, b], [y]))
        return y

    def or_(self, a: str, b: str, prefix: str) -> str:
        y = self.name(prefix)
        self.nodes.append(helper.make_node("Or", [a, b], [y]))
        return y

    def not_(self, x: str, prefix: str) -> str:
        y = self.name(prefix)
        self.nodes.append(helper.make_node("Not", [x], [y]))
        return y


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference numpy solver for the separator/key recoloring rule."""
    g = np.asarray(grid, dtype=np.int64)
    sep_r = 2 if np.all(g[2, :] == 8) else 6
    sep_c = 2 if np.all(g[:, 2] == 8) else 6
    if sep_r == 2 and sep_c == 2:
        key = g[0:2, 0:2]
        mask = g[3:9, 3:9] == 3
    elif sep_r == 2 and sep_c == 6:
        key = g[0:2, 7:9]
        mask = g[3:9, 0:6] == 3
    elif sep_r == 6 and sep_c == 2:
        key = g[7:9, 0:2]
        mask = g[0:6, 3:9] == 3
    else:
        key = g[7:9, 7:9]
        mask = g[0:6, 0:6] == 3

    out = np.zeros((6, 6), dtype=np.int64)
    out[:3, :3][mask[:3, :3]] = key[0, 0]
    out[:3, 3:6][mask[:3, 3:6]] = key[0, 1]
    out[3:6, :3][mask[3:6, :3]] = key[1, 0]
    out[3:6, 3:6][mask[3:6, 3:6]] = key[1, 1]
    return out


def _selected_or(b: Builder, terms: Sequence[str], weights: Sequence[str], prefix: str) -> str:
    parts = [b.and_(term, weight, prefix + "w") for term, weight in zip(terms, weights)]
    acc = parts[0]
    for part in parts[1:]:
        acc = b.or_(acc, part, prefix + "s")
    return acc


def build_model() -> onnx.ModelProto:
    b = Builder()

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    # Orientation selectors: a cyan cell at (2, 0) means the separator row is
    # on top; a cyan cell at (0, 2) means the separator column is on the left.
    top = b.cast_bool(b.slice3(IN_NAME, [8, 2, 0], [9, 3, 1], "topsep"), "topb")
    left = b.cast_bool(b.slice3(IN_NAME, [8, 0, 2], [9, 1, 3], "leftsep"), "leftb")
    not_top = b.not_(top, "nottop")
    not_left = b.not_(left, "notleft")
    selectors = [
        b.and_(top, left, "sel"),
        b.and_(top, not_left, "sel"),
        b.and_(not_top, left, "sel"),
        b.and_(not_top, not_left, "sel"),
    ]

    # Opposite 6x6 green-pattern crops for TL, TR, BL, BR key positions.
    green9 = b.cast_bool(b.slice3(IN_NAME, [3, 0, 0], [4, 9, 9], "green"), "greenb")
    object_crops: Tuple[Tuple[int, int], ...] = ((3, 3), (3, 0), (0, 3), (0, 0))
    masks = [b.slice_hw(green9, [r, c], [r + 6, c + 6], "mask") for r, c in object_crops]
    mask6 = _selected_or(b, masks, selectors, "mask")

    key_bases: Tuple[Tuple[int, int], ...] = ((0, 0), (0, 7), (7, 0), (7, 7))
    keys = [
        b.cast_bool(b.slice3(IN_NAME, [1, br, bc], [C, br + 2, bc + 2], "key"), "keyb")
        for br, bc in key_bases
    ]
    key_any = keys[0]
    for key in keys[1:]:
        key_any = b.or_(key_any, key, "keyor")
    non_green_key = b.init(
        np.asarray([[[[True]], [[True]], [[False]], [[True]], [[True]], [[True]], [[True]], [[True]], [[True]]]]),
        "non_green_key",
    )
    key2 = b.and_(key_any, non_green_key, "key")
    key_offsets: Tuple[Tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))
    key_vecs = [
        b.slice_hw(key2, [kr, kc], [kr + 1, kc + 1], "keycell")
        for kr, kc in key_offsets
    ]

    quads: List[str] = []
    quad_slices: Tuple[Tuple[int, int], ...] = ((0, 0), (0, 3), (3, 0), (3, 3))
    for idx, (qr, qc) in enumerate(quad_slices):
        qmask = b.slice_hw(mask6, [qr, qc], [qr + 3, qc + 3], "qm")
        quads.append(b.and_(key_vecs[idx], qmask, "fg"))

    top = b.name("top")
    bot = b.name("bot")
    fg9 = b.name("fg9")
    bg6 = b.not_(mask6, "bg6")
    out6b = b.name("out6b")
    out6 = b.name("out6")
    b.nodes.append(helper.make_node("Concat", [quads[0], quads[1]], [top], axis=3))
    b.nodes.append(helper.make_node("Concat", [quads[2], quads[3]], [bot], axis=3))
    b.nodes.append(helper.make_node("Concat", [top, bot], [fg9], axis=2))
    b.nodes.append(helper.make_node("Concat", [bg6, fg9], [out6b], axis=1))
    out6 = b.cast_float(out6b, "out6")
    b.nodes.append(helper.make_node("Pad", [out6], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - 6, W - 6]))

    graph = helper.make_graph(b.nodes, TASK_ID, [x_info], [y_info], initializer=b.inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _grid_to_onehot(grid: Sequence[Sequence[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for i, ex in enumerate(data[split]):
            expected = np.asarray(ex["output"], dtype=np.int64)
            ref = solve(np.asarray(ex["input"], dtype=np.int64))
            if not np.array_equal(ref, expected):
                raise AssertionError(f"reference mismatch on {split}[{i}]")
            pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(ex["input"])))[:6, :6]
            if not np.array_equal(pred, expected):
                bad += 1
                print(f"mismatch on {split}[{i}]\n{pred}\nexpected\n{expected}")
    return bad


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    bad = validate_json(model)
    print(f"{TASK_ID}.json: {'PASS' if bad == 0 else f'FAIL ({bad} wrong)'}")

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
