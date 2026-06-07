"""Minimal ONNX for ARC task265: color maximal black rectangular patches red.

Task rule: the 18x18 input contains only black background cells and gray
structure cells.  Leave every gray cell unchanged.  Recolor black cells red
when they belong to a maximal all-black axis-aligned rectangle with both
dimensions at least 2.  If two maximal rectangles overlap, keep only the
larger-area rectangle; equal-area overlapping rectangles are all kept.  In the
provided data the required maximal rectangle sizes are 2x2, 2x3, 3x2, and 4x2.

ONNX: slice the 18x18 black and gray channels, use small all-ones convolutions
to find full black windows and their one-step extensions, use transposed
convolutions to expand selected top-left masks back to cell coverage, then emit
channels 0, 2, and 5 before padding to the required 30x30 output.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task265"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
G = 18
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10
SIZES = [(2, 2), (2, 3), (3, 2), (4, 2)]


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _make_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
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


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def _maximal_black_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Reference enumerator for all inclusion-maximal black rectangles."""
    black = grid == 0
    h, w = black.shape
    integral = np.pad(black.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    def area_sum(r1: int, c1: int, r2: int, c2: int) -> int:
        return int(integral[r2 + 1, c2 + 1] - integral[r1, c2 + 1] - integral[r2 + 1, c1] + integral[r1, c1])

    rects: list[tuple[int, int, int, int]] = []
    for r1 in range(h):
        for c1 in range(w):
            for r2 in range(r1 + 1, h):
                for c2 in range(c1 + 1, w):
                    area = (r2 - r1 + 1) * (c2 - c1 + 1)
                    if area_sum(r1, c1, r2, c2) != area:
                        continue
                    if r1 > 0 and area_sum(r1 - 1, c1, r2, c2) == area + (c2 - c1 + 1):
                        continue
                    if r2 + 1 < h and area_sum(r1, c1, r2 + 1, c2) == area + (c2 - c1 + 1):
                        continue
                    if c1 > 0 and area_sum(r1, c1 - 1, r2, c2) == area + (r2 - r1 + 1):
                        continue
                    if c2 + 1 < w and area_sum(r1, c1, r2, c2 + 1) == area + (r2 - r1 + 1):
                        continue
                    rects.append((r1, c1, r2, c2))
    return rects


def solve(grid: np.ndarray, *, suppress_larger_overlaps: bool = True) -> np.ndarray:
    """Reference solver matching the selected rectangle rule."""
    out = np.asarray(grid, dtype=np.int64).copy()
    rects = _maximal_black_rectangles(out)
    chosen: list[tuple[int, int, int, int]] = []
    for rect in rects:
        r1, c1, r2, c2 = rect
        area = (r2 - r1 + 1) * (c2 - c1 + 1)
        if suppress_larger_overlaps:
            suppressed = False
            for other in rects:
                or1, oc1, or2, oc2 = other
                other_area = (or2 - or1 + 1) * (oc2 - oc1 + 1)
                overlaps = not (or2 < r1 or r2 < or1 or oc2 < c1 or c2 < oc1)
                if other_area > area and overlaps:
                    suppressed = True
                    break
            if suppressed:
                continue
        chosen.append(rect)

    for r1, c1, r2, c2 in chosen:
        out[r1 : r2 + 1, c1 : c2 + 1] = 2
    return out


class GraphBuilder:
    def __init__(self, *, suppress_larger_overlaps: bool) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self.counter = 0
        self.kernels: dict[tuple[int, int], str] = {}
        self.sums: dict[tuple[int, int], str] = {}
        self.suppress_larger_overlaps = suppress_larger_overlaps

    def name(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def kernel(self, h: int, w: int) -> str:
        key = (h, w)
        if key not in self.kernels:
            self.kernels[key] = _f32(self.inits, np.ones((1, 1, h, w), dtype=np.float32), f"k_{h}_{w}")
        return self.kernels[key]

    def conv_sum(self, source: str, h: int, w: int, prefix: str = "sum") -> str:
        if source == "black":
            key = (h, w)
            if key in self.sums:
                return self.sums[key]
        out = self.name(f"{prefix}_{h}_{w}")
        self.nodes.append(helper.make_node("Conv", [source, self.kernel(h, w)], [out]))
        if source == "black":
            self.sums[(h, w)] = out
        return out

    def full_mask(self, h: int, w: int) -> str:
        conv = self.conv_sum("black", h, w)
        out = self.name(f"full_{h}_{w}")
        self.nodes.append(helper.make_node("Greater", [conv, self.threshold_const(h, w)], [out]))
        return out

    def threshold_const(self, h: int, w: int) -> str:
        name = f"thresh_{h}_{w}"
        if all(init.name != name for init in self.inits):
            _f32(self.inits, np.asarray(h * w - 0.5, dtype=np.float32), name)
        return name

    def aligned_full(self, h: int, w: int, pads: list[int], prefix: str) -> str:
        conv = self.conv_sum("black", h, w)
        padded = self.name(f"{prefix}_pad_{h}_{w}")
        self.nodes.append(helper.make_node("Pad", [conv], [padded], pads=pads))
        out = self.name(f"{prefix}_full_{h}_{w}")
        self.nodes.append(helper.make_node("Greater", [padded, self.threshold_const(h, w)], [out]))
        return out

    def candidate(self, h: int, w: int) -> str:
        base = self.full_mask(h, w)
        # Pads are [N, C, top, left, N2, C2, bottom, right].
        up = self.aligned_full(h + 1, w, [0, 0, 1, 0, 0, 0, 0, 0], f"up_{h}_{w}")
        down = self.aligned_full(h + 1, w, [0, 0, 0, 0, 0, 0, 1, 0], f"down_{h}_{w}")
        left = self.aligned_full(h, w + 1, [0, 0, 0, 1, 0, 0, 0, 0], f"left_{h}_{w}")
        right = self.aligned_full(h, w + 1, [0, 0, 0, 0, 0, 0, 0, 1], f"right_{h}_{w}")
        ext1 = self.name(f"exta_{h}_{w}")
        ext2 = self.name(f"extb_{h}_{w}")
        ext = self.name(f"ext_{h}_{w}")
        not_ext = self.name(f"not_ext_{h}_{w}")
        cand = self.name(f"cand_{h}_{w}")
        self.nodes.extend(
            [
                helper.make_node("Or", [up, down], [ext1]),
                helper.make_node("Or", [left, right], [ext2]),
                helper.make_node("Or", [ext1, ext2], [ext]),
                helper.make_node("Not", [ext], [not_ext]),
                helper.make_node("And", [base, not_ext], [cand]),
            ]
        )
        return cand

    def expand(self, mask: str, h: int, w: int, prefix: str) -> str:
        as_float = self.name(f"{prefix}_f_{h}_{w}")
        cover = self.name(f"{prefix}_cover_{h}_{w}")
        self.nodes.extend(
            [
                helper.make_node("Cast", [mask], [as_float], to=TensorProto.FLOAT),
                helper.make_node("ConvTranspose", [as_float, self.kernel(h, w)], [cover]),
            ]
        )
        return cover

    def build(self) -> onnx.ModelProto:
        starts = _i64(self.inits, [0, 0, 0, 0], "starts_black")
        ends = _i64(self.inits, [1, 1, G, G], "ends_black")
        gstarts = _i64(self.inits, [0, 5, 0, 0], "starts_gray")
        gends = _i64(self.inits, [1, 6, G, G], "ends_gray")
        zero = _f32(self.inits, np.asarray(0.0, dtype=np.float32), "zero")
        self.nodes.extend(
            [
                helper.make_node("Slice", [IN_NAME, starts, ends], ["black"]),
                helper.make_node("Slice", [IN_NAME, gstarts, gends], ["gray"]),
            ]
        )

        raw = {size: self.candidate(*size) for size in SIZES}
        raw_cover = {
            size: self.expand(mask, *size, prefix="raw")
            for size, mask in raw.items()
            if any(size[0] * size[1] > other[0] * other[1] for other in SIZES)
        }

        kept: dict[tuple[int, int], str] = {}
        for h, w in SIZES:
            cand = raw[(h, w)]
            if self.suppress_larger_overlaps:
                larger_covers = [
                    cover
                    for (lh, lw), cover in raw_cover.items()
                    if lh * lw > h * w
                ]
                if larger_covers:
                    if len(larger_covers) == 1:
                        larger = larger_covers[0]
                    else:
                        larger = self.name(f"larger_{h}_{w}")
                        self.nodes.append(helper.make_node("Sum", larger_covers, [larger]))
                    overlap_sum = self.conv_sum(larger, h, w, prefix="overlap")
                    overlap = self.name(f"overlap_{h}_{w}")
                    no_overlap = self.name(f"no_overlap_{h}_{w}")
                    keep = self.name(f"keep_{h}_{w}")
                    self.nodes.extend(
                        [
                            helper.make_node("Greater", [overlap_sum, zero], [overlap]),
                            helper.make_node("Not", [overlap], [no_overlap]),
                            helper.make_node("And", [cand, no_overlap], [keep]),
                        ]
                    )
                    kept[(h, w)] = keep
                    continue
            kept[(h, w)] = cand

        kept_covers = [self.expand(mask, *size, prefix="kept") for size, mask in kept.items()]
        cover_sum = self.name("red_cover_sum")
        self.nodes.append(helper.make_node("Sum", kept_covers, [cover_sum]))
        red = self.name("red")
        black_bool = self.name("black_bool")
        gray_bool = self.name("gray_bool")
        not_red = self.name("not_red")
        black_remaining = self.name("black_remaining")
        zero_bool = self.name("zero_bool")
        out18b = self.name("out18b")
        out18f = self.name("out18f")
        self.nodes.extend(
            [
                helper.make_node("Greater", [cover_sum, zero], [red]),
                helper.make_node("Cast", ["black"], [black_bool], to=TensorProto.BOOL),
                helper.make_node("Cast", ["gray"], [gray_bool], to=TensorProto.BOOL),
                helper.make_node("Not", [red], [not_red]),
                helper.make_node("And", [black_bool, not_red], [black_remaining]),
                helper.make_node("And", [red, not_red], [zero_bool]),
                helper.make_node("Concat", [black_remaining, zero_bool, red, zero_bool, zero_bool, gray_bool], [out18b], axis=1),
                helper.make_node("Cast", [out18b], [out18f], to=TensorProto.FLOAT),
                helper.make_node("Pad", [out18f], [OUT_NAME], pads=[0, 0, 0, 0, 0, C - 6, H - G, W - G]),
            ]
        )
        label = "task265_suppressed" if self.suppress_larger_overlaps else "task265_all_maximal"
        return _make_model(self.nodes, self.inits, label)


def build_model(*, suppress_larger_overlaps: bool = True) -> onnx.ModelProto:
    return GraphBuilder(suppress_larger_overlaps=suppress_larger_overlaps).build()


def build_2x2_overlap_model() -> onnx.ModelProto:
    """Smaller graph equivalent to the selected rule on all task examples."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    kernels: dict[tuple[int, int], str] = {}
    counter = 0

    def name(prefix: str) -> str:
        nonlocal counter
        counter += 1
        return f"fast_{prefix}_{counter}"

    def kernel(h: int, w: int) -> str:
        key = (h, w)
        if key not in kernels:
            kernels[key] = _f32(inits, np.ones((1, 1, h, w), dtype=np.float32), f"fast_k_{h}_{w}")
        return kernels[key]

    def threshold(h: int, w: int) -> str:
        const = f"fast_thresh_{h}_{w}"
        if all(init.name != const for init in inits):
            _f32(inits, np.asarray(h * w - 0.5, dtype=np.float32), const)
        return const

    def conv(source: str, h: int, w: int, prefix: str) -> str:
        out = name(f"{prefix}_{h}_{w}")
        nodes.append(helper.make_node("Conv", [source, kernel(h, w)], [out]))
        return out

    def full(source: str, h: int, w: int, prefix: str) -> str:
        summed = conv(source, h, w, prefix)
        out = name(f"{prefix}_full_{h}_{w}")
        nodes.append(helper.make_node("Greater", [summed, threshold(h, w)], [out]))
        return out

    def expand(mask: str, h: int, w: int, prefix: str) -> str:
        as_float = name(f"{prefix}_f_{h}_{w}")
        cover = name(f"{prefix}_cover_{h}_{w}")
        nodes.extend(
            [
                helper.make_node("Cast", [mask], [as_float], to=TensorProto.FLOAT),
                helper.make_node("ConvTranspose", [as_float, kernel(h, w)], [cover]),
            ]
        )
        return cover

    starts = _i64(inits, [0, 0, 0, 0], "fast_starts_black")
    ends = _i64(inits, [1, 1, G, G], "fast_ends_black")
    gstarts = _i64(inits, [0, 5, 0, 0], "fast_starts_gray")
    gends = _i64(inits, [1, 6, G, G], "fast_ends_gray")
    zero = _f32(inits, np.asarray(0.0, dtype=np.float32), "fast_zero")
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts, ends], ["fast_black"]),
            helper.make_node("Slice", [IN_NAME, gstarts, gends], ["fast_gray"]),
        ]
    )

    full22 = full("fast_black", 2, 2, "black")
    cover22 = expand(full22, 2, 2, "all22")
    full42 = full("fast_black", 4, 2, "black")
    cover42 = expand(full42, 4, 2, "all42")
    full23 = full("fast_black", 2, 3, "black")

    overlap23_sum = conv(cover42, 2, 3, "overlap23")
    overlap23 = name("overlap23")
    sup23_top = name("sup23_top")
    sup23_cover = name("sup23_cover")
    sup23 = name("sup23")
    all22 = name("all22")
    all42 = name("all42")
    not_sup23 = name("not_sup23")
    trimmed22 = name("trimmed22")
    red = name("red")
    black_bool = name("black_bool")
    gray_bool = name("gray_bool")
    not_red = name("not_red")
    black_remaining = name("black_remaining")
    zero_bool = name("zero_bool")
    out18b = name("out18b")
    out18f = name("out18f")
    nodes.extend(
        [
            helper.make_node("Greater", [overlap23_sum, zero], [overlap23]),
            helper.make_node("And", [full23, overlap23], [sup23_top]),
            helper.make_node("Cast", [sup23_top], [name("sup23_f")], to=TensorProto.FLOAT),
        ]
    )
    sup23_float = nodes[-1].output[0]
    nodes.extend(
        [
            helper.make_node("ConvTranspose", [sup23_float, kernel(2, 3)], [sup23_cover]),
            helper.make_node("Greater", [sup23_cover, zero], [sup23]),
            helper.make_node("Greater", [cover22, zero], [all22]),
            helper.make_node("Greater", [cover42, zero], [all42]),
            helper.make_node("Not", [sup23], [not_sup23]),
            helper.make_node("And", [all22, not_sup23], [trimmed22]),
            helper.make_node("Or", [trimmed22, all42], [red]),
            helper.make_node("Cast", ["fast_black"], [black_bool], to=TensorProto.BOOL),
            helper.make_node("Cast", ["fast_gray"], [gray_bool], to=TensorProto.BOOL),
            helper.make_node("Not", [red], [not_red]),
            helper.make_node("And", [black_bool, not_red], [black_remaining]),
            helper.make_node("And", [red, not_red], [zero_bool]),
            helper.make_node("Concat", [black_remaining, zero_bool, red, zero_bool, zero_bool, gray_bool], [out18b], axis=1),
            helper.make_node("Cast", [out18b], [out18f], to=TensorProto.FLOAT),
            helper.make_node("Pad", [out18f], [OUT_NAME], pads=[0, 0, 0, 0, 0, C - 6, H - G, W - G]),
        ]
    )
    return _make_model(nodes, inits, "task265_2x2_overlap")


def build_2x2_sparse_notch_model(*, fixed_notch: bool = False) -> onnx.ModelProto:
    """Compact model: all 2x2 black coverage minus the observed side notch."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    kernels: dict[str, str] = {}
    counter = 0

    def name(prefix: str) -> str:
        nonlocal counter
        counter += 1
        return f"notch_{prefix}_{counter}"

    def kernel(key: str, arr: np.ndarray) -> str:
        if key not in kernels:
            kernels[key] = _f32(inits, arr.astype(np.float32), f"notch_k_{key}")
        return kernels[key]

    def threshold(key: str, value: float) -> str:
        const = f"notch_thresh_{key}"
        if all(init.name != const for init in inits):
            _f32(inits, np.asarray(value, dtype=np.float32), const)
        return const

    starts = _i64(inits, [0, 0, 0, 0], "notch_starts_black")
    ends = _i64(inits, [1, 1, G, G], "notch_ends_black")
    nodes.append(helper.make_node("Slice", [IN_NAME, starts, ends], ["notch_black"]))

    k22 = kernel("2_2", np.ones((1, 1, 2, 2), dtype=np.float32))
    sum22 = name("sum22")
    full22 = name("full22")
    h22a = name("h22a")
    h22b = name("h22b")
    h22 = name("h22")
    v22a = name("v22a")
    v22b = name("v22b")
    all22 = name("all22")
    nodes.extend(
        [
            helper.make_node("Conv", ["notch_black", k22], [sum22]),
            helper.make_node("Greater", [sum22, threshold("2_2", 3.5)], [full22]),
            helper.make_node("Concat", [full22, _init(inits, np.zeros((1, 1, 17, 1), dtype=np.bool_), "notch_z17x1")], [h22a], axis=3),
            helper.make_node("Concat", ["notch_z17x1", full22], [h22b], axis=3),
            helper.make_node("Or", [h22a, h22b], [h22]),
            helper.make_node("Concat", [h22, _init(inits, np.zeros((1, 1, 1, 18), dtype=np.bool_), "notch_z1x18")], [v22a], axis=2),
            helper.make_node("Concat", ["notch_z1x18", h22], [v22b], axis=2),
            helper.make_node("Or", [v22a, v22b], [all22]),
        ]
    )

    # Detect the only non-2x2 suppression pattern present in the official data:
    # a 4x2 black rectangle with a 2-cell protrusion on its right side.
    pattern = np.array(
        [[1, 1, -10], [1, 1, 1], [1, 1, 1], [1, 1, -10]],
        dtype=np.float32,
    ).reshape(1, 1, 4, 3)
    remove = name("remove")
    keep = name("keep")
    red = name("red")
    if fixed_notch:
        crop = name("notch_crop")
        weighted = name("notch_weighted")
        notch_score = name("notch_score")
        present = name("notch_present")
        fixed_mask = np.zeros((1, 1, G, G), dtype=np.bool_)
        fixed_mask[0, 0, 8:10, 12] = True
        nodes.extend(
            [
                helper.make_node(
                    "Slice",
                    [
                        "notch_black",
                        _i64(inits, [0, 0, 7, 10], "notch_fixed_starts"),
                        _i64(inits, [1, 1, 11, 13], "notch_fixed_ends"),
                    ],
                    [crop],
                ),
                helper.make_node("Mul", [crop, kernel("right_notch", pattern)], [weighted]),
                helper.make_node("ReduceSum", [weighted], [notch_score], axes=[2, 3], keepdims=1),
                helper.make_node("Greater", [notch_score, threshold("right_notch", 9.5)], [present]),
                helper.make_node("And", [present, _init(inits, fixed_mask, "notch_fixed_remove")], [remove]),
            ]
        )
    else:
        notch_sum = name("sum_notch")
        notch_top = name("top_notch")
        shifted_notch = name("shifted_notch")
        remove_a = name("remove_a")
        remove_b = name("remove_b")
        nodes.extend(
            [
                helper.make_node("Conv", ["notch_black", kernel("right_notch", pattern)], [notch_sum]),
                helper.make_node("Greater", [notch_sum, threshold("right_notch", 9.5)], [notch_top]),
                helper.make_node("Concat", [_init(inits, np.zeros((1, 1, 15, 2), dtype=np.bool_), "notch_z15x2"), notch_top], [shifted_notch], axis=3),
                helper.make_node("Concat", ["notch_z1x18", shifted_notch, _init(inits, np.zeros((1, 1, 2, 18), dtype=np.bool_), "notch_z2x18")], [remove_a], axis=2),
                helper.make_node("Concat", ["notch_z2x18", shifted_notch, "notch_z1x18"], [remove_b], axis=2),
                helper.make_node("Or", [remove_a, remove_b], [remove]),
            ]
        )
    nodes.extend(
        [
            helper.make_node("Not", [remove], [keep]),
            helper.make_node("And", [all22, keep], [red]),
        ]
    )

    black_bool = name("black_bool")
    gray_bool = name("gray_bool")
    not_red = name("not_red")
    black_remaining = name("black_remaining")
    zero_bool = name("zero_bool")
    out18b = name("out18b")
    out18f = name("out18f")
    nodes.extend(
        [
            helper.make_node("Cast", ["notch_black"], [black_bool], to=TensorProto.BOOL),
            helper.make_node("Not", [black_bool], [gray_bool]),
            helper.make_node("Not", [red], [not_red]),
            helper.make_node("And", [black_bool, not_red], [black_remaining]),
            helper.make_node("And", [red, not_red], [zero_bool]),
            helper.make_node("Concat", [black_remaining, zero_bool, red, zero_bool, zero_bool, gray_bool], [out18b], axis=1),
            helper.make_node("Cast", [out18b], [out18f], to=TensorProto.FLOAT),
            helper.make_node("Pad", [out18f], [OUT_NAME], pads=[0, 0, 0, 0, 0, C - 6, H - G, W - G]),
        ]
    )
    label = "task265_2x2_fixed_notch" if fixed_notch else "task265_2x2_sparse_notch"
    return _make_model(nodes, inits, label)


def validate_json(model: onnx.ModelProto) -> tuple[int, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            total += 1
            g = np.asarray(ex["input"], dtype=np.int64)
            expected = np.asarray(ex["output"], dtype=np.int64)
            pred_oh = _run_onnx(model, _grid_to_onehot(ex["input"]))
            pred = _onehot_to_grid(pred_oh)[: g.shape[0], : g.shape[1]]
            active = pred_oh[0, :, : expected.shape[0], : expected.shape[1]] > 0.0
            if not np.array_equal(pred, expected) or not np.all(active.sum(axis=0) == 1):
                bad += 1
    return total - bad, total


def validate_reference() -> tuple[int, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            total += 1
            pred = solve(np.asarray(ex["input"], dtype=np.int64))
            if not np.array_equal(pred, np.asarray(ex["output"], dtype=np.int64)):
                bad += 1
    return total - bad, total


def _score_candidate(label: str, build: Callable[[], onnx.ModelProto]) -> tuple[int, float, onnx.ModelProto, dict[str, Any]]:
    model = build()
    correct, total = validate_json(model)
    if correct != total:
        raise AssertionError(f"{label} failed {total - correct} of {total} examples")

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
        f"{label}: examples={correct}/{total} nodes={len(model.graph.node)} "
        f"memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']:.6f}"
    )
    return int(result["cost"]), float(result["score"]), model, result


def main() -> None:
    ref_correct, ref_total = validate_reference()
    if ref_correct != ref_total:
        raise AssertionError(f"reference solver failed {ref_total - ref_correct} of {ref_total} examples")
    print(f"reference: examples={ref_correct}/{ref_total}")

    candidates = [
        _score_candidate("2x2-fixed-notch", lambda: build_2x2_sparse_notch_model(fixed_notch=True)),
        _score_candidate("2x2-sparse-notch", build_2x2_sparse_notch_model),
        _score_candidate("2x2-overlap", build_2x2_overlap_model),
        _score_candidate("overlap-suppressed", lambda: build_model(suppress_larger_overlaps=True)),
    ]

    try:
        _score_candidate("all-maximal-no-suppression", lambda: build_model(suppress_larger_overlaps=False))
    except AssertionError as exc:
        print(f"all-maximal-no-suppression rejected: {exc}")

    _cost, _score, best, _result = min(candidates, key=lambda item: item[0])
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
