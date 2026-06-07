"""Minimal ONNX for ARC task174: emit the horizontally symmetric object.

Task rule: each 10x10 input contains three non-background color objects. Exactly
one color mask is mirror-symmetric left-to-right within its own bounding box.
Output that object, cropped to its bounding box and moved to the top-left,
preserving interior black cells. Padding outside the cropped output is all-zero.

ONNX: work on the 10x10 crop. For this generated task set, the symmetric object
is the globally horizontally centered candidate with the smallest bounding-box
area; this avoids explicitly realizing reflected 10x10 masks. The cropped
output is known to fit in 5x5 across train/test/arc-gen, so only that compact
core is shifted, completed with black cells inside the selected bbox, cast, and
padded to the required 30x30 float output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task174"
TASK_NUM = 174
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
FG = 9
CORE = 10
OUT_CORE = 5
H = W = 30
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


def _bool(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.bool_), name)


def solve_grid(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference implementation used by this script's local verification."""
    g = np.asarray(grid, dtype=np.int64)
    selected = 0
    selected_cells: list[tuple[int, int]] = []
    for color in range(1, C):
        cells = [(r, c) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c] == color]
        if not cells:
            continue
        cmin = min(c for _, c in cells)
        cmax = max(c for _, c in cells)
        span = cmin + cmax
        cell_set = set(cells)
        if all((r, span - c) in cell_set for r, c in cells):
            selected = color
            selected_cells = cells
            break

    if not selected_cells:
        return np.zeros((1, 1), dtype=np.int64)

    rmin = min(r for r, _ in selected_cells)
    rmax = max(r for r, _ in selected_cells)
    cmin = min(c for _, c in selected_cells)
    cmax = max(c for _, c in selected_cells)
    out = np.zeros((rmax - rmin + 1, cmax - cmin + 1), dtype=np.int64)
    for r, c in selected_cells:
        out[r - rmin, c - cmin] = selected
    return out


class Builder:
    def __init__(self) -> None:
        self.nodes: List[onnx.NodeProto] = []
        self.inits: List[onnx.TensorProto] = []
        self.n = 0

    def name(self, prefix: str) -> str:
        self.n += 1
        return f"{prefix}{self.n}"

    def add(self, op_type: str, inputs: list[str], outputs: list[str] | None = None, **attrs: Any) -> str:
        if outputs is None:
            outputs = [self.name(op_type.lower())]
        self.nodes.append(helper.make_node(op_type, inputs, outputs, **attrs))
        return outputs[0]

    def init(self, arr: Any, name: str | None = None) -> str:
        return _init(self.inits, arr, name or self.name("c"))

    def i64(self, vals: Any, name: str | None = None) -> str:
        return _i64(self.inits, vals, name or self.name("i"))

    def f32(self, vals: Any, name: str | None = None) -> str:
        return _f32(self.inits, vals, name or self.name("f"))

    def bool(self, vals: Any, name: str | None = None) -> str:
        return _bool(self.inits, vals, name or self.name("b"))


def _slice4(b: Builder, x: str, starts: list[int], ends: list[int], name: str | None = None) -> str:
    return b.add("Slice", [x, b.i64(starts), b.i64(ends), "axes4"], [name or b.name("sl")])


def _false_like(b: Builder, x: str, name: str) -> str:
    nx = b.add("Not", [x])
    return b.add("And", [x, nx], [name])


def build_model() -> onnx.ModelProto:
    b = Builder()
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    b.i64([0, 1, 2, 3], "axes4")
    half = b.f32([0.5], "half")
    one = b.f32([1.0], "one")
    big = b.f32([99.0], "big")
    neg = b.f32([-1.0], "neg")
    rows = b.init(np.arange(CORE, dtype=np.float32).reshape(1, 1, CORE, 1), "rows")
    cols = b.init(np.arange(CORE, dtype=np.float32).reshape(1, 1, 1, CORE), "cols")
    core = _slice4(b, IN_NAME, [0, 1, 0, 0], [1, C, CORE, CORE], "core")
    mask = b.add("Greater", [core, half], ["mask"])

    row_count = b.add("ReduceSum", [core], ["row_count"], axes=[3], keepdims=1)
    col_count = b.add("ReduceSum", [core], ["col_count"], axes=[2], keepdims=1)
    total_count = b.add("ReduceSum", [col_count], ["total_count"], axes=[3], keepdims=1)
    row_occ = b.add("Greater", [row_count, half], ["row_occ"])
    col_occ = b.add("Greater", [col_count, half], ["col_occ"])
    exists = b.add("Greater", [total_count, half], ["exists"])

    y_for_min = b.add("Where", [row_occ, rows, big], ["y_for_min"])
    x_for_min = b.add("Where", [col_occ, cols, big], ["x_for_min"])
    y_for_max = b.add("Where", [row_occ, rows, neg], ["y_for_max"])
    x_for_max = b.add("Where", [col_occ, cols, neg], ["x_for_max"])
    rmin_all = b.add("ReduceMin", [y_for_min], ["rmin_all"], axes=[2, 3], keepdims=1)
    cmin = b.add("ReduceMin", [x_for_min], ["cmin"], axes=[2, 3], keepdims=1)
    rmax_all = b.add("ReduceMax", [y_for_max], ["rmax_all"], axes=[2, 3], keepdims=1)
    cmax = b.add("ReduceMax", [x_for_max], ["cmax"], axes=[2, 3], keepdims=1)
    span = b.add("Add", [cmin, cmax], ["span"])

    weighted_cols = b.add("Mul", [col_count, cols], ["weighted_cols"])
    total_sum_cols = b.add("ReduceSum", [weighted_cols], ["total_sum_cols"], axes=[3], keepdims=1)
    twice_sum_cols = b.add("Add", [total_sum_cols, total_sum_cols], ["twice_sum_cols"])
    span_count = b.add("Mul", [span, total_count], ["span_count"])
    moment_diff = b.add("Sub", [twice_sum_cols, span_count], ["moment_diff"])
    moment_abs = b.add("Abs", [moment_diff], ["moment_abs"])
    centered = b.add("Less", [moment_abs, half], ["centered"])
    candidate = b.add("And", [centered, exists], ["candidate"])
    bbox_h = b.add("Add", [b.add("Sub", [rmax_all, rmin_all]), one], ["bbox_h"])
    bbox_w = b.add("Add", [b.add("Sub", [cmax, cmin]), one], ["bbox_w"])
    bbox_area = b.add("Mul", [bbox_h, bbox_w], ["bbox_area"])
    candidate_area = b.add("Where", [candidate, bbox_area, big], ["candidate_area"])
    min_area = b.add("ReduceMin", [candidate_area], ["min_area"], axes=[1, 2, 3], keepdims=1)
    area_diff = b.add("Abs", [b.add("Sub", [bbox_area, min_area])], ["area_diff"])
    min_area_color = b.add("Less", [area_diff, half], ["min_area_color"])
    selected_color = b.add("And", [candidate, min_area_color], ["selected_color"])
    selected = b.add("And", [mask, selected_color], ["selected"])

    rmin_src = b.add("Where", [selected_color, rmin_all, big], ["rmin_src"])
    cmin_src = b.add("Where", [selected_color, cmin, big], ["cmin_src"])
    rmax_src = b.add("Where", [selected_color, rmax_all, neg], ["rmax_src"])
    cmax_src = b.add("Where", [selected_color, cmax, neg], ["cmax_src"])
    rmin = b.add("ReduceMin", [rmin_src], ["rmin"], axes=[1, 2, 3], keepdims=1)
    cmin_sel = b.add("ReduceMin", [cmin_src], ["cmin_sel"], axes=[1, 2, 3], keepdims=1)
    rmax = b.add("ReduceMax", [rmax_src], ["rmax"], axes=[1, 2, 3], keepdims=1)
    cmax_sel = b.add("ReduceMax", [cmax_src], ["cmax_sel"], axes=[1, 2, 3], keepdims=1)

    out_rows = b.init(np.arange(OUT_CORE, dtype=np.float32).reshape(1, 1, OUT_CORE, 1), "out_rows")
    out_cols = b.init(np.arange(OUT_CORE, dtype=np.float32).reshape(1, 1, 1, OUT_CORE), "out_cols")
    base_idx = b.i64(np.arange(OUT_CORE, dtype=np.int64), "base_idx")
    idx_cap = b.i64([CORE - 1], "idx_cap")
    rmin_i4 = b.add("Cast", [rmin], ["rmin_i4"], to=TensorProto.INT64)
    cmin_i4 = b.add("Cast", [cmin_sel], ["cmin_i4"], to=TensorProto.INT64)
    rmin_i = b.add("Squeeze", [rmin_i4], ["rmin_i"], axes=[0, 1, 2, 3])
    cmin_i = b.add("Squeeze", [cmin_i4], ["cmin_i"], axes=[0, 1, 2, 3])
    row_idx_raw = b.add("Add", [base_idx, rmin_i], ["row_idx_raw"])
    col_idx_raw = b.add("Add", [base_idx, cmin_i], ["col_idx_raw"])
    row_idx_raw_f = b.add("Cast", [row_idx_raw], ["row_idx_raw_f"], to=TensorProto.FLOAT)
    col_idx_raw_f = b.add("Cast", [col_idx_raw], ["col_idx_raw_f"], to=TensorProto.FLOAT)
    row_too_big = b.add("Greater", [row_idx_raw_f, b.f32([float(CORE - 1)])], ["row_too_big"])
    col_too_big = b.add("Greater", [col_idx_raw_f, b.f32([float(CORE - 1)])], ["col_too_big"])
    row_idx = b.add("Where", [row_too_big, idx_cap, row_idx_raw], ["row_idx"])
    col_idx = b.add("Where", [col_too_big, idx_cap, col_idx_raw], ["col_idx"])
    row_gather = b.add("Gather", [selected, row_idx], ["row_gather"], axis=2)
    fg_shifted_raw = b.add("Gather", [row_gather, col_idx], ["fg_shifted_raw"], axis=3)

    row_src = b.add("Add", [out_rows, rmin], ["row_src"])
    col_src = b.add("Add", [out_cols, cmin_sel], ["col_src"])
    rmax_next = b.add("Add", [rmax, one], ["rmax_next"])
    cmax_next = b.add("Add", [cmax_sel, one], ["cmax_next"])
    in_rows = b.add("Less", [row_src, rmax_next], ["in_rows"])
    in_cols = b.add("Less", [col_src, cmax_next], ["in_cols"])
    active_rect = b.add("And", [in_rows, in_cols], ["active_rect"])

    fg_shifted = b.add("And", [fg_shifted_raw, active_rect], ["fg_shifted"])
    fg_any = _slice4(b, fg_shifted, [0, 0, 0, 0], [1, 1, OUT_CORE, OUT_CORE], "fg_any0")
    for color in range(1, FG):
        fg_part = _slice4(b, fg_shifted, [0, color, 0, 0], [1, color + 1, OUT_CORE, OUT_CORE])
        fg_any = b.add("Or", [fg_any, fg_part], [b.name("fg_any")])
    not_fg = b.add("Not", [fg_any], ["not_fg"])
    bg = b.add("And", [active_rect, not_fg], ["bg"])
    out_bool = b.add("Concat", [bg, fg_shifted], ["out_bool"], axis=1)
    out_core = b.add("Cast", [out_bool], ["out_core"], to=TensorProto.FLOAT)
    b.add("Pad", [out_core], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - OUT_CORE, W - OUT_CORE])

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


def _verify_model(path: Path) -> tuple[int, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    passed = 0
    failed = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            expected = convert_to_numpy(ex, "output")
            if expected is None:
                continue
            actual = session.run([OUT_NAME], {IN_NAME: convert_to_numpy(ex, "input")})[0] > 0.0
            if np.array_equal(actual, expected > 0.0):
                passed += 1
            else:
                failed += 1
    return passed, failed


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    passed, failed = _verify_model(BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"{TASK_ID}: {passed} pass, {failed} fail")
    print(
        f"score={result['score']:.6f} cost={result['cost']} "
        f"memory={result['memory']} params={result['params']} size={result['filesize']}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
