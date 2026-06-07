"""Build a compact ONNX solver for NeuroGolf task080.

Task rule: the input is a regular pixel lattice whose logical cells are
2x2, 3x3, or 4x4 solid blocks separated by one-pixel colored grid lines.
One 3x3 logical-cell stencil is complete; elsewhere only the center cell of
that stencil appears as a marker. The output copies the complete 3x3 stencil
around every marker center, clipping naturally at the lattice edge.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from score_model import convert_to_numpy, print_report, sanitize_model, score_file


ROOT = Path(__file__).resolve().parent
TASK_PATH = ROOT / "data" / "task080.json"
OUT_PATH = ROOT / "solution.onnx"
TASK_ONNX_PATH = ROOT / "task080.onnx"

IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, 10, 30, 30]
OPSET = 10
IR_VERSION = 10


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self.counter = 0

    def name(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def init(self, name: str, arr: np.ndarray) -> str:
        self.inits.append(numpy_helper.from_array(arr, name))
        return name

    def node(self, op: str, inputs: list[str], prefix: str, **attrs: Any) -> str:
        out = self.name(prefix)
        self.nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    def const_i64(self, name: str, values: list[int] | np.ndarray) -> str:
        return self.init(name, np.asarray(values, dtype=np.int64))

    def const_f32(self, name: str, values: list[float] | np.ndarray) -> str:
        return self.init(name, np.asarray(values, dtype=np.float32))


def slice4(b: Builder, x: str, starts: list[int], ends: list[int], prefix: str) -> str:
    st = b.const_i64(f"{prefix}_st_{b.counter}", starts)
    en = b.const_i64(f"{prefix}_en_{b.counter}", ends)
    ax = b.const_i64(f"{prefix}_ax_{b.counter}", [0, 1, 2, 3])
    return b.node("Slice", [x, st, en, ax], prefix)


def reduce_max(b: Builder, x: str, axes: list[int], prefix: str) -> str:
    return b.node("ReduceMax", [x], prefix, axes=axes, keepdims=1)


def reduce_sum(b: Builder, x: str, axes: list[int], prefix: str) -> str:
    return b.node("ReduceSum", [x], prefix, axes=axes, keepdims=1)


def branch_valid(b: Builder, block: int, zero_f: str, twenty_three_half: str) -> str:
    row = slice4(b, IN_NAME, [0, 1, block, 0], [1, 10, block + 1, 24], f"b{block}_rowline")
    col = slice4(b, IN_NAME, [0, 1, 0, block], [1, 10, 24, block + 1], f"b{block}_colline")
    row_count = reduce_sum(b, row, [1, 2, 3], f"b{block}_rowsum")
    col_count = reduce_sum(b, col, [1, 2, 3], f"b{block}_colsum")
    row_ok = b.node("Greater", [row_count, twenty_three_half], f"b{block}_rowok")
    col_ok = b.node("Greater", [col_count, twenty_three_half], f"b{block}_colok")
    return b.node("And", [row_ok, col_ok], f"b{block}_valid")


def shift_mask(b: Builder, mask_f: str, n: int, dy: int, dx: int, zero_f: str, prefix: str) -> str:
    sy0 = max(0, -dy)
    sy1 = n - max(0, dy)
    sx0 = max(0, -dx)
    sx1 = n - max(0, dx)
    sliced = slice4(b, mask_f, [0, 0, sy0, sx0], [1, 1, sy1, sx1], f"{prefix}_sl")
    pads = [0, 0, max(0, dy), max(0, dx), 0, 0, max(0, -dy), max(0, -dx)]
    padded = b.node("Pad", [sliced], f"{prefix}_pad", pads=pads, mode="constant", value=0.0)
    return b.node("Greater", [padded, zero_f], f"{prefix}_gt")


def build_branch(
    b: Builder,
    block: int,
    n: int,
    zero_i: str,
    neg_i: str,
    zero_f: str,
    ch_idx: str,
    twenty_three_half: str,
    active_area_bool: str,
) -> str:
    starts = np.arange(n, dtype=np.int64) * (block + 1)
    rows = b.init(f"b{block}_rows", starts)
    cols = b.init(f"b{block}_cols", starts)
    cells_h = b.node("Gather", [IN_NAME, rows], f"b{block}_gh", axis=2)
    cells = b.node("Gather", [cells_h, cols], f"b{block}_gw", axis=3)
    k = b.node("ArgMax", [cells], f"b{block}_argmax", axis=1, keepdims=1)

    c = slice4(b, k, [0, 0, 1, 1], [1, 1, n - 1, n - 1], f"b{block}_cen")
    u = slice4(b, k, [0, 0, 0, 1], [1, 1, n - 2, n - 1], f"b{block}_up")
    d = slice4(b, k, [0, 0, 2, 1], [1, 1, n, n - 1], f"b{block}_down")
    l = slice4(b, k, [0, 0, 1, 0], [1, 1, n - 1, n - 2], f"b{block}_left")
    r = slice4(b, k, [0, 0, 1, 2], [1, 1, n - 1, n], f"b{block}_right")
    tl = slice4(b, k, [0, 0, 0, 0], [1, 1, n - 2, n - 2], f"b{block}_tl")
    tr = slice4(b, k, [0, 0, 0, 2], [1, 1, n - 2, n], f"b{block}_tr")
    bl = slice4(b, k, [0, 0, 2, 0], [1, 1, n, n - 2], f"b{block}_bl")
    br = slice4(b, k, [0, 0, 2, 2], [1, 1, n, n], f"b{block}_br")

    center_nonzero = b.node("Greater", [c, zero_i], f"b{block}_cnz")
    arm_nonzero = b.node("Greater", [u, zero_i], f"b{block}_unz")
    arms_ud = b.node("Equal", [u, d], f"b{block}_aud")
    arms_lr = b.node("Equal", [l, r], f"b{block}_alr")
    arms_ul = b.node("Equal", [u, l], f"b{block}_aul")
    corners_lr = b.node("Equal", [tl, tr], f"b{block}_ctlr")
    corners_b = b.node("Equal", [bl, br], f"b{block}_cblr")
    corners_tl = b.node("Equal", [tl, bl], f"b{block}_ctbl")
    m0 = b.node("And", [center_nonzero, arm_nonzero], f"b{block}_m0")
    m1 = b.node("And", [arms_ud, arms_lr], f"b{block}_m1")
    m2 = b.node("And", [m1, arms_ul], f"b{block}_m2")
    m3 = b.node("And", [corners_lr, corners_b], f"b{block}_m3")
    m4 = b.node("And", [m3, corners_tl], f"b{block}_m4")
    src_mask = b.node("And", [b.node("And", [m0, m2], f"b{block}_m5"), m4], f"b{block}_srcmask")

    c_masked = b.node("Where", [src_mask, c, zero_i], f"b{block}_cmasked")
    arm_masked = b.node("Where", [src_mask, u, zero_i], f"b{block}_bmasked")
    corner_masked = b.node("Where", [src_mask, tl, zero_i], f"b{block}_amasked")
    c_val = reduce_max(b, c_masked, [2, 3], f"b{block}_cval")
    arm_val = reduce_max(b, arm_masked, [2, 3], f"b{block}_bval")
    corner_val = reduce_max(b, corner_masked, [2, 3], f"b{block}_aval")

    found = b.node("Greater", [c_val, zero_i], f"b{block}_found")
    valid = branch_valid(b, block, zero_f, twenty_three_half)
    found_valid = b.node("And", [found, valid], f"b{block}_foundvalid")
    marker = b.node("Equal", [k, c_val], f"b{block}_marker_eq")
    marker = b.node("And", [marker, found_valid], f"b{block}_marker")
    marker_f = b.node("Cast", [marker], f"b{block}_markerf", to=TensorProto.FLOAT)

    update_i = neg_i
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            val = c_val if dy == 0 and dx == 0 else (arm_val if dy == 0 or dx == 0 else corner_val)
            shifted = shift_mask(b, marker_f, n, dy, dx, zero_f, f"b{block}_sh_{dy+1}_{dx+1}")
            update_i = b.node("Where", [shifted, val, update_i], f"b{block}_uw_{dy+1}_{dx+1}")

    update = b.node("Equal", [ch_idx, update_i], f"b{block}_onehot")

    row_map = []
    row_in = []
    for p in range(30):
        if p < n * block + (n - 1) and p % (block + 1) < block:
            row_map.append(p // (block + 1))
            row_in.append(1.0)
        else:
            row_map.append(0)
            row_in.append(0.0)
    col_map = row_map[:]
    col_in = row_in[:]
    rm = b.init(f"b{block}_rmap", np.asarray(row_map, dtype=np.int64))
    cm = b.init(f"b{block}_cmap", np.asarray(col_map, dtype=np.int64))
    rin = b.init(f"b{block}_rin", np.asarray(row_in, dtype=bool).reshape(1, 1, 30, 1))
    cin = b.init(f"b{block}_cin", np.asarray(col_in, dtype=bool).reshape(1, 1, 1, 30))
    px_h = b.node("Gather", [update, rm], f"b{block}_pxh", axis=2)
    px = b.node("Gather", [px_h, cm], f"b{block}_pxw", axis=3)
    base_area = b.node("And", [rin, cin], f"b{block}_area")
    cell_area = b.node("And", [base_area, active_area_bool], f"b{block}_active_area")
    return b.node("And", [px, cell_area], f"b{block}_px")


def build_model() -> onnx.ModelProto:
    b = Builder()
    zero_i = b.init("zero_i64", np.asarray([[[[0]]]], dtype=np.int64))
    neg_i = b.init("neg_i64", np.asarray([[[[-1]]]], dtype=np.int64))
    zero_f = b.init("zero_f32", np.asarray([[[[0.0]]]], dtype=np.float32))
    twenty_three_half = b.init("twenty_three_half", np.asarray([[[[23.5]]]], dtype=np.float32))
    ch_idx = b.init("channel_idx", np.arange(10, dtype=np.int64).reshape(1, 10, 1, 1))
    active_area = reduce_max(b, IN_NAME, [1], "input_active_area")
    active_area_bool = b.node("Greater", [active_area, zero_f], "input_active_area_bool")

    branches = [
        build_branch(b, 2, 10, zero_i, neg_i, zero_f, ch_idx, twenty_three_half, active_area_bool),
        build_branch(b, 3, 7, zero_i, neg_i, zero_f, ch_idx, twenty_three_half, active_area_bool),
        build_branch(b, 4, 6, zero_i, neg_i, zero_f, ch_idx, twenty_three_half, active_area_bool),
    ]
    update = branches[0]
    for branch in branches[1:]:
        update = b.node("Or", [update, branch], "all_branch_or")

    update_f = b.node("Cast", [update], "update_float", to=TensorProto.FLOAT)
    update_any = reduce_max(b, update_f, [1], "update_any")
    update_active = b.node("Greater", [update_any, zero_f], "update_active")
    b.nodes.append(helper.make_node("Where", [update_active, update_f, IN_NAME], [OUT_NAME]))

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(b.nodes, "task080_lattice_stencil", [x_info], [y_info], initializer=b.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)])
    model.ir_version = IR_VERSION
    return model


def expected_one_hot(example: dict[str, Any]) -> np.ndarray | None:
    return convert_to_numpy(example, "output")


def run_correctness(path: Path) -> None:
    data = json.loads(TASK_PATH.read_text())
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    total = 0
    ok_total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            x = convert_to_numpy(example, "input")
            y = expected_one_hot(example)
            if x is None or y is None:
                print(f"{split}[{idx}]: skipped >30")
                continue
            pred = session.run([OUT_NAME], {IN_NAME: x})[0]
            ok = bool(np.array_equal(pred > 0.0, y > 0.0))
            total += 1
            ok_total += int(ok)
            print(f"{split}[{idx}]: exact={ok}")
    print(f"correctness: {ok_total}/{total}")


def profile_cost(path: Path) -> None:
    # score_model keys task data by filename stem, so profile a task080-named copy.
    shutil.copyfile(path, TASK_ONNX_PATH)
    result = score_file(TASK_ONNX_PATH)
    print_report(result)


def main() -> None:
    model = build_model()
    checked = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(checked, full_check=True)
    sanitized = sanitize_model(model)
    if sanitized is None:
        raise RuntimeError("model failed sanitizer")
    onnx.save(sanitized, OUT_PATH)
    print(f"saved {OUT_PATH}")
    run_correctness(OUT_PATH)
    profile_cost(OUT_PATH)


if __name__ == "__main__":
    main()
