"""Compact ONNX for task080 lattice-stencil propagation.

Task rule: the input is a regular colored scaffold whose one-pixel lattice
lines separate square logical cells.  One cell neighborhood shows the full
3x3 template: a center color, a repeated orthogonal-neighbor color, and a
repeated diagonal value that may be black.  Every other cell with the same
center color is a marker.  The output copies that 3x3 cell template around
each marker, clipping to the active grid and preserving every lattice line.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import sanitize_model

TASK_ID = "task080"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, 10, 30, 30]
OPSET = 10
IR_VERSION = 10


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self.init_cache: dict[tuple[str, tuple[int, ...], bytes], str] = {}
        self.zero_bool_names: dict[tuple[int, int], str] = {}
        self.counter = 0

    def name(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def init(self, name: str, arr: np.ndarray) -> str:
        self.inits.append(numpy_helper.from_array(arr, name))
        return name

    def cached_init(self, prefix: str, arr: np.ndarray) -> str:
        arr = np.ascontiguousarray(arr)
        key = (arr.dtype.str, tuple(arr.shape), arr.tobytes())
        if key not in self.init_cache:
            self.init_cache[key] = self.init(f"{prefix}_{len(self.init_cache)}", arr)
        return self.init_cache[key]

    def node(self, op_type: str, inputs: list[str], prefix: str, **attrs: Any) -> str:
        out = self.name(prefix)
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out

    def zero_bool(self, h: int, w: int) -> str:
        key = (h, w)
        if key not in self.zero_bool_names:
            self.zero_bool_names[key] = self.init(f"zero_bool_{h}x{w}", np.zeros((1, 1, h, w), dtype=bool))
        return self.zero_bool_names[key]


def slice4(b: Builder, x: str, starts: list[int], ends: list[int], prefix: str) -> str:
    starts_name = b.cached_init("slice_starts", np.asarray(starts, dtype=np.int64))
    ends_name = b.cached_init("slice_ends", np.asarray(ends, dtype=np.int64))
    axes_name = b.cached_init("slice_axes", np.asarray([0, 1, 2, 3], dtype=np.int64))
    return b.node("Slice", [x, starts_name, ends_name, axes_name], prefix)


def reduce_max(b: Builder, x: str, axes: list[int], prefix: str) -> str:
    return b.node("ReduceMax", [x], prefix, axes=axes, keepdims=1)


def shift_mask(b: Builder, mask: str, n: int, dy: int, dx: int, prefix: str) -> str:
    if dy == 0 and dx == 0:
        return mask
    sy0 = max(0, -dy)
    sy1 = n - max(0, dy)
    sx0 = max(0, -dx)
    sx1 = n - max(0, dx)
    out = slice4(b, mask, [0, 0, sy0, sx0], [1, 1, sy1, sx1], f"{prefix}_slice")
    h = sy1 - sy0
    if dx > 0:
        out = b.node("Concat", [b.zero_bool(h, dx), out], f"{prefix}_xpad", axis=3)
    elif dx < 0:
        out = b.node("Concat", [out, b.zero_bool(h, -dx)], f"{prefix}_xpad", axis=3)
    if dy > 0:
        out = b.node("Concat", [b.zero_bool(dy, n), out], f"{prefix}_ypad", axis=2)
    elif dy < 0:
        out = b.node("Concat", [out, b.zero_bool(-dy, n)], f"{prefix}_ypad", axis=2)
    return out


def build_branch(
    b: Builder,
    color_grid: str,
    block: int,
    n: int,
    zero_i: str,
    sentinel_u8: str,
) -> str:
    starts = np.arange(n, dtype=np.int64) * (block + 1)
    row_idx = b.cached_init(f"b{block}_cell_idx", starts)
    col_idx = row_idx
    cells_h = b.node("Gather", [color_grid, row_idx], f"b{block}_cells_h", axis=2)
    colors_u8 = b.node("Gather", [cells_h, col_idx], f"b{block}_colors_u8", axis=3)
    colors = b.node("Cast", [colors_u8], f"b{block}_colors", to=TensorProto.INT32)

    center = slice4(b, colors, [0, 0, 1, 1], [1, 1, n - 1, n - 1], f"b{block}_center")
    up = slice4(b, colors, [0, 0, 0, 1], [1, 1, n - 2, n - 1], f"b{block}_up")
    top_left = slice4(b, colors, [0, 0, 0, 0], [1, 1, n - 2, n - 2], f"b{block}_top_left")

    arm_nonzero = b.node("Greater", [up, zero_i], f"b{block}_arm_nonzero")
    source_mask = arm_nonzero

    center_masked = b.node("Where", [source_mask, center, zero_i], f"b{block}_center_masked")
    arm_masked = b.node("Where", [source_mask, up, zero_i], f"b{block}_arm_masked")
    corner_masked = b.node("Where", [source_mask, top_left, zero_i], f"b{block}_corner_masked")
    center_value_i = reduce_max(b, center_masked, [2, 3], f"b{block}_center_value_i")
    arm_value_i = reduce_max(b, arm_masked, [2, 3], f"b{block}_arm_value_i")
    corner_value_i = reduce_max(b, corner_masked, [2, 3], f"b{block}_corner_value_i")

    found_valid = b.node("Greater", [center_value_i, zero_i], f"b{block}_found_valid")
    marker = b.node("Equal", [colors, center_value_i], f"b{block}_marker_eq")
    marker = b.node("And", [marker, found_valid], f"b{block}_marker")

    center_value = b.node("Cast", [center_value_i], f"b{block}_center_value", to=TensorProto.UINT8)
    arm_value = b.node("Cast", [arm_value_i], f"b{block}_arm_value", to=TensorProto.UINT8)
    corner_value = b.node("Cast", [corner_value_i], f"b{block}_corner_value", to=TensorProto.UINT8)

    update = sentinel_u8
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            value = center_value if dy == 0 and dx == 0 else arm_value if dy == 0 or dx == 0 else corner_value
            shifted = shift_mask(b, marker, n, dy, dx, f"b{block}_shift_{dy + 1}_{dx + 1}")
            update = b.node("Where", [shifted, value, update], f"b{block}_update_{dy + 1}_{dx + 1}")

    row_map: list[int] = []
    for pos in range(30):
        if pos < n * block + (n - 1) and pos % (block + 1) < block:
            row_map.append(pos // (block + 1))
        else:
            row_map.append(0)

    row_map_name = b.cached_init(f"b{block}_row_map", np.asarray(row_map, dtype=np.int64))
    col_map_name = row_map_name
    px_h = b.node("Gather", [update, row_map_name], f"b{block}_px_h", axis=2)
    px = b.node("Gather", [px_h, col_map_name], f"b{block}_px", axis=3)
    return px


def build_model() -> onnx.ModelProto:
    b = Builder()
    zero_i = b.init("zero_i32", np.asarray([[[[0]]]], dtype=np.int32))
    zero_f = b.init("zero_f32", np.asarray([[[[0.0]]]], dtype=np.float32))
    sentinel_u8 = b.init("sentinel_u8", np.asarray([[[[255]]]], dtype=np.uint8))

    active_area = reduce_max(b, IN_NAME, [1], "active_area")
    active_area_bool = b.node("Greater", [active_area, zero_f], "active_area_bool")
    color_grid_i64 = b.node("ArgMax", [IN_NAME], "color_grid_i64", axis=1, keepdims=1)
    color_grid_u8 = b.node("Cast", [color_grid_i64], "color_grid_u8", to=TensorProto.UINT8)

    branches = [
        build_branch(b, color_grid_u8, 2, 10, zero_i, sentinel_u8),
        build_branch(b, color_grid_u8, 3, 7, zero_i, sentinel_u8),
        build_branch(b, color_grid_u8, 4, 6, zero_i, sentinel_u8),
    ]
    branch0_active = b.node("Less", [branches[0], sentinel_u8], "branch0_active")
    update_u8 = b.node("Where", [branch0_active, branches[0], branches[1]], "combine_0")
    combined_active = b.node("Less", [update_u8, sentinel_u8], "combined_active")
    update_u8 = b.node("Where", [combined_active, update_u8, branches[2]], "combine_1")
    active_update_raw = b.node("Less", [update_u8, sentinel_u8], "active_update_raw")
    active_update = b.node("And", [active_update_raw, active_area_bool], "active_update")
    base_code = b.node("Where", [active_area_bool, color_grid_u8, sentinel_u8], "base_code")
    output_code_u8 = b.node("Where", [active_update, update_u8, base_code], "output_code_u8")
    output_code = b.node("Cast", [output_code_u8], "output_code", to=TensorProto.INT32)
    channel_idx_i32 = b.init("channel_idx_i32", np.arange(10, dtype=np.int32).reshape(1, 10, 1, 1))
    output_onehot = b.node("Equal", [channel_idx_i32, output_code], "output_onehot")
    b.nodes.append(helper.make_node("Cast", [output_onehot], [OUT_NAME], to=TensorProto.FLOAT))

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(b.nodes, "task080_lattice_stencil", [x_info], [y_info], initializer=b.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)])
    model.ir_version = IR_VERSION
    return model


def main() -> None:
    model = build_model()
    checked = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(checked, full_check=True)
    sanitized = sanitize_model(model)
    if sanitized is None:
        raise RuntimeError("model failed sanitizer")
    onnx.save(sanitized, BEST_PATH)
    print(f"saved {BEST_PATH}")


if __name__ == "__main__":
    main()
