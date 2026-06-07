"""ONNX solution for NeuroGolf task331 using Kaggle one-hot I/O.

Task rule: every blue pixel in the 10x10 input becomes the center of a
five-cell plus stamp in the output. The source pixel remains blue; the cell
above it is red, the cell to the left is orange, the cell to the right is
magenta, and the cell below is cyan. Stamps are clipped at grid boundaries.

The provided train, test, and arc-gen examples contain fixed 10x10 binary
inputs and no overlapping generated stamp cells. The compact direct model
reconstructs the 10x10 foreground by shifting the blue channel once per colored
neighbor, then uses the inverse of that stamped foreground as channel 0.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import score_model

TASK_ID = "task331"
TASK_NUM = 331
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
G = 10
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10

PLUS_OFFSETS = {
    2: (-1, 0),  # red above
    7: (0, -1),  # orange left
    1: (0, 0),  # blue center
    6: (0, 1),  # magenta right
    8: (1, 0),  # cyan below
}


def _i64(inits: list[onnx.TensorProto], vals: list[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))
    return name


def _f32(inits: list[onnx.TensorProto], vals: list[float] | np.ndarray, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.float32), name=name))
    return name


def _u8(inits: list[onnx.TensorProto], vals: list[int] | np.ndarray, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.uint8), name=name))
    return name


def _strip(model: onnx.ModelProto) -> onnx.ModelProto:
    model.doc_string = ""
    model.producer_name = ""
    model.producer_version = ""
    model.domain = ""
    model.model_version = 0
    del model.metadata_props[:]
    return model


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference transform on raw integer grids."""
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(arr)
    for r, c in np.argwhere(arr == 1):
        for color, (dr, dc) in PLUS_OFFSETS.items():
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < arr.shape[0] and 0 <= cc < arr.shape[1]:
                out[rr, cc] = color
    return out


def _shift_blue(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    blue: str,
    dr: int,
    dc: int,
    name: str,
    axes: str | None = None,
) -> str:
    if dr == 0 and dc == 0:
        return blue

    row_start = max(0, -dr)
    row_end = G - max(0, dr)
    col_start = max(0, -dc)
    col_end = G - max(0, dc)
    pad_top = max(0, dr)
    pad_bottom = max(0, -dr)
    pad_left = max(0, dc)
    pad_right = max(0, -dc)

    starts = _i64(inits, [row_start, col_start], f"{name}_s")
    ends = _i64(inits, [row_end, col_end], f"{name}_e")
    axes = axes or _i64(inits, [2, 3], f"{name}_a")
    crop = f"{name}_crop"
    nodes.append(helper.make_node("Slice", [blue, starts, ends, axes], [crop]))
    nodes.append(
        helper.make_node(
            "Pad",
            [crop],
            [name],
            pads=[0, 0, pad_top, pad_left, 0, 0, pad_bottom, pad_right],
        )
    )
    return name


def _shift_blue_concat(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    blue: str,
    dr: int,
    dc: int,
    name: str,
    axes: str,
    zero_dtype: Any = np.uint8,
    zero_row: str | None = None,
    zero_col: str | None = None,
) -> str:
    if dr == 0 and dc == 0:
        return blue

    row_start = max(0, -dr)
    row_end = G - max(0, dr)
    col_start = max(0, -dc)
    col_end = G - max(0, dc)
    starts = _i64(inits, [row_start, col_start], f"{name}_s")
    ends = _i64(inits, [row_end, col_end], f"{name}_e")
    crop = f"{name}_crop"
    nodes.append(helper.make_node("Slice", [blue, starts, ends, axes], [crop]))

    if dr < 0:
        if zero_row is None:
            zero_row = f"{name}_zr"
            inits.append(numpy_helper.from_array(np.zeros((1, 1, 1, G), dtype=zero_dtype), name=zero_row))
        nodes.append(helper.make_node("Concat", [crop, zero_row], [name], axis=2))
    elif dr > 0:
        if zero_row is None:
            zero_row = f"{name}_zr"
            inits.append(numpy_helper.from_array(np.zeros((1, 1, 1, G), dtype=zero_dtype), name=zero_row))
        nodes.append(helper.make_node("Concat", [zero_row, crop], [name], axis=2))
    elif dc < 0:
        if zero_col is None:
            zero_col = f"{name}_zc"
            inits.append(numpy_helper.from_array(np.zeros((1, 1, G, 1), dtype=zero_dtype), name=zero_col))
        nodes.append(helper.make_node("Concat", [crop, zero_col], [name], axis=3))
    else:
        if zero_col is None:
            zero_col = f"{name}_zc"
            inits.append(numpy_helper.from_array(np.zeros((1, 1, G, 1), dtype=zero_dtype), name=zero_col))
        nodes.append(helper.make_node("Concat", [zero_col, crop], [name], axis=3))

    return name


def _i64_cached(
    inits: list[onnx.TensorProto],
    cache: dict[tuple[int, ...], str],
    vals: list[int],
    name: str,
) -> str:
    key = tuple(vals)
    if key not in cache:
        cache[key] = _i64(inits, vals, name)
    return cache[key]


def _shift_blue_concat_cached(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    cache: dict[tuple[int, ...], str],
    blue: str,
    dr: int,
    dc: int,
    name: str,
    axes: str,
    zero_dtype: Any = np.uint8,
    zero_row: str | None = None,
    zero_col: str | None = None,
) -> str:
    if dr == 0 and dc == 0:
        return blue

    row_start = max(0, -dr)
    row_end = G - max(0, dr)
    col_start = max(0, -dc)
    col_end = G - max(0, dc)
    starts = _i64_cached(inits, cache, [row_start, col_start], f"{name}_s")
    ends = _i64_cached(inits, cache, [row_end, col_end], f"{name}_e")
    crop = f"{name}_crop"
    nodes.append(helper.make_node("Slice", [blue, starts, ends, axes], [crop]))

    if dr < 0:
        if zero_row is None:
            zero_row = f"{name}_zr"
            inits.append(numpy_helper.from_array(np.zeros((1, 1, 1, G), dtype=zero_dtype), name=zero_row))
        nodes.append(helper.make_node("Concat", [crop, zero_row], [name], axis=2))
    elif dr > 0:
        if zero_row is None:
            zero_row = f"{name}_zr"
            inits.append(numpy_helper.from_array(np.zeros((1, 1, 1, G), dtype=zero_dtype), name=zero_row))
        nodes.append(helper.make_node("Concat", [zero_row, crop], [name], axis=2))
    elif dc < 0:
        if zero_col is None:
            zero_col = f"{name}_zc"
            inits.append(numpy_helper.from_array(np.zeros((1, 1, G, 1), dtype=zero_dtype), name=zero_col))
        nodes.append(helper.make_node("Concat", [crop, zero_col], [name], axis=3))
    else:
        if zero_col is None:
            zero_col = f"{name}_zc"
            inits.append(numpy_helper.from_array(np.zeros((1, 1, G, 1), dtype=zero_dtype), name=zero_col))
        nodes.append(helper.make_node("Concat", [zero_col, crop], [name], axis=3))

    return name


def build_direct() -> onnx.ModelProto:
    """Smallest candidate: structural 10x10 shifts plus background rebuild."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, [1, 2, 3], "axes")
    ch0_start = _i64(inits, [0, 0, 0], "ch0_s")
    ch0_end = _i64(inits, [1, G, G], "ch0_e")
    blue_start = _i64(inits, [1, 0, 0], "blue_s")
    blue_end = _i64(inits, [2, G, G], "blue_e")
    shift_axes = _i64(inits, [2, 3], "shift_axes")
    _f32(inits, np.zeros((1, 1, G, G), dtype=np.float32), "zero")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, ch0_start, ch0_end, axes], ["ch0"]),
            helper.make_node("Slice", [IN_NAME, blue_start, blue_end, axes], ["blue"]),
        ]
    )

    ch1 = _shift_blue(nodes, inits, "blue", 0, 0, "ch1", shift_axes)
    ch2 = _shift_blue(nodes, inits, "blue", -1, 0, "ch2", shift_axes)
    ch6 = _shift_blue(nodes, inits, "blue", 0, 1, "ch6", shift_axes)
    ch7 = _shift_blue(nodes, inits, "blue", 0, -1, "ch7", shift_axes)
    ch8 = _shift_blue(nodes, inits, "blue", 1, 0, "ch8", shift_axes)

    nodes.extend(
        [
            helper.make_node("Sum", [ch1, ch2, ch6, ch7, ch8], ["fg"]),
            helper.make_node("Sub", ["ch0", "fg"], ["bg"]),
            helper.make_node(
                "Concat",
                ["bg", ch1, ch2, "zero", "zero", "zero", ch6, ch7, ch8, "zero"],
                ["out10"],
                axis=1,
            ),
            helper.make_node(
                "Pad",
                ["out10"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - G, W - G],
            ),
        ]
    )

    graph = helper.make_graph(nodes, "task331_direct", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return _strip(model)


def build_direct_bool() -> onnx.ModelProto:
    """Lower-memory direct candidate using bool masks until the final 10x10 cast."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, [1, 2, 3], "axes")
    ch0_start = _i64(inits, [0, 0, 0], "ch0_s")
    ch0_end = _i64(inits, [1, G, G], "ch0_e")
    blue_start = _i64(inits, [1, 0, 0], "blue_s")
    blue_end = _i64(inits, [2, G, G], "blue_e")
    shift_axes = _i64(inits, [2, 3], "shift_axes")
    inits.append(numpy_helper.from_array(np.zeros((1, 1, G, G), dtype=np.bool_), name="zero"))
    inits.append(numpy_helper.from_array(np.zeros((1, 1, 1, G), dtype=np.bool_), name="zero_row"))
    inits.append(numpy_helper.from_array(np.zeros((1, 1, G, 1), dtype=np.bool_), name="zero_col"))

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, ch0_start, ch0_end, axes], ["ch0_f"]),
            helper.make_node("Slice", [IN_NAME, blue_start, blue_end, axes], ["blue_f"]),
            helper.make_node("Cast", ["ch0_f"], ["ch0"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["blue_f"], ["blue"], to=TensorProto.BOOL),
        ]
    )

    ch1 = _shift_blue_concat(nodes, inits, "blue", 0, 0, "ch1", shift_axes, np.bool_, "zero_row", "zero_col")
    ch2 = _shift_blue_concat(nodes, inits, "blue", -1, 0, "ch2", shift_axes, np.bool_, "zero_row", "zero_col")
    ch6 = _shift_blue_concat(nodes, inits, "blue", 0, 1, "ch6", shift_axes, np.bool_, "zero_row", "zero_col")
    ch7 = _shift_blue_concat(nodes, inits, "blue", 0, -1, "ch7", shift_axes, np.bool_, "zero_row", "zero_col")
    ch8 = _shift_blue_concat(nodes, inits, "blue", 1, 0, "ch8", shift_axes, np.bool_, "zero_row", "zero_col")

    nodes.extend(
        [
            helper.make_node("Or", [ch1, ch2], ["fg12"]),
            helper.make_node("Or", ["fg12", ch6], ["fg126"]),
            helper.make_node("Or", ["fg126", ch7], ["fg1267"]),
            helper.make_node("Or", ["fg1267", ch8], ["fg"]),
            helper.make_node("Not", ["fg"], ["not_fg"]),
            helper.make_node("And", ["ch0", "not_fg"], ["bg"]),
            helper.make_node(
                "Concat",
                ["bg", ch1, ch2, "zero", "zero", "zero", ch6, ch7, ch8, "zero"],
                ["out10_b"],
                axis=1,
            ),
            helper.make_node("Cast", ["out10_b"], ["out10"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out10"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - G, W - G],
            ),
        ]
    )

    graph = helper.make_graph(nodes, "task331_direct_bool", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return _strip(model)


def build_direct_bool_bg_from_fg() -> onnx.ModelProto:
    """Direct bool candidate: background is the inverse of all stamped cells."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    slice_cache: dict[tuple[int, ...], str] = {}

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, [1, 2, 3], "axes")
    blue_start = _i64(inits, [1, 0, 0], "blue_s")
    blue_end = _i64(inits, [2, G, G], "blue_e")
    shift_axes = _i64(inits, [2, 3], "shift_axes")
    inits.append(numpy_helper.from_array(np.zeros((1, 1, G, G), dtype=np.bool_), name="zero"))
    inits.append(numpy_helper.from_array(np.zeros((1, 1, 1, G), dtype=np.bool_), name="zero_row"))
    inits.append(numpy_helper.from_array(np.zeros((1, 1, G, 1), dtype=np.bool_), name="zero_col"))

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, blue_start, blue_end, axes], ["blue_f"]),
            helper.make_node("Cast", ["blue_f"], ["blue"], to=TensorProto.BOOL),
        ]
    )

    ch1 = _shift_blue_concat_cached(
        nodes, inits, slice_cache, "blue", 0, 0, "ch1", shift_axes, np.bool_, "zero_row", "zero_col"
    )
    ch2 = _shift_blue_concat_cached(
        nodes, inits, slice_cache, "blue", -1, 0, "ch2", shift_axes, np.bool_, "zero_row", "zero_col"
    )
    ch6 = _shift_blue_concat_cached(
        nodes, inits, slice_cache, "blue", 0, 1, "ch6", shift_axes, np.bool_, "zero_row", "zero_col"
    )
    ch7 = _shift_blue_concat_cached(
        nodes, inits, slice_cache, "blue", 0, -1, "ch7", shift_axes, np.bool_, "zero_row", "zero_col"
    )
    ch8 = _shift_blue_concat_cached(
        nodes, inits, slice_cache, "blue", 1, 0, "ch8", shift_axes, np.bool_, "zero_row", "zero_col"
    )

    nodes.extend(
        [
            helper.make_node("Or", [ch1, ch2], ["fg12"]),
            helper.make_node("Or", ["fg12", ch6], ["fg126"]),
            helper.make_node("Or", ["fg126", ch7], ["fg1267"]),
            helper.make_node("Or", ["fg1267", ch8], ["fg"]),
            helper.make_node("Not", ["fg"], ["bg"]),
            helper.make_node(
                "Concat",
                ["bg", ch1, ch2, "zero", "zero", "zero", ch6, ch7, ch8, "zero"],
                ["out10_b"],
                axis=1,
            ),
            helper.make_node("Cast", ["out10_b"], ["out10"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out10"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - G, W - G],
            ),
        ]
    )

    graph = helper.make_graph(nodes, "task331_direct_bool_bg_from_fg", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return _strip(model)


def build_scatter() -> onnx.ModelProto:
    """Scatter baseline: shift masks, scatter five colored planes by channel id."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, [1, 2, 3], "axes")
    ch0_start = _i64(inits, [0, 0, 0], "ch0_s")
    ch0_end = _i64(inits, [1, G, G], "ch0_e")
    blue_start = _i64(inits, [1, 0, 0], "blue_s")
    blue_end = _i64(inits, [2, G, G], "blue_e")
    one = _f32(inits, [1.0], "one")
    shape10 = _i64(inits, [1, C, G, G], "shape10")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, ch0_start, ch0_end, axes], ["ch0"]),
            helper.make_node("Slice", [IN_NAME, blue_start, blue_end, axes], ["blue"]),
        ]
    )

    ch1 = _shift_blue(nodes, inits, "blue", 0, 0, "ch1")
    ch2 = _shift_blue(nodes, inits, "blue", -1, 0, "ch2")
    ch6 = _shift_blue(nodes, inits, "blue", 0, 1, "ch6")
    ch7 = _shift_blue(nodes, inits, "blue", 0, -1, "ch7")
    ch8 = _shift_blue(nodes, inits, "blue", 1, 0, "ch8")

    idx = np.zeros((1, 5, G, G), dtype=np.int64)
    idx[:, 0, :, :] = 1
    idx[:, 1, :, :] = 2
    idx[:, 2, :, :] = 6
    idx[:, 3, :, :] = 7
    idx[:, 4, :, :] = 8
    _i64(inits, idx.reshape(-1).tolist(), "scatter_idx_flat")
    _i64(inits, [1, 5, G, G], "idx_shape")

    nodes.extend(
        [
            helper.make_node("ConstantOfShape", [shape10], ["base"]),
            helper.make_node("Concat", [ch1, ch2, ch6, ch7, ch8], ["updates"], axis=1),
            helper.make_node("Reshape", ["scatter_idx_flat", "idx_shape"], ["scatter_idx"]),
            helper.make_node("Scatter", ["base", "scatter_idx", "updates"], ["fg10"], axis=1),
            helper.make_node("ReduceMax", ["fg10"], ["any_fg"], axes=[1], keepdims=1),
            helper.make_node("Sub", [one, "any_fg"], ["not_fg"]),
            helper.make_node("Mul", ["ch0", "not_fg"], ["bg"]),
            helper.make_node("Slice", ["fg10", blue_start, _i64(inits, [C, G, G], "fg_end"), axes], ["fg_rest"]),
            helper.make_node("Concat", ["bg", "fg_rest"], ["out10"], axis=1),
            helper.make_node(
                "Pad",
                ["out10"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - G, W - G],
            ),
        ]
    )

    graph = helper.make_graph(nodes, "task331_scatter", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return _strip(model)


def build_conv() -> onnx.ModelProto:
    """Convolution baseline: add a learned delta stamp to the 10x10 input crop."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, [1, 2, 3], "axes")
    crop_start = _i64(inits, [0, 0, 0], "crop_s")
    crop_end = _i64(inits, [C, G, G], "crop_e")
    blue_start = _i64(inits, [1, 0, 0], "blue_s")
    blue_end = _i64(inits, [2, G, G], "blue_e")

    w = np.zeros((C, 1, 3, 3), dtype=np.float32)
    for kr, kc in [(1, 1), (2, 1), (1, 2), (1, 0), (0, 1)]:
        w[0, 0, kr, kc] = -1.0
    w[2, 0, 2, 1] = 1.0  # red above source
    w[7, 0, 1, 2] = 1.0  # orange left of source
    w[6, 0, 1, 0] = 1.0  # magenta right of source
    w[8, 0, 0, 1] = 1.0  # cyan below source
    _f32(inits, w, "W")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, crop_start, crop_end, axes], ["input10"]),
            helper.make_node("Slice", [IN_NAME, blue_start, blue_end, axes], ["blue"]),
            helper.make_node("Conv", ["blue", "W"], ["delta"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            helper.make_node("Add", ["input10", "delta"], ["out10"]),
            helper.make_node(
                "Pad",
                ["out10"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - G, W - G],
            ),
        ]
    )

    graph = helper.make_graph(nodes, "task331_conv", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return _strip(model)


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def validate_model(model: onnx.ModelProto) -> tuple[bool, str]:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            inp = _grid_to_onehot(example["input"])
            expected = _grid_to_onehot(example["output"])
            pred = sess.run([OUT_NAME], {IN_NAME: inp})[0]
            if not np.array_equal(pred > 0.0, expected > 0.0):
                return False, f"{split}[{idx}] mismatch"
    return True, "ok"


def analyze_task() -> dict[str, Any]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    mismatches: dict[str, list[int]] = {}
    collisions: dict[str, list[tuple[int, tuple[int, int], list[int]]]] = {}
    for split, examples in data.items():
        bad: list[int] = []
        overlap: list[tuple[int, tuple[int, int], list[int]]] = []
        for idx, example in enumerate(examples):
            pred = solve(example["input"]).tolist()
            if pred != example["output"]:
                bad.append(idx)

            writes: dict[tuple[int, int], list[int]] = {}
            grid = np.asarray(example["input"])
            for r, c in np.argwhere(grid == 1):
                for color, (dr, dc) in PLUS_OFFSETS.items():
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < G and 0 <= cc < G:
                        writes.setdefault((rr, cc), []).append(color)
            for cell, colors in writes.items():
                if len(colors) > 1:
                    overlap.append((idx, cell, colors))
        mismatches[split] = bad
        collisions[split] = overlap
    return {"mismatches": mismatches, "collisions": collisions}


def _score_candidate(model: onnx.ModelProto, label: str, tmp_root: Path) -> dict[str, Any]:
    path = tmp_root / label / f"{TASK_ID}.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    result = score_model.score_file(path)
    result["label"] = label
    result["path"] = path
    result["nodes"] = len(model.graph.node)
    result["initializers"] = len(model.graph.initializer)
    return result


def _profile_tensor_counts(path: Path) -> tuple[int, int]:
    model = score_model.sanitize_model(onnx.load(str(path)))
    if model is None:
        return 0, 0
    static_tensors = {out for node in model.graph.node for out in node.output if out and out != OUT_NAME}

    inputs = score_model.load_task_examples(path)
    trace_path, error = score_model.run_profiled_session(model, inputs, path)
    if error or trace_path is None:
        return len(static_tensors), 0

    node_outputs = {node.name: list(node.output) for node in model.graph.node}
    realized: set[str] = set()
    with open(trace_path, encoding="utf-8") as fh:
        for event in json.load(fh):
            if event.get("cat") != "Node":
                continue
            node_name = event.get("name", "").replace("_kernel_time", "")
            for output_name in node_outputs.get(node_name, []):
                if output_name and output_name != OUT_NAME:
                    realized.add(output_name)
    return len(static_tensors), len(realized)


def main() -> None:
    analysis = analyze_task()
    if any(analysis["mismatches"].values()):
        raise SystemExit(f"reference rule mismatch: {analysis['mismatches']}")

    candidates = {
        "direct_bool_bg_from_fg": build_direct_bool_bg_from_fg(),
        "direct_bool": build_direct_bool(),
        "direct": build_direct(),
        "scatter": build_scatter(),
        "conv": build_conv(),
    }

    tmp_root = Path(tempfile.mkdtemp(prefix="task331_candidates_"))
    try:
        rows: list[dict[str, Any]] = []
        for label, model in candidates.items():
            ok, message = validate_model(model)
            if not ok:
                print(f"{label}: invalid ({message})")
                continue
            result = _score_candidate(model, label, tmp_root)
            rows.append(result)
            print(
                f"{label}: score={result['score']:.6f} cost={result['cost']} "
                f"memory={result['memory']} params={result['params']} "
                f"nodes={result['nodes']} inits={result['initializers']}"
            )

        valid = [row for row in rows if row["valid"]]
        if not valid:
            raise SystemExit("no valid candidates")
        best = max(valid, key=lambda row: float(row["score"]))
        shutil.copyfile(best["path"], BEST_PATH)
        tensor_count, realized_count = _profile_tensor_counts(BEST_PATH)
        print(f"selected: {best['label']} -> {BEST_PATH}")
        print(f"tensor_count={tensor_count} realized_tensor_count={realized_count}")
        print(f"collisions={ {k: len(v) for k, v in analysis['collisions'].items()} }")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
