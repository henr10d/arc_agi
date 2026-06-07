"""Compact ONNX for NeuroGolf task081: complete cyan 2x2 L-shapes.

Task rule: the 7x7 input contains only background 0 and incomplete 2x2
cyan squares. Each square is an L-shape with exactly three color-8 cells
inside a 2x2 window. The output preserves all cyan cells and fills the
missing background corner of every such window with color 1.

ONNX approach: work only on the active 7x7 crop. Several variants are built
and scored. The best one counts cyan cells in each 2x2 window, combines that
with per-corner cyan masks to find the missing corner, creates channels
0, 1, and 8 for a compact 9-channel crop, casts once, and lets the final Pad
add channel 9 and the 23-row/column spatial padding.
"""

from __future__ import annotations

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task081"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
OPSET = 10
IR_VERSION = 10


class Builder:
    def __init__(self, name: str) -> None:
        self.name = name
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.value_infos: list[onnx.ValueInfoProto] = []
        self.counter = 0

    def unique(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def init(self, name: str, array: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(array, name))
        return name

    def vi(self, name: str, dtype: int, shape: tuple[int, ...]) -> str:
        self.value_infos.append(helper.make_tensor_value_info(name, dtype, list(shape)))
        return name

    def node(
        self,
        op_type: str,
        inputs: list[str],
        dtype: int,
        shape: tuple[int, ...],
        prefix: str,
        **attrs: Any,
    ) -> str:
        out = self.vi(self.unique(prefix), dtype, shape)
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out


def add_common_initializers(
    b: Builder,
    *,
    include_ch0: bool = True,
    include_zero6: bool = True,
    include_zero_f: bool = True,
) -> None:
    if include_zero_f:
        b.init("zero_f", np.array(0.0, dtype=np.float32))
    if include_ch0:
        b.init("ch0_st", np.array([0, 0, 0, 0], dtype=np.int64))
        b.init("ch0_en", np.array([1, 1, 7, 7], dtype=np.int64))
    b.init("ch8_st", np.array([0, 8, 0, 0], dtype=np.int64))
    b.init("ch8_en", np.array([1, 9, 7, 7], dtype=np.int64))
    b.init("axes23", np.array([2, 3], dtype=np.int64))
    b.init("win00_st", np.array([0, 0], dtype=np.int64))
    b.init("win00_en", np.array([6, 6], dtype=np.int64))
    b.init("win01_st", np.array([0, 1], dtype=np.int64))
    b.init("win01_en", np.array([6, 7], dtype=np.int64))
    b.init("win10_st", np.array([1, 0], dtype=np.int64))
    b.init("win10_en", np.array([7, 6], dtype=np.int64))
    b.init("win11_st", np.array([1, 1], dtype=np.int64))
    b.init("win11_en", np.array([7, 7], dtype=np.int64))
    if include_zero6:
        b.init("zero6", np.zeros((1, 6, 7, 7), dtype=np.bool_))
    b.init("z6c", np.zeros((1, 1, 6, 1), dtype=np.bool_))
    b.init("z1r7", np.zeros((1, 1, 1, 7), dtype=np.bool_))


def slice4(
    b: Builder,
    x: str,
    dtype: int,
    shape: tuple[int, ...],
    start: str,
    end: str,
    prefix: str,
) -> str:
    return b.node("Slice", [x, start, end], dtype, shape, prefix)


def slice_window(
    b: Builder,
    x: str,
    dtype: int,
    shape: tuple[int, ...],
    start: str,
    end: str,
    prefix: str,
) -> str:
    return b.node("Slice", [x, start, end, "axes23"], dtype, shape, prefix)


def and_all(b: Builder, items: list[str], shape: tuple[int, ...], prefix: str) -> str:
    out = items[0]
    for idx, item in enumerate(items[1:], start=1):
        out = b.node("And", [out, item], TensorProto.BOOL, shape, f"{prefix}_and{idx}")
    return out


def place_6x6(b: Builder, x: str, top: bool, left: bool, prefix: str) -> str:
    col_inputs = ["z6c", x] if left else [x, "z6c"]
    cols = b.node("Concat", col_inputs, TensorProto.BOOL, (1, 1, 6, 7), f"{prefix}_cols", axis=3)
    row_inputs = ["z1r7", cols] if top else [cols, "z1r7"]
    return b.node("Concat", row_inputs, TensorProto.BOOL, (1, 1, 7, 7), prefix, axis=2)


def place_horizontal_pair(b: Builder, left_x: str, right_x: str, top_group: bool, prefix: str) -> str:
    left_cols = b.node("Concat", [left_x, "z6c"], TensorProto.BOOL, (1, 1, 6, 7), f"{prefix}_left", axis=3)
    right_cols = b.node("Concat", ["z6c", right_x], TensorProto.BOOL, (1, 1, 6, 7), f"{prefix}_right", axis=3)
    rows6 = b.node("Or", [left_cols, right_cols], TensorProto.BOOL, (1, 1, 6, 7), f"{prefix}_rows6")
    row_inputs = [rows6, "z1r7"] if top_group else ["z1r7", rows6]
    return b.node("Concat", row_inputs, TensorProto.BOOL, (1, 1, 7, 7), prefix, axis=2)


def finish_model(
    b: Builder,
    ch8_b: str,
    fill_b: str,
    *,
    ch0_b: str | None = None,
    channel_mode: str = "zero6",
) -> onnx.ModelProto:
    if ch0_b is None:
        occupied = b.node("Or", [ch8_b, fill_b], TensorProto.BOOL, (1, 1, 7, 7), "occupied")
        out0 = b.node("Not", [occupied], TensorProto.BOOL, (1, 1, 7, 7), "out0")
    else:
        not_fill = b.node("Not", [fill_b], TensorProto.BOOL, (1, 1, 7, 7), "not_fill")
        out0 = b.node("And", [ch0_b, not_fill], TensorProto.BOOL, (1, 1, 7, 7), "out0")
    if channel_mode == "gather_zero":
        zero = b.node("And", [fill_b, ch8_b], TensorProto.BOOL, (1, 1, 7, 7), "zero")
        compact = b.node(
            "Concat",
            [out0, fill_b, zero, ch8_b],
            TensorProto.BOOL,
            (1, 4, 7, 7),
            "compact_channels",
            axis=1,
        )
        crop_bool = b.node("Gather", [compact, "channel_idx"], TensorProto.BOOL, (1, 9, 7, 7), "crop_bool", axis=1)
    else:
        crop_bool = b.node(
            "Concat",
            [out0, fill_b, "zero6", ch8_b],
            TensorProto.BOOL,
            (1, 9, 7, 7),
            "crop_bool",
            axis=1,
        )
    crop_float = b.node("Cast", [crop_bool], TensorProto.FLOAT, (1, 9, 7, 7), "crop_float", to=TensorProto.FLOAT)
    b.nodes.append(
        helper.make_node(
            "Pad",
            [crop_float],
            ["output"],
            pads=[0, 0, 0, 0, 0, 1, 23, 23],
            mode="constant",
            value=0.0,
        )
    )

    graph = helper.make_graph(
        b.nodes,
        b.name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=b.initializers,
        value_info=b.value_infos,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    return model


def build_slice_logic_model() -> onnx.ModelProto:
    b = Builder("task081_slice_logic")
    add_common_initializers(b, include_ch0=True, include_zero6=True)

    ch0 = slice4(b, "input", TensorProto.FLOAT, (1, 1, 7, 7), "ch0_st", "ch0_en", "ch0")
    ch8 = slice4(b, "input", TensorProto.FLOAT, (1, 1, 7, 7), "ch8_st", "ch8_en", "ch8")
    ch0_b = b.node("Greater", [ch0, "zero_f"], TensorProto.BOOL, (1, 1, 7, 7), "ch0_b")
    ch8_b = b.node("Greater", [ch8, "zero_f"], TensorProto.BOOL, (1, 1, 7, 7), "ch8_b")

    z00 = slice_window(b, ch0_b, TensorProto.BOOL, (1, 1, 6, 6), "win00_st", "win00_en", "z00")
    z01 = slice_window(b, ch0_b, TensorProto.BOOL, (1, 1, 6, 6), "win01_st", "win01_en", "z01")
    z10 = slice_window(b, ch0_b, TensorProto.BOOL, (1, 1, 6, 6), "win10_st", "win10_en", "z10")
    z11 = slice_window(b, ch0_b, TensorProto.BOOL, (1, 1, 6, 6), "win11_st", "win11_en", "z11")
    m00 = slice_window(b, ch8_b, TensorProto.BOOL, (1, 1, 6, 6), "win00_st", "win00_en", "m00")
    m01 = slice_window(b, ch8_b, TensorProto.BOOL, (1, 1, 6, 6), "win01_st", "win01_en", "m01")
    m10 = slice_window(b, ch8_b, TensorProto.BOOL, (1, 1, 6, 6), "win10_st", "win10_en", "m10")
    m11 = slice_window(b, ch8_b, TensorProto.BOOL, (1, 1, 6, 6), "win11_st", "win11_en", "m11")

    tl = and_all(b, [z00, m01, m10, m11], (1, 1, 6, 6), "tl")
    tr = and_all(b, [m00, z01, m10, m11], (1, 1, 6, 6), "tr")
    bl = and_all(b, [m00, m01, z10, m11], (1, 1, 6, 6), "bl")
    br = and_all(b, [m00, m01, m10, z11], (1, 1, 6, 6), "br")

    fill_top = place_horizontal_pair(b, tl, tr, top_group=True, prefix="fill_top")
    fill_bottom = place_horizontal_pair(b, bl, br, top_group=False, prefix="fill_bottom")
    fill = b.node("Or", [fill_top, fill_bottom], TensorProto.BOOL, (1, 1, 7, 7), "fill")
    return finish_model(b, ch8_b, fill, ch0_b=ch0_b)


def build_conv_count_model() -> onnx.ModelProto:
    b = Builder("task081_conv_count")
    add_common_initializers(b, include_ch0=True, include_zero6=True)
    b.init("two_f", np.array(2.0, dtype=np.float32))
    b.init("four_f", np.array(4.0, dtype=np.float32))
    b.init("k2", np.ones((1, 1, 2, 2), dtype=np.float32))

    ch0 = slice4(b, "input", TensorProto.FLOAT, (1, 1, 7, 7), "ch0_st", "ch0_en", "ch0")
    ch8 = slice4(b, "input", TensorProto.FLOAT, (1, 1, 7, 7), "ch8_st", "ch8_en", "ch8")
    ch0_b = b.node("Greater", [ch0, "zero_f"], TensorProto.BOOL, (1, 1, 7, 7), "ch0_b")
    ch8_b = b.node("Greater", [ch8, "zero_f"], TensorProto.BOOL, (1, 1, 7, 7), "ch8_b")

    count = b.node("Conv", [ch8, "k2"], TensorProto.FLOAT, (1, 1, 6, 6), "count")
    gt2 = b.node("Greater", [count, "two_f"], TensorProto.BOOL, (1, 1, 6, 6), "gt2")
    lt4 = b.node("Less", [count, "four_f"], TensorProto.BOOL, (1, 1, 6, 6), "lt4")
    has3 = b.node("And", [gt2, lt4], TensorProto.BOOL, (1, 1, 6, 6), "has3")
    z00 = slice_window(b, ch0_b, TensorProto.BOOL, (1, 1, 6, 6), "win00_st", "win00_en", "z00")
    z01 = slice_window(b, ch0_b, TensorProto.BOOL, (1, 1, 6, 6), "win01_st", "win01_en", "z01")
    z10 = slice_window(b, ch0_b, TensorProto.BOOL, (1, 1, 6, 6), "win10_st", "win10_en", "z10")
    z11 = slice_window(b, ch0_b, TensorProto.BOOL, (1, 1, 6, 6), "win11_st", "win11_en", "z11")

    tl = b.node("And", [has3, z00], TensorProto.BOOL, (1, 1, 6, 6), "tl")
    tr = b.node("And", [has3, z01], TensorProto.BOOL, (1, 1, 6, 6), "tr")
    bl = b.node("And", [has3, z10], TensorProto.BOOL, (1, 1, 6, 6), "bl")
    br = b.node("And", [has3, z11], TensorProto.BOOL, (1, 1, 6, 6), "br")

    fill_top = place_horizontal_pair(b, tl, tr, top_group=True, prefix="fill_top")
    fill_bottom = place_horizontal_pair(b, bl, br, top_group=False, prefix="fill_bottom")
    fill = b.node("Or", [fill_top, fill_bottom], TensorProto.BOOL, (1, 1, 7, 7), "fill")
    return finish_model(b, ch8_b, fill, ch0_b=ch0_b)


def build_conv_cyan_only_model() -> onnx.ModelProto:
    b = Builder("task081_pool_cyan_only")
    add_common_initializers(b, include_ch0=False, include_zero6=False, include_zero_f=False)
    b.init("half_f", np.array(0.5, dtype=np.float32))
    b.init("channel_idx", np.array([0, 1, 2, 2, 2, 2, 2, 2, 3], dtype=np.int64))

    ch8 = slice4(b, "input", TensorProto.FLOAT, (1, 1, 7, 7), "ch8_st", "ch8_en", "ch8")
    ch8_b = b.node("Cast", [ch8], TensorProto.BOOL, (1, 1, 7, 7), "ch8_b", to=TensorProto.BOOL)

    density = b.node(
        "AveragePool",
        [ch8],
        TensorProto.FLOAT,
        (1, 1, 6, 6),
        "density",
        kernel_shape=[2, 2],
        strides=[1, 1],
    )
    has3_or4 = b.node("Greater", [density, "half_f"], TensorProto.BOOL, (1, 1, 6, 6), "has3_or4")
    m00 = slice_window(b, ch8_b, TensorProto.BOOL, (1, 1, 6, 6), "win00_st", "win00_en", "m00")
    m01 = slice_window(b, ch8_b, TensorProto.BOOL, (1, 1, 6, 6), "win01_st", "win01_en", "m01")
    m10 = slice_window(b, ch8_b, TensorProto.BOOL, (1, 1, 6, 6), "win10_st", "win10_en", "m10")
    m11 = slice_window(b, ch8_b, TensorProto.BOOL, (1, 1, 6, 6), "win11_st", "win11_en", "m11")

    n00 = b.node("Not", [m00], TensorProto.BOOL, (1, 1, 6, 6), "n00")
    n01 = b.node("Not", [m01], TensorProto.BOOL, (1, 1, 6, 6), "n01")
    n10 = b.node("Not", [m10], TensorProto.BOOL, (1, 1, 6, 6), "n10")
    n11 = b.node("Not", [m11], TensorProto.BOOL, (1, 1, 6, 6), "n11")
    tl = b.node("And", [has3_or4, n00], TensorProto.BOOL, (1, 1, 6, 6), "tl")
    tr = b.node("And", [has3_or4, n01], TensorProto.BOOL, (1, 1, 6, 6), "tr")
    bl = b.node("And", [has3_or4, n10], TensorProto.BOOL, (1, 1, 6, 6), "bl")
    br = b.node("And", [has3_or4, n11], TensorProto.BOOL, (1, 1, 6, 6), "br")

    fill_top = place_horizontal_pair(b, tl, tr, top_group=True, prefix="fill_top")
    fill_bottom = place_horizontal_pair(b, bl, br, top_group=False, prefix="fill_bottom")
    fill = b.node("Or", [fill_top, fill_bottom], TensorProto.BOOL, (1, 1, 7, 7), "fill")
    return finish_model(b, ch8_b, fill, channel_mode="gather_zero")


def write_model(model: onnx.ModelProto, path: Path) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(path))


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    variants = {
        "slice": build_slice_logic_model(),
        "conv": build_conv_count_model(),
        "conv_cyan_only": build_conv_cyan_only_model(),
    }
    results: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="task081_") as tmp:
        tmp_path = Path(tmp)
        for name, model in variants.items():
            path = tmp_path / f"{TASK_ID}_{name}.onnx"
            write_model(model, path)
            score_path = tmp_path / f"{TASK_ID}.onnx"
            shutil.copyfile(path, score_path)
            results[name] = score_file(score_path)

        best_name = min(results, key=lambda key: int(results[key]["cost"]) if results[key]["valid"] else 10**18)
        if not results[best_name]["valid"]:
            raise RuntimeError(f"no valid variant: {results}")
        write_model(variants[best_name], BEST_PATH)

    for name, result in results.items():
        status = "valid" if result["valid"] else "invalid"
        print(
            f"{name}: {status} memory={result['memory']} params={result['params']} "
            f"cost={result['cost']} score={result['score']}"
        )
        if result["error"]:
            print(f"  error: {str(result['error']).strip()}")
    print(f"kept: {best_name} -> {BEST_PATH}")


if __name__ == "__main__":
    main()
