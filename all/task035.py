"""Build a compact ONNX model for NeuroGolf task035.

Task rule: a cyan (color 8) rectangle occupies rows starting at 3 and
ending at a variable bottom row, with a variable left edge and fixed right
edge at column 5. Isolated non-background marker cells sit on the outer
border rows/columns. Each marker is copied orthogonally onto the nearest
edge cell of the rectangle, while the original marker and the rest of the
grid remain unchanged.

The ONNX graph works on the 10x10 crop as bool one-hot data, projects
markers from row 0, row 9, column 0, and column 9, overlays those projected
colors on the original crop, then casts and pads only at the final output.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402


TASK_NUM = "035"
TASK_NAME = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_NAME}.json"
OUT_PATH = Path(__file__).resolve().parent / f"{TASK_NAME}.onnx"


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.value_infos: list[onnx.ValueInfoProto] = []
        self.false_blocks: dict[tuple[int, ...], str] = {}
        self.counter = 0

    def name(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def init(self, name: str, array: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(array, name))
        return name

    def vi(self, name: str, dtype: int, shape: tuple[int, ...]) -> str:
        self.value_infos.append(helper.make_tensor_value_info(name, dtype, list(shape)))
        return name

    def false_block_name(self, shape: tuple[int, ...]) -> str:
        if shape not in self.false_blocks:
            name = f"false_{'_'.join(map(str, shape))}"
            self.false_blocks[shape] = self.init(name, np.zeros(shape, dtype=np.bool_))
        return self.false_blocks[shape]

    def node(
        self,
        op_type: str,
        inputs: list[str],
        dtype: int,
        shape: tuple[int, ...],
        prefix: str,
        **attrs: object,
    ) -> str:
        out = self.vi(self.name(prefix), dtype, shape)
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out


def grid_from_onehot(arr: np.ndarray) -> np.ndarray:
    return arr[0, :, :10, :10].argmax(axis=0).astype(np.int64)


def make_const_initializers(b: Builder) -> None:
    b.init("zero_f", np.array(0.0, dtype=np.float32))
    b.init("zero_u8", np.array(0, dtype=np.uint8))
    b.init("ten_i", np.array([10], dtype=np.int64))
    b.init("rect_row3_starts", np.array([8, 3], dtype=np.int64))
    b.init("rect_row3_ends", np.array([9, 4], dtype=np.int64))
    b.init("rect_row3_axes", np.array([1, 2], dtype=np.int64))
    b.init("rect_col5_starts", np.array([8, 5], dtype=np.int64))
    b.init("rect_col5_ends", np.array([9, 6], dtype=np.int64))
    b.init("rect_col5_axes", np.array([1, 3], dtype=np.int64))
    b.init("row0_starts", np.array([0], dtype=np.int64))
    b.init("row0_ends", np.array([1], dtype=np.int64))
    b.init("row9_starts", np.array([9], dtype=np.int64))
    b.init("row9_ends", np.array([10], dtype=np.int64))
    b.init("col0_starts", np.array([0], dtype=np.int64))
    b.init("col0_ends", np.array([1], dtype=np.int64))
    b.init("col9_starts", np.array([9], dtype=np.int64))
    b.init("col9_ends", np.array([10], dtype=np.int64))
    b.init("axes_c", np.array([1], dtype=np.int64))
    b.init("axes_h", np.array([2], dtype=np.int64))
    b.init("crop_starts", np.array([0, 0, 0, 0], dtype=np.int64))
    b.init("crop_ends", np.array([1, 10, 10, 10], dtype=np.int64))
    b.init("crop_axes", np.array([0, 1, 2, 3], dtype=np.int64))
    b.init("shape_10", np.array([10], dtype=np.int64))
    b.init("shape_100", np.array([100], dtype=np.int64))
    b.init("shape_1", np.array([1], dtype=np.int64))
    b.init("shape_grid", np.array([1, 1, 10, 10], dtype=np.int64))
    b.init("row10", (np.arange(10, dtype=np.int64) * 10))
    b.init("cols", np.arange(10, dtype=np.int64))
    b.init("right_idx", (np.arange(10, dtype=np.int64) * 10 + 5))
    b.init("top_idx", (np.arange(10, dtype=np.int64) + 30))
    b.init("row_weights", np.arange(10, dtype=np.float32).reshape(1, 1, 10, 1))
    b.init("color_ids", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1))
    b.init("pad30", np.array([0, 0, 0, 0, 0, 0, 20, 20], dtype=np.int64))


def slice_axis(
    b: Builder,
    x: str,
    dtype: int,
    in_shape: tuple[int, ...],
    axis: int,
    start_name: str,
    end_name: str,
    out_shape: tuple[int, ...],
    prefix: str,
) -> str:
    axes_name = {1: "axes_c", 2: "axes_h", 3: "axes_w"}[axis]
    return b.node("Slice", [x, start_name, end_name, axes_name], dtype, out_shape, prefix)


def pad_to_10(
    b: Builder,
    x: str,
    dtype: int,
    shape: tuple[int, ...],
    row_before: int,
    row_after: int,
    col_before: int,
    col_after: int,
    prefix: str,
) -> str:
    assert dtype == TensorProto.BOOL
    inputs = [x]
    axis = 3 if col_before or col_after else 2
    if col_before:
        inputs.insert(0, b.false_block_name((shape[0], shape[1], shape[2], col_before)))
    if col_after:
        inputs.append(b.false_block_name((shape[0], shape[1], shape[2], col_after)))
    if row_before:
        inputs.insert(0, b.false_block_name((shape[0], shape[1], row_before, shape[3])))
    if row_after:
        inputs.append(b.false_block_name((shape[0], shape[1], row_after, shape[3])))
    return b.node("Concat", inputs, dtype, shape, prefix, axis=axis)


def gated(b: Builder, x: str, cond: str, shape: tuple[int, ...], prefix: str) -> str:
    return b.node("And", [x, cond], TensorProto.BOOL, shape, prefix)


def build_model() -> onnx.ModelProto:
    b = Builder()
    make_const_initializers(b)

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])

    crop = b.node(
        "Slice",
        ["input", "crop_starts", "crop_ends", "crop_axes"],
        TensorProto.FLOAT,
        (1, 10, 10, 10),
        "crop",
    )
    colors_i = b.node("ArgMax", [crop], TensorProto.INT64, (1, 10, 10), "colors_i", axis=1, keepdims=0)
    colors = b.node("Cast", [colors_i], TensorProto.UINT8, (1, 10, 10), "colors", to=TensorProto.UINT8)
    flat = b.node("Reshape", [colors, "shape_100"], TensorProto.UINT8, (100,), "flat")

    rect_row3 = b.node(
        "Slice",
        [crop, "rect_row3_starts", "rect_row3_ends", "rect_row3_axes"],
        TensorProto.FLOAT,
        (1, 1, 1, 10),
        "rect_row3",
    )
    left_raw = b.node("ArgMax", [rect_row3], TensorProto.INT64, (1, 1, 1), "left_raw", axis=3, keepdims=0)
    left = b.node("Reshape", [left_raw, "shape_1"], TensorProto.INT64, (1,), "left")

    rect_col5 = b.node(
        "Slice",
        [crop, "rect_col5_starts", "rect_col5_ends", "rect_col5_axes"],
        TensorProto.FLOAT,
        (1, 1, 10, 1),
        "rect_col5",
    )
    weighted_rows = b.node("Mul", [rect_col5, "row_weights"], TensorProto.FLOAT, (1, 1, 10, 1), "weighted_rows")
    bottom_f = b.node("ReduceMax", [weighted_rows], TensorProto.FLOAT, (1, 1, 1), "bottom_f", axes=[2], keepdims=0)
    bottom_i = b.node("Cast", [bottom_f], TensorProto.INT64, (1, 1, 1), "bottom_i", to=TensorProto.INT64)
    bottom = b.node("Reshape", [bottom_i, "shape_1"], TensorProto.INT64, (1,), "bottom")
    bottom_base = b.node("Mul", [bottom, "ten_i"], TensorProto.INT64, (1,), "bottom_base")

    def line(axis: int, start: str, end: str, shape: tuple[int, ...], prefix: str) -> str:
        sliced = slice_axis(b, colors, TensorProto.UINT8, (1, 10, 10), axis, start, end, shape, f"{prefix}_slice")
        return b.node("Reshape", [sliced, "shape_10"], TensorProto.UINT8, (10,), prefix)

    left_colors = line(2, "col0_starts", "col0_ends", (1, 10, 1), "left_colors")
    right_colors = line(2, "col9_starts", "col9_ends", (1, 10, 1), "right_colors")
    top_colors = line(1, "row0_starts", "row0_ends", (1, 1, 10), "top_colors")
    bottom_colors = line(1, "row9_starts", "row9_ends", (1, 1, 10), "bottom_colors")

    left_idx = b.node("Add", ["row10", left], TensorProto.INT64, (10,), "left_idx")
    bottom_idx = b.node("Add", [bottom_base, "cols"], TensorProto.INT64, (10,), "bottom_idx")

    def scatter_markers(data: str, idx: str, src_colors: str, prefix: str) -> str:
        original = b.node("Gather", [data, idx], TensorProto.UINT8, (10,), f"{prefix}_original", axis=0)
        marker = b.node("Greater", [src_colors, "zero_u8"], TensorProto.BOOL, (10,), f"{prefix}_marker")
        updates = b.node("Where", [marker, src_colors, original], TensorProto.UINT8, (10,), f"{prefix}_updates")
        return b.node("ScatterElements", [data, idx, updates], TensorProto.UINT8, (100,), prefix, axis=0)

    s1 = scatter_markers(flat, left_idx, left_colors, "scatter_left")
    s2 = scatter_markers(s1, "right_idx", right_colors, "scatter_right")
    s3 = scatter_markers(s2, "top_idx", top_colors, "scatter_top")
    s4 = scatter_markers(s3, bottom_idx, bottom_colors, "scatter_bottom")

    solved_grid = b.node("Reshape", [s4, "shape_grid"], TensorProto.UINT8, (1, 1, 10, 10), "solved_grid")
    solved_bool = b.node("Equal", ["color_ids", solved_grid], TensorProto.BOOL, (1, 10, 10, 10), "solved_bool")
    final_float = b.node("Cast", [solved_bool], TensorProto.FLOAT, (1, 10, 10, 10), "final_float", to=TensorProto.FLOAT)
    b.nodes.append(helper.make_node("Pad", [final_float, "pad30", "zero_f"], ["output"], mode="constant"))

    graph = helper.make_graph(b.nodes, TASK_NAME, [inp], [out], b.initializers, value_info=b.value_infos)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 11)])
    model.ir_version = 10
    return model


def load_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def validate(model_path: Path, data: dict[str, list[dict[str, list[list[int]]]]]) -> bool:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    ok = True
    for split, examples in data.items():
        for idx, ex in enumerate(examples):
            inp = convert_to_numpy(ex, "input")
            expected = convert_to_numpy(ex, "output")
            if inp is None or expected is None:
                continue
            pred = sess.run(["output"], {"input": inp})[0]
            match = np.array_equal(pred > 0, expected > 0)
            ok = ok and match
            if split == "train":
                pred_grid = grid_from_onehot(pred)
                exp_grid = np.array(ex["output"], dtype=np.int64)
                print(f"train[{idx}] match={match}")
                print("predicted:")
                print(pred_grid)
                print("expected:")
                print(exp_grid)
            elif not match:
                print(f"{split}[{idx}] failed")
    return ok


def main() -> None:
    data = load_data()
    model = build_model()
    onnx.checker.check_model(model, full_check=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUT_PATH)
    valid = validate(OUT_PATH, data)
    with tempfile.TemporaryDirectory() as tmpdir:
        score_path = Path(tmpdir) / f"{TASK_NAME}.onnx"
        onnx.save(model, score_path)
        stats = score_file(score_path)
    print(f"valid_examples={valid}")
    print(
        "score "
        f"memory={stats['memory']} params={stats['params']} cost={stats['cost']} "
        f"points={stats['score'] if stats['score'] is not None else None}"
    )
    if stats["cost"] is not None:
        print(f"manual_points={max(1.0, 25.0 - math.log(max(1.0, stats['cost']))):.6f}")


if __name__ == "__main__":
    main()
