"""ONNX solver for NeuroGolf task121: return the marked 3x3 object.

Task rule: the 13x13 input contains several disconnected 3x3, mostly
monochrome plus/cross-like objects. Exactly one object has its center cell
replaced by color 8. Output that object's 3x3 bounding box in the top-left of
the competition output tensor, replacing the marker with the object's true
color and leaving cells outside the 3x3 task output all-zero.

All available train/test/arc-gen examples place the marker at the center of the
selected 3x3 object, so the cheapest valid graph crops the 3x3 window centered
on the marker and reconstructs the center color from the neighboring cells.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import calculate_params, score, score_file  # noqa: E402

TASK_ID = "task121"
TASK_NUM = 121
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
ROOT_BEST_PATH = ROOT / f"{TASK_ID}_best.onnx"

IN_NAME = "input"
OUT_NAME = "output"
C = 10
H = W = 30
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: list[onnx.TensorProto], name: str, arr: Any, dtype: np.dtype) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=dtype), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    return _init(inits, name, arr, np.int64)


def _f32(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    return _init(inits, name, arr, np.float32)


def _bool(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    return _init(inits, name, arr, np.bool_)


def grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def expected_onehot(grid: list[list[int]]) -> np.ndarray:
    return grid_to_onehot(grid)


def load_examples() -> list[tuple[str, int, dict[str, list[list[int]]]]]:
    data = json.loads(DATA_PATH.read_text())
    return [
        (split, idx, example)
        for split in ("train", "test", "arc-gen")
        for idx, example in enumerate(data.get(split, []))
    ]


def infer_io_shapes(model: onnx.ModelProto) -> tuple[list[int], list[int]]:
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph

    def dims(name: str) -> list[int]:
        for item in list(graph.input) + list(graph.value_info) + list(graph.output):
            if item.name == name:
                return [int(dim.dim_value) for dim in item.type.tensor_type.shape.dim]
        raise KeyError(name)

    return dims(IN_NAME), dims(OUT_NAME)


def verify_model(model_path: Path) -> tuple[bool, str]:
    sess = ort.InferenceSession(
        str(model_path),
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    )
    for split, idx, example in load_examples():
        actual = (sess.run([OUT_NAME], {IN_NAME: grid_to_onehot(example["input"])})[0] > 0).astype(np.float32)
        expected = expected_onehot(example["output"])
        if not np.array_equal(actual, expected):
            return False, f"{split}[{idx}] failed"
    return True, "all examples passed"


def make_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_centered_slice_model() -> onnx.ModelProto:
    """Crop the 3x3 window centered on the unique color-8 marker."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    axes4 = _i64(inits, "axes4", [0, 1, 2, 3])
    marker_st = _i64(inits, "marker_st", [0, 8, 0, 0])
    marker_en = _i64(inits, "marker_en", [1, 9, H, W])
    flat900 = _i64(inits, "flat900", [900])
    thirty = _i64(inits, "thirty", [30])
    one_i = _i64(inits, "one_i", [1])
    three_i = _i64(inits, "three_i", [3])
    zero2 = _i64(inits, "zero2", [0, 0])
    one2 = _i64(inits, "one2", [1, C])
    center = _bool(
        inits,
        "center",
        np.asarray([[[[False, False, False], [False, True, False], [False, False, False]]]]),
    )
    color_mask = _f32(inits, "color_mask", [[[[0]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[0]], [[1]]]])
    pads = [0, 0, 0, 0, 0, 0, H - 3, W - 3]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, marker_st, marker_en, axes4], ["marker_plane"]),
            helper.make_node("Reshape", ["marker_plane", flat900], ["marker_flat"]),
            helper.make_node("ArgMax", ["marker_flat"], ["marker_idx"], axis=0, keepdims=1),
            helper.make_node("Div", ["marker_idx", thirty], ["marker_row"]),
            helper.make_node("Mod", ["marker_idx", thirty], ["marker_col"]),
            helper.make_node("Sub", ["marker_row", one_i], ["r0"]),
            helper.make_node("Sub", ["marker_col", one_i], ["c0"]),
            helper.make_node("Add", ["r0", three_i], ["r3"]),
            helper.make_node("Add", ["c0", three_i], ["c3"]),
            helper.make_node("Concat", [zero2, "r0", "c0"], ["crop_st"], axis=0),
            helper.make_node("Concat", [one2, "r3", "c3"], ["crop_en"], axis=0),
            helper.make_node("Slice", [IN_NAME, "crop_st", "crop_en", axes4], ["crop"]),
            helper.make_node("ReduceMax", ["crop"], ["crop_vec"], axes=[2, 3], keepdims=1),
            helper.make_node("Mul", ["crop_vec", color_mask], ["color_vec"]),
            helper.make_node("Where", [center, "color_vec", "crop"], ["out3"]),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=pads),
        ]
    )
    return make_model(nodes, inits, "task121_centered_slice")


def build_centered_gather_flat_model() -> onnx.ModelProto:
    """Alternative: gather the 3x3 crop from a flattened full-grid tensor."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    axes4 = _i64(inits, "axes4", [0, 1, 2, 3])
    marker_st = _i64(inits, "marker_st", [0, 8, 0, 0])
    marker_en = _i64(inits, "marker_en", [1, 9, H, W])
    flat900 = _i64(inits, "flat900", [900])
    flat_grid = _i64(inits, "flat_grid", [C, 900])
    offsets = _i64(inits, "offsets", [-31, -30, -29, -1, 0, 1, 29, 30, 31])
    crop_shape = _i64(inits, "crop_shape", [1, C, 3, 3])
    center = _bool(
        inits,
        "center",
        np.asarray([[[[False, False, False], [False, True, False], [False, False, False]]]]),
    )
    color_mask = _f32(inits, "color_mask", [[[[0]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[0]], [[1]]]])
    pads = [0, 0, 0, 0, 0, 0, H - 3, W - 3]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, marker_st, marker_en, axes4], ["marker_plane"]),
            helper.make_node("Reshape", ["marker_plane", flat900], ["marker_flat"]),
            helper.make_node("ArgMax", ["marker_flat"], ["marker_idx"], axis=0, keepdims=0),
            helper.make_node("Add", ["marker_idx", offsets], ["crop_idx"]),
            helper.make_node("Reshape", [IN_NAME, flat_grid], ["input_flat"]),
            helper.make_node("Gather", ["input_flat", "crop_idx"], ["crop_flat"], axis=1),
            helper.make_node("Reshape", ["crop_flat", crop_shape], ["crop"]),
            helper.make_node("Mul", ["crop", color_mask], ["colored_crop"]),
            helper.make_node("ReduceMax", ["colored_crop"], ["color_vec"], axes=[2, 3], keepdims=1),
            helper.make_node("Where", [center, "color_vec", "crop"], ["out3"]),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=pads),
        ]
    )
    return make_model(nodes, inits, "task121_centered_gather_flat")


def build_centered_gather_13_bool_model() -> onnx.ModelProto:
    """Flatten only the 13x13 task area and gather the centered 3x3 crop as bool."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    axes4 = _i64(inits, "axes4", [0, 1, 2, 3])
    marker_st = _i64(inits, "marker_st", [0, 8, 0, 0])
    marker_en = _i64(inits, "marker_en", [1, 9, 13, 13])
    input13_st = _i64(inits, "input13_st", [0, 0, 0, 0])
    input13_en = _i64(inits, "input13_en", [1, C, 13, 13])
    flat169 = _i64(inits, "flat169", [169])
    flat_grid = _i64(inits, "flat_grid", [C, 169])
    offsets = _i64(inits, "offsets", [-14, -13, -12, -1, 0, 1, 12, 13, 14])
    crop_shape = _i64(inits, "crop_shape", [1, C, 3, 3])
    zero = _f32(inits, "zero", [0.0])
    center = _bool(
        inits,
        "center",
        np.asarray([[[[False, False, False], [False, True, False], [False, False, False]]]]),
    )
    color_mask = _f32(inits, "color_mask", [[[[0]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[0]], [[1]]]])
    pads = [0, 0, 0, 0, 0, 0, H - 3, W - 3]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, marker_st, marker_en, axes4], ["marker_plane"]),
            helper.make_node("Reshape", ["marker_plane", flat169], ["marker_flat"]),
            helper.make_node("ArgMax", ["marker_flat"], ["marker_idx"], axis=0, keepdims=0),
            helper.make_node("Add", ["marker_idx", offsets], ["crop_idx"]),
            helper.make_node("Slice", [IN_NAME, input13_st, input13_en, axes4], ["input13"]),
            helper.make_node("Greater", ["input13", zero], ["input13b"]),
            helper.make_node("Reshape", ["input13b", flat_grid], ["input_flatb"]),
            helper.make_node("Gather", ["input_flatb", "crop_idx"], ["crop_flatb"], axis=1),
            helper.make_node("Reshape", ["crop_flatb", crop_shape], ["cropb"]),
            helper.make_node("Cast", ["cropb"], ["crop"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["crop", color_mask], ["colored_crop"]),
            helper.make_node("ReduceMax", ["colored_crop"], ["color_vec"], axes=[2, 3], keepdims=1),
            helper.make_node("Where", [center, "color_vec", "crop"], ["out3"]),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=pads),
        ]
    )
    return make_model(nodes, inits, "task121_centered_gather_13_bool")


def build_centered_gather_13_float_model() -> onnx.ModelProto:
    """Flatten only the 13x13 task area and gather the centered 3x3 crop as float."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    axes4 = _i64(inits, "axes4", [0, 1, 2, 3])
    marker_st = _i64(inits, "marker_st", [0, 8, 0, 0])
    marker_en = _i64(inits, "marker_en", [1, 9, 13, 13])
    input13_st = _i64(inits, "input13_st", [0, 0, 0, 0])
    input13_en = _i64(inits, "input13_en", [1, C, 13, 13])
    flat169 = _i64(inits, "flat169", [169])
    flat_grid = _i64(inits, "flat_grid", [C, 169])
    offsets = _i64(inits, "offsets", [-14, -13, -12, -1, 0, 1, 12, 13, 14])
    crop_shape = _i64(inits, "crop_shape", [1, C, 3, 3])
    center = _bool(
        inits,
        "center",
        np.asarray([[[[False, False, False], [False, True, False], [False, False, False]]]]),
    )
    color_mask = _f32(inits, "color_mask", [[[[0]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[0]], [[1]]]])
    pads = [0, 0, 0, 0, 0, 0, H - 3, W - 3]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, marker_st, marker_en, axes4], ["marker_plane"]),
            helper.make_node("Reshape", ["marker_plane", flat169], ["marker_flat"]),
            helper.make_node("ArgMax", ["marker_flat"], ["marker_idx"], axis=0, keepdims=0),
            helper.make_node("Add", ["marker_idx", offsets], ["crop_idx"]),
            helper.make_node("Slice", [IN_NAME, input13_st, input13_en, axes4], ["input13"]),
            helper.make_node("Reshape", ["input13", flat_grid], ["input_flat"]),
            helper.make_node("Gather", ["input_flat", "crop_idx"], ["crop_flat"], axis=1),
            helper.make_node("Reshape", ["crop_flat", crop_shape], ["crop"]),
            helper.make_node("Mul", ["crop", color_mask], ["colored_crop"]),
            helper.make_node("ReduceMax", ["colored_crop"], ["color_vec"], axes=[2, 3], keepdims=1),
            helper.make_node("Where", [center, "color_vec", "crop"], ["out3"]),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=pads),
        ]
    )
    return make_model(nodes, inits, "task121_centered_gather_13_float")


def build_centered_gather_13_bool_reused_marker_model() -> onnx.ModelProto:
    """Best variant: locate color 8 from the bool 13x13 flat tensor reused for crop gather."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    axes4 = _i64(inits, "axes4", [0, 1, 2, 3])
    input13_st = _i64(inits, "input13_st", [0, 0, 0, 0])
    input13_en = _i64(inits, "input13_en", [1, C, 13, 13])
    flat_grid = _i64(inits, "flat_grid", [C, 169])
    color8 = _i64(inits, "color8", np.asarray(8))
    offsets = _i64(inits, "offsets", [-14, -13, -12, -1, 0, 1, 12, 13, 14])
    crop_shape = _i64(inits, "crop_shape", [1, C, 3, 3])
    zero = _f32(inits, "zero", [0.0])
    center = _bool(
        inits,
        "center",
        np.asarray([[[[False, False, False], [False, True, False], [False, False, False]]]]),
    )
    color_mask = _f32(inits, "color_mask", [[[[0]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[0]], [[1]]]])
    pads = [0, 0, 0, 0, 0, 0, H - 3, W - 3]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, input13_st, input13_en, axes4], ["input13"]),
            helper.make_node("Greater", ["input13", zero], ["input13b"]),
            helper.make_node("Reshape", ["input13b", flat_grid], ["input_flatb"]),
            helper.make_node("Gather", ["input_flatb", color8], ["marker_flatb"], axis=0),
            helper.make_node("Cast", ["marker_flatb"], ["marker_flat"], to=TensorProto.FLOAT),
            helper.make_node("ArgMax", ["marker_flat"], ["marker_idx"], axis=0, keepdims=0),
            helper.make_node("Add", ["marker_idx", offsets], ["crop_idx"]),
            helper.make_node("Gather", ["input_flatb", "crop_idx"], ["crop_flatb"], axis=1),
            helper.make_node("Reshape", ["crop_flatb", crop_shape], ["cropb"]),
            helper.make_node("Cast", ["cropb"], ["crop"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["crop", color_mask], ["colored_crop"]),
            helper.make_node("ReduceMax", ["colored_crop"], ["color_vec"], axes=[2, 3], keepdims=1),
            helper.make_node("Where", [center, "color_vec", "crop"], ["out3"]),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=pads),
        ]
    )
    return make_model(nodes, inits, "task121_centered_gather_13_bool_reused_marker")


def build_centered_gather_13_foreground_bool_model() -> onnx.ModelProto:
    """Gather only foreground channels 1..9, then reconstruct background channel 0."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    axes4 = _i64(inits, "axes4", [0, 1, 2, 3])
    fg_st = _i64(inits, "fg_st", [0, 1, 0, 0])
    fg_en = _i64(inits, "fg_en", [1, C, 13, 13])
    flat_grid = _i64(inits, "flat_grid", [9, 169])
    marker_ch = _i64(inits, "marker_ch", np.asarray(7))
    offsets = _i64(inits, "offsets", [-14, -13, -12, -1, 0, 1, 12, 13, 14])
    crop9_shape = _i64(inits, "crop9_shape", [1, 9, 3, 3])
    one = _f32(inits, "one", [1.0])
    zero_color = _f32(inits, "zero_color", [[[[0.0]]]])
    center = _bool(
        inits,
        "center",
        np.asarray([[[[False, False, False], [False, True, False], [False, False, False]]]]),
    )
    color_mask9 = _f32(inits, "color_mask9", [[[[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[0]], [[1]]]])
    pads = [0, 0, 0, 0, 0, 0, H - 3, W - 3]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, fg_st, fg_en, axes4], ["input13fg"]),
            helper.make_node("Cast", ["input13fg"], ["input13fgb"], to=TensorProto.BOOL),
            helper.make_node("Reshape", ["input13fgb", flat_grid], ["input_flatb"]),
            helper.make_node("Gather", ["input_flatb", marker_ch], ["marker_flatb"], axis=0),
            helper.make_node("Cast", ["marker_flatb"], ["marker_flat"], to=TensorProto.FLOAT),
            helper.make_node("ArgMax", ["marker_flat"], ["marker_idx"], axis=0, keepdims=0),
            helper.make_node("Add", ["marker_idx", offsets], ["crop_idx"]),
            helper.make_node("Gather", ["input_flatb", "crop_idx"], ["crop9_flatb"], axis=1),
            helper.make_node("Reshape", ["crop9_flatb", crop9_shape], ["crop9b"]),
            helper.make_node("Cast", ["crop9b"], ["crop9"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", ["crop9"], ["fg_any"], axes=[1], keepdims=1),
            helper.make_node("Sub", [one, "fg_any"], ["bg"]),
            helper.make_node("Concat", ["bg", "crop9"], ["crop"], axis=1),
            helper.make_node("Mul", ["crop9", color_mask9], ["colored_crop9"]),
            helper.make_node("ReduceMax", ["colored_crop9"], ["color_vec9"], axes=[2, 3], keepdims=1),
            helper.make_node("Concat", [zero_color, "color_vec9"], ["color_vec"], axis=1),
            helper.make_node("Where", [center, "color_vec", "crop"], ["out3"]),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=pads),
        ]
    )
    return make_model(nodes, inits, "task121_centered_gather_13_foreground_bool")


def build_centered_gather_13_foreground_u8_model() -> onnx.ModelProto:
    """Variant: locate the marker with uint8 ArgMax, then use compact 3x3 float logic."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    axes4 = _i64(inits, "axes4", [0, 1, 2, 3])
    fg_st = _i64(inits, "fg_st", [0, 1, 0, 0])
    fg_en = _i64(inits, "fg_en", [1, C, 13, 13])
    flat_grid = _i64(inits, "flat_grid", [9, 169])
    marker_ch = _i64(inits, "marker_ch", np.asarray(7))
    offsets = _i64(inits, "offsets", [-14, -13, -12, -1, 0, 1, 12, 13, 14])
    crop9_shape = _i64(inits, "crop9_shape", [1, 9, 3, 3])
    one = _f32(inits, "one", [1.0])
    zero_color = _f32(inits, "zero_color", [[[[0.0]]]])
    center = _bool(
        inits,
        "center",
        np.asarray([[[[False, False, False], [False, True, False], [False, False, False]]]]),
    )
    color_mask9 = _f32(inits, "color_mask9", [[[[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[0]], [[1]]]])
    pads = [0, 0, 0, 0, 0, 0, H - 3, W - 3]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, fg_st, fg_en, axes4], ["input13fg"]),
            helper.make_node("Cast", ["input13fg"], ["input13fgb"], to=TensorProto.BOOL),
            helper.make_node("Reshape", ["input13fgb", flat_grid], ["input_flatb"]),
            helper.make_node("Gather", ["input_flatb", marker_ch], ["marker_flatb"], axis=0),
            helper.make_node("Cast", ["marker_flatb"], ["marker_flat_u8"], to=TensorProto.UINT8),
            helper.make_node("ArgMax", ["marker_flat_u8"], ["marker_idx"], axis=0, keepdims=0),
            helper.make_node("Add", ["marker_idx", offsets], ["crop_idx"]),
            helper.make_node("Gather", ["input_flatb", "crop_idx"], ["crop9_flatb"], axis=1),
            helper.make_node("Reshape", ["crop9_flatb", crop9_shape], ["crop9b"]),
            helper.make_node("Cast", ["crop9b"], ["crop9"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", ["crop9"], ["fg_any"], axes=[1], keepdims=1),
            helper.make_node("Sub", [one, "fg_any"], ["bg"]),
            helper.make_node("Concat", ["bg", "crop9"], ["crop"], axis=1),
            helper.make_node("Mul", ["crop9", color_mask9], ["colored_crop9"]),
            helper.make_node("ReduceMax", ["colored_crop9"], ["color_vec9"], axes=[2, 3], keepdims=1),
            helper.make_node("Concat", [zero_color, "color_vec9"], ["color_vec"], axis=1),
            helper.make_node("Where", [center, "color_vec", "crop"], ["out3"]),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=pads),
        ]
    )
    return make_model(nodes, inits, "task121_centered_gather_13_foreground_u8")


def build_marker_plane_row_col_gather_model() -> onnx.ModelProto:
    """Locate the color-8 marker in 13x13, then gather its 3 rows and columns."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    axes4 = _i64(inits, "axes4", [0, 1, 2, 3])
    marker_st = _i64(inits, "marker_st", [0, 8, 0, 0])
    marker_en = _i64(inits, "marker_en", [1, 9, 13, 13])
    flat169 = _i64(inits, "flat169", [169])
    thirteen = _i64(inits, "thirteen", [13])
    offsets3 = _i64(inits, "offsets3", [-1, 0, 1])
    center = _bool(
        inits,
        "center",
        np.asarray([[[[False, False, False], [False, True, False], [False, False, False]]]]),
    )
    color_mask = _f32(inits, "color_mask", [[[[0]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[0]], [[1]]]])
    pads = [0, 0, 0, 0, 0, 0, H - 3, W - 3]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, marker_st, marker_en, axes4], ["marker_plane"]),
            helper.make_node("Reshape", ["marker_plane", flat169], ["marker_flat"]),
            helper.make_node("ArgMax", ["marker_flat"], ["marker_idx"], axis=0, keepdims=0),
            helper.make_node("Div", ["marker_idx", thirteen], ["marker_row"]),
            helper.make_node("Mod", ["marker_idx", thirteen], ["marker_col"]),
            helper.make_node("Add", ["marker_row", offsets3], ["row_idx"]),
            helper.make_node("Add", ["marker_col", offsets3], ["col_idx"]),
            helper.make_node("Gather", [IN_NAME, "row_idx"], ["rows"], axis=2),
            helper.make_node("Gather", ["rows", "col_idx"], ["crop"], axis=3),
            helper.make_node("Mul", ["crop", color_mask], ["colored_crop"]),
            helper.make_node("ReduceMax", ["colored_crop"], ["color_vec"], axes=[2, 3], keepdims=1),
            helper.make_node("Where", [center, "color_vec", "crop"], ["out3"]),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=pads),
        ]
    )
    return make_model(nodes, inits, "task121_marker_plane_row_col_gather")


def build_dynamic_slice_static_vi_model() -> onnx.ModelProto:
    """Crop the dynamic 3x3 marker window, with static value_info for scoring."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    axes4 = _i64(inits, "axes4", [0, 1, 2, 3])
    marker_st = _i64(inits, "marker_st", [0, 8, 0, 0])
    marker_en = _i64(inits, "marker_en", [1, 9, 13, 13])
    flat169 = _i64(inits, "flat169", [169])
    thirteen = _i64(inits, "thirteen", [13])
    one_i = _i64(inits, "one_i", [1])
    three_i = _i64(inits, "three_i", [3])
    zero2 = _i64(inits, "zero2", [0, 0])
    one2 = _i64(inits, "one2", [1, C])
    center = _bool(
        inits,
        "center",
        np.asarray([[[[False, False, False], [False, True, False], [False, False, False]]]]),
    )
    color_mask = _f32(inits, "color_mask", [[[[0]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[0]], [[1]]]])
    pads = [0, 0, 0, 0, 0, 0, H - 3, W - 3]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, marker_st, marker_en, axes4], ["marker_plane"]),
            helper.make_node("Cast", ["marker_plane"], ["marker_plane_u8"], to=TensorProto.UINT8),
            helper.make_node("Reshape", ["marker_plane_u8", flat169], ["marker_flat_u8"]),
            helper.make_node("ArgMax", ["marker_flat_u8"], ["marker_idx"], axis=0, keepdims=0),
            helper.make_node("Div", ["marker_idx", thirteen], ["marker_row"]),
            helper.make_node("Mod", ["marker_idx", thirteen], ["marker_col"]),
            helper.make_node("Sub", ["marker_row", one_i], ["r0"]),
            helper.make_node("Sub", ["marker_col", one_i], ["c0"]),
            helper.make_node("Add", ["r0", three_i], ["r3"]),
            helper.make_node("Add", ["c0", three_i], ["c3"]),
            helper.make_node("Concat", [zero2, "r0", "c0"], ["crop_st"], axis=0),
            helper.make_node("Concat", [one2, "r3", "c3"], ["crop_en"], axis=0),
            helper.make_node("Slice", [IN_NAME, "crop_st", "crop_en", axes4], ["crop"]),
            helper.make_node("ReduceMax", ["crop"], ["crop_vec"], axes=[2, 3], keepdims=1),
            helper.make_node("Mul", ["crop_vec", color_mask], ["color_vec"]),
            helper.make_node("Where", [center, "color_vec", "crop"], ["out3"]),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=pads),
        ]
    )
    model = make_model(nodes, inits, "task121_dynamic_slice_static_vi")
    for name, elem_type, shape in [
        ("marker_plane_u8", TensorProto.UINT8, [1, 1, 13, 13]),
        ("marker_flat_u8", TensorProto.UINT8, [169]),
        ("marker_row", TensorProto.INT64, [1]),
        ("marker_col", TensorProto.INT64, [1]),
        ("r0", TensorProto.INT64, [1]),
        ("c0", TensorProto.INT64, [1]),
        ("r3", TensorProto.INT64, [1]),
        ("c3", TensorProto.INT64, [1]),
        ("crop_st", TensorProto.INT64, [4]),
        ("crop_en", TensorProto.INT64, [4]),
        ("crop", TensorProto.FLOAT, [1, C, 3, 3]),
        ("crop_vec", TensorProto.FLOAT, [1, C, 1, 1]),
        ("color_vec", TensorProto.FLOAT, [1, C, 1, 1]),
        ("out3", TensorProto.FLOAT, [1, C, 3, 3]),
    ]:
        model.graph.value_info.append(helper.make_tensor_value_info(name, elem_type, shape))
    onnx.checker.check_model(model)
    return model


def build_dynamic_slice_row_col_reduce_model() -> onnx.ModelProto:
    """Best variant: find marker row/column by reducing the 13x13 marker plane."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    axes4 = _i64(inits, "axes4", [0, 1, 2, 3])
    marker_st = _i64(inits, "marker_st", [0, 8, 0, 0])
    marker_en = _i64(inits, "marker_en", [1, 9, 13, 13])
    one_i = _i64(inits, "one_i", [1])
    three_i = _i64(inits, "three_i", [3])
    zero2 = _i64(inits, "zero2", [0, 0])
    one2 = _i64(inits, "one2", [1, C])
    center = _bool(
        inits,
        "center",
        np.asarray([[[[False, False, False], [False, True, False], [False, False, False]]]]),
    )
    color_mask = _f32(inits, "color_mask", [[[[0]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[0]], [[1]]]])
    pads = [0, 0, 0, 0, 0, 0, H - 3, W - 3]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, marker_st, marker_en, axes4], ["marker_plane"]),
            helper.make_node("ReduceMax", ["marker_plane"], ["marker_rows"], axes=[0, 1, 3], keepdims=0),
            helper.make_node("ReduceMax", ["marker_plane"], ["marker_cols"], axes=[0, 1, 2], keepdims=0),
            helper.make_node("ArgMax", ["marker_rows"], ["marker_row"], axis=0, keepdims=1),
            helper.make_node("ArgMax", ["marker_cols"], ["marker_col"], axis=0, keepdims=1),
            helper.make_node("Sub", ["marker_row", one_i], ["r0"]),
            helper.make_node("Sub", ["marker_col", one_i], ["c0"]),
            helper.make_node("Add", ["r0", three_i], ["r3"]),
            helper.make_node("Add", ["c0", three_i], ["c3"]),
            helper.make_node("Concat", [zero2, "r0", "c0"], ["crop_st"], axis=0),
            helper.make_node("Concat", [one2, "r3", "c3"], ["crop_en"], axis=0),
            helper.make_node("Slice", [IN_NAME, "crop_st", "crop_en", axes4], ["crop"]),
            helper.make_node("ReduceMax", ["crop"], ["crop_vec"], axes=[2, 3], keepdims=1),
            helper.make_node("Mul", ["crop_vec", color_mask], ["color_vec"]),
            helper.make_node("Where", [center, "color_vec", "crop"], ["out3"]),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=pads),
        ]
    )
    model = make_model(nodes, inits, "task121_dynamic_slice_row_col_reduce")
    for name, elem_type, shape in [
        ("marker_rows", TensorProto.FLOAT, [13]),
        ("marker_cols", TensorProto.FLOAT, [13]),
        ("marker_row", TensorProto.INT64, [1]),
        ("marker_col", TensorProto.INT64, [1]),
        ("r0", TensorProto.INT64, [1]),
        ("c0", TensorProto.INT64, [1]),
        ("r3", TensorProto.INT64, [1]),
        ("c3", TensorProto.INT64, [1]),
        ("crop_st", TensorProto.INT64, [4]),
        ("crop_en", TensorProto.INT64, [4]),
        ("crop", TensorProto.FLOAT, [1, C, 3, 3]),
        ("crop_vec", TensorProto.FLOAT, [1, C, 1, 1]),
        ("color_vec", TensorProto.FLOAT, [1, C, 1, 1]),
        ("out3", TensorProto.FLOAT, [1, C, 3, 3]),
    ]:
        model.graph.value_info.append(helper.make_tensor_value_info(name, elem_type, shape))
    onnx.checker.check_model(model)
    return model


def build_candidate_models() -> dict[str, onnx.ModelProto]:
    return {
        "A_centered_slice": build_centered_slice_model(),
        "B_centered_gather_13_bool": build_centered_gather_13_bool_model(),
        "B3_centered_gather_13_bool_reused_marker": build_centered_gather_13_bool_reused_marker_model(),
        "B4_centered_gather_13_foreground_bool": build_centered_gather_13_foreground_bool_model(),
        "B5_centered_gather_13_foreground_u8": build_centered_gather_13_foreground_u8_model(),
        "B6_marker_plane_row_col_gather": build_marker_plane_row_col_gather_model(),
        "B7_dynamic_slice_static_vi": build_dynamic_slice_static_vi_model(),
        "B8_dynamic_slice_row_col_reduce": build_dynamic_slice_row_col_reduce_model(),
        "B2_centered_gather_13_float": build_centered_gather_13_float_model(),
        "C_plane_marker_gather_flat": build_centered_gather_flat_model(),
    }


def print_candidate_report(name: str, path: Path, model: onnx.ModelProto, correct: bool, message: str) -> dict[str, Any]:
    try:
        in_shape, out_shape = infer_io_shapes(model)
    except Exception as exc:
        in_shape, out_shape = [], []
        message = f"{message}; shape inference failed: {exc}"

    result = score_file(path)
    params = calculate_params(model)
    print(f"{name}")
    print(f"  input shape:  {in_shape}")
    print(f"  output shape: {out_shape}")
    print(f"  correctness:  {message}")
    print(f"  params:       {params}")
    print(f"  memory:       {result['memory']}")
    print(f"  cost:         {result['cost']}")
    score_text = None if result["score"] is None else f"{result['score']:.6f}"
    print(f"  score:        {score_text}")
    if result["error"]:
        print(f"  error:        {str(result['error']).strip().splitlines()[-1]}")
    result["correct"] = correct
    return result


def main() -> None:
    tmp_dir = OUT_DIR / "_task121_candidates"
    tmp_dir.mkdir(exist_ok=True)

    reports: list[tuple[str, Path, dict[str, Any]]] = []
    for name, model in build_candidate_models().items():
        candidate_dir = tmp_dir / name
        candidate_dir.mkdir(exist_ok=True)
        path = candidate_dir / f"{TASK_ID}.onnx"
        onnx.save(model, path)
        correct, message = verify_model(path)
        report = print_candidate_report(name, path, model, correct, message)
        reports.append((name, path, report))

    valid = [
        (name, path, report)
        for name, path, report in reports
        if report["correct"] and report["valid"] and report["score"] is not None
    ]
    if not valid:
        raise SystemExit("no valid correct task121 model candidates")

    best_name, best_tmp, best_report = min(valid, key=lambda item: int(item[2]["cost"]))
    shutil.copyfile(best_tmp, BEST_PATH)
    shutil.copyfile(best_tmp, ROOT_BEST_PATH)

    print()
    print(f"best: {best_name}")
    print(f"saved: {BEST_PATH}")
    print(f"saved: {ROOT_BEST_PATH}")
    print(
        "best metrics: "
        f"memory={best_report['memory']} params={best_report['params']} "
        f"cost={best_report['cost']} score={score(int(best_report['cost'])):.6f}"
    )


if __name__ == "__main__":
    main()
