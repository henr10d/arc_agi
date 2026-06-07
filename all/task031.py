"""Minimal ONNX for ARC task031: crop the lone colored shape.

Task rule: the input is a black 10-12 by 12 grid with one connected non-black
shape.  Return the tight bounding-box crop of that shape, preserving its color
and any black holes/background inside the box.  Examples have crops up to 7x8.

ONNX: find the foreground row/column min and max inside a compact 12x12 window,
flatten the 10-channel crop, dynamically Gather the top-left-aligned 7x8
bounding box, mask cells beyond the true box, then pad to the fixed NeuroGolf
30x30 output.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import (  # noqa: E402
    calculate_memory,
    calculate_params,
    convert_to_numpy,
    load_task_examples,
    sanitize_model,
    score_file,
)

TASK_ID = "task031"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task031.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
SH = SW = 12
OH = 7
OW = 8
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


def solve(grid: np.ndarray) -> np.ndarray:
    """Return the tight non-zero bounding-box crop of the grid."""
    g = np.asarray(grid, dtype=np.int64)
    ys, xs = np.where(g != 0)
    return g[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _make_model(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], name: str) -> onnx.ModelProto:
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


def build_gather_model(*, compare_half: bool = False) -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    crop_st = _i64(inits, [0, 0, 0, 0], "crop_st")
    crop_en = _i64(inits, [1, C, SH, SW], "crop_en")
    fg_st = _i64(inits, [0, 1, 0, 0], "fg_st")
    fg_en = _i64(inits, [1, C, SH, SW], "fg_en")
    flat_shape = _i64(inits, [1, C, SH * SW], "flat_shape")
    rows12 = _f32(inits, np.arange(SH, dtype=np.float32).reshape(1, 1, SH, 1), "rows12")
    cols12 = _f32(inits, np.arange(SW, dtype=np.float32).reshape(1, 1, 1, SW), "cols12")
    out_rows = _f32(inits, np.arange(OH, dtype=np.float32).reshape(OH, 1), "out_rows")
    out_cols = _f32(inits, np.arange(OW, dtype=np.float32).reshape(1, OW), "out_cols")
    big = _f32(inits, [99.0], "big")
    half = _f32(inits, [0.5], "half")
    zero = _f32(inits, [0.0], "zero")
    one = _f32(inits, [1.0], "one")
    twelve = _f32(inits, [float(SW)], "twelve")
    sq4 = [0, 1, 2, 3]
    pad = [0, 0, 0, 0, 0, 0, H - OH, W - OW]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, crop_st, crop_en, axes4], ["crop"]),
            helper.make_node("Slice", [IN_NAME, fg_st, fg_en, axes4], ["fg_ch"]),
            helper.make_node("ReduceMax", ["fg_ch"], ["fg"], axes=[1], keepdims=1),
        ]
    )
    if compare_half:
        nodes.append(helper.make_node("Greater", ["fg", half], ["fgb"]))
        occ = "fgb"
    else:
        occ = "fg"

    nodes.extend(
        [
            helper.make_node("ReduceMax", [occ], ["row_occ"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", [occ], ["col_occ"], axes=[2], keepdims=1),
            helper.make_node("Mul", ["row_occ", rows12], ["row_weighted"]),
            helper.make_node("Mul", ["col_occ", cols12], ["col_weighted"]),
            helper.make_node("ReduceMax", ["row_weighted"], ["max_y4"], axes=[2], keepdims=1),
            helper.make_node("ReduceMax", ["col_weighted"], ["max_x4"], axes=[3], keepdims=1),
            helper.make_node("Greater", ["row_occ", half], ["rowb"]),
            helper.make_node("Greater", ["col_occ", half], ["colb"]),
            helper.make_node("Where", ["rowb", rows12, big], ["row_min_src"]),
            helper.make_node("Where", ["colb", cols12, big], ["col_min_src"]),
            helper.make_node("ReduceMin", ["row_min_src"], ["min_y4"], axes=[2], keepdims=1),
            helper.make_node("ReduceMin", ["col_min_src"], ["min_x4"], axes=[3], keepdims=1),
            helper.make_node("Squeeze", ["min_y4"], ["min_y"], axes=sq4),
            helper.make_node("Squeeze", ["min_x4"], ["min_x"], axes=sq4),
            helper.make_node("Squeeze", ["max_y4"], ["max_y"], axes=sq4),
            helper.make_node("Squeeze", ["max_x4"], ["max_x"], axes=sq4),
            helper.make_node("Sub", ["max_y", "min_y"], ["h0"]),
            helper.make_node("Sub", ["max_x", "min_x"], ["w0"]),
            helper.make_node("Add", ["h0", one], ["height"]),
            helper.make_node("Add", ["w0", one], ["width"]),
            helper.make_node("Less", [out_rows, "height"], ["valid_y"]),
            helper.make_node("Less", [out_cols, "width"], ["valid_x"]),
            helper.make_node("And", ["valid_y", "valid_x"], ["valid"]),
            helper.make_node("Add", [out_rows, "min_y"], ["abs_y"]),
            helper.make_node("Add", [out_cols, "min_x"], ["abs_x"]),
            helper.make_node("Mul", ["abs_y", twelve], ["row_base"]),
            helper.make_node("Add", ["row_base", "abs_x"], ["idxf"]),
            helper.make_node("Where", ["valid", "idxf", zero], ["safe_idxf"]),
            helper.make_node("Cast", ["safe_idxf"], ["idx"], to=TensorProto.INT64),
            helper.make_node("Reshape", ["crop", flat_shape], ["flat"]),
            helper.make_node("Gather", ["flat", "idx"], ["gathered"], axis=2),
            helper.make_node("Unsqueeze", ["valid"], ["valid4"], axes=[0, 1]),
            helper.make_node("Cast", ["valid4"], ["validf"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["gathered", "validf"], ["out_crop"]),
            helper.make_node("Pad", ["out_crop"], [OUT_NAME], mode="constant", pads=pad),
        ]
    )
    return _make_model(nodes, inits, f"{TASK_ID}_gather{'_half' if compare_half else ''}")


def build_mask_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    fg_st = _i64(inits, [0, 1, 0, 0], "fg_st")
    fg_en = _i64(inits, [1, C, SH, SW], "fg_en")
    flat_shape = _i64(inits, [1, 1, SH * SW], "flat_shape")
    rows12 = _f32(inits, np.arange(SH, dtype=np.float32).reshape(1, 1, SH, 1), "rows12")
    cols12 = _f32(inits, np.arange(SW, dtype=np.float32).reshape(1, 1, 1, SW), "cols12")
    out_rows = _f32(inits, np.arange(OH, dtype=np.float32).reshape(OH, 1), "out_rows")
    out_cols = _f32(inits, np.arange(OW, dtype=np.float32).reshape(1, OW), "out_cols")
    colors9 = _f32(inits, np.arange(1, C, dtype=np.float32).reshape(1, C - 1, 1, 1), "colors9")
    channels = _i64(inits, np.arange(C, dtype=np.int64).reshape(1, C, 1, 1), "channels")
    big = _f32(inits, [99.0], "big")
    half = _f32(inits, [0.5], "half")
    zero = _f32(inits, [0.0], "zero")
    zero_i = _i64(inits, [0], "zero_i")
    one = _f32(inits, [1.0], "one")
    twelve = _f32(inits, [float(SW)], "twelve")
    sq4 = [0, 1, 2, 3]
    pad = [0, 0, 0, 0, 0, 0, H - OH, W - OW]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, fg_st, fg_en, axes4], ["fg_ch"]),
            helper.make_node("ReduceMax", ["fg_ch"], ["fg"], axes=[1], keepdims=1),
            helper.make_node("ReduceMax", ["fg_ch"], ["color_occ"], axes=[2, 3], keepdims=1),
            helper.make_node("Mul", ["color_occ", colors9], ["color_weighted"]),
            helper.make_node("ReduceMax", ["color_weighted"], ["color4"], axes=[1], keepdims=1),
            helper.make_node("Squeeze", ["color4"], ["color"], axes=sq4),
            helper.make_node("ReduceMax", ["fg"], ["row_occ"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["fg"], ["col_occ"], axes=[2], keepdims=1),
            helper.make_node("Mul", ["row_occ", rows12], ["row_weighted"]),
            helper.make_node("Mul", ["col_occ", cols12], ["col_weighted"]),
            helper.make_node("ReduceMax", ["row_weighted"], ["max_y4"], axes=[2], keepdims=1),
            helper.make_node("ReduceMax", ["col_weighted"], ["max_x4"], axes=[3], keepdims=1),
            helper.make_node("Greater", ["row_occ", half], ["rowb"]),
            helper.make_node("Greater", ["col_occ", half], ["colb"]),
            helper.make_node("Where", ["rowb", rows12, big], ["row_min_src"]),
            helper.make_node("Where", ["colb", cols12, big], ["col_min_src"]),
            helper.make_node("ReduceMin", ["row_min_src"], ["min_y4"], axes=[2], keepdims=1),
            helper.make_node("ReduceMin", ["col_min_src"], ["min_x4"], axes=[3], keepdims=1),
            helper.make_node("Squeeze", ["min_y4"], ["min_y"], axes=sq4),
            helper.make_node("Squeeze", ["min_x4"], ["min_x"], axes=sq4),
            helper.make_node("Squeeze", ["max_y4"], ["max_y"], axes=sq4),
            helper.make_node("Squeeze", ["max_x4"], ["max_x"], axes=sq4),
            helper.make_node("Sub", ["max_y", "min_y"], ["h0"]),
            helper.make_node("Sub", ["max_x", "min_x"], ["w0"]),
            helper.make_node("Add", ["h0", one], ["height"]),
            helper.make_node("Add", ["w0", one], ["width"]),
            helper.make_node("Less", [out_rows, "height"], ["valid_y"]),
            helper.make_node("Less", [out_cols, "width"], ["valid_x"]),
            helper.make_node("And", ["valid_y", "valid_x"], ["valid"]),
            helper.make_node("Add", [out_rows, "min_y"], ["abs_y"]),
            helper.make_node("Add", [out_cols, "min_x"], ["abs_x"]),
            helper.make_node("Mul", ["abs_y", twelve], ["row_base"]),
            helper.make_node("Add", ["row_base", "abs_x"], ["idxf"]),
            helper.make_node("Where", ["valid", "idxf", zero], ["safe_idxf"]),
            helper.make_node("Cast", ["safe_idxf"], ["idx"], to=TensorProto.INT64),
            helper.make_node("Reshape", ["fg", flat_shape], ["flat_fg"]),
            helper.make_node("Gather", ["flat_fg", "idx"], ["gathered_fg"], axis=2),
            helper.make_node("Greater", ["gathered_fg", half], ["fgb"]),
            helper.make_node("Cast", ["color"], ["color_i"], to=TensorProto.INT64),
            helper.make_node("Where", ["fgb", "color_i", zero_i], ["color_grid"]),
            helper.make_node("Equal", [channels, "color_grid"], ["onehot"]),
            helper.make_node("Unsqueeze", ["valid"], ["valid4"], axes=[0, 1]),
            helper.make_node("And", ["onehot", "valid4"], ["onehot_valid"]),
            helper.make_node("Cast", ["onehot_valid"], ["out_crop"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out_crop"], [OUT_NAME], mode="constant", pads=pad),
        ]
    )
    return _make_model(nodes, inits, f"{TASK_ID}_mask")


def build_conv_color_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    color_st = _i64(inits, [0, 0, 0, 0], "color_st")
    color_en = _i64(inits, [1, 1, SH, SW], "color_en")
    flat_shape = _i64(inits, [1, 1, SH * SW], "flat_shape")
    rows12 = _f32(inits, np.arange(SH, dtype=np.float32).reshape(1, 1, SH, 1), "rows12")
    cols12 = _f32(inits, np.arange(SW, dtype=np.float32).reshape(1, 1, 1, SW), "cols12")
    out_rows = _f32(inits, np.arange(OH, dtype=np.float32).reshape(OH, 1), "out_rows")
    out_cols = _f32(inits, np.arange(OW, dtype=np.float32).reshape(1, OW), "out_cols")
    channels = _i64(inits, np.arange(C, dtype=np.int64).reshape(1, C, 1, 1), "channels")
    weight = _f32(inits, np.arange(C, dtype=np.float32).reshape(1, C, 1, 1), "color_weight")
    big = _f32(inits, [99.0], "big")
    half = _f32(inits, [0.5], "half")
    zero = _f32(inits, [0.0], "zero")
    zero_i = _i64(inits, [0], "zero_i")
    one = _f32(inits, [1.0], "one")
    twelve = _f32(inits, [float(SW)], "twelve")
    sq4 = [0, 1, 2, 3]
    pad = [0, 0, 0, 0, 0, 0, H - OH, W - OW]

    nodes.extend(
        [
            helper.make_node("Conv", [IN_NAME, weight], ["color30"], kernel_shape=[1, 1]),
            helper.make_node("Slice", ["color30", color_st, color_en, axes4], ["color12"]),
            helper.make_node("ReduceMax", ["color12"], ["row_occ"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["color12"], ["col_occ"], axes=[2], keepdims=1),
            helper.make_node("Greater", ["row_occ", half], ["rowb"]),
            helper.make_node("Greater", ["col_occ", half], ["colb"]),
            helper.make_node("Where", ["rowb", rows12, zero], ["row_max_src"]),
            helper.make_node("Where", ["colb", cols12, zero], ["col_max_src"]),
            helper.make_node("ReduceMax", ["row_max_src"], ["max_y4"], axes=[2], keepdims=1),
            helper.make_node("ReduceMax", ["col_max_src"], ["max_x4"], axes=[3], keepdims=1),
            helper.make_node("Where", ["rowb", rows12, big], ["row_min_src"]),
            helper.make_node("Where", ["colb", cols12, big], ["col_min_src"]),
            helper.make_node("ReduceMin", ["row_min_src"], ["min_y4"], axes=[2], keepdims=1),
            helper.make_node("ReduceMin", ["col_min_src"], ["min_x4"], axes=[3], keepdims=1),
            helper.make_node("Squeeze", ["min_y4"], ["min_y"], axes=sq4),
            helper.make_node("Squeeze", ["min_x4"], ["min_x"], axes=sq4),
            helper.make_node("Squeeze", ["max_y4"], ["max_y"], axes=sq4),
            helper.make_node("Squeeze", ["max_x4"], ["max_x"], axes=sq4),
            helper.make_node("Sub", ["max_y", "min_y"], ["h0"]),
            helper.make_node("Sub", ["max_x", "min_x"], ["w0"]),
            helper.make_node("Add", ["h0", one], ["height"]),
            helper.make_node("Add", ["w0", one], ["width"]),
            helper.make_node("Less", [out_rows, "height"], ["valid_y"]),
            helper.make_node("Less", [out_cols, "width"], ["valid_x"]),
            helper.make_node("And", ["valid_y", "valid_x"], ["valid"]),
            helper.make_node("Add", [out_rows, "min_y"], ["abs_y"]),
            helper.make_node("Add", [out_cols, "min_x"], ["abs_x"]),
            helper.make_node("Mul", ["abs_y", twelve], ["row_base"]),
            helper.make_node("Add", ["row_base", "abs_x"], ["idxf"]),
            helper.make_node("Where", ["valid", "idxf", zero], ["safe_idxf"]),
            helper.make_node("Cast", ["safe_idxf"], ["idx"], to=TensorProto.INT64),
            helper.make_node("Reshape", ["color12", flat_shape], ["flat_color"]),
            helper.make_node("Gather", ["flat_color", "idx"], ["gathered_color"], axis=2),
            helper.make_node("Cast", ["gathered_color"], ["color_grid"], to=TensorProto.INT64),
            helper.make_node("Where", ["valid", "color_grid", zero_i], ["safe_color_grid"]),
            helper.make_node("Equal", [channels, "safe_color_grid"], ["onehot"]),
            helper.make_node("Unsqueeze", ["valid"], ["valid4"], axes=[0, 1]),
            helper.make_node("And", ["onehot", "valid4"], ["onehot_valid"]),
            helper.make_node("Cast", ["onehot_valid"], ["out_crop"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out_crop"], [OUT_NAME], mode="constant", pads=pad),
        ]
    )
    return _make_model(nodes, inits, f"{TASK_ID}_conv_color")


def _load_examples() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _check_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return False, {}
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        sanitized.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    examples = _load_examples()
    counts: dict[str, tuple[int, int]] = {}
    all_ok = True
    for split in ("train", "test", "arc-gen"):
        passed = 0
        total = 0
        for example in examples.get(split, []):
            input_arr = convert_to_numpy(example, "input")
            expected_arr = convert_to_numpy(example, "output")
            if input_arr is None or expected_arr is None:
                continue
            total += 1
            out = session.run([OUT_NAME], {IN_NAME: input_arr})[0]
            if np.array_equal((out > 0.0).astype(np.float32), expected_arr):
                passed += 1
            else:
                all_ok = False
        counts[split] = (passed, total)
    return all_ok, counts


def _profile_largest_internal(model: onnx.ModelProto) -> tuple[int | None, str | None, int | None]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return None, None, None

    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(Path(tempfile.gettempdir()) / f"ng_{TASK_ID}_largest")
    session = ort.InferenceSession(
        sanitized.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    for arr in load_task_examples(BEST_PATH):
        session.run([OUT_NAME], {IN_NAME: arr})
    trace_path = session.end_profiling()

    graph = onnx.shape_inference.infer_shapes(sanitized, strict_mode=True).graph
    outputs_by_node = {node.name: list(node.output) for node in graph.node}
    dtypes = {
        info.name: onnx.helper.tensor_dtype_to_np_dtype(info.type.tensor_type.elem_type)
        for info in list(graph.value_info) + list(graph.output)
        if info.type.HasField("tensor_type")
    }
    largest_name: str | None = None
    largest_bytes = -1
    with open(trace_path, encoding="utf-8") as fh:
        trace = json.load(fh)
    for event in trace:
        if event.get("cat") != "Node" or "output_type_shape" not in event.get("args", {}):
            continue
        node_name = event.get("name", "").replace("_kernel_time", "")
        for idx, shape_dict in enumerate(event["args"]["output_type_shape"]):
            outputs = outputs_by_node.get(node_name, [])
            if idx >= len(outputs):
                continue
            output_name = outputs[idx]
            if output_name == OUT_NAME or output_name not in dtypes:
                continue
            itemsize = np.dtype(dtypes[output_name]).itemsize
            size = itemsize * sum(math.prod(dims) for dims in shape_dict.values())
            if size > largest_bytes:
                largest_name = output_name
                largest_bytes = int(size)
    memory = calculate_memory(sanitized, trace_path)
    params = calculate_params(sanitized)
    return memory, largest_name, largest_bytes


def _format_counts(counts: dict[str, tuple[int, int]]) -> str:
    return ", ".join(f"{split} {ok}/{total}" for split, (ok, total) in counts.items())


def main() -> None:
    variants: Iterable[tuple[str, onnx.ModelProto]] = (
        ("conv_color", build_conv_color_model()),
        ("mask_color", build_mask_model()),
        ("gather_float_occ", build_gather_model(compare_half=False)),
    )

    results: list[tuple[int, str, onnx.ModelProto, dict[str, Any], str | None, int | None, dict[str, tuple[int, int]]]] = []
    for label, model in variants:
        tmp_path = OUT_DIR / f"{TASK_ID}_{label}.onnx"
        onnx.save(model, tmp_path)
        correct, counts = _check_correct(model)
        scored = score_file(tmp_path)
        memory, largest_name, largest_bytes = _profile_largest_internal(model)
        score_text = f"{scored['score']:.6f}" if scored["score"] is not None else "None"
        print(
            f"{label:<18} correct={correct} ({_format_counts(counts)}) "
            f"memory={scored['memory']} params={scored['params']} cost={scored['cost']} "
            f"score={score_text} largest={largest_name}:{largest_bytes}"
        )
        if correct and scored["valid"]:
            results.append((int(scored["cost"]), label, model, scored, largest_name, largest_bytes, counts))
        tmp_path.unlink(missing_ok=True)

    if not results:
        raise SystemExit("no valid correct variants")

    _, label, model, scored, largest_name, largest_bytes, counts = min(results, key=lambda item: item[0])
    onnx.save(model, BEST_PATH)
    print()
    print(f"kept:    {label}")
    print(f"wrote:   {BEST_PATH}")
    print(f"valid:   {scored['valid']}")
    print(f"passes:  {_format_counts(counts)}")
    print(f"memory:  {scored['memory']}")
    print(f"params:  {scored['params']}")
    print(f"cost:    {scored['cost']}")
    print(f"score:   {scored['score']:.6f}")
    print(f"largest: {largest_name} ({largest_bytes} bytes)")


if __name__ == "__main__":
    main()
