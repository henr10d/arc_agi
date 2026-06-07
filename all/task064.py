"""ONNX solver for ARC task064: extend aligned dot lines to a rectangle.

Task rule: each grid has one background color, one solid rectangle color, and
one dot color.  Dots that share a row with the rectangle draw a horizontal dot
colored segment from the rectangle edge to the dot; dots that share a column
draw the analogous vertical segment.  Non-aligned dots and the rectangle stay
unchanged.  The model infers colors from channel counts and row spans, projects
the solid rectangle to compact 1D row/column spans, scans only the dot plane at
full crop resolution, and emits a one-hot output with the original 30x30
padding preserved.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

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

TASK_ID = "task064"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task064.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 11
IR_VERSION = 10


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


def _cum_positive(
    nodes: list[onnx.NodeProto],
    source: str,
    axis_name: str,
    axis: str,
    *,
    reverse: bool = False,
    zero_name: str = "zero",
) -> str:
    summed = f"{source}_{axis_name}_{'rev' if reverse else 'fwd'}_sum"
    out = f"{source}_{axis_name}_{'rev' if reverse else 'fwd'}"
    nodes.append(helper.make_node("CumSum", [source, axis], [summed], reverse=1 if reverse else 0))
    nodes.append(helper.make_node("Greater", [summed, zero_name], [out]))
    return out


def _line_mask(nodes: list[onnx.NodeProto], rect: str, dot: str, *, scan_zero: str = "zero") -> str:
    rect_l = _cum_positive(nodes, rect, "w", "axis_w", zero_name=scan_zero)
    rect_r = _cum_positive(nodes, rect, "w", "axis_w", reverse=True, zero_name=scan_zero)
    dot_l = _cum_positive(nodes, dot, "w", "axis_w", zero_name=scan_zero)
    dot_r = _cum_positive(nodes, dot, "w", "axis_w", reverse=True, zero_name=scan_zero)
    rect_t = _cum_positive(nodes, rect, "h", "axis_h", zero_name=scan_zero)
    rect_b = _cum_positive(nodes, rect, "h", "axis_h", reverse=True, zero_name=scan_zero)
    dot_t = _cum_positive(nodes, dot, "h", "axis_h", zero_name=scan_zero)
    dot_b = _cum_positive(nodes, dot, "h", "axis_h", reverse=True, zero_name=scan_zero)

    nodes.extend(
        [
            helper.make_node("And", [rect_l, dot_r], ["h_right"]),
            helper.make_node("And", [dot_l, rect_r], ["h_left"]),
            helper.make_node("Or", ["h_right", "h_left"], ["h_fill"]),
            helper.make_node("And", [rect_t, dot_b], ["v_down"]),
            helper.make_node("And", [dot_t, rect_b], ["v_up"]),
            helper.make_node("Or", ["v_down", "v_up"], ["v_fill"]),
            helper.make_node("Or", ["h_fill", "v_fill"], ["fill_any"]),
            helper.make_node("Greater", [rect, scan_zero], ["rect_bool"]),
            helper.make_node("Not", ["rect_bool"], ["not_rect"]),
            helper.make_node("And", ["fill_any", "not_rect"], ["changed_crop"]),
        ]
    )
    return "changed_crop"


def _line_mask_projected_rect(nodes: list[onnx.NodeProto], rect: str, dot: str, *, scan_zero: str = "zero") -> str:
    """Build fill mask for rank-3 [1,H,W] planes using 1D rectangle projections."""
    nodes.extend(
        [
            helper.make_node("ReduceMax", [rect], ["rect_rows_f"], axes=[2], keepdims=0),
            helper.make_node("ReduceMax", [rect], ["rect_cols_f"], axes=[1], keepdims=0),
        ]
    )
    rect_l = _cum_positive(nodes, "rect_cols_f", "cols", "axis_h", zero_name=scan_zero)
    rect_r = _cum_positive(nodes, "rect_cols_f", "cols", "axis_h", reverse=True, zero_name=scan_zero)
    rect_t = _cum_positive(nodes, "rect_rows_f", "rows", "axis_h", zero_name=scan_zero)
    rect_b = _cum_positive(nodes, "rect_rows_f", "rows", "axis_h", reverse=True, zero_name=scan_zero)

    dot_l = _cum_positive(nodes, dot, "w", "axis_w", zero_name=scan_zero)
    dot_r = _cum_positive(nodes, dot, "w", "axis_w", reverse=True, zero_name=scan_zero)
    dot_t = _cum_positive(nodes, dot, "h", "axis_h", zero_name=scan_zero)
    dot_b = _cum_positive(nodes, dot, "h", "axis_h", reverse=True, zero_name=scan_zero)

    nodes.extend(
        [
            helper.make_node("Unsqueeze", [rect_l], ["rect_l_w"], axes=[1]),
            helper.make_node("Unsqueeze", [rect_r], ["rect_r_w"], axes=[1]),
            helper.make_node("Unsqueeze", [rect_t], ["rect_t_h"], axes=[2]),
            helper.make_node("Unsqueeze", [rect_b], ["rect_b_h"], axes=[2]),
            helper.make_node("And", ["rect_t_h", "rect_b_h"], ["rect_rows_h"]),
            helper.make_node("And", ["rect_l_w", "rect_r_w"], ["rect_cols_v"]),
            helper.make_node("And", ["rect_l_w", dot_r], ["h_right_raw"]),
            helper.make_node("And", ["h_right_raw", "rect_rows_h"], ["h_right"]),
            helper.make_node("And", [dot_l, "rect_r_w"], ["h_left_raw"]),
            helper.make_node("And", ["h_left_raw", "rect_rows_h"], ["h_left"]),
            helper.make_node("Or", ["h_right", "h_left"], ["h_fill"]),
            helper.make_node("And", ["rect_t_h", dot_b], ["v_down_raw"]),
            helper.make_node("And", ["v_down_raw", "rect_cols_v"], ["v_down"]),
            helper.make_node("And", [dot_t, "rect_b_h"], ["v_up_raw"]),
            helper.make_node("And", ["v_up_raw", "rect_cols_v"], ["v_up"]),
            helper.make_node("Or", ["v_down", "v_up"], ["v_fill"]),
            helper.make_node("Or", ["h_fill", "v_fill"], ["fill_any"]),
            helper.make_node("Greater", [rect, scan_zero], ["rect_bool"]),
            helper.make_node("Not", ["rect_bool"], ["not_rect"]),
            helper.make_node("And", ["fill_any", "not_rect"], ["changed_crop"]),
        ]
    )
    return "changed_crop"


def build_model(
    *,
    crop: int | None,
    use_plane_crop: bool,
    color_strategy: str,
    float16_scans: bool = False,
    bool_pad: bool = False,
    uint8_pad: bool = False,
    squeeze_planes: bool = False,
    mask_dot: bool = False,
    projected_rect: bool = False,
) -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    _f32(inits, [0.0], "zero")
    _f32(inits, [-1.0], "neg_one")
    if float16_scans:
        _init(inits, np.asarray([0.0], dtype=np.float16), "zero_f16")
    if uint8_pad:
        _init(inits, np.asarray([0], dtype=np.uint8), "zero_u8")
    if mask_dot and not uint8_pad:
        _init(inits, np.asarray([0], dtype=np.uint8), "zero_u8")
    _i64(inits, np.arange(C), "colors")
    _i64(inits, np.arange(C).reshape(1, C, 1, 1), "channel_grid")
    _i64(inits, [1 if squeeze_planes else 2], "axis_h")
    _i64(inits, [2 if squeeze_planes else 3], "axis_w")

    nodes.extend(
        [
            helper.make_node("ReduceSum", [IN_NAME], ["counts"], axes=[0, 2, 3], keepdims=0),
            helper.make_node("ArgMax", ["counts"], ["bg_idx"], axis=0, keepdims=0 if squeeze_planes else 1),
            helper.make_node("Equal", ["colors", "bg_idx"], ["is_bg_color"]),
        ]
    )
    if color_strategy == "count":
        nodes.append(helper.make_node("Where", ["is_bg_color", "neg_one", "counts"], ["rect_scores"]))
    elif color_strategy == "row_span":
        nodes.extend(
            [
                helper.make_node("ReduceSum", [IN_NAME], ["row_counts"], axes=[3], keepdims=0),
                helper.make_node("ReduceMax", ["row_counts"], ["rect_scores_raw"], axes=[0, 2], keepdims=0),
                helper.make_node("Where", ["is_bg_color", "neg_one", "rect_scores_raw"], ["rect_scores"]),
            ]
        )
    elif color_strategy == "span":
        nodes.extend(
            [
                helper.make_node("ReduceSum", [IN_NAME], ["row_counts"], axes=[3], keepdims=0),
                helper.make_node("ReduceMax", ["row_counts"], ["max_row_count"], axes=[0, 2], keepdims=0),
                helper.make_node("ReduceSum", [IN_NAME], ["col_counts"], axes=[2], keepdims=0),
                helper.make_node("ReduceMax", ["col_counts"], ["max_col_count"], axes=[0, 2], keepdims=0),
                helper.make_node("Mul", ["max_row_count", "max_col_count"], ["shape_scores"]),
                helper.make_node("Where", ["is_bg_color", "neg_one", "shape_scores"], ["rect_scores"]),
            ]
        )
    else:
        raise ValueError(color_strategy)

    nodes.extend(
        [
            helper.make_node("ArgMax", ["rect_scores"], ["rect_idx"], axis=0, keepdims=0 if squeeze_planes else 1),
            helper.make_node("Equal", ["colors", "rect_idx"], ["is_rect_color"]),
            helper.make_node("Or", ["is_bg_color", "is_rect_color"], ["not_dot_color"]),
        ]
    )
    if mask_dot:
        nodes.extend(
            [
                helper.make_node("Not", ["not_dot_color"], ["is_dot_color"]),
                helper.make_node("Cast", ["is_dot_color"], ["dot_mask_u8"], to=TensorProto.UINT8),
                helper.make_node("ArgMax", ["dot_mask_u8"], ["dot_idx"], axis=0, keepdims=0 if squeeze_planes else 1),
            ]
        )
    else:
        nodes.extend(
            [
                helper.make_node("Where", ["not_dot_color", "neg_one", "counts"], ["dot_counts"]),
                helper.make_node("ArgMax", ["dot_counts"], ["dot_idx"], axis=0, keepdims=0 if squeeze_planes else 1),
            ]
        )
    nodes.extend(
        [
            helper.make_node("Gather", [IN_NAME, "rect_idx"], ["rect30"], axis=1),
            helper.make_node("Gather", [IN_NAME, "dot_idx"], ["dot30"], axis=1),
        ]
    )

    rect = "rect30"
    dot = "dot30"
    if crop is not None and use_plane_crop:
        _i64(inits, [0, 0], "crop_starts")
        _i64(inits, [crop, crop], "crop_ends")
        _i64(inits, [1, 2] if squeeze_planes else [2, 3], "crop_axes")
        nodes.extend(
            [
                helper.make_node("Slice", ["rect30", "crop_starts", "crop_ends", "crop_axes"], ["rect_crop"]),
                helper.make_node("Slice", ["dot30", "crop_starts", "crop_ends", "crop_axes"], ["dot_crop"]),
            ]
        )
        rect = "rect_crop"
        dot = "dot_crop"

    scan_zero = "zero"
    if float16_scans:
        nodes.extend(
            [
                helper.make_node("Cast", [rect], ["rect_f16"], to=TensorProto.FLOAT16),
                helper.make_node("Cast", [dot], ["dot_f16"], to=TensorProto.FLOAT16),
            ]
        )
        rect = "rect_f16"
        dot = "dot_f16"
        scan_zero = "zero_f16"

    if projected_rect:
        if not squeeze_planes:
            raise ValueError("projected_rect requires squeeze_planes")
        changed = _line_mask_projected_rect(nodes, rect, dot, scan_zero=scan_zero)
    else:
        changed = _line_mask(nodes, rect, dot, scan_zero=scan_zero)

    if crop is not None and use_plane_crop:
        pad = H - crop
        _i64(inits, [0, 0, 0, 0, pad, pad] if squeeze_planes else [0, 0, 0, 0, 0, 0, pad, pad], "pads_changed")
        if bool_pad:
            nodes.append(helper.make_node("Pad", [changed, "pads_changed"], ["changed"], mode="constant"))
            changed = "changed"
        elif uint8_pad:
            nodes.extend(
                [
                    helper.make_node("Cast", [changed], ["changed_crop_u8"], to=TensorProto.UINT8),
                    helper.make_node("Pad", ["changed_crop_u8", "pads_changed"], ["changed_u8"], mode="constant"),
                    helper.make_node("Greater", ["changed_u8", "zero_u8"], ["changed"]),
                ]
            )
            changed = "changed"
        else:
            nodes.extend(
                [
                    helper.make_node("Cast", [changed], ["changed_crop_f"], to=TensorProto.FLOAT),
                    helper.make_node("Pad", ["changed_crop_f", "pads_changed"], ["changed_f"], mode="constant"),
                    helper.make_node("Greater", ["changed_f", "zero"], ["changed"]),
                ]
            )
            changed = "changed"

    nodes.extend(
        [
            helper.make_node(
                "Unsqueeze",
                ["dot_idx"],
                ["dot_idx4"],
                axes=[0, 1, 2, 3] if squeeze_planes else [1, 2, 3],
            ),
            helper.make_node("Equal", ["channel_grid", "dot_idx4"], ["dot_color_oh"]),
            helper.make_node("Cast", ["dot_color_oh"], ["dot_color_float"], to=TensorProto.FLOAT),
            helper.make_node("Where", [changed, "dot_color_float", IN_NAME], [OUT_NAME]),
        ]
    )

    label = f"{TASK_ID}_{color_strategy}_{'crop' + str(crop) if crop else 'full'}"
    if float16_scans:
        label += "_f16"
    if bool_pad:
        label += "_boolpad"
    if uint8_pad:
        label += "_u8pad"
    if squeeze_planes:
        label += "_squeeze"
    if mask_dot:
        label += "_maskdot"
    if projected_rect:
        label += "_projrect"
    return _make_model(nodes, inits, label)


def _load_examples() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _max_grid_size() -> int:
    examples = _load_examples()
    return max(max(len(ex["input"]), len(ex["input"][0])) for split in examples.values() for ex in split)


def _check_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return False, {}
    session = ort.InferenceSession(
        sanitized.SerializeToString(),
        sess_options=ort.SessionOptions(),
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
            pred = (out > 0.0).astype(np.float32)
            if np.array_equal(pred, expected_arr):
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
    return memory, largest_name, largest_bytes


def _format_counts(counts: dict[str, tuple[int, int]]) -> str:
    return ", ".join(f"{split} {ok}/{total}" for split, (ok, total) in counts.items())


def main() -> None:
    max_grid = _max_grid_size()
    variants: Iterable[tuple[str, onnx.ModelProto]] = (
        (
            "rowspan_u8pad_squeeze_projrect",
            build_model(
                crop=max_grid,
                use_plane_crop=True,
                color_strategy="row_span",
                uint8_pad=True,
                squeeze_planes=True,
                projected_rect=True,
            ),
        ),
    )

    results: list[tuple[int, str, onnx.ModelProto, dict[str, Any], str | None, int | None, dict[str, tuple[int, int]]]] = []
    for label, model in variants:
        tmp_path = OUT_DIR / f"{TASK_ID}_{label}.onnx"
        onnx.save(model, tmp_path)
        try:
            correct, counts = _check_correct(model)
            scored = score_file(tmp_path)
            memory, largest_name, largest_bytes = _profile_largest_internal(model)
        except Exception as exc:  # noqa: BLE001 - keep experimental variants from aborting the search.
            print(f"{label:<22} invalid: {exc}")
            tmp_path.unlink(missing_ok=True)
            continue
        score_text = f"{scored['score']:.6f}" if scored["score"] is not None else "INVALID"
        print(
            f"{label:<22} correct={correct} ({_format_counts(counts)}) "
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
