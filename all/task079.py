"""ONNX solver for NeuroGolf task079.

Task rule: the 14x14 input contains repeated colored 3x3 glyphs on black.
Valid glyph windows are monochrome, have at least four filled cells, and touch
all three rows and all three columns of their 3x3 window. The output is the
3x3 glyph belonging to the color whose valid glyphs appear most often, drawn
in that color on black.

The graph is specialized to the fixed 14x14 inputs for this task. It uses a
grouped 3x3 convolution to count each color's glyph windows, compact occupancy
logic for row/column span checks, and only expands to float one-hot at the
final 3x3 output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task079"
TASK_JSON = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
NC = 9
H = W = 30
GH = GW = 14
WH = WW = 12
IN_NAME = "input"
OUT_NAME = "output"


def init(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def scalar_f(inits: list[onnx.TensorProto], name: str, value: float) -> str:
    return init(inits, name, np.asarray([value], dtype=np.float32))


def scalar_h(inits: list[onnx.TensorProto], name: str, value: float) -> str:
    return init(inits, name, np.asarray([value], dtype=np.float16))


def scalar_i(inits: list[onnx.TensorProto], name: str, value: int) -> str:
    return init(inits, name, np.asarray([value], dtype=np.int64))


def make_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    starts = init(inits, "starts", np.asarray([0, 1, 0, 0], dtype=np.int64))
    ends = init(inits, "ends", np.asarray([1, 10, GH, GW], dtype=np.int64))
    axes = init(inits, "axes", np.asarray([0, 1, 2, 3], dtype=np.int64))
    valid_shape = init(inits, "valid_shape", np.asarray([NC, WH * WW], dtype=np.int64))
    flat_196 = init(inits, "flat_196", np.asarray([GH * GW], dtype=np.int64))
    tiny_shape = init(inits, "tiny_shape", np.asarray([1, C, 3, 3], dtype=np.int64))
    colors = init(inits, "colors", np.arange(C, dtype=np.int64).reshape(C, 1))
    bg_channel = init(inits, "bg_channel", np.asarray([[1.0]] + [[0.0]] * 9, dtype=np.float16))
    one = scalar_h(inits, "one", 1.0)
    half = scalar_h(inits, "half", 0.5)
    three_half = scalar_h(inits, "three_half", 3.5)
    one_i64 = scalar_i(inits, "one_i64", 1)
    twelve_i64 = scalar_i(inits, "twelve_i64", WW)
    fourteen_i64 = scalar_i(inits, "fourteen_i64", GW)
    mask_offsets = init(
        inits,
        "mask_offsets",
        np.asarray([0, 1, 2, GW, GW + 1, GW + 2, 2 * GW, 2 * GW + 1, 2 * GW + 2], dtype=np.int64),
    )
    channel_axis = init(inits, "channel_axis", np.asarray([1], dtype=np.int64))

    color_kernel = np.ones((NC, 1, 3, 3), dtype=np.float16)
    rowcol_kernel = np.zeros((6, 1, 3, 3), dtype=np.float16)
    for i in range(3):
        rowcol_kernel[i, 0, i, :] = 1.0
        rowcol_kernel[i + 3, 0, :, i] = 1.0
    color_kernel_name = init(inits, "color_kernel", color_kernel)
    rowcol_kernel_name = init(inits, "rowcol_kernel", rowcol_kernel)

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts, ends, axes], ["crop"]),
            helper.make_node("Cast", ["crop"], ["crop_h"], to=TensorProto.FLOAT16),
            helper.make_node(
                "Conv",
                ["crop_h", color_kernel_name],
                ["color_count"],
                group=NC,
                kernel_shape=[3, 3],
            ),
            helper.make_node("ReduceSum", ["color_count"], ["total_count"], axes=[1], keepdims=1),
            helper.make_node("Sub", ["total_count", "color_count"], ["other_count"]),
            helper.make_node("Less", ["other_count", half], ["one_color"]),
            helper.make_node("Greater", ["color_count", three_half], ["enough"]),
            helper.make_node("ReduceSum", ["crop_h"], ["occupancy"], axes=[1], keepdims=1),
            helper.make_node(
                "Conv",
                ["occupancy", rowcol_kernel_name],
                ["rowcol_count"],
                kernel_shape=[3, 3],
            ),
            helper.make_node("Greater", ["rowcol_count", half], ["rowcol_hit"]),
        ]
    )

    rowcol_parts: list[str] = []
    for i in range(6):
        s = init(inits, f"rc_s_{i}", np.asarray([i], dtype=np.int64))
        e = init(inits, f"rc_e_{i}", np.asarray([i + 1], dtype=np.int64))
        out = f"rowcol_{i}"
        nodes.append(helper.make_node("Slice", ["rowcol_hit", s, e, channel_axis], [out]))
        rowcol_parts.append(out)
    acc = rowcol_parts[0]
    for i, part in enumerate(rowcol_parts[1:], start=1):
        out = "span_ok" if i == 5 else f"span_{i}"
        nodes.append(helper.make_node("And", [acc, part], [out]))
        acc = out

    nodes.extend(
        [
            helper.make_node("And", ["one_color", "enough"], ["valid_color"]),
            helper.make_node("And", ["valid_color", "span_ok"], ["valid"]),
            helper.make_node("Reshape", ["valid", valid_shape], ["valid_flat_b"]),
            helper.make_node("Cast", ["valid_flat_b"], ["valid_flat_f"], to=TensorProto.FLOAT16),
            helper.make_node("ReduceSum", ["valid_flat_f"], ["counts"], axes=[1], keepdims=0),
            helper.make_node("ArgMax", ["counts"], ["win_color0"], axis=0, keepdims=0),
            helper.make_node("Add", ["win_color0", one_i64], ["win_color"]),
            helper.make_node("Gather", ["valid_flat_b", "win_color0"], ["winning_valid_b"], axis=0),
            helper.make_node("Cast", ["winning_valid_b"], ["winning_valid"], to=TensorProto.FLOAT),
            helper.make_node("ArgMax", ["winning_valid"], ["win_pos"], axis=0, keepdims=1),
        ]
    )

    nodes.extend(
        [
            helper.make_node("Reshape", ["occupancy", flat_196], ["occupancy_flat"]),
            helper.make_node("Div", ["win_pos", twelve_i64], ["win_row"]),
            helper.make_node("Mul", ["win_row", twelve_i64], ["win_row12"]),
            helper.make_node("Sub", ["win_pos", "win_row12"], ["win_col"]),
            helper.make_node("Mul", ["win_row", fourteen_i64], ["win_row14"]),
            helper.make_node("Add", ["win_row14", "win_col"], ["win_base"]),
            helper.make_node("Add", ["win_base", mask_offsets], ["win_indices"]),
            helper.make_node("Gather", ["occupancy_flat", "win_indices"], ["win_mask_h"], axis=0),
            helper.make_node("Sub", [one, "win_mask_h"], ["inv_mask"]),
            helper.make_node("Equal", [colors, "win_color"], ["color_eq"]),
            helper.make_node("Cast", ["color_eq"], ["color_eq_f"], to=TensorProto.FLOAT16),
            helper.make_node("Mul", ["color_eq_f", "win_mask_h"], ["fg_out"]),
            helper.make_node("Mul", [bg_channel, "inv_mask"], ["bg_out"]),
            helper.make_node("Add", ["fg_out", "bg_out"], ["tiny_10_9"]),
            helper.make_node("Reshape", ["tiny_10_9", tiny_shape], ["tiny_h"]),
            helper.make_node("Cast", ["tiny_h"], ["tiny"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["tiny"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - 3, W - 3],
                value=0.0,
            ),
        ]
    )

    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, [1, C, H, W])],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, [1, C, H, W])],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name=TASK_ID,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 10)],
    )
    onnx.checker.check_model(model)
    return model


def arc_grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros((1, C, H, W), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def validate_examples(model: onnx.ModelProto) -> None:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    task = json.loads(TASK_JSON.read_text(encoding="utf-8"))
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(task.get(split, [])):
            pred = session.run([OUT_NAME], {IN_NAME: arc_grid_to_onehot(ex["input"])})[0]
            rows = len(ex["output"])
            cols = len(ex["output"][0])
            pred_grid = np.argmax(pred[0, :, :rows, :cols], axis=0).astype(np.int64)
            expected = np.asarray(ex["output"], dtype=np.int64)
            if not np.array_equal(pred_grid, expected):
                raise AssertionError(
                    f"{split} {idx} mismatch: {pred_grid.tolist()} != {expected.tolist()}"
                )


def main() -> None:
    model = make_model()
    validate_examples(model)
    onnx.save(model, BEST_PATH)
    result = score_file(BEST_PATH)
    print(
        f"{BEST_PATH}: valid={result['valid']} memory={result['memory']} "
        f"params={result['params']} cost={result['cost']} score={result['score']}"
    )
    if not result["valid"]:
        raise SystemExit(result.get("error") or "score_model reported invalid")


if __name__ == "__main__":
    main()
