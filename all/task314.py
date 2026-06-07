"""Compact ONNX for NeuroGolf task314.

Task rule: the active 8x8 board is a 3x3 array of 2x2 blue patches separated
by black rows and columns at indices 2 and 5. Colored marker cells in matching
within-patch offsets complete straight lines through the middle patch: when the
same marker color appears in the left and right outer patches of a macro-row,
copy it into the corresponding middle patch cell; when it appears in the top
and bottom outer patches of a macro-column, copy it into the middle patch cell.
Existing markers and the black cross stay fixed, and blue cells are cleared
where a copied marker is written.

ONNX approach: crop the 8x8 board, convert it to bool, operate only on marker
channels 2..9, fill middle columns from left/right endpoint conjunctions and
middle rows from top/bottom endpoint conjunctions, rebuild black/blue/marker
channels, cast the compact 8x8 result to float, then pad to the required
30x30 output.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402


TASK_ID = "task314"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task314.onnx"

IR_VERSION = 10
OPSET = 10
FULL_SHAPE = [1, 10, 30, 30]


def _init(initializers: list[onnx.TensorProto], name: str, values: Any, dtype: Any) -> str:
    initializers.append(numpy_helper.from_array(np.asarray(values, dtype=dtype), name=name))
    return name


def _make_model(nodes: list[onnx.NodeProto], initializers: list[onnx.TensorProto]) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, FULL_SHAPE)],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def _base_masks() -> tuple[np.ndarray, np.ndarray]:
    black = np.zeros((1, 1, 8, 8), dtype=np.bool_)
    black[:, :, 2, :] = True
    black[:, :, 5, :] = True
    black[:, :, :, 2] = True
    black[:, :, :, 5] = True
    blue = ~black
    return black, blue


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    axes4 = _init(initializers, "axes4", [0, 1, 2, 3], np.int64)
    axis1 = _init(initializers, "axis1", [1], np.int64)
    marker_st = _init(initializers, "marker_st", [0, 2, 0, 0], np.int64)
    marker_en = _init(initializers, "marker_en", [1, 10, 8, 8], np.int64)
    left_st = _init(initializers, "left_st", [0, 0, 0, 0], np.int64)
    left_en = _init(initializers, "left_en", [1, 8, 8, 2], np.int64)
    right_st = _init(initializers, "right_st", [0, 0, 0, 6], np.int64)
    right_en = _init(initializers, "right_en", [1, 8, 8, 8], np.int64)
    top_st = _init(initializers, "top_st", [0, 0, 0, 0], np.int64)
    top_en = _init(initializers, "top_en", [1, 8, 2, 8], np.int64)
    bottom_st = _init(initializers, "bottom_st", [0, 0, 6, 0], np.int64)
    bottom_en = _init(initializers, "bottom_en", [1, 8, 8, 8], np.int64)
    h_zero = _init(initializers, "h_zero", np.zeros((1, 8, 8, 3), dtype=np.bool_), np.bool_)
    v_zero = _init(initializers, "v_zero", np.zeros((1, 8, 3, 8), dtype=np.bool_), np.bool_)
    black_mask, blue_mask = _base_masks()
    black_const = _init(initializers, "black_const", black_mask, np.bool_)
    blue_const = _init(initializers, "blue_const", blue_mask, np.bool_)
    color_ch_starts = [
        _init(initializers, f"color{i}_st", [i], np.int64) for i in range(8)
    ]
    color_ch_ends = [
        _init(initializers, f"color{i}_en", [i + 1], np.int64) for i in range(8)
    ]

    nodes.extend(
        [
            helper.make_node("Slice", ["input", marker_st, marker_en, axes4], ["markers_f"]),
            helper.make_node("Greater", ["markers_f", _init(initializers, "zero", [0.0], np.float32)], ["markers"]),
            helper.make_node("Slice", ["markers", left_st, left_en, axes4], ["left"]),
            helper.make_node("Slice", ["markers", right_st, right_en, axes4], ["right"]),
            helper.make_node("And", ["left", "right"], ["h_pair"]),
            helper.make_node("Concat", [h_zero, "h_pair", h_zero], ["h_fill"], axis=3),
            helper.make_node("Slice", ["markers", top_st, top_en, axes4], ["top"]),
            helper.make_node("Slice", ["markers", bottom_st, bottom_en, axes4], ["bottom"]),
            helper.make_node("And", ["top", "bottom"], ["v_pair"]),
            helper.make_node("Concat", [v_zero, "v_pair", v_zero], ["v_fill"], axis=2),
            helper.make_node("Or", ["markers", "h_fill"], ["markers_h"]),
            helper.make_node("Or", ["markers_h", "v_fill"], ["markers_out"]),
        ]
    )

    any_color = "color0"
    nodes.append(helper.make_node("Slice", ["markers_out", color_ch_starts[0], color_ch_ends[0], axis1], [any_color]))
    for i in range(1, 8):
        ch = f"color{i}"
        out = f"any_color{i}"
        nodes.append(helper.make_node("Slice", ["markers_out", color_ch_starts[i], color_ch_ends[i], axis1], [ch]))
        nodes.append(helper.make_node("Or", [any_color, ch], [out]))
        any_color = out

    nodes.extend(
        [
            helper.make_node("Not", [any_color], ["no_color"]),
            helper.make_node("And", [blue_const, "no_color"], ["blue"]),
            helper.make_node("Concat", [black_const, "blue", "markers_out"], ["out8b"], axis=1),
            helper.make_node("Cast", ["out8b"], ["out8"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out8"],
                ["output"],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, 22, 22],
                value=0.0,
            ),
        ]
    )

    return _make_model(nodes, initializers)


def _load_examples() -> list[dict[str, list[list[int]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [ex for split in ("train", "test", "arc-gen") for ex in data.get(split, [])]


def solve_grid(grid: list[list[int]]) -> list[list[int]]:
    arr = np.asarray(grid, dtype=np.int64)
    out = arr.copy()
    for color in range(2, 10):
        for oy in (0, 1):
            for ox in (0, 1):
                for block_row in range(3):
                    y = block_row * 3 + oy
                    if arr[y, ox] == color and arr[y, 6 + ox] == color:
                        out[y, 3 + ox] = color
                for block_col in range(3):
                    x = block_col * 3 + ox
                    if arr[oy, x] == color and arr[6 + oy, x] == color:
                        out[3 + oy, x] = color
    return out.tolist()


def validate_rule() -> None:
    for idx, ex in enumerate(_load_examples()):
        actual = solve_grid(ex["input"])
        if actual != ex["output"]:
            raise AssertionError(f"Python rule mismatch on example {idx}")


def validate_onnx(path: Path) -> None:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for idx, ex in enumerate(_load_examples()):
        inp = convert_to_numpy(ex, "input")
        expected = convert_to_numpy(ex, "output")
        if inp is None or expected is None:
            continue
        actual = session.run(["output"], {"input": inp})[0]
        if not np.array_equal(actual > 0.0, expected > 0.0):
            raise AssertionError(f"ONNX mismatch on example {idx}")


def main() -> None:
    validate_rule()
    model = build_model()
    onnx.save(model, BEST_PATH)
    validate_onnx(BEST_PATH)
    result = score_file(BEST_PATH)
    print(
        f"{BEST_PATH}: valid={result['valid']} memory={result['memory']} "
        f"params={result['params']} cost={result['cost']} score={result['score']}"
    )


if __name__ == "__main__":
    main()
