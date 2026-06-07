"""ONNX generator for NeuroGolf task034.

Task rule: the visible 9x9 grid contains one 2x2 seed whose cells are
either marker red (color 2) or one payload color C. Each red marker corner
selects its matching diagonal direction (top-left, top-right, bottom-left,
or bottom-right). The output recolors full 2x2 footprints with C while
dragging the seed along every selected diagonal until the 9x9 canvas clips
it; all red markers disappear and the rest stays black.

The graph keeps the diagonal footprint computation compact: four 8x8 marker
corner masks are concatenated and passed through one cropped ConvTranspose
with a 16x16 binary kernel, producing the final 9x9 footprint mask directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "task034"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task034.onnx"


def diagonal_kernel() -> np.ndarray:
    """Return ConvTranspose weights [input_channels, output_channels, 16, 16]."""
    dirs = [
        (-1, -1),  # marker in seed top-left
        (-1, 1),  # marker in seed top-right
        (1, -1),  # marker in seed bottom-left
        (1, 1),  # marker in seed bottom-right
    ]
    weight = np.zeros((4, 1, 16, 16), dtype=np.float32)
    for channel, (dr, dc) in enumerate(dirs):
        for k in range(9):
            for block_r in (0, 1):
                for block_c in (0, 1):
                    row = 7 + dr * k + block_r
                    col = 7 + dc * k + block_c
                    if 0 <= row < 16 and 0 <= col < 16:
                        weight[channel, 0, row, col] = 1.0
    return weight


def build_model() -> onnx.ModelProto:
    initializers = [
        numpy_helper.from_array(diagonal_kernel(), "diag_weight"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero"),
        numpy_helper.from_array(
            np.array([[[[False]], [[True]], [[False]], [[True]], [[True]], [[True]], [[True]], [[True]], [[True]], [[True]]]]),
            "paintable_colors",
        ),
        numpy_helper.from_array(
            np.array([[[[True]], [[False]], [[False]], [[False]], [[False]], [[False]], [[False]], [[False]], [[False]], [[False]]]]),
            "bg_color",
        ),
    ]

    def slice_node(data: str, output: str, starts: list[int], ends: list[int]) -> onnx.NodeProto:
        return helper.make_node(
            "Slice",
            [data],
            [output],
            starts=starts,
            ends=ends,
            axes=[0, 1, 2, 3],
        )

    nodes = [
        slice_node("input", "input_bg9", [0, 0, 0, 0], [1, 1, 9, 9]),
        helper.make_node("Greater", ["input_bg9", "zero"], ["input_is_bg"]),
        helper.make_node("Not", ["input_is_bg"], ["object9"]),
        slice_node("object9", "object_tl", [0, 0, 0, 0], [1, 1, 8, 8]),
        slice_node("object9", "object_tr", [0, 0, 0, 1], [1, 1, 8, 9]),
        slice_node("object9", "object_bl", [0, 0, 1, 0], [1, 1, 9, 8]),
        slice_node("object9", "object_br", [0, 0, 1, 1], [1, 1, 9, 9]),
        helper.make_node("And", ["object_tl", "object_tr"], ["object_top"]),
        helper.make_node("And", ["object_bl", "object_br"], ["object_bottom"]),
        helper.make_node("And", ["object_top", "object_bottom"], ["valid_seed"]),
        helper.make_node("Cast", ["valid_seed"], ["valid_seed_f"], to=TensorProto.FLOAT),
        slice_node("input", "marker_tl", [0, 2, 0, 0], [1, 3, 8, 8]),
        slice_node("input", "marker_tr", [0, 2, 0, 1], [1, 3, 8, 9]),
        slice_node("input", "marker_bl", [0, 2, 1, 0], [1, 3, 9, 8]),
        slice_node("input", "marker_br", [0, 2, 1, 1], [1, 3, 9, 9]),
        helper.make_node(
            "Concat",
            ["marker_tl", "marker_tr", "marker_bl", "marker_br"],
            ["corner_markers"],
            axis=1,
        ),
        helper.make_node("Mul", ["corner_markers", "valid_seed_f"], ["valid_markers"]),
        helper.make_node(
            "ConvTranspose",
            ["valid_markers", "diag_weight"],
            ["diag_hits"],
            kernel_shape=[16, 16],
            pads=[7, 7, 7, 7],
        ),
        helper.make_node("Greater", ["diag_hits", "zero"], ["diag_mask"]),
        helper.make_node("Not", ["diag_mask"], ["bg_mask"]),
        helper.make_node(
            "ReduceSum",
            ["input"],
            ["channel_sums"],
            axes=[2, 3],
            keepdims=1,
        ),
        helper.make_node("Greater", ["channel_sums", "zero"], ["present_colors"]),
        helper.make_node("And", ["present_colors", "paintable_colors"], ["paint_color"]),
        helper.make_node("And", ["paint_color", "diag_mask"], ["paint9_bool"]),
        helper.make_node("And", ["bg_color", "bg_mask"], ["bg9_bool"]),
        helper.make_node("Or", ["paint9_bool", "bg9_bool"], ["out9_bool"]),
        helper.make_node("Cast", ["out9_bool"], ["paint9"], to=TensorProto.FLOAT),
        helper.make_node(
            "Pad",
            ["paint9"],
            ["output"],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 21, 21],
            value=0.0,
        ),
    ]

    graph = helper.make_graph(
        nodes,
        TASK_ID,
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
        ],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
        ],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 9)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), BEST_PATH)
    print(BEST_PATH)


if __name__ == "__main__":
    main()
