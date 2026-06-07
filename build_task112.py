"""Build an optimized ONNX solution for NeuroGolf task112.

Task rule: a red pattern sits in one quadrant next to a 2x2 green marker.
The output preserves the input and mirrors every red cell across the
horizontal and vertical axes through the 2x2 marker, filling the matching
quadrants around that marker. The green marker and background remain intact.

The graph keeps the core logic as bool 30x30 masks. It finds the marker's
top row/left column with ReduceSum+ArgMax, builds two 30-element reflected
index vectors, gathers the red mask across columns/rows, and only casts back
to float for the final one-hot output tensor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


OUT_PATH = Path("task112.onnx")


def init(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array, name=name)


def scalar_i64(name: str, value: int) -> onnx.TensorProto:
    return init(name, np.array(value, dtype=np.int64))


def scalar_f32(name: str, value: float) -> onnx.TensorProto:
    return init(name, np.array(value, dtype=np.float32))


def main() -> None:
    nodes: list[onnx.NodeProto] = []

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])

    initializers = [
            init("idx2", np.array([2], dtype=np.int64)),
            init("idx3", np.array([3], dtype=np.int64)),
            init(
                "red_onehot",
                np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32).reshape(1, 10, 1, 1),
            ),
            init("coord", np.arange(30, dtype=np.int64)),
            scalar_i64("two_i", 2),
            scalar_i64("one_i", 1),
            scalar_i64("zero_i", 0),
            scalar_i64("thirty_i", 30),
            scalar_f32("zero_f", 0.0),
    ]

    # Extract red/green channels as [1,1,30,30] tensors.
    nodes.extend(
        [
            helper.make_node("Gather", ["input", "idx2"], ["red4"], axis=1),
            helper.make_node("Gather", ["input", "idx3"], ["green4"], axis=1),
        ]
    )

    # Marker top row and left column. The two green rows/cols tie, and ArgMax
    # returns the first max, which is exactly the top/left edge of the marker.
    nodes.extend(
        [
            helper.make_node("ReduceSum", ["green4"], ["green_rows"], axes=[0, 1, 3], keepdims=0),
            helper.make_node("ArgMax", ["green_rows"], ["g_row"], axis=0, keepdims=0),
            helper.make_node("ReduceSum", ["green4"], ["green_cols"], axes=[0, 1, 2], keepdims=0),
            helper.make_node("ArgMax", ["green_cols"], ["g_col"], axis=0, keepdims=0),
        ]
    )

    # Reflected index formula: src = 2 * marker_edge + 1 - output_index.
    for axis_name, marker_name in (("row", "g_row"), ("col", "g_col")):
        nodes.extend(
            [
                helper.make_node("Mul", [marker_name, "two_i"], [f"{axis_name}_twice"]),
                helper.make_node("Add", [f"{axis_name}_twice", "one_i"], [f"{axis_name}_pivot"]),
                helper.make_node("Sub", [f"{axis_name}_pivot", "coord"], [f"{axis_name}_src"]),
                helper.make_node("Less", [f"{axis_name}_src", "thirty_i"], [f"{axis_name}_lt30"]),
                helper.make_node("Less", [f"{axis_name}_src", "zero_i"], [f"{axis_name}_lt0"]),
                helper.make_node("Not", [f"{axis_name}_lt0"], [f"{axis_name}_ge0"]),
                helper.make_node("And", [f"{axis_name}_ge0", f"{axis_name}_lt30"], [f"{axis_name}_valid"]),
                helper.make_node(
                    "Where",
                    [f"{axis_name}_valid", f"{axis_name}_src", marker_name],
                    [f"{axis_name}_safe"],
                ),
            ]
        )

    nodes.extend(
        [
            helper.make_node("Greater", ["red4", "zero_f"], ["red_mask"]),
            helper.make_node("Gather", ["red_mask", "col_safe"], ["h_raw"], axis=3),
            helper.make_node("Gather", ["red_mask", "row_safe"], ["v_raw"], axis=2),
            helper.make_node("Gather", ["h_raw", "row_safe"], ["hv_raw"], axis=2),
            helper.make_node("Or", ["red_mask", "h_raw"], ["red_or_h"]),
            helper.make_node("Or", ["v_raw", "hv_raw"], ["v_or_hv"]),
            helper.make_node("Or", ["red_or_h", "v_or_hv"], ["all_red_mask"]),
            helper.make_node("Where", ["all_red_mask", "red_onehot", "input"], ["output"]),
        ]
    )

    graph = helper.make_graph(nodes, "task112_reflect_red", [x], [y], initializers)
    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 10)],
        producer_name="neurogolf_task112_builder",
    )
    onnx.checker.check_model(model, full_check=True)
    OUT_PATH.write_bytes(model.SerializeToString())
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
