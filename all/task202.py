"""Build an ONNX solution for task202.

Task rule: the grid is partitioned into full-width horizontal bands or
full-height vertical stripes of solid colors. Each black marker belongs to the
colored stripe/band that surrounds it. Replace every marker by a black line
through the marker that spans that whole region: vertical lines in horizontal
bands, horizontal lines in vertical stripes. Preserve every other colored cell.
"""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "task202"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task202.onnx"


def reduce_sum(name: str, source: str, output: str, axes: list[int], keepdims: int = 1) -> onnx.NodeProto:
    return helper.make_node("ReduceSum", [source], [output], name=name, axes=axes, keepdims=keepdims)


def cast(name: str, source: str, output: str, to: int) -> onnx.NodeProto:
    return helper.make_node("Cast", [source], [output], name=name, to=to)


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Gather", ["input", "seed_idx"], ["seed"], name="seed", axis=1),
        reduce_sum("row_sum_all", "input", "row_sum_all", [3]),
        helper.make_node("Slice", ["row_sum_all", "color_starts", "color_ends", "color_axes"], ["row_color_sum"], name="row_color_sum"),
        helper.make_node("ArgMax", ["row_color_sum"], ["row_color_arg"], name="row_color_arg", axis=1, keepdims=0),
        cast("row_color_arg_f", "row_color_arg", "row_color_arg_f", TensorProto.FLOAT),
        helper.make_node("Add", ["row_color_arg_f", "one"], ["row_color_val"], name="row_color_val"),
        helper.make_node("Transpose", ["row_color_val"], ["row_color_row"], name="row_color_row", perm=[0, 2, 1]),
        helper.make_node("Slice", ["row_color_sum", "slice_starts", "slice_ends", "slice_axes"], ["row0_color_sum"], name="row0_color_sum"),
        helper.make_node("Greater", ["row0_color_sum", "half"], ["row0_has_color_b"], name="row0_has_color_b"),
        cast("row0_has_color_f", "row0_has_color_b", "row0_has_color_f", TensorProto.FLOAT),
        reduce_sum("row0_color_count", "row0_has_color_f", "row0_color_count", [1]),
        helper.make_node("Greater", ["row0_color_count", "one"], ["row_mixed"], name="row_mixed"),
        reduce_sum("col_sum_all", "input", "col_sum_all", [2]),
        helper.make_node("Slice", ["col_sum_all", "color_starts", "color_ends", "color_axes"], ["col_color_sum"], name="col_color_sum"),
        helper.make_node("ArgMax", ["col_color_sum"], ["col_color_arg"], name="col_color_arg", axis=1, keepdims=0),
        cast("col_color_arg_f", "col_color_arg", "col_color_arg_f", TensorProto.FLOAT),
        helper.make_node("Add", ["col_color_arg_f", "one"], ["col_color_val"], name="col_color_val"),
        helper.make_node("Transpose", ["col_color_val"], ["col_color_col"], name="col_color_col", perm=[0, 2, 1]),
        helper.make_node("MatMul", ["row_color_row", "seed"], ["seed_cols_val"], name="seed_cols_val"),
        helper.make_node("Sub", ["seed_cols_val", "half"], ["seed_cols_low"], name="seed_cols_low"),
        helper.make_node("Add", ["seed_cols_val", "half"], ["seed_cols_high"], name="seed_cols_high"),
        helper.make_node("Greater", ["row_color_val", "seed_cols_low"], ["vert_gt_low"], name="vert_gt_low"),
        helper.make_node("Less", ["row_color_val", "seed_cols_high"], ["vert_lt_high"], name="vert_lt_high"),
        helper.make_node("And", ["vert_gt_low", "vert_lt_high"], ["vert_line"], name="vert_line"),
        helper.make_node("MatMul", ["seed", "col_color_col"], ["seed_rows_val"], name="seed_rows_val"),
        helper.make_node("Sub", ["seed_rows_val", "half"], ["seed_rows_low"], name="seed_rows_low"),
        helper.make_node("Add", ["seed_rows_val", "half"], ["seed_rows_high"], name="seed_rows_high"),
        helper.make_node("Greater", ["col_color_val", "seed_rows_low"], ["horiz_gt_low"], name="horiz_gt_low"),
        helper.make_node("Less", ["col_color_val", "seed_rows_high"], ["horiz_lt_high"], name="horiz_lt_high"),
        helper.make_node("And", ["horiz_gt_low", "horiz_lt_high"], ["horiz_line"], name="horiz_line"),
        helper.make_node("And", ["row_mixed", "horiz_line"], ["use_horiz"], name="use_horiz"),
        helper.make_node("Not", ["row_mixed"], ["not_row_mixed"], name="not_row_mixed"),
        helper.make_node("And", ["not_row_mixed", "vert_line"], ["use_vert"], name="use_vert"),
        helper.make_node("Or", ["use_horiz", "use_vert"], ["line_b"], name="line_b"),
        helper.make_node("Where", ["line_b", "black_fill", "input"], ["output"], name="output"),
    ]

    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [
            helper.make_tensor("one", TensorProto.FLOAT, [], [1.0]),
            helper.make_tensor("half", TensorProto.FLOAT, [], [0.5]),
            helper.make_tensor("slice_starts", TensorProto.INT64, [1], [0]),
            helper.make_tensor("slice_ends", TensorProto.INT64, [1], [1]),
            helper.make_tensor("slice_axes", TensorProto.INT64, [1], [2]),
            helper.make_tensor("color_starts", TensorProto.INT64, [1], [1]),
            helper.make_tensor("color_ends", TensorProto.INT64, [1], [10]),
            helper.make_tensor("color_axes", TensorProto.INT64, [1], [1]),
            helper.make_tensor("black_fill", TensorProto.FLOAT, [1, 10, 1, 1], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            helper.make_tensor("seed_idx", TensorProto.INT64, [], [0]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 10)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def main() -> None:
    onnx.save(build_model(), BEST_PATH)
    print(BEST_PATH)


if __name__ == "__main__":
    main()
