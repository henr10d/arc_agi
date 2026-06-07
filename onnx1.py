"""Minimal ONNX for ARC task001 using Kaggle one-hot I/O.

Task rule: read the top-left 3x3 input core. For each non-background
cell in that core, copy the full object-color pattern into the
corresponding 3x3 output block, producing a 9x9 tiled output. Empty
blocks are background. The 9x9 one-hot result is padded to the required
30x30 competition tensor.

The graph keeps the final value as UINT8 before Pad because this ONNX
Runtime build rejects Pad(tensor(bool)). The declared output is therefore
UINT8 [1,10,30,30] named exactly "output".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_PATH = Path("/home/filip/Desktop/neuro_golf/all/task001.onnx")

IN_NAME = "input"
OUT_NAME = "output"


def build_einsum_direct_model() -> onnx.ModelProto:
    nodes = []
    inits = []

    def init(name, arr, dtype=np.int64):
        inits.append(numpy_helper.from_array(np.asarray(arr, dtype=dtype), name))

    def mapper_30() -> np.ndarray:
        out = np.zeros((30, 3, 3), dtype=np.uint8)
        for block in range(3):
            for inner in range(3):
                out[block * 3 + inner, block, inner] = 1
        return out

    init("bg_starts", [0, 0, 0, 0])
    init("bg_ends", [1, 1, 3, 3])
    init("obj_starts", [0, 1, 0, 0])
    init("obj_ends", [1, 10, 3, 3])
    init("zero_f", [0.0], dtype=np.float32)
    init("term_1c_shape", [1, 1, 1, 3, 3])
    init("term_9c_shape", [1, 9, 1, 3, 3])
    init("active_9c_shape", [1, 9, 1, 3, 3])
    init("one_term", np.ones((1, 1, 1, 3, 3), dtype=np.uint8), dtype=np.uint8)
    init("map30", mapper_30(), dtype=np.uint8)

    nodes.append(helper.make_node(
        "Slice",
        [IN_NAME, "bg_starts", "bg_ends"],
        ["bg_core"],
        name="slice_bg_direct",
    ))

    nodes.append(helper.make_node(
        "Greater",
        ["bg_core", "zero_f"],
        ["bg_core_bool"],
        name="bg_core_bool",
    ))

    nodes.append(helper.make_node(
        "Cast",
        ["bg_core"],
        ["bg_core_u8"],
        name="cast_bg_core_to_uint8",
        to=TensorProto.UINT8,
    ))

    nodes.append(helper.make_node(
        "Not",
        ["bg_core_bool"],
        ["active_3x3"],
        name="active_3x3",
    ))

    nodes.append(helper.make_node(
        "Cast",
        ["active_3x3"],
        ["active_3x3_u8"],
        name="cast_active_3x3_to_uint8",
        to=TensorProto.UINT8,
    ))

    nodes.append(helper.make_node(
        "Slice",
        [IN_NAME, "obj_starts", "obj_ends"],
        ["obj_core"],
        name="slice_obj_direct",
    ))

    nodes.append(helper.make_node(
        "Cast",
        ["obj_core"],
        ["obj_core_u8"],
        name="cast_obj_core_to_uint8",
        to=TensorProto.UINT8,
    ))

    nodes.append(helper.make_node(
        "Reshape",
        ["bg_core_u8", "term_1c_shape"],
        ["bg_term"],
        name="reshape_bg_term",
    ))

    nodes.append(helper.make_node(
        "Reshape",
        ["active_3x3_u8", "term_1c_shape"],
        ["active_term_1c"],
        name="reshape_active_term_1c",
    ))

    nodes.append(helper.make_node(
        "Expand",
        ["active_term_1c", "active_9c_shape"],
        ["active_term_9c"],
        name="expand_active_term_9c",
    ))

    nodes.append(helper.make_node(
        "Reshape",
        ["obj_core_u8", "term_9c_shape"],
        ["obj_term"],
        name="reshape_obj_term",
    ))

    nodes.append(helper.make_node(
        "Concat",
        ["bg_term", "one_term"],
        ["a_bg"],
        name="concat_a_bg",
        axis=2,
    ))

    nodes.append(helper.make_node(
        "Concat",
        ["active_term_9c", "active_term_9c"],
        ["a_obj"],
        name="concat_a_obj",
        axis=2,
    ))

    nodes.append(helper.make_node(
        "Concat",
        ["a_bg", "a_obj"],
        ["einsum_a"],
        name="concat_einsum_a",
        axis=1,
    ))

    nodes.append(helper.make_node(
        "Concat",
        ["one_term", "bg_term"],
        ["b_bg"],
        name="concat_b_bg",
        axis=2,
    ))

    nodes.append(helper.make_node(
        "Concat",
        ["obj_term", "obj_term"],
        ["b_obj"],
        name="concat_b_obj",
        axis=2,
    ))

    nodes.append(helper.make_node(
        "Concat",
        ["b_bg", "b_obj"],
        ["einsum_b"],
        name="concat_einsum_b",
        axis=1,
    ))

    nodes.append(helper.make_node(
        "Einsum",
        ["einsum_a", "einsum_b", "map30", "map30"],
        [OUT_NAME],
        name="direct_30x30_einsum_output",
        equation="nctij,nctrs,hir,wjs->nchw",
    ))

    input_vi = helper.make_tensor_value_info(
        IN_NAME,
        TensorProto.FLOAT,
        [1, 10, 30, 30],
    )
    output_vi = helper.make_tensor_value_info(
        OUT_NAME,
        TensorProto.UINT8,
        [1, 10, 30, 30],
    )
    graph = helper.make_graph(
        nodes,
        "task001_direct_einsum_variant_G",
        [input_vi],
        [output_vi],
        initializer=inits,
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 14)],
        ir_version=7,
    )


def build_einsum_float_direct_model() -> onnx.ModelProto:
    nodes = []
    inits = []

    def init(name, arr, dtype=np.int64):
        inits.append(numpy_helper.from_array(np.asarray(arr, dtype=dtype), name))

    def mapper_30() -> np.ndarray:
        out = np.zeros((30, 3, 3), dtype=np.float32)
        for block in range(3):
            for inner in range(3):
                out[block * 3 + inner, block, inner] = 1.0
        return out

    init("bg_starts", [0, 0, 0, 0])
    init("bg_ends", [1, 1, 3, 3])
    init("obj_starts", [0, 1, 0, 0])
    init("obj_ends", [1, 10, 3, 3])
    init("zero_f", [0.0], dtype=np.float32)
    init("term_1c_shape", [1, 1, 1, 3, 3])
    init("term_9c_shape", [1, 9, 1, 3, 3])
    init("active_9c_shape", [1, 9, 1, 3, 3])
    init("one_term", np.ones((1, 1, 1, 3, 3), dtype=np.float32), dtype=np.float32)
    init("map30", mapper_30(), dtype=np.float32)

    nodes.append(helper.make_node(
        "Slice",
        [IN_NAME, "bg_starts", "bg_ends"],
        ["bg_core"],
        name="slice_bg_direct",
    ))

    nodes.append(helper.make_node(
        "Greater",
        ["bg_core", "zero_f"],
        ["bg_core_bool"],
        name="bg_core_bool",
    ))

    nodes.append(helper.make_node(
        "Not",
        ["bg_core_bool"],
        ["active_3x3"],
        name="active_3x3",
    ))

    nodes.append(helper.make_node(
        "Cast",
        ["active_3x3"],
        ["active_3x3_f"],
        name="cast_active_3x3_to_float",
        to=TensorProto.FLOAT,
    ))

    nodes.append(helper.make_node(
        "Slice",
        [IN_NAME, "obj_starts", "obj_ends"],
        ["obj_core"],
        name="slice_obj_direct",
    ))

    nodes.append(helper.make_node(
        "Reshape",
        ["bg_core", "term_1c_shape"],
        ["bg_term"],
        name="reshape_bg_term",
    ))

    nodes.append(helper.make_node(
        "Reshape",
        ["active_3x3_f", "term_1c_shape"],
        ["active_term_1c"],
        name="reshape_active_term_1c",
    ))

    nodes.append(helper.make_node(
        "Expand",
        ["active_term_1c", "active_9c_shape"],
        ["active_term_9c"],
        name="expand_active_term_9c",
    ))

    nodes.append(helper.make_node(
        "Reshape",
        ["obj_core", "term_9c_shape"],
        ["obj_term"],
        name="reshape_obj_term",
    ))

    nodes.append(helper.make_node(
        "Concat",
        ["bg_term", "one_term"],
        ["a_bg"],
        name="concat_a_bg",
        axis=2,
    ))

    nodes.append(helper.make_node(
        "Concat",
        ["active_term_9c", "active_term_9c"],
        ["a_obj"],
        name="concat_a_obj",
        axis=2,
    ))

    nodes.append(helper.make_node(
        "Concat",
        ["a_bg", "a_obj"],
        ["einsum_a"],
        name="concat_einsum_a",
        axis=1,
    ))

    nodes.append(helper.make_node(
        "Concat",
        ["one_term", "bg_term"],
        ["b_bg"],
        name="concat_b_bg",
        axis=2,
    ))

    nodes.append(helper.make_node(
        "Concat",
        ["obj_term", "obj_term"],
        ["b_obj"],
        name="concat_b_obj",
        axis=2,
    ))

    nodes.append(helper.make_node(
        "Concat",
        ["b_bg", "b_obj"],
        ["einsum_b"],
        name="concat_einsum_b",
        axis=1,
    ))

    nodes.append(helper.make_node(
        "Einsum",
        ["einsum_a", "einsum_b", "map30", "map30"],
        [OUT_NAME],
        name="direct_30x30_float_einsum_output",
        equation="nctij,nctrs,hir,wjs->nchw",
    ))

    input_vi = helper.make_tensor_value_info(
        IN_NAME,
        TensorProto.FLOAT,
        [1, 10, 30, 30],
    )
    output_vi = helper.make_tensor_value_info(
        OUT_NAME,
        TensorProto.FLOAT,
        [1, 10, 30, 30],
    )
    graph = helper.make_graph(
        nodes,
        "task001_direct_float_einsum_variant_H",
        [input_vi],
        [output_vi],
        initializer=inits,
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 14)],
        ir_version=7,
    )


def build_convtranspose_direct_model() -> onnx.ModelProto:
    nodes = []
    inits = []

    def init(name, arr, dtype=np.int64):
        inits.append(numpy_helper.from_array(np.asarray(arr, dtype=dtype), name))

    def convtranspose_weight() -> np.ndarray:
        weight = np.zeros((100, 1, 24, 24), dtype=np.float32)

        # Group 0 -> output background channel. Channel 0 fills inactive
        # blocks; channels 1..9 place the background pattern pixels.
        weight[0, 0, :3, :3] = 1.0
        for inner in range(9):
            weight[1 + inner, 0, inner // 3, inner % 3] = 1.0

        # Groups 1..9 -> output object color channels 1..9. Each group has
        # 9 useful inner-pixel channels plus one zero dummy to keep groups
        # equal-sized for grouped ConvTranspose.
        for color_group in range(9):
            base = 10 * (color_group + 1)
            for inner in range(9):
                weight[base + inner, 0, inner // 3, inner % 3] = 1.0
        return weight

    init("bg_starts", [0, 0, 0, 0])
    init("bg_ends", [1, 1, 3, 3])
    init("obj_starts", [0, 1, 0, 0])
    init("obj_ends", [1, 10, 3, 3])
    init("zero_f", [0.0], dtype=np.float32)
    init("active_pair_shape", [1, 1, 1, 3, 3])
    init("bg_pattern_pair_shape", [1, 1, 9, 1, 1])
    init("obj_pair_shape", [1, 9, 9, 1, 1])
    init("bg_pattern_shape", [1, 9, 3, 3])
    init("obj_pair_flat_shape", [1, 81, 3, 3])
    init("ct_weight", convtranspose_weight(), dtype=np.float32)

    nodes.append(helper.make_node(
        "Slice",
        [IN_NAME, "bg_starts", "bg_ends"],
        ["bg_core"],
        name="slice_bg_direct",
    ))

    nodes.append(helper.make_node(
        "Greater",
        ["bg_core", "zero_f"],
        ["bg_core_bool"],
        name="bg_core_bool",
    ))

    nodes.append(helper.make_node(
        "Not",
        ["bg_core_bool"],
        ["active_3x3"],
        name="active_3x3",
    ))

    nodes.append(helper.make_node(
        "Cast",
        ["active_3x3"],
        ["active_3x3_f"],
        name="cast_active_3x3_to_float",
        to=TensorProto.FLOAT,
    ))

    nodes.append(helper.make_node(
        "Slice",
        [IN_NAME, "obj_starts", "obj_ends"],
        ["obj_core"],
        name="slice_obj_direct",
    ))

    nodes.append(helper.make_node(
        "Sub",
        ["active_3x3_f", "active_3x3_f"],
        ["zero_map"],
        name="zero_dummy_feature",
    ))

    nodes.append(helper.make_node(
        "Reshape",
        ["active_3x3_f", "active_pair_shape"],
        ["active_pair"],
        name="reshape_active_pair",
    ))

    nodes.append(helper.make_node(
        "Reshape",
        ["bg_core", "bg_pattern_pair_shape"],
        ["bg_pattern_pair"],
        name="reshape_bg_pattern_pair",
    ))

    nodes.append(helper.make_node(
        "Mul",
        ["active_pair", "bg_pattern_pair"],
        ["bg_pattern_pair_masked"],
        name="make_bg_pattern_features",
    ))

    nodes.append(helper.make_node(
        "Reshape",
        ["bg_pattern_pair_masked", "bg_pattern_shape"],
        ["bg_pattern_features"],
        name="reshape_bg_pattern_features",
    ))

    nodes.append(helper.make_node(
        "Reshape",
        ["obj_core", "obj_pair_shape"],
        ["obj_pair"],
        name="reshape_obj_pair",
    ))

    nodes.append(helper.make_node(
        "Mul",
        ["active_pair", "obj_pair"],
        ["obj_pair_masked"],
        name="make_obj_pair_features",
    ))

    nodes.append(helper.make_node(
        "Reshape",
        ["obj_pair_masked", "obj_pair_flat_shape"],
        ["obj_pair_flat"],
        name="reshape_obj_pair_flat",
    ))

    for color in range(9):
        start = color * 9
        end = start + 9
        init(f"color{color}_starts", [0, start, 0, 0])
        init(f"color{color}_ends", [1, end, 3, 3])
        nodes.append(helper.make_node(
            "Slice",
            ["obj_pair_flat", f"color{color}_starts", f"color{color}_ends"],
            [f"obj_color{color}_features"],
            name=f"slice_obj_color{color}_features",
        ))

    concat_inputs = ["bg_core", "bg_pattern_features"]
    for color in range(9):
        concat_inputs.extend([f"obj_color{color}_features", "zero_map"])

    nodes.append(helper.make_node(
        "Concat",
        concat_inputs,
        ["ct_input"],
        name="concat_convtranspose_input",
        axis=1,
    ))

    nodes.append(helper.make_node(
        "ConvTranspose",
        ["ct_input", "ct_weight"],
        [OUT_NAME],
        name="direct_30x30_convtranspose_output",
        group=10,
        kernel_shape=[24, 24],
        strides=[3, 3],
    ))

    input_vi = helper.make_tensor_value_info(
        IN_NAME,
        TensorProto.FLOAT,
        [1, 10, 30, 30],
    )
    output_vi = helper.make_tensor_value_info(
        OUT_NAME,
        TensorProto.FLOAT,
        [1, 10, 30, 30],
    )
    graph = helper.make_graph(
        nodes,
        "task001_direct_convtranspose_variant_I",
        [input_vi],
        [output_vi],
        initializer=inits,
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 14)],
        ir_version=7,
    )


def build_model(variant: str) -> onnx.ModelProto:
    if variant == "G":
        return build_einsum_direct_model()
    if variant == "H":
        return build_einsum_float_direct_model()
    if variant == "I":
        return build_convtranspose_direct_model()

    nodes = []
    inits = []

    def init(name, arr, dtype=np.int64):
        inits.append(numpy_helper.from_array(np.asarray(arr, dtype=dtype), name))

    # Background channel directly from input: [1,1,3,3]
    init("bg_starts", [0, 0, 0, 0])
    init("bg_ends", [1, 1, 3, 3])

    # Object channels directly from input: [1,9,3,3]
    init("obj_starts", [0, 1, 0, 0])
    init("obj_ends", [1, 10, 3, 3])

    init("zero_f", [0.0], dtype=np.float32)

    # Object pattern: [1,9,3,3] -> [1,9,1,3,1,3]
    init("obj_pattern_shape", [1, 9, 1, 3, 1, 3])

    # Active mask: [1,1,3,3] -> [1,1,3,1,3,1]
    init("active_shape", [1, 1, 3, 1, 3, 1])

    # Background pattern: [1,1,3,3] -> [1,1,1,3,1,3]
    init("bg_pattern_shape", [1, 1, 1, 3, 1, 3])

    init("obj9_shape", [1, 9, 9, 9])
    init("bg9_shape", [1, 1, 9, 9])

    # Pad [1,10,9,9] -> [1,10,30,30]
    init("pads", [0, 0, 0, 0, 0, 0, 21, 21])

    nodes.append(helper.make_node(
        "Slice",
        [IN_NAME, "bg_starts", "bg_ends"],
        ["bg_core"],
        name="slice_bg_direct",
    ))

    nodes.append(helper.make_node(
        "Greater",
        ["bg_core", "zero_f"],
        ["bg_core_bool"],
        name="bg_core_bool",
    ))

    nodes.append(helper.make_node(
        "Slice",
        [IN_NAME, "obj_starts", "obj_ends"],
        ["obj_core"],
        name="slice_obj_direct",
    ))

    # active = NOT(background)
    nodes.append(helper.make_node(
        "Not",
        ["bg_core_bool"],
        ["active_3x3"],
        name="active_3x3",
    ))

    if variant in {"D", "E", "F"}:
        nodes.append(helper.make_node(
            "Cast",
            ["active_3x3"],
            ["active_3x3_u8"],
            name="cast_active_3x3_to_uint8",
            to=TensorProto.UINT8,
        ))

        nodes.append(helper.make_node(
            "Reshape",
            ["active_3x3_u8", "active_shape"],
            ["active_6d_u8"],
            name="reshape_active_6d_u8",
        ))

        nodes.append(helper.make_node(
            "Cast",
            ["obj_core"],
            ["obj_core_u8"],
            name="cast_obj_core_to_uint8",
            to=TensorProto.UINT8,
        ))

        nodes.append(helper.make_node(
            "Reshape",
            ["obj_core_u8", "obj_pattern_shape"],
            ["obj_pattern_6d_u8"],
            name="reshape_obj_pattern_6d_u8",
        ))

        nodes.append(helper.make_node(
            "Mul",
            ["active_6d_u8", "obj_pattern_6d_u8"],
            ["obj_blocked_6d_u8"],
            name="apply_object_blocks_u8",
        ))

        nodes.append(helper.make_node(
            "Reshape",
            ["obj_blocked_6d_u8", "obj9_shape"],
            ["obj9"],
            name="reshape_obj9_u8",
        ))
    else:
        nodes.append(helper.make_node(
            "Greater",
            ["obj_core", "zero_f"],
            ["obj_core_bool"],
            name="obj_core_bool",
        ))

        nodes.append(helper.make_node(
            "Reshape",
            ["active_3x3", "active_shape"],
            ["active_6d"],
            name="reshape_active_6d",
        ))

        nodes.append(helper.make_node(
            "Reshape",
            ["obj_core_bool", "obj_pattern_shape"],
            ["obj_pattern_6d"],
            name="reshape_obj_pattern_6d",
        ))

        nodes.append(helper.make_node(
            "And",
            ["obj_pattern_6d", "active_6d"],
            ["obj_blocked_6d"],
            name="apply_object_blocks",
        ))

        nodes.append(helper.make_node(
            "Reshape",
            ["obj_blocked_6d", "obj9_shape"],
            ["obj9"],
            name="reshape_obj9",
        ))

    if variant == "D":
        nodes.append(helper.make_node(
            "Reshape",
            ["active_3x3", "active_shape"],
            ["active_6d"],
            name="reshape_active_6d",
        ))

    if variant == "E":
        nodes.append(helper.make_node(
            "Reshape",
            ["bg_core_bool", "bg_pattern_shape"],
            ["bg_pattern_6d"],
            name="reshape_bg_pattern_6d",
        ))

        nodes.append(helper.make_node(
            "Reshape",
            ["bg_core_bool", "active_shape"],
            ["inactive_block_6d"],
            name="reshape_inactive_block_6d",
        ))

        nodes.append(helper.make_node(
            "Or",
            ["bg_pattern_6d", "inactive_block_6d"],
            ["bg_6d"],
            name="final_bg_6d",
        ))

        nodes.append(helper.make_node(
            "Reshape",
            ["bg_6d", "bg9_shape"],
            ["bg9"],
            name="reshape_bg9",
        ))
    elif variant == "F":
        nodes.append(helper.make_node(
            "Cast",
            ["bg_core"],
            ["bg_core_u8"],
            name="cast_bg_core_to_uint8",
            to=TensorProto.UINT8,
        ))

        nodes.append(helper.make_node(
            "Reshape",
            ["bg_core_u8", "bg_pattern_shape"],
            ["bg_pattern_6d_u8"],
            name="reshape_bg_pattern_6d_u8",
        ))

        nodes.append(helper.make_node(
            "Reshape",
            ["bg_core_u8", "active_shape"],
            ["inactive_block_6d_u8"],
            name="reshape_inactive_block_6d_u8",
        ))

        nodes.append(helper.make_node(
            "Add",
            ["bg_pattern_6d_u8", "inactive_block_6d_u8"],
            ["bg_6d_u8"],
            name="final_bg_6d_u8",
        ))

        nodes.append(helper.make_node(
            "Reshape",
            ["bg_6d_u8", "bg9_shape"],
            ["bg9_u8"],
            name="reshape_bg9_u8",
        ))
    else:
        # For inactive blocks, the whole 3x3 block should be background.
        # For active blocks, background follows the original pattern.
        nodes.append(helper.make_node(
            "Reshape",
            ["bg_core_bool", "bg_pattern_shape"],
            ["bg_pattern_6d"],
            name="reshape_bg_pattern_6d",
        ))

        nodes.append(helper.make_node(
            "And",
            ["bg_pattern_6d", "active_6d"],
            ["active_bg_6d"],
            name="active_block_background",
        ))

        nodes.append(helper.make_node(
            "Not",
            ["active_6d"],
            ["inactive_6d"],
            name="inactive_blocks",
        ))

        nodes.append(helper.make_node(
            "Or",
            ["active_bg_6d", "inactive_6d"],
            ["bg_6d"],
            name="final_bg_6d",
        ))

        nodes.append(helper.make_node(
            "Reshape",
            ["bg_6d", "bg9_shape"],
            ["bg9"],
            name="reshape_bg9",
        ))

    if variant == "A":
        # Variant A: Concat(bool) -> Cast(uint8) -> Pad.
        nodes.append(helper.make_node(
            "Concat",
            ["bg9", "obj9"],
            ["out9_bool"],
            name="concat_bool_out9",
            axis=1,
        ))

        nodes.append(helper.make_node(
            "Cast",
            ["out9_bool"],
            ["out9"],
            name="cast_out9_to_uint8",
            to=TensorProto.UINT8,
        ))
    elif variant == "B":
        # Variant B: Cast both pieces before Concat, then Pad uint8.
        nodes.append(helper.make_node(
            "Cast",
            ["bg9"],
            ["bg9_u8"],
            name="cast_bg9_to_uint8",
            to=TensorProto.UINT8,
        ))

        nodes.append(helper.make_node(
            "Cast",
            ["obj9"],
            ["obj9_u8"],
            name="cast_obj9_to_uint8",
            to=TensorProto.UINT8,
        ))

        nodes.append(helper.make_node(
            "Concat",
            ["bg9_u8", "obj9_u8"],
            ["out9"],
            name="concat_uint8_out9",
            axis=1,
        ))
    elif variant in {"D", "E"}:
        # Variants D/E: object branch is already uint8 via Mul; only cast background.
        nodes.append(helper.make_node(
            "Cast",
            ["bg9"],
            ["bg9_u8"],
            name="cast_bg9_to_uint8",
            to=TensorProto.UINT8,
        ))

        nodes.append(helper.make_node(
            "Concat",
            ["bg9_u8", "obj9"],
            ["out9"],
            name="concat_uint8_out9",
            axis=1,
        ))
    elif variant == "F":
        # Variant F: both branches are already uint8 before Concat.
        nodes.append(helper.make_node(
            "Concat",
            ["bg9_u8", "obj9"],
            ["out9"],
            name="concat_uint8_out9",
            axis=1,
        ))
    else:
        raise ValueError(f"unknown variant: {variant}")

    nodes.append(helper.make_node(
        "Pad",
        ["out9", "pads"],
        [OUT_NAME],
        name="pad_to_30x30",
        mode="constant",
    ))

    input_vi = helper.make_tensor_value_info(
        IN_NAME,
        TensorProto.FLOAT,
        [1, 10, 30, 30],
    )

    output_vi = helper.make_tensor_value_info(
        OUT_NAME,
        TensorProto.UINT8,
        [1, 10, 30, 30],
    )

    graph = helper.make_graph(
        nodes,
        f"task001_uint8_lower_memory_variant_{variant}",
        [input_vi],
        [output_vi],
        initializer=inits,
    )

    # UINT8 Mul/Add are only admitted by the ONNX schema starting at opset 14.
    opset_version = 14 if variant in {"D", "E", "F"} else 11
    return helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", opset_version)],
        ir_version=7,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("A", "B", "D", "E", "F", "G", "H", "I"),
        default="F",
        help=(
            "A: Concat(bool)->Cast(uint8); "
            "B: Cast pieces before Concat; "
            "D: uint8 object Mul before Concat; "
            "E: simplified bool background; "
            "F: uint8 Add background; "
            "G: direct 30x30 Einsum output; "
            "H: direct 30x30 float Einsum output; "
            "I: direct 30x30 grouped ConvTranspose output."
        ),
    )
    args = parser.parse_args()

    model = build_model(args.variant)
    onnx.checker.check_model(model)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUT_PATH)

    print(f"saved: {OUT_PATH}")
    print(f"variant: {args.variant}")


if __name__ == "__main__":
    main()
