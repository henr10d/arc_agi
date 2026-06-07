"""Broadcast-memory ONNX for ARC task001 (3×3 stencil tiling).

Task rule: read the top-left 3×3 core; for each cell (i, j) with a non-background
color, copy the full core into output block [i*3:(i+1)*3, j*3:(j+1)*3]; inactive
blocks stay all-zero. One-hot I/O is [1,10,30,30]; logic runs on bool channels 1–9,
then channel 0 is reconstructed and the 9×9 result is cast to float before pad.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).resolve().parent / "task001.onnx"

IN_NAME = "input"
OUT_NAME = "output"
IR_VERSION = 10
OPSET = 10

nodes: list = []
inits: list = []


def init(name: str, arr, dtype=None) -> None:
    arr = np.asarray(arr)
    if dtype is not None:
        arr = arr.astype(dtype)
    inits.append(numpy_helper.from_array(arr, name=name))


input_vi = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, [1, 10, 30, 30])
output_vi = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, [1, 10, 30, 30])

# 1. Slice core [1,10,3,3]
init("core_starts", [0, 0, 0, 0], np.int64)
init("core_ends", [1, 10, 3, 3], np.int64)
init("core_axes", [0, 1, 2, 3], np.int64)
nodes.append(
    helper.make_node(
        "Slice",
        [IN_NAME, "core_starts", "core_ends", "core_axes"],
        ["core"],
        name="slice_core",
    )
)

# 2. Object channels 1..9 -> [1,9,3,3]
init("obj_starts", [0, 1, 0, 0], np.int64)
init("obj_ends", [1, 10, 3, 3], np.int64)
nodes.append(
    helper.make_node(
        "Slice",
        ["core", "obj_starts", "obj_ends", "core_axes"],
        ["obj"],
        name="slice_objects",
    )
)

init("zero", [0.0], np.float32)
init("zero_i64", [0], np.int64)

# 3. Object pattern bool [1,9,3,3]
nodes.append(
    helper.make_node("Greater", ["obj", "zero"], ["objb"], name="obj_bool")
)

# 4. Active-cell mask from channels 1..9 -> [1,1,3,3] (Cast: ORT opset 10 rejects ReduceSum on bool)
nodes.append(
    helper.make_node("Cast", ["objb"], ["obji"], to=TensorProto.INT64, name="obj_cast_i64")
)
nodes.append(
    helper.make_node(
        "ReduceSum",
        ["obji"],
        ["obj_sum"],
        name="obj_sum",
        axes=[1],
        keepdims=1,
    )
)
nodes.append(
    helper.make_node("Greater", ["obj_sum", "zero_i64"], ["active"], name="active_mask")
)

# 5–6. Reshape for broadcast tiling
init("pattern_shape", [1, 9, 1, 3, 1, 3], np.int64)
init("mask_shape", [1, 1, 3, 1, 3, 1], np.int64)
nodes.append(
    helper.make_node(
        "Reshape",
        ["objb", "pattern_shape"],
        ["pattern_6d"],
        name="reshape_pattern_6d",
    )
)
nodes.append(
    helper.make_node(
        "Reshape",
        ["active", "mask_shape"],
        ["mask_6d"],
        name="reshape_mask_6d",
    )
)

# 7. Broadcast AND -> [1,9,3,3,3,3]
nodes.append(
    helper.make_node(
        "And",
        ["pattern_6d", "mask_6d"],
        ["blocked_6d"],
        name="apply_block_mask",
    )
)

# 8. [1,9,9,9]
init("out9_shape", [1, 9, 9, 9], np.int64)
nodes.append(
    helper.make_node(
        "Reshape",
        ["blocked_6d", "out9_shape"],
        ["fg9"],
        name="reshape_fg9",
    )
)

# 9. Background channel: NOT(any object active) -> [1,1,9,9]
nodes.append(
    helper.make_node("Cast", ["fg9"], ["fgi"], to=TensorProto.INT64, name="fg_cast_i64")
)
nodes.append(
    helper.make_node(
        "ReduceSum",
        ["fgi"],
        ["fg_sum"],
        name="fg_sum",
        axes=[1],
        keepdims=1,
    )
)
nodes.append(
    helper.make_node("Greater", ["fg_sum", "zero_i64"], ["any_fg"], name="any_fg")
)
nodes.append(helper.make_node("Not", ["any_fg"], ["bg9"], name="bg9"))

# 10. [1,10,9,9]
nodes.append(
    helper.make_node("Concat", ["bg9", "fg9"], ["out9b"], axis=1, name="concat_ch")
)

# 11. Cast to float
nodes.append(
    helper.make_node("Cast", ["out9b"], ["out9"], to=TensorProto.FLOAT, name="cast_float")
)

# 12. Pad to [1,10,30,30]
nodes.append(
    helper.make_node(
        "Pad",
        ["out9"],
        [OUT_NAME],
        name="pad_to_30x30",
        mode="constant",
        pads=[0, 0, 0, 0, 0, 0, 21, 21],
    )
)

graph = helper.make_graph(
    nodes,
    "task001_broadcast_bool",
    [input_vi],
    [output_vi],
    initializer=inits,
)

model = helper.make_model(
    graph,
    opset_imports=[helper.make_operatorsetid("", OPSET)],
    ir_version=IR_VERSION,
)

onnx.checker.check_model(model)
onnx.save(model, OUT)
print(f"saved: {OUT}")
