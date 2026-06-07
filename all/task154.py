"""Compact ONNX generator for NeuroGolf task154.

Task rule: the active ARC grid is 15x15. Two opposing red bracket shapes
(color 2) remain fixed. The gray objects (color 5) outside the brackets move
inward along the bracket axis, reflected through the bracket opening: left/right
objects move horizontally into the gap, while top/bottom objects move vertically
into the gap. The original gray cells are removed and background remains color 0.

The graph keeps the top-left 15x15 crop, forms horizontal and vertical
reflected gray candidates, selects the orientation from a red probe cell that
is present only for left/right brackets, and rebuilds channels 0, 2, and 5
before padding back to the required 30x30 competition tensor. Width/height
choices are made on compact strips before padding to avoid realizing extra full
15x15 candidates. The gray source is cropped directly to the fixed boundary
strips where the generated examples place movable objects.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import calculate_params, convert_to_numpy, score_file  # noqa: E402


TASK_NUM = "154"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
IR_VERSION = 10


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []

    def init_i64(self, name: str, values: list[int]) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name))
        return name

    def init_f32(self, name: str, values: list[float]) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.float32), name))
        return name

    def node(self, op_type: str, inputs: list[str], output: str, **attrs: Any) -> str:
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        return output


def make_model(nodes: list[onnx.NodeProto], initializers: list[onnx.TensorProto]) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 10)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def build_model() -> onnx.ModelProto:
    b = Builder()
    b.init_i64("red_starts", [2, 0, 0])
    b.init_i64("red_ends", [3, 15, 15])
    b.init_i64("hleft_starts", [5, 3, 0])
    b.init_i64("hleft_ends", [6, 11, 2])
    b.init_i64("hright_starts", [5, 3, 12])
    b.init_i64("hright_ends", [6, 11, 15])
    b.init_i64("vtop_starts", [5, 0, 3])
    b.init_i64("vtop_ends", [6, 2, 11])
    b.init_i64("vbottom_starts", [5, 12, 3])
    b.init_i64("vbottom_ends", [6, 15, 11])
    b.init_i64("color_crop_axes", [1, 2, 3])
    b.init_i64("rev2", [1, 0])
    b.init_i64("last2of3", [2, 1])
    b.init_i64("probe_starts", [5, 3])
    b.init_i64("probe_ends", [6, 4])
    b.init_i64("probe_axes", [2, 3])
    b.init_i64("wide_probe_starts", [5, 11])
    b.init_i64("wide_probe_ends", [6, 12])
    b.init_i64("tall_probe_starts", [11, 5])
    b.init_i64("tall_probe_ends", [12, 6])
    b.init_f32("one", [1.0])

    red = b.node("Slice", [IN_NAME, "red_starts", "red_ends", "color_crop_axes"], "red")
    hleft = b.node("Slice", [IN_NAME, "hleft_starts", "hleft_ends", "color_crop_axes"], "hleft")
    hright = b.node("Slice", [IN_NAME, "hright_starts", "hright_ends", "color_crop_axes"], "hright")
    vtop = b.node("Slice", [IN_NAME, "vtop_starts", "vtop_ends", "color_crop_axes"], "vtop")
    vbottom = b.node("Slice", [IN_NAME, "vbottom_starts", "vbottom_ends", "color_crop_axes"], "vbottom")

    left_src = b.node("Gather", [hleft, "rev2"], "left_src", axis=3)
    right10_src = b.node("Gather", [hright, "rev2"], "right10_src", axis=3)
    right11_src = b.node("Gather", [hright, "last2of3"], "right11_src", axis=3)
    right10_w = b.node(
        "Pad",
        [right10_src],
        "right10_w",
        mode="constant",
        pads=[0, 0, 0, 0, 0, 0, 0, 1],
        value=0.0,
    )
    right11_w = b.node(
        "Pad",
        [right11_src],
        "right11_w",
        mode="constant",
        pads=[0, 0, 0, 1, 0, 0, 0, 0],
        value=0.0,
    )
    wide_probe = b.node("Slice", [red, "wide_probe_starts", "wide_probe_ends", "probe_axes"], "wide_probe")
    wide = b.node("Cast", [wide_probe], "wide", to=TensorProto.BOOL)
    right_src = b.node("Where", [wide, right11_w, right10_w], "right_src")
    hstrip = b.node("Concat", ["left_src", right_src], "hstrip", axis=3)
    horiz = b.node(
        "Pad",
        [hstrip],
        "horiz",
        mode="constant",
        pads=[0, 0, 3, 5, 0, 0, 4, 5],
        value=0.0,
    )

    top_src = b.node("Gather", [vtop, "rev2"], "top_src", axis=2)
    bottom10_src = b.node("Gather", [vbottom, "rev2"], "bottom10_src", axis=2)
    bottom11_src = b.node("Gather", [vbottom, "last2of3"], "bottom11_src", axis=2)
    bottom10_h = b.node(
        "Pad",
        [bottom10_src],
        "bottom10_h",
        mode="constant",
        pads=[0, 0, 0, 0, 0, 0, 1, 0],
        value=0.0,
    )
    bottom11_h = b.node(
        "Pad",
        [bottom11_src],
        "bottom11_h",
        mode="constant",
        pads=[0, 0, 1, 0, 0, 0, 0, 0],
        value=0.0,
    )
    tall_probe = b.node("Slice", [red, "tall_probe_starts", "tall_probe_ends", "probe_axes"], "tall_probe")
    tall = b.node("Cast", [tall_probe], "tall", to=TensorProto.BOOL)
    bottom_src = b.node("Where", [tall, bottom11_h, bottom10_h], "bottom_src")
    vstrip = b.node("Concat", ["top_src", bottom_src], "vstrip", axis=2)
    vert = b.node(
        "Pad",
        [vstrip],
        "vert",
        mode="constant",
        pads=[0, 0, 5, 3, 0, 0, 5, 4],
        value=0.0,
    )

    probe = b.node("Slice", [red, "probe_starts", "probe_ends", "probe_axes"], "probe")
    side = b.node("Cast", [probe], "side", to=TensorProto.BOOL)
    moved = b.node("Where", [side, horiz, vert], "moved")

    occupied = b.node("Add", [red, moved], "occupied")
    bg = b.node("Sub", ["one", occupied], "bg")
    zero = b.node("Sub", [red, red], "zero")
    small = b.node(
        "Concat",
        ["bg", zero, red, zero, zero, moved, zero, zero, zero, zero],
        "small",
        axis=1,
    )
    b.node(
        "Pad",
        [small],
        OUT_NAME,
        mode="constant",
        pads=[0, 0, 0, 0, 0, 0, 15, 15],
        value=0.0,
    )
    return make_model(b.nodes, b.initializers)


def load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def verify_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    all_ok = True
    splits: dict[str, tuple[int, int]] = {}
    for split, examples in load_task_data().items():
        passed = 0
        checked = 0
        for example in examples:
            inp = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if inp is None or expected is None:
                continue
            pred = session.run([OUT_NAME], {IN_NAME: inp})[0]
            checked += 1
            if np.array_equal(pred > 0.0, expected > 0.0):
                passed += 1
            else:
                all_ok = False
        splits[split] = (passed, checked)
    return all_ok, splits


def inferred_shapes(model: onnx.ModelProto) -> list[tuple[str, str, list[int]]]:
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    out: list[tuple[str, str, list[int]]] = []
    for value in graph.value_info:
        tensor_type = value.type.tensor_type
        dtype = TensorProto.DataType.Name(tensor_type.elem_type)
        shape = [int(dim.dim_value) for dim in tensor_type.shape.dim]
        out.append((value.name, dtype, shape))
    return out


def main() -> None:
    model = build_model()
    ok, splits = verify_correct(model)
    if not ok:
        raise SystemExit(f"validation failed: {splits}")

    onnx.save(model, str(BEST_PATH))
    result = score_file(BEST_PATH)
    params = calculate_params(model)
    shapes = inferred_shapes(model)
    estimated_memory = sum(
        int(np.prod(shape)) * np.dtype(onnx.helper.tensor_dtype_to_np_dtype(getattr(TensorProto, dtype))).itemsize
        for _, dtype, shape in shapes
    )

    print(f"wrote {BEST_PATH}")
    print(f"validation: {splits}")
    print(f"params: {params}")
    print(f"estimated inferred internal memory: {estimated_memory}")
    print(
        "score_model: "
        f"valid={result['valid']} memory={result.get('memory')} params={result.get('params')} "
        f"cost={result.get('cost')} score={result.get('score')}"
    )
    print("inferred internal tensor shapes:")
    for name, dtype, shape in shapes:
        print(f"  {name}: {dtype} {shape}")


if __name__ == "__main__":
    main()
