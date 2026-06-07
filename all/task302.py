"""ONNX generator for ARC task302 using hollow-frame detection.

Task rule: preserve every gray (5) hollow square frame and fill its black
interior according to the interior area observed in the training examples:
1x1 -> purple/magenta (6), 2x2 -> orange/brown (7), and 3x3 -> cyan/teal (8).
The task grids are 12x12 inside the NeuroGolf 30x30 padded tensor, and the
output keeps the same grid size and object positions.

The graph detects complete 3x3, 4x4, and 5x5 gray frames in the active 12x12
region. A signed kernel rewards gray border cells and heavily penalizes gray
interior cells, so one Conv per frame size is enough to locate valid top-left
corners. ConvTranspose expands those detections over the matching interiors.
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

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task302"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task302.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
ACTIVE = 12
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _i32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int32), name)


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _slice(
    nodes: list[onnx.NodeProto],
    data: str,
    starts: str,
    ends: str,
    axes: str | None,
    out: str,
) -> None:
    inputs = [data, starts, ends]
    if axes is not None:
        inputs.append(axes)
    nodes.append(helper.make_node("Slice", inputs, [out]))


def _frame_kernels(size: int) -> tuple[np.ndarray, np.ndarray]:
    inner = size - 2
    border_count = size * size - inner * inner

    detect = np.zeros((1, 1, size, size), dtype=np.float32)
    detect[:, :, 0, :] = 1.0
    detect[:, :, -1, :] = 1.0
    detect[:, :, :, 0] = 1.0
    detect[:, :, :, -1] = 1.0
    detect[:, :, 1:-1, 1:-1] = -float(border_count)

    expand = np.zeros((1, 1, size, size), dtype=np.float32)
    expand[:, :, 1:-1, 1:-1] = 1.0
    return detect, expand


def _pad_shift(
    nodes: list[onnx.NodeProto],
    data: str,
    in_hw: int,
    row_offset: int,
    col_offset: int,
    out: str,
) -> None:
    nodes.append(
        helper.make_node(
            "Pad",
            [data],
            [out],
            mode="constant",
            pads=[
                0,
                0,
                row_offset,
                col_offset,
                0,
                0,
                ACTIVE - in_hw - row_offset,
                ACTIVE - in_hw - col_offset,
            ],
        )
    )


def _detect_fill(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    gray: str,
    size: int,
) -> str:
    detect, expand = _frame_kernels(size)
    inner_area = (size - 2) * (size - 2)
    border_count = size * size - inner_area

    detect_w = _f32(inits, detect, f"detect_w_{size}")
    detect_threshold = _f32(inits, [float(border_count) - 0.5], f"detect_t_{size}")

    nodes.append(helper.make_node("Conv", [gray, detect_w], [f"frame_sum_{size}"]))

    frame_hw = ACTIVE - size + 1
    if size == 3:
        _pad_shift(nodes, f"frame_sum_{size}", frame_hw, 1, 1, f"fill_sum_{size}")
        nodes.append(helper.make_node("Greater", [f"fill_sum_{size}", detect_threshold], [f"fill_{size}"]))
        return f"fill_{size}"

    expand_w = _f32(inits, expand, f"expand_w_{size}")
    half = _f32(inits, [0.5], f"half_{size}")
    nodes.append(helper.make_node("Greater", [f"frame_sum_{size}", detect_threshold], [f"frame_{size}"]))
    nodes.append(helper.make_node("Cast", [f"frame_{size}"], [f"frame_f_{size}"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("ConvTranspose", [f"frame_f_{size}", expand_w], [f"fill_f_{size}"]))
    nodes.append(helper.make_node("Greater", [f"fill_f_{size}", half], [f"fill_{size}"]))
    return f"fill_{size}"


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    gray_st = _i64(inits, [0, 5, 0, 0], "gray_st")
    gray_en = _i64(inits, [1, 6, ACTIVE, ACTIVE], "gray_en")
    half = _f32(inits, [0.5], "half")

    _slice(nodes, IN_NAME, gray_st, gray_en, None, "gray_f")
    nodes.append(helper.make_node("Greater", ["gray_f", half], ["gray"]))

    fill3 = _detect_fill(nodes, inits, "gray_f", 3)
    fill4 = _detect_fill(nodes, inits, "gray_f", 4)
    fill5 = _detect_fill(nodes, inits, "gray_f", 5)

    nodes.append(helper.make_node("Or", ["gray", fill3], ["used_a"]))
    nodes.append(helper.make_node("Or", [fill4, fill5], ["used_b"]))
    nodes.append(helper.make_node("Or", ["used_a", "used_b"], ["used"]))
    nodes.append(helper.make_node("Not", ["used"], ["bg"]))
    nodes.append(helper.make_node("And", ["bg", "gray"], ["false"]))
    nodes.append(
        helper.make_node(
            "Concat",
            ["bg", "false", "false", "false", "false", "gray", fill3, fill4, fill5, "false"],
            ["out_bool"],
            axis=1,
        )
    )
    nodes.append(helper.make_node("Cast", ["out_bool"], ["out_small"], to=TensorProto.FLOAT))
    nodes.append(
        helper.make_node(
            "Pad",
            ["out_small"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - ACTIVE, W - ACTIVE],
        )
    )

    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _examples() -> list[tuple[str, int, dict[str, Any]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [
        (split, idx, example)
        for split in ("train", "test", "arc-gen")
        for idx, example in enumerate(data.get(split, []))
    ]


def verify_correct(model_path: Path) -> tuple[int, int]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    examples = _examples()
    for split, idx, example in examples:
        actual = session.run([OUT_NAME], {IN_NAME: convert_to_numpy(example, "input")})[0]
        expected = convert_to_numpy(example, "output")
        if not np.array_equal(actual > 0.0, expected > 0.0):
            raise AssertionError(f"{split}[{idx}] output mismatch")
    return len(examples), len(examples)


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    passed, total = verify_correct(BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"correct: {passed}/{total}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
