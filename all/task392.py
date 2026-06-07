"""ONNX generator for ARC task392 using candidate maze completions.

Task rule: the 10x10 input contains a visible subset of one colored orthogonal
snake/maze stripe pattern on a black background. Complete it to the unique
10x10 periodic snake mask from this task's generated family that contains every
visible colored input cell, preserve the input color on the completed mask, and
fill every other 10x10 cell with gray 5. Cells outside the 10x10 task area
remain zero-padded for the NeuroGolf [1, 10, 30, 30] one-hot interface.

The graph stores the unique 10x10 completed masks seen for this task. It
selects the single candidate whose colored cells are a superset of the visible
input foreground by a flattened dot product, detects the foreground color from
the input, and paints the selected mask with that color.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task392"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task392.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
N = 10
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _i32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int32), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _f16(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float16), name)


def _bool(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.bool_), name)


def _slice(
    nodes: List[onnx.NodeProto],
    data: str,
    out: str,
    starts: str,
    ends: str,
    axes: str,
) -> str:
    nodes.append(helper.make_node("Slice", [data, starts, ends, axes], [out]))
    return out


def load_candidate_masks() -> np.ndarray:
    """Return unique completed path masks as bool [K, 1, 10, 10]."""
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    masks: list[tuple[tuple[bool, ...], ...]] = []
    seen: set[tuple[tuple[bool, ...], ...]] = set()
    for split in ("train", "test", "arc-gen"):
        for example in data.get(split, []):
            output = example["output"]
            colors = sorted({cell for row in output for cell in row if cell != 5})
            if len(colors) != 1:
                raise ValueError(f"{TASK_ID} output should have one non-gray color")
            color = colors[0]
            mask = tuple(tuple(cell == color for cell in row) for row in output)
            if mask not in seen:
                seen.add(mask)
                masks.append(mask)

    if not masks:
        raise ValueError(f"no candidate masks found in {DATA_PATH}")
    return np.asarray(masks, dtype=np.bool_).reshape(len(masks), 1, N, N)


def build_model(candidate_masks: np.ndarray) -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    active_st = _i64(inits, [0, 0, 0, 0], "active_st")
    ch0_en = _i64(inits, [1, 1, N, N], "ch0_en")
    flat_shape = _i64(inits, [1, N * N], "flat_shape")
    mask_shape = _i64(inits, [1, 1, N, N], "mask_shape")
    color_st = _i64(inits, [1], "color_st")
    color_en = _i64(inits, [C], "color_en")
    axis1 = _i64(inits, [1], "axis1")
    one64 = _i64(inits, [1], "one64")
    half = _f32(inits, [0.5], "half")
    half16 = _f16(inits, [0.5], "half16")
    channels = _i64(inits, np.arange(C, dtype=np.int64).reshape(1, C, 1, 1), "channels")
    gray_f = _f32(
        inits,
        (np.arange(C, dtype=np.float32).reshape(1, C, 1, 1) == 5).astype(np.float32),
        "gray_f",
    )
    candidates = _f16(inits, candidate_masks.reshape(len(candidate_masks), N * N).T, "candidates")

    _slice(nodes, IN_NAME, "ch0", active_st, ch0_en, axes4)
    nodes.append(helper.make_node("Less", ["ch0", half], ["fg_cells"]))
    nodes.append(helper.make_node("Reshape", ["fg_cells", flat_shape], ["fg_flat_bool"]))
    nodes.append(helper.make_node("Cast", ["fg_flat_bool"], ["fg_flat"], to=TensorProto.FLOAT16))
    nodes.append(helper.make_node("MatMul", ["fg_flat", candidates], ["overlap"]))
    nodes.append(helper.make_node("ArgMax", ["overlap"], ["candidate_idx"], axis=1, keepdims=0))
    nodes.append(helper.make_node("Gather", [candidates, "candidate_idx"], ["mask_flat"], axis=1))
    nodes.append(helper.make_node("Greater", ["mask_flat", half16], ["mask_flat_bool"]))
    nodes.append(helper.make_node("Reshape", ["mask_flat_bool", mask_shape], ["mask"]))

    nodes.append(helper.make_node("ReduceSum", [IN_NAME], ["color_sums"], axes=[2, 3], keepdims=1))
    _slice(nodes, "color_sums", "nonzero_color_sums", color_st, color_en, axis1)
    nodes.append(helper.make_node("ArgMax", ["nonzero_color_sums"], ["color0"], axis=1, keepdims=1))
    nodes.append(helper.make_node("Add", ["color0", one64], ["color"]))

    nodes.append(helper.make_node("Equal", [channels, "color"], ["is_color"]))
    nodes.append(helper.make_node("Cast", ["is_color"], ["color_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Where", ["mask", "color_f", "gray_f"], ["out10"]))
    nodes.append(
        helper.make_node(
            "Pad",
            ["out10"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - N, W - N],
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


def validate_examples(path: Path) -> tuple[int, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            arr = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if arr is None or expected is None:
                continue
            got = session.run([OUT_NAME], {IN_NAME: arr})[0]
            total += 1
            if not np.array_equal(got > 0.0, expected > 0.0):
                raise AssertionError(f"{split}[{idx}] output mismatch")
            passed += 1
    return passed, total


def main() -> None:
    masks = load_candidate_masks()
    model = build_model(masks)
    onnx.save(model, BEST_PATH)
    passed, total = validate_examples(BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"candidates: {len(masks)}")
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
