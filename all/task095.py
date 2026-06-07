"""Minimal ONNX for ARC task095: gray-center 3x3 blue halos on a 9x9 grid.

Task rule: input is 9x9 with background 0 and isolated gray cells (color 5).
For each gray cell, paint a full 3x3 neighborhood blue (color 1) while the
center stays gray. Overlapping neighborhoods keep gray centers. Output is the
same 9x9 grid embedded in the competition [1, 10, 30, 30] one-hot tensor.

Observed constraints across train/test/arc-gen: gray centers are always in the
interior 7x7 area and are separated by at least a 3-cell Chebyshev distance.

ONNX approach: slice the 7x7 gray-center core, use one 3x3 ConvTranspose to
write compact 9x9 float logits for background, blue ring, and gray center
channels, then pad to the fixed competition output.
"""

from __future__ import annotations

import copy
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
sys.path.insert(0, str(ROOT))

from score_model import (  # noqa: E402
    calculate_memory,
    calculate_params,
    convert_to_numpy,
    sanitize_model,
    score,
    score_file,
)

TASK_ID = "task095"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task095.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
N = 9
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


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _make_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(nodes, name, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return model


def _gray_slice(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto]) -> str:
    starts = _i64(inits, [0, 5, 0, 0], "gray_st")
    ends = _i64(inits, [1, 6, N, N], "gray_en")
    nodes.append(helper.make_node("Slice", [IN_NAME, starts, ends], ["gray"]))
    return "gray"


def _gray_core7(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto]) -> str:
    starts = _i64(inits, [0, 5, 1, 1], "gray7_st")
    ends = _i64(inits, [1, 6, 8, 8], "gray7_en")
    nodes.append(helper.make_node("Slice", [IN_NAME, starts, ends], ["gray7"]))
    return "gray7"


def _hood_maxpool(nodes: list[onnx.NodeProto], gray: str) -> str:
    nodes.append(
        helper.make_node(
            "MaxPool",
            [gray],
            ["hood_f"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        )
    )
    return "hood_f"


def _hood_conv(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], gray: str) -> str:
    kernel = _f32(inits, np.ones((1, 1, 3, 3), dtype=np.float32), "conv3")
    nodes.append(
        helper.make_node(
            "Conv",
            [gray, kernel],
            ["hood_f"],
            pads=[1, 1, 1, 1],
        )
    )
    return "hood_f"


def _shift_pad(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    gray: str,
    dr: int,
    dc: int,
    name: str,
) -> str:
    starts = [0, 0, max(0, -dr), max(0, -dc)]
    ends = [1, 1, N - max(0, dr), N - max(0, dc)]
    pads = [0, 0, max(0, dr), max(0, dc), 0, 0, max(0, -dr), max(0, -dc)]
    axes = _i64(inits, [0, 1, 2, 3], f"{name}_ax")
    s = _i64(inits, starts, f"{name}_st")
    e = _i64(inits, ends, f"{name}_en")
    cropped = f"{name}_cr"
    shifted = f"{name}_sh"
    nodes.append(helper.make_node("Slice", [gray, s, e, axes], [cropped]))
    nodes.append(helper.make_node("Pad", [cropped], [shifted], mode="constant", pads=pads))
    return shifted


def _hood_shift(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], gray: str) -> str:
    zero = _f32(inits, [0.0], "zero_sh")
    shifted = [
        _shift_pad(nodes, inits, gray, dr, dc, f"sh{dr + 1}{dc + 1}")
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
    ]
    nodes.append(helper.make_node("Max", shifted, ["hood_f"]))
    nodes.append(helper.make_node("Greater", ["hood_f", zero], ["hood_b"]))
    return "hood_b"


def _assemble_output(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    gray: str,
    hood: str,
    *,
    hood_is_bool: bool,
) -> None:
    zero = _f32(inits, [0.0], "zero")
    if hood_is_bool:
        hood_b = hood
    else:
        nodes.append(helper.make_node("Greater", [hood, zero], ["hood_b"]))
        hood_b = "hood_b"

    nodes.extend(
        [
            helper.make_node("Greater", [gray, zero], ["gray_b"]),
            helper.make_node("Not", ["gray_b"], ["not_gray"]),
            helper.make_node("And", [hood_b, "not_gray"], ["blue_b"]),
            helper.make_node("Not", [hood_b], ["bg_b"]),
            helper.make_node("And", ["gray_b", "not_gray"], ["zero_b"]),
            helper.make_node(
                "Concat",
                ["bg_b", "blue_b", "zero_b", "zero_b", "zero_b", "gray_b", "zero_b", "zero_b", "zero_b", "zero_b"],
                ["out9b"],
                axis=1,
            ),
            helper.make_node("Cast", ["out9b"], ["out9"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out9"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - N, W - N],
            ),
        ]
    )


def _assemble_output_less(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    gray: str,
    hood: str,
) -> None:
    zero = _f32(inits, [0.0], "zero")
    nodes.extend(
        [
            helper.make_node("Greater", [hood, zero], ["hood_b"]),
            helper.make_node("Greater", [gray, zero], ["gray_b"]),
            helper.make_node("Less", [gray, hood], ["blue_b"]),
            helper.make_node("Not", ["hood_b"], ["bg_b"]),
            helper.make_node("Less", [hood, gray], ["zero_b"]),
            helper.make_node(
                "Concat",
                ["bg_b", "blue_b", "zero_b", "zero_b", "zero_b", "gray_b", "zero_b", "zero_b", "zero_b", "zero_b"],
                ["out9b"],
                axis=1,
            ),
            helper.make_node("Cast", ["out9b"], ["out9"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out9"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - N, W - N],
            ),
        ]
    )


def _assemble_output_float(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    gray: str,
    hood: str,
) -> None:
    half = _f32(inits, [0.5], "half")
    zeros = _f32(inits, np.zeros((1, 1, N, N), dtype=np.float32), "zeros9")
    nodes.extend(
        [
            helper.make_node("Less", [hood, half], ["bg_b"]),
            helper.make_node("Cast", ["bg_b"], ["bg"], to=TensorProto.FLOAT),
            helper.make_node("Sub", [hood, gray], ["blue"]),
            helper.make_node(
                "Concat",
                ["bg", "blue", zeros, zeros, zeros, gray, zeros, zeros, zeros, zeros],
                ["out9"],
                axis=1,
            ),
            helper.make_node(
                "Pad",
                ["out9"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - N, W - N],
            ),
        ]
    )


def _assemble_output_float_ones(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    gray: str,
    hood: str,
) -> None:
    ones = _f32(inits, np.ones((1, 1, N, N), dtype=np.float32), "ones9")
    zeros = _f32(inits, np.zeros((1, 1, N, N), dtype=np.float32), "zeros9")
    nodes.extend(
        [
            helper.make_node("Sub", [ones, hood], ["bg"]),
            helper.make_node("Sub", [hood, gray], ["blue"]),
            helper.make_node(
                "Concat",
                ["bg", "blue", zeros, zeros, zeros, gray, zeros, zeros, zeros, zeros],
                ["out9"],
                axis=1,
            ),
            helper.make_node(
                "Pad",
                ["out9"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - N, W - N],
            ),
        ]
    )


def _conv_logits(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], gray: str) -> None:
    kernel = np.zeros((10, 1, 3, 3), dtype=np.float32)
    bias = np.zeros((10,), dtype=np.float32)
    kernel[0, 0, :, :] = -1.0
    bias[0] = 1.0
    kernel[1, 0, :, :] = 1.0
    kernel[1, 0, 1, 1] = 0.0
    kernel[5, 0, 1, 1] = 1.0
    weights = _f32(inits, kernel, "conv_logits_w")
    biases = _f32(inits, bias, "conv_logits_b")
    nodes.extend(
        [
            helper.make_node("Conv", [gray, weights, biases], ["out9"], pads=[1, 1, 1, 1]),
            helper.make_node(
                "Pad",
                ["out9"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - N, W - N],
            ),
        ]
    )


def _convtranspose_logits(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], gray7: str) -> None:
    kernel = np.zeros((1, 10, 3, 3), dtype=np.float32)
    bias = np.zeros((10,), dtype=np.float32)
    kernel[0, 0, :, :] = -1.0
    bias[0] = 1.0
    kernel[0, 1, :, :] = 1.0
    kernel[0, 1, 1, 1] = 0.0
    kernel[0, 5, 1, 1] = 1.0
    weights = _f32(inits, kernel, "deconv_logits_w")
    biases = _f32(inits, bias, "deconv_logits_b")
    nodes.extend(
        [
            helper.make_node("ConvTranspose", [gray7, weights, biases], ["out9"]),
            helper.make_node(
                "Pad",
                ["out9"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - N, W - N],
            ),
        ]
    )


def build_model(variant: str) -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    if variant == "convt":
        gray7 = _gray_core7(nodes, inits)
        _convtranspose_logits(nodes, inits, gray7)
    else:
        gray = _gray_slice(nodes, inits)
    if variant == "maxpool":
        hood = _hood_maxpool(nodes, gray)
        _assemble_output(nodes, inits, gray, hood, hood_is_bool=False)
    elif variant == "conv":
        hood = _hood_conv(nodes, inits, gray)
        _assemble_output(nodes, inits, gray, hood, hood_is_bool=False)
    elif variant == "shift":
        hood = _hood_shift(nodes, inits, gray)
        _assemble_output(nodes, inits, gray, hood, hood_is_bool=True)
    elif variant == "less":
        hood = _hood_maxpool(nodes, gray)
        _assemble_output_less(nodes, inits, gray, hood)
    elif variant == "float":
        hood = _hood_maxpool(nodes, gray)
        _assemble_output_float(nodes, inits, gray, hood)
    elif variant == "float_ones":
        hood = _hood_maxpool(nodes, gray)
        _assemble_output_float_ones(nodes, inits, gray, hood)
    elif variant == "conv_logits":
        _conv_logits(nodes, inits, gray)
    elif variant == "convt":
        pass
    else:
        raise ValueError(variant)
    return _make_model(nodes, inits, f"{TASK_ID}_{variant}")


def _load_task() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _check_model(model: onnx.ModelProto) -> tuple[bool, str]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return False, "sanitize_model returned None"
    sess = ort.InferenceSession(sanitized.SerializeToString(), providers=["CPUExecutionProvider"])
    task = _load_task()
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(task.get(split, [])):
            x = convert_to_numpy(ex, "input")
            y = convert_to_numpy(ex, "output")
            if x is None or y is None:
                continue
            pred = sess.run([OUT_NAME], {IN_NAME: x})[0]
            if not np.array_equal(pred > 0.0, y > 0.0):
                return False, f"{split}[{idx}] mismatch"
    return True, "ok"


def _measure_model(model: onnx.ModelProto, path: Path) -> dict[str, Any]:
    sanitized = sanitize_model(copy.deepcopy(model))
    assert sanitized is not None
    inputs = [convert_to_numpy(ex, "input") for values in _load_task().values() for ex in values]
    profile_inputs = [arr for arr in inputs if arr is not None]
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(Path(tempfile.gettempdir()) / f"ng_{path.stem}")
    sess = ort.InferenceSession(sanitized.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    for arr in profile_inputs:
        sess.run([OUT_NAME], {IN_NAME: arr})
    trace = sess.end_profiling()
    memory = calculate_memory(sanitized, trace)
    params = calculate_params(sanitized)
    if memory is None or params is None:
        return {"valid": False, "memory": memory, "params": params, "cost": None, "score": None}
    cost = memory + params
    return {"valid": True, "memory": memory, "params": params, "cost": cost, "score": score(cost)}


def main() -> None:
    results: list[tuple[str, onnx.ModelProto, dict[str, Any]]] = []
    for variant in ("maxpool", "conv", "shift", "less", "float", "float_ones", "conv_logits", "convt"):
        model = build_model(variant)
        ok, reason = _check_model(model)
        if not ok:
            print(f"{variant}: invalid correctness: {reason}")
            continue
        tmp_path = OUT_DIR / f"{TASK_ID}_{variant}.onnx"
        onnx.save(model, tmp_path)
        measured = score_file(tmp_path)
        print(
            f"{variant}: memory={measured['memory']} params={measured['params']} "
            f"cost={measured['cost']} score={measured['score']:.6f}"
        )
        if measured["valid"]:
            results.append((variant, model, measured))
        tmp_path.unlink(missing_ok=True)

    if not results:
        raise SystemExit("no valid task095 variants")

    best_variant, best_model, best = min(results, key=lambda item: int(item[2]["cost"]))
    onnx.save(best_model, BEST_PATH)
    print(
        f"best={best_variant} wrote={BEST_PATH} memory={best['memory']} "
        f"params={best['params']} cost={best['cost']} score={best['score']:.6f}"
    )


if __name__ == "__main__":
    main()
