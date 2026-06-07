"""Minimal ONNX for ARC task068 using unique-color singleton dilation.

Task rule: in the 10x10 input, find the non-background color that appears
exactly once.  The output is black everywhere except the singleton's 3x3
neighborhood: the center keeps its original singleton color and the eight
neighboring cells are red.  The competition tensor is still padded to 30x30,
so only the top-left 10x10 region is populated in the output.
"""

from __future__ import annotations

import copy
import json
import math
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

TASK_ID = "task068"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task068.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
N = 10
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


def _bool(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.bool_), name)


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
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _slice_10(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto]) -> str:
    starts = _i64(inits, [0, 1, 0, 0], "starts10")
    ends = _i64(inits, [1, C, N, N], "ends10")
    axes = _i64(inits, [0, 1, 2, 3], "axes4")
    nodes.append(helper.make_node("Slice", [IN_NAME, starts, ends, axes], ["x9"]))
    return "x9"


def _singleton_masks(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], x10: str) -> tuple[str, str, list[str]]:
    one_half = _f32(inits, [1.5], "one_half")
    nodes.extend(
        [
            helper.make_node("ReduceSum", [x10], ["counts"], axes=[2, 3], keepdims=1),
            helper.make_node("Less", ["counts", one_half], ["unique_any"]),
            helper.make_node("Cast", [x10], ["x10b"], to=TensorProto.BOOL),
            helper.make_node("And", ["x10b", "unique_any"], ["singleton_channels"]),
        ]
    )
    split_names = [f"m{i}" for i in range(1, C)]
    nodes.append(helper.make_node("Split", ["singleton_channels"], split_names, axis=1, split=[1] * (C - 1)))
    current = split_names[0]
    for idx, item in enumerate(split_names[1:], start=2):
        out = f"singleton_or{idx}"
        nodes.append(helper.make_node("Or", [current, item], [out]))
        current = out
    return "singleton_channels", current, split_names


def _neighborhood_maxpool(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], singleton: str) -> str:
    half = _f32(inits, [0.5], "half")
    nodes.extend(
        [
            helper.make_node("Cast", [singleton], ["singleton_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "MaxPool",
                ["singleton_f"],
                ["neighborhood_f"],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                strides=[1, 1],
            ),
            helper.make_node("Greater", ["neighborhood_f", half], ["neighborhood"]),
        ]
    )
    return "neighborhood"


def _neighborhood_conv(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], singleton: str) -> str:
    zero = _f32(inits, [0.0], "zero")
    kernel = _f32(inits, np.ones((1, 1, 3, 3), dtype=np.float32), "conv_kernel")
    nodes.extend(
        [
            helper.make_node("Cast", [singleton], ["singleton_f"], to=TensorProto.FLOAT),
            helper.make_node("Conv", ["singleton_f", kernel], ["neighborhood_f"], pads=[1, 1, 1, 1]),
            helper.make_node("Greater", ["neighborhood_f", zero], ["neighborhood"]),
        ]
    )
    return "neighborhood"


def _shift_pad(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    singleton: str,
    dr: int,
    dc: int,
    name: str,
) -> str:
    starts = [0, 0, max(0, -dr), max(0, -dc)]
    ends = [1, 1, N - max(0, dr), N - max(0, dc)]
    pads = [0, 0, max(0, dr), max(0, dc), 0, 0, max(0, -dr), max(0, -dc)]
    axes = _i64(inits, [0, 1, 2, 3], f"{name}_axes")
    s = _i64(inits, starts, f"{name}_starts")
    e = _i64(inits, ends, f"{name}_ends")
    cropped = f"{name}_crop"
    shifted = f"{name}_shift"
    nodes.append(helper.make_node("Slice", [singleton, s, e, axes], [cropped]))
    nodes.append(helper.make_node("Pad", [cropped], [shifted], mode="constant", pads=pads))
    return shifted


def _neighborhood_shift(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], singleton: str) -> str:
    zero = _f32(inits, [0.0], "zero")
    nodes.append(helper.make_node("Cast", [singleton], ["singleton_f"], to=TensorProto.FLOAT))
    shifted = [
        _shift_pad(nodes, inits, "singleton_f", dr, dc, f"s{dr + 1}{dc + 1}")
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
    ]
    nodes.append(helper.make_node("Max", shifted, ["neighborhood_f"]))
    nodes.append(helper.make_node("Greater", ["neighborhood_f", zero], ["neighborhood"]))
    return "neighborhood"


def _output_from_masks(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    singleton_channels: str,
    singleton: str,
    neighborhood: str,
    split_names: list[str],
) -> None:
    del singleton_channels
    nodes.extend(
        [
            helper.make_node("Not", [singleton], ["not_singleton"]),
            helper.make_node("And", ["neighborhood", "not_singleton"], ["ring"]),
            helper.make_node("Not", ["neighborhood"], ["bg"]),
        ]
    )
    nodes.append(helper.make_node("Or", ["m2", "ring"], ["red"]))
    nodes.append(helper.make_node("Concat", ["bg", "m1", "red", "m3", "m4", "m5", "m6", "m7", "m8", "m9"], ["out10b"], axis=1))
    nodes.append(helper.make_node("Cast", ["out10b"], ["out10"], to=TensorProto.FLOAT))
    nodes.append(
        helper.make_node(
            "Pad",
            ["out10"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - N, W - N],
        )
    )


def build_model(variant: str) -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    x10 = _slice_10(nodes, inits)
    singleton_channels, singleton, split_names = _singleton_masks(nodes, inits, x10)
    if variant == "maxpool":
        neighborhood = _neighborhood_maxpool(nodes, inits, singleton)
    elif variant == "conv":
        neighborhood = _neighborhood_conv(nodes, inits, singleton)
    elif variant == "shift":
        neighborhood = _neighborhood_shift(nodes, inits, singleton)
    else:
        raise ValueError(variant)
    _output_from_masks(nodes, inits, singleton_channels, singleton, neighborhood, split_names)
    return _make_model(nodes, inits, f"{TASK_ID}_{variant}")


def _load_task() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _expected_grid(grid: list[list[int]]) -> list[list[int]]:
    counts = {color: sum(row.count(color) for row in grid) for color in range(1, C)}
    singletons = [color for color, count in counts.items() if count == 1]
    if len(singletons) != 1:
        raise ValueError(f"expected exactly one non-background singleton, found {singletons}")
    color = singletons[0]
    pos = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == color][0]
    out = [[0 for _ in range(N)] for _ in range(N)]
    rr, cc = pos
    for r in range(max(0, rr - 1), min(N, rr + 2)):
        for c in range(max(0, cc - 1), min(N, cc + 2)):
            out[r][c] = 2
    out[rr][cc] = color
    return out


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
    for variant in ("shift", "maxpool", "conv"):
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
            f"cost={measured['cost']} score={measured['score']}"
        )
        if measured["valid"]:
            results.append((variant, model, measured))
        tmp_path.unlink(missing_ok=True)

    if not results:
        raise SystemExit("no valid task068 variants")

    best_variant, best_model, best = min(results, key=lambda item: int(item[2]["cost"]))
    onnx.save(best_model, BEST_PATH)
    print(
        f"best={best_variant} wrote={BEST_PATH} memory={best['memory']} "
        f"params={best['params']} cost={best['cost']} score={best['score']:.6f}"
    )


if __name__ == "__main__":
    main()
