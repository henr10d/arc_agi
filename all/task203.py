"""Reverse concentric square-ring colors for NeuroGolf task203.

Task rule: the input is an even-sized square made from solid concentric
rectangular rings. The output keeps the same geometry and reverses the ring
color order: the center 2x2 color becomes the outer border, the outer border
color becomes the center 2x2 block, and each intermediate ring mirrors to the
opposite depth. Observed task sizes are 6x6 through 18x18, padded to the
competition 30x30 one-hot tensor.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

TASK_ID = "task203"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

BATCH = 1
COLORS = 10
HEIGHT = WIDTH = 30
MAX_SIZE = 18
SHAPE = [BATCH, COLORS, HEIGHT, WIDTH]
SIZES = tuple(range(6, 20, 2))
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def _i64(inits: list[onnx.TensorProto], vals: Iterable[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(list(vals), dtype=np.int64), name=name))
    return name


def _i64_arr(inits: list[onnx.TensorProto], arr: np.ndarray, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.int64), name=name))
    return name


def _i32(inits: list[onnx.TensorProto], vals: Iterable[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(list(vals), dtype=np.int32), name=name))
    return name


def _f32(inits: list[onnx.TensorProto], vals: Iterable[float], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(list(vals), dtype=np.float32), name=name))
    return name


def _f16(inits: list[onnx.TensorProto], vals: Iterable[float], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(list(vals), dtype=np.float16), name=name))
    return name


def _bool_arr(inits: list[onnx.TensorProto], arr: np.ndarray, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=bool), name=name))
    return name


def _layer_mask(size: int, layer: int) -> np.ndarray:
    rows = np.arange(size)[:, None]
    cols = np.arange(size)[None, :]
    dist = np.minimum(np.minimum(rows, cols), np.minimum(size - 1 - rows, size - 1 - cols))
    return (dist == layer).reshape(1, 1, size, size)


def _layer_index_map(size: int) -> np.ndarray:
    rows = np.arange(size)[:, None]
    cols = np.arange(size)[None, :]
    dist = np.minimum(np.minimum(rows, cols), np.minimum(size - 1 - rows, size - 1 - cols))
    return dist.reshape(1, size, size)


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    size = arr.shape[0]
    out = np.zeros_like(arr)
    layers = size // 2
    for layer in range(layers):
        src = layers - 1 - layer
        color = int(arr[src, src])
        mask = _layer_mask(size, layer).reshape(size, size)
        out[mask] = color
    return out


def infer_permutation_from_data() -> dict[int, tuple[int, ...]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    permutations: dict[int, set[tuple[int, ...]]] = {}
    for split in ("train", "test", "arc-gen"):
        for example in data.get(split, []):
            inp = np.asarray(example["input"], dtype=np.int64)
            out = np.asarray(example["output"], dtype=np.int64)
            layers = inp.shape[0] // 2
            in_colors: list[int] = []
            out_colors: list[int] = []
            for layer in range(layers):
                mask = _layer_mask(inp.shape[0], layer).reshape(inp.shape)
                in_colors.append(Counter(inp[mask].tolist()).most_common(1)[0][0])
                out_colors.append(Counter(out[mask].tolist()).most_common(1)[0][0])
            permutations.setdefault(layers, set()).add(tuple(in_colors.index(color) for color in out_colors))

    return {layers: next(iter(values)) for layers, values in sorted(permutations.items())}


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes_hw = _i64(inits, [2, 3], "axes_hw")
    zero = _f32(inits, [0.0], "zero")
    zero_h = _f16(inits, [0.0], "zero_h")
    onehot_depth = _i32(inits, [COLORS], "onehot_depth")
    onehot_values = _f32(inits, [0.0, 1.0], "onehot_values")
    flat_shape = _i64(inits, [1], "flat_shape")

    depth_colors: list[str] = []
    for depth_idx in range(max(SIZES) // 2):
        src_start = _i64(inits, [depth_idx, depth_idx], f"d{depth_idx}_src_start")
        src_end = _i64(inits, [depth_idx + 1, depth_idx + 1], f"d{depth_idx}_src_end")
        nodes.extend(
            [
                helper.make_node("Slice", [IN_NAME, src_start, src_end, axes_hw], [f"d{depth_idx}_onehot"]),
                helper.make_node("ArgMax", [f"d{depth_idx}_onehot"], [f"d{depth_idx}_idx"], axis=1, keepdims=1),
                helper.make_node("Cast", [f"d{depth_idx}_idx"], [f"d{depth_idx}_color"], to=TensorProto.FLOAT16),
                helper.make_node("Reshape", [f"d{depth_idx}_color", flat_shape], [f"d{depth_idx}_flat"]),
            ]
        )
        depth_colors.append(f"d{depth_idx}_flat")

    compact_candidates: list[str] = []
    prev_outside: str | None = None
    for size in SIZES:
        layers = size // 2
        if size == MAX_SIZE:
            if prev_outside is None:
                raise AssertionError("MAX_SIZE must not be the first candidate size")
            active = prev_outside
        else:
            outside_start = _i64(inits, [size, size], f"s{size}_outside_start")
            outside_end = _i64(inits, [size + 1, size + 1], f"s{size}_outside_end")
            nodes.extend(
                [
                    helper.make_node("Slice", [IN_NAME, outside_start, outside_end, axes_hw], [f"s{size}_outside_onehot"]),
                    helper.make_node("ReduceSum", [f"s{size}_outside_onehot"], [f"s{size}_outside_sum"], axes=[1], keepdims=0),
                    helper.make_node("Greater", [f"s{size}_outside_sum", zero], [f"s{size}_outside"]),
                    helper.make_node("Not", [f"s{size}_outside"], [f"s{size}_not_outside"]),
                ]
            )
            if prev_outside is None:
                active = f"s{size}_not_outside"
            else:
                nodes.append(helper.make_node("And", [prev_outside, f"s{size}_not_outside"], [f"s{size}_active"]))
                active = f"s{size}_active"
            prev_outside = f"s{size}_outside"

        colors: list[str] = []
        for layer in range(layers):
            src = layers - 1 - layer
            colors.append(depth_colors[src])

        layer_map = _i64_arr(inits, _layer_index_map(size), f"s{size}_layer_map")
        nodes.extend(
            [
                helper.make_node("Concat", colors, [f"s{size}_colors"], axis=0),
                helper.make_node("Gather", [f"s{size}_colors", layer_map], [f"s{size}_candidate"], axis=0),
            ]
        )
        pad_h = MAX_SIZE - size
        compact_candidate = f"s{size}_candidate"
        if pad_h:
            compact_candidate = f"s{size}_compact_candidate"
            nodes.append(
                helper.make_node(
                    "Pad",
                    [f"s{size}_candidate"],
                    [compact_candidate],
                    pads=[0, 0, 0, 0, pad_h, pad_h],
                    value=float(COLORS),
                )
            )
        nodes.append(
            helper.make_node(
                "Where",
                [active, compact_candidate, zero_h],
                [f"s{size}_compact"],
            )
        )
        compact_candidates.append(f"s{size}_compact")

    nodes.extend(
        [
            helper.make_node("Sum", compact_candidates, ["idx_compact"]),
            helper.make_node(
                "Pad",
                ["idx_compact"],
                ["idx_grid"],
                pads=[0, 0, 0, 0, HEIGHT - MAX_SIZE, WIDTH - MAX_SIZE],
                value=float(COLORS),
            ),
        ]
    )

    nodes.extend(
        [
            helper.make_node("Cast", ["idx_grid"], ["idx_i32"], to=TensorProto.INT32),
            helper.make_node("OneHot", ["idx_i32", onehot_depth, onehot_values], [OUT_NAME], axis=1),
        ]
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


def _grid_to_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def verify_model(path: Path) -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            expected_grid = np.asarray(example["output"], dtype=np.int64)
            got = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(example["input"])})[0]
            expected = _grid_to_onehot(expected_grid)
            if not np.array_equal(got > 0.0, expected > 0.0):
                raise AssertionError(f"{split}[{idx}] failed")
            solved = solve(example["input"])
            if not np.array_equal(solved, expected_grid):
                raise AssertionError(f"reference solve failed for {split}[{idx}]")


def main() -> None:
    permutations = infer_permutation_from_data()
    expected = {layers: tuple(reversed(range(layers))) for layers in range(3, 10)}
    if permutations != expected:
        raise AssertionError(f"unexpected layer permutations: {permutations}")

    model = build_model()
    onnx.save(model, BEST_PATH)
    verify_model(BEST_PATH)

    from score_model import score_file

    result = score_file(BEST_PATH)
    if not result["valid"]:
        raise RuntimeError(result["error"])
    print(f"wrote {BEST_PATH}")
    print(f"permutation by layer count: {permutations}")
    print(f"memory={result['memory']} params={result['params']} cost={result['cost']}")
    print(f"score={float(result['score']):.6f}")


if __name__ == "__main__":
    main()
