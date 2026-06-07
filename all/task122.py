"""ONNX generator for NeuroGolf task122.

Task rule: move the red 3x3 ring two cells along the dashed green guide.
If the green guide has multiple cells in a row, clear the original red cells
and repaint them two columns to the right. Otherwise, clear the original red
cells and repaint them two rows down. The dashed green guide is preserved.
Inputs and outputs use the NeuroGolf one-hot [1, 10, 30, 30] interface; cells
outside the original ARC grid remain all-zero padding.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import calculate_params, convert_to_numpy, print_report, score_file  # noqa: E402

TASK_ID = "task122"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task122.onnx"
ROOT_BEST_PATH = ROOT / "task122_best.onnx"
TMP_DIR = ROOT / ".tmp_task122"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
SHAPE = (1, 10, 30, 30)


@dataclass(frozen=True)
class Variant:
    name: str
    model: onnx.ModelProto
    path: Path


def init_i64(name: str, value: int) -> onnx.TensorProto:
    return numpy_helper.from_array(np.array([value], dtype=np.int64), name)


def init_f32(name: str, value: float) -> onnx.TensorProto:
    return numpy_helper.from_array(np.array(value, dtype=np.float32), name)


def init_bool_array(name: str, shape: tuple[int, ...], value: bool = False) -> onnx.TensorProto:
    return numpy_helper.from_array(np.full(shape, value, dtype=np.bool_), name)


def node(op_type: str, inputs: Iterable[str], outputs: Iterable[str], **attrs: object) -> onnx.NodeProto:
    return helper.make_node(op_type, list(inputs), list(outputs), **attrs)


def slice_channel(nodes: list[onnx.NodeProto], x: str, y: str, channel: int) -> None:
    nodes.append(node("Slice", [x, f"i{channel}", f"i{channel + 1}", "axis_c"], [y]))


def finish_model(nodes: list[onnx.NodeProto], initializers: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(SHAPE))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, list(SHAPE))],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 10)], ir_version=10)
    onnx.checker.check_model(model)
    return onnx.shape_inference.infer_shapes(model, strict_mode=True)


def add_common_prefix(nodes: list[onnx.NodeProto]) -> None:
    slice_channel(nodes, "input", "red_f", 2)
    slice_channel(nodes, "input", "green_f", 3)

    nodes.extend(
        [
            node("ReduceMax", ["input"], ["valid_f"], axes=[1], keepdims=1),
            node("Cast", ["valid_f"], ["valid"], to=TensorProto.BOOL),
            node("Cast", ["red_f"], ["red"], to=TensorProto.BOOL),
            node("Cast", ["green_f"], ["green"], to=TensorProto.BOOL),
            node("ReduceSum", ["green_f"], ["row_counts"], axes=[3], keepdims=1),
            node("ReduceMax", ["row_counts"], ["max_row_count"], axes=[2], keepdims=1),
            node("Greater", ["max_row_count", "f1"], ["is_h"]),
        ]
    )


def add_common_suffix(nodes: list[onnx.NodeProto], shifted_red: str) -> None:
    nodes.extend(
        [
            node("Or", ["green", shifted_red], ["painted"]),
            node("Not", ["painted"], ["not_painted"]),
            node("And", ["valid", "not_painted"], ["out_bg"]),
            node(
                "Concat",
                [
                    "out_bg",
                    "zero_full",
                    shifted_red,
                    "green",
                    "zero_full",
                    "zero_full",
                    "zero_full",
                    "zero_full",
                    "zero_full",
                    "zero_full",
                ],
                ["out_bool"],
                axis=1,
            ),
            node("Cast", ["out_bool"], ["output"], to=TensorProto.FLOAT),
        ]
    )


def build_slice_concat_variant() -> onnx.ModelProto:
    """Shift bool masks using Slice + Concat. Higher params, lower memory."""
    initializers = [
        init_i64("i0", 0),
        init_i64("i2", 2),
        init_i64("i3", 3),
        init_i64("i4", 4),
        init_i64("i28", 28),
        init_i64("axis_c", 1),
        init_i64("axis_h", 2),
        init_i64("axis_w", 3),
        init_f32("f1", 1.0),
        init_bool_array("zero_full", (1, 1, 30, 30)),
    ]
    nodes: list[onnx.NodeProto] = []
    add_common_prefix(nodes)
    nodes.extend(
        [
            node("Slice", ["red", "i0", "i28", "axis_w"], ["red_cols_0_28"]),
            node("Slice", ["zero_full", "i0", "i2", "axis_w"], ["zero_cols_2"]),
            node("Concat", ["zero_cols_2", "red_cols_0_28"], ["red_right"], axis=3),
            node("Slice", ["red", "i0", "i28", "axis_h"], ["red_rows_0_28"]),
            node("Slice", ["zero_full", "i0", "i2", "axis_h"], ["zero_rows_2"]),
            node("Concat", ["zero_rows_2", "red_rows_0_28"], ["red_down"], axis=2),
            node("And", ["is_h", "red_right"], ["red_right_if_h"]),
            node("Not", ["is_h"], ["is_v"]),
            node("And", ["is_v", "red_down"], ["red_down_if_v"]),
            node("Or", ["red_right_if_h", "red_down_if_v"], ["red_out"]),
        ]
    )
    add_common_suffix(nodes, "red_out")
    return finish_model(nodes, initializers, "task122_slice_concat")


def build_pad_float_variant() -> onnx.ModelProto:
    """Shift red with opset-10 Pad negative crops. Lower params, higher memory."""
    initializers = [
        init_i64("i2", 2),
        init_i64("i3", 3),
        init_i64("i4", 4),
        init_i64("axis_c", 1),
        init_f32("f1", 1.0),
        init_bool_array("zero_full", (1, 1, 30, 30)),
    ]
    nodes: list[onnx.NodeProto] = []
    add_common_prefix(nodes)
    nodes.extend(
        [
            node(
                "Pad",
                ["red_f"],
                ["red_right_f"],
                pads=[0, 0, 0, 2, 0, 0, 0, -2],
                mode="constant",
                value=0.0,
            ),
            node(
                "Pad",
                ["red_f"],
                ["red_down_f"],
                pads=[0, 0, 2, 0, 0, 0, -2, 0],
                mode="constant",
                value=0.0,
            ),
            node("Cast", ["red_right_f"], ["red_right"], to=TensorProto.BOOL),
            node("Cast", ["red_down_f"], ["red_down"], to=TensorProto.BOOL),
            node("And", ["is_h", "red_right"], ["red_right_if_h"]),
            node("Not", ["is_h"], ["is_v"]),
            node("And", ["is_v", "red_down"], ["red_down_if_v"]),
            node("Or", ["red_right_if_h", "red_down_if_v"], ["red_out"]),
        ]
    )
    add_common_suffix(nodes, "red_out")
    return finish_model(nodes, initializers, "task122_pad_float")


def grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    arr = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            arr[0, int(color), r, c] = 1.0
    return arr


def onehot_to_grid(arr: np.ndarray, h: int, w: int) -> list[list[int]]:
    cropped = arr[0, :, :h, :w]
    return cropped.argmax(axis=0).astype(int).tolist()


def expected_grid(grid: list[list[int]]) -> list[list[int]]:
    h, w = len(grid), len(grid[0])
    out = [[0 if cell == 2 else cell for cell in row] for row in grid]
    greens = [(r, c) for r, row in enumerate(grid) for c, cell in enumerate(row) if cell == 3]
    horizontal = any(sum(1 for rr, _ in greens if rr == r) > 1 for r in range(h))
    dr, dc = (0, 2) if horizontal else (2, 0)
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == 2 and r + dr < h and c + dc < w:
                out[r + dr][c + dc] = 2
    return out


def synthetic_examples() -> list[list[list[int]]]:
    h = [[0] * 13 for _ in range(7)]
    for c in range(1, 13, 2):
        h[3][c] = 3
    for r in range(2, 5):
        for c in range(0, 3):
            h[r][c] = 2
    h[3][1] = 3

    v = [[0] * 7 for _ in range(13)]
    for r in range(0, 13, 2):
        v[r][4] = 3
    for r in range(3, 6):
        for c in range(3, 6):
            v[r][c] = 2
    v[4][4] = 3

    edge_h = [[0] * 7 for _ in range(7)]
    for c in range(0, 7, 2):
        edge_h[2][c] = 3
    for r in range(1, 4):
        for c in range(5, 7):
            edge_h[r][c] = 2
    edge_h[2][6] = 3

    edge_v = [[0] * 7 for _ in range(7)]
    for r in range(0, 7, 2):
        edge_v[r][2] = 3
    for r in range(5, 7):
        for c in range(1, 4):
            edge_v[r][c] = 2
    edge_v[6][2] = 3
    return [h, v, edge_h, edge_v]


def run_model(model: onnx.ModelProto, arr: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(["output"], {"input": arr})[0]


def assert_examples(model: onnx.ModelProto) -> None:
    for idx, grid in enumerate(synthetic_examples()):
        got = onehot_to_grid(run_model(model, grid_to_onehot(grid)), len(grid), len(grid[0]))
        exp = expected_grid(grid)
        if got != exp:
            raise AssertionError(f"synthetic example {idx} failed\nexpected={exp}\ngot={got}")

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data[split]):
            arr = convert_to_numpy(example, "input")
            if arr is None:
                continue
            got = onehot_to_grid(run_model(model, arr), len(example["output"]), len(example["output"][0]))
            if got != example["output"]:
                raise AssertionError(f"{split} example {idx} failed")


def inferred_tensor_memory(model: onnx.ModelProto) -> tuple[int, list[tuple[str, str, list[int], int]]]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    rows: list[tuple[str, str, list[int], int]] = []
    total = 0
    for value in inferred.graph.value_info:
        t = value.type.tensor_type
        shape = [dim.dim_value for dim in t.shape.dim]
        dtype = helper.tensor_dtype_to_np_dtype(t.elem_type)
        size = int(math.prod(shape) * np.dtype(dtype).itemsize)
        rows.append((value.name, np.dtype(dtype).name, shape, size))
        total += size
    return total, rows


def build_variants() -> list[Variant]:
    TMP_DIR.mkdir(exist_ok=True)
    variants = [
        Variant("slice_concat", build_slice_concat_variant(), TMP_DIR / "task122_slice_concat.onnx"),
        Variant("pad_float", build_pad_float_variant(), TMP_DIR / "task122_pad_float.onnx"),
    ]
    for variant in variants:
        onnx.save(variant.model, variant.path)
    return variants


def main() -> None:
    variants = build_variants()
    scored: list[tuple[dict[str, object], Variant]] = []
    for variant in variants:
        assert_examples(variant.model)
        inferred_mem, rows = inferred_tensor_memory(variant.model)
        result = score_file(variant.path)
        scored.append((result, variant))
        print(f"\nVariant {variant.name}")
        print(f"initializer params: {calculate_params(variant.model)}")
        print(f"inferred internal tensor bytes: {inferred_mem}")
        print("largest inferred tensors:")
        for name, dtype, shape, size in sorted(rows, key=lambda item: item[3], reverse=True)[:8]:
            print(f"  {name:<18} {dtype:<7} {shape} {size}")
        print_report(result)

    valid = [(result, variant) for result, variant in scored if result.get("valid")]
    if not valid:
        raise RuntimeError("no valid task122 variants")
    best_result, best_variant = min(valid, key=lambda item: int(item[0]["cost"]))
    shutil.copyfile(best_variant.path, BEST_PATH)
    shutil.copyfile(best_variant.path, ROOT_BEST_PATH)
    print(f"\nBest variant: {best_variant.name}")
    print(f"saved: {BEST_PATH}")
    print(f"saved: {ROOT_BEST_PATH}")
    print_report(score_file(BEST_PATH))
    print(f"best cost: {best_result['cost']}")


if __name__ == "__main__":
    main()
