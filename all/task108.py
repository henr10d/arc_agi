"""ONNX generators for NeuroGolf task108.

Task rule: the 10x10 input uses colored cells only on odd row/column
positions. Read those odd positions as a compact 5x5 grid, expand each
logical cell to a solid 4x4 block of the same color, then pad the 20x20
result to the competition output tensor's 30x30 canvas.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import score_model
TASK_ID = "task108"
TASK_NUM = 108
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = ROOT / "task108_best.onnx"
CANONICAL_PATH = OUT_DIR / f"{TASK_ID}.onnx"
VARIANT_DIR = ROOT / "onnx" / TASK_ID

INPUT_SHAPE = [1, 10, 30, 30]
OUTPUT_SHAPE = [1, 10, 30, 30]
ODD_INDICES = np.array([1, 3, 5, 7, 9], dtype=np.int64)


def tensor_value_info(name: str, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def init(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array, name)


def make_model(
    name: str,
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    value_info: list[onnx.ValueInfoProto],
) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        name,
        [tensor_value_info("input", INPUT_SHAPE)],
        [tensor_value_info("output", OUTPUT_SHAPE)],
        initializer=initializers,
        value_info=value_info,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 10)],
        producer_name="neurogolf_task108",
    )
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


def common_gather_prefix() -> tuple[list[onnx.NodeProto], list[onnx.TensorProto], list[onnx.ValueInfoProto]]:
    nodes = [
        helper.make_node("Gather", ["input", "row_idx"], ["rows"], axis=2),
        helper.make_node("Gather", ["rows", "col_idx"], ["core"], axis=3),
    ]
    initializers = [
        init("row_idx", ODD_INDICES),
        init("col_idx", ODD_INDICES),
    ]
    value_info = [
        tensor_value_info("rows", [1, 10, 5, 30]),
        tensor_value_info("core", [1, 10, 5, 5]),
    ]
    return nodes, initializers, value_info


def build_resize() -> onnx.ModelProto:
    nodes, initializers, value_info = common_gather_prefix()
    initializers.append(init("scales", np.array([1.0, 1.0, 4.0, 4.0], dtype=np.float32)))
    nodes.extend(
        [
            helper.make_node("Resize", ["core", "scales"], ["expanded"], mode="nearest"),
            helper.make_node(
                "Pad",
                ["expanded"],
                ["output"],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, 10, 10],
                value=0.0,
            ),
        ]
    )
    value_info.append(tensor_value_info("expanded", [1, 10, 20, 20]))
    return make_model("task108_resize", nodes, initializers, value_info)


def build_expand_reshape() -> onnx.ModelProto:
    nodes, initializers, value_info = common_gather_prefix()
    initializers.extend(
        [
            init("expand_shape", np.array([1, 10, 5, 4, 5, 4], dtype=np.int64)),
            init("flat_shape", np.array([1, 10, 20, 20], dtype=np.int64)),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Unsqueeze", ["core"], ["unsqueezed"], axes=[3, 5]),
            helper.make_node("Expand", ["unsqueezed", "expand_shape"], ["expanded6"]),
            helper.make_node("Reshape", ["expanded6", "flat_shape"], ["expanded"]),
            helper.make_node(
                "Pad",
                ["expanded"],
                ["output"],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, 10, 10],
                value=0.0,
            ),
        ]
    )
    value_info.extend(
        [
            tensor_value_info("unsqueezed", [1, 10, 5, 1, 5, 1]),
            tensor_value_info("expanded6", [1, 10, 5, 4, 5, 4]),
            tensor_value_info("expanded", [1, 10, 20, 20]),
        ]
    )
    return make_model("task108_expand_reshape", nodes, initializers, value_info)


def build_tile() -> onnx.ModelProto:
    nodes, initializers, value_info = common_gather_prefix()
    initializers.extend(
        [
            init("repeats_h", np.array([1, 1, 1, 4, 1], dtype=np.int64)),
            init("shape_h", np.array([1, 10, 20, 5], dtype=np.int64)),
            init("repeats_w", np.array([1, 1, 1, 1, 4], dtype=np.int64)),
            init("shape_hw", np.array([1, 10, 20, 20], dtype=np.int64)),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Unsqueeze", ["core"], ["with_h_axis"], axes=[3]),
            helper.make_node("Tile", ["with_h_axis", "repeats_h"], ["tiled_h5"]),
            helper.make_node("Reshape", ["tiled_h5", "shape_h"], ["tiled_h"]),
            helper.make_node("Unsqueeze", ["tiled_h"], ["with_w_axis"], axes=[4]),
            helper.make_node("Tile", ["with_w_axis", "repeats_w"], ["tiled_hw5"]),
            helper.make_node("Reshape", ["tiled_hw5", "shape_hw"], ["expanded"]),
            helper.make_node(
                "Pad",
                ["expanded"],
                ["output"],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, 10, 10],
                value=0.0,
            ),
        ]
    )
    value_info.extend(
        [
            tensor_value_info("with_h_axis", [1, 10, 5, 1, 5]),
            tensor_value_info("tiled_h5", [1, 10, 5, 4, 5]),
            tensor_value_info("tiled_h", [1, 10, 20, 5]),
            tensor_value_info("with_w_axis", [1, 10, 20, 5, 1]),
            tensor_value_info("tiled_hw5", [1, 10, 20, 5, 4]),
            tensor_value_info("expanded", [1, 10, 20, 20]),
        ]
    )
    return make_model("task108_tile", nodes, initializers, value_info)


def slice_prefix() -> tuple[list[onnx.NodeProto], list[onnx.TensorProto], list[onnx.ValueInfoProto]]:
    nodes = [helper.make_node("Slice", ["input", "starts", "ends", "axes", "steps"], ["core"])]
    initializers = [
        init("starts", np.array([1, 1], dtype=np.int64)),
        init("ends", np.array([10, 10], dtype=np.int64)),
        init("axes", np.array([2, 3], dtype=np.int64)),
        init("steps", np.array([2, 2], dtype=np.int64)),
    ]
    value_info = [tensor_value_info("core", [1, 10, 5, 5])]
    return nodes, initializers, value_info


def build_slice_resize() -> onnx.ModelProto:
    nodes, initializers, value_info = slice_prefix()
    initializers.append(init("scales", np.array([1.0, 1.0, 4.0, 4.0], dtype=np.float32)))
    nodes.extend(
        [
            helper.make_node("Resize", ["core", "scales"], ["expanded"], mode="nearest"),
            helper.make_node(
                "Pad",
                ["expanded"],
                ["output"],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, 10, 10],
                value=0.0,
            ),
        ]
    )
    value_info.append(tensor_value_info("expanded", [1, 10, 20, 20]))
    return make_model("task108_slice_resize", nodes, initializers, value_info)


def build_slice_convtranspose() -> onnx.ModelProto:
    one_kernel = np.ones((10, 1, 4, 4), dtype=np.float32)
    nodes, initializers, value_info = slice_prefix()
    initializers.append(init("kernel", one_kernel))
    nodes.extend(
        [
            helper.make_node(
                "ConvTranspose",
                ["core", "kernel"],
                ["expanded"],
                group=10,
                strides=[4, 4],
            ),
            helper.make_node(
                "Pad",
                ["expanded"],
                ["output"],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, 10, 10],
                value=0.0,
            ),
        ]
    )
    value_info.append(tensor_value_info("expanded", [1, 10, 20, 20]))
    return make_model("task108_slice_convtranspose", nodes, initializers, value_info)


def one_hot(grid: list[list[int]], canvas: int = 30) -> np.ndarray:
    arr = np.zeros((1, 10, canvas, canvas), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            arr[0, int(color), r, c] = 1.0
    return arr


def load_examples() -> list[tuple[np.ndarray, np.ndarray]]:
    with (ROOT / "data" / f"{TASK_ID}.json").open(encoding="utf-8") as fh:
        data = json.load(fh)
    examples: list[tuple[np.ndarray, np.ndarray]] = []
    for split in ("train", "test", "arc-gen"):
        for example in data.get(split, []):
            examples.append((one_hot(example["input"]), one_hot(example["output"])))
    return examples


def verify_outputs(path: Path) -> bool:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for inp, expected in load_examples():
        got = session.run(["output"], {"input": inp})[0]
        if not np.array_equal(got > 0.0, expected > 0.0):
            return False
    return True


def main() -> None:
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    variants = {
        "resize": build_resize(),
        "expand_reshape": build_expand_reshape(),
        "tile": build_tile(),
        "slice_resize": build_slice_resize(),
        "slice_convtranspose": build_slice_convtranspose(),
    }

    results = []
    for name, model in variants.items():
        path = VARIANT_DIR / f"{TASK_ID}_{name}.onnx"
        onnx.save(model, path)
        result = score_model.score_file(path)
        result["correct"] = verify_outputs(path) if result["valid"] else False
        results.append(result)

        print(name)
        print(f"  correct: {result['correct']}")
        print(f"  memory:  {result['memory']}")
        print(f"  params:  {result['params']}")
        print(f"  cost:    {result['cost']}")
        print(f"  score:   {result['score']}")
        if result["error"]:
            print(f"  error:   {str(result['error']).strip()}")

    correct_results = [r for r in results if r["valid"] and r["correct"]]
    if not correct_results:
        raise SystemExit("no valid correct variant")

    best = min(correct_results, key=lambda r: int(r["cost"]))
    best_path = best["path"]
    assert isinstance(best_path, Path)
    shutil.copyfile(best_path, BEST_PATH)
    shutil.copyfile(best_path, CANONICAL_PATH)
    print(f"best: {best_path.name} -> {BEST_PATH.name}")


if __name__ == "__main__":
    main()
