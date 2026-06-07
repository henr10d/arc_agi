"""Minimal ONNX for ARC task223 using 3x nearest-neighbor pixel expansion.

Task rule: the input is a 3x3 grid.  Each input cell expands to a solid 3x3
block of the same color, producing a 9x9 output.  Black cells expand to black
blocks.  The competition tensor is padded to 30x30, with padded cells left
all-zero outside the 9x9 task output.

ONNX: compare a few static 3x enlargement graphs and keep the lowest-cost
valid model.  The best variant uses nearest-neighbor Resize on the cropped
3x3 core, avoiding the grouped ConvTranspose kernel parameters while realizing
only the 3x3 crop and 9x9 expanded core before the final unscored output pad.
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
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task223"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task223.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
N = 3
OUT = N * N
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver: repeat every input pixel into a 3x3 output block."""
    core = np.asarray(grid, dtype=np.int64)[:N, :N]
    return np.repeat(np.repeat(core, N, axis=0), N, axis=1)


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _make_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _slice_core(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto]) -> str:
    starts = _i64(inits, [0, 0], "starts")
    ends = _i64(inits, [N, N], "ends")
    axes = _i64(inits, [2, 3], "axes")
    nodes.append(helper.make_node("Slice", [IN_NAME, starts, ends, axes], ["core"]))
    return "core"


def build_resize_model() -> onnx.ModelProto:
    """Lowest-param model: crop the 3x3 core and nearest-resize it to 9x9."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    core = _slice_core(nodes, inits)
    scales = _f32(inits, [1.0, 1.0, float(N), float(N)], "scales")
    nodes.extend(
        [
            helper.make_node("Resize", [core, scales], ["out9"], mode="nearest"),
            helper.make_node("Pad", ["out9"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - OUT, W - OUT]),
        ]
    )
    return _make_model(nodes, inits, "task223_resize")


def build_conv_transpose_model() -> onnx.ModelProto:
    """Best model: grouped transposed convolution expands each channel blockwise."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    core = _slice_core(nodes, inits)
    weights = np.ones((C, 1, N, N), dtype=np.float32)
    weight_name = _f32(inits, weights, "w")
    nodes.extend(
        [
            helper.make_node(
                "ConvTranspose",
                [core, weight_name],
                ["out9"],
                group=C,
                strides=[N, N],
            ),
            helper.make_node("Pad", ["out9"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - OUT, W - OUT]),
        ]
    )
    return _make_model(nodes, inits, "task223_conv_transpose")


def build_gather_model() -> onnx.ModelProto:
    """Direct indexing variant: gather repeated row and column indices."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    core = _slice_core(nodes, inits)
    idx = _i64(inits, [0, 0, 0, 1, 1, 1, 2, 2, 2], "idx")
    nodes.extend(
        [
            helper.make_node("Gather", [core, idx], ["rows"], axis=2),
            helper.make_node("Gather", ["rows", idx], ["out9"], axis=3),
            helper.make_node("Pad", ["out9"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - OUT, W - OUT]),
        ]
    )
    return _make_model(nodes, inits, "task223_gather")


def build_tile_model() -> onnx.ModelProto:
    """Tile variant: reshape to 6D, tile singleton block axes, reshape to 9x9."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    core = _slice_core(nodes, inits)
    r6 = _i64(inits, [1, C, N, 1, N, 1], "r6")
    reps = _i64(inits, [1, 1, 1, N, 1, N], "reps")
    r4 = _i64(inits, [1, C, OUT, OUT], "r4")
    nodes.extend(
        [
            helper.make_node("Reshape", [core, r6], ["six"]),
            helper.make_node("Tile", ["six", reps], ["tiled"]),
            helper.make_node("Reshape", ["tiled", r4], ["out9"]),
            helper.make_node("Pad", ["out9"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - OUT, W - OUT]),
        ]
    )
    return _make_model(nodes, inits, "task223_tile")


def build_expand_model() -> onnx.ModelProto:
    """Expand+reshape variant: broadcast singleton block axes to 3x3 blocks."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    core = _slice_core(nodes, inits)
    r6 = _i64(inits, [1, C, N, 1, N, 1], "r6")
    shape6 = _i64(inits, [1, C, N, N, N, N], "shape6")
    r4 = _i64(inits, [1, C, OUT, OUT], "r4")
    nodes.extend(
        [
            helper.make_node("Reshape", [core, r6], ["six"]),
            helper.make_node("Expand", ["six", shape6], ["expanded"]),
            helper.make_node("Reshape", ["expanded", r4], ["out9"]),
            helper.make_node("Pad", ["out9"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - OUT, W - OUT]),
        ]
    )
    return _make_model(nodes, inits, "task223_expand")


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _expected_onehot(grid: list[list[int]]) -> np.ndarray:
    return _grid_to_onehot(grid)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> dict[str, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    counts = {"train": 0, "test": 0, "arc-gen": 0}
    for split in ("train", "test", "arc-gen"):
        for ex in data.get(split, []):
            expected_grid = np.asarray(ex["output"], dtype=np.int64)
            solved = solve(np.asarray(ex["input"], dtype=np.int64))
            if not np.array_equal(solved, expected_grid):
                raise AssertionError(f"reference mismatch in {split} example {counts[split]}")

            pred = _run_onnx(model, _grid_to_onehot(ex["input"]))
            expected = _expected_onehot(ex["output"])
            if not np.array_equal(pred > 0.0, expected > 0.0):
                raise AssertionError(f"ONNX mismatch in {split} example {counts[split]}")
            counts[split] += 1
    return counts


def _score_variant(label: str, model: onnx.ModelProto) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}_{label}_") as tmp:
        path = Path(tmp) / f"{TASK_ID}.onnx"
        onnx.save(model, path)
        return score_file(path)


def _tensor_count(model: onnx.ModelProto) -> int:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    graph = inferred.graph
    names = {value.name for value in graph.value_info}
    names.update(node.output[0] for node in graph.node if node.output and node.output[0] != OUT_NAME)
    return len(names)


def main() -> None:
    variants = [
        ("resize", build_resize_model()),
        ("conv_transpose", build_conv_transpose_model()),
        ("gather", build_gather_model()),
        ("tile", build_tile_model()),
        ("expand_reshape", build_expand_model()),
    ]

    results: list[tuple[int, str, onnx.ModelProto, dict[str, Any], dict[str, int]]] = []
    for label, model in variants:
        counts = validate_json(model)
        scored = _score_variant(label, model)
        print(
            f"{label:<16} valid={scored['valid']} nodes={len(model.graph.node)} "
            f"tensors={_tensor_count(model)} memory={scored['memory']} params={scored['params']} "
            f"cost={scored['cost']} score={scored['score']:.6f} passes={counts}"
        )
        if scored["valid"]:
            results.append((int(scored["cost"]), label, model, scored, counts))

    if not results:
        raise SystemExit("no valid task223 variants")

    _cost, label, model, scored, counts = min(results, key=lambda item: item[0])
    onnx.save(model, BEST_PATH)
    print()
    print(f"kept:    {label}")
    print(f"wrote:   {BEST_PATH}")
    print(f"passes:  {counts}")
    print(f"nodes:   {len(model.graph.node)}")
    print(f"tensors: {_tensor_count(model)}")
    print(f"input:   {SHAPE}")
    print(f"output:  {SHAPE}")
    print(f"memory:  {scored['memory']}")
    print(f"params:  {scored['params']}")
    print(f"cost:    {scored['cost']}")
    print(f"score:   {scored['score']:.6f}")


if __name__ == "__main__":
    main()
