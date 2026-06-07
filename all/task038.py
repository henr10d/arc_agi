"""Build and validate compact ONNX candidates for NeuroGolf task038.

Task rule: in the active 9x9 grid, count solid blue (color 1) 2x2 blocks.
The output is a 1x5 unary row: the first N cells are blue and the rest are
black, where N is the number of blue 2x2 blocks capped at five. Red blocks and
isolated single cells are distractors.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_PATH = REPO_ROOT / "data" / "task038.json"
OUT_PATH = SCRIPT_DIR / "task038.onnx"

C = 10
H = W = 30
ACTIVE = 9
OUT_W = 5
SHAPE = [1, C, H, W]
IR_VERSION = 10


@dataclass(frozen=True)
class Candidate:
    name: str
    build: Callable[[], onnx.ModelProto]


def solve_grid(grid: list[list[int]]) -> list[list[int]]:
    count = 0
    for row in range(ACTIVE - 1):
        for col in range(ACTIVE - 1):
            if (
                grid[row][col] == 1
                and grid[row][col + 1] == 1
                and grid[row + 1][col] == 1
                and grid[row + 1][col + 1] == 1
            ):
                count += 1
    return [[1 if col < min(count, OUT_W) else 0 for col in range(OUT_W)]]


def grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            out[0, int(color), row, col] = 1.0
    return out


def onehot_to_grid(tensor: np.ndarray, rows: int = 1, cols: int = OUT_W) -> list[list[int]]:
    return tensor[0, :, :rows, :cols].argmax(axis=0).astype(int).tolist()


def expected_onehot(grid: list[list[int]]) -> np.ndarray:
    return grid_to_onehot(grid)


def selected_blocks(grid: list[list[int]]) -> list[tuple[int, int, int]]:
    out = []
    for row in range(ACTIVE - 1):
        for col in range(ACTIVE - 1):
            if (
                grid[row][col] == 1
                and grid[row][col + 1] == 1
                and grid[row + 1][col] == 1
                and grid[row + 1][col + 1] == 1
            ):
                out.append((row, col, 1))
    return out


def _i64(values: list[int], name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)


def _f32(values: list[float], name: str, shape: tuple[int, ...] | None = None) -> onnx.TensorProto:
    arr = np.asarray(values, dtype=np.float32)
    if shape is not None:
        arr = arr.reshape(shape)
    return numpy_helper.from_array(arr, name=name)


def _f16(values: list[float], name: str, shape: tuple[int, ...] | None = None) -> onnx.TensorProto:
    arr = np.asarray(values, dtype=np.float16)
    if shape is not None:
        arr = arr.reshape(shape)
    return numpy_helper.from_array(arr, name=name)


def _u8(values: list[int], name: str, shape: tuple[int, ...] | None = None) -> onnx.TensorProto:
    arr = np.asarray(values, dtype=np.uint8)
    if shape is not None:
        arr = arr.reshape(shape)
    return numpy_helper.from_array(arr, name=name)


def _slice_blue9(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], opset: int) -> str:
    if opset >= 10:
        inits.extend(
            [
                _i64([0, 1, 0, 0], "slice_starts"),
                _i64([1, 2, ACTIVE, ACTIVE], "slice_ends"),
                _i64([0, 1, 2, 3], "slice_axes"),
            ]
        )
        nodes.append(helper.make_node("Slice", ["input", "slice_starts", "slice_ends", "slice_axes"], ["blue9"]))
    else:
        nodes.append(
            helper.make_node(
                "Slice",
                ["input"],
                ["blue9"],
                starts=[0, 1, 0, 0],
                ends=[1, 2, ACTIVE, ACTIVE],
                axes=[0, 1, 2, 3],
            )
        )
    return "blue9"


def _finish_count_graph(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], count_dtype: int) -> None:
    if count_dtype == TensorProto.FLOAT16:
        inits.append(_f16([0.0, 1.0, 2.0, 3.0, 4.0], "thresholds", (1, 1, 1, OUT_W)))
    else:
        inits.append(_f32([0.0, 1.0, 2.0, 3.0, 4.0], "thresholds", (1, 1, 1, OUT_W)))
    nodes.extend(
        [
            helper.make_node("Greater", ["count", "thresholds"], ["blue_row_b"]),
            helper.make_node("Not", ["blue_row_b"], ["black_row_b"]),
            helper.make_node("Concat", ["black_row_b", "blue_row_b"], ["row2_b"], axis=1),
            helper.make_node("Cast", ["row2_b"], ["row2"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["row2"],
                ["output"],
                pads=[0, 0, 0, 0, 0, C - 2, H - 1, W - OUT_W],
                value=0.0,
            ),
        ]
    )


def _make_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], name: str, opset: int) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf-task038",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_average_pool_model(opset: int = 9) -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    blue9 = _slice_blue9(nodes, inits, opset)
    inits.append(_f32([0.99], "full_block_threshold"))
    nodes.extend(
        [
            helper.make_node("AveragePool", [blue9], ["pool"], kernel_shape=[2, 2], strides=[1, 1]),
            helper.make_node("Greater", ["pool", "full_block_threshold"], ["is_block"]),
            helper.make_node("Cast", ["is_block"], ["block_f"], to=TensorProto.FLOAT),
            helper.make_node("ReduceSum", ["block_f"], ["count"], axes=[2, 3], keepdims=1),
        ]
    )
    _finish_count_graph(nodes, inits, TensorProto.FLOAT)
    return _make_model(nodes, inits, f"task038_average_pool_opset{opset}", opset)


def build_average_pool_f16_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    blue9 = _slice_blue9(nodes, inits, 9)
    inits.append(_f32([0.99], "full_block_threshold"))
    nodes.extend(
        [
            helper.make_node("AveragePool", [blue9], ["pool"], kernel_shape=[2, 2], strides=[1, 1]),
            helper.make_node("Greater", ["pool", "full_block_threshold"], ["is_block"]),
            helper.make_node("Cast", ["is_block"], ["block_f16"], to=TensorProto.FLOAT16),
            helper.make_node("ReduceSum", ["block_f16"], ["count"], axes=[2, 3], keepdims=1),
        ]
    )
    _finish_count_graph(nodes, inits, TensorProto.FLOAT16)
    return _make_model(nodes, inits, "task038_average_pool_f16_opset9", 9)


def build_conv_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    blue9 = _slice_blue9(nodes, inits, 10)
    inits.extend(
        [
            _f32([1.0, 1.0, 1.0, 1.0], "kernel", (1, 1, 2, 2)),
            _f32([3.5], "conv_threshold"),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Conv", [blue9, "kernel"], ["hits"], strides=[1, 1]),
            helper.make_node("Greater", ["hits", "conv_threshold"], ["is_block"]),
            helper.make_node("Cast", ["is_block"], ["block_f"], to=TensorProto.FLOAT),
            helper.make_node("ReduceSum", ["block_f"], ["count"], axes=[2, 3], keepdims=1),
        ]
    )
    _finish_count_graph(nodes, inits, TensorProto.FLOAT)
    return _make_model(nodes, inits, "task038_conv_opset10", 10)


def build_neighbor_and_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    blue9 = _slice_blue9(nodes, inits, 10)
    inits.extend(
        [
            _i64([0, 0, 0, 0], "tl_starts"),
            _i64([1, 1, 8, 8], "tl_ends"),
            _i64([0, 0, 0, 1], "tr_starts"),
            _i64([1, 1, 8, 9], "tr_ends"),
            _i64([0, 0, 1, 0], "bl_starts"),
            _i64([1, 1, 9, 8], "bl_ends"),
            _i64([0, 0, 1, 1], "br_starts"),
            _i64([1, 1, 9, 9], "br_ends"),
            _i64([0, 1, 2, 3], "axes4"),
            _f32([0.5], "half"),
        ]
    )
    for name in ("tl", "tr", "bl", "br"):
        nodes.extend(
            [
                helper.make_node("Slice", [blue9, f"{name}_starts", f"{name}_ends", "axes4"], [name]),
                helper.make_node("Greater", [name, "half"], [f"{name}_b"]),
            ]
        )
    nodes.extend(
        [
            helper.make_node("And", ["tl_b", "tr_b"], ["top_b"]),
            helper.make_node("And", ["bl_b", "br_b"], ["bot_b"]),
            helper.make_node("And", ["top_b", "bot_b"], ["is_block"]),
            helper.make_node("Cast", ["is_block"], ["block_f"], to=TensorProto.FLOAT),
            helper.make_node("ReduceSum", ["block_f"], ["count"], axes=[2, 3], keepdims=1),
        ]
    )
    _finish_count_graph(nodes, inits, TensorProto.FLOAT)
    return _make_model(nodes, inits, "task038_neighbor_and_opset10", 10)


def validate(path: Path, examples: dict[str, list[dict[str, Any]]], verbose: bool) -> None:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])

    failures = 0
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(examples.get(split, [])):
            inp = grid_to_onehot(example["input"])
            got = session.run(["output"], {"input": inp})[0]
            exp = expected_onehot(example["output"])
            ok = np.array_equal(got > 0.0, exp > 0.0)
            failures += not ok

            if verbose and (split != "arc-gen" or idx < 8):
                print(
                    f"{split:7s} {idx:3d}: selected={selected_blocks(example['input'])} "
                    f"output={onehot_to_grid(got)} expected={example['output']} ok={ok}"
                )

    if failures:
        raise RuntimeError(f"{path.name} validation failed on {failures} examples")


def score_candidate(path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT))
    from score_model import score_file

    with tempfile.TemporaryDirectory(prefix="task038_score_") as tmp:
        tmp_path = Path(tmp) / "task038.onnx"
        shutil.copy2(path, tmp_path)
        return score_file(tmp_path)


def main() -> None:
    examples = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    for split, split_examples in examples.items():
        for idx, example in enumerate(split_examples):
            solved = solve_grid(example["input"])
            if solved != example["output"]:
                raise SystemExit(f"rule mismatch in {split} example {idx}: {solved} != {example['output']}")

    candidates = [
        Candidate("average_pool_f16_opset9", build_average_pool_f16_model),
        Candidate("average_pool_opset9", lambda: build_average_pool_model(9)),
        Candidate("average_pool_opset10", lambda: build_average_pool_model(10)),
        Candidate("conv_opset10", build_conv_model),
        Candidate("neighbor_and_opset10", build_neighbor_and_model),
    ]

    results: list[tuple[int, Candidate, Path, dict[str, Any]]] = []
    for candidate in candidates:
        path = SCRIPT_DIR / f"task038_{candidate.name}.onnx"
        model = candidate.build()
        onnx.save(model, path)
        validate(path, examples, verbose=False)
        result = score_candidate(path)
        if not result["valid"]:
            raise SystemExit(f"{candidate.name} scored invalid: {result['error']}")
        cost = int(result["cost"])
        results.append((cost, candidate, path, result))
        print(
            f"{candidate.name:22s} memory={result['memory']} params={result['params']} "
            f"cost={result['cost']} score={result['score']:.6f} file={path.name}"
        )

    results.sort(key=lambda item: (item[0], item[3]["filesize"]))
    _, best_candidate, best_path, _ = results[0]
    if best_path.resolve() != OUT_PATH.resolve():
        shutil.copy2(best_path, OUT_PATH)
    print(f"\nwrote best {best_candidate.name} to {OUT_PATH}")
    validate(OUT_PATH, examples, verbose=True)
    print()

    sys.path.insert(0, str(REPO_ROOT))
    from score_model import print_report, score_file

    print_report(score_file(OUT_PATH))


if __name__ == "__main__":
    main()
