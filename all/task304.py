"""Compact ONNX for ARC task304: copy a 3x3 pattern by modal color cells.

Task rule: read the top-left 3x3 input pattern. Find the most frequent color in
those nine cells; in the official task data the pattern contains no black cells
and the dominant color is unique. The 9x9 output is divided into nine 3x3
blocks; a block gets a full copy of the original 3x3 pattern exactly where the
corresponding input cell has that dominant color, and otherwise that block is
black color 0. Everything outside the top-left 9x9 active area is padding.
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

from score_model import convert_to_numpy, sanitize_model, score_file  # noqa: E402

TASK_ID = "task304"
TASK_NUM = 304
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task304.onnx"

C = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    core_st = _i64(inits, [0, 0, 0, 0], "core_st")
    core_en = _i64(inits, [1, C, 3, 3], "core_en")
    core_repeats = _i64(inits, [1, 1, 3, 3], "core_repeats")
    block_rows = _i64(inits, [0, 0, 0, 1, 1, 1, 2, 2, 2], "block_rows")
    block_cols = _i64(inits, [0, 0, 0, 1, 1, 1, 2, 2, 2], "block_cols")
    black = np.zeros((1, C, 1, 1), dtype=np.uint8)
    black[:, 0, :, :] = True
    _init(inits, black, "black_tile")

    nodes.append(helper.make_node("Slice", [IN_NAME, core_st, core_en, axes4], ["core"]))

    nodes.append(helper.make_node("ReduceSum", ["core"], ["counts_all"], axes=[2, 3], keepdims=1))
    nodes.append(helper.make_node("ArgMax", ["counts_all"], ["dominant_color"], axis=1, keepdims=1))
    nodes.append(helper.make_node("ArgMax", ["core"], ["color_grid"], axis=1, keepdims=1))
    nodes.append(helper.make_node("Equal", ["color_grid", "dominant_color"], ["modal_cells"]))
    nodes.append(helper.make_node("Cast", ["core"], ["core_u"], to=TensorProto.UINT8))
    nodes.append(helper.make_node("Tile", ["core_u", core_repeats], ["core_9"]))
    nodes.append(helper.make_node("Gather", ["modal_cells", block_rows], ["mask_r"], axis=2))
    nodes.append(helper.make_node("Gather", ["mask_r", block_cols], ["mask_9"], axis=3))
    nodes.append(helper.make_node("Where", ["mask_9", "core_9", "black_tile"], ["out9_b"]))
    nodes.append(helper.make_node("Cast", ["out9_b"], ["out9"], to=TensorProto.FLOAT))
    nodes.append(
        helper.make_node(
            "Pad",
            ["out9"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - 9, W - 9],
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


def _load_task() -> dict[str, list[dict[str, Any]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _expected_rule_tensor(grid: list[list[int]]) -> np.ndarray:
    counts: dict[int, int] = {}
    for row in grid:
        for color in row:
            if color:
                counts[color] = counts.get(color, 0) + 1
    dominant = min(color for color, count in counts.items() if count == max(counts.values())) if counts else 1
    out = [[0 for _ in range(9)] for _ in range(9)]
    for block_r in range(3):
        for block_c in range(3):
            if grid[block_r][block_c] != dominant:
                continue
            for r in range(3):
                for c in range(3):
                    out[block_r * 3 + r][block_c * 3 + c] = grid[r][c]
    return convert_to_numpy({"input": grid, "output": out}, "output")


def _validate_examples(model: onnx.ModelProto) -> dict[str, tuple[int, int]]:
    sanitized = sanitize_model(model)
    if sanitized is None:
        raise RuntimeError("model failed NeuroGolf name sanitization")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(sanitized.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])

    data = _load_task()
    results: dict[str, tuple[int, int]] = {}
    for split in ("train", "test", "arc-gen"):
        passed = 0
        examples = data.get(split, [])
        for index, example in enumerate(examples):
            inp = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            actual = session.run([OUT_NAME], {IN_NAME: inp})[0]
            if not np.array_equal(actual > 0.0, expected > 0.0):
                raise AssertionError(f"{split} example {index} failed")
            passed += 1
        results[split] = (passed, len(examples))

    synthetic_grid = [[1, 2, 1], [3, 1, 4], [5, 6, 7]]
    synthetic_input = convert_to_numpy({"input": synthetic_grid, "output": synthetic_grid}, "input")
    synthetic_expected = _expected_rule_tensor(synthetic_grid)
    synthetic_actual = session.run([OUT_NAME], {IN_NAME: synthetic_input})[0]
    if not np.array_equal(synthetic_actual > 0.0, synthetic_expected > 0.0):
        raise AssertionError("synthetic no-black dominant-color example failed")
    return results


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    checks = _validate_examples(model)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    for split, (passed, total) in checks.items():
        print(f"{split}:   {passed}/{total} correct")
    print("synthetic: no-black dominant-color example correct")
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
