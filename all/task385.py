"""Minimal ONNX for ARC task385 using Kaggle one-hot I/O.

Task rule: every example is a 10x4 grid whose top five rows are black and whose
bottom five rows contain a colored staircase-like pattern. The output preserves
those bottom five rows and fills the top five rows with a vertical reflection of
the bottom half: output rows 0..4 are input rows 9,8,7,6,5, and output rows 5..9
are input rows 5..9. Padding outside the 10x4 task grid remains all-zero.

ONNX approach: the best model is a single Gather along the height axis over the
full [1,10,30,30] input. The row index vector performs the fixed permutation
[9,8,7,6,5,5,6,7,8,9,10..29]. Since Gather writes directly to the graph output,
NeuroGolf counts no internal activation memory; the only scored cost is the
30-element int32 index initializer.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402


TASK_ID = "task385"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task385.onnx"
IN_NAME = "input"
OUT_NAME = "output"
C = 10
H = W = 30
GRID_H = 10
GRID_W = 4
HALF = 5
FULL_SHAPE = [1, C, H, W]
IR_VERSION = 10
OPSET = 10

Grid = list[list[int]]
Hypothesis = Callable[[Grid], np.ndarray]


def _i64(initializers: list[onnx.TensorProto], values: list[int], name: str) -> str:
    initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name))
    return name


def _i32(initializers: list[onnx.TensorProto], values: list[int], name: str) -> str:
    initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.int32), name=name))
    return name


def load_task_data() -> dict[str, list[dict[str, Grid]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def hypothesis_a_by_color_highest(grid: Grid) -> np.ndarray:
    """For each color and occupied column, extend from its highest cell to row 0."""
    out = np.asarray(grid, dtype=np.int64).copy()
    arr = np.asarray(grid, dtype=np.int64)
    for color in range(1, C):
        rows, cols = np.where(arr == color)
        for col in sorted(set(int(c) for c in cols)):
            top = int(rows[cols == col].min())
            out[:top, col] = np.where(out[:top, col] == 0, color, out[:top, col])
    return out


def hypothesis_b_by_column_bottom(grid: Grid) -> np.ndarray:
    """For each occupied column, extend its bottom-most color upward through black."""
    out = np.asarray(grid, dtype=np.int64).copy()
    arr = np.asarray(grid, dtype=np.int64)
    for col in range(arr.shape[1]):
        colored = np.where(arr[:, col] != 0)[0]
        if colored.size:
            bottom = int(colored.max())
            color = int(arr[bottom, col])
            out[:bottom, col] = np.where(out[:bottom, col] == 0, color, out[:bottom, col])
    return out


def hypothesis_c_by_component(grid: Grid) -> np.ndarray:
    """Extend each connected non-black component upward in its occupied columns."""
    arr = np.asarray(grid, dtype=np.int64)
    out = arr.copy()
    seen = np.zeros(arr.shape, dtype=bool)
    height, width = arr.shape
    for start_r in range(height):
        for start_c in range(width):
            color = int(arr[start_r, start_c])
            if color == 0 or seen[start_r, start_c]:
                continue
            cells: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(start_r, start_c)])
            seen[start_r, start_c] = True
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and arr[nr, nc] == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            for col in sorted({c for _, c in cells}):
                top = min(r for r, c in cells if c == col)
                out[:top, col] = np.where(out[:top, col] == 0, color, out[:top, col])
    return out


def solve_grid(grid: Grid) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    return np.concatenate([arr[HALF:GRID_H][::-1], arr[HALF:GRID_H]], axis=0)


def validate_hypotheses_on_train() -> dict[str, int]:
    hypotheses: dict[str, Hypothesis] = {
        "A_color_highest_upward": hypothesis_a_by_color_highest,
        "B_column_bottom_upward": hypothesis_b_by_column_bottom,
        "C_component_upward": hypothesis_c_by_component,
        "selected_bottom_half_reflection": solve_grid,
    }
    data = load_task_data()
    results: dict[str, int] = {}
    for name, func in hypotheses.items():
        matches = 0
        for example in data["train"]:
            expected = np.asarray(example["output"], dtype=np.int64)
            matches += int(np.array_equal(func(example["input"]), expected))
        results[name] = matches
    if results["selected_bottom_half_reflection"] != len(data["train"]):
        raise ValueError(f"no train-consistent hypothesis found: {results}")
    return results


def assert_rule_matches_all_examples() -> dict[str, tuple[int, int]]:
    data = load_task_data()
    split_counts: dict[str, tuple[int, int]] = {}
    for split in ("train", "test", "arc-gen"):
        passed = 0
        total = 0
        for index, example in enumerate(data.get(split, [])):
            inp = np.asarray(example["input"], dtype=np.int64)
            out = np.asarray(example["output"], dtype=np.int64)
            if inp.shape != (GRID_H, GRID_W) or out.shape != (GRID_H, GRID_W):
                raise ValueError(f"{split}[{index}] is not {GRID_H}x{GRID_W}")
            if np.any(inp[:HALF] != 0):
                raise ValueError(f"{split}[{index}] has non-black top half")
            total += 1
            if np.array_equal(solve_grid(example["input"]), out):
                passed += 1
            else:
                raise ValueError(f"{split}[{index}] does not match reflected-bottom-half rule")
        split_counts[split] = (passed, total)
    return split_counts


def make_model(nodes: list[onnx.NodeProto], initializers: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def build_full_gather_model() -> onnx.ModelProto:
    initializers: list[onnx.TensorProto] = []
    row_indices = list(range(GRID_H - 1, HALF - 1, -1)) + list(range(HALF, GRID_H)) + list(range(GRID_H, H))
    idx = _i32(initializers, row_indices, "row_indices")
    nodes = [helper.make_node("Gather", [IN_NAME, idx], [OUT_NAME], axis=2)]
    return make_model(nodes, initializers, "task385_full_gather")


def build_compact_slice_gather_model() -> onnx.ModelProto:
    initializers: list[onnx.TensorProto] = []
    starts = _i64(initializers, [0, 0, HALF, 0], "starts")
    ends = _i64(initializers, [1, C, GRID_H, GRID_W], "ends")
    gather_idx = _i64(initializers, [4, 3, 2, 1, 0, 0, 1, 2, 3, 4], "gather_idx")
    nodes = [
        helper.make_node("Slice", [IN_NAME, starts, ends], ["bottom"]),
        helper.make_node("Gather", ["bottom", gather_idx], ["compact"], axis=2),
        helper.make_node(
            "Pad",
            ["compact"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - GRID_H, W - GRID_W],
            value=0.0,
        ),
    ]
    return make_model(nodes, initializers, "task385_compact_slice_gather")


def validate_model(model: onnx.ModelProto) -> dict[str, tuple[int, int]]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    data = load_task_data()
    split_counts: dict[str, tuple[int, int]] = {}
    for split in ("train", "test", "arc-gen"):
        passed = 0
        total = 0
        for index, example in enumerate(data.get(split, [])):
            inp = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if inp is None or expected is None:
                continue
            pred = session.run([OUT_NAME], {IN_NAME: inp})[0]
            total += 1
            if not np.array_equal(pred > 0.0, expected > 0.0):
                raise AssertionError(f"{split}[{index}] output mismatch")
            passed += 1
        split_counts[split] = (passed, total)
    return split_counts


def score_candidate(model: onnx.ModelProto, name: str) -> dict[str, object]:
    validate_model(model)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / f"{TASK_ID}.onnx"
        onnx.save(model, str(path))
        result = score_file(path)
    if not result["valid"]:
        raise RuntimeError(f"{name} invalid: {result['error']}")
    return result


def select_best_model() -> tuple[str, onnx.ModelProto, dict[str, object], dict[str, dict[str, object]]]:
    candidates = {
        "full_gather": build_full_gather_model(),
        "compact_slice_gather": build_compact_slice_gather_model(),
    }
    scores = {name: score_candidate(model, name) for name, model in candidates.items()}
    # Lower cost means higher score; filesize only breaks exact score ties.
    best_name = min(scores, key=lambda key: (int(scores[key]["cost"]), int(scores[key]["filesize"])))
    return best_name, candidates[best_name], scores[best_name], scores


def main() -> None:
    hypothesis_results = validate_hypotheses_on_train()
    rule_counts = assert_rule_matches_all_examples()
    best_name, model, best_score, all_scores = select_best_model()
    final_counts = validate_model(model)

    OUT_DIR.mkdir(exist_ok=True)
    onnx.save(model, str(BEST_PATH))
    final_score = score_file(BEST_PATH)
    if not final_score["valid"]:
        raise RuntimeError(final_score["error"])

    print(f"train hypothesis matches: {hypothesis_results}")
    print(f"rule verified: {rule_counts}")
    for name, result in all_scores.items():
        print(
            f"candidate {name}: memory={result['memory']} params={result['params']} "
            f"cost={result['cost']} score={result['score']:.6f}"
        )
    print(f"selected: {best_name}")
    print(f"model verified: {final_counts}")
    print(f"wrote: {BEST_PATH}")
    print(
        f"score: valid={final_score['valid']} memory={final_score['memory']} "
        f"params={final_score['params']} cost={final_score['cost']} score={final_score['score']:.6f}"
    )


if __name__ == "__main__":
    main()
