"""Minimal ONNX for ARC task393 using foreground color counts.

Task rule: the active input is a 12x12 grid with exactly three non-black
foreground colors. Sum the cells for each foreground color, sort colors by
descending total area, and emit a 3x1 vertical output whose rows are the
largest, middle, and smallest colors. All cells outside that 3x1 output are
zero-padded for the NeuroGolf interface.

The training data includes same-color disconnected specks, so ranking
individual 4-connected components is not the correct local rule. The ONNX graph
uses the cheaper equivalent that passes the examples: reduce the one-hot input
to per-channel counts, ignore background channel 0, TopK channels 1..9, scatter
three positive float cells into a compact [1,9,3,1] foreground slab, then pad
once to [1,10,30,30]. Counts are cast to float16 before TopK because all
observed object areas are small exact integers and TopK supports float16.
"""

from __future__ import annotations

from collections import deque
import json
import sys
from pathlib import Path
from typing import Callable, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task393"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task393.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

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

    ch_st = _i64(inits, [1], "ch_st")
    ch_en = _i64(inits, [10], "ch_en")
    top3 = _i64(inits, [3], "top3")
    zero_slab = _init(inits, np.zeros((1, 9, 3, 1), dtype=np.float32), "zero_slab")
    updates = _init(inits, np.ones((1, 1, 3, 1), dtype=np.float32), "updates")

    nodes.append(helper.make_node("ReduceSum", [IN_NAME], ["counts10f"], axes=[0, 2, 3], keepdims=0))
    nodes.append(helper.make_node("Cast", ["counts10f"], ["counts10"], to=TensorProto.FLOAT16))
    nodes.append(helper.make_node("Slice", ["counts10", ch_st, ch_en], ["counts9"]))
    nodes.append(helper.make_node("TopK", ["counts9", top3], ["top_vals", "top_idx"], axis=0))
    nodes.append(helper.make_node("Unsqueeze", ["top_idx"], ["rank_idx"], axes=[0, 1, 3]))
    nodes.append(helper.make_node("Scatter", [zero_slab, "rank_idx", updates], ["out9"], axis=1))
    nodes.append(
        helper.make_node(
            "Pad",
            ["out9"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 1, 0, 0, 0, 0, H - 3, W - 1],
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


def _output_colors(example: dict, key: str) -> list[int]:
    grid = example[key]
    return [row[0] for row in grid]


def _predict_color_counts(example: dict) -> list[int]:
    counts: dict[int, int] = {}
    for row in example["input"]:
        for color in row:
            if color:
                counts[color] = counts.get(color, 0) + 1
    return [color for color, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]]


def _predict_components(example: dict) -> list[int]:
    grid = example["input"]
    h = len(grid)
    w = len(grid[0])
    seen: set[tuple[int, int]] = set()
    components: list[tuple[int, int, int, int, int]] = []

    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if color == 0 or (r, c) in seen:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen.add((r, c))
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and (nr, nc) not in seen
                        and grid[nr][nc] == color
                    ):
                        seen.add((nr, nc))
                        queue.append((nr, nc))

            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            bbox_area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
            components.append((len(cells), bbox_area, min(rows), min(cols), color))

    ranked = sorted(components, key=lambda item: (-item[0], -item[1], item[2], item[3]))
    return [color for _, _, _, _, color in ranked[:3]]


def _compare_reference_rules() -> dict[str, tuple[int, int]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    rules: dict[str, Callable[[dict], list[int]]] = {
        "component": _predict_components,
        "color_count": _predict_color_counts,
    }
    results: dict[str, tuple[int, int]] = {}
    train = data.get("train", [])
    for name, predict in rules.items():
        passed = sum(predict(example) == _output_colors(example, "output") for example in train)
        results[name] = (passed, len(train))
    return results


def _check_examples(path: Path) -> tuple[int, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    passed = 0
    failed = 0
    for split in ("train", "test", "arc-gen"):
        for example in data.get(split, []):
            expected = convert_to_numpy(example, "output")
            actual = session.run([OUT_NAME], {IN_NAME: convert_to_numpy(example, "input")})[0]
            if np.array_equal(actual > 0.0, expected > 0.0):
                passed += 1
            else:
                failed += 1
    return passed, failed


def main() -> None:
    rule_results = _compare_reference_rules()
    model = build_model()
    onnx.save(model, BEST_PATH)

    passed, failed = _check_examples(BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    for name, (rule_passed, rule_total) in rule_results.items():
        print(f"{name} train rule: {rule_passed}/{rule_total}")
    print(f"examples: {passed} pass, {failed} fail")
    print(f"valid:    {result['valid']}")
    if result["error"]:
        print(f"error:    {result['error']}")
    print(f"memory:   {result['memory']}")
    print(f"params:   {result['params']}")
    print(f"cost:     {result['cost']}")
    if result["score"] is not None:
        print(f"score:    {result['score']:.6f}")


if __name__ == "__main__":
    main()
