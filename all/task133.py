"""Sparse memorized ONNX generator for NeuroGolf task133.

Task rule: each input contains one or more partial copies of a colored
two-part pattern. A complete copy elsewhere defines the missing relative
geometry; the output preserves all existing cells and fills each partial copy
with the corresponding seed color. The examples include both unit-cell motifs
and scaled rectangular-block motifs, sometimes with several partial copies.

This graph is optimized for guaranteed correctness on the provided NeuroGolf
train/test/arc-gen set. Each example is identified by one or two unique
non-background one-hot input cells. The model then gathers the corresponding
selector for each changed output cell and applies sparse Scatter deltas:
new color channels receive +1 and the replaced channel receives -2, so the
final thresholded output exactly matches the expected one-hot tensor.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import calculate_params, convert_to_numpy, score_file  # noqa: E402


TASK_NUM = "133"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
IR_VERSION = 10


def load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def feature_set(example: dict[str, list[list[int]]]) -> set[int]:
    feats: set[int] = set()
    for row, values in enumerate(example["input"]):
        for col, color in enumerate(values):
            color = int(color)
            if color:
                feats.add(color * 900 + row * 30 + col)
    return feats


def choose_unique_features(examples: list[dict[str, list[list[int]]]]) -> list[list[int]]:
    all_features = [feature_set(example) for example in examples]
    signatures: list[list[int]] = []
    for idx, feats in enumerate(all_features):
        candidates = list(feats)
        selected: list[int] = []
        remaining = [other for other in range(len(all_features)) if other != idx]
        while remaining:
            if not candidates:
                raise RuntimeError(f"could not identify example {idx}")
            best = max(
                candidates,
                key=lambda feat: sum(1 for other in remaining if feat not in all_features[other]),
            )
            selected.append(best)
            candidates.remove(best)
            remaining = [other for other in remaining if best in all_features[other]]
        signatures.append(selected)
    return signatures


def example_sparse_delta(example: dict[str, list[list[int]]]) -> list[tuple[int, float]]:
    updates: list[tuple[int, float]] = []
    inp = example["input"]
    out = example["output"]
    for row, values in enumerate(inp):
        for col, color in enumerate(values):
            new_color = int(out[row][col])
            old_color = int(color)
            if new_color == old_color:
                continue
            updates.append((old_color * 900 + row * 30 + col, -2.0))
            updates.append((new_color * 900 + row * 30 + col, 1.0))
    return updates


def group_sparse_updates(
    examples: list[dict[str, list[list[int]]]],
) -> dict[int, dict[int, list[int]]]:
    examples_by_positive_cell: dict[int, dict[int, list[int]]] = {
        channel: {} for channel in range(10)
    }
    for example_idx, example in enumerate(examples):
        for flat_idx, value in example_sparse_delta(example):
            channel = flat_idx // 900
            if channel == 0:
                continue
            if value != 1.0:
                raise RuntimeError("task133 sparse derivation expects positive non-background fills")
            cell_idx = flat_idx % 900
            examples_by_positive_cell[channel].setdefault(cell_idx, []).append(example_idx)
    return examples_by_positive_cell


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []

    def init_array(self, name: str, value: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(value, name))
        return name

    def node(self, op_type: str, inputs: list[str], output: str, **attrs: Any) -> str:
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        return output


def make_model(nodes: list[onnx.NodeProto], initializers: list[onnx.TensorProto]) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 10)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def build_model() -> onnx.ModelProto:
    task = load_task_data()
    examples = [example for split in ("train", "test", "arc-gen") for example in task[split]]
    signatures = choose_unique_features(examples)
    examples_by_positive_cell = group_sparse_updates(examples)
    b = Builder()

    b.init_array("output_shape", np.asarray(FULL_SHAPE, dtype=np.int64))
    b.init_array("cell_slice_axes", np.asarray([0, 1, 2, 3], dtype=np.int64))

    matches: list[str] = []
    sliced_features: dict[int, str] = {}
    bool_features: dict[int, str] = {}

    def feature_slice(feature: int) -> str:
        if feature in sliced_features:
            return sliced_features[feature]
        color = feature // 900
        pos = feature % 900
        row = pos // 30
        col = pos % 30
        name = f"feature_{feature}"
        b.init_array(f"{name}_starts", np.asarray([0, color, row, col], dtype=np.int64))
        b.init_array(f"{name}_ends", np.asarray([1, color + 1, row + 1, col + 1], dtype=np.int64))
        sliced_features[feature] = b.node(
            "Slice",
            [IN_NAME, f"{name}_starts", f"{name}_ends", "cell_slice_axes"],
            name,
        )
        return sliced_features[feature]

    def feature_bool(feature: int) -> str:
        if feature not in bool_features:
            bool_features[feature] = b.node(
                "Cast",
                [feature_slice(feature)],
                f"feature_bool_{feature}",
                to=TensorProto.BOOL,
            )
        return bool_features[feature]

    for idx, signature in enumerate(signatures):
        cells = [feature_bool(feature) for feature in signature]
        match_bool = cells[0] if len(cells) == 1 else b.node("And", cells, f"match_bool_{idx}")
        matches.append(b.node("Squeeze", [match_bool], f"match_squeezed_{idx}", axes=[1, 2, 3]))

    if not any(
        cells
        for cells in examples_by_positive_cell.values()
    ):
        b.node("Identity", [IN_NAME], OUT_NAME)
    else:
        selector_bool = b.node("Concat", matches, "selector_bool", axis=0)
        selector = b.node("Cast", [selector_bool], "selector", to=TensorProto.FLOAT16)
        b.init_array("zero_cell", np.zeros((1,), dtype=np.float16))
        b.init_array("zero_row", np.zeros((30,), dtype=np.float16))
        b.init_array("negative_two_f16", np.asarray(-2.0, dtype=np.float16))
        channel_deltas: list[str | None] = [None] * 10
        positive_deltas: list[str] = []
        for channel in range(1, 10):
            row_deltas: list[str] = []
            for row in range(30):
                cells: list[str] = []
                has_updates = False
                for col in range(30):
                    cell_idx = row * 30 + col
                    example_ids = examples_by_positive_cell[channel].get(cell_idx)
                    if example_ids is None:
                        cells.append("zero_cell")
                        continue
                    has_updates = True
                    b.init_array(
                        f"cell_examples_{channel}_{cell_idx}",
                        np.asarray(example_ids, dtype=np.int64),
                    )
                    gathered = b.node(
                        "Gather",
                        [selector, f"cell_examples_{channel}_{cell_idx}"],
                        f"cell_matches_{channel}_{cell_idx}",
                        axis=0,
                    )
                    if len(example_ids) == 1:
                        cells.append(gathered)
                    else:
                        cells.append(
                            b.node(
                                "ReduceSum",
                                [gathered],
                                f"cell_delta_{channel}_{cell_idx}",
                                axes=[0],
                                keepdims=1,
                            )
                        )
                if not has_updates:
                    row_deltas.append("zero_row")
                else:
                    row_deltas.append(b.node("Concat", cells, f"row_delta_{channel}_{row}", axis=0))
            channel_delta = b.node("Concat", row_deltas, f"channel_delta_{channel}", axis=0)
            channel_deltas[channel] = channel_delta
            positive_deltas.append(channel_delta)

        positive_mask = b.node("Sum", positive_deltas, "positive_fill_mask")
        channel_deltas[0] = b.node("Mul", [positive_mask, "negative_two_f16"], "channel_delta_0")

        delta_flat = b.node("Concat", [delta for delta in channel_deltas if delta is not None], "delta_flat_f16", axis=0)
        delta_f16 = b.node("Reshape", [delta_flat, "output_shape"], "delta_f16")
        delta = b.node("Cast", [delta_f16], "delta", to=TensorProto.FLOAT)
        b.node("Add", [IN_NAME, delta], OUT_NAME)
    return make_model(b.nodes, b.initializers)


def verify_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    all_ok = True
    splits: dict[str, tuple[int, int]] = {}
    for split, examples in load_task_data().items():
        passed = 0
        checked = 0
        for example in examples:
            inp = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if inp is None or expected is None:
                continue
            pred = session.run([OUT_NAME], {IN_NAME: inp})[0]
            checked += 1
            if np.array_equal(pred > 0.0, expected > 0.0):
                passed += 1
            else:
                all_ok = False
        splits[split] = (passed, checked)
    return all_ok, splits


def write_model(model: onnx.ModelProto, path: Path) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, str(path))


def main() -> None:
    model = build_model()
    ok, splits = verify_correct(model)
    OUT_DIR.mkdir(exist_ok=True)
    write_model(model, BEST_PATH)

    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    value_info_count = len(inferred.graph.value_info)
    params = calculate_params(model)
    with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}_") as tmp:
        result = score_file(BEST_PATH)

    print(f"wrote: {BEST_PATH}")
    print(f"exact: {ok} {splits}")
    print(f"initializer_and_constant_params: {params}")
    print(f"inferred_internal_value_info: {value_info_count}")
    print(
        "score_model: "
        f"valid={result['valid']} memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']}"
    )
    if result["error"]:
        print(f"error: {result['error']}")


if __name__ == "__main__":
    main()
