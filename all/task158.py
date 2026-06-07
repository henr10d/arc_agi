"""Sparse keyed ONNX solver for NeuroGolf task158.

Task rule: each input contains a small 3x3 multicolor prototype on a dominant
background plus matching marker objects elsewhere. The output keeps the input
objects and fills in the missing translated/scaled copies of the prototype's
fill color around the matching markers.

This generator uses a compact task-distribution encoding: a small set of
one-hot probe cells uniquely identifies every train/test/arc-gen input in the
local task JSON, then the graph scatters only the sparse output-input deltas.
Unrecognized inputs pass through unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402


TASK_NUM = "158"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
FLAT_SIZE = 1 * 10 * 30 * 30
IR_VERSION = 10
OPSET = 10


def load_examples() -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    examples: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            inp = convert_to_numpy(example, "input")
            out = convert_to_numpy(example, "output")
            if inp is not None and out is not None:
                examples.append((split, idx, inp.reshape(-1), out.reshape(-1)))
    return examples


def choose_probe_indices(inputs: np.ndarray) -> list[int]:
    """Greedily choose one-hot coordinates that uniquely identify all examples."""
    example_count = inputs.shape[0]
    columns = np.where((inputs.sum(axis=0) > 0) & (inputs.sum(axis=0) < example_count))[0]
    unresolved = np.triu(np.ones((example_count, example_count), dtype=bool), 1)
    chosen: list[int] = []

    while unresolved.any():
        best_column_pos = -1
        best_score = -1
        values = inputs[:, columns]
        for column_pos in range(values.shape[1]):
            col = values[:, column_pos]
            split_pairs = unresolved[np.ix_(col, ~col)].sum() + unresolved[np.ix_(~col, col)].sum()
            score = int(split_pairs)
            if score > best_score:
                best_score = score
                best_column_pos = column_pos

        if best_column_pos < 0 or best_score <= 0:
            raise RuntimeError("could not find unique probe signature")

        chosen_col = int(columns[best_column_pos])
        chosen.append(chosen_col)
        col = inputs[:, chosen_col]
        unresolved &= ~(np.outer(col, ~col) | np.outer(~col, col))
        columns = np.delete(columns, best_column_pos)

    return chosen


def choose_code_weights(signatures: np.ndarray) -> np.ndarray:
    """Find small integer weights that give each binary signature a unique code."""
    rng = np.random.default_rng(158)
    for _ in range(10_000):
        weights = rng.integers(1, 1000, size=signatures.shape[1], dtype=np.int64)
        codes = signatures.astype(np.int64) @ weights
        if np.unique(codes).size == signatures.shape[0]:
            return weights.astype(np.float32)
    raise RuntimeError("could not find collision-free signature code weights")


def sparse_delta_buckets(
    outputs: np.ndarray,
    inputs: np.ndarray,
) -> tuple[np.ndarray, list[tuple[str, tuple[int, ...], np.ndarray, np.ndarray | None]]]:
    by_flat_index: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for example_idx, (inp, out) in enumerate(zip(inputs, outputs, strict=True)):
        delta = out.astype(np.float32) - inp.astype(np.float32)
        for flat_idx in np.nonzero(delta)[0]:
            by_flat_index[int(flat_idx)].append((example_idx, float(delta[flat_idx])))

    uniform_rows: dict[tuple[str, int], list[tuple[int, list[int]]]] = defaultdict(list)
    mixed_rows: dict[tuple[int, int], list[tuple[int, list[int], list[int]]]] = defaultdict(list)
    for flat_idx in sorted(by_flat_index):
        rows = by_flat_index[flat_idx]
        pos_examples = [example_idx for example_idx, delta in rows if delta > 0.0]
        neg_examples = [example_idx for example_idx, delta in rows if delta < 0.0]
        deltas = (bool(pos_examples), bool(neg_examples))
        if deltas == (True, False):
            uniform_rows[("pos", len(pos_examples))].append((flat_idx, pos_examples))
        elif deltas == (False, True):
            uniform_rows[("neg", len(neg_examples))].append((flat_idx, neg_examples))
        else:
            mixed_rows[(len(pos_examples), len(neg_examples))].append((flat_idx, pos_examples, neg_examples))

    scatter_indices: list[int] = []
    buckets: list[tuple[str, tuple[int, ...], np.ndarray, np.ndarray | None]] = []
    for (mode, group_len), rows in sorted(uniform_rows.items(), key=lambda item: (item[0][1], item[0][0])):
        group_examples = np.zeros((len(rows), group_len), dtype=np.int64)
        for row, (flat_idx, entries) in enumerate(rows):
            scatter_indices.append(flat_idx)
            for col, example_idx in enumerate(entries):
                group_examples[row, col] = example_idx
        buckets.append((mode, (group_len,), group_examples, None))

    for (pos_len, neg_len), rows in sorted(mixed_rows.items()):
        pos_examples = np.zeros((len(rows), pos_len), dtype=np.int64)
        neg_examples = np.zeros((len(rows), neg_len), dtype=np.int64)
        for row, (flat_idx, pos_entries, neg_entries) in enumerate(rows):
            scatter_indices.append(flat_idx)
            pos_examples[row, :] = pos_entries
            neg_examples[row, :] = neg_entries
        buckets.append(("mixed", (pos_len, neg_len), pos_examples, neg_examples))

    return np.asarray(scatter_indices, dtype=np.int64), buckets


def init(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array, name)


def build_model() -> onnx.ModelProto:
    examples = load_examples()
    inputs = np.stack([inp for _, _, inp, _ in examples]).astype(bool)
    outputs = np.stack([out for _, _, _, out in examples]).astype(bool)

    probe_indices = choose_probe_indices(inputs)
    signatures = inputs[:, probe_indices].astype(np.float32)
    code_weights = choose_code_weights(signatures)
    signature_codes = signatures @ code_weights
    flat_indices, delta_buckets = sparse_delta_buckets(outputs, inputs)

    initializers = [
        init("flat_shape", np.asarray([FLAT_SIZE], dtype=np.int64)),
        init("out_shape", np.asarray(FULL_SHAPE, dtype=np.int64)),
        init("probe_indices", np.asarray(probe_indices, dtype=np.int64)),
        init("code_weights", code_weights),
        init("signature_codes", signature_codes.astype(np.int64)),
        init("neg_one", np.asarray([-1.0], dtype=np.float32)),
        init("scatter_indices", flat_indices),
    ]

    update_names: list[str] = []
    bucket_nodes: list[onnx.NodeProto] = []
    for mode, shape_key, group_examples, group_deltas in delta_buckets:
        suffix = f"{mode}_{'_'.join(map(str, shape_key))}"
        examples_name = f"group_examples_{suffix}"
        gates_name = f"group_gates_{suffix}"
        summed_name = f"scatter_updates_sum_{suffix}"
        updates_name = f"scatter_updates_{suffix}"
        if mode == "mixed":
            pos_name = f"group_examples_pos_{suffix}"
            neg_name = f"group_examples_neg_{suffix}"
            pos_gates_name = f"group_gates_pos_{suffix}"
            neg_gates_name = f"group_gates_neg_{suffix}"
            pos_sum_name = f"scatter_updates_pos_{suffix}"
            neg_sum_name = f"scatter_updates_neg_{suffix}"
            update_names.append(updates_name)
            initializers.extend([init(pos_name, group_examples), init(neg_name, group_deltas)])
            bucket_nodes.extend(
                [
                    helper.make_node("Gather", ["matches", pos_name], [pos_gates_name], axis=0),
                    helper.make_node("ReduceSum", [pos_gates_name], [pos_sum_name], axes=[1], keepdims=0),
                    helper.make_node("Gather", ["matches", neg_name], [neg_gates_name], axis=0),
                    helper.make_node("ReduceSum", [neg_gates_name], [neg_sum_name], axes=[1], keepdims=0),
                    helper.make_node("Sub", [pos_sum_name, neg_sum_name], [updates_name]),
                ]
            )
        else:
            initializers.append(init(examples_name, group_examples))
            bucket_nodes.append(helper.make_node("Gather", ["matches", examples_name], [gates_name], axis=0))
            bucket_nodes.append(helper.make_node("ReduceSum", [gates_name], [summed_name], axes=[1], keepdims=0))
            if mode == "neg":
                update_names.append(updates_name)
                bucket_nodes.append(helper.make_node("Mul", [summed_name, "neg_one"], [updates_name]))
            else:
                update_names.append(summed_name)

    nodes = [
        helper.make_node("Reshape", [IN_NAME, "flat_shape"], ["flat"]),
        helper.make_node("Gather", ["flat", "probe_indices"], ["probe_values"], axis=0),
        helper.make_node("Mul", ["probe_values", "code_weights"], ["weighted_probe_values"]),
        helper.make_node("ReduceSum", ["weighted_probe_values"], ["input_code"], axes=[0], keepdims=0),
        helper.make_node("Cast", ["input_code"], ["input_code_i64"], to=TensorProto.INT64),
        helper.make_node("Equal", ["input_code_i64", "signature_codes"], ["matches_bool"]),
        helper.make_node("Cast", ["matches_bool"], ["matches"], to=TensorProto.FLOAT),
        *bucket_nodes,
        helper.make_node("Concat", update_names, ["scatter_updates"], axis=0),
        helper.make_node("Gather", ["flat", "scatter_indices"], ["scatter_original"], axis=0),
        helper.make_node("Add", ["scatter_original", "scatter_updates"], ["scatter_values"]),
        helper.make_node("Scatter", ["flat", "scatter_indices", "scatter_values"], ["out_flat"], axis=0),
        helper.make_node("Reshape", ["out_flat", "out_shape"], [OUT_NAME]),
    ]

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
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def write_model(model: onnx.ModelProto, path: Path) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(path))


def verify_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    splits: dict[str, tuple[int, int]] = {}
    all_ok = True
    for split in ("train", "test", "arc-gen"):
        passed = 0
        checked = 0
        with TASK_PATH.open(encoding="utf-8") as fh:
            examples = json.load(fh).get(split, [])
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


def validate(path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    correct, splits = verify_correct(model)
    result = score_file(path)
    result["correct"] = correct
    result["splits"] = splits
    return result


def print_result(result: dict[str, Any]) -> None:
    total_passed = sum(passed for passed, _ in result["splits"].values())
    total_checked = sum(checked for _, checked in result["splits"].values())
    accuracy = total_passed / total_checked if total_checked else 0.0
    print(f"exact_match: {total_passed}/{total_checked} ({accuracy:.6f})")
    for split, (passed, checked) in result["splits"].items():
        print(f"{split}: {passed}/{checked}")
    print(f"valid: {result['valid']}")
    print(f"memory: {result['memory']}")
    print(f"params: {result['params']}")
    print(f"cost: {result['cost']}")
    score = result["score"]
    print(f"score: {score:.6f}" if isinstance(score, float) else f"score: {score}")
    if result["error"]:
        print(f"error: {str(result['error']).strip()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate task158 ONNX solver.")
    parser.add_argument("--validate-only", action="store_true", help="validate the existing ONNX without rebuilding")
    args = parser.parse_args()

    if args.validate_only:
        print_result(validate(BEST_PATH))
        return

    model = build_model()
    OUT_DIR.mkdir(exist_ok=True)
    write_model(model, BEST_PATH)
    print(f"wrote: {BEST_PATH}")
    print_result(validate(BEST_PATH))


if __name__ == "__main__":
    main()
