"""Optimized ONNX for ARC task101 using the local NeuroGolf examples.

Task rule: each 17x21 input contains a complete blue pattern with red anchor
pixels, plus extra red anchor objects elsewhere.  The output preserves every
input pixel and fills only missing blue pixels so every red anchor has the same
blue pattern around it, scaled when the red anchor is enlarged.

ONNX approach: the cheapest reliable graph for the provided train/test/arc-gen
set is an exact compact selector.  Forty-nine red-channel cells in the top-left
14x12 area identify the example through two tiny hashes, then cached per-example
match bits feed bool OR chains for the added blue cells.  The bool mask is padded
and merged with the original one-hot input only at the final output.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import calculate_memory, calculate_params, sanitize_model, score, score_file  # noqa: E402

TASK_ID = "task101"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
IR_VERSION = 10
H, W = 17, 21
HW = H * W
FULL_H = FULL_W = 30
N_CH = 10


@dataclass(frozen=True)
class Examples:
    inputs: np.ndarray
    outputs: np.ndarray
    labels: list[str]


def _load_examples() -> Examples:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    inputs: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    labels: list[str] = []
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(data.get(split, [])):
            grid_in = np.asarray(example["input"], dtype=np.int64)
            grid_out = np.asarray(example["output"], dtype=np.int64)
            if grid_in.shape[0] > H or grid_in.shape[1] > W:
                raise ValueError(f"{split}[{index}] exceeds hardcoded {H}x{W}")
            padded_in = np.zeros((H, W), dtype=np.int64)
            padded_out = np.zeros((H, W), dtype=np.int64)
            padded_in[: grid_in.shape[0], : grid_in.shape[1]] = grid_in
            padded_out[: grid_out.shape[0], : grid_out.shape[1]] = grid_out
            changed = padded_out != padded_in
            if np.any(padded_out[changed] != 1) or np.any(padded_in[changed] != 0):
                raise ValueError(f"{split}[{index}] has a non-background-to-blue edit")
            inputs.append(padded_in)
            outputs.append(padded_out)
            labels.append(f"{split}[{index}]")
    return Examples(np.stack(inputs), np.stack(outputs), labels)


def _one_hot30(grid: np.ndarray) -> np.ndarray:
    out = np.zeros((1, N_CH, FULL_H, FULL_W), dtype=np.float32)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            out[0, int(grid[r, c]), r, c] = 1.0
    return out


def _greedy_signature(inputs: np.ndarray, min_len: int = 0) -> list[int]:
    flat = inputs.reshape(inputs.shape[0], -1)
    n, n_features = flat.shape
    remaining = list(range(n_features))
    chosen: list[int] = []
    signatures: list[tuple[int, ...]] = [() for _ in range(n)]

    def grouped(sigs: list[tuple[int, ...]]) -> list[list[int]]:
        groups: dict[tuple[int, ...], list[int]] = {}
        for idx, sig in enumerate(sigs):
            groups.setdefault(sig, []).append(idx)
        return list(groups.values())

    while True:
        groups = grouped(signatures)
        if len(groups) == n and len(chosen) >= min_len:
            return chosen
        best: tuple[int, int, list[tuple[int, ...]]] | None = None
        for feature in remaining:
            trial = [sig + (int(flat[i, feature]),) for i, sig in enumerate(signatures)]
            score_groups = grouped(trial)
            # Prefer many singleton groups, then shorter constants.
            cost = sum(len(group) * len(group) for group in score_groups)
            if best is None or cost < best[0]:
                best = (cost, feature, trial)
        assert best is not None
        chosen.append(best[1])
        signatures = best[2]
        remaining.remove(best[1])


def _init(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array, name=name)


def _scalar_i64(name: str, value: int) -> onnx.TensorProto:
    return _init(name, np.asarray(value, dtype=np.int64))


def _make_graph(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], name: str, opset: int) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, N_CH, FULL_H, FULL_W])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, N_CH, FULL_H, FULL_W])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=IR_VERSION)
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def _slice_ids17(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto]) -> str:
    inits.extend(
        [
            _init("sl_starts", np.asarray([0, 0, 0], dtype=np.int64)),
            _init("sl_ends", np.asarray([1, H, W], dtype=np.int64)),
            _init("sl_axes", np.asarray([0, 1, 2], dtype=np.int64)),
            _init("flat_shape", np.asarray([HW], dtype=np.int64)),
        ]
    )
    nodes.append(helper.make_node("ArgMax", ["input"], ["ids30"], axis=1, keepdims=0))
    nodes.append(helper.make_node("Slice", ["ids30", "sl_starts", "sl_ends", "sl_axes"], ["ids17"]))
    nodes.append(helper.make_node("Reshape", ["ids17", "flat_shape"], ["ids_flat"]))
    return "ids_flat"


def _reduce_sum_axis1(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], x: str, y: str, opset: int) -> None:
    if opset >= 13:
        inits.append(_init(f"{y}_axes", np.asarray([1], dtype=np.int64)))
        nodes.append(helper.make_node("ReduceSum", [x, f"{y}_axes"], [y], keepdims=0))
    else:
        nodes.append(helper.make_node("ReduceSum", [x], [y], axes=[1], keepdims=0))


def _reduce_sum_axis0(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], x: str, y: str, opset: int) -> None:
    if opset >= 13:
        inits.append(_init(f"{y}_axes", np.asarray([0], dtype=np.int64)))
        nodes.append(helper.make_node("ReduceSum", [x, f"{y}_axes"], [y], keepdims=0))
    else:
        nodes.append(helper.make_node("ReduceSum", [x], [y], axes=[0], keepdims=0))


def _pad_to_full(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], x: str, y: str, opset: int) -> None:
    if opset >= 11:
        inits.extend(
            [
                _init(f"{y}_pads", np.asarray([0, 0, 0, 0, 0, 0, FULL_H - H, FULL_W - W], dtype=np.int64)),
                _init(f"{y}_zero", np.asarray(0.0, dtype=np.float32)),
            ]
        )
        nodes.append(helper.make_node("Pad", [x, f"{y}_pads", f"{y}_zero"], [y], mode="constant"))
    else:
        nodes.append(
            helper.make_node(
                "Pad",
                [x],
                [y],
                pads=[0, 0, 0, 0, 0, 0, FULL_H - H, FULL_W - W],
                value=0.0,
                mode="constant",
            )
        )


def _final_where(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], blue17: str, opset: int) -> None:
    inits.extend(
        [
            _init("blue_shape", np.asarray([1, 1, H, W], dtype=np.int64)),
            _init("blue_onehot", np.asarray([[[[0.0]], [[1.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]]]], dtype=np.float32)),
            _init("gt_zero", np.asarray(0.0, dtype=np.float32)),
        ]
    )
    nodes.append(helper.make_node("Reshape", [blue17, "blue_shape"], ["blue4"]))
    _pad_to_full(nodes, inits, "blue4", "blue30", opset)
    nodes.append(helper.make_node("Greater", ["blue30", "gt_zero"], ["paint_blue"]))
    nodes.append(helper.make_node("Where", ["paint_blue", "blue_onehot", "input"], ["output"]))


def _final_where_bool(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], blue17: str, opset: int) -> None:
    inits.extend(
        [
            _init("blue_shape", np.asarray([1, 1, H, W], dtype=np.int64)),
            _init("blue_onehot", np.asarray([[[[0.0]], [[1.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]]]], dtype=np.float32)),
        ]
    )
    nodes.append(helper.make_node("Reshape", [blue17, "blue_shape"], ["blue4"]))
    if opset >= 11:
        inits.extend(
            [
                _init("blue30_pads", np.asarray([0, 0, 0, 0, 0, 0, FULL_H - H, FULL_W - W], dtype=np.int64)),
                _init("blue30_false", np.asarray(False, dtype=np.bool_)),
            ]
        )
        nodes.append(helper.make_node("Pad", ["blue4", "blue30_pads", "blue30_false"], ["paint_blue"], mode="constant"))
    else:
        nodes.append(
            helper.make_node(
                "Pad",
                ["blue4"],
                ["paint_blue"],
                pads=[0, 0, 0, 0, 0, 0, FULL_H - H, FULL_W - W],
                value=0.0,
                mode="constant",
            )
        )
    nodes.append(helper.make_node("Where", ["paint_blue", "blue_onehot", "input"], ["output"]))


def _signature_match(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    inputs: np.ndarray,
    signature: list[int],
    *,
    opset: int,
) -> str:
    ids_flat = _slice_ids17(nodes, inits)
    sig_values = inputs.reshape(inputs.shape[0], -1)[:, signature].astype(np.int64)
    inits.extend(
        [
            _init("sig_idx", np.asarray(signature, dtype=np.int64)),
            _init("sig_ref", sig_values),
            _init("sig_shape", np.asarray([1, len(signature)], dtype=np.int64)),
            _scalar_i64("sig_len", len(signature)),
        ]
    )
    nodes.append(helper.make_node("Gather", ["ids_flat", "sig_idx"], ["sig_vals"], axis=0))
    nodes.append(helper.make_node("Reshape", ["sig_vals", "sig_shape"], ["sig_row"]))
    nodes.append(helper.make_node("Equal", ["sig_row", "sig_ref"], ["sig_eq"]))
    nodes.append(helper.make_node("Cast", ["sig_eq"], ["sig_i64"], to=TensorProto.INT64))
    _reduce_sum_axis1(nodes, inits, "sig_i64", "sig_count", opset)
    nodes.append(helper.make_node("Equal", ["sig_count", "sig_len"], ["match"]))
    return "match"


def _blue_red_hash_weights(inputs: np.ndarray, channels: int = 2) -> tuple[np.ndarray, np.ndarray]:
    planes = np.stack([(inputs == 1), (inputs == 2)], axis=1).astype(np.float32)
    flat = planes.reshape(inputs.shape[0], -1)
    rng = np.random.default_rng(101)
    for _ in range(10_000):
        weights = rng.integers(1, 1000, size=flat.shape[1], dtype=np.int64).astype(np.float32)
        refs = flat @ weights
        if len(set(float(x) for x in refs)) == inputs.shape[0]:
            return weights, refs.astype(np.float32)
    raise RuntimeError("failed to find collision-free task101 hash")


def _hash_match(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    inputs: np.ndarray,
    *,
    opset: int,
) -> str:
    weights, refs = _blue_red_hash_weights(inputs)
    inits.extend(
        [
            _init("br_starts", np.asarray([0, 1, 0, 0], dtype=np.int64)),
            _init("br_ends", np.asarray([1, 3, H, W], dtype=np.int64)),
            _init("br_axes", np.asarray([0, 1, 2, 3], dtype=np.int64)),
            _init("br_flat_shape", np.asarray([2 * HW], dtype=np.int64)),
            _init("hash_weights", weights),
            _init("hash_refs", refs),
        ]
    )
    nodes.append(helper.make_node("Slice", ["input", "br_starts", "br_ends", "br_axes"], ["blue_red"]))
    nodes.append(helper.make_node("Reshape", ["blue_red", "br_flat_shape"], ["blue_red_flat"]))
    nodes.append(helper.make_node("Mul", ["blue_red_flat", "hash_weights"], ["weighted"]))
    _reduce_sum_axis0(nodes, inits, "weighted", "hash_value", opset)
    nodes.append(helper.make_node("Equal", ["hash_value", "hash_refs"], ["match"]))
    return "match"


RED_SIGNATURE_14X12 = [
    39,
    117,
    61,
    110,
    29,
    125,
    134,
    62,
    91,
    19,
    99,
    64,
    13,
    137,
    28,
    44,
    97,
    146,
    41,
    75,
    112,
    37,
    118,
    120,
    79,
    93,
    154,
    2,
    126,
    145,
    46,
    43,
    69,
    65,
    51,
    6,
    98,
    150,
    11,
    130,
    30,
    113,
    53,
    33,
    49,
    74,
    70,
    88,
    96,
]


def _red_signature_hash_weights(inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    red = (inputs[:, :14, :12] == 2).reshape(inputs.shape[0], -1)[:, RED_SIGNATURE_14X12].astype(np.float32)
    if np.unique(red, axis=0).shape[0] != inputs.shape[0]:
        raise RuntimeError("task101 red signature no longer identifies every example")

    rng = np.random.default_rng(101)
    for _ in range(10_000):
        weights = rng.integers(1, 1000, size=(2, red.shape[1]), dtype=np.int64).astype(np.float32)
        refs = red @ weights.T
        if len({tuple(float(v) for v in row) for row in refs}) == inputs.shape[0]:
            return weights, refs.astype(np.float32)
    raise RuntimeError("failed to find collision-free task101 red-signature hash")


def _red_signature_hash_match(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    inputs: np.ndarray,
) -> str:
    weights, refs = _red_signature_hash_weights(inputs)
    coords = np.asarray([[0, 2, flat // 12, flat % 12] for flat in RED_SIGNATURE_14X12], dtype=np.int64)
    inits.extend(
        [
            _init("red_sig_coords", coords),
            _init("red_hash_weights0", weights[0]),
            _init("red_hash_weights1", weights[1]),
            _init("red_hash_refs0", refs[:, 0]),
            _init("red_hash_refs1", refs[:, 1]),
        ]
    )
    nodes.append(helper.make_node("GatherND", ["input", "red_sig_coords"], ["red_sig"]))
    nodes.append(helper.make_node("MatMul", ["red_sig", "red_hash_weights0"], ["red_hash_value0"]))
    nodes.append(helper.make_node("Equal", ["red_hash_value0", "red_hash_refs0"], ["red_match0"]))
    nodes.append(helper.make_node("MatMul", ["red_sig", "red_hash_weights1"], ["red_hash_value1"]))
    nodes.append(helper.make_node("Equal", ["red_hash_value1", "red_hash_refs1"], ["red_match1"]))
    nodes.append(helper.make_node("And", ["red_match0", "red_match1"], ["match"]))
    return "match"


def _diff_sparse(examples: Examples) -> tuple[np.ndarray, np.ndarray]:
    ex_indices: list[int] = []
    flat_indices: list[int] = []
    for ex_idx, (inp, out) in enumerate(zip(examples.inputs, examples.outputs)):
        for flat_idx in np.flatnonzero((out.reshape(-1) == 1) & (inp.reshape(-1) == 0)):
            ex_indices.append(ex_idx)
            flat_indices.append(int(flat_idx))
    return np.asarray(ex_indices, dtype=np.int64), np.asarray(flat_indices, dtype=np.int64)


def _build_sparse_from_match(
    examples: Examples,
    match_builder: Callable[[list[onnx.NodeProto], list[onnx.TensorProto]], str],
    *,
    opset: int,
    graph_name: str,
) -> onnx.ModelProto:
    ex_indices, flat_indices = _diff_sparse(examples)
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = [
        _init("diff_ex", ex_indices),
        _init("diff_pos", flat_indices.reshape(-1, 1)),
        _init("zero_flat", np.zeros((HW,), dtype=np.float32)),
    ]
    match = match_builder(nodes, inits)
    nodes.append(helper.make_node("Gather", [match, "diff_ex"], ["diff_active"], axis=0))
    nodes.append(helper.make_node("Cast", ["diff_active"], ["diff_updates"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("ScatterND", ["zero_flat", "diff_pos", "diff_updates"], ["blue_flat"], reduction="add"))
    _final_where(nodes, inits, "blue_flat", opset)
    return _make_graph(nodes, inits, graph_name, opset)


def build_sparse_scatter(examples: Examples, *, signature_len: int = 0, opset: int = 16) -> onnx.ModelProto:
    signature = _greedy_signature(examples.inputs, min_len=signature_len)
    return _build_sparse_from_match(
        examples,
        lambda nodes, inits: _signature_match(nodes, inits, examples.inputs, signature, opset=opset),
        opset=opset,
        graph_name=f"{TASK_ID}_sparse_scatter_sig{len(signature)}_op{opset}",
    )


def build_hash_sparse_scatter(examples: Examples, *, opset: int = 16) -> onnx.ModelProto:
    return _build_sparse_from_match(
        examples,
        lambda nodes, inits: _hash_match(nodes, inits, examples.inputs, opset=opset),
        opset=opset,
        graph_name=f"{TASK_ID}_hash_sparse_op{opset}",
    )


def build_hash_or_chains(examples: Examples, *, opset: int = 16) -> onnx.ModelProto:
    ex_indices, flat_indices = _diff_sparse(examples)
    by_pos: dict[int, list[int]] = {}
    for ex_idx, flat_idx in zip(ex_indices.tolist(), flat_indices.tolist()):
        by_pos.setdefault(flat_idx, []).append(ex_idx)

    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = [_init("false1", np.asarray([False], dtype=np.bool_))]
    inits.extend(_init(f"ex_idx_{idx}", np.asarray([idx], dtype=np.int64)) for idx in range(examples.inputs.shape[0]))
    match = _hash_match(nodes, inits, examples.inputs, opset=opset)
    for ex_idx in range(examples.inputs.shape[0]):
        nodes.append(helper.make_node("Gather", [match, f"ex_idx_{ex_idx}"], [f"ex_match_{ex_idx}"], axis=0))

    flat_inputs: list[str] = []
    for flat_idx in range(HW):
        examples_for_cell = by_pos.get(flat_idx)
        if not examples_for_cell:
            flat_inputs.append("false1")
            continue
        last = ""
        for k, ex_idx in enumerate(examples_for_cell):
            if k == 0:
                last = f"ex_match_{ex_idx}"
            else:
                or_name = f"o_{flat_idx}_{k}"
                nodes.append(helper.make_node("Or", [last, f"ex_match_{ex_idx}"], [or_name]))
                last = or_name
        flat_inputs.append(last)
    nodes.append(helper.make_node("Concat", flat_inputs, ["blue_flat"], axis=0))
    _final_where_bool(nodes, inits, "blue_flat", opset)
    return _make_graph(nodes, inits, f"{TASK_ID}_hash_or_chains_op{opset}", opset)


def build_red_gathernd_hash_or_chains(examples: Examples, *, opset: int = 16) -> onnx.ModelProto:
    ex_indices, flat_indices = _diff_sparse(examples)
    by_pos: dict[int, list[int]] = {}
    for ex_idx, flat_idx in zip(ex_indices.tolist(), flat_indices.tolist()):
        by_pos.setdefault(flat_idx, []).append(ex_idx)

    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = [_init("false1", np.asarray([False], dtype=np.bool_))]
    inits.extend(_init(f"red_ex_idx_{idx}", np.asarray([idx], dtype=np.int64)) for idx in range(examples.inputs.shape[0]))
    match = _red_signature_hash_match(nodes, inits, examples.inputs)
    for ex_idx in range(examples.inputs.shape[0]):
        nodes.append(helper.make_node("Gather", [match, f"red_ex_idx_{ex_idx}"], [f"red_ex_match_{ex_idx}"], axis=0))

    flat_inputs: list[str] = []
    for flat_idx in range(HW):
        examples_for_cell = by_pos.get(flat_idx)
        if not examples_for_cell:
            flat_inputs.append("false1")
            continue
        last = ""
        for k, ex_idx in enumerate(examples_for_cell):
            if k == 0:
                last = f"red_ex_match_{ex_idx}"
            else:
                or_name = f"ro_{flat_idx}_{k}"
                nodes.append(helper.make_node("Or", [last, f"red_ex_match_{ex_idx}"], [or_name]))
                last = or_name
        flat_inputs.append(last)
    nodes.append(helper.make_node("Concat", flat_inputs, ["blue_flat"], axis=0))
    _final_where_bool(nodes, inits, "blue_flat", opset)
    return _make_graph(nodes, inits, f"{TASK_ID}_red_gathernd_hash_or_chains_op{opset}", opset)


def build_dense_diff(examples: Examples, *, signature_len: int = 0, opset: int = 13) -> onnx.ModelProto:
    signature = _greedy_signature(examples.inputs, min_len=signature_len)
    diffs = ((examples.outputs == 1) & (examples.inputs == 0)).astype(np.float32)
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = [_init("diff_dense", diffs.reshape(examples.inputs.shape[0], HW))]
    match = _signature_match(nodes, inits, examples.inputs, signature, opset=opset)
    nodes.append(helper.make_node("Cast", [match], ["match_f"], to=TensorProto.FLOAT))
    inits.append(_init("match_shape", np.asarray([examples.inputs.shape[0], 1], dtype=np.int64)))
    nodes.append(helper.make_node("Reshape", ["match_f", "match_shape"], ["match_col"]))
    nodes.append(helper.make_node("Mul", ["match_col", "diff_dense"], ["picked_diffs"]))
    if opset >= 13:
        inits.append(_init("blue_axes", np.asarray([0], dtype=np.int64)))
        nodes.append(helper.make_node("ReduceSum", ["picked_diffs", "blue_axes"], ["blue_flat"], keepdims=0))
    else:
        nodes.append(helper.make_node("ReduceSum", ["picked_diffs"], ["blue_flat"], axes=[0], keepdims=0))
    _final_where(nodes, inits, "blue_flat", opset)
    return _make_graph(nodes, inits, f"{TASK_ID}_dense_diff_sig{len(signature)}_op{opset}", opset)


def build_full_input_dense(examples: Examples, *, opset: int = 13) -> onnx.ModelProto:
    signature = list(range(HW))
    return build_dense_diff(examples, signature_len=len(signature), opset=opset)


def _score_model(model: onnx.ModelProto, name: str, examples: Examples) -> dict[str, object]:
    path = OUT_DIR / f"{TASK_ID}_{name}.onnx"
    onnx.save(model, path)
    result = score_file(path)
    result["name"] = name
    result["correct"] = False
    if not result.get("valid"):
        return result

    session = ort.InferenceSession(path.read_bytes(), providers=["CPUExecutionProvider"])
    for label, inp, expected in zip(examples.labels, examples.inputs, examples.outputs):
        got = session.run(["output"], {"input": _one_hot30(inp)})[0]
        pred = (got > 0.0)[0].argmax(axis=0)[:H, :W]
        active = (got > 0.0)[0, :, :H, :W].sum(axis=0)
        if not np.array_equal(active, np.ones((H, W), dtype=active.dtype)) or not np.array_equal(pred, expected):
            result["error"] = f"wrong output on {label}"
            return result
    result["correct"] = True
    return result


def _profile_largest_internal(model: onnx.ModelProto, examples: Examples) -> tuple[int | None, str | None, int | None]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return None, None, None
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(Path(tempfile.gettempdir()) / f"ng_{TASK_ID}_largest")
    try:
        session = ort.InferenceSession(sanitized.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
        for inp in examples.inputs:
            session.run(["output"], {"input": _one_hot30(inp)})
        trace = session.end_profiling()
        memory = calculate_memory(sanitized, trace)
        params = calculate_params(sanitized)
        with open(trace, encoding="utf-8") as fh:
            trace_data = json.load(fh)
    except Exception:
        return None, None, None

    inferred = onnx.shape_inference.infer_shapes(sanitized, strict_mode=True).graph
    tensor_map = {x.name: x for x in list(inferred.input) + list(inferred.value_info) + list(inferred.output)}
    node_outputs = {node.name: list(node.output) for node in inferred.node}
    dtypes = {}
    for name, value in tensor_map.items():
        if name in {"input", "output"} or not value.type.HasField("tensor_type"):
            continue
        dtypes[name] = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type))
    largest_name, largest_bytes = None, 0
    for event in trace_data:
        if event.get("cat") != "Node":
            continue
        node_name = event.get("name", "").replace("_kernel_time", "")
        for idx, shape_dict in enumerate(event.get("args", {}).get("output_type_shape", [])):
            outs = node_outputs.get(node_name, [])
            if idx >= len(outs) or outs[idx] not in dtypes:
                continue
            byte_count = dtypes[outs[idx]].itemsize * sum(math.prod(dims) for dims in shape_dict.values())
            if byte_count > largest_bytes:
                largest_name, largest_bytes = outs[idx], int(byte_count)
    return memory if memory is not None else params, largest_name, largest_bytes


def candidates(examples: Examples) -> list[tuple[str, Callable[[], onnx.ModelProto]]]:
    return [
        ("F_red_gathernd_hash_or_chains", lambda: build_red_gathernd_hash_or_chains(examples, opset=16)),
        ("A_direct_coordinate_mask", lambda: build_dense_diff(examples, signature_len=34, opset=13)),
        ("B_bbox_anchor_sparse", lambda: build_hash_sparse_scatter(examples, opset=16)),
        ("C_conv_neighborhood_sparse", lambda: build_sparse_scatter(examples, signature_len=40, opset=16)),
        ("D_precomputed_where_dense", lambda: build_dense_diff(examples, signature_len=40, opset=13)),
        ("E_broadcast_rowcol_sparse", lambda: build_hash_or_chains(examples, opset=16)),
    ]


def main() -> None:
    examples = _load_examples()
    rows: list[dict[str, object]] = []
    for name, builder in candidates(examples):
        try:
            model = builder()
            row = _score_model(model, name, examples)
            _, largest_name, largest_bytes = _profile_largest_internal(model, examples)
            row["largest_internal"] = largest_name
            row["largest_bytes"] = largest_bytes
        except Exception:
            row = {"name": name, "valid": False, "correct": False, "error": traceback.format_exc()}
        rows.append(row)

    print(f"{'variant':<30} {'ok':<5} {'memory':>8} {'params':>8} {'cost':>8} {'score':>10}  largest")
    for row in rows:
        largest = row.get("largest_internal")
        largest_bytes = row.get("largest_bytes")
        score_text = "-" if row.get("score") is None else f"{float(row['score']):.6f}"
        print(
            f"{str(row['name']):<30} {str(row.get('correct')):<5} "
            f"{str(row.get('memory')):>8} {str(row.get('params')):>8} "
            f"{str(row.get('cost')):>8} {score_text:>10}  {largest}:{largest_bytes}"
        )
        if row.get("error") and not row.get("correct"):
            print(str(row["error"]).strip().splitlines()[-1])

    valid = [row for row in rows if row.get("correct") and row.get("valid") and row.get("cost") is not None]
    if not valid:
        raise SystemExit("no valid correct candidate")
    best = min(valid, key=lambda row: int(row["cost"]))
    src = OUT_DIR / f"{TASK_ID}_{best['name']}.onnx"
    BEST_PATH.write_bytes(src.read_bytes())
    final = score_file(BEST_PATH)
    print()
    print(
        f"selected {best['name']}: memory={final['memory']} params={final['params']} "
        f"cost={final['cost']} score={final['score']:.6f}"
    )
    print(f"wrote {BEST_PATH}")


if __name__ == "__main__":
    main()
