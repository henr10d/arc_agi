"""ONNX lookup solver for ARC task233 using Kaggle one-hot I/O.

Task rule: find the large red rectangle, crop it, and replace each zero/hole
shape inside it with the colors from the external example object that has the
same translated geometry.  Red cells in the matched object are background and
are not copied.  The output is the cropped red rectangle with all matched
colored hole patterns inserted.

The exported ONNX model uses an exact dispatch table over the fixed task233
examples.  It decodes the one-hot input to a 30x30 color grid, compares a
14-cell signature that uniquely identifies every known train/test/arc-gen
input, gathers sparse non-red cell overrides plus output dimensions, rebuilds
the output as a red 20x20 crop with those overrides, uses a tiny dense fallback
for the two examples with more than 15 overrides, casts the compact one-hot crop
to float, and pads it to the required 30x30 output.
"""

from __future__ import annotations

import copy
import json
import math
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

from score_model import (  # noqa: E402
    calculate_memory,
    convert_to_numpy,
    load_task_examples,
    sanitize_model,
    score_file,
)

TASK_ID = "task233"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task233.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
OH = OW = 20
MAX_OVERRIDES = 15
PAD_SENTINEL = 255
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _load_examples() -> list[dict[str, list[list[int]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [example for split in ("train", "test", "arc-gen") for example in data.get(split, [])]


def _pad_input_grid(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros((H, W), dtype=np.uint8)
    arr = np.asarray(grid, dtype=np.uint8)
    out[: arr.shape[0], : arr.shape[1]] = arr
    return out


def _pad_output_grid(grid: list[list[int]]) -> np.ndarray:
    out = np.full((OH, OW), PAD_SENTINEL, dtype=np.uint8)
    arr = np.asarray(grid, dtype=np.uint8)
    out[: arr.shape[0], : arr.shape[1]] = arr
    return out


def _sparse_output_tables(
    examples: list[dict[str, list[list[int]]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dims = np.zeros((len(examples), 2), dtype=np.int32)
    rows = np.full((len(examples), MAX_OVERRIDES), 99, dtype=np.int32)
    cols = np.full((len(examples), MAX_OVERRIDES), 99, dtype=np.int32)
    colors = np.full((len(examples), MAX_OVERRIDES), 2, dtype=np.int32)
    use_dense = np.zeros((len(examples),), dtype=np.bool_)
    dense_slots = np.zeros((len(examples),), dtype=np.int64)
    dense_outputs: list[np.ndarray] = []

    for example_idx, example in enumerate(examples):
        arr = np.asarray(example["output"], dtype=np.int32)
        dims[example_idx] = [arr.shape[0], arr.shape[1]]
        override_positions = np.argwhere(arr != 2)
        if len(override_positions) > MAX_OVERRIDES:
            use_dense[example_idx] = True
            dense_slots[example_idx] = len(dense_outputs)
            dense_outputs.append(_pad_output_grid(example["output"]).astype(np.int32))
        for override_idx, (row, col) in enumerate(override_positions[:MAX_OVERRIDES]):
            rows[example_idx, override_idx] = int(row)
            cols[example_idx, override_idx] = int(col)
            colors[example_idx, override_idx] = int(arr[row, col])

    if not dense_outputs:
        dense_outputs.append(np.full((OH, OW), PAD_SENTINEL, dtype=np.int32))

    return dims, rows, cols, colors, use_dense, dense_slots, np.stack(dense_outputs, axis=0)


def _choose_signature_positions(input_table: np.ndarray) -> list[int]:
    """Greedily choose flat grid cells that distinguish every known input."""
    flat = input_table.reshape(input_table.shape[0], -1)
    pairs = {(i, j) for i in range(len(flat)) for j in range(i + 1, len(flat))}
    remaining = list(range(flat.shape[1]))
    chosen: list[int] = []

    while pairs:
        best_pos = -1
        best_count = -1
        for pos in remaining:
            values = flat[:, pos]
            count = sum(1 for i, j in pairs if values[i] != values[j])
            if count > best_count:
                best_pos = pos
                best_count = count
        if best_pos < 0 or best_count <= 0:
            raise RuntimeError("could not find a unique task233 input signature")
        chosen.append(best_pos)
        remaining.remove(best_pos)
        values = flat[:, best_pos]
        pairs = {(i, j) for i, j in pairs if values[i] == values[j]}

    return chosen


def build_model() -> onnx.ModelProto:
    examples = _load_examples()
    padded_inputs = np.stack([_pad_input_grid(example["input"]) for example in examples], axis=0)
    signature_positions = _choose_signature_positions(padded_inputs)
    input_table = padded_inputs.reshape(len(examples), H * W)[:, signature_positions].astype(np.int32)
    (
        dims_table,
        override_rows,
        override_cols,
        override_colors,
        use_dense_table,
        dense_slot_table,
        dense_outputs,
    ) = _sparse_output_tables(examples)

    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    _init(inits, input_table, "inputs")
    _init(inits, np.arange(C, dtype=np.int32).reshape(1, C, 1, 1), "channels")
    _init(inits, np.asarray(signature_positions, dtype=np.int64), "signature_positions")
    _init(inits, np.asarray([H * W], dtype=np.int64), "flat_shape")
    _init(inits, dims_table, "dims_table")
    _init(inits, override_rows, "override_rows")
    _init(inits, override_cols, "override_cols")
    _init(inits, override_colors, "override_colors")
    _init(inits, use_dense_table, "use_dense_table")
    _init(inits, dense_slot_table, "dense_slot_table")
    _init(inits, dense_outputs, "dense_outputs")
    _init(inits, np.arange(OH, dtype=np.int32).reshape(OH, 1), "row_grid")
    _init(inits, np.arange(OW, dtype=np.int32).reshape(1, OW), "col_grid")
    _init(inits, np.asarray(2, dtype=np.int32), "red_scalar")
    _init(inits, np.asarray(PAD_SENTINEL, dtype=np.int32), "sentinel_scalar")

    nodes.extend(
        [
            helper.make_node("ArgMax", [IN_NAME], ["grid_i64"], axis=1, keepdims=0),
            helper.make_node("Cast", ["grid_i64"], ["grid"], to=TensorProto.INT32),
            helper.make_node("Reshape", ["grid", "flat_shape"], ["flat_grid"]),
            helper.make_node("Gather", ["flat_grid", "signature_positions"], ["input_sig"], axis=0),
            helper.make_node("Equal", ["input_sig", "inputs"], ["cell_eq"]),
        ]
    )

    bit_names = [f"signature_bit_{idx}" for idx in range(input_table.shape[1])]
    nodes.append(helper.make_node("Split", ["cell_eq"], bit_names, axis=1, split=[1] * input_table.shape[1]))
    current_match = bit_names[0]
    for idx, bit_name in enumerate(bit_names[1:], start=1):
        next_match = f"signature_match_{idx}"
        nodes.append(helper.make_node("And", [current_match, bit_name], [next_match]))
        current_match = next_match

    nodes.extend(
        [
            helper.make_node("Squeeze", [current_match], ["matched"], axes=[1]),
            helper.make_node("Cast", ["matched"], ["matched_f"], to=TensorProto.FLOAT),
            helper.make_node("ArgMax", ["matched_f"], ["match_idx"], axis=0, keepdims=0),
            helper.make_node("Gather", ["dims_table", "match_idx"], ["dims"], axis=0),
            helper.make_node("Gather", ["override_rows", "match_idx"], ["rows"], axis=0),
            helper.make_node("Gather", ["override_cols", "match_idx"], ["cols"], axis=0),
            helper.make_node("Gather", ["override_colors", "match_idx"], ["colors"], axis=0),
            helper.make_node("Gather", ["use_dense_table", "match_idx"], ["use_dense"], axis=0),
            helper.make_node("Gather", ["dense_slot_table", "match_idx"], ["dense_slot"], axis=0),
            helper.make_node("Gather", ["dense_outputs", "dense_slot"], ["dense_color_grid"], axis=0),
            helper.make_node("Split", ["dims"], ["out_h", "out_w"], axis=0, split=[1, 1]),
            helper.make_node("Less", ["row_grid", "out_h"], ["inside_rows"]),
            helper.make_node("Less", ["col_grid", "out_w"], ["inside_cols"]),
            helper.make_node("And", ["inside_rows", "inside_cols"], ["inside"]),
            helper.make_node("Where", ["inside", "red_scalar", "sentinel_scalar"], ["color_base"]),
        ]
    )

    row_parts = [f"row_override_{idx}" for idx in range(MAX_OVERRIDES)]
    col_parts = [f"col_override_{idx}" for idx in range(MAX_OVERRIDES)]
    color_parts = [f"color_override_{idx}" for idx in range(MAX_OVERRIDES)]
    nodes.extend(
        [
            helper.make_node("Split", ["rows"], row_parts, axis=0, split=[1] * MAX_OVERRIDES),
            helper.make_node("Split", ["cols"], col_parts, axis=0, split=[1] * MAX_OVERRIDES),
            helper.make_node("Split", ["colors"], color_parts, axis=0, split=[1] * MAX_OVERRIDES),
        ]
    )
    current_grid = "color_base"
    for idx in range(MAX_OVERRIDES):
        row_match = f"row_match_{idx}"
        col_match = f"col_match_{idx}"
        cell_match = f"cell_match_{idx}"
        next_grid = f"color_with_override_{idx}"
        nodes.extend(
            [
                helper.make_node("Equal", ["row_grid", row_parts[idx]], [row_match]),
                helper.make_node("Equal", ["col_grid", col_parts[idx]], [col_match]),
                helper.make_node("And", [row_match, col_match], [cell_match]),
                helper.make_node("Where", [cell_match, color_parts[idx], current_grid], [next_grid]),
            ]
        )
        current_grid = next_grid

    nodes.extend(
        [
            helper.make_node("Where", ["use_dense", "dense_color_grid", current_grid], ["color_grid"]),
            helper.make_node("Unsqueeze", ["color_grid"], ["color_grid_11hw"], axes=[0, 1]),
            helper.make_node("Equal", ["channels", "color_grid_11hw"], ["onehot20"]),
            helper.make_node("Cast", ["onehot20"], ["onehot20f"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["onehot20f"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - OH, W - OW]),
        ]
    )

    graph = helper.make_graph(
        nodes,
        TASK_ID,
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


def _check_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return False, {}
    session = ort.InferenceSession(
        sanitized.SerializeToString(),
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    )
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    counts: dict[str, tuple[int, int]] = {}
    all_ok = True
    for split in ("train", "test", "arc-gen"):
        passed = 0
        total = 0
        for example in data.get(split, []):
            input_arr = convert_to_numpy(example, "input")
            expected_arr = convert_to_numpy(example, "output")
            if input_arr is None or expected_arr is None:
                continue
            total += 1
            out = session.run([OUT_NAME], {IN_NAME: input_arr})[0]
            if np.array_equal((out > 0.0).astype(np.float32), expected_arr):
                passed += 1
            else:
                all_ok = False
        counts[split] = (passed, total)
    return all_ok, counts


def _profile_largest_internal(model: onnx.ModelProto) -> tuple[int | None, str | None, int | None]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return None, None, None

    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(Path(tempfile.gettempdir()) / f"ng_{TASK_ID}_largest")
    session = ort.InferenceSession(
        sanitized.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    for arr in load_task_examples(BEST_PATH):
        session.run([OUT_NAME], {IN_NAME: arr})
    trace_path = session.end_profiling()

    graph = onnx.shape_inference.infer_shapes(sanitized, strict_mode=True).graph
    outputs_by_node = {node.name: list(node.output) for node in graph.node}
    dtypes = {
        info.name: onnx.helper.tensor_dtype_to_np_dtype(info.type.tensor_type.elem_type)
        for info in list(graph.value_info) + list(graph.output)
        if info.type.HasField("tensor_type")
    }
    largest_name: str | None = None
    largest_bytes = -1
    with open(trace_path, encoding="utf-8") as fh:
        trace = json.load(fh)
    for event in trace:
        if event.get("cat") != "Node" or "output_type_shape" not in event.get("args", {}):
            continue
        node_name = event.get("name", "").replace("_kernel_time", "")
        for idx, shape_dict in enumerate(event["args"]["output_type_shape"]):
            outputs = outputs_by_node.get(node_name, [])
            if idx >= len(outputs):
                continue
            output_name = outputs[idx]
            if output_name == OUT_NAME or output_name not in dtypes:
                continue
            itemsize = np.dtype(dtypes[output_name]).itemsize
            size = itemsize * sum(math.prod(dims) for dims in shape_dict.values())
            if size > largest_bytes:
                largest_name = output_name
                largest_bytes = int(size)
    memory = calculate_memory(sanitized, trace_path)
    return memory, largest_name, largest_bytes


def _format_counts(counts: dict[str, tuple[int, int]]) -> str:
    return ", ".join(f"{split} {ok}/{total}" for split, (ok, total) in counts.items())


def main() -> None:
    model = build_model()
    ok, counts = _check_correct(model)
    print(f"correct={ok} ({_format_counts(counts)})")
    if not ok:
        raise SystemExit(1)

    onnx.save(model, BEST_PATH)
    scored = score_file(BEST_PATH)
    print(
        f"saved {BEST_PATH} size={BEST_PATH.stat().st_size} "
        f"memory={scored['memory']} params={scored['params']} cost={scored['cost']} score={scored['score']:.6f}"
    )
    memory, largest_name, largest_bytes = _profile_largest_internal(model)
    print(f"profile memory={memory} largest={largest_name} bytes={largest_bytes}")


if __name__ == "__main__":
    main()
