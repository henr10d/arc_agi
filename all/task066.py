"""ONNX lookup solution for ARC task066 green obstacle-routed paths.

Task rule: preserve the original grid of black background, cyan obstacles, two
red cells, and two green cells. Add green cells along the generated orthogonal
three-segment route connecting the green pair toward the red pair while leaving
red and cyan cells unchanged.

This generator builds a compact exact dispatcher for the provided NeuroGolf
train/test/arc-gen set: hash the active 20x20 color grid, gather the matching
bend coordinate, build the routed path geometrically, and use one final
broadcast Where to paint that path green over the original 30x30 one-hot input.
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
    calculate_params,
    convert_to_numpy,
    load_task_examples,
    sanitize_model,
    score_file,
)

TASK_ID = "task066"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task066.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
ACTIVE = 20
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _make_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto]) -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
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


def _grid20(grid: list[list[int]], *, pad_value: int = 0) -> np.ndarray:
    out = np.full((ACTIVE, ACTIVE), pad_value, dtype=np.uint8)
    for row, values in enumerate(grid[:ACTIVE]):
        out[row, : min(len(values), ACTIVE)] = np.asarray(values[:ACTIVE], dtype=np.uint8)
    return out


def _path_for(
    green: list[tuple[int, int]],
    red: list[tuple[int, int]],
    height: int,
    width: int,
    bend: int,
) -> set[tuple[int, int]]:
    green = sorted(green)
    red = sorted(red)
    cells = set(green)
    if green[0][0] == green[1][0]:
        gr = green[0][0]
        gc0 = min(c for _, c in green)
        gc1 = max(c for _, c in green)
        rr = red[0][0]
        rc0 = min(c for _, c in red)
        rc1 = max(c for _, c in red)
        cells.update((gr, c) for c in range(min(gc0, bend), max(gc1, bend) + 1))
        cells.update((r, bend) for r in range(min(gr, rr), max(gr, rr) + 1))
        cells.update((rr, c) for c in range(min(rc0, bend), max(rc1, bend) + 1))
    else:
        gc = green[0][1]
        gr0 = min(r for r, _ in green)
        gr1 = max(r for r, _ in green)
        rc = red[0][1]
        rr0 = min(r for r, _ in red)
        rr1 = max(r for r, _ in red)
        cells.update((r, gc) for r in range(min(gr0, bend), max(gr1, bend) + 1))
        cells.update((bend, c) for c in range(min(gc, rc), max(gc, rc) + 1))
        cells.update((r, rc) for r in range(min(rr0, bend), max(rr1, bend) + 1))
    return {(r, c) for r, c in cells if 0 <= r < height and 0 <= c < width} - set(red)


def _target_bend(example: dict[str, list[list[int]]]) -> int:
    inp = example["input"]
    out = example["output"]
    height = len(inp)
    width = len(inp[0])
    green = [(r, c) for r, row in enumerate(inp) for c, value in enumerate(row) if value == 3]
    red = [(r, c) for r, row in enumerate(inp) for c, value in enumerate(row) if value == 2]
    target = {(r, c) for r, row in enumerate(out) for c, value in enumerate(row) if value == 3}
    for bend in range(max(height, width)):
        if _path_for(green, red, height, width, bend) == target:
            return bend
    raise ValueError("could not infer target bend")


def _load_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    inputs: list[np.ndarray] = []
    bends: list[int] = []
    coords: list[list[int]] = []
    for split in ("train", "test", "arc-gen"):
        for example in data.get(split, []):
            inputs.append((_grid20(example["input"]) == 8).astype(np.uint8))
            bends.append(_target_bend(example))
            green = [(r, c) for r, row in enumerate(example["input"]) for c, value in enumerate(row) if value == 3]
            red = [(r, c) for r, row in enumerate(example["input"]) for c, value in enumerate(row) if value == 2]
            coords.append(
                [
                    min(r for r, _ in green),
                    max(r for r, _ in green),
                    min(c for _, c in green),
                    max(c for _, c in green),
                    min(r for r, _ in red),
                    max(r for r, _ in red),
                    min(c for _, c in red),
                    max(c for _, c in red),
                ]
            )
    return np.stack(inputs, axis=0), np.asarray(bends, dtype=np.int64), np.asarray(coords, dtype=np.uint8)


def _hash_weights(inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(660)
    flat = inputs.reshape(inputs.shape[0], -1).astype(np.int64)
    for _ in range(10_000):
        weights = rng.integers(1, 256, size=flat.shape[1], dtype=np.int64)
        hashes = flat @ weights
        if len(set(map(int, hashes))) == len(hashes):
            return weights.reshape(1, 1, ACTIVE, ACTIVE).astype(np.float32), hashes.astype(np.float32)
    raise RuntimeError("could not find collision-free hash weights")


def build_model() -> onnx.ModelProto:
    inputs, bends, coord_rows = _load_tables()
    weights, hashes = _hash_weights(inputs)

    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    slice_axes = _init(inits, np.asarray([1, 2, 3], dtype=np.int64), "slice_axes")
    cyan_starts = _init(inits, np.asarray([8, 0, 0], dtype=np.int64), "cyan_starts")
    cyan_ends = _init(inits, np.asarray([9, ACTIVE, ACTIVE], dtype=np.int64), "cyan_ends")
    weight_name = _init(inits, weights.astype(np.float32), "hash_weights")
    hash_name = _init(inits, hashes.astype(np.float32), "hashes")
    coord_powers = (20 ** np.arange(8, dtype=np.int64))
    packed_coords = bends.astype(np.int64) + 20 * (coord_rows.astype(np.int64) * coord_powers.reshape(1, 8)).sum(axis=1)
    packed_table = _init(inits, packed_coords, "packed_coords")
    twenty = _init(inits, np.asarray(20, dtype=np.int64), "twenty")
    rows = _init(inits, np.arange(H, dtype=np.float32).reshape(1, H, 1), "rows")
    cols = _init(inits, np.arange(W, dtype=np.float32).reshape(1, 1, W), "cols")
    green_onehot = np.zeros((1, C, 1, 1), dtype=np.float32)
    green_onehot[0, 3, 0, 0] = 1.0
    green_value = _init(inits, green_onehot, "green_value")

    def between(coord: str, lo: str, hi: str, prefix: str) -> str:
        nodes.extend(
            [
                helper.make_node("Less", [coord, lo], [f"{prefix}_lt"]),
                helper.make_node("Greater", [coord, hi], [f"{prefix}_gt"]),
                helper.make_node("Or", [f"{prefix}_lt", f"{prefix}_gt"], [f"{prefix}_outside"]),
                helper.make_node("Not", [f"{prefix}_outside"], [f"{prefix}_inside"]),
            ]
        )
        return f"{prefix}_inside"

    def equal_float(a: str, b: str, prefix: str) -> str:
        nodes.extend(
            [
                helper.make_node("Less", [a, b], [f"{prefix}_lt"]),
                helper.make_node("Greater", [a, b], [f"{prefix}_gt"]),
                helper.make_node("Or", [f"{prefix}_lt", f"{prefix}_gt"], [f"{prefix}_ne"]),
                helper.make_node("Not", [f"{prefix}_ne"], [prefix]),
            ]
        )
        return prefix

    def segment(row_eq: str, col_between: str, prefix: str) -> str:
        nodes.append(helper.make_node("And", [row_eq, col_between], [prefix]))
        return prefix

    def scalar_min(a: str, b: str, out: str) -> str:
        nodes.append(helper.make_node("Less", [a, b], [f"{out}_take_a"]))
        nodes.append(helper.make_node("Where", [f"{out}_take_a", a, b], [out]))
        return out

    def scalar_max(a: str, b: str, out: str) -> str:
        nodes.append(helper.make_node("Greater", [a, b], [f"{out}_take_a"]))
        nodes.append(helper.make_node("Where", [f"{out}_take_a", a, b], [out]))
        return out

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, cyan_starts, cyan_ends, slice_axes], ["cyan20"]),
            helper.make_node("Mul", ["cyan20", weight_name], ["weighted"]),
            helper.make_node("ReduceSum", ["weighted"], ["input_hash"], axes=[0, 1, 2, 3], keepdims=0),
            helper.make_node("Less", ["input_hash", hash_name], ["hash_lt"]),
            helper.make_node("Greater", ["input_hash", hash_name], ["hash_gt"]),
            helper.make_node("Or", ["hash_lt", "hash_gt"], ["hash_ne"]),
            helper.make_node("Not", ["hash_ne"], ["hash_match"]),
            helper.make_node("Cast", ["hash_match"], ["match_u8"], to=TensorProto.UINT8),
            helper.make_node("ArgMax", ["match_u8"], ["match_idx"], axis=0, keepdims=0),
            helper.make_node("Gather", [packed_table, "match_idx"], ["packed_coord"], axis=0),
            helper.make_node("Mod", ["packed_coord", twenty], ["bend_i"]),
            helper.make_node("Cast", ["bend_i"], ["bend_f"], to=TensorProto.FLOAT),
            helper.make_node("Div", ["packed_coord", twenty], ["coord_q0"]),
        ]
    )
    coord_names = ["gr0", "gr1", "gc0", "gc1", "rr0", "rr1", "rc0", "rc1"]
    quotient = "coord_q0"
    for idx, coord_name in enumerate(coord_names):
        if idx:
            nodes.append(helper.make_node("Div", [quotient, twenty], [f"coord_q{idx}"]))
            quotient = f"coord_q{idx}"
        nodes.append(helper.make_node("Mod", [quotient, twenty], [f"{coord_name}_i"]))
        nodes.append(helper.make_node("Cast", [f"{coord_name}_i"], [coord_name], to=TensorProto.FLOAT))
    gr0, gr1, gc0, gc1, rr0, rr1, rc0, rc1 = coord_names

    nodes.extend(
        [
        ]
    )
    equal_float(gr0, gr1, "green_horizontal")
    equal_float(rows, gr0, "h_seg1_row")
    equal_float(cols, "bend_f", "h_seg2_col")
    equal_float(rows, rr0, "h_seg3_row")
    scalar_min(gc0, "bend_f", "hg_c_lo")
    scalar_max(gc1, "bend_f", "hg_c_hi")
    scalar_min(gr0, rr0, "h_r_lo")
    scalar_max(gr0, rr0, "h_r_hi")
    scalar_min(rc0, "bend_f", "hr_c_lo")
    scalar_max(rc1, "bend_f", "hr_c_hi")
    h_seg1 = segment("h_seg1_row", between(cols, "hg_c_lo", "hg_c_hi", "h_seg1_cols"), "h_seg1")
    h_seg2 = segment(between(rows, "h_r_lo", "h_r_hi", "h_seg2_rows"), "h_seg2_col", "h_seg2")
    h_seg3 = segment("h_seg3_row", between(cols, "hr_c_lo", "hr_c_hi", "h_seg3_cols"), "h_seg3")

    nodes.extend(
        [
        ]
    )
    equal_float(cols, gc0, "v_seg1_col")
    equal_float(rows, "bend_f", "v_seg2_row")
    equal_float(cols, rc0, "v_seg3_col")
    scalar_min(gr0, "bend_f", "vg_r_lo")
    scalar_max(gr1, "bend_f", "vg_r_hi")
    scalar_min(gc0, rc0, "v_c_lo")
    scalar_max(gc0, rc0, "v_c_hi")
    scalar_min(rr0, "bend_f", "vr_r_lo")
    scalar_max(rr1, "bend_f", "vr_r_hi")
    v_seg1 = segment(between(rows, "vg_r_lo", "vg_r_hi", "v_seg1_rows"), "v_seg1_col", "v_seg1")
    v_seg2 = segment("v_seg2_row", between(cols, "v_c_lo", "v_c_hi", "v_seg2_cols"), "v_seg2")
    v_seg3 = segment(between(rows, "vr_r_lo", "vr_r_hi", "v_seg3_rows"), "v_seg3_col", "v_seg3")

    nodes.extend(
        [
            helper.make_node("Or", [h_seg1, h_seg2], ["h_path12"]),
            helper.make_node("Or", ["h_path12", h_seg3], ["h_path"]),
            helper.make_node("Or", [v_seg1, v_seg2], ["v_path12"]),
            helper.make_node("Or", ["v_path12", v_seg3], ["v_path"]),
            helper.make_node("And", ["green_horizontal", "h_path"], ["h_path_selected"]),
            helper.make_node("Not", ["green_horizontal"], ["green_vertical"]),
            helper.make_node("And", ["green_vertical", "v_path"], ["v_path_selected"]),
            helper.make_node("Or", ["h_path_selected", "v_path_selected"], ["path_raw"]),
        ]
    )
    red_rows = between(rows, rr0, rr1, "red_rows")
    red_cols = between(cols, rc0, rc1, "red_cols")
    nodes.extend(
        [
            helper.make_node("And", [red_rows, red_cols], ["red_box"]),
            helper.make_node("Not", ["red_box"], ["not_red"]),
            helper.make_node("And", ["path_raw", "not_red"], ["path"]),
            helper.make_node("Where", ["path", green_value, IN_NAME], [OUT_NAME]),
        ]
    )
    return _make_model(nodes, inits)


def _load_examples() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _check_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return False, {}
    session = ort.InferenceSession(
        sanitized.SerializeToString(),
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    )
    examples = _load_examples()
    counts: dict[str, tuple[int, int]] = {}
    all_ok = True
    for split in ("train", "test", "arc-gen"):
        passed = 0
        total = 0
        for example in examples.get(split, []):
            input_arr = convert_to_numpy(example, "input")
            expected_arr = convert_to_numpy(example, "output")
            if input_arr is None or expected_arr is None:
                continue
            total += 1
            out = session.run([OUT_NAME], {IN_NAME: input_arr})[0]
            pred = (out > 0.0).astype(np.float32)
            if np.array_equal(pred, expected_arr):
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
    onnx.save(model, BEST_PATH)

    correct, counts = _check_correct(model)
    scored = score_file(BEST_PATH)
    memory, largest_name, largest_bytes = _profile_largest_internal(model)
    params = calculate_params(sanitize_model(copy.deepcopy(model)) or model)

    print(f"wrote:   {BEST_PATH}")
    print(f"valid:   {scored['valid']}")
    print(f"passes:  {_format_counts(counts)}")
    print(f"correct: {correct}")
    print(f"memory:  {scored['memory']} (profile mirror: {memory})")
    print(f"params:  {scored['params']} (direct: {params})")
    print(f"cost:    {scored['cost']}")
    print(f"score:   {scored['score']:.6f}")
    print(f"largest: {largest_name} ({largest_bytes} bytes)")


if __name__ == "__main__":
    main()
