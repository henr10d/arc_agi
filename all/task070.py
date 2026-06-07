"""Minimal ONNX for ARC task070: fill the cyan object's bounding box.

Task rule: inputs are 17x17 grids containing only background/dark-blue pattern
cells plus cyan (8) cells embedded in one rectangular region.  Find the tight
bounding box of all cyan cells.  Inside that box, keep cyan cells cyan and turn
every other cell green (3); outside the box, preserve the original grid.

ONNX: work on a 17x17 cyan mask, compute its bbox with ReduceMax/ArgMax and
coordinate comparisons, then use a padded single-channel fill mask to overwrite
the original input with a broadcast green one-hot vector only where needed.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import (  # noqa: E402
    calculate_memory,
    convert_to_numpy,
    sanitize_model,
    score_file,
)

TASK_ID = "task070"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task070.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
G = 17
H = W = 30
PAD = H - G
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _make_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(nodes, name, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _bbox_from_cyan(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    cyan_f: str,
    cyan_b: str,
    *,
    size: int = G,
) -> tuple[str, str]:
    rev = _i64(inits, np.arange(size - 1, -1, -1), "rev")
    last = _i64(inits, [size - 1], "last")
    rows = _i64(inits, np.arange(size).reshape(1, 1, size, 1), "rows")
    cols = _i64(inits, np.arange(size).reshape(1, 1, 1, size), "cols")

    nodes.extend(
        [
            helper.make_node("ReduceMax", [cyan_f], ["row_has"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", [cyan_f], ["col_has"], axes=[2], keepdims=1),
            helper.make_node("ArgMax", ["row_has"], ["rmin"], axis=2, keepdims=1),
            helper.make_node("ArgMax", ["col_has"], ["cmin"], axis=3, keepdims=1),
            helper.make_node("Gather", ["row_has", rev], ["row_rev"], axis=2),
            helper.make_node("Gather", ["col_has", rev], ["col_rev"], axis=3),
            helper.make_node("ArgMax", ["row_rev"], ["rrev"], axis=2, keepdims=1),
            helper.make_node("ArgMax", ["col_rev"], ["crev"], axis=3, keepdims=1),
            helper.make_node("Sub", [last, "rrev"], ["rmax"]),
            helper.make_node("Sub", [last, "crev"], ["cmax"]),
            helper.make_node("Less", [rows, "rmin"], ["r_before"]),
            helper.make_node("Less", ["rmax", rows], ["r_after"]),
            helper.make_node("Or", ["r_before", "r_after"], ["r_out"]),
            helper.make_node("Not", ["r_out"], ["r_in"]),
            helper.make_node("Less", [cols, "cmin"], ["c_before"]),
            helper.make_node("Less", ["cmax", cols], ["c_after"]),
            helper.make_node("Or", ["c_before", "c_after"], ["c_out"]),
            helper.make_node("Not", ["c_out"], ["c_in"]),
            helper.make_node("And", ["r_in", "c_in"], ["bbox"]),
            helper.make_node("Not", [cyan_b], ["not_cyan"]),
            helper.make_node("And", ["bbox", "not_cyan"], ["fill"]),
        ]
    )
    return "fill"


def _finalize_bool_planes(nodes: list[onnx.NodeProto], planes: list[str], name: str) -> onnx.ModelProto:
    nodes.extend(
        [
            helper.make_node("Concat", planes, ["out17b"], axis=1),
            helper.make_node("Cast", ["out17b"], ["out17"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out17"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, C - len(planes), PAD, PAD],
            ),
        ]
    )
    return name


def _finalize_overlay_green(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], fill: str) -> str:
    green = np.zeros((1, C, 1, 1), dtype=np.float32)
    green[0, 3, 0, 0] = 1.0
    green_name = _init(inits, green, "green")
    nodes.extend(
        [
            helper.make_node("Cast", [fill], ["fill_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["fill_f"],
                ["fill30_f"],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, PAD, PAD],
            ),
            helper.make_node("Cast", ["fill30_f"], ["fill30"], to=TensorProto.BOOL),
            helper.make_node("Where", ["fill30", green_name, IN_NAME], [OUT_NAME]),
        ]
    )
    return "overlay_green"


def build_overlay_green() -> onnx.ModelProto:
    """Lowest-memory candidate: preserve input and overwrite only green fill cells."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    ch8_st = _i64(inits, [0, 8, 0, 0], "ch8_st")
    ch8_en = _i64(inits, [1, 9, G, G], "ch8_en")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, ch8_st, ch8_en], ["cyan_f"]),
            helper.make_node("Cast", ["cyan_f"], ["cyan"], to=TensorProto.BOOL),
        ]
    )
    fill = _bbox_from_cyan(nodes, inits, "cyan_f", "cyan")
    _finalize_overlay_green(nodes, inits, fill)
    return _make_model(nodes, inits, "task070_overlay_green")


def build_overlay_green30() -> onnx.ModelProto:
    """Control candidate: compute the fill mask directly at full 30x30 size."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    ch8_st = _i64(inits, [0, 8, 0, 0], "ch8_st")
    ch8_en = _i64(inits, [1, 9, H, W], "ch8_en")
    green = np.zeros((1, C, 1, 1), dtype=np.float32)
    green[0, 3, 0, 0] = 1.0
    green_name = _init(inits, green, "green")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, ch8_st, ch8_en], ["cyan_f"]),
            helper.make_node("Cast", ["cyan_f"], ["cyan"], to=TensorProto.BOOL),
        ]
    )
    fill = _bbox_from_cyan(nodes, inits, "cyan_f", "cyan", size=H)
    nodes.append(helper.make_node("Where", [fill, green_name, IN_NAME], [OUT_NAME]))
    return _make_model(nodes, inits, "task070_overlay_green30")


def build_direct_derive0() -> onnx.ModelProto:
    """Best candidate: read only color-1 and color-8 planes; derive color 0."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    ch1_st = _i64(inits, [0, 1, 0, 0], "ch1_st")
    ch1_en = _i64(inits, [1, 2, G, G], "ch1_en")
    ch8_st = _i64(inits, [0, 8, 0, 0], "ch8_st")
    ch8_en = _i64(inits, [1, 9, G, G], "ch8_en")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, ch8_st, ch8_en], ["cyan_f"]),
            helper.make_node("Cast", ["cyan_f"], ["cyan"], to=TensorProto.BOOL),
        ]
    )
    fill = _bbox_from_cyan(nodes, inits, "cyan_f", "cyan")
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, ch1_st, ch1_en], ["blue_f"]),
            helper.make_node("Cast", ["blue_f"], ["blue0"], to=TensorProto.BOOL),
            helper.make_node("Not", ["bbox"], ["not_bbox"]),
            helper.make_node("And", ["bbox", "not_bbox"], ["z1"]),
            helper.make_node("And", ["blue0", "not_bbox"], ["blue"]),
            helper.make_node("Or", ["blue0", "bbox"], ["not_black"]),
            helper.make_node("Not", ["not_black"], ["black"]),
        ]
    )
    _finalize_bool_planes(
        nodes,
        ["black", "blue", "z1", fill, "z1", "z1", "z1", "z1", "cyan"],
        "direct_derive0",
    )
    return _make_model(nodes, inits, "task070_direct_derive0")


def build_direct_slice01() -> onnx.ModelProto:
    """Slightly larger candidate: preserve color 0 by slicing the input plane."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    ch0_st = _i64(inits, [0, 0, 0, 0], "ch0_st")
    ch0_en = _i64(inits, [1, 1, G, G], "ch0_en")
    ch1_st = _i64(inits, [0, 1, 0, 0], "ch1_st")
    ch1_en = _i64(inits, [1, 2, G, G], "ch1_en")
    ch8_st = _i64(inits, [0, 8, 0, 0], "ch8_st")
    ch8_en = _i64(inits, [1, 9, G, G], "ch8_en")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, ch8_st, ch8_en], ["cyan_f"]),
            helper.make_node("Cast", ["cyan_f"], ["cyan"], to=TensorProto.BOOL),
        ]
    )
    fill = _bbox_from_cyan(nodes, inits, "cyan_f", "cyan")
    nodes.extend(
        [
            helper.make_node("Not", [fill], ["not_fill"]),
            helper.make_node("Slice", [IN_NAME, ch0_st, ch0_en], ["black_f"]),
            helper.make_node("Slice", [IN_NAME, ch1_st, ch1_en], ["blue_f"]),
            helper.make_node("Cast", ["black_f"], ["black0"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["blue_f"], ["blue0"], to=TensorProto.BOOL),
            helper.make_node("And", [fill, "not_fill"], ["z1"]),
            helper.make_node("And", ["black0", "not_fill"], ["black"]),
            helper.make_node("And", ["blue0", "not_fill"], ["blue"]),
        ]
    )
    _finalize_bool_planes(
        nodes,
        ["black", "blue", "z1", fill, "z1", "z1", "z1", "z1", "cyan"],
        "direct_slice01",
    )
    return _make_model(nodes, inits, "task070_direct_slice01")


def build_ids_4plane() -> onnx.ModelProto:
    """Control candidate: ArgMax color ids, then decode only live planes."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    core_st = _i64(inits, [0, 0, 0, 0], "core_st")
    core_en = _i64(inits, [1, C, G, G], "core_en")
    ch8_st = _i64(inits, [0, 8, 0, 0], "ch8_st")
    ch8_en = _i64(inits, [1, 9, G, G], "ch8_en")
    v0 = _i64(inits, [0], "v0")
    v1 = _i64(inits, [1], "v1")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, core_st, core_en], ["core"]),
            helper.make_node("ArgMax", ["core"], ["ids"], axis=1, keepdims=1),
            helper.make_node("Slice", [IN_NAME, ch8_st, ch8_en], ["cyan_f"]),
            helper.make_node("Cast", ["cyan_f"], ["cyan"], to=TensorProto.BOOL),
        ]
    )
    fill = _bbox_from_cyan(nodes, inits, "cyan_f", "cyan")
    nodes.extend(
        [
            helper.make_node("Equal", ["ids", v0], ["black0"]),
            helper.make_node("Equal", ["ids", v1], ["blue0"]),
            helper.make_node("Not", [fill], ["not_fill"]),
            helper.make_node("And", [fill, "not_fill"], ["z1"]),
            helper.make_node("And", ["black0", "not_fill"], ["black"]),
            helper.make_node("And", ["blue0", "not_fill"], ["blue"]),
        ]
    )
    _finalize_bool_planes(
        nodes,
        ["black", "blue", "z1", fill, "z1", "z1", "z1", "z1", "cyan"],
        "ids_4plane",
    )
    return _make_model(nodes, inits, "task070_ids_4plane")


def _load_examples() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _check_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return False, {}
    session = ort.InferenceSession(sanitized.SerializeToString(), providers=["CPUExecutionProvider"])
    task = _load_examples()
    counts: dict[str, tuple[int, int]] = {}
    for split in ("train", "test", "arc-gen"):
        ok_count = 0
        examples = task.get(split, [])
        for idx, example in enumerate(examples):
            x = convert_to_numpy(example, "input")
            y = convert_to_numpy(example, "output")
            if x is None or y is None:
                continue
            pred = session.run([OUT_NAME], {IN_NAME: x})[0]
            if not np.array_equal(pred > 0.0, y > 0.0):
                return False, {**counts, split: (ok_count, len(examples))}
            ok_count += 1
        counts[split] = (ok_count, len(examples))
    return True, counts


def _profile_largest_internal(model: onnx.ModelProto) -> tuple[int | None, str | None, int | None]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return None, None, None
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(Path(tempfile.gettempdir()) / f"ng_{TASK_ID}")
    session = ort.InferenceSession(sanitized.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    for split_examples in _load_examples().values():
        for example in split_examples:
            x = convert_to_numpy(example, "input")
            if x is not None:
                session.run([OUT_NAME], {IN_NAME: x})
    trace_path = session.end_profiling()
    memory = calculate_memory(sanitized, trace_path)
    graph = onnx.shape_inference.infer_shapes(sanitized, strict_mode=True).graph
    largest_name: str | None = None
    largest_bytes = -1
    for value in graph.value_info:
        if value.name == OUT_NAME or not value.type.HasField("tensor_type"):
            continue
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        n = 1
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                n *= dim.dim_value
        itemsize = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)).itemsize
        size = int(n * itemsize)
        if size > largest_bytes:
            largest_name = value.name
            largest_bytes = size
    return memory, largest_name, largest_bytes


def _format_counts(counts: dict[str, tuple[int, int]]) -> str:
    return ", ".join(f"{split}={ok}/{total}" for split, (ok, total) in counts.items())


def main() -> None:
    builders: list[tuple[str, Callable[[], onnx.ModelProto]]] = [
        ("overlay_green", build_overlay_green),
        ("overlay_green30", build_overlay_green30),
        ("direct_derive0", build_direct_derive0),
        ("direct_slice01", build_direct_slice01),
        ("ids_4plane", build_ids_4plane),
    ]
    results: list[tuple[int, str, onnx.ModelProto, dict[str, Any], str | None, int | None, dict[str, tuple[int, int]]]] = []
    for label, builder in builders:
        model = builder()
        tmp_path = OUT_DIR / f"{TASK_ID}_{label}.onnx"
        onnx.save(model, tmp_path)
        correct, counts = _check_correct(model)
        scored = score_file(tmp_path)
        memory, largest_name, largest_bytes = _profile_largest_internal(model)
        score_text = f"{scored['score']:.6f}" if scored.get("score") is not None else "INVALID"
        print(
            f"{label:<16} correct={correct} ({_format_counts(counts)}) "
            f"memory={scored['memory']} params={scored['params']} cost={scored['cost']} "
            f"score={score_text} largest={largest_name}:{largest_bytes}"
        )
        if correct and scored["valid"]:
            results.append((int(scored["cost"]), label, model, scored, largest_name, largest_bytes, counts))
        tmp_path.unlink(missing_ok=True)

    if not results:
        raise SystemExit("no valid correct variants")

    _, label, model, scored, largest_name, largest_bytes, counts = min(results, key=lambda item: item[0])
    onnx.save(model, BEST_PATH)
    print()
    print(f"kept:    {label}")
    print(f"wrote:   {BEST_PATH}")
    print(f"valid:   {scored['valid']}")
    print(f"passes:  {_format_counts(counts)}")
    print(f"memory:  {scored['memory']}")
    print(f"params:  {scored['params']}")
    print(f"cost:    {scored['cost']}")
    print(f"score:   {scored['score']:.6f}")
    print(f"largest: {largest_name} ({largest_bytes} bytes)")


if __name__ == "__main__":
    main()
