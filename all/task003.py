"""Minimal ONNX for ARC task003 using vertical period extension.

Task rule: the 6x3 input contains a vertically periodic foreground pattern on
black.  Infer the shortest period visible in the six rows (period 2, 3, or 4),
continue that row sequence to a 9x3 output, and recolor every foreground cell
from blue/non-black to red while keeping black background cells black.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, List

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

TASK_ID = "task003"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task003.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
OH = 9
OW = 3
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _i32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int32), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _make_model(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], name: str) -> onnx.ModelProto:
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


def _all_true(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], name: str, eq_name: str, count: int) -> str:
    shape = _i64(inits, [count], f"{name}_shape")
    flat = f"{name}_flat"
    parts = [f"{name}_bit{i}" for i in range(count)]
    nodes.append(helper.make_node("Reshape", [eq_name, shape], [flat]))
    nodes.append(helper.make_node("Split", [flat], parts, axis=0, split=[1] * count))
    current = parts[0]
    for idx, part in enumerate(parts[1:], start=1):
        out = f"{name}_and{idx}"
        nodes.append(helper.make_node("And", [current, part], [out]))
        current = out
    return current


def _all_true_reduce_min(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], name: str, eq_name: str) -> str:
    one = _init(inits, np.asarray(1, dtype=np.uint8), f"{name}_one")
    cast = f"{name}_u8"
    reduced = f"{name}_min"
    out = f"{name}_ok"
    nodes.extend(
        [
            helper.make_node("Cast", [eq_name], [cast], to=TensorProto.UINT8),
            helper.make_node("ReduceMin", [cast], [reduced], axes=[0, 1, 2, 3], keepdims=0),
            helper.make_node("Equal", [reduced, one], [out]),
        ]
    )
    return out


def _row_split_conditions(
    nodes: List[onnx.NodeProto],
    fg: str,
) -> tuple[dict[str, str], list[str]]:
    row_names = [f"row{i}" for i in range(6)]
    nodes.append(helper.make_node("Split", [fg], row_names, axis=2, split=[1] * 6))

    def period_condition(label: str, pairs: list[tuple[int, int]]) -> str:
        eq_names: list[str] = []
        for idx, (left, right) in enumerate(pairs):
            eq_name = f"{label}_eq{idx}"
            nodes.append(helper.make_node("Equal", [row_names[left], row_names[right]], [eq_name]))
            eq_names.append(eq_name)

        current = eq_names[0]
        for idx, eq_name in enumerate(eq_names[1:], start=1):
            out = f"{label}_row_and{idx}"
            nodes.append(helper.make_node("And", [current, eq_name], [out]))
            current = out

        col_parts = [f"{label}_col{i}" for i in range(3)]
        nodes.append(helper.make_node("Split", [current], col_parts, axis=3, split=[1] * 3))
        col01 = f"{label}_col01"
        cond4d = f"{label}_cond4d"
        cond = f"{label}_cond"
        nodes.extend(
            [
                helper.make_node("And", [col_parts[0], col_parts[1]], [col01]),
                helper.make_node("And", [col01, col_parts[2]], [cond4d]),
                helper.make_node("Squeeze", [cond4d], [cond], axes=[0, 1, 2]),
            ]
        )
        return cond

    return (
        {
            "p2": period_condition("p2", [(0, 2), (1, 3), (2, 4), (3, 5)]),
            "p3": period_condition("p3", [(0, 3), (1, 4), (2, 5)]),
        },
        row_names,
    )


def _compare_period_nodes(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    fg: str,
    *,
    compare_style: str,
    reduce_style: str,
) -> dict[str, str]:
    conds: dict[str, str] = {}
    if compare_style == "gather":
        specs = (
            ("p2", _i64(inits, [0, 1, 2, 3], "p2_a"), _i64(inits, [2, 3, 4, 5], "p2_b"), 12),
            ("p3", _i64(inits, [0, 1, 2], "p3_a"), _i64(inits, [3, 4, 5], "p3_b"), 9),
        )
        for label, a_idx, b_idx, total in specs:
            nodes.extend(
                [
                    helper.make_node("Gather", [fg, a_idx], [f"{label}_left"], axis=2),
                    helper.make_node("Gather", [fg, b_idx], [f"{label}_right"], axis=2),
                    helper.make_node("Equal", [f"{label}_left", f"{label}_right"], [f"{label}_eq"]),
                ]
            )
            conds[label] = (
                _all_true(nodes, inits, f"{label}_ok", f"{label}_eq", total)
                if reduce_style == "split"
                else _all_true_reduce_min(nodes, inits, f"{label}_ok", f"{label}_eq")
            )
    elif compare_style == "slice":
        axis_row = _i64(inits, [2], "axis_row")
        row_bounds = {
            "r0": _i64(inits, [0], "r0"),
            "r2": _i64(inits, [2], "r2"),
            "r3": _i64(inits, [3], "r3"),
            "r4": _i64(inits, [4], "r4"),
            "r6": _i64(inits, [6], "r6"),
        }
        specs = (
            ("p2", "r0", "r4", "r2", "r6", 12),
            ("p3", "r0", "r3", "r3", "r6", 9),
        )
        for label, a_start, a_end, b_start, b_end, total in specs:
            nodes.extend(
                [
                    helper.make_node(
                        "Slice",
                        [fg, row_bounds[a_start], row_bounds[a_end], axis_row],
                        [f"{label}_left"],
                    ),
                    helper.make_node(
                        "Slice",
                        [fg, row_bounds[b_start], row_bounds[b_end], axis_row],
                        [f"{label}_right"],
                    ),
                    helper.make_node("Equal", [f"{label}_left", f"{label}_right"], [f"{label}_eq"]),
                ]
            )
            conds[label] = (
                _all_true(nodes, inits, f"{label}_ok", f"{label}_eq", total)
                if reduce_style == "split"
                else _all_true_reduce_min(nodes, inits, f"{label}_ok", f"{label}_eq")
            )
    elif compare_style == "row_split":
        conds, _row_names = _row_split_conditions(nodes, fg)
    else:
        raise ValueError(compare_style)
    return conds


def _period_nodes(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    fg: str,
    *,
    compare_style: str = "gather",
    reduce_style: str = "split",
    index_dtype: str = "i64",
    index_style: str = "lookup",
) -> str:
    """Return a length-9 row index vector for the shortest period in fg."""
    make_idx = _i32 if index_dtype == "i32" else _i64

    conds = _compare_period_nodes(
        nodes,
        inits,
        fg,
        compare_style=compare_style,
        reduce_style=reduce_style,
    )

    if index_style == "lookup":
        idx2 = make_idx(inits, [0, 1, 0, 1, 0, 1, 0, 1, 0], "idx2")
        idx3 = make_idx(inits, [0, 1, 2, 0, 1, 2, 0, 1, 2], "idx3")
        idx4 = make_idx(inits, [0, 1, 2, 3, 0, 1, 2, 3, 0], "idx4")
        nodes.extend(
            [
                helper.make_node("Where", [conds["p3"], idx3, idx4], ["idx34"]),
                helper.make_node("Where", [conds["p2"], idx2, "idx34"], ["row_idx"]),
            ]
        )
    elif index_style == "mod":
        base = make_idx(inits, np.arange(OH, dtype=np.int32), "idx_base")
        div2 = make_idx(inits, [2], "div2")
        div3 = make_idx(inits, [3], "div3")
        div4 = make_idx(inits, [4], "div4")
        nodes.extend(
            [
                helper.make_node("Where", [conds["p3"], div3, div4], ["div34"]),
                helper.make_node("Where", [conds["p2"], div2, "div34"], ["period_div"]),
                helper.make_node("Mod", [base, "period_div"], ["row_idx"], fmod=0),
            ]
        )
    else:
        raise ValueError(index_style)
    return "row_idx"


def build_model(
    *,
    detect_nonblack: bool,
    output_style: str,
    compare_style: str = "gather",
    reduce_style: str = "split",
    index_dtype: str = "i64",
    index_style: str = "lookup",
    motif_style: str = "gather",
) -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    starts = _i64(inits, [0, 1 if detect_nonblack else 1, 0, 0], "starts")
    ends = _i64(inits, [1, C if detect_nonblack else 2, 6, OW], "ends")

    nodes.append(helper.make_node("Slice", [IN_NAME, starts, ends, axes4], ["raw_fg"]))
    fg_float = "raw_fg"
    if detect_nonblack:
        nodes.append(helper.make_node("ReduceMax", ["raw_fg"], ["fg"], axes=[1], keepdims=1))
        fg_float = "fg"
    nodes.append(helper.make_node("Cast", [fg_float], ["fgb6"], to=TensorProto.BOOL))
    fg = "fgb6"

    if motif_style == "gather":
        row_idx = _period_nodes(
            nodes,
            inits,
            fg,
            compare_style=compare_style,
            reduce_style=reduce_style,
            index_dtype=index_dtype,
            index_style=index_style,
        )
        motif_source = fg_float if output_style in {"concat_float", "concat3_float"} else fg
        nodes.append(helper.make_node("Gather", [motif_source, row_idx], ["motif"], axis=2))
    elif motif_style == "row_extend":
        conds, rows = _row_split_conditions(nodes, fg)
        nodes.extend(
            [
                helper.make_node("Not", [conds["p2"]], ["not_p2"]),
                helper.make_node("Not", [conds["p3"]], ["not_p3"]),
            ]
        )

        def mux(cond: str, not_cond: str, a: str, b: str, prefix: str) -> str:
            left = f"{prefix}_left"
            right = f"{prefix}_right"
            out = f"{prefix}_out"
            nodes.extend(
                [
                    helper.make_node("And", [cond, a], [left]),
                    helper.make_node("And", [not_cond, b], [right]),
                    helper.make_node("Or", [left, right], [out]),
                ]
            )
            return out

        row6_p34 = mux(conds["p3"], "not_p3", rows[0], rows[2], "row6_p34")
        row6 = mux(conds["p2"], "not_p2", rows[0], row6_p34, "row6")
        row7_p34 = mux(conds["p3"], "not_p3", rows[1], rows[3], "row7_p34")
        row7 = mux(conds["p2"], "not_p2", rows[1], row7_p34, "row7")
        row8_p34 = mux(conds["p3"], "not_p3", rows[2], rows[0], "row8_p34")
        row8 = mux(conds["p2"], "not_p2", rows[0], row8_p34, "row8")
        nodes.extend(
            [
                helper.make_node(
                    "Concat",
                    [rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], row6, row7, row8],
                    ["motif"],
                    axis=2,
                ),
            ]
        )
    else:
        raise ValueError(motif_style)

    if output_style == "onehot":
        channels = _i64(inits, np.arange(C).reshape(1, C, 1, 1), "channels")
        zero_i = _i64(inits, [0], "zero_i")
        two_i = _i64(inits, [2], "two_i")
        nodes.extend(
            [
                helper.make_node("Where", ["motif", two_i, zero_i], ["color_grid"]),
                helper.make_node("Equal", [channels, "color_grid"], ["onehot"]),
                helper.make_node("Cast", ["onehot"], ["out9"], to=TensorProto.FLOAT),
            ]
        )
    elif output_style in {"concat", "concat3"}:
        zero1 = _f32(inits, np.zeros((1, 1, OH, OW), dtype=np.float32), "zero1")
        if output_style == "concat":
            zero7 = _f32(inits, np.zeros((1, 7, OH, OW), dtype=np.float32), "zero7")
        nodes.extend(
            [
                helper.make_node("Not", ["motif"], ["bgb"]),
                helper.make_node("Cast", ["bgb"], ["bg"], to=TensorProto.FLOAT),
                helper.make_node("Cast", ["motif"], ["red"], to=TensorProto.FLOAT),
            ]
        )
        if output_style == "concat":
            nodes.append(helper.make_node("Concat", ["bg", zero1, "red", zero7], ["out9"], axis=1))
        else:
            nodes.append(helper.make_node("Concat", ["bg", zero1, "red"], ["out9"], axis=1))
    elif output_style in {"concat_float", "concat3_float"}:
        zero1 = _f32(inits, np.zeros((1, 1, OH, OW), dtype=np.float32), "zero1")
        one = _f32(inits, [1.0], "one")
        if output_style == "concat_float":
            zero7 = _f32(inits, np.zeros((1, 7, OH, OW), dtype=np.float32), "zero7")
        nodes.append(helper.make_node("Sub", [one, "motif"], ["bg"]))
        if output_style == "concat_float":
            nodes.append(helper.make_node("Concat", ["bg", zero1, "motif", zero7], ["out9"], axis=1))
        else:
            nodes.append(helper.make_node("Concat", ["bg", zero1, "motif"], ["out9"], axis=1))
    elif output_style == "bool3_cast":
        nodes.extend(
            [
                helper.make_node("Not", ["motif"], ["bgb"]),
                helper.make_node("And", ["motif", "bgb"], ["zerob"]),
                helper.make_node("Concat", ["bgb", "zerob", "motif"], ["out9_bool"], axis=1),
                helper.make_node("Cast", ["out9_bool"], ["out9"], to=TensorProto.FLOAT),
            ]
        )
    else:
        raise ValueError(output_style)

    nodes.append(
        helper.make_node(
            "Pad",
            ["out9"],
            [OUT_NAME],
            mode="constant",
            pads=[
                0,
                0,
                0,
                0,
                0,
                7 if output_style in {"concat3", "concat3_float", "bool3_cast"} else 0,
                H - OH,
                W - OW,
            ],
        )
    )
    return _make_model(nodes, inits, f"{TASK_ID}_{'nonblack' if detect_nonblack else 'blue'}_{output_style}")


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
    params = calculate_params(sanitized)
    return memory, largest_name, largest_bytes


def _format_counts(counts: dict[str, tuple[int, int]]) -> str:
    return ", ".join(f"{split} {ok}/{total}" for split, (ok, total) in counts.items())


def main() -> None:
    variants: Iterable[tuple[str, onnx.ModelProto]] = (
        ("blue_onehot", build_model(detect_nonblack=False, output_style="onehot")),
        ("blue_concat", build_model(detect_nonblack=False, output_style="concat")),
        ("blue_concat3", build_model(detect_nonblack=False, output_style="concat3")),
        ("blue_concat3f", build_model(detect_nonblack=False, output_style="concat3_float")),
        (
            "blue_bool3",
            build_model(detect_nonblack=False, output_style="bool3_cast"),
        ),
        (
            "blue_bool3_i32",
            build_model(detect_nonblack=False, output_style="bool3_cast", index_dtype="i32"),
        ),
        (
            "blue_bool3_slice_i32",
            build_model(
                detect_nonblack=False,
                output_style="bool3_cast",
                compare_style="slice",
                index_dtype="i32",
            ),
        ),
        (
            "blue_bool3_mod_i32",
            build_model(
                detect_nonblack=False,
                output_style="bool3_cast",
                index_dtype="i32",
                index_style="mod",
            ),
        ),
        (
            "blue_bool3_slice_mod_i32",
            build_model(
                detect_nonblack=False,
                output_style="bool3_cast",
                compare_style="slice",
                index_dtype="i32",
                index_style="mod",
            ),
        ),
        (
            "blue_bool3_rows_mod_i32",
            build_model(
                detect_nonblack=False,
                output_style="bool3_cast",
                compare_style="row_split",
                index_dtype="i32",
                index_style="mod",
            ),
        ),
        (
            "blue_bool3_row_extend",
            build_model(
                detect_nonblack=False,
                output_style="bool3_cast",
                motif_style="row_extend",
            ),
        ),
        (
            "blue_bool3_i32_reduce",
            build_model(
                detect_nonblack=False,
                output_style="bool3_cast",
                reduce_style="reduce_min",
                index_dtype="i32",
            ),
        ),
        (
            "blue_bool3_slice_i32_reduce",
            build_model(
                detect_nonblack=False,
                output_style="bool3_cast",
                compare_style="slice",
                reduce_style="reduce_min",
                index_dtype="i32",
            ),
        ),
        (
            "blue_concat3f_i32",
            build_model(detect_nonblack=False, output_style="concat3_float", index_dtype="i32"),
        ),
        (
            "blue_concat3f_slice_i32",
            build_model(
                detect_nonblack=False,
                output_style="concat3_float",
                compare_style="slice",
                index_dtype="i32",
            ),
        ),
        (
            "blue_concat3f_slice_mod_i32",
            build_model(
                detect_nonblack=False,
                output_style="concat3_float",
                compare_style="slice",
                index_dtype="i32",
                index_style="mod",
            ),
        ),
        (
            "blue_concat3f_rows_mod_i32",
            build_model(
                detect_nonblack=False,
                output_style="concat3_float",
                compare_style="row_split",
                index_dtype="i32",
                index_style="mod",
            ),
        ),
        (
            "blue_concat3f_slice_i32_reduce",
            build_model(
                detect_nonblack=False,
                output_style="concat3_float",
                compare_style="slice",
                reduce_style="reduce_min",
                index_dtype="i32",
            ),
        ),
        ("nonblack_onehot", build_model(detect_nonblack=True, output_style="onehot")),
        ("nonblack_concat", build_model(detect_nonblack=True, output_style="concat")),
        ("nonblack_concat3", build_model(detect_nonblack=True, output_style="concat3")),
        ("nonblack_concat3f", build_model(detect_nonblack=True, output_style="concat3_float")),
    )

    results: list[tuple[int, str, onnx.ModelProto, dict[str, Any], str | None, int | None, dict[str, tuple[int, int]]]] = []
    for label, model in variants:
        tmp_path = OUT_DIR / f"{TASK_ID}_{label}.onnx"
        onnx.save(model, tmp_path)
        try:
            correct, counts = _check_correct(model)
            scored = score_file(tmp_path)
            memory, largest_name, largest_bytes = _profile_largest_internal(model)
            print(
                f"{label:<32} correct={correct} ({_format_counts(counts)}) "
                f"memory={scored['memory']} params={scored['params']} cost={scored['cost']} "
                f"score={scored['score']:.6f} largest={largest_name}:{largest_bytes}"
            )
            if correct and scored["valid"]:
                results.append((int(scored["cost"]), label, model, scored, largest_name, largest_bytes, counts))
        except Exception as exc:
            print(f"{label:<32} invalid/error={type(exc).__name__}: {exc}")
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
