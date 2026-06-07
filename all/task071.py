"""ONNX solver for ARC task071 using rectangle removal and mirror repair.

Task rule: the active 16x16 grid contains one left-right symmetric colored
pattern plus one intrusive solid rectangle of another color.  Remove the
rectangle color, infer the pattern color as the other non-background color, and
repair the occluded cells by mirroring visible pattern cells across the
pattern's vertical symmetry axis.  The graph tests all possible vertical mirror
axes and accepts only mirrored cells whose new positions lie inside the removed
rectangle.
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

TASK_ID = "task071"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task071.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
ACTIVE = 16
H = W = 30
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


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _i32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int32), name)


def _bool(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.bool_), name)


def _make_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        name,
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


def _or_all(nodes: list[onnx.NodeProto], names: list[str], prefix: str) -> str:
    current = names[0]
    for idx, name in enumerate(names[1:], start=1):
        out = f"{prefix}_{idx}"
        nodes.append(helper.make_node("Or", [current, name], [out]))
        current = out
    return current


def _build_symmetry_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    starts = _i64(inits, [0, 1, 0, 0], "starts")
    ends = _i64(inits, [1, C, ACTIVE, ACTIVE], "ends")
    zero_f = _f32(inits, [0.0], "zero_f")
    zero_i = _i32(inits, [0], "zero_i")
    four_i = _i32(inits, [4], "four_i")
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts, ends, axes4], ["nonbg"]),
            helper.make_node("Cast", ["nonbg"], ["nonbgb"], to=TensorProto.BOOL),
            helper.make_node("ReduceSum", ["nonbg"], ["counts"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceSum", ["nonbg"], ["row_counts"], axes=[3], keepdims=1),
            helper.make_node("ReduceSum", ["nonbg"], ["col_counts"], axes=[2], keepdims=1),
            helper.make_node("Cast", ["counts"], ["counts_i"], to=TensorProto.INT32),
            helper.make_node("Cast", ["row_counts"], ["row_counts_i"], to=TensorProto.INT32),
            helper.make_node("Cast", ["col_counts"], ["col_counts_i"], to=TensorProto.INT32),
            helper.make_node("Greater", ["counts", zero_f], ["present"]),
            helper.make_node("Equal", ["row_counts_i", four_i], ["row4"]),
            helper.make_node("Cast", ["row4"], ["row4_i"], to=TensorProto.INT32),
            helper.make_node("ReduceSum", ["row4_i"], ["row_total"], axes=[2, 3], keepdims=1),
        ]
    )

    rect_tests: list[str] = []
    for height, area in ((3, 12), (4, 16), (5, 20)):
        area_name = _i32(inits, [area], f"area{height}")
        height_name = _i32(inits, [height], f"height{height}")
        nodes.extend(
            [
                helper.make_node("Equal", ["counts_i", area_name], [f"rect{height}_count"]),
                helper.make_node("Equal", ["row_total", height_name], [f"rect{height}_rows"]),
                helper.make_node("Equal", ["col_counts_i", height_name], [f"rect{height}_colh"]),
                helper.make_node("Cast", [f"rect{height}_colh"], [f"rect{height}_colh_i"], to=TensorProto.INT32),
                helper.make_node("ReduceSum", [f"rect{height}_colh_i"], [f"rect{height}_col_total"], axes=[2, 3], keepdims=1),
                helper.make_node("Equal", [f"rect{height}_col_total", four_i], [f"rect{height}_cols"]),
                helper.make_node("And", [f"rect{height}_count", f"rect{height}_rows"], [f"rect{height}_count_rows"]),
                helper.make_node("And", [f"rect{height}_count_rows", f"rect{height}_cols"], [f"rect{height}_ok"]),
            ]
        )
        rect_tests.append(f"rect{height}_ok")

    rect_color = _or_all(nodes, rect_tests, "rect_color")
    nodes.extend(
        [
            helper.make_node("Not", [rect_color], ["not_rect_color"]),
            helper.make_node("And", ["present", "not_rect_color"], ["pattern_color"]),
            helper.make_node("Cast", ["pattern_color"], ["pattern_color_i"], to=TensorProto.INT32),
            helper.make_node("Cast", [rect_color], ["rect_color_i"], to=TensorProto.INT32),
            helper.make_node("ArgMax", ["pattern_color_i"], ["pattern_idx3"], axis=1, keepdims=0),
            helper.make_node("ArgMax", ["rect_color_i"], ["rect_idx3"], axis=1, keepdims=0),
            helper.make_node("Squeeze", ["pattern_idx3"], ["pattern_idx"], axes=[0, 1, 2]),
            helper.make_node("Squeeze", ["rect_idx3"], ["rect_idx"], axes=[0, 1, 2]),
            helper.make_node("Gather", ["nonbgb", "pattern_idx"], ["pattern3"], axis=1),
            helper.make_node("Gather", ["nonbgb", "rect_idx"], ["rect3"], axis=1),
            helper.make_node("Unsqueeze", ["pattern3"], ["pattern"], axes=[1]),
            helper.make_node("Unsqueeze", ["rect3"], ["rect"], axes=[1]),
            helper.make_node("Or", ["pattern", "rect"], ["allowed"]),
            helper.make_node("Not", ["allowed"], ["not_allowed"]),
        ]
    )

    final = "pattern"
    for axis2 in range(9, 21):
        gather_idx = []
        for col in range(ACTIVE):
            source = axis2 - col
            gather_idx.append(source if 0 <= source < ACTIVE else 0)
        idx_name = _i64(inits, gather_idx, f"axis{axis2}_idx")
        nodes.extend(
            [
                helper.make_node("Gather", ["pattern", idx_name], [f"axis{axis2}_raw"], axis=3),
                helper.make_node("And", [f"axis{axis2}_raw", "not_allowed"], [f"axis{axis2}_bad"]),
                helper.make_node("Cast", [f"axis{axis2}_bad"], [f"axis{axis2}_bad_i"], to=TensorProto.INT32),
                helper.make_node("ReduceSum", [f"axis{axis2}_bad_i"], [f"axis{axis2}_bad_count"], axes=[0, 1, 2, 3], keepdims=1),
                helper.make_node("Equal", [f"axis{axis2}_bad_count", zero_i], [f"axis{axis2}_ok"]),
                helper.make_node("And", [f"axis{axis2}_raw", f"axis{axis2}_ok"], [f"axis{axis2}_accepted"]),
                helper.make_node("Or", [final, f"axis{axis2}_accepted"], [f"final_{axis2}"]),
            ]
        )
        final = f"final_{axis2}"

    pads = [0, 0, 0, 0, 0, 0, H - ACTIVE, W - ACTIVE]
    nodes.extend(
        [
            helper.make_node("Not", [final], ["background16"]),
            helper.make_node("And", [final, "pattern_color"], ["out_nonbg16"]),
            helper.make_node("Concat", ["background16", "out_nonbg16"], ["out_bool16"], axis=1),
            helper.make_node("Cast", ["out_bool16"], ["out_float16"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out_float16"], [OUT_NAME], mode="constant", pads=pads),
        ]
    )
    return _make_model(nodes, inits, f"{TASK_ID}_symmetry")


def _load_json_examples() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _check_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return False, {}
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        sanitized.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    examples = _load_json_examples()
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
            output = session.run([OUT_NAME], {IN_NAME: input_arr})[0]
            if np.array_equal((output > 0.0).astype(np.float32), expected_arr):
                passed += 1
            else:
                all_ok = False
        counts[split] = (passed, total)
    return all_ok, counts


def _measure_model(model: onnx.ModelProto, path: Path) -> dict[str, Any]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return {"valid": False, "error": "sanitize failed"}
    inputs = load_task_examples(path)
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(Path(tempfile.gettempdir()) / f"ng_{TASK_ID}")
    session = ort.InferenceSession(
        sanitized.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    for arr in inputs:
        session.run([OUT_NAME], {IN_NAME: arr})
    trace_path = session.end_profiling()
    memory = calculate_memory(sanitized, trace_path)
    params = calculate_params(sanitized)
    cost = None if memory is None or params is None else memory + params
    score = None if cost is None else max(1.0, 25.0 - math.log(max(1.0, float(cost))))
    return {"valid": memory is not None and params is not None, "memory": memory, "params": params, "cost": cost, "score": score}


def main() -> None:
    variants = {"symmetry_rowcol_rect_axes_9_20": _build_symmetry_model()}
    results: list[tuple[str, onnx.ModelProto, bool, dict[str, tuple[int, int]], dict[str, Any]]] = []
    for name, model in variants.items():
        ok, counts = _check_correct(model)
        metrics = _measure_model(model, BEST_PATH)
        results.append((name, model, ok, counts, metrics))
        score_text = "INVALID" if metrics.get("score") is None else f"{metrics['score']:.6f}"
        print(
            f"{name}: correct={ok} splits={counts} "
            f"memory={metrics.get('memory')} params={metrics.get('params')} "
            f"cost={metrics.get('cost')} score={score_text}"
        )

    valid = [item for item in results if item[2] and item[4].get("valid")]
    if not valid:
        raise SystemExit("no correct measurable variant")
    best = min(valid, key=lambda item: int(item[4]["cost"]))
    onnx.save(best[1], BEST_PATH)
    report = score_file(BEST_PATH)
    print(f"saved {BEST_PATH}")
    print(
        f"best={best[0]} memory={report['memory']} params={report['params']} "
        f"cost={report['cost']} score={report['score']:.6f}"
    )


if __name__ == "__main__":
    main()
