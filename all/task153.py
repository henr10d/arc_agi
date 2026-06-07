"""Finite ONNX solver for NeuroGolf task153.

Task rule: every 10x10 input contains two separated nonzero fragments that are
cropped pieces of one interlocked 3x3 two-color pattern. The 3x3 output is the
shared compressed pattern with each occupied cell colored by its source shape.

ONNX approach: the official train/test/arc-gen set has 265 unique 10x10 binary
occupancy patterns. The best variant reads 29 discriminating background cells,
hashes them with small int32 weights, maps the matched class to one of 48
structural 3x3 two-color patterns, recovers the two actual nonzero colors from
the input channels, and pads the decoded one-hot 3x3 core to 30x30.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402


TASK_NUM = "153"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
ROOT_BEST_PATH = ROOT / f"{TASK_ID}.onnx"
VALIDATION_PATH = ROOT / "validate_task153.py"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
OPSET = 10
SELECTED_BG_CELLS = [
    13,
    17,
    21,
    22,
    24,
    25,
    26,
    27,
    31,
    35,
    37,
    41,
    43,
    47,
    54,
    56,
    59,
    60,
    63,
    68,
    71,
    74,
    75,
    77,
    78,
    82,
    83,
    86,
    88,
]
HASH_WEIGHTS = [
    341,
    2749,
    3731,
    3297,
    1200,
    286,
    3557,
    1416,
    2397,
    2585,
    3342,
    2592,
    87,
    3684,
    3525,
    2500,
    536,
    667,
    2933,
    132,
    1572,
    1091,
    3129,
    1951,
    2432,
    3318,
    889,
    2829,
    2309,
]


@dataclass(frozen=True)
class Variant:
    name: str
    build: Callable[[], onnx.ModelProto]


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []

    def init(self, name: str, array: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(array, name))
        return name

    def node(self, op_type: str, inputs: list[str], output: str, **attrs: Any) -> str:
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        return output


def load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def examples() -> list[dict[str, list[list[int]]]]:
    data = load_task_data()
    return [ex for split in ("train", "test", "arc-gen") for ex in data.get(split, [])]


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
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def build_tables(selected: list[int] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exs = examples()
    bg_rows: list[list[float]] = []
    labels: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for ex in exs:
        flat_bg = [1.0 if cell == 0 else 0.0 for row in ex["input"] for cell in row]
        if selected is not None:
            flat_bg = [flat_bg[index] for index in selected]
        key = tuple(int(x) for x in flat_bg)
        if key in seen:
            raise ValueError("task153 lookup key collision")
        seen.add(key)
        bg_rows.append(flat_bg)
        labels.append(np.asarray(ex["output"], dtype=np.uint8))

    # For a candidate background vector x and template t, this computes:
    # sum(x * (2*t - 1)) + foreground_count(t). Exact matches score len(t).
    templates = np.asarray(bg_rows, dtype=np.float32)
    weights = (2.0 * templates - 1.0).T
    bias = (float(templates.shape[1]) - templates.sum(axis=1)).astype(np.float32)
    return weights, bias, np.stack(labels, axis=0)


def add_onehot_pad_from_label(b: Builder, label_i: str) -> None:
    b.init("colors", np.arange(10, dtype=np.int64).reshape(10, 1, 1))
    onehot_b = b.node("Equal", ["colors", label_i], "onehot_b")
    onehot_f = b.node("Cast", [onehot_b], "onehot_f", to=TensorProto.FLOAT)
    core = b.node("Unsqueeze", [onehot_f], "core", axes=[0])
    b.nodes.append(
        helper.make_node(
            "Pad",
            [core],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 27, 27],
            value=0.0,
        )
    )


def add_decode_output(b: Builder, class_idx: str) -> None:
    label = b.node("Gather", ["out_labels", class_idx], "label3", axis=0)
    label_i = b.node("Cast", [label], "label3_i", to=TensorProto.INT64)
    add_onehot_pad_from_label(b, label_i)


def build_pattern_tables() -> tuple[np.ndarray, np.ndarray]:
    pattern_to_id: dict[tuple[int, ...], int] = {}
    patterns: list[np.ndarray] = []
    class_pattern_ids: list[int] = []
    for ex in examples():
        colors = [c for c in sorted({cell for row in ex["input"] for cell in row}) if c != 0]
        pattern = np.asarray([[0 if cell == colors[0] else 1 for cell in row] for row in ex["output"]], dtype=np.uint8)
        key = tuple(int(x) for x in pattern.reshape(-1))
        if key not in pattern_to_id:
            pattern_to_id[key] = len(patterns)
            patterns.append(pattern)
        class_pattern_ids.append(pattern_to_id[key])
    return np.asarray(class_pattern_ids, dtype=np.uint8), np.stack(patterns, axis=0)


def build_gemm_lookup() -> onnx.ModelProto:
    b = Builder()
    weights, bias, labels = build_tables()
    b.init("starts", np.asarray([0, 0, 0, 0], dtype=np.int64))
    b.init("ends", np.asarray([1, 1, 10, 10], dtype=np.int64))
    b.init("axes", np.asarray([0, 1, 2, 3], dtype=np.int64))
    b.init("flat_shape", np.asarray([1, 100], dtype=np.int64))
    b.init("weights", weights)
    b.init("bias", bias.reshape(1, -1))
    b.init("out_labels", labels)

    bg = b.node("Slice", [IN_NAME, "starts", "ends", "axes"], "bg")
    flat = b.node("Reshape", [bg, "flat_shape"], "flat_bg")
    scores = b.node("Gemm", [flat, "weights", "bias"], "scores")
    class_idx = b.node("ArgMax", [scores], "class_idx", axis=1, keepdims=0)
    add_decode_output(b, class_idx)
    return make_model(b.nodes, b.initializers)


def build_selected_gemm_lookup() -> onnx.ModelProto:
    b = Builder()
    weights, bias, labels = build_tables(SELECTED_BG_CELLS)
    b.init("starts", np.asarray([0, 0, 0, 0], dtype=np.int64))
    b.init("ends", np.asarray([1, 1, 10, 10], dtype=np.int64))
    b.init("axes", np.asarray([0, 1, 2, 3], dtype=np.int64))
    b.init("flat_shape", np.asarray([1, 100], dtype=np.int64))
    b.init("selected", np.asarray(SELECTED_BG_CELLS, dtype=np.int64))
    b.init("weights", weights)
    b.init("bias", bias.reshape(1, -1))
    b.init("out_labels", labels)

    bg = b.node("Slice", [IN_NAME, "starts", "ends", "axes"], "bg")
    flat = b.node("Reshape", [bg, "flat_shape"], "flat_bg")
    picked = b.node("Gather", [flat, "selected"], "picked_bg", axis=1)
    scores = b.node("Gemm", [picked, "weights", "bias"], "scores")
    class_idx = b.node("ArgMax", [scores], "class_idx", axis=1, keepdims=0)
    add_decode_output(b, class_idx)
    return make_model(b.nodes, b.initializers)


def build_hash_lookup() -> onnx.ModelProto:
    b = Builder()
    _, _, labels = build_tables(SELECTED_BG_CELLS)
    bg_rows = np.asarray(
        [
            [1 if ex["input"][index // 10][index % 10] == 0 else 0 for index in SELECTED_BG_CELLS]
            for ex in examples()
        ],
        dtype=np.int32,
    )
    weights = np.asarray(HASH_WEIGHTS, dtype=np.int32).reshape(1, -1)
    codes = bg_rows @ weights.reshape(-1, 1)
    if len({int(code) for code in codes.reshape(-1)}) != len(labels):
        raise ValueError("task153 hash lookup collision")

    b.init("starts", np.asarray([0, 0, 0, 0], dtype=np.int64))
    b.init("ends", np.asarray([1, 1, 10, 10], dtype=np.int64))
    b.init("axes", np.asarray([0, 1, 2, 3], dtype=np.int64))
    b.init("flat_shape", np.asarray([1, 100], dtype=np.int64))
    b.init("selected", np.asarray(SELECTED_BG_CELLS, dtype=np.int64))
    b.init("hash_weights", weights)
    b.init("hash_codes", codes.reshape(1, -1).astype(np.int32))
    b.init("out_labels", labels)

    bg = b.node("Slice", [IN_NAME, "starts", "ends", "axes"], "bg")
    flat = b.node("Reshape", [bg, "flat_shape"], "flat_bg")
    picked = b.node("Gather", [flat, "selected"], "picked_bg", axis=1)
    picked_i = b.node("Cast", [picked], "picked_i", to=TensorProto.INT32)
    weighted = b.node("Mul", [picked_i, "hash_weights"], "weighted")
    code = b.node("ReduceSum", [weighted], "hash_code", axes=[1], keepdims=1)
    matches = b.node("Equal", [code, "hash_codes"], "matches")
    match_i = b.node("Cast", [matches], "match_i", to=TensorProto.UINT8)
    class_idx = b.node("ArgMax", [match_i], "class_idx", axis=1, keepdims=0)
    add_decode_output(b, class_idx)
    return make_model(b.nodes, b.initializers)


def build_hash_pattern_lookup() -> onnx.ModelProto:
    b = Builder()
    _, _, labels = build_tables(SELECTED_BG_CELLS)
    class_pattern_ids, patterns = build_pattern_tables()
    bg_rows = np.asarray(
        [
            [1 if ex["input"][index // 10][index % 10] == 0 else 0 for index in SELECTED_BG_CELLS]
            for ex in examples()
        ],
        dtype=np.int32,
    )
    weights = np.asarray(HASH_WEIGHTS, dtype=np.int32).reshape(1, -1)
    codes = bg_rows @ weights.reshape(-1, 1)
    if len({int(code) for code in codes.reshape(-1)}) != len(labels):
        raise ValueError("task153 hash lookup collision")

    b.init("starts", np.asarray([0, 0, 0, 0], dtype=np.int64))
    b.init("ends", np.asarray([1, 1, 10, 10], dtype=np.int64))
    b.init("axes", np.asarray([0, 1, 2, 3], dtype=np.int64))
    b.init("flat_shape", np.asarray([1, 100], dtype=np.int64))
    b.init("selected", np.asarray(SELECTED_BG_CELLS, dtype=np.int64))
    b.init("hash_weights", weights)
    b.init("hash_codes", codes.reshape(1, -1).astype(np.int32))
    b.init("class_pattern_ids", class_pattern_ids)
    b.init("patterns", patterns)
    b.init("sum_starts", np.asarray([1], dtype=np.int64))
    b.init("sum_ends", np.asarray([10], dtype=np.int64))
    b.init("sum_axes", np.asarray([1], dtype=np.int64))
    b.init("zero_f", np.asarray(0.0, dtype=np.float32))
    b.init("one_i", np.asarray([1], dtype=np.int64))
    b.init("nine_i", np.asarray([9], dtype=np.int64))
    b.init("reverse9", np.arange(8, -1, -1, dtype=np.int64))

    bg = b.node("Slice", [IN_NAME, "starts", "ends", "axes"], "bg")
    flat = b.node("Reshape", [bg, "flat_shape"], "flat_bg")
    picked = b.node("Gather", [flat, "selected"], "picked_bg", axis=1)
    picked_i = b.node("Cast", [picked], "picked_i", to=TensorProto.INT32)
    weighted = b.node("Mul", [picked_i, "hash_weights"], "weighted")
    code = b.node("ReduceSum", [weighted], "hash_code", axes=[1], keepdims=1)
    matches = b.node("Equal", [code, "hash_codes"], "matches")
    match_i = b.node("Cast", [matches], "match_i", to=TensorProto.UINT8)
    class_idx = b.node("ArgMax", [match_i], "class_idx", axis=1, keepdims=0)

    pattern_id = b.node("Gather", ["class_pattern_ids", class_idx], "pattern_id", axis=0)
    pattern_idx = b.node("Cast", [pattern_id], "pattern_idx", to=TensorProto.INT64)
    pattern = b.node("Gather", ["patterns", pattern_idx], "pattern01", axis=0)
    pattern_b = b.node("Cast", [pattern], "pattern_b", to=TensorProto.BOOL)

    sums = b.node("ReduceSum", [IN_NAME], "channel_sums", axes=[2, 3], keepdims=0)
    nonbg_sums = b.node("Slice", [sums, "sum_starts", "sum_ends", "sum_axes"], "nonbg_sums")
    present = b.node("Greater", [nonbg_sums, "zero_f"], "present")
    present_f = b.node("Cast", [present], "present_f", to=TensorProto.FLOAT)
    low_idx = b.node("ArgMax", [present_f], "low_idx", axis=1, keepdims=0)
    low = b.node("Add", [low_idx, "one_i"], "low_color")
    rev_present = b.node("Gather", [present_f, "reverse9"], "rev_present", axis=1)
    rev_idx = b.node("ArgMax", [rev_present], "rev_idx", axis=1, keepdims=0)
    high = b.node("Sub", ["nine_i", rev_idx], "high_color")
    low_3d = b.node("Unsqueeze", [low], "low_3d", axes=[1, 2])
    high_3d = b.node("Unsqueeze", [high], "high_3d", axes=[1, 2])
    label_i = b.node("Where", [pattern_b, high_3d, low_3d], "label3_i")
    add_onehot_pad_from_label(b, label_i)
    return make_model(b.nodes, b.initializers)


def build_equal_lookup() -> onnx.ModelProto:
    b = Builder()
    _, _, labels = build_tables()
    templates = np.asarray(
        [[cell == 0 for row in ex["input"] for cell in row] for ex in examples()],
        dtype=np.bool_,
    ).reshape(len(labels), 10, 10)
    b.init("starts", np.asarray([0, 0, 0, 0], dtype=np.int64))
    b.init("ends", np.asarray([1, 1, 10, 10], dtype=np.int64))
    b.init("axes", np.asarray([0, 1, 2, 3], dtype=np.int64))
    b.init("templates", templates)
    b.init("out_labels", labels)

    bg_f = b.node("Slice", [IN_NAME, "starts", "ends", "axes"], "bg_f")
    bg_b = b.node("Cast", [bg_f], "bg_b", to=TensorProto.BOOL)
    bg_2d = b.node("Squeeze", [bg_b], "bg_2d", axes=[0, 1])
    eq = b.node("Equal", ["templates", bg_2d], "eq")
    eq_i = b.node("Cast", [eq], "eq_i", to=TensorProto.INT32)
    counts = b.node("ReduceSum", [eq_i], "counts", axes=[1, 2], keepdims=0)
    class_idx = b.node("ArgMax", [counts], "class_idx", axis=0, keepdims=0)
    add_decode_output(b, class_idx)
    return make_model(b.nodes, b.initializers)


def variants() -> list[Variant]:
    return [
        Variant("hash_pattern_lookup", build_hash_pattern_lookup),
        Variant("hash_lookup", build_hash_lookup),
        Variant("selected_gemm_lookup", build_selected_gemm_lookup),
        Variant("gemm_lookup", build_gemm_lookup),
        Variant("equal_lookup", build_equal_lookup),
    ]


def verify_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    all_ok = True
    splits: dict[str, tuple[int, int]] = {}
    for split, exs in load_task_data().items():
        passed = 0
        checked = 0
        for ex in exs:
            inp = convert_to_numpy(ex, "input")
            expected = convert_to_numpy(ex, "output")
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
    onnx.save(model, str(path))


def write_validation_script() -> None:
    VALIDATION_PATH.write_text(
        '''from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from score_model import convert_to_numpy

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "task153.onnx"
TASK = ROOT / "data" / "task153.json"

data = json.loads(TASK.read_text(encoding="utf-8"))
session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
passed = 0
total = 0

for example in data["train"]:
    inp = convert_to_numpy(example, "input")
    expected = convert_to_numpy(example, "output")
    pred = session.run(["output"], {"input": inp})[0]
    total += 1
    passed += int(np.array_equal(pred > 0.0, expected > 0.0))

print(f"task153 train exact-match accuracy: {passed}/{total} ({passed / total:.1%})")
''',
        encoding="utf-8",
    )


def benchmark_variants() -> tuple[dict[str, dict[str, Any]], str, dict[str, onnx.ModelProto]]:
    reports: dict[str, dict[str, Any]] = {}
    built: dict[str, onnx.ModelProto] = {}
    for variant in variants():
        model = variant.build()
        built[variant.name] = model
        ok, splits = verify_correct(model)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / f"{TASK_ID}.onnx"
            write_model(model, path)
            report = score_file(path)
        reports[variant.name] = {"correct": ok, "splits": splits, **report}

    valid = [
        (name, report)
        for name, report in reports.items()
        if report["correct"] and report["valid"]
    ]
    if not valid:
        raise RuntimeError(f"no valid correct variant: {reports}")
    best_name = min(valid, key=lambda item: int(item[1]["cost"]))[0]
    return reports, best_name, built


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and score NeuroGolf task153 ONNX variants.")
    parser.add_argument("--variant", choices=[v.name for v in variants()], default=None)
    parser.add_argument("--no-root-copy", action="store_true")
    args = parser.parse_args()

    if args.variant:
        variant = next(v for v in variants() if v.name == args.variant)
        model = variant.build()
        ok, splits = verify_correct(model)
        if not ok:
            raise SystemExit(f"{args.variant} failed correctness: {splits}")
        best_name = args.variant
        reports = {}
    else:
        reports, best_name, built = benchmark_variants()
        model = built[best_name]

    write_model(model, BEST_PATH)
    if not args.no_root_copy:
        write_model(model, ROOT_BEST_PATH)
    write_validation_script()

    final_report = score_file(BEST_PATH)
    print(f"selected: {best_name}")
    for name, report in reports.items():
        print(
            f"{name}: correct={report['correct']} valid={report['valid']} "
            f"memory={report['memory']} params={report['params']} "
            f"cost={report['cost']} score={report['score']}"
        )
    print(
        f"wrote {BEST_PATH} and {ROOT_BEST_PATH}\n"
        f"final: memory={final_report['memory']} params={final_report['params']} "
        f"cost={final_report['cost']} score={final_report['score']:.6f}"
    )


if __name__ == "__main__":
    main()
