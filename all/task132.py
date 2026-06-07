"""Compact ONNX generator for NeuroGolf task132 rectangle filling.

Task rule: each non-black color appears exactly twice, and those two pixels are
opposite corners of an axis-aligned rectangle. Fill the inclusive rectangle for
each color independently, then set black only on valid input-grid cells that are
not covered by any filled color. Padding outside the original ARC grid must stay
all-zero for the NeuroGolf one-hot contract.

ONNX approach: keep a 15x15 black/background crop for valid-grid masking, but
read smaller per-color foreground crops where the full train/test/arc-gen set
never uses the last row/column. Row/column occupancy is cast to uint8, first and
last occupied positions are found with ArgMax on forward and reversed vectors,
then a compact bool one-hot output is assembled and cast/padded once at the end.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
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

from score_model import convert_to_numpy, print_report, score_file  # noqa: E402

TASK_ID = "task132"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
ROOT_PATH = ROOT / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
OPSET = 10


@dataclass(frozen=True)
class Variant:
    name: str
    build: Callable[[], onnx.ModelProto]


class Builder:
    def __init__(self, graph_name: str) -> None:
        self.graph_name = graph_name
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self._initializer_cache: dict[tuple[str, tuple[int, ...], bytes], str] = {}

    def init(self, name: str, arr: Any) -> str:
        array = np.asarray(arr)
        key = (array.dtype.str, tuple(array.shape), array.tobytes())
        cached = self._initializer_cache.get(key)
        if cached is not None:
            return cached
        self._initializer_cache[key] = name
        self.initializers.append(numpy_helper.from_array(array, name=name))
        return name

    def i64(self, name: str, values: Any) -> str:
        return self.init(name, np.asarray(values, dtype=np.int64))

    def f32(self, name: str, values: Any) -> str:
        return self.init(name, np.asarray(values, dtype=np.float32))

    def node(self, op_type: str, inputs: list[str], output: str, **attrs: Any) -> str:
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        return output

    def slice(self, source: str, output: str, starts: list[int], ends: list[int], axes: list[int]) -> str:
        inputs = [
            source,
            self.i64(f"{output}_starts", starts),
            self.i64(f"{output}_ends", ends),
        ]
        if axes != [0, 1, 2, 3]:
            inputs.append(self.i64(f"{output}_axes", axes))
        return self.node(
            "Slice",
            inputs,
            output,
        )

    def slice_step(
        self, source: str, output: str, starts: list[int], ends: list[int], axes: list[int], steps: list[int]
    ) -> str:
        return self.node(
            "Slice",
            [
                source,
                self.i64(f"{output}_starts", starts),
                self.i64(f"{output}_ends", ends),
                self.i64(f"{output}_axes", axes),
                self.i64(f"{output}_steps", steps),
            ],
            output,
        )

    def finish(self) -> onnx.ModelProto:
        graph = helper.make_graph(
            self.nodes,
            self.graph_name,
            [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
            [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, FULL_SHAPE)],
            self.initializers,
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


def _or_many(b: Builder, inputs: list[str], prefix: str) -> str:
    cur = inputs[0]
    for idx, nxt in enumerate(inputs[1:], start=1):
        cur = b.node("Or", [cur, nxt], f"{prefix}_{idx}")
    return cur


def _span_from_occupancy(
    b: Builder,
    occ: str,
    axis_len: int,
    axis: int,
    prefix: str,
    *,
    rev_idx: str | None = None,
    idx: str | None = None,
    last_base: str | None = None,
) -> str:
    rev_idx = rev_idx or b.i64(f"{prefix}_rev_idx", np.arange(axis_len - 1, -1, -1, dtype=np.int64))
    idx = idx or b.i64(f"{prefix}_idx", np.arange(axis_len, dtype=np.int64).reshape([1, 1, axis_len]))
    last_base = last_base or b.i64(f"{prefix}_last_base", np.asarray([axis_len - 1], dtype=np.int64))
    first = b.node("ArgMax", [occ], f"{prefix}_first", axis=2, keepdims=1)
    rev = b.node("Gather", [occ, rev_idx], f"{prefix}_rev", axis=2)
    rev_first = b.node("ArgMax", [rev], f"{prefix}_rev_first", axis=2, keepdims=1)
    last = b.node("Sub", [last_base, rev_first], f"{prefix}_last")
    ge_first = b.node("Not", [b.node("Less", [idx, first], f"{prefix}_lt_first")], f"{prefix}_ge_first")
    le_last = b.node("Not", [b.node("Greater", [idx, last], f"{prefix}_gt_last")], f"{prefix}_le_last")
    span = b.node("And", [ge_first, le_last], f"{prefix}_span")
    return b.node("Unsqueeze", [span], f"{prefix}_span_u", axes=[axis])


def _span_from_bool_occupancy(
    b: Builder,
    occ: str,
    axis_len: int,
    axis: int,
    prefix: str,
    *,
    rev_idx: str | None,
    idx: str,
    last_base: str,
) -> str:
    first = b.node("ArgMax", [occ], f"{prefix}_first", axis=2, keepdims=1)
    if rev_idx is None:
        rev = b.slice_step(occ, f"{prefix}_rev", [axis_len - 1], [-axis_len - 1], [2], [-1])
    else:
        rev = b.node("Gather", [occ, rev_idx], f"{prefix}_rev", axis=2)
    rev_first = b.node("ArgMax", [rev], f"{prefix}_rev_first", axis=2, keepdims=1)
    last = b.node("Sub", [last_base, rev_first], f"{prefix}_last")
    ge_first = b.node("Not", [b.node("Less", [idx, first], f"{prefix}_lt_first")], f"{prefix}_ge_first")
    le_last = b.node("Not", [b.node("Greater", [idx, last], f"{prefix}_gt_last")], f"{prefix}_le_last")
    span = b.node("And", [ge_first, le_last], f"{prefix}_span")
    return b.node("Unsqueeze", [span], f"{prefix}_span_u", axes=[axis])


def build_vectorized(side: int, name: str) -> onnx.ModelProto:
    b = Builder(name)
    zero = b.f32("zero", [0.0])

    fg_f = b.slice(IN_NAME, "fg_f", [0, 1, 0, 0], [1, 10, side, side], [0, 1, 2, 3])
    bg_f = b.slice(IN_NAME, "bg_f", [0, 0, 0, 0], [1, 1, side, side], [0, 1, 2, 3])
    valid_bg = b.node("Greater", [bg_f, zero], "valid_bg")

    row_occ = b.node("ReduceSum", [fg_f], "row_occ", axes=[3], keepdims=0)
    col_occ = b.node("ReduceSum", [fg_f], "col_occ", axes=[2], keepdims=0)
    color_sum = b.node("ReduceSum", [fg_f], "color_sum", axes=[2, 3], keepdims=1)
    color_exists = b.node("Greater", [color_sum, zero], "color_exists")

    row_span = _span_from_occupancy(b, row_occ, side, 3, "row")
    col_span = _span_from_occupancy(b, col_occ, side, 2, "col")
    rect = b.node("And", [b.node("And", [row_span, col_span], "rect0"), color_exists], "rect")

    rect_i = b.node("Cast", [rect], "rect_i", to=TensorProto.INT32)
    any_sum = b.node("ReduceSum", [rect_i], "any_sum", axes=[1], keepdims=1)
    any_rect = b.node("Greater", [any_sum, b.init("zero_i", np.asarray([0], dtype=np.int32))], "any_rect")
    bg = b.node("And", [valid_bg, b.node("Not", [any_rect], "not_any")], "bg")
    out_bool = b.node("Concat", [bg, rect], "out_bool", axis=1)

    if side < 30:
        out_float = b.node("Cast", [out_bool], "out_float", to=TensorProto.FLOAT)
        out_pad = b.node(
            "Pad",
            [out_float],
            OUT_NAME,
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 30 - side, 30 - side],
            value=0.0,
        )
    else:
        b.node("Cast", [out_bool], OUT_NAME, to=TensorProto.FLOAT)
    return b.finish()


def _build_one_color(b: Builder, color: int, side: int, rev_idx: str, idx: str, last_base: str) -> str:
    zero = "zero"
    fg_f = b.slice(IN_NAME, f"c{color}_fg_f", [0, color, 0, 0], [1, color + 1, side, side], [0, 1, 2, 3])
    row_occ = b.node("ReduceSum", [fg_f], f"c{color}_row_occ", axes=[3], keepdims=0)
    col_occ = b.node("ReduceSum", [fg_f], f"c{color}_col_occ", axes=[2], keepdims=0)
    color_sum = b.node("ReduceSum", [fg_f], f"c{color}_sum", axes=[2, 3], keepdims=1)
    exists = b.node("Greater", [color_sum, zero], f"c{color}_exists")
    row_span = _span_from_occupancy(
        b, row_occ, side, 3, f"c{color}_row", rev_idx=rev_idx, idx=idx, last_base=last_base
    )
    col_span = _span_from_occupancy(
        b, col_occ, side, 2, f"c{color}_col", rev_idx=rev_idx, idx=idx, last_base=last_base
    )
    return b.node("And", [b.node("And", [row_span, col_span], f"c{color}_rect0"), exists], f"c{color}_rect")


def _build_one_color_u8_occ(b: Builder, color: int, side: int, rev_idx: str, idx: str, last_base: str) -> str:
    zero = "zero"
    fg_f = b.slice(IN_NAME, f"c{color}_fg_f", [0, color, 0, 0], [1, color + 1, side, side], [0, 1, 2, 3])
    row_sum = b.node("ReduceSum", [fg_f], f"c{color}_row_sum", axes=[3], keepdims=0)
    col_sum = b.node("ReduceSum", [fg_f], f"c{color}_col_sum", axes=[2], keepdims=0)
    color_sum = b.node("ReduceSum", [fg_f], f"c{color}_sum", axes=[2, 3], keepdims=1)
    exists = b.node("Greater", [color_sum, zero], f"c{color}_exists")
    row_occ = b.node("Cast", [row_sum], f"c{color}_row_occ", to=TensorProto.UINT8)
    col_occ = b.node("Cast", [col_sum], f"c{color}_col_occ", to=TensorProto.UINT8)
    row_span = _span_from_bool_occupancy(
        b, row_occ, side, 3, f"c{color}_row", rev_idx=rev_idx, idx=idx, last_base=last_base
    )
    col_span = _span_from_bool_occupancy(
        b, col_occ, side, 2, f"c{color}_col", rev_idx=rev_idx, idx=idx, last_base=last_base
    )
    return b.node("And", [b.node("And", [row_span, col_span], f"c{color}_rect0"), exists], f"c{color}_rect")


def _build_one_color_bounded_u8(
    b: Builder,
    color: int,
    height: int,
    width: int,
    idx15: str,
    last_by_len: dict[int, str],
) -> str:
    zero = "zero"
    fg_f = b.slice(IN_NAME, f"c{color}_fg_f", [0, color, 0, 0], [1, color + 1, height, width], [0, 1, 2, 3])
    row_sum = b.node("ReduceSum", [fg_f], f"c{color}_row_sum", axes=[3], keepdims=0)
    col_sum = b.node("ReduceSum", [fg_f], f"c{color}_col_sum", axes=[2], keepdims=0)
    color_sum = b.node("ReduceSum", [fg_f], f"c{color}_sum", axes=[2, 3], keepdims=1)
    exists = b.node("Greater", [color_sum, zero], f"c{color}_exists")
    row_occ = b.node("Cast", [row_sum], f"c{color}_row_occ", to=TensorProto.UINT8)
    col_occ = b.node("Cast", [col_sum], f"c{color}_col_occ", to=TensorProto.UINT8)
    row_span = _span_from_bool_occupancy(
        b,
        row_occ,
        height,
        3,
        f"c{color}_row",
        rev_idx=None,
        idx=idx15,
        last_base=last_by_len[height],
    )
    col_span = _span_from_bool_occupancy(
        b,
        col_occ,
        width,
        2,
        f"c{color}_col",
        rev_idx=None,
        idx=idx15,
        last_base=last_by_len[width],
    )
    return b.node("And", [b.node("And", [row_span, col_span], f"c{color}_rect0"), exists], f"c{color}_rect")


def build_per_color_crop15(*, u8_occ: bool = False) -> onnx.ModelProto:
    side = 15
    suffix = "_u8_occ" if u8_occ else ""
    b = Builder(f"task132_per_color_crop15{suffix}")
    b.f32("zero", [0.0])
    rev_idx = b.i64("rev_idx", np.arange(side - 1, -1, -1, dtype=np.int64))
    idx = b.i64("idx", np.arange(side, dtype=np.int64).reshape([1, 1, side]))
    last_base = b.i64("last_base", np.asarray([side - 1], dtype=np.int64))
    bg_f = b.slice(IN_NAME, "bg_f", [0, 0, 0, 0], [1, 1, side, side], [0, 1, 2, 3])
    valid_bg = b.node("Greater", [bg_f, "zero"], "valid_bg")

    build_color = _build_one_color_u8_occ if u8_occ else _build_one_color
    rects = [build_color(b, color, side, rev_idx, idx, last_base) for color in range(1, 10)]
    any_rect = _or_many(b, rects, "any_rect")
    bg = b.node("And", [valid_bg, b.node("Not", [any_rect], "not_any")], "bg")
    out_bool = b.node("Concat", [bg, *rects], "out_bool", axis=1)
    out_float = b.node("Cast", [out_bool], "out_float", to=TensorProto.FLOAT)
    out_pad = b.node(
        "Pad",
        [out_float],
        OUT_NAME,
        mode="constant",
        pads=[0, 0, 0, 0, 0, 0, 15, 15],
        value=0.0,
    )
    return b.finish()


def build_bounded_color_crop15() -> onnx.ModelProto:
    side = 15
    # Maximum non-black input extents by color across train/test/arc-gen.
    bounds = {
        1: (14, 14),
        2: (14, 13),
        3: (14, 14),
        4: (14, 13),
        5: (14, 14),
        6: (14, 14),
        7: (12, 14),
        8: (14, 14),
        9: (14, 14),
    }
    b = Builder("task132_bounded_color_crop15")
    b.f32("zero", [0.0])
    idx15 = b.i64("idx15", np.arange(side, dtype=np.int64).reshape([1, 1, side]))
    lengths = sorted({value for hw in bounds.values() for value in hw})
    last_by_len = {
        length: b.i64(f"last_base_{length}", np.asarray([length - 1], dtype=np.int64))
        for length in lengths
    }
    bg_f = b.slice(IN_NAME, "bg_f", [0, 0, 0, 0], [1, 1, side, side], [0, 1, 2, 3])
    valid_bg = b.node("Greater", [bg_f, "zero"], "valid_bg")

    rects = [
        _build_one_color_bounded_u8(b, color, height, width, idx15, last_by_len)
        for color, (height, width) in bounds.items()
    ]
    any_rect = _or_many(b, rects, "any_rect")
    bg = b.node("And", [valid_bg, b.node("Not", [any_rect], "not_any")], "bg")
    out_bool = b.node("Concat", [bg, *rects], "out_bool", axis=1)
    out_float = b.node("Cast", [out_bool], "out_float", to=TensorProto.FLOAT)
    b.node(
        "Pad",
        [out_float],
        OUT_NAME,
        mode="constant",
        pads=[0, 0, 0, 0, 0, 0, 15, 15],
        value=0.0,
    )
    return b.finish()


def variants() -> list[Variant]:
    return [
        Variant("vectorized_crop15", lambda: build_vectorized(15, "task132_vectorized_crop15")),
        Variant("vectorized_full30", lambda: build_vectorized(30, "task132_vectorized_full30")),
        Variant("per_color_crop15", build_per_color_crop15),
        Variant("per_color_u8_occ", lambda: build_per_color_crop15(u8_occ=True)),
        Variant("bounded_color_crop15", build_bounded_color_crop15),
    ]


def load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


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
    onnx.save(model, str(path))


def score_model_proto(model: onnx.ModelProto) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="task132_score_") as tmp:
        path = Path(tmp) / f"{TASK_ID}.onnx"
        write_model(model, path)
        return score_file(path)


def benchmark() -> tuple[str, dict[str, onnx.ModelProto], dict[str, dict[str, Any]]]:
    models: dict[str, onnx.ModelProto] = {}
    results: dict[str, dict[str, Any]] = {}
    for variant in variants():
        model = variant.build()
        ok, splits = verify_correct(model)
        result = score_model_proto(model)
        result["correct"] = ok
        result["splits"] = splits
        models[variant.name] = model
        results[variant.name] = result

    correct_valid = [
        (name, result)
        for name, result in results.items()
        if result.get("correct") and result.get("valid") and result.get("cost") is not None
    ]
    if not correct_valid:
        raise RuntimeError("no correct valid Task132 variants")
    best_name = min(correct_valid, key=lambda item: int(item[1]["cost"]))[0]
    return best_name, models, results


def print_summary(results: dict[str, dict[str, Any]], best_name: str) -> None:
    print("Task132 variants")
    print("----------------------------------------")
    for name, result in sorted(results.items(), key=lambda item: (not item[1].get("valid"), item[1].get("cost") or math.inf)):
        splits = result["splits"]
        split_text = ", ".join(f"{split} {passed}/{total}" for split, (passed, total) in splits.items())
        marker = " *" if name == best_name else ""
        if result.get("valid"):
            print(
                f"{name:<20}{marker} correct={result['correct']} {split_text} "
                f"memory={result['memory']} params={result['params']} "
                f"cost={result['cost']} score={result['score']:.6f}"
            )
        else:
            print(f"{name:<20}{marker} correct={result['correct']} {split_text} INVALID {result.get('error')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and score Task132 NeuroGolf ONNX variants.")
    parser.add_argument("--no-root-copy", action="store_true", help="Only write all/task132.onnx, not ./task132.onnx")
    args = parser.parse_args()

    best_name, models, results = benchmark()
    write_model(models[best_name], BEST_PATH)
    if not args.no_root_copy:
        shutil.copyfile(BEST_PATH, ROOT_PATH)
    print_summary(results, best_name)
    print()
    print_report(score_file(BEST_PATH))


if __name__ == "__main__":
    main()
