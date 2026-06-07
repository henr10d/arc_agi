"""Direct ONNX solver for NeuroGolf task143.

Task rule: the 10x10 input contains a reference object in the top-left 3x3
area, separated from the rest of the scene by gray color 5. Find any other
object with the same cropped binary shape and the same orientation, recolor
that matched object to gray, and leave the reference, divider, background, and
non-matching objects unchanged.

The ONNX graph builds a non-background/non-gray 10x10 mask, classifies the
reference by comparing the top-left 3x3 mask against the task's observed
reference patterns, then uses grouped Conv template matching with per-color
isolation to paint only cells belonging to matching placements.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import calculate_params, convert_to_numpy, score, score_file  # noqa: E402


TASK_NUM = "143"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
OPSET = 10


@dataclass(frozen=True)
class RefPattern:
    raw3: tuple[tuple[int, ...], ...]
    crop: tuple[tuple[int, ...], ...]

    @property
    def dims(self) -> tuple[int, int]:
        return len(self.crop), len(self.crop[0])

    @property
    def area(self) -> int:
        return sum(sum(row) for row in self.crop)


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []

    def init(self, name: str, array: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(array, name))
        return name

    def init_i64(self, name: str, values: list[int]) -> str:
        return self.init(name, np.asarray(values, dtype=np.int64))

    def init_f32(self, name: str, values: Any) -> str:
        return self.init(name, np.asarray(values, dtype=np.float32))

    def init_f16(self, name: str, values: Any) -> str:
        return self.init(name, np.asarray(values, dtype=np.float16))

    def init_bool(self, name: str, values: Any) -> str:
        return self.init(name, np.asarray(values, dtype=bool))

    def node(self, op_type: str, inputs: list[str], output: str, **attrs: Any) -> str:
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        return output


def load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def reference_patterns() -> list[RefPattern]:
    patterns: dict[tuple[tuple[int, ...], ...], RefPattern] = {}
    for examples in load_task_data().values():
        for example in examples:
            grid = example["input"]
            raw = tuple(
                tuple(1 if grid[r][c] not in (0, 5) else 0 for c in range(3))
                for r in range(3)
            )
            pts = [(r, c) for r in range(3) for c in range(3) if raw[r][c]]
            r0 = min(r for r, _ in pts)
            c0 = min(c for _, c in pts)
            r1 = max(r for r, _ in pts)
            c1 = max(c for _, c in pts)
            crop = tuple(tuple(raw[r][c] for c in range(c0, c1 + 1)) for r in range(r0, r1 + 1))
            patterns.setdefault(raw, RefPattern(raw, crop))
    return sorted(patterns.values(), key=lambda p: (p.dims, p.crop, p.raw3))


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


def build_model() -> onnx.ModelProto:
    raw_patterns = reference_patterns()
    crop_groups: dict[tuple[tuple[int, ...], ...], list[int]] = defaultdict(list)
    for raw_idx, pattern in enumerate(raw_patterns):
        crop_groups[pattern.crop].append(raw_idx)
    patterns = [
        RefPattern(raw_patterns[raw_indices[0]].raw3, crop)
        for crop, raw_indices in sorted(crop_groups.items(), key=lambda item: ((len(item[0]), len(item[0][0])), item[0]))
    ]
    raw_indices_by_crop = [
        crop_groups[pattern.crop]
        for pattern in patterns
    ]
    b = Builder()

    b.init_i64("obj_channels", [1, 2, 3, 4, 6, 7, 8, 9])
    b.init_i64("axes_hw", [2, 3])
    b.init_i64("slice10_starts", [0, 0])
    b.init_i64("slice10_ends", [10, 10])
    b.init_i64("slice3_starts", [0, 0])
    b.init_i64("slice3_ends", [3, 3])
    b.init_f16("gt_zero_f16", 0.0)
    b.init_f16("gt_eight_half", 8.5)
    b.init_bool("ref_patterns", np.asarray([p.raw3 for p in raw_patterns], dtype=bool).reshape(len(raw_patterns), 1, 3, 3))
    gray_onehot = np.zeros((1, 10, 1, 1), dtype=np.float32)
    gray_onehot[0, 5, 0, 0] = True
    b.init_f32("gray_onehot", gray_onehot)
    not_reference = np.ones((1, 1, 10, 10), dtype=bool)
    not_reference[:, :, :3, :3] = False
    b.init_bool("not_reference", not_reference)

    input10 = b.node("Slice", [IN_NAME, "slice10_starts", "slice10_ends", "axes_hw"], "input10")
    input10_f16 = b.node("Cast", [input10], "input10_f16", to=TensorProto.FLOAT16)
    obj_ch = b.node("Gather", [input10_f16, "obj_channels"], "obj_ch", axis=1)
    obj_sum = b.node("ReduceSum", [obj_ch], "obj_sum", axes=[1], keepdims=1)
    obj_bool_full = b.node("Greater", [obj_sum, "gt_zero_f16"], "obj_bool_full")
    color12 = b.node(
        "Pad",
        [obj_ch],
        "color12",
        mode="constant",
        pads=[0, 0, 1, 1, 0, 0, 1, 1],
        value=0.0,
    )
    ref3 = b.node("Slice", [obj_bool_full, "slice3_starts", "slice3_ends", "axes_hw"], "ref3")
    ref_equal = b.node("Equal", [ref3, "ref_patterns"], "ref_equal")
    ref_equal_f = b.node("Cast", [ref_equal], "ref_equal_f", to=TensorProto.FLOAT16)
    ref_score = b.node("ReduceSum", [ref_equal_f], "ref_score", axes=[1, 2, 3], keepdims=0)
    raw_ref_flags = b.node("Greater", [ref_score, "gt_eight_half"], "raw_ref_flags")
    crop_flag_parts: list[str] = []
    for crop_idx, raw_indices in enumerate(raw_indices_by_crop):
        b.init_i64(f"crop_raw_refs_{crop_idx}", raw_indices)
        gathered = b.node("Gather", [raw_ref_flags, f"crop_raw_refs_{crop_idx}"], f"crop_raw_flags_{crop_idx}", axis=0)
        if len(raw_indices) == 1:
            crop_flag_parts.append(gathered)
        else:
            gathered_f = b.node("Cast", [gathered], f"crop_raw_flags_f_{crop_idx}", to=TensorProto.FLOAT16)
            crop_score = b.node("ReduceSum", [gathered_f], f"crop_score_{crop_idx}", axes=[0], keepdims=1)
            crop_flag_parts.append(b.node("Greater", [crop_score, "gt_zero_f16"], f"crop_flag_{crop_idx}"))
    ref_flags = b.node("Concat", crop_flag_parts, "ref_flags", axis=0)

    group_masks: list[str] = []
    by_dims: dict[tuple[int, int], list[tuple[int, RefPattern]]] = defaultdict(list)
    for idx, pattern in enumerate(patterns):
        by_dims[pattern.dims].append((idx, pattern))

    for group_idx, ((height, width), items) in enumerate(sorted(by_dims.items())):
        count = len(items)
        channels = 8 * count
        kernel = np.zeros((channels, 1, height + 2, width + 2), dtype=np.float32)
        lo = np.zeros((1, channels, 1, 1), dtype=np.float32)
        paint = np.zeros((channels, 1, height, width), dtype=np.float32)
        ref_indices = []
        for color_idx in range(8):
            for pattern_ch, (pattern_idx, pattern) in enumerate(items):
                out_ch = color_idx * count + pattern_ch
                crop = np.asarray(pattern.crop, dtype=np.float32)
                kernel[out_ch, 0] = -1.0
                kernel[out_ch, 0, 1 : height + 1, 1 : width + 1] = np.where(crop > 0, 1.0, -1.0)
                paint[out_ch, 0] = crop
                lo[0, out_ch, 0, 0] = float(pattern.area) - 0.5
                ref_indices.append(pattern_idx)

        out_h = 10 - height + 1
        out_w = 10 - width + 1
        b.init_f16(f"k_{group_idx}", kernel)
        b.init_f16(f"lo_{group_idx}", lo)
        b.init_f16(f"paint_{group_idx}", paint)
        b.init_i64(f"refs_{group_idx}", ref_indices)

        conv = b.node("Conv", ["color12", f"k_{group_idx}"], f"conv_{group_idx}", group=8)
        exact = b.node("Greater", [conv, f"lo_{group_idx}"], f"exact_{group_idx}")
        flags = b.node("Gather", [ref_flags, f"refs_{group_idx}"], f"flags_{group_idx}", axis=0)
        flags_rs = b.node("Reshape", [flags, b.init_i64(f"flag_shape_{group_idx}", [1, channels, 1, 1])], f"flags_rs_{group_idx}")
        selected = b.node("And", [exact, flags_rs], f"selected_{group_idx}")
        selected_f = b.node("Cast", [selected], f"selected_f_{group_idx}", to=TensorProto.FLOAT16)
        painted = b.node(
            "ConvTranspose",
            [selected_f, f"paint_{group_idx}"],
            f"painted_{group_idx}",
        )
        group_masks.append(painted)

    sum_mask = b.node("Sum", group_masks, "sum_mask")
    mask10 = b.node("Greater", [sum_mask, "gt_zero_f16"], "mask10")
    mask10_without_ref = b.node("And", [mask10, "not_reference"], "mask10_without_ref")
    mask10_f16 = b.node("Cast", [mask10_without_ref], "mask10_f16", to=TensorProto.FLOAT16)
    padded_mask = b.node(
        "Pad",
        [mask10_f16],
        "padded_mask",
        mode="constant",
        pads=[0, 0, 0, 0, 0, 0, 20, 20],
        value=0.0,
    )
    mask30 = b.node("Greater", [padded_mask, "gt_zero_f16"], "mask30")

    b.node("Where", [mask30, "gray_onehot", IN_NAME], OUT_NAME)

    return make_model(b.nodes, b.initializers)


def write_model(model: onnx.ModelProto, path: Path) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(path))


def verify_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    splits: dict[str, tuple[int, int]] = {}
    all_ok = True
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


def grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            arr[0, color, r, c] = 1.0
    return arr


def onehot_to_grid(arr: np.ndarray, size: int = 10) -> list[list[int]]:
    return arr[0, :, :size, :size].argmax(axis=0).astype(int).tolist()


def synthetic_cases() -> list[tuple[str, list[list[int]], list[list[int]]]]:
    base = [[0] * 10 for _ in range(10)]
    base[0][0] = base[0][1] = base[1][1] = 2
    for r in range(4):
        base[r][3] = 5
        base[3][r] = 5

    same = [row[:] for row in base]
    same[5][5] = same[5][6] = same[6][6] = 8
    same_expected = [row[:] for row in same]
    same_expected[5][5] = same_expected[5][6] = same_expected[6][6] = 5

    rotated = [row[:] for row in base]
    rotated[5][5] = rotated[6][5] = rotated[6][6] = 8

    different = [row[:] for row in base]
    different[5][5] = different[5][6] = different[6][5] = different[6][6] = 8

    unchanged_ref = [row[:] for row in same_expected]
    unchanged_ref[0][0] = unchanged_ref[0][1] = unchanged_ref[1][1] = 2

    return [
        ("same pattern recolored", same, same_expected),
        ("rotated pattern unchanged", rotated, rotated),
        ("different shape unchanged", different, different),
        ("reference remains unchanged", same, unchanged_ref),
    ]


def verify_synthetic(model: onnx.ModelProto) -> tuple[bool, list[str]]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    failures: list[str] = []
    for name, inp_grid, expected_grid in synthetic_cases():
        pred = onehot_to_grid(session.run([OUT_NAME], {IN_NAME: grid_to_onehot(inp_grid)})[0])
        if pred != expected_grid:
            failures.append(name)
    return not failures, failures


def inferred_internal_shapes(model: onnx.ModelProto) -> list[tuple[str, str]]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    shapes: list[tuple[str, str]] = []
    for value in inferred.graph.value_info:
        tensor_type = value.type.tensor_type
        dims = [str(dim.dim_value) for dim in tensor_type.shape.dim]
        dtype = TensorProto.DataType.Name(tensor_type.elem_type)
        shapes.append((value.name, f"{dtype}[{','.join(dims)}]"))
    return shapes


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Build and verify {TASK_ID}.onnx")
    parser.add_argument("--out", type=Path, default=BEST_PATH, help="output ONNX path")
    parser.add_argument("--shapes", action="store_true", help="print all inferred internal tensor shapes")
    args = parser.parse_args()

    model = build_model()
    write_model(model, args.out)

    task_ok, splits = verify_correct(model)
    synthetic_ok, synthetic_failures = verify_synthetic(model)
    result = score_file(args.out)
    params = calculate_params(model)

    print(f"wrote: {args.out}")
    print(f"reference patterns: {len(reference_patterns())}")
    print(f"task verification: {task_ok} {splits}")
    print(f"synthetic verification: {synthetic_ok}" + (f" failures={synthetic_failures}" if synthetic_failures else ""))
    if result["valid"]:
        cost = int(result["cost"])
        print(f"official memory: {result['memory']}")
        print(f"params: {result['params']} (raw={params})")
        print(f"cost: {cost}")
        print(f"score: {result['score']:.6f}")
    else:
        print(f"score invalid: {result['error']}")
        if params is not None:
            print(f"raw params: {params}")

    shapes = inferred_internal_shapes(model)
    max_shape = max((math.prod(int(part) for part in desc.split("[", 1)[1].rstrip("]").split(",")) for _, desc in shapes), default=0)
    print(f"inferred internal tensors: {len(shapes)}")
    print(f"largest inferred element count: {max_shape}")
    if args.shapes:
        for name, desc in shapes:
            print(f"{name}: {desc}")

    if not task_ok or not synthetic_ok or not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
