"""Compact ONNX generator for NeuroGolf task087.

Task rule: the input is always a 3x3 ARC grid in the top-left of the 30x30
canvas. The output is the same grid rotated 180 degrees (flip both axes):
out[r, c] = in[2 - r, 2 - c]. All cells outside the 3x3 region stay unactivated.

ONNX approach: the best variant uses one negative-step Slice to both crop and
reverse the 3x3 core, then pads that compact tensor directly to the required
30x30 output.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
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

TASK_NUM = "087"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
CORE = 3
PAD_BOTTOM = 30 - CORE
PAD_RIGHT = 30 - CORE


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.counter = 0

    def init_i64(self, name: str, values: list[int]) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name))
        return name

    def name(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def node(self, op_type: str, inputs: list[str], prefix: str, **attrs: Any) -> str:
        out = self.name(prefix)
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out


def make_model(nodes: list[onnx.NodeProto], initializers: list[onnx.TensorProto], opset: int) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=10)
    onnx.checker.check_model(model, full_check=True)
    return model


def add_pad_to_output(b: Builder, x: str, *, opset: int) -> None:
    pads = [0, 0, 0, 0, 0, 0, PAD_BOTTOM, PAD_RIGHT]
    if opset >= 11:
        b.init_i64("pad_to_30", pads)
        b.nodes.append(helper.make_node("Pad", [x, "pad_to_30"], [OUT_NAME], mode="constant"))
    else:
        b.nodes.append(helper.make_node("Pad", [x], [OUT_NAME], mode="constant", pads=pads, value=0.0))


def build_crop_float_slice(*, opset: int = 10, spatial_crop: bool = True) -> onnx.ModelProto:
    """Crop top-left 3x3, reverse H/W with one negative-step Slice, pad to 30x30."""
    b = Builder()
    if spatial_crop:
        b.init_i64("crop_starts", [0, 0])
        b.init_i64("crop_ends", [CORE, CORE])
        b.init_i64("crop_axes", [2, 3])
        core = b.node("Slice", [IN_NAME, "crop_starts", "crop_ends", "crop_axes"], "core")
    else:
        b.init_i64("crop_starts", [0, 0, 0, 0])
        b.init_i64("crop_ends", [1, 10, CORE, CORE])
        core = b.node("Slice", [IN_NAME, "crop_starts", "crop_ends"], "core")

    b.init_i64("rev_starts", [2, 2])
    b.init_i64("rev_ends", [-4, -4])
    b.init_i64("rev_axes", [2, 3])
    b.init_i64("rev_steps", [-1, -1])
    rev = b.node("Slice", [core, "rev_starts", "rev_ends", "rev_axes", "rev_steps"], "rev")
    add_pad_to_output(b, rev, opset=opset)
    return make_model(b.nodes, b.initializers, opset)


def build_single_reverse_slice(*, opset: int = 10) -> onnx.ModelProto:
    """Reverse-crop the top-left 3x3 directly, then pad to 30x30."""
    b = Builder()
    b.init_i64("rev_starts", [CORE - 1, CORE - 1])
    b.init_i64("rev_ends", [-31, -31])
    b.init_i64("rev_axes", [2, 3])
    b.init_i64("rev_steps", [-1, -1])

    rev = b.node("Slice", [IN_NAME, "rev_starts", "rev_ends", "rev_axes", "rev_steps"], "rev")
    add_pad_to_output(b, rev, opset=opset)
    return make_model(b.nodes, b.initializers, opset)


def build_crop_bool_slice(*, opset: int = 10) -> onnx.ModelProto:
    """Bool 3x3 crop + negative-step Slice + cast back before pad."""
    b = Builder()
    b.init_i64("crop_starts", [0, 0])
    b.init_i64("crop_ends", [CORE, CORE])
    b.init_i64("crop_axes", [2, 3])
    b.init_i64("rev_starts", [2, 2])
    b.init_i64("rev_ends", [-4, -4])
    b.init_i64("rev_axes", [2, 3])
    b.init_i64("rev_steps", [-1, -1])

    core = b.node("Slice", [IN_NAME, "crop_starts", "crop_ends", "crop_axes"], "core_f")
    core_b = b.node("Cast", [core], "core_b", to=TensorProto.BOOL)
    rev_b = b.node("Slice", [core_b, "rev_starts", "rev_ends", "rev_axes", "rev_steps"], "rev_b")
    rev_f = b.node("Cast", [rev_b], "rev_f", to=TensorProto.FLOAT)
    add_pad_to_output(b, rev_f, opset=opset)
    return make_model(b.nodes, b.initializers, opset)


def build_float_gather(*, opset: int = 10) -> onnx.ModelProto:
    """Crop 3x3 then reverse with two Gather ops using indices [2, 1, 0]."""
    b = Builder()
    b.init_i64("crop_starts", [0, 0])
    b.init_i64("crop_ends", [CORE, CORE])
    b.init_i64("crop_axes", [2, 3])
    b.init_i64("rev_idx", [2, 1, 0])

    core = b.node("Slice", [IN_NAME, "crop_starts", "crop_ends", "crop_axes"], "core")
    rev_h = b.node("Gather", [core, "rev_idx"], "rev_h", axis=2)
    rev = b.node("Gather", [rev_h, "rev_idx"], "rev", axis=3)
    add_pad_to_output(b, rev, opset=opset)
    return make_model(b.nodes, b.initializers, opset)


def build_bool_gather(*, opset: int = 10) -> onnx.ModelProto:
    """Bool crop + two Gather reversals."""
    b = Builder()
    b.init_i64("crop_starts", [0, 0])
    b.init_i64("crop_ends", [CORE, CORE])
    b.init_i64("crop_axes", [2, 3])
    b.init_i64("rev_idx", [2, 1, 0])

    core = b.node("Slice", [IN_NAME, "crop_starts", "crop_ends", "crop_axes"], "core_f")
    core_b = b.node("Cast", [core], "core_b", to=TensorProto.BOOL)
    rev_h = b.node("Gather", [core_b, "rev_idx"], "rev_h", axis=2)
    rev_b = b.node("Gather", [rev_h, "rev_idx"], "rev_b", axis=3)
    rev_f = b.node("Cast", [rev_b], "rev_f", to=TensorProto.FLOAT)
    add_pad_to_output(b, rev_f, opset=opset)
    return make_model(b.nodes, b.initializers, opset)


def build_flatten_gather(*, opset: int = 10, use_bool: bool = False) -> onnx.ModelProto:
    """Flatten 3x3, Gather reverse order [8..0], reshape back."""
    b = Builder()
    b.init_i64("crop_starts", [0, 0])
    b.init_i64("crop_ends", [CORE, CORE])
    b.init_i64("crop_axes", [2, 3])
    b.init_i64("flat_shape", [1, 10, 9])
    b.init_i64("out_shape", [1, 10, CORE, CORE])
    b.init_i64("rev_flat", [8, 7, 6, 5, 4, 3, 2, 1, 0])

    core = b.node("Slice", [IN_NAME, "crop_starts", "crop_ends", "crop_axes"], "core_f")
    if use_bool:
        core = b.node("Cast", [core], "core_b", to=TensorProto.BOOL)
    flat = b.node("Reshape", [core, "flat_shape"], "flat")
    gathered = b.node("Gather", [flat, "rev_flat"], "gathered", axis=2)
    rev = b.node("Reshape", [gathered, "out_shape"], "rev")
    if use_bool:
        rev = b.node("Cast", [rev], "rev_f", to=TensorProto.FLOAT)
    add_pad_to_output(b, rev, opset=opset)
    return make_model(b.nodes, b.initializers, opset)


def build_variants() -> dict[str, onnx.ModelProto]:
    return {
        "single_reverse_slice": build_single_reverse_slice(opset=10),
        "crop_float_slice": build_crop_float_slice(opset=10, spatial_crop=True),
        "crop_float_slice_full": build_crop_float_slice(opset=10, spatial_crop=False),
        "crop_bool_slice": build_crop_bool_slice(opset=10),
        "float_gather": build_float_gather(opset=10),
        "bool_gather": build_bool_gather(opset=10),
        "flatten_gather_f32": build_flatten_gather(opset=10, use_bool=False),
        "flatten_gather_bool": build_flatten_gather(opset=10, use_bool=True),
        "crop_float_slice_opset11": build_crop_float_slice(opset=11, spatial_crop=True),
    }


def load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def verify_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    split_counts: dict[str, tuple[int, int]] = {}
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
        split_counts[split] = (passed, checked)
    return all_ok, split_counts


def write_model(model: onnx.ModelProto, path: Path) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(path))


def benchmark_variants() -> tuple[dict[str, dict[str, Any]], str, dict[str, onnx.ModelProto]]:
    variants = build_variants()
    results: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}_") as tmp:
        tmp_path = Path(tmp)
        for name, model in variants.items():
            ok, splits = verify_correct(model)
            path = tmp_path / f"{TASK_ID}.onnx"
            write_model(model, path)
            result = score_file(path)
            result["correct"] = ok
            result["splits"] = splits
            results[name] = result

    def sort_key(name: str) -> int:
        result = results[name]
        if not result["valid"] or not result["correct"]:
            return 10**18
        return int(result["cost"])

    best_name = min(results, key=sort_key)
    if sort_key(best_name) >= 10**18:
        raise RuntimeError(f"no valid correct variant: {results}")
    return results, best_name, variants


def print_benchmark(results: dict[str, dict[str, Any]], best_name: str) -> None:
    print(f"{'variant':<28} {'ok':<5} {'valid':<6} {'memory':>8} {'params':>7} {'cost':>8} {'score':>10}")
    for name, result in sorted(results.items(), key=lambda item: (item[0] != best_name, item[0])):
        score = result["score"]
        score_text = f"{score:.6f}" if isinstance(score, float) else "None"
        print(
            f"{name:<28} {str(result['correct']):<5} {str(result['valid']):<6} "
            f"{str(result['memory']):>8} {str(result['params']):>7} "
            f"{str(result['cost']):>8} {score_text:>10}"
        )
        if not result["correct"]:
            print(f"  splits: {result['splits']}")
        if result["error"]:
            print(f"  error: {str(result['error']).strip()}")
    print(f"best: {best_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and benchmark task087 ONNX variants.")
    parser.add_argument("--benchmark-only", action="store_true", help="print variant scores without writing best model")
    args = parser.parse_args()

    results, best_name, variants = benchmark_variants()
    print_benchmark(results, best_name)
    if not args.benchmark_only:
        OUT_DIR.mkdir(exist_ok=True)
        write_model(variants[best_name], BEST_PATH)
        print(f"wrote: {BEST_PATH}")


if __name__ == "__main__":
    main()
