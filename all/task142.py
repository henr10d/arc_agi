"""Optimized ONNX generator for NeuroGolf task142.

Task rule: the ARC input is always a 3x3 grid using colors 0-3 in the top-left
of the competition one-hot tensor. The output is a 6x6 mirror tile:
top-left is the original 3x3 pattern, top-right is the left-right flip,
bottom-left is the top-bottom flip, and bottom-right is flipped both ways.

The official I/O contract still requires a [1, 10, 30, 30] output tensor, so
the best submission graph builds only the compact color-0..3 6x6 region and
pads channels 4-9 plus the unused spatial area as the final graph output. The
final output tensor is excluded from memory scoring.
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


TASK_NUM = "142"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
COMPACT_SHAPE = [1, 10, 6, 6]
IR_VERSION = 10
OPSET = 10
I64_MIN = -9223372036854775808


@dataclass(frozen=True)
class Variant:
    name: str
    build: Callable[[], onnx.ModelProto]
    verifier_correct_required: bool = True


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []

    def init_i64(self, name: str, values: list[int]) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name))
        return name

    def node(self, op_type: str, inputs: list[str], output: str, **attrs: Any) -> str:
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        return output


def make_model(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    *,
    output_shape: list[int] = FULL_SHAPE,
) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, output_shape)],
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


def pad_to_full_output(b: Builder, x: str) -> None:
    b.nodes.append(
        helper.make_node(
            "Pad",
            [x],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 24, 24],
            value=0.0,
        )
    )


def pad_color4_to_full_output(b: Builder, x: str) -> None:
    b.nodes.append(
        helper.make_node(
            "Pad",
            [x],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 6, 24, 24],
            value=0.0,
        )
    )


def build_slice_negative_steps_full() -> onnx.ModelProto:
    """A: Slice crop plus negative-step Slice flips, then final full-size Pad."""
    b = Builder()
    b.init_i64("crop_starts", [0, 0])
    b.init_i64("crop_ends", [3, 3])
    b.init_i64("crop_axes", [2, 3])
    b.init_i64("lr_starts", [2])
    b.init_i64("lr_ends", [I64_MIN])
    b.init_i64("lr_axes", [3])
    b.init_i64("lr_steps", [-1])
    b.init_i64("tb_starts", [2])
    b.init_i64("tb_ends", [I64_MIN])
    b.init_i64("tb_axes", [2])
    b.init_i64("tb_steps", [-1])

    x = b.node("Slice", [IN_NAME, "crop_starts", "crop_ends", "crop_axes"], "x")
    x_lr = b.node("Slice", [x, "lr_starts", "lr_ends", "lr_axes", "lr_steps"], "x_lr")
    top = b.node("Concat", [x, x_lr], "top", axis=3)
    bottom = b.node("Slice", [top, "tb_starts", "tb_ends", "tb_axes", "tb_steps"], "bottom")
    small = b.node("Concat", [top, bottom], "small", axis=2)
    pad_to_full_output(b, small)
    return make_model(b.nodes, b.initializers)


def build_gather_flips_full() -> onnx.ModelProto:
    """B: compact crop plus Gather flips with [2, 1, 0] indices."""
    b = Builder()
    b.init_i64("crop_starts", [0, 0])
    b.init_i64("crop_ends", [3, 3])
    b.init_i64("crop_axes", [2, 3])
    b.init_i64("rev3", [2, 1, 0])

    x = b.node("Slice", [IN_NAME, "crop_starts", "crop_ends", "crop_axes"], "x")
    x_lr = b.node("Gather", [x, "rev3"], "x_lr", axis=3)
    top = b.node("Concat", [x, x_lr], "top", axis=3)
    bottom = b.node("Gather", [top, "rev3"], "bottom", axis=2)
    small = b.node("Concat", [top, bottom], "small", axis=2)
    pad_to_full_output(b, small)
    return make_model(b.nodes, b.initializers)


def build_gather_flips_bool_full() -> onnx.ModelProto:
    """B2: Gather flips on bool compact tensors, cast to float just before Pad."""
    b = Builder()
    b.init_i64("crop_starts", [0, 0])
    b.init_i64("crop_ends", [3, 3])
    b.init_i64("crop_axes", [2, 3])
    b.init_i64("rev3", [2, 1, 0])

    x_f = b.node("Slice", [IN_NAME, "crop_starts", "crop_ends", "crop_axes"], "x_f")
    x = b.node("Cast", [x_f], "x", to=TensorProto.BOOL)
    x_lr = b.node("Gather", [x, "rev3"], "x_lr", axis=3)
    top = b.node("Concat", [x, x_lr], "top", axis=3)
    bottom = b.node("Gather", [top, "rev3"], "bottom", axis=2)
    small_b = b.node("Concat", [top, bottom], "small_b", axis=2)
    small = b.node("Cast", [small_b], "small", to=TensorProto.FLOAT)
    pad_to_full_output(b, small)
    return make_model(b.nodes, b.initializers)


def build_shape_specialized_gather_full() -> onnx.ModelProto:
    """C: one row Gather and one compact column Gather before final Pad."""
    b = Builder()
    b.init_i64("rows6", [0, 1, 2, 2, 1, 0])
    b.init_i64("cols6", [0, 1, 2, 2, 1, 0])

    rows = b.node("Gather", [IN_NAME, "rows6"], "rows", axis=2)
    small = b.node("Gather", [rows, "cols6"], "small", axis=3)
    pad_to_full_output(b, small)
    return make_model(b.nodes, b.initializers)


def build_color4_gather_flips_bool_full() -> onnx.ModelProto:
    """D: crop colors 0-3 and the 3x3 core, mirror as bool, then final Pad."""
    b = Builder()
    b.init_i64("crop_starts", [0, 0, 0])
    b.init_i64("crop_ends", [4, 3, 3])
    b.init_i64("crop_axes", [1, 2, 3])
    b.init_i64("rev3", [2, 1, 0])

    x_f = b.node("Slice", [IN_NAME, "crop_starts", "crop_ends", "crop_axes"], "x_f")
    x = b.node("Cast", [x_f], "x", to=TensorProto.BOOL)
    x_lr = b.node("Gather", [x, "rev3"], "x_lr", axis=3)
    top = b.node("Concat", [x, x_lr], "top", axis=3)
    bottom = b.node("Gather", [top, "rev3"], "bottom", axis=2)
    small_b = b.node("Concat", [top, bottom], "small_b", axis=2)
    small = b.node("Cast", [small_b], "small", to=TensorProto.FLOAT)
    pad_color4_to_full_output(b, small)
    return make_model(b.nodes, b.initializers)


def build_color4_shape_gather_bool_full() -> onnx.ModelProto:
    """E: crop colors 0-3/3x3, then gather mirrored row and column indices."""
    b = Builder()
    b.init_i64("crop_starts", [0, 0, 0])
    b.init_i64("crop_ends", [4, 3, 3])
    b.init_i64("crop_axes", [1, 2, 3])
    b.init_i64("mirror6", [0, 1, 2, 2, 1, 0])

    x_f = b.node("Slice", [IN_NAME, "crop_starts", "crop_ends", "crop_axes"], "x_f")
    x = b.node("Cast", [x_f], "x", to=TensorProto.BOOL)
    rows = b.node("Gather", [x, "mirror6"], "rows", axis=2)
    small_b = b.node("Gather", [rows, "mirror6"], "small_b", axis=3)
    small = b.node("Cast", [small_b], "small", to=TensorProto.FLOAT)
    pad_color4_to_full_output(b, small)
    return make_model(b.nodes, b.initializers)


def build_color4_shape_gather_bool_full_no_axes() -> onnx.ModelProto:
    """F: same as E, with Slice default axes to save one initializer element."""
    b = Builder()
    b.init_i64("crop_starts", [0, 0, 0, 0])
    b.init_i64("crop_ends", [1, 4, 3, 3])
    b.init_i64("mirror6", [0, 1, 2, 2, 1, 0])

    x_f = b.node("Slice", [IN_NAME, "crop_starts", "crop_ends"], "x_f")
    x = b.node("Cast", [x_f], "x", to=TensorProto.BOOL)
    rows = b.node("Gather", [x, "mirror6"], "rows", axis=2)
    small_b = b.node("Gather", [rows, "mirror6"], "small_b", axis=3)
    small = b.node("Cast", [small_b], "small", to=TensorProto.FLOAT)
    pad_color4_to_full_output(b, small)
    return make_model(b.nodes, b.initializers)


def build_compact_slice_negative_steps() -> onnx.ModelProto:
    """Measured only: compact [1, 10, 6, 6] output, invalid for official I/O."""
    b = Builder()
    b.init_i64("crop_starts", [0, 0])
    b.init_i64("crop_ends", [3, 3])
    b.init_i64("crop_axes", [2, 3])
    b.init_i64("lr_starts", [2])
    b.init_i64("lr_ends", [I64_MIN])
    b.init_i64("lr_axes", [3])
    b.init_i64("lr_steps", [-1])
    b.init_i64("tb_starts", [2])
    b.init_i64("tb_ends", [I64_MIN])
    b.init_i64("tb_axes", [2])
    b.init_i64("tb_steps", [-1])

    x = b.node("Slice", [IN_NAME, "crop_starts", "crop_ends", "crop_axes"], "x")
    x_lr = b.node("Slice", [x, "lr_starts", "lr_ends", "lr_axes", "lr_steps"], "x_lr")
    top = b.node("Concat", [x, x_lr], "top", axis=3)
    bottom = b.node("Slice", [top, "tb_starts", "tb_ends", "tb_axes", "tb_steps"], "bottom")
    b.node("Concat", [top, bottom], OUT_NAME, axis=2)
    return make_model(b.nodes, b.initializers, output_shape=COMPACT_SHAPE)


def variants() -> list[Variant]:
    return [
        Variant("A_slice_negative_steps_full", build_slice_negative_steps_full),
        Variant("B_gather_flips_full", build_gather_flips_full),
        Variant("B2_gather_flips_bool_full", build_gather_flips_bool_full),
        Variant("C_shape_specialized_gather_full", build_shape_specialized_gather_full),
        Variant("D_color4_gather_flips_bool_full", build_color4_gather_flips_bool_full),
        Variant("E_color4_shape_gather_bool_full", build_color4_shape_gather_bool_full),
        Variant("F_color4_shape_gather_bool_no_axes", build_color4_shape_gather_bool_full_no_axes),
        Variant("compact_slice_negative_steps_6x6_invalid_io", build_compact_slice_negative_steps, False),
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
            if pred.shape == expected.shape and np.array_equal(pred > 0.0, expected > 0.0):
                passed += 1
            else:
                all_ok = False
        splits[split] = (passed, checked)
    return all_ok, splits


def write_model(model: onnx.ModelProto, path: Path) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(path))


def benchmark_variants() -> tuple[dict[str, dict[str, Any]], str, dict[str, onnx.ModelProto]]:
    built = {variant.name: variant.build() for variant in variants()}
    results: dict[str, dict[str, Any]] = {}
    best_name = ""
    best_cost: int | None = None
    with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}_") as tmp:
        tmpdir = Path(tmp)
        for variant in variants():
            path = tmpdir / f"{TASK_ID}.onnx"
            write_model(built[variant.name], path)
            result = score_file(path)
            ok, splits = verify_correct(built[variant.name])
            result["correct"] = ok
            result["splits"] = splits
            result["verifier_correct_required"] = variant.verifier_correct_required
            results[variant.name] = result
            if ok and variant.verifier_correct_required and result["valid"]:
                cost = int(result["cost"])
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_name = variant.name
    if not best_name:
        raise RuntimeError("no verifier-correct valid variant")
    return results, best_name, built


def print_results(results: dict[str, dict[str, Any]], best_name: str) -> None:
    for name, result in results.items():
        marker = " <-- best" if name == best_name else ""
        splits = ", ".join(f"{split} {passed}/{total}" for split, (passed, total) in result["splits"].items())
        if result["valid"]:
            print(
                f"{name}{marker}: correct={result['correct']} ({splits}), "
                f"memory={result['memory']}, params={result['params']}, "
                f"cost={result['cost']}, score={result['score']:.6f}, "
                f"filesize={result['filesize']}"
            )
        else:
            print(f"{name}: INVALID score, correct={result['correct']} ({splits}), error={result['error']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Build optimized ONNX for {TASK_ID}.")
    parser.add_argument("--benchmark-only", action="store_true", help="do not overwrite the best ONNX")
    args = parser.parse_args()

    results, best_name, built = benchmark_variants()
    print_results(results, best_name)

    if not args.benchmark_only:
        write_model(built[best_name], BEST_PATH)
        final = score_file(BEST_PATH)
        print(f"\nwrote {BEST_PATH.relative_to(ROOT)}")
        print(
            f"final: memory={final['memory']}, params={final['params']}, "
            f"cost={final['cost']}, score={final['score']:.6f}, filesize={final['filesize']}"
        )


if __name__ == "__main__":
    main()
