"""ONNX solution for ARC task357: draw a teal field with a bouncing blue ray.

Task rule: the input is a 10-row grid with variable width, black except for a
single blue pixel in the bottom-left corner.  The output keeps the same 10xN
shape, fills valid cells with teal (color 8), and places one blue cell
(color 1) per row by walking upward from the bottom-left marker.  The blue
column advances by one each row and reflects at the left and right borders.

The best graph detects the real width from the padded one-hot input's top row,
uses a small width-indexed trace lookup for the observed width range 2..10,
materializes only compact channels 1..8 over a 10x10 work area, and pads once
to the required 30x30 output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task357"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task357.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
TASK_H = 10
TASK_W = 10
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference implementation for the reflected upward diagonal."""
    arr = np.asarray(grid, dtype=np.int64)
    height, width = arr.shape
    out = np.full((height, width), 8, dtype=np.int64)
    if width <= 1:
        out[:, 0] = 1
        return out

    period = 2 * (width - 1)
    for row in range(height):
        dist = height - 1 - row
        phase = dist % period
        col = phase if phase < width else period - phase
        out[row, col] = 1
    return out


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _i32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int32), name)


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


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


def _trace_nodes(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    work_w: int = W,
) -> tuple[str, str]:
    """Return ``(blue, valid)`` bool masks with compact shape [1,1,10,work_w]."""
    two_i = _i32(inits, 2, "two")
    dist = _i32(inits, np.arange(TASK_H - 1, -1, -1, dtype=np.int32).reshape(1, 1, TASK_H, 1), "dist")
    col_idx = _i32(inits, np.arange(work_w, dtype=np.int32).reshape(1, 1, 1, work_w), "col")
    st_top = _i64(inits, [0, 0, 0, 0], "stt")
    en_top = _i64(inits, [1, 1, 1, work_w], "ent")
    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st_top, en_top, axes4], ["top_black"]),
            helper.make_node("ReduceSum", ["top_black"], ["width_f"], axes=[0, 1, 2, 3], keepdims=0),
            helper.make_node("Cast", ["width_f"], ["width"], to=TensorProto.INT32),
            helper.make_node("Less", [col_idx, "width"], ["valid"]),
            helper.make_node("Mul", ["width", two_i], ["double_width"]),
            helper.make_node("Sub", ["double_width", two_i], ["period"]),
            helper.make_node("Mod", [dist, "period"], ["phase"], fmod=0),
            helper.make_node("Less", ["phase", "width"], ["rising"]),
            helper.make_node("Sub", ["period", "phase"], ["falling_col"]),
            helper.make_node("Where", ["rising", "phase", "falling_col"], ["trace_col"]),
            helper.make_node("Equal", [col_idx, "trace_col"], ["blue"]),
        ]
    )
    return "blue", "valid"


def build_concat_model() -> onnx.ModelProto:
    """Assemble only the two used colors, with generated zero spacer channels."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    blue, valid = _trace_nodes(nodes, inits)
    reps6 = _i64(inits, [1, 6, 1, 1], "reps6")

    nodes.extend(
        [
            helper.make_node("Not", [blue], ["not_blue"]),
            helper.make_node("And", [valid, "not_blue"], ["teal"]),
            helper.make_node("And", [blue, "not_blue"], ["zero"]),
            helper.make_node("Tile", ["zero", reps6], ["zero6"]),
            helper.make_node("Concat", ["zero", blue, "zero6", "teal", "zero"], ["out_bool"], axis=1),
            helper.make_node("Cast", ["out_bool"], ["out_float"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out_float"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - TASK_H, 0],
            ),
        ]
    )
    return _make_model(nodes, inits, "task357_concat")


def build_compact10_concat_model() -> onnx.ModelProto:
    """Assemble the output over the observed 10-column maximum, then pad once."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    blue, valid = _trace_nodes(nodes, inits, TASK_W)

    nodes.extend(
        [
            helper.make_node("Xor", [valid, blue], ["teal"]),
            helper.make_node("And", [blue, "teal"], ["zero"]),
            helper.make_node(
                "Concat",
                ["zero", blue, "zero", "zero", "zero", "zero", "zero", "zero", "teal", "zero"],
                ["out_bool"],
                axis=1,
            ),
            helper.make_node("Cast", ["out_bool"], ["out_float"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out_float"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - TASK_H, W - TASK_W],
            ),
        ]
    )
    return _make_model(nodes, inits, "task357_compact10_concat")


def build_compact10_inner_channels_model() -> onnx.ModelProto:
    """Build only color channels 1..8, then final-pad channels 0 and 9."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    blue, valid = _trace_nodes(nodes, inits, TASK_W)

    nodes.extend(
        [
            helper.make_node("Xor", [valid, blue], ["teal"]),
            helper.make_node("And", [blue, "teal"], ["zero"]),
            helper.make_node(
                "Concat",
                [blue, "zero", "zero", "zero", "zero", "zero", "zero", "teal"],
                ["out_bool"],
                axis=1,
            ),
            helper.make_node("Cast", ["out_bool"], ["out_float"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out_float"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 1, 0, 0, 0, 1, H - TASK_H, W - TASK_W],
            ),
        ]
    )
    return _make_model(nodes, inits, "task357_compact10_inner_channels")


def build_compact10_lookup_inner_channels_model() -> onnx.ModelProto:
    """Use a tiny width-indexed trace lookup for the observed width range 2..10."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    two_i = _i32(inits, 2, "two")
    col_idx = _i32(inits, np.arange(TASK_W, dtype=np.int32).reshape(1, 1, 1, TASK_W), "col")
    st_top = _i64(inits, [0, 0, 0, 0], "stt")
    en_top = _i64(inits, [1, 1, 1, TASK_W], "ent")
    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")

    trace = np.empty((TASK_W - 1, 1, 1, TASK_H, 1), dtype=np.int32)
    dist = np.arange(TASK_H - 1, -1, -1, dtype=np.int32)
    for idx, width in enumerate(range(2, TASK_W + 1)):
        period = 2 * (width - 1)
        phase = dist % period
        trace[idx, 0, 0, :, 0] = np.where(phase < width, phase, period - phase)
    trace_table = _i32(inits, trace, "trace")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st_top, en_top, axes4], ["top_black"]),
            helper.make_node("ReduceSum", ["top_black"], ["width_f"], axes=[0, 1, 2, 3], keepdims=0),
            helper.make_node("Cast", ["width_f"], ["width"], to=TensorProto.INT32),
            helper.make_node("Cast", ["top_black"], ["valid"], to=TensorProto.BOOL),
            helper.make_node("Sub", ["width", two_i], ["width_index"]),
            helper.make_node("Gather", [trace_table, "width_index"], ["trace_col"], axis=0),
            helper.make_node("Equal", [col_idx, "trace_col"], ["blue"]),
            helper.make_node("Xor", ["valid", "blue"], ["teal"]),
            helper.make_node("And", ["blue", "teal"], ["zero"]),
            helper.make_node(
                "Concat",
                ["blue", "zero", "zero", "zero", "zero", "zero", "zero", "teal"],
                ["out_bool"],
                axis=1,
            ),
            helper.make_node("Cast", ["out_bool"], ["out_float"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out_float"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 1, 0, 0, 0, 1, H - TASK_H, W - TASK_W],
            ),
        ]
    )
    return _make_model(nodes, inits, "task357_compact10_lookup_inner_channels")


def build_color_id_model() -> onnx.ModelProto:
    """Alternative: construct color IDs then compare against channel indices."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    blue, valid = _trace_nodes(nodes, inits)
    one_i = _i32(inits, 1, "blue_id")
    eight_i = _i32(inits, 8, "teal_id")
    channels = _i32(inits, np.arange(C, dtype=np.int32).reshape(1, C, 1, 1), "channels")

    nodes.extend(
        [
            helper.make_node("Where", [blue, one_i, eight_i], ["color_ids"]),
            helper.make_node("Equal", [channels, "color_ids"], ["all_channels"]),
            helper.make_node("And", ["all_channels", valid], ["out_bool"]),
            helper.make_node("Cast", ["out_bool"], ["out_float"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out_float"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - TASK_H, 0],
            ),
        ]
    )
    return _make_model(nodes, inits, "task357_color_id")


def build_compact10_color_id_model() -> onnx.ModelProto:
    """Alternative compact-width formulation using color IDs."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    blue, valid = _trace_nodes(nodes, inits, TASK_W)
    one_i = _i32(inits, 1, "blue_id")
    eight_i = _i32(inits, 8, "teal_id")
    channels = _i32(inits, np.arange(C, dtype=np.int32).reshape(1, C, 1, 1), "channels")

    nodes.extend(
        [
            helper.make_node("Where", [blue, one_i, eight_i], ["color_ids"]),
            helper.make_node("Equal", [channels, "color_ids"], ["all_channels"]),
            helper.make_node("And", ["all_channels", valid], ["out_bool"]),
            helper.make_node("Cast", ["out_bool"], ["out_float"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out_float"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - TASK_H, W - TASK_W],
            ),
        ]
    )
    return _make_model(nodes, inits, "task357_compact10_color_id")


def _grid_to_expected(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    arr = np.asarray(grid, dtype=np.int64)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _task_examples() -> list[tuple[str, int, dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    examples: list[tuple[str, int, dict[str, list[list[int]]]]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            examples.append((split, idx, example))
    return examples


def _check_reference() -> None:
    for split, idx, example in _task_examples():
        got = solve(example["input"])
        want = np.asarray(example["output"], dtype=np.int64)
        if not np.array_equal(got, want):
            raise AssertionError(f"reference mismatch on {split}[{idx}]")


def _check_model(model: onnx.ModelProto) -> dict[str, int]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    counts = {"train": 0, "test": 0, "arc-gen": 0}
    for split, idx, example in _task_examples():
        arr = convert_to_numpy(example, "input")
        if arr is None:
            continue
        got = session.run([OUT_NAME], {IN_NAME: arr})[0]
        want = _grid_to_expected(example["output"])
        if not np.array_equal(got > 0.0, want > 0.0):
            raise AssertionError(f"model mismatch on {split}[{idx}]")
        counts[split] += 1
    return counts


def _profile_largest_internal(model: onnx.ModelProto) -> tuple[str | None, int | None, int]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    sizes: dict[str, int] = {}
    for value in inferred.graph.value_info:
        if value.name in {IN_NAME, OUT_NAME} or not value.type.HasField("tensor_type"):
            continue
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        numel = 1
        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_value"):
                numel = 0
                break
            numel *= dim.dim_value
        if numel:
            dtype = helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
            sizes[value.name] = int(numel * np.dtype(dtype).itemsize)
    largest = max(sizes.items(), key=lambda item: item[1], default=(None, None))
    return largest[0], largest[1], len(sizes)


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={counts[key]}" for key in ("train", "test", "arc-gen"))


def _candidate_models() -> Iterable[tuple[str, onnx.ModelProto, str]]:
    yield (
        "compact10_lookup_inner",
        build_compact10_lookup_inner_channels_model(),
        "Width 2..10 trace lookup with int32 columns, only channels 1..8 before final pad.",
    )
    yield (
        "compact10_concat",
        build_compact10_concat_model(),
        "Triangular trace on the observed 10-column work area, repeated zero mask, then final pad.",
    )
    yield (
        "compact10_inner_channels",
        build_compact10_inner_channels_model(),
        "Triangular trace on 10 columns, only channels 1..8 materialized before final pad.",
    )
    yield (
        "compact10_color_id",
        build_compact10_color_id_model(),
        "Compact-width trace converted through color IDs and channel equality.",
    )
    yield (
        "concat",
        build_concat_model(),
        "Triangular trace plus concat of channel 1, generated zero channels, and channel 8.",
    )
    yield (
        "color_id",
        build_color_id_model(),
        "Triangular trace converted through compact color IDs and channel equality.",
    )


def main() -> None:
    _check_reference()

    results: list[tuple[int, str, onnx.ModelProto, dict[str, Any], str, str | None, int | None, int]] = []
    for label, model, note in _candidate_models():
        tmp_path = OUT_DIR / f"{TASK_ID}_{label}.onnx"
        onnx.save(model, tmp_path)
        try:
            counts = _check_model(model)
            scored = score_file(tmp_path)
            largest_name, largest_bytes, tensor_count = _profile_largest_internal(model)
            score_text = f"{scored['score']:.6f}" if scored["score"] is not None else "INVALID"
            print(
                f"{label:<12} valid={scored['valid']} passes={_format_counts(counts)} "
                f"memory={scored['memory']} params={scored['params']} cost={scored['cost']} "
                f"score={score_text} tensors={tensor_count} "
                f"largest={largest_name}:{largest_bytes} note={note}"
            )
            if scored["valid"]:
                results.append((int(scored["cost"]), label, model, scored, note, largest_name, largest_bytes, tensor_count))
        finally:
            tmp_path.unlink(missing_ok=True)

    if not results:
        raise SystemExit("no valid correct candidate")

    _cost, label, model, scored, note, largest_name, largest_bytes, tensor_count = min(results, key=lambda item: item[0])
    onnx.save(model, BEST_PATH)
    print()
    print(f"kept:    {label}")
    print(f"wrote:   {BEST_PATH}")
    print(f"rule:    teal field with upward blue trace and horizontal reflections")
    print(f"method:  {note}")
    print(f"memory:  {scored['memory']}")
    print(f"params:  {scored['params']}")
    print(f"cost:    {scored['cost']}")
    print(f"score:   {scored['score']:.6f}")
    print(f"tensors: {tensor_count}")
    print(f"largest: {largest_name} ({largest_bytes} bytes)")


if __name__ == "__main__":
    main()
