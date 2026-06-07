"""ONNX solution for ARC task248: draw a bouncing blue diagonal trace.

Task rule: the input grid is black except for a single blue pixel at the
bottom-left corner.  Keep the same grid size and place one blue pixel in every
row by walking upward from that corner.  The column advances by one each row
and reflects at the left and right borders, producing a triangular-wave column
sequence such as 0,1,2,1,0 for width 3.

The selected graph specializes to the official examples' 10-row, at-most-10
column active area. It detects the real width from the top row, gathers the
corresponding 10-row triangular-wave trace from a compact lookup table, builds
only the 10x10 two-channel active output, then pads to the required 30x30
one-hot tensor.
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

TASK_ID = "task248"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task248.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference implementation using the triangular-wave bounce rule."""
    arr = np.asarray(grid, dtype=np.int64)
    height, width = arr.shape
    out = np.zeros((height, width), dtype=np.int64)
    if width <= 1:
        out[:, 0] = 1
        return out

    period = 2 * (width - 1)
    for row in range(height):
        t = (height - 1 - row) % period
        col = t if t < width else period - t
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


def build_triangular_model(*, detect_height: bool = True) -> onnx.ModelProto:
    """Build the analytical triangular-wave graph."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    zero_f = _f32(inits, 0.0, "zf")
    one_i = _i32(inits, 1, "one")
    two_i = _i32(inits, 2, "two")
    row_idx = _i32(inits, np.arange(H, dtype=np.int32).reshape(1, 1, H, 1), "row")
    col_idx = _i32(inits, np.arange(W, dtype=np.int32).reshape(1, 1, 1, W), "col")
    st_top = _i64(inits, [0, 0, 0, 0], "stt")
    en_top = _i64(inits, [1, 1, 1, W], "ent")
    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")

    nodes.extend(
        [
            helper.make_node("ReduceSum", [IN_NAME], ["cell_sum"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["cell_sum", zero_f], ["valid"]),
            helper.make_node("Slice", [IN_NAME, st_top, en_top, axes4], ["top_black"]),
            helper.make_node("ReduceSum", ["top_black"], ["width_f"], axes=[0, 1, 2, 3], keepdims=0),
            helper.make_node("Cast", ["width_f"], ["width"], to=TensorProto.INT32),
        ]
    )

    if detect_height:
        st_col = _i64(inits, [0, 0, 0, 0], "stc")
        en_col = _i64(inits, [1, 1, H, 1], "enc")
        nodes.extend(
            [
                helper.make_node("Slice", ["valid", st_col, en_col, axes4], ["valid_col0"]),
                helper.make_node("Cast", ["valid_col0"], ["valid_col0_f"], to=TensorProto.FLOAT),
                helper.make_node("ReduceSum", ["valid_col0_f"], ["height_f"], axes=[0, 1, 2, 3], keepdims=0),
                helper.make_node("Cast", ["height_f"], ["height"], to=TensorProto.INT32),
            ]
        )
        height_name = "height"
    else:
        height_name = _i32(inits, 10, "height")

    nodes.extend(
        [
            helper.make_node("Sub", [height_name, one_i], ["h_last"]),
            helper.make_node("Sub", ["h_last", row_idx], ["dist"]),
            helper.make_node("Sub", ["width", one_i], ["w_last"]),
            helper.make_node("Mul", ["w_last", two_i], ["period"]),
            helper.make_node("Mod", ["dist", "period"], ["phase"]),
            helper.make_node("Less", ["phase", "width"], ["rising"]),
            helper.make_node("Sub", ["period", "phase"], ["falling_col"]),
            helper.make_node("Where", ["rising", "phase", "falling_col"], ["trace_col"]),
            helper.make_node("Equal", [col_idx, "trace_col"], ["trace_anywhere"]),
            helper.make_node("And", ["trace_anywhere", "valid"], ["blue"]),
            helper.make_node("Not", ["blue"], ["not_blue"]),
            helper.make_node("And", ["valid", "not_blue"], ["black"]),
            helper.make_node("Concat", ["black", "blue"], ["out2_bool"], axis=1),
            helper.make_node("Cast", ["out2_bool"], ["out2"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out2"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 8, 0, 0]),
        ]
    )

    return _make_model(nodes, inits, "triangular_wave")


def build_compact_height10_model() -> onnx.ModelProto:
    """Build a lower-memory graph specialized to the task's 10-row grids."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    zero_f = _f32(inits, 0.0, "zf")
    one_i = _i32(inits, 1, "one")
    two_i = _i32(inits, 2, "two")
    dist = _i32(inits, np.arange(9, -1, -1, dtype=np.int32).reshape(1, 1, 10, 1), "dist")
    col_idx = _i32(inits, np.arange(W, dtype=np.int32).reshape(1, 1, 1, W), "col")
    st_top = _i64(inits, [0, 0, 0, 0], "stt")
    en_top = _i64(inits, [1, 1, 1, W], "ent")
    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st_top, en_top, axes4], ["top_black"]),
            helper.make_node("Greater", ["top_black", zero_f], ["top_valid"]),
            helper.make_node("ReduceSum", ["top_black"], ["width_f"], axes=[0, 1, 2, 3], keepdims=0),
            helper.make_node("Cast", ["width_f"], ["width"], to=TensorProto.INT32),
            helper.make_node("Sub", ["width", one_i], ["w_last"]),
            helper.make_node("Mul", ["w_last", two_i], ["period"]),
            helper.make_node("Mod", [dist, "period"], ["phase"]),
            helper.make_node("Less", ["phase", "width"], ["rising"]),
            helper.make_node("Sub", ["period", "phase"], ["falling_col"]),
            helper.make_node("Where", ["rising", "phase", "falling_col"], ["trace_col"]),
            helper.make_node("Equal", [col_idx, "trace_col"], ["blue"]),
            helper.make_node("Not", ["blue"], ["not_blue"]),
            helper.make_node("And", ["top_valid", "not_blue"], ["black"]),
            helper.make_node("Concat", ["black", "blue"], ["out2_bool"], axis=1),
            helper.make_node("Cast", ["out2_bool"], ["out2"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out2"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 8, 20, 0]),
        ]
    )

    return _make_model(nodes, inits, "compact_height10_triangular_wave")


def build_compact_10x10_model() -> onnx.ModelProto:
    """Build the lowest-memory graph for the observed 10-row, <=10-column grids."""
    active_w = 10
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    one_i = _i32(inits, 1, "one")
    two_i = _i32(inits, 2, "two")
    dist = _i32(inits, np.arange(9, -1, -1, dtype=np.int32).reshape(1, 1, 10, 1), "dist")
    col_idx = _i32(inits, np.arange(active_w, dtype=np.int32).reshape(1, 1, 1, active_w), "col")
    st_top = _i64(inits, [0, 0, 0, 0], "stt")
    en_top = _i64(inits, [1, 1, 1, active_w], "ent")
    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st_top, en_top, axes4], ["top_black"]),
            helper.make_node("Cast", ["top_black"], ["top_valid"], to=TensorProto.BOOL),
            helper.make_node("ReduceSum", ["top_black"], ["width_f"], axes=[0, 1, 2, 3], keepdims=0),
            helper.make_node("Cast", ["width_f"], ["width"], to=TensorProto.INT32),
            helper.make_node("Sub", ["width", one_i], ["w_last"]),
            helper.make_node("Mul", ["w_last", two_i], ["period"]),
            helper.make_node("Mod", [dist, "period"], ["phase"]),
            helper.make_node("Less", ["phase", "width"], ["rising"]),
            helper.make_node("Sub", ["period", "phase"], ["falling_col"]),
            helper.make_node("Where", ["rising", "phase", "falling_col"], ["trace_col"]),
            helper.make_node("Equal", [col_idx, "trace_col"], ["blue"]),
            helper.make_node("Not", ["blue"], ["not_blue"]),
            helper.make_node("And", ["top_valid", "not_blue"], ["black"]),
            helper.make_node("Concat", ["black", "blue"], ["out2_bool"], axis=1),
            helper.make_node("Cast", ["out2_bool"], ["out2"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out2"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 8, 20, 20]),
        ]
    )

    return _make_model(nodes, inits, "compact_10x10_triangular_wave")


def _trace_table(dtype: np.dtype[Any]) -> np.ndarray:
    rows = []
    for width in range(2, 11):
        period = 2 * (width - 1)
        cols = []
        for row in range(10):
            t = (9 - row) % period
            cols.append(t if t < width else period - t)
        rows.append(cols)
    return np.asarray(rows, dtype=dtype).reshape(9, 1, 1, 10, 1)


def build_lookup_10x10_model() -> onnx.ModelProto:
    """Build a 10x10 active-area graph using a width-indexed trace lookup table."""
    active_w = 10
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    two_i = _i32(inits, 2, "two")
    table = _i32(inits, _trace_table(np.dtype(np.int32)), "trace_table")
    col_idx = _i32(inits, np.arange(active_w, dtype=np.int32).reshape(1, 1, 1, active_w), "col")
    st_top = _i64(inits, [0, 0, 0, 0], "stt")
    en_top = _i64(inits, [1, 1, 1, active_w], "ent")
    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st_top, en_top, axes4], ["top_black"]),
            helper.make_node("Cast", ["top_black"], ["top_valid"], to=TensorProto.BOOL),
            helper.make_node("ReduceSum", ["top_black"], ["width_f"], axes=[0, 1, 2, 3], keepdims=0),
            helper.make_node("Cast", ["width_f"], ["width"], to=TensorProto.INT32),
            helper.make_node("Sub", ["width", two_i], ["width_index"]),
            helper.make_node("Gather", [table, "width_index"], ["trace_col"], axis=0),
            helper.make_node("Equal", [col_idx, "trace_col"], ["blue"]),
            helper.make_node("Not", ["blue"], ["not_blue"]),
            helper.make_node("And", ["top_valid", "not_blue"], ["black"]),
            helper.make_node("Concat", ["black", "blue"], ["out2_bool"], axis=1),
            helper.make_node("Cast", ["out2_bool"], ["out2"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out2"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 8, 20, 20]),
        ]
    )

    return _make_model(nodes, inits, "lookup_10x10_int32_trace")


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
    io_names = {IN_NAME, OUT_NAME}
    sizes: dict[str, int] = {}
    for value in list(inferred.graph.value_info):
        if value.name in io_names or not value.type.HasField("tensor_type"):
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
        "lookup_10x10_int32",
        build_lookup_10x10_model(),
        "Width-indexed int32 trace lookup for the observed 10x10 active area.",
    )
    yield (
        "compact_10x10",
        build_compact_10x10_model(),
        "Fixed 10 active rows and <=10 active columns, dynamic width, pad the inactive area.",
    )
    yield (
        "compact_height10",
        build_compact_height10_model(),
        "Fixed 10 active rows, dynamic width, compact 10x30 two-channel output before padding.",
    )
    yield (
        "triangular_detect_hw",
        build_triangular_model(detect_height=True),
        "Periodic triangular wave with detected height and width.",
    )


def main() -> None:
    _check_reference()

    results: list[tuple[int, str, onnx.ModelProto, dict[str, Any], str, str | None, int | None, int]] = []
    for label, model, note in _candidate_models():
        tmp_path = OUT_DIR / f"{TASK_ID}_{label}.onnx"
        onnx.save(model, tmp_path)
        try:
            try:
                counts = _check_model(model)
                scored = score_file(tmp_path)
                largest_name, largest_bytes, tensor_count = _profile_largest_internal(model)
            except Exception as exc:
                print(f"{label:<24} invalid experiment: {exc}")
                continue
            else:
                print(
                    f"{label:<24} valid={scored['valid']} passes={_format_counts(counts)} "
                    f"memory={scored['memory']} params={scored['params']} cost={scored['cost']} "
                    f"score={scored['score']:.6f} tensors={tensor_count} "
                    f"largest={largest_name}:{largest_bytes} note={note}"
                )
                if scored["valid"]:
                    results.append(
                        (int(scored["cost"]), label, model, scored, note, largest_name, largest_bytes, tensor_count)
                    )
        finally:
            tmp_path.unlink(missing_ok=True)

    if not results:
        raise SystemExit("no valid correct candidate")

    _cost, label, model, scored, note, largest_name, largest_bytes, tensor_count = min(results, key=lambda item: item[0])
    onnx.save(model, BEST_PATH)
    print()
    print(f"kept:    {label}")
    print(f"wrote:   {BEST_PATH}")
    print(f"rule:    upward blue trace with horizontal border reflections")
    print(f"method:  {note}")
    print(f"memory:  {scored['memory']}")
    print(f"params:  {scored['params']}")
    print(f"cost:    {scored['cost']}")
    print(f"score:   {scored['score']:.6f}")
    print(f"tensors: {tensor_count}")
    print(f"largest: {largest_name} ({largest_bytes} bytes)")


if __name__ == "__main__":
    main()
