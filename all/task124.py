"""Ultra-compact ONNX for ARC task124 periodic row continuation.

Task rule: the visible Hx10 input is the prefix of a 10-row colored pattern on
black background. Continue the same row sequence down to height 10, preserving
the single foreground color. The sequence is generated from the first 1, 2, or
3 rows; every period may either repeat in place or shift right by 1 or 2 cells,
with cells shifted beyond column 9 becoming black. Output is exactly 10x10 and
the NeuroGolf padding outside that area stays all-zero.

ONNX approach: detect the single foreground color from row 0, derive the 10x10
shape mask from the black channel, then build six flat 1-channel Gather
candidates. Pick the first candidate whose generated visible prefix matches the
input rows, broadcast the chosen mask onto the detected color channel, synthesize
black as no-foreground, cast the compact 10x10 result to float, and Pad directly
to the required 30x30 output.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_all_onnx import verify_correctness  # noqa: E402
from score_model import calculate_memory, calculate_params, print_report, score_file  # noqa: E402

TASK_ID = "task124"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
ROOT_PATH = ROOT / f"{TASK_ID}.onnx"

C = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10
VARIANTS = [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0)]


def _init(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _slice(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    source: str,
    out: str,
    starts: list[int],
    ends: list[int],
    axes: list[int],
) -> str:
    s = _init(inits, f"{out}_s", np.asarray(starts, dtype=np.int64))
    e = _init(inits, f"{out}_e", np.asarray(ends, dtype=np.int64))
    a = _init(inits, f"{out}_a", np.asarray(axes, dtype=np.int64))
    nodes.append(helper.make_node("Slice", [source, s, e, a], [out]))
    return out


def _reshape(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], source: str, out: str, shape: list[int]) -> str:
    shp = _init(inits, f"{out}_shape", np.asarray(shape, dtype=np.int64))
    nodes.append(helper.make_node("Reshape", [source, shp], [out]))
    return out


def _candidate_indices(period: int, dx: int) -> np.ndarray:
    idx = np.zeros(100, dtype=np.int32)
    for row in range(10):
        src_row = row % period
        shift = (row // period) * dx
        for col in range(10):
            src_col = col - shift
            flat = row * 10 + col
            idx[flat] = src_row * 10 + src_col if 0 <= src_col < 10 else 100
    return idx


def _build_candidate(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    flat_fg_pad: str,
    period: int,
    dx: int,
    name: str,
    idx_name: str | None = None,
) -> str:
    idx = idx_name or _init(inits, f"{name}_idx", _candidate_indices(period, dx))
    nodes.append(helper.make_node("Gather", [flat_fg_pad, idx], [f"{name}_flat"], axis=2))
    channels = 9 if flat_fg_pad == "flat_fg_pad" else 1
    return _reshape(nodes, inits, f"{name}_flat", name, [1, channels, 10, 10])


def _build_candidate_flat(
    nodes: list[onnx.NodeProto],
    flat_mask_pad: str,
    name: str,
    idx_name: str,
) -> str:
    nodes.append(helper.make_node("Gather", [flat_mask_pad, idx_name], [name], axis=2))
    return name


def _match_candidate(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    cand: str,
    fg: str,
    valid_mask: str,
    name: str,
) -> str:
    zero = _init(inits, f"{name}_zero", np.asarray([0], dtype=np.int32))
    nodes.extend(
        [
            helper.make_node("Xor", [cand, fg], [f"{name}_neq"]),
            helper.make_node("And", [f"{name}_neq", valid_mask], [f"{name}_bad"]),
            helper.make_node("Cast", [f"{name}_bad"], [f"{name}_bad_i"], to=TensorProto.INT32),
            helper.make_node("ReduceSum", [f"{name}_bad_i"], [f"{name}_bad_sum"], axes=[1, 2], keepdims=0),
            helper.make_node("Equal", [f"{name}_bad_sum", zero], [name]),
        ]
    )
    return name


def _finish(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
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


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    zf = _init(inits, "zf", np.asarray([0.0], dtype=np.float32))
    false_mask_col = _init(inits, "false_mask_col", np.zeros((1, 1, 1), dtype=bool))

    fg_first = _slice(nodes, inits, IN_NAME, "fg_first", [1, 0, 0], [10, 1, 10], [1, 2, 3])
    black10 = _slice(nodes, inits, IN_NAME, "black10", [0, 0, 0], [1, 10, 10], [1, 2, 3])
    nodes.extend(
        [
            helper.make_node("ReduceSum", [fg_first], ["color_sum"], axes=[2, 3], keepdims=0),
            helper.make_node("Greater", ["color_sum", zf], ["color_present"]),
            helper.make_node("Greater", [black10, zf], ["black_in"]),
            helper.make_node("Not", ["black_in"], ["not_black"]),
            helper.make_node("ReduceSum", [black10], ["row_sum"], axes=[1, 3], keepdims=0),
            helper.make_node("Greater", ["row_sum", zf], ["valid_rows"]),
        ]
    )
    color_present = _reshape(nodes, inits, "color_present", "color_present_4d", [1, 9, 1, 1])
    valid_rows = _reshape(nodes, inits, "valid_rows", "valid_rows_4d", [1, 1, 10, 1])
    nodes.append(helper.make_node("And", ["not_black", valid_rows], ["fg_mask"]))
    flat_mask = _reshape(nodes, inits, "fg_mask", "flat_mask", [1, 1, 100])
    row_idx = _init(inits, "row_idx", np.repeat(np.arange(10, dtype=np.int32), 10))
    nodes.extend(
        [
            helper.make_node("Concat", ["flat_mask", false_mask_col], ["flat_mask_pad"], axis=2),
            helper.make_node("Gather", ["valid_rows", row_idx], ["valid_flat_2d"], axis=1),
        ]
    )

    idx_names: list[str] = []
    matches: list[str] = []
    for index, (period, dx) in enumerate(VARIANTS):
        idx = _init(inits, f"cand{index}_idx", _candidate_indices(period, dx))
        idx_names.append(idx)
        if index < len(VARIANTS) - 1:
            mask_cand = _build_candidate_flat(nodes, "flat_mask_pad", f"mask_cand{index}", idx)
            matches.append(_match_candidate(nodes, inits, mask_cand, "flat_mask", "valid_flat_2d", f"match{index}"))

    selected_idx = idx_names[-1]
    for index in range(len(idx_names) - 2, -1, -1):
        out = f"sel_idx{index}"
        nodes.append(helper.make_node("Where", [matches[index], idx_names[index], selected_idx], [out]))
        selected_idx = out

    nodes.append(helper.make_node("Gather", ["flat_mask_pad", selected_idx], ["selected_mask_flat"], axis=2))
    selected_mask = _reshape(nodes, inits, "selected_mask_flat", "selected_mask", [1, 1, 10, 10])
    nodes.extend(
        [
            helper.make_node("And", [selected_mask, color_present], ["selected"]),
            helper.make_node("Not", [selected_mask], ["black"]),
            helper.make_node("Concat", ["black", "selected"], ["out_bool"], axis=1),
            helper.make_node("Cast", ["out_bool"], ["out10"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out10"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, 20, 20],
                value=0.0,
            ),
        ]
    )

    return _finish(nodes, inits, "task124_periodic_rows")


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            arr[0, color, r, c] = 1.0
    return arr


def _decode(out: np.ndarray) -> np.ndarray:
    active = out[0, :, :10, :10] > 0
    colors = np.argmax(active, axis=0).astype(np.int64)
    invalid = active.sum(axis=0) != 1
    colors[invalid] = -1
    return colors


def verify_examples(path: Path) -> None:
    data = json.loads(DATA_PATH.read_text())
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])

    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            total += 1
            got = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(example["input"])})[0]
            pred = _decode(got)
            expected = np.asarray(example["output"], dtype=np.int64)
            if not np.array_equal(pred, expected):
                raise AssertionError(f"{split}[{idx}] failed")
            if np.any(got[0, :, 10:, :] > 0) or np.any(got[0, :, :, 10:] > 0):
                raise AssertionError(f"{split}[{idx}] has nonzero padding")
    print(f"local verifier:   {total}/{total}")


def print_inferred_stats(model: onnx.ModelProto) -> None:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    init_params = calculate_params(model)
    print(f"initializer params: {init_params}")
    print("inferred internal tensors:")
    total = 0
    for value in inferred.graph.value_info:
        tensor_type = value.type.tensor_type
        shape = [dim.dim_value for dim in tensor_type.shape.dim]
        dtype = helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        mem = math.prod(shape) * np.dtype(dtype).itemsize
        total += mem
        print(f"  {value.name:<18} {str(np.dtype(dtype)):<7} {shape} bytes={mem}")
    print(f"static internal memory estimate: {total}")


def profiled_memory(model: onnx.ModelProto, path: Path) -> int | None:
    data = json.loads(DATA_PATH.read_text())
    inputs = []
    for split in ("train", "test", "arc-gen"):
        inputs.extend(_grid_to_onehot(example["input"]) for example in data.get(split, []))
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(Path(tempfile.gettempdir()) / f"ng_{TASK_ID}")
    session = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    for arr in inputs:
        session.run([OUT_NAME], {IN_NAME: arr})
    trace = session.end_profiling()
    return calculate_memory(model, trace)


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    shutil.copyfile(BEST_PATH, ROOT_PATH)
    print(f"saved:            {BEST_PATH}")
    print(f"saved:            {ROOT_PATH}")
    verify_examples(BEST_PATH)

    ok, summary, _, _ = verify_correctness(BEST_PATH)
    print(f"repo verifier:    {summary} ({'ok' if ok else 'wrong'})")
    print_inferred_stats(model)
    print(f"profiled memory:  {profiled_memory(model, BEST_PATH)}")
    print_report(score_file(BEST_PATH))


if __name__ == "__main__":
    main()
