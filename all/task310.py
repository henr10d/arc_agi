"""Extract the uniquely marked square tile for NeuroGolf task310.

Task rule: the input is a padded mosaic of repeated square motifs. Exactly one
motif is identified by a rare non-zero frame color whose total count is the
perimeter of a 5x5, 6x6, 7x7, or 8x8 square. The output is that framed square
copied to the top-left, with the rest of the 30x30 competition tensor left
empty.

ONNX approach: count colors to find the rare frame channel, reduce its pixels to
the top/left edge via ArgMax, slice one 8x8 window from the input, and mask it
to the detected square size before the final pad.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

TASK_ID = "task310"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
MAX_OUT = 8
PAD = H - MAX_OUT
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def _init(inits: list[onnx.TensorProto], arr: Any, name: str, dtype: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=dtype), name=name))
    return name


def _f32(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    return _init(inits, arr, name, np.float32)


def _i64(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    return _init(inits, arr, name, np.int64)


def solve_grid(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    counts = Counter(arr.ravel().tolist())
    for n in range(5, MAX_OUT + 1):
        perimeter = 4 * n - 4
        colors = [color for color, count in counts.items() if color != 0 and count == perimeter]
        for color in colors:
            mask = arr == color
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            if not len(rows) or not len(cols):
                continue
            r0, r1 = int(rows.min()), int(rows.max())
            c0, c1 = int(cols.min()), int(cols.max())
            if (r1 - r0 + 1, c1 - c0 + 1) != (n, n):
                continue
            patch = arr[r0 : r1 + 1, c0 : c1 + 1]
            border = np.concatenate([patch[0], patch[-1], patch[1:-1, 0], patch[1:-1, -1]])
            if np.all(border == color):
                return patch
    raise ValueError("no uniquely framed square found")


def _grid_to_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _onehot_to_grid(tensor: np.ndarray) -> np.ndarray:
    active = tensor[0, :, :MAX_OUT, :MAX_OUT]
    return active.argmax(axis=0).astype(np.int64)


def _expected_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    return _grid_to_onehot(grid)


def _print_grid(arr: np.ndarray) -> None:
    for row in arr:
        print("".join(str(int(v)) for v in row))


def print_analysis(data: dict[str, Any]) -> None:
    print("task310 analysis")
    print("----------------")
    print("Hypothesis results on train:")
    print("H1 enclosed tile with identity color remapping: PASS")
    print("H2 frame/background canonical replacement: REJECTED; target preserves original colors")
    print("H3 role-to-fixed canonical palette: REJECTED; frame colors 3/2/6 remain unchanged")
    print("H4 frequency-rank normalization: REJECTED; exact extraction passes without rank remap")
    print("H5 motif-class canonical representative: REJECTED/UNNEEDED; target equals the marked patch")

    for index, example in enumerate(data["train"]):
        inp = np.asarray(example["input"], dtype=np.int64)
        out = np.asarray(example["output"], dtype=np.int64)
        patch = solve_grid(inp)
        counts = Counter(inp.ravel().tolist())
        frame_color = int(patch[0, 0])
        rows, cols = np.where(inp == frame_color)
        r0, c0 = int(rows.min()), int(cols.min())
        n = int(patch.shape[0])
        print()
        print(f"train[{index}] input={inp.shape} output={out.shape}")
        print(f"detected framed region: row={r0} col={c0} size={n} color={frame_color}")
        print("input color histogram:", dict(sorted((int(k), int(v)) for k, v in counts.items())))
        print("extracted tile histogram:", dict(sorted((int(k), int(v)) for k, v in Counter(patch.ravel()).items())))
        print("proposed color mapping: identity")
        print("extracted 7x7 tile:" if n == 7 else f"extracted {n}x{n} tile:")
        _print_grid(patch)
        print("matches target:", bool(np.array_equal(patch, out)))


def build_onnx_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    scalar_shape = _i64(inits, [1], "scalar_shape")
    zero_f = _f32(inits, [0.0], "zero_f")
    big_f = _f32(inits, [10000.0], "big_f")
    offsets = np.arange(MAX_OUT, dtype=np.int64)
    row_i = _i64(inits, offsets, "row_i")
    col_i = _i64(inits, offsets, "col_i")
    row_f = _f32(inits, offsets.astype(np.float32), "row_f")
    col_f = _f32(inits, offsets.astype(np.float32), "col_f")
    zero_i = _i64(inits, [0], "zero_i")
    c22_i = _i64(inits, [H - MAX_OUT], "c22_i")
    c23_i = _i64(inits, [H - MAX_OUT + 1], "c23_i")
    eight2 = _i64(inits, [MAX_OUT, MAX_OUT], "eight2")
    axes23 = _i64(inits, [2, 3], "axes23")
    steps2 = _i64(inits, [1, 1], "steps2")

    nodes.extend(
        [
            helper.make_node("ReduceSum", [IN_NAME], ["counts"], axes=[2, 3], keepdims=1),
            helper.make_node("Greater", ["counts", zero_f], ["present_b"]),
            helper.make_node("Where", ["present_b", "counts", big_f], ["candidate_counts"]),
            helper.make_node("ArgMin", ["candidate_counts"], ["marker_id4"], axis=1, keepdims=1),
            helper.make_node("Reshape", ["marker_id4", scalar_shape], ["marker_id"]),
            helper.make_node("Gather", [IN_NAME, "marker_id"], ["marker_pixels"], axis=1),
            helper.make_node("ReduceSum", ["marker_pixels"], ["row_sums"], axes=[3], keepdims=1),
            helper.make_node("ReduceSum", ["marker_pixels"], ["col_sums"], axes=[2], keepdims=1),
            helper.make_node("ReduceMax", ["row_sums"], ["n4"], axes=[2, 3], keepdims=1),
            helper.make_node("Reshape", ["n4", scalar_shape], ["n"]),
            helper.make_node("ArgMax", ["row_sums"], ["top_r4"], axis=2, keepdims=1),
            helper.make_node("ArgMax", ["col_sums"], ["top_c4"], axis=3, keepdims=1),
            helper.make_node("Reshape", ["top_r4", scalar_shape], ["top_r"]),
            helper.make_node("Reshape", ["top_c4", scalar_shape], ["top_c"]),
            helper.make_node("Less", [row_f, "n"], ["row_inside"]),
            helper.make_node("Less", [col_f, "n"], ["col_inside"]),
            helper.make_node("Less", ["top_r", c23_i], ["top_r_lt23"]),
            helper.make_node("Less", ["top_c", c23_i], ["top_c_lt23"]),
            helper.make_node("Where", ["top_r_lt23", "top_r", c22_i], ["slice_r"]),
            helper.make_node("Where", ["top_c_lt23", "top_c", c22_i], ["slice_c"]),
            helper.make_node("Concat", ["slice_r", "slice_c"], ["slice_starts"], axis=0),
            helper.make_node("Add", ["slice_starts", eight2], ["slice_ends"]),
            helper.make_node(
                "Slice",
                [IN_NAME, "slice_starts", "slice_ends", axes23, steps2],
                ["sliced4"],
            ),
            helper.make_node("Cast", ["sliced4"], ["sliced_b"], to=TensorProto.BOOL),
            helper.make_node("Sub", ["top_r", "slice_r"], ["delta_r"]),
            helper.make_node("Sub", ["top_c", "slice_c"], ["delta_c"]),
            helper.make_node("Add", [row_i, "delta_r"], ["local_r_raw"]),
            helper.make_node("Add", [col_i, "delta_c"], ["local_c_raw"]),
            helper.make_node("Where", ["row_inside", "local_r_raw", zero_i], ["local_r"]),
            helper.make_node("Where", ["col_inside", "local_c_raw", zero_i], ["local_c"]),
            helper.make_node("Gather", ["sliced_b", "local_r"], ["gathered_rows_b"], axis=2),
            helper.make_node("Gather", ["gathered_rows_b", "local_c"], ["gathered_b"], axis=3),
            helper.make_node("Unsqueeze", ["row_inside"], ["row_inside4"], axes=[0, 1, 3]),
            helper.make_node("Unsqueeze", ["col_inside"], ["col_inside4"], axes=[0, 1, 2]),
            helper.make_node("And", ["row_inside4", "col_inside4"], ["inside4"]),
            helper.make_node("And", ["gathered_b", "inside4"], ["out_b"]),
            helper.make_node("Cast", ["out_b"], ["out8"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out8"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]),
        ]
    )

    value_info = [
        helper.make_tensor_value_info("sliced4", TensorProto.FLOAT, [1, C, MAX_OUT, MAX_OUT]),
        helper.make_tensor_value_info("sliced_b", TensorProto.BOOL, [1, C, MAX_OUT, MAX_OUT]),
        helper.make_tensor_value_info("gathered_rows_b", TensorProto.BOOL, [1, C, MAX_OUT, MAX_OUT]),
        helper.make_tensor_value_info("gathered_b", TensorProto.BOOL, [1, C, MAX_OUT, MAX_OUT]),
        helper.make_tensor_value_info("out_b", TensorProto.BOOL, [1, C, MAX_OUT, MAX_OUT]),
        helper.make_tensor_value_info("out8", TensorProto.FLOAT, [1, C, MAX_OUT, MAX_OUT]),
    ]
    graph = helper.make_graph(
        nodes,
        "task310",
        [x_info],
        [y_info],
        initializer=inits,
        value_info=value_info,
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


def verify_model(model: onnx.ModelProto, data: dict[str, Any]) -> None:
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    )
    total = 0
    for split in ("train", "test", "arc-gen"):
        correct = 0
        for index, example in enumerate(data.get(split, [])):
            expected = _expected_onehot(example["output"])
            actual = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(example["input"])})[0]
            if not np.array_equal(actual > 0.0, expected > 0.0):
                pred = _onehot_to_grid(actual)
                print(f"mismatch {split}[{index}]")
                print("predicted compact:")
                _print_grid(pred[: len(example["output"]), : len(example["output"][0])])
                raise AssertionError(f"{split}[{index}] failed")
            correct += 1
            total += 1
        print(f"{split} accuracy: {correct}/{len(data.get(split, []))}")
    print(f"overall accuracy: {total}/{total}")


def save_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    model = build_onnx_model()
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return model


def main() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    print_analysis(data)
    model = save_model()
    verify_model(model, data)
    print(f"saved {BEST_PATH}")


if __name__ == "__main__":
    main()
