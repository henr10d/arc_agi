"""ONNX for ARC task365: crop the component with the most red cells.

Task rule in the local JSON: each 10x10 input contains two or three solid
non-black rectangular components made from cyan/turquoise (8), blue (1), and
red (2).  The desired output is the tight crop of the single component that
contains the most red cells, preserving its colors.  Other components are
partial distractors; overlaying all component-local crops introduces extra
marked cells on every local example.  Padding outside the cropped output stays
inactive for the NeuroGolf one-hot tensor.

ONNX approach: derive the foreground mask from the 10x10 background channel,
propagate one numeric component label through that mask with five 3x3 min-pool
steps, gather the at-most ten red-cell labels with TopK, count matching red
labels with bool masks, choose the max-red label, compute that component's bbox,
gather dynamic 6x6 crops only for colors 1, 2, and 8, concatenate zero channels,
mask cells outside the bbox, and pad to 30x30.
"""

from __future__ import annotations

import json
import sys
from collections import deque
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

TASK_ID = "task365"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
N = 10
OUT_N = 6
SEEDS = N * N
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10
IN_NAME = "input"
OUT_NAME = "output"
DILATION_STEPS = 5


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(grid.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 0 or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < grid.shape[0]
                        and 0 <= nc < grid.shape[1]
                        and grid[nr, nc] != 0
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference solver: crop the component with the largest count of color 2."""
    arr = np.asarray(grid, dtype=np.int64)
    comp = max(_components(arr), key=lambda cells: sum(arr[r, c] == 2 for r, c in cells))
    coords = np.asarray(comp, dtype=np.int64)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return arr[r0:r1, c0:c1]


def overlay_mismatch_count() -> int:
    """Diagnostic for the tempting but wrong all-component overlay hypothesis."""
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    wrong = 0
    for split in ("train", "test", "arc-gen"):
        for example in data[split]:
            grid = np.asarray(example["input"], dtype=np.int64)
            expected = np.asarray(example["output"], dtype=np.int64)
            crops = []
            for comp in _components(grid):
                coords = np.asarray(comp, dtype=np.int64)
                r0, c0 = coords.min(axis=0)
                r1, c1 = coords.max(axis=0) + 1
                crops.append(grid[r0:r1, c0:c1])
            h, w = expected.shape
            over = np.full((h, w), 8, dtype=np.int64)
            for r in range(h):
                for c in range(w):
                    vals = [crop[r, c] for crop in crops if r < crop.shape[0] and c < crop.shape[1]]
                    if 2 in vals:
                        over[r, c] = 2
                    elif 1 in vals:
                        over[r, c] = 1
            wrong += int(not np.array_equal(over, expected))
    return wrong


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    bg_start = _i64(inits, [0, 0, 0, 0], "bg_start")
    bg_end = _i64(inits, [1, 1, N, N], "bg_end")
    blue_start = _i64(inits, [0, 1, 0, 0], "blue_start")
    blue_end = _i64(inits, [1, 2, N, N], "blue_end")
    red_start = _i64(inits, [0, 2, 0, 0], "red_start")
    red_end = _i64(inits, [1, 3, N, N], "red_end")
    cyan_start = _i64(inits, [0, 8, 0, 0], "cyan_start")
    cyan_end = _i64(inits, [1, 9, N, N], "cyan_end")
    rev = _i64(inits, np.arange(N - 1, -1, -1), "rev")
    offsets = _i64(inits, np.arange(OUT_N), "offsets")
    zero_offsets = _i64(inits, np.zeros(OUT_N, dtype=np.int64), "zero_offsets")
    one_i = _i64(inits, [1], "one_i")
    last_i = _i64(inits, [N - 1], "last_i")
    row_shape = _i64(inits, [1, 1, OUT_N, 1], "row_shape")
    col_shape = _i64(inits, [1, 1, 1, OUT_N], "col_shape")
    flat_shape = _i64(inits, [SEEDS], "flat_shape")
    red_col_shape = _i64(inits, [10, 1], "red_col_shape")
    red_row_shape = _i64(inits, [1, 10], "red_row_shape")
    topk = _i64(inits, [10], "topk")
    zero = _f32(inits, [0.0], "zero")
    half = _f32(inits, [0.5], "half")
    zero_crop = _f32(inits, np.zeros((1, 1, OUT_N, OUT_N), dtype=np.float32), "zero_crop")
    _f32(inits, np.arange(SEEDS, dtype=np.float32).reshape(1, 1, N, N), "label_grid")
    _f32(inits, np.full((1, 1, N, N), 1000.0, dtype=np.float32), "inf_grid")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, bg_start, bg_end, axes4], ["bg"]),
            helper.make_node("Slice", [IN_NAME, blue_start, blue_end, axes4], ["blue"]),
            helper.make_node("Less", ["bg", half], ["active_b"]),
            helper.make_node("Slice", [IN_NAME, red_start, red_end, axes4], ["red"]),
            helper.make_node("Slice", [IN_NAME, cyan_start, cyan_end, axes4], ["cyan"]),
            helper.make_node("Where", ["active_b", "label_grid", "inf_grid"], ["labels0"]),
        ]
    )

    current = "labels0"
    for step in range(DILATION_STEPS):
        neg = f"neg{step}"
        pooled = f"pool{step}"
        local_min = f"min{step}"
        reached = f"labels{step + 1}"
        nodes.extend(
            [
                helper.make_node("Neg", [current], [neg]),
                helper.make_node(
                    "MaxPool",
                    [neg],
                    [pooled],
                    kernel_shape=[3, 3],
                    pads=[1, 1, 1, 1],
                    strides=[1, 1],
                ),
                helper.make_node("Neg", [pooled], [local_min]),
                helper.make_node("Where", ["active_b", local_min, "inf_grid"], [reached]),
            ]
        )
        current = reached

    nodes.extend(
        [
            helper.make_node("Cast", [current], ["labels_i"], to=TensorProto.INT32),
            helper.make_node("Reshape", ["labels_i", flat_shape], ["flat_labels"]),
            helper.make_node("Reshape", ["red", flat_shape], ["flat_red"]),
            helper.make_node("TopK", ["flat_red", topk], ["red_vals", "red_idx"], axis=0),
            helper.make_node("Greater", ["red_vals", zero], ["red_present"]),
            helper.make_node("Gather", ["flat_labels", "red_idx"], ["red_labels"], axis=0),
            helper.make_node("Reshape", ["red_labels", red_col_shape], ["red_labels_col"]),
            helper.make_node("Reshape", ["red_labels", red_row_shape], ["red_labels_row"]),
            helper.make_node("Reshape", ["red_present", red_row_shape], ["red_present_row"]),
            helper.make_node("Equal", ["red_labels_col", "red_labels_row"], ["same_label"]),
            helper.make_node("And", ["same_label", "red_present_row"], ["red_same_b"]),
            helper.make_node("Cast", ["red_same_b"], ["red_same"], to=TensorProto.FLOAT),
            helper.make_node("ReduceSum", ["red_same"], ["red_count"], axes=[1], keepdims=0),
            helper.make_node("ArgMax", ["red_count"], ["target_index"], axis=0, keepdims=0),
            helper.make_node("Gather", ["red_labels", "target_index"], ["target_label"], axis=0),
            helper.make_node("Equal", ["labels_i", "target_label"], ["target_mask"]),
            helper.make_node("Cast", ["target_mask"], ["target_f"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", ["target_f"], ["row_has"], axes=[3], keepdims=0),
            helper.make_node("ReduceMax", ["target_f"], ["col_has"], axes=[2], keepdims=0),
            helper.make_node("ArgMax", ["row_has"], ["rmin_2d"], axis=2, keepdims=0),
            helper.make_node("ArgMax", ["col_has"], ["cmin_2d"], axis=2, keepdims=0),
            helper.make_node("Gather", ["row_has", rev], ["row_rev"], axis=2),
            helper.make_node("Gather", ["col_has", rev], ["col_rev"], axis=2),
            helper.make_node("ArgMax", ["row_rev"], ["rrev_2d"], axis=2, keepdims=0),
            helper.make_node("ArgMax", ["col_rev"], ["crev_2d"], axis=2, keepdims=0),
            helper.make_node("Squeeze", ["rmin_2d"], ["rmin"], axes=[0, 1]),
            helper.make_node("Squeeze", ["cmin_2d"], ["cmin"], axes=[0, 1]),
            helper.make_node("Squeeze", ["rrev_2d"], ["rrev"], axes=[0, 1]),
            helper.make_node("Squeeze", ["crev_2d"], ["crev"], axes=[0, 1]),
            helper.make_node("Sub", [last_i, "rrev"], ["rmax"]),
            helper.make_node("Sub", [last_i, "crev"], ["cmax"]),
            helper.make_node("Sub", ["rmax", "rmin"], ["height_m1"]),
            helper.make_node("Sub", ["cmax", "cmin"], ["width_m1"]),
            helper.make_node("Add", ["height_m1", one_i], ["height"]),
            helper.make_node("Add", ["width_m1", one_i], ["width"]),
            helper.make_node("Add", ["rmin", offsets], ["row_idx"]),
            helper.make_node("Add", ["cmin", offsets], ["col_idx"]),
            helper.make_node("Less", [offsets, "height"], ["row_valid_1d"]),
            helper.make_node("Less", [offsets, "width"], ["col_valid_1d"]),
            helper.make_node("Where", ["row_valid_1d", "row_idx", zero_offsets], ["row_idx_safe"]),
            helper.make_node("Where", ["col_valid_1d", "col_idx", zero_offsets], ["col_idx_safe"]),
            helper.make_node("Reshape", ["row_valid_1d", row_shape], ["row_valid"]),
            helper.make_node("Reshape", ["col_valid_1d", col_shape], ["col_valid"]),
            helper.make_node("And", ["row_valid", "col_valid"], ["valid_b"]),
            helper.make_node("Cast", ["valid_b"], ["valid"], to=TensorProto.FLOAT),
            helper.make_node("Gather", ["blue", "row_idx_safe"], ["blue_rows"], axis=2),
            helper.make_node("Gather", ["blue_rows", "col_idx_safe"], ["blue_crop"], axis=3),
            helper.make_node("Gather", ["red", "row_idx_safe"], ["red_rows"], axis=2),
            helper.make_node("Gather", ["red_rows", "col_idx_safe"], ["red_crop"], axis=3),
            helper.make_node("Gather", ["cyan", "row_idx_safe"], ["cyan_rows"], axis=2),
            helper.make_node("Gather", ["cyan_rows", "col_idx_safe"], ["cyan_crop"], axis=3),
            helper.make_node("Mul", ["blue_crop", "valid"], ["blue_masked"]),
            helper.make_node("Mul", ["red_crop", "valid"], ["red_masked"]),
            helper.make_node("Mul", ["cyan_crop", "valid"], ["cyan_masked"]),
            helper.make_node(
                "Concat",
                [
                    zero_crop,
                    "blue_masked",
                    "red_masked",
                    zero_crop,
                    zero_crop,
                    zero_crop,
                    zero_crop,
                    zero_crop,
                    "cyan_masked",
                    zero_crop,
                ],
                ["crop"],
                axis=1,
            ),
            helper.make_node("Pad", ["crop"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - OUT_N, W - OUT_N]),
        ]
    )

    graph = helper.make_graph(
        nodes,
        TASK_ID,
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
    onnx.checker.check_model(model, full_check=True)
    return model


def verify_reference() -> tuple[int, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data[split]):
            actual = solve(example["input"])
            expected = np.asarray(example["output"], dtype=np.int64)
            total += 1
            if not np.array_equal(actual, expected):
                raise AssertionError(f"reference mismatch on {split}[{idx}]")
            passed += 1
    return passed, total


def verify_model(path: Path) -> tuple[int, int]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data[split]):
            x = convert_to_numpy(example, "input")
            y = convert_to_numpy(example, "output")
            if x is None or y is None:
                continue
            pred = session.run([OUT_NAME], {IN_NAME: x})[0]
            total += 1
            if not np.array_equal(pred > 0.0, y > 0.0):
                raise AssertionError(f"ONNX mismatch on {split}[{idx}]")
            passed += 1
    return passed, total


def main() -> None:
    ref_passed, ref_total = verify_reference()
    overlay_wrong = overlay_mismatch_count()
    model = build_model()
    onnx.save(model, BEST_PATH)
    model_passed, model_total = verify_model(BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"reference: {ref_passed}/{ref_total}")
    print(f"overlay hypothesis mismatches: {overlay_wrong}/{ref_total}")
    print(f"correct: {model_passed}/{model_total}")
    if not result["valid"]:
        raise SystemExit(f"invalid model: {result['error']}")
    print(
        "score: "
        f"memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']:.6f}"
    )


if __name__ == "__main__":
    main()
