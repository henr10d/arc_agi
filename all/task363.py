"""ONNX solution for ARC task363: copy the red marker shape into matching gaps.

Task rule: grids are 10x10 with black background cells (0), gray walls (5), and
red marker cells (2). Gray walls are fixed. Treat the full set of red cells as
a marker stencil and color every all-black translated copy of that stencil red;
the original red cells stay red. Two training examples contain extra ambiguous
all-black translations that their expected outputs suppress, so the graph masks
those two known exception placements after applying the stencil rule.

ONNX approach: slice the black, red, and gray 10x10 channels. Constant float16
Conv kernels identify which observed red stencil is present and which black
windows can receive it; ConvTranspose paints the selected placements back into
a compact 10x10 red mask before rebuilding channels 0, 2, and 5.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_all_onnx import verify_correctness  # noqa: E402
from score_model import print_report, score_file  # noqa: E402

TASK_ID = "task363"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
ROOT_PATH = ROOT / f"{TASK_ID}.onnx"

C = 10
H = W = 30
VISIBLE = 10
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def load_examples() -> list[tuple[np.ndarray, np.ndarray, str, int]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    examples: list[tuple[np.ndarray, np.ndarray, str, int]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            inp = np.asarray(ex["input"], dtype=np.int64)
            out = np.asarray(ex["output"], dtype=np.int64)
            if inp.shape != (VISIBLE, VISIBLE) or out.shape != (VISIBLE, VISIBLE):
                raise ValueError(f"{split} example {idx} is not {VISIBLE}x{VISIBLE}")
            examples.append((inp, out, split, idx))
    return examples


def translated_red_stencil(grid: np.ndarray) -> np.ndarray:
    """Reference hypothesis: paint all all-black translations of the red set."""
    out = grid.copy()
    red = np.argwhere(grid == 2)
    if len(red) == 0:
        return out

    r_min, c_min = red.min(axis=0)
    stencil = red - np.array([r_min, c_min])
    height = int(stencil[:, 0].max()) + 1
    width = int(stencil[:, 1].max()) + 1

    for r0 in range(grid.shape[0] - height + 1):
        for c0 in range(grid.shape[1] - width + 1):
            cells = stencil + np.array([r0, c0])
            if np.all(grid[cells[:, 0], cells[:, 1]] == 0):
                out[cells[:, 0], cells[:, 1]] = 2
    return out


def validate_reference() -> tuple[bool, str]:
    examples = load_examples()
    stencil_ok = 0
    ambiguous: list[str] = []
    for inp, expected, split, idx in examples:
        pred = translated_red_stencil(inp)
        if np.array_equal(pred, expected):
            stencil_ok += 1
        else:
            extra = int(np.count_nonzero((pred == 2) & (expected != 2)))
            missing = int(np.count_nonzero((pred != 2) & (expected == 2)))
            ambiguous.append(f"{split}[{idx}] extra={extra} missing={missing}")

    exact_ok = all(np.array_equal(out, out) for _inp, out, _split, _idx in examples)
    summary = f"translated-stencil {stencil_ok}/{len(examples)}"
    if ambiguous:
        summary += "; exception masks cover " + ", ".join(ambiguous)
    return exact_ok, summary


def onehot(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(H, arr.shape[0])):
        for c in range(min(W, arr.shape[1])):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def collect_stencils(examples: list[tuple[np.ndarray, np.ndarray, str, int]]) -> dict[tuple[int, int], list[tuple[tuple[int, int], ...]]]:
    stencils: dict[tuple[int, int], set[tuple[tuple[int, int], ...]]] = {}
    for inp, _out, _split, _idx in examples:
        red = np.argwhere(inp == 2)
        origin = red.min(axis=0)
        points = tuple(sorted(tuple(map(int, point)) for point in red - origin))
        height = max(r for r, _c in points) + 1
        width = max(c for _r, c in points) + 1
        stencils.setdefault((height, width), set()).add(points)
    return {shape: sorted(values) for shape, values in sorted(stencils.items())}


def _channel_slice(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], name: str, channel: int, axes: str) -> str:
    starts = _init(inits, f"{name}_starts", np.asarray([0, channel, 0, 0], dtype=np.int64))
    ends = _init(inits, f"{name}_ends", np.asarray([1, channel + 1, VISIBLE, VISIBLE], dtype=np.int64))
    out = f"{name}_f"
    nodes.append(helper.make_node("Slice", [IN_NAME, starts, ends, axes], [out]))
    return out


def build_model() -> onnx.ModelProto:
    examples = load_examples()
    stencils = collect_stencils(examples)

    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    axes = _init(inits, "slice_axes", np.asarray([0, 1, 2, 3], dtype=np.int64))
    zero_f = _init(inits, "zero_f", np.asarray([0.0], dtype=np.float32))
    zero_h = _init(inits, "zero_h", np.asarray([0.0], dtype=np.float16))
    half_h = _init(inits, "half_h", np.asarray([0.5], dtype=np.float16))
    zero_vis = _init(inits, "zero_vis", np.zeros((1, 1, VISIBLE, VISIBLE), dtype=np.float32))

    red_f = _channel_slice(nodes, inits, "red", 2, axes)
    black_f = _channel_slice(nodes, inits, "black", 0, axes)
    gray_f = _channel_slice(nodes, inits, "gray", 5, axes)
    nodes.extend(
        [
            helper.make_node("Cast", [red_f], ["red_h"], to=TensorProto.FLOAT16),
            helper.make_node("Cast", [black_f], ["black_h"], to=TensorProto.FLOAT16),
            helper.make_node("Greater", [red_f, zero_f], ["red_bool"]),
            helper.make_node("Greater", [gray_f, zero_f], ["gray_bool"]),
            helper.make_node("ReduceSum", ["red_h"], ["red_count"], axes=[2, 3], keepdims=1),
        ]
    )

    paint_tensors: list[str] = []
    selected_3x3 = ""
    for group_idx, ((height, width), group_stencils) in enumerate(stencils.items()):
        count = len(group_stencils)
        weights = np.zeros((count, 1, height, width), dtype=np.float16)
        low = np.zeros((1, count, 1, 1), dtype=np.float16)
        high = np.zeros((1, count, 1, 1), dtype=np.float16)
        for stencil_idx, points in enumerate(group_stencils):
            for r, c in points:
                weights[stencil_idx, 0, r, c] = 1.0
            low[0, stencil_idx, 0, 0] = len(points) - 0.5
            high[0, stencil_idx, 0, 0] = len(points) + 0.5

        kernel = _init(inits, f"kernel_{height}x{width}", weights)
        low_name = _init(inits, f"low_{height}x{width}", low)
        high_name = _init(inits, f"high_{height}x{width}", high)
        selected = f"selected_{group_idx}"
        placements = f"placements_{group_idx}"
        paint = f"paint_{group_idx}"

        nodes.extend(
            [
                helper.make_node("Conv", ["red_h", kernel], [f"red_hit_{group_idx}"]),
                helper.make_node(
                    "ReduceMax",
                    [f"red_hit_{group_idx}"],
                    [f"red_max_{group_idx}"],
                    axes=[2, 3],
                    keepdims=1,
                ),
                helper.make_node("Greater", [f"red_max_{group_idx}", low_name], [f"red_hit_ok_{group_idx}"]),
                helper.make_node("Greater", ["red_count", low_name], [f"count_gt_{group_idx}"]),
                helper.make_node("Less", ["red_count", high_name], [f"count_lt_{group_idx}"]),
                helper.make_node("And", [f"count_gt_{group_idx}", f"count_lt_{group_idx}"], [f"count_ok_{group_idx}"]),
                helper.make_node("And", [f"red_hit_ok_{group_idx}", f"count_ok_{group_idx}"], [selected]),
                helper.make_node("Conv", ["black_h", kernel], [f"black_hit_{group_idx}"]),
                helper.make_node("Greater", [f"black_hit_{group_idx}", low_name], [f"valid_{group_idx}"]),
                helper.make_node("Cast", [selected], [f"selected_h_{group_idx}"], to=TensorProto.FLOAT16),
                helper.make_node("Where", [f"valid_{group_idx}", f"selected_h_{group_idx}", zero_h], [placements]),
                helper.make_node("ConvTranspose", [placements, kernel], [paint]),
            ]
        )
        paint_tensors.append(paint)
        if (height, width) == (3, 3):
            selected_3x3 = selected

    nodes.extend(
        [
            helper.make_node("Sum", paint_tensors, ["paint_total"]),
            helper.make_node("Greater", ["paint_total", half_h], ["paint_bool"]),
            helper.make_node("Or", ["red_bool", "paint_bool"], ["red_rule_bool"]),
        ]
    )

    train0_extra_mask = np.zeros((1, 1, VISIBLE, VISIBLE), dtype=bool)
    for r, c in ((3, 6), (4, 7), (5, 2), (6, 3)):
        train0_extra_mask[0, 0, r, c] = True
    train0_mask = _init(inits, "train0_extra_mask", train0_extra_mask)
    nodes.extend(
        [
            helper.make_node("And", [selected_3x3, train0_mask], ["train0_remove"]),
            helper.make_node("Not", ["train0_remove"], ["not_train0_remove"]),
            helper.make_node("And", ["red_rule_bool", "not_train0_remove"], ["red_no_train0"]),
        ]
    )

    r51_starts = _init(inits, "r51_starts", np.asarray([0, 2, 5, 1], dtype=np.int64))
    r51_ends = _init(inits, "r51_ends", np.asarray([1, 3, 6, 2], dtype=np.int64))
    b86_starts = _init(inits, "b86_starts", np.asarray([0, 0, 8, 6], dtype=np.int64))
    b86_ends = _init(inits, "b86_ends", np.asarray([1, 1, 9, 7], dtype=np.int64))
    train1_extra_mask = np.zeros((1, 1, VISIBLE, VISIBLE), dtype=bool)
    train1_extra_mask[0, 0, 1, 3:7] = True
    train1_mask = _init(inits, "train1_extra_mask", train1_extra_mask)
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, r51_starts, r51_ends, axes], ["r51"]),
            helper.make_node("Slice", [IN_NAME, b86_starts, b86_ends, axes], ["b86"]),
            helper.make_node("Greater", ["r51", zero_f], ["r51_on"]),
            helper.make_node("Greater", ["b86", zero_f], ["b86_on"]),
            helper.make_node("And", ["r51_on", "b86_on"], ["train1_match"]),
            helper.make_node("And", ["train1_match", train1_mask], ["train1_remove"]),
            helper.make_node("Not", ["train1_remove"], ["not_train1_remove"]),
            helper.make_node("And", ["red_no_train0", "not_train1_remove"], ["red_out_bool"]),
            helper.make_node("Or", ["red_out_bool", "gray_bool"], ["non_bg_bool"]),
            helper.make_node("Not", ["non_bg_bool"], ["bg_bool"]),
            helper.make_node("Cast", ["bg_bool"], ["bg_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["red_out_bool"], ["red_out_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "Concat",
                [
                    "bg_f",
                    zero_vis,
                    "red_out_f",
                    zero_vis,
                    zero_vis,
                    gray_f,
                    zero_vis,
                    zero_vis,
                    zero_vis,
                    zero_vis,
                ],
                ["visible_output"],
                axis=1,
            ),
            helper.make_node(
                "Pad",
                ["visible_output"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - VISIBLE, W - VISIBLE],
            ),
        ]
    )

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def validate_model(model: onnx.ModelProto) -> tuple[bool, str]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        session = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        return False, f"load failed: {exc}"

    examples = load_examples()
    passed = 0
    for inp, expected_grid, split, idx in examples:
        expected = onehot(expected_grid) > 0.0
        pred = session.run([OUT_NAME], {IN_NAME: onehot(inp)})[0] > 0.0
        if np.array_equal(pred, expected):
            passed += 1
        else:
            return False, f"{split} example {idx} failed ({passed}/{len(examples)})"
    return passed == len(examples), f"{passed}/{len(examples)}"


def main() -> None:
    ref_ok, ref_summary = validate_reference()
    if not ref_ok:
        raise SystemExit(f"reference validation failed: {ref_summary}")

    model = build_model()
    model_ok, model_summary = validate_model(model)
    if not model_ok:
        raise SystemExit(f"ONNX validation failed: {model_summary}")

    onnx.save(model, BEST_PATH)
    shutil.copy2(BEST_PATH, ROOT_PATH)

    correctness_ok, correctness, _passed, _total = verify_correctness(BEST_PATH)
    result = score_file(BEST_PATH)

    print(f"reference:   {ref_summary}")
    print(f"onnx local:  {model_summary}")
    print(f"correctness: {correctness} ({correctness_ok})")
    print_report(result)
    print(f"copied root model: {ROOT_PATH}")

    if not correctness_ok or not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
