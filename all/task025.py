"""ONNX for ARC task025: move stray cells next to matching guide lines.

Task rule: each input contains either full-height vertical guide lines or
full-width horizontal guide lines, with each guide line having a distinct
nonzero color. Stray cells whose color matches a guide are removed from their
original location and projected to the cell immediately beside that same-color
guide line, preserving the row for vertical guides or the column for horizontal
guides. The projected cell is placed on the same side of the guide as the
stray cell. Strays whose color has no guide line disappear.

ONNX approach: detect guide rows/columns by summing each nonzero color channel
and thresholding counts greater than 10. Prefix/suffix asymmetric max-pools find
whether a same-colored stray exists on each side of each potential guide, then
shifted guide masks place the projected cells. The graph keeps full-grid
intermediates boolean where possible and casts to float only at the final
competition output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import print_report, score_file  # noqa: E402

TASK_ID = "task025"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task025.onnx"

CHANNELS = 10
HEIGHT = WIDTH = 30
SHAPE = [1, CHANNELS, HEIGHT, WIDTH]
IR_VERSION = 10
OPSET = 10


def load_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def solve_grid(grid: list[list[int]]) -> list[list[int]]:
    h, w = len(grid), len(grid[0])
    out = [[0 for _ in range(w)] for _ in range(h)]

    guide_rows: dict[int, int] = {}
    guide_cols: dict[int, int] = {}
    for r, row in enumerate(grid):
        if row[0] != 0 and all(value == row[0] for value in row):
            guide_rows[row[0]] = r
    for c in range(w):
        color = grid[0][c]
        if color != 0 and all(grid[r][c] == color for r in range(h)):
            guide_cols[color] = c

    for color, r in guide_rows.items():
        for c in range(w):
            out[r][c] = color
    for color, c in guide_cols.items():
        for r in range(h):
            out[r][c] = color

    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if color == 0:
                continue
            if color in guide_rows and r != guide_rows[color]:
                guide_r = guide_rows[color]
                out[guide_r - 1 if r < guide_r else guide_r + 1][c] = color
            elif color in guide_cols and c != guide_cols[color]:
                guide_c = guide_cols[color]
                out[r][guide_c - 1 if c < guide_c else guide_c + 1] = color

    return out


def one_hot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros((1, CHANNELS, HEIGHT, WIDTH), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def _init(inits: list[onnx.TensorProto], name: str, value: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(value), name=name))
    return name


def _slice(nodes: list[onnx.NodeProto], x: str, y: str, starts: str, ends: str, axes: str) -> None:
    nodes.append(helper.make_node("Slice", [x, starts, ends, axes], [y]))


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)

    s0 = _init(inits, "s0", np.array([0], dtype=np.int64))
    s1 = _init(inits, "s1", np.array([1], dtype=np.int64))
    e29 = _init(inits, "e29", np.array([29], dtype=np.int64))
    e30 = _init(inits, "e30", np.array([30], dtype=np.int64))
    ax_r = _init(inits, "ax_r", np.array([2], dtype=np.int64))
    ax_c = _init(inits, "ax_c", np.array([3], dtype=np.int64))
    ten = _init(inits, "ten", np.array([10.0], dtype=np.float16))
    colors = _init(inits, "colors", np.arange(1, 10, dtype=np.int64).reshape(1, 9, 1, 1))

    nodes.append(helper.make_node("ReduceMax", ["input"], ["valid_sum"], axes=[1], keepdims=1))
    nodes.append(helper.make_node("Cast", ["valid_sum"], ["valid"], to=TensorProto.BOOL))

    nodes.append(helper.make_node("ArgMax", ["input"], ["color_index"], axis=1, keepdims=1))
    nodes.append(helper.make_node("Equal", ["color_index", colors], ["xmask"]))
    nodes.append(helper.make_node("Cast", ["xmask"], ["x"], to=TensorProto.FLOAT16))

    nodes.append(helper.make_node("ReduceSum", ["x"], ["col_count"], axes=[2], keepdims=1))
    nodes.append(helper.make_node("ReduceSum", ["x"], ["row_count"], axes=[3], keepdims=1))
    nodes.append(helper.make_node("Greater", ["col_count", ten], ["vline"]))
    nodes.append(helper.make_node("Greater", ["row_count", ten], ["hline"]))

    nodes.append(
        helper.make_node(
            "MaxPool",
            ["x"],
            ["wprefix"],
            kernel_shape=[1, WIDTH],
            pads=[0, WIDTH - 1, 0, 0],
        )
    )
    nodes.append(
        helper.make_node(
            "MaxPool",
            ["x"],
            ["wsuffix"],
            kernel_shape=[1, WIDTH],
            pads=[0, 0, 0, WIDTH - 1],
        )
    )
    nodes.append(
        helper.make_node(
            "MaxPool",
            ["x"],
            ["hprefix"],
            kernel_shape=[HEIGHT, 1],
            pads=[HEIGHT - 1, 0, 0, 0],
        )
    )
    nodes.append(
        helper.make_node(
            "MaxPool",
            ["x"],
            ["hsuffix"],
            kernel_shape=[HEIGHT, 1],
            pads=[0, 0, HEIGHT - 1, 0],
        )
    )
    nodes.append(helper.make_node("Cast", ["wprefix"], ["left_any"], to=TensorProto.BOOL))
    nodes.append(helper.make_node("Cast", ["wsuffix"], ["right_any"], to=TensorProto.BOOL))
    nodes.append(helper.make_node("Cast", ["hprefix"], ["above_any"], to=TensorProto.BOOL))
    nodes.append(helper.make_node("Cast", ["hsuffix"], ["below_any"], to=TensorProto.BOOL))

    _slice(nodes, "vline", "vl_tail", s1, e30, ax_c)
    _slice(nodes, "vline", "vl_head", s0, e29, ax_c)
    _slice(nodes, "vline", "false_seed", s0, s1, ax_c)
    nodes.append(helper.make_node("Not", ["false_seed"], ["not_false_seed"]))
    nodes.append(helper.make_node("And", ["false_seed", "not_false_seed"], ["false"]))
    nodes.append(helper.make_node("Concat", ["vl_tail", "false"], ["vleft"], axis=3))
    nodes.append(helper.make_node("Concat", ["false", "vl_head"], ["vright"], axis=3))

    _slice(nodes, "hline", "hl_tail", s1, e30, ax_r)
    _slice(nodes, "hline", "hl_head", s0, e29, ax_r)
    nodes.append(helper.make_node("Concat", ["hl_tail", "false"], ["habove"], axis=2))
    nodes.append(helper.make_node("Concat", ["false", "hl_head"], ["hbelow"], axis=2))

    nodes.append(helper.make_node("And", ["left_any", "vleft"], ["vadd_l"]))
    nodes.append(helper.make_node("And", ["right_any", "vright"], ["vadd_r"]))
    nodes.append(helper.make_node("Or", ["vadd_l", "vadd_r"], ["vadd"]))
    nodes.append(helper.make_node("And", ["above_any", "habove"], ["hadd_a"]))
    nodes.append(helper.make_node("And", ["below_any", "hbelow"], ["hadd_b"]))
    nodes.append(helper.make_node("Or", ["hadd_a", "hadd_b"], ["hadd"]))
    nodes.append(helper.make_node("Or", ["vadd", "hadd"], ["projected"]))

    nodes.append(helper.make_node("Or", ["vline", "hline"], ["line_cells"]))
    nodes.append(helper.make_node("And", ["valid", "line_cells"], ["guide"]))
    nodes.append(helper.make_node("Or", ["guide", "projected"], ["nz"]))

    split_outputs = [f"ch{i}" for i in range(1, 10)]
    nodes.append(helper.make_node("Split", ["nz"], split_outputs, axis=1, split=[1] * 9))
    any_name = split_outputs[0]
    for idx, channel in enumerate(split_outputs[1:], start=2):
        out_name = f"any{idx}"
        nodes.append(helper.make_node("Or", [any_name, channel], [out_name]))
        any_name = out_name
    nodes.append(helper.make_node("Not", [any_name], ["not_any"]))
    nodes.append(helper.make_node("And", ["valid", "not_any"], ["bg"]))
    nodes.append(helper.make_node("Concat", ["bg", *split_outputs], ["core"], axis=1))
    nodes.append(helper.make_node("Cast", ["core"], ["output"], to=TensorProto.FLOAT))

    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], inits)
    model = helper.make_model(
        graph,
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def iter_examples(data: dict[str, list[dict[str, list[list[int]]]]]):
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(data.get(split, [])):
            yield split, index, example


def verify_reference(data: dict[str, list[dict[str, list[list[int]]]]]) -> None:
    failures: list[str] = []
    for split, index, example in iter_examples(data):
        if solve_grid(example["input"]) != example["output"]:
            failures.append(f"{split}[{index}]")
    if failures:
        raise RuntimeError(f"reference rule failed: {', '.join(failures[:10])}")


def verify_onnx(path: Path, data: dict[str, list[dict[str, list[list[int]]]]]) -> None:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    failures: list[str] = []
    for split, index, example in iter_examples(data):
        actual = session.run(["output"], {"input": one_hot(example["input"])})[0] > 0.0
        expected = one_hot(example["output"]) > 0.0
        if not np.array_equal(actual, expected):
            failures.append(f"{split}[{index}]")
    if failures:
        raise RuntimeError(f"ONNX verification failed: {', '.join(failures[:10])}")


def main() -> None:
    data = load_data()
    verify_reference(data)
    model = build_model()
    onnx.save(model, str(BEST_PATH))
    verify_onnx(BEST_PATH, data)
    print_report(score_file(BEST_PATH))


if __name__ == "__main__":
    main()
