"""ONNX for ARC task293: repair the hidden crossing of two solid bars.

Task rule: the grid contains a horizontal solid bar and a perpendicular
vertical solid bar, each with its own non-black color.  At their rectangular
overlap, the input shows only one bar's color.  Restore the hidden bar by
recoloring just that overlap to the other bar color, leaving every other cell
unchanged.  Bar colors, grid size, and bar thickness are not fixed.

ONNX approach: for each color, compare its maximum row occupancy with its
maximum column occupancy to classify it as the horizontal or vertical stripe.
The overlap is horizontal occupied rows times vertical occupied columns.  At
that overlap, switch from the visible stripe color to the other stripe color.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task293"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task293.onnx"
DATA_PATH = ROOT / "data" / "task293.json"

C = 10
H = W = 30
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _make_model(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], name: str) -> onnx.ModelProto:
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


def load_examples() -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    examples: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            examples.append(
                (
                    split,
                    idx,
                    np.asarray(ex["input"], dtype=np.int64),
                    np.asarray(ex["output"], dtype=np.int64),
                )
            )
    return examples


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver matching the dynamic ONNX graph."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    colors = [int(c) for c in np.unique(g) if c != 0]
    assert len(colors) == 2

    orient: dict[int, str] = {}
    for color in colors:
        mask = g == color
        row_max = int(mask.sum(axis=1).max())
        col_max = int(mask.sum(axis=0).max())
        orient[color] = "h" if row_max > col_max else "v"

    h_color = next(color for color in colors if orient[color] == "h")
    v_color = next(color for color in colors if orient[color] == "v")
    h_rows = np.any(g == h_color, axis=1)
    v_cols = np.any(g == v_color, axis=0)
    overlap = np.ix_(h_rows, v_cols)
    visible = g[overlap]
    out[overlap] = np.where(visible == h_color, v_color, h_color)
    return out


def _grid_to_onehot(grid: np.ndarray) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            out[0, int(grid[r, c]), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    active = onehot > 0.0
    return active.argmax(axis=1)[0].astype(np.int64)


def _run_onnx(model: onnx.ModelProto, grid: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: _grid_to_onehot(grid)})[0]


def build_model(switch_visible: bool) -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    zero = _f32(inits, [0.0], "zero")
    color_mask = _init(inits, np.asarray([0.0] + [1.0] * 9, dtype=np.float32).reshape(1, C, 1, 1), "color_mask")

    nodes.append(helper.make_node("ReduceSum", [IN_NAME], ["row_counts"], axes=[3], keepdims=1))
    nodes.append(helper.make_node("ReduceSum", [IN_NAME], ["col_counts"], axes=[2], keepdims=1))
    nodes.append(helper.make_node("ReduceMax", ["row_counts"], ["row_max"], axes=[2], keepdims=1))
    nodes.append(helper.make_node("ReduceMax", ["col_counts"], ["col_max"], axes=[3], keepdims=1))

    nodes.append(helper.make_node("Greater", ["row_max", "col_max"], ["h_color_bool"]))
    nodes.append(helper.make_node("Greater", ["col_max", "row_max"], ["v_color_bool"]))
    nodes.append(helper.make_node("Cast", ["h_color_bool"], ["h_color"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Cast", ["v_color_bool"], ["v_color"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Mul", ["h_color", color_mask], ["h_color_nz"]))
    nodes.append(helper.make_node("Mul", ["v_color", color_mask], ["v_color_nz"]))

    nodes.append(helper.make_node("Mul", ["row_counts", "h_color_nz"], ["h_row_counts"]))
    nodes.append(helper.make_node("Mul", ["col_counts", "v_color_nz"], ["v_col_counts"]))
    nodes.append(helper.make_node("ReduceSum", ["h_row_counts"], ["h_rows_score"], axes=[1], keepdims=1))
    nodes.append(helper.make_node("ReduceSum", ["v_col_counts"], ["v_cols_score"], axes=[1], keepdims=1))
    nodes.append(helper.make_node("Greater", ["h_rows_score", zero], ["h_rows"]))
    nodes.append(helper.make_node("Greater", ["v_cols_score", zero], ["v_cols"]))
    nodes.append(helper.make_node("And", ["h_rows", "v_cols"], ["overlap"]))

    if switch_visible:
        nodes.append(helper.make_node("Add", ["h_color_nz", "v_color_nz"], ["bar_colors"]))
        nodes.append(helper.make_node("Sub", ["bar_colors", IN_NAME], ["replacement"]))
    else:
        nodes.append(helper.make_node("Identity", [IN_NAME], ["replacement"]))
    nodes.append(helper.make_node("Where", ["overlap", "replacement", IN_NAME], [OUT_NAME]))

    suffix = "switch" if switch_visible else "keep_visible"
    return _make_model(nodes, inits, f"{TASK_ID}_{suffix}")


def validate_reference(examples: list[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    for split, idx, inp, expected in examples:
        got = solve(inp)
        if not np.array_equal(got, expected):
            raise AssertionError(f"reference failed {split}[{idx}]")


def validate_model(model: onnx.ModelProto, examples: list[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    for split, idx, inp, expected in examples:
        got = _onehot_to_grid(_run_onnx(model, inp))[: expected.shape[0], : expected.shape[1]]
        if not np.array_equal(got, expected):
            raise AssertionError(f"model failed {split}[{idx}]")


def main() -> None:
    examples = load_examples()
    validate_reference(examples)

    scored: list[tuple[float, dict[str, Any], onnx.ModelProto, str]] = []
    for switch_visible in (True, False):
        model = build_model(switch_visible)
        try:
            validate_model(model, examples)
        except AssertionError:
            continue
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
            onnx.save(model, tmp.name)
            metrics = score_file(Path(tmp.name))
        scored.append((float(metrics["score"]), metrics, model, "switch_visible" if switch_visible else "keep_visible"))

    if not scored:
        raise AssertionError("no variant passed all examples")

    scored.sort(key=lambda item: (item[0], -float(item[1]["cost"])), reverse=True)
    _score, metrics, model, name = scored[0]
    onnx.save(model, BEST_PATH)
    print(f"saved {BEST_PATH} using {name}")
    print(
        f"memory={metrics['memory']} params={metrics['params']} "
        f"cost={metrics['cost']} score={metrics['score']:.6f}"
    )


if __name__ == "__main__":
    main()
