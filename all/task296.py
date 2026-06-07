"""ONNX for ARC task296: squeeze a sparse 5x7 marker into a 3x3 grid.

Task rule: the 5x7 input has black spacer row 2 and black spacer columns 2..4,
with one non-black color appearing only in the four corner 2x2 areas.  Output a
3x3 grid using that same color wherever the corresponding squeezed bin contains
any non-black pixel:

    row bins = [[0], [1, 3], [4]]
    col bins = [[0], [1, 5], [6]]

Padding outside the 3x3 output remains all-zero for the NeuroGolf one-hot I/O.
The ONNX graph slices the 16 relevant foreground one-hot pixels directly, pools
only the multi-pixel bins, builds a compact bool 3x3, then casts/pads once.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task296"
BEST_PATH = OUT_DIR / "task296.onnx"
DATA_PATH = ROOT / "data" / "task296.json"

C = 10
H = W = 30
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10

ROW_BINS = [[0], [1, 3], [4]]
COL_BINS = [[0], [1, 5], [6]]


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver matching the validated JSON rule."""
    g = np.asarray(grid, dtype=np.int64)
    assert g.shape == (5, 7), g.shape
    colors = np.unique(g[g != 0])
    assert colors.size == 1, colors
    color = int(colors[0])

    out = np.zeros((3, 3), dtype=np.int64)
    for out_r, rows in enumerate(ROW_BINS):
        for out_c, cols in enumerate(COL_BINS):
            if np.any(g[np.ix_(rows, cols)] != 0):
                out[out_r, out_c] = color
    return out


def _grid_to_onehot(grid: Sequence[Sequence[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    active = onehot > 0.0
    bad = active.sum(axis=1)[0, : shape[0], : shape[1]] != 1
    decoded = active.argmax(axis=1)[0, : shape[0], : shape[1]].astype(np.int64)
    decoded[bad] = -1
    return decoded


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


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes_chw = _i64(inits, [1, 2, 3], "axes_chw")
    half = _f32(inits, [0.5], "half")
    pads = [0, 0, 0, 0, 0, 0, H - 3, W - 3]
    pixel_cache: dict[tuple[int, int], str] = {}

    def pixel(row: int, col: int) -> str:
        key = (row, col)
        if key in pixel_cache:
            return pixel_cache[key]
        name = f"p{row}_{col}"
        starts = _i64(inits, [1, row, col], f"{name}_st")
        ends = _i64(inits, [C, row + 1, col + 1], f"{name}_en")
        nodes.append(helper.make_node("Slice", [IN_NAME, starts, ends, axes_chw], [name]))
        pixel_cache[key] = name
        return name

    def pooled_cell(rows: list[int], cols: list[int], out_r: int, out_c: int) -> str:
        sources = [pixel(row, col) for row in rows for col in cols]
        raw = sources[0]
        if len(sources) > 1:
            raw = f"sum_{out_r}_{out_c}"
            nodes.append(helper.make_node("Sum", sources, [raw]))
        out = f"cell_{out_r}_{out_c}"
        nodes.append(helper.make_node("Greater", [raw, half], [out]))
        return out

    row_names = []
    for out_r, rows in enumerate(ROW_BINS):
        cells = [pooled_cell(rows, cols, out_r, out_c) for out_c, cols in enumerate(COL_BINS)]
        row_name = f"row_{out_r}"
        nodes.append(helper.make_node("Concat", cells, [row_name], axis=3))
        row_names.append(row_name)

    nodes.extend(
        [
            helper.make_node("Concat", row_names, ["fg_bool"], axis=2),
            helper.make_node("Cast", ["fg_bool"], ["fg"], to=TensorProto.FLOAT),
            helper.make_node("ReduceSum", ["fg"], ["occupied_f"], axes=[1], keepdims=1),
            helper.make_node("Less", ["occupied_f", half], ["black_bool"]),
            helper.make_node("Cast", ["black_bool"], ["black"], to=TensorProto.FLOAT),
            helper.make_node("Concat", ["black", "fg"], ["out3"], axis=1),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=pads),
        ]
    )

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


def validate_model(model: onnx.ModelProto, examples: list[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    for split, idx, inp, expected in examples:
        solved = solve(inp)
        if not np.array_equal(solved, expected):
            raise AssertionError((split, idx, "reference mismatch", solved, expected))
        pred = sess.run([OUT_NAME], {IN_NAME: _grid_to_onehot(inp)})[0]
        got = _onehot_to_grid(pred, expected.shape)
        if not np.array_equal(got, expected):
            raise AssertionError((split, idx, got, expected))


def main() -> None:
    examples = load_examples()
    model = build_model()
    validate_model(model, examples)
    onnx.save(model, BEST_PATH)
    result = score_file(BEST_PATH)
    print(
        f"{BEST_PATH.name}: valid={result['valid']} memory={result['memory']} "
        f"params={result['params']} cost={result['cost']} score={result['score']}"
    )
    if not result["valid"]:
        raise SystemExit(result["error"])


if __name__ == "__main__":
    main()
