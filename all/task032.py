"""ONNX generator for NeuroGolf task032 gravity columns.

Task rule: within the active 4x4, 5x5, or 6x6 grid, each non-black cell
falls straight down in its own column until the column's colored cells are
packed against the bottom. Colors are preserved. In the provided train/test
and arc-gen data every non-empty column contains only one color, so the
exported graph uses that constraint to avoid a more expensive stable scatter.
Padding outside the active grid remains all-zero in the Kaggle one-hot tensor.

The ONNX graph works only on the top-left 6x6 region. It infers each column's
active height from the one-hot validity mask, counts colored cells per column,
fills the bottom count rows with the column color, then pads directly to the
required float output tensor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task032.onnx"
DATA_PATH = ROOT / "data" / "task032.json"

IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, 10, 30, 30]
CORE = 6
IR_VERSION = 10
OPSET = 10


def _init(inits: list[onnx.TensorProto], name: str, arr: np.ndarray) -> str:
    inits.append(numpy_helper.from_array(arr, name=name))
    return name


def _i64(inits: list[onnx.TensorProto], name: str, vals: Iterable[int]) -> str:
    return _init(inits, name, np.asarray(list(vals), dtype=np.int64))


def _f32(inits: list[onnx.TensorProto], name: str, vals: Iterable[float]) -> str:
    return _init(inits, name, np.asarray(list(vals), dtype=np.float32))


def _slice(
    nodes: list[onnx.NodeProto],
    data: str,
    out: str,
    starts: str,
    ends: str,
    axes: str,
) -> None:
    nodes.append(helper.make_node("Slice", [data, starts, ends, axes], [out]))


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def _onehot_to_grid(arr: np.ndarray, h: int, w: int) -> list[list[int]]:
    return arr[0, :, :h, :w].argmax(axis=0).astype(int).tolist()


def solve_grid(grid: list[list[int]]) -> list[list[int]]:
    h, w = len(grid), len(grid[0])
    out = [[0 for _ in range(w)] for _ in range(h)]
    for c in range(w):
        vals = [grid[r][c] for r in range(h) if grid[r][c] != 0]
        for i, color in enumerate(vals):
            out[h - len(vals) + i][c] = color
    return out


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, "axes4", [0, 1, 2, 3])
    st0 = _i64(inits, "st0", [0, 0, 0, 0])
    en_ch0 = _i64(inits, "en_ch0", [1, 1, CORE, CORE])
    st_fg = _i64(inits, "st_fg", [0, 1, 0, 0])
    en_fg = _i64(inits, "en_fg", [1, 10, CORE, CORE])
    zero = _f32(inits, "zero", [0.0])
    row_ids = [_f32(inits, f"row{i}", [float(i)]) for i in range(CORE)]
    row_next = [_f32(inits, f"row_next{i}", [float(i + 1)]) for i in range(CORE)]

    _slice(nodes, IN_NAME, "ch0", st0, en_ch0, axes4)
    _slice(nodes, IN_NAME, "xfg", st_fg, en_fg, axes4)
    nodes.append(helper.make_node("ReduceSum", ["xfg"], ["nz_sum"], axes=[1], keepdims=1))
    nodes.append(helper.make_node("Add", ["ch0", "nz_sum"], ["valid_sum"]))
    nodes.append(helper.make_node("Greater", ["valid_sum", zero], ["valid"]))
    nodes.append(helper.make_node("ReduceSum", ["valid_sum"], ["height"], axes=[2], keepdims=1))
    nodes.append(helper.make_node("ReduceSum", ["nz_sum"], ["count"], axes=[2], keepdims=1))
    nodes.append(helper.make_node("Sub", ["height", "count"], ["start_row"]))
    nodes.append(helper.make_node("ReduceSum", ["xfg"], ["col_color_sum"], axes=[2], keepdims=1))
    nodes.append(helper.make_node("Greater", ["col_color_sum", zero], ["col_color"]))

    out_rows: list[str] = []
    occ_rows: list[str] = []
    for r in range(CORE):
        nodes.append(helper.make_node("Greater", [row_next[r], "start_row"], [f"after_start{r}"]))
        nodes.append(helper.make_node("Greater", ["height", row_ids[r]], [f"in_height{r}"]))
        nodes.append(helper.make_node("And", [f"after_start{r}", f"in_height{r}"], [f"occ{r}"]))
        nodes.append(helper.make_node("And", ["col_color", f"occ{r}"], [f"out_row{r}"]))
        out_rows.append(f"out_row{r}")
        occ_rows.append(f"occ{r}")

    nodes.append(helper.make_node("Concat", out_rows, ["fg_out"], axis=2))
    nodes.append(helper.make_node("Concat", occ_rows, ["occ_grid"], axis=2))
    nodes.append(helper.make_node("Not", ["occ_grid"], ["no_fg"]))
    nodes.append(helper.make_node("And", ["valid", "no_fg"], ["bg"]))
    nodes.append(helper.make_node("Concat", ["bg", "fg_out"], ["out6b"], axis=1))
    nodes.append(helper.make_node("Cast", ["out6b"], ["out6"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Pad", ["out6"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, 24, 24]))

    graph = helper.make_graph(nodes, "task032_gravity", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task032",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def validate(path: Path) -> None:
    data = json.loads(DATA_PATH.read_text())
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for split in ("train", "test", "arc-gen"):
        for i, example in enumerate(data[split]):
            h, w = len(example["input"]), len(example["input"][0])
            got = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(example["input"])})[0]
            grid = _onehot_to_grid(got, h, w)
            if grid != example["output"]:
                raise AssertionError((split, i, grid, example["output"]))
            expected = _grid_to_onehot(example["output"])
            if not np.array_equal(got > 0.0, expected > 0.0):
                raise AssertionError((split, i, "one-hot mismatch"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = build_model()
    onnx.save(model, BEST_PATH)
    validate(BEST_PATH)
    print(f"wrote {BEST_PATH}")


if __name__ == "__main__":
    main()
