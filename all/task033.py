"""Generate a compact ONNX solution for NeuroGolf task033.

Task rule: the full shape in the top-left compartment is the template. The
input is split by complete separator-color rows and columns into equal
compartments; every compartment must contain the same relative template cells.
Existing object-color cells stay unchanged, missing template cells are filled
with the separator color, separator lines stay separator-colored, and all other
cells remain background.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "task033"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
IR_VERSION = 10


def _load_task() -> dict:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _separator_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h_rows = [r for r in range(grid.shape[0]) if grid[r, 0] and np.all(grid[r] == grid[r, 0])]
    v_cols = [c for c in range(grid.shape[1]) if grid[0, c] and np.all(grid[:, c] == grid[0, c])]
    if not h_rows or h_rows != v_cols:
        raise ValueError(f"unexpected separator geometry: rows={h_rows}, cols={v_cols}")
    return h_rows, v_cols


def _derive_geometry() -> tuple[int, list[int], np.ndarray]:
    train = _load_task()["train"]
    first = np.asarray(train[0]["input"], dtype=np.int64)
    sep_rows, sep_cols = _separator_lines(first)
    for ex in train[1:]:
        rows, cols = _separator_lines(np.asarray(ex["input"], dtype=np.int64))
        if rows != sep_rows or cols != sep_cols:
            raise ValueError("training examples disagree on separator geometry")

    grid_h, grid_w = first.shape
    starts_r = [0] + [r + 1 for r in sep_rows]
    starts_c = [0] + [c + 1 for c in sep_cols]
    stops_r = sep_rows + [grid_h]
    stops_c = sep_cols + [grid_w]
    cell_h = stops_r[0] - starts_r[0]
    cell_w = stops_c[0] - starts_c[0]
    if cell_h != cell_w:
        raise ValueError("task033 expected square compartments")

    # Gather map from the flattened template plus one appended false value.
    false_idx = cell_h * cell_w
    gather = np.full((grid_h, grid_w), false_idx, dtype=np.int64)
    for r0, r1 in zip(starts_r, stops_r):
        for c0, c1 in zip(starts_c, stops_c):
            for rr in range(r1 - r0):
                for cc in range(c1 - c0):
                    gather[r0 + rr, c0 + cc] = rr * cell_w + cc
    return grid_h, sep_rows, gather


def _i64(inits: list[onnx.TensorProto], vals: Iterable[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(list(vals), dtype=np.int64), name=name))
    return name


def _arr(inits: list[onnx.TensorProto], arr: np.ndarray, name: str) -> str:
    inits.append(numpy_helper.from_array(arr, name=name))
    return name


def build_model(opset: int = 13) -> onnx.ModelProto:
    active, sep_rows, gather_idx = _derive_geometry()
    cell = sep_rows[0]
    false_idx = cell * cell
    pad = H - active

    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    st0 = _i64(inits, [0, 0, 0, 0], "st0")
    end_active = _i64(inits, [1, C, active, active], "end_active")
    end_tl = _i64(inits, [1, 1, cell, cell], "end_tl")
    ch0_end = _i64(inits, [1, 1, active, active], "ch0_end")
    sep_start = _i64(inits, [0, 0, sep_rows[0], 0], "sep_start")
    sep_end = _i64(inits, [1, C, sep_rows[0] + 1, 1], "sep_end")
    flat_shape = _i64(inits, [false_idx], "flat_shape")
    template_shape = _i64(inits, [1, 1, false_idx + 1], "template_shape")
    pad_bool = _i64(inits, [0, 0, 0, 0, 0, 0, pad, pad], "pad_bool")
    false_b = _arr(inits, np.asarray([False], dtype=bool), "false_b")
    ch0_sel = _arr(inits, (np.arange(C).reshape(1, C, 1, 1) == 0).astype(np.uint8), "ch0_sel")
    gather = _arr(inits, gather_idx, "gather")

    nodes.extend(
        [
            helper.make_node("Cast", [IN_NAME], ["xallu"], to=TensorProto.UINT8),
            helper.make_node("Slice", ["xallu", st0, end_active, axes4], ["x17u"]),
            helper.make_node("Slice", ["x17u", st0, ch0_end, axes4], ["x0u"]),
            helper.make_node("Cast", ["x0u"], ["x0"], to=TensorProto.BOOL),
            helper.make_node("Not", ["x0"], ["non_bg"]),
            helper.make_node("Slice", ["x0", st0, end_tl, axes4], ["tl_bg"]),
            helper.make_node("Not", ["tl_bg"], ["tl_mask"]),
            helper.make_node("Reshape", ["tl_mask", flat_shape], ["tl_flat"]),
            helper.make_node("Concat", ["tl_flat", false_b], ["tl_plus"], axis=0),
            helper.make_node("Reshape", ["tl_plus", template_shape], ["tl_plus3"]),
            helper.make_node("Gather", ["tl_plus3", gather], ["need"], axis=2),
            helper.make_node("Not", ["non_bg"], ["missing"]),
            helper.make_node("And", ["need", "missing"], ["fill_cond"]),
            helper.make_node("Slice", ["x17u", sep_start, sep_end, axes4], ["sep_color_u"]),
            helper.make_node("Where", ["fill_cond", "sep_color_u", "ch0_sel"], ["missing_color"]),
            helper.make_node("Where", ["non_bg", "x17u", "missing_color"], ["y17u"]),
        ]
    )
    if opset >= 11:
        nodes.append(helper.make_node("Pad", ["y17u", pad_bool], ["y30u"]))
        nodes.append(helper.make_node("Cast", ["y30u"], [OUT_NAME], to=TensorProto.FLOAT))
    else:
        nodes.append(helper.make_node("Cast", ["y17u"], ["y17f"], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Pad", ["y17f"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, pad, pad]))

    graph = helper.make_graph(nodes, "task033", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def verify(path: Path) -> None:
    task = _load_task()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(task[split]):
            got = sess.run([OUT_NAME], {IN_NAME: _onehot(ex["input"])})[0]
            expected = _onehot(ex["output"])
            if not np.array_equal(got > 0.0, expected > 0.0):
                raise AssertionError(f"{split}[{idx}] failed")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Opset 13 is measurably smaller for this graph: Pad can consume the
    # uint8 working tensor directly, avoiding an internal compact float copy.
    model = build_model(opset=13)
    onnx.save(model, str(BEST_PATH))
    verify(BEST_PATH)
    print(f"wrote {BEST_PATH}")


if __name__ == "__main__":
    main()
