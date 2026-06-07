"""ONNX for ARC task288: add upward-outward arms from the base marker.

Task rule: keep the input grid unchanged.  The last non-black row contains a
central segment of color A surrounded by color B, and the row above contains a
color-B segment aligned over that central A segment.  Starting just above the
left and right ends of the upper segment, fill the two outward upward diagonals
with color A until the grid boundary.  The task grids are odd square grids up to
9x9, embedded in the NeuroGolf 30x30 one-hot canvas.

ONNX approach: work only on the top-left 9x9 region.  Detect the six possible
upper segment geometries from their occupied cells, slice color A from the
matching bottom segment endpoint, combine the selected arm updates at the unique
affected positions, scatter them into the boolean core, then cast and pad to
30x30.
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
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task288"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task288.onnx"
DATA_PATH = ROOT / "data" / "task288.json"

C = 10
H = W = 30
N = 9
SHAPE = [1, C, H, W]
CORE_SHAPE = [1, C, N, N]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def _init(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], name: str, vals: Any) -> str:
    return _init(inits, name, np.asarray(vals, dtype=np.int64))


def _f32(inits: list[onnx.TensorProto], name: str, vals: Any) -> str:
    return _init(inits, name, np.asarray(vals, dtype=np.float32))


def _bool(inits: list[onnx.TensorProto], name: str, vals: Any) -> str:
    return _init(inits, name, np.asarray(vals, dtype=np.bool_))


def _arm_mask(size: int, bottom_row: int, left: int, right: int) -> np.ndarray:
    mask = np.zeros((1, 1, N, N), dtype=np.bool_)
    upper_row = bottom_row - 1
    for step in range(1, size):
        row = upper_row - step
        left_col = left - step
        right_col = right + step
        if row < 0:
            break
        if 0 <= left_col < size:
            mask[0, 0, row, left_col] = True
        if 0 <= right_col < size:
            mask[0, 0, row, right_col] = True
    return mask


def _arm_positions(size: int, bottom_row: int, left: int, right: int) -> np.ndarray:
    mask = _arm_mask(size, bottom_row, left, right)[0, 0]
    return np.asarray([r * N + c for r, c in np.argwhere(mask)], dtype=np.int64)


def _make_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto]) -> onnx.ModelProto:
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


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    zero = _f32(inits, "z", np.array(0.0, dtype=np.float32))
    core_st = _i64(inits, "cs", [0, 0, 0, 0])
    core_en = _i64(inits, "ce", [1, C, N, N])
    axes_sp = _i64(inits, "ax", [2, 3])
    axes_csp = _i64(inits, "acsp", [1, 2, 3])
    flat_shape = _i64(inits, "flat_shape", [1, C, N * N])
    core_shape = _i64(inits, "core_shape", CORE_SHAPE)
    shape_101 = _i64(inits, "shape_101", [1, C, 1])
    shape_111 = _i64(inits, "shape_111", [1, 1, 1])

    nodes.append(helper.make_node("Slice", [IN_NAME, core_st, core_en], ["x"]))
    nodes.append(helper.make_node("Greater", ["x", zero], ["xb"]))
    nodes.append(helper.make_node("Reshape", ["xb", flat_shape], ["cur0"]))

    # (actual grid size, upper segment left column, upper segment right column)
    cases = [
        (3, 1, 1),
        (5, 1, 3),
        (5, 2, 2),
        (7, 2, 4),
        (7, 3, 3),
        (9, 3, 5),
    ]

    size_active: dict[int, str] = {}
    for size in sorted({case[0] for case in cases}):
        bottom = size - 1
        size_st = _i64(inits, f"szs{size}", [1, bottom, 0])
        size_en = _i64(inits, f"sze{size}", [C, bottom + 1, 1])
        nodes.append(helper.make_node("Slice", ["x", size_st, size_en, axes_csp], [f"szc{size}"]))
        nodes.append(helper.make_node("ReduceSum", [f"szc{size}"], [f"szsum{size}"], axes=[1], keepdims=1))
        nodes.append(helper.make_node("Greater", [f"szsum{size}", zero], [f"sz{size}"]))
        size_active[size] = f"sz{size}"

    all_positions = sorted(
        {
            int(pos)
            for size, left, right in cases
            for pos in _arm_positions(size, size - 1, left, right)
        }
    )
    pos_lookup = {pos: idx for idx, pos in enumerate(all_positions)}
    unique_pos = _i64(inits, "upos", all_positions)
    scatter_idx = _i64(inits, "scatter_idx", np.tile(np.asarray(all_positions, dtype=np.int64), (1, C, 1)))

    fill_color: str | None = None
    fill_any: str | None = None
    active_by_geometry: dict[tuple[int, int, int], str] = {}
    for idx, (size, left, right) in enumerate(cases):
        bottom = size - 1
        upper = bottom - 1

        wider = active_by_geometry.get((size, left - 1, right + 1))
        if left == right or size == 9:
            active = size_active[size]
        else:
            black_cell_st = _i64(inits, f"s{idx}", [0, upper, left])
            black_cell_en = _i64(inits, f"e{idx}", [1, upper + 1, left + 1])
            nodes.append(helper.make_node("Slice", ["xb", black_cell_st, black_cell_en, axes_csp], [f"bc{idx}"]))
            nodes.append(helper.make_node("Not", [f"bc{idx}"], [f"nb{idx}"]))
            nodes.append(helper.make_node("And", [size_active[size], f"nb{idx}"], [f"a{idx}"]))
            active = f"a{idx}"
        if wider is not None:
            nodes.append(helper.make_node("Not", [wider], [f"ng{idx}"]))
            nodes.append(helper.make_node("And", [active, f"ng{idx}"], [f"ok{idx}"]))
            active = f"ok{idx}"
        active_by_geometry[(size, left, right)] = active

        color_st = _i64(inits, f"ks{idx}", [bottom, left])
        color_en = _i64(inits, f"ke{idx}", [bottom + 1, left + 1])
        membership = np.zeros((1, 1, len(all_positions)), dtype=np.bool_)
        for pos in _arm_positions(size, bottom, left, right):
            membership[0, 0, pos_lookup[int(pos)]] = True
        mem = _bool(inits, f"mem{idx}", membership)

        nodes.append(helper.make_node("Slice", ["xb", color_st, color_en, axes_sp], [f"k{idx}"]))
        nodes.append(helper.make_node("And", [f"k{idx}", active], [f"ka4_{idx}"]))
        nodes.append(helper.make_node("Reshape", [f"ka4_{idx}", shape_101], [f"ka{idx}"]))
        nodes.append(helper.make_node("Reshape", [active, shape_111], [f"act{idx}"]))
        nodes.append(helper.make_node("And", [f"act{idx}", mem], [f"cm{idx}"]))
        nodes.append(helper.make_node("And", [f"ka{idx}", f"cm{idx}"], [f"fill{idx}"]))
        if fill_color is None or fill_any is None:
            fill_color = f"fill{idx}"
            fill_any = f"cm{idx}"
        else:
            nodes.append(helper.make_node("Or", [fill_color, f"fill{idx}"], [f"fc{idx}"]))
            nodes.append(helper.make_node("Or", [fill_any, f"cm{idx}"], [f"fa{idx}"]))
            fill_color = f"fc{idx}"
            fill_any = f"fa{idx}"

    if fill_color is None or fill_any is None:
        raise AssertionError("no geometry cases configured")
    nodes.append(helper.make_node("Gather", ["cur0", unique_pos], ["old"], axis=2))
    nodes.append(helper.make_node("Not", [fill_any], ["nfill"]))
    nodes.append(helper.make_node("And", ["old", "nfill"], ["keep"]))
    nodes.append(helper.make_node("Or", ["keep", fill_color], ["upd"]))
    nodes.append(helper.make_node("Scatter", ["cur0", scatter_idx, "upd"], ["cur"], axis=2))
    nodes.append(helper.make_node("Reshape", ["cur", core_shape], ["yb"]))
    nodes.append(helper.make_node("Cast", ["yb"], ["yf"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Pad", ["yf"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - N, W - N]))
    return _make_model(nodes, inits)


def solve_grid(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = arr.copy()
    bottom = max(r for r in range(arr.shape[0]) if np.any(arr[r] != 0))
    upper = bottom - 1
    upper_cols = np.flatnonzero(arr[upper] != 0)
    left, right = int(upper_cols[0]), int(upper_cols[-1])
    color = int(arr[bottom, left])
    for step in range(1, max(arr.shape) + 1):
        row = upper - step
        if row < 0:
            break
        lc = left - step
        rc = right + step
        if lc >= 0 and out[row, lc] == 0:
            out[row, lc] = color
        if rc < arr.shape[1] and out[row, rc] == 0:
            out[row, rc] = color
    return out


def _onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _decoded(pred: np.ndarray) -> np.ndarray:
    active = pred[0] > 0.0
    return np.argmax(active, axis=0)


def validate(path: Path) -> None:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            expected_grid = np.asarray(ex["output"], dtype=np.int64)
            ref = solve_grid(ex["input"])
            if not np.array_equal(ref, expected_grid):
                raise AssertionError(f"reference solver failed {split}[{idx}]")

            pred = session.run([OUT_NAME], {IN_NAME: _onehot(ex["input"])})[0]
            expected = _onehot(expected_grid)
            if not np.array_equal(pred > 0.0, expected > 0.0):
                diff = np.argwhere((pred > 0.0) != (expected > 0.0))
                got = _decoded(pred)[: expected_grid.shape[0], : expected_grid.shape[1]]
                raise AssertionError(
                    f"{split}[{idx}] failed at {diff[:8].tolist()}\n"
                    f"got:\n{got}\nexpected:\n{expected_grid}"
                )


def tensor_count(model: onnx.ModelProto) -> int:
    names = {value.name for value in model.graph.input}
    names.update(value.name for value in model.graph.output)
    names.update(init.name for init in model.graph.initializer)
    for node in model.graph.node:
        names.update(name for name in node.output if name)
    return len(names)


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    validate(BEST_PATH)
    result = score_file(BEST_PATH)
    if not result["valid"]:
        raise RuntimeError(result["error"])
    print(
        f"wrote {BEST_PATH}: tensors={tensor_count(model)} nodes={len(model.graph.node)} "
        f"memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={float(result['score']):.6f}"
    )


if __name__ == "__main__":
    main()
