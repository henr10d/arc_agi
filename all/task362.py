"""ONNX solution for task362: move a colored 10x10 cross by the gray marker.

Task rule: the input is a 10x10 grid containing one non-gray colored cross and
a gray marker in column 9 starting at the top row. Remove the gray marker and
translate the cross down and left by the marker height (1, 2, or 3 cells),
keeping the same foreground color and clipping to the 10x10 grid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "task362"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
N = 10
PAD = H - N
GRAY = 5
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def solve(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    """Reference solver for the JSON grids."""
    x = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(x)
    colors = sorted(int(v) for v in np.unique(x) if v not in (0, GRAY))
    if not colors:
        return out
    fg = colors[0]
    gray_height = int(np.count_nonzero(x == GRAY))
    fg_cells = np.argwhere(x == fg)

    row_counts = np.bincount(fg_cells[:, 0], minlength=x.shape[0])
    col_counts = np.bincount(fg_cells[:, 1], minlength=x.shape[1])
    center_row = int(np.argmax(row_counts))
    center_col = int(np.argmax(col_counts))
    target_row = center_row + gray_height
    target_col = center_col - gray_height

    if 0 <= target_row < x.shape[0]:
        out[target_row, :] = fg
    if 0 <= target_col < x.shape[1]:
        out[:, target_col] = fg
    return out


def _i64(inits: list[onnx.TensorProto], vals: list[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))
    return name


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.float32), name=name))
    return name


def _bool(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=bool), name=name))
    return name


def _grid_to_onehot(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _expected_onehot(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    return _grid_to_onehot(grid)


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _slice_node(inp: str, out: str, starts: str, ends: str, axes: str) -> onnx.NodeProto:
    return helper.make_node("Slice", [inp, starts, ends, axes], [out])


def build_onnx_model() -> onnx.ModelProto:
    """Build a static opset-10 graph for the marker-height diagonal shift."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes3 = _i64(inits, [1, 2, 3], "a3")
    ch0_st = _i64(inits, [0, 0, 0], "ch0_s")
    ch0_en = _i64(inits, [1, N, N], "ch0_e")
    gray_st = _i64(inits, [GRAY, 0, N - 1], "gray_s")
    gray_en = _i64(inits, [GRAY + 1, 4, N], "gray_e")
    row_grid = _i64(inits, np.arange(N, dtype=np.int64).reshape(1, 1, N, 1), "row_grid")
    col_grid = _i64(inits, np.arange(N, dtype=np.int64).reshape(1, 1, 1, N), "col_grid")
    bg_channel = np.zeros((1, C, 1, 1), dtype=np.float32)
    bg_channel[:, 0, :, :] = 1.0
    bg_channel_name = _f32(inits, bg_channel, "bg_ch")
    fg_channel_mask = np.ones((1, C, 1, 1), dtype=bool)
    fg_channel_mask[:, [0, GRAY], :, :] = False
    channel_mask = _bool(inits, fg_channel_mask, "fg_ch_mask")

    nodes.extend(
        [
            _slice_node(IN_NAME, "ch0", ch0_st, ch0_en, axes3),
            helper.make_node("ReduceSum", ["ch0"], ["row_black_sum"], axes=[3], keepdims=1),
            helper.make_node("ReduceSum", ["ch0"], ["col_black_sum"], axes=[2], keepdims=1),
            helper.make_node("ArgMin", ["row_black_sum"], ["row_idx"], axis=2, keepdims=0),
            helper.make_node("ArgMin", ["col_black_sum"], ["col_idx"], axis=3, keepdims=0),
            helper.make_node("ReduceMax", [IN_NAME], ["all_channels"], axes=[2, 3], keepdims=1),
            helper.make_node("Cast", ["all_channels"], ["all_channels_b"], to=TensorProto.BOOL),
            helper.make_node("And", ["all_channels_b", channel_mask], ["fg_channel_b"]),
            helper.make_node("Cast", ["fg_channel_b"], ["fg_channel"], to=TensorProto.FLOAT),
            _slice_node(IN_NAME, "gray_core", gray_st, gray_en, axes3),
            helper.make_node("ArgMin", ["gray_core"], ["gray_count"], axis=2, keepdims=0),
            helper.make_node("Add", ["row_idx", "gray_count"], ["target_row_idx"]),
            helper.make_node("Sub", ["col_idx", "gray_count"], ["target_col_idx"]),
        ]
    )

    nodes.extend(
        [
            helper.make_node("Equal", ["target_row_idx", row_grid], ["target_row"]),
            helper.make_node("Equal", ["target_col_idx", col_grid], ["target_col"]),
            helper.make_node("Or", ["target_row", "target_col"], ["cross_mask_b"]),
            helper.make_node("Where", ["cross_mask_b", "fg_channel", bg_channel_name], ["out10"]),
            helper.make_node("Pad", ["out10"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]),
        ]
    )

    graph = helper.make_graph(nodes, "task362", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def save_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    model = build_onnx_model()
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return model


def validate_training_mapping() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    pairs: list[tuple[int, int, int, int, int]] = []
    for ex in data["train"]:
        inp = np.asarray(ex["input"], dtype=np.int64)
        out = np.asarray(ex["output"], dtype=np.int64)
        fg = next(int(v) for v in np.unique(inp) if v not in (0, GRAY))
        cells = np.argwhere(inp == fg)
        out_cells = np.argwhere(out == fg)
        in_row = int(np.argmax(np.bincount(cells[:, 0], minlength=N)))
        in_col = int(np.argmax(np.bincount(cells[:, 1], minlength=N)))
        out_row = int(np.argmax(np.bincount(out_cells[:, 0], minlength=N)))
        out_col = int(np.argmax(np.bincount(out_cells[:, 1], minlength=N)))
        pairs.append((int(np.count_nonzero(inp == GRAY)), in_row, in_col, out_row, out_col))
    if not all((r + g, c - g) == (orow, ocol) for g, r, c, orow, ocol in pairs):
        raise AssertionError(f"marker-height shift hypothesis failed: {pairs}")


def validate_model(model: onnx.ModelProto) -> None:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            pred = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(ex["input"])})[0]
            expected = _expected_onehot(ex["output"])
            if not np.array_equal(pred > 0.0, expected > 0.0):
                grid = _onehot_to_grid(pred[0])
                raise AssertionError(
                    f"{split}#{idx} mismatch\nexpected={ex['output']}\npred={grid[:N, :N].tolist()}"
                )


def main() -> None:
    validate_training_mapping()
    model = save_model()
    validate_model(model)
    print(f"saved {BEST_PATH}")


if __name__ == "__main__":
    main()
