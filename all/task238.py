"""ONNX solver for ARC task238: project a cyan stencil into a colored frame.

Task rule: the non-cyan foreground is a hollow rectangular frame with four
side colors.  The output is that frame moved to the top-left, with black
corners and the same side colors.  The separate cyan component has the same
bounding-box size as the frame interior; each cyan cell is projected into the
interior and recolored by the uniquely nearest stencil side.  If the nearest
side is tied, the projected cell stays cyan.  Non-cyan stencil cells are black.
Interior sizes in the data are 3x3, 4x4, or 5x5.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task238"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task238.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
SCAN = 18
MAX_OUT = 7
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _i32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int32), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _bool(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.bool_), name)


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def _onehot_to_grid(x: np.ndarray) -> np.ndarray:
    return x.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    frame = (g != 0) & (g != 8)
    fy, fx = np.where(frame)
    r0, r1 = int(fy.min()), int(fy.max())
    c0, c1 = int(fx.min()), int(fx.max())
    ih, iw = r1 - r0 - 1, c1 - c0 - 1
    t, b = int(g[r0, c0 + 1]), int(g[r1, c0 + 1])
    l, r = int(g[r0 + 1, c0]), int(g[r0 + 1, c1])

    cy, cx = np.where(g == 8)
    y0, x0 = int(cy.min()), int(cx.min())
    out = np.zeros((ih + 2, iw + 2), dtype=np.int64)
    out[0, 1:-1] = t
    out[-1, 1:-1] = b
    out[1:-1, 0] = l
    out[1:-1, -1] = r
    side_colors = {"T": t, "L": l, "R": r, "B": b}
    for rr in range(ih):
        for cc in range(iw):
            if g[y0 + rr, x0 + cc] != 8:
                continue
            distances = [("T", rr), ("L", cc), ("R", iw - 1 - cc), ("B", ih - 1 - rr)]
            nearest = min(d for _, d in distances)
            winners = [side for side, d in distances if d == nearest]
            out[rr + 1, cc + 1] = side_colors[winners[0]] if len(winners) == 1 else 8
    return out


def _side_index(nodes: List[onnx.NodeProto], inp: str, row: str, col: str, name: str) -> str:
    gathered_r = f"{name}_gather_r"
    gathered = f"{name}_gather"
    idx64 = f"{name}_idx64"
    out = f"{name}_idx"
    nodes.extend(
        [
            helper.make_node("Gather", [inp, row], [gathered_r], axis=2),
            helper.make_node("Gather", [gathered_r, col], [gathered], axis=3),
            helper.make_node("ArgMax", [gathered], [idx64], axis=1, keepdims=1),
            helper.make_node("Cast", [idx64], [out], to=TensorProto.INT32),
        ]
    )
    return out


def _make_masks(k: int) -> dict[str, np.ndarray]:
    masks = {name: np.zeros((1, 1, MAX_OUT, MAX_OUT), dtype=np.bool_) for name in ("BT", "BL", "BR", "BB", "T", "L", "R", "B", "8", "A", "BA")}
    masks["BT"][0, 0, 0, 1 : k + 1] = 1
    masks["BB"][0, 0, k + 1, 1 : k + 1] = 1
    masks["BL"][0, 0, 1 : k + 1, 0] = 1
    masks["BR"][0, 0, 1 : k + 1, k + 1] = 1
    masks["A"][0, 0, : k + 2, : k + 2] = 1
    masks["BA"] = masks["BT"] | masks["BL"] | masks["BR"] | masks["BB"]
    for rr in range(k):
        for cc in range(k):
            distances = [("T", rr), ("L", cc), ("R", k - 1 - cc), ("B", k - 1 - rr)]
            nearest = min(d for _, d in distances)
            winners = [side for side, d in distances if d == nearest]
            masks[winners[0] if len(winners) == 1 else "8"][0, 0, rr + 1, cc + 1] = 1
    return masks


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    _i64(inits, [1], "one_i")
    _i64(inits, [SCAN - 1], "last_i")
    _i32(inits, [0], "z32")
    _i32(inits, [8], "ch8_32")
    _i32(inits, [10], "ch10_32")
    _i64(inits, list(range(SCAN - 1, -1, -1)), "rev_scan")
    _f32(inits, np.array([0.0], dtype=np.float32), "zero_f")
    _i64(inits, [0, 0], "spatial_start")
    _i64(inits, [SCAN, SCAN], "spatial_end")
    _i64(inits, [2, 3], "spatial_axes")
    _i64(inits, [0, 0, 0], "bg_start")
    _i64(inits, [1, SCAN, SCAN], "bg_end")
    _i64(inits, [8, 0, 0], "cyan_start")
    _i64(inits, [9, SCAN, SCAN], "cyan_end")
    _i64(inits, [1, 2, 3], "channel_spatial_axes")
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, "bg_start", "bg_end", "channel_spatial_axes"], ["bg_in"]),
            helper.make_node("Slice", [IN_NAME, "cyan_start", "cyan_end", "channel_spatial_axes"], ["cyan"]),
            helper.make_node("ReduceMax", [IN_NAME], ["active"], axes=[1], keepdims=1),
            helper.make_node("Slice", ["active", "spatial_start", "spatial_end", "spatial_axes"], ["active_scan"]),
            helper.make_node("Greater", ["active_scan", "zero_f"], ["active_b"]),
            helper.make_node("Greater", ["bg_in", "zero_f"], ["bg_b"]),
            helper.make_node("Greater", ["cyan", "zero_f"], ["cyan_b"]),
            helper.make_node("Not", ["bg_b"], ["not_bg"]),
            helper.make_node("Not", ["cyan_b"], ["not_cyan"]),
            helper.make_node("And", ["active_b", "not_bg"], ["non_bg"]),
            helper.make_node("And", ["non_bg", "not_cyan"], ["frame_b"]),
            helper.make_node("Cast", ["frame_b"], ["frame"], to=TensorProto.FLOAT16),
            helper.make_node("ReduceMax", ["frame"], ["frame_rows4"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["frame"], ["frame_cols4"], axes=[2], keepdims=1),
            helper.make_node("Reshape", ["frame_rows4", "shape_scan"], ["frame_rows"]),
            helper.make_node("Reshape", ["frame_cols4", "shape_scan"], ["frame_cols"]),
            helper.make_node("ArgMax", ["frame_rows"], ["r0"], axis=0, keepdims=1),
            helper.make_node("ArgMax", ["frame_cols"], ["c0"], axis=0, keepdims=1),
            helper.make_node("Gather", ["frame_rows", "rev_scan"], ["frame_rows_rev"], axis=0),
            helper.make_node("Gather", ["frame_cols", "rev_scan"], ["frame_cols_rev"], axis=0),
            helper.make_node("ArgMax", ["frame_rows_rev"], ["rrev"], axis=0, keepdims=1),
            helper.make_node("ArgMax", ["frame_cols_rev"], ["crev"], axis=0, keepdims=1),
            helper.make_node("Sub", ["last_i", "rrev"], ["r1"]),
            helper.make_node("Sub", ["last_i", "crev"], ["c1"]),
            helper.make_node("Sub", ["r1", "r0"], ["rh_outer_minus1"]),
            helper.make_node("Sub", ["rh_outer_minus1", "one_i"], ["inner"]),
            helper.make_node("ReduceMax", ["cyan"], ["cyan_rows4"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["cyan"], ["cyan_cols4"], axes=[2], keepdims=1),
            helper.make_node("Reshape", ["cyan_rows4", "shape_scan"], ["cyan_rows"]),
            helper.make_node("Reshape", ["cyan_cols4", "shape_scan"], ["cyan_cols"]),
            helper.make_node("ArgMax", ["cyan_rows"], ["cy0"], axis=0, keepdims=1),
            helper.make_node("ArgMax", ["cyan_cols"], ["cx0"], axis=0, keepdims=1),
        ]
    )
    _i64(inits, [SCAN], "shape_scan")

    nodes.extend(
        [
            helper.make_node("Add", ["r0", "one_i"], ["r0p1"]),
            helper.make_node("Add", ["c0", "one_i"], ["c0p1"]),
        ]
    )
    top = _side_index(nodes, IN_NAME, "r0", "c0p1", "top")
    bottom = _side_index(nodes, IN_NAME, "r1", "c0p1", "bottom")
    left = _side_index(nodes, IN_NAME, "r0p1", "c0", "left")
    right = _side_index(nodes, IN_NAME, "r0p1", "c1", "right")

    candidates: list[str] = []
    for k in (3, 4, 5):
        if k < 5:
            _i64(inits, [k], f"k{k}_i")
        _i64(inits, list(range(k)), f"k{k}_rng")
        masks = _make_masks(k)
        for key in ("BT", "BL", "BR", "BB", "T", "L", "R", "B", "8", "A"):
            arr = masks[key]
            _bool(inits, arr, f"k{k}_{key}")
        nodes.extend(
            [
                helper.make_node("Add", ["cy0", f"k{k}_rng"], [f"k{k}_ry"]),
                helper.make_node("Add", ["cx0", f"k{k}_rng"], [f"k{k}_rx"]),
                helper.make_node("Gather", ["cyan", f"k{k}_ry"], [f"k{k}_st_rows"], axis=2),
                helper.make_node("Gather", [f"k{k}_st_rows", f"k{k}_rx"], [f"k{k}_st"], axis=3),
                helper.make_node(
                    "Pad",
                    [f"k{k}_st"],
                    [f"k{k}_st7"],
                    pads=[0, 0, 1, 1, 0, 0, MAX_OUT - k - 1, MAX_OUT - k - 1],
                ),
                helper.make_node("Greater", [f"k{k}_st7", "zero_f"], [f"k{k}_st7_b"]),
                helper.make_node("And", [f"k{k}_st7_b", f"k{k}_T"], [f"k{k}_st_t"]),
                helper.make_node("And", [f"k{k}_st7_b", f"k{k}_L"], [f"k{k}_st_l"]),
                helper.make_node("And", [f"k{k}_st7_b", f"k{k}_R"], [f"k{k}_st_r"]),
                helper.make_node("And", [f"k{k}_st7_b", f"k{k}_B"], [f"k{k}_st_b"]),
                helper.make_node("And", [f"k{k}_st7_b", f"k{k}_8"], [f"k{k}_st_8m"]),
                helper.make_node("Or", [f"k{k}_BT", f"k{k}_st_t"], [f"k{k}_top_m"]),
                helper.make_node("Or", [f"k{k}_BL", f"k{k}_st_l"], [f"k{k}_left_m"]),
                helper.make_node("Or", [f"k{k}_BR", f"k{k}_st_r"], [f"k{k}_right_m"]),
                helper.make_node("Or", [f"k{k}_BB", f"k{k}_st_b"], [f"k{k}_bottom_m"]),
                helper.make_node("Where", [f"k{k}_A", "z32", "ch10_32"], [f"k{k}_bg_idx"]),
                helper.make_node("Where", [f"k{k}_top_m", top, f"k{k}_bg_idx"], [f"k{k}_top_idx"]),
                helper.make_node("Where", [f"k{k}_left_m", left, f"k{k}_top_idx"], [f"k{k}_left_idx"]),
                helper.make_node("Where", [f"k{k}_right_m", right, f"k{k}_left_idx"], [f"k{k}_right_idx"]),
                helper.make_node("Where", [f"k{k}_bottom_m", bottom, f"k{k}_right_idx"], [f"k{k}_bottom_idx"]),
                helper.make_node("Where", [f"k{k}_st_8m", "ch8_32", f"k{k}_bottom_idx"], [f"k{k}_out_idx"]),
            ]
        )
        candidates.append(f"k{k}_out_idx")

    _i32(inits, np.arange(C, dtype=np.int32).reshape(1, C, 1, 1), "color_axis")

    nodes.extend(
        [
            helper.make_node("Equal", ["inner", "k3_i"], ["is3"]),
            helper.make_node("Equal", ["inner", "k4_i"], ["is4"]),
            helper.make_node("Where", ["is4", "k4_out_idx", "k5_out_idx"], ["out45_idx"]),
            helper.make_node("Where", ["is3", "k3_out_idx", "out45_idx"], ["out7_idx"]),
            helper.make_node("Equal", ["out7_idx", "color_axis"], ["out7_b"]),
            helper.make_node("Cast", ["out7_b"], ["out7"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out7"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - MAX_OUT, W - MAX_OUT]),
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


def _run(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate(model: onnx.ModelProto) -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            pred = _onehot_to_grid(_run(model, _grid_to_onehot(ex["input"])))
            expected = np.asarray(ex["output"], dtype=np.int64)
            actual = pred[: expected.shape[0], : expected.shape[1]]
            if not np.array_equal(actual, expected):
                raise AssertionError(f"{split} {idx} mismatch:\n{actual}\n{expected}")


def main() -> None:
    model = build_model()
    validate(model)
    onnx.save(model, BEST_PATH)
    result = score_file(BEST_PATH)
    print(result)


if __name__ == "__main__":
    main()
