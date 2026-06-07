"""Optimized ONNX for ARC task030: align objects to the blue object's row.

Task rule: the input contains three colored objects, using blue (1), red (2),
and yellow (4). Keep each object's shape and x coordinates, but vertically move
the red and yellow objects so their top row matches the blue object's top row.
The blue object remains fixed. The output canvas size matches the input canvas;
padding outside the original grid remains all-zero.

ONNX: operate on the task's 10x10 maximum canvas, find top rows with compact
ReduceMin masks, synthesize shifted red/yellow channels by broadcasting each
source row to its target row, preserve blue plus any non-moving channels, then
pad back to the NeuroGolf 30x30 I/O contract.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task030"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task030.onnx"
DATA_PATH = ROOT / "data" / "task030.json"

C = 10
H = W = 30
SH = SW = 10
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def solve(grid: np.ndarray) -> np.ndarray:
    """Move red/yellow vertically so their top rows equal blue's top row."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    blue_rows = np.where(g == 1)[0]
    if blue_rows.size == 0:
        return out
    target = int(blue_rows.min())
    for color in (2, 4):
        mask = g == color
        rows = np.where(mask)[0]
        if rows.size == 0:
            continue
        shifted = np.zeros_like(mask)
        delta = target - int(rows.min())
        src_r, src_c = np.where(mask)
        dst_r = src_r + delta
        keep = (0 <= dst_r) & (dst_r < g.shape[0])
        out[mask] = 0
        shifted[dst_r[keep], src_c[keep]] = True
        out[shifted] = color
    return out


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _top_row(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    mask: str,
    tag: str,
    rows: str,
    big: str,
    half: str,
    axes_w: list[int],
) -> str:
    occ_f = f"{tag}_occf"
    occ = f"{tag}_occ"
    where = f"{tag}_where"
    top4 = f"{tag}_top4"
    top = f"{tag}_top"
    nodes.extend(
        [
            helper.make_node("Cast", [mask], [occ_f], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", [occ_f], [occ], axes=axes_w, keepdims=1),
            helper.make_node("Greater", [occ, half], [f"{tag}_has"]),
            helper.make_node("Where", [f"{tag}_has", rows, big], [where]),
            helper.make_node("ReduceMin", [where], [top4], axes=[2], keepdims=1),
            helper.make_node("Squeeze", [top4], [top], axes=[0, 1, 2]),
        ]
    )
    return top


def _scalar(inits: list[onnx.TensorProto], val: float, name: str) -> str:
    return _f32(inits, [val], name)


def _shift_to_blue(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    ch: str,
    src_top: str,
    blue_top: str,
    tag: str,
    rows1: str,
    zero10: str,
    nine10: str,
    valid_shape: str,
    half: str,
) -> str:
    delta = f"{tag}_delta"
    src_idx_f = f"{tag}_srcidxf"
    low = f"{tag}_low"
    high = f"{tag}_high"
    not_low = f"{tag}_notlow"
    not_high = f"{tag}_nothigh"
    valid = f"{tag}_valid"
    valid4 = f"{tag}_valid4"
    valid_f = f"{tag}_validf"
    clamp_low = f"{tag}_clamplow"
    clamp = f"{tag}_clamp"
    idx = f"{tag}_idx"
    gathered = f"{tag}_gather"
    masked = f"{tag}_masked"
    out = f"{tag}_out"
    nodes.extend(
        [
            helper.make_node("Sub", [blue_top, src_top], [delta]),
            helper.make_node("Sub", [rows1, delta], [src_idx_f]),
            helper.make_node("Less", [src_idx_f, zero10], [low]),
            helper.make_node("Greater", [src_idx_f, nine10], [high]),
            helper.make_node("Where", [low, zero10, src_idx_f], [clamp_low]),
            helper.make_node("Where", [high, nine10, clamp_low], [clamp]),
            helper.make_node("Cast", [clamp], [idx], to=TensorProto.INT64),
            helper.make_node("Gather", [ch, idx], [gathered], axis=2),
            helper.make_node("Not", [low], [not_low]),
            helper.make_node("Not", [high], [not_high]),
            helper.make_node("And", [not_low, not_high], [valid]),
            helper.make_node("Reshape", [valid, valid_shape], [valid4]),
            helper.make_node("Cast", [valid4], [valid_f], to=TensorProto.FLOAT),
            helper.make_node("Mul", [gathered, valid_f], [masked]),
            helper.make_node("Greater", [masked, half], [out]),
        ]
    )
    return out


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    half = _f32(inits, [0.5], "half")
    rows = _f32(inits, np.arange(SH, dtype=np.float32).reshape(1, 1, SH, 1), "rows")
    rows1 = _f32(inits, np.arange(SH, dtype=np.float32), "rows1")
    zero10 = _f32(inits, np.zeros((SH,), dtype=np.float32), "zero10")
    nine10 = _f32(inits, np.full((SH,), SH - 1, dtype=np.float32), "nine10")
    big = _f32(inits, np.full((1, 1, SH, 1), 99.0, dtype=np.float32), "big")
    valid_shape = _i64(inits, [1, 1, SH, 1], "valid_shape")

    def channel_slice(c: int, name: str) -> str:
        st = _i64(inits, [0, c, 0, 0], f"{name}_st")
        en = _i64(inits, [1, c + 1, SH, SW], f"{name}_en")
        nodes.append(helper.make_node("Slice", [IN_NAME, st, en, axes4], [name]))
        return name

    ch0 = channel_slice(0, "ch0")
    ch1 = channel_slice(1, "ch1")
    ch2 = channel_slice(2, "ch2")
    ch3 = channel_slice(3, "ch3")
    ch4 = channel_slice(4, "ch4")
    ch5_st = _i64(inits, [0, 5, 0, 0], "ch5_st")
    ch5_en = _i64(inits, [1, 10, SH, SW], "ch5_en")
    nodes.append(helper.make_node("Slice", [IN_NAME, ch5_st, ch5_en, axes4], ["ch5_9"]))

    nodes.extend(
        [
            helper.make_node("Greater", [ch0, half], ["m0"]),
            helper.make_node("Greater", [ch1, half], ["m1"]),
            helper.make_node("Greater", [ch2, half], ["m2"]),
            helper.make_node("Greater", [ch3, half], ["m3"]),
            helper.make_node("Greater", [ch4, half], ["m4"]),
            helper.make_node("ReduceMax", ["ch5_9"], ["ch5_any_f"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["ch5_any_f", half], ["m5_9"]),
            helper.make_node("Or", ["m0", "m1"], ["act1"]),
            helper.make_node("Or", ["act1", "m2"], ["act2"]),
            helper.make_node("Or", ["act2", "m3"], ["act3"]),
            helper.make_node("Or", ["act3", "m4"], ["act4"]),
            helper.make_node("Or", ["act4", "m5_9"], ["active"]),
        ]
    )

    blue_top = _top_row(nodes, inits, "m1", "blue", rows, big, half, [3])
    red_top = _top_row(nodes, inits, "m2", "red", rows, big, half, [3])
    yellow_top = _top_row(nodes, inits, "m4", "yellow", rows, big, half, [3])

    out2 = _shift_to_blue(nodes, inits, ch2, red_top, blue_top, "red", rows1, zero10, nine10, valid_shape, half)
    out4 = _shift_to_blue(nodes, inits, ch4, yellow_top, blue_top, "yellow", rows1, zero10, nine10, valid_shape, half)

    nodes.extend(
        [
            helper.make_node("Cast", [out2], ["out2f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", [out4], ["out4f"], to=TensorProto.FLOAT),
            helper.make_node("Concat", [ch1, "out2f", ch3, "out4f", "ch5_9"], ["nonzero"], axis=1),
            helper.make_node("ReduceMax", ["nonzero"], ["any_fg_f"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["any_fg_f", half], ["any_fg"]),
            helper.make_node("Not", ["any_fg"], ["no_fg"]),
            helper.make_node("And", ["no_fg", "active"], ["bg"]),
            helper.make_node("Cast", ["bg"], ["bgf"], to=TensorProto.FLOAT),
        ]
    )

    out_channels = ["bgf", ch1, "out2f", ch3, "out4f", "ch5_9"]

    nodes.extend(
        [
            helper.make_node("Concat", out_channels, ["out10"], axis=1),
            helper.make_node("Pad", ["out10"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - SH, W - SW]),
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


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> tuple[int, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            g = np.array(ex["input"], dtype=np.int64)
            if g.shape[0] > H or g.shape[1] > W:
                continue
            total += 1
            pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(ex["input"])))[: g.shape[0], : g.shape[1]]
            exp = np.array(ex["output"], dtype=np.int64)
            ref = solve(g)
            if not np.array_equal(ref, exp) or not np.array_equal(pred, exp):
                bad += 1
                if bad <= 3:
                    print(f"mismatch {split} #{total}")
                    print("pred:\n", pred)
                    print("exp:\n", exp)
                    print("ref:\n", ref)
    return bad, total


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    bad, total = validate_json(model)
    print(f"{DATA_PATH.name}: {'PASS' if bad == 0 else f'FAIL ({bad}/{total} wrong)'}")

    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"nodes:   {len(model.graph.node)}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
