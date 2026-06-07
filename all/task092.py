"""Minimal ONNX for ARC task092: connect paired same-color cells with line segments.

Task rule: each nonzero color appears in exactly two cells aligned horizontally
(same row) or vertically (same column). Output fills the inclusive straight
segment between each pair with that color. When horizontal and vertical segments
cross, the vertical segment wins at the intersection. Background is 0.

ONNX: ArgMax to a compact id grid, per-color scalar min/max row/col bounds,
horizontal and vertical bool line masks, compose one bool output channel per
color, and Cast the final bool one-hot tensor directly to the float output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task092.onnx"
DATA_PATH = ROOT / "data" / "task092.json"

C = 10
H = W = 30
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


def _i32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int32), name)


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference: horiz segments first, then vertical segments overwrite."""
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    horiz = np.zeros((h, w), dtype=np.int64)
    vert = np.zeros((h, w), dtype=np.int64)
    for color in range(1, 10):
        pts = list(zip(*np.where(g == color)))
        if len(pts) != 2:
            continue
        (r0, c0), (r1, c1) = pts
        if r0 == r1:
            cmin, cmax = min(c0, c1), max(c0, c1)
            horiz[r0, cmin : cmax + 1] = color
        elif c0 == c1:
            rmin, rmax = min(r0, r1), max(r0, r1)
            vert[rmin : rmax + 1, c0] = color
    out = horiz.copy()
    mask = vert != 0
    out[mask] = vert[mask]
    return out


def _line_masks(
    nodes: List[onnx.NodeProto],
    nzf: str,
    tag: str,
    rows: str,
    cols: str,
    big: str,
    zero_i: str,
    zero_f: str,
    one_half_f: str,
) -> tuple[str, str]:
    """Horizontal and vertical inclusive segment masks for one color."""
    min_r = f"{tag}_mr"
    max_r = f"{tag}_xr"
    min_c = f"{tag}_mc"
    max_c = f"{tag}_xc"
    hz = f"{tag}_hz"
    vt = f"{tag}_vt"

    nodes.extend(
        [
            helper.make_node("ReduceSum", [nzf], [f"{tag}_rowv"], axes=[3], keepdims=1),
            helper.make_node("ReduceSum", [nzf], [f"{tag}_colv"], axes=[2], keepdims=1),
            helper.make_node("Greater", [f"{tag}_rowv", zero_f], [f"{tag}_rh"]),
            helper.make_node("Greater", [f"{tag}_colv", zero_f], [f"{tag}_ch"]),
            helper.make_node("Where", [f"{tag}_rh", rows, big], [f"{tag}_mrsrc"]),
            helper.make_node("Where", [f"{tag}_rh", rows, zero_i], [f"{tag}_xrsrc"]),
            helper.make_node("Where", [f"{tag}_ch", cols, big], [f"{tag}_mcsrc"]),
            helper.make_node("Where", [f"{tag}_ch", cols, zero_i], [f"{tag}_xcsrc"]),
            helper.make_node("ReduceMin", [f"{tag}_mrsrc"], [min_r], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMax", [f"{tag}_xrsrc"], [max_r], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMin", [f"{tag}_mcsrc"], [min_c], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMax", [f"{tag}_xcsrc"], [max_c], axes=[2, 3], keepdims=1),
            helper.make_node("Greater", [f"{tag}_rowv", one_half_f], [f"{tag}_r2"]),
            helper.make_node("Greater", [f"{tag}_colv", one_half_f], [f"{tag}_c2"]),
            helper.make_node("Less", [cols, min_c], [f"{tag}_ltc"]),
            helper.make_node("Greater", [cols, max_c], [f"{tag}_gtc"]),
            helper.make_node("Less", [rows, min_r], [f"{tag}_ltr"]),
            helper.make_node("Greater", [rows, max_r], [f"{tag}_gtr"]),
            helper.make_node("Not", [f"{tag}_ltc"], [f"{tag}_cge"]),
            helper.make_node("Not", [f"{tag}_gtc"], [f"{tag}_cle"]),
            helper.make_node("And", [f"{tag}_cge", f"{tag}_cle"], [f"{tag}_crng"]),
            helper.make_node("Not", [f"{tag}_ltr"], [f"{tag}_rge"]),
            helper.make_node("Not", [f"{tag}_gtr"], [f"{tag}_rle"]),
            helper.make_node("And", [f"{tag}_rge", f"{tag}_rle"], [f"{tag}_rrng"]),
            helper.make_node("And", [f"{tag}_r2", f"{tag}_crng"], [hz]),
            helper.make_node("And", [f"{tag}_c2", f"{tag}_rrng"], [vt]),
        ]
    )
    return hz, vt


def _or_chain(nodes: List[onnx.NodeProto], inputs: List[str], prefix: str) -> str:
    if not inputs:
        raise ValueError("empty Or chain")
    acc = inputs[0]
    for idx, item in enumerate(inputs[1:], start=1):
        out = f"{prefix}_{idx}"
        nodes.append(helper.make_node("Or", [acc, item], [out]))
        acc = out
    return acc


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    half = _f32(inits, [0.5], "half")
    zero = _f32(inits, [0.0], "zero")
    one_half = _f32(inits, [1.5], "one_half")
    big = _i32(inits, [99], "big")
    zero_i = _i32(inits, [0], "zero_i")
    rows = _i32(inits, np.arange(H, dtype=np.int32).reshape(1, 1, H, 1), "rows")
    cols = _i32(inits, np.arange(W, dtype=np.int32).reshape(1, 1, 1, W), "cols")

    hz_masks: List[str] = []
    vt_masks: List[str] = []
    for color in range(1, C):
        _i64(inits, [color], f"ci{color}")
        plane = f"p{color}"
        nodes.append(helper.make_node("Gather", [IN_NAME, f"ci{color}"], [plane], axis=1))
        hz, vt = _line_masks(nodes, plane, f"c{color}", rows, cols, big, zero_i, zero, one_half)
        hz_masks.append(hz)
        vt_masks.append(vt)

    any_vt = _or_chain(nodes, vt_masks, "any_vt")
    nodes.append(helper.make_node("Not", [any_vt], ["no_vt"]))

    color_channels: List[str] = []
    for color, (hz, vt) in enumerate(zip(hz_masks, vt_masks), start=1):
        hz_visible = f"c{color}_hzvis"
        ch = f"out_c{color}"
        nodes.extend(
            [
                helper.make_node("And", [hz, "no_vt"], [hz_visible]),
                helper.make_node("Or", [vt, hz_visible], [ch]),
            ]
        )
        color_channels.append(ch)
    any_color = _or_chain(nodes, color_channels, "any_color")
    nodes.append(helper.make_node("Not", [any_color], ["no_color"]))
    nodes.extend(
        [
            helper.make_node("ReduceSum", [IN_NAME], ["in_sm"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["in_sm", half], ["valid"]),
            helper.make_node("And", ["valid", "no_color"], ["out_c0"]),
            helper.make_node("Concat", ["out_c0", *color_channels], ["out_bool"], axis=1),
            helper.make_node("Cast", ["out_bool"], [OUT_NAME], to=TensorProto.FLOAT),
        ]
    )

    graph = helper.make_graph(nodes, "task092", [x_info], [y_info], initializer=inits)
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
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def _grid_from_pred(pred: np.ndarray) -> np.ndarray:
    return (pred[0].argmax(axis=0)).astype(np.int64)


def validate_model(model: onnx.ModelProto) -> tuple[bool, str]:
    if not DATA_PATH.is_file():
        return True, "no data"
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            arr = convert_to_numpy(ex, "input")
            if arr is None:
                continue
            g = np.asarray(ex["input"], dtype=np.int64)
            exp = solve(g)
            pred = _run_onnx(model, arr)
            got = _grid_from_pred(pred)[: g.shape[0], : g.shape[1]]
            if not np.array_equal(got, exp):
                return False, f"{split}#{idx}"
            exp_oh = convert_to_numpy(ex, "output")
            if exp_oh is not None and not np.array_equal(pred > 0, exp_oh > 0):
                return False, f"{split}#{idx} one-hot"
    return True, "PASS"


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    ok, msg = validate_model(model)
    scored = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"validation: {msg} ({ok})")
    print(f"nodes:    {len(model.graph.node)}")
    print(f"memory:   {scored['memory']}")
    print(f"params:   {scored['params']}")
    print(f"cost:     {scored['cost']}")
    if scored["score"] is not None:
        print(f"score:    {scored['score']:.6f}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
