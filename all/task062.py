"""Minimal ONNX for ARC task062: mirror colored object across red marker side.

Task rule: 10x10 input with black (0) background. A small nonzero pattern (one
color) includes red (2) direction markers on one side. Reflect the non-red
object across the object bbox edge facing the marker (right/left/up/down), keep
original colors, remove red, fill with green (3).

ONNX: ArgMax 10x10 ids, object/red masks, bbox extrema, cardinal direction from
red bbox vs object bbox, inverse-reflect gather for mirrored colors, green
background, OneHot encode, then pad to 30x30 output.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task062"
BEST_PATH = OUT_DIR / "task062.onnx"
SOLUTION_PATH = OUT_DIR / "solution.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
GH = GW = 10
H = W = 30
PAD = H - GH
BG = 3
RED = 2
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def solve(grid: np.ndarray) -> np.ndarray:
    """Mirror object across bbox edge on the red-marker side; background green."""
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    out = np.full((h, w), BG, dtype=np.int64)
    red = g == RED
    obj = (g != 0) & (g != RED)
    if not obj.any():
        return out
    or_, oc = np.where(obj)
    rmin, rmax = int(or_.min()), int(or_.max())
    cmin, cmax = int(oc.min()), int(oc.max())
    octr_r, octr_c = or_.mean(), oc.mean()
    if red.any():
        rr, rc = np.where(red)
        dr = rr.mean() - octr_r
        dc = rc.mean() - octr_c
    else:
        dr = dc = 0.0
    horiz = abs(dc) >= abs(dr)
    for r, c in zip(or_, oc):
        out[r, c] = g[r, c]
    for r, c in zip(or_, oc):
        color = g[r, c]
        nr, nc = int(r), int(c)
        if horiz:
            if dc > 0:
                nc = 2 * cmax - c + 1
            else:
                nc = 2 * cmin - c - 1
        else:
            if dr > 0:
                nr = 2 * rmax - r + 1
            else:
                nr = 2 * rmin - r - 1
        if 0 <= nr < h and 0 <= nc < w:
            out[nr, nc] = color
    return out


def _grid_to_onehot(grid: Sequence[Sequence[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            color = int(val)
            if 0 <= color < C:
                out[0, color, r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _reduce_spatial(op: str, data: str, out: str, *, inits: List[onnx.TensorProto], nodes: List[onnx.NodeProto], opset: int, tag: str) -> None:
    nodes.append(helper.make_node(op, [data], [out], axes=[2, 3], keepdims=1))


def _reshape_scalar(val: str, out: str, *, inits: List[onnx.TensorProto], nodes: List[onnx.NodeProto]) -> None:
    sh = _i64(inits, [1, 1, 1, 1], f"{out}_sh")
    nodes.append(helper.make_node("Reshape", [val, sh], [out]))


def build_model(*, opset: int = 10) -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    st = _i64(inits, [0, 0, 0, 0], "st")
    en = _i64(inits, [1, C, GH, GW], "en")
    pads = _i64(inits, [0, 0, 0, 0, 0, 0, PAD, PAD], "pads")
    pad_attr = [0, 0, 0, 0, 0, 0, PAD, PAD]
    zf = _f32(inits, [0.0], "zf")
    one = _f32(inits, [1.0], "one")
    big = _f32(inits, [99.0], "big")
    sml = _f32(inits, [-1.0], "sml")
    half = _f32(inits, [0.5], "half")
    nine = _f32(inits, [9.0], "nine")
    ten = _f32(inits, [10.0], "ten")
    redf = _f32(inits, [float(RED)], "redf")
    bgf = _f32(inits, [float(BG)], "bgf")
    rows = np.arange(GH, dtype=np.float32).reshape(1, 1, GH, 1)
    cols = np.arange(GW, dtype=np.float32).reshape(1, 1, 1, GW)
    _f32(inits, rows, "rows")
    _f32(inits, cols, "cols")
    _f32(inits, np.ones((1, 1, GH, 1), dtype=np.float32), "row1")
    _f32(inits, np.ones((1, 1, 1, GW), dtype=np.float32), "col1")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st, en, axes4], ["core"]),
            helper.make_node("ArgMax", ["core"], ["ids"], axis=1, keepdims=0),
        ]
    )
    if opset >= 13:
        nodes.append(helper.make_node("Unsqueeze", ["ids", _i64(inits, [1], "sq1")], ["ids4"]))
    else:
        nodes.append(helper.make_node("Unsqueeze", ["ids"], ["ids4"], axes=[1]))
    zc = _i64(inits, [0], "zc")
    redc = _i64(inits, [RED], "redc")

    nodes.extend(
        [
            helper.make_node("Equal", ["ids4", zc], ["is0"]),
            helper.make_node("Equal", ["ids4", redc], ["is2"]),
            helper.make_node("Not", ["is0"], ["nz"]),
            helper.make_node("Not", ["is2"], ["not2"]),
            helper.make_node("And", ["nz", "not2"], ["obj"]),
            # object bbox
            helper.make_node(
                "Where",
                ["obj", "rows", big],
                ["r_hi"],
            ),
            helper.make_node(
                "Where",
                ["obj", "rows", sml],
                ["r_lo"],
            ),
            helper.make_node(
                "Where",
                ["obj", "cols", big],
                ["c_hi"],
            ),
            helper.make_node(
                "Where",
                ["obj", "cols", sml],
                ["c_lo"],
            ),
            helper.make_node(
                "Where",
                ["is2", "rows", big],
                ["rr_hi"],
            ),
            helper.make_node(
                "Where",
                ["is2", "rows", sml],
                ["rr_lo"],
            ),
            helper.make_node(
                "Where",
                ["is2", "cols", big],
                ["rc_hi"],
            ),
            helper.make_node(
                "Where",
                ["is2", "cols", sml],
                ["rc_lo"],
            ),
        ]
    )

    _reduce_spatial("ReduceMin", "r_hi", "rmin", inits=inits, nodes=nodes, opset=opset, tag="rhi")
    _reduce_spatial("ReduceMax", "r_lo", "rmax", inits=inits, nodes=nodes, opset=opset, tag="rlo")
    _reduce_spatial("ReduceMin", "c_hi", "cmin", inits=inits, nodes=nodes, opset=opset, tag="chi")
    _reduce_spatial("ReduceMax", "c_lo", "cmax", inits=inits, nodes=nodes, opset=opset, tag="clo")
    _reduce_spatial("ReduceMin", "rr_hi", "rrmin", inits=inits, nodes=nodes, opset=opset, tag="rrhi")
    _reduce_spatial("ReduceMax", "rr_lo", "rrmax", inits=inits, nodes=nodes, opset=opset, tag="rrlo")
    _reduce_spatial("ReduceMin", "rc_hi", "rcmin", inits=inits, nodes=nodes, opset=opset, tag="rchi")
    _reduce_spatial("ReduceMax", "rc_lo", "rcmax", inits=inits, nodes=nodes, opset=opset, tag="rclo")

    for name in ("rmin", "rmax", "cmin", "cmax", "rrmin", "rrmax", "rcmin", "rcmax"):
        _reshape_scalar(name, f"{name}4", inits=inits, nodes=nodes)

    nodes.append(helper.make_node("Cast", ["ids4"], ["ids4f"], to=TensorProto.FLOAT))

    nodes.extend(
        [
            helper.make_node("Greater", ["rcmin4", "cmax4"], ["right"]),
            helper.make_node("Less", ["rcmax4", "cmin4"], ["left"]),
            helper.make_node("Or", ["right", "left"], ["horiz"]),
            helper.make_node("Greater", ["rrmin4", "rmax4"], ["down"]),
            # inverse reflect source coordinates for each output cell
            helper.make_node("Add", ["cmax4", "cmax4"], ["c2max"]),
            helper.make_node("Add", ["cmin4", "cmin4"], ["c2min"]),
            helper.make_node("Add", ["rmax4", "rmax4"], ["r2max"]),
            helper.make_node("Add", ["rmin4", "rmin4"], ["r2min"]),
            helper.make_node("Add", ["c2max", one], ["c2maxp1"]),
            helper.make_node("Sub", ["c2min", one], ["c2minm1"]),
            helper.make_node("Add", ["r2max", one], ["r2maxp1"]),
            helper.make_node("Sub", ["r2min", one], ["r2minm1"]),
            helper.make_node("Sub", ["c2maxp1", "cols"], ["src_c_r"]),
            helper.make_node("Sub", ["c2minm1", "cols"], ["src_c_l"]),
            helper.make_node("Sub", ["r2maxp1", "rows"], ["src_r_d"]),
            helper.make_node("Sub", ["r2minm1", "rows"], ["src_r_u"]),
            helper.make_node("Where", ["right", "src_c_r", "src_c_l"], ["src_c_h"]),
            helper.make_node("Where", ["down", "src_r_d", "src_r_u"], ["src_r_v"]),
            helper.make_node("Mul", ["src_c_h", "row1"], ["src_c_full"]),
            helper.make_node("Mul", ["src_r_v", "col1"], ["src_r_full"]),
            helper.make_node("Where", ["horiz", "src_c_full", "cols"], ["src_c"]),
            helper.make_node("Where", ["horiz", "rows", "src_r_full"], ["src_r"]),
            helper.make_node("Greater", ["src_r", sml], ["vr0"]),
            helper.make_node("Less", ["src_r", ten], ["vr1"]),
            helper.make_node("Greater", ["src_c", sml], ["vc0"]),
            helper.make_node("Less", ["src_c", ten], ["vc1"]),
            helper.make_node("And", ["vr0", "vr1"], ["vra"]),
            helper.make_node("And", ["vc0", "vc1"], ["vca"]),
            helper.make_node("And", ["vra", "vca"], ["vin"]),
            helper.make_node("Clip", ["src_c", zf, nine], ["src_cc"]),
            helper.make_node("Clip", ["src_r", zf, nine], ["src_rc"]),
            helper.make_node("Cast", ["src_cc"], ["src_c_i"], to=TensorProto.INT64),
            helper.make_node("Cast", ["src_rc"], ["src_r_i"], to=TensorProto.INT64),
        ]
    )
    if opset >= 11:
        nodes.extend(
            [
                helper.make_node("GatherElements", ["ids4f", "src_c_i"], ["g1"], axis=3),
                helper.make_node("GatherElements", ["g1", "src_r_i"], ["refl"], axis=2),
            ]
        )
    else:
        nodes.extend(
            [
                helper.make_node("Gather", ["ids4f", "src_c_i"], ["g1"], axis=3),
                helper.make_node("Gather", ["g1", "src_r_i"], ["refl"], axis=2),
            ]
        )

    nodes.extend(
        [
            helper.make_node("Greater", ["refl", half], ["refl_nz"]),
            helper.make_node("Equal", ["refl", redf], ["refl_red"]),
            helper.make_node("Not", ["refl_red"], ["refl_not_red"]),
            helper.make_node("And", ["refl_nz", "refl_not_red"], ["refl_obj"]),
            helper.make_node("And", ["vin", "refl_obj"], ["refl_ok"]),
            helper.make_node("Where", ["obj", "ids4f", bgf], ["outv"]),
            helper.make_node("Where", ["refl_ok", "refl", "outv"], ["outv2"]),
            # one-hot encode scalar grid outv2
        ]
    )

    nodes.append(helper.make_node("Cast", ["outv2"], ["out_i"], to=TensorProto.INT64))
    nodes.append(helper.make_node("OneHot", ["out_i", _i64(inits, [C], "depth"), _f32(inits, [0.0, 1.0], "hot_vals")], ["oh5"], axis=1))
    nodes.append(helper.make_node("Squeeze", ["oh5"], ["oh10f"], axes=[2]))
    if opset >= 11:
        nodes.append(helper.make_node("Pad", ["oh10f", pads], [OUT_NAME]))
    else:
        nodes.append(helper.make_node("Pad", ["oh10f"], [OUT_NAME], pads=pad_attr))

    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    onnx.checker.check_model(model)
    return model

def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
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
            total += 1
            oh = _grid_to_onehot(ex["input"])
            pred = _onehot_to_grid(_run_onnx(model, oh))[:GH, :GW]
            if not np.array_equal(pred, ex["output"]):
                bad += 1
    return bad, total


@dataclass(frozen=True)
class Variant:
    name: str
    builder: Callable[[], onnx.ModelProto]


VARIANTS = [
    Variant("op11", lambda: build_model(opset=11)),
]


def main() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            g = np.array(ex["input"], dtype=np.int64)
            assert np.array_equal(solve(g), ex["output"]), f"reference fail {split}"

    best_name = None
    best_result = None
    best_model = None
    for variant in VARIANTS:
        print(f"building {variant.name}...")
        model = variant.builder()
        bad, total = validate_json(model)
        path = OUT_DIR / f"task062_{variant.name}.onnx"
        onnx.save(model, path)
        result = score_file(path)
        print(
            f"{variant.name}: nodes={len(model.graph.node)} json={'PASS' if bad == 0 else f'FAIL({bad}/{total})'} "
            f"mem={result.get('memory')} params={result.get('params')} "
            f"cost={result.get('cost')} score={result.get('score')}"
        )
        if bad != 0 or not result.get("valid"):
            continue
        cost = result.get("cost")
        if cost is None:
            continue
        if best_result is None or cost < best_result.get("cost", float("inf")):
            best_name = variant.name
            best_result = result
            best_model = model

    if best_model is None:
        raise RuntimeError("no valid ONNX variant")

    onnx.save(best_model, BEST_PATH)
    onnx.save(best_model, SOLUTION_PATH)
    print(f"BEST: {best_name}")
    print(f"wrote {BEST_PATH}")
    print(f"wrote {SOLUTION_PATH}")
    print(f"memory:  {best_result['memory']}")
    print(f"params:  {best_result['params']}")
    print(f"cost:    {best_result['cost']}")
    print(f"score:   {best_result['score']:.6f}")


if __name__ == "__main__":
    main()
