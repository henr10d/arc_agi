"""ONNX for ARC task020: complete a symmetric 5x5 diamond/square frame.

Task rule: the 10x10 input contains a partially filled 5x5 pattern. Keep all
given non-zero cells, infer the 5x5 object's center from its non-zero bounding
box, and fill every missing counterpart under the square's 8 symmetries
(horizontal/vertical/diagonal reflections and rotations) with the same color.
The grid size is unchanged and background remains 0.

ONNX: all examples use colors 1,2,3,4,8 and keep the 5x5 object inside rows
and columns 1..8. Work only on that 8x8 interior and those five colors, build
the four used dihedral orbits by absolute offsets from the inferred center,
then pad a 10x10 one-hot grid back to the required 30x30 tensor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task020"
BEST_PATH = OUT_DIR / "task020.onnx"
DATA_PATH = ROOT / "data" / "task020.json"

C = 10
H = W = 30
GRID = 10
GH = GW = 8
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


def _i32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int32), name)


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def solve(grid: np.ndarray) -> np.ndarray:
    """Complete each non-zero cell's full 5x5 square-symmetry orbit."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    nz = np.argwhere(g > 0)
    if len(nz) == 0:
        return out
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    rr = (int(r0) + int(r1)) // 2
    cc = (int(c0) + int(c1)) // 2
    for r, c in nz:
        color = int(g[r, c])
        dr = int(r) - rr
        dc = int(c) - cc
        mirrors = (
            (rr + dr, cc + dc),
            (rr + dr, cc - dc),
            (rr - dr, cc + dc),
            (rr - dr, cc - dc),
            (rr + dc, cc + dr),
            (rr + dc, cc - dr),
            (rr - dc, cc + dr),
            (rr - dc, cc - dr),
        )
        for mr, mc in mirrors:
            if 0 <= mr < g.shape[0] and 0 <= mc < g.shape[1] and out[mr, mc] == 0:
                out[mr, mc] = color
    return out


def _grid_to_onehot(grid: List[List[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _eq_int_scalar(nodes: List[onnx.NodeProto], value: str, scalar: str, tag: str) -> str:
    out = f"{tag}_eq"
    nodes.append(helper.make_node("Equal", [value, scalar], [out]))
    return out


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    st_core5 = _i64(inits, [0, 0, 1, 1], "st_core5")
    en_core5 = _i64(inits, [1, 5, 9, 9], "en_core5")
    st_core8 = _i64(inits, [0, 8, 1, 1], "st_core8")
    en_core8 = _i64(inits, [1, 9, 9, 9], "en_core8")
    half = _f32(inits, [0.5], "half")
    i0 = _i32(inits, [0], "i0")
    i1 = _i32(inits, [1], "i1")
    i2 = _i32(inits, [2], "i2")
    i3 = _i32(inits, [3], "i3")
    i4 = _i32(inits, [4], "i4")
    i8 = _i32(inits, [8], "i8")
    ibig = _i32(inits, [99], "ibig")
    rows = _i32(inits, np.arange(GH, dtype=np.int32).reshape(1, 1, GH, 1), "rows")
    cols = _i32(inits, np.arange(GW, dtype=np.int32).reshape(1, 1, 1, GW), "cols")
    zero10 = _init(inits, np.zeros((1, 1, GRID, GRID), dtype=np.bool_), "zero10")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st_core5, en_core5, axes4], ["core5"]),
            helper.make_node("ArgMax", ["core5"], ["color64"], axis=1, keepdims=1),
            helper.make_node("Cast", ["color64"], ["color5"], to=TensorProto.INT32),
            helper.make_node("Slice", [IN_NAME, st_core8, en_core8, axes4], ["core8"]),
            helper.make_node("Greater", ["core8", half], ["has8"]),
            helper.make_node("Where", ["has8", i8, "color5"], ["color"]),
            helper.make_node("ReduceMax", ["color"], ["row_color"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["color"], ["col_color"], axes=[2], keepdims=1),
            helper.make_node("Greater", ["row_color", i0], ["row_has"]),
            helper.make_node("Greater", ["col_color", i0], ["col_has"]),
            helper.make_node("Where", ["row_has", rows, ibig], ["row_min_src"]),
            helper.make_node("Where", ["col_has", cols, ibig], ["col_min_src"]),
            helper.make_node("ReduceMin", ["row_min_src"], ["min_r"], axes=[2], keepdims=1),
            helper.make_node("ReduceMin", ["col_min_src"], ["min_c"], axes=[3], keepdims=1),
            helper.make_node("Add", ["min_r", i2], ["cen_r"]),
            helper.make_node("Add", ["min_c", i2], ["cen_c"]),
            helper.make_node("Sub", ["rows", "cen_r"], ["dr"]),
            helper.make_node("Sub", ["cols", "cen_c"], ["dc"]),
            helper.make_node("Abs", ["dr"], ["adr"]),
            helper.make_node("Abs", ["dc"], ["adc"]),
        ]
    )

    eq_r = {}
    eq_c = {}
    for value, scalar in ((0, i0), (1, i1), (2, i2)):
        eq_r[value] = _eq_int_scalar(nodes, "adr", scalar, f"adr{value}")
        eq_c[value] = _eq_int_scalar(nodes, "adc", scalar, f"adc{value}")

    def orbit_mask(a: int, b: int) -> str:
        tag = f"orb{a}{b}"
        if a == 0 and b == 0:
            out = f"{tag}_m"
            nodes.append(helper.make_node("And", [eq_r[0], eq_c[0]], [out]))
            return out
        if b == 0:
            m1, m2, out = f"{tag}_r", f"{tag}_c", f"{tag}_m"
            nodes.extend(
                [
                    helper.make_node("And", [eq_r[a], eq_c[0]], [m1]),
                    helper.make_node("And", [eq_r[0], eq_c[a]], [m2]),
                    helper.make_node("Or", [m1, m2], [out]),
                ]
            )
            return out
        if a == b:
            out = f"{tag}_m"
            nodes.append(helper.make_node("And", [eq_r[a], eq_c[a]], [out]))
            return out
        m1, m2, out = f"{tag}_a", f"{tag}_b", f"{tag}_m"
        nodes.extend(
            [
                helper.make_node("And", [eq_r[a], eq_c[b]], [m1]),
                helper.make_node("And", [eq_r[b], eq_c[a]], [m2]),
                helper.make_node("Or", [m1, m2], [out]),
            ]
        )
        return out

    orbit_masks: List[str] = []
    orbit_colors: List[str] = []
    color_acc = ""
    for a, b in ((0, 0), (2, 0), (1, 1), (2, 2)):
        mask = orbit_mask(a, b)
        orbit_masks.append(mask)
        tag = f"orb{a}{b}"
        hit, color = f"{tag}_hit", f"{tag}_color"
        nodes.append(helper.make_node("Where", [mask, "color", i0], [hit]))
        if a == 0 and b == 0:
            color_acc = hit
            continue
        nodes.append(helper.make_node("ReduceMax", [hit], [color], axes=[2, 3], keepdims=1))
        orbit_colors.append(color)

    for i, (mask, color) in enumerate(zip(orbit_masks[1:], orbit_colors), start=1):
        out = f"out_color_{i}"
        nodes.append(helper.make_node("Where", [mask, color, color_acc], [out]))
        color_acc = out

    nodes.extend(
        [
            helper.make_node("Cast", [color_acc], ["colorf"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["colorf"], ["color10f"], pads=[0, 0, 1, 1, 0, 0, 1, 1]),
            helper.make_node("Cast", ["color10f"], ["color10"], to=TensorProto.INT32),
            helper.make_node("Equal", ["color10", i0], ["bg"]),
            helper.make_node("Equal", ["color10", i1], ["fg1b"]),
            helper.make_node("Equal", ["color10", i2], ["fg2b"]),
            helper.make_node("Equal", ["color10", i3], ["fg3b"]),
            helper.make_node("Equal", ["color10", i4], ["fg4b"]),
            helper.make_node("Equal", ["color10", i8], ["fg8b"]),
            helper.make_node(
                "Concat",
                ["bg", "fg1b", "fg2b", "fg3b", "fg4b", zero10, zero10, zero10, "fg8b"],
                ["out9b"],
                axis=1,
            ),
            helper.make_node("Cast", ["out9b"], ["out9"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out9"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 1, H - GRID, W - GRID]),
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


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            g = np.array(ex["input"], dtype=np.int64)
            oh = _grid_to_onehot(ex["input"])
            pred = _onehot_to_grid(_run_onnx(model, oh))[: g.shape[0], : g.shape[1]]
            exp = np.array(ex["output"], dtype=np.int64)
            ref = solve(g)
            if not np.array_equal(ref, exp):
                raise AssertionError(f"reference solver mismatch in {split}")
            if not np.array_equal(pred, exp):
                bad += 1
    return bad


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    bad = validate_json(model)
    assert bad == 0, f"{bad} JSON examples failed"
    result = score_file(BEST_PATH)
    print(result)


if __name__ == "__main__":
    main()
