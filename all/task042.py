"""ONNX solver for ARC task042: complete the opposite diagonal in cyan.

Task rule: inputs are 10x10 grids with green (3) shapes on one diagonal. Each
4-connected green component is either a singleton, a 2x2 block, or a 3x3 block,
and a matching component of the same size appears one block-step away on a
diagonal. Preserve all green cells. For each matching diagonal pair, add cyan
(8) copies on the other diagonal: singleton pairs produce the other two corners
of the surrounding 4x4 diagonal pattern; 2x2 and 3x3 block pairs produce clipped
same-size cyan block copies at the corresponding opposite-diagonal positions.

ONNX: crop the 10x10 green/background planes, work in float16, classify green
cells into singleton/2x2/3x3 components with tiny convolutional neighbor-count
masks, use one-hot convolution kernels for shifts where they are cheaper than
Slice+Pad, then concatenate a compact 10x10 one-hot output and cast/pad once to
the NeuroGolf [1,10,30,30] float output.
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

TASK_ID = "task042"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
ACTIVE = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
IR_VERSION = 10
OPSET = 10


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference implementation using 4-connected square components."""
    g = np.asarray(grid, dtype=np.int64)
    green = g == 3
    h, w = green.shape
    seen = np.zeros_like(green, dtype=bool)
    comps: list[dict[str, object]] = []

    for r0, c0 in zip(*np.nonzero(green)):
        r0, c0 = int(r0), int(c0)
        if seen[r0, c0]:
            continue
        todo = [(r0, c0)]
        seen[r0, c0] = True
        cells: list[tuple[int, int]] = []
        for r, c in todo:
            cells.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and green[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    todo.append((nr, nc))
        arr = np.asarray(cells, dtype=np.int64)
        mins = arr.min(axis=0)
        maxs = arr.max(axis=0)
        comps.append(
            {
                "cells": cells,
                "r": int(mins[0]),
                "c": int(mins[1]),
                "n": int(maxs[0] - mins[0] + 1),
                "m": int(maxs[1] - mins[1] + 1),
            }
        )

    cyan = np.zeros_like(green)
    for a in comps:
        n = int(a["n"])
        if n != int(a["m"]):
            continue
        for b in comps:
            if a is b or int(b["n"]) != n or int(b["m"]) != n:
                continue
            dr = int(b["r"]) - int(a["r"])
            dc = int(b["c"]) - int(a["c"])
            if dr == n and dc == n:
                placements = ((a, -n, 2 * n), (b, n, -2 * n))
            elif dr == n and dc == -n:
                placements = ((a, 2 * n, n), (b, -2 * n, -n))
            else:
                continue
            for comp, sr, sc in placements:
                for r, c in comp["cells"]:  # type: ignore[index]
                    nr, nc = int(r) + sr, int(c) + sc
                    if 0 <= nr < h and 0 <= nc < w:
                        cyan[nr, nc] = True

    out = g.copy()
    out[(out == 0) & cyan] = 8
    return out


def _i64(inits: List[onnx.TensorProto], name: str, values: list[int]) -> str:
    inits.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name))
    return name


def _f32(inits: List[onnx.TensorProto], name: str, values: list[float]) -> str:
    inits.append(numpy_helper.from_array(np.asarray(values, dtype=np.float32), name=name))
    return name


def _f16_array(inits: List[onnx.TensorProto], name: str, values: np.ndarray | list[float]) -> str:
    inits.append(numpy_helper.from_array(np.asarray(values, dtype=np.float16), name=name))
    return name


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self.counter = 0
        self.axes = _i64(self.inits, "axes", [0, 1, 2, 3])
        self.zero = _f32(self.inits, "zero", [0.0])
        self.half = _f32(self.inits, "half", [0.5])
        self.one_half = _f32(self.inits, "one_half", [1.5])
        self.two_half = _f32(self.inits, "two_half", [2.5])

    def name(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}{self.counter}"

    def init(self, values: list[int], prefix: str) -> str:
        return _i64(self.inits, self.name(prefix), values)

    def node(self, op: str, inputs: list[str], prefix: str, **attrs: object) -> str:
        out = self.name(prefix)
        self.nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    def slice(self, x: str, starts: list[int], ends: list[int], prefix: str) -> str:
        return self.node(
            "Slice",
            [x, self.init(starts, "st"), self.init(ends, "en"), self.axes],
            prefix,
        )

    def shift(self, x: str, dr: int, dc: int, prefix: str) -> str:
        rs = max(0, -dr)
        re = ACTIVE - max(0, dr)
        cs = max(0, -dc)
        ce = ACTIVE - max(0, dc)
        body = self.slice(x, [0, 0, rs, cs], [1, 1, re, ce], f"{prefix}s")
        pads = [
            0,
            0,
            max(0, dr),
            max(0, dc),
            0,
            0,
            max(0, -dr),
            max(0, -dc),
        ]
        return self.node("Pad", [body], prefix, mode="constant", pads=pads)

    def shift_bool(self, x: str, dr: int, dc: int, prefix: str) -> str:
        xf = self.node("Cast", [x], f"{prefix}f", to=TensorProto.FLOAT)
        shifted = self.shift(xf, dr, dc, prefix)
        return self.node("Greater", [shifted, self.half], f"{prefix}b")

    def and_(self, a: str, b: str, prefix: str) -> str:
        return self.node("And", [a, b], prefix)

    def or_(self, a: str, b: str, prefix: str) -> str:
        return self.node("Or", [a, b], prefix)

    def not_(self, x: str, prefix: str) -> str:
        return self.node("Not", [x], prefix)


def _make_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], graph_name: str) -> onnx.ModelProto:
    used_inputs = {name for node in nodes for name in node.input if name}
    inits = [init for init in inits if init.name in used_inputs]
    graph = helper.make_graph(
        nodes,
        graph_name,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_hybrid_model() -> onnx.ModelProto:
    b = Builder()

    zero10 = _f16_array(b.inits, "zero10", np.zeros((1, 1, ACTIVE, ACTIVE), dtype=np.float16))
    half16 = _f16_array(b.inits, "half16", [0.5])
    one_half16 = _f16_array(b.inits, "one_half16", [1.5])
    two_half16 = _f16_array(b.inits, "two_half16", [2.5])

    k3 = np.zeros((1, 1, 3, 3), dtype=np.float16)
    k3[0, 0, 0, 1] = k3[0, 0, 1, 0] = k3[0, 0, 1, 2] = k3[0, 0, 2, 1] = 1
    k5 = np.zeros((1, 1, 5, 5), dtype=np.float16)
    k5[0, 0, 0, 2] = k5[0, 0, 2, 0] = k5[0, 0, 2, 4] = k5[0, 0, 4, 2] = 1
    neigh_kernel = _f16_array(b.inits, "neigh_kernel", k3)
    dist2_kernel = _f16_array(b.inits, "dist2_kernel", k5)

    conv_offsets: set[tuple[int, int]] = set()
    for n in (1, 2, 3):
        conv_offsets.update({(-n, -n), (n, n), (-n, n), (n, -n)})
    conv_offsets.update({(-1, 2), (1, -2), (2, 1), (-2, -1)})
    conv_offsets.update({(-2, 4), (2, -4), (4, 2), (-4, -2)})
    shift_kernels: dict[tuple[int, int], str] = {}

    def cast16(x: str, prefix: str) -> str:
        return b.node("Cast", [x], prefix, to=TensorProto.FLOAT16)

    def shift(x: str, dr: int, dc: int, prefix: str) -> str:
        if (dr, dc) not in conv_offsets:
            return b.shift(x, dr, dc, prefix)

        rh = abs(dr)
        rw = abs(dc)
        key = (dr, dc)
        if key not in shift_kernels:
            kernel = np.zeros((1, 1, 2 * rh + 1, 2 * rw + 1), dtype=np.float16)
            kernel[0, 0, rh - dr, rw - dc] = 1
            shift_kernels[key] = _f16_array(b.inits, f"shift_{dr}_{dc}", kernel)
        return b.node("Conv", [x, shift_kernels[key]], prefix, pads=[rh, rw, rh, rw])

    ch0 = cast16(b.slice(IN_NAME, [0, 0, 0, 0], [1, 1, ACTIVE, ACTIVE], "ch0f"), "ch0h")
    g = cast16(b.slice(IN_NAME, [0, 3, 0, 0], [1, 4, ACTIVE, ACTIVE], "gf"), "gh")

    neigh = b.node("Conv", [g, neigh_kernel], "n1", pads=[1, 1, 1, 1])
    iso = b.node(
        "Mul",
        [g, b.node("Cast", [b.node("Less", [neigh, half16], "no_nb")], "no_nbf", to=TensorProto.FLOAT16)],
        "iso",
    )

    dist2 = b.node("Conv", [g, dist2_kernel], "dst2", pads=[2, 2, 2, 2])
    count2 = b.node(
        "Cast",
        [
            b.and_(
                b.node("Greater", [neigh, one_half16], "gt15"),
                b.node("Less", [neigh, two_half16], "lt25"),
                "eq2",
            )
        ],
        "eq2f",
        to=TensorProto.FLOAT16,
    )
    no_dist2 = b.node("Cast", [b.node("Less", [dist2, half16], "nd2")], "nd2f", to=TensorProto.FLOAT16)
    m2 = b.node("Mul", [b.node("Mul", [g, count2], "g2a"), no_dist2], "m2")
    m3 = b.node("Sub", [b.node("Sub", [g, iso], "gni"), m2], "m3")

    cyan = zero10

    def add_pairs(mask: str, n: int) -> None:
        nonlocal cyan
        tl = b.node("Mul", [mask, shift(mask, -n, -n, f"tl{n}a")], f"tl{n}")
        br = b.node("Mul", [mask, shift(mask, n, n, f"br{n}a")], f"br{n}")
        tr = b.node("Mul", [mask, shift(mask, -n, n, f"tr{n}a")], f"tr{n}")
        bl = b.node("Mul", [mask, shift(mask, n, -n, f"bl{n}a")], f"bl{n}")
        cyan = b.node("Add", [cyan, shift(tl, -n, 2 * n, f"ctl{n}")], f"ca{n}")
        cyan = b.node("Add", [cyan, shift(br, n, -2 * n, f"cbr{n}")], f"cb{n}")
        cyan = b.node("Add", [cyan, shift(tr, 2 * n, n, f"ctr{n}")], f"cc{n}")
        cyan = b.node("Add", [cyan, shift(bl, -2 * n, -n, f"cbl{n}")], f"cd{n}")

    add_pairs(iso, 1)
    add_pairs(m2, 2)
    add_pairs(m3, 3)

    clean_b = b.and_(b.node("Greater", [cyan, half16], "cyanb"), b.node("Less", [g, half16], "ng"), "cyb")
    clean = b.node("Cast", [clean_b], "cy", to=TensorProto.FLOAT16)
    ch0_out = b.node("Where", [clean_b, zero10, ch0], "ch0o")
    out10h = b.node(
        "Concat",
        [ch0_out, zero10, zero10, g, zero10, zero10, zero10, zero10, clean, zero10],
        "out10h",
        axis=1,
    )
    out10 = b.node("Cast", [out10h], "out10", to=TensorProto.FLOAT)
    b.nodes.append(
        helper.make_node(
            "Pad",
            [out10],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - ACTIVE, W - ACTIVE],
        )
    )
    return _make_model(b.nodes, b.inits, f"{TASK_ID}_hybrid")


def build_bool_model() -> onnx.ModelProto:
    b = Builder()

    ch0 = b.slice(IN_NAME, [0, 0, 0, 0], [1, 1, ACTIVE, ACTIVE], "ch0f")
    ch3 = b.slice(IN_NAME, [0, 3, 0, 0], [1, 4, ACTIVE, ACTIVE], "ch3f")
    bg = b.node("Greater", [ch0, b.half], "bg")
    green = b.node("Greater", [ch3, b.half], "green")

    up = b.shift(ch3, -1, 0, "up")
    down = b.shift(ch3, 1, 0, "dn")
    left = b.shift(ch3, 0, -1, "lf")
    right = b.shift(ch3, 0, 1, "rt")
    ud = b.node("Add", [up, down], "ud")
    lr = b.node("Add", [left, right], "lr")
    neigh = b.node("Add", [ud, lr], "n1")
    has_neigh = b.node("Greater", [neigh, b.half], "hn")
    iso = b.and_(green, b.not_(has_neigh, "nh"), "iso")

    up2 = b.shift(ch3, -2, 0, "u2")
    down2 = b.shift(ch3, 2, 0, "d2")
    left2 = b.shift(ch3, 0, -2, "l2")
    right2 = b.shift(ch3, 0, 2, "r2")
    ud2 = b.node("Add", [up2, down2], "u2d")
    lr2 = b.node("Add", [left2, right2], "l2r")
    dist2 = b.node("Add", [ud2, lr2], "dst2")
    no_dist2 = b.node("Less", [dist2, b.half], "nd2")
    count2 = b.and_(
        b.node("Greater", [neigh, b.one_half], "gt15"),
        b.node("Less", [neigh, b.two_half], "lt25"),
        "eq2",
    )
    m2 = b.and_(b.and_(green, count2, "g2a"), no_dist2, "m2")
    m3 = b.and_(b.and_(green, b.not_(iso, "niso"), "m3a"), b.not_(m2, "nm2"), "m3")

    zero_bool = b.and_(green, b.not_(green, "ng0"), "zb")
    cyan = zero_bool

    def add_pairs(mask: str, n: int) -> None:
        nonlocal cyan
        tl = b.and_(mask, b.shift_bool(mask, -n, -n, f"tl{n}a"), f"tl{n}")
        br = b.and_(mask, b.shift_bool(mask, n, n, f"br{n}a"), f"br{n}")
        tr = b.and_(mask, b.shift_bool(mask, -n, n, f"tr{n}a"), f"tr{n}")
        bl = b.and_(mask, b.shift_bool(mask, n, -n, f"bl{n}a"), f"bl{n}")
        cyan = b.or_(cyan, b.shift_bool(tl, -n, 2 * n, f"ctl{n}"), f"co{n}a")
        cyan = b.or_(cyan, b.shift_bool(br, n, -2 * n, f"cbr{n}"), f"co{n}b")
        cyan = b.or_(cyan, b.shift_bool(tr, 2 * n, n, f"ctr{n}"), f"co{n}c")
        cyan = b.or_(cyan, b.shift_bool(bl, -2 * n, -n, f"cbl{n}"), f"co{n}d")

    add_pairs(iso, 1)
    add_pairs(m2, 2)
    add_pairs(m3, 3)

    cyan = b.and_(cyan, b.not_(green, "cng"), "cyan")
    ch0_out = b.and_(bg, b.not_(cyan, "ncy"), "ch0")
    out10b = b.node(
        "Concat",
        [ch0_out, zero_bool, zero_bool, green, zero_bool, zero_bool, zero_bool, zero_bool, cyan, zero_bool],
        "out10b",
        axis=1,
    )
    out10 = b.node("Cast", [out10b], "out10", to=TensorProto.FLOAT)
    b.nodes.append(
        helper.make_node(
            "Pad",
            [out10],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - ACTIVE, W - ACTIVE],
        )
    )

    return _make_model(b.nodes, b.inits, f"{TASK_ID}_bool")


def build_float_model() -> onnx.ModelProto:
    b = Builder()

    ch0 = b.slice(IN_NAME, [0, 0, 0, 0], [1, 1, ACTIVE, ACTIVE], "ch0")
    g = b.slice(IN_NAME, [0, 3, 0, 0], [1, 4, ACTIVE, ACTIVE], "g")
    z = b.node("Mul", [g, b.zero], "z")

    up = b.shift(g, -1, 0, "up")
    down = b.shift(g, 1, 0, "dn")
    left = b.shift(g, 0, -1, "lf")
    right = b.shift(g, 0, 1, "rt")
    neigh = b.node("Add", [b.node("Add", [up, down], "ud"), b.node("Add", [left, right], "lr")], "n1")
    iso = b.node("Mul", [g, b.node("Cast", [b.node("Less", [neigh, b.half], "no_nb")], "no_nbf", to=TensorProto.FLOAT)], "iso")

    up2 = b.shift(g, -2, 0, "u2")
    down2 = b.shift(g, 2, 0, "d2")
    left2 = b.shift(g, 0, -2, "l2")
    right2 = b.shift(g, 0, 2, "r2")
    dist2 = b.node("Add", [b.node("Add", [up2, down2], "u2d"), b.node("Add", [left2, right2], "l2r")], "dst2")
    count2b = b.and_(
        b.node("Greater", [neigh, b.one_half], "gt15"),
        b.node("Less", [neigh, b.two_half], "lt25"),
        "eq2",
    )
    count2 = b.node("Cast", [count2b], "eq2f", to=TensorProto.FLOAT)
    no_dist2 = b.node("Cast", [b.node("Less", [dist2, b.half], "nd2")], "nd2f", to=TensorProto.FLOAT)
    m2 = b.node("Mul", [b.node("Mul", [g, count2], "g2a"), no_dist2], "m2")
    m3 = b.node("Sub", [b.node("Sub", [g, iso], "gni"), m2], "m3")

    cyan = z

    def add_pairs(mask: str, n: int) -> None:
        nonlocal cyan
        tl = b.node("Mul", [mask, b.shift(mask, -n, -n, f"tl{n}a")], f"tl{n}")
        br = b.node("Mul", [mask, b.shift(mask, n, n, f"br{n}a")], f"br{n}")
        tr = b.node("Mul", [mask, b.shift(mask, -n, n, f"tr{n}a")], f"tr{n}")
        bl = b.node("Mul", [mask, b.shift(mask, n, -n, f"bl{n}a")], f"bl{n}")
        cyan = b.node("Add", [cyan, b.shift(tl, -n, 2 * n, f"ctl{n}")], f"ca{n}")
        cyan = b.node("Add", [cyan, b.shift(br, n, -2 * n, f"cbr{n}")], f"cb{n}")
        cyan = b.node("Add", [cyan, b.shift(tr, 2 * n, n, f"ctr{n}")], f"cc{n}")
        cyan = b.node("Add", [cyan, b.shift(bl, -2 * n, -n, f"cbl{n}")], f"cd{n}")

    add_pairs(iso, 1)
    add_pairs(m2, 2)
    add_pairs(m3, 3)

    cyanb = b.node("Greater", [cyan, b.half], "cyanb")
    not_green = b.node("Less", [g, b.half], "ng")
    clean_cyanb = b.and_(cyanb, not_green, "cyb")
    clean_cyan = b.node("Cast", [clean_cyanb], "cy", to=TensorProto.FLOAT)
    ch0_out = b.node("Where", [clean_cyanb, z, ch0], "ch0o")
    out10 = b.node(
        "Concat",
        [ch0_out, z, z, g, z, z, z, z, clean_cyan, z],
        "out10",
        axis=1,
    )
    b.nodes.append(
        helper.make_node(
            "Pad",
            [out10],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - ACTIVE, W - ACTIVE],
        )
    )
    return _make_model(b.nodes, b.inits, f"{TASK_ID}_float")


def build_model() -> onnx.ModelProto:
    return build_hybrid_model()


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def validate(model: onnx.ModelProto) -> int:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, []), start=1):
            expected = np.asarray(ex["output"], dtype=np.int64)
            reference = solve(np.asarray(ex["input"], dtype=np.int64))
            assert np.array_equal(reference, expected), f"reference mismatch {split} #{idx}"
            x = convert_to_numpy(ex, "input")
            y = sess.run([OUT_NAME], {IN_NAME: x})[0]
            pred = _onehot_to_grid(y)[: expected.shape[0], : expected.shape[1]]
            active = (y[0, :, : expected.shape[0], : expected.shape[1]] > 0).sum(axis=0)
            if not np.array_equal(pred, expected) or not np.all(active == 1):
                bad += 1
    return bad


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    bad = validate(model)
    print(f"{TASK_ID}.json: {'PASS' if bad == 0 else f'FAIL ({bad} wrong)'}")

    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {str(result['error']).strip()}")
    print(f"nodes:   {len(model.graph.node)}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
