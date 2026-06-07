"""ONNX solution for ARC task225: diagonal corner stamping from a 2x2 block.

Task rule: the 6x6 input contains exactly one non-black 2x2 block.  If the
block is

    A B
    C D

then preserve it in place and stamp monochrome 2x2 blocks diagonally outward:
D up-left, C up-right, B down-left, and A down-right.  Stamps are clipped to
the 6x6 output boundary; all other output cells are black.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task225"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task225.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
G = 6
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Iterable[int], name: str) -> str:
    return _init(inits, np.asarray(list(vals), dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals: Iterable[float], name: str) -> str:
    return _init(inits, np.asarray(list(vals), dtype=np.float32), name)


def _shift(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    data: str,
    dy: int,
    dx: int,
    channels: int,
    name: str,
) -> str:
    current = data
    if dy:
        k = abs(dy)
        start = _i64(inits, [0 if dy > 0 else k], f"{name}_ys")
        end = _i64(inits, [G - k if dy > 0 else G], f"{name}_ye")
        axis = _i64(inits, [2], f"{name}_ya")
        sliced = f"{name}_yslice"
        zero = _init(inits, np.zeros((1, channels, k, G), dtype=bool), f"{name}_yz")
        current_y = f"{name}_y"
        nodes.append(helper.make_node("Slice", [current, start, end, axis], [sliced]))
        inputs = [zero, sliced] if dy > 0 else [sliced, zero]
        nodes.append(helper.make_node("Concat", inputs, [current_y], axis=2))
        current = current_y
    if dx:
        k = abs(dx)
        start = _i64(inits, [0 if dx > 0 else k], f"{name}_xs")
        end = _i64(inits, [G - k if dx > 0 else G], f"{name}_xe")
        axis = _i64(inits, [3], f"{name}_xa")
        sliced = f"{name}_xslice"
        zero = _init(inits, np.zeros((1, channels, G, k), dtype=bool), f"{name}_xz")
        current_x = f"{name}_x"
        nodes.append(helper.make_node("Slice", [current, start, end, axis], [sliced]))
        inputs = [zero, sliced] if dx > 0 else [sliced, zero]
        nodes.append(helper.make_node("Concat", inputs, [current_x], axis=3))
        current = current_x
    return current


def _or_all(nodes: List[onnx.NodeProto], terms: list[str], prefix: str) -> str:
    current = terms[0]
    for idx, term in enumerate(terms[1:], start=1):
        out = f"{prefix}{idx}"
        nodes.append(helper.make_node("Or", [current, term], [out]))
        current = out
    return current


def solve_reference(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    out = np.zeros((G, G), dtype=np.int64)
    for r in range(G - 1):
        for c in range(G - 1):
            block = g[r : r + 2, c : c + 2]
            if not np.all(block != 0):
                continue
            a, b = int(block[0, 0]), int(block[0, 1])
            cc, d = int(block[1, 0]), int(block[1, 1])
            out[r : r + 2, c : c + 2] = block
            for rr in range(r - 2, r):
                for col in range(c - 2, c):
                    if 0 <= rr < G and 0 <= col < G:
                        out[rr, col] = d
            for rr in range(r - 2, r):
                for col in range(c + 2, c + 4):
                    if 0 <= rr < G and 0 <= col < G:
                        out[rr, col] = cc
            for rr in range(r + 2, r + 4):
                for col in range(c - 2, c):
                    if 0 <= rr < G and 0 <= col < G:
                        out[rr, col] = b
            for rr in range(r + 2, r + 4):
                for col in range(c + 2, c + 4):
                    if 0 <= rr < G and 0 <= col < G:
                        out[rr, col] = a
            return out
    raise ValueError("no 2x2 non-black block found")


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    s6_all = _i64(inits, [0, 0, 0, 0], "s6_all")
    e6_all = _i64(inits, [1, C, G, G], "e6_all")
    s6_fg = _i64(inits, [0, 1, 0, 0], "s6_fg")
    e6_fg = _i64(inits, [1, C, G, G], "e6_fg")
    s6_bg = _i64(inits, [0, 0, 0, 0], "s6_bg")
    e6_bg = _i64(inits, [1, 1, G, G], "e6_bg")
    zero_f = _f32(inits, [0.0], "zero_f")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, s6_all, e6_all, axes4], ["x6"]),
            helper.make_node("Slice", ["x6", s6_fg, e6_fg, axes4], ["xfg"]),
            helper.make_node("Slice", ["x6", s6_bg, e6_bg, axes4], ["xbg"]),
            helper.make_node("Greater", ["xfg", zero_f], ["fg"]),
            helper.make_node("Greater", ["xbg", zero_f], ["bg_in"]),
            helper.make_node("Not", ["bg_in"], ["nonblack"]),
        ]
    )

    right = _shift(nodes, inits, "nonblack", 0, -1, 1, "nb_right")
    left = _shift(nodes, inits, "nonblack", 0, 1, 1, "nb_left")
    down = _shift(nodes, inits, "nonblack", -1, 0, 1, "nb_down")
    up = _shift(nodes, inits, "nonblack", 1, 0, 1, "nb_up")

    nodes.extend(
        [
            helper.make_node("And", ["fg", right], ["a_r"]),
            helper.make_node("And", ["a_r", down], ["a"]),
            helper.make_node("And", ["fg", left], ["b_l"]),
            helper.make_node("And", ["b_l", down], ["b"]),
            helper.make_node("And", ["fg", right], ["c_r"]),
            helper.make_node("And", ["c_r", up], ["c"]),
            helper.make_node("And", ["fg", left], ["d_l"]),
            helper.make_node("And", ["d_l", up], ["d"]),
        ]
    )

    terms = ["fg"]
    for dy in (2, 3):
        for dx in (2, 3):
            terms.append(_shift(nodes, inits, "a", dy, dx, 9, f"a_{dy}_{dx}"))
        for dx in (-3, -2):
            terms.append(_shift(nodes, inits, "b", dy, dx, 9, f"b_{dy}_{dx}"))
    for dy in (-3, -2):
        for dx in (2, 3):
            terms.append(_shift(nodes, inits, "c", dy, dx, 9, f"c_{dy}_{dx}"))
        for dx in (-3, -2):
            terms.append(_shift(nodes, inits, "d", dy, dx, 9, f"d_{dy}_{dx}"))

    current = _or_all(nodes, terms, "fg_acc")
    channels = [f"fg_ch{i}" for i in range(9)]
    nodes.append(helper.make_node("Split", [current], channels, axis=1, split=[1] * 9))
    occupied = _or_all(nodes, channels, "occ_acc")

    nodes.extend(
        [
            helper.make_node("Not", [occupied], ["out_bg"]),
            helper.make_node("Concat", ["out_bg", current], ["out_bool"], axis=1),
            helper.make_node("Cast", ["out_bool"], ["out6"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out6"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - G, W - G]),
        ]
    )

    graph = helper.make_graph(nodes, "task225", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _load_examples() -> list[dict[str, list[list[int]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [ex for split in ("train", "test", "arc-gen") for ex in data.get(split, [])]


def verify_reference() -> None:
    for idx, ex in enumerate(_load_examples()):
        got = solve_reference(ex["input"])
        exp = np.asarray(ex["output"], dtype=np.int64)
        if not np.array_equal(got, exp):
            raise AssertionError(f"reference mismatch on example {idx}")


def verify_model(path: Path) -> None:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for idx, ex in enumerate(_load_examples()):
        inp = convert_to_numpy(ex, "input")
        exp = convert_to_numpy(ex, "output")
        assert inp is not None and exp is not None
        got = session.run([OUT_NAME], {IN_NAME: inp})[0]
        if not np.array_equal(got > 0.0, exp > 0.0):
            raise AssertionError(f"model mismatch on example {idx}")


def main() -> None:
    verify_reference()
    model = build_model()
    BEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, BEST_PATH)
    verify_model(BEST_PATH)
    result = score_file(BEST_PATH)
    if not result["valid"]:
        raise SystemExit(result["error"])
    print(
        f"{BEST_PATH} memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']:.6f}"
    )


if __name__ == "__main__":
    main()
