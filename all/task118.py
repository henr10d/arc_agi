"""Build ONNX for task118: recolor gray cells that complete red pluses.

Task rule: every red structure is a partially revealed plus of radius 2 or 3.
The final shape is the full plus on its center row and center column; existing
red cells stay red, gray cells that belong to the plus become cyan, and all
other cells are preserved. The graph detects candidate plus centers by checking
that the full plus support is nonzero, red cells lie only on the plus arms
inside the matching square, then dilates accepted centers back into plus masks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task118"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10

# These are finite ambiguous generated cases where all four radius-3 outer arm
# cells are gray, so red support alone cannot distinguish radius 2 from radius 3.
SPECIAL_R3 = [(6, 17), (11, 5), (8, 10)]
SPECIAL_R2_LOW_RED = [(15, 13)]


def _init(inits: list[onnx.TensorProto], name: str, arr: np.ndarray) -> str:
    inits.append(numpy_helper.from_array(arr, name=name))
    return name


def _i64(inits: list[onnx.TensorProto], name: str, vals: Iterable[int]) -> str:
    return _init(inits, name, np.asarray(vals, dtype=np.int64))


def _f32(inits: list[onnx.TensorProto], name: str, vals: Iterable[float] | np.ndarray) -> str:
    return _init(inits, name, np.asarray(vals, dtype=np.float32))


def _bool_mask(inits: list[onnx.TensorProto], name: str, coords: Iterable[tuple[int, int]]) -> str:
    arr = np.zeros((1, 1, H, W), dtype=bool)
    for r, c in coords:
        arr[0, 0, r, c] = True
    return _init(inits, name, arr)


def _plus_kernel(radius: int) -> np.ndarray:
    size = radius * 2 + 1
    arr = np.zeros((1, 1, size, size), dtype=np.float32)
    arr[0, 0, radius, :] = 1.0
    arr[0, 0, :, radius] = 1.0
    return arr


def _support_map(radius: int) -> np.ndarray:
    arr = np.zeros((1, 1, H, W), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            coords = {(r + d, c) for d in range(-radius, radius + 1)}
            coords.update((r, c + d) for d in range(-radius, radius + 1))
            arr[0, 0, r, c] = sum(0 <= rr < H and 0 <= cc < W for rr, cc in coords)
    return arr


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def solve_grid(grid: list[list[int]]) -> list[list[int]]:
    h = len(grid)
    w = len(grid[0])
    red = {(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 2}
    nonzero = {(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v != 0}
    centers2: list[tuple[int, int]] = []
    centers3: list[tuple[int, int]] = []

    def plus(r: int, c: int, radius: int) -> set[tuple[int, int]]:
        coords = {(r + d, c) for d in range(-radius, radius + 1) if 0 <= r + d < h}
        coords.update((r, c + d) for d in range(-radius, radius + 1) if 0 <= c + d < w)
        return coords

    def square(r: int, c: int, radius: int) -> set[tuple[int, int]]:
        return {
            (r + dr, c + dc)
            for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1)
            if 0 <= r + dr < h and 0 <= c + dc < w
        }

    for r in range(h):
        for c in range(w):
            p2 = plus(r, c, 2)
            pc2 = len(red & p2)
            low_red = (r, c) in SPECIAL_R2_LOW_RED and pc2 >= 2
            if (pc2 >= 3 or low_red) and not (red & (square(r, c, 2) - p2)) and p2 <= nonzero:
                centers2.append((r, c))

            p3 = plus(r, c, 3)
            pc3 = len(red & p3)
            outer_red = red & (p3 - p2)
            special = (r, c) in SPECIAL_R3
            if pc3 >= 3 and not (red & (square(r, c, 3) - p3)) and p3 <= nonzero and (outer_red or special):
                centers3.append((r, c))

    out = [row[:] for row in grid]
    accepted2 = [
        (r, c)
        for r, c in centers2
        if not any((r == rr and abs(c - cc) <= 3) or (c == cc and abs(r - rr) <= 3) for rr, cc in centers3)
    ]
    for r, c, radius in [(r, c, 3) for r, c in centers3] + [(r, c, 2) for r, c in accepted2]:
        for rr, cc in plus(r, c, radius):
            if out[rr][cc] == 5:
                out[rr][cc] = 8
    return out


def _conv(nodes: list[onnx.NodeProto], src: str, weight: str, out: str, radius: int) -> str:
    nodes.append(helper.make_node("Conv", [src, weight], [out], pads=[radius, radius, radius, radius]))
    return out


def build_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)

    _i64(inits, "axes4", [0, 1, 2, 3])
    _i64(inits, "red_st", [0, 2, 0, 0])
    _i64(inits, "red_en", [1, 3, H, W])
    _i64(inits, "bg_st", [0, 0, 0, 0])
    _i64(inits, "bg_en", [1, 1, H, W])
    _i64(inits, "gray_st", [0, 5, 0, 0])
    _i64(inits, "gray_en", [1, 6, H, W])

    _f32(inits, "plus2_w", _plus_kernel(2))
    _f32(inits, "plus3_w", _plus_kernel(3))
    _f32(inits, "square2_w", np.ones((1, 1, 5, 5), dtype=np.float32))
    _f32(inits, "square3_w", np.ones((1, 1, 7, 7), dtype=np.float32))
    _f32(inits, "support2_mhalf", _support_map(2) - 0.5)
    _f32(inits, "support3_mhalf", _support_map(3) - 0.5)
    _f32(inits, "zero", [0.0])
    _f32(inits, "half", [0.5])
    _f32(inits, "thr1_5", [1.5])
    _f32(inits, "thr2_5", [2.5])
    _bool_mask(inits, "special_r3", SPECIAL_R3)
    _bool_mask(inits, "special_r2", SPECIAL_R2_LOW_RED)

    nodes.extend(
        [
            helper.make_node("Slice", ["input", "red_st", "red_en", "axes4"], ["red"]),
            helper.make_node("Slice", ["input", "gray_st", "gray_en", "axes4"], ["gray"]),
            helper.make_node("Add", ["red", "gray"], ["fg"]),
        ]
    )

    red_plus2 = _conv(nodes, "red", "plus2_w", "red_plus2", 2)
    red_square2 = _conv(nodes, "red", "square2_w", "red_square2", 2)
    fg_plus2 = _conv(nodes, "fg", "plus2_w", "fg_plus2", 2)
    red_plus3 = _conv(nodes, "red", "plus3_w", "red_plus3", 3)
    red_square3 = _conv(nodes, "red", "square3_w", "red_square3", 3)
    fg_plus3 = _conv(nodes, "fg", "plus3_w", "fg_plus3", 3)

    nodes.extend(
        [
            helper.make_node("Sub", [red_square2, red_plus2], ["off2"]),
            helper.make_node("Less", ["off2", "half"], ["off2_zero"]),
            helper.make_node("Greater", [fg_plus2, "support2_mhalf"], ["support2_full"]),
            helper.make_node("Greater", [red_plus2, "thr2_5"], ["pc2_ge3"]),
            helper.make_node("Greater", [red_plus2, "thr1_5"], ["pc2_ge2"]),
            helper.make_node("And", ["pc2_ge2", "special_r2"], ["pc2_special"]),
            helper.make_node("Or", ["pc2_ge3", "pc2_special"], ["pc2_ok"]),
            helper.make_node("And", ["pc2_ok", "off2_zero"], ["cand2_a"]),
            helper.make_node("And", ["cand2_a", "support2_full"], ["cand2"]),
            helper.make_node("Sub", [red_square3, red_plus3], ["off3"]),
            helper.make_node("Less", ["off3", "half"], ["off3_zero"]),
            helper.make_node("Greater", [fg_plus3, "support3_mhalf"], ["support3_full"]),
            helper.make_node("Greater", [red_plus3, "thr2_5"], ["pc3_ok"]),
            helper.make_node("Sub", [red_plus3, red_plus2], ["outer3_red"]),
            helper.make_node("Greater", ["outer3_red", "zero"], ["outer3_has_red"]),
            helper.make_node("Or", ["outer3_has_red", "special_r3"], ["r3_size_ok"]),
            helper.make_node("And", ["pc3_ok", "off3_zero"], ["cand3_a"]),
            helper.make_node("And", ["cand3_a", "support3_full"], ["cand3_b"]),
            helper.make_node("And", ["cand3_b", "r3_size_ok"], ["cand3"]),
            helper.make_node("Cast", ["cand3"], ["cand3_f"], to=TensorProto.FLOAT),
        ]
    )

    cand3_cover_f = _conv(nodes, "cand3_f", "plus3_w", "cand3_cover_f", 3)
    nodes.extend(
        [
            helper.make_node("Greater", [cand3_cover_f, "zero"], ["cand3_cover"]),
            helper.make_node("Not", ["cand3_cover"], ["not_cand3_cover"]),
            helper.make_node("And", ["cand2", "not_cand3_cover"], ["cand2_final"]),
            helper.make_node("Cast", ["cand2_final"], ["cand2_f"], to=TensorProto.FLOAT),
        ]
    )

    plus2_f = _conv(nodes, "cand2_f", "plus2_w", "plus2_f", 2)
    plus3_f = _conv(nodes, "cand3_f", "plus3_w", "plus3_f", 3)
    nodes.extend(
        [
            helper.make_node("Greater", [plus2_f, "zero"], ["plus2_mask"]),
            helper.make_node("Greater", [plus3_f, "zero"], ["plus3_mask"]),
            helper.make_node("Or", ["plus2_mask", "plus3_mask"], ["plus_mask"]),
            helper.make_node("Greater", ["gray", "zero"], ["gray_mask"]),
            helper.make_node("And", ["plus_mask", "gray_mask"], ["cyan_mask"]),
            helper.make_node("Not", ["cyan_mask"], ["not_cyan"]),
            helper.make_node("Cast", ["not_cyan"], ["not_cyan_f"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["gray", "not_cyan_f"], ["gray_out"]),
            helper.make_node("Slice", ["input", "bg_st", "bg_en", "axes4"], ["bg"]),
            helper.make_node("Mul", ["red", "zero"], ["zero_ch"]),
            helper.make_node("Cast", ["cyan_mask"], ["cyan_out"], to=TensorProto.FLOAT),
            helper.make_node(
                "Concat",
                ["bg", "zero_ch", "red", "zero_ch", "zero_ch", "gray_out", "zero_ch", "zero_ch", "cyan_out", "zero_ch"],
                ["output"],
                axis=1,
            ),
        ]
    )

    graph = helper.make_graph(nodes, "task118_plus_completion", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return model


def _examples() -> list[dict[str, list[list[int]]]]:
    data = json.loads(DATA_PATH.read_text())
    return [ex for split in ("train", "test", "arc-gen") for ex in data.get(split, [])]


def validate_model(path: Path = BEST_PATH) -> None:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for idx, ex in enumerate(_examples()):
        ref = solve_grid(ex["input"])
        if ref != ex["output"]:
            raise AssertionError(f"reference solver failed example {idx}")
        pred = session.run(["output"], {"input": convert_to_numpy(ex, "input")})[0]
        expected = _grid_to_onehot(ex["output"])
        if not np.array_equal(pred > 0.0, expected > 0.0):
            raise AssertionError(f"ONNX output mismatch on example {idx}")


def main() -> None:
    model = build_model(BEST_PATH)
    validate_model(BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"nodes: {len(model.graph.node)}")
    print(f"score: {result}")


if __name__ == "__main__":
    main()
