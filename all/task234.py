"""ONNX solution for task234: remove a connector and attach the carried block.

Task rule: two non-black colors are present.  One color forms a solid
rectangular body with a one-cell-wide stem pointing toward the other solid
rectangle.  Delete the stem, translate that color's rectangular body in the
stem direction until it touches the other rectangle, and keep the other
rectangle fixed.  The output grid size equals the input grid size.
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

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task234"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task234.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
FG = 9
IOH = IOW = 30
H = W = 20
MAX_SHIFT = 11
CANDIDATES = tuple(
    (direction, amount)
    for direction, amounts in (
        ("up", (10, 9, 8, 7, 6, 5, 4, 3, 2, 1)),
        ("down", (10, 9, 8, 7, 6, 5, 4, 3, 2, 1)),
        ("left", (10, 9, 8, 7, 6, 5, 4, 3, 2, 1)),
        ("right", (11, 10, 8, 7, 6, 5, 4, 3, 2, 1)),
    )
    for amount in amounts
)
SHAPE = [1, C, IOH, IOW]
FG_SHAPE = [1, FG, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    if any(init.name == name for init in inits):
        return name
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _bool(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=bool), name)


def _make_model(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto]) -> onnx.ModelProto:
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


def _slice(
    nodes: List[onnx.NodeProto],
    data: str,
    starts: str,
    ends: str,
    axes: str,
    out: str,
) -> str:
    nodes.append(helper.make_node("Slice", [data, starts, ends, axes], [out]))
    return out


def _pad(nodes: List[onnx.NodeProto], data: str, pads: list[int], out: str) -> str:
    nodes.append(helper.make_node("Pad", [data], [out], pads=pads))
    return out


def _shift(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    x: str,
    direction: str,
    amount: int,
    name: str,
    channels: int = FG,
) -> str:
    """Zero-fill shift for a [1, channels, 30, 30] bool tensor."""
    axes_row = "ax_r"
    axes_col = "ax_c"
    full_end = "full_end"
    zero = "zero_i"
    if amount == 0:
        nodes.append(helper.make_node("Identity", [x], [name]))
        return name

    if direction == "up":
        start = _i64(inits, [amount], f"idx_{amount}")
        sliced = _slice(nodes, x, start, full_end, axes_row, f"{name}_sl")
        pad = _bool(inits, np.zeros((1, channels, amount, W), dtype=bool), f"z_row_{channels}_{amount}")
        nodes.append(helper.make_node("Concat", [sliced, pad], [name], axis=2))
        return name
    if direction == "left":
        start = _i64(inits, [amount], f"idx_{amount}")
        sliced = _slice(nodes, x, start, full_end, axes_col, f"{name}_sl")
        pad = _bool(inits, np.zeros((1, channels, H, amount), dtype=bool), f"z_col_{channels}_{amount}")
        nodes.append(helper.make_node("Concat", [sliced, pad], [name], axis=3))
        return name
    if direction == "down":
        end = _i64(inits, [H - amount], f"idx_{H - amount}")
        sliced = _slice(nodes, x, zero, end, axes_row, f"{name}_sl")
        pad = _bool(inits, np.zeros((1, channels, amount, W), dtype=bool), f"z_row_{channels}_{amount}")
        nodes.append(helper.make_node("Concat", [pad, sliced], [name], axis=2))
        return name
    if direction == "right":
        end = _i64(inits, [W - amount], f"idx_{W - amount}")
        sliced = _slice(nodes, x, zero, end, axes_col, f"{name}_sl")
        pad = _bool(inits, np.zeros((1, channels, H, amount), dtype=bool), f"z_col_{channels}_{amount}")
        nodes.append(helper.make_node("Concat", [pad, sliced], [name], axis=3))
        return name
    raise ValueError(direction)


def build_onnx_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    _i64(inits, [0, 1, 2, 3], "axes_all")
    _i64(inits, [0, 0, 0, 0], "input_start")
    _i64(inits, [0, 1, 0, 0], "fg_start")
    _i64(inits, [1, C, H, W], "chw_end")
    _i64(inits, [1, 1, H, W], "bg_end")
    _i64(inits, [2], "ax_r")
    _i64(inits, [3], "ax_c")
    _i64(inits, [0], "zero_i")
    _i64(inits, [H], "full_end")
    zero_f = _f32(inits, [0.0], "zero_f")
    zero_occ = _bool(inits, np.zeros((1, 1, H, W), dtype=bool), "zero_occ")

    nodes.extend(
        [
            helper.make_node("Greater", [IN_NAME, zero_f], ["in_b"]),
            helper.make_node("Slice", ["in_b", "input_start", "bg_end", "axes_all"], ["bg_in"]),
            helper.make_node("Slice", ["in_b", "fg_start", "chw_end", "axes_all"], ["fg"]),
            helper.make_node("Cast", ["bg_in"], ["bg_in_f"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", ["bg_in_f"], ["valid_rows_f"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["bg_in_f"], ["valid_cols_f"], axes=[2], keepdims=1),
            helper.make_node("Greater", ["valid_rows_f", zero_f], ["valid_rows"]),
            helper.make_node("Greater", ["valid_cols_f", zero_f], ["valid_cols"]),
            helper.make_node("And", ["valid_rows", "valid_cols"], ["valid"]),
        ]
    )

    left = _shift(nodes, inits, "fg", "left", 1, "fg_left")
    right = _shift(nodes, inits, "fg", "right", 1, "fg_right")
    up = _shift(nodes, inits, "fg", "up", 1, "fg_up")
    down = _shift(nodes, inits, "fg", "down", 1, "fg_down")
    nodes.extend(
        [
            helper.make_node("Or", [left, right], ["has_h"]),
            helper.make_node("Or", [up, down], ["has_v"]),
            helper.make_node("And", ["fg", "has_h"], ["body_h"]),
            helper.make_node("And", ["body_h", "has_v"], ["body"]),
            helper.make_node("Not", ["body"], ["not_body"]),
            helper.make_node("And", ["fg", "not_body"], ["removed"]),
            helper.make_node("Cast", ["removed"], ["removed_f"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", ["removed_f"], ["removed_any_f"], axes=[2, 3], keepdims=1),
            helper.make_node("Greater", ["removed_any_f", zero_f], ["has_removed"]),
            helper.make_node("Not", ["has_removed"], ["not_has_removed"]),
            helper.make_node("And", ["fg", "not_has_removed"], ["base"]),
            helper.make_node("Cast", ["base"], ["base_f"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", ["base_f"], ["anchor_any_f"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["anchor_any_f", zero_f], ["anchor_any"]),
            helper.make_node("And", ["body", "has_removed"], ["mover_body"]),
            helper.make_node("Cast", ["mover_body"], ["mover_body_f"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", ["mover_body_f"], ["body_occ_f"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["body_occ_f", zero_f], ["body_occ"]),
        ]
    )

    moved = zero_occ
    touch_masks = {
        "up": _shift(nodes, inits, "anchor_any", "down", 1, "touch_up", channels=1),
        "down": _shift(nodes, inits, "anchor_any", "up", 1, "touch_down", channels=1),
        "left": _shift(nodes, inits, "anchor_any", "right", 1, "touch_left", channels=1),
        "right": _shift(nodes, inits, "anchor_any", "left", 1, "touch_right", channels=1),
    }
    # A valid candidate is the shifted body whose next one-cell step in the
    # same direction intersects the fixed anchor.  This uniquely identifies the
    # connector length without measuring bounding boxes.
    for direction, amount in CANDIDATES:
        shifted = _shift(nodes, inits, "body_occ", direction, amount, f"{direction}{amount}_body", channels=1)
        nodes.extend(
            [
                helper.make_node("And", [shifted, touch_masks[direction]], [f"{direction}{amount}_touch"]),
                helper.make_node(
                    "Cast",
                    [f"{direction}{amount}_touch"],
                    [f"{direction}{amount}_touch_f"],
                    to=TensorProto.FLOAT,
                ),
                helper.make_node(
                    "ReduceMax",
                    [f"{direction}{amount}_touch_f"],
                    [f"{direction}{amount}_touch_any_f"],
                    axes=[2, 3],
                    keepdims=1,
                ),
                helper.make_node(
                    "Greater",
                    [f"{direction}{amount}_touch_any_f", zero_f],
                    [f"{direction}{amount}_touch_any"],
                ),
                helper.make_node("And", [f"{direction}{amount}_touch_any", shifted], [f"{direction}{amount}_take"]),
                helper.make_node("Not", [f"{direction}{amount}_touch_any"], [f"{direction}{amount}_not_cond"]),
                helper.make_node("And", [f"{direction}{amount}_not_cond", moved], [f"{direction}{amount}_keep"]),
                helper.make_node("Or", [f"{direction}{amount}_take", f"{direction}{amount}_keep"], [f"{direction}{amount}_moved"]),
            ]
        )
        moved = f"{direction}{amount}_moved"

    nodes.extend(
        [
            helper.make_node("And", ["has_removed", moved], ["moved_color"]),
            helper.make_node("Or", ["base", "moved_color"], ["fg_out"]),
            helper.make_node("Or", ["anchor_any", moved], ["out_any"]),
            helper.make_node("Not", ["out_any"], ["no_fg"]),
            helper.make_node("And", ["valid", "no_fg"], ["bg_out"]),
            helper.make_node("Concat", ["bg_out", "fg_out"], ["out_b"], axis=1),
            helper.make_node("Cast", ["out_b"], ["out20_f"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out20_f"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, IOH - H, IOW - W]),
        ]
    )

    return _make_model(nodes, inits)


def solve(grid: list[list[int]]) -> np.ndarray:
    """Python reference used for local sanity checks."""
    arr = np.asarray(grid)
    out = np.zeros_like(arr)
    colors = [int(c) for c in np.unique(arr) if c != 0]

    def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
        rows, cols = np.where(mask)
        return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())

    infos: list[tuple[int, int, int, int, int, int, int]] = []
    for color in colors:
        mask = arr == color
        r0, r1, c0, c1 = bbox(mask)
        area = int(mask.sum())
        infos.append((color, area, (r1 - r0 + 1) * (c1 - c0 + 1), r0, r1, c0, c1))

    mover_info = [info for info in infos if info[1] < info[2]][0]
    anchor_info = [info for info in infos if info[0] != mover_info[0]][0]
    mover_color = mover_info[0]
    anchor_color = anchor_info[0]
    ar0, ar1, ac0, ac1 = anchor_info[3:]
    mask = arr == mover_color

    best: tuple[int, int, int, int, int] | None = None
    for r0 in range(mover_info[3], mover_info[4] + 1):
        for r1 in range(r0, mover_info[4] + 1):
            for c0 in range(mover_info[5], mover_info[6] + 1):
                for c1 in range(c0, mover_info[6] + 1):
                    h = r1 - r0 + 1
                    w = c1 - c0 + 1
                    if h >= 2 and w >= 2 and mask[r0 : r1 + 1, c0 : c1 + 1].all():
                        area = h * w
                        if best is None or area > best[0]:
                            best = (area, r0, r1, c0, c1)
    assert best is not None
    _, br0, br1, bc0, bc1 = best

    dr = dc = 0
    if br1 < ar0:
        dr = ar0 - 1 - br1
    elif br0 > ar1:
        dr = ar1 + 1 - br0
    elif bc1 < ac0:
        dc = ac0 - 1 - bc1
    else:
        dc = ac1 + 1 - bc0

    out[arr == anchor_color] = anchor_color
    out[br0 + dr : br1 + dr + 1, bc0 + dc : bc1 + dc + 1] = mover_color
    return out


def _validate_reference() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            pred = solve(example["input"])
            expected = np.asarray(example["output"])
            if not np.array_equal(pred, expected):
                raise AssertionError(f"reference mismatch: {split} {idx}")


def _validate_onnx(path: Path) -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            inp = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if inp is None or expected is None:
                continue
            pred = sess.run(None, {IN_NAME: inp})[0]
            if not np.array_equal(pred > 0.0, expected > 0.0):
                raise AssertionError(f"onnx mismatch: {split} {idx}")


def main() -> None:
    _validate_reference()
    model = build_onnx_model()
    onnx.save(model, BEST_PATH)
    _validate_onnx(BEST_PATH)
    result = score_file(BEST_PATH)
    print(result)


if __name__ == "__main__":
    main()
