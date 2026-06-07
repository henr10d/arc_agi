"""ONNX solution for NeuroGolf task192 using local rectangle-damage repair.

Task rule: keep the dominant non-background color, erase all other colors, and
recolor a non-dominant pixel to the dominant color only when it is a damaged
corner/interior cell of a rectangle. Equivalently, a minority-colored pixel is
filled when it has at least one dominant-color horizontal neighbor and at least
one dominant-color vertical neighbor. This completes noisy rectangles without
bridging two separate rectangles across a straight one-cell gap.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TASK_ID = "task192"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
WORK = 20
PAD = H - WORK
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def solve(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    """Reference implementation of the local dominant-color repair rule."""
    arr = np.asarray(grid, dtype=np.int64)
    counts = Counter(arr.ravel())
    counts.pop(0, None)
    color = int(counts.most_common(1)[0][0])
    dom = arr == color
    out = dom.copy()
    h, w = arr.shape
    for r, c in zip(*np.where((arr != 0) & (arr != color))):
        left = c > 0 and dom[r, c - 1]
        right = c + 1 < w and dom[r, c + 1]
        up = r > 0 and dom[r - 1, c]
        down = r + 1 < h and dom[r + 1, c]
        out[r, c] = bool((left or right) and (up or down))
    return np.where(out, color, 0).astype(np.int64)


def _init_i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))
    return name


def _init_bool(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=bool), name=name))
    return name


def _slice(
    nodes: List[onnx.NodeProto],
    data: str,
    output: str,
    starts: str,
    ends: str,
    axes: str,
) -> None:
    nodes.append(helper.make_node("Slice", [data, starts, ends, axes], [output]))


def build_onnx_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes_c_hw = _init_i64(inits, [1, 2, 3], "achw")
    count_start = _init_i64(inits, [1], "cts")
    count_end = _init_i64(inits, [C], "cte")
    count_axis = _init_i64(inits, [1], "cta")
    dom_hw_start = _init_i64(inits, [0, 0], "dhs")
    dom_hw_end = _init_i64(inits, [WORK, WORK], "dhe")
    dom_hw_axes = _init_i64(inits, [2, 3], "dha")
    bg_start = _init_i64(inits, [0, 0, 0], "bgs")
    bg_end = _init_i64(inits, [1, WORK, WORK], "bge")
    one_i64 = _init_i64(inits, [1], "one")
    channel_ids = _init_i64(inits, np.arange(9, dtype=np.int64).reshape(1, 9), "ch")

    s_c0 = _init_i64(inits, [0], "sc0")
    e_c_last = _init_i64(inits, [WORK - 1], "ecl")
    s_c1 = _init_i64(inits, [1], "sc1")
    e_c_work = _init_i64(inits, [WORK], "ecw")
    ax_w = _init_i64(inits, [3], "aw")
    s_r0 = _init_i64(inits, [0], "sr0")
    e_r_last = _init_i64(inits, [WORK - 1], "erl")
    s_r1 = _init_i64(inits, [1], "sr1")
    e_r_work = _init_i64(inits, [WORK], "erw")
    ax_h = _init_i64(inits, [2], "ar")
    false_col = _init_bool(inits, np.zeros((1, 1, WORK, 1), dtype=bool), "fc")
    false_row = _init_bool(inits, np.zeros((1, 1, 1, WORK), dtype=bool), "fr")

    nodes.extend(
        [
            helper.make_node("ReduceSum", [IN_NAME], ["counts10"], axes=[2, 3], keepdims=0),
            helper.make_node("Slice", ["counts10", count_start, count_end, count_axis], ["counts"]),
            helper.make_node("ArgMax", ["counts"], ["idx"], axis=1, keepdims=0),
            helper.make_node("Add", ["idx", one_i64], ["coloridx"]),
            helper.make_node("Gather", [IN_NAME, "coloridx"], ["dom30"], axis=1),
            helper.make_node("Cast", ["dom30"], ["db30"], to=TensorProto.BOOL),
            helper.make_node("Slice", ["db30", dom_hw_start, dom_hw_end, dom_hw_axes], ["db"]),
            helper.make_node("ReduceSum", [IN_NAME], ["inside30"], axes=[1], keepdims=1),
            helper.make_node("Cast", ["inside30"], ["inside30b"], to=TensorProto.BOOL),
            helper.make_node("Slice", ["inside30b", dom_hw_start, dom_hw_end, dom_hw_axes], ["inside"]),
            helper.make_node("Slice", [IN_NAME, bg_start, bg_end, axes_c_hw], ["bg0"]),
            helper.make_node("Cast", ["bg0"], ["bg0b"], to=TensorProto.BOOL),
            helper.make_node("Not", ["bg0b"], ["nbg"]),
            helper.make_node("And", ["inside", "nbg"], ["fgb"]),
            helper.make_node("Not", ["db"], ["ndb"]),
            helper.make_node("And", ["fgb", "ndb"], ["minor"]),
        ]
    )

    _slice(nodes, "db", "lcore", s_c0, e_c_last, ax_w)
    _slice(nodes, "db", "rcore", s_c1, e_c_work, ax_w)
    _slice(nodes, "db", "ucore", s_r0, e_r_last, ax_h)
    _slice(nodes, "db", "dcore", s_r1, e_r_work, ax_h)

    nodes.extend(
        [
            helper.make_node("Concat", [false_col, "lcore"], ["left"], axis=3),
            helper.make_node("Concat", ["rcore", false_col], ["right"], axis=3),
            helper.make_node("Concat", [false_row, "ucore"], ["up"], axis=2),
            helper.make_node("Concat", ["dcore", false_row], ["down"], axis=2),
            helper.make_node("Or", ["left", "right"], ["hnb"]),
            helper.make_node("Or", ["up", "down"], ["vnb"]),
            helper.make_node("And", ["hnb", "vnb"], ["corner"]),
            helper.make_node("And", ["minor", "corner"], ["repair"]),
            helper.make_node("Or", ["db", "repair"], ["maskb"]),
            helper.make_node("Equal", ["idx", channel_ids], ["oh"]),
            helper.make_node("Unsqueeze", ["oh"], ["ohb"], axes=[2, 3]),
            helper.make_node("And", ["ohb", "maskb"], ["outfgb"]),
            helper.make_node("Not", ["maskb"], ["notmask"]),
            helper.make_node("And", ["inside", "notmask"], ["bgb"]),
            helper.make_node("Concat", ["bgb", "outfgb"], ["out20b"], axis=1),
            helper.make_node("Cast", ["out20b"], ["out20"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out20"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]),
        ]
    )

    graph = helper.make_graph(nodes, "task192", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _grid_to_onehot(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _verify_model(path: Path) -> None:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for i, example in enumerate(data.get(split, [])):
            expected = _grid_to_onehot(example["output"])
            pred = sess.run([OUT_NAME], {IN_NAME: _grid_to_onehot(example["input"])})[0]
            if not np.array_equal(pred > 0.0, expected > 0.0):
                raise AssertionError(f"{split} example {i} failed")


def main() -> None:
    model = build_onnx_model()
    onnx.save(model, BEST_PATH)
    _verify_model(BEST_PATH)
    print(f"wrote {BEST_PATH}")


if __name__ == "__main__":
    main()
