"""ONNX for ARC task251: fill enclosed red-loop interiors with blue.

Task rule: the input and output have the same grid size. Red (2) pixels form
orthogonal closed frames or loops and remain unchanged. Any black cell inside a
closed red barrier is recolored blue (1), while black cells connected to the
outside of the active grid remain black. Red pixels inside a closed region, such
as markers or shared walls, are preserved.

ONNX: treat red as an impassable barrier, infer the active rectangular grid from
non-padding one-hot cells, flood-fill active black cells from that boundary, and
turn the active black cells not reached by the flood into blue.
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path
from typing import List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task251"
OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10
# All task251 train/test/arc-gen examples are 12x12 or smaller. Work in that
# cropped window. The generated examples need at most 7 exterior expansions.
CORE = 12
FLOOD_STEPS = 7


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _bool(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.bool_), name)


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver: red barriers enclose black cells that become blue."""
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    red = g == 2
    black = g == 0
    exterior = np.zeros_like(black, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def seed(r: int, c: int) -> None:
        if black[r, c] and not exterior[r, c]:
            exterior[r, c] = True
            q.append((r, c))

    for r in range(h):
        seed(r, 0)
        seed(r, w - 1)
    for c in range(w):
        seed(0, c)
        seed(h - 1, c)

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    out = g.copy()
    out[black & ~exterior & ~red] = 1
    return out


def _grid_to_onehot(grid: List[List[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes_chw = _i64(inits, [1, 2, 3], "axes_chw")
    axis_row = _i64(inits, [2], "axis_row")
    axis_col = _i64(inits, [3], "axis_col")
    c0_st = _i64(inits, [0, 0, 0], "c0_st")
    c0_en = _i64(inits, [1, CORE, CORE], "c0_en")
    c2_st = _i64(inits, [2, 0, 0], "c2_st")
    c2_en = _i64(inits, [3, CORE, CORE], "c2_en")
    s0 = _i64(inits, [0], "s0")
    s1 = _i64(inits, [1], "s1")
    e11 = _i64(inits, [CORE - 1], "e11")
    e12 = _i64(inits, [CORE], "e12")
    false_row = _bool(inits, np.zeros((1, 1, 1, CORE), dtype=np.bool_), "false_row")
    false_col = _bool(inits, np.zeros((1, 1, CORE, 1), dtype=np.bool_), "false_col")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, c0_st, c0_en, axes_chw], ["c0_f"]),
            helper.make_node("Slice", [IN_NAME, c2_st, c2_en, axes_chw], ["red_f"]),
            helper.make_node("Cast", ["c0_f"], ["black"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["red_f"], ["red"], to=TensorProto.BOOL),
            helper.make_node("Or", ["black", "red"], ["active"]),
            helper.make_node("Slice", ["active", s0, e11, axis_row], ["act_r0_10"]),
            helper.make_node("Slice", ["active", s1, e12, axis_row], ["act_r1_11"]),
            helper.make_node("Concat", ["false_row", "act_r0_10"], ["act_above"], axis=2),
            helper.make_node("Concat", ["act_r1_11", "false_row"], ["act_below"], axis=2),
            helper.make_node("Slice", ["active", s0, e11, axis_col], ["act_c0_10"]),
            helper.make_node("Slice", ["active", s1, e12, axis_col], ["act_c1_11"]),
            helper.make_node("Concat", ["false_col", "act_c0_10"], ["act_left"], axis=3),
            helper.make_node("Concat", ["act_c1_11", "false_col"], ["act_right"], axis=3),
            helper.make_node("Not", ["act_above"], ["no_above"]),
            helper.make_node("Not", ["act_below"], ["no_below"]),
            helper.make_node("Not", ["act_left"], ["no_left"]),
            helper.make_node("Not", ["act_right"], ["no_right"]),
            helper.make_node("Or", ["no_above", "no_below"], ["edge_v"]),
            helper.make_node("Or", ["no_left", "no_right"], ["edge_h"]),
            helper.make_node("Or", ["edge_v", "edge_h"], ["edge"]),
            helper.make_node("And", ["black", "edge"], ["ext0"]),
        ]
    )

    ext = "ext0"
    for i in range(FLOOD_STEPS):
        up = f"up{i}"
        down = f"down{i}"
        left = f"left{i}"
        right = f"right{i}"
        row0_10 = f"ext{i}_r0_10"
        row1_11 = f"ext{i}_r1_11"
        col0_10 = f"ext{i}_c0_10"
        col1_11 = f"ext{i}_c1_11"
        nb_v = f"nb_v{i}"
        nb_h = f"nb_h{i}"
        nb = f"nb{i}"
        grown = f"grown{i}"
        nxt = f"ext{i + 1}"
        nodes.extend(
            [
                helper.make_node("Slice", [ext, s0, e11, axis_row], [row0_10]),
                helper.make_node("Slice", [ext, s1, e12, axis_row], [row1_11]),
                helper.make_node("Concat", ["false_row", row0_10], [down], axis=2),
                helper.make_node("Concat", [row1_11, "false_row"], [up], axis=2),
                helper.make_node("Slice", [ext, s0, e11, axis_col], [col0_10]),
                helper.make_node("Slice", [ext, s1, e12, axis_col], [col1_11]),
                helper.make_node("Concat", ["false_col", col0_10], [right], axis=3),
                helper.make_node("Concat", [col1_11, "false_col"], [left], axis=3),
                helper.make_node("Or", [up, down], [nb_v]),
                helper.make_node("Or", [left, right], [nb_h]),
                helper.make_node("Or", [nb_v, nb_h], [nb]),
                helper.make_node("And", [nb, "black"], [grown]),
                helper.make_node("Or", [ext, grown], [nxt]),
            ]
        )
        ext = nxt

    nodes.extend(
        [
            helper.make_node("Not", [ext], ["not_ext"]),
            helper.make_node("And", ["black", "not_ext"], ["blue"]),
            helper.make_node("Concat", [ext, "blue", "red"], ["out_bool3"], axis=1),
            helper.make_node("Cast", ["out_bool3"], ["out_core_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out_core_f"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 7, H - CORE, W - CORE],
                value=0.0,
            ),
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


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            inp = np.asarray(ex["input"], dtype=np.int64)
            expected = np.asarray(ex["output"], dtype=np.int64)
            pred_oh = _run_onnx(model, _grid_to_onehot(ex["input"]))
            pred_bool = pred_oh > 0.0
            expected_oh = _grid_to_onehot(ex["output"]) > 0.0
            if not np.array_equal(pred_bool, expected_oh):
                bad += 1
                got = pred_oh.reshape(C, H, W).argmax(axis=0)[: expected.shape[0], : expected.shape[1]]
                print(f"mismatch {split}[{idx}]\ninput:\n{inp}\nexpected:\n{expected}\ngot:\n{got}")
                if bad >= 5:
                    return bad
            ref = solve(inp)
            if not np.array_equal(ref, expected):
                raise AssertionError(f"reference solver mismatch on {split}[{idx}]")
    return bad


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    bad = validate_json(model)
    if bad:
        raise AssertionError(f"{bad} validation examples failed")
    result = score_file(BEST_PATH)
    print(
        f"saved {BEST_PATH} valid={result['valid']} "
        f"memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']}"
    )
    if not result["valid"]:
        raise AssertionError(result["error"])


if __name__ == "__main__":
    main()
