"""Minimal ONNX for ARC task266: replace one red marker by directional marks.

Task rule: the visible grid is always 3x5.  The single red cell is a position
code.  Paint the diagonal neighbors one row away from the red cell: green and
magenta go on the row above at columns c-1 and c+1, while cyan and orange go on
the row below at columns c-1 and c+1.  A middle-row red emits both pairs, a
top-row red emits only the lower pair, and a bottom-row red emits only the
upper pair.  Marks whose columns fall outside the 3x5 grid are omitted; all
other visible cells remain black.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task266"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task266.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
GH = 3
GW = 5
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _make_model(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], name: str) -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(nodes, name, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver for the 3x5 marker rule."""
    g = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(g)
    red = np.argwhere(g == 2)
    if len(red) != 1:
        raise ValueError(f"expected one red cell, found {len(red)}")
    r, c = (int(v) for v in red[0])

    def paint(rr: int, cc: int, color: int) -> None:
        if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
            out[rr, cc] = color

    if r == 1:
        paint(0, c - 1, 3)
        paint(0, c + 1, 6)
        paint(2, c - 1, 8)
        paint(2, c + 1, 7)
    elif r == 0:
        paint(1, c - 1, 8)
        paint(1, c + 1, 7)
    elif r == 2:
        paint(1, c - 1, 3)
        paint(1, c + 1, 6)
    return out


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _examples() -> list[dict[str, Any]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [ex for split in ("train", "test", "arc-gen") for ex in data.get(split, [])]


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def build_conv_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    red_st = _i64(inits, [0, 2, 0, 0], "red_st")
    red_en = _i64(inits, [1, 3, GH, GW], "red_en")

    weights = np.zeros((9, 1, 3, 3), dtype=np.float32)
    bias = np.zeros((9,), dtype=np.float32)
    bias[0] = 1.0

    # Conv is cross-correlation: out[y, x] reads red[y + ky - 1, x + kx - 1].
    weights[0, 0, 2, 2] = -1.0
    weights[0, 0, 2, 0] = -1.0
    weights[0, 0, 0, 2] = -1.0
    weights[0, 0, 0, 0] = -1.0
    weights[3, 0, 2, 2] = 1.0
    weights[6, 0, 2, 0] = 1.0
    weights[8, 0, 0, 2] = 1.0
    weights[7, 0, 0, 0] = 1.0

    w_name = _f32(inits, weights, "w")
    b_name = _f32(inits, bias, "b")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, red_st, red_en], ["redf"]),
            helper.make_node("Conv", ["redf", w_name, b_name], ["out9"], pads=[1, 1, 1, 1]),
            helper.make_node("Pad", ["out9"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 1, H - GH, W - GW]),
        ]
    )
    return _make_model(nodes, inits, "task266_conv")


def build_structural_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    ax123 = _i64(inits, [1, 2, 3], "ax123")
    ax3 = _i64(inits, [3], "ax3")
    rt_st = _i64(inits, [2, 0, 0], "rt_st")
    rt_en = _i64(inits, [3, 1, GW], "rt_en")
    rm_st = _i64(inits, [2, 1, 0], "rm_st")
    rm_en = _i64(inits, [3, 2, GW], "rm_en")
    rb_st = _i64(inits, [2, 2, 0], "rb_st")
    rb_en = _i64(inits, [3, 3, GW], "rb_en")
    c0 = _i64(inits, [0], "c0")
    c1 = _i64(inits, [1], "c1")
    c4 = _i64(inits, [4], "c4")
    c5 = _i64(inits, [5], "c5")

    _init(inits, np.zeros((1, 1, 1, 1), dtype=bool), "zcol1")
    _init(inits, np.zeros((1, 1, 1, GW), dtype=bool), "zrow")
    _init(inits, np.zeros((1, 1, GH, GW), dtype=bool), "zchan")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, rt_st, rt_en, ax123], ["rtf"]),
            helper.make_node("Slice", [IN_NAME, rm_st, rm_en, ax123], ["rmf"]),
            helper.make_node("Slice", [IN_NAME, rb_st, rb_en, ax123], ["rbf"]),
            helper.make_node("Cast", ["rtf"], ["rt"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["rmf"], ["rm"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["rbf"], ["rb"], to=TensorProto.BOOL),
            helper.make_node("Slice", ["rm", c1, c5, ax3], ["rm1_5"]),
            helper.make_node("Slice", ["rm", c0, c4, ax3], ["rm0_4"]),
            helper.make_node("Slice", ["rt", c1, c5, ax3], ["rt1_5"]),
            helper.make_node("Slice", ["rt", c0, c4, ax3], ["rt0_4"]),
            helper.make_node("Slice", ["rb", c1, c5, ax3], ["rb1_5"]),
            helper.make_node("Slice", ["rb", c0, c4, ax3], ["rb0_4"]),
            helper.make_node("Concat", ["rm1_5", "zcol1"], ["rm_l1"], axis=3),
            helper.make_node("Concat", ["zcol1", "rm0_4"], ["rm_r1"], axis=3),
            helper.make_node("Concat", ["rt1_5", "zcol1"], ["rt_l1"], axis=3),
            helper.make_node("Concat", ["zcol1", "rt0_4"], ["rt_r1"], axis=3),
            helper.make_node("Concat", ["rb1_5", "zcol1"], ["rb_l1"], axis=3),
            helper.make_node("Concat", ["zcol1", "rb0_4"], ["rb_r1"], axis=3),
            helper.make_node("Concat", ["rm_l1", "rb_l1", "zrow"], ["green"], axis=2),
            helper.make_node("Concat", ["rm_r1", "rb_r1", "zrow"], ["magenta"], axis=2),
            helper.make_node("Concat", ["zrow", "rt_l1", "rm_l1"], ["cyan"], axis=2),
            helper.make_node("Concat", ["zrow", "rt_r1", "rm_r1"], ["orange"], axis=2),
            helper.make_node("Or", ["rm_l1", "rm_r1"], ["edge_paint"]),
            helper.make_node("Not", ["edge_paint"], ["edge_black"]),
            helper.make_node("Or", ["rt_l1", "rt_r1"], ["top_paint"]),
            helper.make_node("Or", ["rb_l1", "rb_r1"], ["bottom_paint"]),
            helper.make_node("Or", ["top_paint", "bottom_paint"], ["mid_paint"]),
            helper.make_node("Not", ["mid_paint"], ["mid_black"]),
            helper.make_node("Concat", ["edge_black", "mid_black", "edge_black"], ["black"], axis=2),
            helper.make_node(
                "Concat",
                ["black", "zchan", "zchan", "green", "zchan", "zchan", "magenta", "orange", "cyan"],
                ["out9b"],
                axis=1,
            ),
            helper.make_node("Cast", ["out9b"], ["out9"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out9"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 1, H - GH, W - GW]),
        ]
    )
    return _make_model(nodes, inits, "task266_structural")


def validate_json(model: onnx.ModelProto) -> int:
    bad = 0
    for idx, ex in enumerate(_examples()):
        g = np.asarray(ex["input"], dtype=np.int64)
        expected = np.asarray(ex["output"], dtype=np.int64)
        ref = solve(g)
        if not np.array_equal(ref, expected):
            raise AssertionError(f"reference solver mismatch on example {idx}")
        pred_oh = _run_onnx(model, _grid_to_onehot(ex["input"]))
        pred = _onehot_to_grid(pred_oh)[: expected.shape[0], : expected.shape[1]]
        active = pred_oh[0, :, : expected.shape[0], : expected.shape[1]] > 0.0
        if not np.array_equal(pred, expected) or not np.all(active.sum(axis=0) == 1):
            bad += 1
    return bad


def _score_candidate(label: str, build: Callable[[], onnx.ModelProto]) -> tuple[int, float, onnx.ModelProto]:
    model = build()
    bad = validate_json(model)
    if bad:
        raise AssertionError(f"{label} failed {bad} examples")

    with tempfile.NamedTemporaryFile(suffix=f"_{TASK_ID}.onnx", dir=OUT_DIR, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        onnx.save(model, tmp_path)
        result = score_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not result["valid"]:
        raise AssertionError(f"{label} invalid: {result['error']}")
    assert result["cost"] is not None and result["score"] is not None
    print(
        f"{label}: nodes={len(model.graph.node)} memory={result['memory']} "
        f"params={result['params']} cost={result['cost']} score={result['score']:.6f}"
    )
    return int(result["cost"]), float(result["score"]), model


def main() -> None:
    candidates = [
        _score_candidate("conv", build_conv_model),
        _score_candidate("structural", build_structural_model),
    ]
    _cost, _score, best = min(candidates, key=lambda item: item[0])
    onnx.save(best, BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"nodes:   {len(best.graph.node)}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
