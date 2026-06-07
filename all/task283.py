"""Minimal ONNX for ARC task283: recolor solid gray rectangles by layer.

Task rule: each 10x10 input has one or more filled gray rectangles on a black
background.  Keep each rectangle in place, recoloring its four corners blue
(1), its non-corner border yellow (4), and its interior red (2).  Background
cells remain black.

ONNX approach: crop the gray channel in the active 10x10 area and use one
cross-shaped convolution with center weight 5 to encode center+neighbor state.
The only observed codes are 0/1/2 for black, 7 for corners, 8 for borders, and
9 for interiors; three nested thresholds plus Xor recover the output masks.
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

TASK_ID = "task283"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task283.onnx"
DATA_PATH = ROOT / "data" / "task283.json"

C = 10
H = W = 30
G = 10
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
    """Reference solver for the rectangle layer recoloring rule."""
    g = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(g)
    seen = np.zeros_like(g, dtype=bool)
    for r, c in zip(*np.where(g == 5)):
        if seen[r, c]:
            continue
        stack = [(int(r), int(c))]
        seen[r, c] = True
        cells: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            cells.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and g[nr, nc] == 5 and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        rs = [cell[0] for cell in cells]
        cs = [cell[1] for cell in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == 5)
        out[r0 : r1 + 1, c0 : c1 + 1] = 4
        if r1 - r0 > 1 and c1 - c0 > 1:
            out[r0 + 1 : r1, c0 + 1 : c1] = 2
        out[r0, c0] = out[r0, c1] = out[r1, c0] = out[r1, c1] = 1
    return out


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def _gray_crop(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto]) -> str:
    starts = _i64(inits, [0, 5, 0, 0], "starts")
    ends = _i64(inits, [1, 6, G, G], "ends")
    nodes.append(helper.make_node("Slice", [IN_NAME, starts, ends], ["gray"]))
    return "gray"


def _finish(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    grayb: str,
    red: str,
    blue: str,
    name: str,
) -> onnx.ModelProto:
    nodes.extend(
        [
            helper.make_node("Not", [grayb], ["black"]),
            helper.make_node("Or", [blue, red], ["blue_or_red"]),
            helper.make_node("Not", ["blue_or_red"], ["not_blue_or_red"]),
            helper.make_node("And", [grayb, "not_blue_or_red"], ["yellow"]),
            helper.make_node("And", [grayb, "black"], ["zero"]),
            helper.make_node("Concat", ["black", blue, red, "zero", "yellow"], ["out5b"], axis=1),
            helper.make_node("Cast", ["out5b"], ["out5"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out5"], [OUT_NAME], pads=[0, 0, 0, 0, 0, C - 5, H - G, W - G]),
        ]
    )
    return _make_model(nodes, inits, name)


def build_conv_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    gray = _gray_crop(nodes, inits)
    kernel = np.asarray([[[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]], dtype=np.float32)
    weights = _f32(inits, kernel, "cross")
    one_half = _f32(inits, [1.5], "one_half")
    two_half = _f32(inits, [2.5], "two_half")
    three_half = _f32(inits, [3.5], "three_half")
    nodes.extend(
        [
            helper.make_node("Conv", [gray, weights], ["neighbor_count"], pads=[1, 1, 1, 1]),
            helper.make_node("Greater", ["neighbor_count", one_half], ["gt1"]),
            helper.make_node("Less", ["neighbor_count", two_half], ["lt3"]),
            helper.make_node("And", ["gt1", "lt3"], ["count2"]),
            helper.make_node("Greater", ["neighbor_count", three_half], ["count4"]),
            helper.make_node("Cast", [gray], ["grayb0"], to=TensorProto.BOOL),
            helper.make_node("And", ["grayb0", "count2"], ["blue"]),
            helper.make_node("And", ["grayb0", "count4"], ["red"]),
        ]
    )
    return _finish(nodes, inits, "grayb0", "red", "blue", "task283_conv")


def build_encoded_conv_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    gray = _gray_crop(nodes, inits)
    kernel = np.asarray([[[[0, 1, 0], [1, 5, 1], [0, 1, 0]]]], dtype=np.float32)
    weights = _f32(inits, kernel, "encoded_cross")
    black_max = _f32(inits, [4.5], "black_max")
    corner_lo = _f32(inits, [6.5], "corner_lo")
    corner_hi = _f32(inits, [7.5], "corner_hi")
    red_lo = _f32(inits, [8.5], "red_lo")
    nodes.extend(
        [
            helper.make_node("Conv", [gray, weights], ["code"], pads=[1, 1, 1, 1]),
            helper.make_node("Less", ["code", black_max], ["black"]),
            helper.make_node("Greater", ["code", corner_lo], ["gt_corner_lo"]),
            helper.make_node("Less", ["code", corner_hi], ["lt_corner_hi"]),
            helper.make_node("And", ["gt_corner_lo", "lt_corner_hi"], ["blue"]),
            helper.make_node("Greater", ["code", red_lo], ["red"]),
            helper.make_node("Or", ["blue", "red"], ["blue_or_red"]),
            helper.make_node("Or", ["black", "blue_or_red"], ["not_yellow"]),
            helper.make_node("Not", ["not_yellow"], ["yellow"]),
            helper.make_node("And", ["blue", "red"], ["zero"]),
            helper.make_node("Concat", ["black", "blue", "red", "zero", "yellow"], ["out5b"], axis=1),
            helper.make_node("Cast", ["out5b"], ["out5"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out5"], [OUT_NAME], pads=[0, 0, 0, 0, 0, C - 5, H - G, W - G]),
        ]
    )
    return _make_model(nodes, inits, "task283_encoded_conv")


def build_threshold_xor_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    gray = _gray_crop(nodes, inits)
    kernel = np.asarray([[[[0, 1, 0], [1, 5, 1], [0, 1, 0]]]], dtype=np.float32)
    weights = _f32(inits, kernel, "encoded_cross")
    gray_min = _f32(inits, [6.5], "gray_min")
    edge_min = _f32(inits, [7.5], "edge_min")
    interior_min = _f32(inits, [8.5], "interior_min")
    nodes.extend(
        [
            helper.make_node("Conv", [gray, weights], ["code"], pads=[1, 1, 1, 1]),
            helper.make_node("Greater", ["code", gray_min], ["gray_or_color"]),
            helper.make_node("Greater", ["code", edge_min], ["edge_or_red"]),
            helper.make_node("Greater", ["code", interior_min], ["red"]),
            helper.make_node("Not", ["gray_or_color"], ["black"]),
            helper.make_node("Xor", ["gray_or_color", "edge_or_red"], ["blue"]),
            helper.make_node("Xor", ["edge_or_red", "red"], ["yellow"]),
            helper.make_node("And", ["black", "red"], ["zero"]),
            helper.make_node("Concat", ["black", "blue", "red", "zero", "yellow"], ["out5b"], axis=1),
            helper.make_node("Cast", ["out5b"], ["out5"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out5"], [OUT_NAME], pads=[0, 0, 0, 0, 0, C - 5, H - G, W - G]),
        ]
    )
    return _make_model(nodes, inits, "task283_threshold_xor")


def build_shift_model() -> onnx.ModelProto:
    """Alternative candidate using bool shifts instead of convolution."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    gray = _gray_crop(nodes, inits)

    row0 = _i64(inits, [0, 0, 0, 0], "row0")
    row1 = _i64(inits, [1, 1, G - 1, G], "row1")
    row2 = _i64(inits, [0, 0, 1, 0], "row2")
    row3 = _i64(inits, [1, 1, G, G], "row3")
    col0 = _i64(inits, [0, 0, 0, 0], "col0")
    col1 = _i64(inits, [1, 1, G, G - 1], "col1")
    col2 = _i64(inits, [0, 0, 0, 1], "col2")
    col3 = _i64(inits, [1, 1, G, G], "col3")
    axes = _i64(inits, [0, 1, 2, 3], "axes")

    nodes.extend(
        [
            helper.make_node("Slice", [gray, row0, row1, axes], ["up_src"]),
            helper.make_node("Pad", ["up_src"], ["upf"], pads=[0, 0, 1, 0, 0, 0, 0, 0]),
            helper.make_node("Cast", ["upf"], ["up"], to=TensorProto.BOOL),
            helper.make_node("Slice", [gray, row2, row3, axes], ["down_src"]),
            helper.make_node("Pad", ["down_src"], ["downf"], pads=[0, 0, 0, 0, 0, 0, 1, 0]),
            helper.make_node("Cast", ["downf"], ["down"], to=TensorProto.BOOL),
            helper.make_node("Slice", [gray, col0, col1, axes], ["left_src"]),
            helper.make_node("Pad", ["left_src"], ["leftf"], pads=[0, 0, 0, 1, 0, 0, 0, 0]),
            helper.make_node("Cast", ["leftf"], ["left"], to=TensorProto.BOOL),
            helper.make_node("Slice", [gray, col2, col3, axes], ["right_src"]),
            helper.make_node("Pad", ["right_src"], ["rightf"], pads=[0, 0, 0, 0, 0, 0, 0, 1]),
            helper.make_node("Cast", ["rightf"], ["right"], to=TensorProto.BOOL),
            helper.make_node("And", ["up", "down"], ["v_both"]),
            helper.make_node("And", ["left", "right"], ["h_both"]),
            helper.make_node("And", ["v_both", "h_both"], ["red"]),
            helper.make_node("Or", ["up", "down"], ["v_any"]),
            helper.make_node("Or", ["left", "right"], ["h_any"]),
            helper.make_node("Not", ["v_both"], ["not_v_both"]),
            helper.make_node("Not", ["h_both"], ["not_h_both"]),
            helper.make_node("And", ["v_any", "not_v_both"], ["v_one"]),
            helper.make_node("And", ["h_any", "not_h_both"], ["h_one"]),
            helper.make_node("And", ["v_one", "h_one"], ["blue0"]),
            helper.make_node("And", ["grayb0", "blue0"], ["blue"]),
            helper.make_node("And", ["grayb0", "red"], ["red_gray"]),
        ]
    )
    nodes.insert(1, helper.make_node("Cast", [gray], ["grayb0"], to=TensorProto.BOOL))
    return _finish(nodes, inits, "grayb0", "red_gray", "blue", "task283_shift")


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            g = np.asarray(ex["input"], dtype=np.int64)
            expected = np.asarray(ex["output"], dtype=np.int64)
            ref = solve(g)
            if not np.array_equal(ref, expected):
                raise AssertionError(f"reference solver mismatch on {split}[{idx}]")
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
        _score_candidate("threshold-xor", build_threshold_xor_model),
        _score_candidate("encoded-conv", build_encoded_conv_model),
        _score_candidate("conv", build_conv_model),
        _score_candidate("shift", build_shift_model),
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
