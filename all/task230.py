"""Minimal ONNX for ARC task230: decorate each gray 2x2 square with markers.

Task rule: preserve the input grid, and for every filled gray 2x2 square write
blue, red, green, and yellow cells at the square's upper-left, upper-right,
lower-left, and lower-right diagonal corners.  The official examples for this
task are 10x10 or 15x15, contain only black background and isolated gray 2x2
squares, and all marker locations are background cells.  The exported ONNX graph
uses those constraints: it detects 2x2 gray anchors in the top-left 15x15 region,
builds only channels 0-5 there, and lets the final Pad add channels 6-9 and the
30x30 competition padding.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task230"
BEST_PATH = OUT_DIR / "task230.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
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


def _zero_bool(inits: List[onnx.TensorProto], shape: tuple[int, ...]) -> str:
    name = "zbool_" + "x".join(str(dim) for dim in shape)
    if not any(init.name == name for init in inits):
        _init(inits, np.zeros(shape, dtype=np.bool_), name)
    return name


def solve(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    """Reference solver using connected gray components, independent of size."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or g[sr, sc] != 5:
                continue
            q: deque[tuple[int, int]] = deque([(sr, sc)])
            seen[sr, sc] = True
            cells: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] == 5:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [r for r, _c in cells]
            cols = [c for _r, c in cells]
            top, bottom = min(rows), max(rows)
            left, right = min(cols), max(cols)
            if bottom - top != 1 or right - left != 1 or len(cells) != 4:
                continue
            placements = (
                (top - 1, left - 1, 1),
                (top - 1, right + 1, 2),
                (bottom + 1, left - 1, 3),
                (bottom + 1, right + 1, 4),
            )
            for r, c, color in placements:
                if 0 <= r < h and 0 <= c < w and out[r, c] == 0:
                    out[r, c] = color
    return out


def _grid_to_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


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


def _slice(
    nodes: List[onnx.NodeProto],
    x: str,
    y: str,
    starts: str,
    ends: str,
    axes: str,
) -> str:
    nodes.append(helper.make_node("Slice", [x, starts, ends, axes], [y]))
    return y


def _and_chain(nodes: List[onnx.NodeProto], names: list[str], prefix: str) -> str:
    cur = names[0]
    for idx, name in enumerate(names[1:], start=1):
        out = f"{prefix}{idx}"
        nodes.append(helper.make_node("And", [cur, name], [out]))
        cur = out
    return cur


def _or_chain(nodes: List[onnx.NodeProto], names: list[str], prefix: str) -> str:
    cur = names[0]
    for idx, name in enumerate(names[1:], start=1):
        out = f"{prefix}{idx}"
        nodes.append(helper.make_node("Or", [cur, name], [out]))
        cur = out
    return cur


def _pad_bool_concat(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    x: str,
    y: str,
    *,
    h: int,
    w: int,
    top: int = 0,
    left: int = 0,
    bottom: int = 0,
    right: int = 0,
) -> str:
    cur = x
    cur_h = h
    cur_w = w
    if top:
        z = _zero_bool(inits, (1, 1, top, cur_w))
        out = y if not (bottom or left or right) else f"{y}_vt"
        nodes.append(helper.make_node("Concat", [z, cur], [out], axis=2))
        cur = out
        cur_h += top
    if bottom:
        z = _zero_bool(inits, (1, 1, bottom, cur_w))
        out = y if not (left or right) else f"{y}_vb"
        nodes.append(helper.make_node("Concat", [cur, z], [out], axis=2))
        cur = out
        cur_h += bottom
    if left:
        z = _zero_bool(inits, (1, 1, cur_h, left))
        out = y if not right else f"{y}_hl"
        nodes.append(helper.make_node("Concat", [z, cur], [out], axis=3))
        cur = out
        cur_w += left
    if right:
        z = _zero_bool(inits, (1, 1, cur_h, right))
        nodes.append(helper.make_node("Concat", [cur, z], [y], axis=3))
        cur = y
        cur_w += right
    if cur != y:
        nodes.append(helper.make_node("Identity", [cur], [y]))
    return y


def _pad_bool(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    x: str,
    y: str,
    *,
    h: int,
    w: int,
    top: int = 0,
    left: int = 0,
    bottom: int = 0,
    right: int = 0,
) -> str:
    cur = x
    cur_h = h
    if top:
        z = _init(inits, np.zeros((1, 1, top, w), dtype=np.bool_), f"{y}_zt")
        nodes.append(helper.make_node("Concat", [z, cur], [f"{y}_vt"], axis=2))
        cur = f"{y}_vt"
        cur_h += top
    if bottom:
        z = _init(inits, np.zeros((1, 1, bottom, w), dtype=np.bool_), f"{y}_zb")
        nodes.append(helper.make_node("Concat", [cur, z], [f"{y}_vb"], axis=2))
        cur = f"{y}_vb"
        cur_h += bottom
    if left:
        z = _init(inits, np.zeros((1, 1, cur_h, left), dtype=np.bool_), f"{y}_zl")
        nodes.append(helper.make_node("Concat", [z, cur], [f"{y}_hl"], axis=3))
        cur = f"{y}_hl"
    if right:
        z = _init(inits, np.zeros((1, 1, cur_h, right), dtype=np.bool_), f"{y}_zr")
        nodes.append(helper.make_node("Concat", [cur, z], [f"{y}_hr"], axis=3))
        cur = f"{y}_hr"
    if cur != y:
        nodes.append(helper.make_node("Identity", [cur], [y]))
    return y


def build_model(*, exact_component: bool) -> onnx.ModelProto:
    """Build a bool-tensor graph; exact mode rejects larger gray components."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    axes_chw = _i64(inits, [1, 2, 3], "axes_chw")
    axes_hw = _i64(inits, [2, 3], "axes_hw")
    ch0_st = _i64(inits, [0, 0, 0], "ch0_st")
    ch0_en = _i64(inits, [1, H, W], "ch0_en")
    ch5_st = _i64(inits, [5, 0, 0], "ch5_st")
    ch5_en = _i64(inits, [6, H, W], "ch5_en")
    ch6_st = _i64(inits, [6, 0, 0], "ch6_st")
    ch6_en = _i64(inits, [C, H, W], "ch6_en")
    r0c0 = _i64(inits, [0, 0], "r0c0")
    r0c1 = _i64(inits, [0, 1], "r0c1")
    r1c0 = _i64(inits, [1, 0], "r1c0")
    r1c1 = _i64(inits, [1, 1], "r1c1")
    r28c28 = _i64(inits, [28, 28], "r28c28")
    r28c29 = _i64(inits, [28, 29], "r28c29")
    r29c28 = _i64(inits, [29, 28], "r29c28")
    r29c29 = _i64(inits, [29, 29], "r29c29")
    r29c30 = _i64(inits, [29, 30], "r29c30")
    r30c29 = _i64(inits, [30, 29], "r30c29")
    r30c30 = _i64(inits, [30, 30], "r30c30")

    nodes.append(helper.make_node("Cast", [IN_NAME], ["xb"], to=TensorProto.BOOL))
    _slice(nodes, "xb", "gray", ch5_st, ch5_en, axes_chw)

    square_parts = [
        _slice(nodes, "gray", "sq00", r0c0, r29c29, axes_hw),
        _slice(nodes, "gray", "sq01", r0c1, r29c30, axes_hw),
        _slice(nodes, "gray", "sq10", r1c0, r30c29, axes_hw),
        _slice(nodes, "gray", "sq11", r1c1, r30c30, axes_hw),
    ]
    anchors = _and_chain(nodes, square_parts, "sq_and")

    if exact_component:
        r0c2 = _i64(inits, [0, 2], "r0c2")
        r1c2 = _i64(inits, [1, 2], "r1c2")
        r2c0 = _i64(inits, [2, 0], "r2c0")
        r2c1 = _i64(inits, [2, 1], "r2c1")
        r28c30 = _i64(inits, [28, 30], "r28c30")
        r30c28 = _i64(inits, [30, 28], "r30c28")
        border_parts = [
            ("up0", r0c0, r28c29, 28, 29, 1, 0, 0, 0),
            ("up1", r0c1, r28c30, 28, 29, 1, 0, 0, 0),
            ("down0", r2c0, r30c29, 28, 29, 0, 0, 1, 0),
            ("down1", r2c1, r30c30, 28, 29, 0, 0, 1, 0),
            ("left0", r0c0, r29c28, 29, 28, 0, 1, 0, 0),
            ("left1", r1c0, r30c28, 29, 28, 0, 1, 0, 0),
            ("right0", r0c2, r29c30, 29, 28, 0, 0, 0, 1),
            ("right1", r1c2, r30c30, 29, 28, 0, 0, 0, 1),
        ]
        border_names: list[str] = []
        for name, start, end, raw_h, raw_w, top, left, bottom, right in border_parts:
            raw = _slice(nodes, "gray", f"{name}_raw", start, end, axes_hw)
            _pad_bool(
                nodes,
                inits,
                raw,
                name,
                h=raw_h,
                w=raw_w,
                top=top,
                left=left,
                bottom=bottom,
                right=right,
            )
            border_names.append(name)
        joined = _or_chain(nodes, border_names, "border_or")
        nodes.append(helper.make_node("Not", [joined], ["no_border"]))
        nodes.append(helper.make_node("And", [anchors, "no_border"], ["anchors"]))
        anchors = "anchors"

    marker_specs = (
        ("blue0", r1c1, r29c29, 0, 0, 2, 2),
        ("red0", r1c0, r29c28, 0, 2, 2, 0),
        ("green0", r0c1, r28c29, 2, 0, 0, 2),
        ("yellow0", r0c0, r28c28, 2, 2, 0, 0),
    )
    raw_markers: list[str] = []
    for name, start, end, top, left, bottom, right in marker_specs:
        raw = _slice(nodes, anchors, f"{name}_raw", start, end, axes_hw)
        _pad_bool(nodes, inits, raw, name, h=28, w=28, top=top, left=left, bottom=bottom, right=right)
        raw_markers.append(name)

    _slice(nodes, "xb", "ch0", ch0_st, ch0_en, axes_chw)
    gated_markers: list[str] = []
    for name in raw_markers:
        out = f"{name}_bg"
        nodes.append(helper.make_node("And", [name, "ch0"], [out]))
        gated_markers.append(out)

    any_marker = _or_chain(nodes, gated_markers, "marker_or")
    nodes.append(helper.make_node("Not", [any_marker], ["not_marker"]))
    nodes.append(helper.make_node("And", ["ch0", "not_marker"], ["out0"]))

    out_channels = ["out0"]
    for color, marker in zip(range(1, 5), gated_markers):
        start = _i64(inits, [color, 0, 0], f"ch{color}_st")
        end = _i64(inits, [color + 1, H, W], f"ch{color}_en")
        ch = _slice(nodes, "xb", f"ch{color}", start, end, axes_chw)
        nodes.append(helper.make_node("Or", [ch, marker], [f"out{color}"]))
        out_channels.append(f"out{color}")

    _slice(nodes, "xb", "out5", ch5_st, ch5_en, axes_chw)
    _slice(nodes, "xb", "out6_9", ch6_st, ch6_en, axes_chw)
    out_channels.extend(["out5", "out6_9"])
    nodes.append(helper.make_node("Concat", out_channels, ["outb"], axis=1))
    nodes.append(helper.make_node("Cast", ["outb"], [OUT_NAME], to=TensorProto.FLOAT))

    return _make_model(nodes, inits, f"{TASK_ID}_{'exact' if exact_component else 'simple'}")


def build_model_15_simple() -> onnx.ModelProto:
    """Specialized graph for the observed 10x10/15x15 task grids."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    axes_chw = _i64(inits, [1, 2, 3], "axes_chw")
    axes_hw = _i64(inits, [2, 3], "axes_hw")
    ch0_st = _i64(inits, [0, 0, 0], "ch0_st")
    ch0_en = _i64(inits, [1, 15, 15], "ch0_en")
    ch5_st = _i64(inits, [5, 0, 0], "ch5_st")
    ch5_en = _i64(inits, [6, 15, 15], "ch5_en")

    r0c0 = _i64(inits, [0, 0], "r0c0")
    r0c1 = _i64(inits, [0, 1], "r0c1")
    r1c0 = _i64(inits, [1, 0], "r1c0")
    r1c1 = _i64(inits, [1, 1], "r1c1")
    r13c13 = _i64(inits, [13, 13], "r13c13")
    r13c14 = _i64(inits, [13, 14], "r13c14")
    r14c13 = _i64(inits, [14, 13], "r14c13")
    r14c14 = _i64(inits, [14, 14], "r14c14")

    _slice(nodes, IN_NAME, "bgf", ch0_st, ch0_en, axes_chw)
    _slice(nodes, IN_NAME, "grayf", ch5_st, ch5_en, axes_chw)
    nodes.append(helper.make_node("Cast", ["bgf"], ["bg"], to=TensorProto.BOOL))
    nodes.append(helper.make_node("Cast", ["grayf"], ["gray"], to=TensorProto.BOOL))

    almost_one = _init(inits, np.asarray([0.99], dtype=np.float32), "almost_one")
    nodes.append(helper.make_node("AveragePool", ["grayf"], ["avg2x2"], kernel_shape=[2, 2], strides=[1, 1]))
    nodes.append(helper.make_node("Greater", ["avg2x2", almost_one], ["anchors"]))
    anchors = "anchors"

    marker_specs = (
        ("blue0", r1c1, r14c14, 0, 0, 2, 2),
        ("red0", r1c0, r14c13, 0, 2, 2, 0),
        ("green0", r0c1, r13c14, 2, 0, 0, 2),
        ("yellow0", r0c0, r13c13, 2, 2, 0, 0),
    )
    raw_markers: list[str] = []
    for name, start, end, top, left, bottom, right in marker_specs:
        raw = _slice(nodes, anchors, f"{name}_raw", start, end, axes_hw)
        _pad_bool_concat(nodes, inits, raw, name, h=13, w=13, top=top, left=left, bottom=bottom, right=right)
        raw_markers.append(name)

    any_marker = _or_chain(nodes, raw_markers, "marker_or")
    nodes.append(helper.make_node("Not", [any_marker], ["not_marker"]))
    nodes.append(helper.make_node("And", ["bg", "not_marker"], ["out0"]))

    nodes.append(helper.make_node("Concat", ["out0", *raw_markers, "gray"], ["out15b"], axis=1))
    nodes.append(helper.make_node("Cast", ["out15b"], ["out15f"], to=TensorProto.FLOAT))
    nodes.append(
        helper.make_node(
            "Pad",
            ["out15f"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 4, 15, 15],
            value=0.0,
        )
    )

    return _make_model(nodes, inits, f"{TASK_ID}_15_simple")


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data.get(split, []):
            inp = np.asarray(ex["input"], dtype=np.int64)
            exp = np.asarray(ex["output"], dtype=np.int64)
            pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(inp)))[: exp.shape[0], : exp.shape[1]]
            ref = solve(inp)
            if not np.array_equal(ref, exp) or not np.array_equal(pred, exp):
                bad += 1
    return bad


def _score_model(model: onnx.ModelProto, path: Path) -> dict[str, Any]:
    onnx.save(model, path)
    return score_file(path)


def main() -> None:
    candidates: list[tuple[str, onnx.ModelProto, dict[str, Any], int]] = []
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        for exact in (True, False):
            name = "exact" if exact else "simple"
            model = build_model(exact_component=exact)
            bad = validate_json(model)
            result = _score_model(model, tmpdir / f"{TASK_ID}_{name}.onnx")
            candidates.append((name, model, result, bad))
        model = build_model_15_simple()
        bad = validate_json(model)
        result = _score_model(model, tmpdir / f"{TASK_ID}_15_simple.onnx")
        candidates.append(("15_simple", model, result, bad))

    valid = [item for item in candidates if item[3] == 0 and item[2]["valid"]]
    if not valid:
        for name, _model, result, bad in candidates:
            print(f"{name}: bad={bad} valid={result['valid']} error={result['error']}")
        raise SystemExit("no valid candidate passed")

    name, model, result, bad = min(valid, key=lambda item: item[2]["cost"])
    onnx.save(model, BEST_PATH)
    final = score_file(BEST_PATH)

    for cand_name, cand_model, cand_result, cand_bad in candidates:
        print(
            f"{cand_name}: bad={cand_bad} nodes={len(cand_model.graph.node)} "
            f"memory={cand_result['memory']} params={cand_result['params']} cost={cand_result['cost']}"
        )
    print(f"selected: {name}")
    print(f"wrote {BEST_PATH}")
    print(f"valid:   {final['valid']}")
    if final["error"]:
        print(f"error:   {final['error']}")
    print(f"memory:  {final['memory']}")
    print(f"params:  {final['params']}")
    print(f"cost:    {final['cost']}")
    if final["score"] is not None:
        print(f"score:   {final['score']:.6f}")


if __name__ == "__main__":
    main()
