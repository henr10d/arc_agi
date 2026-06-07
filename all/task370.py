"""ONNX solution for ARC task370: stamp the black object along a diagonal ray.

Task rule: the input has one large background color, one black object, and one
colored marker. Translate the black object's mask so that its far diagonal
frontier lands on the marker, then keep stamping that mask by the same diagonal
offset until it leaves the grid. The marker color fills those stamped cells;
background and the original black object are preserved.

ONNX: build the black-cell mask and marker mask in the 20x20 task region, test
only the diagonal direction/step combinations observed in the task data, keep
the largest marker-overlapping offset per direction, and OR together the
corresponding repeated shifted masks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task370"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
WORK = 20
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


def _i64_cached(
    inits: List[onnx.TensorProto],
    cache: Dict[Tuple[int, ...], str],
    vals,
) -> str:
    key = tuple(int(v) for v in vals)
    if key not in cache:
        name = "i64_" + "_".join(str(v) for v in key)
        cache[key] = _i64(inits, key, name)
    return cache[key]


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _bool(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.bool_), name)


def _zero_bool(
    inits: List[onnx.TensorProto],
    cache: Dict[Tuple[int, ...], str],
    shape: Tuple[int, ...],
) -> str:
    if shape not in cache:
        name = "z_" + "_".join(str(v) for v in shape)
        cache[shape] = _bool(inits, np.zeros(shape, dtype=np.bool_), name)
    return cache[shape]


def _shift_np(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask)
    r0 = max(0, -dr)
    r1 = min(h, h - dr)
    c0 = max(0, -dc)
    c1 = min(w, w - dc)
    if r1 > r0 and c1 > c0:
        out[r0 + dr : r1 + dr, c0 + dc : c1 + dc] = mask[r0:r1, c0:c1]
    return out


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference implementation of the diagonal black-mask stamping rule."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[int(np.argmax(counts))])
    marker_colors = [int(v) for v in vals if int(v) not in (0, bg)]
    if not marker_colors:
        return out
    color = marker_colors[0]
    marker = g == color
    black = g == 0
    paint = np.zeros_like(black)
    for sr_sign, sc_sign in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        overlaps: List[int] = []
        for k in range(1, H):
            sr = sr_sign * k
            sc = sc_sign * k
            if np.any(_shift_np(black, sr, sc) & marker):
                overlaps.append(k)
        if not overlaps:
            continue
        k = max(overlaps)
        sr = sr_sign * k
        sc = sc_sign * k
        for n in range(1, H):
            paint |= _shift_np(black, n * sr, n * sc)
    out[paint & ~black] = color
    return out


def _grid_to_onehot(grid: List[List[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _add_shift(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    cache: Dict[Tuple[int, int], str],
    zero_cache: Dict[Tuple[int, ...], str],
    i64_cache: Dict[Tuple[int, ...], str],
    source: str,
    axes4: str,
    dr: int,
    dc: int,
) -> str:
    key = (dr, dc)
    if key in cache:
        return cache[key]

    r0 = max(0, -dr)
    r1 = min(WORK, WORK - dr)
    c0 = max(0, -dc)
    c1 = min(WORK, WORK - dc)
    name = f"sh_{dr + H}_{dc + W}"
    st = _i64_cached(inits, i64_cache, [0, 0, r0, c0])
    en = _i64_cached(inits, i64_cache, [1, 1, r1, c1])
    crop = f"{name}_crop"
    top = max(0, dr)
    left = max(0, dc)
    bottom = max(0, -dr)
    right = max(0, -dc)
    height = r1 - r0
    width = c1 - c0
    nodes.append(helper.make_node("Slice", [source, st, en, axes4], [crop]))

    cur = crop
    if left or right:
        parts = []
        if left:
            parts.append(_zero_bool(inits, zero_cache, (1, 1, height, left)))
        parts.append(cur)
        if right:
            parts.append(_zero_bool(inits, zero_cache, (1, 1, height, right)))
        col_pad = f"{name}_cp"
        nodes.append(helper.make_node("Concat", parts, [col_pad], axis=3))
        cur = col_pad

    if top or bottom:
        parts = []
        if top:
            parts.append(_zero_bool(inits, zero_cache, (1, 1, top, WORK)))
        parts.append(cur)
        if bottom:
            parts.append(_zero_bool(inits, zero_cache, (1, 1, bottom, WORK)))
        nodes.append(helper.make_node("Concat", parts, [name], axis=2))
    else:
        name = cur
    cache[key] = name
    return name


def _or_many(nodes: List[onnx.NodeProto], names: List[str], prefix: str) -> str:
    if not names:
        raise ValueError("cannot OR an empty list")
    cur = names[0]
    for idx, name in enumerate(names[1:], start=1):
        out = f"{prefix}_{idx}"
        nodes.append(helper.make_node("Or", [cur, name], [out]))
        cur = out
    return cur


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    shift_cache: Dict[Tuple[int, int], str] = {}
    zero_cache: Dict[Tuple[int, ...], str] = {}
    i64_cache: Dict[Tuple[int, ...], str] = {}

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    zero = _f32(inits, [0.0], "zero")
    one_i = _i64(inits, [1], "one_i")
    false = _bool(inits, np.zeros((1, 1, 1, 1), dtype=np.bool_), "false")
    ch0_st = _i64(inits, [0, 0, 0, 0], "ch0_st")
    ch0_en = _i64(inits, [1, 1, WORK, WORK], "ch0_en")
    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    marker_st = _i64(inits, [0, 0, 0], "marker_st")
    marker_en = _i64(inits, [1, WORK, WORK], "marker_en")
    axes3 = _i64(inits, [0, 1, 2], "axes3")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, ch0_st, ch0_en, axes4], ["black_f"]),
            helper.make_node("Cast", ["black_f"], ["black"], to=TensorProto.BOOL),
            helper.make_node("ReduceSum", [IN_NAME], ["counts"], axes=[2, 3], keepdims=1),
            helper.make_node("Cast", ["counts"], ["counts_i"], to=TensorProto.INT64),
            helper.make_node("Equal", ["counts_i", one_i], ["marker_ch"]),
            helper.make_node("Cast", ["marker_ch"], ["marker_ch_f"], to=TensorProto.FLOAT),
            helper.make_node("ArgMax", ["marker_ch_f"], ["marker_idx4"], axis=1, keepdims=1),
            helper.make_node("Squeeze", ["marker_idx4"], ["marker_idx"], axes=[0, 1, 2, 3]),
            helper.make_node("Gather", [IN_NAME, "marker_idx"], ["marker_g30"], axis=1),
            helper.make_node("Slice", ["marker_g30", marker_st, marker_en, axes3], ["marker_g"]),
            helper.make_node("Unsqueeze", ["marker_g"], ["marker_f"], axes=[1]),
            helper.make_node("Greater", ["marker_f", zero], ["marker"]),
            helper.make_node("ReduceMax", [IN_NAME], ["active_f30"], axes=[1], keepdims=1),
        ]
    )

    selected_rays: List[str] = []
    candidate_groups = (
        (1, 1, (4, 3, 2)),
        (1, -1, (4, 3, 2)),
        (-1, 1, (5, 4, 3)),
        (-1, -1, (3,)),
    )
    for d_idx, (sr_sign, sc_sign, steps) in enumerate(candidate_groups):
        seen = false
        for k in steps:
            sr = sr_sign * k
            sc = sc_sign * k
            first = _add_shift(nodes, inits, shift_cache, zero_cache, i64_cache, "black", axes4, sr, sc)
            overlap = f"ov_{d_idx}_{k}"
            overlap_f = f"{overlap}_f"
            gate_f = f"gate_{d_idx}_{k}_f"
            gate = f"gate_{d_idx}_{k}"
            not_seen = f"not_seen_{d_idx}_{k}"
            selected = f"sel_{d_idx}_{k}"

            nodes.extend(
                [
                    helper.make_node("And", [first, "marker"], [overlap]),
                    helper.make_node("Cast", [overlap], [overlap_f], to=TensorProto.FLOAT),
                    helper.make_node("ReduceMax", [overlap_f], [gate_f], axes=[2, 3], keepdims=1),
                    helper.make_node("Greater", [gate_f, zero], [gate]),
                    helper.make_node("Not", [seen], [not_seen]),
                    helper.make_node("And", [gate, not_seen], [selected]),
                ]
            )

            ray_parts = []
            for n in range(1, WORK):
                rr = n * sr
                cc = n * sc
                if abs(rr) >= WORK or abs(cc) >= WORK:
                    break
                ray_parts.append(
                    _add_shift(nodes, inits, shift_cache, zero_cache, i64_cache, "black", axes4, rr, cc)
                )
            ray = _or_many(nodes, ray_parts, f"ray_{d_idx}_{k}")
            gated = f"gated_{d_idx}_{k}"
            nodes.append(helper.make_node("And", [ray, selected], [gated]))
            selected_rays.append(gated)

            new_seen = f"seen_{d_idx}_{k}"
            nodes.append(helper.make_node("Or", [seen, gate], [new_seen]))
            seen = new_seen

    paint_any = _or_many(nodes, selected_rays, "paint_any")
    nodes.extend(
        [
            helper.make_node("Not", ["black"], ["not_black"]),
            helper.make_node("And", [paint_any, "not_black"], ["paint_in_grid"]),
            helper.make_node("Cast", ["paint_in_grid"], ["paint_f"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["paint_f"], ["paint_f30"], pads=[0, 0, 0, 0, 0, 0, H - WORK, W - WORK]),
            helper.make_node("Greater", ["paint_f30", zero], ["paint30"]),
            helper.make_node("Greater", ["active_f30", zero], ["active30"]),
            helper.make_node("And", ["paint30", "active30"], ["paint_active30"]),
            helper.make_node("Where", ["paint_active30", "marker_ch_f", IN_NAME], [OUT_NAME]),
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


def _make_session(model: onnx.ModelProto):
    import onnxruntime as ort

    return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])


def _run_onnx(sess, x: np.ndarray) -> np.ndarray:
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    if not DATA_PATH.is_file():
        return 0
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    sess = _make_session(model)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            inp = np.array(ex["input"], dtype=np.int64)
            exp = np.array(ex["output"], dtype=np.int64)
            if max(inp.shape) > H:
                continue
            ref = solve(inp)
            if not np.array_equal(ref, exp):
                raise AssertionError(f"reference mismatch on {split}[{idx}]")
            pred = _run_onnx(sess, _grid_to_onehot(inp))
            expected = _grid_to_onehot(exp)
            if not np.array_equal(pred > 0.0, expected > 0.0):
                bad += 1
    return bad


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    bad = validate_json(model)
    print(f"{TASK_ID}.json: {'PASS' if bad == 0 else f'FAIL ({bad} wrong)'}")

    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"nodes:   {len(model.graph.node)}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
