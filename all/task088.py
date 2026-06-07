"""Minimal ONNX for NeuroGolf ARC task088: crop pattern, recolor to marker.

Task rule: each grid has background (0), one pattern color (more cells), and
corner marker cells of another color (fewer cells, usually four corners around
the pattern).  Output is the tight crop around the pattern cells, expanded
one cell inward from the marker frame (crop excludes marker pixels but keeps
the interior frame width/height).  Pattern cells become the marker color;
other cells inside the crop are 0.

ONNX: slice a compact crop, infer pattern/marker channels from per-channel counts,
compute pattern and marker bboxes, apply the marker inset rule, gather up to
OUT×OUT output cells from the pattern mask, build marker one-hot on a small
canvas, then pad to 30×30.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task088"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task088.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
IR_VERSION = 10

TOY_INPUT = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 2, 2, 0, 0],
    [0, 0, 2, 0, 2, 0, 0],
    [0, 0, 2, 2, 2, 0, 0],
    [0, 4, 0, 0, 0, 4, 0],
    [0, 0, 0, 0, 0, 0, 0],
]
TOY_OUTPUT = [
    [4, 4, 4],
    [4, 0, 4],
    [4, 4, 4],
]


def solve(grid: list[list[int]] | np.ndarray) -> list[list[int]]:
    """Reference solver used for local checks."""
    g = np.asarray(grid, dtype=np.int64)
    colors = [int(c) for c in np.unique(g) if c != 0]
    if len(colors) != 2:
        return []
    c0, c1 = colors
    pattern_c = c0 if (g == c0).sum() >= (g == c1).sum() else c1
    marker_c = c1 if pattern_c == c0 else c0
    py, px = np.where(g == pattern_c)
    my, mx = np.where(g == marker_c)
    pr0, pr1 = int(py.min()), int(py.max())
    pc0, pc1 = int(px.min()), int(px.max())
    mr0, mr1 = int(my.min()), int(my.max())
    mc0, mc1 = int(mx.min()), int(mx.max())
    r0 = mr0 + 1 if mr0 < pr0 else pr0
    r1 = mr1 - 1 if mr1 > pr1 else pr1
    c0 = mc0 + 1 if mc0 < pc0 else pc0
    c1 = mc1 - 1 if mc1 > pc1 else pc1
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.int64)
    for y, x in zip(py, px):
        out[y - r0, x - c0] = marker_c
    return out.tolist()


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            color = int(val)
            if 0 <= color < C:
                out[0, color, r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _make_model(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    *,
    opset: int,
    graph_name: str,
) -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(nodes, graph_name, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _bbox_from_mask(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    mask_f: str,
    *,
    prefix: str,
) -> tuple[str, str, str, str]:
    row_has = f"{prefix}_row_has"
    col_has = f"{prefix}_col_has"
    nodes.extend(
        [
            helper.make_node("ReduceMax", [mask_f], [row_has], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", [mask_f], [col_has], axes=[2], keepdims=1),
            helper.make_node("ArgMax", [row_has], [f"{prefix}_rmin"], axis=2, keepdims=1),
            helper.make_node("ArgMax", [col_has], [f"{prefix}_cmin"], axis=3, keepdims=1),
            helper.make_node("Gather", [row_has, "rev_h"], [f"{prefix}_row_rev"], axis=2),
            helper.make_node("Gather", [col_has, "rev_w"], [f"{prefix}_col_rev"], axis=3),
            helper.make_node("ArgMax", [f"{prefix}_row_rev"], [f"{prefix}_rrev"], axis=2, keepdims=1),
            helper.make_node("ArgMax", [f"{prefix}_col_rev"], [f"{prefix}_crev"], axis=3, keepdims=1),
            helper.make_node("Sub", ["last_h", f"{prefix}_rrev"], [f"{prefix}_rmax"]),
            helper.make_node("Sub", ["last_w", f"{prefix}_crev"], [f"{prefix}_cmax"]),
        ]
    )
    return f"{prefix}_rmin", f"{prefix}_rmax", f"{prefix}_cmin", f"{prefix}_cmax"


def _inset_bbox(
    nodes: list[onnx.NodeProto],
    pr0: str,
    pr1: str,
    pc0: str,
    pc1: str,
    mr0: str,
    mr1: str,
    mc0: str,
    mc1: str,
) -> tuple[str, str, str, str]:
    nodes.extend(
        [
            helper.make_node("Less", [mr0, pr0], ["m_above"]),
            helper.make_node("Greater", [mr1, pr1], ["m_below"]),
            helper.make_node("Less", [mc0, pc0], ["m_left"]),
            helper.make_node("Greater", [mc1, pc1], ["m_right"]),
            helper.make_node("Add", [mr0, "one_i"], ["mr0p1"]),
            helper.make_node("Sub", [mr1, "one_i"], ["mr1m1"]),
            helper.make_node("Add", [mc0, "one_i"], ["mc0p1"]),
            helper.make_node("Sub", [mc1, "one_i"], ["mc1m1"]),
            helper.make_node("Where", ["m_above", "mr0p1", pr0], ["r0"]),
            helper.make_node("Where", ["m_below", "mr1m1", pr1], ["r1"]),
            helper.make_node("Where", ["m_left", "mc0p1", pc0], ["c0"]),
            helper.make_node("Where", ["m_right", "mc1m1", pc1], ["c1"]),
        ]
    )
    return "r0", "r1", "c0", "c1"


def build_model(
    *,
    grid: int = 24,
    grid_h: int | None = None,
    grid_w: int | None = None,
    out_size: int = 10,
    opset: int = 10,
) -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    grid_h = grid if grid_h is None else grid_h
    grid_w = grid if grid_w is None else grid_w

    _f32(inits, [0.0], "zero")
    _f32(inits, [-1.0], "neg_one")
    _i64(inits, [0, 1, 2, 3], "axes4")
    _i64(inits, [0, 0, 0, 0], "crop_st")
    _i64(inits, [1, 1, grid_h, grid_w], "crop_en")
    _i64(inits, np.arange(C), "colors")
    _i64(inits, [1], "one_i")
    _i64(inits, [1], "ch_idx_shape")
    _i64(inits, [grid_h - 1], "last_h")
    _i64(inits, [grid_w - 1], "last_w")
    _i64(inits, np.arange(grid_h - 1, -1, -1), "rev_h")
    _i64(inits, np.arange(grid_w - 1, -1, -1), "rev_w")
    _init(inits, np.asarray([grid_w], dtype=np.int32), "grid_w_i32")
    if opset >= 11:
        _i64(inits, [0, 0, 0, 0, 0, 0, H - out_size, W - out_size], "pad30")
    _init(inits, np.arange(out_size, dtype=np.int32).reshape(1, 1, out_size, 1), "out_rows")
    _init(inits, np.arange(out_size, dtype=np.int32).reshape(1, 1, 1, out_size), "out_cols")

    nodes.extend(
        [
            helper.make_node("ReduceSum", [IN_NAME], ["counts"], axes=[0, 2, 3], keepdims=0),
            helper.make_node("ArgMax", ["counts"], ["bg_idx"], axis=0, keepdims=0),
            helper.make_node("Equal", ["colors", "bg_idx"], ["is_bg"]),
            helper.make_node("Where", ["is_bg", "neg_one", "counts"], ["pat_scores"]),
            helper.make_node("ArgMax", ["pat_scores"], ["pat_idx"], axis=0, keepdims=0),
            helper.make_node("Equal", ["colors", "pat_idx"], ["is_pat"]),
            helper.make_node("Or", ["is_bg", "is_pat"], ["not_mark"]),
            helper.make_node("Where", ["not_mark", "neg_one", "counts"], ["mark_scores"]),
            helper.make_node("ArgMax", ["mark_scores"], ["mark_idx"], axis=0, keepdims=0),
            helper.make_node("Reshape", ["pat_idx", "ch_idx_shape"], ["pat_i"]),
            helper.make_node("Reshape", ["mark_idx", "ch_idx_shape"], ["mark_i"]),
            helper.make_node("Gather", [IN_NAME, "pat_i"], ["pat30"], axis=1),
            helper.make_node("Gather", [IN_NAME, "mark_i"], ["mark30"], axis=1),
            helper.make_node("Slice", ["pat30", "crop_st", "crop_en", "axes4"], ["pat_plane"]),
            helper.make_node("Slice", ["mark30", "crop_st", "crop_en", "axes4"], ["mark_plane"]),
            helper.make_node("Greater", ["pat_plane", "zero"], ["pat_b"]),
            helper.make_node("Greater", ["mark_plane", "zero"], ["mark_b"]),
            helper.make_node("Cast", ["pat_b"], ["pat_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["mark_b"], ["mark_f"], to=TensorProto.FLOAT),
        ]
    )

    pr0, pr1, pc0, pc1 = _bbox_from_mask(nodes, inits, "pat_f", prefix="pat")
    mr0, mr1, mc0, mc1 = _bbox_from_mask(nodes, inits, "mark_f", prefix="mark")
    r0, r1, c0, c1 = _inset_bbox(nodes, pr0, pr1, pc0, pc1, mr0, mr1, mc0, mc1)

    _i64(inits, [1, 1, grid_h * grid_w], "pat_flat_shape")
    _i64(inits, [1, out_size * out_size], "flat_idx_shape")
    _i64(inits, [1, 1, out_size, out_size], "fg_shape")
    nodes.extend(
        [
            helper.make_node("Reshape", ["pat_b", "pat_flat_shape"], ["pat_flat"]),
            helper.make_node("Cast", [r0], ["r0_i32"], to=TensorProto.INT32),
            helper.make_node("Cast", [c0], ["c0_i32"], to=TensorProto.INT32),
            helper.make_node("Add", ["r0_i32", "out_rows"], ["src_r"]),
            helper.make_node("Add", ["c0_i32", "out_cols"], ["src_c"]),
            helper.make_node("Mul", ["src_r", "grid_w_i32"], ["src_ri"]),
            helper.make_node("Add", ["src_ri", "src_c"], ["src_idx"]),
            helper.make_node("Reshape", ["src_idx", "flat_idx_shape"], ["idx_flat"]),
            helper.make_node("Gather", ["pat_flat", "idx_flat"], ["picked"], axis=2),
            helper.make_node("Reshape", ["picked", "fg_shape"], ["fg_out_b"]),
        ]
    )

    nodes.extend(
        [
            helper.make_node("Sub", ["r1", "r0"], ["rh"]),
            helper.make_node("Sub", ["c1", "c0"], ["rw"]),
            helper.make_node("Add", ["rh", "one_i"], ["out_h"]),
            helper.make_node("Add", ["rw", "one_i"], ["out_w"]),
            helper.make_node("Cast", ["out_h"], ["out_h_i32"], to=TensorProto.INT32),
            helper.make_node("Cast", ["out_w"], ["out_w_i32"], to=TensorProto.INT32),
            helper.make_node("Less", ["out_rows", "out_h_i32"], ["ok_r"]),
            helper.make_node("Less", ["out_cols", "out_w_i32"], ["ok_c"]),
            helper.make_node("And", ["ok_r", "ok_c"], ["ok_rc"]),
            helper.make_node("And", ["fg_out_b", "ok_rc"], ["fg_crop_b"]),
        ]
    )

    nodes.extend(
        [
            helper.make_node("Not", ["fg_crop_b"], ["not_fg_b"]),
            helper.make_node("And", ["not_fg_b", "ok_rc"], ["black_b"]),
        ]
    )

    planes = ["black_b"]
    for c in range(1, C):
        _i64(inits, [c], f"cid{c}")
        is_c = f"isc{c}"
        plane = f"pl{c}"
        nodes.extend(
            [
                helper.make_node("Equal", ["mark_i", f"cid{c}"], [is_c]),
                helper.make_node("And", ["fg_crop_b", is_c], [plane]),
            ]
        )
        planes.append(plane)

    nodes.extend(
        [
            helper.make_node("Concat", planes, ["out_small_b"], axis=1),
            helper.make_node("Cast", ["out_small_b"], ["out_small"], to=TensorProto.FLOAT),
        ]
    )
    pad_attr = [0, 0, 0, 0, 0, 0, H - out_size, W - out_size]
    if opset >= 11:
        nodes.append(helper.make_node("Pad", ["out_small", "pad30"], [OUT_NAME]))
    else:
        nodes.append(helper.make_node("Pad", ["out_small"], [OUT_NAME], pads=pad_attr))
    return _make_model(nodes, inits, opset=opset, graph_name=f"task088_g{grid_h}x{grid_w}_o{out_size}")


def save_model(path: Path = BEST_PATH, **kwargs: Any) -> onnx.ModelProto:
    model = build_model(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return model


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def test_hardcoded() -> None:
    x = _grid_to_onehot(TOY_INPUT)
    model = build_model()
    y = _run_onnx(model, x)
    pred = _onehot_to_grid(y[0])
    exp = np.array(TOY_OUTPUT, dtype=np.int64)
    assert pred[: exp.shape[0], : exp.shape[1]].tolist() == exp.tolist(), pred[:3, :3]


def test_task_json() -> None:
    with DATA_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    model = onnx.load(str(BEST_PATH))
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            x = convert_to_numpy(ex, "input")
            y = _run_onnx(model, x)
            exp = convert_to_numpy(ex, "output")
            if x is None or exp is None:
                continue
            if not np.array_equal((y > 0.0), exp):
                raise AssertionError(f"failed on {split}")


def _score_variant(name: str, builder: Callable[[], onnx.ModelProto]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{name}.onnx"
        onnx.save(builder(), str(path))
        return score_file(path)


def main() -> None:
    test_hardcoded()
    variants: list[tuple[str, dict[str, Any]]] = [
        ("g22x20_o10_op10", {"grid_h": 22, "grid_w": 20, "out_size": 10, "opset": 10}),
        ("g22_o10_op10", {"grid": 22, "out_size": 10, "opset": 10}),
        ("g24_o10_op10", {"grid": 24, "out_size": 10, "opset": 10}),
        ("g30_o10_op10", {"grid": 30, "out_size": 10, "opset": 10}),
    ]
    best_name = "g22x20_o10_op10"
    best_score = -1.0
    best_kwargs: dict[str, Any] = {"grid_h": 22, "grid_w": 20, "out_size": 10, "opset": 10}
    print("Scoring variants...")
    for name, kwargs in variants:
        try:
            stats = _score_variant(name, lambda k=kwargs: build_model(**k))
        except Exception as exc:
            print(f"  {name}: FAIL {exc}")
            continue
        ok = bool(stats.get("valid"))
        sc = float(stats.get("score", 0.0) or 0.0)
        print(
            f"  {name}: ok={ok} score={sc:.3f} cost={stats.get('cost')} "
            f"mem={stats.get('memory')} params={stats.get('params')}"
        )
        if ok and sc > best_score:
            best_score = sc
            best_name = name
            best_kwargs = kwargs

    print(f"Best: {best_name} ({best_kwargs}) score={best_score:.3f}")
    save_model(BEST_PATH, **best_kwargs)
    test_hardcoded()
    test_task_json()
    final = score_file(BEST_PATH)
    print(
        f"Saved {BEST_PATH}: score={final.get('score'):.3f} cost={final.get('cost')} "
        f"memory={final.get('memory')} params={final.get('params')}"
    )


if __name__ == "__main__":
    main()
