"""Build, validate, and score optimized ONNX models for NeuroGolf task049.

Task rule: several solid colored rectangles sit on a black background. Output is
the smallest rectangle object (minimum bounding-box area per color), copied as a
compact filled rectangle in its original color at the top-left of the 30x30 grid.

Selection tie-break: smaller bbox area, then fewer pixels of that color, then
lower color index.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file, sanitize_model  # noqa: E402

TASK_JSON = ROOT / "data" / "task049.json"
OUT_PATH = ROOT / "task049.onnx"

C = 10
NC = 9
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
BIG = 999.0
AREA_MUL = 10_000.0


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def solve_reference(grid: np.ndarray) -> np.ndarray:
    """Per-color bbox area minimum; crop filled with winning color at origin."""
    g = np.asarray(grid, dtype=np.int64)
    best_key: tuple[int, int, int] | None = None
    best_color: int | None = None
    best_box: tuple[int, int, int, int] | None = None

    for color in np.unique(g):
        color = int(color)
        if color == 0:
            continue
        ys, xs = np.where(g == color)
        min_y, max_y = int(ys.min()), int(ys.max())
        min_x, max_x = int(xs.min()), int(xs.max())
        area = (max_y - min_y + 1) * (max_x - min_x + 1)
        key = (area, len(ys), color)
        if best_key is None or key < best_key:
            best_key = key
            best_color = color
            best_box = (min_y, max_y, min_x, max_x)

    out = np.zeros((H, W), dtype=np.int64)
    if best_color is None or best_box is None:
        return out

    min_y, max_y, min_x, max_x = best_box
    ch = max_y - min_y + 1
    cw = max_x - min_x + 1
    out[:ch, :cw] = best_color
    return out


def _make_model(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    *,
    name: str,
    opset: int,
    ir_version: int,
) -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(nodes, name, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="task049",
        ir_version=ir_version,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    model.doc_string = name
    onnx.checker.check_model(model)
    return model


def _bbox_and_select_nodes(
    nodes: List[onnx.NodeProto],
    fg_name: str,
    *,
    gh: int,
    gw: int,
    use_bool: bool,
    prefix: str,
) -> tuple[str, str, str, str, str]:
    """Return win_idx, height, width, min_y, min_x tensor names."""
    inits: List[onnx.TensorProto] = []
    rows = _f32(inits, np.arange(gh, dtype=np.float32).reshape(1, 1, gh, 1), f"{prefix}_rows")
    cols = _f32(inits, np.arange(gw, dtype=np.float32).reshape(1, 1, 1, gw), f"{prefix}_cols")
    ch_idx = _f32(inits, np.arange(NC, dtype=np.float32).reshape(1, NC, 1, 1), f"{prefix}_chidx")
    big = _f32(inits, [BIG], f"{prefix}_big")
    half = _f32(inits, [0.5], f"{prefix}_half")
    one = _f32(inits, [1.0], f"{prefix}_one")
    zero = _f32(inits, [0.0], f"{prefix}_zero")
    area_mul = _f32(inits, [AREA_MUL], f"{prefix}_amul")
    ten = _f32(inits, [10.0], f"{prefix}_ten")

    p = prefix
    work = fg_name
    if use_bool:
        nodes.append(helper.make_node("Greater", [fg_name, half], [f"{p}_fgb"]))
        nodes.append(helper.make_node("Cast", [f"{p}_fgb"], [f"{p}_fgf"], to=TensorProto.FLOAT))
        work = f"{p}_fgf"
        cnt_in = work
    else:
        cnt_in = work

    nodes.extend(
        [
            helper.make_node("ReduceSum", [cnt_in], [f"{p}_cnt"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMax", [work], [f"{p}_rowocc"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", [work], [f"{p}_colocc"], axes=[2], keepdims=1),
            helper.make_node("Greater", [f"{p}_rowocc", half], [f"{p}_rowb"]),
            helper.make_node("Greater", [f"{p}_colocc", half], [f"{p}_colb"]),
            helper.make_node("Where", [f"{p}_rowb", rows, big], [f"{p}_rowmins"]),
            helper.make_node("Where", [f"{p}_colb", cols, big], [f"{p}_colmins"]),
            helper.make_node("ReduceMin", [f"{p}_rowmins"], [f"{p}_miny"], axes=[2], keepdims=1),
            helper.make_node("ReduceMin", [f"{p}_colmins"], [f"{p}_minx"], axes=[3], keepdims=1),
            helper.make_node("Where", [f"{p}_rowb", rows, zero], [f"{p}_rowmaxs"]),
            helper.make_node("Where", [f"{p}_colb", cols, zero], [f"{p}_colmaxs"]),
            helper.make_node("ReduceMax", [f"{p}_rowmaxs"], [f"{p}_maxy"], axes=[2], keepdims=1),
            helper.make_node("ReduceMax", [f"{p}_colmaxs"], [f"{p}_maxx"], axes=[3], keepdims=1),
            helper.make_node("Sub", [f"{p}_maxy", f"{p}_miny"], [f"{p}_h0"]),
            helper.make_node("Sub", [f"{p}_maxx", f"{p}_minx"], [f"{p}_w0"]),
            helper.make_node("Add", [f"{p}_h0", one], [f"{p}_area_h"]),
            helper.make_node("Add", [f"{p}_w0", one], [f"{p}_area_w"]),
            helper.make_node("Mul", [f"{p}_area_h", f"{p}_area_w"], [f"{p}_area"]),
            helper.make_node("Greater", [f"{p}_cnt", half], [f"{p}_has"]),
            helper.make_node("Where", [f"{p}_has", f"{p}_area", big], [f"{p}_areap"]),
            helper.make_node("Mul", [f"{p}_areap", area_mul], [f"{p}_k0"]),
            helper.make_node("Mul", [f"{p}_cnt", ten], [f"{p}_k1"]),
            helper.make_node("Add", [f"{p}_k0", f"{p}_k1"], [f"{p}_k2"]),
            helper.make_node("Add", [f"{p}_k2", ch_idx], [f"{p}_key"]),
            helper.make_node("ArgMin", [f"{p}_key"], [f"{p}_win"], axis=1, keepdims=1),
            helper.make_node("Gather", [f"{p}_miny", f"{p}_win"], [f"{p}_wminy"], axis=1),
            helper.make_node("Gather", [f"{p}_minx", f"{p}_win"], [f"{p}_wminx"], axis=1),
            helper.make_node("Gather", [f"{p}_maxy", f"{p}_win"], [f"{p}_wmaxy"], axis=1),
            helper.make_node("Gather", [f"{p}_maxx", f"{p}_win"], [f"{p}_wmaxx"], axis=1),
            helper.make_node("Sub", [f"{p}_wmaxy", f"{p}_wminy"], [f"{p}_wh0"]),
            helper.make_node("Sub", [f"{p}_wmaxx", f"{p}_wminx"], [f"{p}_ww0"]),
            helper.make_node("Add", [f"{p}_wh0", one], [f"{p}_height"]),
            helper.make_node("Add", [f"{p}_ww0", one], [f"{p}_width"]),
        ]
    )
    return f"{p}_win", f"{p}_height", f"{p}_width", f"{p}_wminy", f"{p}_wminx", inits


def _fill_output_nodes(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    *,
    win_idx: str,
    height: str,
    width: str,
    prefix: str,
) -> None:
    out_rows = _f32(inits, np.arange(H, dtype=np.float32).reshape(1, 1, H, 1), f"{prefix}_orows")
    out_cols = _f32(inits, np.arange(W, dtype=np.float32).reshape(1, 1, 1, W), f"{prefix}_ocols")
    colors = _i64(inits, np.arange(C, dtype=np.int64).reshape(1, C, 1, 1), f"{prefix}_colors")
    one_i = _i64(inits, [1], f"{prefix}_onei")
    sq_all = [0, 1, 2, 3]

    p = prefix
    nodes.extend(
        [
            helper.make_node("Squeeze", [height], [f"{p}_hf"], axes=sq_all),
            helper.make_node("Squeeze", [width], [f"{p}_wf"], axes=sq_all),
            helper.make_node("Less", [out_rows, f"{p}_hf"], [f"{p}_iny"]),
            helper.make_node("Less", [out_cols, f"{p}_wf"], [f"{p}_inx"]),
            helper.make_node("And", [f"{p}_iny", f"{p}_inx"], [f"{p}_inside"]),
            helper.make_node("Cast", [f"{p}_inside"], [f"{p}_insidef"], to=TensorProto.FLOAT),
            helper.make_node("Cast", [win_idx], [f"{p}_wini"], to=TensorProto.INT64),
            helper.make_node("Squeeze", [f"{p}_wini"], [f"{p}_wins"], axes=sq_all),
            helper.make_node("Add", [f"{p}_wins", one_i], [f"{p}_winc"]),
            helper.make_node("Equal", [colors, f"{p}_winc"], [f"{p}_match"]),
            helper.make_node("Cast", [f"{p}_match"], [f"{p}_matchf"], to=TensorProto.FLOAT),
            helper.make_node("Mul", [f"{p}_matchf", f"{p}_insidef"], [OUT_NAME]),
        ]
    )


def build_parallel(
    *,
    crop: int | None,
    use_bool: bool,
    opset: int = 10,
    ir_version: int = 10,
) -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    ch_st = _i64(inits, [0, 1, 0, 0], "ch_st")
    ch_en = _i64(inits, [1, C, H, W], "ch_en")
    fg_st = _i64(inits, [0, 1, 0, 0], "fg_st")
    fg_en = _i64(inits, [1, C, H, W], "fg_en")

    src = IN_NAME
    gh, gw = H, W
    if crop is not None:
        gh = gw = crop
        crop_st = _i64(inits, [0, 0, 0, 0], "crop_st")
        crop_en = _i64(inits, [1, C, crop, crop], "crop_en")
        nodes.append(helper.make_node("Slice", [IN_NAME, crop_st, crop_en, axes4], ["crop"]))
        src = "crop"
        fg_en = _i64(inits, [1, C, crop, crop], "fg_en_c")

    nodes.append(helper.make_node("Slice", [src, fg_st, fg_en, axes4], ["fg"]))
    extra_inits: List[onnx.TensorProto]
    win_idx, height, width, _, _, extra_inits = _bbox_and_select_nodes(
        nodes, "fg", gh=gh, gw=gw, use_bool=use_bool, prefix="sel"
    )
    inits.extend(extra_inits)
    _fill_output_nodes(nodes, inits, win_idx=win_idx, height=height, width=width, prefix="out")

    tag = f"p{crop or H}_{'bool' if use_bool else 'float'}_op{opset}"
    return _make_model(nodes, inits, name=tag, opset=opset, ir_version=ir_version)


def build_parallel_direct(*, use_bool: bool, opset: int = 10, ir_version: int = 10) -> onnx.ModelProto:
    """Slice foreground channels directly from input without a full 10-channel crop."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    crop = 20
    gh = gw = crop

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    fg_st = _i64(inits, [0, 1, 0, 0], "fg_st")
    fg_en = _i64(inits, [1, C, crop, crop], "fg_en")

    nodes.append(helper.make_node("Slice", [IN_NAME, fg_st, fg_en, axes4], ["fg"]))
    extra_inits: List[onnx.TensorProto]
    win_idx, height, width, _, _, extra_inits = _bbox_and_select_nodes(
        nodes, "fg", gh=gh, gw=gw, use_bool=use_bool, prefix="sel"
    )
    inits.extend(extra_inits)
    _fill_output_nodes(nodes, inits, win_idx=win_idx, height=height, width=width, prefix="out")
    tag = f"pd20_{'bool' if use_bool else 'float'}_op{opset}"
    return _make_model(nodes, inits, name=tag, opset=opset, ir_version=ir_version)


def build_sequential(*, crop: int = 20, opset: int = 10, ir_version: int = 10) -> onnx.ModelProto:
    """Fold best color with scalar state; avoids [1,9,H,W] parallel tensors."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    gh = gw = crop
    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    crop_st = _i64(inits, [0, 0, 0, 0], "crop_st")
    crop_en = _i64(inits, [1, C, crop, crop], "crop_en")
    fg_st = _i64(inits, [0, 1, 0, 0], "fg_st")
    fg_en = _i64(inits, [1, C, crop, crop], "fg_en_c")
    rows = _f32(inits, np.arange(gh, dtype=np.float32).reshape(1, 1, gh, 1), "rows")
    cols = _f32(inits, np.arange(gw, dtype=np.float32).reshape(1, 1, 1, gw), "cols")
    big = _f32(inits, [BIG], "big")
    half = _f32(inits, [0.5], "half")
    one = _f32(inits, [1.0], "one")
    zero = _f32(inits, [0.0], "zero")
    area_mul = _f32(inits, [AREA_MUL], "amul")
    ten = _f32(inits, [10.0], "ten")
    one_i = _i64(inits, [1], "one_i")
    ch1 = _f32(inits, [1.0], "ch1f")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, crop_st, crop_en, axes4], ["crop"]),
            helper.make_node("Slice", ["crop", fg_st, fg_en, axes4], ["fg"]),
        ]
    )

    best_area = _f32(inits, [BIG + 1.0], "best_area")
    best_cnt = _f32(inits, [BIG + 1.0], "best_cnt")
    best_color = _f32(inits, [float(C)], "best_color")
    best_miny = _f32(inits, [0.0], "best_miny")
    best_minx = _f32(inits, [0.0], "best_minx")
    best_maxy = _f32(inits, [0.0], "best_maxy")
    best_maxx = _f32(inits, [0.0], "best_maxx")

    for c in range(1, C):
        cf = _f32(inits, [float(c)], f"c{c}")
        cs = _i64(inits, [0, c, 0, 0], f"cs{c}")
        ce = _i64(inits, [1, c + 1, crop, crop], f"ce{c}")
        nodes.append(helper.make_node("Slice", ["crop", cs, ce, axes4], [f"ch{c}"]))
        ch = f"ch{c}"
        nodes.extend(
            [
                helper.make_node("Greater", [ch, half], [f"mb{c}"]),
                helper.make_node("Cast", [f"mb{c}"], [f"mf{c}"], to=TensorProto.FLOAT),
                helper.make_node("ReduceSum", [f"mf{c}"], [f"cnt{c}"], axes=[2, 3], keepdims=1),
                helper.make_node("ReduceMax", [f"mf{c}"], [f"rowocc{c}"], axes=[3], keepdims=1),
                helper.make_node("ReduceMax", [f"mf{c}"], [f"colocc{c}"], axes=[2], keepdims=1),
                helper.make_node("Greater", [f"rowocc{c}", half], [f"rowb{c}"]),
                helper.make_node("Greater", [f"colocc{c}", half], [f"colb{c}"]),
                helper.make_node("Where", [f"rowb{c}", rows, big], [f"ryi{c}"]),
                helper.make_node("Where", [f"colb{c}", cols, big], [f"cxi{c}"]),
                helper.make_node("Where", [f"rowb{c}", rows, zero], [f"ryn{c}"]),
                helper.make_node("Where", [f"colb{c}", cols, zero], [f"cxn{c}"]),
                helper.make_node("ReduceMin", [f"ryi{c}"], [f"miny{c}"], axes=[2], keepdims=1),
                helper.make_node("ReduceMax", [f"ryn{c}"], [f"maxy{c}"], axes=[2], keepdims=1),
                helper.make_node("ReduceMin", [f"cxi{c}"], [f"minx{c}"], axes=[3], keepdims=1),
                helper.make_node("ReduceMax", [f"cxn{c}"], [f"maxx{c}"], axes=[3], keepdims=1),
                helper.make_node("Sub", [f"maxy{c}", f"miny{c}"], [f"h0{c}"]),
                helper.make_node("Sub", [f"maxx{c}", f"minx{c}"], [f"w0{c}"]),
                helper.make_node("Add", [f"h0{c}", one], [f"ah{c}"]),
                helper.make_node("Add", [f"w0{c}", one], [f"aw{c}"]),
                helper.make_node("Mul", [f"ah{c}", f"aw{c}"], [f"area{c}"]),
                helper.make_node("Greater", [f"cnt{c}", half], [f"has{c}"]),
                helper.make_node("Where", [f"has{c}", f"area{c}", big], [f"areap{c}"]),
                helper.make_node("Mul", [f"areap{c}", area_mul], [f"k0{c}"]),
                helper.make_node("Mul", [f"cnt{c}", ten], [f"k1{c}"]),
                helper.make_node("Add", [f"k0{c}", f"k1{c}"], [f"k2{c}"]),
                helper.make_node("Add", [f"k2{c}", cf], [f"key{c}"]),
                helper.make_node("Mul", [best_area, area_mul], [f"bk0{c}"]),
                helper.make_node("Mul", [best_cnt, ten], [f"bk1{c}"]),
                helper.make_node("Add", [f"bk0{c}", f"bk1{c}"], [f"bk2{c}"]),
                helper.make_node("Add", [f"bk2{c}", best_color], [f"bkey{c}"]),
                helper.make_node("Less", [f"key{c}", f"bkey{c}"], [f"better{c}"]),
                helper.make_node("Where", [f"better{c}", f"areap{c}", best_area], [f"na{c}"]),
                helper.make_node("Where", [f"better{c}", f"cnt{c}", best_cnt], [f"nc{c}"]),
                helper.make_node("Where", [f"better{c}", cf, best_color], [f"ncol{c}"]),
                helper.make_node("Where", [f"better{c}", f"miny{c}", best_miny], [f"nmy{c}"]),
                helper.make_node("Where", [f"better{c}", f"minx{c}", best_minx], [f"nmx{c}"]),
                helper.make_node("Where", [f"better{c}", f"maxy{c}", best_maxy], [f"nmay{c}"]),
                helper.make_node("Where", [f"better{c}", f"maxx{c}", best_maxx], [f"nmxx{c}"]),
            ]
        )
        best_area, best_cnt, best_color = f"na{c}", f"nc{c}", f"ncol{c}"
        best_miny, best_minx, best_maxy, best_maxx = f"nmy{c}", f"nmx{c}", f"nmay{c}", f"nmxx{c}"

    nodes.extend(
        [
            helper.make_node("Sub", [best_maxy, best_miny], ["wh0"]),
            helper.make_node("Sub", [best_maxx, best_minx], ["ww0"]),
            helper.make_node("Add", ["wh0", one], ["height"]),
            helper.make_node("Add", ["ww0", one], ["width"]),
            helper.make_node("Sub", [best_color, ch1], ["win_idx"]),
        ]
    )
    _fill_output_nodes(nodes, inits, win_idx="win_idx", height="height", width="width", prefix="fill")
    return _make_model(nodes, inits, name="seq20", opset=opset, ir_version=ir_version)


@dataclass
class VariantResult:
    label: str
    model: onnx.ModelProto
    correct: bool
    counts: dict[str, tuple[int, int]]
    memory: int | None
    params: int | None
    cost: int | None
    score: float | None
    valid: bool


def _grid_from_onehot(arr: np.ndarray) -> np.ndarray:
    flat = arr.reshape(C, H, W)
    active = flat > 0.0
    out = flat.argmax(axis=0).astype(np.int64)
    out[~active.any(axis=0)] = 0
    return out


def _check_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return False, {}
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        sanitized.SerializeToString(),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    with TASK_JSON.open(encoding="utf-8") as fh:
        data = json.load(fh)

    counts: dict[str, tuple[int, int]] = {}
    all_ok = True
    for split in ("train", "test", "arc-gen"):
        passed = 0
        total = 0
        for ex in data.get(split, []):
            inp = convert_to_numpy(ex, "input")
            exp = convert_to_numpy(ex, "output")
            if inp is None or exp is None:
                continue
            total += 1
            pred = sess.run([OUT_NAME], {IN_NAME: inp})[0]
            if not np.array_equal((pred > 0.0).astype(np.float32), exp):
                all_ok = False
            else:
                passed += 1
        counts[split] = (passed, total)
    return all_ok, counts


def _score_model(model: onnx.ModelProto) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        path = Path(tmp.name)
        onnx.save(model, path)
        try:
            return score_file(path)
        finally:
            path.unlink(missing_ok=True)


def _format_counts(counts: dict[str, tuple[int, int]]) -> str:
    return ", ".join(f"{split} {ok}/{total}" for split, (ok, total) in counts.items())


def benchmark_variants() -> VariantResult:
    builders: Iterable[tuple[str, Callable[[], onnx.ModelProto]]] = (
        ("pd20_float", lambda: build_parallel_direct(use_bool=False)),
        ("p20_float", lambda: build_parallel(crop=20, use_bool=False)),
        ("p20_bool", lambda: build_parallel(crop=20, use_bool=True)),
        ("seq20", build_sequential),
        ("p30_float", lambda: build_parallel(crop=None, use_bool=False)),
    )

    results: list[VariantResult] = []
    print("Benchmarking variants:")
    print("-" * 88)
    for label, builder in builders:
        try:
            model = builder()
        except Exception as exc:  # pragma: no cover
            print(f"{label:<16} BUILD FAIL: {exc}")
            continue

        correct, counts = _check_correct(model)
        scored = _score_model(model)
        vr = VariantResult(
            label=label,
            model=model,
            correct=correct,
            counts=counts,
            memory=scored.get("memory"),
            params=scored.get("params"),
            cost=scored.get("cost"),
            score=scored.get("score"),
            valid=bool(scored.get("valid")),
        )
        score_txt = f"{vr.score:.6f}" if vr.score is not None else "None"
        print(
            f"{label:<16} ok={correct} ({_format_counts(counts)}) "
            f"mem={vr.memory} par={vr.params} cost={vr.cost} score={score_txt}"
        )
        if correct and vr.valid and vr.cost is not None:
            results.append(vr)

    if not results:
        raise SystemExit("No valid correct variant found")

    best = min(results, key=lambda r: int(r.cost))
    print("-" * 88)
    print(f"Best variant: {best.label}  cost={best.cost}  score={best.score:.6f}")
    return best


def validate_train(model: onnx.ModelProto) -> None:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        sanitize_model(copy.deepcopy(model)).SerializeToString(),  # type: ignore[union-attr]
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    with TASK_JSON.open(encoding="utf-8") as fh:
        data = json.load(fh)

    print("\nTrain validation:")
    for idx, ex in enumerate(data["train"]):
        inp = convert_to_numpy(ex, "input")
        exp = np.asarray(ex["output"], dtype=np.int64)
        pred = _grid_from_onehot(sess.run([OUT_NAME], {IN_NAME: inp})[0])
        crop = pred[: exp.shape[0], : exp.shape[1]]
        ok = np.array_equal(crop, exp)
        print(f"  train[{idx}] in={len(ex['input'])}x{len(ex['input'][0])} out={exp.shape[1]}x{exp.shape[0]} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            print("    expected:\n", exp)
            print("    got:\n", crop)


def main() -> None:
    # Sanity-check reference solver against all splits.
    with TASK_JSON.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            g = np.asarray(ex["input"], dtype=np.int64)
            oh, ow = g.shape
            ref = solve_reference(g)[:oh, :ow]
            exp = np.asarray(ex["output"], dtype=np.int64)
            if not np.array_equal(ref[: exp.shape[0], : exp.shape[1]], exp):
                raise SystemExit(f"Reference solver failed on {split}")

    best = benchmark_variants()
    onnx.save(best.model, OUT_PATH)
    validate_train(best.model)
    scored = _score_model(best.model)

    print("\nFinal report")
    print("=" * 40)
    print(f"ONNX path:  {OUT_PATH}")
    print(f"Variant:    {best.label}")
    print(f"Memory:     {scored.get('memory')}")
    print(f"Params:     {scored.get('params')}")
    print(f"Cost:       {scored.get('cost')}")
    if scored.get("score") is not None:
        print(f"Score:      {scored['score']:.6f}")
    print(f"Correct:    {_format_counts(best.counts)}")

    print(
        "\nGraph summary: slice foreground channels 1..9, compute per-color bbox area "
        "with ReduceMin/ReduceMax on row/col coordinate grids, pick the minimum "
        "composite key (area, pixel count, color index) via ArgMin, then emit a solid "
        f"top-left rectangle via channel equality masking. Best variant '{best.label}' "
        "minimizes internal tensor footprint."
    )


if __name__ == "__main__":
    main()
