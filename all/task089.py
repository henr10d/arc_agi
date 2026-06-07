"""ONNX generator for ARC task089: copy anchored two-color objects.

Task rule: the 13x13 input contains one complete object for anchor color 2
and/or 3, plus isolated pixels of those same anchor colors.  A complete object
is the connected nonzero shape touching its anchor pixel.  Each isolated anchor
pixel receives a translated copy of the corresponding object; color 2 copies
are horizontally mirrored relative to the source, while color 3 copies preserve
the source orientation.  The original objects and isolated anchor pixels remain.

ONNX approach: work on the compact 13x13 core, locate each source anchor by
testing for a neighboring nonzero object cell, recover the connected source
stencil inside a 5x5 anchor window, generate all translated target-anchor masks
with fixed 25-channel convolutions, and build the final one-hot output only at
the end.  A single small convolution computes the shared "has any nonzero
neighbor" mask.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task089"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
N = 13
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
IR_VERSION = 10
OPSET = 10
ANCHORS = (2, 3)
OFFS = tuple(range(-2, 3))
NEIGHBORS = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
OFFSET_PAIRS = tuple((dr, dc) for dr in OFFS for dc in OFFS)


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _paint_kernel() -> np.ndarray:
    kernel = np.zeros((len(OFFSET_PAIRS), 1, 5, 5), dtype=np.float32)
    for idx, (dr, dc) in enumerate(OFFSET_PAIRS):
        kernel[idx, 0, 2 - dr, 2 - dc] = 1.0
    return kernel


def _or_chain(nodes: list[onnx.NodeProto], parts: list[str], name: str) -> str:
    cur = parts[0]
    for idx, part in enumerate(parts[1:], start=1):
        nxt = f"{name}_{idx}"
        nodes.append(helper.make_node("Or", [cur, part], [nxt]))
        cur = nxt
    return cur


def _max_chain(nodes: list[onnx.NodeProto], parts: list[str], name: str) -> str:
    cur = parts[0]
    for idx, part in enumerate(parts[1:], start=1):
        nxt = f"{name}_{idx}"
        nodes.append(helper.make_node("Max", [cur, part], [nxt]))
        cur = nxt
    return cur


def _shift13(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    cache: dict[tuple[int, int, int, int], tuple[str, str]],
    tensor: str,
    dr: int,
    dc: int,
    name: str,
) -> str:
    """Shift a float [1,1,13,13] tensor by (dr, dc), padding zeros."""
    r0 = max(0, -dr)
    r1 = N - max(0, dr)
    c0 = max(0, -dc)
    c1 = N - max(0, dc)
    top = max(0, dr)
    bottom = max(0, -dr)
    left = max(0, dc)
    right = max(0, -dc)
    key = (r0, r1, c0, c1)
    if key not in cache:
        cache[key] = (
            _i64(inits, [0, 0, r0, c0], f"shift_{r0}_{r1}_{c0}_{c1}_st"),
            _i64(inits, [1, 1, r1, c1], f"shift_{r0}_{r1}_{c0}_{c1}_en"),
        )
    starts, ends = cache[key]
    cropped = f"{name}_cr"
    shifted = f"{name}_sh"
    nodes.append(helper.make_node("Slice", [tensor, starts, ends, "axes4"], [cropped]))
    nodes.append(
        helper.make_node(
            "Pad",
            [cropped],
            [shifted],
            mode="constant",
            pads=[0, 0, top, left, 0, 0, bottom, right],
        )
    )
    return shifted


def solve(grid: list[list[int]] | np.ndarray, *, reach_iters: int = 4) -> list[list[int]]:
    """Reference implementation of the selected hypothesis."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()

    def has_partner(r: int, c: int, marker: int) -> bool:
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < N and 0 <= nc < N and g[nr, nc]:
                return True
        return False

    for marker in ANCHORS:
        sources: list[tuple[int, int]] = []
        targets: list[tuple[int, int]] = []
        for r in range(N):
            for c in range(N):
                if g[r, c] == marker:
                    (sources if has_partner(r, c, marker) else targets).append((r, c))
        if not sources:
            continue

        sr, sc = max(sources, key=lambda rc: rc[0] * N + rc[1])
        colors: dict[tuple[int, int], int] = {}
        present: dict[tuple[int, int], bool] = {}
        for dr in OFFS:
            for dc in OFFS:
                rr, cc = sr + dr, sc + dc
                color = int(g[rr, cc]) if 0 <= rr < N and 0 <= cc < N else 0
                colors[(dr, dc)] = color
                present[(dr, dc)] = color != 0

        reached: dict[tuple[int, int], bool] = {(0, 0): present[(0, 0)]}
        for _ in range(reach_iters):
            nxt: dict[tuple[int, int], bool] = {(0, 0): present[(0, 0)]}
            for dr in OFFS:
                for dc in OFFS:
                    if (dr, dc) == (0, 0):
                        continue
                    touches = any(reached.get((dr + nr, dc + nc), False) for nr, nc in NEIGHBORS)
                    nxt[(dr, dc)] = present[(dr, dc)] and touches
            reached = nxt

        for tr, tc in targets:
            for dr in OFFS:
                for dc in OFFS:
                    if not reached.get((dr, dc), False):
                        continue
                    out_dc = -dc if marker == 2 else dc
                    rr, cc = tr + dr, tc + out_dc
                    if 0 <= rr < N and 0 <= cc < N:
                        out[rr, cc] = colors[(dr, dc)]
    return out.tolist()


def build_model(*, reach_iters: int = 4) -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    shift_cache: dict[tuple[int, int, int, int], tuple[str, str]] = {}
    scalar_cache: dict[int, str] = {}

    def i64_scalar(value: int) -> str:
        if value not in scalar_cache:
            scalar_cache[value] = _i64(inits, [value], f"i64_{value}".replace("-", "m"))
        return scalar_cache[value]

    _i64(inits, [0, 1, 2, 3], "axes4")
    _i64(inits, [0, 0, 0, 0], "core_st")
    _i64(inits, [1, C, N, N], "core_en")
    _i64(inits, [N * N], "flat169")
    _i64(inits, [1, len(OFFSET_PAIRS), 1, 1], "paint_value_shape")
    _i64(inits, [0], "zero_i")
    _i64(inits, [N], "n_i")
    _f32(inits, [0.0], "zero_f")
    _f32(inits, [float(N * N - 1)], "last_flat_f")
    _f32(
        inits,
        np.asarray([[[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]], dtype=np.float32),
        "neighbor_kernel",
    )
    _i64(inits, np.arange(C, dtype=np.int64).reshape(1, C, 1, 1), "chans")

    _f32(inits, _paint_kernel(), "paint_kernel")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, "core_st", "core_en", "axes4"], ["core"]),
            helper.make_node("ArgMax", ["core"], ["ids64"], axis=1, keepdims=1),
            helper.make_node("Cast", ["ids64"], ["ids"], to=TensorProto.FLOAT),
            helper.make_node("Reshape", ["ids64", "flat169"], ["ids_flat"]),
            helper.make_node("Greater", ["ids", "zero_f"], ["nonzero_core"]),
            helper.make_node("Cast", ["nonzero_core"], ["nonzero_core_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "Conv",
                ["nonzero_core_f", "neighbor_kernel"],
                ["neighbor_count"],
                pads=[1, 1, 1, 1],
            ),
            helper.make_node("Greater", ["neighbor_count", "zero_f"], ["has_neighbor"]),
            helper.make_node("Not", ["has_neighbor"], ["no_neighbor"]),
        ]
    )

    combined = "ids"

    for marker in ANCHORS:
        marker_i = i64_scalar(marker)
        nodes.append(helper.make_node("Equal", ["ids64", marker_i], [f"occ{marker}"]))
        nodes.append(helper.make_node("And", [f"occ{marker}", "has_neighbor"], [f"src{marker}"]))
        nodes.append(helper.make_node("And", [f"occ{marker}", "no_neighbor"], [f"tgt{marker}"]))
        nodes.append(helper.make_node("Cast", [f"src{marker}"], [f"srcf{marker}"], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Cast", [f"tgt{marker}"], [f"tgtf{marker}"], to=TensorProto.FLOAT))

        nodes.extend(
            [
                helper.make_node("Reshape", [f"srcf{marker}", "flat169"], [f"srcf_flat{marker}"]),
                helper.make_node("ArgMax", [f"srcf_flat{marker}"], [f"anchor_idx{marker}"], axis=0, keepdims=0),
                helper.make_node("Div", [f"anchor_idx{marker}", "n_i"], [f"anchor_r{marker}"]),
                helper.make_node("Mod", [f"anchor_idx{marker}", "n_i"], [f"anchor_c{marker}"]),
            ]
        )

        present: dict[tuple[int, int], str] = {}
        colors: dict[tuple[int, int], str] = {}
        for dr in OFFS:
            for dc in OFFS:
                tag = f"{marker}_{dr}_{dc}".replace("-", "m")
                dr_i = i64_scalar(dr)
                dc_i = i64_scalar(dc)
                nodes.append(helper.make_node("Add", [f"anchor_r{marker}", dr_i], [f"rr{tag}"]))
                nodes.append(helper.make_node("Add", [f"anchor_c{marker}", dc_i], [f"cc{tag}"]))

                nodes.append(helper.make_node("Less", [f"rr{tag}", "zero_i"], [f"rr_neg{tag}"]))
                nodes.append(helper.make_node("Not", [f"rr_neg{tag}"], [f"rr_ge0{tag}"]))
                nodes.append(helper.make_node("Less", [f"cc{tag}", "zero_i"], [f"cc_neg{tag}"]))
                nodes.append(helper.make_node("Not", [f"cc_neg{tag}"], [f"cc_ge0{tag}"]))
                nodes.append(helper.make_node("Less", [f"rr{tag}", "n_i"], [f"rr_lt_n{tag}"]))
                nodes.append(helper.make_node("Less", [f"cc{tag}", "n_i"], [f"cc_lt_n{tag}"]))
                nodes.append(helper.make_node("And", [f"rr_ge0{tag}", f"rr_lt_n{tag}"], [f"rr_ok{tag}"]))
                nodes.append(helper.make_node("And", [f"cc_ge0{tag}", f"cc_lt_n{tag}"], [f"cc_ok{tag}"]))
                nodes.append(helper.make_node("And", [f"rr_ok{tag}", f"cc_ok{tag}"], [f"ok{tag}"]))

                offset = i64_scalar(dr * N + dc)
                nodes.append(helper.make_node("Add", [f"anchor_idx{marker}", offset], [f"raw_idx{tag}"]))
                nodes.append(helper.make_node("Cast", [f"raw_idx{tag}"], [f"raw_idx_f{tag}"], to=TensorProto.FLOAT))
                nodes.append(helper.make_node("Max", [f"raw_idx_f{tag}", "zero_f"], [f"idx_pos{tag}"]))
                nodes.append(helper.make_node("Min", [f"idx_pos{tag}", "last_flat_f"], [f"idx_clip_f{tag}"]))
                nodes.append(helper.make_node("Cast", [f"idx_clip_f{tag}"], [f"idx_clip{tag}"], to=TensorProto.INT64))
                nodes.append(helper.make_node("Gather", ["ids_flat", f"idx_clip{tag}"], [f"col_i{tag}"]))
                nodes.append(helper.make_node("Greater", [f"col_i{tag}", "zero_i"], [f"nonzero{tag}"]))
                nodes.append(helper.make_node("And", [f"nonzero{tag}", f"ok{tag}"], [f"present_b{tag}"]))
                nodes.append(helper.make_node("Cast", [f"present_b{tag}"], [f"present{tag}"], to=TensorProto.FLOAT))
                nodes.append(helper.make_node("Cast", [f"col_i{tag}"], [f"col{tag}"], to=TensorProto.FLOAT))
                present[(dr, dc)] = f"present{tag}"
                colors[(dr, dc)] = f"col{tag}"

        if reach_iters == 0:
            reached = dict(present)
        else:
            reached: dict[tuple[int, int], str] = {(0, 0): present[(0, 0)]}
            for step in range(reach_iters):
                nxt: dict[tuple[int, int], str] = {(0, 0): present[(0, 0)]}
                for dr in OFFS:
                    for dc in OFFS:
                        if (dr, dc) == (0, 0):
                            continue
                        nbrs = [reached.get((dr + nr, dc + nc), "zero_f") for nr, nc in NEIGHBORS]
                        touching = _max_chain(nodes, nbrs, f"touch{marker}_{step}_{dr}_{dc}".replace("-", "m"))
                        out_name = f"reach{marker}_{step}_{dr}_{dc}".replace("-", "m")
                        nodes.append(helper.make_node("Mul", [present[(dr, dc)], touching], [out_name]))
                        nxt[(dr, dc)] = out_name
                reached = nxt

        paint_values: list[str] = []
        for dr, out_dc in OFFSET_PAIRS:
            dc = -out_dc if marker == 2 else out_dc
            tag = f"{marker}_{dr}_{dc}".replace("-", "m")
            nodes.append(helper.make_node("Mul", [reached[(dr, dc)], colors[(dr, dc)]], [f"paint_value{tag}"]))
            paint_values.append(f"paint_value{tag}")

        nodes.extend(
            [
                helper.make_node(
                    "Conv",
                    [f"tgtf{marker}", "paint_kernel"],
                    [f"target_offsets{marker}"],
                    pads=[2, 2, 2, 2],
                ),
                helper.make_node("Concat", paint_values, [f"paint_values{marker}"], axis=0),
                helper.make_node(
                    "Reshape",
                    [f"paint_values{marker}", "paint_value_shape"],
                    [f"paint_values4{marker}"],
                ),
                helper.make_node(
                    "Mul",
                    [f"target_offsets{marker}", f"paint_values4{marker}"],
                    [f"paint_weighted{marker}"],
                ),
                helper.make_node(
                    "ReduceMax",
                    [f"paint_weighted{marker}"],
                    [f"paint_reduced{marker}"],
                    axes=[1],
                    keepdims=1,
                ),
                helper.make_node("Max", [combined, f"paint_reduced{marker}"], [f"combined_marker{marker}"]),
            ]
        )
        combined = f"combined_marker{marker}"

    nodes.extend(
        [
            helper.make_node("Cast", [combined], ["ids_out"], to=TensorProto.INT64),
            helper.make_node("Equal", ["chans", "ids_out"], ["oh_b"]),
            helper.make_node("Cast", ["oh_b"], ["oh_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["oh_f"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - N, W - N],
            ),
        ]
    )

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


def validate(model: onnx.ModelProto, data: dict[str, Any]) -> tuple[bool, dict[str, tuple[int, int]]]:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    counts: dict[str, tuple[int, int]] = {}
    ok = True
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
            if np.array_equal((pred > 0).astype(np.float32), exp):
                passed += 1
            else:
                ok = False
        counts[split] = (passed, total)
    return ok, counts


def validate_reference(data: dict[str, Any], *, reach_iters: int) -> bool:
    for split in ("train", "test", "arc-gen"):
        for ex in data.get(split, []):
            pred = np.asarray(solve(ex["input"], reach_iters=reach_iters), dtype=np.int64)
            if not np.array_equal(pred, np.asarray(ex["output"], dtype=np.int64)):
                return False
    return True


def main() -> None:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))

    candidates = [
        ("raw_5x5_window", 0),
        ("two_step_connected_window", 2),
        ("three_step_connected_window", 3),
        ("four_step_connected_window", 4),
    ]
    best: tuple[int, float, str, int, onnx.ModelProto] | None = None

    with tempfile.TemporaryDirectory(prefix="task089_candidates_") as tmpdir:
        tmp = Path(tmpdir)
        for name, reach_iters in candidates:
            ref_ok = validate_reference(data, reach_iters=reach_iters)
            model = build_model(reach_iters=reach_iters)
            path = tmp / f"{TASK_ID}_{name}.onnx"
            onnx.save(model, path)
            ok, counts = validate(model, data)
            stats = score_file(path)
            score_text = "invalid"
            if stats["valid"]:
                score_text = (
                    f"memory={stats['memory']} params={stats['params']} "
                    f"cost={stats['cost']} score={float(stats['score']):.6f}"
                )
            print(f"{name}: reference={ref_ok} onnx={ok} counts={counts} {score_text}")
            if ok and stats["valid"]:
                cost = int(stats["cost"])
                score = float(stats["score"])
                if best is None or cost < best[0]:
                    best = (cost, score, name, reach_iters, model)

    if best is None:
        raise SystemExit("no correct valid task089 candidate")

    cost, score, name, reach_iters, model = best
    onnx.save(model, BEST_PATH)
    final_stats = score_file(BEST_PATH)
    print(f"\nSelected {name} (reach_iters={reach_iters})")
    print(f"Wrote {BEST_PATH}")
    print(f"memory: {final_stats['memory']}")
    print(f"params: {final_stats['params']}")
    print(f"cost:   {final_stats['cost']}")
    print(f"score:  {float(final_stats['score']):.6f}")


if __name__ == "__main__":
    main()
