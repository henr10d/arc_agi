"""Optimized ONNX for ARC task098 rectangle outline extraction.

Task rule: each input grid contains solid nonzero colored rectangles on
background color 0. The output preserves every rectangle position and color, but
only the one-cell-thick outline remains; each strict same-color 4-neighbor
interior cell is changed to background 0. The rule is color-independent and is
applied across the full padded NeuroGolf grid, so variable visible sizes remain
safe.

ONNX approach: convert the one-hot input to a compact color-id grid, cast it to
float16, and run a 3x3 AveragePool. For the separated solid rectangles in this
task, a cell is interior exactly when the 3x3 average equals the center color
(background areas may also match, but zeroing them is harmless). The float16
pooled grid is then zeroed at interiors, invalid padded cells are mapped to a
sentinel class, and the result is one-hot decoded to float output.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_all_onnx import verify_correctness  # noqa: E402
from score_model import calculate_memory, calculate_params, print_report, sanitize_model, score, score_file  # noqa: E402

TASK_ID = "task098"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
ROOT_PATH = ROOT / f"{TASK_ID}.onnx"

C = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _slice(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    source: str,
    out: str,
    starts: list[int],
    ends: list[int],
    axes: list[int],
) -> str:
    s = _init(inits, f"{out}_s", np.asarray(starts, dtype=np.int64))
    e = _init(inits, f"{out}_e", np.asarray(ends, dtype=np.int64))
    a = _init(inits, f"{out}_a", np.asarray(axes, dtype=np.int64))
    nodes.append(helper.make_node("Slice", [source, s, e, a], [out]))
    return out


def _finish(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
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


def _pad_bool_28_to_30(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    source: str,
    out: str,
    *,
    channels: int = 1,
) -> str:
    left = _init(inits, f"{out}_zl", np.zeros((1, channels, 28, 1), dtype=bool))
    right = _init(inits, f"{out}_zr", np.zeros((1, channels, 28, 1), dtype=bool))
    top = _init(inits, f"{out}_zt", np.zeros((1, channels, 1, 30), dtype=bool))
    bottom = _init(inits, f"{out}_zb", np.zeros((1, channels, 1, 30), dtype=bool))
    nodes.append(helper.make_node("Concat", [left, source, right], [f"{out}_mid"], axis=3))
    nodes.append(helper.make_node("Concat", [top, f"{out}_mid", bottom], [out], axis=2))
    return out


def build_int32_argmax_model() -> onnx.ModelProto:
    """Best measured variant: compact int32 color ids and one-hot decode."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    zero = _init(inits, "z_i32", np.asarray([0], dtype=np.int32))
    invalid = _init(inits, "invalid_i32", np.asarray([10], dtype=np.int32))
    zero_f = _init(inits, "z_f", np.asarray([0.0], dtype=np.float32))
    classes = _init(inits, "cls_i32", np.arange(C, dtype=np.int32).reshape(1, C, 1, 1))

    nodes.extend(
        [
            helper.make_node("ArgMax", [IN_NAME], ["ids64"], axis=1, keepdims=1),
            helper.make_node("Cast", ["ids64"], ["ids"], to=TensorProto.INT32),
            helper.make_node("ReduceSum", [IN_NAME], ["valid_f"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["valid_f", zero_f], ["valid"]),
        ]
    )

    center = _slice(nodes, inits, "ids", "c", [1, 1], [29, 29], [2, 3])
    up = _slice(nodes, inits, "ids", "u", [0, 1], [28, 29], [2, 3])
    down = _slice(nodes, inits, "ids", "d", [2, 1], [30, 29], [2, 3])
    left = _slice(nodes, inits, "ids", "l", [1, 0], [29, 28], [2, 3])
    right = _slice(nodes, inits, "ids", "r", [1, 2], [29, 30], [2, 3])

    nodes.extend(
        [
            helper.make_node("Greater", [center, zero], ["nz"]),
            helper.make_node("Equal", [center, up], ["same_u"]),
            helper.make_node("Equal", [center, down], ["same_d"]),
            helper.make_node("Equal", [center, left], ["same_l"]),
            helper.make_node("Equal", [center, right], ["same_r"]),
            helper.make_node("And", ["same_u", "same_d"], ["same_ud"]),
            helper.make_node("And", ["same_l", "same_r"], ["same_lr"]),
            helper.make_node("And", ["same_ud", "same_lr"], ["same4"]),
            helper.make_node("And", ["same4", "nz"], ["interior28"]),
        ]
    )
    _pad_bool_28_to_30(nodes, inits, "interior28", "interior")
    nodes.extend(
        [
            helper.make_node("Where", ["interior", zero, "ids"], ["out_ids"]),
            helper.make_node("Where", ["valid", "out_ids", invalid], ["valid_ids"]),
            helper.make_node("Equal", ["valid_ids", classes], ["out_bool"]),
            helper.make_node("Cast", ["out_bool"], [OUT_NAME], to=TensorProto.FLOAT),
        ]
    )
    return _finish(nodes, inits, "task098_int32_argmax")


def build_float16_avg_id_model() -> onnx.ModelProto:
    """Best measured variant: float16 pooled color ids for solid rectangles."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    zero_f16 = _init(inits, "z_f16", np.asarray([0.0], dtype=np.float16))
    invalid_f16 = _init(inits, "invalid_f16", np.asarray([10.0], dtype=np.float16))
    zero_f = _init(inits, "z_f", np.asarray([0.0], dtype=np.float32))
    classes = _init(inits, "cls_i32", np.arange(C, dtype=np.int32).reshape(1, C, 1, 1))

    nodes.extend(
        [
            helper.make_node("ArgMax", [IN_NAME], ["ids64"], axis=1, keepdims=1),
            helper.make_node("Cast", ["ids64"], ["idsf"], to=TensorProto.FLOAT16),
            helper.make_node("ReduceSum", [IN_NAME], ["valid_f"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["valid_f", zero_f], ["valid"]),
            helper.make_node(
                "AveragePool",
                ["idsf"],
                ["pool"],
                kernel_shape=[3, 3],
                strides=[1, 1],
                count_include_pad=0,
            ),
        ]
    )

    center = _slice(nodes, inits, "idsf", "c", [1, 1], [29, 29], [2, 3])
    nodes.extend(
        [
            helper.make_node("Less", ["pool", center], ["lt"]),
            helper.make_node("Greater", ["pool", center], ["gt"]),
            helper.make_node("Or", ["lt", "gt"], ["diff"]),
            helper.make_node("Not", ["diff"], ["interior28"]),
        ]
    )
    _pad_bool_28_to_30(nodes, inits, "interior28", "interior")
    nodes.extend(
        [
            helper.make_node("Where", ["interior", zero_f16, "idsf"], ["out_idsf"]),
            helper.make_node("Where", ["valid", "out_idsf", invalid_f16], ["valid_idsf"]),
            helper.make_node("Cast", ["valid_idsf"], ["valid_ids"], to=TensorProto.INT32),
            helper.make_node("Equal", ["valid_ids", classes], ["out_bool"]),
            helper.make_node("Cast", ["out_bool"], [OUT_NAME], to=TensorProto.FLOAT),
        ]
    )
    return _finish(nodes, inits, "task098_float16_avg_id")


def build_int64_argmax_model() -> onnx.ModelProto:
    """Comparison variant without the uint8 cast; correct but higher memory."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    zero = _init(inits, "z_i64", np.asarray([0], dtype=np.int64))
    invalid = _init(inits, "invalid_i64", np.asarray([10], dtype=np.int64))
    zero_f = _init(inits, "z_f", np.asarray([0.0], dtype=np.float32))
    classes = _init(inits, "cls_i64", np.arange(C, dtype=np.int64).reshape(1, C, 1, 1))
    nodes.extend(
        [
            helper.make_node("ArgMax", [IN_NAME], ["ids"], axis=1, keepdims=1),
            helper.make_node("ReduceSum", [IN_NAME], ["valid_f"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["valid_f", zero_f], ["valid"]),
        ]
    )

    center = _slice(nodes, inits, "ids", "c", [1, 1], [29, 29], [2, 3])
    up = _slice(nodes, inits, "ids", "u", [0, 1], [28, 29], [2, 3])
    down = _slice(nodes, inits, "ids", "d", [2, 1], [30, 29], [2, 3])
    left = _slice(nodes, inits, "ids", "l", [1, 0], [29, 28], [2, 3])
    right = _slice(nodes, inits, "ids", "r", [1, 2], [29, 30], [2, 3])

    nodes.extend(
        [
            helper.make_node("Greater", [center, zero], ["nz"]),
            helper.make_node("Equal", [center, up], ["same_u"]),
            helper.make_node("Equal", [center, down], ["same_d"]),
            helper.make_node("Equal", [center, left], ["same_l"]),
            helper.make_node("Equal", [center, right], ["same_r"]),
            helper.make_node("And", ["same_u", "same_d"], ["same_ud"]),
            helper.make_node("And", ["same_l", "same_r"], ["same_lr"]),
            helper.make_node("And", ["same_ud", "same_lr"], ["same4"]),
            helper.make_node("And", ["same4", "nz"], ["interior28"]),
        ]
    )
    _pad_bool_28_to_30(nodes, inits, "interior28", "interior")
    nodes.extend(
        [
            helper.make_node("Where", ["interior", zero, "ids"], ["out_ids"]),
            helper.make_node("Where", ["valid", "out_ids", invalid], ["valid_ids"]),
            helper.make_node("Equal", ["valid_ids", classes], ["out_bool"]),
            helper.make_node("Cast", ["out_bool"], [OUT_NAME], to=TensorProto.FLOAT),
        ]
    )
    return _finish(nodes, inits, "task098_int64_argmax")


def build_channel_bool_model() -> onnx.ModelProto:
    """Comparison variant: do all four-neighbor tests directly per color plane."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    zero_f = _init(inits, "z_f", np.asarray([0.0], dtype=np.float32))
    zero_i = _init(inits, "z_i", np.asarray([0], dtype=np.int32))
    nodes.append(helper.make_node("Greater", [IN_NAME, zero_f], ["inb"]))
    fg = _slice(nodes, inits, "inb", "fg", [1], [10], [1])
    center = _slice(nodes, inits, fg, "c", [1, 1], [29, 29], [2, 3])
    up = _slice(nodes, inits, fg, "u", [0, 1], [28, 29], [2, 3])
    down = _slice(nodes, inits, fg, "d", [2, 1], [30, 29], [2, 3])
    left = _slice(nodes, inits, fg, "l", [1, 0], [29, 28], [2, 3])
    right = _slice(nodes, inits, fg, "r", [1, 2], [29, 30], [2, 3])

    nodes.extend(
        [
            helper.make_node("And", [center, up], ["cu"]),
            helper.make_node("And", ["cu", down], ["cud"]),
            helper.make_node("And", ["cud", left], ["cudl"]),
            helper.make_node("And", ["cudl", right], ["interior28"]),
        ]
    )
    _pad_bool_28_to_30(nodes, inits, "interior28", "interior", channels=9)
    ch0 = _slice(nodes, inits, "inb", "ch0", [0], [1], [1])
    nodes.extend(
        [
            helper.make_node("Cast", ["interior"], ["interior_i"], to=TensorProto.INT32),
            helper.make_node("ReduceSum", ["interior_i"], ["removed"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["removed", zero_i], ["removed_b"]),
            helper.make_node("Or", [ch0, "removed_b"], ["out_bg"]),
            helper.make_node("Not", ["interior"], ["keep_fg"]),
            helper.make_node("And", [fg, "keep_fg"], ["out_fg"]),
            helper.make_node("Concat", ["out_bg", "out_fg"], ["out_bool"], axis=1),
            helper.make_node("Cast", ["out_bool"], [OUT_NAME], to=TensorProto.FLOAT),
        ]
    )
    return _finish(nodes, inits, "task098_channel_bool")


def build_avgpool_model() -> onnx.ModelProto:
    """Comparison variant for solid rectangles: 3x3 AveragePool erosion."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    zero_f = _init(inits, "z_f", np.asarray([0.0], dtype=np.float32))
    zero_i = _init(inits, "z_i", np.asarray([0], dtype=np.int32))
    thresh = _init(inits, "th", np.asarray([0.999], dtype=np.float32))
    nodes.append(helper.make_node("Greater", [IN_NAME, zero_f], ["inb"]))
    fg = _slice(nodes, inits, IN_NAME, "fgf", [1], [10], [1])
    fgb = _slice(nodes, inits, "inb", "fgb", [1], [10], [1])
    nodes.extend(
        [
            helper.make_node(
                "AveragePool",
                [fg],
                ["pool"],
                kernel_shape=[3, 3],
                strides=[1, 1],
                count_include_pad=1,
            ),
            helper.make_node("Greater", ["pool", thresh], ["interior28"]),
        ]
    )
    _pad_bool_28_to_30(nodes, inits, "interior28", "interior", channels=9)
    ch0 = _slice(nodes, inits, "inb", "ch0", [0], [1], [1])
    nodes.extend(
        [
            helper.make_node("Cast", ["interior"], ["interior_i"], to=TensorProto.INT32),
            helper.make_node("ReduceSum", ["interior_i"], ["removed"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["removed", zero_i], ["removed_b"]),
            helper.make_node("Or", [ch0, "removed_b"], ["out_bg"]),
            helper.make_node("Not", ["interior"], ["keep_fg"]),
            helper.make_node("And", [fgb, "keep_fg"], ["out_fg"]),
            helper.make_node("Concat", ["out_bg", "out_fg"], ["out_bool"], axis=1),
            helper.make_node("Cast", ["out_bool"], [OUT_NAME], to=TensorProto.FLOAT),
        ]
    )
    return _finish(nodes, inits, "task098_avgpool")


VARIANTS: dict[str, Callable[[], onnx.ModelProto]] = {
    "float16_avg_id": build_float16_avg_id_model,
    "int32_argmax": build_int32_argmax_model,
    "int64_argmax": build_int64_argmax_model,
    "channel_bool": build_channel_bool_model,
    "avgpool": build_avgpool_model,
}


def solve_grid(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.uint8)
    out = arr.copy()
    h, w = arr.shape
    if h < 3 or w < 3:
        return out
    c = arr[1:-1, 1:-1]
    interior = (
        (c != 0)
        & (c == arr[:-2, 1:-1])
        & (c == arr[2:, 1:-1])
        & (c == arr[1:-1, :-2])
        & (c == arr[1:-1, 2:])
    )
    out[1:-1, 1:-1][interior] = 0
    return out


def onehot(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(H, arr.shape[0])):
        for c in range(min(W, arr.shape[1])):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def validate_model(model: onnx.ModelProto, *, random_cases: int = 64) -> tuple[bool, str]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        session = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        return False, f"ORT load failed: {exc}"

    cases: list[np.ndarray] = []
    if DATA_PATH.is_file():
        with DATA_PATH.open(encoding="utf-8") as fh:
            data = json.load(fh)
        for split in ("train", "test", "arc-gen"):
            for ex in data.get(split, []):
                grid = np.asarray(ex["input"], dtype=np.uint8)
                if max(grid.shape) <= 30:
                    cases.append(grid)

    rng = np.random.default_rng(98)
    for _ in range(random_cases):
        cases.append(random_rect_grid(rng))

    for idx, grid in enumerate(cases):
        expected = onehot(solve_grid(grid))
        pred = session.run([OUT_NAME], {IN_NAME: onehot(grid)})[0]
        if not np.array_equal(pred > 0.0, expected > 0.0):
            return False, f"case {idx} failed shape={grid.shape}"
    return True, f"{len(cases)}/{len(cases)}"


def random_rect_grid(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(3, 31))
    w = int(rng.integers(3, 31))
    grid = np.zeros((h, w), dtype=np.uint8)
    colors = list(range(1, C))
    rng.shuffle(colors)
    attempts = 0
    placed = 0
    target = int(rng.integers(1, 7))
    while placed < target and attempts < 200:
        attempts += 1
        rh = int(rng.integers(1, min(8, h) + 1))
        rw = int(rng.integers(1, min(8, w) + 1))
        r0 = int(rng.integers(0, h - rh + 1))
        c0 = int(rng.integers(0, w - rw + 1))
        # Keep a one-cell gap so random fixtures match the task's separated rectangles.
        rr0, rr1 = max(0, r0 - 1), min(h, r0 + rh + 1)
        cc0, cc1 = max(0, c0 - 1), min(w, c0 + rw + 1)
        if np.any(grid[rr0:rr1, cc0:cc1]):
            continue
        grid[r0 : r0 + rh, c0 : c0 + rw] = colors[placed % len(colors)]
        placed += 1
    return grid


def _profile_largest_internal(path: Path) -> tuple[int | None, str, int]:
    import graph_onnx_memory

    try:
        _model, tensors, _nodes, _inputs = graph_onnx_memory.analyze(path, fast=False)
    except Exception:  # noqa: BLE001
        return None, "", 0
    scored = [item for item in tensors.values() if item.scored]
    if not scored:
        return 0, "", 0
    largest = max(scored, key=lambda item: item.bytes)
    return sum(item.bytes for item in scored), largest.name, largest.bytes


def benchmark_variants() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="task098_") as td:
        tmpdir = Path(td)
        for name, builder in VARIANTS.items():
            path = tmpdir / f"{TASK_ID}.onnx"
            model = builder()
            onnx.save(model, path)
            ok, detail = validate_model(model)
            result = score_file(path)
            memory2, largest_name, largest_bytes = _profile_largest_internal(path)
            rows.append(
                {
                    "variant": name,
                    "ok": ok,
                    "detail": detail,
                    "model": model,
                    "path": path,
                    "valid": result["valid"],
                    "memory": result.get("memory"),
                    "params": result.get("params"),
                    "cost": result.get("cost"),
                    "score": result.get("score"),
                    "error": result.get("error"),
                    "largest": largest_name,
                    "largest_bytes": largest_bytes,
                    "profile_memory": memory2,
                }
            )
    return rows


def choose_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [r for r in rows if r["ok"] and r["valid"] and r["cost"] is not None]
    if not candidates:
        raise SystemExit("no correct and measurable task098 variant")
    return min(candidates, key=lambda r: int(r["cost"]))


def main() -> None:
    rows = benchmark_variants()
    print(f"{'variant':<16} {'ok':<5} {'valid':<6} {'memory':>8} {'params':>7} {'cost':>8} {'score':>10} largest")
    for r in rows:
        score_s = "-" if r["score"] is None else f"{float(r['score']):.6f}"
        print(
            f"{r['variant']:<16} {str(r['ok']):<5} {str(r['valid']):<6} "
            f"{str(r['memory']):>8} {str(r['params']):>7} {str(r['cost']):>8} "
            f"{score_s:>10} {r['largest']}:{r['largest_bytes']}"
        )
        if not r["ok"] or not r["valid"]:
            error_lines = str(r["error"] or "").strip().splitlines()
            error = error_lines[0] if error_lines else ""
            print(f"  note: {r['detail']} {error}")

    best = choose_best(rows)
    model = best["model"]
    onnx.save(model, BEST_PATH)
    shutil.copy2(BEST_PATH, ROOT_PATH)

    correctness_ok, correctness, _passed, _total = verify_correctness(BEST_PATH)
    result = score_file(BEST_PATH)
    params = calculate_params(sanitize_model(model) or model)
    print()
    print(f"best variant: {best['variant']}")
    print(f"correctness:  {correctness} ({correctness_ok})")
    print(f"params check: {params}")
    print_report(result)
    print(f"copied root model: {ROOT_PATH}")

    if not correctness_ok or not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
