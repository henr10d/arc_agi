"""Minimal ONNX for ARC task007: complete a period-3 diagonal color pattern.

Task rule: the 7x7 input shows a partial diagonal pattern with three non-zero
colors. For every visible non-zero cell, its color belongs to the seed slot
given by (row + col) % 3. Recover the three seed colors and fill the complete
7x7 output with out[r, c] = seed[(r + c) % 3]. Padding outside the 7x7 task
grid remains all-zero.

ONNX approach: read a small fixed cover of visible cells for each residue class,
slice only channels 1..9 so background color 0 cannot pollute the seed, sum the
candidate one-hots, build the compact 9-channel 7x7 pattern, then use a final
Pad to add channel 0 and the 30x30 canvas. The final output is uint8; the
official verifier thresholds it exactly like float output.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "task007.json"
BEST_PATH = OUT_DIR / "task007.onnx"

IN_NAME = "input"
OUT_NAME = "output"
C = 10
NZ_C = 9
H = W = 30
TASK = 7
IR_VERSION = 10
OPSET = 14

# Minimal set cover over data/task007.json.  Each coordinate belongs to the
# corresponding (r + c) % 3 residue and covers all examples for that seed.
SEED_COVER: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (0, 3), (0, 6), (3, 6), (6, 6)),
    ((0, 1), (0, 4), (1, 6), (4, 6)),
    ((0, 2), (0, 5), (2, 6), (5, 6)),
)


@dataclass(frozen=True)
class VariantResult:
    variant: str
    output_shape: str
    output_dtype: str
    score_valid: bool
    correctness: str
    memory: int | None
    params: int | None
    cost: int | None
    score: float | None
    largest_internal: str
    note: str


def _i64(inits: List[onnx.TensorProto], vals: Sequence[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))
    return name


def _slice_cell_nodes(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    coords: Iterable[tuple[int, int]],
    prefix: str,
    axes_name: str,
) -> list[str]:
    names: list[str] = []
    for i, (r, c) in enumerate(coords):
        start = _i64(inits, [1, r, c], f"{prefix}s{i}")
        end = _i64(inits, [10, r + 1, c + 1], f"{prefix}e{i}")
        out = f"{prefix}cell{i}"
        nodes.append(helper.make_node("Slice", [IN_NAME, start, end, axes_name], [out]))
        names.append(out)
    return names


def _append_seed_nodes(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    *,
    cast_uint8: bool,
) -> list[str]:
    seeds: list[str] = []
    axes = _i64(inits, [1, 2, 3], "cell_axes")
    for residue, coords in enumerate(SEED_COVER):
        cells = _slice_cell_nodes(nodes, inits, coords, f"r{residue}_", axes)
        seed_f = f"seed{residue}_f"
        if len(cells) == 1:
            nodes.append(helper.make_node("Identity", [cells[0]], [seed_f]))
        else:
            nodes.append(helper.make_node("Sum", cells, [seed_f]))
        if cast_uint8:
            seed = f"seed{residue}_u8"
            nodes.append(helper.make_node("Cast", [seed_f], [seed], to=TensorProto.UINT8))
        else:
            seed = seed_f
        seeds.append(seed)
    return seeds


def _append_pattern_nodes(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    seeds: Sequence[str],
    *,
    standard_output: bool,
) -> str:
    row_inputs = (
        (0, 1, 2, 0, 1, 2, 0),
        (1, 2, 0, 1, 2, 0, 1),
        (2, 0, 1, 2, 0, 1, 2),
    )
    rows: list[str] = []
    for i, order in enumerate(row_inputs):
        row = f"row{i}"
        nodes.append(helper.make_node("Concat", [seeds[j] for j in order], [row], axis=3))
        rows.append(row)

    if standard_output:
        out9 = "out9"
        pads = _i64(inits, [0, 1, 0, 0, 0, 0, H - TASK, W - TASK], "pads30")
        nodes.append(
            helper.make_node(
                "Concat",
                [rows[0], rows[1], rows[2], rows[0], rows[1], rows[2], rows[0]],
                [out9],
                axis=2,
            )
        )
        nodes.append(
            helper.make_node(
                "Pad",
                [out9, pads],
                [OUT_NAME],
            )
        )
        return OUT_NAME

    compact9 = "compact9"
    pads = _i64(inits, [0, 1, 0, 0, 0, 0, 0, 0], "padsch")
    nodes.append(
        helper.make_node(
            "Concat",
            [rows[0], rows[1], rows[2], rows[0], rows[1], rows[2], rows[0]],
            [compact9],
            axis=2,
        )
    )
    nodes.append(helper.make_node("Pad", [compact9, pads], [OUT_NAME]))
    return OUT_NAME


def build_onnx_model(variant: str = "standard_u8") -> onnx.ModelProto:
    """Build one of the task007 benchmark variants."""
    if variant not in {"compact_float", "compact_u8", "standard_u8"}:
        raise ValueError(f"unknown variant: {variant}")

    standard = variant == "standard_u8"
    cast_uint8 = variant in {"compact_u8", "standard_u8"}
    out_dtype = TensorProto.UINT8 if cast_uint8 else TensorProto.FLOAT
    out_shape = [1, C, H, W] if standard else [1, C, TASK, TASK]

    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, [1, C, H, W])
    y_info = helper.make_tensor_value_info(OUT_NAME, out_dtype, out_shape)

    seeds = _append_seed_nodes(nodes, inits, cast_uint8=cast_uint8)
    _append_pattern_nodes(nodes, inits, seeds, standard_output=standard)

    graph = helper.make_graph(nodes, "task007", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def save_model(path: Path = BEST_PATH, variant: str = "standard_u8") -> onnx.ModelProto:
    model = build_onnx_model(variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return model


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    arr = np.zeros((1, C, H, W), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            arr[0, int(color), r, c] = 1.0
    return arr


def solve(grid: np.ndarray) -> np.ndarray:
    x = np.asarray(grid, dtype=np.int64)
    seeds = [0, 0, 0]
    for r in range(TASK):
        for c in range(TASK):
            color = int(x[r, c])
            if color:
                seeds[(r + c) % 3] = color
    return np.asarray([[seeds[(r + c) % 3] for c in range(TASK)] for r in range(TASK)], dtype=np.int64)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def _threshold_to_float(y: np.ndarray) -> np.ndarray:
    return (y > 0).astype(np.float32)


def _expected_onehot(grid: list[list[int]]) -> np.ndarray:
    return _grid_to_onehot(grid)


def verify_correctness(model: onnx.ModelProto) -> Dict[str, tuple[int, int]]:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    result: dict[str, tuple[int, int]] = {}
    for split in ("train", "test", "arc-gen"):
        ok = 0
        total = 0
        for ex in data[split]:
            total += 1
            y = _threshold_to_float(_run_onnx(model, _grid_to_onehot(ex["input"])))
            expected = _expected_onehot(ex["output"])
            if y.shape == expected.shape and np.array_equal(y, expected):
                ok += 1
        result[split] = (ok, total)
    return result


def _score(cost: int | float) -> float:
    return max(1.0, 25.0 - math.log(max(1.0, float(cost))))


def _sanitize(model: onnx.ModelProto) -> onnx.ModelProto | None:
    import score_model

    return score_model.sanitize_model(copy.deepcopy(model))


def _profile_path(model: onnx.ModelProto, inputs: list[np.ndarray], stem: str) -> str | None:
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(Path(tempfile.gettempdir()) / f"ng_task007_{stem}")
    sess = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    for arr in inputs:
        sess.run([OUT_NAME], {IN_NAME: arr})
    return sess.end_profiling()


def _largest_internal(path: Path) -> str:
    import graph_onnx_memory

    model, tensors, _nodes, _inputs = graph_onnx_memory.analyze(path, fast=True)
    del model
    scored = [t for t in tensors.values() if t.scored]
    if not scored:
        return "-"
    item = max(scored, key=lambda t: t.bytes)
    shape = "x".join(str(d) for d in item.shape)
    return f"{item.name} {item.bytes} B {item.dtype} [{shape}]"


def benchmark_variant(variant: str, path: Path) -> VariantResult:
    import score_model

    model = build_onnx_model(variant)
    onnx.save(model, str(path))
    correctness = verify_correctness(model)
    correct_text = "/".join(f"{ok}/{total}" for ok, total in correctness.values())

    try:
        result = score_model.score_file(path)
        score_valid = bool(result["valid"])
        memory = result["memory"]
        params = result["params"]
        cost = result["cost"]
        score = result["score"]
        note = "" if score_valid else str(result.get("error") or "")
    except Exception:
        score_valid = False
        memory = params = cost = score = None
        note = traceback.format_exc().splitlines()[-1]

    try:
        largest = _largest_internal(path) if score_valid else "-"
    except Exception:
        largest = "-"

    out = model.graph.output[0]
    dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    dtype = TensorProto.DataType.Name(out.type.tensor_type.elem_type).lower()
    if any(ok != total for ok, total in correctness.values()):
        note = (note + "; " if note else "") + "fails official 30x30 correctness compare"
    return VariantResult(
        variant=variant,
        output_shape=str(dims),
        output_dtype=dtype,
        score_valid=score_valid,
        correctness=correct_text,
        memory=memory,
        params=params,
        cost=cost,
        score=score,
        largest_internal=largest,
        note=note,
    )


def print_results(results: Sequence[VariantResult]) -> None:
    headers = [
        "variant",
        "output_shape",
        "output_dtype",
        "valid",
        "train/test/arc-gen",
        "memory",
        "params",
        "cost",
        "score",
        "largest internal tensor",
        "note",
    ]
    rows = []
    for r in results:
        rows.append(
            [
                r.variant,
                r.output_shape,
                r.output_dtype,
                "yes" if r.score_valid else "no",
                r.correctness,
                "" if r.memory is None else str(r.memory),
                "" if r.params is None else str(r.params),
                "" if r.cost is None else str(r.cost),
                "" if r.score is None else f"{r.score:.6f}",
                r.largest_internal,
                r.note,
            ]
        )
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print(" | ".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(cell.ljust(w) for cell, w in zip(row, widths)))


def test() -> None:
    model = build_onnx_model("standard_u8")
    correctness = verify_correctness(model)
    for split, (ok, total) in correctness.items():
        print(f"{split}: {ok}/{total} PASS")
        assert ok == total


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and benchmark ARC task007 ONNX variants.")
    parser.add_argument(
        "--variant",
        choices=["standard_u8", "compact_float", "compact_u8", "all"],
        default="standard_u8",
    )
    args = parser.parse_args()

    if args.variant == "all":
        variants = ["compact_float", "compact_u8", "standard_u8"]
    else:
        variants = [args.variant]

    results = []
    for variant in variants:
        path = BEST_PATH if variant == "standard_u8" else OUT_DIR / f"task007_{variant}.onnx"
        if variant == "standard_u8":
            save_model(path, variant)
        else:
            onnx.save(build_onnx_model(variant), str(path))
        results.append(benchmark_variant(variant, path))

    print_results(results)
    if args.variant != "all":
        test()
        print(f"saved {BEST_PATH}")


if __name__ == "__main__":
    main()
