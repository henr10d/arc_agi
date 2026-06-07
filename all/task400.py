"""Recover a masked 5x5 patch from a 180-degree symmetric 24x24 grid.

Task rule: the input contains one solid 5x5 marker of color 1 that hides part of
a rotationally symmetric pattern.  Find the marker's top-left corner, take the
5x5 patch at the 180-degree opposite location, rotate that patch by 180 degrees,
and emit it as the 5x5 output.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TASK_ID = "task400"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task400.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

IN_NAME = "input"
OUT_NAME = "output"
C = 10
H = W = 30
N = 24
P = 5
IR_VERSION = 10
OPSET = 10


@dataclass
class Row:
    variant: str
    valid: bool
    splits: dict[str, tuple[int, int]]
    memory: int | None
    params: int | None
    cost: int | None
    score: float | None
    largest: str
    reason: str
    path: Path | None = None


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    arr = np.asarray(vals, dtype=np.int64)
    inits.append(numpy_helper.from_array(arr, name))
    return name


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    arr = np.asarray(vals, dtype=np.float32)
    inits.append(numpy_helper.from_array(arr, name))
    return name


def _bool(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    arr = np.asarray(vals, dtype=np.bool_)
    inits.append(numpy_helper.from_array(arr, name))
    return name


def _slice(
    nodes: list[onnx.NodeProto],
    x: str,
    out: str,
    starts: str,
    ends: str,
    axes: str,
) -> str:
    nodes.append(helper.make_node("Slice", [x, starts, ends, axes], [out]))
    return out


def _gather(nodes: list[onnx.NodeProto], x: str, idx: str, axis: int, out: str) -> str:
    nodes.append(helper.make_node("Gather", [x, idx], [out], axis=axis))
    return out


def _transform(nodes: list[onnx.NodeProto], x: str, rev: str, kind: str, prefix: str) -> str:
    if kind == "id":
        return x
    if kind == "h":
        return _gather(nodes, x, rev, 3, f"{prefix}_h")
    if kind == "v":
        return _gather(nodes, x, rev, 2, f"{prefix}_v")
    if kind == "rot":
        v = _gather(nodes, x, rev, 2, f"{prefix}_v")
        return _gather(nodes, v, rev, 3, f"{prefix}_r")
    if kind == "t":
        nodes.append(helper.make_node("Transpose", [x], [f"{prefix}_t"], perm=[0, 1, 3, 2]))
        return f"{prefix}_t"
    if kind == "t_h":
        t = _transform(nodes, x, rev, "t", f"{prefix}_t0")
        return _gather(nodes, t, rev, 3, f"{prefix}_th")
    if kind == "t_v":
        t = _transform(nodes, x, rev, "t", f"{prefix}_t0")
        return _gather(nodes, t, rev, 2, f"{prefix}_tv")
    if kind == "t_rot":
        t = _transform(nodes, x, rev, "t", f"{prefix}_t0")
        v = _gather(nodes, t, rev, 2, f"{prefix}_tv")
        return _gather(nodes, v, rev, 3, f"{prefix}_tr")
    raise ValueError(kind)


def _shift_down_right(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    x: str,
    dr: int,
    dc: int,
    prefix: str,
    threshold: str = "z",
) -> str:
    if dr == 0 and dc == 0:
        return x
    starts = _i64(inits, [0, 0, 0, 0], f"{prefix}_s")
    ends = _i64(inits, [1, 1, N - dr, N - dc], f"{prefix}_e")
    axes = _i64(inits, [0, 1, 2, 3], f"{prefix}_a")
    sliced = f"{prefix}_sl"
    nodes.append(helper.make_node("Slice", [x, starts, ends, axes], [sliced]))
    cast = f"{prefix}_sf"
    nodes.append(helper.make_node("Cast", [sliced], [cast], to=TensorProto.FLOAT))
    padded = f"{prefix}_pf"
    nodes.append(
        helper.make_node(
            "Pad",
            [cast],
            [padded],
            mode="constant",
            pads=[0, 0, dr, dc, 0, 0, 0, 0],
            value=0.0,
        )
    )
    out = f"{prefix}_sh"
    nodes.append(helper.make_node("Greater", [padded, threshold], [out]))
    return out


def _or_many(nodes: list[onnx.NodeProto], xs: list[str], prefix: str) -> str:
    cur = xs[0]
    for i, item in enumerate(xs[1:], start=1):
        out = f"{prefix}_{i}"
        nodes.append(helper.make_node("Or", [cur, item], [out]))
        cur = out
    return cur


def build_onnx_model(transforms: tuple[str, ...] = ("id", "h", "v", "rot", "t", "t_h", "t_v", "t_rot")) -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, [1, C, H, W])
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, [1, C, H, W])

    z = _f32(inits, 0.5, "z")
    s24 = _i64(inits, [0, 0, 0, 0], "s24")
    e24 = _i64(inits, [1, C, N, N], "e24")
    ax4 = _i64(inits, [0, 1, 2, 3], "ax4")
    rev = _i64(inits, np.arange(N - 1, -1, -1), "rev")
    keep = _bool(inits, np.array([[[[i != 1]] for i in range(C)]], dtype=np.bool_), "keep")

    _slice(nodes, IN_NAME, "x24", s24, e24, ax4)
    nodes.append(helper.make_node("Greater", ["x24", z], ["xb"]))

    sblue = _i64(inits, [0, 1, 0, 0], "sblue")
    eblue = _i64(inits, [1, 2, N, N], "eblue")
    _slice(nodes, "xb", "blue", sblue, eblue, ax4)

    up = _shift_down_right(nodes, inits, "blue", 1, 0, "up")
    left = _shift_down_right(nodes, inits, "blue", 0, 1, "left")
    nodes.append(helper.make_node("Not", [up], ["nup"]))
    nodes.append(helper.make_node("Not", [left], ["nleft"]))
    nodes.append(helper.make_node("And", ["blue", "nup"], ["top_edge"]))
    nodes.append(helper.make_node("And", ["top_edge", "nleft"], ["tl"]))

    candidates: list[str] = []
    for i, kind in enumerate(transforms):
        tx = _transform(nodes, "xb", rev, kind, f"tr{i}_{kind}")
        kept = f"k{i}"
        nodes.append(helper.make_node("And", [tx, keep], [kept]))
        candidates.append(kept)
    cand = _or_many(nodes, candidates, "cand") if len(candidates) > 1 else candidates[0]

    cells: list[list[str]] = []
    for r in range(P):
        row: list[str] = []
        for c in range(P):
            mask = _shift_down_right(nodes, inits, "tl", r, c, f"m{r}{c}")
            sel = f"sel{r}{c}"
            nodes.append(helper.make_node("And", [cand, mask], [sel]))
            flt = f"flt{r}{c}"
            nodes.append(helper.make_node("Cast", [sel], [flt], to=TensorProto.FLOAT))
            cell = f"cell{r}{c}"
            nodes.append(helper.make_node("ReduceSum", [flt], [cell], axes=[2, 3], keepdims=1))
            row.append(cell)
        cells.append(row)

    row_names: list[str] = []
    for r, row in enumerate(cells):
        row_name = f"row{r}"
        nodes.append(helper.make_node("Concat", row, [row_name], axis=3))
        row_names.append(row_name)
    nodes.append(helper.make_node("Concat", row_names, ["patch"], axis=2))
    nodes.append(
        helper.make_node(
            "Pad",
            ["patch"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - P, W - P],
            value=0.0,
        )
    )

    graph = helper.make_graph(nodes, "task400", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_dynamic_rot_model() -> onnx.ModelProto:
    """Build the compact model used for the delivered ONNX.

    The marker's top-left must lie in the 20x20 region of possible 5x5
    placements.  Row/column projections over just that region find the marker
    start.  The opposite 5x5 patch is then sliced directly in reverse order,
    with a sentinel end value for edge positions where a normal -1 end would be
    interpreted relative to the input dimension by Slice.
    """
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, [1, C, H, W])
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, [1, C, H, W])

    possible = N - P + 1
    axes_spatial = _i64(inits, [2, 3], "axes_spatial")
    sblue = _i64(inits, [0, 1, 0, 0], "sblue")
    eblue = _i64(inits, [1, 2, possible, possible], "eblue")

    nodes.append(helper.make_node("Slice", [IN_NAME, sblue, eblue], ["blue_f"]))
    nodes.append(helper.make_node("ReduceSum", ["blue_f"], ["row_sums"], axes=[0, 1, 3], keepdims=0))
    nodes.append(helper.make_node("ArgMax", ["row_sums"], ["row_idx"], axis=0, keepdims=1))
    nodes.append(helper.make_node("ReduceSum", ["blue_f"], ["col_sums"], axes=[0, 1, 2], keepdims=0))
    nodes.append(helper.make_node("ArgMax", ["col_sums"], ["col_idx"], axis=0, keepdims=1))
    nodes.append(helper.make_node("Concat", ["row_idx", "col_idx"], ["idx"], axis=0))

    rev_start_const = _i64(inits, [N - 1, N - 1], "rev_start_const")
    rev_end_const = _i64(inits, [N - P - 1, N - P - 1], "rev_end_const")
    last_idx = _i64(inits, [possible - 1, possible - 1], "last_idx")
    neg_inf = _i64(inits, [-9223372036854775807, -9223372036854775807], "neg_inf")
    rev_steps = _i64(inits, [-1, -1], "rev_steps")

    nodes.append(helper.make_node("Sub", [rev_start_const, "idx"], ["starts"]))
    nodes.append(helper.make_node("Sub", [rev_end_const, "idx"], ["raw_ends"]))
    nodes.append(helper.make_node("Equal", ["idx", last_idx], ["at_edge"]))
    nodes.append(helper.make_node("Where", ["at_edge", neg_inf, "raw_ends"], ["ends"]))
    nodes.append(helper.make_node("Slice", [IN_NAME, "starts", "ends", axes_spatial, rev_steps], ["patch"]))
    nodes.append(
        helper.make_node(
            "Pad",
            ["patch"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - P, W - P],
            value=0.0,
        )
    )

    value_info = [
        helper.make_tensor_value_info("patch", TensorProto.FLOAT, [1, C, P, P]),
    ]
    graph = helper.make_graph(
        nodes,
        "task400_dynamic_rot",
        [x_info],
        [y_info],
        initializer=inits,
        value_info=value_info,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros((1, C, H, W), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def _expected_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros((1, C, H, W), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def _load_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def validate_model(model: onnx.ModelProto) -> dict[str, tuple[int, int]]:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    data = _load_data()
    result: dict[str, tuple[int, int]] = {}
    for split, examples in data.items():
        ok = 0
        for ex in examples:
            pred = sess.run([OUT_NAME], {IN_NAME: _grid_to_onehot(ex["input"])})[0]
            exp = _expected_onehot(ex["output"])
            ok += bool(np.array_equal(pred > 0.0, exp > 0.0))
        result[split] = (ok, len(examples))
    return result


def _largest_internal(path: Path) -> str:
    try:
        import graph_onnx_memory

        _model, tensors, _nodes, _inputs = graph_onnx_memory.analyze(path, fast=False)
        scored = [t for t in tensors.values() if t.scored]
        if not scored:
            return "none"
        largest = max(scored, key=lambda t: t.bytes)
        return f"{largest.name}:{largest.bytes}B {largest.dtype}{largest.shape}"
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"unavailable: {type(exc).__name__}"


def _score_path(path: Path) -> dict[str, Any]:
    import score_model

    return score_model.score_file(path)


def _score(cost: int | None) -> float | None:
    if cost is None:
        return None
    return max(1.0, 25.0 - math.log(max(1.0, float(cost))))


def run_variant(tmp: Path, name: str, transforms: tuple[str, ...] | None, reason: str) -> Row:
    path = tmp / f"task400_{name}.onnx"
    model = build_dynamic_rot_model() if transforms is None else build_onnx_model(transforms)
    onnx.save(model, str(path))
    splits = validate_model(model)
    valid = all(a == b for a, b in splits.values())
    score = _score_path(path)
    memory = score.get("memory")
    params = score.get("params")
    cost = score.get("cost")
    largest = _largest_internal(path) if score.get("valid") else "-"
    return Row(
        name,
        valid,
        splits,
        memory,
        params,
        cost,
        _score(cost),
        largest,
        "passes" if valid else reason,
        path,
    )


def print_rows(rows: list[Row]) -> None:
    print("variant | valid | train/test/arc-gen | memory | params | cost | score | largest internal tensor | reason")
    for row in rows:
        splits = "/".join(f"{row.splits.get(k, (0, 0))[0]}/{row.splits.get(k, (0, 0))[1]}" for k in ("train", "test", "arc-gen"))
        score = "-" if row.score is None else f"{row.score:.6f}"
        print(
            f"{row.variant} | {row.valid} | {splits} | {row.memory} | {row.params} | "
            f"{row.cost} | {score} | {row.largest} | {row.reason}"
        )


def main() -> None:
    variants: list[tuple[str, tuple[str, ...] | None, str]] = [
        ("h_mirror", ("h",), "horizontal mirror alone disagrees with many examples"),
        ("v_mirror", ("v",), "vertical mirror alone disagrees with many examples"),
        ("rot180", ("rot",), "180-degree mirror alone disagrees with many examples"),
        ("d4_vote", ("id", "h", "v", "rot", "t", "t_h", "t_v", "t_rot"), "all D4 non-blue candidates agree"),
        ("rot180_dynamic_slice", None, "detect top-left and dynamically slice the 180-degree source patch"),
    ]
    rows: list[Row] = []
    with tempfile.TemporaryDirectory(prefix="task400_") as td:
        tmp = Path(td)
        for name, transforms, reason in variants:
            rows.append(run_variant(tmp, name, transforms, reason))
        passing = [row for row in rows if row.valid and row.cost is not None]
        if not passing:
            print_rows(rows)
            raise SystemExit("no passing variant")
        best = min(passing, key=lambda row: int(row.cost or 10**18))
        assert best.path is not None
        BEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(onnx.load(str(best.path)), str(BEST_PATH))

    # Re-score the saved file so the table's best row describes the delivered artifact.
    saved = _score_path(BEST_PATH)
    for row in rows:
        if row.variant == best.variant:
            row.memory = saved.get("memory")
            row.params = saved.get("params")
            row.cost = saved.get("cost")
            row.score = _score(row.cost)
            row.largest = _largest_internal(BEST_PATH)
    print_rows(rows)
    print(f"saved {BEST_PATH}")


if __name__ == "__main__":
    main()
