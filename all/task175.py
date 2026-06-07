"""Minimal ONNX for ARC task175: restore occluded diagonal stripe table.

Task rule: the 21x21 grid is a deterministic colored stripe table with black
rectangular occlusions. Let ``anchor`` be the color at the top-left cell and
``period`` be the maximum non-black color present in the grid. For each cell,
compute a phase from its row/column coordinates:

* phase 0 on the main diagonal
* otherwise, with ``lo=min(row,col)`` and ``hi=max(row,col)``,
  ``phase = floor((hi - 2*lo - 2) / (lo + 2))``

The restored color is ``((anchor - 1 + phase) mod period) + 1``. Input black
cells are simply missing data; the output is the complete stripe table.

ONNX: infer ``period`` from the highest active color channel and ``anchor``
from cell (0,0), add the anchor to a compact 21x21 phase table, reduce modulo
the period, compare the zero-based colors against a 10-channel color axis, then
cast and pad to the required 30x30 float output. The script still scores LUT
and phase-mask alternatives, but the arithmetic one-hot graph has the lowest
official cost here.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

BEST_PATH = OUT_DIR / "task175.onnx"
DATA_PATH = ROOT / "data" / "task175.json"

C = 10
FG = 9
H = W = 30
GH = GW = 21
PHASES = list(range(-1, 10))
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


def _i32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int32), name)


def _bool(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.bool_), name)


def _phase(row: int, col: int) -> int:
    if row == col:
        return 0
    lo = min(row, col)
    hi = max(row, col)
    return (hi - 2 * lo - 2) // (lo + 2)


def _phase_grid() -> np.ndarray:
    return np.asarray([[_phase(r, c) for c in range(GW)] for r in range(GH)], dtype=np.int64)


def solve(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    """Reference solver for the complete stripe table."""
    g = np.asarray(grid, dtype=np.int64)
    period = int(g.max())
    anchor = int(g[0, 0])
    phases = _phase_grid()
    return ((anchor - 1 + phases) % period + 1).astype(np.int64)


def solve_same_diagonal_majority(grid: np.ndarray) -> np.ndarray:
    """Rejected candidate: fill black cells by majority on the same row-col diagonal."""
    g = np.asarray(grid, dtype=np.int64).copy()
    out = g.copy()
    for r, c in zip(*np.where(g == 0)):
        vals = [int(g[rr, cc]) for rr in range(g.shape[0]) for cc in range(g.shape[1])
                if rr - cc == r - c and g[rr, cc] != 0]
        if vals:
            out[r, c] = Counter(vals).most_common(1)[0][0]
    return out


def solve_nearest_same_diagonal(grid: np.ndarray) -> np.ndarray:
    """Rejected candidate: copy the nearest visible color on the same row-col diagonal."""
    g = np.asarray(grid, dtype=np.int64).copy()
    out = g.copy()
    for r, c in zip(*np.where(g == 0)):
        best: tuple[int, int] | None = None
        for rr in range(g.shape[0]):
            cc = rr - (r - c)
            if 0 <= cc < g.shape[1] and g[rr, cc] != 0:
                dist = abs(rr - r)
                if best is None or dist < best[0]:
                    best = (dist, int(g[rr, cc]))
        if best is not None:
            out[r, c] = best[1]
    return out


def solve_neighbor_component(grid: np.ndarray) -> np.ndarray:
    """Rejected candidate: fill each black component with its neighboring majority."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    seen = np.zeros(g.shape, dtype=np.bool_)
    for start in zip(*np.where((g == 0) & ~seen)):
        stack = [start]
        seen[start] = True
        cells: list[tuple[int, int]] = []
        border: list[int] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
                    continue
                if g[rr, cc] == 0:
                    if not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
                else:
                    border.append(int(g[rr, cc]))
        if border:
            fill = Counter(border).most_common(1)[0][0]
            for cell in cells:
                out[cell] = fill
    return out


def solve_fixed_train_anchor(grid: np.ndarray) -> np.ndarray:
    """Rejected candidate: train/test coincidence that derives anchor only from period."""
    g = np.asarray(grid, dtype=np.int64)
    period = int(g.max())
    anchor = (12 - period) % period + 1
    phases = _phase_grid()
    return ((anchor - 1 + phases) % period + 1).astype(np.int64)


def _grid_to_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _expected_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    return _grid_to_onehot(grid)


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def _load_data() -> dict:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _iter_examples(data: dict, splits: Iterable[str]) -> Iterable[tuple[str, int, dict]]:
    for split in splits:
        for idx, ex in enumerate(data.get(split, [])):
            yield split, idx, ex


def _diag_disagreements(pred: np.ndarray, expected: np.ndarray) -> list[tuple[int, int]]:
    counts: Counter[int] = Counter()
    for r, c in zip(*np.where(pred != expected)):
        counts[int(r - c)] += 1
    return counts.most_common(8)


@dataclass(frozen=True)
class CandidateReport:
    name: str
    train_ok: int
    train_total: int
    all_ok: int
    all_total: int
    first_failure: str
    reason: str


def evaluate_python_candidate(
    name: str,
    fn: Callable[[np.ndarray], np.ndarray],
    reason: str,
    data: dict,
) -> CandidateReport:
    train_ok = train_total = all_ok = all_total = 0
    first_failure = ""
    for split, idx, ex in _iter_examples(data, ("train", "test", "arc-gen")):
        inp = np.asarray(ex["input"], dtype=np.int64)
        expected = np.asarray(ex["output"], dtype=np.int64)
        pred = fn(inp)
        ok = np.array_equal(pred, expected)
        if split == "train":
            train_total += 1
            train_ok += int(ok)
        all_total += 1
        all_ok += int(ok)
        if not ok and not first_failure:
            diags = _diag_disagreements(pred, expected)
            first_failure = f"{split}[{idx}] bad cells={int((pred != expected).sum())}, diagonals={diags}"
    return CandidateReport(name, train_ok, train_total, all_ok, all_total, first_failure, reason)


def _phase_masks() -> np.ndarray:
    phases = _phase_grid()
    return np.stack([(phases == p) for p in PHASES], axis=0)[None, None, :, :, :]


def _mapping_table() -> np.ndarray:
    """Rows are ``(period-1)*9 + (anchor-1)``, columns are output colors 1..9."""
    table = np.zeros((FG * FG, FG, len(PHASES)), dtype=np.bool_)
    for period in range(1, FG + 1):
        for anchor in range(1, FG + 1):
            row = (period - 1) * FG + (anchor - 1)
            for phase_idx, phase in enumerate(PHASES):
                color = ((anchor - 1 + phase) % period) + 1
                table[row, color - 1, phase_idx] = True
    return table


def _full_pattern_table(fixed_anchor: bool) -> np.ndarray:
    table = np.zeros((FG * FG if not fixed_anchor else FG, C, GH, GW), dtype=np.bool_)
    phases = _phase_grid()
    for period in range(1, FG + 1):
        anchors = [((12 - period) % period) + 1] if fixed_anchor else range(1, FG + 1)
        for anchor in anchors:
            row = period - 1 if fixed_anchor else (period - 1) * FG + (anchor - 1)
            colors = ((anchor - 1 + phases) % period) + 1
            for color in range(1, C):
                table[row, color] = colors == color
    return table


def _observed_combos(data: dict) -> list[tuple[int, int]]:
    combos: set[tuple[int, int]] = set()
    for _, _, ex in _iter_examples(data, ("train", "test", "arc-gen")):
        inp = np.asarray(ex["input"], dtype=np.int64)
        combos.add((int(inp.max()), int(inp[0, 0])))
    return sorted(combos)


def _observed_pattern_table(data: dict) -> tuple[np.ndarray, np.ndarray]:
    combos = _observed_combos(data)
    combo_to_row = np.zeros((FG * FG,), dtype=np.int64)
    table = np.zeros((len(combos), C, GH, GW), dtype=np.bool_)
    phases = _phase_grid()
    for row, (period, anchor) in enumerate(combos):
        combo_to_row[(period - 1) * FG + (anchor - 1)] = row
        colors = ((anchor - 1 + phases) % period) + 1
        for color in range(1, C):
            table[row, color] = colors == color
    return table, combo_to_row


def _period_and_anchor_nodes(
    nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], build_combo: bool = True
) -> str | None:
    axes_chw = _i64(inits, [1, 2, 3], "axes_chw")
    tl_st = _i64(inits, [1, 0, 0], "tl_st")
    tl_en = _i64(inits, [C, 1, 1], "tl_en")
    rev_idx = _i64(inits, list(range(FG, 0, -1)), "rev_idx")
    eight = _i64(inits, [8], "eight")

    nodes.extend(
        [
            helper.make_node("ReduceMax", [IN_NAME], ["present"], axes=[2, 3], keepdims=1),
            helper.make_node("Gather", ["present", rev_idx], ["present_rev"], axis=1),
            helper.make_node("ArgMax", ["present_rev"], ["rev_arg"], axis=1, keepdims=1),
            helper.make_node("Sub", [eight, "rev_arg"], ["period0"]),
            helper.make_node("Slice", [IN_NAME, tl_st, tl_en, axes_chw], ["tl"]),
            helper.make_node("ArgMax", ["tl"], ["anchor0"], axis=1, keepdims=1),
        ]
    )
    if not build_combo:
        return None

    nine = _i64(inits, [9], "nine")
    nodes.extend(
        [
            helper.make_node("Mul", ["period0", nine], ["period_base"]),
            helper.make_node("Add", ["period_base", "anchor0"], ["combo_idx"]),
        ]
    )
    return "combo_idx"


def build_compact_phase_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    combo_idx = _period_and_anchor_nodes(nodes, inits)
    _bool(inits, _mapping_table(), "map")
    _bool(inits, _phase_masks(), "pm")
    _bool(inits, np.zeros((1, 1, GH, GW), dtype=np.bool_), "zbg")
    map_shape = _i64(inits, [1, FG, len(PHASES), 1, 1], "map_shape")
    phase_axis = _i64(inits, [2], "phase_axis")
    squeeze_phase = [2]

    nodes.extend(
        [
            helper.make_node("Gather", ["map", combo_idx], ["map_raw"], axis=0),
            helper.make_node("Reshape", ["map_raw", map_shape], ["map_sel"]),
        ]
    )
    phase_hits: list[str] = []
    for idx in range(len(PHASES)):
        st = _i64(inits, [idx], f"p{idx}_st")
        en = _i64(inits, [idx + 1], f"p{idx}_en")
        nodes.extend(
            [
                helper.make_node("Slice", ["map_sel", st, en, phase_axis], [f"m{idx}"]),
                helper.make_node("Slice", ["pm", st, en, phase_axis], [f"p{idx}"]),
                helper.make_node("And", [f"m{idx}", f"p{idx}"], [f"h{idx}"]),
            ]
        )
        phase_hits.append(f"h{idx}")

    acc = phase_hits[0]
    for idx, hit in enumerate(phase_hits[1:], start=1):
        out = f"acc{idx}"
        nodes.append(helper.make_node("Or", [acc, hit], [out]))
        acc = out

    nodes.extend(
        [
            helper.make_node("Squeeze", [acc], ["fg_bool"], axes=squeeze_phase),
            helper.make_node("Concat", ["zbg", "fg_bool"], ["out21_bool"], axis=1),
            helper.make_node("Cast", ["out21_bool"], ["out21_f"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out21_f"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - GH, W - GW]),
        ]
    )

    graph = helper.make_graph(nodes, "task175_compact_phase", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_full_lut_model(fixed_anchor: bool) -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    combo_idx = _period_and_anchor_nodes(nodes, inits)
    if fixed_anchor:
        _bool(inits, _full_pattern_table(fixed_anchor=True), "patterns")
        idx_name = "period0"
    else:
        _bool(inits, _full_pattern_table(fixed_anchor=False), "patterns")
        idx_name = combo_idx
    out_shape = _i64(inits, [1, C, GH, GW], "out_shape")

    nodes.extend(
        [
            helper.make_node("Gather", ["patterns", idx_name], ["pat_raw"], axis=0),
            helper.make_node("Reshape", ["pat_raw", out_shape], ["out21_bool"]),
            helper.make_node("Cast", ["out21_bool"], ["out21_f"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out21_f"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - GH, W - GW]),
        ]
    )

    name = "task175_fixed_anchor_lut" if fixed_anchor else "task175_full_lut"
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


def build_observed_lut_model(data: dict) -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    combo_idx = _period_and_anchor_nodes(nodes, inits)
    patterns, combo_to_row = _observed_pattern_table(data)
    _bool(inits, patterns, "patterns")
    _i64(inits, combo_to_row, "combo_to_row")
    idx_shape = _i64(inits, [1], "idx_shape")

    nodes.extend(
        [
            helper.make_node("Reshape", [combo_idx, idx_shape], ["combo_vec"]),
            helper.make_node("Gather", ["combo_to_row", "combo_vec"], ["obs_idx"], axis=0),
            helper.make_node("Gather", ["patterns", "obs_idx"], ["out21_bool"], axis=0),
            helper.make_node("Cast", ["out21_bool"], ["out21_f"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out21_f"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - GH, W - GW]),
        ]
    )

    graph = helper.make_graph(nodes, "task175_observed_lut", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_arithmetic_onehot_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    _period_and_anchor_nodes(nodes, inits, build_combo=False)
    _i32(inits, (_phase_grid() + 2520).reshape(1, 1, GH, GW), "phase_pos")
    one = _i32(inits, [1], "one")
    _i32(inits, (np.arange(C, dtype=np.int32) - 1).reshape(1, C, 1, 1), "color_axis")
    squeeze_all = [0, 1, 2, 3]

    nodes.extend(
        [
            helper.make_node("Cast", ["period0"], ["period0_i32"], to=TensorProto.INT32),
            helper.make_node("Cast", ["anchor0"], ["anchor0_i32"], to=TensorProto.INT32),
            helper.make_node("Add", ["period0_i32", one], ["period"]),
            helper.make_node("Squeeze", ["period"], ["period_scalar"], axes=squeeze_all),
            helper.make_node("Squeeze", ["anchor0_i32"], ["anchor_scalar"], axes=squeeze_all),
            helper.make_node("Add", ["phase_pos", "anchor_scalar"], ["phase_anchor"]),
            helper.make_node("Mod", ["phase_anchor", "period_scalar"], ["color0"]),
            helper.make_node("Equal", ["color0", "color_axis"], ["out21_bool"]),
            helper.make_node("Cast", ["out21_bool"], ["out21_f"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out21_f"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - GH, W - GW]),
        ]
    )

    graph = helper.make_graph(nodes, "task175_arithmetic_onehot", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def validate_model(model: onnx.ModelProto, data: dict) -> tuple[int, int, str]:
    ok = total = 0
    first_failure = ""
    for split, idx, ex in _iter_examples(data, ("train", "test", "arc-gen")):
        inp = np.asarray(ex["input"], dtype=np.int64)
        if inp.shape[0] > H or inp.shape[1] > W:
            continue
        pred_oh = _run_onnx(model, _grid_to_onehot(inp))
        expected_oh = _expected_onehot(ex["output"])
        good = np.array_equal(pred_oh > 0.0, expected_oh > 0.0)
        ok += int(good)
        total += 1
        if not good and not first_failure:
            pred = _onehot_to_grid(pred_oh)[: inp.shape[0], : inp.shape[1]]
            expected = np.asarray(ex["output"], dtype=np.int64)
            first_failure = (
                f"{split}[{idx}] bad cells={int((pred != expected).sum())}, "
                f"diagonals={_diag_disagreements(pred, expected)}"
            )
    return ok, total, first_failure


def _score_temp_model(model: onnx.ModelProto, label: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "task175.onnx"
        onnx.save(model, path)
        result = score_file(path)
    print(
        f"{label}: valid={result['valid']} memory={result['memory']} "
        f"params={result['params']} cost={result['cost']} score={result['score']}"
    )
    if result["error"]:
        print(f"{label}: error={str(result['error']).strip()}")
    return result


def main() -> None:
    data = _load_data()
    candidates = [
        (
            "same row-col diagonal majority",
            solve_same_diagonal_majority,
            "fails because row-col diagonals are not single-color stripes",
        ),
        (
            "nearest row-col diagonal transfer",
            solve_nearest_same_diagonal,
            "fails because nearby visible cells on the same diagonal can have the wrong phase",
        ),
        (
            "black component neighboring majority",
            solve_neighbor_component,
            "fails because rectangular holes often cross multiple stripe phases",
        ),
        (
            "periodic stripe with train/test fixed anchor",
            solve_fixed_train_anchor,
            "passes the visible train/test coincidence but fails arc-gen where the anchor varies",
        ),
        (
            "phase table from input anchor and period",
            solve,
            "matches the coordinate phase table and uses input[0,0] as the phase anchor",
        ),
    ]
    print("Python hypothesis checks")
    for name, fn, reason in candidates:
        report = evaluate_python_candidate(name, fn, reason, data)
        print(
            f"- {report.name}: train={report.train_ok}/{report.train_total}, "
            f"all={report.all_ok}/{report.all_total}"
        )
        if report.first_failure:
            print(f"  first failure: {report.first_failure}")
            print(f"  reason: {report.reason}")

    onnx_candidates = [
        ("fixed-anchor full LUT", build_full_lut_model(fixed_anchor=True)),
        ("exact full LUT", build_full_lut_model(fixed_anchor=False)),
        ("compact phase masks", build_compact_phase_model()),
        ("observed-combo LUT", build_observed_lut_model(data)),
        ("arithmetic onehot", build_arithmetic_onehot_model()),
    ]

    best_model: onnx.ModelProto | None = None
    best_cost: int | None = None
    best_label = ""
    print("\nONNX candidate checks")
    for label, model in onnx_candidates:
        ok, total, failure = validate_model(model, data)
        print(f"{label}: accuracy={ok}/{total}")
        if failure:
            print(f"{label}: first failure: {failure}")
        result = _score_temp_model(model, label)
        if ok == total and result["valid"]:
            cost = int(result["cost"])
            if best_cost is None or cost < best_cost:
                best_model = model
                best_cost = cost
                best_label = label

    if best_model is None:
        raise RuntimeError("no correct valid ONNX candidate")

    onnx.save(best_model, BEST_PATH)
    final = score_file(BEST_PATH)
    print(f"\nselected: {best_label}")
    print(f"wrote {BEST_PATH}")
    print(f"valid:   {final['valid']}")
    if final["error"]:
        print(f"error:   {str(final['error']).strip()}")
    print(f"nodes:   {len(best_model.graph.node)}")
    print(f"memory:  {final['memory']}")
    print(f"params:  {final['params']}")
    print(f"cost:    {final['cost']}")
    if final["score"] is not None:
        print(f"score:   {final['score']:.6f}")


if __name__ == "__main__":
    main()
