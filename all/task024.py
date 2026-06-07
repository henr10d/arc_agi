"""Minimal ONNX for ARC task024: extend sparse color seeds into full lines.

Task rule: infer from the training pairs that colors 1 (blue) and 3 (green)
paint their whole row, color 2 (red) paints its whole column, and row paints
take precedence at row/column intersections. The padded NeuroGolf tensor keeps
the original grid extent via channel 0, so the graph builds row/column validity
masks and emits all-zero padding outside the ARC grid.

ONNX: reduce the input once by rows and once by columns. The selected model
builds a compact bool row-side condition [1,10,30,1] and a compact float
column-side value [1,10,1,30], then uses a final broadcast Where named output.
The full 30x30x10 tensor is only the graph output, which is excluded from
NeuroGolf memory scoring.
"""

from __future__ import annotations

import json
import math
import shutil
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
sys.path.insert(0, str(ROOT))

from score_model import calculate_params, sanitize_model, score, score_file  # noqa: E402

TASK_ID = "task024"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C, H, W = 10, 30, 30
SHAPE = [1, C, H, W]
IR_VERSION = 10


@dataclass(frozen=True)
class Rule:
    row_colors: tuple[int, ...]
    col_colors: tuple[int, ...]
    row_over_col: bool


@dataclass(frozen=True)
class Candidate:
    name: str
    opset: int
    robust_validity: bool
    strategy: str


def load_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def solve_grid(grid: list[list[int]], rule: Rule) -> list[list[int]]:
    h, w = len(grid), len(grid[0])
    row_paints: dict[int, int] = {}
    col_paints: dict[int, int] = {}
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if color in rule.row_colors:
                row_paints[r] = color
            elif color in rule.col_colors:
                col_paints[c] = color

    out = [[0 for _ in range(w)] for _ in range(h)]
    if rule.row_over_col:
        for c, color in col_paints.items():
            for r in range(h):
                out[r][c] = color
        for r, color in row_paints.items():
            for c in range(w):
                out[r][c] = color
    else:
        for r, color in row_paints.items():
            for c in range(w):
                out[r][c] = color
        for c, color in col_paints.items():
            for r in range(h):
                out[r][c] = color
    return out


def infer_rule(data: dict[str, list[dict[str, list[list[int]]]]]) -> Rule:
    train = data["train"]
    colors = sorted({v for ex in train for row in ex["input"] for v in row if v})
    best: tuple[int, Rule] | None = None
    for row_mask in range(1, 1 << len(colors)):
        row_colors = tuple(colors[i] for i in range(len(colors)) if row_mask & (1 << i))
        col_colors = tuple(c for c in colors if c not in row_colors)
        if not col_colors:
            continue
        for row_over_col in (True, False):
            rule = Rule(row_colors, col_colors, row_over_col)
            hits = sum(solve_grid(ex["input"], rule) == ex["output"] for ex in train)
            if best is None or hits > best[0]:
                best = (hits, rule)
            if hits == len(train):
                return rule
    if best is None:
        raise RuntimeError("could not infer any line-fill rule")
    raise RuntimeError(f"best inferred rule only matched {best[0]}/{len(train)} train examples: {best[1]}")


def grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    arr = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            arr[0, int(color), r, c] = 1.0
    return arr


def expected_onehot(example: dict[str, list[list[int]]]) -> np.ndarray:
    return grid_to_onehot(example["output"])


def synthetic_examples(rule: Rule, count: int = 128) -> list[dict[str, list[list[int]]]]:
    rng = np.random.default_rng(24024)
    examples: list[dict[str, list[list[int]]]] = []
    row_colors = list(rule.row_colors)
    col_color = rule.col_colors[0]
    for _ in range(count):
        h = int(rng.integers(5, 16))
        w = int(rng.integers(5, 16))
        grid = [[0 for _ in range(w)] for _ in range(h)]
        used: set[tuple[int, int]] = set()
        n_rows = int(rng.integers(1, min(4, h) + 1))
        n_cols = int(rng.integers(1, min(4, w) + 1))
        for r in rng.choice(h, size=n_rows, replace=False):
            c = int(rng.integers(0, w))
            while (int(r), c) in used:
                c = int(rng.integers(0, w))
            used.add((int(r), c))
            grid[int(r)][c] = int(rng.choice(row_colors))
        for c in rng.choice(w, size=n_cols, replace=False):
            r = int(rng.integers(0, h))
            while (r, int(c)) in used:
                r = int(rng.integers(0, h))
            used.add((r, int(c)))
            grid[r][int(c)] = col_color
        examples.append({"input": grid, "output": solve_grid(grid, rule)})
    return examples


def _idx(inits: list[onnx.TensorProto], value: int, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray([value], dtype=np.int64), name=name))
    return name


def _gather(nodes: list[onnx.NodeProto], source: str, idx_name: str, out: str) -> None:
    nodes.append(helper.make_node("Gather", [source, idx_name], [out], axis=1))


def build_model(candidate: Candidate, rule: Rule) -> onnx.ModelProto:
    if tuple(sorted(rule.row_colors)) != (1, 3) or tuple(rule.col_colors) != (2,) or not rule.row_over_col:
        raise ValueError(f"task024 graph builder expects the inferred canonical rule, got {rule}")

    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)

    row_any, col_any = "ra", "ca"
    nodes.append(helper.make_node("ReduceMax", ["input"], [row_any], axes=[3], keepdims=1))
    nodes.append(helper.make_node("ReduceMax", ["input"], [col_any], axes=[2], keepdims=1))

    i1 = _idx(inits, 1, "i1")
    i2 = _idx(inits, 2, "i2")
    i3 = _idx(inits, 3, "i3")

    r1, r3 = "r1", "r3"
    c2 = "c2"
    _gather(nodes, row_any, i1, r1)
    _gather(nodes, row_any, i3, r3)
    _gather(nodes, col_any, i2, c2)

    if candidate.robust_validity:
        rvalid, cvalid = "rv", "cv"
        nodes.append(helper.make_node("ReduceMax", [row_any], [rvalid], axes=[1], keepdims=1))
        nodes.append(helper.make_node("ReduceMax", [col_any], [cvalid], axes=[1], keepdims=1))
    else:
        rvalid, cvalid = "r0", "c0"
        i0 = _idx(inits, 0, "i0")
        _gather(nodes, row_any, i0, rvalid)
        _gather(nodes, col_any, i0, cvalid)

    not_red = "nc"
    nodes.append(helper.make_node("Sub", [cvalid, c2], [not_red]))

    if candidate.strategy == "where_bool_rows":
        r1b, r3b = "r1b", "r3b"
        n1b, n3b, no_row_b, false_b = "n1b", "n3b", "nrb", "fb"
        nodes.append(helper.make_node("Cast", [r1], [r1b], to=TensorProto.BOOL))
        nodes.append(helper.make_node("Cast", [r3], [r3b], to=TensorProto.BOOL))
        nodes.append(helper.make_node("Less", [r1, rvalid], [n1b]))
        nodes.append(helper.make_node("Less", [r3, rvalid], [n3b]))
        nodes.append(helper.make_node("And", [n1b, n3b], [no_row_b]))
        nodes.append(helper.make_node("Less", [r1, r1], [false_b]))

        a_inputs = [
            no_row_b,
            r1b,
            no_row_b,
            r3b,
            false_b,
            false_b,
            false_b,
            false_b,
            false_b,
            false_b,
        ]
        b_inputs = [
            not_red,
            cvalid,
            c2,
            cvalid,
            cvalid,
            cvalid,
            cvalid,
            cvalid,
            cvalid,
            cvalid,
        ]
        inits.append(numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), name="zero"))
        nodes.append(helper.make_node("Concat", a_inputs, ["A"], axis=1))
        nodes.append(helper.make_node("Concat", b_inputs, ["B"], axis=1))
        nodes.append(helper.make_node("Where", ["A", "B", "zero"], ["output"]))
    elif candidate.strategy == "product":
        row_seed, no_row = "rs", "nr"
        nodes.append(helper.make_node("Add", [r1, r3], [row_seed]))
        nodes.append(helper.make_node("Sub", [rvalid, row_seed], [no_row]))

        zrow = "zr"
        nodes.append(helper.make_node("Sub", [r1, r1], [zrow]))

        a_inputs = [no_row, r1, no_row, r3, zrow, zrow, zrow, zrow, zrow, zrow]
        b_inputs = [
            not_red,
            cvalid,
            c2,
            cvalid,
            cvalid,
            cvalid,
            cvalid,
            cvalid,
            cvalid,
            cvalid,
        ]
        nodes.append(helper.make_node("Concat", a_inputs, ["A"], axis=1))
        nodes.append(helper.make_node("Concat", b_inputs, ["B"], axis=1))
        nodes.append(helper.make_node("Mul", ["A", "B"], ["output"]))
    else:
        raise ValueError(f"unknown task024 graph strategy: {candidate.strategy}")

    graph = helper.make_graph(nodes, f"{TASK_ID}_{candidate.name}", [x], [y], inits)
    model = helper.make_model(
        graph,
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", candidate.opset)],
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def run_pass_rate(model_path: Path, examples: list[dict[str, list[list[int]]]]) -> tuple[int, int]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    passed = 0
    for ex in examples:
        got = (session.run(["output"], {"input": grid_to_onehot(ex["input"])})[0] > 0).astype(np.float32)
        if np.array_equal(got, expected_onehot(ex)):
            passed += 1
    return passed, len(examples)


def largest_internal_tensor(model_path: Path) -> int | None:
    sanitized = sanitize_model(onnx.load(str(model_path)))
    if sanitized is None:
        return None
    try:
        graph = onnx.shape_inference.infer_shapes(sanitized, strict_mode=True).graph
    except Exception:
        return None
    max_bytes = 0
    for value in graph.value_info:
        if value.name in {"input", "output"} or not value.type.HasField("tensor_type"):
            continue
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            return None
        numel = 1
        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                return None
            numel *= dim.dim_value
        dtype = helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        max_bytes = max(max_bytes, int(numel * np.dtype(dtype).itemsize))
    return max_bytes


def evaluate_candidate(
    candidate: Candidate,
    rule: Rule,
    data: dict[str, list[dict[str, list[list[int]]]]],
    synth: list[dict[str, list[list[int]]]],
    workdir: Path,
) -> dict[str, Any]:
    path = workdir / f"{TASK_ID}_{candidate.name}_opset{candidate.opset}.onnx"
    result: dict[str, Any] = {"candidate": candidate, "path": path, "validity": "invalid"}
    try:
        model = build_model(candidate, rule)
        onnx.save(model, path)
        train_pass = run_pass_rate(path, data["train"])
        test_pass = run_pass_rate(path, data.get("test", []))
        arc_pass = run_pass_rate(path, data.get("arc-gen", []))
        synth_pass = run_pass_rate(path, synth)
        scored = score_file(path)
        result.update(scored)
        result.update(
            {
                "validity": "valid" if scored["valid"] else "invalid",
                "train_pass": train_pass,
                "test_pass": test_pass,
                "arc_pass": arc_pass,
                "synthetic_pass": synth_pass,
                "largest_internal": largest_internal_tensor(path),
                "params_direct": calculate_params(onnx.load(str(path))),
            }
        )
    except Exception as exc:  # keep search running across incompatible opsets
        result["error"] = repr(exc)
    return result


def print_candidate(result: dict[str, Any]) -> None:
    cand = result["candidate"]
    train = result.get("train_pass", (0, 0))
    test = result.get("test_pass", (0, 0))
    synth = result.get("synthetic_pass", (0, 0))
    arc = result.get("arc_pass", (0, 0))
    valid = result.get("valid", False)
    memory = result.get("memory")
    params = result.get("params")
    cost = result.get("cost")
    points = result.get("score")
    print(
        f"{cand.name:<16} opset={cand.opset:<2} validity={result.get('validity')} "
        f"train={train[0]}/{train[1]} test={test[0]}/{test[1]} "
        f"arc-gen={arc[0]}/{arc[1]} synthetic={synth[0]}/{synth[1]} "
        f"params={params} largest_internal={result.get('largest_internal')} "
        f"memory={memory} cost={cost} score={points if points is None else f'{points:.6f}'}"
    )
    if not valid and result.get("error"):
        print(f"  error: {str(result['error']).splitlines()[-1]}")


def main() -> None:
    data = load_data()
    rule = infer_rule(data)
    synth = synthetic_examples(rule)
    print(f"inferred rule: row_colors={rule.row_colors} col_colors={rule.col_colors} row_over_col={rule.row_over_col}")

    candidates = [
        Candidate("where_bool", 10, True, "where_bool_rows"),
        Candidate("product", 10, True, "product"),
        Candidate("lean_where_bool", 10, False, "where_bool_rows"),
        Candidate("lean_product", 10, False, "product"),
    ]

    best: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}_search_") as tmp:
        workdir = Path(tmp)
        for candidate in candidates:
            result = evaluate_candidate(candidate, rule, data, synth, workdir)
            print_candidate(result)
            all_pass = all(
                result.get(key, (0, 1))[0] == result.get(key, (0, 1))[1]
                for key in ("train_pass", "test_pass", "arc_pass", "synthetic_pass")
            )
            if result.get("valid") and all_pass:
                if best is None or (
                    int(result["cost"]),
                    0 if result["candidate"].robust_validity else 1,
                    result["candidate"].opset,
                ) < (
                    int(best["cost"]),
                    0 if best["candidate"].robust_validity else 1,
                    best["candidate"].opset,
                ):
                    best = result

        if best is None:
            raise SystemExit("no valid fully-correct candidate found")

        shutil.copyfile(best["path"], BEST_PATH)
        chosen = best["candidate"]
        print()
        print(
            f"selected {chosen.name} opset={chosen.opset}: "
            f"memory={best['memory']} params={best['params']} "
            f"cost={best['cost']} score={best['score']:.6f}"
        )
        print(f"wrote {BEST_PATH.relative_to(ROOT)}")

    # Re-score the persisted file so the final line is from the kept artifact.
    final = score_file(BEST_PATH)
    if not final["valid"]:
        raise SystemExit(f"persisted model is invalid: {final['error']}")
    print(
        f"persisted score: memory={final['memory']} params={final['params']} "
        f"cost={final['cost']} score={final['score']:.6f}"
    )
    if not math.isclose(float(final["score"]), float(best["score"]), rel_tol=0.0, abs_tol=1e-9):
        raise SystemExit("persisted model score changed unexpectedly")


if __name__ == "__main__":
    main()
