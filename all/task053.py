"""ONNX generator for ARC task053: shift the visible grid down one row.

Task rule: create an output grid with the same HxW shape as the input. Every
non-black cell moves from (r, c) to (r + 1, c), preserving color; cells that
would move below the bottom edge are discarded. Black cells remain black, so
equivalently output[1:, :] = input[:-1, :] and output[0, :] = 0.

All official task053 train/test/arc-gen examples are 3x3 and use only colors
0, 1, and 2. The Python reference solver below works for arbitrary HxW grids,
while the competition ONNX graph is specialized to the confirmed 3x3 shape and
three used color channels to keep memory small. The graph slices input channels
0:3 and rows 0:2, prepends a one-hot black row, and pads channels/rows/columns
back to the required [1, 10, 30, 30] output tensor.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task053"
TASK_NUM = 53
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
CORE_H = CORE_W = 3
USED_C = 3
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


@dataclass(frozen=True)
class Variant:
    name: str
    builder: Callable[[], onnx.ModelProto]
    note: str


def solve_grid(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Dimension-general reference implementation for the ARC rule."""
    arr = np.asarray(grid, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D grid, got shape {arr.shape}")
    out = np.zeros_like(arr)
    if arr.shape[0] > 1:
        out[1:, :] = arr[:-1, :]
    return out


def _i64(inits: list[onnx.TensorProto], values: list[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name))
    return name


def _f32(inits: list[onnx.TensorProto], array: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(array, dtype=np.float32), name=name))
    return name


def _make_model(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    *,
    graph_name: str,
    opset: int = OPSET,
) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        graph_name,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    model.doc_string = ""
    model.producer_version = ""
    model.domain = ""
    model.model_version = 0
    del model.metadata_props[:]
    onnx.checker.check_model(model, full_check=True)
    return model


def build_model() -> onnx.ModelProto:
    """Best submission-ready model: exact one-hot output, opset 10 / IR 10."""
    inits: list[onnx.TensorProto] = []
    starts = _i64(inits, [0, 0, 0, 0], "starts")
    ends = _i64(inits, [1, USED_C, CORE_H - 1, CORE_W], "ends")

    top = np.zeros((1, USED_C, 1, CORE_W), dtype=np.float32)
    top[:, 0, :, :] = 1.0
    top_name = _f32(inits, top, "black_top")

    nodes = [
        helper.make_node("Slice", [IN_NAME, starts, ends], ["body"]),
        helper.make_node("Concat", [top_name, "body"], ["core"], axis=2),
        helper.make_node(
            "Pad",
            ["core"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, C - USED_C, H - CORE_H, W - CORE_W],
        ),
    ]
    return _make_model(nodes, inits, graph_name=TASK_ID)


def build_argmax_only_baseline() -> onnx.ModelProto:
    """Historical tiny graph: cheap, but not exact one-hot for black top row."""
    inits: list[onnx.TensorProto] = []
    starts = _i64(inits, [0, 0], "starts")
    ends = _i64(inits, [CORE_H - 1, CORE_W], "ends")
    axes = _i64(inits, [2, 3], "axes")
    nodes = [
        helper.make_node("Slice", [IN_NAME, starts, ends, axes], ["body"]),
        helper.make_node(
            "Pad",
            ["body"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 1, 0, 0, 0, H - CORE_H, W - CORE_W],
        ),
    ]
    return _make_model(nodes, inits, graph_name="argmax_only_baseline")


def _task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _examples() -> list[tuple[str, int, dict[str, list[list[int]]]]]:
    data = _task_data()
    return [
        (split, idx, ex)
        for split in ("train", "test", "arc-gen")
        for idx, ex in enumerate(data.get(split, []))
    ]


def _grid_dims() -> dict[str, list[tuple[int, int]]]:
    data = _task_data()
    dims: dict[str, list[tuple[int, int]]] = {}
    for split in ("train", "test", "arc-gen"):
        split_dims = {
            (len(ex["input"]), len(ex["input"][0]))
            for ex in data.get(split, [])
            if ex.get("input")
        }
        dims[split] = sorted(split_dims)
    return dims


def _run_model(model: onnx.ModelProto, arr: np.ndarray) -> np.ndarray:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    return session.run([OUT_NAME], {IN_NAME: arr})[0]


def verify_reference_rule() -> None:
    for split, idx, ex in _examples():
        expected = np.asarray(ex["output"], dtype=np.int64)
        actual = solve_grid(ex["input"])
        if not np.array_equal(actual, expected):
            raise AssertionError(f"reference mismatch: {split} #{idx}")


def verify_model(model: onnx.ModelProto) -> tuple[bool, str]:
    for split, idx, ex in _examples():
        inp = convert_to_numpy(ex, "input")
        expected = convert_to_numpy(ex, "output")
        if inp is None or expected is None:
            continue
        actual = _run_model(model, inp)
        if not np.array_equal(actual > 0.0, expected > 0.0):
            return False, f"{split} #{idx}: thresholded one-hot mismatch"
        h, w = len(ex["output"]), len(ex["output"][0])
        active = actual[0, :, :h, :w] > 0.0
        if not np.all(active.sum(axis=0) == 1):
            return False, f"{split} #{idx}: invalid one-hot cell"
        if np.any(actual[0, :, h:, :] > 0.0) or np.any(actual[0, :, :, w:] > 0.0):
            return False, f"{split} #{idx}: nonzero padding"
    return True, f"all {len(_examples())} examples matched"


def _score_model(model: onnx.ModelProto, name: str) -> dict[str, Any]:
    tmp = Path(tempfile.mkdtemp(prefix=f"{TASK_ID}_")) / f"{name}.onnx"
    onnx.save(model, str(tmp))
    result = score_file(tmp)
    result["path"] = tmp
    return result


def run_variants() -> list[dict[str, Any]]:
    variants = [
        Variant("exact_onehot_op10", build_model, "submission-ready"),
        Variant("argmax_only_baseline", build_argmax_only_baseline, "fails exact black row"),
    ]
    rows: list[dict[str, Any]] = []
    for variant in variants:
        row: dict[str, Any] = {"name": variant.name, "note": variant.note}
        try:
            model = variant.builder()
            ok, msg = verify_model(model)
            row["correct"] = ok
            row["validation"] = msg
            scored = _score_model(model, variant.name)
            row.update(
                {
                    "memory": scored.get("memory"),
                    "params": scored.get("params"),
                    "cost": scored.get("cost"),
                    "score": scored.get("score"),
                    "score_error": scored.get("error"),
                    "model": model,
                }
            )
        except Exception as exc:
            row["correct"] = False
            row["validation"] = f"build failed: {exc}"
        rows.append(row)
    return rows


def main() -> None:
    verify_reference_rule()
    dims = _grid_dims()
    if any(dim_list != [(CORE_H, CORE_W)] for dim_list in dims.values()):
        raise SystemExit(f"{TASK_ID} ONNX specialization expects 3x3 examples, got {dims}")

    rows = run_variants()
    print(f"{TASK_ID}: grid dimensions {dims}")
    print(f"{'variant':<22} {'ok':<5} {'memory':>7} {'params':>7} {'cost':>7} {'score':>9}  note")
    for row in rows:
        score = row.get("score")
        score_s = f"{score:.6f}" if isinstance(score, float) else "INVALID"
        print(
            f"{row['name']:<22} {str(row.get('correct')):<5} "
            f"{str(row.get('memory', '-')):>7} "
            f"{str(row.get('params', '-')):>7} "
            f"{str(row.get('cost', '-')):>7} "
            f"{score_s:>9}  {row.get('note', '')}"
        )
        if row.get("validation") != f"all {len(_examples())} examples matched":
            print(f"  validation: {row.get('validation')}")
        if row.get("score_error"):
            print(f"  scorer: {row['score_error']}")

    valid = [
        row
        for row in rows
        if row.get("correct") and row.get("cost") is not None and row.get("score_error") is None
    ]
    if not valid:
        raise SystemExit("no correct scored variant")

    best = min(valid, key=lambda row: int(row["cost"]))
    onnx.save(best["model"], str(BEST_PATH))
    print(f"\nSaved {BEST_PATH}")


if __name__ == "__main__":
    main()
