"""Minimal ONNX for ARC task272: recolor isolated red cells blue.

Task rule: keep the grid size unchanged and preserve every black cell.  For
each red cell (2), inspect only its four orthogonal neighbors.  If none of
those up/down/left/right neighbors is also red, recolor that cell blue (1);
otherwise keep it red.  Diagonal red cells do not count as connected.

ONNX approach: all official examples are at most 5x5 and use only black/red
inputs.  Slice the compact 5x5 channels 0..2, convolve that crop with a kernel
that reads only the red channel and computes center - orthogonal neighbors.  A
positive score identifies exactly an isolated red cell.  Broadcast Where
replaces those positions with the blue one-hot vector in the 3-channel crop,
then a final Pad output restores the required 10x30x30 tensor without scoring
the large full-grid tensor as internal memory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task272"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task272.onnx"
DATA_PATH = ROOT / "data" / "task272.json"

C = 10
H = W = 30
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _make_model(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto]) -> onnx.ModelProto:
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


def _slice(nodes: List[onnx.NodeProto], data: str, start: str, end: str, axis: str, out: str) -> str:
    nodes.append(helper.make_node("Slice", [data, start, end, axis], [out]))
    return out


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    crop_starts = _i64(inits, [0, 0, 0], "crop_starts")
    crop_ends = _i64(inits, [3, 5, 5], "crop_ends")
    crop_axes = _i64(inits, [1, 2, 3], "crop_axes")
    zero = _f32(inits, [0.0], "zero")
    blue_replacement = np.zeros((1, 3, 1, 1), dtype=np.float32)
    blue_replacement[0, 1, 0, 0] = 1.0
    blue_onehot = _f32(inits, blue_replacement, "blue_onehot")
    kernel_arr = np.zeros((1, 3, 3, 3), dtype=np.float32)
    kernel_arr[0, 2] = np.asarray([[0.0, -1.0, 0.0], [-1.0, 1.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float32)
    kernel = _f32(
        inits,
        kernel_arr,
        "isolation_kernel",
    )

    _slice(nodes, IN_NAME, crop_starts, crop_ends, crop_axes, "crop3")
    nodes.extend(
        [
            helper.make_node("Conv", ["crop3", kernel], ["isolation_score"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            helper.make_node("Greater", ["isolation_score", zero], ["isolated_red"]),
            helper.make_node("Where", ["isolated_red", blue_onehot, "crop3"], ["out3"]),
            helper.make_node(
                "Pad",
                ["out3"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 7, 25, 25],
                value=0.0,
            ),
        ]
    )
    return _make_model(nodes, inits)


def solve(grid: np.ndarray, offsets: Iterable[tuple[int, int]]) -> np.ndarray:
    """Reference solver used only for local diagnostics."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    rows, cols = g.shape
    for r in range(rows):
        for c in range(cols):
            if g[r, c] != 2:
                continue
            has_red_neighbor = False
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and g[nr, nc] == 2:
                    has_red_neighbor = True
                    break
            if not has_red_neighbor:
                out[r, c] = 1
    return out


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            expected = _grid_to_onehot(ex["output"]) > 0.0
            pred = _run_onnx(model, _grid_to_onehot(ex["input"])) > 0.0
            if not np.array_equal(pred, expected):
                print(f"mismatch: {split}[{idx}]")
                bad += 1
    return bad


def diagnose_connectivity_variants() -> None:
    variants = {
        "orthogonal": [(-1, 0), (1, 0), (0, -1), (0, 1)],
        "diagonal": [(-1, -1), (-1, 1), (1, -1), (1, 1)],
        "eight-way": [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
    }
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for name, offsets in variants.items():
        failures = 0
        for split in ("train", "test", "arc-gen"):
            for ex in data[split]:
                if not np.array_equal(
                    solve(np.asarray(ex["input"], dtype=np.int64), offsets),
                    np.asarray(ex["output"], dtype=np.int64),
                ):
                    failures += 1
        print(f"{name}: {failures} failures")


def main() -> None:
    model = build_model()
    bad = validate_json(model)
    if bad:
        diagnose_connectivity_variants()
        raise AssertionError(f"{TASK_ID} failed {bad} examples")

    onnx.save(model, BEST_PATH)
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
