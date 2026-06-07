"""ONNX for ARC task287: repair purple mask cells from 180-degree symmetry.

Task rule: the 16x16 grid is a nested pattern that is symmetric under both
horizontal and vertical reflection.  Some cells are overwritten with color 4;
all other cells are already correct.  Replace every color-4 cell with the color
at its 180-degree reflected position, keep all non-4 cells unchanged, and leave
the competition padding outside the 16x16 grid all-zero.

ONNX approach: a dilated 2x2 Conv reads only the top-left cell of each 15x15
window, converting the 16x16 one-hot core to compact color-minus-one indices.
The repair then runs on one channel: reverse that index plane, threshold index
3 as the color-4 mask, and choose reflected indices with Where.  A depth-9
OneHot reconstructs colors 1..9; final Pad adds the all-zero channel 0 and the
competition padding.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task287"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task287.onnx"
DATA_PATH = ROOT / "data" / "task287.json"

C = 10
H = W = 30
N = 16
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


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    weights = np.zeros((1, C, 2, 2), dtype=np.float32)
    weights[0, :, 0, 0] = np.arange(-1, C - 1, dtype=np.float32)
    conv_w = _f32(inits, weights, "w")
    rev_st = _i64(inits, [N - 1, N - 1], "c")
    rev_en = _i64(inits, [-N - 1, -N - 1], "d")
    rev_axes = _i64(inits, [2, 3], "e")
    rev_steps = _i64(inits, [-1, -1], "f")
    mask_lo = _f32(inits, [2.5], "g")
    mask_hi = _f32(inits, [3.5], "h")
    depth = _i64(inits, [C - 1], "i")
    onehot_values = _f32(inits, [0.0, 1.0], "j")

    nodes.extend(
        [
            helper.make_node("Conv", [IN_NAME, conv_w], ["x"], dilations=[H - N, W - N]),
            helper.make_node("Slice", ["x", rev_st, rev_en, rev_axes, rev_steps], ["r"]),
            helper.make_node("Greater", ["x", mask_lo], ["ml"]),
            helper.make_node("Less", ["x", mask_hi], ["mh"]),
            helper.make_node("And", ["ml", "mh"], ["m"]),
            helper.make_node("Where", ["m", "r", "x"], ["y"]),
            helper.make_node("Squeeze", ["y"], ["ys"], axes=[1]),
            helper.make_node("Cast", ["ys"], ["yi"], to=TensorProto.INT64),
            helper.make_node("OneHot", ["yi", depth, onehot_values], ["oh"], axis=1),
            helper.make_node("Pad", ["oh"], [OUT_NAME], pads=[0, 1, 0, 0, 0, 0, H - N, W - N]),
        ]
    )
    return _make_model(nodes, inits)


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference implementation used by local validation."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    for r, c in np.argwhere(g == 4):
        out[r, c] = g[N - 1 - r, N - 1 - c]
    return out


def _grid_to_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    arr = np.asarray(grid, dtype=np.int64)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            expected_grid = np.asarray(ex["output"], dtype=np.int64)
            ref = solve(np.asarray(ex["input"], dtype=np.int64))
            if not np.array_equal(ref, expected_grid):
                raise AssertionError(f"reference solver failed {split}[{idx}]")

            x = _grid_to_onehot(ex["input"])
            pred = session.run([OUT_NAME], {IN_NAME: x})[0] > 0.0
            expected = _grid_to_onehot(expected_grid) > 0.0
            if not np.array_equal(pred, expected):
                print(f"mismatch: {split}[{idx}]")
                bad += 1
    return bad


def tensor_count(model: onnx.ModelProto) -> int:
    names = {value.name for value in model.graph.input}
    names.update(value.name for value in model.graph.output)
    names.update(init.name for init in model.graph.initializer)
    for node in model.graph.node:
        names.update(name for name in node.output if name)
    return len(names)


def main() -> None:
    model = build_model()
    bad = validate_json(model)
    if bad:
        raise AssertionError(f"{TASK_ID} failed {bad} examples")

    onnx.save(model, BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"tensors: {tensor_count(model)}")
    print(f"nodes:   {len(model.graph.node)}")
    print(f"size:    {result['filesize']}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
