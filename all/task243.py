"""Minimal ONNX for ARC task243: flood-fill black regions from blue.

Task rule: preserve every non-black cell exactly as-is, and recolor to blue
only the black cells in a 4-connected black/blue region that already contains
blue. Black regions separated from blue by other colors stay black. The output
grid has the same shape as the input grid. All examples fit in the top-left
18x18 area of the NeuroGolf 30x30 one-hot tensor, so the ONNX graph performs a
fixed 28-step flood fill on that compact crop and pads the remainder back to
all-zero output padding. The fill loop uses float16 Conv tensors and tracks
only reached black cells; the original blue cells are OR-ed back into channel 1
at the end.
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

TASK_ID = "task243"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task243.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
SH = SW = 18
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10
FILL_STEPS = 28


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference grid solver: flood blue through 4-connected black cells."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    reach = g == 1
    black = g == 0
    for _ in range(g.shape[0] + g.shape[1]):
        nbr = np.zeros_like(reach)
        nbr[1:, :] |= reach[:-1, :]
        nbr[:-1, :] |= reach[1:, :]
        nbr[:, 1:] |= reach[:, :-1]
        nbr[:, :-1] |= reach[:, 1:]
        nxt = reach | (black & nbr)
        if np.array_equal(nxt, reach):
            break
        reach = nxt
    out[black & reach] = 1
    return out


def _grid_to_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    arr = np.asarray(grid, dtype=np.int64)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes_chw = _i64(inits, [1, 2, 3], "axes_chw")
    ch0_start = _i64(inits, [0, 0, 0], "ch0_start")
    ch0_end = _i64(inits, [1, SH, SW], "ch0_end")
    ch1_start = _i64(inits, [1, 0, 0], "ch1_start")
    ch1_end = _i64(inits, [2, SH, SW], "ch1_end")
    ch2_start = _i64(inits, [2, 0, 0], "ch2_start")
    ch2_end = _i64(inits, [C, SH, SW], "ch2_end")
    zero = _init(inits, np.asarray(0.0, dtype=np.float16), "zero")
    cross = np.asarray([[[[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]]], dtype=np.float16)
    kernel = _init(inits, cross, "cross")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, ch0_start, ch0_end, axes_chw], ["black"]),
            helper.make_node("Slice", [IN_NAME, ch1_start, ch1_end, axes_chw], ["blue"]),
            helper.make_node("Slice", [IN_NAME, ch2_start, ch2_end, axes_chw], ["rest"]),
            helper.make_node("Cast", ["black"], ["black_b"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["blue"], ["blue_b"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["blue"], ["reachf0"], to=TensorProto.FLOAT16),
        ]
    )

    reach_b = "blue_b"
    reach_f = "reachf0"
    for idx in range(FILL_STEPS):
        count = f"count{idx}"
        nbr = f"nbr{idx}"
        nxt_b = f"reach{idx + 1}"
        nxt_f = f"reachf{idx + 1}"
        nodes.extend(
            [
                helper.make_node("Conv", [reach_f, kernel], [count], pads=[1, 1, 1, 1]),
                helper.make_node("Greater", [count, zero], [nbr]),
                helper.make_node("And", ["black_b", nbr], [nxt_b]),
                helper.make_node("Cast", [nxt_b], [nxt_f], to=TensorProto.FLOAT16),
            ]
        )
        reach_b = nxt_b
        reach_f = nxt_f

    nodes.extend(
        [
            helper.make_node("Xor", ["black_b", reach_b], ["black_out_b"]),
            helper.make_node("Or", ["blue_b", reach_b], ["blue_out_b"]),
            helper.make_node("Cast", ["black_out_b"], ["black_out"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["blue_out_b"], ["blue_out"], to=TensorProto.FLOAT),
            helper.make_node("Concat", ["black_out", "blue_out", "rest"], ["out18"], axis=1),
            helper.make_node("Pad", ["out18"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - SH, W - SW]),
        ]
    )

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


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            inp = np.asarray(ex["input"], dtype=np.int64)
            expected = np.asarray(ex["output"], dtype=np.int64)
            ref = solve(inp)
            if not np.array_equal(ref, expected):
                raise AssertionError(f"reference mismatch in {split}[{idx}]")
            if inp.shape[0] > SH or inp.shape[1] > SW:
                raise AssertionError(f"{split}[{idx}] exceeds compact crop: {inp.shape}")

            pred_oh = _run_onnx(model, _grid_to_onehot(inp))
            expected_oh = _grid_to_onehot(expected)
            pred = _onehot_to_grid(pred_oh)[: inp.shape[0], : inp.shape[1]]
            if not np.array_equal(pred_oh > 0.0, expected_oh > 0.0) or not np.array_equal(pred, expected):
                bad += 1
    return bad


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    bad = validate_json(model)
    print(f"{TASK_ID}.json: {'PASS' if bad == 0 else f'FAIL ({bad} wrong)'}")

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
