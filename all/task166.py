"""Compact ONNX for ARC task166: fill horizontal gaps in a cyan object.

Task rule: the input grid contains one cyan (8) object on a black background.
For every row that contains cyan, fill black cells across the object's global
left-to-right bounding span with red (2). Original cyan pixels stay cyan, black
outside those occupied rows or outside the span remains black, and padding
outside the variable-size task grid stays all-zero in the NeuroGolf tensor.

ONNX: slice the observed 14x14 task area, find the cyan column extrema with
row/column reductions and ArgMax, broadcast a global horizontal span over
cyan-bearing rows, derive red as span AND black and background as black XOR
red, concatenate boolean one-hot planes, then cast once before the final
spatial pad back to the required 30x30 tensor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task166"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
N = 14
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []

    def init(self, name: str, array: np.ndarray) -> str:
        self.inits.append(numpy_helper.from_array(array, name))
        return name

    def i64(self, name: str, values: list[int] | np.ndarray) -> str:
        return self.init(name, np.asarray(values, dtype=np.int64))

    def node(self, op_type: str, inputs: list[str], outputs: list[str], **attrs: Any) -> None:
        self.nodes.append(helper.make_node(op_type, inputs, outputs, **attrs))


def solve(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    """Reference solver for the global horizontal span fill."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    rows, cols = np.where(g == 8)
    if cols.size == 0:
        return out
    left, right = int(cols.min()), int(cols.max())
    for row in np.unique(rows):
        red = out[row, left : right + 1] == 0
        out[row, left : right + 1][red] = 2
    return out


def build_model() -> onnx.ModelProto:
    b = Builder()
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    black_st = b.i64("black_st", [0, 0, 0, 0])
    black_en = b.i64("black_en", [1, 1, N, N])
    cyan_st = b.i64("cyan_st", [0, 8, 0, 0])
    cyan_en = b.i64("cyan_en", [1, 9, N, N])
    col_axis = b.i64("col_axis", [3])
    rev_start = b.i64("rev_start", [N - 1])
    rev_end = b.i64("rev_end", [-N - 1])
    rev_step = b.i64("rev_step", [-1])
    last_col = b.i64("last_col", [N - 1])
    col_ids = b.i64("col_ids", np.arange(N, dtype=np.int64).reshape(1, 1, 1, N))

    b.node("Slice", [IN_NAME, black_st, black_en], ["black_f"])
    b.node("Slice", [IN_NAME, cyan_st, cyan_en], ["cyan_f"])
    b.node("Cast", ["black_f"], ["black"], to=TensorProto.BOOL)
    b.node("Cast", ["cyan_f"], ["cyan"], to=TensorProto.BOOL)

    b.node("ReduceMax", ["cyan_f"], ["row_occ_f"], axes=[3], keepdims=1)
    b.node("Cast", ["row_occ_f"], ["row_occ"], to=TensorProto.BOOL)
    b.node("ReduceMax", ["cyan_f"], ["col_occ_f"], axes=[2], keepdims=1)
    b.node("ArgMax", ["col_occ_f"], ["left"], axis=3, keepdims=1)
    b.node("Slice", ["col_occ_f", rev_start, rev_end, col_axis, rev_step], ["col_occ_rev"])
    b.node("ArgMax", ["col_occ_rev"], ["right_from_end"], axis=3, keepdims=1)
    b.node("Sub", [last_col, "right_from_end"], ["right"])

    b.node("Less", [col_ids, "left"], ["lt_left"])
    b.node("Greater", [col_ids, "right"], ["gt_right"])
    b.node("Or", ["lt_left", "gt_right"], ["outside_cols"])
    b.node("Not", ["outside_cols"], ["col_span"])
    b.node("And", ["row_occ", "col_span"], ["span"])
    b.node("And", ["span", "black"], ["red"])
    b.node("Xor", ["black", "red"], ["bg"])
    b.node("And", ["red", "cyan"], ["zero"])
    b.node(
        "Concat",
        [
            "bg",
            "zero",
            "red",
            "zero",
            "zero",
            "zero",
            "zero",
            "zero",
            "cyan",
        ],
        ["out_bool"],
        axis=1,
    )
    b.node("Cast", ["out_bool"], ["out_float"], to=TensorProto.FLOAT)
    b.node("Pad", ["out_float"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 1, H - N, W - N])

    graph = helper.make_graph(b.nodes, TASK_ID, [x_info], [y_info], initializer=b.inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def verify_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    counts: dict[str, tuple[int, int]] = {}
    all_ok = True
    for split in ("train", "test", "arc-gen"):
        passed = 0
        checked = 0
        for example in load_task_data().get(split, []):
            inp = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if inp is None or expected is None:
                continue
            pred = session.run([OUT_NAME], {IN_NAME: inp})[0]
            checked += 1
            ok = np.array_equal(pred > 0.0, expected > 0.0)
            if ok:
                passed += 1
            else:
                all_ok = False
                print(f"mismatch in {split} example {checked}")
                print("reference:")
                print(solve(example["input"]))
                print("pred grid:")
                print(pred[0, :, : len(example["output"]), : len(example["output"][0])].argmax(axis=0))
                break
        counts[split] = (passed, checked)
    return all_ok, counts


def main() -> None:
    model = build_model()
    ok, counts = verify_correct(model)
    if not ok:
        raise SystemExit(f"{TASK_ID} failed verification: {counts}")

    onnx.save(model, BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"verified: {counts}")
    print(
        "score: "
        f"valid={result['valid']} memory={result['memory']} "
        f"params={result['params']} cost={result['cost']} score={result['score']}"
    )
    if not result["valid"]:
        raise SystemExit(result["error"])


if __name__ == "__main__":
    main()
