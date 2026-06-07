"""Compact ONNX generator for NeuroGolf task297.

Task rule: preserve row 0, which is a width-sized color palette, and preserve
the gray row 1 exactly. Rows 2 onward are black in the input and become solid
stripes: output row r uses the palette color at column ``(r - 2) % W`` across
the active width W. Padding outside the variable-size ARC grid remains all-zero.

ONNX approach: work only on the observed maximum 14x6 region and the nonblack
channels 1..9, compute W from the occupied palette row, gather twelve cyclic
palette rows, broadcast each row across six columns, mask to the active
``2*W`` by ``W`` rectangle, concatenate the preserved palette row plus
reconstructed gray row, cast to float, and let the final Pad add channel 0 and
produce the required 30x30 output tensor.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402


TASK_ID = "task297"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
WORK_H = 14
WORK_W = 6
BODY_H = WORK_H - 2
IR_VERSION = 10
OPSET = 10


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []

    def init(self, name: str, array: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(array, name))
        return name

    def i64(self, name: str, values: list[int] | np.ndarray) -> str:
        return self.init(name, np.asarray(values, dtype=np.int64))

    def i32(self, name: str, values: list[int] | np.ndarray) -> str:
        return self.init(name, np.asarray(values, dtype=np.int32))

    def node(self, op_type: str, inputs: list[str], output: str, **attrs: Any) -> str:
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        return output


def make_model(nodes: list[onnx.NodeProto], initializers: list[onnx.TensorProto]) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def build_model() -> onnx.ModelProto:
    b = Builder()
    b.i64("row0_starts", [1, 0, 0])
    b.i64("row0_ends", [10, 1, WORK_W])
    b.i64("axes_chw", [1, 2, 3])
    b.i32("row_offsets", list(range(BODY_H)))
    b.i32("body_rows", list(range(BODY_H)))
    b.i32("two_i32", [2])
    b.i64("tile_repeats", [1, 1, 1, WORK_W])
    gray_channel = np.zeros((1, 9, 1, 1), dtype=np.bool_)
    gray_channel[0, 4, 0, 0] = True
    b.init("gray_channel", gray_channel)

    row0_f = b.node("Slice", [IN_NAME, "row0_starts", "row0_ends", "axes_chw"], "row0_f")
    row0_b = b.node("Cast", [row0_f], "row0_b", to=TensorProto.BOOL)
    col_occupied = b.node("ReduceSum", [row0_f], "col_occupied", axes=[1, 2], keepdims=0)
    width_f = b.node("ReduceSum", [col_occupied], "width_f", axes=[1], keepdims=0)
    width_i = b.node("Cast", [width_f], "width_i", to=TensorProto.INT32)

    gray_cols = b.node("Cast", [col_occupied], "gray_cols", to=TensorProto.BOOL)
    gray_cols4 = b.node("Unsqueeze", [gray_cols], "gray_cols4", axes=[1, 2])
    gray_row = b.node("And", ["gray_cols4", "gray_channel"], "gray_row")
    prefix_b = b.node("Concat", [row0_b, gray_row], "prefix_b", axis=2)

    palette = b.node("Squeeze", [row0_b], "palette", axes=[2])
    cyclic_cols = b.node("Mod", ["row_offsets", width_i], "cyclic_cols", fmod=0)
    rows = b.node("Gather", [palette, cyclic_cols], "rows", axis=2)
    rows4 = b.node("Unsqueeze", [rows], "rows4", axes=[3])
    fill = b.node("Tile", [rows4, "tile_repeats"], "fill")

    body_limit = b.node("Mul", [width_i, "two_i32"], "body_limit")
    body_rows_valid = b.node("Less", ["body_rows", body_limit], "body_rows_valid")
    body_rows_valid4 = b.node("Unsqueeze", [body_rows_valid], "body_rows_valid4", axes=[0, 1, 3])
    spatial_mask = b.node("And", [body_rows_valid4, "gray_cols4"], "spatial_mask")
    body_b = b.node("And", [fill, spatial_mask], "body_b")
    compact_b = b.node("Concat", [prefix_b, body_b], "compact_b", axis=2)
    compact_f = b.node("Cast", [compact_b], "compact_f", to=TensorProto.FLOAT)
    b.nodes.append(
        helper.make_node(
            "Pad",
            [compact_f],
            [OUT_NAME],
            mode="constant",
            pads=[0, 1, 0, 0, 0, 0, 30 - WORK_H, 30 - WORK_W],
            value=0.0,
        )
    )
    return make_model(b.nodes, b.initializers)


def _load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def verify_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]], list[str]]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    splits: dict[str, tuple[int, int]] = {}
    mismatches: list[str] = []
    all_ok = True
    for split in ("train", "test", "arc-gen"):
        passed = 0
        checked = 0
        for idx, example in enumerate(_load_task_data().get(split, [])):
            inp = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if inp is None or expected is None:
                continue
            pred = session.run([OUT_NAME], {IN_NAME: inp})[0]
            checked += 1
            if np.array_equal(pred > 0.0, expected > 0.0):
                passed += 1
            else:
                all_ok = False
                mismatches.append(f"{split}[{idx}]")
        splits[split] = (passed, checked)
    return all_ok, splits, mismatches


def save_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    model = build_model()
    onnx.save(model, str(path))
    return model


def main() -> None:
    model = save_model()
    ok, splits, mismatches = verify_correct(model)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"correct: {ok} {splits}")
    if mismatches:
        print("mismatches:", ", ".join(mismatches[:20]))
    if result["valid"]:
        print(
            f"score: {result['score']:.6f} "
            f"cost={result['cost']} memory={result['memory']} params={result['params']}"
        )
    else:
        print(f"score: INVALID {result['error']}")


if __name__ == "__main__":
    main()
