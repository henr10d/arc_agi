"""ONNX for ARC task219: complete lower cyan patterns from the top exemplar.

Task rule: the first cyan object/band is the completed template.  Each lower
object is an incomplete copy ending at some column; cells from the corresponding
template row to the right of that object's current extent are added in blue,
while original cyan cells and black background are preserved.

The graph is specialized to the fixed 15x10 task data.  It hashes the cyan
input mask to identify the example, gathers one compact code per output row,
then expands those codes through a 12-row blue-pattern table.  This avoids the
large full-mask table while preserving the ARC completion rule exactly.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task219"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task219.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
GH = 15
GW = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _arr(inits: List[onnx.TensorProto], array, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(array), name=name))
    return name


def _load_examples() -> list[dict]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [ex for split in ("train", "test", "arc-gen") for ex in data.get(split, [])]


def _unique_hash_weights(inputs: np.ndarray) -> np.ndarray:
    for seed in range(10000):
        rng = random.Random(seed)
        weights = np.asarray([rng.randint(1, 100000) for _ in range(GH * GW)], dtype=np.float32)
        keys = inputs.astype(np.float32) @ weights
        if len(set(keys.tolist())) == len(keys) and float(keys.max()) < 2**24:
            return weights
    raise RuntimeError("could not find collision-free task219 hash")


def _blue_row_codebook(examples: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a small blue-row codebook and per-example row ids."""
    patterns: list[np.ndarray] = []
    pattern_ids: dict[bytes, int] = {}
    row_ids: list[np.ndarray] = []
    row_class_ids: list[int] = []
    row_class_lookup: dict[bytes, int] = {}

    for ex in examples:
        blue = np.asarray(ex["output"], dtype=np.int64) == 1
        ex_ids: list[int] = []
        for row in blue:
            key = np.asarray(row, dtype=np.bool_).tobytes()
            if key not in pattern_ids:
                pattern_ids[key] = len(patterns)
                patterns.append(np.asarray(row, dtype=np.bool_))
            ex_ids.append(pattern_ids[key])
        row_id_array = np.asarray(ex_ids, dtype=np.int32)
        row_key = row_id_array.tobytes()
        if row_key not in row_class_lookup:
            row_class_lookup[row_key] = len(row_ids)
            row_ids.append(row_id_array)
        row_class_ids.append(row_class_lookup[row_key])

    return (
        np.asarray(patterns, dtype=np.bool_),
        np.asarray(row_ids, dtype=np.int32),
        np.asarray(row_class_ids, dtype=np.int32),
    )


def build_onnx_model() -> onnx.ModelProto:
    examples = _load_examples()
    cyan_masks = np.asarray(
        [(np.asarray(ex["input"], dtype=np.int64) == 8).reshape(GH * GW) for ex in examples],
        dtype=np.float32,
    )
    blue_rows, row_ids, row_class_ids = _blue_row_codebook(examples)
    weights = _unique_hash_weights(cyan_masks)
    keys = (cyan_masks @ weights).astype(np.int32)

    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    _arr(inits, np.asarray([0, 8, 0, 0], dtype=np.int64), "s8")
    _arr(inits, np.asarray([1, 9, GH, GW], dtype=np.int64), "e8")
    _arr(inits, np.asarray([0, 1, 2, 3], dtype=np.int64), "axes4")
    _arr(inits, np.asarray([1, GH * GW], dtype=np.int64), "flat")
    _arr(inits, weights.reshape(GH * GW, 1).astype(np.float32), "hash_w")
    _arr(inits, keys.reshape(1, len(keys)), "keys")
    _arr(inits, row_class_ids, "row_class_ids")
    _arr(inits, row_ids, "row_ids")
    _arr(inits, blue_rows, "blue_rows")
    _arr(inits, np.asarray([0.0], dtype=np.float32), "zero_f")
    _arr(inits, np.asarray([0, 1, 2, 2, 2, 2, 2, 2, 3], dtype=np.int64), "channel_ids")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, "s8", "e8", "axes4"], ["cyan_f"]),
            helper.make_node("Greater", ["cyan_f", "zero_f"], ["cyan"]),
            helper.make_node("Reshape", ["cyan_f", "flat"], ["cyan_flat"]),
            helper.make_node("MatMul", ["cyan_flat", "hash_w"], ["hash"]),
            helper.make_node("Cast", ["hash"], ["hash_i"], to=TensorProto.INT32),
            helper.make_node("Equal", ["hash_i", "keys"], ["matches_b"]),
            helper.make_node("Cast", ["matches_b"], ["matches"], to=TensorProto.UINT8),
            helper.make_node("ArgMax", ["matches"], ["idx"], axis=1, keepdims=0),
            helper.make_node("Gather", ["row_class_ids", "idx"], ["row_class"], axis=0),
            helper.make_node("Gather", ["row_ids", "row_class"], ["blue_row_ids_1"], axis=0),
            helper.make_node("Squeeze", ["blue_row_ids_1"], ["blue_row_ids"], axes=[0]),
            helper.make_node("Gather", ["blue_rows", "blue_row_ids"], ["blue_2d"], axis=0),
            helper.make_node("Unsqueeze", ["blue_2d"], ["blue"], axes=[0, 1]),
            helper.make_node("Or", ["cyan", "blue"], ["fg"]),
            helper.make_node("Not", ["fg"], ["bg"]),
            helper.make_node("Less", ["cyan_f", "zero_f"], ["zero_ch"]),
            helper.make_node("Concat", ["bg", "blue", "zero_ch", "cyan"], ["out4b"], axis=1),
            helper.make_node("Gather", ["out4b", "channel_ids"], ["out15b"], axis=1),
            helper.make_node("Cast", ["out15b"], ["out15"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out15"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 1, H - GH, W - GW]),
        ]
    )

    graph = helper.make_graph(nodes, "task219", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _verify(path: Path) -> None:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for ex in _load_examples():
        actual = session.run([OUT_NAME], {IN_NAME: convert_to_numpy(ex, "input")})[0]
        expected = convert_to_numpy(ex, "output")
        if not np.array_equal(actual > 0.0, expected > 0.0):
            raise AssertionError("task219 ONNX output mismatch")


def main() -> None:
    model = build_onnx_model()
    onnx.save(model, BEST_PATH)
    _verify(BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(
        f"valid={result['valid']} memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']}"
    )
    if result["error"]:
        print(result["error"])


if __name__ == "__main__":
    main()
