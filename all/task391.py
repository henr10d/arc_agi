"""Compact ONNX generator for ARC task391 using horizontal pair frequencies.

Task rule: the input is a sparse table of horizontal 1x2 bars. One color is a
high-frequency filler; the other three non-black colors are markers. Count the
bars of each color, omit the filler, and emit the three marker colors in
descending frequency as a 3x1 vertical output. Cells outside that 3x1 answer
remain zero-padded for NeuroGolf one-hot I/O.

Because every visible object is exactly a horizontal 1x2 pair, per-color pixel
counts and object counts induce the same ordering. The graph reduces the
one-hot input to all ten channel counts, uses TopK to take black/background,
filler, and the three marker colors, drops the first two ranks, turns the
remaining color indices into a compact [1,10,3,1] slab with OneHot, and pads
that slab to the required [1,10,30,30] output. The JSON outputs are always 3x1,
not full-width proportional bands.
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

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task391"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task391.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    five = _i64(inits, [5], "five")
    rank_st = _i64(inits, [2], "rank_st")
    depth = _i64(inits, [C], "depth")
    values = _init(inits, np.asarray([0.0, 1.0], dtype=np.float32), "values")

    nodes.append(helper.make_node("ReduceSum", [IN_NAME], ["counts"], axes=[0, 2, 3], keepdims=0))
    nodes.append(helper.make_node("TopK", ["counts", five], ["top_vals", "top_idx"], axis=0))
    nodes.append(helper.make_node("Slice", ["top_idx", rank_st, five], ["rank_idx"]))
    nodes.append(helper.make_node("Unsqueeze", ["rank_idx"], ["rank_ix3d"], axes=[0, 2]))
    nodes.append(helper.make_node("OneHot", ["rank_ix3d", depth, values], ["out3"], axis=1))
    nodes.append(
        helper.make_node(
            "Pad",
            ["out3"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - 3, W - 1],
        )
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


def validate_examples(path: Path) -> tuple[int, int, dict[str, int]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    split_passed: dict[str, int] = {}
    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        split_passed[split] = 0
        for idx, example in enumerate(data.get(split, [])):
            arr = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if arr is None or expected is None:
                continue
            actual = session.run([OUT_NAME], {IN_NAME: arr})[0]
            total += 1
            if not np.array_equal(actual > 0.0, expected > 0.0):
                raise AssertionError(f"{split}[{idx}] output mismatch")
            passed += 1
            split_passed[split] += 1
    return passed, total, split_passed


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    passed, total, split_passed = validate_examples(BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"correct: {passed}/{total}")
    print(
        "splits:  "
        f"train {split_passed['train']}, "
        f"test {split_passed['test']}, "
        f"arc-gen {split_passed['arc-gen']}"
    )
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
