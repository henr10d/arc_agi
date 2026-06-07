"""Minimal ONNX for ARC task395 using two overlaid 3x3 masks.

Task rule: the 6x3 input is two stacked 3x3 layers. Treat any non-zero cell in
the top layer and any non-zero cell in the bottom layer as occupied in the same
coordinate frame. The 3x3 output marks color 2 exactly where both layers are
empty; all other output cells are black. Cells outside the 3x3 task output
remain zero-padded for the NeuroGolf one-hot interface.

ONNX approach: slice the color-0 one-hot channel from both 3x3 layers, cast
those occupancy-of-empty masks to bool, AND them into the color-2 condition,
then use a tiny three-channel Where template before padding channels 3-9 and
the spatial border directly into the excluded graph output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task395"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task395.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
N = 3
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


def _slice(
    nodes: List[onnx.NodeProto],
    data: str,
    out: str,
    starts: str,
    ends: str,
    axes: str,
) -> str:
    nodes.append(helper.make_node("Slice", [data, starts, ends, axes], [out]))
    return out


def _output_color() -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    colors = sorted(
        {
            cell
            for split in ("train", "test", "arc-gen")
            for example in data.get(split, [])
            for row in example["output"]
            for cell in row
            if cell != 0
        }
    )
    if len(colors) != 1:
        raise ValueError(f"{TASK_ID} expected exactly one non-black output color, got {colors}")
    return int(colors[0])


def build_model(red_color: int) -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, [1, 2, 3], "axes")
    top_st = _i64(inits, [0, 0, 0], "top_st")
    top_en = _i64(inits, [1, N, N], "top_en")
    bot_st = _i64(inits, [0, N, 0], "bot_st")
    bot_en = _i64(inits, [1, 2 * N, N], "bot_en")
    red_template = _init(
        inits,
        np.asarray([0.0, 0.0, 1.0], dtype=np.float32).reshape(1, 3, 1, 1),
        "red_template",
    )
    black_template = _init(
        inits,
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32).reshape(1, 3, 1, 1),
        "black_template",
    )

    _slice(nodes, IN_NAME, "top0", top_st, top_en, axes)
    _slice(nodes, IN_NAME, "bot0", bot_st, bot_en, axes)
    nodes.append(helper.make_node("Cast", ["top0"], ["top_empty"], to=TensorProto.BOOL))
    nodes.append(helper.make_node("Cast", ["bot0"], ["bot_empty"], to=TensorProto.BOOL))
    nodes.append(helper.make_node("And", ["top_empty", "bot_empty"], ["red_mask"]))
    if red_color != 2:
        raise ValueError(f"{TASK_ID} expected output color 2, got {red_color}")
    nodes.append(helper.make_node("Where", ["red_mask", red_template, black_template], ["out3"]))
    nodes.append(
        helper.make_node(
            "Pad",
            ["out3"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, C - 3, H - N, W - N],
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


def _predict(example: dict, rule: str, red_color: int) -> list[list[int]]:
    x = np.asarray(example["input"])
    top = x[:N] != 0
    bottom = x[N : 2 * N] != 0
    flipped = np.flipud(bottom)

    rules: dict[str, Callable[[], np.ndarray]] = {
        "xor_flip": lambda: top ^ flipped,
        "xor_no_flip": lambda: top ^ bottom,
        "red_not_blue_flip": lambda: top & ~flipped,
        "not_red_blue_flip": lambda: ~top & flipped,
        "empty_both_no_flip": lambda: ~(top | bottom),
    }
    mask = rules[rule]()
    return np.where(mask, red_color, 0).tolist()


def compare_reference_rules(red_color: int) -> dict[str, tuple[int, int]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    rules = (
        "xor_flip",
        "xor_no_flip",
        "red_not_blue_flip",
        "not_red_blue_flip",
        "empty_both_no_flip",
    )
    train = data.get("train", [])
    return {
        rule: (sum(_predict(example, rule, red_color) == example["output"] for example in train), len(train))
        for rule in rules
    }


def validate_examples(path: Path) -> tuple[int, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            expected = convert_to_numpy(example, "output")
            actual = session.run([OUT_NAME], {IN_NAME: convert_to_numpy(example, "input")})[0]
            total += 1
            if not np.array_equal(actual > 0.0, expected > 0.0):
                raise AssertionError(f"{split}[{idx}] output mismatch")
            passed += 1
    return passed, total


def main() -> None:
    red_color = _output_color()
    model = build_model(red_color)
    onnx.save(model, BEST_PATH)

    rule_results = compare_reference_rules(red_color)
    passed, total = validate_examples(BEST_PATH)
    result = score_file(BEST_PATH)

    print(f"wrote {BEST_PATH}")
    print(f"output red color: {red_color}")
    for name, (ok, count) in rule_results.items():
        print(f"{name} train rule: {ok}/{count}")
    print(f"correct: {passed}/{total}")
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
