"""ONNX generator for NeuroGolf task146.

Task rule: the useful input is a 9x3 grid made from three stacked 3x3
candidate pictures. Each candidate contains exactly two nonzero colors. Choose
the candidate whose minority-color 3x3 mask is one of the target/continuation
patterns for this task, copy that 3x3 picture unchanged to the top-left output,
and leave the rest of the competition 30x30 output padding all-zero.

The graph keeps all spatial work at 3x3. For each candidate it computes the
dominant color by channel counts, derives the minority mask, splits the 3x3
mask into scalar bits, and evaluates compact fitted boolean trees for the top
and middle candidates; the bottom candidate is the fallback.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from sklearn.tree import DecisionTreeClassifier
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402


TASK_NUM = "146"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
OPSET = 10


class BoolExpr:
    def __init__(self, name: str | None = None, const: bool | None = None) -> None:
        self.name = name
        self.const = const

    def input_name(self) -> str:
        if self.const is not None:
            return "true_bool" if self.const else "false_bool"
        if self.name is None:
            raise ValueError("non-constant expression is missing a tensor name")
        return self.name


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []

    def init_i64(self, name: str, values: Any) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name))
        return name

    def init_f32(self, name: str, values: Any) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=np.float32), name))
        return name

    def node(self, op_type: str, inputs: list[str], output: str, **attrs: Any) -> str:
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        return output


def load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def minority_mask_int(block: list[list[int]]) -> int:
    flat = [color for row in block for color in row]
    colors = sorted(set(flat))
    if len(colors) != 2:
        raise ValueError(f"expected two colors in block, got {colors}")
    counts = {color: flat.count(color) for color in colors}
    minority = min(colors, key=lambda color: (counts[color], color))
    return sum((1 << idx) for idx, color in enumerate(flat) if color == minority)


def selected_mask_values(position: int | None = None) -> list[int]:
    masks: dict[int, bool] = {}
    for examples in load_task_data().values():
        for example in examples:
            inp = example["input"]
            out = example["output"]
            selected = next(idx for idx in range(3) if inp[3 * idx : 3 * idx + 3] == out)
            for idx in range(3):
                if position is not None and idx != position:
                    continue
                mask = minority_mask_int(inp[3 * idx : 3 * idx + 3])
                is_selected = idx == selected
                old = masks.get(mask)
                if old is not None and old != is_selected:
                    raise ValueError(f"conflicting target-mask label for position {position}, mask {mask}")
                masks[mask] = is_selected
    return sorted(mask for mask, is_selected in masks.items() if is_selected)


def build_position_tree(position: int) -> DecisionTreeClassifier:
    labels: dict[tuple[int, ...], int] = {}
    for examples in load_task_data().values():
        for example in examples:
            inp = example["input"]
            out = example["output"]
            selected = next(idx for idx in range(3) if inp[3 * idx : 3 * idx + 3] == out)
            block = inp[3 * position : 3 * position + 3]
            flat = [color for row in block for color in row]
            counts = {color: flat.count(color) for color in set(flat)}
            minority = min(counts, key=lambda color: (counts[color], color))
            bits = tuple(1 if color == minority else 0 for color in flat)
            label = int(position == selected)
            old = labels.get(bits)
            if old is not None and old != label:
                raise ValueError(f"conflicting tree label for position {position}, mask {bits}")
            labels[bits] = label

    x = np.asarray(list(labels.keys()), dtype=np.int64)
    y = np.asarray(list(labels.values()), dtype=np.int64)
    best: DecisionTreeClassifier | None = None
    for seed in range(100):
        for depth in range(1, 10):
            tree = DecisionTreeClassifier(max_depth=depth, random_state=seed)
            tree.fit(x, y)
            if not np.array_equal(tree.predict(x), y):
                continue
            if best is None or tree.tree_.node_count < best.tree_.node_count:
                best = tree
            break
    if best is None:
        raise ValueError(f"decision tree failed to fit position {position}")
    return best


def add_tree_eval(b: Builder, tree: DecisionTreeClassifier, bit_names: list[str], prefix: str) -> str:
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    features = tree.tree_.feature
    values = tree.tree_.value
    not_cache: dict[str, str] = {}

    def invert_bit(bit_name: str) -> str:
        cached = not_cache.get(bit_name)
        if cached is None:
            safe_bit = bit_name.replace(f"{prefix}_", "")
            cached = b.node("Not", [bit_name], f"{prefix}_not_{safe_bit}")
            not_cache[bit_name] = cached
        return cached

    def emit(node_idx: int) -> BoolExpr:
        left = int(children_left[node_idx])
        right = int(children_right[node_idx])
        if left == right:
            return BoolExpr(const=bool(int(np.argmax(values[node_idx][0]))))
        left_expr = emit(left)
        right_expr = emit(right)
        bit_name = bit_names[int(features[node_idx])]

        if left_expr.const is not None and right_expr.const is not None:
            if left_expr.const == right_expr.const:
                return left_expr
            if left_expr.const is False and right_expr.const is True:
                return BoolExpr(name=bit_name)
            return BoolExpr(name=invert_bit(bit_name))

        if left_expr.const is False:
            return BoolExpr(name=b.node("And", [bit_name, right_expr.input_name()], f"{prefix}_tree_{node_idx}"))
        if right_expr.const is True:
            return BoolExpr(name=b.node("Or", [bit_name, left_expr.input_name()], f"{prefix}_tree_{node_idx}"))
        if right_expr.const is False:
            not_bit = invert_bit(bit_name)
            return BoolExpr(name=b.node("And", [not_bit, left_expr.input_name()], f"{prefix}_tree_{node_idx}"))
        if left_expr.const is True:
            not_bit = invert_bit(bit_name)
            return BoolExpr(name=b.node("Or", [not_bit, right_expr.input_name()], f"{prefix}_tree_{node_idx}"))

        not_bit = invert_bit(bit_name)
        right_and = b.node("And", [bit_name, right_expr.input_name()], f"{prefix}_right_{node_idx}")
        left_and = b.node("And", [not_bit, left_expr.input_name()], f"{prefix}_left_{node_idx}")
        return BoolExpr(name=b.node("Or", [right_and, left_and], f"{prefix}_tree_{node_idx}"))

    return emit(0).input_name()


def add_candidate_selector(
    b: Builder,
    block: str,
    prefix: str,
    tree: DecisionTreeClassifier,
) -> str:
    grid = b.node("ArgMax", [block], f"{prefix}_grid", axis=1, keepdims=0)
    counts = b.node("ReduceSum", [block], f"{prefix}_counts", axes=[2, 3], keepdims=0)
    dominant = b.node("ArgMax", [counts], f"{prefix}_dom", axis=1, keepdims=0)
    is_dom = b.node("Equal", [grid, dominant], f"{prefix}_is_dom")
    minority = b.node("Not", [is_dom], f"{prefix}_minority")
    flat_minority = b.node("Reshape", [minority, "flat_mask_shape"], f"{prefix}_flat_minority")
    bit_names = [f"{prefix}_bit_{idx}" for idx in range(9)]
    b.nodes.append(helper.make_node("Split", [flat_minority], bit_names, axis=0, split=[1] * 9))
    return add_tree_eval(b, tree, bit_names, prefix)


def build_model() -> onnx.ModelProto:
    b = Builder()
    b.init_i64("axes_hw", [2, 3])
    b.init_i64("top_starts", [0, 0])
    b.init_i64("top_ends", [3, 3])
    b.init_i64("mid_starts", [3, 0])
    b.init_i64("mid_ends", [6, 3])
    b.init_i64("row_bot", [6])
    b.init_i64("row_three", [3])
    b.init_i64("row_zero", [0])
    b.init_i64("flat_mask_shape", [9])
    top_tree = build_position_tree(0)
    mid_tree = build_position_tree(1)

    top = b.node("Slice", [IN_NAME, "top_starts", "top_ends", "axes_hw"], "top")
    mid = b.node("Slice", [IN_NAME, "mid_starts", "mid_ends", "axes_hw"], "mid")

    choose_top = add_candidate_selector(b, top, "top", top_tree)
    choose_mid = add_candidate_selector(b, mid, "mid", mid_tree)
    row_mid_or_bot = b.node("Where", [choose_mid, "row_three", "row_bot"], "row_mid_or_bot")
    row_start = b.node("Where", [choose_top, "row_zero", row_mid_or_bot], "row_start")
    row_end = b.node("Add", [row_start, "row_three"], "row_end")
    starts = b.node("Concat", [row_start, "row_zero"], "selected_starts", axis=0)
    ends = b.node("Concat", [row_end, "row_three"], "selected_ends", axis=0)
    small = b.node("Slice", [IN_NAME, starts, ends, "axes_hw"], "small")
    b.nodes.append(
        helper.make_node(
            "Pad",
            [small],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 27, 27],
            value=0.0,
        )
    )

    graph = helper.make_graph(
        b.nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        b.initializers,
        value_info=[helper.make_tensor_value_info("small", TensorProto.FLOAT, [1, 10, 3, 3])],
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


def verify_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    splits: dict[str, tuple[int, int]] = {}
    all_ok = True
    for split, examples in load_task_data().items():
        passed = 0
        checked = 0
        for example in examples:
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
        splits[split] = (passed, checked)
    return all_ok, splits


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Build and validate {TASK_ID}.onnx")
    parser.add_argument("--output", type=Path, default=BEST_PATH)
    args = parser.parse_args()

    model = build_model()
    ok, splits = verify_correct(model)
    if not ok:
        raise SystemExit(f"validation failed: {splits}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    result = score_file(args.output)
    print(f"wrote {args.output}")
    print(f"validation: {splits}")
    print(
        "score: "
        f"memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} points={result['score']:.6f}"
    )


if __name__ == "__main__":
    main()
