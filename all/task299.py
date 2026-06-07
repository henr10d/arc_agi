"""Minimal ONNX for ARC task299 red/cyan line completion.

Task rule: in the 6x6 active grid, the input has exactly one horizontal red
(2) segment and exactly one vertical cyan (8) segment. Fill the red segment's
row with red, fill the cyan segment's column with cyan, paint their crossing
cell yellow (4), and leave every other active cell black. Cells outside the
6x6 grid stay NeuroGolf padding.
"""

from __future__ import annotations

import json
import sys
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

TASK_ID = "task299"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task299.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
TMP_DIR = Path("/tmp")

C = 10
H = W = 30
N = 6
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


@dataclass(frozen=True)
class Variant:
    name: str
    builder: Callable[[], onnx.ModelProto]


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _slice(
    nodes: list[onnx.NodeProto],
    data: str,
    out: str,
    starts: str,
    ends: str,
    axes: str,
) -> str:
    nodes.append(helper.make_node("Slice", [data, starts, ends, axes], [out]))
    return out


def _and(nodes: list[onnx.NodeProto], a: str, b: str, out: str) -> str:
    nodes.append(helper.make_node("And", [a, b], [out]))
    return out


def _make_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto]) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _line_masks(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto]) -> tuple[str, str]:
    axes = _i64(inits, [1, 2, 3], "axes")
    red_st = _i64(inits, [2, 0, 0], "red_st")
    red_en = _i64(inits, [3, N, N], "red_en")
    cyan_st = _i64(inits, [8, 0, 0], "cyan_st")
    cyan_en = _i64(inits, [9, N, N], "cyan_en")

    _slice(nodes, IN_NAME, "red6", red_st, red_en, axes)
    _slice(nodes, IN_NAME, "cyan6", cyan_st, cyan_en, axes)
    nodes.append(helper.make_node("ReduceSum", ["red6"], ["red_rows_f"], axes=[3], keepdims=1))
    nodes.append(helper.make_node("ReduceSum", ["cyan6"], ["cyan_cols_f"], axes=[2], keepdims=1))
    nodes.append(helper.make_node("Cast", ["red_rows_f"], ["red_rows"], to=TensorProto.BOOL))
    nodes.append(helper.make_node("Cast", ["cyan_cols_f"], ["cyan_cols"], to=TensorProto.BOOL))
    return "red_rows", "cyan_cols"


def _cropped_line_masks(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto]) -> tuple[str, str]:
    axes = _i64(inits, [1, 2, 3], "axes")
    red_st = _i64(inits, [2, 2, 0], "red_st")
    red_en = _i64(inits, [3, 5, N], "red_en")
    cyan_st = _i64(inits, [8, 0, 1], "cyan_st")
    cyan_en = _i64(inits, [9, 2, 5], "cyan_en")

    _slice(nodes, IN_NAME, "red_core", red_st, red_en, axes)
    _slice(nodes, IN_NAME, "cyan_core", cyan_st, cyan_en, axes)
    nodes.append(helper.make_node("ReduceSum", ["red_core"], ["red_core_rows_f"], axes=[3], keepdims=1))
    nodes.append(helper.make_node("ReduceSum", ["cyan_core"], ["cyan_core_cols_f"], axes=[2], keepdims=1))
    nodes.append(
        helper.make_node(
            "Pad",
            ["red_core_rows_f"],
            ["red_rows_f"],
            mode="constant",
            pads=[0, 0, 2, 0, 0, 0, 1, 0],
        )
    )
    nodes.append(
        helper.make_node(
            "Pad",
            ["cyan_core_cols_f"],
            ["cyan_cols_f"],
            mode="constant",
            pads=[0, 0, 0, 1, 0, 0, 0, 1],
        )
    )
    nodes.append(helper.make_node("Cast", ["red_rows_f"], ["red_rows"], to=TensorProto.BOOL))
    nodes.append(helper.make_node("Cast", ["cyan_cols_f"], ["cyan_cols"], to=TensorProto.BOOL))
    return "red_rows", "cyan_cols"


def _concat_from_masks(nodes: list[onnx.NodeProto], red_rows: str, cyan_cols: str) -> None:
    nodes.append(helper.make_node("Not", [red_rows], ["not_red_rows"]))
    nodes.append(helper.make_node("Not", [cyan_cols], ["not_cyan_cols"]))

    _and(nodes, "not_red_rows", "not_cyan_cols", "black")
    _and(nodes, red_rows, "not_cyan_cols", "red")
    _and(nodes, red_rows, cyan_cols, "yellow")
    _and(nodes, "not_red_rows", cyan_cols, "cyan")
    _and(nodes, "red", "cyan", "blank")

    nodes.append(
        helper.make_node(
            "Concat",
            ["black", "blank", "red", "blank", "yellow", "blank", "blank", "blank", "cyan", "blank"],
            ["out6_bool"],
            axis=1,
        )
    )
    nodes.append(helper.make_node("Cast", ["out6_bool"], ["out6"], to=TensorProto.FLOAT))
    nodes.append(
        helper.make_node(
            "Pad",
            ["out6"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - N, W - N],
        )
    )


def build_concat_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    red_rows, cyan_cols = _line_masks(nodes, inits)
    _concat_from_masks(nodes, red_rows, cyan_cols)
    return _make_model(nodes, inits)


def build_cropped_concat_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    red_rows, cyan_cols = _cropped_line_masks(nodes, inits)
    _concat_from_masks(nodes, red_rows, cyan_cols)
    return _make_model(nodes, inits)


def build_color_grid_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    red_rows, cyan_cols = _line_masks(nodes, inits)
    _and(nodes, red_rows, cyan_cols, "intersection")

    zero = _i64(inits, [0], "zero")
    red = _i64(inits, [2], "red")
    yellow = _i64(inits, [4], "yellow")
    cyan = _i64(inits, [8], "cyan")
    channels = _i64(inits, np.arange(C, dtype=np.int64).reshape(1, C, 1, 1), "channels")

    nodes.append(helper.make_node("Where", [red_rows, red, zero], ["red_grid"]))
    nodes.append(helper.make_node("Where", [cyan_cols, cyan, "red_grid"], ["line_grid"]))
    nodes.append(helper.make_node("Where", ["intersection", yellow, "line_grid"], ["color_grid"]))
    nodes.append(helper.make_node("Equal", [channels, "color_grid"], ["out6_bool"]))
    nodes.append(helper.make_node("Cast", ["out6_bool"], ["out6"], to=TensorProto.FLOAT))
    nodes.append(
        helper.make_node(
            "Pad",
            ["out6"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - N, W - N],
        )
    )
    return _make_model(nodes, inits)


VARIANTS = [
    Variant("cropped_concat", build_cropped_concat_model),
    Variant("concat_masks", build_concat_model),
    Variant("color_grid", build_color_grid_model),
]


def _examples() -> list[tuple[str, int, dict[str, Any]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    out: list[tuple[str, int, dict[str, Any]]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            out.append((split, idx, example))
    return out


def validate_task_structure() -> None:
    for split, idx, example in _examples():
        grid = example["input"]
        reds = [(r, c) for r, row in enumerate(grid) for c, color in enumerate(row) if color == 2]
        cyans = [(r, c) for r, row in enumerate(grid) for c, color in enumerate(row) if color == 8]
        if len({r for r, _ in reds}) != 1 or len({c for _, c in cyans}) != 1:
            raise AssertionError(f"{split}[{idx}] does not have unique red row and cyan column")
        if any(color not in {0, 2, 8} for row in grid for color in row):
            raise AssertionError(f"{split}[{idx}] contains an unexpected input color")


def validate_examples(path: Path) -> tuple[int, int, dict[str, int]]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    split_passed = {"train": 0, "test": 0, "arc-gen": 0}
    passed = 0
    total = 0
    for split, idx, example in _examples():
        inp = convert_to_numpy(example, "input")
        expected = convert_to_numpy(example, "output")
        if inp is None or expected is None:
            continue
        actual = session.run([OUT_NAME], {IN_NAME: inp})[0]
        total += 1
        if not np.array_equal(actual > 0.0, expected > 0.0):
            raise AssertionError(f"{split}[{idx}] output mismatch")
        passed += 1
        split_passed[split] += 1
    return passed, total, split_passed


def main() -> None:
    validate_task_structure()

    results = []
    for variant in VARIANTS:
        path = TMP_DIR / f"{TASK_ID}_{variant.name}.onnx"
        onnx.save(variant.builder(), path)
        try:
            passed, total, split_passed = validate_examples(path)
            correct = True
            note = f"{passed}/{total}"
        except Exception as exc:
            split_passed = {"train": 0, "test": 0, "arc-gen": 0}
            correct = False
            note = str(exc)
        result = score_file(path)
        results.append((variant, path, correct, note, split_passed, result))

    valid = [item for item in results if item[2] and item[5]["valid"]]
    if not valid:
        for variant, _, correct, note, _, result in results:
            print(
                f"{variant.name}: correct={correct} valid={result['valid']} "
                f"memory={result['memory']} params={result['params']} "
                f"cost={result['cost']} error={result['error']} note={note}"
            )
        raise SystemExit("no valid task299 variant")

    best = min(valid, key=lambda item: int(item[5]["cost"]))
    onnx.save(onnx.load(str(best[1])), BEST_PATH)

    print("variant comparison")
    for variant, _, correct, note, _, result in sorted(
        results, key=lambda item: (not item[5]["valid"], item[5]["cost"] or 10**18)
    ):
        score_text = "INVALID" if not result["valid"] else f"{result['score']:.6f}"
        print(
            f"{variant.name:<12} correct={correct} valid={result['valid']} "
            f"memory={result['memory']} params={result['params']} cost={result['cost']} "
            f"score={score_text} note={note}"
        )

    _, _, _, note, split_passed, result = best
    print(f"kept {best[0].name} -> {BEST_PATH}")
    print(f"correct: {note}")
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
