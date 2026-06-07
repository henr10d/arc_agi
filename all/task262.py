"""Minimal ONNX for ARC task262: marker column chooses row stripe color.

Task rule: the logical input is a 3x3 grid containing black background and one
gray marker in each row.  A row whose marker is in the left column becomes a
solid red row, a marker in the middle column becomes a solid yellow row, and a
marker in the right column becomes a solid green row.  Padding outside the 3x3
task grid remains all-zero in the competition one-hot tensor.

ONNX: slice only the gray 3x3 crop, use one 1x3 Conv to map marker column to
compact output channels [red, green, yellow], edge-pad the singleton width to
three columns, then constant-pad channels/spatial dimensions to [1,10,30,30].
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task262"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
OPSET = 10


@dataclass(frozen=True)
class Variant:
    name: str
    build: Callable[[], onnx.ModelProto]


def _init(array: Any, name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(array), name=name)


def _make_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        initializer=inits,
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


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference solver: left/middle/right gray marker maps to red/yellow/green."""
    arr = np.asarray(grid, dtype=np.int64)
    out = np.full(arr.shape, 4, dtype=np.int64)
    out[arr[:, 0] == 5, :] = 2
    out[arr[:, 2] == 5, :] = 3
    return out


def _common_inits() -> list[onnx.TensorProto]:
    # Conv output channels are compact [red(2), green(3), yellow(4)].
    weights = np.asarray(
        [
            [[[1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 1.0]]],
            [[[0.0, 1.0, 0.0]]],
        ],
        dtype=np.float32,
    )
    return [
        _init(np.asarray([0, 5, 0, 0], dtype=np.int64), "starts"),
        _init(np.asarray([1, 6, 3, 3], dtype=np.int64), "ends"),
        _init(weights, "weights"),
    ]


def _final_pad(nodes: list[onnx.NodeProto], small: str) -> None:
    nodes.append(
        helper.make_node(
            "Pad",
            [small],
            [OUT_NAME],
            mode="constant",
            pads=[0, 2, 0, 0, 0, 5, 27, 27],
        )
    )


def build_edge_pad_model() -> onnx.ModelProto:
    """Lowest-cost graph found: Pad(mode=edge) repeats the singleton column."""
    inits = _common_inits()
    nodes = [
        helper.make_node("Slice", [IN_NAME, "starts", "ends"], ["gray"]),
        helper.make_node("Conv", ["gray", "weights"], ["row_colors"]),
        helper.make_node(
            "Pad",
            ["row_colors"],
            ["stripes"],
            mode="edge",
            pads=[0, 0, 0, 0, 0, 0, 0, 2],
        ),
    ]
    _final_pad(nodes, "stripes")
    return _make_model(nodes, inits, f"{TASK_ID}_edge_pad")


def variants() -> list[Variant]:
    return [
        Variant("edge-pad", build_edge_pad_model),
    ]


def load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def verify_rule() -> None:
    data = load_task_data()
    for split, examples in data.items():
        for idx, example in enumerate(examples):
            expected = np.asarray(example["output"], dtype=np.int64)
            actual = solve(example["input"])
            if not np.array_equal(actual, expected):
                raise AssertionError(f"rule mismatch in {split}[{idx}]")


def verify_model(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
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
            active = pred[0, :, :3, :3] > 0.0
            ok = np.array_equal(pred > 0.0, expected > 0.0) and np.all(active.sum(axis=0) == 1)
            passed += int(ok)
            checked += 1
            all_ok = all_ok and ok
        splits[split] = (passed, checked)
    return all_ok, splits


def benchmark_variants() -> tuple[dict[str, dict[str, Any]], str, dict[str, onnx.ModelProto]]:
    verify_rule()
    built = {variant.name: variant.build() for variant in variants()}
    results: dict[str, dict[str, Any]] = {}

    with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}_") as tmp:
        model_path = Path(tmp) / f"{TASK_ID}.onnx"
        for name, model in built.items():
            ok, splits = verify_model(model)
            onnx.save(model, model_path)
            result = score_file(model_path)
            result["correct"] = ok
            result["splits"] = splits
            results[name] = result

    valid = [
        (name, result)
        for name, result in results.items()
        if result["correct"] and result["valid"] and result["cost"] is not None
    ]
    if not valid:
        raise AssertionError(f"no valid correct variant: {results}")
    best_name, _ = min(valid, key=lambda item: int(item[1]["cost"]))
    return results, best_name, built


def main() -> None:
    results, best_name, built = benchmark_variants()
    best = built[best_name]
    onnx.save(best, BEST_PATH)
    final = score_file(BEST_PATH)

    for name, result in results.items():
        print(
            f"{name}: correct={result['correct']} splits={result['splits']} "
            f"memory={result['memory']} params={result['params']} "
            f"cost={result['cost']} score={result['score']}"
        )
        if result["error"]:
            print(f"{name} error: {result['error']}")

    print(f"wrote {BEST_PATH}")
    print(f"best:    {best_name}")
    print(f"valid:   {final['valid']}")
    if final["error"]:
        print(f"error:   {final['error']}")
    print(f"memory:  {final['memory']}")
    print(f"params:  {final['params']}")
    print(f"cost:    {final['cost']}")
    if final["score"] is not None:
        print(f"score:   {final['score']:.6f}")


if __name__ == "__main__":
    main()
