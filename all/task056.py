"""Compact ONNX solver for NeuroGolf task056.

Task rule: the input is always a 3x3 grid with black background and one
non-black foreground color. Ignore the foreground color and classify only the
occupied-cell mask: blue for ``XX./X.X/.X.``, red for ``X.X/.X./X.X``, green
for ``.XX/.XX/X..``, and magenta for ``.X./XXX/.X.``. The output is that class
color as a 1x1 cell, padded to the required NeuroGolf one-hot tensor.

ONNX approach: the best candidate reads the black-channel values at three
decisive cells. For the known valid masks, the center-black bit is directly
the color-1 class; the other classes are derived with two tiny boolean
intersections. A second candidate builds the full 3x3 nonzero mask, encodes it
as a 9-bit integer, and maps the known bitmasks to output colors.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task056"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
OPSET = 10

PATTERN_TO_COLOR = {
    (1, 1, 0, 1, 0, 1, 0, 1, 0): 1,
    (1, 0, 1, 0, 1, 0, 1, 0, 1): 2,
    (0, 1, 1, 0, 1, 1, 1, 0, 0): 3,
    (0, 1, 0, 1, 1, 1, 0, 1, 0): 6,
}


def _init(array: np.ndarray | list[int] | list[float] | list[bool] | bool, name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(array), name=name)


def _common_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        name,
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


def solve_grid(grid: list[list[int]] | np.ndarray) -> int:
    mask = tuple(int(v != 0) for v in np.asarray(grid, dtype=np.int64).reshape(-1)[:9])
    try:
        return PATTERN_TO_COLOR[mask]
    except KeyError as exc:
        raise ValueError(f"unknown task056 mask {mask}") from exc


def build_three_cell_model() -> onnx.ModelProto:
    """Classify from the three decisive black-channel bits at positions 2, 4, and 8."""
    inits = [
        _init([0, 0, 0, 2], "s2"),
        _init([1, 1, 1, 3], "e2"),
        _init([0, 0, 1, 1], "s4"),
        _init([1, 1, 2, 2], "e4"),
        _init([0, 0, 2, 2], "s8"),
        _init([1, 1, 3, 3], "e8"),
        _init(np.zeros((1, 1, 1, 1), dtype=np.bool_), "false4"),
    ]
    nodes = [
        helper.make_node("Slice", [IN_NAME, "s2", "e2"], ["k2"]),
        helper.make_node("Slice", [IN_NAME, "s4", "e4"], ["k4"]),
        helper.make_node("Slice", [IN_NAME, "s8", "e8"], ["k8"]),
        helper.make_node("Cast", ["k2"], ["b2"], to=TensorProto.BOOL),
        helper.make_node("Cast", ["k4"], ["b4"], to=TensorProto.BOOL),
        helper.make_node("Cast", ["k8"], ["b8"], to=TensorProto.BOOL),
        helper.make_node("Not", ["b8"], ["red"]),
        helper.make_node("Not", ["b2"], ["nb2"]),
        helper.make_node("Not", ["b4"], ["nb4"]),
        helper.make_node("And", ["nb2", "b8"], ["green"]),
        helper.make_node("And", ["nb4", "b2"], ["magenta"]),
        helper.make_node(
            "Concat",
            ["b4", "red", "green", "false4", "false4", "magenta"],
            ["out6b"],
            axis=1,
        ),
        helper.make_node("Cast", ["out6b"], ["out6"], to=TensorProto.FLOAT),
        helper.make_node("Pad", ["out6"], [OUT_NAME], pads=[0, 1, 0, 0, 0, 3, 29, 29]),
    ]
    return _common_model(nodes, inits, "task056_three_cell")


def build_nonzero_bitmask_model() -> onnx.ModelProto:
    """Build the full nonzero mask, encode row-major bits, then compare known codes."""
    inits = [
        _init([0, 0, 0, 0], "crop_starts"),
        _init([1, 1, 3, 3], "crop_ends"),
        _init(np.asarray([[1], [2], [4], [8], [16], [32], [64], [128], [256]], dtype=np.float32), "bit_weights"),
        _init(np.asarray([[170.5, 340.5, 117.5, 185.5]], dtype=np.float32), "code_lo"),
        _init(np.asarray([[171.5, 341.5, 118.5, 186.5]], dtype=np.float32), "code_hi"),
        _init(np.zeros((1, 1), dtype=np.bool_), "false2"),
        _init([1, 7, 1, 1], "out_shape"),
    ]
    nodes = [
        helper.make_node("Slice", [IN_NAME, "crop_starts", "crop_ends"], ["black3"]),
        helper.make_node("Cast", ["black3"], ["black3b"], to=TensorProto.BOOL),
        helper.make_node("Not", ["black3b"], ["fg3b"]),
        helper.make_node("Flatten", ["fg3b"], ["fg_flatb"]),
        helper.make_node("Cast", ["fg_flatb"], ["fg_flat"], to=TensorProto.FLOAT),
        helper.make_node("MatMul", ["fg_flat", "bit_weights"], ["code"]),
        helper.make_node("Greater", ["code", "code_lo"], ["above"]),
        helper.make_node("Less", ["code", "code_hi"], ["below"]),
        helper.make_node("And", ["above", "below"], ["matches"]),
        helper.make_node("Split", ["matches"], ["blue", "red", "green", "magenta"], axis=1, split=[1, 1, 1, 1]),
        helper.make_node(
            "Concat",
            ["false2", "blue", "red", "green", "false2", "false2", "magenta"],
            ["out7b_flat"],
            axis=1,
        ),
        helper.make_node("Reshape", ["out7b_flat", "out_shape"], ["out7b"]),
        helper.make_node("Cast", ["out7b"], ["out7"], to=TensorProto.FLOAT),
        helper.make_node("Pad", ["out7"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 3, 29, 29]),
    ]
    return _common_model(nodes, inits, "task056_nonzero_bitmask")


def _grid_to_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _expected_onehot(color: int) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    out[0, color, 0, 0] = 1.0
    return out


def validate_model(model: onnx.ModelProto) -> None:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            expected_color = int(example["output"][0][0])
            ref_color = solve_grid(example["input"])
            if ref_color != expected_color:
                raise AssertionError(f"rule mismatch in {split}[{idx}]: expected {expected_color}, ref {ref_color}")

            pred = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(example["input"])})[0]
            expected = _expected_onehot(expected_color)
            if not np.array_equal(pred > 0.0, expected > 0.0):
                active = np.argwhere(pred > 0.0).tolist()
                raise AssertionError(f"ONNX mismatch in {split}[{idx}]: expected {expected_color}, got {active}")


def _save_and_score(builder: Callable[[], onnx.ModelProto], path: Path) -> tuple[onnx.ModelProto, dict[str, object]]:
    model = builder()
    validate_model(model)
    onnx.save(model, path)
    return model, score_file(path)


def main() -> None:
    candidates = {
        "three_cell": build_three_cell_model,
        "nonzero_bitmask": build_nonzero_bitmask_model,
    }
    reports: list[tuple[str, onnx.ModelProto, dict[str, object]]] = []
    for name, builder in candidates.items():
        model, report = _save_and_score(builder, OUT_DIR / f"{TASK_ID}_{name}.onnx")
        reports.append((name, model, report))

    valid = [item for item in reports if item[2].get("valid")]
    if not valid:
        raise RuntimeError(f"no valid candidate: {[report for _, _, report in reports]}")

    best_name, best_model, _best_report = min(valid, key=lambda item: int(item[2]["cost"]))
    onnx.save(best_model, BEST_PATH)

    for name, _model, report in reports:
        marker = " *" if name == best_name else "  "
        print(
            f"{marker} {name}: memory={report.get('memory')} params={report.get('params')} "
            f"cost={report.get('cost')} score={float(report.get('score') or math.nan):.6f}"
        )
    print(f"wrote {BEST_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
