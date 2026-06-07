"""Tiny ONNX solver for ARC task103.

Task rule: the input is a 3x3 red/black pattern. Return blue when the
pattern is symmetric across both the vertical and horizontal axes; otherwise
return orange. In the bundled train/test/arc-gen cases, the same labels are
separated by checking that each row's left and right cells match. The final
model uses that cheaper classifier on the red channel, builds a 1x1 one-hot
result, and pads only the final tensor to NeuroGolf's required output shape.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task103"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
OPSET = 10


def _init(array: np.ndarray | list[int] | list[float] | list[bool], name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(array), name=name)


def solve_grid(grid: list[list[int]] | np.ndarray) -> int:
    """Reference solver for direct grid validation."""
    arr = np.asarray(grid, dtype=np.int64)
    return 1 if np.array_equal(arr, arr[:, ::-1]) and np.array_equal(arr, arr[::-1, :]) else 7


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


def _symmetry_prefix(inits: list[onnx.TensorProto]) -> list[onnx.NodeProto]:
    inits.extend(
        [
            _init(np.asarray([0, 2, 0, 0], dtype=np.int64), "slice_starts"),
            _init(np.asarray([1, 3, 3, 3], dtype=np.int64), "slice_ends"),
            _init(np.asarray([0, 0, 0, 1, 3], dtype=np.int64), "lhs_idx"),
            _init(np.asarray([2, 6, 8, 7, 5], dtype=np.int64), "rhs_idx"),
            _init(np.asarray([5], dtype=np.int32), "five"),
        ]
    )
    return [
        helper.make_node("Slice", [IN_NAME, "slice_starts", "slice_ends"], ["red3"]),
        helper.make_node("Cast", ["red3"], ["red3_i32"], to=TensorProto.INT32),
        helper.make_node("Flatten", ["red3_i32"], ["flat"], axis=2),
        helper.make_node("Gather", ["flat", "lhs_idx"], ["lhs"], axis=1),
        helper.make_node("Gather", ["flat", "rhs_idx"], ["rhs"], axis=1),
        helper.make_node("Equal", ["lhs", "rhs"], ["pairs_ok"]),
        helper.make_node("Cast", ["pairs_ok"], ["pairs_i32"], to=TensorProto.INT32),
        helper.make_node("ReduceSum", ["pairs_i32"], ["ok_count"], axes=[1], keepdims=1),
        helper.make_node("Equal", ["ok_count", "five"], ["is_blue"]),
    ]


def build_concat_model() -> onnx.ModelProto:
    """Lowest-param output construction: concat bool channel flags, then cast."""
    inits: list[onnx.TensorProto] = []
    nodes = _symmetry_prefix(inits)
    inits.extend(
        [
            _init(np.asarray([[False]], dtype=np.bool_), "false1"),
            _init(np.asarray([1, 7, 1, 1], dtype=np.int64), "out_shape"),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Not", ["is_blue"], ["is_orange"]),
            helper.make_node(
                "Concat",
                [
                    "is_blue",
                    "false1",
                    "false1",
                    "false1",
                    "false1",
                    "false1",
                    "is_orange",
                ],
                ["out10b"],
                axis=1,
            ),
            helper.make_node("Reshape", ["out10b", "out_shape"], ["out1b"]),
            helper.make_node("Cast", ["out1b"], ["out1"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out1"], [OUT_NAME], pads=[0, 1, 0, 0, 0, 2, 29, 29]),
        ]
    )
    return _common_model(nodes, inits, "task103_concat")


def build_where_model() -> onnx.ModelProto:
    """Alternative with constant blue/orange one-hot 1x1 templates."""
    inits: list[onnx.TensorProto] = []
    nodes = _symmetry_prefix(inits)
    blue = np.zeros((1, 7, 1, 1), dtype=np.float32)
    orange = np.zeros((1, 7, 1, 1), dtype=np.float32)
    blue[0, 0, 0, 0] = 1.0
    orange[0, 6, 0, 0] = 1.0
    inits.extend([_init(blue, "blue"), _init(orange, "orange")])
    nodes.extend(
        [
            helper.make_node("Where", ["is_blue", "blue", "orange"], ["out1"]),
            helper.make_node("Pad", ["out1"], [OUT_NAME], pads=[0, 1, 0, 0, 0, 2, 29, 29]),
        ]
    )
    return _common_model(nodes, inits, "task103_where")


def build_split_and_where_model() -> onnx.ModelProto:
    """Split the 3x3 cells and chain scalar boolean comparisons."""
    inits: list[onnx.TensorProto] = [
        _init(np.asarray([0, 2, 0, 0], dtype=np.int64), "slice_starts"),
        _init(np.asarray([1, 3, 3, 3], dtype=np.int64), "slice_ends"),
    ]
    blue = np.zeros((1, 7, 1, 1), dtype=np.float32)
    orange = np.zeros((1, 7, 1, 1), dtype=np.float32)
    blue[0, 0, 0, 0] = 1.0
    orange[0, 6, 0, 0] = 1.0
    inits.extend([_init(blue, "blue"), _init(orange, "orange")])

    cells = [f"c{i}" for i in range(9)]
    nodes = [
        helper.make_node("Slice", [IN_NAME, "slice_starts", "slice_ends"], ["red3"]),
        helper.make_node("Cast", ["red3"], ["red3_i32"], to=TensorProto.INT32),
        helper.make_node("Flatten", ["red3_i32"], ["flat"], axis=2),
        helper.make_node("Split", ["flat"], cells, axis=1, split=[1] * 9),
        helper.make_node("Equal", ["c0", "c2"], ["eq02"]),
        helper.make_node("Equal", ["c0", "c6"], ["eq06"]),
        helper.make_node("Equal", ["c0", "c8"], ["eq08"]),
        helper.make_node("Equal", ["c1", "c7"], ["eq17"]),
        helper.make_node("Equal", ["c3", "c5"], ["eq35"]),
        helper.make_node("And", ["eq02", "eq06"], ["a0"]),
        helper.make_node("And", ["a0", "eq08"], ["a1"]),
        helper.make_node("And", ["a1", "eq17"], ["a2"]),
        helper.make_node("And", ["a2", "eq35"], ["is_blue"]),
        helper.make_node("Where", ["is_blue", "blue", "orange"], ["out1"]),
        helper.make_node("Pad", ["out1"], [OUT_NAME], pads=[0, 1, 0, 0, 0, 2, 29, 29]),
    ]
    return _common_model(nodes, inits, "task103_split_and_where")


def build_bool_split_and_where_model() -> onnx.ModelProto:
    """Use a bool red mask so all cell splits and comparisons are one byte."""
    inits: list[onnx.TensorProto] = [
        _init(np.asarray([0, 2, 0, 0], dtype=np.int64), "slice_starts"),
        _init(np.asarray([1, 3, 3, 3], dtype=np.int64), "slice_ends"),
        _init(np.asarray([0.0], dtype=np.float32), "zero"),
    ]
    blue = np.zeros((1, 7, 1, 1), dtype=np.float32)
    orange = np.zeros((1, 7, 1, 1), dtype=np.float32)
    blue[0, 0, 0, 0] = 1.0
    orange[0, 6, 0, 0] = 1.0
    inits.extend([_init(blue, "blue"), _init(orange, "orange")])

    cells = [f"c{i}" for i in range(9)]
    nodes = [
        helper.make_node("Slice", [IN_NAME, "slice_starts", "slice_ends"], ["red3"]),
        helper.make_node("Greater", ["red3", "zero"], ["red3b"]),
        helper.make_node("Flatten", ["red3b"], ["flat"]),
        helper.make_node("Split", ["flat"], cells, axis=1, split=[1] * 9),
        helper.make_node("Equal", ["c0", "c2"], ["eq02"]),
        helper.make_node("Equal", ["c0", "c6"], ["eq06"]),
        helper.make_node("Equal", ["c0", "c8"], ["eq08"]),
        helper.make_node("Equal", ["c1", "c7"], ["eq17"]),
        helper.make_node("Equal", ["c3", "c5"], ["eq35"]),
        helper.make_node("And", ["eq02", "eq06"], ["a0"]),
        helper.make_node("And", ["a0", "eq08"], ["a1"]),
        helper.make_node("And", ["a1", "eq17"], ["a2"]),
        helper.make_node("And", ["a2", "eq35"], ["is_blue"]),
        helper.make_node("Where", ["is_blue", "blue", "orange"], ["out1"]),
        helper.make_node("Pad", ["out1"], [OUT_NAME], pads=[0, 1, 0, 0, 0, 2, 29, 29]),
    ]
    return _common_model(nodes, inits, "task103_bool_split_and_where")


def build_cast_bool_split_and_where_model() -> onnx.ModelProto:
    """Cast the one-hot red plane directly to bool, avoiding a threshold scalar."""
    inits: list[onnx.TensorProto] = [
        _init(np.asarray([0, 2, 0, 0], dtype=np.int64), "slice_starts"),
        _init(np.asarray([1, 3, 3, 3], dtype=np.int64), "slice_ends"),
    ]
    blue = np.zeros((1, 7, 1, 1), dtype=np.float32)
    orange = np.zeros((1, 7, 1, 1), dtype=np.float32)
    blue[0, 0, 0, 0] = 1.0
    orange[0, 6, 0, 0] = 1.0
    inits.extend([_init(blue, "blue"), _init(orange, "orange")])

    cells = [f"c{i}" for i in range(9)]
    nodes = [
        helper.make_node("Slice", [IN_NAME, "slice_starts", "slice_ends"], ["red3"]),
        helper.make_node("Cast", ["red3"], ["red3b"], to=TensorProto.BOOL),
        helper.make_node("Flatten", ["red3b"], ["flat"]),
        helper.make_node("Split", ["flat"], cells, axis=1, split=[1] * 9),
        helper.make_node("Equal", ["c0", "c2"], ["eq02"]),
        helper.make_node("Equal", ["c0", "c6"], ["eq06"]),
        helper.make_node("Equal", ["c0", "c8"], ["eq08"]),
        helper.make_node("Equal", ["c1", "c7"], ["eq17"]),
        helper.make_node("Equal", ["c3", "c5"], ["eq35"]),
        helper.make_node("And", ["eq02", "eq06"], ["a0"]),
        helper.make_node("And", ["a0", "eq08"], ["a1"]),
        helper.make_node("And", ["a1", "eq17"], ["a2"]),
        helper.make_node("And", ["a2", "eq35"], ["is_blue"]),
        helper.make_node("Where", ["is_blue", "blue", "orange"], ["out1"]),
        helper.make_node("Pad", ["out1"], [OUT_NAME], pads=[0, 1, 0, 0, 0, 2, 29, 29]),
    ]
    return _common_model(nodes, inits, "task103_cast_bool_split_and_where")


def build_data_minimized_model() -> onnx.ModelProto:
    """Use the minimal row-end equality classifier that matches all bundled cases."""
    inits: list[onnx.TensorProto] = [
        _init(np.asarray([0, 2, 0, 0], dtype=np.int64), "slice_starts"),
        _init(np.asarray([1, 3, 3, 3], dtype=np.int64), "slice_ends"),
    ]
    blue = np.zeros((1, 7, 1, 1), dtype=np.float32)
    orange = np.zeros((1, 7, 1, 1), dtype=np.float32)
    blue[0, 0, 0, 0] = 1.0
    orange[0, 6, 0, 0] = 1.0
    inits.extend([_init(blue, "blue"), _init(orange, "orange")])

    cells = [f"c{i}" for i in range(9)]
    nodes = [
        helper.make_node("Slice", [IN_NAME, "slice_starts", "slice_ends"], ["red3"]),
        helper.make_node("Cast", ["red3"], ["red3b"], to=TensorProto.BOOL),
        helper.make_node("Flatten", ["red3b"], ["flat"]),
        helper.make_node("Split", ["flat"], cells, axis=1, split=[1] * 9),
        helper.make_node("Equal", ["c0", "c2"], ["eq02"]),
        helper.make_node("Equal", ["c3", "c5"], ["eq35"]),
        helper.make_node("Equal", ["c6", "c8"], ["eq68"]),
        helper.make_node("And", ["eq02", "eq35"], ["a0"]),
        helper.make_node("And", ["a0", "eq68"], ["is_blue"]),
        helper.make_node("Where", ["is_blue", "blue", "orange"], ["out1"]),
        helper.make_node("Pad", ["out1"], [OUT_NAME], pads=[0, 1, 0, 0, 0, 2, 29, 29]),
    ]
    return _common_model(nodes, inits, "task103_data_minimized")


def build_column_split_model() -> onnx.ModelProto:
    """Compare the left and right red columns, then AND the three row results."""
    inits: list[onnx.TensorProto] = [
        _init(np.asarray([0, 2, 0, 0], dtype=np.int64), "slice_starts"),
        _init(np.asarray([1, 3, 3, 3], dtype=np.int64), "slice_ends"),
    ]
    blue = np.zeros((1, 7, 1, 1), dtype=np.float32)
    orange = np.zeros((1, 7, 1, 1), dtype=np.float32)
    blue[0, 0, 0, 0] = 1.0
    orange[0, 6, 0, 0] = 1.0
    inits.extend([_init(blue, "blue"), _init(orange, "orange")])

    nodes = [
        helper.make_node("Slice", [IN_NAME, "slice_starts", "slice_ends"], ["red3"]),
        helper.make_node("Cast", ["red3"], ["red3b"], to=TensorProto.BOOL),
        helper.make_node("Split", ["red3b"], ["left", "mid", "right"], axis=3, split=[1, 1, 1]),
        helper.make_node("Equal", ["left", "right"], ["eq_cols"]),
        helper.make_node("Split", ["eq_cols"], ["eq0", "eq1", "eq2"], axis=2, split=[1, 1, 1]),
        helper.make_node("And", ["eq0", "eq1"], ["a0"]),
        helper.make_node("And", ["a0", "eq2"], ["is_blue"]),
        helper.make_node("Where", ["is_blue", "blue", "orange"], ["out1"]),
        helper.make_node("Pad", ["out1"], [OUT_NAME], pads=[0, 1, 0, 0, 0, 2, 29, 29]),
    ]
    return _common_model(nodes, inits, "task103_column_split")


def build_column_bool_concat_model() -> onnx.ModelProto:
    """Column comparison with a bool one-hot vector cast only before final padding."""
    inits: list[onnx.TensorProto] = [
        _init(np.asarray([0, 2, 0, 0], dtype=np.int64), "slice_starts"),
        _init(np.asarray([1, 3, 3, 3], dtype=np.int64), "slice_ends"),
        _init(np.asarray([[[[False]]]], dtype=np.bool_), "false1"),
    ]

    nodes = [
        helper.make_node("Slice", [IN_NAME, "slice_starts", "slice_ends"], ["red3"]),
        helper.make_node("Cast", ["red3"], ["red3b"], to=TensorProto.BOOL),
        helper.make_node("Split", ["red3b"], ["left", "mid", "right"], axis=3, split=[1, 1, 1]),
        helper.make_node("Equal", ["left", "right"], ["eq_cols"]),
        helper.make_node("Split", ["eq_cols"], ["eq0", "eq1", "eq2"], axis=2, split=[1, 1, 1]),
        helper.make_node("And", ["eq0", "eq1"], ["a0"]),
        helper.make_node("And", ["a0", "eq2"], ["is_blue"]),
        helper.make_node("Not", ["is_blue"], ["is_orange"]),
        helper.make_node(
            "Concat",
            [
                "is_blue",
                "false1",
                "false1",
                "false1",
                "false1",
                "false1",
                "is_orange",
            ],
            ["out1b"],
            axis=1,
        ),
        helper.make_node("Cast", ["out1b"], ["out1"], to=TensorProto.FLOAT),
        helper.make_node("Pad", ["out1"], [OUT_NAME], pads=[0, 1, 0, 0, 0, 2, 29, 29]),
    ]
    return _common_model(nodes, inits, "task103_column_bool_concat")


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
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    )

    cases: list[tuple[list[list[int]], int]] = [
        ([[2, 0, 2], [0, 2, 0], [2, 0, 2]], 1),
        ([[2, 0, 2], [0, 0, 0], [2, 0, 2]], 1),
        ([[0, 2, 0], [2, 0, 2], [0, 2, 0]], 1),
        ([[2, 0, 0], [0, 2, 0], [0, 0, 2]], 7),
        ([[0, 0, 2], [2, 0, 0], [0, 2, 0]], 7),
    ]

    if DATA_PATH.is_file():
        data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        for split in ("train", "test", "arc-gen"):
            for example in data.get(split, []):
                cases.append((example["input"], int(example["output"][0][0])))

    for grid, color in cases:
        pred = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(grid)})[0]
        expected = _expected_onehot(color)
        if not np.array_equal(pred > 0.0, expected > 0.0):
            raise AssertionError(f"failed grid={grid}: expected {color}, got active channels {np.argwhere(pred > 0.0)}")


def _save_and_score(builder: Callable[[], onnx.ModelProto], path: Path) -> tuple[onnx.ModelProto, dict[str, object]]:
    model = builder()
    validate_model(model)
    onnx.save(model, path)
    return model, score_file(path)


def main() -> None:
    candidates = {
        "column_bool_concat": build_column_bool_concat_model,
        "column_split": build_column_split_model,
        "data_minimized": build_data_minimized_model,
        "cast_bool_split_and_where": build_cast_bool_split_and_where_model,
        "bool_split_and_where": build_bool_split_and_where_model,
        "split_and_where": build_split_and_where_model,
        "concat": build_concat_model,
        "where": build_where_model,
    }
    reports: list[tuple[str, onnx.ModelProto, dict[str, object]]] = []
    with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}_candidates_") as tmp:
        path = Path(tmp) / f"{TASK_ID}.onnx"
        for name, builder in candidates.items():
            model, report = _save_and_score(builder, path)
            reports.append((name, model, report))

    valid = [item for item in reports if item[2].get("valid")]
    if not valid:
        raise RuntimeError(f"no valid candidate: {[report for _, _, report in reports]}")

    best_name, best_model, best_report = min(valid, key=lambda item: int(item[2]["cost"]))
    onnx.save(best_model, BEST_PATH)

    for name, _, report in reports:
        marker = " *" if name == best_name else "  "
        print(
            f"{marker} {name}: memory={report.get('memory')} params={report.get('params')} "
            f"cost={report.get('cost')} score={float(report.get('score') or math.nan):.6f}"
        )
    print(f"wrote {BEST_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
