"""ONNX for ARC task292: recolor every third olive column marker.

Task rule: the input is a 3 x N grid containing black background and olive
cells.  In each of the three rows, olive cells in zero-based columns divisible
by 3 become magenta; black cells and olive cells in all other columns remain
unchanged.  The padded area outside the original grid remains all-zero for the
competition tensor contract.

ONNX approach: slice the olive and magenta channels over the first three rows,
multiply the olive slice by a small fixed column mask, subtract those marked
olive cells from channel 4, add them to channel 6, then concatenate the original
unchanged channel slices around the two edited channels.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task292"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task292.onnx"
DATA_PATH = ROOT / "data" / "task292.json"

C = 10
H = W = 30
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10
OLIVE = 4
MAGENTA = 6


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _make_model(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto]) -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
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


def load_examples() -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    examples: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            examples.append(
                (
                    split,
                    idx,
                    np.asarray(ex["input"], dtype=np.int64),
                    np.asarray(ex["output"], dtype=np.int64),
                )
            )
    return examples


def solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    out[:, ::3] = np.where(out[:, ::3] == OLIVE, MAGENTA, out[:, ::3])
    return out


def _grid_to_onehot(grid: np.ndarray) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            out[0, int(grid[r, c]), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    active = onehot > 0.0
    return active.argmax(axis=1)[0].astype(np.int64)


def _run_onnx(model: onnx.ModelProto, grid: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: _grid_to_onehot(grid)})[0]


def _slice(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    x: str,
    out: str,
    starts: list[int],
    ends: list[int],
    axes: list[int],
) -> str:
    nodes.append(
        helper.make_node(
            "Slice",
            [
                x,
                _i64(inits, starts, f"{out}_starts"),
                _i64(inits, ends, f"{out}_ends"),
                _i64(inits, axes, f"{out}_axes"),
            ],
            [out],
        )
    )
    return out


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    ch0_4 = _slice(nodes, inits, IN_NAME, "ch0_4", [0], [4], [1])
    ch4 = _slice(nodes, inits, IN_NAME, "ch4", [4], [5], [1])
    ch5 = _slice(nodes, inits, IN_NAME, "ch5", [5], [6], [1])
    ch6 = _slice(nodes, inits, IN_NAME, "ch6", [6], [7], [1])
    ch7_10 = _slice(nodes, inits, IN_NAME, "ch7_10", [7], [10], [1])

    ch4_top = _slice(nodes, inits, ch4, "ch4_top", [0], [3], [2])
    ch4_bottom = _slice(nodes, inits, ch4, "ch4_bottom", [3], [30], [2])
    ch6_top = _slice(nodes, inits, ch6, "ch6_top", [0], [3], [2])
    ch6_bottom = _slice(nodes, inits, ch6, "ch6_bottom", [3], [30], [2])

    mask = np.zeros((1, 1, 3, W), dtype=np.float32)
    mask[:, :, :, ::3] = 1.0
    nodes.append(helper.make_node("Mul", [ch4_top, _f32(inits, mask, "marker_mask")], ["moved"]))
    nodes.append(helper.make_node("Sub", [ch4_top, "moved"], ["ch4_top_out"]))
    nodes.append(helper.make_node("Add", [ch6_top, "moved"], ["ch6_top_out"]))
    nodes.append(helper.make_node("Concat", ["ch4_top_out", ch4_bottom], ["ch4_out"], axis=2))
    nodes.append(helper.make_node("Concat", ["ch6_top_out", ch6_bottom], ["ch6_out"], axis=2))
    nodes.append(helper.make_node("Concat", [ch0_4, "ch4_out", ch5, "ch6_out", ch7_10], [OUT_NAME], axis=1))
    return _make_model(nodes, inits)


def validate_reference(examples: list[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    for split, idx, inp, expected in examples:
        got = solve(inp)
        if not np.array_equal(got, expected):
            raise AssertionError(f"reference failed {split}[{idx}]")


def validate_model(model: onnx.ModelProto, examples: list[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    for split, idx, inp, expected in examples:
        pred = _onehot_to_grid(_run_onnx(model, inp))[: expected.shape[0], : expected.shape[1]]
        if not np.array_equal(pred, expected):
            raise AssertionError(f"ONNX failed {split}[{idx}]\nexpected:\n{expected}\ngot:\n{pred}")


def main() -> None:
    examples = load_examples()
    validate_reference(examples)
    model = build_model()
    validate_model(model, examples)

    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
        onnx.save(model, tmp.name)
        score_file(Path(tmp.name))

    onnx.save(model, BEST_PATH)
    print(f"saved {BEST_PATH}")
    score_file(BEST_PATH)


if __name__ == "__main__":
    main()
