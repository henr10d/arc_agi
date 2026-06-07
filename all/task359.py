"""ONNX for ARC task359: remove sparse noise from stripe/block grids.

Task rule: each input is a clean set of constant-color bands corrupted by rare
outlier pixels.  The clean image is either vertical bands, where every column
should be filled with that column's dominant color, or horizontal bands, where
every row should be filled with that row's dominant color.  The examples include
both orientations, so the model chooses the orientation whose row/column
majority counts explain more input cells.

ONNX approach: sum one-hot counts per row and per column, take the color ArgMax
for both hypotheses, compare total majority support, broadcast the selected
hypothesis back to a compact index grid, mark padded cells with out-of-range
index 10, then use OneHot as the final float output so the full one-hot tensor
is not realized as a scored internal activation.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task359"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
IR_VERSION = 10
OPSET = 10


def load_examples() -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    examples: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            examples.append(
                (
                    split,
                    idx,
                    np.asarray(example["input"], dtype=np.int64),
                    np.asarray(example["output"], dtype=np.int64),
                )
            )
    return examples


def _dominant(values: np.ndarray) -> int:
    return int(np.bincount(values.reshape(-1), minlength=C).argmax())


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver: choose row-majority or column-majority denoising."""
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape

    col_colors = np.asarray([_dominant(g[:, c]) for c in range(w)], dtype=np.int64)
    row_colors = np.asarray([_dominant(g[r, :]) for r in range(h)], dtype=np.int64)
    col_out = np.tile(col_colors.reshape(1, w), (h, 1))
    row_out = np.tile(row_colors.reshape(h, 1), (1, w))

    col_score = int(sum(np.count_nonzero(g[:, c] == col_colors[c]) for c in range(w)))
    row_score = int(sum(np.count_nonzero(g[r, :] == row_colors[r]) for r in range(h)))
    return col_out if col_score > row_score else row_out


def one_hot(grid: np.ndarray) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            out[0, int(grid[r, c]), r, c] = 1.0
    return out


def decode(onehot: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return (onehot[0, :, :h, :w] > 0.0).argmax(axis=0).astype(np.int64)


def _init(inits: list[onnx.TensorProto], name: str, value: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(value), name=name))
    return name


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    zero = _init(inits, "zero", np.asarray([0.0], dtype=np.float32))
    invalid_id = _init(inits, "invalid_id", np.asarray([C], dtype=np.int32))
    depth = _init(inits, "depth", np.asarray(C, dtype=np.int64))
    onehot_values = _init(inits, "onehot_values", np.asarray([0.0, 1.0], dtype=np.float32))

    nodes.extend(
        [
            helper.make_node("ReduceSum", [IN_NAME], ["col_counts"], axes=[2], keepdims=1),
            helper.make_node("ReduceSum", [IN_NAME], ["row_counts"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["col_counts"], ["col_max"], axes=[1], keepdims=0),
            helper.make_node("ReduceMax", ["row_counts"], ["row_max"], axes=[1], keepdims=0),
            helper.make_node("ReduceSum", ["col_max"], ["col_score"], axes=[0, 1, 2], keepdims=0),
            helper.make_node("ReduceSum", ["row_max"], ["row_score"], axes=[0, 1, 2], keepdims=0),
            helper.make_node("Greater", ["col_score", "row_score"], ["choose_col"]),
            helper.make_node("ArgMax", ["col_counts"], ["col_ids"], axis=1, keepdims=0),
            helper.make_node("ArgMax", ["row_counts"], ["row_ids"], axis=1, keepdims=0),
            helper.make_node("Cast", ["col_ids"], ["col_ids_i32"], to=TensorProto.INT32),
            helper.make_node("Cast", ["row_ids"], ["row_ids_i32"], to=TensorProto.INT32),
            helper.make_node("Where", ["choose_col", "col_ids_i32", "row_ids_i32"], ["chosen_ids"]),
            helper.make_node("Greater", ["col_max", zero], ["col_active"]),
            helper.make_node("Greater", ["row_max", zero], ["row_active"]),
            helper.make_node("And", ["row_active", "col_active"], ["valid_cell"]),
            helper.make_node("Where", ["valid_cell", "chosen_ids", invalid_id], ["valid_ids"]),
            helper.make_node("Cast", ["valid_ids"], ["valid_ids_i64"], to=TensorProto.INT64),
            helper.make_node("OneHot", ["valid_ids_i64", depth, onehot_values], [OUT_NAME], axis=1),
        ]
    )

    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def run_model(model: onnx.ModelProto, grid: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run([OUT_NAME], {IN_NAME: one_hot(grid)})[0]


def validate_reference(examples: list[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    for split, idx, inp, expected in examples:
        got = solve(inp)
        if not np.array_equal(got, expected):
            raise AssertionError(f"reference failed {split}[{idx}]")


def validate_model(model: onnx.ModelProto, examples: list[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    for split, idx, inp, expected in examples:
        got = run_model(model, inp)
        target = one_hot(expected)
        if not np.array_equal(got > 0.0, target > 0.0):
            raise AssertionError(f"model failed {split}[{idx}]")


def main() -> None:
    examples = load_examples()
    validate_reference(examples)

    model = build_model()
    validate_model(model, examples)

    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
        onnx.save(model, tmp.name)
        metrics = score_file(Path(tmp.name))
    if not metrics["valid"]:
        raise AssertionError(f"scoring failed: {metrics['error']}")

    onnx.save(model, BEST_PATH)
    print(f"saved {BEST_PATH}")
    print(
        f"memory={metrics['memory']} params={metrics['params']} "
        f"cost={metrics['cost']} score={metrics['score']:.6f}"
    )


if __name__ == "__main__":
    main()
