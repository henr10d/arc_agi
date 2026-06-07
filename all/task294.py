"""ONNX for ARC task294: recolor interiors of gray rectangles red.

Task rule: the input is a 10x10 grid containing one or more black-separated,
axis-aligned filled gray rectangles.  Preserve each rectangle's gray outer
border and recolor only strict interior cells to red; rectangles shorter than
3 cells or narrower than 3 cells have no interior and remain gray.

ONNX approach: on the 10x10 gray channel, a cell is strict rectangle interior
exactly when it is gray and its four cardinal neighbors are also gray.  Build
that boolean mask in the compact 10x10 area, switch gray to red there, then
pad back to the required 30x30 competition tensor.
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

TASK_ID = "task294"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task294.onnx"
DATA_PATH = ROOT / "data" / "task294.json"

C = 10
H = W = 30
N = 10
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


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
    """Reference implementation for the four-neighbor interior rule."""
    g = np.asarray(grid, dtype=np.int64)
    gray = g == 5
    interior = np.zeros_like(gray)
    interior[1:-1, 1:-1] = (
        gray[1:-1, 1:-1]
        & gray[:-2, 1:-1]
        & gray[2:, 1:-1]
        & gray[1:-1, :-2]
        & gray[1:-1, 2:]
    )
    out = g.copy()
    out[interior] = 2
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
    starts: list[int],
    ends: list[int],
    axes: list[int],
    out: str,
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


def _pad(nodes: List[onnx.NodeProto], x: str, pads: list[int], out: str) -> str:
    nodes.append(helper.make_node("Pad", [x], [out], mode="constant", pads=pads, value=0.0))
    return out


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    one = _f32(inits, [1.0], "one")

    gray = _slice(nodes, inits, IN_NAME, [0, 5, 0, 0], [1, 6, N, N], [0, 1, 2, 3], "gray")

    up = _pad(nodes, _slice(nodes, inits, gray, [0], [N - 1], [2], "up_slice"), [0, 0, 1, 0, 0, 0, 0, 0], "up")
    down = _pad(nodes, _slice(nodes, inits, gray, [1], [N], [2], "down_slice"), [0, 0, 0, 0, 0, 0, 1, 0], "down")
    left = _pad(nodes, _slice(nodes, inits, gray, [0], [N - 1], [3], "left_slice"), [0, 0, 0, 1, 0, 0, 0, 0], "left")
    right = _pad(nodes, _slice(nodes, inits, gray, [1], [N], [3], "right_slice"), [0, 0, 0, 0, 0, 0, 0, 1], "right")

    nodes.append(helper.make_node("Mul", [gray, up], ["interior_a"]))
    nodes.append(helper.make_node("Mul", ["interior_a", down], ["interior_b"]))
    nodes.append(helper.make_node("Mul", ["interior_b", left], ["interior_c"]))
    nodes.append(helper.make_node("Mul", ["interior_c", right], ["interior"]))

    ch0_2 = _slice(nodes, inits, IN_NAME, [0, 0, 0, 0], [1, 2, N, N], [0, 1, 2, 3], "ch0_2")
    red_in = _slice(nodes, inits, IN_NAME, [0, 2, 0, 0], [1, 3, N, N], [0, 1, 2, 3], "red_in")
    nodes.append(helper.make_node("Add", [red_in, "interior"], ["red"]))
    ch3_5 = _slice(nodes, inits, IN_NAME, [0, 3, 0, 0], [1, 5, N, N], [0, 1, 2, 3], "ch3_5")
    nodes.append(helper.make_node("Sub", [one, "interior"], ["not_interior"]))
    nodes.append(helper.make_node("Mul", [gray, "not_interior"], ["gray_border"]))
    ch6_10 = _slice(nodes, inits, IN_NAME, [0, 6, 0, 0], [1, 10, N, N], [0, 1, 2, 3], "ch6_10")
    nodes.append(helper.make_node("Concat", [ch0_2, "red", ch3_5, "gray_border", ch6_10], ["core_out"], axis=1))
    _pad(nodes, "core_out", [0, 0, 0, 0, 0, 0, H - N, W - N], OUT_NAME)

    return _make_model(nodes, inits)


def validate_reference(examples: list[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    for split, idx, inp, expected in examples:
        got = solve(inp)
        if not np.array_equal(got, expected):
            raise AssertionError(f"reference failed {split}[{idx}]")


def validate_model(model: onnx.ModelProto, examples: list[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    for split, idx, inp, expected in examples:
        got = _onehot_to_grid(_run_onnx(model, inp))[: expected.shape[0], : expected.shape[1]]
        if not np.array_equal(got, expected):
            raise AssertionError(f"model failed {split}[{idx}]")


def main() -> None:
    examples = load_examples()
    validate_reference(examples)
    model = build_model()
    validate_model(model, examples)

    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
        onnx.save(model, tmp.name)
        metrics = score_file(Path(tmp.name))

    onnx.save(model, BEST_PATH)
    print(f"saved {BEST_PATH}")
    print(
        f"memory={metrics['memory']} params={metrics['params']} "
        f"cost={metrics['cost']} score={metrics['score']:.6f}"
    )


if __name__ == "__main__":
    main()
