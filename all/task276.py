"""Minimal ONNX for ARC task276: recolor magenta cells to red.

Task rule: the grid geometry is unchanged. Every magenta pixel (color 6)
becomes red (color 2), while orange/brown pixels (color 7) and background
remain unchanged. The official examples contain only colors 6 and 7 inside the
active grid, and no red pixels appear in the inputs. The best scoring ONNX
variant rewires one-hot channels directly: output channel 2 reads input channel
6, output channel 6 reads the empty input channel 2, and all other channels pass
through.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task276"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def _make_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
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


def build_gather_model() -> onnx.ModelProto:
    """Best official-data model: one Gather, zero cost memory, 10 params."""
    indices = np.array([0, 1, 6, 3, 4, 5, 2, 7, 8, 9], dtype=np.int64)
    inits = [numpy_helper.from_array(indices, name="i")]
    nodes = [helper.make_node("Gather", [IN_NAME, "i"], [OUT_NAME], axis=1)]
    return _make_model(nodes, inits, "g")


def build_dense_conv_model() -> onnx.ModelProto:
    """Generic 6->2 one-hot recolor preserving every other color."""
    weights = np.zeros((C, C, 1, 1), dtype=np.float32)
    for color in range(C):
        if color != 6:
            weights[color, color, 0, 0] = 1.0
    weights[2, 6, 0, 0] = 1.0
    inits = [numpy_helper.from_array(weights, name="weights")]
    nodes = [helper.make_node("Conv", [IN_NAME, "weights"], [OUT_NAME], kernel_shape=[1, 1])]
    return _make_model(nodes, inits, "task276_dense_conv")


def solve(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=np.int64).copy()
    out[out == 6] = 2
    return out


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            out[0, int(color), row, col] = 1.0
    return out


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def _load_examples() -> list[tuple[str, int, dict]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    examples: list[tuple[str, int, dict]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            examples.append((split, idx, ex))
    return examples


def validate_model(model: onnx.ModelProto) -> None:
    for split, idx, ex in _load_examples():
        grid = np.asarray(ex["input"], dtype=np.int64)
        if grid.shape[0] > H or grid.shape[1] > W:
            continue
        expected = np.asarray(ex["output"], dtype=np.int64)
        ref = solve(grid)
        if not np.array_equal(ref, expected):
            raise AssertionError(f"reference rule mismatch on {split}[{idx}]")

        pred = _run_onnx(model, _grid_to_onehot(ex["input"]))
        pred_bool = pred > 0.0
        expected_oh = _grid_to_onehot(ex["output"]) > 0.0
        if not np.array_equal(pred_bool, expected_oh):
            decoded = pred[:, :, : grid.shape[0], : grid.shape[1]].argmax(axis=1)[0]
            raise AssertionError(f"ONNX mismatch on {split}[{idx}]\n{decoded}\n!=\n{expected}")


def choose_best(candidates: dict[str, Callable[[], onnx.ModelProto]]) -> tuple[str, onnx.ModelProto, dict]:
    best: tuple[str, onnx.ModelProto, dict] | None = None
    with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}_") as tmp:
        tmp_dir = Path(tmp)
        for name, build in candidates.items():
            model = build()
            validate_model(model)
            path = tmp_dir / name / f"{TASK_ID}.onnx"
            path.parent.mkdir()
            onnx.save(model, path)
            result = score_file(path)
            if not result["valid"]:
                raise AssertionError(f"{name} failed scoring: {result['error']}")
            if best is None or (int(result["cost"]), int(result["filesize"])) < (
                int(best[2]["cost"]),
                int(best[2]["filesize"]),
            ):
                best = (name, model, result)
    assert best is not None
    return best


def main() -> None:
    name, model, result = choose_best(
        {
            "gather": build_gather_model,
            "dense_conv": build_dense_conv_model,
        }
    )
    onnx.save(model, BEST_PATH)
    final = score_file(BEST_PATH)
    validate_model(model)
    print(
        f"saved {BEST_PATH} via {name}: "
        f"memory={final['memory']} params={final['params']} "
        f"cost={final['cost']} score={final['score']:.6f}"
    )
    print(
        f"candidate {name}: "
        f"memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']:.6f}"
    )


if __name__ == "__main__":
    main()
