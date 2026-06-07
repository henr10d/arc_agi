"""Build a compact ONNX model for NeuroGolf task073.

Task rule: the active grid is 5x5. Copy the input unchanged except that
blue cells (color 1) in row index 2 are moved to row index 4 in the same
columns. Those source cells become black/background (color 0), and bottom-row
cells are overwritten by blue only in the moved columns.

ONNX approach: slice the five row-2 blue values as a compact mask, build only
the 20 scalar channel updates that can change, and ScatterND them into the
full [1,10,30,30] input. The ScatterND output is the graph output, so the full
output tensor is not charged as internal memory.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402


TASK_ID = "task073"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
IR_VERSION = 10
SHAPE = [1, 10, 30, 30]


@dataclass(frozen=True)
class Candidate:
    name: str
    build: Callable[[], onnx.ModelProto]


class Builder:
    def __init__(self, opset: int) -> None:
        self.opset = opset
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self.value_infos: list[onnx.ValueInfoProto] = []

    def init(self, name: str, arr: np.ndarray) -> str:
        self.inits.append(numpy_helper.from_array(arr, name=name))
        return name

    def vi(self, name: str, dtype: int, shape: list[int]) -> str:
        self.value_infos.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    def node(self, op: str, inputs: list[str], output: str, dtype: int, shape: list[int], **attrs: object) -> str:
        self.vi(output, dtype, shape)
        self.nodes.append(helper.make_node(op, inputs, [output], **attrs))
        return output

    def model(self, name: str) -> onnx.ModelProto:
        graph = helper.make_graph(
            self.nodes,
            name,
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)],
            self.inits,
            value_info=self.value_infos,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", self.opset)])
        model.ir_version = IR_VERSION
        return model


def _replacement_rows() -> np.ndarray:
    rows = np.zeros((1, 10, 2, 1), dtype=np.float32)
    rows[0, 0, 0, 0] = 1.0
    rows[0, 1, 1, 0] = 1.0
    return rows


def _scatter_indices() -> np.ndarray:
    idx = np.empty((1, 10, 2, 5), dtype=np.int64)
    idx[:, :, 0, :] = 2
    idx[:, :, 1, :] = 4
    return idx


def _scatternd_indices() -> np.ndarray:
    idx = np.empty((1, 1, 1, 20, 4), dtype=np.int64)
    k = 0
    for channel, row in ((0, 2), (1, 2), (1, 4), (5, 4)):
        for col in range(5):
            idx[0, 0, 0, k] = [0, channel, row, col]
            k += 1
    return idx


def build_scatternd_sparse_model() -> onnx.ModelProto:
    b = Builder(opset=11)
    b.init("mask_starts", np.array([1, 2, 0], dtype=np.int64))
    b.init("mask_ends", np.array([2, 3, 5], dtype=np.int64))
    b.init("axes3", np.array([1, 2, 3], dtype=np.int64))
    b.init("ones5", np.ones((1, 1, 1, 5), dtype=np.float32))
    b.init("zeros5", np.zeros((1, 1, 1, 5), dtype=np.float32))
    b.init("scatternd_idx", _scatternd_indices())

    mask_f = b.node(
        "Slice",
        ["input", "mask_starts", "mask_ends", "axes3"],
        "mask_f",
        TensorProto.FLOAT,
        [1, 1, 1, 5],
    )
    inv_mask = b.node(
        "Sub",
        ["ones5", mask_f],
        "inv_mask",
        TensorProto.FLOAT,
        [1, 1, 1, 5],
    )
    updates = b.node(
        "Concat",
        ["ones5", "zeros5", mask_f, inv_mask],
        "updates",
        TensorProto.FLOAT,
        [1, 1, 1, 20],
        axis=3,
    )
    b.nodes.append(helper.make_node("ScatterND", ["input", "scatternd_idx", updates], ["output"]))
    return b.model("task073_scatternd_sparse")


def build_scatter_step_model() -> onnx.ModelProto:
    b = Builder(opset=11)
    b.init("rows24_starts", np.array([0, 0, 2, 0], dtype=np.int64))
    b.init("rows24_ends", np.array([1, 10, 5, 5], dtype=np.int64))
    b.init("axes4", np.array([0, 1, 2, 3], dtype=np.int64))
    b.init("rows24_steps", np.array([1, 1, 2, 1], dtype=np.int64))
    b.init("mask_starts", np.array([0, 1, 0, 0], dtype=np.int64))
    b.init("mask_ends", np.array([1, 2, 1, 5], dtype=np.int64))
    b.init("replacement", _replacement_rows())
    b.init("scatter_idx", _scatter_indices())

    rows24 = b.node(
        "Slice",
        ["input", "rows24_starts", "rows24_ends", "axes4", "rows24_steps"],
        "rows24",
        TensorProto.FLOAT,
        [1, 10, 2, 5],
    )
    mask_f = b.node(
        "Slice",
        [rows24, "mask_starts", "mask_ends", "axes4"],
        "mask_f",
        TensorProto.FLOAT,
        [1, 1, 1, 5],
    )
    mask = b.node("Cast", [mask_f], "mask", TensorProto.BOOL, [1, 1, 1, 5], to=TensorProto.BOOL)
    updates = b.node(
        "Where",
        [mask, "replacement", rows24],
        "updates",
        TensorProto.FLOAT,
        [1, 10, 2, 5],
    )
    b.nodes.append(helper.make_node("ScatterElements", ["input", "scatter_idx", updates], ["output"], axis=2))
    return b.model("task073_scatter_step")


def build_concat_pad_model() -> onnx.ModelProto:
    """Baseline: build the solved 5x5 crop, then Pad it to the output."""
    b = Builder(opset=10)
    b.init("axes4", np.array([0, 1, 2, 3], dtype=np.int64))
    b.init("top_starts", np.array([0, 0, 0, 0], dtype=np.int64))
    b.init("top_ends", np.array([1, 10, 2, 5], dtype=np.int64))
    b.init("row2_starts", np.array([0, 0, 2, 0], dtype=np.int64))
    b.init("row2_ends", np.array([1, 10, 3, 5], dtype=np.int64))
    b.init("row3_starts", np.array([0, 0, 3, 0], dtype=np.int64))
    b.init("row3_ends", np.array([1, 10, 4, 5], dtype=np.int64))
    b.init("row4_starts", np.array([0, 0, 4, 0], dtype=np.int64))
    b.init("row4_ends", np.array([1, 10, 5, 5], dtype=np.int64))
    b.init("mask_starts", np.array([0, 1, 0, 0], dtype=np.int64))
    b.init("mask_ends", np.array([1, 2, 1, 5], dtype=np.int64))
    b.init("black", _replacement_rows()[:, :, :1, :])
    b.init("blue", _replacement_rows()[:, :, 1:, :])

    top = b.node("Slice", ["input", "top_starts", "top_ends", "axes4"], "top", TensorProto.FLOAT, [1, 10, 2, 5])
    row2 = b.node("Slice", ["input", "row2_starts", "row2_ends", "axes4"], "row2", TensorProto.FLOAT, [1, 10, 1, 5])
    row3 = b.node("Slice", ["input", "row3_starts", "row3_ends", "axes4"], "row3", TensorProto.FLOAT, [1, 10, 1, 5])
    row4 = b.node("Slice", ["input", "row4_starts", "row4_ends", "axes4"], "row4", TensorProto.FLOAT, [1, 10, 1, 5])
    mask_f = b.node("Slice", [row2, "mask_starts", "mask_ends", "axes4"], "mask_f", TensorProto.FLOAT, [1, 1, 1, 5])
    mask = b.node("Cast", [mask_f], "mask", TensorProto.BOOL, [1, 1, 1, 5], to=TensorProto.BOOL)
    row2_out = b.node("Where", [mask, "black", row2], "row2_out", TensorProto.FLOAT, [1, 10, 1, 5])
    row4_out = b.node("Where", [mask, "blue", row4], "row4_out", TensorProto.FLOAT, [1, 10, 1, 5])
    crop = b.node("Concat", [top, row2_out, row3, row4_out], "crop", TensorProto.FLOAT, [1, 10, 5, 5], axis=2)
    b.nodes.append(helper.make_node("Pad", [crop], ["output"], mode="constant", pads=[0, 0, 0, 0, 0, 0, 25, 25], value=0.0))
    return b.model("task073_concat_pad")


def load_examples() -> list[dict[str, list[list[int]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [ex for split in ("train", "test", "arc-gen") for ex in data.get(split, [])]


def validate_model(path: Path) -> bool:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for ex in load_examples():
        inp = convert_to_numpy(ex, "input")
        expected = convert_to_numpy(ex, "output")
        if inp is None or expected is None:
            continue
        actual = session.run(["output"], {"input": inp})[0]
        if not np.array_equal(actual > 0.0, expected > 0.0):
            return False
    return True


def evaluate(candidate: Candidate) -> tuple[dict[str, object], onnx.ModelProto]:
    model = candidate.build()
    onnx.checker.check_model(model, full_check=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / f"{TASK_ID}.onnx"
        onnx.save(model, path)
        valid_outputs = validate_temp(path)
        stats = score_file(path)
    stats["candidate"] = candidate.name
    stats["valid_outputs"] = valid_outputs
    return stats, model


def validate_temp(path: Path) -> bool:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for ex in load_examples():
        inp = convert_to_numpy(ex, "input")
        expected = convert_to_numpy(ex, "output")
        if inp is None or expected is None:
            continue
        actual = session.run(["output"], {"input": inp})[0]
        if not np.array_equal(actual > 0.0, expected > 0.0):
            return False
    return True


def main() -> None:
    candidates = [
        Candidate("scatternd_sparse", build_scatternd_sparse_model),
        Candidate("scatter_step", build_scatter_step_model),
        Candidate("concat_pad", build_concat_pad_model),
    ]
    results: list[tuple[dict[str, object], onnx.ModelProto]] = []
    for candidate in candidates:
        stats, model = evaluate(candidate)
        results.append((stats, model))
        print(
            f"{candidate.name}: valid_outputs={stats['valid_outputs']} "
            f"valid_score={stats['valid']} memory={stats['memory']} params={stats['params']} "
            f"cost={stats['cost']} score={stats['score']}"
        )
        if stats.get("error"):
            print(f"  error={str(stats['error']).strip()}")

    valid = [
        (stats, model)
        for stats, model in results
        if stats.get("valid_outputs") and stats.get("valid") and stats.get("cost") is not None
    ]
    if not valid:
        raise SystemExit("no valid candidate")

    best_stats, best_model = min(valid, key=lambda item: int(item[0]["cost"]))
    BEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(best_model, BEST_PATH)
    assert validate_model(BEST_PATH)
    final_stats = score_file(BEST_PATH)
    print(
        f"best={best_stats['candidate']} saved={BEST_PATH} "
        f"memory={final_stats['memory']} params={final_stats['params']} "
        f"cost={final_stats['cost']} score={float(final_stats['score']):.6f}"
    )
    print(f"manual_points={max(1.0, 25.0 - math.log(max(1.0, int(final_stats['cost'])))):.6f}")


if __name__ == "__main__":
    main()
