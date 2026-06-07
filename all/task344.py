"""Build an ONNX solution for NeuroGolf task344.

Task rule: in the active grid, every 4-neighbor red/green adjacent pair is
collapsed onto the green cell. A green cell adjacent to at least one red cell
becomes teal (color 8), and a red cell adjacent to at least one green cell is
removed to black (color 0). Black and gray cells are copied unchanged. The task
data uses only colors 0, 2, 3, and 5 in inputs; padding outside each grid stays
all-zero.

ONNX approach: use a single 3x3 Conv over the full one-hot input and write it
directly to the graph output, so no internal activations are scored. The Conv
uses linear thresholds that are positive exactly for copied black/gray cells,
surviving red/green cells, removed red cells becoming black, and green cells
becoming teal; padding cells have all-zero input and stay inactive.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task344"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10

BLACK = 0
RED = 2
GREEN = 3
GRAY = 5
TEAL = 8


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []

    def init(self, name: str, values: np.ndarray) -> str:
        self.inits.append(numpy_helper.from_array(values, name))
        return name


def solve(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    """Reference solver for the adjacent red/green pair replacement rule."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    h, w = g.shape
    make_teal = np.zeros((h, w), dtype=bool)
    remove_red = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if g[r, c] not in {RED, GREEN}:
                continue
            has_red = False
            has_green = False
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    has_red |= g[rr, cc] == RED
                    has_green |= g[rr, cc] == GREEN
            make_teal[r, c] = g[r, c] == GREEN and has_red
            remove_red[r, c] = g[r, c] == RED and has_green
    out[remove_red] = BLACK
    out[make_teal] = TEAL
    return out


def _linear_rule_kernel() -> tuple[np.ndarray, np.ndarray]:
    kernel = np.zeros((C, C, 3, 3), dtype=np.float32)
    bias = np.zeros((C,), dtype=np.float32)
    cardinals = ((0, 1), (1, 0), (1, 2), (2, 1))

    # Black is copied, and red cells with any green neighbor become black.
    kernel[BLACK, BLACK, 1, 1] = 5.0
    kernel[BLACK, RED, 1, 1] = 4.0
    for r, c in cardinals:
        kernel[BLACK, GREEN, r, c] = 1.0
    bias[BLACK] = -4.5

    # Red/green survive only when they have no opposite-color cardinal neighbor.
    kernel[RED, RED, 1, 1] = 1.0
    for r, c in cardinals:
        kernel[RED, GREEN, r, c] = -1.0
    bias[RED] = 0.0

    kernel[GREEN, GREEN, 1, 1] = 1.0
    for r, c in cardinals:
        kernel[GREEN, RED, r, c] = -1.0
    bias[GREEN] = 0.0

    # Gray is copied unchanged.
    kernel[GRAY, GRAY, 1, 1] = 1.0
    bias[GRAY] = 0.0

    # Green cells with any red cardinal neighbor become teal.
    kernel[TEAL, GREEN, 1, 1] = 4.0
    for r, c in cardinals:
        kernel[TEAL, RED, r, c] = 1.0
    bias[TEAL] = -4.5
    return kernel, bias


def build_model() -> onnx.ModelProto:
    b = Builder()
    kernel, bias = _linear_rule_kernel()
    b.init("Wrule", kernel)
    b.init("Brule", bias)

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)

    b.nodes.append(helper.make_node("Conv", ["input", "Wrule", "Brule"], ["output"], pads=[1, 1, 1, 1]))

    graph = helper.make_graph(
        b.nodes,
        TASK_ID,
        [inp],
        [out],
        initializer=b.inits,
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


def _validate_examples(path: Path) -> tuple[int, int]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(data[split]):
            x = convert_to_numpy(example, "input")
            y = convert_to_numpy(example, "output")
            if x is None or y is None:
                continue
            expected_grid = np.asarray(example["output"], dtype=np.int64)
            if not np.array_equal(solve(example["input"]), expected_grid):
                raise AssertionError(f"reference solver failed {split} example {index}")
            pred = session.run(["output"], {"input": x})[0]
            total += 1
            if not np.array_equal(pred > 0.0, y > 0.0):
                raise AssertionError(f"ONNX mismatch on {split} example {index}")
            passed += 1
    return passed, total


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    passed, total = _validate_examples(BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"correct: {passed}/{total}")
    print(
        "score: "
        f"memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']:.6f}"
    )


if __name__ == "__main__":
    main()
