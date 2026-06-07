"""Compact ONNX for ARC task180: overlay four 4x4 quadrants by priority.

Task rule: the 8x8 input is split into four 4x4 blocks:
A=top-left, B=top-right, C=bottom-left, D=bottom-right. For each local cell,
black means absence and the output takes the first non-black cell in priority
order B, then C, then D, then A. The selected color is preserved in a 4x4
output; the remaining NeuroGolf tensor area is padded with all-zero cells.

ONNX approach: the data fixes quadrant colors as A=4, B=5, C=6, D=9. Slice
only those four compact color planes, concatenate them, and use a 1x1 Conv
as a linear priority encoder: channel 6 is C-B, channel 9 is D-B-C, channel
4 is A-B-C-D, and background is 0.5-A-B-C-D. Thresholding at >0 gives the
required one-hot 4x4 output, which is then padded to 30x30.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task180"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, 10, 30, 30]
H = W = 30
N = 4
IR_VERSION = 10
OPSET = 10

QUADS = {
    "A": (0, 0),
    "B": (0, 4),
    "C": (4, 0),
    "D": (4, 4),
}
PRIORITY = ("B", "C", "D", "A")


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []

    def i64(self, name: str, values: list[int]) -> str:
        self.inits.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name))
        return name

    def tensor(self, name: str, values: np.ndarray) -> str:
        self.inits.append(numpy_helper.from_array(values, name))
        return name

    def node(self, op_type: str, inputs: list[str], outputs: list[str], **attrs: Any) -> None:
        self.nodes.append(helper.make_node(op_type, inputs, outputs, **attrs))


def load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def solve(grid: np.ndarray | list[list[int]], order: tuple[str, ...] = PRIORITY) -> np.ndarray:
    """Reference solver for a quadrant priority overlay."""
    g = np.asarray(grid, dtype=np.int64)
    out = np.zeros((N, N), dtype=np.int64)
    for r in range(N):
        for c in range(N):
            vals = {q: int(g[r + dr, c + dc]) for q, (dr, dc) in QUADS.items()}
            for q in order:
                if vals[q] != 0:
                    out[r, c] = vals[q]
                    break
    return out


def evaluate_priority_orders() -> list[tuple[int, int, str]]:
    """Return all 24 quadrant priorities scored by examples and matching cells."""
    examples = []
    for split in ("train", "test", "arc-gen"):
        for example in load_task_data().get(split, []):
            examples.append((np.asarray(example["input"]), np.asarray(example["output"])))

    results: list[tuple[int, int, str]] = []
    for order in itertools.permutations("ABCD"):
        passed = 0
        cells = 0
        for inp, expected in examples:
            pred = solve(inp, order)
            passed += int(np.array_equal(pred, expected))
            cells += int((pred == expected).sum())
        results.append((passed, cells, "".join(order)))
    return sorted(results, reverse=True)


def build_model() -> onnx.ModelProto:
    b = Builder()
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = b.i64("axes", [0, 1, 2, 3])
    color_by_quad = {"A": 4, "B": 5, "C": 6, "D": 9}
    for name, (row, col) in QUADS.items():
        color = color_by_quad[name]
        st = b.i64(f"{name.lower()}_st", [0, color, row, col])
        en = b.i64(f"{name.lower()}_en", [1, color + 1, row + N, col + N])
        b.node("Slice", [IN_NAME, st, en, axes], [f"{name}_f"])

    # Stack order is B, C, D, A. Positive logits after thresholding encode
    # the B>C>D>A overlay without materializing boolean priority masks.
    b.node("Concat", ["B_f", "C_f", "D_f", "A_f"], ["quad_stack"], axis=1)
    weights = np.zeros((10, 4, 1, 1), dtype=np.float32)
    bias = np.zeros((10,), dtype=np.float32)
    weights[0, :, 0, 0] = -1.0
    bias[0] = 0.5
    weights[4, :, 0, 0] = [-1.0, -1.0, -1.0, 1.0]
    weights[5, 0, 0, 0] = 1.0
    weights[6, :, 0, 0] = [-1.0, 1.0, 0.0, 0.0]
    weights[9, :, 0, 0] = [-1.0, -1.0, 1.0, 0.0]
    b.tensor("priority_w", weights)
    b.tensor("priority_b", bias)
    b.node("Conv", ["quad_stack", "priority_w", "priority_b"], ["out4"])
    b.node(
        "Pad",
        ["out4"],
        [OUT_NAME],
        mode="constant",
        pads=[0, 0, 0, 0, 0, 0, H - N, W - N],
        value=0.0,
    )

    graph = helper.make_graph(b.nodes, TASK_ID, [x_info], [y_info], initializer=b.inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def verify_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    counts: dict[str, tuple[int, int]] = {}
    all_ok = True
    for split in ("train", "test", "arc-gen"):
        passed = 0
        checked = 0
        for idx, example in enumerate(load_task_data().get(split, [])):
            inp = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if inp is None or expected is None:
                continue
            pred = session.run([OUT_NAME], {IN_NAME: inp})[0]
            checked += 1
            if np.array_equal(pred > 0.0, expected > 0.0):
                passed += 1
            else:
                all_ok = False
                h = len(example["output"])
                w = len(example["output"][0])
                decoded = pred[0, :, :h, :w].argmax(axis=0)
                mismatches = np.argwhere(decoded != np.asarray(example["output"]))
                print(f"mismatch in {split}[{idx}]")
                print("best reference:")
                print(solve(example["input"]))
                print("pred grid:")
                print(decoded)
                print("mismatching cells:")
                for r, c in mismatches[:16]:
                    print(f"  ({int(r)}, {int(c)}): got {decoded[r, c]}, expected {example['output'][r][c]}")
                break
        counts[split] = (passed, checked)
    return all_ok, counts


def main() -> None:
    priority_results = evaluate_priority_orders()
    print("best priority orders:")
    for passed, cells, order in priority_results[:5]:
        print(f"  {order}: examples={passed} cells={cells}")
    if priority_results[0][2] != "".join(PRIORITY):
        raise SystemExit(f"expected {''.join(PRIORITY)} to be best, got {priority_results[0]}")

    model = build_model()
    ok, counts = verify_correct(model)
    if not ok:
        raise SystemExit(f"{TASK_ID} failed verification: {counts}")

    onnx.save(model, BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"verified: {counts}")
    print(
        "score: "
        f"valid={result['valid']} memory={result['memory']} "
        f"params={result['params']} cost={result['cost']} score={result['score']}"
    )
    if not result["valid"]:
        raise SystemExit(result["error"])


if __name__ == "__main__":
    main()
