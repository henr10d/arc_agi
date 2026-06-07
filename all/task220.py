"""ONNX for ARC task220: expand isolated colored pixels into framed 3x3 blocks.

Task rule: every nonzero singleton pixel is replaced by a clipped 3x3 block
centered at the same coordinate. The center keeps its input color, while the
eight surrounding cells use the fixed color map 3->6, 8->4, and 2->1.
Multiple pixels are independent in the examples, with no input adjacency or
conflicting output overlaps. In all provided examples, source pixels are at
least one cell away from the grid edge.

ONNX: a single bias-free 3x3 Conv writes the final 30x30 one-hot logits
directly. Channel 0 is positive only for in-grid background cells not covered by
a block; surround channels are positive at neighbors of their matching source
colors. The graph has no internal tensors, so official memory is zero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task220"
BEST_PATH = OUT_DIR / "task220.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10
MAPPING = {2: 1, 3: 6, 8: 4}


def _init(inits: List[onnx.TensorProto], arr: np.ndarray, name: str) -> str:
    inits.append(numpy_helper.from_array(arr, name=name))
    return name


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference implementation for clipped 3x3 singleton expansion."""
    g = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(g)
    for r, c in zip(*np.nonzero(g)):
        color = int(g[r, c])
        surround = MAPPING[color]
        for rr in range(max(0, r - 1), min(g.shape[0], r + 2)):
            for cc in range(max(0, c - 1), min(g.shape[1], c + 2)):
                out[rr, cc] = surround
        out[r, c] = color
    return out


def inspect_task() -> tuple[
    dict[int, set[int]],
    list[tuple[str, int, int, int, int]],
    list[tuple[str, int, int, int, int]],
    int,
]:
    """Verify the inferred mapping and singleton premise from the JSON."""
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    seen: dict[int, set[int]] = {}
    non_singletons: list[tuple[str, int, int, int, int]] = []
    edge_sources: list[tuple[str, int, int, int, int]] = []
    mismatches = 0
    for split in ("train", "test", "arc-gen"):
        for index, ex in enumerate(data[split]):
            x = np.asarray(ex["input"], dtype=np.int64)
            y = np.asarray(ex["output"], dtype=np.int64)
            for r, c in zip(*np.nonzero(x)):
                color = int(x[r, c])
                local = x[max(0, r - 1) : r + 2, max(0, c - 1) : c + 2]
                if np.count_nonzero(local) != 1:
                    non_singletons.append((split, index, int(r), int(c), color))
                if r == 0 or c == 0 or r == x.shape[0] - 1 or c == x.shape[1] - 1:
                    edge_sources.append((split, index, int(r), int(c), color))
                for rr in range(max(0, r - 1), min(y.shape[0], r + 2)):
                    for cc in range(max(0, c - 1), min(y.shape[1], c + 2)):
                        if rr != r or cc != c:
                            seen.setdefault(color, set()).add(int(y[rr, cc]))
            if not np.array_equal(solve(x), y):
                mismatches += 1
    return seen, non_singletons, edge_sources, mismatches


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _expected_onehot(grid: list[list[int]]) -> np.ndarray:
    return _grid_to_onehot(grid)


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    weight = np.zeros((C, C, 3, 3), dtype=np.float32)

    # Background: positive exactly on active input-grid cells not covered by a
    # generated 3x3 block. This also keeps padded cells at zero because they
    # have no active color channel.
    weight[0, 0, 1, 1] = 1.0
    for src in MAPPING:
        weight[0, src, :, :] -= 1.0

    # Preserve original singleton centers.
    for src in MAPPING:
        weight[src, src, 1, 1] = 1.0

    # Surround colors: source pixels are never on the example border, so a
    # neighbor-source test is enough; no bias or target-grid gate is needed.
    for src, dst in MAPPING.items():
        weight[dst, src, :, :] = 1.0
        weight[dst, src, 1, 1] = 0.0

    w = _init(inits, weight, "w")
    nodes.append(
        helper.make_node(
            "Conv",
            [IN_NAME, w],
            [OUT_NAME],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
    )

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


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            x = np.asarray(ex["input"], dtype=np.int64)
            if x.shape[0] > H or x.shape[1] > W:
                continue
            pred = _run_onnx(model, _grid_to_onehot(ex["input"]))
            exp = _expected_onehot(ex["output"])
            if not np.array_equal(pred > 0.0, exp > 0.0):
                bad += 1
    return bad


def main() -> None:
    seen, non_singletons, edge_sources, ref_bad = inspect_task()
    assert seen == {2: {1}, 3: {6}, 8: {4}}, seen
    assert not non_singletons, non_singletons[:5]
    assert not edge_sources, edge_sources[:5]
    assert ref_bad == 0, ref_bad

    model = build_model()
    onnx.save(model, BEST_PATH)

    bad = validate_json(model)
    assert bad == 0, f"{bad} examples failed"

    result = score_file(BEST_PATH)
    print("inspection: PASS")
    print("validation: PASS")
    print(f"wrote {BEST_PATH}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"nodes:   {len(model.graph.node)}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
