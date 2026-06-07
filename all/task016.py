"""Minimal ONNX for ARC task016: fixed 3x3 global color remapping.

Task rule: every input is a 3x3 grid whose positions and geometry are
unchanged. Each cell color is replaced by a globally learned color map from the
training pairs: 1->5, 2->6, 3->4, 4->3, 5->1, 6->2, 8->9, and 9->8. Color 0
stays 0 and the unobserved color 7 is left unchanged.

ONNX: apply the remap as a channel Gather on the one-hot NeuroGolf tensor. The
single Gather node writes directly to the required [1,10,30,30] output, so
there are no scored activation tensors; only the 10-element int32 channel index
initializer counts as params. Padding cells are all-zero in the input and
therefore remain all-zero in the output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task016"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def learn_mapping() -> Dict[int, int]:
    """Infer and validate the global color remap from train examples."""
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    mapping: Dict[int, int] = {0: 0}
    for ex in data["train"]:
        inp = np.asarray(ex["input"], dtype=np.int64)
        out = np.asarray(ex["output"], dtype=np.int64)
        if inp.shape != (3, 3) or out.shape != (3, 3):
            raise ValueError(f"{TASK_ID} expected 3x3 train grids, got {inp.shape} -> {out.shape}")
        for src, dst in zip(inp.ravel(), out.ravel()):
            src_i, dst_i = int(src), int(dst)
            old = mapping.get(src_i)
            if old is not None and old != dst_i:
                raise ValueError(f"inconsistent color mapping for {src_i}: {old} vs {dst_i}")
            mapping[src_i] = dst_i

    for color in range(C):
        mapping.setdefault(color, color if color == 7 else 0)
    return mapping


def solve(grid: np.ndarray, mapping: Dict[int, int] | None = None) -> np.ndarray:
    """Apply the learned color remap without moving any cells."""
    lut = learn_mapping() if mapping is None else mapping
    return np.vectorize(lut.__getitem__, otypes=[np.int64])(np.asarray(grid, dtype=np.int64))


def _grid_to_onehot(grid: List[List[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def build_model(mapping: Dict[int, int] | None = None) -> onnx.ModelProto:
    lut = learn_mapping() if mapping is None else mapping

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    channel_order = np.zeros(C, dtype=np.int32)
    for src, dst in lut.items():
        channel_order[int(dst)] = int(src)

    graph = helper.make_graph(
        [helper.make_node("Gather", [IN_NAME, "channel_order"], [OUT_NAME], axis=1)],
        TASK_ID,
        [x_info],
        [y_info],
        initializer=[numpy_helper.from_array(channel_order, name="channel_order")],
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


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto, mapping: Dict[int, int]) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    bad = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data.get(split, []):
            oh = _grid_to_onehot(ex["input"])
            pred = _onehot_to_grid(_run_onnx(model, oh))[:3, :3]
            exp = np.asarray(ex["output"], dtype=np.int64)
            ref = solve(np.asarray(ex["input"], dtype=np.int64), mapping)
            if not np.array_equal(ref, exp) or not np.array_equal(pred, exp):
                bad += 1
    return bad


def main() -> None:
    mapping = learn_mapping()
    model = build_model(mapping)
    onnx.save(model, BEST_PATH)

    bad = validate_json(model, mapping)
    print(f"{TASK_ID}.json: {'PASS' if bad == 0 else f'FAIL ({bad} wrong)'}")

    result = score_file(BEST_PATH)
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
