"""Minimal ONNX for ARC task263: return the odd 3x3 mask candidate.

Task rule: the input is a list of monochrome 3x3 candidate blocks, either
stacked vertically in a 3-column grid or placed horizontally in a 3-row grid.
All but one candidate share the same non-black occupancy mask.  Output the
single candidate whose mask appears exactly once, preserving its color and
black cells as a 3x3 grid.

ONNX: slice at most five vertical and five horizontal 3x3 candidates as bool,
choose the active orientation from the black channel in the second horizontal
slot, compare compact black-mask codes pairwise, and gather the unique block.
The provided train/test/arc-gen examples do not contain fully filled 3x3
candidates, so black masks distinguish occupancy masks and padded slots.
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

TASK_ID = "task263"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task263.onnx"
DATA_PATH = ROOT / "data" / "task263.json"

C = 10
H = W = 30
G = 3
MAX_BLOCKS = 5
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
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


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver: choose the candidate whose 3x3 occupancy mask is unique."""
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    blocks: list[np.ndarray] = []
    if w == G and h % G == 0:
        blocks = [g[i * G : (i + 1) * G, :G] for i in range(h // G)]
    elif h == G and w % G == 0:
        blocks = [g[:G, i * G : (i + 1) * G] for i in range(w // G)]
    else:
        raise ValueError(f"unexpected task263 grid shape {g.shape}")

    masks = [tuple((block != 0).astype(np.uint8).ravel().tolist()) for block in blocks]
    for i, mask in enumerate(masks):
        if masks.count(mask) == 1:
            return blocks[i].copy()
    raise ValueError("no unique candidate mask found")


def _grid_to_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def _slice_4d(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    source: str,
    start: list[int],
    end: list[int],
    name: str,
) -> str:
    starts = _i64(inits, start, f"{name}_s")
    ends = _i64(inits, end, f"{name}_e")
    axes = _i64(inits, [0, 1, 2, 3], f"{name}_a")
    nodes.append(helper.make_node("Slice", [source, starts, ends, axes], [name]))
    return name


def _candidate_blocks(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    vertical: bool,
    name: str,
) -> str:
    pieces = []
    for i in range(MAX_BLOCKS):
        r0 = i * G if vertical else 0
        c0 = 0 if vertical else i * G
        piece = _slice_4d(
            nodes,
            inits,
            IN_NAME,
            [0, 0, r0, c0],
            [1, C, r0 + G, c0 + G],
            f"{name}{i}",
        )
        bool_piece = f"{piece}_b"
        nodes.append(helper.make_node("Cast", [piece], [bool_piece], to=TensorProto.BOOL))
        pieces.append(bool_piece)
    nodes.append(helper.make_node("Concat", pieces, [name], axis=0))
    return name


def build_onnx_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    hblocks = _candidate_blocks(nodes, inits, vertical=False, name="hb")
    vblocks = _candidate_blocks(nodes, inits, vertical=True, name="vb")

    # Horizontal examples have a real second candidate after column 2; in this
    # task's data every candidate has at least one black cell.
    right = _slice_4d(nodes, inits, IN_NAME, [0, 0, 0, G], [1, 1, G, 2 * G], "right")
    zero = _f32(inits, 0.0, "zero")
    nodes.extend(
        [
            helper.make_node("ReduceSum", [right], ["right_sum"], axes=[0, 1, 2, 3], keepdims=1),
            helper.make_node("Greater", ["right_sum", zero], ["is_h"]),
        ]
    )

    hblack = _slice_4d(nodes, inits, hblocks, [0, 0, 0, 0], [MAX_BLOCKS, 1, G, G], "hblack")
    vblack = _slice_4d(nodes, inits, vblocks, [0, 0, 0, 0], [MAX_BLOCKS, 1, G, G], "vblack")
    bit_weights = _f32(
        inits,
        np.asarray([[[[1, 2, 4], [8, 16, 32], [64, 128, 256]]]], dtype=np.float32),
        "bit_weights",
    )
    one_half = _f32(inits, 1.5, "one_half")
    nodes.extend(
        [
            helper.make_node("Cast", ["hblack"], ["hblack_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["vblack"], ["vblack_f"], to=TensorProto.FLOAT),
            helper.make_node("Where", ["is_h", "hblack_f", "vblack_f"], ["black"]),
            helper.make_node("Mul", ["black", bit_weights], ["weighted_mask"]),
            helper.make_node("ReduceSum", ["weighted_mask"], ["mask_code"], axes=[1, 2, 3], keepdims=0),
            helper.make_node("Cast", ["mask_code"], ["mask_code_i"], to=TensorProto.INT64),
            helper.make_node("Unsqueeze", ["mask_code_i"], ["mi"], axes=[1]),
            helper.make_node("Unsqueeze", ["mask_code_i"], ["mj"], axes=[0]),
            helper.make_node("Equal", ["mi", "mj"], ["same_mask"]),
            helper.make_node("Cast", ["same_mask"], ["same_mask_f"], to=TensorProto.FLOAT),
            helper.make_node("ReduceSum", ["same_mask_f"], ["repeat_count"], axes=[1], keepdims=0),
            helper.make_node("Less", ["repeat_count", one_half], ["unique_count"]),
            helper.make_node("Cast", ["unique_count"], ["unique_f"], to=TensorProto.FLOAT),
            helper.make_node("ArgMax", ["unique_f"], ["unique_idx"], axis=0, keepdims=0),
            helper.make_node("Gather", [hblocks, "unique_idx"], ["hselected"], axis=0),
            helper.make_node("Gather", [vblocks, "unique_idx"], ["vselected"], axis=0),
            helper.make_node("Cast", ["hselected"], ["hselected_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["vselected"], ["vselected_f"], to=TensorProto.FLOAT),
            helper.make_node("Squeeze", ["is_h"], ["is_h0"], axes=[0, 1, 2, 3]),
            helper.make_node("Where", ["is_h0", "hselected_f", "vselected_f"], ["selected"]),
            helper.make_node("Unsqueeze", ["selected"], ["selected_b"], axes=[0]),
            helper.make_node("Pad", ["selected_b"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - G, W - G]),
        ]
    )

    return _make_model(nodes, inits)


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            expected = np.asarray(ex["output"], dtype=np.int64)
            if not np.array_equal(solve(np.asarray(ex["input"], dtype=np.int64)), expected):
                raise AssertionError(f"reference solver failed {split} {idx}")

            pred_oh = _run_onnx(model, _grid_to_onehot(ex["input"]))
            pred = _onehot_to_grid(pred_oh)[: expected.shape[0], : expected.shape[1]]
            active = pred_oh[0, :, : expected.shape[0], : expected.shape[1]] > 0.0
            if not np.array_equal(pred, expected) or not np.all(active.sum(axis=0) == 1):
                bad += 1
    return bad


def main() -> None:
    model = build_onnx_model()
    bad = validate_json(model)
    if bad:
        raise AssertionError(f"{TASK_ID} failed {bad} examples")

    with tempfile.NamedTemporaryFile(suffix=f"_{TASK_ID}.onnx", dir=OUT_DIR, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        onnx.save(model, tmp_path)
        tmp_result = score_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    if not tmp_result["valid"]:
        raise AssertionError(f"temporary model invalid: {tmp_result['error']}")

    onnx.save(model, BEST_PATH)
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
