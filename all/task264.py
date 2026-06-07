"""ONNX for ARC task264: sort 3x3 gray-backed symbols by symbol shape.

Task rule: the input contains nine 3x3 non-background tiles scattered in the
grid.  Each tile is either all gray (5), or gray with one non-gray color drawn
in one of the eight compass-position masks around the 3x3 tile.  Ignore the
tile's position in the input; place each tile into the matching 3x3 slot of a
fixed 9x9 output according to its internal mask.  The all-gray tile occupies
the center slot, and all uncolored cells in every output tile are gray.

ONNX: one 3x3 Conv bank detects every (mask, color) tile in the compact
16x16 task area.  ReduceMax turns each detector into a scalar presence flag;
the best model then structurally concatenates small per-slot color vectors and
a shared gray cell into the 9x9 one-hot output before padding to 30x30.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task264"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
G = 9
T = 3
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

MASKS: list[tuple[tuple[int, int, int], ...]] = [
    ((1, 1, 0), (1, 0, 0), (0, 0, 0)),
    ((1, 1, 1), (0, 1, 0), (0, 0, 0)),
    ((0, 1, 1), (0, 0, 1), (0, 0, 0)),
    ((1, 0, 0), (1, 1, 0), (1, 0, 0)),
    ((0, 0, 1), (0, 1, 1), (0, 0, 1)),
    ((0, 0, 0), (1, 0, 0), (1, 1, 0)),
    ((0, 0, 0), (0, 1, 0), (1, 1, 1)),
    ((0, 0, 0), (0, 0, 1), (0, 1, 1)),
]
SLOTS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _make_model(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], name: str) -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(nodes, name, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _tile_windows(grid: np.ndarray) -> list[tuple[int, int, int, tuple[tuple[int, int, int], ...]]]:
    """Return all valid 3x3 tile windows as row, col, color, non-gray mask."""
    out = []
    h, w = grid.shape
    for r in range(h - T + 1):
        for c in range(w - T + 1):
            tile = grid[r : r + T, c : c + T]
            if np.any(tile == 0):
                continue
            colors = sorted(int(v) for v in np.unique(tile) if int(v) != 5)
            if len(colors) > 1:
                continue
            mask = tuple(tuple(int(tile[i, j] != 5) for j in range(T)) for i in range(T))
            out.append((r, c, colors[0] if colors else 5, mask))
    return out


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver for the shape-to-slot rule."""
    out = np.full((G, G), 5, dtype=np.int64)
    slot_by_mask = {mask: slot for mask, slot in zip(MASKS, SLOTS)}
    slot_by_mask[((0, 0, 0), (0, 0, 0), (0, 0, 0))] = (1, 1)

    seen: set[tuple[int, int]] = set()
    for _r, _c, color, mask in _tile_windows(np.asarray(grid, dtype=np.int64)):
        br, bc = slot_by_mask[mask]
        seen.add((br, bc))
        for i in range(T):
            for j in range(T):
                out[br * T + i, bc * T + j] = color if mask[i][j] else 5

    if len(seen) != 9:
        raise ValueError(f"expected nine task264 tiles, found {len(seen)}")
    return out


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def build_conv_decode_model(crop_size: int = H) -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    detector_count = len(MASKS) * len(COLORS)
    conv_w = np.zeros((detector_count, C, T, T), dtype=np.float32)
    decoder_w = np.zeros((detector_count, C * G * G), dtype=np.float32)
    row = 0
    for slot_idx, (mask, (br, bc)) in enumerate(zip(MASKS, SLOTS)):
        del slot_idx
        mask_arr = np.asarray(mask, dtype=bool)
        for color in COLORS:
            for i in range(T):
                for j in range(T):
                    conv_w[row, color if mask_arr[i, j] else 5, i, j] = 1.0
                    rr = br * T + i
                    cc = bc * T + j
                    out_color = color if mask_arr[i, j] else 5
                    decoder_w[row, (out_color * G + rr) * G + cc] = 1.0
            row += 1

    center = np.zeros((C * G * G,), dtype=np.float32)
    for rr in range(T, 2 * T):
        for cc in range(T, 2 * T):
            center[(5 * G + rr) * G + cc] = 1.0

    conv_name = _f32(inits, conv_w, "detector_w")
    threshold_name = _f32(inits, np.array(8.5, dtype=np.float32), "threshold")
    decoder_name = _f32(inits, decoder_w, "decoder_w")
    center_name = _f32(inits, center, "center_gray")
    shape_name = _i64(inits, [1, C, G, G], "out_shape")
    conv_input = IN_NAME
    if crop_size != H:
        starts_name = _i64(inits, [0, 0, 0, 0], "crop_starts")
        ends_name = _i64(inits, [1, C, crop_size, crop_size], "crop_ends")
        nodes.append(helper.make_node("Slice", [IN_NAME, starts_name, ends_name], ["crop"]))
        conv_input = "crop"

    nodes.extend(
        [
            helper.make_node("Conv", [conv_input, conv_name], ["scores"]),
            helper.make_node("ReduceMax", ["scores"], ["best_scores"], axes=[2, 3], keepdims=0),
            helper.make_node("Greater", ["best_scores", threshold_name], ["present_b"]),
            helper.make_node("Cast", ["present_b"], ["present"], to=TensorProto.FLOAT),
            helper.make_node("MatMul", ["present", decoder_name], ["decoded_flat0"]),
            helper.make_node("Add", ["decoded_flat0", center_name], ["decoded_flat"]),
            helper.make_node("Reshape", ["decoded_flat", shape_name], ["out9"]),
            helper.make_node("Pad", ["out9"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - G, W - G]),
        ]
    )
    return _make_model(nodes, inits, f"task264_conv_decode_{crop_size}")


def build_full_conv_decode_model() -> onnx.ModelProto:
    return build_conv_decode_model(H)


def build_cropped_conv_decode_model() -> onnx.ModelProto:
    return build_conv_decode_model(16)


def build_structural_decode_model(crop_size: int = 16) -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    detector_count = len(MASKS) * len(COLORS)
    conv_w = np.zeros((detector_count, C - 1, T, T), dtype=np.float32)
    row = 0
    for mask in MASKS:
        mask_arr = np.asarray(mask, dtype=bool)
        for color in COLORS:
            for i in range(T):
                for j in range(T):
                    conv_w[row, (color if mask_arr[i, j] else 5) - 1, i, j] = 1.0
            row += 1

    conv_name = _f32(inits, conv_w, "detector_w")
    threshold_name = _f32(inits, np.array(8.5, dtype=np.float32), "threshold")
    present_shape = _i64(inits, [1, len(MASKS), len(COLORS)], "present_shape")
    zero = _init(inits, np.zeros((1, 1, 1, 1), dtype=np.bool_), "zero_cell")
    gray = np.zeros((1, C, 1, 1), dtype=np.bool_)
    gray[0, 5, 0, 0] = True
    gray_cell = _init(inits, gray, "gray_cell")
    slot_shape = _i64(inits, [1, len(COLORS), 1, 1], "slot_shape")
    first4_starts = _i64(inits, [0, 0, 0, 0], "first4_starts")
    first4_ends = _i64(inits, [1, 4, 1, 1], "first4_ends")
    last4_starts = _i64(inits, [0, 4, 0, 0], "last4_starts")
    last4_ends = _i64(inits, [1, 8, 1, 1], "last4_ends")

    starts_name = _i64(inits, [0, 1, 0, 0], "crop_starts")
    ends_name = _i64(inits, [1, C, crop_size, crop_size], "crop_ends")
    nodes.append(helper.make_node("Slice", [IN_NAME, starts_name, ends_name], ["crop"]))
    conv_input = "crop"

    nodes.extend(
        [
            helper.make_node("Conv", [conv_input, conv_name], ["scores"]),
            helper.make_node("ReduceMax", ["scores"], ["best_scores"], axes=[2, 3], keepdims=0),
            helper.make_node("Greater", ["best_scores", threshold_name], ["present_b"]),
            helper.make_node("Reshape", ["present_b", present_shape], ["present8"]),
        ]
    )

    slot_cells: list[str] = []
    for slot_idx in range(len(MASKS)):
        slot_starts = _i64(inits, [0, slot_idx, 0], f"slot{slot_idx}_starts")
        slot_ends = _i64(inits, [1, slot_idx + 1, len(COLORS)], f"slot{slot_idx}_ends")
        nodes.extend(
            [
                helper.make_node("Slice", ["present8", slot_starts, slot_ends], [f"slot{slot_idx}_8_flat"]),
                helper.make_node("Reshape", [f"slot{slot_idx}_8_flat", slot_shape], [f"slot{slot_idx}_8"]),
                helper.make_node("Slice", [f"slot{slot_idx}_8", first4_starts, first4_ends], [f"slot{slot_idx}_first4"]),
                helper.make_node("Slice", [f"slot{slot_idx}_8", last4_starts, last4_ends], [f"slot{slot_idx}_last4"]),
                helper.make_node(
                    "Concat",
                    [zero, f"slot{slot_idx}_first4", zero, f"slot{slot_idx}_last4"],
                    [f"slot{slot_idx}_cell"],
                    axis=1,
                ),
            ]
        )
        slot_cells.append(f"slot{slot_idx}_cell")

    cell_by_block: dict[tuple[int, int], tuple[str, tuple[tuple[int, int, int], ...]]] = {
        slot: (slot_cells[idx], MASKS[idx]) for idx, slot in enumerate(SLOTS)
    }
    row_names: list[str] = []
    for rr in range(G):
        row_cells: list[str] = []
        for cc in range(G):
            br, bc = rr // T, cc // T
            local_r, local_c = rr % T, cc % T
            if (br, bc) == (1, 1):
                row_cells.append(gray_cell)
                continue
            slot_cell, mask = cell_by_block[(br, bc)]
            row_cells.append(slot_cell if mask[local_r][local_c] else gray_cell)
        row_name = f"out_row{rr}"
        nodes.append(helper.make_node("Concat", row_cells, [row_name], axis=3))
        row_names.append(row_name)

    nodes.extend(
        [
            helper.make_node("Concat", row_names, ["out9_b"], axis=2),
            helper.make_node("Cast", ["out9_b"], ["out9"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out9"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - G, W - G]),
        ]
    )
    return _make_model(nodes, inits, f"task264_structural_decode_{crop_size}")


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            expected = np.asarray(ex["output"], dtype=np.int64)
            ref = solve(np.asarray(ex["input"], dtype=np.int64))
            if not np.array_equal(ref, expected):
                raise AssertionError(f"reference mismatch on {split}[{idx}]")

            pred_oh = _run_onnx(model, _grid_to_onehot(ex["input"]))
            pred = _onehot_to_grid(pred_oh)[: expected.shape[0], : expected.shape[1]]
            active = pred_oh[0, :, : expected.shape[0], : expected.shape[1]] > 0.0
            if not np.array_equal(pred, expected) or not np.all(active.sum(axis=0) == 1):
                bad += 1
    return bad


def _score_candidate(label: str, build: Callable[[], onnx.ModelProto]) -> tuple[int, float, onnx.ModelProto]:
    model = build()
    bad = validate_json(model)
    if bad:
        raise AssertionError(f"{label} failed {bad} examples")

    with tempfile.NamedTemporaryFile(suffix=f"_{TASK_ID}.onnx", dir=OUT_DIR, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        onnx.save(model, tmp_path)
        result = score_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not result["valid"]:
        raise AssertionError(f"{label} invalid: {result['error']}")
    assert result["cost"] is not None and result["score"] is not None
    print(
        f"{label}: nodes={len(model.graph.node)} memory={result['memory']} "
        f"params={result['params']} cost={result['cost']} score={result['score']:.6f}"
    )
    return int(result["cost"]), float(result["score"]), model


def main() -> None:
    candidates = [
        _score_candidate("conv-decode-full", build_full_conv_decode_model),
        _score_candidate("conv-decode-crop16", build_cropped_conv_decode_model),
        _score_candidate("structural-decode-crop16", build_structural_decode_model),
    ]
    _cost, _score, best = min(candidates, key=lambda item: item[0])
    onnx.save(best, BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"nodes:   {len(best.graph.node)}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
