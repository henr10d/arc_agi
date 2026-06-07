"""Minimal ONNX for ARC task260: complete diagonals around a gray mask.

Task rule: the 10x10 input contains one nonzero, non-gray color on a slope +1
diagonal.  Gray cells form an occluding mask next to that diagonal.  Remove all
gray cells, keep the original colored diagonal, and add complete parallel
slope +1 diagonals on the outside edge of each side of the gray mask: if gray
appears at diagonal keys smaller than the colored key, fill key min(gray)-2;
if gray appears at larger keys, fill key max(gray)+2.  Output only the target
color and black background.
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

TASK_ID = "task260"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task260.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
G = 10
KEYS = 2 * G - 1
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


def _i32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int32), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _f16(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float16), name)


def _bool(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=bool), name)


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


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver for the gray-mask diagonal rule."""
    g = np.asarray(grid, dtype=np.int64)
    colors = [int(v) for v in sorted(set(g.ravel())) if v not in (0, 5)]
    color = colors[0]
    colored = np.argwhere(g == color)
    main = int(np.bincount(colored[:, 1] - colored[:, 0] + (G - 1), minlength=KEYS).argmax()) - (G - 1)
    gray = np.argwhere(g == 5)
    keys = {main}
    if gray.size:
        gray_keys = gray[:, 1] - gray[:, 0]
        if int(gray_keys.min()) < main:
            keys.add(int(gray_keys.min()) - 2)
        if int(gray_keys.max()) > main:
            keys.add(int(gray_keys.max()) + 2)

    out = np.zeros_like(g)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if c - r in keys:
                out[r, c] = color
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


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    zero_f = _f32(inits, 0.0, "zero_f")
    two_i = _f16(inits, 2, "two_i")
    big_i = _f16(inits, 99, "big_i")
    small_i = _f16(inits, -99, "small_i")
    ten_i64 = _i64(inits, 10, "ten_i64")
    nine_i64 = _i64(inits, G - 1, "nine_i64")

    d = np.fromfunction(lambda r, c: c - r, (G, G), dtype=int).astype(np.int64)
    key_index_map = _f16(inits, (d + (G - 1)).reshape(1, 1, G, G), "key_index_map")
    key_index_map_i32 = _i32(inits, (d + (G - 1)).reshape(1, 1, G, G), "key_index_map_i32")

    gray_starts = _i64(inits, [0, 5, 0, 0], "gray_starts")
    gray_ends = _i64(inits, [1, 6, G, G], "gray_ends")
    bg_starts = _i64(inits, [0, 0, 0, 0], "bg_starts")
    bg_ends = _i64(inits, [1, 1, G, G], "bg_ends")
    false_gray = _bool(inits, np.zeros((1, 1, 1, 1), dtype=bool), "false_gray")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, gray_starts, gray_ends], ["gray"]),
            helper.make_node("Greater", ["gray", zero_f], ["gray_present"]),
            helper.make_node("Slice", [IN_NAME, bg_starts, bg_ends], ["background_in"]),
            helper.make_node("Greater", ["background_in", zero_f], ["background_in_present"]),
            helper.make_node("Or", ["background_in_present", "gray_present"], ["not_target"]),
            helper.make_node("Not", ["not_target"], ["target_spatial"]),
            helper.make_node("Flatten", ["target_spatial"], ["target_flatb"], axis=1),
            helper.make_node("Cast", ["target_flatb"], ["target_flat"], to=TensorProto.UINT8),
            helper.make_node("ArgMax", ["target_flat"], ["main_pos"], axis=1, keepdims=1),
            helper.make_node("Div", ["main_pos", ten_i64], ["main_row"]),
            helper.make_node("Mod", ["main_pos", ten_i64], ["main_col"], fmod=0),
            helper.make_node("Sub", ["main_col", "main_row"], ["main_key_signed"]),
            helper.make_node("Add", ["main_key_signed", nine_i64], ["main_key_i64"]),
            helper.make_node("Cast", ["main_key_i64"], ["main_idx"], to=TensorProto.FLOAT16),
            helper.make_node("Cast", ["main_key_i64"], ["main_idx_i32"], to=TensorProto.INT32),
            helper.make_node("ReduceSum", [IN_NAME], ["channel_count"], axes=[2, 3], keepdims=1),
            helper.make_node("Greater", ["channel_count", zero_f], ["channel_present"]),
            helper.make_node(
                "Split",
                ["channel_present"],
                ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"],
                axis=1,
                split=[1] * C,
            ),
            helper.make_node("Concat", ["c1", "c2", "c3", "c4", false_gray, "c6", "c7", "c8", "c9"], ["target_color9"], axis=1),
            helper.make_node("Less", [key_index_map, "main_idx"], ["left_of_main"]),
            helper.make_node("Greater", [key_index_map, "main_idx"], ["right_of_main"]),
            helper.make_node("And", ["gray_present", "left_of_main"], ["left_gray"]),
            helper.make_node("And", ["gray_present", "right_of_main"], ["right_gray"]),
            helper.make_node("Where", ["left_gray", key_index_map, big_i], ["left_source"]),
            helper.make_node("ReduceMin", ["left_source"], ["left_gray_min"], axes=[2, 3], keepdims=1),
            helper.make_node("Sub", ["left_gray_min", two_i], ["left_out_idx"]),
            helper.make_node("Cast", ["left_out_idx"], ["left_out_idx_i32"], to=TensorProto.INT32),
            helper.make_node("Where", ["right_gray", key_index_map, small_i], ["right_source"]),
            helper.make_node("ReduceMax", ["right_source"], ["right_gray_max"], axes=[2, 3], keepdims=1),
            helper.make_node("Add", ["right_gray_max", two_i], ["right_out_idx"]),
            helper.make_node("Cast", ["right_out_idx"], ["right_out_idx_i32"], to=TensorProto.INT32),
            helper.make_node("Equal", [key_index_map_i32, "main_idx_i32"], ["main_paint"]),
            helper.make_node("Equal", [key_index_map_i32, "left_out_idx_i32"], ["left_paint"]),
            helper.make_node("Equal", [key_index_map_i32, "right_out_idx_i32"], ["right_paint"]),
            helper.make_node("Or", ["main_paint", "left_paint"], ["paint_lr"]),
            helper.make_node("Or", ["paint_lr", "right_paint"], ["paint"]),
            helper.make_node("Not", ["paint"], ["background"]),
            helper.make_node("And", ["target_color9", "paint"], ["color_out9"]),
            helper.make_node("Concat", ["background", "color_out9"], ["out10b"], axis=1),
            helper.make_node("Cast", ["out10b"], ["out10"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out10"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - G, W - G]),
        ]
    )

    return _make_model(nodes, inits, "task260_diagonal_mask")


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            g = np.asarray(ex["input"], dtype=np.int64)
            expected = np.asarray(ex["output"], dtype=np.int64)
            ref = solve(g)
            if not np.array_equal(ref, expected):
                raise AssertionError(f"reference solver failed {split}")
            pred_oh = _run_onnx(model, _grid_to_onehot(ex["input"]))
            pred = _onehot_to_grid(pred_oh)[: expected.shape[0], : expected.shape[1]]
            active = pred_oh[0, :, : expected.shape[0], : expected.shape[1]] > 0.0
            if not np.array_equal(pred, expected) or not np.all(active.sum(axis=0) == 1):
                bad += 1
    return bad


def main() -> None:
    model = build_model()
    bad = validate_json(model)
    if bad:
        raise AssertionError(f"{TASK_ID} failed {bad} examples")

    with tempfile.NamedTemporaryFile(suffix=f"_{TASK_ID}.onnx", dir=OUT_DIR, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        onnx.save(model, tmp_path)
        candidate = score_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    if not candidate["valid"]:
        raise AssertionError(f"candidate invalid: {candidate['error']}")

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
