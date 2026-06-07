"""ONNX solution for NeuroGolf task384: crop and double a small object.

Task rule: each 9x9 input contains a black background and one connected
non-black object. Find the bounding box of the non-black cells, crop that box,
and scale the cropped pattern by 2x in both axes. Every source cell, including
black cells inside the crop, becomes a solid 2x2 block. The output occupies the
top-left 2*bbox_height by 2*bbox_width area and the rest of the 30x30
competition canvas is left unused. In the provided train/test/arc-gen data the
only object color is 4, every nonzero cell is inside the inner 7x7 window, and
the largest crop is 4x5, so the doubled useful area is at most 8x10.

ONNX approach: slice only the color-4 channel, compute the bbox using compact
int32 reductions, gather an 8x10 nearest-neighbor color-4 mask, synthesize the
black/color-4 channels in a 5-channel tensor, and use the final Pad to expand to
the required 10-channel 30x30 output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task384"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
CORE = 9
CROP = 7
CROP_START = 1
MAX_OUT_H = 8
MAX_OUT_W = 10
USED_CHANNELS = 5
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def solve(grid: Sequence[Sequence[int]]) -> np.ndarray:
    """Reference numpy implementation of bbox crop plus 2x expansion."""
    g = np.asarray(grid, dtype=np.int64)
    rows, cols = np.nonzero(g != 0)
    top, bottom = rows.min(), rows.max()
    left, right = cols.min(), cols.max()
    crop = g[top : bottom + 1, left : right + 1]
    return np.repeat(np.repeat(crop, 2, axis=0), 2, axis=1)


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _init(inits, np.asarray([0, 1, 2, 3], dtype=np.int64), "axes4")
    st_ch4 = _init(inits, np.asarray([0, 4, CROP_START, CROP_START], dtype=np.int64), "st_ch4")
    end_ch4 = _init(
        inits,
        np.asarray([1, 5, CROP_START + CROP, CROP_START + CROP], dtype=np.int64),
        "end_ch4",
    )
    flat_shape = _init(inits, np.asarray([1, 1, CROP * CROP], dtype=np.int64), "flat_shape")
    row_coord = _init(
        inits,
        np.arange(CROP, dtype=np.int32).reshape(1, 1, CROP, 1),
        "row_coord",
    )
    col_coord = _init(
        inits,
        np.arange(CROP, dtype=np.int32).reshape(1, 1, 1, CROP),
        "col_coord",
    )
    row_rel = _init(
        inits,
        np.repeat(np.arange(MAX_OUT_H // 2, dtype=np.int32), 2).reshape(MAX_OUT_H, 1),
        "row_rel",
    )
    col_rel = _init(
        inits,
        np.repeat(np.arange(MAX_OUT_W // 2, dtype=np.int32), 2).reshape(1, MAX_OUT_W),
        "col_rel",
    )
    zero = _init(inits, np.asarray(0, dtype=np.int32), "zero")
    zero_f = _init(inits, np.asarray(0.0, dtype=np.float32), "zero_f")
    big = _init(inits, np.asarray(CROP, dtype=np.int32), "big")
    crop_w = _init(inits, np.asarray(CROP, dtype=np.int32), "crop_w")
    zero3 = _init(
        inits,
        np.zeros((1, 3, MAX_OUT_H, MAX_OUT_W), dtype=bool),
        "zero3",
    )

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st_ch4, end_ch4, axes4], ["x4"]),
            helper.make_node("Cast", ["x4"], ["x4b"], to=TensorProto.BOOL),
            helper.make_node("ReduceMax", ["x4"], ["row_hit"], axes=[3], keepdims=1),
            helper.make_node("Greater", ["row_hit", zero_f], ["row_has"]),
            helper.make_node("ReduceMax", ["x4"], ["col_hit"], axes=[2], keepdims=1),
            helper.make_node("Greater", ["col_hit", zero_f], ["col_has"]),
            helper.make_node("Where", ["row_has", row_coord, big], ["r_min_map"]),
            helper.make_node("ReduceMin", ["r_min_map"], ["top"], axes=[0, 1, 2, 3], keepdims=0),
            helper.make_node("Where", ["row_has", row_coord, zero], ["r_max_map"]),
            helper.make_node("ReduceMax", ["r_max_map"], ["bottom"], axes=[0, 1, 2, 3], keepdims=0),
            helper.make_node("Where", ["col_has", col_coord, big], ["c_min_map"]),
            helper.make_node("ReduceMin", ["c_min_map"], ["left"], axes=[0, 1, 2, 3], keepdims=0),
            helper.make_node("Where", ["col_has", col_coord, zero], ["c_max_map"]),
            helper.make_node("ReduceMax", ["c_max_map"], ["right"], axes=[0, 1, 2, 3], keepdims=0),
            helper.make_node("Reshape", ["x4b", flat_shape], ["xflat"]),
            helper.make_node("Add", ["top", row_rel], ["src_r"]),
            helper.make_node("Add", ["left", col_rel], ["src_c"]),
            helper.make_node("Greater", ["src_r", "bottom"], ["r_after"]),
            helper.make_node("Not", ["r_after"], ["r_ok"]),
            helper.make_node("Greater", ["src_c", "right"], ["c_after"]),
            helper.make_node("Not", ["c_after"], ["c_ok"]),
            helper.make_node("Where", ["r_ok", "src_r", "bottom"], ["src_r_clip"]),
            helper.make_node("Where", ["c_ok", "src_c", "right"], ["src_c_clip"]),
            helper.make_node("Mul", ["src_r_clip", crop_w], ["src_r7"]),
            helper.make_node("Add", ["src_r7", "src_c_clip"], ["src_idx"]),
            helper.make_node("And", ["r_ok", "c_ok"], ["active2"]),
            helper.make_node("Gather", ["xflat", "src_idx"], ["gathered"], axis=2),
            helper.make_node("And", ["gathered", "active2"], ["ch4"]),
            helper.make_node("Xor", ["active2", "ch4"], ["ch0"]),
            helper.make_node("Concat", ["ch0", zero3, "ch4"], ["out5b"], axis=1),
            helper.make_node("Cast", ["out5b"], ["out5"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out5"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, C - USED_CHANNELS, H - MAX_OUT_H, W - MAX_OUT_W],
            ),
        ]
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


def _grid_to_onehot(grid: Sequence[Sequence[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def _load_task() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def verify_model(path: Path) -> None:
    import onnxruntime as ort

    task = _load_task()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(task[split]):
            expected_grid = solve(ex["input"])
            json_expected = np.asarray(ex["output"], dtype=np.int64)
            if not np.array_equal(expected_grid, json_expected):
                raise AssertionError(f"reference solver disagrees with JSON on {split}[{idx}]")

            expected = _grid_to_onehot(json_expected)
            got = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(ex["input"])})[0]
            if not np.array_equal(got > 0.0, expected > 0.0):
                raise AssertionError(f"ONNX output mismatch on {split}[{idx}]")


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    verify_model(BEST_PATH)

    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"{TASK_ID}.json: PASS")
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
