"""ONNX for ARC task208: copy the completed rectangular frame.

Task rule: a 21x21 grid contains one colored rectangular ring with a black
interior, plus another all-black rectangle of the same interior size elsewhere.
Keep the grid unchanged except draw the same colored border around the second
black rectangle. Ring outer sizes in the examples are 4..7 cells in each axis,
excluding 4x4.

ONNX: count foreground colors to identify the frame color by its border area,
detect the source ring shape from a compact non-black plane plus black
interior candidates, then expand matching candidates back to border pixels and
overwrite only those border cells.
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

BEST_PATH = OUT_DIR / "task208.onnx"
DATA_PATH = ROOT / "data" / "task208.json"

C = 10
NC = 9
G = 21
H = W = 30
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _border_kernel(oh: int, ow: int) -> np.ndarray:
    k = np.zeros((oh, ow), dtype=np.float32)
    k[0, :] = 1.0
    k[-1, :] = 1.0
    k[:, 0] = 1.0
    k[:, -1] = 1.0
    return k


def _inner_kernel(oh: int, ow: int) -> np.ndarray:
    k = np.zeros((oh, ow), dtype=np.float32)
    k[1:-1, 1:-1] = 1.0
    return k


def solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    for oh in range(4, 8):
        for ow in range(4, 8):
            ih, iw = oh - 2, ow - 2
            border = _border_kernel(oh, ow).astype(bool)
            for color in range(1, 10):
                found = False
                for r in range(G - oh + 1):
                    for c in range(G - ow + 1):
                        patch = g[r : r + oh, c : c + ow]
                        if np.all(patch[border] == color) and np.all(patch[1:-1, 1:-1] == 0):
                            found = True
                            break
                    if found:
                        break
                if not found:
                    continue
                for r in range(G - oh + 1):
                    for c in range(G - ow + 1):
                        if np.all(g[r + 1 : r + 1 + ih, c + 1 : c + 1 + iw] == 0):
                            out[r : r + oh, c : c + ow][border] = color
                return out
    return out


def _grid_to_onehot(grid: List[List[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes_chw = _i64(inits, [1, 2, 3], "axes_chw")
    black_st = _i64(inits, [0, 0, 0], "black_st")
    black_en = _i64(inits, [1, G, G], "black_en")
    fg_st = _i64(inits, [1, 0, 0], "fg_st")
    fg_en = _i64(inits, [C, G, G], "fg_en")
    zero = _f32(inits, [0.0], "zero")
    pads = [0, 0, 0, 0, 0, 0, H - G, W - G]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, black_st, black_en, axes_chw], ["black"]),
            helper.make_node("Slice", [IN_NAME, fg_st, fg_en, axes_chw], ["fg"]),
            helper.make_node("Cast", ["fg"], ["fg_b"], to=TensorProto.BOOL),
            helper.make_node("ReduceSum", ["fg"], ["color_counts"], axes=[2, 3], keepdims=1),
            helper.make_node("ReduceMax", ["fg"], ["fg_any"], axes=[1], keepdims=1),
        ]
    )

    color_lo = _f32(inits, [13.5], "color_lo")
    color_hi = _f32(inits, [24.5], "color_hi")
    nodes.extend(
        [
            helper.make_node("Greater", ["color_counts", color_lo], ["color_gt"]),
            helper.make_node("Less", ["color_counts", color_hi], ["color_lt"]),
            helper.make_node("And", ["color_gt", "color_lt"], ["color_mask"]),
        ]
    )

    paint_parts: List[str] = []
    for oh in range(4, 8):
        for ow in range(4, 8):
            ih, iw = oh - 2, ow - 2
            suffix = f"{oh}x{ow}"
            border_area = oh * 2 + ow * 2 - 4
            if border_area == 12:
                continue
            cand_h = G - oh + 1
            cand_w = G - ow + 1
            inner_area = float(ih * iw)
            border_area_f = float(border_area)

            inner_w = _f32(inits, _inner_kernel(oh, ow).reshape(1, 1, oh, ow), f"inner_w_{suffix}")
            inner_need = _f32(inits, [inner_area - 0.5], f"inner_need_{suffix}")
            border = _border_kernel(oh, ow)
            border_w = _f32(inits, border.reshape(1, 1, oh, ow), f"border_w_{suffix}")
            source_need = _f32(inits, [inner_area + border_area_f - 0.5], f"source_need_{suffix}")

            nodes.extend(
                [
                    helper.make_node("Conv", ["black", inner_w], [f"inner_sum_{suffix}"]),
                    helper.make_node("Greater", [f"inner_sum_{suffix}", inner_need], [f"cand_{suffix}"]),
                    helper.make_node(
                        "Conv",
                        ["fg_any", border_w],
                        [f"border_sum_{suffix}"],
                    ),
                    helper.make_node("Add", [f"inner_sum_{suffix}", f"border_sum_{suffix}"], [f"ring_score_{suffix}"]),
                    helper.make_node(
                        "ReduceMax",
                        [f"ring_score_{suffix}"],
                        [f"ring_exists_f_{suffix}"],
                        axes=[2, 3],
                        keepdims=1,
                    ),
                    helper.make_node("Greater", [f"ring_exists_f_{suffix}", source_need], [f"ring_exists_{suffix}"]),
                    helper.make_node("Cast", [f"cand_{suffix}"], [f"cand_f_{suffix}"], to=TensorProto.FLOAT),
                    helper.make_node("ConvTranspose", [f"cand_f_{suffix}", border_w], [f"paint_f_{suffix}"]),
                    helper.make_node("Greater", [f"paint_f_{suffix}", zero], [f"paint_b_{suffix}"]),
                    helper.make_node("And", [f"ring_exists_{suffix}", f"paint_b_{suffix}"], [f"paint_{suffix}"]),
                ]
            )
            paint_parts.append(f"paint_{suffix}")

    paint_acc = paint_parts[0]
    for idx, name in enumerate(paint_parts[1:], 1):
        out = f"paint_shape_acc_{idx}"
        nodes.append(helper.make_node("Or", [paint_acc, name], [out]))
        paint_acc = out

    acc = "paint_colored"
    nodes.append(helper.make_node("And", [paint_acc, "color_mask"], [acc]))

    nodes.extend(
        [
            helper.make_node("Not", [paint_acc], ["keep_mask"]),
            helper.make_node("And", ["fg_b", "keep_mask"], ["kept_fg"]),
            helper.make_node("Or", ["kept_fg", acc], ["out_fg_b"]),
            helper.make_node("Cast", ["black"], ["bg_b"], to=TensorProto.BOOL),
            helper.make_node("And", ["bg_b", "keep_mask"], ["out_bg_b"]),
            helper.make_node("Concat", ["out_bg_b", "out_fg_b"], ["out21_b"], axis=1),
            helper.make_node("Cast", ["out21_b"], ["out21"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out21"], [OUT_NAME], pads=pads),
        ]
    )

    graph = helper.make_graph(nodes, "task208", [x_info], [y_info], initializer=inits)
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
        for idx, ex in enumerate(data[split]):
            g = np.array(ex["input"], dtype=np.int64)
            expected = np.array(ex["output"], dtype=np.int64)
            if not np.array_equal(solve(g), expected):
                raise AssertionError(f"reference mismatch {split} {idx}")
            pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(ex["input"])))[: G, : G]
            if not np.array_equal(pred, expected):
                bad += 1
                print(f"mismatch {split} {idx}")
    return bad


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    bad = validate_json(model)
    assert bad == 0, f"{bad} JSON examples failed"
    print(score_file(BEST_PATH))


if __name__ == "__main__":
    main()
