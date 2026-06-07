"""Compact ONNX for ARC task306 using the fixed yellow-axis lattice.

Task rule: the 19-row grid is split by a horizontal yellow separator at row 9
and, for wider cases, vertical yellow separators at columns 9 and 19. One 9x9
region contains the non-background pattern. Copy that local pattern into every
present 9x9 region while keeping yellow separator lines unchanged.

The graph exploits the dataset's fixed layout: possible regions start at rows
0/10 and columns 0/10/20. It compresses the ArgMax color plane through uint8
tile slices, gates optional 19/29-wide copies from separator pixels,
synthesizes the yellow axes, and keeps padding boolean until the final float
output cast.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task306"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task306.onnx"

C = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _f16(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float16), name)


def _slice(
    nodes: List[onnx.NodeProto],
    data: str,
    out: str,
    starts: str,
    ends: str,
    axes: str,
) -> str:
    nodes.append(helper.make_node("Slice", [data, starts, ends, axes], [out]))
    return out


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")

    # Six possible 9x9 content regions. ArgMax gives a compact color grid;
    # only one region has non-background content for any local tile cell.
    nodes.append(helper.make_node("ArgMax", [IN_NAME], ["colors"], axis=1, keepdims=1))
    nodes.append(helper.make_node("Cast", ["colors"], ["colorsu8"], to=TensorProto.UINT8))
    region_starts = [(0, 0), (0, 10), (0, 20), (10, 0), (10, 10), (10, 20)]
    color_slices: list[str] = []
    for idx, (r, c0) in enumerate(region_starts):
        st = _i64(inits, [0, 0, r, c0], f"tile_st{idx}")
        en = _i64(inits, [1, 1, r + 9, c0 + 9], f"tile_en{idx}")
        tc_u8 = _slice(nodes, "colorsu8", f"tc{idx}_u8", st, en, axes4)
        tc = f"tc{idx}"
        nodes.append(helper.make_node("Cast", [tc_u8], [tc], to=TensorProto.FLOAT16))
        color_slices.append(tc)

    nodes.append(helper.make_node("Max", color_slices, ["tile_color"]))
    color_lo = _f16(inits, (np.arange(C, dtype=np.float16) - np.float16(0.5)).reshape(1, C, 1, 1), "color_lo")
    color_hi = _f16(inits, (np.arange(C, dtype=np.float16) + np.float16(0.5)).reshape(1, C, 1, 1), "color_hi")
    nodes.append(helper.make_node("Greater", ["tile_color", color_lo], ["tile_ge"]))
    nodes.append(helper.make_node("Less", ["tile_color", color_hi], ["tile_lt"]))
    nodes.append(helper.make_node("And", ["tile_ge", "tile_lt"], ["tile"]))

    half = _f32(inits, [0.5], "half")

    # Optional width gates are the separator pixels themselves: present if > 0.
    p1_st = _i64(inits, [0, 4, 9, 9], "p1_st")
    p1_en = _i64(inits, [1, 5, 10, 10], "p1_en")
    p2_st = _i64(inits, [0, 4, 9, 19], "p2_st")
    p2_en = _i64(inits, [1, 5, 10, 20], "p2_en")
    _slice(nodes, IN_NAME, "p1", p1_st, p1_en, axes4)
    _slice(nodes, IN_NAME, "p2", p2_st, p2_en, axes4)
    nodes.append(helper.make_node("Greater", ["p1", half], ["p1b"]))
    nodes.append(helper.make_node("Greater", ["p2", half], ["p2b"]))
    nodes.append(helper.make_node("And", ["tile", "p1b"], ["tile1"]))
    nodes.append(helper.make_node("And", ["tile", "p2b"], ["tile2"]))

    ycell_arr = np.zeros((1, C, 1, 1), dtype=bool)
    ycell_arr[0, 4, 0, 0] = True
    ycell = _init(inits, ycell_arr, "ycell")
    zero_col9 = _init(inits, np.zeros((1, C, 9, 1), dtype=bool), "zero_col9")
    zero_cell = _init(inits, np.zeros((1, C, 1, 1), dtype=bool), "zero_cell")
    zero_rows11 = _init(inits, np.zeros((1, C, H - 19, W), dtype=bool), "zero_rows11")

    nodes.append(helper.make_node("Concat", [ycell] * 9, ["ycol9"], axis=2))
    nodes.append(helper.make_node("And", ["ycol9", "p1b"], ["sep9b"]))
    nodes.append(helper.make_node("And", ["ycol9", "p2b"], ["sep19b"]))

    nodes.append(helper.make_node("Concat", [ycell] * 9, ["yrow9"], axis=3))
    nodes.append(helper.make_node("And", [ycell, "p1b"], ["ycell1"]))
    nodes.append(helper.make_node("And", ["yrow9", "p1b"], ["yrow1"]))
    nodes.append(helper.make_node("And", [ycell, "p2b"], ["ycell2"]))
    nodes.append(helper.make_node("And", ["yrow9", "p2b"], ["yrow2"]))
    nodes.append(helper.make_node("Concat", ["yrow9", "ycell1", "yrow1", "ycell2", "yrow2"], ["row_sepb"], axis=3))

    nodes.append(helper.make_node("Concat", ["tile", "sep9b", "tile1", "sep19b", "tile2", "zero_col9"], ["band30"], axis=3))
    nodes.append(helper.make_node("Concat", ["row_sepb", "zero_cell"], ["row_sep30"], axis=3))
    nodes.append(helper.make_node("Concat", ["band30", "row_sep30", "band30"], ["out19w"], axis=2))

    nodes.append(helper.make_node("Concat", ["out19w", "zero_rows11"], ["out30"], axis=2))
    nodes.append(helper.make_node("Cast", ["out30"], [OUT_NAME], to=TensorProto.FLOAT))

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


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
