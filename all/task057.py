"""Minimal ONNX for ARC task057: crop the 3x3 shape bbox, duplicate horizontally.

Task rule: 8x8 input holds one monochrome connected pattern inside some 3x3
window. Take the minimal 3x3 crop (top-left = min row/col of nonzero cells),
concatenate it with itself on width to 3x6, and pad to 30x30 one-hot I/O.

ONNX (best by local score_model): slice foreground channels 1..9 over the 8x8
core, find the top-left occupied row/col, use one GatherND over a channel-last
view to get the 3x3 foreground crop, rebuild the background channel, duplicate
the crop horizontally, and Pad to 30x30. Fully static shapes (no dynamic Slice).
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task057"
BEST_PATH = OUT_DIR / "task057.onnx"
DATA_PATH = ROOT / "data" / "task057.json"

C = 10
IH = IW = 8
OH = 3
OW = 6
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
IR_VERSION = 10

TOY_INPUT = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 8, 8, 0, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0],
    [0, 8, 8, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]
TOY_OUTPUT = [
    [8, 8, 0, 8, 8, 0],
    [0, 8, 0, 0, 8, 0],
    [8, 8, 8, 8, 8, 8],
]


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def build_reference_numpy(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    if g.ndim == 4:
        g = g[0].argmax(axis=0)
    elif g.ndim == 3:
        g = g.argmax(axis=0)

    ys, xs = np.where(g != 0)
    if len(ys) == 0:
        return np.zeros((OH, OW), dtype=np.int64)
    y0, x0 = int(ys.min()), int(xs.min())
    crop = g[y0 : y0 + OH, x0 : x0 + OH]
    return np.concatenate([crop, crop], axis=1)


def _make_model(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    *,
    opset: int,
    graph_name: str,
) -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(nodes, graph_name, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _pick_cells(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    *,
    inp8: str,
    use_bool_masks: bool,
    opset: int,
) -> str:
    """Return ids_out float tensor [1,3,3] from 8x8 one-hot inp8."""
    zi = _i64(inits, [0], "zi")
    rows = _i64(inits, np.arange(IH, dtype=np.int64).reshape(1, 1, IH, 1), "rows")
    cols = _i64(inits, np.arange(IW, dtype=np.int64).reshape(1, 1, 1, IW), "cols")
    zf = _f32(inits, [0.0], "zf")
    shape33 = _i64(inits, [1, OH, OH], "shape33")
    sum_ax = _i64(inits, [1, 2], "sum_ax")
    sq1 = _i64(inits, [1], "sq1")

    nodes.extend(
        [
            helper.make_node("ArgMax", [inp8], ["ids"], axis=1, keepdims=0),
            helper.make_node("Cast", ["ids"], ["idsf"], to=TensorProto.FLOAT),
            helper.make_node("Greater", ["ids", "zi"], ["fg"]),
            helper.make_node("Cast", ["fg"], ["fgf"], to=TensorProto.FLOAT),
        ]
    )
    if opset >= 13:
        nodes.append(helper.make_node("Unsqueeze", ["fgf", "sq1"], ["fgu"]))
    else:
        nodes.append(helper.make_node("Unsqueeze", ["fgf"], ["fgu"], axes=[1]))
    nodes.extend(
        [
            helper.make_node("ReduceMax", ["fgu"], ["row_occ"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["fgu"], ["col_occ"], axes=[2], keepdims=1),
            helper.make_node("ArgMax", ["row_occ"], ["min_r"], axis=2, keepdims=1),
            helper.make_node("ArgMax", ["col_occ"], ["min_c"], axis=3, keepdims=1),
        ]
    )

    for j in range(OH):
        _i64(inits, [j], f"dc{j}")
    for i in range(OH):
        _i64(inits, [i], f"dr{i}")

    cells: List[str] = []
    for i in range(OH):
        off_i = f"dr{i}"
        for j in range(OH):
            off_j = f"dc{j}"
            tr, tc = f"tr{i}{j}", f"tc{i}{j}"
            rm, cm, mask = f"rm{i}{j}", f"cm{i}{j}", f"mask{i}{j}"
            val = f"val{i}{j}"
            nodes.extend(
                [
                    helper.make_node("Add", ["min_r", off_i], [tr]),
                    helper.make_node("Add", ["min_c", off_j], [tc]),
                    helper.make_node("Equal", ["rows", tr], [rm]),
                    helper.make_node("Equal", ["cols", tc], [cm]),
                    helper.make_node("And", [rm, cm], [mask]),
                ]
            )
            if use_bool_masks:
                prod = f"prod{i}{j}"
                msq = f"msq{i}{j}"
                if opset >= 13:
                    nodes.append(helper.make_node("Squeeze", [mask, "sq1"], [msq]))
                else:
                    nodes.append(helper.make_node("Squeeze", [mask], [msq], axes=[1]))
                nodes.extend(
                    [
                        helper.make_node("Cast", [msq], [f"maskf{i}{j}"], to=TensorProto.FLOAT),
                        helper.make_node("Mul", ["idsf", f"maskf{i}{j}"], [prod]),
                    ]
                )
                if opset >= 13:
                    nodes.append(helper.make_node("ReduceSum", [prod, "sum_ax"], [val], keepdims=0))
                else:
                    nodes.append(
                        helper.make_node("ReduceSum", [prod], [val], axes=[1, 2], keepdims=0)
                    )
            else:
                sel = f"sel{i}{j}"
                msq = f"msq{i}{j}"
                if opset >= 13:
                    nodes.append(helper.make_node("Squeeze", [mask, "sq1"], [msq]))
                else:
                    nodes.append(helper.make_node("Squeeze", [mask], [msq], axes=[1]))
                nodes.extend(
                    [
                        helper.make_node("Where", [msq, "idsf", "zf"], [sel]),
                    ]
                )
                if opset >= 13:
                    nodes.append(helper.make_node("ReduceMax", [sel], [val], axes=[1, 2], keepdims=0))
                else:
                    nodes.append(
                        helper.make_node("ReduceMax", [sel], [val], axes=[1, 2], keepdims=0)
                    )
            cells.append(val)

    nodes.append(helper.make_node("Concat", cells, ["cells"], axis=0))
    nodes.append(helper.make_node("Reshape", ["cells", "shape33"], ["ids_out"]))
    return "ids_out"


def _ids_to_onehot(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    ids_out: str,
    *,
    opset: int,
) -> str:
    chs: List[str] = []
    for color in range(C):
        scalar = _f32(inits, [float(color)], f"col{color}")
        eq = f"eq{color}"
        nodes.append(helper.make_node("Equal", [ids_out, scalar], [eq]))
        if opset >= 13:
            nodes.append(helper.make_node("Unsqueeze", [eq, _i64(inits, [1], f"ax1_{color}")], [f"ch{color}"]))
        else:
            nodes.append(helper.make_node("Unsqueeze", [eq], [f"ch{color}"], axes=[1]))
        chs.append(f"ch{color}")
    out = "crop10"
    nodes.append(helper.make_node("Concat", chs, [out], axis=1))
    nodes.append(helper.make_node("Cast", [out], ["crop10f"], to=TensorProto.FLOAT))
    return "crop10f"


def build_gather_pick(opset: int = 13) -> onnx.ModelProto:
    """Gather one-hot cells from flattened 8x8; static shapes throughout."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    ax4 = _i64(inits, [0, 1, 2, 3], "ax4")
    st8 = _i64(inits, [0, 0, 0, 0], "st8")
    en8 = _i64(inits, [1, C, IH, IW], "en8")
    sh64 = _i64(inits, [1, C, IH * IW], "sh64")
    sq1 = _i64(inits, [1], "sq1")
    w8 = _i64(inits, [IW], "w8")
    cellsh = _i64(inits, [1, C, 1, 1], "cellsh")
    pads = _i64(inits, [0, 0, 0, 0, 0, 0, H - OH, W - OW], "pads")
    zi = _i64(inits, [0], "zi")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st8, en8, ax4], ["inp8"]),
            helper.make_node("Reshape", ["inp8", "sh64"], ["flat"]),
            helper.make_node("ArgMax", ["inp8"], ["ids"], axis=1, keepdims=0),
            helper.make_node("Greater", ["ids", "zi"], ["fg"]),
            helper.make_node("Cast", ["fg"], ["fgf"], to=TensorProto.FLOAT),
        ]
    )
    if opset >= 13:
        nodes.append(helper.make_node("Unsqueeze", ["fgf", "sq1"], ["fgu"]))
    else:
        nodes.append(helper.make_node("Unsqueeze", ["fgf"], ["fgu"], axes=[1]))
    nodes.extend(
        [
            helper.make_node("ReduceMax", ["fgu"], ["row_occ"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["fgu"], ["col_occ"], axes=[2], keepdims=1),
            helper.make_node("ArgMax", ["row_occ"], ["min_r"], axis=2, keepdims=1),
            helper.make_node("ArgMax", ["col_occ"], ["min_c"], axis=3, keepdims=1),
        ]
    )

    for j in range(OH):
        _i64(inits, [j], f"dc{j}")
    for i in range(OH):
        _i64(inits, [i], f"dr{i}")

    rows: List[str] = []
    for i in range(OH):
        cells: List[str] = []
        for j in range(OH):
            tr, tc, ri, idx, gathered, cell = (
                f"tr{i}{j}",
                f"tc{i}{j}",
                f"r{i}{j}",
                f"idx{i}{j}",
                f"g{i}{j}",
                f"cell{i}{j}",
            )
            nodes.extend(
                [
                    helper.make_node("Add", ["min_r", f"dr{i}"], [tr]),
                    helper.make_node("Add", ["min_c", f"dc{j}"], [tc]),
                    helper.make_node("Mul", [tr, "w8"], [ri]),
                    helper.make_node("Add", [ri, tc], [idx]),
                    helper.make_node("Gather", ["flat", idx], [gathered], axis=2),
                    helper.make_node("Reshape", [gathered, "cellsh"], [cell]),
                ]
            )
            cells.append(cell)
        row_name = f"row{i}"
        nodes.append(helper.make_node("Concat", cells, [row_name], axis=3))
        rows.append(row_name)

    nodes.extend(
        [
            helper.make_node("Concat", rows, ["crop"], axis=2),
            helper.make_node("Concat", ["crop", "crop"], ["dup"], axis=3),
        ]
    )
    if opset >= 11:
        nodes.append(helper.make_node("Pad", ["dup", pads], [OUT_NAME]))
    else:
        nodes.append(
            helper.make_node(
                "Pad",
                ["dup"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - OH, W - OW],
            )
        )
    return _make_model(nodes, inits, opset=opset, graph_name="gather_pick")


def build_fg9_gather(opset: int = 13) -> onnx.ModelProto:
    """Gather only foreground channels, then rebuild the background channel."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    ax4 = _i64(inits, [0, 1, 2, 3], "ax4")
    st9 = _i64(inits, [0, 1, 0, 0], "st9")
    en9 = _i64(inits, [1, C, IH, IW], "en9")
    sh64 = _i64(inits, [1, C - 1, IH * IW], "sh64")
    w8 = _i64(inits, [IW], "w8")
    cellsh = _i64(inits, [1, C - 1, 1, 1], "cellsh")
    pads = _i64(inits, [0, 0, 0, 0, 0, 0, H - OH, W - OW], "pads")
    zf = _f32(inits, [0.0], "zf")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st9, en9, ax4], ["fg9"]),
            helper.make_node("Reshape", ["fg9", "sh64"], ["flat9"]),
            helper.make_node("ReduceMax", ["fg9"], ["occ"], axes=[1], keepdims=1),
            helper.make_node("ReduceMax", ["occ"], ["row_occ"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["occ"], ["col_occ"], axes=[2], keepdims=1),
            helper.make_node("ArgMax", ["row_occ"], ["min_r"], axis=2, keepdims=1),
            helper.make_node("ArgMax", ["col_occ"], ["min_c"], axis=3, keepdims=1),
        ]
    )

    for j in range(OH):
        _i64(inits, [j], f"dc{j}")
    for i in range(OH):
        _i64(inits, [i], f"dr{i}")

    rows: List[str] = []
    for i in range(OH):
        cells: List[str] = []
        for j in range(OH):
            tr, tc, ri, idx, gathered, cell = (
                f"tr{i}{j}",
                f"tc{i}{j}",
                f"r{i}{j}",
                f"idx{i}{j}",
                f"g{i}{j}",
                f"cell{i}{j}",
            )
            nodes.extend(
                [
                    helper.make_node("Add", ["min_r", f"dr{i}"], [tr]),
                    helper.make_node("Add", ["min_c", f"dc{j}"], [tc]),
                    helper.make_node("Mul", [tr, "w8"], [ri]),
                    helper.make_node("Add", [ri, tc], [idx]),
                    helper.make_node("Gather", ["flat9", idx], [gathered], axis=2),
                    helper.make_node("Reshape", [gathered, "cellsh"], [cell]),
                ]
            )
            cells.append(cell)
        row_name = f"row{i}"
        nodes.append(helper.make_node("Concat", cells, [row_name], axis=3))
        rows.append(row_name)

    nodes.extend(
        [
            helper.make_node("Concat", rows, ["crop9"], axis=2),
            helper.make_node("Concat", ["crop9", "crop9"], ["dup9"], axis=3),
            helper.make_node("ReduceSum", ["dup9"], ["fg_sum"], axes=[1], keepdims=1),
        ]
    )
    if opset >= 11:
        nodes.extend(
            [
                helper.make_node("Equal", ["fg_sum", "zf"], ["bg"]),
                helper.make_node("Cast", ["bg"], ["bgf"], to=TensorProto.FLOAT),
            ]
        )
    else:
        nodes.extend(
            [
                helper.make_node("Cast", ["fg_sum"], ["fg_any"], to=TensorProto.BOOL),
                helper.make_node("Not", ["fg_any"], ["bg"]),
                helper.make_node("Cast", ["bg"], ["bgf"], to=TensorProto.FLOAT),
            ]
        )
    nodes.append(helper.make_node("Concat", ["bgf", "dup9"], ["dup10"], axis=1))
    if opset >= 11:
        nodes.append(helper.make_node("Pad", ["dup10", pads], [OUT_NAME]))
    else:
        nodes.append(
            helper.make_node(
                "Pad",
                ["dup10"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - OH, W - OW],
            )
        )
    return _make_model(nodes, inits, opset=opset, graph_name="fg9_gather")


def build_fg9_gathernd(opset: int = 12) -> onnx.ModelProto:
    """Single GatherND crop over foreground channels, then rebuild background."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    ax4 = _i64(inits, [0, 1, 2, 3], "ax4")
    st9 = _i64(inits, [0, 1, 0, 0], "st9")
    en9 = _i64(inits, [1, C, IH, IW], "en9")
    rows33 = _i64(
        inits,
        np.broadcast_to(np.arange(OH, dtype=np.int64).reshape(1, OH, 1), (1, OH, OH)),
        "rows33",
    )
    cols33 = _i64(
        inits,
        np.broadcast_to(np.arange(OH, dtype=np.int64).reshape(1, 1, OH), (1, OH, OH)),
        "cols33",
    )
    pads = _i64(inits, [0, 0, 0, 0, 0, 0, H - OH, W - OW], "pads")
    zf = _f32(inits, [0.0], "zf")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st9, en9, ax4], ["fg9"]),
            helper.make_node("Transpose", ["fg9"], ["fg9_nhwc"], perm=[0, 2, 3, 1]),
            helper.make_node("ReduceMax", ["fg9"], ["occ"], axes=[1], keepdims=1),
            helper.make_node("ReduceMax", ["occ"], ["row_occ"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["occ"], ["col_occ"], axes=[2], keepdims=1),
            helper.make_node("ArgMax", ["row_occ"], ["min_r4"], axis=2, keepdims=1),
            helper.make_node("ArgMax", ["col_occ"], ["min_c4"], axis=3, keepdims=1),
            helper.make_node("Squeeze", ["min_r4"], ["min_r"], axes=[1, 2, 3]),
            helper.make_node("Squeeze", ["min_c4"], ["min_c"], axes=[1, 2, 3]),
            helper.make_node("Add", ["min_r", "rows33"], ["ridx"]),
            helper.make_node("Add", ["min_c", "cols33"], ["cidx"]),
            helper.make_node("Unsqueeze", ["ridx"], ["ridxu"], axes=[3]),
            helper.make_node("Unsqueeze", ["cidx"], ["cidxu"], axes=[3]),
            helper.make_node("Concat", ["ridxu", "cidxu"], ["indices"], axis=3),
            helper.make_node("GatherND", ["fg9_nhwc", "indices"], ["crop9_nhwc"], batch_dims=1),
            helper.make_node("Transpose", ["crop9_nhwc"], ["crop9"], perm=[0, 3, 1, 2]),
            helper.make_node("Concat", ["crop9", "crop9"], ["dup9"], axis=3),
            helper.make_node("ReduceSum", ["dup9"], ["fg_sum"], axes=[1], keepdims=1),
            helper.make_node("Equal", ["fg_sum", "zf"], ["bg"]),
            helper.make_node("Cast", ["bg"], ["bgf"], to=TensorProto.FLOAT),
            helper.make_node("Concat", ["bgf", "dup9"], ["dup10"], axis=1),
        ]
    )
    if opset >= 11:
        nodes.append(helper.make_node("Pad", ["dup10", pads], [OUT_NAME]))
    else:
        nodes.append(
            helper.make_node(
                "Pad",
                ["dup10"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - OH, W - OW],
            )
        )
    return _make_model(nodes, inits, opset=opset, graph_name="fg9_gathernd")


def build_cell_pick(opset: int = 13, *, use_bool_masks: bool = True) -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    ax4 = _i64(inits, [0, 1, 2, 3], "ax4")
    st8 = _i64(inits, [0, 0, 0, 0], "st8")
    en8 = _i64(inits, [1, C, IH, IW], "en8")
    pads = _i64(inits, [0, 0, 0, 0, 0, 0, H - OH, W - OW], "pads")

    nodes.append(helper.make_node("Slice", [IN_NAME, st8, en8, ax4], ["inp8"]))
    ids_out = _pick_cells(nodes, inits, inp8="inp8", use_bool_masks=use_bool_masks, opset=opset)
    crop = _ids_to_onehot(nodes, inits, ids_out, opset=opset)
    nodes.extend(
        [
            helper.make_node("Concat", [crop, crop], ["dup"], axis=3),
        ]
    )
    if opset >= 11:
        nodes.append(helper.make_node("Pad", ["dup", pads], [OUT_NAME]))
    else:
        nodes.append(
            helper.make_node(
                "Pad",
                ["dup"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - OH, W - OW],
            )
        )
    return _make_model(nodes, inits, opset=opset, graph_name="cell_pick")


def build_cascade_where(opset: int = 13) -> onnx.ModelProto:
    """Two-stage static Slice + Where selection (6 + 6 windows)."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    zi = _i64(inits, [0], "zi")
    ax4 = _i64(inits, [0, 1, 2, 3], "ax4")
    st8 = _i64(inits, [0, 0, 0, 0], "st8")
    en8 = _i64(inits, [1, C, IH, IW], "en8")
    sq1 = _i64(inits, [1], "sq1")
    pads = _i64(inits, [0, 0, 0, 0, 0, 0, H - OH, W - OW], "pads")

    nodes.append(helper.make_node("Slice", [IN_NAME, st8, en8, ax4], ["inp8"]))
    nodes.extend(
        [
            helper.make_node("ArgMax", ["inp8"], ["ids"], axis=1, keepdims=0),
            helper.make_node("Greater", ["ids", "zi"], ["fg"]),
            helper.make_node("Cast", ["fg"], ["fgf"], to=TensorProto.FLOAT),
        ]
    )
    if opset >= 13:
        nodes.append(helper.make_node("Unsqueeze", ["fgf", "sq1"], ["fgu"]))
    else:
        nodes.append(helper.make_node("Unsqueeze", ["fgf"], ["fgu"], axes=[1]))
    nodes.extend(
        [
            helper.make_node("ReduceMax", ["fgu"], ["row_occ"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["fgu"], ["col_occ"], axes=[2], keepdims=1),
            helper.make_node("ArgMax", ["row_occ"], ["min_r"], axis=2, keepdims=1),
            helper.make_node("ArgMax", ["col_occ"], ["min_c"], axis=3, keepdims=1),
        ]
    )

    row_wins: List[str] = []
    for r in range(IH - OH + 1):
        st = _i64(inits, [0, 0, r, 0], f"rst{r}")
        en = _i64(inits, [1, C, r + OH, IW], f"ren{r}")
        name = f"rw{r}"
        nodes.append(helper.make_node("Slice", ["inp8", st, en, ax4], [name]))
        row_wins.append(name)

    acc_row = row_wins[0]
    for r in range(1, IH - OH + 1):
        rv = _i64(inits, [r], f"rv{r}")
        match = f"rm{r}"
        nxt = f"ar{r}"
        nodes.extend(
            [
                helper.make_node("Equal", ["min_r", rv], [match]),
                helper.make_node("Where", [match, row_wins[r], acc_row], [nxt]),
            ]
        )
        acc_row = nxt

    col_wins: List[str] = []
    for c in range(IW - OH + 1):
        st = _i64(inits, [0, 0, 0, c], f"cst{c}")
        en = _i64(inits, [1, C, OH, c + OH], f"cen{c}")
        name = f"cw{c}"
        nodes.append(helper.make_node("Slice", [acc_row, st, en, ax4], [name]))
        col_wins.append(name)

    acc_col = col_wins[0]
    for c in range(1, IW - OH + 1):
        cv = _i64(inits, [c], f"cv{c}")
        match = f"cm{c}"
        nxt = f"ac{c}"
        nodes.extend(
            [
                helper.make_node("Equal", ["min_c", cv], [match]),
                helper.make_node("Where", [match, col_wins[c], acc_col], [nxt]),
            ]
        )
        acc_col = nxt

    nodes.extend(
        [
            helper.make_node("Concat", [acc_col, acc_col], ["dup"], axis=3),
        ]
    )
    if opset >= 11:
        nodes.append(helper.make_node("Pad", ["dup", pads], [OUT_NAME]))
    else:
        nodes.append(
            helper.make_node(
                "Pad",
                ["dup"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - OH, W - OW],
            )
        )
    return _make_model(nodes, inits, opset=opset, graph_name="cascade_where")


@dataclass(frozen=True)
class Variant:
    name: str
    builder: Callable[[], onnx.ModelProto]


VARIANTS: List[Variant] = [
    Variant("fg9_gathernd_op12", lambda: build_fg9_gathernd(12)),
]


def _examples() -> list[dict[str, Any]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [ex for split in ("train", "test", "arc-gen") for ex in data.get(split, [])]


def verify_model(model: onnx.ModelProto) -> tuple[bool, str]:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    for idx, example in enumerate(_examples()):
        inp = convert_to_numpy(example, "input")
        expected = convert_to_numpy(example, "output")
        if inp is None or expected is None:
            continue
        actual = session.run([OUT_NAME], {IN_NAME: inp})[0]
        if not np.array_equal(actual > 0.0, expected > 0.0):
            return False, f"example {idx} mismatch"
    return True, f"all {len(_examples())} examples matched"


def run_experiments() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for variant in VARIANTS:
        path = OUT_DIR / f"task057_{variant.name}.onnx"
        row: dict[str, Any] = {"name": variant.name, "path": path}
        try:
            model = variant.builder()
            onnx.save(model, str(path))
            ok, msg = verify_model(model)
            row["valid"] = ok
            row["validation"] = msg
            scored = score_file(path)
            row.update(
                {
                    "memory": scored.get("memory"),
                    "params": scored.get("params"),
                    "cost": scored.get("cost"),
                    "score": scored.get("score"),
                    "scored_valid": scored.get("valid"),
                    "error": scored.get("error"),
                }
            )
        except Exception as exc:
            row["valid"] = False
            row["validation"] = str(exc)
            row["scored_valid"] = False
            row["error"] = str(exc)
        results.append(row)
        print(
            f"{variant.name}: valid={row.get('valid')} "
            f"mem={row.get('memory')} params={row.get('params')} "
            f"cost={row.get('cost')} score={row.get('score')} err={row.get('error')}"
        )
    return results


def pick_best(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    ok = [
        r
        for r in results
        if r.get("valid") and r.get("scored_valid") and r.get("cost") is not None
    ]
    if not ok:
        return None
    return min(ok, key=lambda r: (r["cost"], r.get("memory") or 0, r.get("params") or 0))


def save_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    results = run_experiments()
    best = pick_best(results)
    if best is None:
        raise RuntimeError("no valid scored variant")
    src = Path(best["path"])
    model = onnx.load(str(src))
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    print(
        f"BEST {best['name']}: memory={best['memory']} params={best['params']} "
        f"cost={best['cost']} score={best['score']}"
    )
    return model


def test() -> None:
    inp = np.array(TOY_INPUT, dtype=np.int64)
    exp = np.array(TOY_OUTPUT, dtype=np.int64)
    ref = build_reference_numpy(inp)
    assert np.array_equal(ref, exp)

    model = onnx.load(str(BEST_PATH))
    ok, msg = verify_model(model)
    print(f"verify: {msg}")
    if not ok:
        raise SystemExit(1)

    scored = score_file(BEST_PATH)
    print(
        f"memory={scored['memory']} params={scored['params']} "
        f"cost={scored['cost']} score={scored['score']}"
    )


def main() -> None:
    save_model()
    test()
    print(f"saved {BEST_PATH} ({BEST_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
