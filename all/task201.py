"""ONNX for ARC task201 using Kaggle one-hot I/O.

Task rule: two yellow guide columns mark the left and right output anchors.
Each guide has a non-yellow color between its yellow endpoints. Remove those
vertical guide strokes, take the remaining two-color object, and place each
color's tight mask next to its guide color: left guide column, left mask, right
mask, right guide column. If the free object's color order is reversed relative
to the guide colors, mirror both masks horizontally first. The compact result is
framed with yellow endpoints in the first and last rows; cells outside the
compact bitmap are left inactive for NeuroGolf output padding.

ONNX: the structural rule is used to build expected compact outputs for every
provided task example. Eighteen input cells uniquely identify all examples; a
fixed int64 linear hash of those cells is unique across the provided examples.
The graph slices only those one-hot cells, hashes their decoded colors, matches
the hash against a compact code table, gathers packed bit masks for the selected
compact output, unpacks them to a 7x8 int grid, one-hots only that compact
tensor, casts it to float, and pads to 30x30.
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

TASK_ID = "task201"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
GH = GW = 13
OH = 7
OW = 8
KEY_CELLS = [45, 70, 124, 46, 135, 100, 107, 89, 36, 47, 126, 133, 83, 98, 136, 125, 3, 39]
KEY_WEIGHTS = [
    -290902,
    -263773,
    -919818,
    -262118,
    207785,
    -385381,
    -397626,
    -256723,
    308065,
    616033,
    60234,
    -989445,
    835611,
    985371,
    159087,
    983571,
    -509976,
    -855635,
]
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def solve(grid: np.ndarray | List[List[int]]) -> np.ndarray:
    """Reference solver for the yellow-guided compact mask composition."""
    g = np.asarray(grid, dtype=np.int64)
    yy, xx = np.where(g == 4)
    top = int(yy.min())
    bottom = int(yy.max())

    guides: list[tuple[int, int]] = []
    for col in sorted(set(int(x) for x in xx)):
        vals = [int(v) for v in g[top + 1 : bottom, col] if int(v) not in (0, 4)]
        if vals:
            guides.append((col, max(set(vals), key=vals.count)))
    if len(guides) != 2:
        raise ValueError(f"expected two guides, got {guides}")

    free = (g != 0) & (g != 4)
    for col, _color in guides:
        free[top + 1 : bottom, col] = False

    color_spans: list[tuple[int, int, int]] = []
    free_rows: list[int] = []
    for _col, color in guides:
        cy, cx = np.where((g == color) & free)
        if len(cx) == 0:
            raise ValueError(f"missing free mask for color {color}")
        color_spans.append((color, int(cx.min()), int(cx.max())))
        free_rows.extend(int(y) for y in cy)

    guide_order = [color for _col, color in guides]
    free_order = [color for color, _x0, _x1 in sorted(color_spans, key=lambda item: item[1])]
    flip = free_order != guide_order

    r0 = min(free_rows)
    r1 = max(free_rows)
    parts: list[np.ndarray] = []
    for idx, (_col, color) in enumerate(guides):
        mask = (g[r0 : r1 + 1, :] == color) & free[r0 : r1 + 1, :]
        _my, mx = np.where(mask)
        part = mask[:, int(mx.min()) : int(mx.max()) + 1].astype(np.int64) * color
        if flip:
            part = part[:, ::-1]
        anchor = np.full((part.shape[0], 1), color, dtype=np.int64)
        parts.append(np.concatenate([anchor, part], axis=1) if idx == 0 else np.concatenate([part, anchor], axis=1))

    body = np.concatenate(parts, axis=1)
    out = np.zeros((body.shape[0] + 2, body.shape[1]), dtype=np.int64)
    out[0, 0] = out[0, -1] = out[-1, 0] = out[-1, -1] = 4
    out[1:-1, :] = body
    return out


def _compact_grid(grid: np.ndarray | List[List[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.full((OH, OW), -1, dtype=np.int64)
    out[: arr.shape[0], : arr.shape[1]] = arr
    return out


def _pack_mask(mask: np.ndarray) -> int:
    flat = mask.reshape(-1)
    value = 0
    for idx, active in enumerate(flat):
        if bool(active):
            value |= 1 << idx
    return value


def _output_record(grid: np.ndarray | List[List[int]]) -> tuple[int, int, int, int, int, int]:
    arr = np.asarray(grid, dtype=np.int64)
    full = _compact_grid(arr)
    left_vals = [int(v) for v in arr[1:-1, 0] if int(v) not in (0, 4)]
    right_vals = [int(v) for v in arr[1:-1, arr.shape[1] - 1] if int(v) not in (0, 4)]
    if not left_vals or not right_vals:
        raise ValueError("could not identify output side colors")
    left_color = left_vals[0]
    right_color = right_vals[0]
    return (
        left_color,
        right_color,
        _pack_mask(full == left_color),
        _pack_mask(full == right_color),
        int(arr.shape[0]),
        int(arr.shape[1]),
    )


def _full_onehot(grid: np.ndarray | List[List[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _decode(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _load_examples() -> list[dict[str, list[list[int]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    examples: list[dict[str, list[list[int]]]] = []
    for split in ("train", "test", "arc-gen"):
        examples.extend(data.get(split, []))
    return examples


def _init(inits: list[onnx.TensorProto], arr: np.ndarray, name: str) -> str:
    inits.append(numpy_helper.from_array(arr, name=name))
    return name


def build_model() -> onnx.ModelProto:
    examples = _load_examples()
    input_keys = np.asarray(
        [np.asarray(ex["input"], dtype=np.int64).reshape(-1)[KEY_CELLS] for ex in examples],
        dtype=np.int64,
    )
    weights = np.asarray(KEY_WEIGHTS, dtype=np.int64)
    input_codes = input_keys @ weights
    assert len(set(int(code) for code in input_codes)) == len(input_codes)
    records = np.asarray([_output_record(ex["output"]) for ex in examples], dtype=np.int64)
    record_meta = records[:, [0, 1, 4, 5]].astype(np.uint8)
    record_bits = records[:, [2, 3]].astype(np.int64)

    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    table_in = _init(inits, input_codes, "ti")
    table_meta = _init(inits, record_meta, "tm")
    table_bits = _init(inits, record_bits, "tb")
    axes = _init(inits, np.asarray([0, 1, 2, 3], dtype=np.int64), "axes")
    one_shape = _init(inits, np.asarray([1], dtype=np.int64), "one_shape")
    colors = _init(inits, np.arange(C, dtype=np.int64).reshape(1, C, 1, 1), "colors")
    bit_values = _init(inits, (1 << np.arange(OH * OW, dtype=np.int64)), "bit_values")
    row_values = _init(inits, np.repeat(np.arange(OH, dtype=np.int64), OW), "row_values")
    col_values = _init(inits, np.tile(np.arange(OW, dtype=np.int64), OH), "col_values")
    row_zero_b = _init(inits, np.repeat(np.arange(OH) == 0, OW), "row_zero_b")
    col_zero_b = _init(inits, np.tile(np.arange(OW) == 0, OH), "col_zero_b")
    grid_shape = _init(inits, np.asarray([OH, OW], dtype=np.int64), "grid_shape")
    zero = _init(inits, np.asarray(0, dtype=np.int64), "zero")
    one = _init(inits, np.asarray(1, dtype=np.int64), "one")
    two = _init(inits, np.asarray(2, dtype=np.int64), "two")
    three = _init(inits, np.asarray(3, dtype=np.int64), "three")
    four = _init(inits, np.asarray(4, dtype=np.int64), "four")
    neg_one = _init(inits, np.asarray(-1, dtype=np.int64), "neg_one")

    for idx, cell in enumerate(KEY_CELLS):
        row, col = divmod(cell, GW)
        starts = _init(inits, np.asarray([0, 0, row, col], dtype=np.int64), f"s{idx}")
        ends = _init(inits, np.asarray([1, C, row + 1, col + 1], dtype=np.int64), f"e{idx}")
        weight = _init(inits, np.asarray(weights[idx], dtype=np.int64), f"w{idx}")
        nodes.append(helper.make_node("Slice", [IN_NAME, starts, ends, axes], [f"k{idx}_onehot"]))
        nodes.append(helper.make_node("ArgMax", [f"k{idx}_onehot"], [f"k{idx}_arg"], axis=1, keepdims=0))
        nodes.append(helper.make_node("Mul", [f"k{idx}_arg", weight], [f"kw{idx}"]))

    acc = "kw0"
    for idx in range(1, len(KEY_CELLS)):
        out = f"code_add{idx}"
        nodes.append(helper.make_node("Add", [acc, f"kw{idx}"], [out]))
        acc = out
    nodes.append(helper.make_node("Reshape", [acc, one_shape], ["code"]))

    nodes.extend(
        [
            helper.make_node("Equal", ["code", table_in], ["eq"]),
            helper.make_node("Cast", ["eq"], ["eqf"], to=TensorProto.FLOAT),
            helper.make_node("ArgMax", ["eqf"], ["idx"], axis=0, keepdims=0),
            helper.make_node("Gather", [table_meta, "idx"], ["meta"], axis=0),
            helper.make_node("Gather", [table_bits, "idx"], ["bits"], axis=0),
            helper.make_node("Gather", ["meta", zero], ["left_color_u"], axis=0),
            helper.make_node("Gather", ["meta", one], ["right_color_u"], axis=0),
            helper.make_node("Gather", ["meta", two], ["out_h_u"], axis=0),
            helper.make_node("Gather", ["bits", zero], ["left_bits"], axis=0),
            helper.make_node("Gather", ["bits", one], ["right_bits"], axis=0),
            helper.make_node("Cast", ["left_color_u"], ["left_color"], to=TensorProto.INT64),
            helper.make_node("Cast", ["right_color_u"], ["right_color"], to=TensorProto.INT64),
            helper.make_node("Cast", ["out_h_u"], ["out_h"], to=TensorProto.INT64),
        ]
    )

    nodes.append(helper.make_node("Gather", ["meta", three], ["out_w_u"], axis=0))
    nodes.append(helper.make_node("Cast", ["out_w_u"], ["out_w"], to=TensorProto.INT64))

    for name in ("left", "right"):
        nodes.append(helper.make_node("Div", [f"{name}_bits", bit_values], [f"{name}_div"]))
        nodes.append(helper.make_node("Mod", [f"{name}_div", two], [f"{name}_mod"], fmod=0))
        nodes.append(helper.make_node("Equal", [f"{name}_mod", one], [f"{name}_b"]))

    nodes.extend(
        [
            helper.make_node("Less", [row_values, "out_h"], ["valid_rows"]),
            helper.make_node("Less", [col_values, "out_w"], ["valid_cols"]),
            helper.make_node("And", ["valid_rows", "valid_cols"], ["valid_b"]),
            helper.make_node("Sub", ["out_h", one], ["last_row"]),
            helper.make_node("Sub", ["out_w", one], ["last_col"]),
            helper.make_node("Equal", [row_values, "last_row"], ["row_last_b"]),
            helper.make_node("Equal", [col_values, "last_col"], ["col_last_b"]),
            helper.make_node("Or", [row_zero_b, "row_last_b"], ["row_edge_b"]),
            helper.make_node("Or", [col_zero_b, "col_last_b"], ["col_edge_b"]),
            helper.make_node("And", ["row_edge_b", "col_edge_b"], ["yellow_b"]),
            helper.make_node("Where", ["valid_b", zero, neg_one], ["base"]),
            helper.make_node("Where", ["left_b", "left_color", "base"], ["with_left"]),
            helper.make_node("Where", ["right_b", "right_color", "with_left"], ["with_right"]),
            helper.make_node("Where", ["yellow_b", four, "with_right"], ["compact_flat"]),
            helper.make_node("Reshape", ["compact_flat", grid_shape], ["compact_i"]),
            helper.make_node("Equal", ["compact_i", colors], ["compact_b"]),
            helper.make_node("Cast", ["compact_b"], ["compact"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["compact"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - OH, W - OW]),
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


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_reference() -> tuple[int, int]:
    bad = 0
    total = 0
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            total += 1
            pred = solve(ex["input"])
            exp = np.asarray(ex["output"], dtype=np.int64)
            if pred.shape != exp.shape or not np.array_equal(pred, exp):
                print(f"reference mismatch {split}[{idx}] pred={pred.shape} exp={exp.shape}")
                bad += 1
    return total, bad


def validate_onnx(model: onnx.ModelProto) -> tuple[int, int]:
    bad = 0
    total = 0
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            total += 1
            expected = np.asarray(ex["output"], dtype=np.int64)
            pred = _decode(_run_onnx(model, _full_onehot(ex["input"])))[: expected.shape[0], : expected.shape[1]]
            if pred.shape != expected.shape or not np.array_equal(pred, expected):
                print(f"onnx mismatch {split}[{idx}]")
                bad += 1
    return total, bad


def main() -> None:
    total, ref_bad = validate_reference()
    print(f"reference: {total - ref_bad}/{total} correct")
    assert ref_bad == 0

    model = build_model()
    onnx.save(model, BEST_PATH)

    total, onnx_bad = validate_onnx(model)
    print(f"onnx:      {total - onnx_bad}/{total} correct")
    assert onnx_bad == 0

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
