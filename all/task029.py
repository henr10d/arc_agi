"""ONNX for ARC task029: crop the contents inside a monochrome rectangle.

Task rule: the input is random colored noise containing exactly one rectangular
frame made from a single non-zero color. The output is the frame interior,
cropped to the top-left of the competition output tensor; the frame border and
all surrounding noise are discarded. The frame color, size, and position vary.

ONNX: find length-3-or-longer run starts/ends in each non-background channel.
The true frame has four cells that are simultaneously horizontal and vertical
run corners; choose the channel with the most such corners, reduce those corner
coordinates to y0/y1/x0/x1, then crop the one-hot input with static row/column
selection masks so the graph remains valid at opset 10.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import calculate_params, score_file  # noqa: E402

TASK_ID = "task029"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
NC = 9
H = W = 30
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _bool(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.bool_), name)


def _slice(
    nodes: List[onnx.NodeProto],
    data: str,
    out: str,
    starts: str,
    ends: str,
    axes: str,
) -> None:
    nodes.append(helper.make_node("Slice", [data, starts, ends, axes], [out]))


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver used for script validation."""
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    for color in range(1, 10):
        m = g == color
        for y0 in range(h - 2):
            for y1 in range(y0 + 2, h):
                for x0 in range(w - 2):
                    for x1 in range(x0 + 2, w):
                        if (
                            m[y0, x0 : x1 + 1].all()
                            and m[y1, x0 : x1 + 1].all()
                            and m[y0 : y1 + 1, x0].all()
                            and m[y0 : y1 + 1, x1].all()
                        ):
                            return g[y0 + 1 : y1, x0 + 1 : x1]
    raise ValueError("no frame found")


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

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    half = _f32(inits, [0.5], "half")
    big = _f32(inits, [1000.0], "big")
    one = _f32(inits, [1.0], "one")

    st_h0 = _i64(inits, [0, 0, 0, 0], "st_h0")
    en_h0 = _i64(inits, [1, NC, H, W - 2], "en_h0")
    st_h1 = _i64(inits, [0, 0, 0, 1], "st_h1")
    en_h1 = _i64(inits, [1, NC, H, W - 1], "en_h1")
    st_h2 = _i64(inits, [0, 0, 0, 2], "st_h2")
    en_h2 = _i64(inits, [1, NC, H, W], "en_h2")
    st_hp = _i64(inits, [0, 0, 0, 0], "st_hp")
    en_hp = _i64(inits, [1, NC, H, W - 3], "en_hp")
    st_hn = _i64(inits, [0, 0, 0, 3], "st_hn")
    en_hn = _i64(inits, [1, NC, H, W], "en_hn")

    st_v0 = _i64(inits, [0, 0, 0, 0], "st_v0")
    en_v0 = _i64(inits, [1, NC, H - 2, W], "en_v0")
    st_v1 = _i64(inits, [0, 0, 1, 0], "st_v1")
    en_v1 = _i64(inits, [1, NC, H - 1, W], "en_v1")
    st_v2 = _i64(inits, [0, 0, 2, 0], "st_v2")
    en_v2 = _i64(inits, [1, NC, H, W], "en_v2")
    st_vp = _i64(inits, [0, 0, 0, 0], "st_vp")
    en_vp = _i64(inits, [1, NC, H - 3, W], "en_vp")
    st_vn = _i64(inits, [0, 0, 3, 0], "st_vn")
    en_vn = _i64(inits, [1, NC, H, W], "en_vn")

    t_hcol = _bool(inits, np.ones((1, NC, H, 1), dtype=np.bool_), "t_hcol")
    f_h2 = _bool(inits, np.zeros((1, NC, H, 2), dtype=np.bool_), "f_h2")
    t_vrow = _bool(inits, np.ones((1, NC, 1, W), dtype=np.bool_), "t_vrow")
    f_v2 = _bool(inits, np.zeros((1, NC, 2, W), dtype=np.bool_), "f_v2")

    y_coord = _f32(inits, np.arange(H, dtype=np.float32).reshape(1, 1, H, 1), "y_coord")
    x_coord = _f32(inits, np.arange(W, dtype=np.float32).reshape(1, 1, 1, W), "x_coord")
    rows = _f32(inits, np.arange(H, dtype=np.float32).reshape(1, H, 1), "rows")
    cols = _f32(inits, np.arange(W, dtype=np.float32).reshape(1, 1, W), "cols")
    ch_idx = _i64(inits, np.arange(NC, dtype=np.int64).reshape(1, NC, 1, 1), "ch_idx")
    colors_nz_i = _i64(inits, np.arange(1, C, dtype=np.int64).reshape(1, NC, 1, 1), "colors_nz_i")
    src_rows5 = _i64(inits, np.arange(H, dtype=np.int64).reshape(1, 1, 1, H, 1), "src_rows5")
    dst_rows5 = _i64(inits, np.arange(H, dtype=np.int64).reshape(1, 1, H, 1, 1), "dst_rows5")
    src_cols5 = _i64(inits, np.arange(W, dtype=np.int64).reshape(1, 1, 1, 1, W), "src_cols5")
    dst_cols5 = _i64(inits, np.arange(W, dtype=np.int64).reshape(1, 1, 1, W, 1), "dst_cols5")

    nodes.extend(
        [
            helper.make_node("ArgMax", [IN_NAME], ["color_grid"], axis=1, keepdims=0),
            helper.make_node("Unsqueeze", ["color_grid"], ["color_grid_ch"], axes=[1]),
            helper.make_node("Equal", ["color_grid_ch", colors_nz_i], ["m"]),
        ]
    )

    _slice(nodes, "m", "h0", st_h0, en_h0, axes4)
    _slice(nodes, "m", "h1", st_h1, en_h1, axes4)
    _slice(nodes, "m", "h2", st_h2, en_h2, axes4)
    _slice(nodes, "m", "hp", st_hp, en_hp, axes4)
    _slice(nodes, "m", "hn", st_hn, en_hn, axes4)
    _slice(nodes, "m", "v0", st_v0, en_v0, axes4)
    _slice(nodes, "m", "v1", st_v1, en_v1, axes4)
    _slice(nodes, "m", "v2", st_v2, en_v2, axes4)
    _slice(nodes, "m", "vp", st_vp, en_vp, axes4)
    _slice(nodes, "m", "vn", st_vn, en_vn, axes4)

    nodes.extend(
        [
            helper.make_node("And", ["h0", "h1"], ["h01"]),
            helper.make_node("And", ["h01", "h2"], ["h3"]),
            helper.make_node("Not", ["hp"], ["nhp"]),
            helper.make_node("Concat", [t_hcol, "nhp"], ["hprev"], axis=3),
            helper.make_node("And", ["h3", "hprev"], ["hs28"]),
            helper.make_node("Concat", ["hs28", f_h2], ["hs"], axis=3),
            helper.make_node("Not", ["hn"], ["nhn"]),
            helper.make_node("Concat", ["nhn", t_hcol], ["hnext"], axis=3),
            helper.make_node("And", ["h3", "hnext"], ["he28"]),
            helper.make_node("Concat", [f_h2, "he28"], ["he"], axis=3),
            helper.make_node("And", ["v0", "v1"], ["v01"]),
            helper.make_node("And", ["v01", "v2"], ["v3"]),
            helper.make_node("Not", ["vp"], ["nvp"]),
            helper.make_node("Concat", [t_vrow, "nvp"], ["vprev"], axis=2),
            helper.make_node("And", ["v3", "vprev"], ["vs28"]),
            helper.make_node("Concat", ["vs28", f_v2], ["vs"], axis=2),
            helper.make_node("Not", ["vn"], ["nvn"]),
            helper.make_node("Concat", ["nvn", t_vrow], ["vnext"], axis=2),
            helper.make_node("And", ["v3", "vnext"], ["ve28"]),
            helper.make_node("Concat", [f_v2, "ve28"], ["ve"], axis=2),
            helper.make_node("And", ["hs", "vs"], ["tl"]),
            helper.make_node("And", ["he", "vs"], ["tr"]),
            helper.make_node("And", ["hs", "ve"], ["bl"]),
            helper.make_node("And", ["he", "ve"], ["br"]),
            helper.make_node("Or", ["tl", "tr"], ["ct"]),
            helper.make_node("Or", ["bl", "br"], ["cb"]),
            helper.make_node("Or", ["ct", "cb"], ["corner"]),
            helper.make_node("Cast", ["corner"], ["corner_f"], to=TensorProto.FLOAT),
            helper.make_node("ReduceSum", ["corner_f"], ["score"], axes=[2, 3], keepdims=1),
            helper.make_node("ArgMax", ["score"], ["sel_ch"], axis=1, keepdims=1),
            helper.make_node("Equal", ["ch_idx", "sel_ch"], ["sel"]),
            helper.make_node("And", ["corner", "sel"], ["sel_corner"]),
            helper.make_node("Cast", ["sel_corner"], ["sc9_f"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", ["sc9_f"], ["sc_f"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["sc_f", half], ["sc1"]),
            helper.make_node("Where", ["sc1", y_coord, big], ["y_min_grid"]),
            helper.make_node("Where", ["sc1", x_coord, big], ["x_min_grid"]),
            helper.make_node("Mul", ["sc_f", y_coord], ["y_max_grid"]),
            helper.make_node("Mul", ["sc_f", x_coord], ["x_max_grid"]),
            helper.make_node("ReduceMin", ["y_min_grid"], ["y0"], axes=[1, 2, 3], keepdims=0),
            helper.make_node("ReduceMin", ["x_min_grid"], ["x0"], axes=[1, 2, 3], keepdims=0),
            helper.make_node("ReduceMax", ["y_max_grid"], ["y1"], axes=[1, 2, 3], keepdims=0),
            helper.make_node("ReduceMax", ["x_max_grid"], ["x1"], axes=[1, 2, 3], keepdims=0),
            helper.make_node("Sub", ["y1", "y0"], ["fh1"]),
            helper.make_node("Sub", ["x1", "x0"], ["fw1"]),
            helper.make_node("Sub", ["fh1", one], ["out_h"]),
            helper.make_node("Sub", ["fw1", one], ["out_w"]),
            helper.make_node("Less", ["rows", "out_h"], ["vy"]),
            helper.make_node("Less", ["cols", "out_w"], ["vx"]),
            helper.make_node("And", ["vy", "vx"], ["valid"]),
            helper.make_node("Add", ["y0", one], ["y_start"]),
            helper.make_node("Add", ["x0", one], ["x_start"]),
            helper.make_node("Add", ["rows", "y_start"], ["sy_f"]),
            helper.make_node("Add", ["cols", "x_start"], ["sx_f"]),
            helper.make_node("Cast", ["y_start"], ["y_start_i"], to=TensorProto.INT64),
            helper.make_node("Cast", ["x_start"], ["x_start_i"], to=TensorProto.INT64),
            helper.make_node("Add", ["dst_rows5", "y_start_i"], ["target_rows5"]),
            helper.make_node("Equal", ["target_rows5", "src_rows5"], ["row_mask"]),
            helper.make_node("Cast", ["row_mask"], ["row_mask_f"], to=TensorProto.FLOAT),
            helper.make_node("Unsqueeze", [IN_NAME], ["input_row5"], axes=[2]),
            helper.make_node("Mul", ["input_row5", "row_mask_f"], ["row_prod"]),
            helper.make_node("ReduceSum", ["row_prod"], ["row_shifted"], axes=[3], keepdims=0),
            helper.make_node("Add", ["dst_cols5", "x_start_i"], ["target_cols5"]),
            helper.make_node("Equal", ["target_cols5", "src_cols5"], ["col_mask"]),
            helper.make_node("Cast", ["col_mask"], ["col_mask_f"], to=TensorProto.FLOAT),
            helper.make_node("Unsqueeze", ["row_shifted"], ["row_shifted5"], axes=[3]),
            helper.make_node("Mul", ["row_shifted5", "col_mask_f"], ["col_prod"]),
            helper.make_node("ReduceSum", ["col_prod"], ["cropped"], axes=[4], keepdims=0),
            helper.make_node("Unsqueeze", ["valid"], ["valid_ch"], axes=[1]),
            helper.make_node("Cast", ["valid_ch"], ["valid_f"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["cropped", "valid_f"], [OUT_NAME]),
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


def validate_json(model: onnx.ModelProto, splits: tuple[str, ...] = ("train",)) -> int:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    total = 0
    for split in splits:
        for idx, ex in enumerate(data.get(split, [])):
            g = np.array(ex["input"], dtype=np.int64)
            if g.shape[0] > H or g.shape[1] > W:
                continue
            expected = np.array(ex["output"], dtype=np.int64)
            ref = solve(g)
            if not np.array_equal(ref, expected):
                raise AssertionError(f"reference mismatch on {split}[{idx}]")
            out = sess.run([OUT_NAME], {IN_NAME: _grid_to_onehot(ex["input"])})[0]
            pred = _onehot_to_grid(out)
            got = pred[: expected.shape[0], : expected.shape[1]]
            pad_ok = not np.any(out[:, :, expected.shape[0] :, :])
            pad_ok = pad_ok and not np.any(out[:, :, :, expected.shape[1] :])
            total += 1
            if not np.array_equal(got, expected) or not pad_ok:
                bad += 1
                print(f"mismatch {split}[{idx}] expected {expected.shape} got crop:\n{got}")
    print(f"validated {total - bad}/{total} examples for {', '.join(splits)}")
    return bad


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    bad_train = validate_json(model, ("train",))
    bad_all = validate_json(model, ("train", "test", "arc-gen"))
    if bad_train:
        raise SystemExit(f"{bad_train} train examples failed")
    if bad_all:
        raise SystemExit(f"{bad_all} total examples failed")

    params = calculate_params(model)
    print(f"raw parameter count before sanitizer: {params}")
    result = score_file(BEST_PATH)
    print(f"generated: {BEST_PATH}")
    print(
        "score attempt 1 (dynamic Slice+Pad) was rejected conceptually: "
        "shape inference cannot prove a static 30x30 output after dynamic Slice."
    )
    print(
        "score attempt 2 (GatherND from 10-channel one-hot) avoids dynamic shapes "
        "but creates a 1x10x30x30x4 index tensor, making index memory dominant."
    )
    print(
        "score attempt 3 used GatherND from an ArgMax color grid, but GatherND "
        "requires a newer opset than Kaggle accepts for NeuroGolf."
    )
    print(
        "score attempt 4 (kept) uses opset-10 static row/column masks to crop "
        "the one-hot input. It is more memory-heavy but submission-valid."
    )
    print(
        "likely memory-heavy intermediates: the bool run/corner tensors "
        "(1x9x30x30), selected coordinate grids, and the row/column broadcast "
        "products used for opset-10-compatible cropping."
    )
    print(
        f"official-style score: valid={result['valid']} memory={result['memory']} "
        f"params={result['params']} cost={result['cost']} score={result['score']}"
    )


if __name__ == "__main__":
    main()
