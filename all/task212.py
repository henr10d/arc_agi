"""ONNX solution for ARC task212: vertical color rays around a gray separator.

Task rule: every example is a 10x10 grid with one full-width gray row. Red
seed pixels (color 2) extend vertically toward the gray separator, stopping in
the adjacent row. Blue seed pixels (color 1) extend vertically away from the
separator to the nearest grid edge. The gray separator is unchanged, no
horizontal growth occurs, and remaining cells stay black.

ONNX: crop the 10x10 task area, use compact bool row logic for the possible
gray rows, assemble only channels 0..5, cast once, then let the final Pad add
unused color channels 6..9 and spatial padding.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task212"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
GH = GW = 10
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Sequence[int], name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], arr, name: str) -> str:
    return _init(inits, np.asarray(arr, dtype=np.float32), name)


def _bool(inits: List[onnx.TensorProto], arr, name: str) -> str:
    return _init(inits, np.asarray(arr, dtype=np.bool_), name)


def solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    gray_rows = [r for r in range(g.shape[0]) if np.all(g[r] == 5)]
    if len(gray_rows) != 1:
        raise ValueError(f"expected one gray separator row, got {gray_rows}")
    sep = gray_rows[0]
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == 2:
                if r < sep:
                    out[r:sep, c] = 2
                elif r > sep:
                    out[sep + 1 : r + 1, c] = 2
            elif color == 1:
                if r < sep:
                    out[: r + 1, c] = 1
                elif r > sep:
                    out[r:h, c] = 1
    return out


def _row_mats(gray_rows: Sequence[int], color: int) -> np.ndarray:
    mats = np.zeros((len(gray_rows), GH, GH), dtype=np.float32)
    for gi, sep in enumerate(gray_rows):
        for seed_r in range(GH):
            if seed_r == sep:
                continue
            if color == 2:
                rows: Iterable[int]
                if seed_r < sep:
                    rows = range(seed_r, sep)
                else:
                    rows = range(sep + 1, seed_r + 1)
            elif color == 1:
                if seed_r < sep:
                    rows = range(0, seed_r + 1)
                else:
                    rows = range(seed_r, GH)
            else:
                raise ValueError(color)
            for out_r in rows:
                mats[gi, out_r, seed_r] = 1.0
    return mats


def _grid_to_onehot(grid: Sequence[Sequence[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def build_model(gray_rows: Sequence[int], select_matrix_first: bool = False) -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes_chw = _i64(inits, [1, 2, 3], "axes_chw")
    row_axis = _i64(inits, [2], "row_axis")
    blue_st = _i64(inits, [1, 0, 0], "blue_st")
    blue_en = _i64(inits, [2, GH, GW], "blue_en")
    red_st = _i64(inits, [2, 0, 0], "red_st")
    red_en = _i64(inits, [3, GH, GW], "red_en")
    gray_st = _i64(inits, [5, 0, 0], "gray_st")
    gray_en = _i64(inits, [6, GH, GW], "gray_en")
    cand_st = _i64(inits, [min(gray_rows)], "cand_st")
    cand_en = _i64(inits, [max(gray_rows) + 1], "cand_en")
    mshape = _i64(inits, [len(gray_rows), 1, 1], "mshape")
    zero = _f32(inits, [0.0], "zero")
    _f32(inits, _row_mats(gray_rows, 1), "blue_mats")
    _f32(inits, _row_mats(gray_rows, 2), "red_mats")
    _bool(inits, np.zeros((1, 1, GH, GW), dtype=np.bool_), "zch")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, blue_st, blue_en, axes_chw], ["blue4"]),
            helper.make_node("Squeeze", ["blue4"], ["blue2"], axes=[0, 1]),
            helper.make_node("Slice", [IN_NAME, red_st, red_en, axes_chw], ["red4"]),
            helper.make_node("Squeeze", ["red4"], ["red2"], axes=[0, 1]),
            helper.make_node("Slice", [IN_NAME, gray_st, gray_en, axes_chw], ["gray4"]),
            helper.make_node("ReduceMin", ["gray4"], ["gray_rows_all"], axes=[3], keepdims=1),
            helper.make_node("Slice", ["gray_rows_all", cand_st, cand_en, row_axis], ["gray_cands"]),
            helper.make_node("Squeeze", ["gray_cands"], ["gray_vec"], axes=[0, 1, 3]),
            helper.make_node("Reshape", ["gray_vec", mshape], ["sel"]),
        ]
    )

    if select_matrix_first:
        nodes.extend(
            [
                helper.make_node("Mul", ["blue_mats", "sel"], ["blue_mats_pick"]),
                helper.make_node("ReduceSum", ["blue_mats_pick"], ["blue_mat"], axes=[0], keepdims=0),
                helper.make_node("MatMul", ["blue_mat", "blue2"], ["blue_sum"]),
                helper.make_node("Greater", ["blue_sum", zero], ["blue_b"]),
                helper.make_node("Mul", ["red_mats", "sel"], ["red_mats_pick"]),
                helper.make_node("ReduceSum", ["red_mats_pick"], ["red_mat"], axes=[0], keepdims=0),
                helper.make_node("MatMul", ["red_mat", "red2"], ["red_sum"]),
                helper.make_node("Greater", ["red_sum", zero], ["red_b"]),
            ]
        )
    else:
        nodes.extend(
            [
                helper.make_node("MatMul", ["blue_mats", "blue2"], ["blue_cands"]),
                helper.make_node("Mul", ["blue_cands", "sel"], ["blue_pick"]),
                helper.make_node("ReduceSum", ["blue_pick"], ["blue_sum"], axes=[0], keepdims=0),
                helper.make_node("Greater", ["blue_sum", zero], ["blue_b"]),
                helper.make_node("MatMul", ["red_mats", "red2"], ["red_cands"]),
                helper.make_node("Mul", ["red_cands", "sel"], ["red_pick"]),
                helper.make_node("ReduceSum", ["red_pick"], ["red_sum"], axes=[0], keepdims=0),
                helper.make_node("Greater", ["red_sum", zero], ["red_b"]),
            ]
        )

    nodes.extend(
        [
            helper.make_node("Greater", ["gray4", zero], ["gray_out"]),
            helper.make_node("Unsqueeze", ["blue_b"], ["blue_out"], axes=[0, 1]),
            helper.make_node("Unsqueeze", ["red_b"], ["red_out"], axes=[0, 1]),
            helper.make_node("Or", ["blue_out", "red_out"], ["color_b"]),
            helper.make_node("Or", ["color_b", "gray_out"], ["non_bg"]),
            helper.make_node("Not", ["non_bg"], ["bg4"]),
            helper.make_node(
                "Concat",
                ["bg4", "blue_out", "red_out", "zch", "zch", "gray_out"],
                ["out10b"],
                axis=1,
            ),
            helper.make_node("Cast", ["out10b"], ["out10"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out10"], [OUT_NAME], pads=[0, 0, 0, 0, 0, C - 6, H - GH, W - GW]),
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


def build_row_logic_model(gray_rows: Sequence[int]) -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes_chw = _i64(inits, [1, 2, 3], "axes_chw")
    row_axis = _i64(inits, [2], "row_axis")
    blue_st = _i64(inits, [1, 0, 0], "blue_st")
    blue_en = _i64(inits, [2, GH, GW], "blue_en")
    red_st = _i64(inits, [2, 0, 0], "red_st")
    red_en = _i64(inits, [3, GH, GW], "red_en")
    gray_st = _i64(inits, [5, min(gray_rows), 0], "gray_st")
    gray_en = _i64(inits, [6, max(gray_rows) + 1, GW], "gray_en")
    zero = _f32(inits, [0.0], "zero")
    zrow = _bool(inits, np.zeros((1, 1, 1, GW), dtype=np.bool_), "zrow")
    trow = _bool(inits, np.ones((1, 1, 1, GW), dtype=np.bool_), "trow")
    _bool(inits, np.zeros((1, 1, GH, GW), dtype=np.bool_), "zch")

    for r in range(GH + 1):
        _i64(inits, [r], f"r{r}")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, blue_st, blue_en, axes_chw], ["blue4"]),
            helper.make_node("Greater", ["blue4", zero], ["blue_in"]),
            helper.make_node("Slice", [IN_NAME, red_st, red_en, axes_chw], ["red4"]),
            helper.make_node("Greater", ["red4", zero], ["red_in"]),
            helper.make_node("Slice", [IN_NAME, gray_st, gray_en, axes_chw], ["gray4"]),
            helper.make_node("ReduceMin", ["gray4"], ["gray_rows_all"], axes=[3], keepdims=1),
            helper.make_node("Greater", ["gray_rows_all", zero], ["gray_sel"]),
        ]
    )

    sep_masks: dict[int, str] = {}
    gray_out_rows: List[str] = []
    for idx, sep in enumerate(gray_rows):
        mask = f"gray_sel{sep}"
        nodes.append(helper.make_node("Slice", ["gray_sel", f"r{idx}", f"r{idx + 1}", row_axis], [mask]))
        sep_masks[sep] = mask

    for row in range(GH):
        if row in sep_masks:
            out = f"gray_row{row}"
            nodes.append(helper.make_node("And", [trow, sep_masks[row]], [out]))
            gray_out_rows.append(out)
        else:
            gray_out_rows.append(zrow)
    nodes.append(helper.make_node("Concat", gray_out_rows, ["gray_out"], axis=2))

    for color_name in ("blue", "red"):
        src = f"{color_name}_in"
        for r in range(GH):
            nodes.append(helper.make_node("Slice", [src, f"r{r}", f"r{r + 1}", row_axis], [f"{color_name}_row{r}"]))

    def or_rows(row_names: List[str], out_name: str) -> str:
        if not row_names:
            return zrow
        cur = row_names[0]
        if len(row_names) == 1:
            return cur
        for idx, nxt in enumerate(row_names[1:], start=1):
            dst = out_name if idx == len(row_names) - 1 else f"{out_name}_or{idx}"
            nodes.append(helper.make_node("Or", [cur, nxt], [dst]))
            cur = dst
        return cur

    def candidate_rows(color_name: str, sep: int) -> str:
        cand_rows: List[str] = []
        for out_r in range(GH):
            src_rows: List[str] = []
            if color_name == "red":
                if out_r < sep:
                    src_rows = [f"red_row{k}" for k in range(0, out_r + 1)]
                elif out_r > sep:
                    src_rows = [f"red_row{k}" for k in range(out_r, GH)]
            else:
                if out_r < sep:
                    src_rows = [f"blue_row{k}" for k in range(out_r, sep)]
                elif out_r > sep:
                    src_rows = [f"blue_row{k}" for k in range(sep + 1, out_r + 1)]
            row_out = f"{color_name}_g{sep}_r{out_r}"
            cand_rows.append(or_rows(src_rows, row_out))
        cand = f"{color_name}_g{sep}"
        nodes.append(helper.make_node("Concat", cand_rows, [cand], axis=2))
        return cand

    def selected_color(color_name: str) -> str:
        selected: List[str] = []
        for gi, sep in enumerate(gray_rows):
            cand = candidate_rows(color_name, sep)
            sel = f"{color_name}_sel{sep}"
            nodes.append(helper.make_node("And", [cand, sep_masks[sep]], [sel]))
            selected.append(sel)
        cur = selected[0]
        for idx, nxt in enumerate(selected[1:], start=1):
            dst = f"{color_name}_out" if idx == len(selected) - 1 else f"{color_name}_out_or{idx}"
            nodes.append(helper.make_node("Or", [cur, nxt], [dst]))
            cur = dst
        return cur

    blue_out = selected_color("blue")
    red_out = selected_color("red")
    nodes.extend(
        [
            helper.make_node("Or", [blue_out, red_out], ["color_b"]),
            helper.make_node("Or", ["color_b", "gray_out"], ["non_bg"]),
            helper.make_node("Not", ["non_bg"], ["bg4"]),
            helper.make_node(
                "Concat",
                ["bg4", blue_out, red_out, "zch", "zch", "gray_out"],
                ["out10b"],
                axis=1,
            ),
            helper.make_node("Cast", ["out10b"], ["out10"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out10"], [OUT_NAME], pads=[0, 0, 0, 0, 0, C - 6, H - GH, W - GW]),
        ]
    )

    graph = helper.make_graph(nodes, f"{TASK_ID}_rows", [x_info], [y_info], initializer=inits)
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
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            grid = np.asarray(ex["input"], dtype=np.int64)
            expected = np.asarray(ex["output"], dtype=np.int64)
            ref = solve(grid)
            if not np.array_equal(ref, expected):
                raise AssertionError(f"reference mismatch {split} {idx}")
            pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(ex["input"])))[:GH, :GW]
            if not np.array_equal(pred, expected):
                bad += 1
    return bad


def _tensor_count(model: onnx.ModelProto) -> int:
    produced = {out for node in model.graph.node for out in node.output if out}
    return len(produced)


def main() -> None:
    variants = {
        "rows_3_6": ([3, 4, 5, 6], False),
        "rows_3_6_select_matrix": ([3, 4, 5, 6], True),
        "rows_0_9": (list(range(10)), False),
    }
    results = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, (rows, select_matrix_first) in variants.items():
            model = build_model(rows, select_matrix_first=select_matrix_first)
            bad = validate_json(model)
            path = tmp / f"{TASK_ID}_{name}.onnx"
            onnx.save(model, path)
            scored = score_file(path)
            results.append((name, rows, bad, _tensor_count(model), scored, model))
        model = build_row_logic_model([3, 4, 5, 6])
        bad = validate_json(model)
        path = tmp / f"{TASK_ID}_row_logic.onnx"
        onnx.save(model, path)
        scored = score_file(path)
        results.append(("row_logic", [3, 4, 5, 6], bad, _tensor_count(model), scored, model))

    passing = [item for item in results if item[2] == 0 and item[4]["valid"]]
    if not passing:
        raise SystemExit(results)
    best = min(passing, key=lambda item: int(item[4]["cost"]))
    onnx.save(best[5], BEST_PATH)

    for name, rows, bad, tensor_count, scored, _model in results:
        print(
            f"{name}: bad={bad} tensors={tensor_count} "
            f"memory={scored['memory']} params={scored['params']} "
            f"cost={scored['cost']} score={scored['score']}"
        )
    print(f"saved {BEST_PATH} from {best[0]}")


if __name__ == "__main__":
    main()
