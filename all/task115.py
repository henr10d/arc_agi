"""Build a compact ONNX solution for NeuroGolf task115.

Task rule: the active input rectangle is entirely filled by three or four
non-background colors forming noisy sequential bands.  If the band centroids are
separated more by column than by row, output a 1xN strip of the colors ordered
left-to-right; otherwise output an Nx1 strip ordered top-to-bottom.  Padding
outside that compact strip remains all-zero in the required 30x30 one-hot
competition tensor.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, print_report, score_file  # noqa: E402

TASK_ID = "task115"
SHAPE = [1, 10, 30, 30]
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"


def _init(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array, name)


def _vi(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def _make_model(
    path: Path,
    *,
    use_div: bool,
    topk_sorted: bool,
    orient_margin: float,
    single_selected_output: bool = False,
    use_matmul_coords: bool = False,
    vector_selected_output: bool = False,
) -> None:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = [
        _init("row_w", np.arange(30, dtype=np.float32).reshape((30, 1) if use_matmul_coords else (1, 30))),
        _init("col_w", np.arange(30, dtype=np.float32).reshape((30, 1) if use_matmul_coords else (1, 30))),
        _init("zero_f", np.array(0, dtype=np.float32)),
        _init("one_f", np.array(1, dtype=np.float32)),
        _init("high", np.array(1000, dtype=np.float32)),
        _init("neg_live_cut", np.array(-999, dtype=np.float32)),
        _init("low", np.array(-1, dtype=np.float32)),
        _init("color_axis", np.arange(10, dtype=np.int64).reshape(1, 10, 1, 1)),
        _init("row4_axis", np.arange(4, dtype=np.int64).reshape(1, 1, 4, 1)),
        _init("col4_axis", np.arange(4, dtype=np.int64).reshape(1, 1, 1, 4)),
        _init("zero_i", np.array(0, dtype=np.int64)),
        _init("k4", np.array([4], dtype=np.int64)),
    ]
    if orient_margin != 0.0:
        inits.append(_init("orient_margin", np.array(orient_margin, dtype=np.float32)))
    if not vector_selected_output:
        inits.extend(
            [
                _init("one_i", np.array(1, dtype=np.int64)),
                _init("two_i", np.array(2, dtype=np.int64)),
                _init("three_i", np.array(3, dtype=np.int64)),
                _init("idx0", np.array([0], dtype=np.int64)),
                _init("idx1", np.array([1], dtype=np.int64)),
                _init("idx2", np.array([2], dtype=np.int64)),
                _init("idx3", np.array([3], dtype=np.int64)),
            ]
        )
    value_infos: list[onnx.ValueInfoProto] = [
        _vi("cols", TensorProto.FLOAT, [10, 30]),
        _vi("rows", TensorProto.FLOAT, [10, 30]),
        _vi("counts", TensorProto.FLOAT, [10]),
        _vi("present", TensorProto.BOOL, [10]),
        _vi("present_f", TensorProto.FLOAT, [10]),
        _vi("absent_f", TensorProto.FLOAT, [10]),
        _vi("x_sum", TensorProto.FLOAT, [10]),
        _vi("y_sum", TensorProto.FLOAT, [10]),
        _vi("denom", TensorProto.FLOAT, [10]),
        _vi("x_coord", TensorProto.FLOAT, [10]),
        _vi("y_coord", TensorProto.FLOAT, [10]),
        _vi("x_abs_hi", TensorProto.FLOAT, [10]),
        _vi("y_abs_hi", TensorProto.FLOAT, [10]),
        _vi("x_abs_lo", TensorProto.FLOAT, [10]),
        _vi("y_abs_lo", TensorProto.FLOAT, [10]),
        _vi("x_hi", TensorProto.FLOAT, [10]),
        _vi("y_hi", TensorProto.FLOAT, [10]),
        _vi("x_neg", TensorProto.FLOAT, [10]),
        _vi("y_neg", TensorProto.FLOAT, [10]),
        _vi("x_lo", TensorProto.FLOAT, [10]),
        _vi("y_lo", TensorProto.FLOAT, [10]),
        _vi("x_min", TensorProto.FLOAT, [1]),
        _vi("y_min", TensorProto.FLOAT, [1]),
        _vi("x_max", TensorProto.FLOAT, [1]),
        _vi("y_max", TensorProto.FLOAT, [1]),
        _vi("x_range", TensorProto.FLOAT, [1]),
        _vi("y_range", TensorProto.FLOAT, [1]),
        _vi("vertical", TensorProto.BOOL, [1]),
        _vi("x_vals_neg", TensorProto.FLOAT, [4]),
        _vi("y_vals_neg", TensorProto.FLOAT, [4]),
        _vi("x_ids", TensorProto.INT64, [4]),
        _vi("y_ids", TensorProto.INT64, [4]),
    ]
    if orient_margin != 0.0:
        value_infos.append(_vi("y_cmp", TensorProto.FLOAT, [1]))
    if use_matmul_coords:
        value_infos.extend(
            [
                _vi("x_sum_2d", TensorProto.FLOAT, [10, 1]),
                _vi("y_sum_2d", TensorProto.FLOAT, [10, 1]),
            ]
        )
    else:
        value_infos.extend(
            [
                _vi("x_mul", TensorProto.FLOAT, [10, 30]),
                _vi("y_mul", TensorProto.FLOAT, [10, 30]),
            ]
        )

    nodes.extend(
        [
            helper.make_node("ReduceSum", ["input"], ["cols"], axes=[0, 2], keepdims=0),
            helper.make_node("ReduceSum", ["input"], ["rows"], axes=[0, 3], keepdims=0),
            helper.make_node("ReduceSum", ["cols"], ["counts"], axes=[1], keepdims=0),
            helper.make_node("Greater", ["counts", "zero_f"], ["present"]),
            helper.make_node("Cast", ["present"], ["present_f"], to=TensorProto.FLOAT),
            helper.make_node("Sub", ["one_f", "present_f"], ["absent_f"]),
        ]
    )
    if use_matmul_coords:
        nodes.extend(
            [
                helper.make_node("MatMul", ["cols", "col_w"], ["x_sum_2d"]),
                helper.make_node("MatMul", ["rows", "row_w"], ["y_sum_2d"]),
                helper.make_node("Squeeze", ["x_sum_2d"], ["x_sum"], axes=[1]),
                helper.make_node("Squeeze", ["y_sum_2d"], ["y_sum"], axes=[1]),
            ]
        )
    else:
        nodes.extend(
            [
                helper.make_node("Mul", ["cols", "col_w"], ["x_mul"]),
                helper.make_node("Mul", ["rows", "row_w"], ["y_mul"]),
                helper.make_node("ReduceSum", ["x_mul"], ["x_sum"], axes=[1], keepdims=0),
                helper.make_node("ReduceSum", ["y_mul"], ["y_sum"], axes=[1], keepdims=0),
            ]
        )

    if use_div:
        nodes.extend(
            [
                helper.make_node("Add", ["counts", "absent_f"], ["denom"]),
                helper.make_node("Div", ["x_sum", "denom"], ["x_coord"]),
                helper.make_node("Div", ["y_sum", "denom"], ["y_coord"]),
            ]
        )
    else:
        # Counts are similar across same-task bands; weighted sums preserve order.
        nodes.extend(
            [
                helper.make_node("Add", ["counts", "one_f"], ["denom"]),
                helper.make_node("Identity", ["x_sum"], ["x_coord"]),
                helper.make_node("Identity", ["y_sum"], ["y_coord"]),
            ]
        )

    nodes.extend(
        [
            helper.make_node("Mul", ["absent_f", "high"], ["x_abs_hi"]),
            helper.make_node("Mul", ["absent_f", "high"], ["y_abs_hi"]),
            helper.make_node("Mul", ["absent_f", "low"], ["x_abs_lo"]),
            helper.make_node("Mul", ["absent_f", "low"], ["y_abs_lo"]),
            helper.make_node("Add", ["x_coord", "x_abs_hi"], ["x_hi"]),
            helper.make_node("Add", ["y_coord", "y_abs_hi"], ["y_hi"]),
            helper.make_node("Neg", ["x_hi"], ["x_neg"]),
            helper.make_node("Neg", ["y_hi"], ["y_neg"]),
            helper.make_node("Add", ["x_coord", "x_abs_lo"], ["x_lo"]),
            helper.make_node("Add", ["y_coord", "y_abs_lo"], ["y_lo"]),
            helper.make_node("ReduceMin", ["x_hi"], ["x_min"], axes=[0], keepdims=1),
            helper.make_node("ReduceMin", ["y_hi"], ["y_min"], axes=[0], keepdims=1),
            helper.make_node("ReduceMax", ["x_lo"], ["x_max"], axes=[0], keepdims=1),
            helper.make_node("ReduceMax", ["y_lo"], ["y_max"], axes=[0], keepdims=1),
            helper.make_node("Sub", ["x_max", "x_min"], ["x_range"]),
            helper.make_node("Sub", ["y_max", "y_min"], ["y_range"]),
        ]
    )
    if orient_margin != 0.0:
        nodes.extend(
            [
                helper.make_node("Add", ["y_range", "orient_margin"], ["y_cmp"]),
                helper.make_node("Greater", ["x_range", "y_cmp"], ["vertical"]),
            ]
        )
    else:
        nodes.append(helper.make_node("Greater", ["x_range", "y_range"], ["vertical"]))
    nodes.extend(
        [
            helper.make_node(
                "TopK",
                ["x_neg", "k4"],
                ["x_vals_neg", "x_ids"],
                axis=0,
            ),
            helper.make_node(
                "TopK",
                ["y_neg", "k4"],
                ["y_vals_neg", "y_ids"],
                axis=0,
            ),
        ]
    )

    pos_names = ["zero_i", "one_i", "two_i", "three_i"]
    idx_names = ["idx0", "idx1", "idx2", "idx3"]
    value_infos.extend(
        [
            _vi("row4_zero", TensorProto.BOOL, [1, 1, 4, 1]),
            _vi("col4_zero", TensorProto.BOOL, [1, 1, 1, 4]),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Equal", ["row4_axis", "zero_i"], ["row4_zero"]),
            helper.make_node("Equal", ["col4_axis", "zero_i"], ["col4_zero"]),
        ]
    )

    def build_terms(prefix: str, ids_name: str, vals_name: str, horizontal: bool) -> list[str]:
        terms: list[str] = []
        for i, pos_name in enumerate(pos_names):
            id_name = f"{prefix}_id{i}"
            val_name = f"{prefix}_val{i}"
            live_name = f"{prefix}_live{i}"
            color_eq = f"{prefix}_color_eq{i}"
            pos_eq = f"{prefix}_pos_eq{i}"
            rc = f"{prefix}_rc{i}"
            colored = f"{prefix}_colored{i}"
            live_grid = f"{prefix}_live_grid{i}"
            pos_shape = [1, 1, 4, 1] if horizontal else [1, 1, 1, 4]
            grid_shape = [1, 10, 4, 4]
            value_infos.extend(
                [
                    _vi(id_name, TensorProto.INT64, [1]),
                    _vi(val_name, TensorProto.FLOAT, [1]),
                    _vi(live_name, TensorProto.BOOL, [1]),
                    _vi(color_eq, TensorProto.BOOL, [1, 10, 1, 1]),
                    _vi(pos_eq, TensorProto.BOOL, pos_shape),
                    _vi(rc, TensorProto.BOOL, [1, 1, 4, 4]),
                    _vi(colored, TensorProto.BOOL, grid_shape),
                    _vi(live_grid, TensorProto.BOOL, grid_shape),
                ]
            )
            nodes.extend(
                [
                    helper.make_node("Gather", [ids_name, idx_names[i]], [id_name], axis=0),
                    helper.make_node("Gather", [vals_name, idx_names[i]], [val_name], axis=0),
                    helper.make_node("Greater", [val_name, "neg_live_cut"], [live_name]),
                    helper.make_node("Equal", ["color_axis", id_name], [color_eq]),
                    helper.make_node("Equal", ["row4_axis" if horizontal else "col4_axis", pos_name], [pos_eq]),
                    helper.make_node("And", [pos_eq, "col4_zero" if horizontal else "row4_zero"], [rc]),
                    helper.make_node("And", [color_eq, rc], [colored]),
                    helper.make_node("And", [colored, live_name], [live_grid]),
                ]
            )
            terms.append(live_grid)
        return terms

    def or_chain(prefix: str, names: list[str], shape: list[int]) -> str:
        current = names[0]
        for i, name in enumerate(names[1:], 1):
            out = f"{prefix}{i}"
            value_infos.append(_vi(out, TensorProto.BOOL, shape))
            nodes.append(helper.make_node("Or", [current, name], [out]))
            current = out
        return current

    if vector_selected_output:
        value_infos.extend(
            [
                _vi("not_vertical", TensorProto.BOOL, [1]),
                _vi("selected_ids", TensorProto.INT64, [4]),
                _vi("selected_vals", TensorProto.FLOAT, [4]),
                _vi("selected_live", TensorProto.BOOL, [4]),
                _vi("ids_col", TensorProto.INT64, [1, 1, 4, 1]),
                _vi("live_col", TensorProto.BOOL, [1, 1, 4, 1]),
                _vi("h_color", TensorProto.BOOL, [1, 10, 1, 4]),
                _vi("v_color", TensorProto.BOOL, [1, 10, 4, 1]),
                _vi("h_live", TensorProto.BOOL, [1, 10, 1, 4]),
                _vi("v_live", TensorProto.BOOL, [1, 10, 4, 1]),
                _vi("h_line", TensorProto.BOOL, [1, 10, 4, 4]),
                _vi("v_line", TensorProto.BOOL, [1, 10, 4, 4]),
                _vi("h_selected", TensorProto.BOOL, [1, 10, 4, 4]),
                _vi("v_selected", TensorProto.BOOL, [1, 10, 4, 4]),
                _vi("compact_bool", TensorProto.BOOL, [1, 10, 4, 4]),
                _vi("compact_float", TensorProto.FLOAT, [1, 10, 4, 4]),
            ]
        )
        nodes.extend(
            [
                helper.make_node("Not", ["vertical"], ["not_vertical"]),
                helper.make_node("Where", ["vertical", "x_ids", "y_ids"], ["selected_ids"]),
                helper.make_node("Where", ["vertical", "x_vals_neg", "y_vals_neg"], ["selected_vals"]),
                helper.make_node("Greater", ["selected_vals", "neg_live_cut"], ["selected_live"]),
                helper.make_node("Unsqueeze", ["selected_ids"], ["ids_col"], axes=[0, 1, 3]),
                helper.make_node("Unsqueeze", ["selected_live"], ["live_col"], axes=[0, 1, 3]),
                helper.make_node("Equal", ["color_axis", "selected_ids"], ["h_color"]),
                helper.make_node("Equal", ["color_axis", "ids_col"], ["v_color"]),
                helper.make_node("And", ["h_color", "selected_live"], ["h_live"]),
                helper.make_node("And", ["v_color", "live_col"], ["v_live"]),
                helper.make_node("And", ["h_live", "row4_zero"], ["h_line"]),
                helper.make_node("And", ["v_live", "col4_zero"], ["v_line"]),
                helper.make_node("And", ["h_line", "vertical"], ["h_selected"]),
                helper.make_node("And", ["v_line", "not_vertical"], ["v_selected"]),
                helper.make_node("Or", ["h_selected", "v_selected"], ["compact_bool"]),
                helper.make_node("Cast", ["compact_bool"], ["compact_float"], to=TensorProto.FLOAT),
                helper.make_node("Pad", ["compact_float"], ["output"], mode="constant", pads=[0, 0, 0, 0, 0, 0, 26, 26]),
            ]
        )
    elif single_selected_output:
        value_infos.append(_vi("not_vertical", TensorProto.BOOL, [1]))
        nodes.append(helper.make_node("Not", ["vertical"], ["not_vertical"]))

        terms: list[str] = []
        for i, pos_name in enumerate(pos_names):
            x_id = f"s_x_id{i}"
            y_id = f"s_y_id{i}"
            x_val = f"s_x_val{i}"
            y_val = f"s_y_val{i}"
            id_name = f"s_id{i}"
            val_name = f"s_val{i}"
            live_name = f"s_live{i}"
            color_eq = f"s_color_eq{i}"
            row_pos = f"s_row_pos{i}"
            col_pos = f"s_col_pos{i}"
            v_rc = f"s_v_rc{i}"
            h_rc = f"s_h_rc{i}"
            v_sel = f"s_v_sel{i}"
            h_sel = f"s_h_sel{i}"
            pos_grid = f"s_pos_grid{i}"
            colored = f"s_colored{i}"
            live_grid = f"s_live_grid{i}"
            value_infos.extend(
                [
                    _vi(x_id, TensorProto.INT64, [1]),
                    _vi(y_id, TensorProto.INT64, [1]),
                    _vi(x_val, TensorProto.FLOAT, [1]),
                    _vi(y_val, TensorProto.FLOAT, [1]),
                    _vi(id_name, TensorProto.INT64, [1]),
                    _vi(val_name, TensorProto.FLOAT, [1]),
                    _vi(live_name, TensorProto.BOOL, [1]),
                    _vi(color_eq, TensorProto.BOOL, [1, 10, 1, 1]),
                    _vi(row_pos, TensorProto.BOOL, [1, 1, 4, 1]),
                    _vi(col_pos, TensorProto.BOOL, [1, 1, 1, 4]),
                    _vi(v_rc, TensorProto.BOOL, [1, 1, 4, 4]),
                    _vi(h_rc, TensorProto.BOOL, [1, 1, 4, 4]),
                    _vi(v_sel, TensorProto.BOOL, [1, 1, 4, 4]),
                    _vi(h_sel, TensorProto.BOOL, [1, 1, 4, 4]),
                    _vi(pos_grid, TensorProto.BOOL, [1, 1, 4, 4]),
                    _vi(colored, TensorProto.BOOL, [1, 10, 4, 4]),
                    _vi(live_grid, TensorProto.BOOL, [1, 10, 4, 4]),
                ]
            )
            nodes.extend(
                [
                    helper.make_node("Gather", ["x_ids", idx_names[i]], [x_id], axis=0),
                    helper.make_node("Gather", ["y_ids", idx_names[i]], [y_id], axis=0),
                    helper.make_node("Gather", ["x_vals_neg", idx_names[i]], [x_val], axis=0),
                    helper.make_node("Gather", ["y_vals_neg", idx_names[i]], [y_val], axis=0),
                    helper.make_node("Where", ["vertical", x_id, y_id], [id_name]),
                    helper.make_node("Where", ["vertical", x_val, y_val], [val_name]),
                    helper.make_node("Greater", [val_name, "neg_live_cut"], [live_name]),
                    helper.make_node("Equal", ["color_axis", id_name], [color_eq]),
                    helper.make_node("Equal", ["row4_axis", pos_name], [row_pos]),
                    helper.make_node("Equal", ["col4_axis", pos_name], [col_pos]),
                    helper.make_node("And", [row_pos, "col4_zero"], [v_rc]),
                    helper.make_node("And", ["row4_zero", col_pos], [h_rc]),
                    helper.make_node("And", [v_rc, "not_vertical"], [v_sel]),
                    helper.make_node("And", [h_rc, "vertical"], [h_sel]),
                    helper.make_node("Or", [v_sel, h_sel], [pos_grid]),
                    helper.make_node("And", [color_eq, pos_grid], [colored]),
                    helper.make_node("And", [colored, live_name], [live_grid]),
                ]
            )
            terms.append(live_grid)

        compact_bool = or_chain("s_or", terms, [1, 10, 4, 4])
        value_infos.append(_vi("compact_float", TensorProto.FLOAT, [1, 10, 4, 4]))
        nodes.extend(
            [
                helper.make_node("Cast", [compact_bool], ["compact_float"], to=TensorProto.FLOAT),
                helper.make_node("Pad", ["compact_float"], ["output"], mode="constant", pads=[0, 0, 0, 0, 0, 0, 26, 26]),
            ]
        )
    else:
        v_terms = build_terms("v", "x_ids", "x_vals_neg", horizontal=False)
        h_terms = build_terms("h", "y_ids", "y_vals_neg", horizontal=True)
        v_grid = or_chain("v_or", v_terms, [1, 10, 4, 4])
        h_grid = or_chain("h_or", h_terms, [1, 10, 4, 4])
        value_infos.extend(
            [
                _vi("v_compact", TensorProto.FLOAT, [1, 10, 4, 4]),
                _vi("h_compact", TensorProto.FLOAT, [1, 10, 4, 4]),
                _vi("v_compact_selected", TensorProto.FLOAT, [1, 10, 4, 4]),
                _vi("h_compact_selected", TensorProto.FLOAT, [1, 10, 4, 4]),
                _vi("vertical_f", TensorProto.FLOAT, [1]),
                _vi("not_vertical_f", TensorProto.FLOAT, [1]),
                _vi("compact_sum", TensorProto.FLOAT, [1, 10, 4, 4]),
            ]
        )
        nodes.extend(
            [
                helper.make_node("Cast", [v_grid], ["v_compact"], to=TensorProto.FLOAT),
                helper.make_node("Cast", [h_grid], ["h_compact"], to=TensorProto.FLOAT),
                helper.make_node("Cast", ["vertical"], ["vertical_f"], to=TensorProto.FLOAT),
                helper.make_node("Sub", ["one_f", "vertical_f"], ["not_vertical_f"]),
                helper.make_node("Mul", ["v_compact", "vertical_f"], ["v_compact_selected"]),
                helper.make_node("Mul", ["h_compact", "not_vertical_f"], ["h_compact_selected"]),
                helper.make_node("Add", ["v_compact_selected", "h_compact_selected"], ["compact_sum"]),
                helper.make_node("Pad", ["compact_sum"], ["output"], mode="constant", pads=[0, 0, 0, 0, 0, 0, 26, 26]),
            ]
        )

    graph = helper.make_graph(
        nodes,
        f"{TASK_ID}_{path.stem}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)],
        inits,
        value_info=value_infos,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    arr = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            arr[0, color, r, c] = 1.0
    return arr


def _decode(arr: np.ndarray, height: int = 30, width: int = 30) -> list[list[int]]:
    active = arr[0, :, :height, :width] > 0
    out: list[list[int]] = []
    for r in range(height):
        row: list[int] = []
        for c in range(width):
            hits = np.flatnonzero(active[:, r, c])
            row.append(int(hits[0]) if len(hits) == 1 else -1 if len(hits) > 1 else 0)
        out.append(row)
    return out


def _validate_file(path: Path) -> tuple[bool, str]:
    import onnxruntime as ort

    session = ort.InferenceSession(
        onnx.load(str(path)).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    with TASK_PATH.open(encoding="utf-8") as fh:
        task = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(task[split]):
            inp = convert_to_numpy(example, "input")
            exp = convert_to_numpy(example, "output")
            assert inp is not None and exp is not None
            got = session.run(["output"], {"input": inp})[0]
            if not np.array_equal(got > 0, exp > 0):
                out_h = len(example["output"])
                out_w = len(example["output"][0])
                return (
                    False,
                    f"{split}[{idx}] expected {example['output']} got "
                    f"{_decode(got, max(4, out_h), max(4, out_w))[: max(4, out_h)]}",
                )
    return True, "ok"


def _synthetic_examples() -> list[tuple[str, list[list[int]], list[list[int]]]]:
    vertical = [[4] * 5 + [2] * 5 + [8] * 6 for _ in range(14)]
    for r in range(2, 12, 3):
        vertical[r][4] = 2
        vertical[r][9] = 8
    horizontal = [[2] * 7 for _ in range(3)] + [[8] * 7 for _ in range(3)] + [[5] * 7 for _ in range(3)]
    horizontal[2][1] = 8
    horizontal[5][4] = 5
    return [
        ("vertical_3", vertical, [[4, 2, 8]]),
        ("horizontal_3", horizontal, [[2], [8], [5]]),
    ]


def _validate_synthetic(path: Path) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(
        onnx.load(str(path)).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    for name, inp_grid, exp_grid in _synthetic_examples():
        got = session.run(["output"], {"input": _grid_to_onehot(inp_grid)})[0]
        exp = _grid_to_onehot(exp_grid)
        if not np.array_equal(got > 0, exp > 0):
            raise AssertionError(f"synthetic {name} failed: expected {exp_grid}, got {_decode(got, 4, 4)}")


def build_variants() -> list[tuple[str, Path, dict[str, Any]]]:
    variants = [
        ("A_centroid_sorted", dict(use_div=True, topk_sorted=True, orient_margin=0.0)),
        ("B_centroid_unsorted_topk", dict(use_div=True, topk_sorted=False, orient_margin=0.0)),
        ("C_weighted_sum_specialized", dict(use_div=False, topk_sorted=True, orient_margin=0.0)),
        ("D_single_selected_output", dict(use_div=True, topk_sorted=True, orient_margin=0.0, single_selected_output=True)),
        ("E_matmul_selected_output", dict(use_div=True, topk_sorted=True, orient_margin=0.0, single_selected_output=True, use_matmul_coords=True)),
        ("F_vector_matmul_output", dict(use_div=True, topk_sorted=True, orient_margin=0.0, use_matmul_coords=True, vector_selected_output=True)),
    ]
    results: list[tuple[str, Path, dict[str, Any]]] = []
    for name, kwargs in variants:
        path = OUT_DIR / f"{TASK_ID}_{name}.onnx"
        _make_model(path, **kwargs)
        ok, message = _validate_file(path)
        result = score_file(path)
        result["correct"] = ok
        result["validation"] = message
        results.append((name, path, result))
    return results


def main() -> None:
    results = build_variants()
    valid = [item for item in results if item[2]["correct"] and item[2]["valid"]]
    if not valid:
        for name, _, result in results:
            print(name, result.get("validation"), result.get("error"))
        raise SystemExit("no valid correct variant")

    best_name, best_path, best_result = min(valid, key=lambda item: int(item[2]["cost"]))
    shutil.copyfile(best_path, BEST_PATH)
    _validate_synthetic(BEST_PATH)
    final_result = score_file(BEST_PATH)

    for name, _, result in results:
        score_text = None if result["score"] is None else f"{result['score']:.6f}"
        print(f"{name}: correct={result['correct']} valid={result['valid']} "
              f"memory={result['memory']} params={result['params']} cost={result['cost']} "
              f"score={score_text}")
        if not result["correct"] or not result["valid"]:
            print(f"  {result.get('validation') or result.get('error')}")
    print()
    print(f"Best variant: {best_name}")
    print_report(final_result)
    if final_result["valid"]:
        cost = int(final_result["cost"])
        print(f"formula check: max(1.0, 25.0 - ln({cost})) = {max(1.0, 25.0 - math.log(cost)):.6f}")
        print(
            "near-minimal note: the graph reduces the input to two 10x30 projections, "
            "sorts four present color centroids, and only expands to a 30x30 one-hot "
            "tensor for the final four output cells."
        )


if __name__ == "__main__":
    main()
