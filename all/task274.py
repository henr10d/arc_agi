"""ONNX for ARC task274: count empty rows above the cyan fill in a gray U.

Task rule: the input contains a gray U-shaped container and a solid cyan
rectangle touching the bottom of the U.  The 3x3 output encodes how many empty
interior rows sit above the cyan rectangle: one to three rows fill the output's
top row from left to right, and four rows additionally fill the middle-right
cell.  All other output cells are black.

Diagnostics compare the initially plausible cyan-bounding-box and gray-cavity
normalization hypotheses.  The winning hypothesis is the simpler empty-row
count, computed as cyan_top_row - gray_top_row.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task274"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task274.onnx"
DATA_PATH = ROOT / "data" / "task274.json"

C = 10
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


def _make_model(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto]) -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
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


def load_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def iter_examples(
    data: dict[str, list[dict[str, list[list[int]]]]],
    splits: Iterable[str] = ("train", "test", "arc-gen"),
) -> Iterable[tuple[str, int, np.ndarray, np.ndarray]]:
    for split in splits:
        for idx, ex in enumerate(data.get(split, [])):
            yield split, idx, np.asarray(ex["input"], dtype=np.int64), np.asarray(ex["output"], dtype=np.int64)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def cyan_bbox_region(grid: np.ndarray) -> tuple[int, int, int, int]:
    return bbox(grid == 8)


def gray_cavity_region(grid: np.ndarray) -> tuple[int, int, int, int]:
    gray = grid == 5
    rows, cols = np.where(gray)
    top = int(rows.min())
    bottom = int(rows.max())
    left = int(cols.min())
    right = int(cols.max())
    return top, bottom, left + 1, right


def normalized_occupancy(grid: np.ndarray, region: tuple[int, int, int, int], rule: str) -> np.ndarray:
    r0, r1, c0, c1 = region
    height = r1 - r0
    width = c1 - c0
    cyan = grid == 8
    out = np.zeros((3, 3), dtype=np.int64)

    for rr in range(3):
        for cc in range(3):
            if rule == "center":
                row = r0 + int((rr + 0.5) * height / 3.0)
                col = c0 + int((cc + 0.5) * width / 3.0)
                occupied = bool(cyan[row, col])
            else:
                rs = r0 + (rr * height) // 3
                re = r0 + ((rr + 1) * height) // 3
                cs = c0 + (cc * width) // 3
                ce = c0 + ((cc + 1) * width) // 3
                block = cyan[rs:re, cs:ce]
                if rule == "majority":
                    occupied = bool(block.size and block.sum() * 2 >= block.size)
                elif rule == "any-overlap":
                    occupied = bool(block.any())
                else:
                    raise ValueError(rule)
            if occupied:
                out[rr, cc] = 8
    return out


def solve_empty_rows(grid: np.ndarray) -> np.ndarray:
    gray_top = int(np.where(grid == 5)[0].min())
    cyan_top = int(np.where(grid == 8)[0].min())
    empty_rows = cyan_top - gray_top
    out = np.zeros((3, 3), dtype=np.int64)
    if empty_rows > 0:
        out[0, 0] = 8
    if empty_rows > 1:
        out[0, 1] = 8
    if empty_rows > 2:
        out[0, 2] = 8
    if empty_rows > 3:
        out[1, 2] = 8
    return out


def hypothesis_accuracy(
    data: dict[str, list[dict[str, list[list[int]]]]],
    solver: Callable[[np.ndarray], np.ndarray],
    splits: Iterable[str] = ("train", "test", "arc-gen"),
) -> tuple[int, int]:
    good = 0
    total = 0
    for _split, _idx, grid, expected in iter_examples(data, splits):
        total += 1
        good += int(np.array_equal(solver(grid), expected))
    return good, total


def print_diagnostics(data: dict[str, list[dict[str, list[list[int]]]]]) -> None:
    rules = ("center", "majority", "any-overlap")
    regions = {
        "cyan_bbox": cyan_bbox_region,
        "gray_cavity": gray_cavity_region,
    }

    print("training diagnostics")
    for split, idx, grid, expected in iter_examples(data, ("train",)):
        cb = cyan_bbox_region(grid)
        cav = gray_cavity_region(grid)
        gray_top = int(np.where(grid == 5)[0].min())
        cyan_top = int(np.where(grid == 8)[0].min())
        occ = solve_empty_rows(grid)
        print(
            f"{split}[{idx}] cyan_bbox={cb} gray_cavity={cav} "
            f"empty_rows={cyan_top - gray_top}"
        )
        print(f"  normalized occupancy matrix={occ.tolist()}")
        assert np.array_equal(occ, expected)

    for region_name, region_fn in regions.items():
        for rule in rules:
            solver = lambda grid, rf=region_fn, r=rule: normalized_occupancy(grid, rf(grid), r)
            train_good, train_total = hypothesis_accuracy(data, solver, ("train",))
            all_good, all_total = hypothesis_accuracy(data, solver)
            print(
                f"hypothesis={region_name}/{rule}: "
                f"train_accuracy={train_good}/{train_total} all_accuracy={all_good}/{all_total}"
            )

    train_good, train_total = hypothesis_accuracy(data, solve_empty_rows, ("train",))
    all_good, all_total = hypothesis_accuracy(data, solve_empty_rows)
    print(
        "selected rule=gray-cavity empty-row count "
        f"training accuracy={train_good}/{train_total} all_accuracy={all_good}/{all_total}"
    )


def finish_from_empty_rows(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    name: str,
) -> onnx.ModelProto:
    t0 = _i64(inits, 0, f"{name}_t0")
    t1 = _i64(inits, 1, f"{name}_t1")
    t2 = _i64(inits, 2, f"{name}_t2")
    t3 = _i64(inits, 3, f"{name}_t3")

    nodes.extend(
        [
            helper.make_node("Greater", ["empty_rows", t0], ["p1"]),
            helper.make_node("Greater", ["empty_rows", t1], ["p2"]),
            helper.make_node("Greater", ["empty_rows", t2], ["p3"]),
            helper.make_node("Greater", ["empty_rows", t3], ["p4"]),
            helper.make_node("Less", ["empty_rows", t0], ["false"]),
            helper.make_node("Concat", ["p1", "p2", "p3"], ["row0"], axis=3),
            helper.make_node("Concat", ["false", "false", "p4"], ["row1"], axis=3),
            helper.make_node("Concat", ["false", "false", "false"], ["row2"], axis=3),
            helper.make_node("Concat", ["row0", "row1", "row2"], ["cyan3"], axis=2),
            helper.make_node("Not", ["cyan3"], ["bg3"]),
            helper.make_node("And", ["cyan3", "bg3"], ["zero3"]),
            helper.make_node(
                "Concat",
                ["bg3", "zero3", "zero3", "zero3", "zero3", "zero3", "zero3", "zero3", "cyan3"],
                ["out9b"],
                axis=1,
            ),
            helper.make_node("Cast", ["out9b"], ["out9"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out9"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 1, H - 3, W - 3]),
        ]
    )
    return _make_model(nodes, inits)


def finish_from_empty_rows_thresholds(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    name: str,
) -> onnx.ModelProto:
    thresholds = _i64(
        inits,
        [[[[0, 1, 2], [9, 9, 3], [9, 9, 9]]]],
        f"{name}_thresholds",
    )

    nodes.extend(
        [
            helper.make_node("Greater", ["empty_rows", thresholds], ["cyan3"]),
            helper.make_node("Not", ["cyan3"], ["bg3"]),
            helper.make_node("And", ["cyan3", "bg3"], ["zero3"]),
            helper.make_node(
                "Concat",
                ["bg3", "zero3", "zero3", "zero3", "zero3", "zero3", "zero3", "zero3", "cyan3"],
                ["out9b"],
                axis=1,
            ),
            helper.make_node("Cast", ["out9b"], ["out9"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out9"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 1, H - 3, W - 3]),
        ]
    )
    return _make_model(nodes, inits)


def finish_from_empty_rows_float_thresholds(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    name: str,
) -> onnx.ModelProto:
    thresholds = _f32(
        inits,
        [[[[0, 1, 2], [9, 9, 3], [9, 9, 9]]]],
        f"{name}_thresholds",
    )

    nodes.extend(
        [
            helper.make_node("Greater", ["empty_rows_f", thresholds], ["cyan3"]),
            helper.make_node("Not", ["cyan3"], ["bg3"]),
            helper.make_node("And", ["cyan3", "bg3"], ["zero3"]),
            helper.make_node(
                "Concat",
                ["bg3", "zero3", "zero3", "zero3", "zero3", "zero3", "zero3", "zero3", "cyan3"],
                ["out9b"],
                axis=1,
            ),
            helper.make_node("Cast", ["out9b"], ["out9"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out9"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 1, H - 3, W - 3]),
        ]
    )
    return _make_model(nodes, inits)


def build_reducemax_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    gray_idx = _i64(inits, [5], "gray_idx")
    cyan_idx = _i64(inits, [8], "cyan_idx")

    nodes.extend(
        [
            helper.make_node("ReduceMax", [IN_NAME], ["row_by_color"], axes=[3], keepdims=1),
            helper.make_node("Gather", ["row_by_color", gray_idx], ["gray_rows"], axis=1),
            helper.make_node("Gather", ["row_by_color", cyan_idx], ["cyan_rows"], axis=1),
            helper.make_node("ArgMax", ["gray_rows"], ["gray_top"], axis=2, keepdims=1),
            helper.make_node("ArgMax", ["cyan_rows"], ["cyan_top"], axis=2, keepdims=1),
            helper.make_node("Sub", ["cyan_top", "gray_top"], ["empty_rows"]),
        ]
    )
    return finish_from_empty_rows(nodes, inits, "reduce")


def build_conv_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    weights = np.zeros((2, C, 1, W), dtype=np.float32)
    weights[0, 5, 0, :] = 1.0
    weights[1, 8, 0, :] = 1.0
    conv_w = _f32(inits, weights, "row_count_w")
    zero_f = _f32(inits, 0.0, "zero_f")

    nodes.extend(
        [
            helper.make_node("Conv", [IN_NAME, conv_w], ["row_counts"], kernel_shape=[1, W]),
            helper.make_node("Greater", ["row_counts", zero_f], ["has_rows_b"]),
            helper.make_node("Cast", ["has_rows_b"], ["has_rows"], to=TensorProto.FLOAT),
            helper.make_node("ArgMax", ["has_rows"], ["top_rows"], axis=2, keepdims=1),
            helper.make_node("Split", ["top_rows"], ["gray_top", "cyan_top"], axis=1, split=[1, 1]),
            helper.make_node("Sub", ["cyan_top", "gray_top"], ["empty_rows"]),
        ]
    )

    return finish_from_empty_rows(nodes, inits, "conv")


def build_column_probe_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    axes4 = _i64(inits, [0, 1, 2, 3], "probe_axes4")
    cyan_col_st = _i64(inits, [0, 8, 0, 3], "cyan_col_st")
    cyan_col_en = _i64(inits, [1, 9, H, 4], "cyan_col_en")
    gray_row_st = _i64(inits, [0, 5, 1, 0], "gray_row_st")
    gray_row_en = _i64(inits, [1, 6, 2, W], "gray_row_en")
    two = _i64(inits, 2, "probe_two")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, cyan_col_st, cyan_col_en, axes4], ["cyan_col3"]),
            helper.make_node("ArgMax", ["cyan_col3"], ["cyan_top"], axis=2, keepdims=1),
            helper.make_node("Slice", [IN_NAME, gray_row_st, gray_row_en, axes4], ["gray_row1"]),
            helper.make_node("ReduceMax", ["gray_row1"], ["has_gray_row1_f"], axes=[3], keepdims=1),
            helper.make_node("Cast", ["has_gray_row1_f"], ["has_gray_row1"], to=TensorProto.INT64),
            helper.make_node("Sub", ["cyan_top", two], ["cyan_top_minus_two"]),
            helper.make_node("Add", ["cyan_top_minus_two", "has_gray_row1"], ["empty_rows"]),
        ]
    )

    return finish_from_empty_rows(nodes, inits, "probe")


def build_compact_probe_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    cyan_col_st = _i64(inits, [0, 8, 2, 3], "compact_cyan_st")
    cyan_col_en = _i64(inits, [1, 9, 7, 4], "compact_cyan_en")
    gray_row_st = _i64(inits, [0, 5, 1, 1], "compact_gray_st")
    gray_row_en = _i64(inits, [1, 6, 2, 3], "compact_gray_en")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, cyan_col_st, cyan_col_en], ["cyan_col3_rows2_6"]),
            helper.make_node("ArgMax", ["cyan_col3_rows2_6"], ["cyan_top_minus_two"], axis=2, keepdims=1),
            helper.make_node("Slice", [IN_NAME, gray_row_st, gray_row_en], ["gray_row1_left"]),
            helper.make_node("ReduceMax", ["gray_row1_left"], ["has_gray_row1_f"], axes=[3], keepdims=1),
            helper.make_node("Cast", ["has_gray_row1_f"], ["has_gray_row1"], to=TensorProto.INT64),
            helper.make_node("Add", ["cyan_top_minus_two", "has_gray_row1"], ["empty_rows"]),
        ]
    )

    return finish_from_empty_rows(nodes, inits, "compact")


def build_compact_probe_threshold_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    cyan_col_st = _i64(inits, [0, 8, 2, 3], "compact_t_cyan_st")
    cyan_col_en = _i64(inits, [1, 9, 7, 4], "compact_t_cyan_en")
    gray_row_st = _i64(inits, [0, 5, 1, 1], "compact_t_gray_st")
    gray_row_en = _i64(inits, [1, 6, 2, 3], "compact_t_gray_en")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, cyan_col_st, cyan_col_en], ["cyan_col3_rows2_6"]),
            helper.make_node("ArgMax", ["cyan_col3_rows2_6"], ["cyan_top_minus_two"], axis=2, keepdims=1),
            helper.make_node("Slice", [IN_NAME, gray_row_st, gray_row_en], ["gray_row1_left"]),
            helper.make_node("ReduceMax", ["gray_row1_left"], ["has_gray_row1_f"], axes=[3], keepdims=1),
            helper.make_node("Cast", ["has_gray_row1_f"], ["has_gray_row1"], to=TensorProto.INT64),
            helper.make_node("Add", ["cyan_top_minus_two", "has_gray_row1"], ["empty_rows"]),
        ]
    )

    return finish_from_empty_rows_thresholds(nodes, inits, "compact_t")


def build_compact_probe_float_threshold_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    cyan_col_st = _i64(inits, [0, 8, 2, 3], "compact_f_cyan_st")
    cyan_col_en = _i64(inits, [1, 9, 7, 4], "compact_f_cyan_en")
    gray_row_st = _i64(inits, [0, 5, 1, 1], "compact_f_gray_st")
    gray_row_en = _i64(inits, [1, 6, 2, 3], "compact_f_gray_en")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, cyan_col_st, cyan_col_en], ["cyan_col3_rows2_6"]),
            helper.make_node("ArgMax", ["cyan_col3_rows2_6"], ["cyan_top_minus_two"], axis=2, keepdims=1),
            helper.make_node("Cast", ["cyan_top_minus_two"], ["cyan_top_minus_two_f"], to=TensorProto.FLOAT),
            helper.make_node("Slice", [IN_NAME, gray_row_st, gray_row_en], ["gray_row1_left"]),
            helper.make_node("ReduceMax", ["gray_row1_left"], ["has_gray_row1_f"], axes=[3], keepdims=1),
            helper.make_node("Add", ["cyan_top_minus_two_f", "has_gray_row1_f"], ["empty_rows_f"]),
        ]
    )

    return finish_from_empty_rows_float_thresholds(nodes, inits, "compact_f")


def grid_to_onehot(grid: np.ndarray) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            out[0, int(grid[r, c]), r, c] = 1.0
    return out


def onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def validate_model(model: onnx.ModelProto, data: dict[str, list[dict[str, list[list[int]]]]]) -> None:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    bad: list[str] = []

    for split, idx, grid, expected in iter_examples(data):
        pred_oh = session.run([OUT_NAME], {IN_NAME: grid_to_onehot(grid)})[0]
        pred = onehot_to_grid(pred_oh)[: expected.shape[0], : expected.shape[1]]
        active = pred_oh[0, :, : expected.shape[0], : expected.shape[1]] > 0.0
        if not np.array_equal(pred, expected) or not np.all(active.sum(axis=0) == 1):
            bad.append(f"{split}[{idx}]")

    if bad:
        raise AssertionError(f"failed {len(bad)} examples: {bad[:10]}")


def score_temp(model: onnx.ModelProto) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=f"_{TASK_ID}.onnx", dir=OUT_DIR, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        onnx.save(model, tmp_path)
        return score_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> None:
    data = load_data()
    print_diagnostics(data)

    candidates = [
        ("compact_probe_float_threshold", build_compact_probe_float_threshold_model()),
        ("compact_probe_threshold", build_compact_probe_threshold_model()),
        ("compact_probe", build_compact_probe_model()),
        ("column_probe", build_column_probe_model()),
        ("reducemax", build_reducemax_model()),
        ("conv", build_conv_model()),
    ]
    scored: list[tuple[int, str, onnx.ModelProto, dict[str, Any]]] = []
    for label, candidate in candidates:
        validate_model(candidate, data)
        result = score_temp(candidate)
        if not result["valid"]:
            raise AssertionError(f"{label}: {result['error']}")
        scored.append((int(result["cost"]), label, candidate, result))
        print(
            f"candidate={label} nodes={len(candidate.graph.node)} "
            f"memory={result['memory']} params={result['params']} "
            f"cost={result['cost']} score={result['score']:.6f}"
        )

    _cost, label, model, _result = min(scored, key=lambda item: item[0])
    onnx.save(model, BEST_PATH)
    final = score_file(BEST_PATH)
    if not final["valid"]:
        raise AssertionError(final["error"])

    print(
        f"selected candidate={label}; saved {BEST_PATH} nodes={len(model.graph.node)} "
        f"memory={final['memory']} params={final['params']} "
        f"cost={final['cost']} score={final['score']:.6f}"
    )


if __name__ == "__main__":
    main()
