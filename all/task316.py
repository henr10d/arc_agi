"""ONNX solution for ARC task316: pack sparse pixels by input-column order.

Task rule: the input is a 10x10 grid with 6 to 9 isolated non-black pixels,
with at most one colored pixel in any column.  Ignore the input rows.  Sort the
colored pixels by their input column and copy their colors into a 3x3 output in
snake order: the first row is left-to-right, the second row right-to-left, and
the third row left-to-right.  Any unused output cells are black, and the 3x3
result is padded to the NeuroGolf 30x30 one-hot tensor.

The graph slices foreground channels 1..9 from the 10x10 input, reduces each
column to a compact color vector, computes each column's occupied-column rank
from cumulative occupancy, and sums the matching column vector into the nine
snake positions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task316"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
FG = C - 1
H = W = 30
CORE = 10
OUT = 3
PAD = H - OUT
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _load_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def non_black_pixels(grid: list[list[int]] | np.ndarray) -> list[tuple[int, int, int]]:
    arr = np.asarray(grid, dtype=np.int64)
    return [
        (int(r), int(c), int(color))
        for r, row in enumerate(arr)
        for c, color in enumerate(row)
        if int(color) != 0
    ]


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference solver: sort non-black pixels by column and fill a 3x3 snake."""
    out = np.zeros((OUT, OUT), dtype=np.int64)
    for rank, (_row, _col, color) in enumerate(sorted(non_black_pixels(grid), key=lambda item: item[1])):
        row = rank // OUT
        col = rank % OUT if row % 2 == 0 else OUT - 1 - (rank % OUT)
        out[row, col] = color
    return out


def spatial_binning(grid: list[list[int]] | np.ndarray, *, flip_r: bool, flip_c: bool, transpose: bool) -> np.ndarray | None:
    out = np.zeros((OUT, OUT), dtype=np.int64)
    for r, c, color in non_black_pixels(grid):
        rr = min(OUT - 1, r * OUT // CORE)
        cc = min(OUT - 1, c * OUT // CORE)
        if flip_r:
            rr = OUT - 1 - rr
        if flip_c:
            cc = OUT - 1 - cc
        if transpose:
            rr, cc = cc, rr
        if out[rr, cc] != 0:
            return None
        out[rr, cc] = color
    return out


def rank_compression(grid: list[list[int]] | np.ndarray, *, flip_r: bool, flip_c: bool, transpose: bool) -> np.ndarray | None:
    pixels = non_black_pixels(grid)
    rows = sorted({r for r, _c, _color in pixels})
    cols = sorted({c for _r, c, _color in pixels})
    out = np.zeros((OUT, OUT), dtype=np.int64)
    for r, c, color in pixels:
        rr = rows.index(r) if len(rows) <= OUT else rows.index(r) * OUT // len(rows)
        cc = cols.index(c) if len(cols) <= OUT else cols.index(c) * OUT // len(cols)
        if flip_r:
            rr = OUT - 1 - rr
        if flip_c:
            cc = OUT - 1 - cc
        if transpose:
            rr, cc = cc, rr
        if out[rr, cc] != 0:
            return None
        out[rr, cc] = color
    return out


def ordered_fill(
    grid: list[list[int]] | np.ndarray,
    *,
    key: Callable[[tuple[int, int, int]], tuple[int, ...]],
    column_major: bool,
) -> np.ndarray:
    out = np.zeros((OUT, OUT), dtype=np.int64)
    for rank, (_r, _c, color) in enumerate(sorted(non_black_pixels(grid), key=key)):
        if column_major:
            col, row = divmod(rank, OUT)
        else:
            row, col = divmod(rank, OUT)
        out[row, col] = color
    return out


def verify_hypotheses() -> dict[str, int]:
    """Return train mismatch counts for requested hypotheses and the final rule."""
    data = _load_data()
    hypotheses: dict[str, Callable[[list[list[int]]], np.ndarray | None]] = {}

    for flip_r in (False, True):
        for flip_c in (False, True):
            for transpose in (False, True):
                name = f"spatial_binning flip_r={flip_r} flip_c={flip_c} transpose={transpose}"
                hypotheses[name] = (
                    lambda grid, flip_r=flip_r, flip_c=flip_c, transpose=transpose: spatial_binning(
                        grid, flip_r=flip_r, flip_c=flip_c, transpose=transpose
                    )
                )

                name = f"rank_compression flip_r={flip_r} flip_c={flip_c} transpose={transpose}"
                hypotheses[name] = (
                    lambda grid, flip_r=flip_r, flip_c=flip_c, transpose=transpose: rank_compression(
                        grid, flip_r=flip_r, flip_c=flip_c, transpose=transpose
                    )
                )

    order_keys: dict[str, Callable[[tuple[int, int, int]], tuple[int, ...]]] = {
        "row_col": lambda item: (item[0], item[1]),
        "col_row": lambda item: (item[1], item[0]),
        "row_plus_col_row": lambda item: (item[0] + item[1], item[0]),
        "row_plus_col_col": lambda item: (item[0] + item[1], item[1]),
        "row_minus_col": lambda item: (item[0] - item[1], item[0]),
        "col_minus_row": lambda item: (item[1] - item[0], item[1]),
    }
    for key_name, key in order_keys.items():
        hypotheses[f"ordered_fill {key_name} row_major"] = (
            lambda grid, key=key: ordered_fill(grid, key=key, column_major=False)
        )
        hypotheses[f"ordered_fill {key_name} column_major"] = (
            lambda grid, key=key: ordered_fill(grid, key=key, column_major=True)
        )
    hypotheses["column_snake"] = solve

    results: dict[str, int] = {}
    for name, fn in hypotheses.items():
        mismatches = 0
        for ex in data["train"]:
            actual = fn(ex["input"])
            expected = np.asarray(ex["output"], dtype=np.int64)
            if actual is None or actual.shape != expected.shape or not np.array_equal(actual, expected):
                mismatches += 1
        results[name] = mismatches
    return results


def print_train_coordinate_tables() -> None:
    data = _load_data()
    print("train coordinate tables:")
    for idx, ex in enumerate(data["train"]):
        print(f"train[{idx}] by input column:")
        for rank, (row, col, color) in enumerate(sorted(non_black_pixels(ex["input"]), key=lambda item: item[1])):
            out_row = rank // OUT
            out_col = rank % OUT if out_row % 2 == 0 else OUT - 1 - (rank % OUT)
            print(f"  rank {rank}: input(r={row}, c={col}, color={color}) -> output({out_row},{out_col})")
        print(f"  predicted: {solve(ex['input']).tolist()}")
        print(f"  expected:  {ex['output']}")


def _grid_to_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(arr):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def _expected_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    return _grid_to_onehot(grid)


def _strict_onehot_matches(pred: np.ndarray, expected: np.ndarray) -> bool:
    return pred.shape == expected.shape and np.array_equal(pred > 0.0, expected > 0.0)


def validate_reference() -> dict[str, tuple[int, int]]:
    data = _load_data()
    results: dict[str, tuple[int, int]] = {}
    for split in ("train", "test", "arc-gen"):
        total = len(data.get(split, []))
        passed = 0
        for ex in data.get(split, []):
            expected = np.asarray(ex["output"], dtype=np.int64)
            actual = solve(ex["input"])
            passed += int(actual.shape == expected.shape and np.array_equal(actual, expected))
        results[split] = (passed, total)
    return results


def build_column_snake_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, [0, 1, 2, 3], "axes")
    fg_st = _i64(inits, [0, 1, 0, 0], "fg_st")
    fg_en = _i64(inits, [1, C, CORE, CORE], "fg_en")
    zero = _f32(inits, np.zeros((1, 1, 1, 1), dtype=np.float32), "zero")
    rank_lowers = [
        _f32(inits, np.full((1, 1, 1, 1), rank - 0.5, dtype=np.float32), f"rank{rank}_lo")
        for rank in range(OUT * OUT)
    ]
    rank_uppers = [
        _f32(inits, np.full((1, 1, 1, 1), rank + 0.5, dtype=np.float32), f"rank{rank}_hi")
        for rank in range(OUT * OUT)
    ]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, fg_st, fg_en, axes], ["fg"]),
            helper.make_node("ReduceSum", ["fg"], ["col_sums"], axes=[2], keepdims=1),
            helper.make_node(
                "Split",
                ["col_sums"],
                [f"col{col}" for col in range(CORE)],
                axis=3,
                split=[1] * CORE,
            ),
        ]
    )

    prev_names: list[str] = []
    occ_names: list[str] = []
    running = zero
    for col in range(CORE):
        prev_names.append(running)
        occ = f"occ{col}"
        nodes.append(helper.make_node("ReduceSum", [f"col{col}"], [occ], axes=[1], keepdims=1))
        occ_names.append(occ)
        if col < CORE - 1:
            running_next = f"rank_before_col{col + 1}"
            nodes.append(helper.make_node("Add", [running, occ], [running_next]))
            running = running_next

    rank_tail_names: list[str] = []
    for rank, (rank_lower, rank_upper) in enumerate(zip(rank_lowers, rank_uppers)):
        parts: list[str] = []
        for col, prev in enumerate(prev_names):
            gt = f"rank{rank}_col{col}_gt"
            lt = f"rank{rank}_col{col}_lt"
            eq = f"rank{rank}_col{col}_eq"
            mask = f"rank{rank}_col{col}_mask"
            part = f"rank{rank}_col{col}_part"
            nodes.extend(
                [
                    helper.make_node("Greater", [prev, rank_lower], [gt]),
                    helper.make_node("Less", [prev, rank_upper], [lt]),
                    helper.make_node("And", [gt, lt], [eq]),
                    helper.make_node("Cast", [eq], [mask], to=TensorProto.FLOAT),
                    helper.make_node("Mul", [f"col{col}", mask], [part]),
                ]
            )
            parts.append(part)
        tail = f"tail{rank}"
        nodes.append(helper.make_node("Sum", parts, [tail]))
        rank_tail_names.append(tail)

    cell_names: list[str] = []
    for rank, tail in enumerate(rank_tail_names):
        active = f"active{rank}"
        active_bool = f"active{rank}_bool"
        bg_bool = f"bg{rank}_bool"
        bg = f"bg{rank}"
        cell = f"cell{rank}"
        nodes.extend(
            [
                helper.make_node("ReduceSum", [tail], [active], axes=[1], keepdims=1),
                helper.make_node("Greater", [active, zero], [active_bool]),
                helper.make_node("Not", [active_bool], [bg_bool]),
                helper.make_node("Cast", [bg_bool], [bg], to=TensorProto.FLOAT),
                helper.make_node("Concat", [bg, tail], [cell], axis=1),
            ]
        )
        cell_names.append(cell)

    snake_ranks = [
        [0, 1, 2],
        [5, 4, 3],
        [6, 7, 8],
    ]
    row_names: list[str] = []
    for row, ranks in enumerate(snake_ranks):
        row_name = f"row{row}"
        nodes.append(helper.make_node("Concat", [cell_names[rank] for rank in ranks], [row_name], axis=3))
        row_names.append(row_name)

    nodes.extend(
        [
            helper.make_node("Concat", row_names, ["out3"], axis=2),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]),
        ]
    )

    graph = helper.make_graph(nodes, "task316_column_snake", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def validate_model(model: onnx.ModelProto) -> tuple[bool, str]:
    try:
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    except Exception as exc:
        return False, f"ORT load failed: {exc}"

    data = _load_data()
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            expected = _expected_onehot(ex["output"])
            pred = sess.run([OUT_NAME], {IN_NAME: _grid_to_onehot(ex["input"])})[0]
            if not _strict_onehot_matches(pred, expected):
                return False, f"{split}#{idx} strict one-hot mismatch"
    return True, "PASS"


def save_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    model = build_column_snake_model()
    onnx.save(model, str(path))
    return model


def main() -> None:
    print_train_coordinate_tables()

    hypothesis_results = verify_hypotheses()
    print("hypothesis train mismatches:")
    for name, mismatches in hypothesis_results.items():
        status = "PASS" if mismatches == 0 else f"fail ({mismatches}/3)"
        print(f"  {status}: {name}")

    print(f"reference validation: {validate_reference()}")
    model = save_model()
    ok, message = validate_model(model)
    print(f"onnx validation: {message}")
    result = score_file(BEST_PATH)
    score = result.get("score")
    score_text = f"{score:.6f}" if isinstance(score, float) else "INVALID"
    print(f"saved: {BEST_PATH}")
    print(
        f"score_model: valid={result.get('valid')} memory={result.get('memory')} "
        f"params={result.get('params')} cost={result.get('cost')} score={score_text}"
    )
    if not ok:
        raise SystemExit(message)
    if not result.get("valid"):
        raise SystemExit(result.get("error") or "score_model reported invalid")


if __name__ == "__main__":
    main()
