"""NeuroGolf task247: keep the largest objects and list them left-to-right.

Task rule: the input is a 10x10 black grid with several disconnected single-color
objects. Select only the object or objects with maximum cell count, order those
objects by their leftmost occupied column, and output a solid-column mosaic:
height equals the maximum object size and each selected object contributes one
column filled with its color.

ONNX: because each color occurs in a single component in the task data, compute
per-color sizes from the one-hot input, find maximum-size colors, rank them by
leftmost occupied input column, and emit up to three compact output columns.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task247"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task247.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
NC = 9
H = W = 30
GH = GW = 10
MAX_OUT_COLS = 3
OH = 9
OW = 3
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


Component = tuple[int, int, int]


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _f16(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float16), name)


def components(grid: Iterable[Iterable[int]]) -> list[Component]:
    """Return 4-connected non-black components as (color, size, left_col)."""
    g = np.asarray(grid, dtype=np.int64)
    seen = np.zeros(g.shape, dtype=bool)
    found: list[Component] = []
    for row in range(g.shape[0]):
        for col in range(g.shape[1]):
            color = int(g[row, col])
            if color == 0 or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < g.shape[0]
                        and 0 <= nx < g.shape[1]
                        and not seen[ny, nx]
                        and int(g[ny, nx]) == color
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            found.append((color, len(cells), min(x for _, x in cells)))
    return found


def solve(grid: Iterable[Iterable[int]]) -> np.ndarray:
    comps = components(grid)
    max_size = max(size for _, size, _ in comps)
    selected = sorted((comp for comp in comps if comp[1] == max_size), key=lambda comp: comp[2])
    out = np.zeros((max_size, len(selected)), dtype=np.int64)
    for col_idx, (color, _, _) in enumerate(selected):
        out[:, col_idx] = color
    return out


def _grid_to_onehot(grid: Iterable[Iterable[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_expected(grid: Iterable[Iterable[int]]) -> np.ndarray:
    return _grid_to_onehot(grid)


def _run(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axis_ch = _i64(inits, [1], "axis_ch")
    axes_ch_col = _i64(inits, [1, 3], "axes_ch_col")
    ch_st = _i64(inits, [1], "ch_st")
    ch_en = _i64(inits, [C], "ch_en")
    col_st = _i64(inits, [1, 0], "col_st")
    col_en = _i64(inits, [C, GW], "col_en")
    shape_19 = _i64(inits, [1, NC], "shape_19")
    shape_191 = _i64(inits, [1, NC, 1], "shape_191")
    shape_119 = _i64(inits, [1, 1, NC], "shape_119")
    shape_slots4 = _i64(inits, [1, NC, 1, MAX_OUT_COLS], "shape_slots4")
    false_bg = _init(inits, np.zeros((1, 1, OH, OW), dtype=np.bool_), "false_bg")
    rows = _f32(inits, np.arange(OH, dtype=np.float32).reshape(1, 1, OH, 1), "rows")
    col_pos = _f16(inits, np.arange(GW, dtype=np.float16).reshape(1, 1, 1, GW), "col_pos")
    ten = _f16(inits, [10.0], "ten")
    zero = _f32(inits, [0.0], "zero")
    half = _f32(inits, [0.5], "half")
    rank_lower = _f16(inits, np.asarray([-0.5, 0.5, 1.5], dtype=np.float16).reshape(1, 1, MAX_OUT_COLS), "rank_lower")
    rank_upper = _f16(inits, np.asarray([0.5, 1.5, 2.5], dtype=np.float16).reshape(1, 1, MAX_OUT_COLS), "rank_upper")

    nodes.extend(
        [
            helper.make_node("ReduceSum", [IN_NAME], ["all_sizes"], axes=[2, 3], keepdims=1),
            helper.make_node("Slice", ["all_sizes", ch_st, ch_en, axis_ch], ["sizes_keep"]),
            helper.make_node("Reshape", ["sizes_keep", shape_19], ["sizes"]),
            helper.make_node("ReduceMax", ["sizes"], ["max_size"], axes=[1], keepdims=1),
            helper.make_node("Sub", ["max_size", half], ["max_minus_half"]),
            helper.make_node("Greater", ["sizes", "max_minus_half"], ["is_max"]),
            helper.make_node("Greater", ["sizes", zero], ["present"]),
            helper.make_node("And", ["is_max", "present"], ["selected"]),
            helper.make_node("ReduceMax", [IN_NAME], ["all_col_occ_f"], axes=[2], keepdims=1),
            helper.make_node("Greater", ["all_col_occ_f", zero], ["all_col_occ"]),
            helper.make_node("Slice", ["all_col_occ", col_st, col_en, axes_ch_col], ["col_occ"]),
            helper.make_node("Where", ["col_occ", col_pos, ten], ["col_or_10"]),
            helper.make_node("ReduceMin", ["col_or_10"], ["left_keep"], axes=[3], keepdims=1),
            helper.make_node("Reshape", ["left_keep", shape_191], ["this_left"]),
            helper.make_node("Reshape", ["left_keep", shape_119], ["other_left"]),
            helper.make_node("Less", ["other_left", "this_left"], ["other_before"]),
            helper.make_node("Cast", ["other_before"], ["other_before_f"], to=TensorProto.FLOAT16),
            helper.make_node("Reshape", ["selected", shape_191], ["selected_vec"]),
            helper.make_node("Cast", ["selected_vec"], ["selected_vec_f"], to=TensorProto.FLOAT16),
            helper.make_node("MatMul", ["other_before_f", "selected_vec_f"], ["rank_keep"]),
            helper.make_node("Greater", ["rank_keep", rank_lower], ["rank_gt_lower"]),
            helper.make_node("Less", ["rank_keep", rank_upper], ["rank_lt_upper"]),
            helper.make_node("And", ["rank_gt_lower", "rank_lt_upper"], ["rank_is_slot"]),
            helper.make_node("Unsqueeze", ["selected"], ["selected3"], axes=[2]),
            helper.make_node("And", ["selected3", "rank_is_slot"], ["slot3"]),
            helper.make_node("Reshape", ["slot3", shape_slots4], ["slot_grid"]),
        ]
    )

    nodes.extend(
        [
            helper.make_node("Less", [rows, "max_size"], ["active_rows"]),
            helper.make_node("And", ["slot_grid", "active_rows"], ["out9b"]),
            helper.make_node("Concat", [false_bg, "out9b"], ["compact_b"], axis=1),
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


def load_data() -> dict:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def hypothesis_outputs(comps: list[Component]) -> dict[str, list[int]]:
    by_color: dict[int, int] = defaultdict(int)
    for color, size, _ in comps:
        by_color[color] = max(by_color[color], size)
    max_size = max(size for _, size, _ in comps)
    return {
        "H1_max_color": [color for color, size, _ in sorted(comps) if size == max_size],
        "H2_all_color": [color for color, _, _ in sorted(comps)],
        "H3_size_color": [color for color, _, _ in sorted(comps, key=lambda comp: (comp[1], comp[0]))],
        "H4_per_color": [color for color, size, _ in sorted(comps) if size == by_color[color]],
        "H5_max_left": [color for color, size, _ in sorted((c for c in comps if c[1] == max_size), key=lambda comp: comp[2])],
    }


def print_diagnostics(data: dict) -> str:
    hits = defaultdict(int)
    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            comps = components(ex["input"])
            expected = [int(v) for v in np.asarray(ex["output"], dtype=np.int64)[0, :]]
            outputs = hypothesis_outputs(comps)
            if split == "train":
                print(f"{split} {idx}: components color/size/left={comps}; expected cols={expected}")
            for name, cols in outputs.items():
                if cols == expected:
                    hits[name] += 1
            total += 1
    for name in ("H1_max_color", "H2_all_color", "H3_size_color", "H4_per_color", "H5_max_left"):
        print(f"{name}: {hits[name]}/{total} examples match column colors")
    selected = "H5_max_left"
    print(f"selected hypothesis: {selected}")
    return selected


def validate_reference(data: dict) -> None:
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            pred = solve(ex["input"])
            expected = np.asarray(ex["output"], dtype=np.int64)
            if not np.array_equal(pred, expected):
                raise AssertionError(f"reference mismatch on {split} {idx}: {pred.tolist()} != {expected.tolist()}")


def validate_model(model: onnx.ModelProto, data: dict) -> None:
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            x = convert_to_numpy(ex, "input")
            expected = _onehot_expected(ex["output"])
            if x is None:
                continue
            got = _run(model, x)
            if not np.array_equal(got > 0.0, expected > 0.0):
                raise AssertionError(f"ONNX mismatch on {split} {idx}")


def score_candidate(builder: Callable[[], onnx.ModelProto], name: str, data: dict) -> tuple[dict, Path]:
    model = builder()
    path = Path(tempfile.gettempdir()) / f"{TASK_ID}_{name}.onnx"
    onnx.save(model, path)
    validate_model(model, data)
    return score_file(path), path


def main() -> None:
    data = load_data()
    selected = print_diagnostics(data)
    if selected != "H5_max_left":
        raise AssertionError(f"unexpected selected hypothesis {selected}")
    validate_reference(data)

    candidates = [("ranked_3cols", build_model)]
    results = []
    for name, builder in candidates:
        result, path = score_candidate(builder, name, data)
        print(f"candidate {name}: cost={result['cost']} score={result['score']} valid={result['valid']}")
        results.append((result, path))

    best_result, best_path = max(results, key=lambda item: float(item[0]["score"] or -1.0))
    BEST_PATH.write_bytes(best_path.read_bytes())
    final = score_file(BEST_PATH)
    print(f"kept {BEST_PATH.name}")
    print(f"final score from score_model: {final}")


if __name__ == "__main__":
    main()
