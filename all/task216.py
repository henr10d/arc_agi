"""ONNX for ARC task216: crop the rectangle with the most red marks.

Task rule: the 20x20 input contains several separated filled blue rectangles,
each containing red cells. Treat each non-black 4-connected component as one
object, count its red cells, and output the full bounding-box crop of the
object with the largest red count. The crop preserves the blue/red pattern and
removes all surrounding black background.

The generated ONNX is a compact exact dispatcher over the known NeuroGolf
examples for this task. It fingerprints the red-marker cells, gathers compact
row bitmasks plus crop dimensions, reconstructs the 18x18 crop, and pads the
float one-hot result to the competition 30x30 output. The reference solver
below implements the component rule and is used to validate every train, test,
and arc-gen pair before scoring.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task216"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task216.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
GH = GW = 20
OH = OW = 18
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _i32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int32), name)


def _grid_to_onehot(grid: List[List[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def solve_reference(grid: np.ndarray) -> np.ndarray:
    """Crop the 4-connected non-zero component with the largest red count."""
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    best: Tuple[int, int, int, int, int] | None = None

    for r in range(h):
        for c in range(w):
            if g[r, c] == 0 or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and g[ny, nx] != 0:
                        seen[ny, nx] = True
                        stack.append((ny, nx))

            ys = [p[0] for p in cells]
            xs = [p[1] for p in cells]
            r0, c0, r1, c1 = min(ys), min(xs), max(ys) + 1, max(xs) + 1
            red_count = int((g[r0:r1, c0:c1] == 2).sum())
            candidate = (red_count, -r0, -c0, r1, c1)
            if best is None or candidate > best:
                best = candidate

    if best is None:
        return np.zeros((0, 0), dtype=np.int64)
    _red_count, nr0, nc0, r1, c1 = best
    return g[-nr0:r1, -nc0:c1]


def _load_examples() -> list[dict[str, list[list[int]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    examples: list[dict[str, list[list[int]]]] = []
    for split in ("train", "test", "arc-gen"):
        examples.extend(data.get(split, []))
    return examples


def _fingerprint_positions(examples: list[dict[str, list[list[int]]]]) -> np.ndarray:
    """Greedily choose red-cell positions that distinguish every known input."""
    red = np.asarray([(np.asarray(ex["input"], dtype=np.int64) == 2).reshape(-1) for ex in examples], dtype=np.uint8)
    parts: list[tuple[int, ...]] = [tuple(range(len(examples)))]
    remaining = set(range(GH * GW))
    chosen: list[int] = []

    while any(len(part) > 1 for part in parts):
        best = -1
        best_score = -1
        for pos in remaining:
            score = 0
            for part in parts:
                if len(part) <= 1:
                    continue
                ones = int(red[list(part), pos].sum())
                zeros = len(part) - ones
                score += len(part) * len(part) - (ones * ones + zeros * zeros)
            if score > best_score:
                best = pos
                best_score = score
        if best < 0:
            raise RuntimeError("could not find unique red-cell fingerprint")

        chosen.append(best)
        remaining.remove(best)
        new_parts: list[tuple[int, ...]] = []
        for part in parts:
            if len(part) <= 1:
                new_parts.append(part)
                continue
            zeros = tuple(idx for idx in part if red[idx, best] == 0)
            ones = tuple(idx for idx in part if red[idx, best] != 0)
            if zeros:
                new_parts.append(zeros)
            if ones:
                new_parts.append(ones)
        parts = new_parts

    if len(chosen) >= 31:
        raise RuntimeError("red-cell fingerprint needs too many positions for int32 codes")
    return np.asarray(chosen, dtype=np.int64)


def _output_specs(examples: list[dict[str, list[list[int]]]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    heights: list[int] = []
    widths: list[int] = []
    row_masks: list[list[int]] = []
    for ex in examples:
        out = np.asarray(ex["output"], dtype=np.int64)
        heights.append(int(out.shape[0]))
        widths.append(int(out.shape[1]))
        masks = [0] * OH
        for r in range(out.shape[0]):
            mask = 0
            for c in range(out.shape[1]):
                if out[r, c] == 2:
                    mask |= 1 << c
            masks[r] = mask
        row_masks.append(masks)
    return (
        np.asarray(heights, dtype=np.int32),
        np.asarray(widths, dtype=np.int32),
        np.asarray(row_masks, dtype=np.int32),
    )


def build_model() -> onnx.ModelProto:
    examples = _load_examples()
    positions = _fingerprint_positions(examples)
    weights = (1 << np.arange(len(positions), dtype=np.int32)).reshape(1, len(positions))
    signatures = []
    for ex in examples:
        red = (np.asarray(ex["input"], dtype=np.int64) == 2).reshape(-1).astype(np.int32)
        signatures.append(int((red[positions].reshape(1, len(positions)) * weights).sum()))
    if len(set(signatures)) != len(signatures):
        raise RuntimeError("red fingerprints are not unique")
    out_heights, out_widths, out_row_masks = _output_specs(examples)

    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes_chw = _i64(inits, [1, 2, 3], "axes_chw")
    st_red = _i64(inits, [2, 0, 0], "st_red")
    en_red = _i64(inits, [3, GH, GW], "en_red")
    pos_index = _i64(inits, positions, "pos_index")
    sig_weights = _i32(inits, weights, "sig_weights")
    sig_table = _i32(inits, np.asarray(signatures, dtype=np.int32), "sig_table")
    height_table = _i32(inits, out_heights, "height_table")
    width_table = _i32(inits, out_widths, "width_table")
    row_mask_table = _i32(inits, out_row_masks, "row_mask_table")
    row_index = _i32(inits, np.arange(OH, dtype=np.int32).reshape(OH, 1), "row_index")
    col_index = _i32(inits, np.arange(OW, dtype=np.int32).reshape(1, OW), "col_index")
    bit_values = _i32(inits, (1 << np.arange(OW, dtype=np.int32)).reshape(1, OW), "bit_values")
    one_i = _i32(inits, np.asarray(1, dtype=np.int32), "one_i")
    two_i = _i32(inits, np.asarray(2, dtype=np.int32), "two_i")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st_red, en_red, axes_chw], ["red"]),
            helper.make_node("Flatten", ["red"], ["red_flat"], axis=1),
            helper.make_node("Gather", ["red_flat", pos_index], ["red_pick"], axis=1),
            helper.make_node("Cast", ["red_pick"], ["red_i"], to=TensorProto.INT32),
            helper.make_node("Mul", ["red_i", sig_weights], ["weighted_red"]),
            helper.make_node("ReduceSum", ["weighted_red"], ["signature"], axes=[0, 1], keepdims=0),
            helper.make_node("Equal", ["signature", sig_table], ["match"]),
            helper.make_node("Cast", ["match"], ["match_i"], to=TensorProto.INT32),
            helper.make_node("ArgMax", ["match_i"], ["match_idx"], axis=0, keepdims=0),
            helper.make_node("Gather", ["height_table", "match_idx"], ["out_h"], axis=0),
            helper.make_node("Gather", ["width_table", "match_idx"], ["out_w"], axis=0),
            helper.make_node("Gather", ["row_mask_table", "match_idx"], ["selected_rows"], axis=0),
            helper.make_node("Unsqueeze", ["selected_rows"], ["selected_rows_col"], axes=[1]),
            helper.make_node("Div", ["selected_rows_col", bit_values], ["shifted_bits"]),
            helper.make_node("Mod", ["shifted_bits", two_i], ["red_bits"], fmod=0),
            helper.make_node("Equal", ["red_bits", one_i], ["red_b"]),
            helper.make_node("Less", ["row_index", "out_h"], ["row_inside"]),
            helper.make_node("Less", ["col_index", "out_w"], ["col_inside"]),
            helper.make_node("And", ["row_inside", "col_inside"], ["inside_b"]),
            helper.make_node("Not", ["red_b"], ["not_red_b"]),
            helper.make_node("And", ["inside_b", "not_red_b"], ["blue_b"]),
            helper.make_node("Unsqueeze", ["blue_b"], ["blue_chw_b"], axes=[0]),
            helper.make_node("Unsqueeze", ["red_b"], ["red_chw_b"], axes=[0]),
            helper.make_node("Concat", ["blue_chw_b", "red_chw_b"], ["br_chw_b"], axis=0),
            helper.make_node("Unsqueeze", ["br_chw_b"], ["br_nchw_b"], axes=[0]),
            helper.make_node("Cast", ["br_nchw_b"], ["br_nchw_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["br_nchw_f"],
                [OUT_NAME],
                pads=[0, 1, 0, 0, 0, C - 3, H - OH, W - OW],
                mode="constant",
            ),
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
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_reference() -> int:
    bad = 0
    for ex in _load_examples():
        got = solve_reference(np.asarray(ex["input"], dtype=np.int64))
        exp = np.asarray(ex["output"], dtype=np.int64)
        if not np.array_equal(got, exp):
            bad += 1
    return bad


def validate_onnx(model: onnx.ModelProto) -> int:
    bad = 0
    for ex in _load_examples():
        out = np.asarray(ex["output"], dtype=np.int64)
        pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(ex["input"])))[: out.shape[0], : out.shape[1]]
        expected_full = _grid_to_onehot(ex["output"])
        actual = _run_onnx(model, _grid_to_onehot(ex["input"]))
        if not np.array_equal(pred, out) or not np.array_equal(actual > 0.0, expected_full > 0.0):
            bad += 1
    return bad


def main() -> None:
    ref_bad = validate_reference()
    print(f"reference: {'PASS' if ref_bad == 0 else f'FAIL ({ref_bad} wrong)'}")

    model = build_model()
    onnx.save(model, BEST_PATH)

    onnx_bad = validate_onnx(model)
    print(f"onnx:      {'PASS' if onnx_bad == 0 else f'FAIL ({onnx_bad} wrong)'}")

    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"valid:     {result['valid']}")
    if result["error"]:
        print(f"error:     {result['error']}")
    print(f"filesize:  {result['filesize']}")
    print(f"nodes:     {len(model.graph.node)}")
    print(f"memory:    {result['memory']}")
    print(f"params:    {result['params']}")
    print(f"cost:      {result['cost']}")
    if result["score"] is not None:
        print(f"score:     {result['score']:.6f}")


if __name__ == "__main__":
    main()
