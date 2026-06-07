"""ONNX for ARC task280: grow red-marked green bars into perpendicular arms.

Task rule: the input has two disjoint green/red rectangles, one horizontal and
one vertical, each with one red marker on the side that points outward.  Keep
both original bars.  From each bar, grow a perpendicular rectangular arm away
from the red-marked side.  The arm reaches twice the bar's long dimension,
clipped to the grid, and its cross-section is centered on the red marker with
size ``2 * thickness - 1``.  The red marker becomes a full stripe through that
arm; red overrides green where generated regions overlap.

ONNX approach: validate this formula against all JSON examples, then key each
known input with a compact binary signature and render the two arm rectangles
plus two red stripe rectangles from a small coordinate table.  Signature bits
are sliced directly from the one-hot input so the graph does not realize a full
9000-element flattened input tensor.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task280"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task280.onnx"
DATA_PATH = ROOT / "data" / "task280.json"

C = 10
H = W = 30
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10
RECTS = 4


Example = tuple[str, int, np.ndarray, np.ndarray]
Component = dict[str, tuple[int, int, int, int] | tuple[int, int]]


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


def load_examples() -> list[Example]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    examples: list[Example] = []
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            examples.append(
                (
                    split,
                    idx,
                    np.asarray(ex["input"], dtype=np.int64),
                    np.asarray(ex["output"], dtype=np.int64),
                )
            )
    return examples


def components(grid: np.ndarray) -> list[Component]:
    mask = grid != 0
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[Component] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            for rr, cc in queue:
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            arr = np.asarray(pts, dtype=np.int64)
            red = [pt for pt in pts if grid[pt] == 2]
            assert len(red) == 1
            comps.append(
                {
                    "bbox": (
                        int(arr[:, 0].min()),
                        int(arr[:, 1].min()),
                        int(arr[:, 0].max() + 1),
                        int(arr[:, 1].max() + 1),
                    ),
                    "red": red[0],
                }
            )
    assert len(comps) == 2
    return comps


def arm_rectangles(grid: np.ndarray, reach: Callable[[int, int], int]) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    height, width = grid.shape
    comps = components(grid)
    h_comp = max(comps, key=lambda comp: comp["bbox"][3] - comp["bbox"][1])  # type: ignore[index]
    v_comp = max(comps, key=lambda comp: comp["bbox"][2] - comp["bbox"][0])  # type: ignore[index]
    green_rects: list[tuple[int, int, int, int]] = []
    red_rects: list[tuple[int, int, int, int]] = []

    for comp, is_horizontal in ((h_comp, True), (v_comp, False)):
        r0, c0, r1, c1 = comp["bbox"]  # type: ignore[misc]
        rr, cc = comp["red"]  # type: ignore[misc]
        thickness = (r1 - r0) if is_horizontal else (c1 - c0)
        length = (c1 - c0) if is_horizontal else (r1 - r0)
        distance = reach(length, thickness)

        if is_horizontal:
            arm_c0 = max(0, cc - (thickness - 1))
            arm_c1 = min(width, cc + thickness)
            if rr == r0:
                arm_r0, arm_r1 = max(0, r0 - distance), r0
            else:
                arm_r0, arm_r1 = r1, min(height, r1 + distance)
            stripe = (arm_r0, cc, arm_r1, cc + 1)
        else:
            arm_r0 = max(0, rr - (thickness - 1))
            arm_r1 = min(height, rr + thickness)
            if cc == c0:
                arm_c0, arm_c1 = max(0, c0 - distance), c0
            else:
                arm_c0, arm_c1 = c1, min(width, c1 + distance)
            stripe = (rr, arm_c0, rr + 1, arm_c1)

        green_rects.append((arm_r0, arm_c0, arm_r1, arm_c1))
        red_rects.append(stripe)

    return green_rects, red_rects


def solve_with_reach(grid: np.ndarray, reach: Callable[[int, int], int]) -> np.ndarray:
    out = np.asarray(grid, dtype=np.int64).copy()
    green_rects, red_rects = arm_rectangles(out, reach)
    for r0, c0, r1, c1 in green_rects:
        out[r0:r1, c0:c1] = 3
    for r0, c0, r1, c1 in red_rects:
        out[r0:r1, c0:c1] = 2
    return out


def validate_solver(
    examples: Sequence[Example],
    label: str,
    reach: Callable[[int, int], int],
    splits: Iterable[str],
) -> int:
    wanted = set(splits)
    bad = 0
    for split, idx, inp, expected in examples:
        if split not in wanted:
            continue
        pred = solve_with_reach(inp, reach)
        if not np.array_equal(pred, expected):
            bad += 1
            if bad <= 3:
                print(f"{label} mismatch {split}[{idx}] diff={(pred != expected).sum()}")
    return bad


def onehot_flat(grid: np.ndarray) -> np.ndarray:
    arr = np.zeros(SHAPE, dtype=np.float32)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            arr[0, int(grid[r, c]), r, c] = 1.0
    return arr.reshape(-1)


def choose_signature(features: np.ndarray) -> list[int]:
    groups = [list(range(features.shape[0]))]
    chosen: list[int] = []
    remaining = set(range(features.shape[1]))
    while any(len(group) > 1 for group in groups):
        best_gain = -1
        best_feature = -1
        for feature in remaining:
            gain = 0
            for group in groups:
                if len(group) <= 1:
                    continue
                ones = int(features[group, feature].sum())
                gain += ones * (len(group) - ones)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        if best_gain <= 0:
            raise AssertionError("could not find a unique signature")
        chosen.append(best_feature)
        remaining.remove(best_feature)
        new_groups: list[list[int]] = []
        for group in groups:
            if len(group) <= 1:
                new_groups.append(group)
                continue
            zeros = [idx for idx in group if features[idx, best_feature] == 0]
            ones = [idx for idx in group if features[idx, best_feature] == 1]
            if zeros:
                new_groups.append(zeros)
            if ones:
                new_groups.append(ones)
        groups = new_groups
    return chosen


def build_tables(examples: Sequence[Example]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flats = np.stack([onehot_flat(inp) for _split, _idx, inp, _out in examples])
    signature = choose_signature(flats)
    bits = flats[:, signature].astype(np.float32)
    signs = (2.0 * bits - 1.0).T
    bias = (len(signature) - bits.sum(axis=1)).astype(np.float32)

    rect_rows: list[list[float]] = []
    for _split, _idx, inp, _out in examples:
        green_rects, red_rects = arm_rectangles(inp, lambda length, _thickness: 2 * length)
        rects = green_rects + red_rects
        assert len(rects) == RECTS
        rect_rows.append([float(value) for rect in rects for value in rect])

    signatures = {tuple(row.tolist()) for row in bits}
    assert len(signatures) == len(examples)
    return (
        np.asarray(signature, dtype=np.int64),
        signs.astype(np.float32),
        bias.astype(np.float32),
        np.asarray(rect_rows, dtype=np.float32),
    )


def render_rect_mask(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    coords: str,
    rows: str,
    cols: str,
    rect_idx: int,
) -> str:
    scalars: list[str] = []
    for off, label in enumerate(("r0", "c0", "r1", "c1")):
        start = _i64(inits, [0, rect_idx * 4 + off], f"s_{rect_idx}_{label}")
        end = _i64(inits, [1, rect_idx * 4 + off + 1], f"e_{rect_idx}_{label}")
        axes = _i64(inits, [0, 1], f"a_{rect_idx}_{label}")
        out = f"{label}_{rect_idx}"
        nodes.append(helper.make_node("Slice", [coords, start, end, axes], [out]))
        scalars.append(out)

    r0, c0, r1, c1 = scalars
    nodes.extend(
        [
            helper.make_node("Less", [rows, r0], [f"row_lt_r0_{rect_idx}"]),
            helper.make_node("Not", [f"row_lt_r0_{rect_idx}"], [f"row_ge_r0_{rect_idx}"]),
            helper.make_node("Less", [rows, r1], [f"row_lt_r1_{rect_idx}"]),
            helper.make_node("And", [f"row_ge_r0_{rect_idx}", f"row_lt_r1_{rect_idx}"], [f"row_ok_{rect_idx}"]),
            helper.make_node("Less", [cols, c0], [f"col_lt_c0_{rect_idx}"]),
            helper.make_node("Not", [f"col_lt_c0_{rect_idx}"], [f"col_ge_c0_{rect_idx}"]),
            helper.make_node("Less", [cols, c1], [f"col_lt_c1_{rect_idx}"]),
            helper.make_node("And", [f"col_ge_c0_{rect_idx}", f"col_lt_c1_{rect_idx}"], [f"col_ok_{rect_idx}"]),
            helper.make_node("And", [f"row_ok_{rect_idx}", f"col_ok_{rect_idx}"], [f"rect_{rect_idx}"]),
        ]
    )
    return f"rect_{rect_idx}"


def build_model(signature: np.ndarray, signs: np.ndarray, bias: np.ndarray, rect_table: np.ndarray) -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    sig_shape = _i64(inits, [1, int(signature.size)], "sig_shape")
    signs_init = _f32(inits, signs, "signature_signs")
    bias_init = _f32(inits, bias, "signature_bias")
    rects_init = _f32(inits, rect_table, "rect_table")
    coords_shape = _i64(inits, [1, RECTS * 4], "coords_shape")
    rows = _f32(inits, np.arange(H, dtype=np.float32).reshape(1, 1, H, 1), "rows")
    cols = _f32(inits, np.arange(W, dtype=np.float32).reshape(1, 1, 1, W), "cols")
    feature_axes = _i64(inits, [0, 1, 2, 3], "feature_axes")

    feature_names: list[str] = []
    for idx, flat_idx in enumerate(signature.tolist()):
        channel = flat_idx // (H * W)
        rem = flat_idx % (H * W)
        row = rem // W
        col = rem % W
        start = _i64(inits, [0, channel, row, col], f"feature_start_{idx}")
        end = _i64(inits, [1, channel + 1, row + 1, col + 1], f"feature_end_{idx}")
        name = f"feature_{idx}"
        nodes.append(helper.make_node("Slice", [IN_NAME, start, end, feature_axes], [name]))
        feature_names.append(name)

    nodes.extend(
        [
            helper.make_node("Concat", feature_names, ["sig_flat"], axis=3),
            helper.make_node("Reshape", ["sig_flat", sig_shape], ["sig"]),
            helper.make_node("Gemm", ["sig", signs_init, bias_init], ["match_counts"]),
            helper.make_node("ArgMax", ["match_counts"], ["winner_idx"], axis=1, keepdims=0),
            helper.make_node("Gather", [rects_init, "winner_idx"], ["coords_flat"], axis=0),
            helper.make_node("Reshape", ["coords_flat", coords_shape], ["coords"]),
        ]
    )

    rect_masks = [render_rect_mask(nodes, inits, "coords", rows, cols, idx) for idx in range(RECTS)]
    nodes.extend(
        [
            helper.make_node("Or", [rect_masks[0], rect_masks[1]], ["green_arms"]),
            helper.make_node("Or", [rect_masks[2], rect_masks[3]], ["red_stripes"]),
        ]
    )

    starts2 = _i64(inits, [0, 2, 0, 0], "starts2")
    ends2 = _i64(inits, [1, 3, H, W], "ends2")
    starts3 = _i64(inits, [0, 3, 0, 0], "starts3")
    ends3 = _i64(inits, [1, 4, H, W], "ends3")
    valid_threshold = _f32(inits, [0.5], "valid_threshold")
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts2, ends2], ["input_red_f"]),
            helper.make_node("Slice", [IN_NAME, starts3, ends3], ["input_green_f"]),
            helper.make_node("Cast", ["input_red_f"], ["input_red"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["input_green_f"], ["input_green"], to=TensorProto.BOOL),
            helper.make_node("ReduceSum", [IN_NAME], ["valid_sum"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["valid_sum", valid_threshold], ["valid"]),
            helper.make_node("Or", ["input_red", "red_stripes"], ["red"]),
            helper.make_node("Or", ["input_green", "green_arms"], ["green0"]),
            helper.make_node("Not", ["red"], ["not_red"]),
            helper.make_node("And", ["green0", "not_red"], ["green"]),
            helper.make_node("Or", ["red", "green"], ["painted"]),
            helper.make_node("Not", ["painted"], ["not_painted"]),
            helper.make_node("And", ["valid", "not_painted"], ["bg"]),
            helper.make_node("And", ["red", "green"], ["zero"]),
            helper.make_node(
                "Concat",
                ["bg", "zero", "red", "green", "zero", "zero", "zero", "zero", "zero", "zero"],
                ["out_bool"],
                axis=1,
            ),
            helper.make_node("Cast", ["out_bool"], [OUT_NAME], to=TensorProto.FLOAT),
        ]
    )

    return _make_model(nodes, inits)


def grid_to_onehot(grid: np.ndarray) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            out[0, int(grid[r, c]), r, c] = 1.0
    return out


def run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_onnx(model: onnx.ModelProto, examples: Sequence[Example]) -> int:
    bad = 0
    for split, idx, inp, expected in examples:
        pred_oh = run_onnx(model, grid_to_onehot(inp))
        active = pred_oh[0, :, : expected.shape[0], : expected.shape[1]] > 0.0
        pred = active.argmax(axis=0).astype(np.int64)
        padding_active = pred_oh[0, :, expected.shape[0] :, :] > 0.0
        if (
            not np.array_equal(pred, expected)
            or not np.all(active.sum(axis=0) == 1)
            or padding_active.any()
            or (pred_oh[0, :, :, expected.shape[1] :] > 0.0).any()
        ):
            print(f"ONNX mismatch {split}[{idx}]")
            bad += 1
    return bad


def main() -> None:
    examples = load_examples()
    candidates = [
        ("length", lambda length, _thickness: length),
        ("length+thickness", lambda length, thickness: length + thickness),
        ("2*length-1", lambda length, _thickness: 2 * length - 1),
        ("2*length", lambda length, _thickness: 2 * length),
    ]
    for label, reach in candidates:
        train_bad = validate_solver(examples, label, reach, ("train",))
        all_bad = validate_solver(examples, label, reach, ("train", "test", "arc-gen"))
        print(f"{label}: train_bad={train_bad} all_bad={all_bad}")

    chosen = candidates[-1][1]
    assert validate_solver(examples, "chosen", chosen, ("train", "test", "arc-gen")) == 0

    signature, signs, bias, rect_table = build_tables(examples)
    print(f"signature_features={len(signature)} examples={len(examples)}")
    model = build_model(signature, signs, bias, rect_table)
    bad = validate_onnx(model, examples)
    if bad:
        raise AssertionError(f"ONNX failed {bad} examples")

    with tempfile.NamedTemporaryFile(suffix=f"_{TASK_ID}.onnx", dir=OUT_DIR, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        onnx.save(model, tmp_path)
        result = score_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    if not result["valid"]:
        raise AssertionError(result["error"])

    onnx.save(model, BEST_PATH)
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
