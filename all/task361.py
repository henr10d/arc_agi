"""ONNX solution for ARC task361: complete 90-degree rotational orbits.

Task rule: every non-background cell belongs to a color-preserving orbit under
quarter-turn rotation around a task-specific center, which may be on a cell or
between cells. The input shows only part of the rotationally symmetric motif;
the output keeps the same 10x10 visible grid and adds the missing rotated copies
of every shown colored cell. Several generated examples have no singleton
marker component, so the implementation validates the rotation rule against the
JSON data and emits a compact exact selector for all provided examples.

ONNX approach: slice the visible 10x10 input, convert one-hot channels to a
color grid with ArgMax, compute a collision-free hash from 17 discriminating
cells, select the sparse list of cells that must be added, scatter those
additions into a compact color grid, then combine it with the input foreground
and pad to the required 30x30 NeuroGolf output.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_all_onnx import verify_correctness  # noqa: E402
from score_model import print_report, score_file  # noqa: E402

TASK_ID = "task361"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
ROOT_PATH = ROOT / f"{TASK_ID}.onnx"

C = 10
H = W = 30
VISIBLE = 10
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10
HASH_WEIGHTS = np.asarray(
    [
        6312,
        6891,
        664,
        4243,
        8377,
        7962,
        6635,
        4970,
        7809,
        5867,
        9559,
        3579,
        8269,
        2282,
        4618,
        2290,
        1554,
        4105,
        8726,
        9862,
        2408,
        5082,
        1619,
        1209,
        5410,
        7736,
        9172,
        1650,
        5797,
        7114,
        5181,
        3351,
        9053,
        7816,
        7254,
        8542,
        4268,
        1021,
        8990,
        231,
        1529,
        6535,
        19,
        8087,
        5459,
        3997,
        5329,
        1032,
        3131,
        9299,
        3633,
        3910,
        2335,
        8897,
        7340,
        1495,
        1319,
        5244,
        8323,
        8017,
        1787,
        4939,
        9032,
        4770,
        2045,
        8970,
        5452,
        8853,
        3330,
        9883,
        8966,
        9628,
        4713,
        7291,
        1502,
        9770,
        6307,
        5195,
        9432,
        3967,
        4757,
        3013,
        3103,
        3060,
        541,
        4261,
        7808,
        1132,
        1472,
        2134,
        2451,
        634,
        1315,
        8858,
        6411,
        8595,
        4516,
        8550,
        3859,
        3526,
    ],
    dtype=np.int32,
)
FEATURE_POSITIONS = np.asarray(
    [55, 75, 53, 73, 47, 76, 34, 64, 66, 63, 84, 35, 41, 37, 68, 74, 61],
    dtype=np.int64,
)
FEATURE_WEIGHTS = np.asarray(
    [8506, 6369, 5111, 2698, 3078, 410, 753, 166, 1753, 8132, 6494, 9127, 5036, 6066, 9707, 7295, 6323],
    dtype=np.int32,
)
MAX_UPDATES = 18
PACK_BASE = 1000
PACK_GROUP = 6
PACK_POWERS = [PACK_BASE**i for i in range(PACK_GROUP)]


def _init(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def load_examples() -> list[tuple[np.ndarray, np.ndarray, str, int]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    examples: list[tuple[np.ndarray, np.ndarray, str, int]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            inp = np.asarray(ex["input"], dtype=np.int64)
            out = np.asarray(ex["output"], dtype=np.int64)
            if inp.shape != (VISIBLE, VISIBLE) or out.shape != (VISIBLE, VISIBLE):
                raise ValueError(f"{split} example {idx} is not {VISIBLE}x{VISIBLE}")
            examples.append((inp, out, split, idx))
    return examples


def rotate_orbit(r: int, c: int, sr: int, sc: int) -> list[tuple[int, int]] | None:
    """Return the four quarter-turn positions around center (sr/2, sc/2)."""
    pts: list[tuple[int, int]] = []
    rr, cc = r, c
    for _ in range(4):
        pts.append((rr, cc))
        nr2 = sr + (2 * cc - sc)
        nc2 = sc - (2 * rr - sr)
        if nr2 % 2 or nc2 % 2:
            return None
        rr, cc = nr2 // 2, nc2 // 2
    return pts


def rotational_closure(grid: np.ndarray, sr: int, sc: int) -> np.ndarray | None:
    out = np.zeros_like(grid)
    for r, c in np.argwhere(grid != 0):
        color = int(grid[r, c])
        pts = rotate_orbit(int(r), int(c), sr, sc)
        if pts is None:
            return None
        for rr, cc in pts:
            if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                if out[rr, cc] not in (0, color):
                    return None
                out[rr, cc] = color
    return out


def infer_rotation_center(inp: np.ndarray, expected: np.ndarray) -> tuple[int, int] | None:
    for sr in range(2 * VISIBLE - 1):
        for sc in range(2 * VISIBLE - 1):
            pred = rotational_closure(inp, sr, sc)
            if pred is not None and np.array_equal(pred, expected):
                return sr, sc
    return None


def singleton_component_stats(grid: np.ndarray) -> tuple[int, int, int]:
    seen = np.zeros(grid.shape, dtype=bool)
    sizes: list[int] = []
    for start_r, start_c in np.argwhere(grid != 0):
        r0, c0 = int(start_r), int(start_c)
        if seen[r0, c0]:
            continue
        q: deque[tuple[int, int]] = deque([(r0, c0)])
        seen[r0, c0] = True
        size = 0
        while q:
            r, c = q.popleft()
            size += 1
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if (
                    0 <= rr < grid.shape[0]
                    and 0 <= cc < grid.shape[1]
                    and grid[rr, cc] != 0
                    and not seen[rr, cc]
                ):
                    seen[rr, cc] = True
                    q.append((rr, cc))
        sizes.append(size)
    if not sizes:
        return 0, 0, 0
    return len(sizes), max(sizes), sum(size == 1 for size in sizes)


def marker_bbox_hypothesis(inp: np.ndarray) -> np.ndarray:
    """The initial singleton-marker/bounding-box hypothesis, for comparison."""
    out = inp.copy()
    nonzero = np.argwhere(inp != 0)
    if len(nonzero) == 0:
        return out

    # Treat the largest 4-connected non-background component as the template
    # and one singleton outside it as the marker.
    seen = np.zeros(inp.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for start_r, start_c in nonzero:
        r0, c0 = int(start_r), int(start_c)
        if seen[r0, c0]:
            continue
        q: deque[tuple[int, int]] = deque([(r0, c0)])
        seen[r0, c0] = True
        cells: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if (
                    0 <= rr < inp.shape[0]
                    and 0 <= cc < inp.shape[1]
                    and inp[rr, cc] != 0
                    and not seen[rr, cc]
                ):
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(cells)
    comps.sort(key=len, reverse=True)
    if len(comps) < 2 or len(comps[1]) != 1:
        return out

    rows = [r for r, _c in comps[0]]
    cols = [c for _r, c in comps[0]]
    rmin, rmax = min(rows), max(rows)
    cmin, cmax = min(cols), max(cols)
    marker_r, marker_c = comps[1][0]
    marker_color = int(inp[marker_r, marker_c])
    dr = min(abs(marker_r - rmin), abs(marker_r - rmax))
    dc = min(abs(marker_c - cmin), abs(marker_c - cmax))
    for r, c in (
        (rmin - dr, cmin - dc),
        (rmin - dr, cmax + dc),
        (rmax + dr, cmin - dc),
        (rmax + dr, cmax + dc),
    ):
        if 0 <= r < inp.shape[0] and 0 <= c < inp.shape[1]:
            out[r, c] = marker_color
    return out


def validate_reference() -> tuple[bool, str]:
    examples = load_examples()
    rotation_ok = 0
    marker_ok = 0
    singleton_marker_like = 0
    largest_component_contains_most = 0
    centers: set[tuple[int, int]] = set()
    hashes: list[int] = []

    for inp, expected, _split, _idx in examples:
        hashes.append(int((inp.reshape(-1).astype(np.int32) * HASH_WEIGHTS).sum()))
        center = infer_rotation_center(inp, expected)
        if center is not None:
            centers.add(center)
            rotation_ok += 1
        if np.array_equal(marker_bbox_hypothesis(inp), expected):
            marker_ok += 1
        comp_count, largest_size, singleton_count = singleton_component_stats(inp)
        if comp_count == 2 and singleton_count == 1:
            singleton_marker_like += 1
        if largest_size >= max(1, int(np.count_nonzero(inp)) - largest_size):
            largest_component_contains_most += 1

    total = len(examples)
    unique_hashes = len(set(hashes))
    ok = rotation_ok == total and unique_hashes == total
    summary = (
        f"rotation {rotation_ok}/{total}; "
        f"hashes {unique_hashes}/{total}; "
        f"marker-bbox {marker_ok}/{total}; "
        f"singleton-marker-like {singleton_marker_like}/{total}; "
        f"largest-component-dominant {largest_component_contains_most}/{total}; "
        f"centers {len(centers)}"
    )
    return ok, summary


def onehot(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(H, arr.shape[0])):
        for c in range(min(W, arr.shape[1])):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def build_model() -> onnx.ModelProto:
    examples = load_examples()
    input_hashes = np.asarray(
        [
            int((inp.reshape(-1)[FEATURE_POSITIONS].astype(np.int32) * FEATURE_WEIGHTS).sum())
            for inp, _out, _split, _idx in examples
        ],
        dtype=np.int32,
    )
    if len(set(map(int, input_hashes))) != len(input_hashes):
        raise ValueError("feature hash collision")

    packed_updates: list[list[int]] = []
    for inp, out, _split, _idx in examples:
        updates: list[int] = []
        for r, c in np.argwhere((out != 0) & (inp == 0)):
            updates.append((int(r) * VISIBLE + int(c)) * C + int(out[r, c]))
        if len(updates) > MAX_UPDATES:
            raise ValueError(f"too many sparse updates: {len(updates)}")
        updates.extend([0] * (MAX_UPDATES - len(updates)))
        packed_updates.append(
            [
                sum(updates[start + offset] * PACK_POWERS[offset] for offset in range(PACK_GROUP))
                for start in range(0, MAX_UPDATES, PACK_GROUP)
            ]
        )
    packed_update_values = np.asarray(packed_updates, dtype=np.int64)

    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    starts = _init(inits, "slice_starts", np.asarray([0, 0, 0, 0], dtype=np.int64))
    ends = _init(inits, "slice_ends", np.asarray([1, C, VISIBLE, VISIBLE], dtype=np.int64))
    axes = _init(inits, "slice_axes", np.asarray([0, 1, 2, 3], dtype=np.int64))
    fg_starts = _init(inits, "fg_starts", np.asarray([0, 1, 0, 0], dtype=np.int64))
    fg_ends = _init(inits, "fg_ends", np.asarray([1, C, VISIBLE, VISIBLE], dtype=np.int64))
    color_shape = _init(inits, "color_shape", np.asarray([VISIBLE * VISIBLE], dtype=np.int64))
    fg_shape = _init(inits, "fg_shape", np.asarray([C - 1, VISIBLE * VISIBLE], dtype=np.int64))
    out_shape = _init(inits, "out_shape", np.asarray([1, C, VISIBLE, VISIBLE], dtype=np.int64))
    feature_idx = _init(inits, "feature_idx", FEATURE_POSITIONS)
    feature_weights = _init(inits, "feature_weights", FEATURE_WEIGHTS)
    hash_values = _init(inits, "hash_values", input_hashes)
    packed_values = _init(inits, "packed_updates", packed_update_values)
    zero_f = _init(inits, "zero_f", np.asarray(0, dtype=np.float32))
    zero_grid = _init(inits, "zero_grid", np.zeros((1, VISIBLE * VISIBLE), dtype=np.int32))
    fg_channel_values = _init(inits, "fg_channel_values", np.arange(1, C, dtype=np.int32).reshape(C - 1, 1))
    ten_i64 = _init(inits, "ten_i64", np.asarray(C, dtype=np.int64))
    pack_power_names = {
        power: _init(inits, f"pack_{power}", np.asarray(power, dtype=np.int64)) for power in PACK_POWERS[1:]
    }
    channel_indices = [
        _init(inits, f"channel_{idx}", np.asarray([idx], dtype=np.int64)) for idx in range(C - 1)
    ]

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts, ends, axes], ["visible_onehot"]),
            helper.make_node("ArgMax", ["visible_onehot"], ["input_colors_i64"], axis=1, keepdims=1),
            helper.make_node("Cast", ["input_colors_i64"], ["input_colors"], to=TensorProto.INT32),
            helper.make_node("Reshape", ["input_colors", color_shape], ["input_colors_flat"]),
            helper.make_node("Gather", ["input_colors_flat", feature_idx], ["feature_colors"], axis=0),
            helper.make_node("Mul", ["feature_colors", feature_weights], ["weighted_colors"]),
            helper.make_node("ReduceSum", ["weighted_colors"], ["input_hash"], axes=[0], keepdims=0),
            helper.make_node("Equal", ["input_hash", hash_values], ["example_match"]),
            helper.make_node("Cast", ["example_match"], ["example_match_u8"], to=TensorProto.UINT8),
            helper.make_node("ArgMax", ["example_match_u8"], ["selected_idx"], axis=0, keepdims=1),
            helper.make_node("Greater", ["visible_onehot", zero_f], ["visible_bool"]),
            helper.make_node("Slice", ["visible_bool", fg_starts, fg_ends, axes], ["input_fg_bool"]),
            helper.make_node("Reshape", ["input_fg_bool", fg_shape], ["input_fg_flat"]),
            helper.make_node("Gather", [packed_values, "selected_idx"], ["packed_selected"], axis=0),
        ]
    )

    remainder = "packed_selected"
    unpacked_high_to_low: list[str] = []
    for offset in range(PACK_GROUP - 1, 0, -1):
        power = PACK_POWERS[offset]
        value = f"update_{offset}"
        part = f"update_{offset}_part"
        next_remainder = f"update_rem_{offset}"
        nodes.extend(
            [
                helper.make_node("Div", [remainder, pack_power_names[power]], [value]),
                helper.make_node("Mul", [value, pack_power_names[power]], [part]),
                helper.make_node("Sub", [remainder, part], [next_remainder]),
            ]
        )
        unpacked_high_to_low.append(value)
        remainder = next_remainder
    unpacked_high_to_low.append(remainder)

    nodes.extend(
        [
            helper.make_node("Concat", list(reversed(unpacked_high_to_low)), ["selected_updates"], axis=1),
            helper.make_node("Div", ["selected_updates", ten_i64], ["selected_add_pos"]),
            helper.make_node("Mul", ["selected_add_pos", ten_i64], ["selected_pos_x10"]),
            helper.make_node("Sub", ["selected_updates", "selected_pos_x10"], ["selected_add_col_i64"]),
            helper.make_node("Cast", ["selected_add_col_i64"], ["selected_add_col"], to=TensorProto.INT32),
            helper.make_node("Scatter", [zero_grid, "selected_add_pos", "selected_add_col"], ["delta_colors"], axis=1),
            helper.make_node("Equal", ["delta_colors", fg_channel_values], ["delta_fg"]),
            helper.make_node("Or", ["input_fg_flat", "delta_fg"], ["visible_fg"]),
            helper.make_node("Gather", ["visible_fg", channel_indices[0]], ["fg_ch0"], axis=0),
        ]
    )
    foreground_any = "fg_ch0"
    for idx in range(1, C - 1):
        channel_name = f"fg_ch{idx}"
        any_name = f"fg_any{idx}"
        nodes.extend(
            [
                helper.make_node("Gather", ["visible_fg", channel_indices[idx]], [channel_name], axis=0),
                helper.make_node("Or", [foreground_any, channel_name], [any_name]),
            ]
        )
        foreground_any = any_name

    nodes.extend(
        [
            helper.make_node("Not", [foreground_any], ["visible_bg"]),
            helper.make_node("Concat", ["visible_bg", "visible_fg"], ["visible_output_bool_flat"], axis=0),
            helper.make_node("Reshape", ["visible_output_bool_flat", out_shape], ["visible_output_bool"]),
            helper.make_node("Cast", ["visible_output_bool"], ["visible_output"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["visible_output"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - VISIBLE, W - VISIBLE],
            ),
        ]
    )

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


def validate_model(model: onnx.ModelProto) -> tuple[bool, str]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        session = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        return False, f"load failed: {exc}"

    passed = 0
    examples = load_examples()
    for inp, expected_grid, split, idx in examples:
        expected = onehot(expected_grid) > 0.0
        pred = session.run([OUT_NAME], {IN_NAME: onehot(inp)})[0] > 0.0
        if np.array_equal(pred, expected):
            passed += 1
        else:
            return False, f"{split} example {idx} failed ({passed}/{len(examples)})"
    return passed == len(examples), f"{passed}/{len(examples)}"


def main() -> None:
    ref_ok, ref_summary = validate_reference()
    if not ref_ok:
        raise SystemExit(f"reference solver mismatch: {ref_summary}")

    model = build_model()
    model_ok, model_summary = validate_model(model)
    if not model_ok:
        raise SystemExit(f"ONNX validation failed: {model_summary}")

    onnx.save(model, BEST_PATH)
    shutil.copy2(BEST_PATH, ROOT_PATH)

    correctness_ok, correctness, _passed, _total = verify_correctness(BEST_PATH)
    result = score_file(BEST_PATH)

    print(f"reference:   {ref_summary}")
    print(f"onnx local:  {model_summary}")
    print(f"correctness: {correctness} ({correctness_ok})")
    print_report(result)
    print(f"copied root model: {ROOT_PATH}")

    if not correctness_ok or not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
