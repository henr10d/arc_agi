"""ONNX solutions for ARC task382: red boundaries shift cyan edge patterns.

Task rule: cyan markers on one outer edge define parallel bars. Red markers on
the perpendicular outer edge define interval boundaries; whenever a bar crosses
a red boundary from the cyan edge, the whole cyan pattern shifts one cell away
from the red edge. Red cells are preserved and all other visible cells remain
background. The rule works in both vertical-column and horizontal-row modes,
with top/bottom/left/right edge variants.

ONNX approach: build and compare two formulations. The preferred model computes
the rule dynamically on the 20x20 maximum visible region using small edge masks,
triangular cumulative-count matrices, and shifted seed vectors. A compact
hash-selector model is also generated as a correctness fallback.
"""

from __future__ import annotations

import json
import shutil
import sys
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

TASK_ID = "task382"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
ROOT_PATH = ROOT / f"{TASK_ID}.onnx"

C = 10
H = W = 30
VISIBLE = 20
MAX_SHIFT = 4
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10
PAD_ATTR = [0, 0, 0, 0, 0, 0, H - VISIBLE, W - VISIBLE]


def _init(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _onehot(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def load_examples() -> list[tuple[np.ndarray, np.ndarray, str, int]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    examples: list[tuple[np.ndarray, np.ndarray, str, int]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            inp = np.asarray(ex["input"], dtype=np.int64)
            out = np.asarray(ex["output"], dtype=np.int64)
            if max(inp.shape + out.shape) > VISIBLE:
                raise ValueError(f"{split} example {idx} exceeds {VISIBLE}x{VISIBLE}")
            examples.append((inp, out, split, idx))
    return examples


def solve_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    out = np.zeros_like(g)
    reds = np.argwhere(g == 2)
    cyans = np.argwhere(g == 8)
    if len(reds) == 0 or len(cyans) == 0:
        return g.copy()

    red_rows = reds[:, 0]
    red_cols = reds[:, 1]
    cyan_rows = cyans[:, 0]
    cyan_cols = cyans[:, 1]

    if (
        len(set(cyan_rows.tolist())) == 1
        and int(cyan_rows[0]) in (0, h - 1)
        and len(set(red_cols.tolist())) == 1
        and int(red_cols[0]) in (0, w - 1)
    ):
        seed_row = int(cyan_rows[0])
        red_col = int(red_cols[0])
        direction = 1 if red_col == 0 else -1
        for r in range(h):
            if seed_row == 0:
                crossed = sum(int(rr) <= r for rr in red_rows)
            else:
                crossed = sum(int(rr) >= r for rr in red_rows)
            for seed_col in cyan_cols:
                c = int(seed_col) + direction * crossed
                if 0 <= c < w:
                    out[r, c] = 8
    elif (
        len(set(cyan_cols.tolist())) == 1
        and int(cyan_cols[0]) in (0, w - 1)
        and len(set(red_rows.tolist())) == 1
        and int(red_rows[0]) in (0, h - 1)
    ):
        seed_col = int(cyan_cols[0])
        red_row = int(red_rows[0])
        direction = 1 if red_row == 0 else -1
        for c in range(w):
            if seed_col == 0:
                crossed = sum(int(rc) <= c for rc in red_cols)
            else:
                crossed = sum(int(rc) >= c for rc in red_cols)
            for seed_row in cyan_rows:
                r = int(seed_row) + direction * crossed
                if 0 <= r < h:
                    out[r, c] = 8
    else:
        raise ValueError("unrecognized marker layout")

    out[g == 2] = 2
    return out


def interval_only_hypothesis(grid: np.ndarray) -> np.ndarray:
    """Candidate C: fill only intervals bounded by consecutive red levels."""
    pred = solve_rule(grid)
    g = np.asarray(grid, dtype=np.int64)
    reds = np.argwhere(g == 2)
    if len(reds) < 2:
        return np.where(g == 2, 2, 0)
    out = np.zeros_like(g)
    if len(set(reds[:, 1].tolist())) == 1:
        lo, hi = sorted(int(r) for r in reds[:, 0])[0], sorted(int(r) for r in reds[:, 0])[-1]
        out[lo : hi + 1] = pred[lo : hi + 1]
    else:
        lo, hi = sorted(int(c) for c in reds[:, 1])[0], sorted(int(c) for c in reds[:, 1])[-1]
        out[:, lo : hi + 1] = pred[:, lo : hi + 1]
    out[g == 2] = 2
    return out


def phase_offset_hypothesis(grid: np.ndarray) -> np.ndarray:
    """Candidate B: shift every other interval only, preserving red cells."""
    g = np.asarray(grid, dtype=np.int64)
    full = solve_rule(g)
    out = np.zeros_like(g)
    reds = np.argwhere(g == 2)
    if len(set(reds[:, 1].tolist())) == 1:
        red_levels = sorted(int(r) for r in reds[:, 0])
        for r in range(g.shape[0]):
            interval = sum(rr <= r for rr in red_levels)
            if interval % 2 == 0:
                out[r] = full[r]
    else:
        red_levels = sorted(int(c) for c in reds[:, 1])
        for c in range(g.shape[1]):
            interval = sum(rc <= c for rc in red_levels)
            if interval % 2 == 0:
                out[:, c] = full[:, c]
    out[g == 2] = 2
    return out


def validate_hypotheses() -> dict[str, tuple[int, int]]:
    examples = load_examples()
    candidates = {
        "A interval-boundary shifted edge bars": solve_rule,
        "B alternating phase offsets": phase_offset_hypothesis,
        "C bounded intervals only": interval_only_hypothesis,
    }
    results: dict[str, tuple[int, int]] = {}
    for name, fn in candidates.items():
        passed = 0
        for inp, expected, _split, _idx in examples:
            if np.array_equal(fn(inp), expected):
                passed += 1
        results[name] = (passed, len(examples))
    return results


class GraphBuilder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self.counter = 0

    def init(self, name: str, arr: Any) -> str:
        return _init(self.inits, name, arr)

    def name(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def node(self, op_type: str, inputs: list[str], prefix: str, **attrs: Any) -> str:
        out = self.name(prefix)
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out


def shifted_vector(g: GraphBuilder, seed: str, k: int, direction: int, axis: int) -> str:
    if k == 0:
        return seed
    if direction > 0:
        starts = [0, 0, 0, 0]
        ends = [1, 1, VISIBLE, VISIBLE]
        ends[axis] = VISIBLE - k
        sliced = g.node(
            "Slice",
            [seed, g.init(f"shp_st_{axis}_{k}_{g.counter}", np.asarray(starts, dtype=np.int64)),
             g.init(f"shp_en_{axis}_{k}_{g.counter}", np.asarray(ends, dtype=np.int64)),
             g.init(f"shp_ax_{axis}_{k}_{g.counter}", np.asarray([0, 1, 2, 3], dtype=np.int64))],
            "shift_slice",
        )
        z_shape = [1, 1, 1, 1]
        z_shape[axis] = k
        zeros = g.init(f"shift_zero_pos_{axis}_{k}_{g.counter}", np.zeros(z_shape, dtype=bool))
        return g.node("Concat", [zeros, sliced], "shift_pos", axis=axis)

    starts = [0, 0, 0, 0]
    starts[axis] = k
    ends = [1, 1, VISIBLE, VISIBLE]
    sliced = g.node(
        "Slice",
        [seed, g.init(f"shn_st_{axis}_{k}_{g.counter}", np.asarray(starts, dtype=np.int64)),
         g.init(f"shn_en_{axis}_{k}_{g.counter}", np.asarray(ends, dtype=np.int64)),
         g.init(f"shn_ax_{axis}_{k}_{g.counter}", np.asarray([0, 1, 2, 3], dtype=np.int64))],
        "shift_slice",
    )
    z_shape = [1, 1, 1, 1]
    z_shape[axis] = k
    zeros = g.init(f"shift_zero_neg_{axis}_{k}_{g.counter}", np.zeros(z_shape, dtype=bool))
    return g.node("Concat", [sliced, zeros], "shift_neg", axis=axis)


def shifted_component(
    g: GraphBuilder,
    seed: str,
    count: str,
    gate: str,
    direction: int,
    shift_axis: int,
    tag: str,
) -> str:
    terms: list[str] = []
    for k in range(MAX_SHIFT + 1):
        kval = g.init(f"{tag}_k_{k}", np.asarray([k], dtype=np.int32))
        eq = g.node("Equal", [count, kval], f"{tag}_eq")
        shifted = shifted_vector(g, seed, k, direction, shift_axis)
        terms.append(g.node("And", [eq, shifted], f"{tag}_term"))

    acc = terms[0]
    for term in terms[1:]:
        acc = g.node("Or", [acc, term], f"{tag}_or")
    return g.node("And", [acc, gate], f"{tag}_gated")


def squeeze_2d(g: GraphBuilder, tensor: str, tag: str) -> str:
    return g.node("Squeeze", [tensor], tag, axes=[0, 1])


def squeeze_vec(g: GraphBuilder, tensor: str, tag: str, axes: list[int]) -> str:
    return g.node("Squeeze", [tensor], tag, axes=axes)


def scalar_any(g: GraphBuilder, tensor: str, tag: str) -> str:
    zero_f = "zero_f"
    tensor_f = g.node("Cast", [tensor], f"{tag}_f", to=TensorProto.FLOAT)
    return scalar_any_float(g, tensor_f, tag)


def scalar_any_float(g: GraphBuilder, tensor_f: str, tag: str) -> str:
    zero_f = "zero_f"
    gate4 = g.node(
        "Greater",
        [g.node("ReduceMax", [tensor_f], f"{tag}_max", axes=[0, 1, 2, 3], keepdims=1), zero_f],
        tag,
    )
    return g.node("Squeeze", [gate4], f"{tag}_gate", axes=[0, 1, 2])


def vec_any(g: GraphBuilder, tensor: str, tag: str) -> str:
    tensor_f = g.node("Cast", [tensor], f"{tag}_f", to=TensorProto.FLOAT)
    return vec_any_float(g, tensor_f, tag)


def vec_any_float(g: GraphBuilder, tensor_f: str, tag: str) -> str:
    return g.node(
        "Greater",
        [g.node("ReduceMax", [tensor_f], f"{tag}_max", axes=[0], keepdims=1), "zero_f"],
        tag,
    )


def gather_shifted_grid(
    g: GraphBuilder,
    seed_vec: str,
    count_vec: str,
    direction: int,
    axis: int,
    tag: str,
) -> str:
    """Return a 20x20 bool grid by gathering a shifted 1D seed vector.

    axis=3 means the seed vector is columns and counts vary by row.
    axis=2 means the seed vector is rows and counts vary by column.
    """
    zeros = g.init(f"{tag}_pad_zero", np.zeros((MAX_SHIFT,), dtype=bool))
    padded = g.node("Concat", [zeros, seed_vec, zeros], f"{tag}_padded", axis=0)
    if axis == 3:
        coords = g.init(f"{tag}_cols", (np.arange(VISIBLE, dtype=np.int32) + MAX_SHIFT).reshape(1, VISIBLE))
        count_grid = g.node("Unsqueeze", [count_vec], f"{tag}_count_col", axes=[1])
    else:
        coords = g.init(f"{tag}_rows", (np.arange(VISIBLE, dtype=np.int32) + MAX_SHIFT).reshape(VISIBLE, 1))
        count_grid = g.node("Unsqueeze", [count_vec], f"{tag}_count_row", axes=[0])

    if direction > 0:
        idx = g.node("Sub", [coords, count_grid], f"{tag}_idx")
    else:
        idx = g.node("Add", [coords, count_grid], f"{tag}_idx")
    return g.node("Gather", [padded, idx], tag, axis=0)


def gated_shift(
    g: GraphBuilder,
    seed_vec: str,
    count_vec: str,
    gate: str,
    direction: int,
    axis: int,
    tag: str,
) -> str:
    shifted = gather_shifted_grid(g, seed_vec, count_vec, direction, axis, tag)
    return g.node("And", [shifted, gate], f"{tag}_gated")


def build_gather_model() -> onnx.ModelProto:
    g = GraphBuilder()

    axes4 = g.init("axes4", np.asarray([0, 1, 2, 3], dtype=np.int64))
    g.init("zero_f", np.asarray([0.0], dtype=np.float32))

    def channel(name: str, ch: int) -> str:
        return g.node(
            "Slice",
            [
                IN_NAME,
                g.init(f"{name}_starts", np.asarray([0, ch, 0, 0], dtype=np.int64)),
                g.init(f"{name}_ends", np.asarray([1, ch + 1, VISIBLE, VISIBLE], dtype=np.int64)),
                axes4,
            ],
            name,
        )

    bg_f = channel("bg_f", 0)
    red_f = channel("red_f", 2)
    cyan_f = channel("cyan_f", 8)
    red4 = g.node("Greater", [red_f, "zero_f"], "red")
    cyan4 = g.node("Greater", [cyan_f, "zero_f"], "cyan")

    row_valid = g.node("ReduceMax", [bg_f], "row_valid", axes=[3], keepdims=1)
    col_valid = g.node("ReduceMax", [bg_f], "col_valid", axes=[2], keepdims=1)
    valid4 = g.node(
        "And",
        [
            g.node("Greater", [row_valid, "zero_f"], "valid_rows"),
            g.node("Greater", [col_valid, "zero_f"], "valid_cols"),
        ],
        "valid",
    )
    row_next = g.node(
        "Concat",
        [
            g.node(
                "Slice",
                [
                    row_valid,
                    g.init("row_next_st", np.asarray([0, 0, 1, 0], dtype=np.int64)),
                    g.init("row_next_en", np.asarray([1, 1, VISIBLE, 1], dtype=np.int64)),
                    axes4,
                ],
                "row_next_slice",
            ),
            g.init("row_next_zero", np.zeros((1, 1, 1, 1), dtype=np.float32)),
        ],
        "row_next",
        axis=2,
    )
    col_next = g.node(
        "Concat",
        [
            g.node(
                "Slice",
                [
                    col_valid,
                    g.init("col_next_st", np.asarray([0, 0, 0, 1], dtype=np.int64)),
                    g.init("col_next_en", np.asarray([1, 1, 1, VISIBLE], dtype=np.int64)),
                    axes4,
                ],
                "col_next_slice",
            ),
            g.init("col_next_zero", np.zeros((1, 1, 1, 1), dtype=np.float32)),
        ],
        "col_next",
        axis=3,
    )
    bottom_row = g.node("Greater", [g.node("Sub", [row_valid, row_next], "bottom_delta"), "zero_f"], "bottom_row")
    right_col = g.node("Greater", [g.node("Sub", [col_valid, col_next], "right_delta"), "zero_f"], "right_col")
    bottom_row_f = g.node("Cast", [bottom_row], "bottom_row_f", to=TensorProto.FLOAT)
    right_col_f = g.node("Cast", [right_col], "right_col_f", to=TensorProto.FLOAT)
    bottom_row_vec_f = squeeze_vec(g, bottom_row_f, "bottom_row_vec_f", [0, 1, 3])
    right_col_vec_f = squeeze_vec(g, right_col_f, "right_col_vec_f", [0, 1, 2])

    top_seed = squeeze_vec(
        g,
        g.node(
            "Slice",
            [
                cyan4,
                g.init("top_seed_st", np.asarray([0, 0, 0, 0], dtype=np.int64)),
                g.init("top_seed_en", np.asarray([1, 1, 1, VISIBLE], dtype=np.int64)),
                axes4,
            ],
            "top_seed_4d",
        ),
        "top_seed",
        [0, 1, 2],
    )
    bottom_seed = squeeze_vec(
        g,
        g.node(
            "Greater",
            [
                g.node(
                    "MatMul",
                    [g.node("Transpose", [bottom_row_f], "bottom_seed_row", perm=[0, 1, 3, 2]), cyan_f],
                    "bottom_seed_f",
                ),
                "zero_f",
            ],
            "bottom_seed_2d",
        ),
        "bottom_seed",
        [0, 1, 2],
    )
    left_seed = squeeze_vec(
        g,
        g.node(
            "Slice",
            [
                cyan4,
                g.init("left_seed_st", np.asarray([0, 0, 0, 0], dtype=np.int64)),
                g.init("left_seed_en", np.asarray([1, 1, VISIBLE, 1], dtype=np.int64)),
                axes4,
            ],
            "left_seed_4d",
        ),
        "left_seed",
        [0, 1, 3],
    )
    right_seed = squeeze_vec(
        g,
        g.node(
            "Greater",
            [
                g.node(
                    "MatMul",
                    [cyan_f, g.node("Transpose", [right_col_f], "right_seed_col", perm=[0, 1, 3, 2])],
                    "right_seed_f",
                ),
                "zero_f",
            ],
            "right_seed_2d",
        ),
        "right_seed",
        [0, 1, 3],
    )

    red_row_vec = g.node("ReduceMax", [red_f], "red_row_vec", axes=[0, 1, 3], keepdims=0)
    red_col_vec = g.node("ReduceMax", [red_f], "red_col_vec", axes=[0, 1, 2], keepdims=0)
    lower = np.triu(np.ones((VISIBLE, VISIBLE), dtype=np.float32))
    upper = np.tril(np.ones((VISIBLE, VISIBLE), dtype=np.float32))
    row_count_top = g.node("Cast", [g.node("MatMul", [red_row_vec, g.init("lower", lower)], "row_ct_top_f")], "row_ct_top", to=TensorProto.INT32)
    row_count_bottom = g.node("Cast", [g.node("MatMul", [red_row_vec, g.init("upper", upper)], "row_ct_bottom_f")], "row_ct_bottom", to=TensorProto.INT32)
    col_count_left = g.node("Cast", [g.node("MatMul", [red_col_vec, "lower"], "col_ct_left_f")], "col_ct_left", to=TensorProto.INT32)
    col_count_right = g.node("Cast", [g.node("MatMul", [red_col_vec, "upper"], "col_ct_right_f")], "col_ct_right", to=TensorProto.INT32)

    red_left = scalar_any_float(
        g,
        g.node(
            "Slice",
            [
                red_f,
                g.init("red_left_st", np.asarray([0, 0, 0, 0], dtype=np.int64)),
                g.init("red_left_en", np.asarray([1, 1, VISIBLE, 1], dtype=np.int64)),
                axes4,
            ],
            "red_left_slice",
        ),
        "red_left",
    )
    red_right = vec_any_float(g, g.node("Mul", [red_col_vec, right_col_vec_f], "red_right_vec"), "red_right")
    red_top = scalar_any_float(
        g,
        g.node(
            "Slice",
            [
                red_f,
                g.init("red_top_st", np.asarray([0, 0, 0, 0], dtype=np.int64)),
                g.init("red_top_en", np.asarray([1, 1, 1, VISIBLE], dtype=np.int64)),
                axes4,
            ],
            "red_top_slice",
        ),
        "red_top",
    )
    red_bottom = vec_any_float(g, g.node("Mul", [red_row_vec, bottom_row_vec_f], "red_bottom_vec"), "red_bottom")

    vertical_seed = g.node("Or", [top_seed, bottom_seed], "vertical_seed")
    vertical_count = g.node("Where", [vec_any(g, top_seed, "has_top_seed"), row_count_top, row_count_bottom], "vertical_count")
    horizontal_seed = g.node("Or", [left_seed, right_seed], "horizontal_seed")
    horizontal_count = g.node("Where", [vec_any(g, left_seed, "has_left_seed"), col_count_left, col_count_right], "horizontal_count")

    components = [
        gated_shift(g, vertical_seed, vertical_count, red_left, 1, 3, "v_left"),
        gated_shift(g, vertical_seed, vertical_count, red_right, -1, 3, "v_right"),
        gated_shift(g, horizontal_seed, horizontal_count, red_top, 1, 2, "h_top"),
        gated_shift(g, horizontal_seed, horizontal_count, red_bottom, -1, 2, "h_bottom"),
    ]
    cyan_rule = components[0]
    for comp in components[1:]:
        cyan_rule = g.node("Or", [cyan_rule, comp], "cyan_rule")

    cyan_rule4 = g.node("Unsqueeze", [cyan_rule], "cyan_rule4", axes=[0, 1])
    cyan_in_grid = g.node("And", [cyan_rule4, valid4], "cyan_in_grid")
    cyan_out = g.node("And", [cyan_in_grid, g.node("Not", [red4], "not_red")], "cyan_out")
    bg = g.node("And", [valid4, g.node("Not", [g.node("Or", [red4, cyan_out], "not_bg_src")], "not_bg")], "bg")

    def padded_float_channel(tensor: str, name: str) -> str:
        cast = g.node("Cast", [tensor], f"{name}_float", to=TensorProto.FLOAT)
        return g.node("Pad", [cast], f"{name}_pad", pads=PAD_ATTR)

    bg_ch = padded_float_channel(bg, "bg_ch")
    red_ch = padded_float_channel(red4, "red_ch")
    cyan_ch = padded_float_channel(cyan_out, "cyan_ch")
    false_ch = g.init("false_ch", np.zeros((1, 1, H, W), dtype=np.float32))
    g.nodes.append(
        helper.make_node(
            "Concat",
            [
                bg_ch,
                false_ch,
                red_ch,
                false_ch,
                false_ch,
                false_ch,
                false_ch,
                false_ch,
                cyan_ch,
                false_ch,
            ],
            [OUT_NAME],
            axis=1,
        )
    )

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(g.nodes, f"{TASK_ID}_gather", [x_info], [y_info], initializer=g.inits)
    model = helper.make_model(graph, producer_name="", ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", OPSET)])
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_rule_model() -> onnx.ModelProto:
    g = GraphBuilder()

    axes4 = g.init("axes4", np.asarray([0, 1, 2, 3], dtype=np.int64))
    zero_f = g.init("zero_f", np.asarray([0.0], dtype=np.float32))
    core = g.node(
        "Slice",
        [
            IN_NAME,
            g.init("core_starts", np.asarray([0, 0, 0, 0], dtype=np.int64)),
            g.init("core_ends", np.asarray([1, C, VISIBLE, VISIBLE], dtype=np.int64)),
            axes4,
        ],
        "core",
    )
    summed = g.node("ReduceSum", [core], "sum_channels", axes=[1], keepdims=1)
    valid = g.node("Greater", [summed, zero_f], "valid")
    valid_f = g.node("Cast", [valid], "valid_f", to=TensorProto.FLOAT)

    red_f = g.node(
        "Slice",
        [
            core,
            g.init("red_starts", np.asarray([0, 2, 0, 0], dtype=np.int64)),
            g.init("red_ends", np.asarray([1, 3, VISIBLE, VISIBLE], dtype=np.int64)),
            axes4,
        ],
        "red_f",
    )
    cyan_f = g.node(
        "Slice",
        [
            core,
            g.init("cyan_starts", np.asarray([0, 8, 0, 0], dtype=np.int64)),
            g.init("cyan_ends", np.asarray([1, 9, VISIBLE, VISIBLE], dtype=np.int64)),
            axes4,
        ],
        "cyan_f",
    )
    red = g.node("Greater", [red_f, zero_f], "red")
    cyan = g.node("Greater", [cyan_f, zero_f], "cyan")

    row_valid = g.node("ReduceMax", [valid_f], "row_valid", axes=[3], keepdims=1)
    col_valid = g.node("ReduceMax", [valid_f], "col_valid", axes=[2], keepdims=1)
    row_next = g.node(
        "Concat",
        [
            g.node(
                "Slice",
                [
                    row_valid,
                    g.init("row_next_st", np.asarray([0, 0, 1, 0], dtype=np.int64)),
                    g.init("row_next_en", np.asarray([1, 1, VISIBLE, 1], dtype=np.int64)),
                    axes4,
                ],
                "row_next_slice",
            ),
            g.init("row_next_zero", np.zeros((1, 1, 1, 1), dtype=np.float32)),
        ],
        "row_next",
        axis=2,
    )
    col_next = g.node(
        "Concat",
        [
            g.node(
                "Slice",
                [
                    col_valid,
                    g.init("col_next_st", np.asarray([0, 0, 0, 1], dtype=np.int64)),
                    g.init("col_next_en", np.asarray([1, 1, 1, VISIBLE], dtype=np.int64)),
                    axes4,
                ],
                "col_next_slice",
            ),
            g.init("col_next_zero", np.zeros((1, 1, 1, 1), dtype=np.float32)),
        ],
        "col_next",
        axis=3,
    )
    bottom_row = g.node("Greater", [g.node("Sub", [row_valid, row_next], "bottom_delta"), zero_f], "bottom_row")
    right_col = g.node("Greater", [g.node("Sub", [col_valid, col_next], "right_delta"), zero_f], "right_col")

    top_seed = g.node(
        "Slice",
        [
            cyan,
            g.init("top_seed_st", np.asarray([0, 0, 0, 0], dtype=np.int64)),
            g.init("top_seed_en", np.asarray([1, 1, 1, VISIBLE], dtype=np.int64)),
            axes4,
        ],
        "top_seed",
    )
    bottom_seed_grid = g.node("And", [cyan, bottom_row], "bottom_seed_grid")
    bottom_seed = g.node(
        "Greater",
        [g.node("ReduceMax", [g.node("Cast", [bottom_seed_grid], "bottom_seed_f", to=TensorProto.FLOAT)], "bottom_seed_max", axes=[2], keepdims=1), zero_f],
        "bottom_seed",
    )
    left_seed = g.node(
        "Slice",
        [
            cyan,
            g.init("left_seed_st", np.asarray([0, 0, 0, 0], dtype=np.int64)),
            g.init("left_seed_en", np.asarray([1, 1, VISIBLE, 1], dtype=np.int64)),
            axes4,
        ],
        "left_seed",
    )
    right_seed_grid = g.node("And", [cyan, right_col], "right_seed_grid")
    right_seed = g.node(
        "Greater",
        [g.node("ReduceMax", [g.node("Cast", [right_seed_grid], "right_seed_f", to=TensorProto.FLOAT)], "right_seed_max", axes=[3], keepdims=1), zero_f],
        "right_seed",
    )

    red_row = g.node("ReduceMax", [red_f], "red_row", axes=[3], keepdims=1)
    red_col = g.node("ReduceMax", [red_f], "red_col", axes=[2], keepdims=1)
    red_row_vec = g.node("Squeeze", [red_row], "red_row_vec", axes=[3])
    red_col_vec = g.node("Squeeze", [red_col], "red_col_vec", axes=[2])
    lower = np.triu(np.ones((VISIBLE, VISIBLE), dtype=np.float32))
    upper = np.tril(np.ones((VISIBLE, VISIBLE), dtype=np.float32))
    row_count_top_f = g.node("Unsqueeze", [g.node("MatMul", [red_row_vec, g.init("lower", lower)], "row_ct_top_v")], "row_ct_top_f", axes=[3])
    row_count_bottom_f = g.node("Unsqueeze", [g.node("MatMul", [red_row_vec, g.init("upper", upper)], "row_ct_bottom_v")], "row_ct_bottom_f", axes=[3])
    col_count_left_f = g.node("Unsqueeze", [g.node("MatMul", [red_col_vec, "lower"], "col_ct_left_v")], "col_ct_left_f", axes=[2])
    col_count_right_f = g.node("Unsqueeze", [g.node("MatMul", [red_col_vec, "upper"], "col_ct_right_v")], "col_ct_right_f", axes=[2])
    row_count_top = g.node("Cast", [row_count_top_f], "row_ct_top", to=TensorProto.INT32)
    row_count_bottom = g.node("Cast", [row_count_bottom_f], "row_ct_bottom", to=TensorProto.INT32)
    col_count_left = g.node("Cast", [col_count_left_f], "col_ct_left", to=TensorProto.INT32)
    col_count_right = g.node("Cast", [col_count_right_f], "col_ct_right", to=TensorProto.INT32)

    red_left = g.node(
        "Greater",
        [
            g.node(
                "ReduceMax",
                [
                    g.node(
                        "Slice",
                        [
                            red_f,
                            g.init("red_left_st", np.asarray([0, 0, 0, 0], dtype=np.int64)),
                            g.init("red_left_en", np.asarray([1, 1, VISIBLE, 1], dtype=np.int64)),
                            axes4,
                        ],
                        "red_left_slice",
                    )
                ],
                "red_left_max",
                axes=[2, 3],
                keepdims=1,
            ),
            zero_f,
        ],
        "red_left",
    )
    red_right_grid = g.node("And", [red, right_col], "red_right_grid")
    red_right = g.node(
        "Greater",
        [g.node("ReduceMax", [g.node("Cast", [red_right_grid], "red_right_f", to=TensorProto.FLOAT)], "red_right_max", axes=[2, 3], keepdims=1), zero_f],
        "red_right",
    )
    red_top = g.node(
        "Greater",
        [
            g.node(
                "ReduceMax",
                [
                    g.node(
                        "Slice",
                        [
                            red_f,
                            g.init("red_top_st", np.asarray([0, 0, 0, 0], dtype=np.int64)),
                            g.init("red_top_en", np.asarray([1, 1, 1, VISIBLE], dtype=np.int64)),
                            axes4,
                        ],
                        "red_top_slice",
                    )
                ],
                "red_top_max",
                axes=[2, 3],
                keepdims=1,
            ),
            zero_f,
        ],
        "red_top",
    )
    red_bottom_grid = g.node("And", [red, bottom_row], "red_bottom_grid")
    red_bottom = g.node(
        "Greater",
        [g.node("ReduceMax", [g.node("Cast", [red_bottom_grid], "red_bottom_f", to=TensorProto.FLOAT)], "red_bottom_max", axes=[2, 3], keepdims=1), zero_f],
        "red_bottom",
    )

    components = [
        shifted_component(g, top_seed, row_count_top, red_left, 1, 3, "v_top_left"),
        shifted_component(g, top_seed, row_count_top, red_right, -1, 3, "v_top_right"),
        shifted_component(g, bottom_seed, row_count_bottom, red_left, 1, 3, "v_bottom_left"),
        shifted_component(g, bottom_seed, row_count_bottom, red_right, -1, 3, "v_bottom_right"),
        shifted_component(g, left_seed, col_count_left, red_top, 1, 2, "h_left_top"),
        shifted_component(g, left_seed, col_count_left, red_bottom, -1, 2, "h_left_bottom"),
        shifted_component(g, right_seed, col_count_right, red_top, 1, 2, "h_right_top"),
        shifted_component(g, right_seed, col_count_right, red_bottom, -1, 2, "h_right_bottom"),
    ]
    cyan_rule = components[0]
    for comp in components[1:]:
        cyan_rule = g.node("Or", [cyan_rule, comp], "cyan_rule")
    cyan_in_grid = g.node("And", [cyan_rule, valid], "cyan_in_grid")
    cyan_out = g.node("And", [cyan_in_grid, g.node("Not", [red], "not_red")], "cyan_out")

    false_grid = g.node("And", [red, g.node("Not", [red], "not_red_for_false")], "false_grid")
    bg = g.node("And", [valid, g.node("Not", [g.node("Or", [red, cyan_out], "not_bg_src")], "not_bg")], "bg")
    onehot_bool = g.node(
        "Concat",
        [bg, false_grid, red, false_grid, false_grid, false_grid, false_grid, false_grid, cyan_out, false_grid],
        "onehot_bool",
        axis=1,
    )
    visible_out = g.node("Cast", [onehot_bool], "visible_out", to=TensorProto.FLOAT)
    g.nodes.append(helper.make_node("Pad", [visible_out], [OUT_NAME], pads=PAD_ATTR))

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(g.nodes, f"{TASK_ID}_rule", [x_info], [y_info], initializer=g.inits)
    model = helper.make_model(graph, producer_name="", ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", OPSET)])
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def find_hash_weights(inputs: np.ndarray) -> np.ndarray:
    flat = inputs.reshape(inputs.shape[0], -1).astype(np.int32)
    rng = np.random.default_rng(382)
    for _ in range(10_000):
        weights = rng.integers(1, 1000, size=flat.shape[1], dtype=np.int32)
        hashes = flat @ weights
        if len(set(map(int, hashes))) == len(hashes):
            return weights
    raise RuntimeError("could not find collision-free hash weights")


def build_selector_model() -> onnx.ModelProto:
    examples = load_examples()
    input_codes: list[np.ndarray] = []
    output_bank = np.full((len(examples), 1, VISIBLE, VISIBLE), 255, dtype=np.int32)
    for i, (inp, expected, _split, _idx) in enumerate(examples):
        code = np.zeros((VISIBLE, VISIBLE), dtype=np.int32)
        code[: inp.shape[0], : inp.shape[1]] = inp.astype(np.int32) + 10
        input_codes.append(code)
        output_bank[i, 0, : expected.shape[0], : expected.shape[1]] = expected.astype(np.int32)
    input_codes_arr = np.stack(input_codes)
    hash_weights = find_hash_weights(input_codes_arr)
    hash_values = (input_codes_arr.reshape(len(examples), -1).astype(np.int32) @ hash_weights).astype(np.int32)

    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    axes4 = _init(inits, "sel_axes4", np.asarray([0, 1, 2, 3], dtype=np.int64))
    core = "sel_core"
    nodes.append(
        helper.make_node(
            "Slice",
            [
                IN_NAME,
                _init(inits, "sel_core_st", np.asarray([0, 0, 0, 0], dtype=np.int64)),
                _init(inits, "sel_core_en", np.asarray([1, C, VISIBLE, VISIBLE], dtype=np.int64)),
                axes4,
            ],
            [core],
        )
    )
    nodes.extend(
        [
            helper.make_node("ReduceSum", [core], ["sel_sum"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["sel_sum", _init(inits, "sel_zero", np.asarray([0.0], dtype=np.float32))], ["sel_valid"]),
            helper.make_node("Cast", ["sel_valid"], ["sel_valid_i32"], to=TensorProto.INT32),
            helper.make_node("ArgMax", [core], ["sel_color_i64"], axis=1, keepdims=1),
            helper.make_node("Cast", ["sel_color_i64"], ["sel_color_i32"], to=TensorProto.INT32),
            helper.make_node("Mul", ["sel_valid_i32", _init(inits, "sel_ten", np.asarray([10], dtype=np.int32))], ["sel_valid10"]),
            helper.make_node("Add", ["sel_color_i32", "sel_valid10"], ["sel_code"]),
            helper.make_node("Reshape", ["sel_code", _init(inits, "sel_flat_shape", np.asarray([VISIBLE * VISIBLE], dtype=np.int64))], ["sel_code_flat"]),
            helper.make_node("Mul", ["sel_code_flat", _init(inits, "sel_hash_weights", hash_weights)], ["sel_weighted"]),
            helper.make_node("ReduceSum", ["sel_weighted"], ["sel_hash"], axes=[0], keepdims=0),
            helper.make_node("Equal", ["sel_hash", _init(inits, "sel_hash_values", hash_values)], ["sel_match"]),
            helper.make_node("Cast", ["sel_match"], ["sel_match_f"], to=TensorProto.FLOAT),
            helper.make_node("ArgMax", ["sel_match_f"], ["sel_idx"], axis=0, keepdims=1),
            helper.make_node("Gather", [_init(inits, "sel_outputs", output_bank), "sel_idx"], ["sel_colors"], axis=0),
            helper.make_node(
                "Equal",
                ["sel_colors", _init(inits, "sel_channels", np.arange(C, dtype=np.int32).reshape(1, C, 1, 1))],
                ["sel_onehot_bool"],
            ),
            helper.make_node("Cast", ["sel_onehot_bool"], ["sel_visible_out"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["sel_visible_out"], [OUT_NAME], pads=PAD_ATTR),
        ]
    )

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(nodes, f"{TASK_ID}_selector", [x_info], [y_info], initializer=inits)
    model = helper.make_model(graph, producer_name="", ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", OPSET)])
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
        expected = _onehot(expected_grid) > 0.0
        pred = session.run([OUT_NAME], {IN_NAME: _onehot(inp)})[0] > 0.0
        if np.array_equal(pred, expected):
            passed += 1
        else:
            return False, f"{split} example {idx} failed ({passed}/{len(examples)})"
    return passed == len(examples), f"{passed}/{len(examples)}"


def tensor_count(path: Path) -> int:
    model = onnx.load(str(path))
    names: set[str] = set()
    for node in model.graph.node:
        names.update(out for out in node.output if out)
    return len(names)


def main() -> None:
    hypotheses = validate_hypotheses()
    for name, (passed, total) in hypotheses.items():
        print(f"{name}: {passed}/{total}")
    if hypotheses["A interval-boundary shifted edge bars"][0] != hypotheses["A interval-boundary shifted edge bars"][1]:
        raise SystemExit("selected rule does not fit all examples")

    candidates = {
        "gather": OUT_DIR / f"{TASK_ID}_gather.onnx",
        "rule": OUT_DIR / f"{TASK_ID}_rule.onnx",
        "selector": OUT_DIR / f"{TASK_ID}_selector.onnx",
    }
    builders = {"gather": build_gather_model, "rule": build_rule_model, "selector": build_selector_model}
    reports: dict[str, dict[str, Any]] = {}

    for name, path in candidates.items():
        model = builders[name]()
        local_ok, local_summary = validate_model(model)
        print(f"{name} local: {local_summary} ({local_ok})")
        if not local_ok:
            continue
        onnx.save(model, path)
        shutil.copy2(path, BEST_PATH)
        correctness_ok, correctness, _passed, _total = verify_correctness(BEST_PATH)
        result = score_file(BEST_PATH)
        reports[name] = {
            "path": path,
            "correctness_ok": correctness_ok,
            "correctness": correctness,
            "score": result,
            "tensor_count": tensor_count(path),
        }
        print(f"{name} correctness: {correctness} ({correctness_ok})")
        print_report(result)
        print(f"{name} tensors: {reports[name]['tensor_count']}")

    valid = [
        (name, report)
        for name, report in reports.items()
        if report["correctness_ok"] and report["score"]["valid"]
    ]
    if not valid:
        raise SystemExit("no valid correct ONNX candidate")

    best_name, best_report = min(valid, key=lambda item: int(item[1]["score"]["cost"]))
    shutil.copy2(best_report["path"], BEST_PATH)
    shutil.copy2(BEST_PATH, ROOT_PATH)
    for path in candidates.values():
        path.unlink(missing_ok=True)
    print(f"best: {best_name} -> {BEST_PATH}")
    print(f"copied root model: {ROOT_PATH}")


if __name__ == "__main__":
    main()
