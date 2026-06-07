"""Minimal ONNX for ARC task270: aligned marker pixels become center arms.

Task rule: the 15x15 input contains one blue pixel (color 1), one red pixel
(color 2), and isolated green/orange marker pixels (colors 3 and 7).  The red
center is kept and receives green arms; the blue center is kept and receives
orange arms.  A marker contributes an arm only when it lies in the same row or
column as its center: above/left/right/below markers become the adjacent
up/left/right/down cell around that center.  Diagonal markers disappear.

ONNX: crop only the four relevant 15x15 channels.  For each center/marker
pair, reduce the center to row/column vectors, Gather the corresponding marker
row/column, use directional 1-D MaxPool to detect markers on each side, keep arm
geometry boolean until the final channel cast, then concatenate compact float
channels before the final 30x30 pad.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task270"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task270.onnx"
DATA_PATH = ROOT / "data" / "task270.json"

C = 10
G = 15
H = W = 30
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10
DIRS = {
    "up": (-1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "down": (1, 0),
}
ORDER = ("up", "left", "right", "down")


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


def solve_aligned(grid: np.ndarray) -> np.ndarray:
    """Reference solver selected by hypothesis validation."""
    g = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(g)
    pairs = ((2, 3), (1, 7))
    for center_color, arm_color in pairs:
        centers = np.argwhere(g == center_color)
        if len(centers) != 1:
            continue
        r, c = map(int, centers[0])
        out[r, c] = center_color
        for mr, mc in np.argwhere(g == arm_color):
            mr, mc = int(mr), int(mc)
            if mc == c and mr < r and r > 0:
                out[r - 1, c] = arm_color
            elif mc == c and mr > r and r + 1 < g.shape[0]:
                out[r + 1, c] = arm_color
            elif mr == r and mc < c and c > 0:
                out[r, c - 1] = arm_color
            elif mr == r and mc > c and c + 1 < g.shape[1]:
                out[r, c + 1] = arm_color
    return out


def solve_count_order(grid: np.ndarray) -> np.ndarray:
    """H1: use marker count and fixed up/left/right/down arm order."""
    g = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(g)
    for center_color, arm_color in ((2, 3), (1, 7)):
        centers = np.argwhere(g == center_color)
        if len(centers) != 1:
            continue
        r, c = map(int, centers[0])
        out[r, c] = center_color
        for name in ORDER[: int((g == arm_color).sum())]:
            dr, dc = DIRS[name]
            if 0 <= r + dr < g.shape[0] and 0 <= c + dc < g.shape[1]:
                out[r + dr, c + dc] = arm_color
    return out


def solve_count_shapes(grid: np.ndarray) -> np.ndarray:
    """H3: count-based canonical shapes."""
    g = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(g)
    patterns = {
        1: ("up",),
        2: ("up", "down"),
        3: ORDER,
        4: ORDER,
    }
    for center_color, arm_color in ((2, 3), (1, 7)):
        centers = np.argwhere(g == center_color)
        if len(centers) != 1:
            continue
        r, c = map(int, centers[0])
        out[r, c] = center_color
        count = int((g == arm_color).sum())
        for name in patterns.get(count, ORDER):
            dr, dc = DIRS[name]
            if 0 <= r + dr < g.shape[0] and 0 <= c + dc < g.shape[1]:
                out[r + dr, c + dc] = arm_color
    return out


def _load_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _hypothesis_errors(
    data: dict[str, list[dict[str, list[list[int]]]]],
    solver: Callable[[np.ndarray], np.ndarray],
    splits: Iterable[str],
) -> int:
    bad = 0
    for split in splits:
        for ex in data[split]:
            pred = solver(np.asarray(ex["input"], dtype=np.int64))
            expected = np.asarray(ex["output"], dtype=np.int64)
            if not np.array_equal(pred, expected):
                bad += 1
    return bad


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _expected_onehot(grid: list[list[int]]) -> np.ndarray:
    return _grid_to_onehot(grid)[:, :, : len(grid), : len(grid[0])]


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def _conv_kernel(direction: str) -> np.ndarray:
    if direction == "up":
        k = np.ones((1, 1, G, 1), dtype=np.float32)
        k[0, 0, -1, 0] = 0.0
    elif direction == "down":
        k = np.ones((1, 1, G, 1), dtype=np.float32)
        k[0, 0, 0, 0] = 0.0
    elif direction == "left":
        k = np.ones((1, 1, 1, G), dtype=np.float32)
        k[0, 0, 0, -1] = 0.0
    elif direction == "right":
        k = np.ones((1, 1, 1, G), dtype=np.float32)
        k[0, 0, 0, 0] = 0.0
    else:
        raise ValueError(direction)
    return k


def _conv_pads(direction: str) -> list[int]:
    return {
        "up": [G - 1, 0, 0, 0],
        "down": [0, 0, G - 1, 0],
        "left": [0, G - 1, 0, 0],
        "right": [0, 0, 0, G - 1],
    }[direction]


def _slice_shift(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], src: str, direction: str, prefix: str) -> str:
    axes = _i64(inits, [0, 1, 2, 3], f"{prefix}_{direction}_axes")
    if direction == "up":
        starts, ends, pads = [0, 0, 1, 0], [1, 1, G, G], [0, 0, 0, 0, 0, 0, 1, 0]
    elif direction == "down":
        starts, ends, pads = [0, 0, 0, 0], [1, 1, G - 1, G], [0, 0, 1, 0, 0, 0, 0, 0]
    elif direction == "left":
        starts, ends, pads = [0, 0, 0, 1], [1, 1, G, G], [0, 0, 0, 0, 0, 0, 0, 1]
    elif direction == "right":
        starts, ends, pads = [0, 0, 0, 0], [1, 1, G, G - 1], [0, 0, 0, 1, 0, 0, 0, 0]
    else:
        raise ValueError(direction)
    starts_name = _i64(inits, starts, f"{prefix}_{direction}_starts")
    ends_name = _i64(inits, ends, f"{prefix}_{direction}_ends")
    sliced = f"{prefix}_{direction}_slice"
    shifted = f"{prefix}_{direction}_shift"
    nodes.append(helper.make_node("Slice", [src, starts_name, ends_name, axes], [sliced]))
    nodes.append(helper.make_node("Pad", [sliced], [shifted], pads=pads))
    return shifted


def _slice_shift_vec(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    src: str,
    direction: str,
    prefix: str,
) -> tuple[str, str]:
    axes = _i64(inits, [0, 1, 2, 3], f"{prefix}_{direction}_axes")
    if direction == "up":
        starts, ends, pads = [0, 0, 1, 0], [1, 1, G, 1], [0, 0, 0, 0, 0, 0, 1, 0]
    elif direction == "down":
        starts, ends, pads = [0, 0, 0, 0], [1, 1, G - 1, 1], [0, 0, 1, 0, 0, 0, 0, 0]
    elif direction == "left":
        starts, ends, pads = [0, 0, 0, 1], [1, 1, 1, G], [0, 0, 0, 0, 0, 0, 0, 1]
    elif direction == "right":
        starts, ends, pads = [0, 0, 0, 0], [1, 1, 1, G - 1], [0, 0, 0, 1, 0, 0, 0, 0]
    else:
        raise ValueError(direction)
    starts_name = _i64(inits, starts, f"{prefix}_{direction}_starts")
    ends_name = _i64(inits, ends, f"{prefix}_{direction}_ends")
    sliced = f"{prefix}_{direction}_slice"
    shifted = f"{prefix}_{direction}_shift"
    nodes.append(helper.make_node("Slice", [src, starts_name, ends_name, axes], [sliced]))
    nodes.append(helper.make_node("Pad", [sliced], [shifted], pads=pads))
    return shifted


def _arm_sum(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    center: str,
    markers: str,
    prefix: str,
) -> str:
    shifted: list[str] = []
    for direction in ORDER:
        weight = _f32(inits, _conv_kernel(direction), f"{prefix}_{direction}_w")
        count = f"{prefix}_{direction}_count"
        at_center = f"{prefix}_{direction}_at_center"
        nodes.append(helper.make_node("Conv", [markers, weight], [count], pads=_conv_pads(direction)))
        nodes.append(helper.make_node("Mul", [center, count], [at_center]))
        shifted.append(_slice_shift(nodes, inits, at_center, direction, prefix))
    arms = f"{prefix}_arms"
    nodes.append(helper.make_node("Sum", shifted, [arms]))
    return arms


def _arm_sum_projected(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    center: str,
    markers: str,
    prefix: str,
    zero: str,
) -> tuple[str, str]:
    center_row = f"{prefix}_center_row"
    center_col = f"{prefix}_center_col"
    marker_col = f"{prefix}_marker_col"
    marker_row = f"{prefix}_marker_row"
    center_col_idx = f"{prefix}_center_col_idx"
    center_row_idx = f"{prefix}_center_row_idx"
    marker_col_gather = f"{prefix}_marker_col_gather"
    marker_row_gather = f"{prefix}_marker_row_gather"
    marker_col_shape = _i64(inits, [1, 1, G, 1], f"{prefix}_marker_col_shape")
    marker_row_shape = _i64(inits, [1, 1, 1, G], f"{prefix}_marker_row_shape")
    nodes.extend(
        [
            helper.make_node("ReduceSum", [center], [center_row], axes=[3], keepdims=1),
            helper.make_node("ReduceSum", [center], [center_col], axes=[2], keepdims=1),
            helper.make_node("ArgMax", [center_col], [center_col_idx], axis=3, keepdims=0),
            helper.make_node("ArgMax", [center_row], [center_row_idx], axis=2, keepdims=0),
            helper.make_node("Gather", [markers, center_col_idx], [marker_col_gather], axis=3),
            helper.make_node("Gather", [markers, center_row_idx], [marker_row_gather], axis=2),
            helper.make_node("Reshape", [marker_col_gather, marker_col_shape], [marker_col]),
            helper.make_node("Reshape", [marker_row_gather, marker_row_shape], [marker_row]),
        ]
    )

    vertical: list[str] = []
    for direction in ("up", "down"):
        count = f"{prefix}_{direction}_count"
        at_center = f"{prefix}_{direction}_at_center"
        nodes.append(
            helper.make_node(
                "MaxPool",
                [marker_col],
                [count],
                kernel_shape=[G, 1],
                pads=_conv_pads(direction),
                strides=[1, 1],
            )
        )
        nodes.append(helper.make_node("Mul", [center_row, count], [at_center]))
        vertical.append(_slice_shift_vec(nodes, inits, at_center, direction, prefix))
    nodes.append(helper.make_node("Sum", vertical, [f"{prefix}_vvec"]))

    horizontal: list[str] = []
    for direction in ("left", "right"):
        count = f"{prefix}_{direction}_count"
        at_center = f"{prefix}_{direction}_at_center"
        nodes.append(
            helper.make_node(
                "MaxPool",
                [marker_row],
                [count],
                kernel_shape=[1, G],
                pads=_conv_pads(direction),
                strides=[1, 1],
            )
        )
        nodes.append(helper.make_node("Mul", [center_col, count], [at_center]))
        horizontal.append(_slice_shift_vec(nodes, inits, at_center, direction, prefix))
    nodes.append(helper.make_node("Sum", horizontal, [f"{prefix}_hvec"]))

    nodes.extend(
        [
            helper.make_node("Greater", [f"{prefix}_vvec", zero], [f"{prefix}_vvec_b"]),
            helper.make_node("Greater", [f"{prefix}_hvec", zero], [f"{prefix}_hvec_b"]),
            helper.make_node("Greater", [center_row, zero], [f"{prefix}_center_row_b"]),
            helper.make_node("Greater", [center_col, zero], [f"{prefix}_center_col_b"]),
            helper.make_node("And", [f"{prefix}_vvec_b", f"{prefix}_center_col_b"], [f"{prefix}_varms_b"]),
            helper.make_node("And", [f"{prefix}_center_row_b", f"{prefix}_hvec_b"], [f"{prefix}_harms_b"]),
            helper.make_node("Or", [f"{prefix}_varms_b", f"{prefix}_harms_b"], [f"{prefix}_arms_b"]),
        ]
    )
    arms = f"{prefix}_arms"
    nodes.append(helper.make_node("Cast", [f"{prefix}_arms_b"], [arms], to=TensorProto.FLOAT))
    return arms, f"{prefix}_arms_b"


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    axes = _i64(inits, [0, 1, 2, 3], "channel_axes")
    for color, name in ((1, "c1"), (2, "c2"), (3, "c3"), (7, "c7")):
        starts = _i64(inits, [0, color, 0, 0], f"{name}_starts")
        ends = _i64(inits, [1, color + 1, G, G], f"{name}_ends")
        nodes.append(helper.make_node("Slice", [IN_NAME, starts, ends, axes], [name]))

    zero_scalar = _f32(inits, 0.0, "zero_scalar")
    arm3, arm3_b = _arm_sum_projected(nodes, inits, "c2", "c3", "green", zero_scalar)
    arm7, arm7_b = _arm_sum_projected(nodes, inits, "c1", "c7", "orange", zero_scalar)

    nodes.extend(
        [
            helper.make_node("Sub", ["c1", "c1"], ["zero"]),
            helper.make_node("Greater", ["c1", zero_scalar], ["c1_b"]),
            helper.make_node("Greater", ["c2", zero_scalar], ["c2_b"]),
            helper.make_node("Or", ["c1_b", "c2_b"], ["centers_b"]),
            helper.make_node("Or", [arm3_b, arm7_b], ["arms_b"]),
            helper.make_node("Or", ["centers_b", "arms_b"], ["painted_b"]),
            helper.make_node("Not", ["painted_b"], ["bg_b"]),
            helper.make_node("Cast", ["bg_b"], ["bg"], to=TensorProto.FLOAT),
            helper.make_node(
                "Concat",
                ["bg", "c1", "c2", arm3, "zero", "zero", "zero", arm7, "zero", "zero"],
                ["out15"],
                axis=1,
            ),
            helper.make_node("Pad", ["out15"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - G, W - G]),
        ]
    )
    return _make_model(nodes, inits)


def validate_onnx(model: onnx.ModelProto, data: dict[str, list[dict[str, list[list[int]]]]]) -> int:
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            pred = _run_onnx(model, _grid_to_onehot(ex["input"]))[:, :, : len(ex["output"]), : len(ex["output"][0])]
            expected = _expected_onehot(ex["output"])
            if not np.array_equal(pred > 0.0, expected > 0.0):
                bad += 1
    return bad


def main() -> None:
    data = _load_data()
    hypotheses = [
        ("H1 count/order", solve_count_order),
        ("H2 aligned directions", solve_aligned),
        ("H3 count/shapes", solve_count_shapes),
    ]
    errors = [(name, _hypothesis_errors(data, solver, ("train",))) for name, solver in hypotheses]
    for name, bad in errors:
        print(f"{name}: train errors={bad}")
    winners = [name for name, bad in errors if bad == 0]
    if winners != ["H2 aligned directions"]:
        raise AssertionError(f"unexpected zero-error hypotheses: {winners}")

    all_bad = _hypothesis_errors(data, solve_aligned, ("train", "test", "arc-gen"))
    if all_bad:
        raise AssertionError(f"reference solver failed {all_bad} examples")

    model = build_model()
    onnx_bad = validate_onnx(model, data)
    if onnx_bad:
        raise AssertionError(f"ONNX model failed {onnx_bad} examples")

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
