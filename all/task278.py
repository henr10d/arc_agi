"""ONNX for ARC task278: surround every red domino with a green frame.

Task rule: red cells are either isolated singletons or two-cell orthogonal
dominoes.  Isolated reds remain unchanged.  A horizontal red domino becomes
the center of a clipped 3x4 green rectangle, and a vertical red domino becomes
the center of a clipped 4x3 green rectangle.  The red domino cells stay red;
only black cells inside the frame turn green.  Frames are clipped to the
original grid, not to the 30x30 padded tensor.

ONNX approach: detect adjacent red pairs in the compact red channel, dilate
those pair masks into frame masks, clip by the valid input-cell mask, then
emit bool one-hot channels and cast only the final tensor to float.
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

TASK_ID = "task278"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task278.onnx"
DATA_PATH = ROOT / "data" / "task278.json"

C = 10
H = W = 30
G = 18
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    arr = np.asarray(vals, dtype=np.int64)
    for init in inits:
        if init.data_type == TensorProto.INT64 and np.array_equal(numpy_helper.to_array(init), arr):
            return init.name
    return _init(inits, arr, name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _make_model(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], name: str) -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(nodes, name, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _slice(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    data: str,
    out: str,
    starts: Sequence[int],
    ends: Sequence[int],
    axes: Sequence[int] | None = None,
) -> str:
    starts_name = _i64(inits, starts, f"{out}_starts")
    ends_name = _i64(inits, ends, f"{out}_ends")
    inputs = [data, starts_name, ends_name]
    if axes is not None:
        inputs.append(_i64(inits, axes, f"{out}_axes"))
    nodes.append(helper.make_node("Slice", inputs, [out]))
    return out


def _load_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver for the red-domino frame rule."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    red = g == 2
    h, w = g.shape

    for r in range(h):
        for c in range(w - 1):
            if red[r, c] and red[r, c + 1]:
                block = out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 3)]
                block[block == 0] = 3

    for r in range(h - 1):
        for c in range(w):
            if red[r, c] and red[r + 1, c]:
                block = out[max(0, r - 1) : min(h, r + 3), max(0, c - 1) : min(w, c + 2)]
                block[block == 0] = 3

    return out


def _components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    red = grid == 2
    seen = np.zeros(red.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in zip(*np.where(red)):
        if seen[r, c]:
            continue
        stack = [(int(r), int(c))]
        seen[r, c] = True
        cells: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            cells.append((cr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < red.shape[0] and 0 <= nc < red.shape[1] and red[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(cells)
    return comps


def solve_vertical_only(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    red = g == 2
    h, w = g.shape
    for r in range(h - 1):
        for c in range(w):
            if red[r, c] and red[r + 1, c]:
                block = out[max(0, r - 1) : min(h, r + 3), max(0, c - 1) : min(w, c + 2)]
                block[block == 0] = 3
    return out


def solve_size2_components(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    h, w = g.shape
    for cells in _components(g):
        if len(cells) != 2:
            continue
        (r0, c0), (r1, c1) = sorted(cells)
        if r0 == r1 and abs(c0 - c1) == 1:
            c = min(c0, c1)
            block = out[max(0, r0 - 1) : min(h, r0 + 2), max(0, c - 1) : min(w, c + 3)]
            block[block == 0] = 3
        elif c0 == c1 and abs(r0 - r1) == 1:
            r = min(r0, r1)
            block = out[max(0, r - 1) : min(h, r + 3), max(0, c0 - 1) : min(w, c0 + 2)]
            block[block == 0] = 3
    return out


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return onehot[0, :, : shape[0], : shape[1]].argmax(axis=0).astype(np.int64)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_reference(fn: Callable[[np.ndarray], np.ndarray]) -> dict[str, int]:
    data = _load_data()
    bad: dict[str, int] = {}
    for split in ("train", "test", "arc-gen"):
        count = 0
        for ex in data[split]:
            inp = np.asarray(ex["input"], dtype=np.int64)
            expected = np.asarray(ex["output"], dtype=np.int64)
            if not np.array_equal(fn(inp), expected):
                count += 1
        bad[split] = count
    return bad


def validate_onnx(model: onnx.ModelProto) -> int:
    data = _load_data()
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            inp = np.asarray(ex["input"], dtype=np.int64)
            expected = np.asarray(ex["output"], dtype=np.int64)
            pred_oh = _run_onnx(model, _grid_to_onehot(ex["input"]))
            pred = _onehot_to_grid(pred_oh, expected.shape)
            active = pred_oh[0, :, : expected.shape[0], : expected.shape[1]] > 0.0
            pad_active = bool((pred_oh[0, :, expected.shape[0] :, :] > 0.0).any()) or bool(
                (pred_oh[0, :, :, expected.shape[1] :] > 0.0).any()
            )
            if not np.array_equal(pred, expected) or not np.all(active.sum(axis=0) == 1) or pad_active:
                bad += 1
    return bad


def _input_channels(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto]) -> tuple[str, str, str, str]:
    black = _slice(nodes, inits, IN_NAME, "black", [0, 0, 0, 0], [1, 1, H, W])
    red = _slice(nodes, inits, IN_NAME, "red", [0, 2, 0, 0], [1, 3, H, W])
    nodes.extend(
        [
            helper.make_node("Cast", [black], ["black_b"], to=TensorProto.BOOL),
            helper.make_node("Cast", [red], ["red_b"], to=TensorProto.BOOL),
        ]
    )
    return black, red, "black_b", "red_b"


def _input_channels_crop(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], size: int) -> tuple[str, str, str, str]:
    black = _slice(nodes, inits, IN_NAME, "black", [0, 0, 0, 0], [1, 1, size, size])
    red = _slice(nodes, inits, IN_NAME, "red", [0, 2, 0, 0], [1, 3, size, size])
    nodes.extend(
        [
            helper.make_node("Cast", [black], ["black_b"], to=TensorProto.BOOL),
            helper.make_node("Cast", [red], ["red_b"], to=TensorProto.BOOL),
        ]
    )
    return black, red, "black_b", "red_b"


def _finish_bool(
    nodes: List[onnx.NodeProto],
    black_b: str,
    red_b: str,
    spread_b: str,
    name: str,
    inits: List[onnx.TensorProto],
) -> onnx.ModelProto:
    nodes.extend(
        [
            helper.make_node("And", [spread_b, black_b], ["green_b"]),
            helper.make_node("Xor", [black_b, "green_b"], ["black_out_b"]),
            helper.make_node("And", [black_b, red_b], ["zero_b"]),
            helper.make_node(
                "Concat",
                [
                    "black_out_b",
                    "zero_b",
                    red_b,
                    "green_b",
                    "zero_b",
                    "zero_b",
                    "zero_b",
                    "zero_b",
                    "zero_b",
                    "zero_b",
                ],
                ["out_b"],
                axis=1,
            ),
            helper.make_node("Cast", ["out_b"], [OUT_NAME], to=TensorProto.FLOAT),
        ]
    )
    return _make_model(nodes, inits, name)


def _finish_crop_float_pad(
    nodes: List[onnx.NodeProto],
    black_b: str,
    red_b: str,
    spread_b: str,
    name: str,
    inits: List[onnx.TensorProto],
    size: int,
) -> onnx.ModelProto:
    nodes.extend(
        [
            helper.make_node("And", [spread_b, black_b], ["green_b"]),
            helper.make_node("Xor", [black_b, "green_b"], ["black_out_b"]),
            helper.make_node("And", [black_b, red_b], ["zero_b"]),
            helper.make_node("Concat", ["black_out_b", "zero_b", red_b, "green_b"], ["out4_b"], axis=1),
            helper.make_node("Cast", ["out4_b"], ["out4_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out4_f"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, C - 4, H - size, W - size],
            ),
        ]
    )
    return _make_model(nodes, inits, name)


def _pair_masks_bool(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], red_b: str) -> tuple[str, str]:
    h_left = _slice(nodes, inits, red_b, "h_left", [0, 0, 0, 0], [1, 1, H, W - 1])
    h_right = _slice(nodes, inits, red_b, "h_right", [0, 0, 0, 1], [1, 1, H, W])
    v_top = _slice(nodes, inits, red_b, "v_top", [0, 0, 0, 0], [1, 1, H - 1, W])
    v_bottom = _slice(nodes, inits, red_b, "v_bottom", [0, 0, 1, 0], [1, 1, H, W])
    nodes.extend(
        [
            helper.make_node("And", [h_left, h_right], ["h_pair_b"]),
            helper.make_node("And", [v_top, v_bottom], ["v_pair_b"]),
        ]
    )
    return "h_pair_b", "v_pair_b"


def _pair_masks_bool_crop(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    red_b: str,
    size: int,
) -> tuple[str, str]:
    h_left = _slice(nodes, inits, red_b, "h_left", [0, 0, 0, 0], [1, 1, size, size - 1])
    h_right = _slice(nodes, inits, red_b, "h_right", [0, 0, 0, 1], [1, 1, size, size])
    v_top = _slice(nodes, inits, red_b, "v_top", [0, 0, 0, 0], [1, 1, size - 1, size])
    v_bottom = _slice(nodes, inits, red_b, "v_bottom", [0, 0, 1, 0], [1, 1, size, size])
    nodes.extend(
        [
            helper.make_node("And", [h_left, h_right], ["h_pair_b"]),
            helper.make_node("And", [v_top, v_bottom], ["v_pair_b"]),
        ]
    )
    return "h_pair_b", "v_pair_b"


def build_maxpool_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    _black, _red, black_b, red_b = _input_channels(nodes, inits)
    h_pair_b, v_pair_b = _pair_masks_bool(nodes, inits, red_b)
    nodes.extend(
        [
            helper.make_node("Cast", [h_pair_b], ["h_pair_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", [v_pair_b], ["v_pair_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "MaxPool",
                ["h_pair_f"],
                ["h_spread_f"],
                kernel_shape=[3, 4],
                strides=[1, 1],
                pads=[1, 2, 1, 2],
            ),
            helper.make_node(
                "MaxPool",
                ["v_pair_f"],
                ["v_spread_f"],
                kernel_shape=[4, 3],
                strides=[1, 1],
                pads=[2, 1, 2, 1],
            ),
            helper.make_node("Cast", ["h_spread_f"], ["h_spread_b"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["v_spread_f"], ["v_spread_b"], to=TensorProto.BOOL),
            helper.make_node("Or", ["h_spread_b", "v_spread_b"], ["spread_b"]),
        ]
    )
    return _finish_bool(nodes, black_b, red_b, "spread_b", "task278_maxpool", inits)


def build_maxpool18_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    _black, _red, black_b, red_b = _input_channels_crop(nodes, inits, G)
    h_pair_b, v_pair_b = _pair_masks_bool_crop(nodes, inits, red_b, G)
    nodes.extend(
        [
            helper.make_node("Cast", [h_pair_b], ["h_pair_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", [v_pair_b], ["v_pair_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "MaxPool",
                ["h_pair_f"],
                ["h_spread_f"],
                kernel_shape=[3, 4],
                strides=[1, 1],
                pads=[1, 2, 1, 2],
            ),
            helper.make_node(
                "MaxPool",
                ["v_pair_f"],
                ["v_spread_f"],
                kernel_shape=[4, 3],
                strides=[1, 1],
                pads=[2, 1, 2, 1],
            ),
            helper.make_node("Cast", ["h_spread_f"], ["h_spread_b"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["v_spread_f"], ["v_spread_b"], to=TensorProto.BOOL),
            helper.make_node("Or", ["h_spread_b", "v_spread_b"], ["spread_b"]),
        ]
    )
    return _finish_crop_float_pad(nodes, black_b, red_b, "spread_b", "task278_maxpool18", inits, G)


def build_maxpool18_u8_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    _black, _red, black_b, red_b = _input_channels_crop(nodes, inits, G)
    h_pair_b, v_pair_b = _pair_masks_bool_crop(nodes, inits, red_b, G)
    nodes.extend(
        [
            helper.make_node("Cast", [h_pair_b], ["h_pair_u8"], to=TensorProto.UINT8),
            helper.make_node("Cast", [v_pair_b], ["v_pair_u8"], to=TensorProto.UINT8),
            helper.make_node(
                "MaxPool",
                ["h_pair_u8"],
                ["h_spread_u8"],
                kernel_shape=[3, 4],
                strides=[1, 1],
                pads=[1, 2, 1, 2],
            ),
            helper.make_node(
                "MaxPool",
                ["v_pair_u8"],
                ["v_spread_u8"],
                kernel_shape=[4, 3],
                strides=[1, 1],
                pads=[2, 1, 2, 1],
            ),
            helper.make_node("Cast", ["h_spread_u8"], ["h_spread_b"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["v_spread_u8"], ["v_spread_b"], to=TensorProto.BOOL),
            helper.make_node("Or", ["h_spread_b", "v_spread_b"], ["spread_b"]),
        ]
    )
    return _finish_crop_float_pad(nodes, black_b, red_b, "spread_b", "task278_maxpool18_u8", inits, G)


def build_maxpool18_f16_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    _black, _red, black_b, red_b = _input_channels_crop(nodes, inits, G)
    h_pair_b, v_pair_b = _pair_masks_bool_crop(nodes, inits, red_b, G)
    nodes.extend(
        [
            helper.make_node("Cast", [h_pair_b], ["h_pair_f16"], to=TensorProto.FLOAT16),
            helper.make_node("Cast", [v_pair_b], ["v_pair_f16"], to=TensorProto.FLOAT16),
            helper.make_node(
                "MaxPool",
                ["h_pair_f16"],
                ["h_spread_f16"],
                kernel_shape=[3, 4],
                strides=[1, 1],
                pads=[1, 2, 1, 2],
            ),
            helper.make_node(
                "MaxPool",
                ["v_pair_f16"],
                ["v_spread_f16"],
                kernel_shape=[4, 3],
                strides=[1, 1],
                pads=[2, 1, 2, 1],
            ),
            helper.make_node("Cast", ["h_spread_f16"], ["h_spread_b"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["v_spread_f16"], ["v_spread_b"], to=TensorProto.BOOL),
            helper.make_node("Or", ["h_spread_b", "v_spread_b"], ["spread_b"]),
        ]
    )
    return _finish_crop_float_pad(nodes, black_b, red_b, "spread_b", "task278_maxpool18_f16", inits, G)


def build_convtranspose_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    _black, red, black_b, red_b = _input_channels(nodes, inits)

    _init(inits, np.ones((1, 1, 1, 2), dtype=np.float32), "h_adj_w")
    _init(inits, np.ones((1, 1, 2, 1), dtype=np.float32), "v_adj_w")
    _init(inits, np.ones((1, 1, 3, 4), dtype=np.float32), "h_frame_w")
    _init(inits, np.ones((1, 1, 4, 3), dtype=np.float32), "v_frame_w")
    thresh = _f32(inits, [1.5], "pair_thresh")

    nodes.extend(
        [
            helper.make_node("Conv", [red, "h_adj_w"], ["h_sum"]),
            helper.make_node("Conv", [red, "v_adj_w"], ["v_sum"]),
            helper.make_node("Greater", ["h_sum", thresh], ["h_pair_b"]),
            helper.make_node("Greater", ["v_sum", thresh], ["v_pair_b"]),
            helper.make_node("Cast", ["h_pair_b"], ["h_pair_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["v_pair_b"], ["v_pair_f"], to=TensorProto.FLOAT),
            helper.make_node("ConvTranspose", ["h_pair_f", "h_frame_w"], ["h_spread_f"], pads=[1, 1, 1, 1]),
            helper.make_node("ConvTranspose", ["v_pair_f", "v_frame_w"], ["v_spread_f"], pads=[1, 1, 1, 1]),
            helper.make_node("Cast", ["h_spread_f"], ["h_spread_b"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["v_spread_f"], ["v_spread_b"], to=TensorProto.BOOL),
            helper.make_node("Or", ["h_spread_b", "v_spread_b"], ["spread_b"]),
        ]
    )
    return _finish_bool(nodes, black_b, red_b, "spread_b", "task278_convtranspose", inits)


def _shifted_full_mask(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    src: str,
    src_h: int,
    src_w: int,
    dr: int,
    dc: int,
    out: str,
) -> str:
    r0 = max(0, -dr)
    r1 = min(src_h, H - dr)
    c0 = max(0, -dc)
    c1 = min(src_w, W - dc)
    y0 = r0 + dr
    x0 = c0 + dc
    sliced = _slice(nodes, inits, src, f"{out}_crop", [0, 0, r0, c0], [1, 1, r1, c1])
    pads = [0, 0, y0, x0, 0, 0, H - y0 - (r1 - r0), W - x0 - (c1 - c0)]
    nodes.append(helper.make_node("Pad", [sliced], [out], pads=pads))
    return out


def _or_all(nodes: List[onnx.NodeProto], names: Sequence[str], prefix: str) -> str:
    current = names[0]
    for idx, name in enumerate(names[1:], start=1):
        out = f"{prefix}_{idx}"
        nodes.append(helper.make_node("Or", [current, name], [out]))
        current = out
    return current


def build_shift_or_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    _black, _red, black_b, red_b = _input_channels(nodes, inits)
    h_pair_b, v_pair_b = _pair_masks_bool(nodes, inits, red_b)
    nodes.extend(
        [
            helper.make_node("Cast", [h_pair_b], ["h_pair_shift_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", [v_pair_b], ["v_pair_shift_f"], to=TensorProto.FLOAT),
        ]
    )

    shifted: list[str] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1, 2):
            shifted.append(
                _shifted_full_mask(nodes, inits, "h_pair_shift_f", H, W - 1, dr, dc, f"h_shift_{dr + 1}_{dc + 1}")
            )
    for dr in (-1, 0, 1, 2):
        for dc in (-1, 0, 1):
            shifted.append(
                _shifted_full_mask(nodes, inits, "v_pair_shift_f", H - 1, W, dr, dc, f"v_shift_{dr + 1}_{dc + 1}")
            )

    nodes.extend(
        [
            helper.make_node("Max", shifted, ["spread_shift_f"]),
            helper.make_node("Cast", ["spread_shift_f"], ["spread_b"], to=TensorProto.BOOL),
        ]
    )
    return _finish_bool(nodes, black_b, red_b, "spread_b", "task278_shift_or", inits)


def print_hypothesis_diagnostics() -> None:
    hypotheses: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
        ("H1 vertical red dominoes only", solve_vertical_only),
        ("H2 maximal connected red components of size 2", solve_size2_components),
        ("H3 vertical 1x2 components only", solve_vertical_only),
        ("H4 every adjacent horizontal/vertical red domino", solve),
    ]
    for label, fn in hypotheses:
        bad = validate_reference(fn)
        print(f"{label}: train_bad={bad['train']} test_bad={bad['test']} arc_gen_bad={bad['arc-gen']}")

    data = _load_data()
    sizes: dict[int, int] = {}
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            for comp in _components(np.asarray(ex["input"], dtype=np.int64)):
                sizes[len(comp)] = sizes.get(len(comp), 0) + 1
    print(f"red component sizes: {sizes}")


def _score_candidate(label: str, build: Callable[[], onnx.ModelProto]) -> tuple[int, float, onnx.ModelProto, dict[str, Any]]:
    model = build()
    bad = validate_onnx(model)
    if bad:
        raise AssertionError(f"{label} failed {bad} examples")

    with tempfile.NamedTemporaryFile(suffix=f"_{TASK_ID}.onnx", dir=OUT_DIR, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        onnx.save(model, tmp_path)
        result = score_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not result["valid"]:
        raise AssertionError(f"{label} invalid: {result['error']}")
    assert result["cost"] is not None and result["score"] is not None
    print(
        f"{label}: nodes={len(model.graph.node)} tensors={len(model.graph.node)} "
        f"filesize={result['filesize']} memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']:.6f}"
    )
    return int(result["cost"]), float(result["score"]), model, result


def main() -> None:
    print_hypothesis_diagnostics()
    candidates = []
    for label, build in [
        ("maxpool18-f16", build_maxpool18_f16_model),
        ("maxpool18-u8", build_maxpool18_u8_model),
        ("maxpool18", build_maxpool18_model),
        ("maxpool", build_maxpool_model),
        ("convtranspose", build_convtranspose_model),
        ("shift-or", build_shift_or_model),
    ]:
        try:
            candidates.append(_score_candidate(label, build))
        except Exception as exc:
            print(f"{label}: INVALID {exc}")
    if not candidates:
        raise AssertionError("no valid candidates")
    _cost, _score, best, _result = min(candidates, key=lambda item: item[0])
    onnx.save(best, BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"nodes:   {len(best.graph.node)}")
    print(f"filesize:{result['filesize']}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
