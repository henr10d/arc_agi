"""ONNX for ARC task222: keep only the solid monochrome rectangle.

Task rule: each 16x16 input contains random colored noise plus one salient
filled axis-aligned rectangle of a single non-background color. The output
keeps that rectangle at the same coordinates, size, and color, and changes
every other in-grid cell to background 0. All provided rectangles are at least
2x3 or 3x2, and no noise-only foreground region forms such a solid patch.

ONNX approach: slice the 16x16 foreground channels, use grouped Conv to find
solid same-color 2x3 and 3x2 patches, paint their spatial union with
ConvTranspose, then multiply the compact mask by the original foreground
one-hot slice before padding back to the 30x30 competition tensor.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task222"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task222.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
FG = 9
H = W = 30
N = 16
PAD = H - N
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10

RECT_SIZES: tuple[tuple[int, int], ...] = (
    (2, 5),
    (5, 2),
    (2, 6),
    (6, 2),
    (2, 7),
    (7, 2),
    (2, 8),
    (8, 2),
    (3, 3),
    (3, 4),
    (4, 3),
    (3, 5),
    (5, 3),
    (4, 4),
)

PATCH_SIZES: tuple[tuple[int, int], ...] = ((2, 3), (3, 2))


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _diag_kernel(h: int, w: int) -> np.ndarray:
    kernel = np.ones((FG, 1, h, w), dtype=np.float32)
    return kernel


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference implementation: union of solid 2x3/3x2 foreground patches."""
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(arr)
    for color in range(1, C):
        mask = arr == color
        for h, w in PATCH_SIZES:
            for r in range(arr.shape[0] - h + 1):
                for c in range(arr.shape[1] - w + 1):
                    if mask[r : r + h, c : c + w].all():
                        out[r : r + h, c : c + w] = color
    return out


def _grid_to_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(arr[:H]):
        for c, color in enumerate(row[:W]):
            out[0, int(color), r, c] = 1.0
    return out


def _onehot_expected(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    return _grid_to_onehot(grid)


def _make_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
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


def _prefix(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto]) -> str:
    axes = _i64(inits, [0, 1, 2, 3], "axes")
    start_fg = _i64(inits, [0, 1, 0, 0], "start_fg")
    end_fg = _i64(inits, [1, C, N, N], "end_fg")
    zero = _f32(inits, [0.0], "zero")
    nodes.append(helper.make_node("Slice", [IN_NAME, start_fg, end_fg, axes], ["fg"]))
    return zero


def build_sum_model() -> onnx.ModelProto:
    """Build a compact float-sum graph; this was kept for scoring comparison."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    zero = _prefix(nodes, inits)
    painted: list[str] = []

    for idx, (h, w) in enumerate(RECT_SIZES):
        area = _f32(inits, [float(h * w) - 0.5], f"area{idx}")
        weight = _f32(inits, _diag_kernel(h, w), f"w{idx}")
        conv = f"conv{idx}"
        hit = f"hit{idx}"
        hitf = f"hitf{idx}"
        paint = f"paint{idx}"
        nodes.extend(
            [
                helper.make_node("Conv", ["fg", weight], [conv], group=FG),
                helper.make_node("Greater", [conv, area], [hit]),
                helper.make_node("Cast", [hit], [hitf], to=TensorProto.FLOAT),
                helper.make_node("ConvTranspose", [hitf, weight], [paint], group=FG),
            ]
        )
        painted.append(paint)

    nodes.extend(
        [
            helper.make_node("Sum", painted, ["sum_fg"]),
            helper.make_node("Greater", ["sum_fg", zero], ["fg_mask"]),
            helper.make_node("Cast", ["fg_mask"], ["fg_out"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", ["fg_out"], ["any_fg"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["any_fg", zero], ["has_fg"]),
            helper.make_node("Not", ["has_fg"], ["bg_bool"]),
            helper.make_node("Cast", ["bg_bool"], ["bg"], to=TensorProto.FLOAT),
            helper.make_node("Concat", ["bg", "fg_out"], ["out16"], axis=1),
            helper.make_node("Pad", ["out16"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]),
        ]
    )
    return _make_model(nodes, inits, "task222_sum")


def build_or_model() -> onnx.ModelProto:
    """Build the selected graph with bool OR fusion after each paint operation."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    zero = _prefix(nodes, inits)
    masks: list[str] = []

    for idx, (h, w) in enumerate(RECT_SIZES):
        area = _f32(inits, [float(h * w) - 0.5], f"area{idx}")
        weight = _f32(inits, _diag_kernel(h, w), f"w{idx}")
        conv = f"conv{idx}"
        hit = f"hit{idx}"
        hitf = f"hitf{idx}"
        paint = f"paint{idx}"
        mask = f"mask{idx}"
        nodes.extend(
            [
                helper.make_node("Conv", ["fg", weight], [conv], group=FG),
                helper.make_node("Greater", [conv, area], [hit]),
                helper.make_node("Cast", [hit], [hitf], to=TensorProto.FLOAT),
                helper.make_node("ConvTranspose", [hitf, weight], [paint], group=FG),
                helper.make_node("Greater", [paint, zero], [mask]),
            ]
        )
        masks.append(mask)

    current = masks[0]
    for idx, mask in enumerate(masks[1:], start=1):
        merged = f"or{idx}"
        nodes.append(helper.make_node("Or", [current, mask], [merged]))
        current = merged

    nodes.extend(
        [
            helper.make_node("Cast", [current], ["fg_out"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", ["fg_out"], ["any_fg"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["any_fg", zero], ["has_fg"]),
            helper.make_node("Not", ["has_fg"], ["bg_bool"]),
            helper.make_node("Cast", ["bg_bool"], ["bg"], to=TensorProto.FLOAT),
            helper.make_node("Concat", ["bg", "fg_out"], ["out16"], axis=1),
            helper.make_node("Pad", ["out16"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]),
        ]
    )
    return _make_model(nodes, inits, "task222_or")


def build_spatial_model() -> onnx.ModelProto:
    """Build the selected low-memory graph using a single-channel spatial mask."""
    return build_patch_model(PATCH_SIZES, "task222_patch", or_reduce=True, bool_output=True)


def _or_reduce_channels(nodes: list[onnx.NodeProto], source: str, prefix: str) -> str:
    channels = [f"{prefix}_c{channel}" for channel in range(FG)]
    nodes.append(helper.make_node("Split", [source], channels, axis=1, split=[1] * FG))
    current = channels[0]
    for channel, next_name in enumerate(channels[1:], start=1):
        merged = f"{prefix}_or{channel}"
        nodes.append(helper.make_node("Or", [current, next_name], [merged]))
        current = merged
    return current


def _append_bool_output(
    nodes: list[onnx.NodeProto],
    zero: str,
    spatial_sum: str,
) -> None:
    nodes.extend(
        [
            helper.make_node("Greater", [spatial_sum, zero], ["spatial_bool"]),
            helper.make_node("Greater", ["fg", zero], ["fg_bool"]),
            helper.make_node("And", ["fg_bool", "spatial_bool"], ["fg_out_bool"]),
            helper.make_node("Not", ["spatial_bool"], ["bg_bool"]),
            helper.make_node("Concat", ["bg_bool", "fg_out_bool"], ["out16_bool"], axis=1),
            helper.make_node("Cast", ["out16_bool"], ["out16"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out16"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]),
        ]
    )


def build_adjacent_2x2_model() -> onnx.ModelProto:
    """Build a graph that keeps cells covered by adjacent same-color 2x2 hits."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    zero = _prefix(nodes, inits)
    area = _f32(inits, [3.5], "area")
    detect_weight = _f32(inits, _diag_kernel(2, 2), "dw")
    h_weight = _f32(inits, np.ones((1, 1, 2, 3), dtype=np.float32), "hpw")
    v_weight = _f32(inits, np.ones((1, 1, 3, 2), dtype=np.float32), "vpw")

    nodes.extend(
        [
            helper.make_node("Conv", ["fg", detect_weight], ["conv"], group=FG),
            helper.make_node("Greater", ["conv", area], ["hit"]),
        ]
    )

    axes = _i64(inits, [0, 1, 2, 3], "pair_axes")
    h_left_start = _i64(inits, [0, 0, 0, 0], "h_left_start")
    h_left_end = _i64(inits, [1, FG, N - 1, N - 2], "h_left_end")
    h_right_start = _i64(inits, [0, 0, 0, 1], "h_right_start")
    h_right_end = _i64(inits, [1, FG, N - 1, N - 1], "h_right_end")
    v_top_start = _i64(inits, [0, 0, 0, 0], "v_top_start")
    v_top_end = _i64(inits, [1, FG, N - 2, N - 1], "v_top_end")
    v_bot_start = _i64(inits, [0, 0, 1, 0], "v_bot_start")
    v_bot_end = _i64(inits, [1, FG, N - 1, N - 1], "v_bot_end")

    nodes.extend(
        [
            helper.make_node("Slice", ["hit", h_left_start, h_left_end, axes], ["h_left"]),
            helper.make_node("Slice", ["hit", h_right_start, h_right_end, axes], ["h_right"]),
            helper.make_node("And", ["h_left", "h_right"], ["h_pair"]),
            helper.make_node("Slice", ["hit", v_top_start, v_top_end, axes], ["v_top"]),
            helper.make_node("Slice", ["hit", v_bot_start, v_bot_end, axes], ["v_bot"]),
            helper.make_node("And", ["v_top", "v_bot"], ["v_pair"]),
        ]
    )

    h_any_bool = _or_reduce_channels(nodes, "h_pair", "h_any")
    v_any_bool = _or_reduce_channels(nodes, "v_pair", "v_any")
    nodes.extend(
        [
            helper.make_node("Cast", [h_any_bool], ["h_any"], to=TensorProto.FLOAT),
            helper.make_node("Cast", [v_any_bool], ["v_any"], to=TensorProto.FLOAT),
            helper.make_node("ConvTranspose", ["h_any", h_weight], ["h_paint"]),
            helper.make_node("ConvTranspose", ["v_any", v_weight], ["v_paint"]),
            helper.make_node("Sum", ["h_paint", "v_paint"], ["sum_spatial"]),
        ]
    )
    _append_bool_output(nodes, zero, "sum_spatial")
    return _make_model(nodes, inits, "task222_adjacent_2x2")


def build_patch_model(
    sizes: Iterable[tuple[int, int]],
    name: str,
    *,
    or_reduce: bool = False,
    bool_output: bool = False,
) -> onnx.ModelProto:
    """Build a low-memory graph from a small set of solid-patch detectors."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    zero = _prefix(nodes, inits)
    painted: list[str] = []

    for idx, (h, w) in enumerate(sizes):
        area = _f32(inits, [float(h * w) - 0.5], f"area{idx}")
        detect_weight = _f32(inits, _diag_kernel(h, w), f"dw{idx}")
        paint_weight = _f32(inits, np.ones((1, 1, h, w), dtype=np.float32), f"pw{idx}")
        conv = f"conv{idx}"
        hit = f"hit{idx}"
        hitf = f"hitf{idx}"
        any_hit = f"any_hit{idx}"
        paint = f"paint{idx}"
        nodes.extend(
            [
                helper.make_node("Conv", ["fg", detect_weight], [conv], group=FG),
                helper.make_node("Greater", [conv, area], [hit]),
            ]
        )
        if or_reduce:
            channels = [f"hit{idx}_c{channel}" for channel in range(FG)]
            nodes.append(helper.make_node("Split", [hit], channels, axis=1, split=[1] * FG))
            current = channels[0]
            for channel, next_name in enumerate(channels[1:], start=1):
                merged = f"hit{idx}_or{channel}"
                nodes.append(helper.make_node("Or", [current, next_name], [merged]))
                current = merged
            nodes.extend(
                [
                    helper.make_node("Cast", [current], [any_hit], to=TensorProto.FLOAT),
                ]
            )
        else:
            nodes.extend(
                [
                    helper.make_node("Cast", [hit], [hitf], to=TensorProto.FLOAT),
                    helper.make_node("ReduceMax", [hitf], [any_hit], axes=[1], keepdims=1),
                ]
            )
        nodes.append(helper.make_node("ConvTranspose", [any_hit, paint_weight], [paint]))
        painted.append(paint)

    if len(painted) == 1:
        spatial_sum = painted[0]
    else:
        spatial_sum = "sum_spatial"
        nodes.append(helper.make_node("Sum", painted, [spatial_sum]))

    if bool_output:
        _append_bool_output(nodes, zero, spatial_sum)
    else:
        nodes.append(helper.make_node("Greater", [spatial_sum, zero], ["spatial_bool"]))
        nodes.extend(
            [
                helper.make_node("Cast", ["spatial_bool"], ["spatial"], to=TensorProto.FLOAT),
                helper.make_node("Mul", ["fg", "spatial"], ["fg_out"]),
                helper.make_node("Not", ["spatial_bool"], ["bg_bool"]),
                helper.make_node("Cast", ["bg_bool"], ["bg"], to=TensorProto.FLOAT),
                helper.make_node("Concat", ["bg", "fg_out"], ["out16"], axis=1),
                helper.make_node("Pad", ["out16"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]),
            ]
        )
    return _make_model(nodes, inits, name)


def verify_reference() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            actual = solve(example["input"])
            expected = np.asarray(example["output"], dtype=np.int64)
            if not np.array_equal(actual, expected):
                raise AssertionError(f"reference mismatch on {split}[{idx}]")


def verify_model(model: onnx.ModelProto) -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            pred = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(example["input"])})[0]
            expected = _onehot_expected(example["output"])
            if not np.array_equal(pred > 0.0, expected > 0.0):
                raise AssertionError(f"ONNX mismatch on {split}[{idx}]")


def _save_and_score(name: str, model: onnx.ModelProto, out_dir: Path = OUT_DIR) -> dict[str, Any]:
    path = out_dir / f"{TASK_ID}_{name}.onnx"
    onnx.save(model, path)
    verify_model(model)
    return score_file(path)


def _format_result(label: str, result: dict[str, Any]) -> str:
    if not result["valid"]:
        return f"{label}: INVALID ({result['error']})"
    return (
        f"{label}: memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']:.6f}"
    )


def main() -> None:
    verify_reference()
    attempts = {
        "sum": build_sum_model(),
        "or": build_or_model(),
        "patch": build_spatial_model(),
        "patch_float": build_patch_model(PATCH_SIZES, "task222_patch_float"),
        "patch_or_reduce": build_patch_model(PATCH_SIZES, "task222_patch_or_reduce", or_reduce=True),
        "patch_bool_out": build_patch_model(
            PATCH_SIZES, "task222_patch_bool_out", or_reduce=True, bool_output=True
        ),
        "rect_spatial": build_patch_model(RECT_SIZES, "task222_rect_spatial"),
    }

    with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}_") as tmp:
        tmpdir = Path(tmp)
        results = [(name, _save_and_score(name, model, tmpdir), model) for name, model in attempts.items()]
    valid = [(name, result, model) for name, result, model in results if result["valid"]]
    if not valid:
        raise SystemExit("no valid model variants")
    best_name, best_result, best_model = min(valid, key=lambda item: int(item[1]["cost"]))
    onnx.save(best_model, BEST_PATH)

    for name, result, _model in results:
        print(_format_result(name, result))
    print(_format_result(f"best={best_name}", best_result))
    print(f"saved {BEST_PATH}")


if __name__ == "__main__":
    main()
