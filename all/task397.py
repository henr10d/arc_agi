"""ONNX generator for NeuroGolf task397.

Task rule: the 10x10 grid contains only compact non-background 2x2 blocks.
Preserve every block. For each 2x2 block, count how many distinct
non-background colors appear in its four cells, then draw a vertical green
(color 3) rectangle directly below the block with width 2 and height equal to
that distinct-color count. The examples are arranged so these added green bars
fall on background cells.

ONNX: the best variant slices the 10x10 grid, converts one-hot cells to compact
color IDs with a 1x1 convolution, compares the four IDs in each candidate 2x2
window to derive the distinct-color thresholds, and stamps the corresponding
rows below each block with one ConvTranspose. The final 10x10 result is padded
to the competition output shape.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task397"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
C = 10
N = 10
H = W = 30
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


@dataclass(frozen=True)
class Variant:
    name: str
    build: Callable[[], onnx.ModelProto]


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Iterable[int], name: str) -> str:
    return _init(inits, np.asarray(list(vals), dtype=np.int64), name)


def _i32(inits: list[onnx.TensorProto], vals: Iterable[int], name: str) -> str:
    return _init(inits, np.asarray(list(vals), dtype=np.int32), name)


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def solve(grid: np.ndarray | list[list[int]], *, mask_background: bool = False) -> np.ndarray:
    """Reference solver for the distinct-color-height green bars."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    height, width = g.shape
    for row in range(height - 1):
        for col in range(width - 1):
            block = g[row : row + 2, col : col + 2]
            vals = block.reshape(-1)
            if np.all(vals != 0):
                length = len(set(int(v) for v in vals))
                for rr in range(row + 2, min(height, row + 2 + length)):
                    for cc in (col, col + 1):
                        if not mask_background or out[rr, cc] == 0:
                            out[rr, cc] = 3
    return out


def _distinct_kernel() -> np.ndarray:
    """Grouped Conv weights: sum each foreground color over every 2x2 block."""
    return np.ones((C - 1, 1, 2, 2), dtype=np.float32)


def _paint_kernel() -> np.ndarray:
    """ConvTranspose weights mapping four height-threshold channels to bars."""
    weight = np.zeros((4, 1, 6, 2), dtype=np.float32)
    for channel, row_offset in enumerate((2, 3, 4, 5)):
        weight[channel, 0, row_offset, 0] = 1.0
        weight[channel, 0, row_offset, 1] = 1.0
    return weight


def _green_selector() -> np.ndarray:
    selector = np.zeros((1, C, 1, 1), dtype=np.float32)
    selector[0, 3, 0, 0] = 1.0
    return selector


def _id_kernel() -> np.ndarray:
    """1x1 Conv weights turning a one-hot cell into its color index."""
    return np.arange(C, dtype=np.float32).reshape(1, C, 1, 1)


def build_model(*, mask_background: bool = False) -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    zero = _f32(inits, [0.0], "zero")
    one = _f32(inits, [1.0], "one")
    two = _f32(inits, [2.0], "two")
    three = _f32(inits, [3.0], "three")
    fg_start = _i64(inits, [0, 1, 0, 0], "fg_start")
    fg_end = _i64(inits, [1, C, N, N], "fg_end")
    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    distinct_w = _f32(inits, _distinct_kernel(), "distinct_w")
    paint_w = _f32(inits, _paint_kernel(), "paint_w")
    selector = _f32(inits, _green_selector(), "selector")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, fg_start, fg_end, axes4], ["fg"]),
            helper.make_node("Conv", ["fg", distinct_w], ["color_counts"], group=C - 1, kernel_shape=[2, 2]),
            helper.make_node("Clip", ["color_counts"], ["present"], min=0.0, max=1.0),
            helper.make_node("ReduceSum", ["present"], ["distinct"], axes=[1], keepdims=1),
        ]
    )

    if mask_background:
        bg_start = _i64(inits, [0, 0, 0, 0], "bg_start")
        bg_end = _i64(inits, [1, 1, N, N], "bg_end")
        bg_conv_w = _f32(inits, np.ones((1, 1, 2, 2), dtype=np.float32), "bg_conv_w")
        nodes.extend(
            [
                helper.make_node("Slice", [IN_NAME, bg_start, bg_end, axes4], ["bg10"]),
                helper.make_node("Conv", ["bg10", bg_conv_w], ["bg_counts"], kernel_shape=[2, 2]),
                helper.make_node("Less", ["bg_counts", one], ["is_block"]),
            ]
        )
    else:
        nodes.extend(
            [
                helper.make_node("ReduceSum", ["color_counts"], ["filled"], axes=[1], keepdims=1),
                helper.make_node("Greater", ["filled", three], ["is_block"]),
            ]
        )

    nodes.extend(
        [
            helper.make_node("Greater", ["distinct", one], ["gt1"]),
            helper.make_node("Greater", ["distinct", two], ["gt2"]),
            helper.make_node("Greater", ["distinct", three], ["gt3"]),
            helper.make_node("And", ["is_block", "gt1"], ["h2"]),
            helper.make_node("And", ["is_block", "gt2"], ["h3"]),
            helper.make_node("And", ["is_block", "gt3"], ["h4"]),
            helper.make_node("Concat", ["is_block", "h2", "h3", "h4"], ["height_bits"], axis=1),
            helper.make_node("Cast", ["height_bits"], ["height_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "ConvTranspose",
                ["height_f", paint_w],
                ["green10_f"],
                kernel_shape=[6, 2],
                pads=[0, 0, 4, 0],
            ),
        ]
    )

    if mask_background:
        nodes.extend(
            [
                helper.make_node("Greater", ["bg10", zero], ["bg10_b"]),
                helper.make_node("Greater", ["green10_f", zero], ["green10_raw"]),
                helper.make_node("And", ["green10_raw", "bg10_b"], ["green10_b"]),
                helper.make_node("Cast", ["green10_b"], ["green10_mask_f"], to=TensorProto.FLOAT),
            ]
        )
        green_for_pad = "green10_mask_f"
    else:
        green_for_pad = "green10_f"

    nodes.extend(
        [
            helper.make_node("Pad", [green_for_pad], ["green30_f"], pads=[0, 0, 0, 0, 0, 0, H - N, W - N]),
            helper.make_node("Greater", ["green30_f", zero], ["green30_b"]),
            helper.make_node("Where", ["green30_b", selector, IN_NAME], [OUT_NAME]),
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


def build_conv_id_model() -> onnx.ModelProto:
    """Best candidate: derive color IDs with a small 1x1 Conv, then compare IDs."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    x_start = _i64(inits, [0, 0, 0, 0], "conv_x_start")
    x_end = _i64(inits, [1, C, N, N], "conv_x_end")
    axes23 = _i64(inits, [2, 3], "conv_axes23")
    s00 = _i64(inits, [0, 0], "conv_s00")
    e99 = _i64(inits, [N - 1, N - 1], "conv_e99")
    s01 = _i64(inits, [0, 1], "conv_s01")
    e9n = _i64(inits, [N - 1, N], "conv_e9n")
    s10 = _i64(inits, [1, 0], "conv_s10")
    en9 = _i64(inits, [N, N - 1], "conv_en9")
    s11 = _i64(inits, [1, 1], "conv_s11")
    enn = _i64(inits, [N, N], "conv_enn")
    zero_f = _f32(inits, [0.0], "conv_zero_f")
    zero_i = _i32(inits, [0], "conv_zero_i")
    id_w = _f32(inits, _id_kernel(), "conv_id_w")
    paint_w = _f32(inits, _paint_kernel(), "conv_paint_w")
    selector = _f32(inits, _green_selector(), "conv_selector")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, x_start, x_end], ["conv_x10"]),
            helper.make_node("Conv", ["conv_x10", id_w], ["conv_ids_f"], kernel_shape=[1, 1]),
            helper.make_node("Cast", ["conv_ids_f"], ["conv_ids"], to=TensorProto.INT32),
            helper.make_node("Slice", ["conv_ids", s00, e99, axes23], ["conv_tl"]),
            helper.make_node("Slice", ["conv_ids", s01, e9n, axes23], ["conv_tr"]),
            helper.make_node("Slice", ["conv_ids", s10, en9, axes23], ["conv_bl"]),
            helper.make_node("Slice", ["conv_ids", s11, enn, axes23], ["conv_br"]),
            helper.make_node("Greater", ["conv_tl", zero_i], ["conv_ntl"]),
            helper.make_node("Greater", ["conv_tr", zero_i], ["conv_ntr"]),
            helper.make_node("Greater", ["conv_bl", zero_i], ["conv_nbl"]),
            helper.make_node("Greater", ["conv_br", zero_i], ["conv_nbr"]),
            helper.make_node("And", ["conv_ntl", "conv_ntr"], ["conv_ntop"]),
            helper.make_node("And", ["conv_nbl", "conv_nbr"], ["conv_nbot"]),
            helper.make_node("And", ["conv_ntop", "conv_nbot"], ["conv_is_block"]),
            helper.make_node("Equal", ["conv_tr", "conv_tl"], ["conv_tr_tl_eq"]),
            helper.make_node("Not", ["conv_tr_tl_eq"], ["conv_u2"]),
            helper.make_node("Equal", ["conv_bl", "conv_tl"], ["conv_bl_tl_eq"]),
            helper.make_node("Equal", ["conv_bl", "conv_tr"], ["conv_bl_tr_eq"]),
            helper.make_node("Or", ["conv_bl_tl_eq", "conv_bl_tr_eq"], ["conv_bl_seen"]),
            helper.make_node("Not", ["conv_bl_seen"], ["conv_u3"]),
            helper.make_node("Equal", ["conv_br", "conv_tl"], ["conv_br_tl_eq"]),
            helper.make_node("Equal", ["conv_br", "conv_tr"], ["conv_br_tr_eq"]),
            helper.make_node("Equal", ["conv_br", "conv_bl"], ["conv_br_bl_eq"]),
            helper.make_node("Or", ["conv_br_tl_eq", "conv_br_tr_eq"], ["conv_br_seen_a"]),
            helper.make_node("Or", ["conv_br_seen_a", "conv_br_bl_eq"], ["conv_br_seen"]),
            helper.make_node("Not", ["conv_br_seen"], ["conv_u4"]),
            helper.make_node("Or", ["conv_u2", "conv_u3"], ["conv_h2_a"]),
            helper.make_node("Or", ["conv_h2_a", "conv_u4"], ["conv_h2_any"]),
            helper.make_node("And", ["conv_u2", "conv_u3"], ["conv_p23"]),
            helper.make_node("And", ["conv_u2", "conv_u4"], ["conv_p24"]),
            helper.make_node("And", ["conv_u3", "conv_u4"], ["conv_p34"]),
            helper.make_node("Or", ["conv_p23", "conv_p24"], ["conv_h3_a"]),
            helper.make_node("Or", ["conv_h3_a", "conv_p34"], ["conv_h3_any"]),
            helper.make_node("And", ["conv_p23", "conv_u4"], ["conv_h4_any"]),
            helper.make_node("And", ["conv_is_block", "conv_h2_any"], ["conv_h2"]),
            helper.make_node("And", ["conv_is_block", "conv_h3_any"], ["conv_h3"]),
            helper.make_node("And", ["conv_is_block", "conv_h4_any"], ["conv_h4"]),
            helper.make_node(
                "Concat",
                ["conv_is_block", "conv_h2", "conv_h3", "conv_h4"],
                ["conv_height_bits"],
                axis=1,
            ),
            helper.make_node("Cast", ["conv_height_bits"], ["conv_height_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "ConvTranspose",
                ["conv_height_f", paint_w],
                ["conv_green10_f"],
                kernel_shape=[6, 2],
                pads=[0, 0, 4, 0],
            ),
            helper.make_node("Greater", ["conv_green10_f", zero_f], ["conv_green10_b"]),
            helper.make_node("Where", ["conv_green10_b", selector, "conv_x10"], ["conv_out10"]),
            helper.make_node("Pad", ["conv_out10"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - N, W - N]),
        ]
    )

    graph = helper.make_graph(nodes, f"{TASK_ID}_conv_id", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_id_model() -> onnx.ModelProto:
    """Alternative candidate: count distinct colors with ArgMax IDs and Equal."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    x_start = _i64(inits, [0, 0, 0, 0], "id_x_start")
    x_end = _i64(inits, [1, C, N, N], "id_x_end")
    axes4 = _i64(inits, [0, 1, 2, 3], "id_axes4")
    axes23 = _i64(inits, [2, 3], "id_axes23")
    s00 = _i64(inits, [0, 0], "s00")
    e99 = _i64(inits, [N - 1, N - 1], "e99")
    s01 = _i64(inits, [0, 1], "s01")
    e9n = _i64(inits, [N - 1, N], "e9n")
    s10 = _i64(inits, [1, 0], "s10")
    en9 = _i64(inits, [N, N - 1], "en9")
    s11 = _i64(inits, [1, 1], "s11")
    enn = _i64(inits, [N, N], "enn")
    zero_f = _f32(inits, [0.0], "id_zero_f")
    zero_i = _i64(inits, [0], "id_zero_i")
    paint_w = _f32(inits, _paint_kernel(), "id_paint_w")
    selector = _f32(inits, _green_selector(), "id_selector")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, x_start, x_end, axes4], ["x10"]),
            helper.make_node("ArgMax", ["x10"], ["ids"], axis=1, keepdims=1),
            helper.make_node("Slice", ["ids", s00, e99, axes23], ["tl"]),
            helper.make_node("Slice", ["ids", s01, e9n, axes23], ["tr"]),
            helper.make_node("Slice", ["ids", s10, en9, axes23], ["bl"]),
            helper.make_node("Slice", ["ids", s11, enn, axes23], ["br"]),
            helper.make_node("Greater", ["tl", zero_i], ["ntl"]),
            helper.make_node("Greater", ["tr", zero_i], ["ntr"]),
            helper.make_node("Greater", ["bl", zero_i], ["nbl"]),
            helper.make_node("Greater", ["br", zero_i], ["nbr"]),
            helper.make_node("And", ["ntl", "ntr"], ["ntop"]),
            helper.make_node("And", ["nbl", "nbr"], ["nbot"]),
            helper.make_node("And", ["ntop", "nbot"], ["is_block"]),
            helper.make_node("Equal", ["tr", "tl"], ["tr_tl_eq"]),
            helper.make_node("Not", ["tr_tl_eq"], ["u2"]),
            helper.make_node("Equal", ["bl", "tl"], ["bl_tl_eq"]),
            helper.make_node("Equal", ["bl", "tr"], ["bl_tr_eq"]),
            helper.make_node("Or", ["bl_tl_eq", "bl_tr_eq"], ["bl_seen"]),
            helper.make_node("Not", ["bl_seen"], ["u3"]),
            helper.make_node("Equal", ["br", "tl"], ["br_tl_eq"]),
            helper.make_node("Equal", ["br", "tr"], ["br_tr_eq"]),
            helper.make_node("Equal", ["br", "bl"], ["br_bl_eq"]),
            helper.make_node("Or", ["br_tl_eq", "br_tr_eq"], ["br_seen_a"]),
            helper.make_node("Or", ["br_seen_a", "br_bl_eq"], ["br_seen"]),
            helper.make_node("Not", ["br_seen"], ["u4"]),
            helper.make_node("Or", ["u2", "u3"], ["h2_a"]),
            helper.make_node("Or", ["h2_a", "u4"], ["h2_any"]),
            helper.make_node("And", ["u2", "u3"], ["p23"]),
            helper.make_node("And", ["u2", "u4"], ["p24"]),
            helper.make_node("And", ["u3", "u4"], ["p34"]),
            helper.make_node("Or", ["p23", "p24"], ["h3_a"]),
            helper.make_node("Or", ["h3_a", "p34"], ["h3_any"]),
            helper.make_node("And", ["p23", "u4"], ["h4_any"]),
            helper.make_node("And", ["is_block", "h2_any"], ["h2"]),
            helper.make_node("And", ["is_block", "h3_any"], ["h3"]),
            helper.make_node("And", ["is_block", "h4_any"], ["h4"]),
            helper.make_node("Concat", ["is_block", "h2", "h3", "h4"], ["height_bits"], axis=1),
            helper.make_node("Cast", ["height_bits"], ["height_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "ConvTranspose",
                ["height_f", paint_w],
                ["green10_f"],
                kernel_shape=[6, 2],
                pads=[0, 0, 4, 0],
            ),
            helper.make_node("Greater", ["green10_f", zero_f], ["green10_b"]),
            helper.make_node("Where", ["green10_b", selector, "x10"], ["out10"]),
            helper.make_node("Pad", ["out10"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - N, W - N]),
        ]
    )

    graph = helper.make_graph(nodes, f"{TASK_ID}_id", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def load_task() -> dict[str, list[dict[str, list[list[int]]]]]:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def validate_hypothesis(data: dict[str, list[dict[str, list[list[int]]]]]) -> dict[str, int]:
    """Hypothesis test: extension height equals 2x2 distinct-color count."""
    counts: dict[str, int] = {}
    for split in ("train", "test", "arc-gen"):
        bad = 0
        for example in data.get(split, []):
            plain = solve(example["input"], mask_background=False)
            masked = solve(example["input"], mask_background=True)
            expected = np.asarray(example["output"], dtype=np.int64)
            if not np.array_equal(plain, expected) or not np.array_equal(masked, expected):
                bad += 1
        if bad:
            raise AssertionError(f"distinct-color-height hypothesis failed on {bad} {split} examples")
        counts[split] = len(data.get(split, []))
    return counts


def validate_model(model: onnx.ModelProto, data: dict[str, list[dict[str, list[list[int]]]]]) -> tuple[bool, str]:
    try:
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    except Exception as exc:
        return False, f"ORT load failed: {exc}"

    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            inp = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if inp is None or expected is None:
                continue
            pred = session.run([OUT_NAME], {IN_NAME: inp})[0]
            if not np.array_equal(pred > 0.0, expected > 0.0):
                return False, f"wrong output on {split}[{idx}]"
    return True, "ok"


def score_candidate(model: onnx.ModelProto) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / f"{TASK_ID}.onnx"
        onnx.save(model, path)
        return score_file(path)


def candidate_variants() -> list[Variant]:
    return [
        Variant("conv_id_overlay", build_conv_id_model),
        Variant("unmasked_green_overlay", lambda: build_model(mask_background=False)),
        Variant("background_guarded_overlay", lambda: build_model(mask_background=True)),
        Variant("argmax_id_overlay", build_id_model),
    ]


def main() -> None:
    data = load_task()
    hypothesis_counts = validate_hypothesis(data)

    results: list[tuple[float, str, onnx.ModelProto, dict[str, Any], str]] = []
    for variant in candidate_variants():
        model = variant.build()
        ok, message = validate_model(model, data)
        if not ok:
            print(f"{variant.name}: invalid ({message})")
            continue
        score = score_candidate(model)
        rank_cost = math.inf if not score.get("valid") else int(score["cost"])
        results.append((rank_cost, variant.name, model, score, message))
        print(
            f"{variant.name}: cost={score.get('cost')} memory={score.get('memory')} "
            f"params={score.get('params')} score={score.get('score')}"
        )

    if not results:
        raise AssertionError("no valid task397 ONNX variant")

    _, name, best_model, best_score, _ = min(results, key=lambda item: item[0])
    onnx.save(best_model, BEST_PATH)
    print(f"hypothesis passed: {hypothesis_counts}")
    print(
        f"selected {name}; wrote {BEST_PATH}; cost={best_score['cost']} "
        f"memory={best_score['memory']} params={best_score['params']} "
        f"score={best_score['score']:.6f}"
    )


if __name__ == "__main__":
    main()
