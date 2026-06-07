"""Analytical ONNX solver for ARC task379 cyan-line hubs.

Task rule: the grid contains one or two parallel cyan lines and red marker
cells. For each red marker, draw a perpendicular red segment to the nearest
cyan line on each side of the marker. Where such a segment crosses a cyan
line, the crossing cell is red and the eight surrounding cells become cyan,
forming a 3x3 hub around the crossing. The hub cyan cells overwrite adjacent
red segment cells, while the crossing center remains red.

ONNX approach: slice only colors 0/2/8 in the observed 20x20 task region.
For each orientation, prefix/suffix masks identify the nearest cyan-line
region without enumerating ray distances. The vertical case reuses the same
logic on transposed 1-channel masks, then a small orientation test selects
red/cyan/black masks and expands them to one-hot output once.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task379"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task379.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
FULL_N = 30
N = 20
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _i32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int32), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _slice_channel(
    nodes: List[onnx.NodeProto],
    data: str,
    channel: int,
    prefix: str,
    starts: str,
    ends_by_channel: dict[int, str],
    axes4: str,
    half: str,
) -> str:
    end = ends_by_channel[channel + 1]
    nodes.append(helper.make_node("Slice", [data, starts, end, axes4], [f"{prefix}_raw_c{channel}"]))
    nodes.append(helper.make_node("Greater", [f"{prefix}_raw_c{channel}", half], [f"{prefix}_c{channel}"]))
    return f"{prefix}_c{channel}"


def _make_color_masks(
    nodes: List[onnx.NodeProto],
    active: str,
    segment: str,
    cyan: str,
    prefix: str,
) -> tuple[str, str, str]:
    nodes.append(helper.make_node("And", [segment, cyan], [f"{prefix}_cross"]))
    nodes.append(helper.make_node("Cast", [f"{prefix}_cross"], [f"{prefix}_cross_f"], to=TensorProto.FLOAT))
    nodes.append(
        helper.make_node(
            "Conv",
            [f"{prefix}_cross_f", "hub_kernel"],
            [f"{prefix}_hub_count"],
            pads=[1, 1, 1, 1],
        )
    )
    nodes.append(helper.make_node("Greater", [f"{prefix}_hub_count", "zero_f"], [f"{prefix}_hub3"]))
    nodes.append(helper.make_node("Not", [f"{prefix}_cross"], [f"{prefix}_not_cross"]))
    nodes.append(helper.make_node("And", [f"{prefix}_hub3", f"{prefix}_not_cross"], [f"{prefix}_hub_cyan"]))
    nodes.append(helper.make_node("Not", [f"{prefix}_hub_cyan"], [f"{prefix}_not_hub_cyan"]))
    nodes.append(helper.make_node("And", [segment, f"{prefix}_not_hub_cyan"], [f"{prefix}_red"]))
    nodes.append(helper.make_node("And", [cyan, f"{prefix}_not_cross"], [f"{prefix}_cyan_base"]))
    nodes.append(helper.make_node("Or", [f"{prefix}_cyan_base", f"{prefix}_hub_cyan"], [f"{prefix}_cyan"]))
    nodes.append(helper.make_node("Or", [f"{prefix}_red", f"{prefix}_cyan"], [f"{prefix}_non_black"]))
    nodes.append(helper.make_node("Not", [f"{prefix}_non_black"], [f"{prefix}_not_non_black"]))
    nodes.append(helper.make_node("And", [active, f"{prefix}_not_non_black"], [f"{prefix}_black"]))
    return f"{prefix}_red", f"{prefix}_cyan", f"{prefix}_black"


def _make_onehot(
    nodes: List[onnx.NodeProto],
    red: str,
    cyan: str,
    black: str,
    prefix: str,
) -> str:
    nodes.append(helper.make_node("And", [red, black], [f"{prefix}_zero"]))
    nodes.append(
        helper.make_node(
            "Concat",
            [
                black,
                f"{prefix}_zero",
                red,
                f"{prefix}_zero",
                f"{prefix}_zero",
                f"{prefix}_zero",
                f"{prefix}_zero",
                f"{prefix}_zero",
                cyan,
                f"{prefix}_zero",
            ],
            [f"{prefix}_onehot"],
            axis=1,
        )
    )
    return f"{prefix}_onehot"


def _shift_vertical(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    data: str,
    out: str,
    d: int,
    direction: str,
    axes4: str,
) -> str:
    if d == 0:
        nodes.append(helper.make_node("Identity", [data], [out]))
        return out

    if direction == "down":
        starts = _i64(inits, [0, 0, 0, 0], f"{out}_starts")
        ends = _i64(inits, [1, 1, N - d, N], f"{out}_ends")
        pads = [0, 0, d, 0, 0, 0, 0, 0]
    elif direction == "up":
        starts = _i64(inits, [0, 0, d, 0], f"{out}_starts")
        ends = _i64(inits, [1, 1, N, N], f"{out}_ends")
        pads = [0, 0, 0, 0, 0, 0, d, 0]
    else:
        raise ValueError(f"unsupported shift direction: {direction}")

    nodes.append(helper.make_node("Slice", [data, starts, ends, axes4], [f"{out}_slice"]))
    nodes.append(helper.make_node("Pad", [f"{out}_slice"], [out], mode="constant", pads=pads))
    return out


def _horizontal_solution(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    red: str,
    cyan: str,
    active: str,
    prefix: str,
    names: dict[str, str],
) -> tuple[str, str, str]:
    nodes.append(helper.make_node("Cast", [red], [f"{prefix}_red_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Cast", [cyan], [f"{prefix}_cyan_f"], to=TensorProto.FLOAT))

    nodes.append(
        helper.make_node(
            "Conv",
            [f"{prefix}_cyan_f", "future_kernel"],
            [f"{prefix}_cyan_above_count"],
            pads=[N - 1, 0, 0, 0],
        )
    )
    nodes.append(
        helper.make_node(
            "Conv",
            [f"{prefix}_cyan_f", "future_kernel"],
            [f"{prefix}_cyan_below_count"],
            pads=[0, 0, N - 1, 0],
        )
    )
    nodes.append(helper.make_node("Greater", [f"{prefix}_cyan_above_count", "zero_f"], [f"{prefix}_cyan_above"]))
    nodes.append(helper.make_node("Greater", [f"{prefix}_cyan_below_count", "zero_f"], [f"{prefix}_cyan_below"]))

    _shift_vertical(
        nodes,
        inits,
        f"{prefix}_cyan_above_count",
        f"{prefix}_cyan_above_strict_count",
        1,
        "down",
        names["axes4"],
    )
    _shift_vertical(
        nodes,
        inits,
        f"{prefix}_cyan_below_count",
        f"{prefix}_cyan_below_strict_count",
        1,
        "up",
        names["axes4"],
    )
    nodes.append(
        helper.make_node("Greater", [f"{prefix}_cyan_above_strict_count", "zero_f"], [f"{prefix}_cyan_above_strict"])
    )
    nodes.append(
        helper.make_node("Greater", [f"{prefix}_cyan_below_strict_count", "zero_f"], [f"{prefix}_cyan_below_strict"])
    )

    nodes.append(
        helper.make_node(
            "Conv",
            [f"{prefix}_red_f", "future_kernel"],
            [f"{prefix}_red_above_count"],
            pads=[N - 1, 0, 0, 0],
        )
    )
    nodes.append(
        helper.make_node(
            "Conv",
            [f"{prefix}_red_f", "future_kernel"],
            [f"{prefix}_red_below_count"],
            pads=[0, 0, N - 1, 0],
        )
    )
    nodes.append(helper.make_node("Greater", [f"{prefix}_red_above_count", "zero_f"], [f"{prefix}_red_above"]))
    nodes.append(helper.make_node("Greater", [f"{prefix}_red_below_count", "zero_f"], [f"{prefix}_red_below"]))

    nodes.append(helper.make_node("And", [f"{prefix}_cyan_above", f"{prefix}_cyan_below"], [f"{prefix}_between_lines"]))
    nodes.append(helper.make_node("And", [red, f"{prefix}_between_lines"], [f"{prefix}_red_between"]))
    nodes.append(helper.make_node("Cast", [f"{prefix}_red_between"], [f"{prefix}_red_between_i"], to=TensorProto.INT32))
    nodes.append(
        helper.make_node(
            "ReduceSum",
            [f"{prefix}_red_between_i"],
            [f"{prefix}_red_between_col_count"],
            axes=[2],
            keepdims=1,
        )
    )
    nodes.append(
        helper.make_node("Greater", [f"{prefix}_red_between_col_count", names["zero_i"]], [f"{prefix}_has_red_between"])
    )
    nodes.append(helper.make_node("And", [f"{prefix}_between_lines", f"{prefix}_has_red_between"], [f"{prefix}_middle_segment"]))

    nodes.append(helper.make_node("Not", [f"{prefix}_cyan_above_strict"], [f"{prefix}_not_cyan_above_strict"]))
    nodes.append(helper.make_node("Not", [f"{prefix}_cyan_below_strict"], [f"{prefix}_not_cyan_below_strict"]))
    nodes.append(helper.make_node("And", [f"{prefix}_red_above", f"{prefix}_cyan_below"], [f"{prefix}_top_pre"]))
    nodes.append(helper.make_node("And", [f"{prefix}_top_pre", f"{prefix}_not_cyan_above_strict"], [f"{prefix}_top_segment"]))
    nodes.append(helper.make_node("And", [f"{prefix}_red_below", f"{prefix}_cyan_above"], [f"{prefix}_bottom_pre"]))
    nodes.append(
        helper.make_node("And", [f"{prefix}_bottom_pre", f"{prefix}_not_cyan_below_strict"], [f"{prefix}_bottom_segment"])
    )
    nodes.append(helper.make_node("Or", [f"{prefix}_top_segment", f"{prefix}_middle_segment"], [f"{prefix}_segment_a"]))
    nodes.append(helper.make_node("Or", [f"{prefix}_segment_a", f"{prefix}_bottom_segment"], [f"{prefix}_segment"]))
    segment = f"{prefix}_segment"

    return _make_color_masks(
        nodes,
        active,
        segment,
        cyan,
        prefix,
    )


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    black_start = _i64(inits, [0, 0, 0, 0], "black_start")
    red_start = _i64(inits, [0, 2, 0, 0], "red_start")
    cyan_start = _i64(inits, [0, 8, 0, 0], "cyan_start")
    channel_ends = {
        1: _i64(inits, [1, 1, N, N], "end_c1"),
        3: _i64(inits, [1, 3, N, N], "end_c3"),
        9: _i64(inits, [1, 9, N, N], "end_c9"),
    }
    half = _f32(inits, [0.5], "half")
    zero_f = _f32(inits, [0.0], "zero_f")
    zero_i = _i32(inits, [0], "zero_i")
    _f32(inits, np.ones((1, 1, 3, 3), dtype=np.float32), "hub_kernel")
    _f32(inits, np.ones((1, 1, N, 1), dtype=np.float32), "future_kernel")

    names = {
        "axes4": axes4,
        "black_start": black_start,
        "red_start": red_start,
        "cyan_start": cyan_start,
        "channel_ends": channel_ends,
        "half": half,
        "zero_i": zero_i,
    }

    black = _slice_channel(nodes, IN_NAME, 0, "input_black", black_start, channel_ends, axes4, half)
    red = _slice_channel(nodes, IN_NAME, 2, "input_red", red_start, channel_ends, axes4, half)
    cyan = _slice_channel(nodes, IN_NAME, 8, "input_cyan", cyan_start, channel_ends, axes4, half)
    nodes.append(helper.make_node("Or", [black, red], ["active_a"]))
    nodes.append(helper.make_node("Or", ["active_a", cyan], ["active"]))

    nodes.append(helper.make_node("Transpose", [red], ["red_t"], perm=[0, 1, 3, 2]))
    nodes.append(helper.make_node("Transpose", [cyan], ["cyan_t"], perm=[0, 1, 3, 2]))
    nodes.append(helper.make_node("Transpose", ["active"], ["active_t"], perm=[0, 1, 3, 2]))

    h_red, h_cyan, h_black = _horizontal_solution(nodes, inits, red, cyan, "active", "h", names)
    v_red_t, v_cyan_t, v_black_t = _horizontal_solution(nodes, inits, "red_t", "cyan_t", "active_t", "v", names)
    nodes.append(helper.make_node("Transpose", [v_red_t], ["v_red_back"], perm=[0, 1, 3, 2]))
    nodes.append(helper.make_node("Transpose", [v_cyan_t], ["v_cyan_back"], perm=[0, 1, 3, 2]))
    nodes.append(helper.make_node("Transpose", [v_black_t], ["v_black_back"], perm=[0, 1, 3, 2]))

    nodes.append(helper.make_node("Slice", [cyan, "adj_left_start", "adj_left_end", axes4], ["cyan_left"]))
    nodes.append(helper.make_node("Slice", [cyan, "adj_right_start", "adj_right_end", axes4], ["cyan_right"]))
    nodes.append(helper.make_node("And", ["cyan_left", "cyan_right"], ["cyan_hadj"]))
    nodes.append(helper.make_node("Cast", ["cyan_hadj"], ["cyan_hadj_i"], to=TensorProto.INT32))
    nodes.append(helper.make_node("ReduceSum", ["cyan_hadj_i"], ["cyan_hadj_count"], axes=[0, 1, 2, 3], keepdims=0))
    nodes.append(helper.make_node("Greater", ["cyan_hadj_count", zero_i], ["is_horizontal"]))
    nodes.append(helper.make_node("Not", ["is_horizontal"], ["is_vertical"]))

    selected_masks: list[str] = []
    for color, h_mask, v_mask in (
        ("red", h_red, "v_red_back"),
        ("cyan", h_cyan, "v_cyan_back"),
        ("black", h_black, "v_black_back"),
    ):
        nodes.append(helper.make_node("And", ["is_horizontal", h_mask], [f"h_{color}_selected"]))
        nodes.append(helper.make_node("And", ["is_vertical", v_mask], [f"v_{color}_selected"]))
        nodes.append(helper.make_node("Or", [f"h_{color}_selected", f"v_{color}_selected"], [f"selected_{color}"]))
        selected_masks.append(f"selected_{color}")

    core_onehot = _make_onehot(
        nodes,
        selected_masks[0],
        selected_masks[1],
        selected_masks[2],
        "out",
    )
    nodes.append(helper.make_node("Cast", [core_onehot], ["core_onehot_f"], to=TensorProto.FLOAT))
    nodes.append(
        helper.make_node(
            "Pad",
            ["core_onehot_f"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, FULL_N - N, FULL_N - N],
        )
    )

    _i64(inits, [0, 0, 0, 0], "adj_left_start")
    _i64(inits, [1, 1, N, N - 1], "adj_left_end")
    _i64(inits, [0, 0, 0, 1], "adj_right_start")
    _i64(inits, [1, 1, N, N], "adj_right_end")

    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def validate_examples(path: Path) -> tuple[int, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            arr = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if arr is None or expected is None:
                continue
            got = session.run([OUT_NAME], {IN_NAME: arr})[0]
            total += 1
            if not np.array_equal(got > 0.0, expected > 0.0):
                raise AssertionError(f"{split}[{idx}] output mismatch")
            passed += 1
    return passed, total


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    passed, total = validate_examples(BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print("variant: analytical nearest-line crossing")
    print(f"correct: {passed}/{total}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
