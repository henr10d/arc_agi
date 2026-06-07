"""ONNX generator for ARC task349: frame maroon blocks and add stems.

Task rule: the input contains solid color-9 square blocks on black.  A block
of visible width k gets a green (color 3) rectangular frame whose thickness is
k/2, clipped to the grid; the original color-9 cells remain; and a blue
(color 1) vertical stem with the same columns as the block starts immediately
below the framed region and continues to the grid bottom.  Edge-clipped blocks
may appear shorter than k, but their visible width still determines the frame
thickness.  Overlaps use the data's effective priority: blue stems are drawn
first, green frames overwrite stems, and maroon interiors overwrite frames.

ONNX: detect exact horizontal color-9 runs of widths 2, 4, 6, 8, and 10, use
ConvTranspose to reconstruct per-width block masks, dilate each mask by k/2 for
green frames, project each per-width bottom edge downward for blue stems, and
assemble one-hot output.  Generated colors are clipped with the input black
plane, which is equivalent to active-grid clipping because this task uses only
black and maroon inputs.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task349"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task349.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10
WIDTHS = (2, 4, 6, 8, 10)


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _f16(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float16), name)


def _slice(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    source: str,
    starts: list[int],
    ends: list[int],
    name: str,
) -> str:
    starts_name = _i64(inits, starts, f"{name}_starts")
    ends_name = _i64(inits, ends, f"{name}_ends")
    nodes.append(helper.make_node("Slice", [source, starts_name, ends_name], [name]))
    return name


def _or(nodes: List[onnx.NodeProto], a: str, b: str, name: str) -> str:
    nodes.append(helper.make_node("Or", [a, b], [name]))
    return name


def _and(nodes: List[onnx.NodeProto], a: str, b: str, name: str) -> str:
    nodes.append(helper.make_node("And", [a, b], [name]))
    return name


def _not(nodes: List[onnx.NodeProto], x: str, name: str) -> str:
    nodes.append(helper.make_node("Not", [x], [name]))
    return name


def _cast(nodes: List[onnx.NodeProto], x: str, name: str, to: int) -> str:
    nodes.append(helper.make_node("Cast", [x], [name], to=to))
    return name


def _greater(nodes: List[onnx.NodeProto], x: str, threshold: str, name: str) -> str:
    nodes.append(helper.make_node("Greater", [x, threshold], [name]))
    return name


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    half = _f32(inits, 0.5, "half")
    half16 = _f16(inits, 0.5, "half16")
    zero_plane = _init(inits, np.zeros((1, 1, H, W), dtype=np.bool_), "zero_plane")
    ch0_st = _i64(inits, [0, 0, 0, 0], "ch0_st")
    ch0_en = _i64(inits, [1, 1, H, W], "ch0_en")
    ch9_st = _i64(inits, [0, 9, 0, 0], "ch9_st")
    ch9_en = _i64(inits, [1, 10, H, W], "ch9_en")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, ch0_st, ch0_en], ["black_f"]),
            helper.make_node("Greater", ["black_f", half], ["black_b"]),
            helper.make_node("Slice", [IN_NAME, ch9_st, ch9_en], ["red_f"]),
        ]
    )
    _cast(nodes, "red_f", "red_h", TensorProto.FLOAT16)

    # Pad one black column on both horizontal sides so exact run detectors work
    # at grid edges and reject shorter sub-runs inside wider blocks.
    nodes.append(
        helper.make_node(
            "Pad",
            ["red_h"],
            ["red_lr_pad"],
            pads=[0, 0, 0, 1, 0, 0, 0, 1],
            mode="constant",
            value=0.0,
        )
    )

    green_masks: list[str] = []
    blue_masks: list[str] = []

    for k in WIDTHS:
        t = k // 2
        prefix = f"w{k}"

        run_kernel = np.zeros((1, 1, 1, k + 2), dtype=np.float32)
        run_kernel[0, 0, 0, 1 : k + 1] = 1.0
        run_kernel[0, 0, 0, 0] = -100.0
        run_kernel[0, 0, 0, k + 1] = -100.0
        run_w = _f16(inits, run_kernel, f"{prefix}_run_w")
        run_thr = _f16(inits, float(k) - 0.5, f"{prefix}_run_thr")
        nodes.append(helper.make_node("Conv", ["red_lr_pad", run_w], [f"{prefix}_run_score"]))
        _greater(nodes, f"{prefix}_run_score", run_thr, f"{prefix}_run_b")
        _cast(nodes, f"{prefix}_run_b", f"{prefix}_run_f", TensorProto.FLOAT16)

        green_w = _f16(inits, np.ones((1, 1, 2 * t + 1, k + 2 * t), dtype=np.float16), f"{prefix}_green_w")
        nodes.append(
            helper.make_node(
                "ConvTranspose",
                [f"{prefix}_run_f", green_w],
                [f"{prefix}_green"],
                pads=[t, t, t, t],
            )
        )
        green_masks.append(_greater(nodes, f"{prefix}_green", half16, f"{prefix}_green_b"))

        run_cols = W - k + 1
        run_tail = _slice(nodes, inits, f"{prefix}_run_b", [0, 0, 1, 0], [1, 1, H, run_cols], f"{prefix}_run_tail")
        false_row = _init(inits, np.zeros((1, 1, 1, run_cols), dtype=np.bool_), f"{prefix}_false_row")
        nodes.append(helper.make_node("Concat", [run_tail, false_row], [f"{prefix}_below_b"], axis=2))
        _not(nodes, f"{prefix}_below_b", f"{prefix}_not_below")
        _and(nodes, f"{prefix}_run_b", f"{prefix}_not_below", f"{prefix}_bottom_b")
        _cast(nodes, f"{prefix}_bottom_b", f"{prefix}_bottom_f", TensorProto.FLOAT16)

        stem_kernel = np.zeros((1, 1, H + t + 1, k), dtype=np.float16)
        stem_kernel[0, 0, t + 1 :, :] = 1.0
        stem_w = _f16(inits, stem_kernel, f"{prefix}_stem_w")
        nodes.append(
            helper.make_node(
                "ConvTranspose",
                [f"{prefix}_bottom_f", stem_w],
                [f"{prefix}_stem"],
                pads=[0, 0, H + t, 0],
            )
        )
        blue_masks.append(_greater(nodes, f"{prefix}_stem", half16, f"{prefix}_blue_b"))

    green_any = green_masks[0]
    for idx, mask in enumerate(green_masks[1:], start=1):
        green_any = _or(nodes, green_any, mask, f"green_any_{idx}")

    blue_any = blue_masks[0]
    for idx, mask in enumerate(blue_masks[1:], start=1):
        blue_any = _or(nodes, blue_any, mask, f"blue_any_{idx}")

    # Data priority is blue first, then green, then maroon/red interiors.  The
    # input black plane clips all generated color to the true grid area and
    # excludes maroon cells, so no separate active-grid mask is needed.
    green_no_red = _and(nodes, green_any, "black_b", "green_no_red")
    _not(nodes, green_any, "not_green_any")
    blue_no_green = _and(nodes, blue_any, "not_green_any", "blue_no_green")
    blue_out_b = _and(nodes, blue_no_green, "black_b", "blue_out_b")
    used_gb = _or(nodes, green_any, blue_any, "used_gb")
    _not(nodes, used_gb, "not_used")
    black_out_b = _and(nodes, "black_b", "not_used", "black_out_b")

    _greater(nodes, "red_f", half, "red_b")
    nodes.append(
        helper.make_node(
            "Concat",
            [
                "black_out_b",
                "blue_out_b",
                zero_plane,
                "green_no_red",
                zero_plane,
                zero_plane,
                zero_plane,
                zero_plane,
                zero_plane,
                "red_b",
            ],
            ["output_b"],
            axis=1,
        )
    )
    _cast(nodes, "output_b", OUT_NAME, TensorProto.FLOAT)

    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name=TASK_ID,
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return model


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x})[0]


def verify_all(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    checked = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            x = convert_to_numpy(ex, "input")
            y = convert_to_numpy(ex, "output")
            if x is None or y is None:
                continue
            pred = _run_onnx(model, x)
            if not np.array_equal(pred > 0.0, y > 0.0):
                diff = np.argwhere((pred > 0.0) != (y > 0.0))
                raise AssertionError(f"{split} example {idx} failed at {diff[:8].tolist()}")
            checked += 1
    return checked


def main() -> None:
    model = build_model()
    checked = verify_all(model)
    onnx.save(model, BEST_PATH)
    with tempfile.NamedTemporaryFile(suffix=".onnx", dir=OUT_DIR, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        onnx.save(model, tmp_path)
        result = score_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    print(f"verified {checked} examples")
    print(f"saved {BEST_PATH}")
    if result["valid"]:
        print(
            f"score={result['score']:.6f} cost={result['cost']} "
            f"memory={result['memory']} params={result['params']}"
        )
    else:
        print(f"score invalid: {result['error']}")


if __name__ == "__main__":
    main()
