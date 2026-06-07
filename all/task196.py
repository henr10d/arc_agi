"""Recolor complete hollow blue rectangles to green for NeuroGolf task196.

Task rule: every non-background object is blue.  Connected components that are
exactly a one-pixel-thick hollow rectangle are changed from blue (1) to green
(3); incomplete frames, bars, dots, filled fragments, and open shapes stay blue.
The available rectangles are 3..5 cells high and 3..5 cells wide, with geometry
and positions otherwise unchanged.

ONNX approach: slice the blue channel, detect each 3..5 by 3..5 hollow rectangle
with small convolution kernels, paint the detected rectangle borders, then move
those pixels from channel 1 to channel 3.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task196"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
CORE = 15
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10
SIZES = tuple((h, w) for h in range(3, 6) for w in range(3, 6))


def _init(inits: list[onnx.TensorProto], arr: np.ndarray | Iterable[int] | Iterable[float], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _border_kernel(h: int, w: int, signed: bool) -> np.ndarray:
    kernel = np.zeros((1, 1, h, w), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                kernel[0, 0, r, c] = 1.0
            elif signed:
                kernel[0, 0, r, c] = -100.0
    return kernel


def _model(
    signed: bool,
    delta_output: bool = False,
    signed_paint: bool = False,
    core: int = H,
    compact_output: bool = False,
    compact_channels: int = C,
    direct_compact: bool = False,
    direct_heat: bool = False,
) -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    axes = _init(inits, np.array([0, 1, 2, 3], dtype=np.int64), "axes")
    s1 = _init(inits, np.array([0, 1, 0, 0], dtype=np.int64), "s1")
    e2 = _init(inits, np.array([1, 2, core, core], dtype=np.int64), "e2")

    nodes.append(helper.make_node("Slice", [IN_NAME, s1, e2, axes], ["blue"]))

    painted: list[str] = []
    thresholds: dict[int, str] = {}
    for h, w in SIZES:
        border = 2 * h + 2 * w - 4
        suffix = f"{h}{w}"
        threshold = thresholds.get(border)
        if threshold is None:
            threshold = _init(inits, np.array([border - 0.5], dtype=np.float32), f"t{border}")
            thresholds[border] = threshold
        if signed:
            kernel = _init(inits, _border_kernel(h, w, signed=True), f"k{suffix}")
            nodes.append(helper.make_node("Conv", ["blue", kernel], [f"score{suffix}"]))
            nodes.append(helper.make_node("Greater", [f"score{suffix}", threshold], [f"hit{suffix}"]))
        else:
            kb = _init(inits, _border_kernel(h, w, signed=False), f"kb{suffix}")
            ka = _init(inits, np.ones((1, 1, h, w), dtype=np.float32), f"ka{suffix}")
            full_threshold = _init(inits, np.array([border + 0.5], dtype=np.float32), f"u{suffix}")
            nodes.append(helper.make_node("Conv", ["blue", kb], [f"bs{suffix}"]))
            nodes.append(helper.make_node("Conv", ["blue", ka], [f"as{suffix}"]))
            nodes.append(helper.make_node("Greater", [f"bs{suffix}", threshold], [f"hb{suffix}"]))
            nodes.append(helper.make_node("Less", [f"as{suffix}", full_threshold], [f"ha{suffix}"]))
            nodes.append(helper.make_node("And", [f"hb{suffix}", f"ha{suffix}"], [f"hit{suffix}"]))
        hitf = f"hitf{suffix}"
        paint = f"paint{suffix}"
        nodes.append(helper.make_node("Cast", [f"hit{suffix}"], [hitf], to=TensorProto.FLOAT))
        if signed and signed_paint:
            pk = kernel
        else:
            pk = _init(inits, _border_kernel(h, w, signed=False), f"p{suffix}")
        nodes.append(helper.make_node("ConvTranspose", [hitf, pk], [paint]))
        painted.append(paint)

    nodes.append(helper.make_node("Sum", painted, ["heat"]))
    if direct_heat:
        maskf = "heat"
    else:
        zero = _init(inits, np.array([0.0], dtype=np.float32), "zero")
        nodes.append(helper.make_node("Greater", ["heat", zero], ["mask"]))
        nodes.append(helper.make_node("Cast", ["mask"], ["maskf_core"], to=TensorProto.FLOAT))
        maskf = "maskf_core"
    if core != H and not compact_output:
        nodes.append(
            helper.make_node(
                "Pad",
                ["maskf_core"],
                ["maskf"],
                pads=[0, 0, 0, 0, 0, 0, H - core, W - core],
            )
        )
        maskf = "maskf"
    if direct_compact:
        s0 = _init(inits, np.array([0, 0, 0, 0], dtype=np.int64), "s0")
        e1 = _init(inits, np.array([1, 1, core, core], dtype=np.int64), "e1")
        nodes.extend(
            [
                helper.make_node("Slice", [IN_NAME, s0, e1, axes], ["c0"]),
                helper.make_node("Sub", ["blue", maskf], ["blue_out"]),
                helper.make_node("Sub", ["blue", "blue"], ["zero_ch"]),
                helper.make_node("Concat", ["c0", "blue_out", "zero_ch", maskf], ["out_core"], axis=1),
                helper.make_node(
                    "Pad",
                    ["out_core"],
                    [OUT_NAME],
                    pads=[0, 0, 0, 0, 0, C - 4, H - core, W - core],
                ),
            ]
        )
    elif delta_output:
        out_channels = compact_channels if compact_output else C
        coeff = np.zeros((1, out_channels, 1, 1), dtype=np.float32)
        coeff[0, 1, 0, 0] = -1.0
        coeff[0, 3, 0, 0] = 1.0
        delta_coeff = _init(inits, coeff, "delta_coeff")
        nodes.append(helper.make_node("Mul", [maskf, delta_coeff], ["delta"]))
        if compact_output:
            s_all = _init(inits, np.array([0, 0, 0, 0], dtype=np.int64), "s_all")
            e_all = _init(inits, np.array([1, out_channels, core, core], dtype=np.int64), "e_all")
            nodes.append(helper.make_node("Slice", [IN_NAME, s_all, e_all, axes], ["input_core"]))
            nodes.append(helper.make_node("Add", ["input_core", "delta"], ["out_core"]))
            nodes.append(
                helper.make_node(
                    "Pad",
                    ["out_core"],
                    [OUT_NAME],
                    pads=[0, 0, 0, 0, 0, C - out_channels, H - core, W - core],
                )
            )
        else:
            nodes.append(helper.make_node("Add", [IN_NAME, "delta"], [OUT_NAME]))
    else:
        s0 = _init(inits, np.array([0, 0, 0, 0], dtype=np.int64), "s0")
        e1 = _init(inits, np.array([1, 1, H, W], dtype=np.int64), "e1")
        s2 = _init(inits, np.array([0, 2, 0, 0], dtype=np.int64), "s2")
        e3 = _init(inits, np.array([1, 3, H, W], dtype=np.int64), "e3")
        s3 = _init(inits, np.array([0, 3, 0, 0], dtype=np.int64), "s3")
        e4 = _init(inits, np.array([1, 4, H, W], dtype=np.int64), "e4")
        s4 = _init(inits, np.array([0, 4, 0, 0], dtype=np.int64), "s4")
        e10 = _init(inits, np.array([1, C, H, W], dtype=np.int64), "e10")
        nodes.extend(
            [
                helper.make_node("Slice", [IN_NAME, s0, e1, axes], ["c0"]),
                helper.make_node("Slice", [IN_NAME, s2, e3, axes], ["c2"]),
                helper.make_node("Slice", [IN_NAME, s3, e4, axes], ["c3"]),
                helper.make_node("Slice", [IN_NAME, s4, e10, axes], ["c4_9"]),
                helper.make_node("Sub", ["blue", maskf], ["blue_out"]),
                helper.make_node("Add", ["c3", maskf], ["green_out"]),
                helper.make_node("Concat", ["c0", "blue_out", "c2", "green_out", "c4_9"], [OUT_NAME], axis=1),
            ]
        )

    graph = helper.make_graph(
        nodes,
        f"{TASK_ID}_{'signed' if signed else 'literal'}_{'direct' if direct_compact else ('delta' if delta_output else 'concat')}_{core}_{'compact' if compact_output else 'full'}_{compact_channels}",
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def solve_grid(grid: list[list[int]]) -> list[list[int]]:
    arr = np.asarray(grid, dtype=np.int64)
    out = arr.copy()
    for h, w in SIZES:
        border_count = 2 * h + 2 * w - 4
        for r in range(arr.shape[0] - h + 1):
            for c in range(arr.shape[1] - w + 1):
                box = arr[r : r + h, c : c + w]
                border = np.zeros((h, w), dtype=bool)
                border[0, :] = border[-1, :] = True
                border[:, 0] = border[:, -1] = True
                if np.count_nonzero(box == 1) == border_count and np.all(box[border] == 1):
                    out[r : r + h, c : c + w][border] = 3
    return out.tolist()


def _expected_onehot(grid: list[list[int]]) -> np.ndarray:
    return convert_to_numpy({"input": grid, "output": grid}, "input")


def validate_reference() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    failures: list[str] = []
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            got = solve_grid(example["input"])
            if got != example["output"]:
                failures.append(f"{split}[{idx}]")
    if failures:
        raise AssertionError(f"reference rule failed: {', '.join(failures[:10])}")


def validate_model(model: onnx.ModelProto) -> None:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    failures: list[str] = []
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            arr = convert_to_numpy(example, "input")
            if arr is None:
                continue
            got = session.run([OUT_NAME], {IN_NAME: arr})[0]
            expected = _expected_onehot(example["output"])
            if not np.array_equal(got > 0.0, expected > 0.0):
                failures.append(f"{split}[{idx}]")
    if failures:
        raise AssertionError(f"ONNX failed: {', '.join(failures[:10])}")


def main() -> None:
    validate_reference()
    candidates = [
        ("signed_concat", _model(True)),
        ("signed_delta", _model(True, delta_output=True)),
        ("signed_delta_reuse", _model(True, delta_output=True, signed_paint=True)),
        ("signed_delta_reuse_core15", _model(True, delta_output=True, signed_paint=True, core=CORE)),
        (
            "signed_delta_reuse_core15_compact",
            _model(True, delta_output=True, signed_paint=True, core=CORE, compact_output=True),
        ),
        (
            "signed_delta_reuse_core15_compact4",
            _model(
                True,
                delta_output=True,
                signed_paint=True,
                core=CORE,
                compact_output=True,
                compact_channels=4,
            ),
        ),
        (
            "signed_direct_core15",
            _model(True, signed_paint=True, core=CORE, compact_output=True, compact_channels=4, direct_compact=True),
        ),
        (
            "signed_direct_core15_heat",
            _model(
                True,
                signed_paint=False,
                core=CORE,
                compact_output=True,
                compact_channels=4,
                direct_compact=True,
                direct_heat=True,
            ),
        ),
        ("literal_concat", _model(False)),
    ]
    results = []
    for name, model in candidates:
        validate_model(model)
        path = Path(tempfile.gettempdir()) / f"{TASK_ID}_{name}.onnx"
        onnx.save(model, path)
        result = score_file(path)
        results.append((result["score"] or 0.0, -(result["filesize"] or 0), name, model, result))
        print(
            f"{name}: valid={result['valid']} memory={result['memory']} "
            f"params={result['params']} cost={result['cost']} score={result['score']}"
        )
    _, _, name, model, result = max(results)
    onnx.save(model, BEST_PATH)
    print(
        f"saved {BEST_PATH} from {name}: memory={result['memory']} "
        f"params={result['params']} cost={result['cost']} score={result['score']}"
    )


if __name__ == "__main__":
    main()
