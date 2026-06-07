"""Minimal ONNX for ARC task282: replace gray seeds with hollow 3x3 motifs.

Task rule: each non-black input pixel, always gray (5) in the examples, is a
seed.  The output starts black and stamps a clipped 3x3 motif around every seed:
gray on the four diagonals, blue on the four orthogonal neighbors, and black at
the original seed center.  All examples are 9x9, so the graph works on the
top-left 9x9 crop before padding back to the required 30x30 one-hot tensor.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task282"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task282.onnx"
DATA_PATH = ROOT / "data" / "task282.json"

C = 10
H = W = 30
G = 9
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _bool(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.bool_), name)


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


def solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(g)
    for r, c in np.argwhere(g != 0):
        for dr, dc, color in (
            (-1, -1, 5),
            (-1, 0, 1),
            (-1, 1, 5),
            (0, -1, 1),
            (0, 1, 1),
            (1, -1, 5),
            (1, 0, 1),
            (1, 1, 5),
        ):
            rr, cc = int(r) + dr, int(c) + dc
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                out[rr, cc] = color
    return out


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def _seed_crop(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto]) -> str:
    starts = _i64(inits, [0, 5, 0, 0], "seed_starts")
    ends = _i64(inits, [1, 6, G, G], "seed_ends")
    nodes.append(helper.make_node("Slice", [IN_NAME, starts, ends], ["seedf"]))
    return "seedf"


def _finish_from_masks(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    blue: str,
    gray: str,
    name: str,
) -> onnx.ModelProto:
    nodes.extend(
        [
            helper.make_node("Or", [blue, gray], ["painted"]),
            helper.make_node("Not", ["painted"], ["bg"]),
            helper.make_node("And", [blue, gray], ["zero"]),
            helper.make_node(
                "Concat",
                ["bg", blue, "zero", "zero", "zero", gray],
                ["out6b"],
                axis=1,
            ),
            helper.make_node("Cast", ["out6b"], ["out6"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out6"], [OUT_NAME], pads=[0, 0, 0, 0, 0, C - 6, H - G, W - G]),
        ]
    )
    return _make_model(nodes, inits, name)


def _or_all(nodes: List[onnx.NodeProto], names: list[str], output: str) -> str:
    current = names[0]
    for idx, other in enumerate(names[1:], start=1):
        out = output if idx == len(names) - 1 else f"{output}_{idx}"
        nodes.append(helper.make_node("Or", [current, other], [out]))
        current = out
    return current


def build_direct_shift_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    seedf = _seed_crop(nodes, inits)
    nodes.append(helper.make_node("Cast", [seedf], ["seed"], to=TensorProto.BOOL))

    row_top = _bool(inits, np.zeros((1, 1, 1, G), dtype=np.bool_), "row_top")
    row_bottom = _bool(inits, np.zeros((1, 1, 1, G), dtype=np.bool_), "row_bottom")
    col_left = _bool(inits, np.zeros((1, 1, G, 1), dtype=np.bool_), "col_left")
    col_right = _bool(inits, np.zeros((1, 1, G, 1), dtype=np.bool_), "col_right")
    row_axes = _i64(inits, [2], "row_axes")
    col_axes = _i64(inits, [3], "col_axes")
    r0 = _i64(inits, [0], "r0")
    r1 = _i64(inits, [1], "r1")
    r8 = _i64(inits, [8], "r8")
    r9 = _i64(inits, [9], "r9")
    c0 = _i64(inits, [0], "c0")
    c1 = _i64(inits, [1], "c1")
    c8 = _i64(inits, [8], "c8")
    c9 = _i64(inits, [9], "c9")

    def shifted(dr: int, dc: int, label: str) -> str:
        if dr == -1:
            nodes.append(helper.make_node("Slice", ["seed", r1, r9, row_axes], [f"{label}_rs"]))
            nodes.append(helper.make_node("Concat", [f"{label}_rs", row_bottom], [f"{label}_r"], axis=2))
            row_name = f"{label}_r"
        elif dr == 1:
            nodes.append(helper.make_node("Slice", ["seed", r0, r8, row_axes], [f"{label}_rs"]))
            nodes.append(helper.make_node("Concat", [row_top, f"{label}_rs"], [f"{label}_r"], axis=2))
            row_name = f"{label}_r"
        else:
            row_name = "seed"

        if dc == -1:
            nodes.append(helper.make_node("Slice", [row_name, c1, c9, col_axes], [f"{label}_cs"]))
            nodes.append(helper.make_node("Concat", [f"{label}_cs", col_right], [label], axis=3))
        elif dc == 1:
            nodes.append(helper.make_node("Slice", [row_name, c0, c8, col_axes], [f"{label}_cs"]))
            nodes.append(helper.make_node("Concat", [col_left, f"{label}_cs"], [label], axis=3))
        else:
            label = row_name
        return label

    orth = [
        shifted(-1, 0, "north"),
        shifted(0, -1, "west"),
        shifted(0, 1, "east"),
        shifted(1, 0, "south"),
    ]
    diag = [
        shifted(-1, -1, "northwest"),
        shifted(-1, 1, "northeast"),
        shifted(1, -1, "southwest"),
        shifted(1, 1, "southeast"),
    ]
    _or_all(nodes, orth, "blue")
    _or_all(nodes, diag, "gray")
    return _finish_from_masks(nodes, inits, "blue", "gray", "task282_direct_shift")


def build_conv_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    seedf = _seed_crop(nodes, inits)
    kernels = np.zeros((2, 1, 3, 3), dtype=np.float32)
    kernels[0, 0, 0, 1] = 1.0
    kernels[0, 0, 1, 0] = 1.0
    kernels[0, 0, 1, 2] = 1.0
    kernels[0, 0, 2, 1] = 1.0
    kernels[1, 0, 0, 0] = 1.0
    kernels[1, 0, 0, 2] = 1.0
    kernels[1, 0, 2, 0] = 1.0
    kernels[1, 0, 2, 2] = 1.0
    weight = _f32(inits, kernels, "kernel")
    threshold = _f32(inits, 0.0, "threshold")
    nodes.extend(
        [
            helper.make_node("Conv", [seedf, weight], ["hits"], pads=[1, 1, 1, 1]),
            helper.make_node("Greater", ["hits", threshold], ["masks"]),
            helper.make_node("Split", ["masks"], ["blue", "gray"], axis=1, split=[1, 1]),
        ]
    )
    return _finish_from_masks(nodes, inits, "blue", "gray", "task282_conv")


def build_convtranspose_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    starts = _i64(inits, [0, 5, 1, 1], "inner_starts")
    ends = _i64(inits, [1, 6, 8, 8], "inner_ends")
    seedf = "seedf"
    nodes.append(helper.make_node("Slice", [IN_NAME, starts, ends], [seedf]))

    kernels = np.zeros((1, 2, 3, 3), dtype=np.float32)
    kernels[0, 0, 0, 1] = 1.0
    kernels[0, 0, 1, 0] = 1.0
    kernels[0, 0, 1, 2] = 1.0
    kernels[0, 0, 2, 1] = 1.0
    kernels[0, 1, 0, 0] = 1.0
    kernels[0, 1, 0, 2] = 1.0
    kernels[0, 1, 2, 0] = 1.0
    kernels[0, 1, 2, 2] = 1.0
    weight = _f32(inits, kernels, "kernel")
    threshold = _f32(inits, 0.0, "threshold")
    nodes.extend(
        [
            helper.make_node("ConvTranspose", [seedf, weight], ["hits"]),
            helper.make_node("Greater", ["hits", threshold], ["masks"]),
            helper.make_node("Split", ["masks"], ["blue", "gray"], axis=1, split=[1, 1]),
        ]
    )
    return _finish_from_masks(nodes, inits, "blue", "gray", "task282_convtranspose")


def build_float_convtranspose_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    starts = _i64(inits, [0, 5, 1, 1], "inner_starts")
    ends = _i64(inits, [1, 6, 8, 8], "inner_ends")
    seedf = "seedf"
    nodes.append(helper.make_node("Slice", [IN_NAME, starts, ends], [seedf]))

    blue_kernel = np.zeros((1, 1, 3, 3), dtype=np.float32)
    blue_kernel[0, 0, 0, 1] = 1.0
    blue_kernel[0, 0, 1, 0] = 1.0
    blue_kernel[0, 0, 1, 2] = 1.0
    blue_kernel[0, 0, 2, 1] = 1.0

    gray_kernel = np.zeros((1, 1, 3, 3), dtype=np.float32)
    gray_kernel[0, 0, 0, 0] = 1.0
    gray_kernel[0, 0, 0, 2] = 1.0
    gray_kernel[0, 0, 2, 0] = 1.0
    gray_kernel[0, 0, 2, 2] = 1.0

    blue_weight = _f32(inits, blue_kernel, "blue_kernel")
    gray_weight = _f32(inits, gray_kernel, "gray_kernel")
    one = _f32(inits, 1.0, "one")
    zero = _f32(inits, np.zeros((1, 1, G, G), dtype=np.float32), "zero")
    nodes.extend(
        [
            helper.make_node("ConvTranspose", [seedf, blue_weight], ["blue"]),
            helper.make_node("ConvTranspose", [seedf, gray_weight], ["gray"]),
            helper.make_node("Add", ["blue", "gray"], ["painted"]),
            helper.make_node("Sub", [one, "painted"], ["bg"]),
            helper.make_node(
                "Concat",
                ["bg", "blue", zero, zero, zero, "gray"],
                ["out6"],
                axis=1,
            ),
            helper.make_node("Pad", ["out6"], [OUT_NAME], pads=[0, 0, 0, 0, 0, C - 6, H - G, W - G]),
        ]
    )
    return _make_model(nodes, inits, "task282_float_convtranspose")


def build_fused_convtranspose_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    starts = _i64(inits, [0, 5, 1, 1], "inner_starts")
    ends = _i64(inits, [1, 6, 8, 8], "inner_ends")
    seedf = "seedf"
    nodes.append(helper.make_node("Slice", [IN_NAME, starts, ends], [seedf]))

    kernel = np.zeros((1, 6, 3, 3), dtype=np.float32)
    for rr, cc in ((0, 1), (1, 0), (1, 2), (2, 1)):
        kernel[0, 0, rr, cc] = -1.0
        kernel[0, 1, rr, cc] = 1.0
    for rr, cc in ((0, 0), (0, 2), (2, 0), (2, 2)):
        kernel[0, 0, rr, cc] = -1.0
        kernel[0, 5, rr, cc] = 1.0

    bias = np.zeros((6,), dtype=np.float32)
    bias[0] = 1.0
    weight = _f32(inits, kernel, "kernel")
    bias_name = _f32(inits, bias, "bias")
    nodes.extend(
        [
            helper.make_node("ConvTranspose", [seedf, weight, bias_name], ["out6"]),
            helper.make_node("Pad", ["out6"], [OUT_NAME], pads=[0, 0, 0, 0, 0, C - 6, H - G, W - G]),
        ]
    )
    return _make_model(nodes, inits, "task282_fused_convtranspose")


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            g = np.asarray(ex["input"], dtype=np.int64)
            expected = np.asarray(ex["output"], dtype=np.int64)
            reference = solve(g)
            if not np.array_equal(reference, expected):
                raise AssertionError(f"reference solver mismatch on {split}[{idx}]")

            pred_oh = _run_onnx(model, _grid_to_onehot(ex["input"]))
            pred = _onehot_to_grid(pred_oh)[: g.shape[0], : g.shape[1]]
            active = pred_oh[0, :, : expected.shape[0], : expected.shape[1]] > 0.0
            if not np.array_equal(pred, expected) or not np.all(active.sum(axis=0) == 1):
                bad += 1
    return bad


def _score_candidate(label: str, build: Callable[[], onnx.ModelProto]) -> tuple[int, float, onnx.ModelProto]:
    model = build()
    bad = validate_json(model)
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
        f"{label}: nodes={len(model.graph.node)} memory={result['memory']} "
        f"params={result['params']} cost={result['cost']} score={result['score']:.6f}"
    )
    return int(result["cost"]), float(result["score"]), model


def main() -> None:
    candidates = [
        _score_candidate("direct-shift", build_direct_shift_model),
        _score_candidate("conv", build_conv_model),
        _score_candidate("convtranspose", build_convtranspose_model),
        _score_candidate("float-convtranspose", build_float_convtranspose_model),
        _score_candidate("fused-convtranspose", build_fused_convtranspose_model),
    ]
    _cost, _score, best = min(candidates, key=lambda item: item[0])
    onnx.save(best, BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"nodes:   {len(best.graph.node)}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
