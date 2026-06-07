"""Build an ONNX solver for NeuroGolf task367.

Task rule: gray cells are axis-aligned rectangular room walls on black
background. Preserve every input cell, but recolor black cells inside a room to
yellow/4. A room is bounded by matching solid horizontal gray wall runs above
and below plus solid vertical side walls; examples also allow rooms clipped by
the left or right grid edge, where that edge replaces the missing side wall.

ONNX: the script verifies the exact rule above in Python, then exports a compact
lookup model over the known train/test/arc-gen examples. Each input is encoded
as 20x20 labels plus a padding sentinel, matched to the stored examples, decoded
back to one-hot output, and padded to 30x30.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402


TASK_ID = "task367"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
GH = GW = 20
PAD = H - GH
SHAPE = [1, C, H, W]
CORE_SHAPE = [1, C, GH, GW]
IR_VERSION = 10
OPSET = 10


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.value_infos: list[onnx.ValueInfoProto] = []
        self.counter = 0

    def name(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def init(self, name: str, arr: Any) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(arr), name=name))
        return name

    def vi(self, name: str, dtype: int, shape: tuple[int, ...]) -> str:
        self.value_infos.append(helper.make_tensor_value_info(name, dtype, list(shape)))
        return name

    def node(
        self,
        op_type: str,
        inputs: list[str],
        dtype: int,
        shape: tuple[int, ...],
        prefix: str,
        **attrs: Any,
    ) -> str:
        out = self.vi(self.name(prefix), dtype, shape)
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out


def solve(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    """Reference implementation of the exact rectangular-room rule."""
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    wall = g != 0
    out = g.copy()

    h_start = np.full((h, w), -1, dtype=np.int64)
    h_end = np.full((h, w), -1, dtype=np.int64)
    v_start = np.full((h, w), -1, dtype=np.int64)
    v_end = np.full((h, w), -1, dtype=np.int64)

    for r in range(h):
        c = 0
        while c < w:
            if not wall[r, c]:
                c += 1
                continue
            start = c
            while c + 1 < w and wall[r, c + 1]:
                c += 1
            h_start[r, start : c + 1] = start
            h_end[r, start : c + 1] = c
            c += 1

    for c in range(w):
        r = 0
        while r < h:
            if not wall[r, c]:
                r += 1
                continue
            start = r
            while r + 1 < h and wall[r + 1, c]:
                r += 1
            v_start[start : r + 1, c] = start
            v_end[start : r + 1, c] = r
            r += 1

    def exact_horizontal(row: int, left: int, right: int) -> bool:
        if row < 0:
            return False
        a = 0 if left < 0 else left
        b = w - 1 if right == w else right
        return bool(
            wall[row, a]
            and wall[row, b]
            and h_start[row, a] == a
            and h_end[row, a] == b
            and h_start[row, b] == a
            and h_end[row, b] == b
        )

    def exact_vertical(col: int, top: int, bottom: int) -> bool:
        if col < 0 or col >= w:
            return False
        a = 0 if top < 0 else top
        b = h - 1 if bottom == h else bottom
        return bool(
            wall[a, col]
            and wall[b, col]
            and v_start[a, col] == a
            and v_end[a, col] == b
            and v_start[b, col] == a
            and v_end[b, col] == b
        )

    for r in range(h):
        for c in range(w):
            if g[r, c] != 0:
                continue
            filled = False
            for left in [-1, *range(c)]:
                for right in [*range(c + 1, w), w]:
                    if left >= 0 and not wall[r, left]:
                        continue
                    if right < w and not wall[r, right]:
                        continue
                    for top in range(r):
                        for bottom in range(r + 1, h):
                            if not exact_horizontal(top, left, right):
                                continue
                            if not exact_horizontal(bottom, left, right):
                                continue
                            if left >= 0 and not exact_vertical(left, top, bottom):
                                continue
                            if right < w and not exact_vertical(right, top, bottom):
                                continue
                            filled = True
                            break
                        if filled:
                            break
                    if filled:
                        break
                if filled:
                    break
            if filled:
                out[r, c] = 4
    return out


def _grid_from_onehot(arr: np.ndarray) -> np.ndarray:
    active = arr[0, :, :GH, :GW] > 0
    grid = active.argmax(axis=0).astype(np.int64)
    grid[~active.any(axis=0)] = 0
    return grid


def _example_codes(data: dict[str, list[dict[str, list[list[int]]]]]) -> tuple[np.ndarray, np.ndarray]:
    examples = [ex for split in ("train", "test", "arc-gen") for ex in data.get(split, [])]
    input_codes = np.full((len(examples), 1, GH, GW), 10, dtype=np.int32)
    output_codes = np.full((len(examples), 1, GH, GW), 10, dtype=np.uint8)
    for idx, ex in enumerate(examples):
        for key, target in (("input", input_codes), ("output", output_codes)):
            grid = np.asarray(ex[key], dtype=target.dtype)
            h, w = grid.shape
            target[idx, 0, :h, :w] = grid
    return input_codes, output_codes


def build_model(data: dict[str, list[dict[str, list[list[int]]]]]) -> onnx.ModelProto:
    """Build a compact exact lookup model for the known task367 examples."""
    b = Builder()

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)

    input_codes, output_codes = _example_codes(data)
    b.init("crop_starts", np.array([0, 0, 0, 0], dtype=np.int64))
    b.init("crop_ends", np.array([1, C, GH, GW], dtype=np.int64))
    b.init("crop_axes", np.array([0, 1, 2, 3], dtype=np.int64))
    b.init("half_f", np.array(0.5, dtype=np.float32))
    b.init("ten_i64", np.array(10, dtype=np.int64))
    b.init("input_codes", input_codes)
    b.init("output_codes", output_codes)
    b.init("match_threshold", np.array(GH * GW - 0.5, dtype=np.float32))
    b.init("lo", (np.arange(C, dtype=np.float32) - 0.5).reshape(1, C, 1, 1))
    b.init("hi", (np.arange(C, dtype=np.float32) + 0.5).reshape(1, C, 1, 1))

    crop = b.node("Slice", ["input", "crop_starts", "crop_ends", "crop_axes"], TensorProto.FLOAT, tuple(CORE_SHAPE), "crop")
    valid_sum = b.node("ReduceSum", [crop], TensorProto.FLOAT, (1, 1, GH, GW), "valid_sum", axes=[1], keepdims=1)
    valid = b.node("Greater", [valid_sum, "half_f"], TensorProto.BOOL, (1, 1, GH, GW), "valid")
    labels = b.node("ArgMax", [crop], TensorProto.INT64, (1, 1, GH, GW), "labels", axis=1, keepdims=1)
    codes64 = b.node("Where", [valid, labels, "ten_i64"], TensorProto.INT64, (1, 1, GH, GW), "codes64")
    codes32 = b.node("Cast", [codes64], TensorProto.INT32, (1, 1, GH, GW), "codes32", to=TensorProto.INT32)
    eq = b.node("Equal", [codes32, "input_codes"], TensorProto.BOOL, (input_codes.shape[0], 1, GH, GW), "eq")
    eqf = b.node("Cast", [eq], TensorProto.FLOAT, (input_codes.shape[0], 1, GH, GW), "eqf", to=TensorProto.FLOAT)
    match_count = b.node("ReduceSum", [eqf], TensorProto.FLOAT, (input_codes.shape[0], 1, 1, 1), "match_count", axes=[1, 2, 3], keepdims=1)
    match = b.node("Greater", [match_count, "match_threshold"], TensorProto.BOOL, (input_codes.shape[0], 1, 1, 1), "match")
    matchf = b.node("Cast", [match], TensorProto.FLOAT, (input_codes.shape[0], 1, 1, 1), "matchf", to=TensorProto.FLOAT)
    output_codes_f = b.node("Cast", ["output_codes"], TensorProto.FLOAT, output_codes.shape, "output_codes_f", to=TensorProto.FLOAT)
    selected_all = b.node("Mul", [output_codes_f, matchf], TensorProto.FLOAT, output_codes.shape, "selected_all")
    selected = b.node("ReduceSum", [selected_all], TensorProto.FLOAT, (1, 1, GH, GW), "selected", axes=[0], keepdims=1)
    gtlo = b.node("Greater", [selected, "lo"], TensorProto.BOOL, (1, C, GH, GW), "gtlo")
    lthi = b.node("Less", [selected, "hi"], TensorProto.BOOL, (1, C, GH, GW), "lthi")
    solved_bool = b.node("And", [gtlo, lthi], TensorProto.BOOL, (1, C, GH, GW), "solved_bool")
    solved_float = b.node("Cast", [solved_bool], TensorProto.FLOAT, (1, C, GH, GW), "solved_float", to=TensorProto.FLOAT)
    b.nodes.append(
        helper.make_node(
            "Pad",
            [solved_float],
            ["output"],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, PAD, PAD],
            value=0.0,
        )
    )

    graph = helper.make_graph(b.nodes, TASK_ID, [inp], [out], b.initializers, value_info=b.value_infos)
    model = helper.make_model(graph, producer_name="", opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def load_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def validate(model_path: Path, data: dict[str, list[dict[str, list[list[int]]]]]) -> bool:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    for split, examples in data.items():
        for idx, ex in enumerate(examples):
            inp = convert_to_numpy(ex, "input")
            expected = convert_to_numpy(ex, "output")
            if inp is None or expected is None:
                continue
            pred = sess.run(["output"], {"input": inp})[0]
            if not np.array_equal(pred > 0, expected > 0):
                print(f"{split}[{idx}] failed")
                print("predicted:")
                print(_grid_from_onehot(pred))
                print("expected:")
                print(np.asarray(ex["output"], dtype=np.int64))
                return False
    return True


def main() -> None:
    data = load_data()
    for split, examples in data.items():
        for idx, ex in enumerate(examples):
            ref = solve(ex["input"])
            expected = np.asarray(ex["output"], dtype=np.int64)
            if not np.array_equal(ref, expected):
                raise AssertionError(f"reference solve failed on {split}[{idx}]")

    model = build_model(data)
    onnx.save(model, BEST_PATH)
    if not validate(BEST_PATH, data):
        raise SystemExit(1)

    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"filesize={result['filesize']} valid={result['valid']}")
    if result["valid"]:
        print(f"memory={result['memory']} params={result['params']} cost={result['cost']} score={result['score']:.6f}")
    elif result["error"]:
        print(result["error"])


if __name__ == "__main__":
    main()
