"""ONNX solution for ARC task253 using rotation-specific L-triomino detection.

Task rule: the 13x13 input contains four non-black L triominoes, one in
each 2x2 rotation. The output is a 4x4 grid whose quadrants hold those
objects by orientation: missing lower-right in the upper-left quadrant,
missing lower-left in the upper-right, missing upper-right in the lower-left,
and missing upper-left in the lower-right. The four missing cells form the
central 2x2 black square. All examples use four distinct object colors.

The ONNX graph slices the 13x13 foreground channels and uses one grouped
2x2 convolution with fractional weights to encode each L orientation as a
distinct per-color score. Any non-anchor partial window scores lower than
the orientation codes, so a spatial ReduceMax followed by small interval
comparisons gives the color/orientation table for the final paint MatMul.
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
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task253"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
FG = 9
H = W = 30
IN_HW = 13
OUT_HW = 4
SHAPE = [1, C, H, W]
IR_VERSION = 10
OPSET = 10


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros((OUT_HW, OUT_HW), dtype=np.int64)
    placements = {
        ((1, 1), (1, 0)): (0, 0),
        ((1, 1), (0, 1)): (0, 2),
        ((1, 0), (1, 1)): (2, 0),
        ((0, 1), (1, 1)): (2, 2),
    }
    seen: set[tuple[int, int]] = set()
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            color = int(arr[r, c])
            if color == 0 or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < arr.shape[0]
                        and 0 <= nc < arr.shape[1]
                        and (nr, nc) not in seen
                        and int(arr[nr, nc]) == color
                    ):
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            min_r = min(x for x, _ in cells)
            min_c = min(y for _, y in cells)
            mask = tuple(
                tuple(1 if (min_r + i, min_c + j) in cells else 0 for j in range(2))
                for i in range(2)
            )
            out_r, out_c = placements[mask]
            for i in range(2):
                for j in range(2):
                    if mask[i][j]:
                        out[out_r + i, out_c + j] = color
    return out


def _init(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)

    axes = _init(inits, "axes", np.array([0, 1, 2, 3], dtype=np.int64))
    fg_st = _init(inits, "fg_st", np.array([0, 1, 0, 0], dtype=np.int64))
    fg_en = _init(inits, "fg_en", np.array([1, C, IN_HW, IN_HW], dtype=np.int64))
    shape_out = _init(inits, "shape_out", np.array([1, FG, OUT_HW, OUT_HW], dtype=np.int64))
    code_shape = _init(inits, "code_shape", np.array([1, FG, 1], dtype=np.int64))

    rotations = np.array(
        [
            [[1, 1], [1, 0]],
            [[1, 1], [0, 1]],
            [[1, 0], [1, 1]],
            [[0, 1], [1, 1]],
        ],
        dtype=np.float32,
    )
    code_kernel = np.array([[1.0, 1.25], [1.5, 1.75]], dtype=np.float32)
    lower = np.array([3.625, 3.875, 4.125, 4.375], dtype=np.float32).reshape(1, 1, 4)
    upper = np.array([3.875, 4.125, 4.375, 4.625], dtype=np.float32).reshape(1, 1, 4)
    weights = np.zeros((FG, 1, 2, 2), dtype=np.float32)
    for color in range(FG):
        weights[color, 0] = code_kernel
    _init(inits, "weights", weights)
    _init(inits, "lower", lower)
    _init(inits, "upper", upper)

    place = np.zeros((4, OUT_HW * OUT_HW), dtype=np.float32)
    for orient, (r0, c0) in enumerate(((0, 0), (0, 2), (2, 0), (2, 2))):
        for i in range(2):
            for j in range(2):
                if rotations[orient, i, j] > 0:
                    place[orient, (r0 + i) * OUT_HW + c0 + j] = 1.0
    _init(inits, "place", place)

    black = np.zeros((1, 1, OUT_HW, OUT_HW), dtype=np.float32)
    for r, c in ((1, 1), (1, 2), (2, 1), (2, 2)):
        black[0, 0, r, c] = 1.0
    _init(inits, "black", black)

    nodes.extend(
        [
            helper.make_node("Slice", ["input", fg_st, fg_en, axes], ["fg13"]),
            helper.make_node(
                "Conv",
                ["fg13", "weights"],
                ["hits"],
                group=FG,
                kernel_shape=[2, 2],
            ),
            helper.make_node("ReduceMax", ["hits"], ["code"], axes=[2, 3], keepdims=0),
            helper.make_node("Reshape", ["code", "code_shape"], ["code3"]),
            helper.make_node("Greater", ["code3", "lower"], ["above"]),
            helper.make_node("Less", ["code3", "upper"], ["below"]),
            helper.make_node("And", ["above", "below"], ["detb"]),
            helper.make_node("Cast", ["detb"], ["det"], to=TensorProto.FLOAT),
            helper.make_node("MatMul", ["det", "place"], ["fgflat"]),
            helper.make_node("Reshape", ["fgflat", "shape_out"], ["fg4"]),
            helper.make_node("Concat", ["black", "fg4"], ["full4"], axis=1),
            helper.make_node(
                "Pad",
                ["full4"],
                ["output"],
                pads=[0, 0, 0, 0, 0, 0, H - OUT_HW, W - OUT_HW],
            ),
        ]
    )

    graph = helper.make_graph(nodes, "task253", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _load_examples() -> list[tuple[str, int, dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [
        (split, idx, ex)
        for split in ("train", "test", "arc-gen")
        for idx, ex in enumerate(data.get(split, []))
    ]


def _verify_python() -> tuple[int, int]:
    examples = _load_examples()
    ok = 0
    for _, _, ex in examples:
        ok += int(np.array_equal(solve(ex["input"]), np.asarray(ex["output"], dtype=np.int64)))
    return ok, len(examples)


def _verify_onnx(path: Path) -> tuple[dict[str, int], dict[str, int]]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    examples = _load_examples()
    ok = {"train": 0, "test": 0, "arc-gen": 0}
    total = {"train": 0, "test": 0, "arc-gen": 0}
    for split, _, ex in examples:
        inp = convert_to_numpy(ex, "input")
        exp = convert_to_numpy(ex, "output")
        assert inp is not None and exp is not None
        pred = session.run(["output"], {"input": inp})[0]
        ok[split] += int(np.array_equal(pred > 0.0, exp > 0.0))
        total[split] += 1
    return ok, total


def _realized_tensor_count(model: onnx.ModelProto) -> int:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    io_names = {item.name for item in list(inferred.graph.input) + list(inferred.graph.output)}
    return sum(
        1
        for item in inferred.graph.value_info
        if item.name not in io_names and item.type.HasField("tensor_type")
    )


def main() -> None:
    py_ok, py_total = _verify_python()
    if py_ok != py_total:
        raise SystemExit(f"python reference failed: {py_ok}/{py_total}")

    model = build_model()
    onnx.save(model, BEST_PATH)

    onnx_ok, onnx_total = _verify_onnx(BEST_PATH)
    all_ok = sum(onnx_ok.values())
    all_total = sum(onnx_total.values())
    result = score_file(BEST_PATH)
    print(f"training accuracy: {onnx_ok['train']}/{onnx_total['train']}")
    print(f"all accuracy:      {all_ok}/{all_total}")
    print(f"python reference:  {py_ok}/{py_total}")
    print(f"tensor count:      {len(model.graph.node)} nodes")
    print(f"realized tensors:  {_realized_tensor_count(model)}")
    print(f"ONNX score:        {result.get('score')}")
    print(f"cost:              {result.get('cost')}")
    print(f"memory:            {result.get('memory')}")
    print(f"params:            {result.get('params')}")
    if all_ok != all_total:
        raise SystemExit(f"onnx verification failed: {all_ok}/{all_total}")
    if not result.get("valid"):
        raise SystemExit(f"score_model invalid: {result.get('error')}")


if __name__ == "__main__":
    main()
