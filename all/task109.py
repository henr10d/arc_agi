"""Build a compact ONNX solution for NeuroGolf task109.

Task rule: the input contains a colored separator cross splitting the visible
grid into quadrants. Copy the non-background pattern in the top-left quadrant
into a separator-free output: top-left original, top-right horizontally
mirrored, bottom-left vertically mirrored, and bottom-right mirrored both ways.
All copied foreground cells are recolored to the separator color; output
background cells are color 0 and padding outside the output grid is all-zero.
The task data uses quadrant sizes 3, 4, 5, and 6.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file

TASK_ID = "task109"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task109.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

IN_NAME = "input"
OUT_NAME = "output"
C = 10
H = W = 30
SHAPE = [1, C, H, W]
MAX_Q = 6
MAX_OUT = 2 * MAX_Q
OPSET = 10
IR_VERSION = 10


def _init(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], name: str, values: list[int]) -> str:
    return _init(inits, name, np.asarray(values, dtype=np.int64))


def _f32(inits: list[onnx.TensorProto], name: str, values: list[float]) -> str:
    return _init(inits, name, np.asarray(values, dtype=np.float32))


def _bool_mask(inits: list[onnx.TensorProto], name: str, size: int) -> str:
    arr = np.zeros((1, 1, MAX_OUT, MAX_OUT), dtype=bool)
    arr[:, :, : 2 * size, : 2 * size] = True
    return _init(inits, name, arr)


def _slice_chw(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    axes_name: str,
    source: str,
    output: str,
    starts: list[int],
    ends: list[int],
) -> str:
    starts_name = _i64(inits, f"{output}_s", starts)
    ends_name = _i64(inits, f"{output}_e", ends)
    nodes.append(helper.make_node("Slice", [source, starts_name, ends_name, axes_name], [output]))
    return output


def _false_const(inits: list[onnx.TensorProto], name: str, shape: tuple[int, ...]) -> str:
    return _init(inits, name, np.zeros(shape, dtype=bool))


def _pad12(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    source: str,
    output: str,
    size: int,
) -> str:
    pad = MAX_OUT - 2 * size
    if pad == 0:
        return source
    right = _false_const(inits, f"{output}_right", (1, 1, 2 * size, pad))
    bottom = _false_const(inits, f"{output}_bottom", (1, 1, pad, MAX_OUT))
    nodes.append(helper.make_node("Concat", [source, right], [f"{output}_wide"], axis=3))
    nodes.append(helper.make_node("Concat", [f"{output}_wide", bottom], [output], axis=2))
    return output


def _candidate(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    axes_name: str,
    pattern: str,
    size: int,
) -> str:
    if size == MAX_Q:
        p = pattern
    else:
        p = _slice_chw(nodes, inits, axes_name, pattern, f"p{size}", [0, 0, 0], [1, size, size])
    rev = _i64(inits, f"rev{size}", list(range(size - 1, -1, -1)))
    nodes.append(helper.make_node("Gather", [p, rev], [f"h{size}"], axis=3))
    nodes.append(helper.make_node("Gather", [p, rev], [f"v{size}"], axis=2))
    nodes.append(helper.make_node("Gather", [f"h{size}", rev], [f"vh{size}"], axis=2))
    nodes.append(helper.make_node("Concat", [p, f"h{size}"], [f"top{size}"], axis=3))
    nodes.append(helper.make_node("Concat", [f"v{size}", f"vh{size}"], [f"bot{size}"], axis=3))
    nodes.append(helper.make_node("Concat", [f"top{size}", f"bot{size}"], [f"o{size}"], axis=2))
    return _pad12(nodes, inits, f"o{size}", f"o{size}p", size)


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    axes_chw = _i64(inits, "axes_chw", [1, 2, 3])
    half = _f32(inits, "half", [0.5])

    # Visible-grid size detection via the background cell at row 2n, column 0.
    # That cell is color 0 inside the visible grid, but all-zero in padding.
    row_flags: dict[int, str] = {}
    for row in (6, 8, 10, 12):
        r = _slice_chw(nodes, inits, axes_chw, IN_NAME, f"r{row}", [0, row, 0], [1, row + 1, 1])
        nodes.append(helper.make_node("Cast", [r], [f"has{row}"], to=TensorProto.BOOL))
        row_flags[row] = f"has{row}"

    nodes.append(helper.make_node("Not", [row_flags[8]], ["not8"]))
    nodes.append(helper.make_node("Not", [row_flags[10]], ["not10"]))
    nodes.append(helper.make_node("Not", [row_flags[12]], ["not12"]))
    nodes.append(helper.make_node("And", [row_flags[6], "not8"], ["is3"]))
    nodes.append(helper.make_node("And", [row_flags[8], "not10"], ["is4"]))
    nodes.append(helper.make_node("And", [row_flags[10], "not12"], ["is5"]))
    # is6 is has12.

    # Separator color is the non-background one-hot value at (size, 0).
    for size, flag in ((3, "is3"), (4, "is4"), (5, "is5"), (6, row_flags[12])):
        s = _slice_chw(nodes, inits, axes_chw, IN_NAME, f"sep{size}x", [1, size, 0], [C, size + 1, 1])
        nodes.append(helper.make_node("Cast", [s], [f"sep{size}"], to=TensorProto.BOOL))
        nodes.append(helper.make_node("And", [flag, f"sep{size}"], [f"sep{size}s"]))
    nodes.append(helper.make_node("Or", ["sep3s", "sep4s"], ["sep34"]))
    nodes.append(helper.make_node("Or", ["sep5s", "sep6s"], ["sep56"]))
    nodes.append(helper.make_node("Or", ["sep34", "sep56"], ["sep"]))

    _slice_chw(nodes, inits, axes_chw, IN_NAME, "x6", [0, 0, 0], [1, MAX_Q, MAX_Q])
    nodes.append(helper.make_node("Less", ["x6", half], ["p6"]))
    candidates = {size: _candidate(nodes, inits, axes_chw, "p6", size) for size in (3, 4, 5, 6)}
    valid_masks = {size: _bool_mask(inits, f"valid{size}", size) for size in (3, 4, 5, 6)}

    for size, flag in ((3, "is3"), (4, "is4"), (5, "is5"), (6, row_flags[12])):
        nodes.append(helper.make_node("And", [flag, candidates[size]], [f"fg{size}s"]))
        nodes.append(helper.make_node("And", [flag, valid_masks[size]], [f"va{size}s"]))
    nodes.append(helper.make_node("Or", ["fg3s", "fg4s"], ["fg34"]))
    nodes.append(helper.make_node("Or", ["fg5s", "fg6s"], ["fg56"]))
    nodes.append(helper.make_node("Or", ["fg34", "fg56"], ["fg"]))
    nodes.append(helper.make_node("Or", ["va3s", "va4s"], ["va34"]))
    nodes.append(helper.make_node("Or", ["va5s", "va6s"], ["va56"]))
    nodes.append(helper.make_node("Or", ["va34", "va56"], ["valid"]))

    nodes.append(helper.make_node("Not", ["fg"], ["notfg"]))
    nodes.append(helper.make_node("And", ["valid", "notfg"], ["bg"]))
    nodes.append(helper.make_node("And", ["fg", "sep"], ["fgc"]))
    nodes.append(helper.make_node("Concat", ["bg", "fgc"], ["out12b"], axis=1))
    nodes.append(helper.make_node("Cast", ["out12b"], ["out12"], to=TensorProto.FLOAT))
    nodes.append(
        helper.make_node(
            "Pad",
            ["out12"],
            [OUT_NAME],
            pads=[0, 0, 0, 0, 0, 0, H - MAX_OUT, W - MAX_OUT],
        )
    )

    graph = helper.make_graph(nodes, "task109", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="ng109",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def validate(path: Path) -> tuple[int, int, str | None]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            total += 1
            got = session.run([OUT_NAME], {IN_NAME: grid_to_onehot(ex["input"])})[0] > 0.0
            want = grid_to_onehot(ex["output"]) > 0.0
            if np.array_equal(got, want):
                passed += 1
                continue
            diff = np.argwhere(got != want)
            return passed, total, f"{split}[{idx}] mismatch at {diff[0].tolist()}"
    return passed, total, None


def op_counts(model: onnx.ModelProto) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    passed, total, error = validate(BEST_PATH)
    result = score_file(BEST_PATH)

    print(f"wrote {BEST_PATH}")
    print(f"correctness: {passed}/{total}" + (f" ({error})" if error else ""))
    print(f"ops: {op_counts(model)}")
    print(f"filesize: {BEST_PATH.stat().st_size}")
    if result["valid"]:
        print(f"params: {result['params']}")
        print(f"memory: {result['memory']}")
        print(f"cost: {result['cost']}")
        print(f"score: {float(result['score']):.6f}")
    else:
        print(f"score invalid: {result['error']}")
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
