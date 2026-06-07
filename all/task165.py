"""Build ONNX for task165: grow marker columns under the roof object.

Task rule: the input is a 20x20 grid with two foreground colors. One color is
the fixed 10-pixel roof shape; the other color is a sparse set of marker
pixels. The roof color is the color with exactly 10 cells, 8 of which have an
orthogonal same-color neighbor. For each marker column that has a marker below
the local bottom of the roof and whose x coordinate is occupied by the roof,
draw the marker color downward from just below that local roof bottom to the
bottom of the 20x20 grid. Preserve all other input cells.
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

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

TASK_ID = "task165"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
FG = C - 1
N = 20
H = W = 30
PAD = H - N
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def _i64(inits: list[onnx.TensorProto], vals: list[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))
    return name


def _f32(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name))
    return name


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference implementation used for local hypothesis validation."""
    arr = np.asarray(grid, dtype=np.int64)
    out = arr.copy()
    colors = [int(c) for c in sorted(set(arr.ravel())) if c != 0]

    roof_color = None
    for color in colors:
        coords = np.argwhere(arr == color)
        adjacent = 0
        for row, col in coords:
            has_neighbor = False
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr = int(row) + dr
                cc = int(col) + dc
                if 0 <= rr < arr.shape[0] and 0 <= cc < arr.shape[1] and arr[rr, cc] == color:
                    has_neighbor = True
                    break
            adjacent += int(has_neighbor)
        if len(coords) == 10 and adjacent == 8:
            roof_color = color
            break

    if roof_color is None:
        return out

    marker_colors = [color for color in colors if color != roof_color]
    if not marker_colors:
        return out
    marker_color = marker_colors[0]

    roof = arr == roof_color
    marker = arr == marker_color
    for col in sorted(set(np.where(marker)[1].tolist())):
        roof_rows = np.where(roof[:, col])[0]
        if len(roof_rows) == 0:
            continue
        start = int(roof_rows.max()) + 1
        if not np.any(np.where(marker[:, col])[0] >= start):
            continue
        for row in range(start, arr.shape[0]):
            if out[row, col] in (0, marker_color):
                out[row, col] = marker_color
    return out


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    start20 = _i64(inits, [0, 0, 0, 0], "start20")
    fg_start = _i64(inits, [0, 1, 0, 0], "fg_start")
    fg_end = _i64(inits, [1, C, N, N], "fg_end")
    ch0_end = _i64(inits, [1, 1, N, N], "ch0_end")
    zero = _f32(inits, [0.0], "zero")
    nine_half = _f32(inits, [9.5], "nine_half")

    roof_template = np.zeros((FG, 1, 4, 7), dtype=np.float32)
    roof_cells = [(0, 3), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 4), (2, 5), (3, 0), (3, 6)]
    for row, col in roof_cells:
        roof_template[:, 0, row, col] = 1.0
    _f32(inits, roof_template, "roof_template")

    after_template = np.zeros((1, 1, N, 7), dtype=np.float32)
    local_bottom = {0: 3, 1: 2, 2: 2, 3: 1, 4: 2, 5: 2, 6: 3}
    for col, bottom in local_bottom.items():
        after_template[0, 0, bottom + 1 :, col] = 1.0
    _f32(inits, after_template, "after_template")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, fg_start, fg_end, axes4], ["fg"]),
            helper.make_node(
                "Conv",
                ["fg", "roof_template"],
                ["roof_match_score"],
                group=FG,
                kernel_shape=[4, 7],
            ),
            helper.make_node("Greater", ["roof_match_score", "nine_half"], ["roof_hit"]),
            helper.make_node("Cast", ["roof_hit"], ["roof_hit_f"], to=TensorProto.FLOAT),
            helper.make_node("ReduceSum", ["roof_hit_f"], ["roof_loc"], axes=[1], keepdims=1),
            helper.make_node(
                "ConvTranspose",
                ["roof_loc", "after_template"],
                ["after_big"],
                kernel_shape=[N, 7],
            ),
            helper.make_node("Slice", ["after_big", "start20", "ch0_end", "axes4"], ["afterf"]),
            helper.make_node("Mul", ["fg", "afterf"], ["fg_after"]),
            helper.make_node("ReduceSum", ["fg_after"], ["marker_cols"], axes=[2], keepdims=1),
            helper.make_node("Greater", ["marker_cols", zero], ["selected_cols"]),
            helper.make_node("Greater", ["fg", zero], ["fg_bool"]),
            helper.make_node("Greater", ["afterf", zero], ["after_bool"]),
            helper.make_node("And", ["selected_cols", "after_bool"], ["line_fg_bool"]),
            helper.make_node("Or", ["fg_bool", "line_fg_bool"], ["yfg"]),
            helper.make_node("ReduceSum", ["marker_cols"], ["selected_any_f"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["selected_any_f", zero], ["selected_any"]),
            helper.make_node("And", ["selected_any", "after_bool"], ["line_any"]),
            helper.make_node("Slice", [IN_NAME, start20, ch0_end, axes4], ["ch0"]),
            helper.make_node("Greater", ["ch0", zero], ["ch0_bool"]),
            helper.make_node("Not", ["line_any"], ["not_line_any"]),
            helper.make_node("And", ["ch0_bool", "not_line_any"], ["y0"]),
            helper.make_node("Concat", ["y0", "yfg"], ["y20_bool"], axis=1),
            helper.make_node("Cast", ["y20_bool"], ["y20"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["y20"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]),
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


def _grid_to_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            out[0, int(arr[row, col]), row, col] = 1.0
    return out


def verify_reference() -> None:
    data = json.loads(DATA_PATH.read_text())
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            pred = solve(example["input"])
            expected = np.asarray(example["output"], dtype=np.int64)
            if not np.array_equal(pred, expected):
                raise AssertionError(f"reference mismatch on {split} example {idx}")


def verify_onnx(path: Path) -> None:
    data = json.loads(DATA_PATH.read_text())
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            pred = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(example["input"])})[0]
            expected = _grid_to_onehot(example["output"])
            if not np.array_equal(pred > 0.0, expected > 0.0):
                raise AssertionError(f"ONNX mismatch on {split} example {idx}")


def save_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    model = build_model()
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return model


def main() -> None:
    verify_reference()
    save_model(BEST_PATH)
    verify_onnx(BEST_PATH)
    print(f"wrote {BEST_PATH}")


if __name__ == "__main__":
    main()
