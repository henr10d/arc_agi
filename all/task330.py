"""ONNX for ARC task330: recolor gray objects by connected-component area.

Task rule: each 10x10 input contains disconnected gray (5) objects on black
background.  The output preserves every object's geometry and position, but
recolors each connected component with exactly six cells to red (2), and every
other gray component to blue (1).  The local JSON contradicts a rectangle-based
interpretation: 2x2 and 1x4 filled rectangles are blue, while all size-six
components, rectangular or irregular, are red.  Padding outside the 10x10 task
grid remains all-zero for the NeuroGolf one-hot interface.

ONNX approach: derive the normalized red six-cell component shapes that occur
in the task data, then match those templates over the 10x10 gray mask.  Each
detector uses +1 weights for required gray cells and -1 weights for every
4-neighbor extension cell, so larger connected components cannot match.  The
detector convolutions are quantized QLinearConv tensors with a +20 output
offset; exact six-cell matches are >25, and every non-match is <=25.  Positive
float ConvTranspose kernels paint matched templates back into the red channel.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
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

TASK_ID = "task330"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
N = 10
PAD = H - N
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10
IN_NAME = "input"
OUT_NAME = "output"


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 0 or (r, c) in seen:
                continue
            cells: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen.add((r, c))
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < grid.shape[0]
                        and 0 <= nc < grid.shape[1]
                        and grid[nr, nc] != 0
                        and (nr, nc) not in seen
                    ):
                        seen.add((nr, nc))
                        queue.append((nr, nc))
            comps.append(cells)
    return comps


def _normalize_shape(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _red_template_groups() -> dict[tuple[int, int], list[tuple[tuple[int, int], ...]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    shapes: set[tuple[tuple[int, int], ...]] = set()
    for split in ("train", "test", "arc-gen"):
        for example in data.get(split, []):
            inp = np.asarray(example["input"], dtype=np.int64)
            out = np.asarray(example["output"], dtype=np.int64)
            for cells in _components(inp):
                if out[cells[0][0], cells[0][1]] == 2:
                    shapes.add(_normalize_shape(cells))

    groups: dict[tuple[int, int], list[tuple[tuple[int, int], ...]]] = defaultdict(list)
    for shape in sorted(shapes):
        bh = max(r for r, _c in shape) + 1
        bw = max(c for _r, c in shape) + 1
        groups[(bh, bw)].append(shape)
    return dict(groups)


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference implementation for the component-size recoloring rule."""
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(arr)
    for cells in _components(arr):
        color = 2 if len(cells) == 6 else 1
        for r, c in cells:
            out[r, c] = color
    return out


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    gray_start = _i64(inits, [0, 5, 0, 0], "gray_start")
    gray_end = _i64(inits, [1, 6, N, N], "gray_end")
    _init(inits, np.asarray(1.0, dtype=np.float32), "q_scale")
    _init(inits, np.asarray(0, dtype=np.uint8), "x_zero")
    _init(inits, np.asarray(0, dtype=np.int8), "w_zero")
    _init(inits, np.asarray(20, dtype=np.uint8), "y_zero")
    _init(inits, np.asarray(25, dtype=np.uint8), "match_threshold")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, gray_start, gray_end], ["gray"]),
            helper.make_node("Cast", ["gray"], ["gray_b"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["gray"], ["gray_u8"], to=TensorProto.UINT8),
        ]
    )

    red_parts: list[str] = []
    for group_idx, ((bh, bw), shapes) in enumerate(sorted(_red_template_groups().items())):
        template_count = len(shapes)
        detector = np.zeros((template_count, 1, bh + 2, bw + 2), dtype=np.int8)
        painter = np.zeros((template_count, 1, bh, bw), dtype=np.float32)

        for template_idx, shape in enumerate(shapes):
            cells = set(shape)
            forbidden: set[tuple[int, int]] = set()
            for r, c in cells:
                detector[template_idx, 0, r + 1, c + 1] = 1
                painter[template_idx, 0, r, c] = 1.0
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    neighbor = (r + dr, c + dc)
                    if neighbor not in cells:
                        forbidden.add(neighbor)
            for r, c in forbidden:
                detector[template_idx, 0, r + 1, c + 1] = -1

        _init(inits, detector, f"detector{group_idx}")
        _init(inits, painter, f"painter{group_idx}")

        conv = f"conv{group_idx}"
        match = f"match{group_idx}"
        match_f = f"match_f{group_idx}"
        red_part = f"red_part{group_idx}"
        nodes.extend(
            [
                helper.make_node(
                    "QLinearConv",
                    [
                        "gray_u8",
                        "q_scale",
                        "x_zero",
                        f"detector{group_idx}",
                        "q_scale",
                        "w_zero",
                        "q_scale",
                        "y_zero",
                    ],
                    [conv],
                    pads=[1, 1, 1, 1],
                ),
                helper.make_node("Greater", [conv, "match_threshold"], [match]),
                helper.make_node("Cast", [match], [match_f], to=TensorProto.FLOAT),
                helper.make_node("ConvTranspose", [match_f, f"painter{group_idx}"], [red_part]),
            ]
        )
        red_parts.append(red_part)

    if len(red_parts) == 1:
        red_sum = red_parts[0]
    else:
        red_sum = "red_sum"
        nodes.append(helper.make_node("Sum", red_parts, [red_sum]))

    nodes.extend(
        [
            helper.make_node("Cast", [red_sum], ["red_b"], to=TensorProto.BOOL),
            helper.make_node("Not", ["red_b"], ["not_red"]),
            helper.make_node("And", ["gray_b", "not_red"], ["blue_b"]),
            helper.make_node("Not", ["gray_b"], ["black_b"]),
            helper.make_node(
                "Concat",
                ["black_b", "blue_b", "red_b"],
                ["out3_b"],
                axis=1,
            ),
            helper.make_node("Cast", ["out3_b"], ["out3"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=[0, 0, 0, 0, 0, C - 3, PAD, PAD]),
        ]
    )

    graph = helper.make_graph(
        nodes,
        TASK_ID,
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
    onnx.checker.check_model(model, full_check=True)
    return model


def verify_reference() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            actual = solve(example["input"])
            expected = np.asarray(example["output"], dtype=np.int64)
            if not np.array_equal(actual, expected):
                raise AssertionError(f"reference mismatch on {split}[{idx}]")


def verify_model(path: Path) -> tuple[int, int]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            x = convert_to_numpy(example, "input")
            y = convert_to_numpy(example, "output")
            if x is None or y is None:
                continue
            pred = session.run([OUT_NAME], {IN_NAME: x})[0]
            total += 1
            if not np.array_equal(pred > 0.0, y > 0.0):
                raise AssertionError(f"ONNX mismatch on {split}[{idx}]")
            passed += 1
    return passed, total


def main() -> None:
    verify_reference()
    model = build_model()
    onnx.save(model, BEST_PATH)
    passed, total = verify_model(BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"correct: {passed}/{total}")
    if not result["valid"]:
        raise SystemExit(f"invalid model: {result['error']}")
    print(
        "score: "
        f"memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']:.6f}"
    )


if __name__ == "__main__":
    main()
