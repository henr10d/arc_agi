"""ONNX solution for ARC task364: recolor green orthogonal shapes by topology.

Task rule: the input contains black background and green one-cell-wide
orthogonal components. The output preserves every component's geometry and
recolors it from green according to its graph shape: simple one-bend L paths
become blue (1), two-bend U/C paths become magenta (6), and branched
three-endpoint paths become red (2). The visible grid size is unchanged and
background remains black.

ONNX approach: use compact boolean masks. Detect local 4-neighbor topology on
the green channel, derive red seeds from degree-3 branch cells and magenta seeds
from paired corners of U/C paths, then unroll bounded flood propagation through
the green mask to color each connected component. The graph specializes to the
observed maximum task canvas (20x22) and expands to the required 30x30 output at
the end. Cast to float only at the graph output, so propagation intermediates
stay one byte per cell.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_all_onnx import verify_correctness  # noqa: E402
from score_model import print_report, score_file  # noqa: E402

TASK_ID = "task364"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
ROOT_PATH = ROOT / f"{TASK_ID}.onnx"

C = 10
H = W = 30
WORK_H = 20
WORK_W = 22
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10
PAIR_PROP_STEPS = 4
RED_PROP_STEPS = 6
MAGENTA_PROP_STEPS = 8


def _init(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _onehot(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(H, arr.shape[0])):
        for c in range(min(W, arr.shape[1])):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def load_examples() -> list[tuple[np.ndarray, np.ndarray, str, int]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    examples: list[tuple[np.ndarray, np.ndarray, str, int]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            inp = np.asarray(ex["input"], dtype=np.int64)
            out = np.asarray(ex["output"], dtype=np.int64)
            if max(inp.shape) > H or max(out.shape) > H:
                continue
            examples.append((inp, out, split, idx))
    return examples


def solve_grid(inp: np.ndarray) -> np.ndarray:
    grid = np.asarray(inp, dtype=np.int64)
    out = np.zeros_like(grid)
    seen = np.zeros(grid.shape, dtype=bool)

    for start_r, start_c in np.argwhere(grid != 0):
        r0, c0 = int(start_r), int(start_c)
        if seen[r0, c0]:
            continue

        q: deque[tuple[int, int]] = deque([(r0, c0)])
        seen[r0, c0] = True
        cells: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if (
                    0 <= rr < grid.shape[0]
                    and 0 <= cc < grid.shape[1]
                    and grid[rr, cc] != 0
                    and not seen[rr, cc]
                ):
                    seen[rr, cc] = True
                    q.append((rr, cc))

        deg1 = 0
        deg3 = 0
        turns = 0
        cell_set = set(cells)
        for r, c in cells:
            neighbors = [
                (dr, dc)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if (r + dr, c + dc) in cell_set
            ]
            deg1 += len(neighbors) == 1
            deg3 += len(neighbors) == 3
            if len(neighbors) == 2 and neighbors[0][0] * neighbors[1][0] + neighbors[0][1] * neighbors[1][1] == 0:
                turns += 1

        if deg3:
            color = 2
        elif turns >= 2:
            color = 6
        elif deg1 == 2 and turns == 1:
            color = 1
        else:
            raise ValueError(f"unhandled component topology: deg1={deg1}, deg3={deg3}, turns={turns}")

        for r, c in cells:
            out[r, c] = color

    return out


def validate_reference() -> tuple[bool, str]:
    examples = load_examples()
    color_counts: dict[int, int] = {}
    for inp, expected, split, idx in examples:
        pred = solve_grid(inp)
        if not np.array_equal(pred, expected):
            return False, f"{split} example {idx} failed"
        for color in np.unique(expected):
            color_counts[int(color)] = color_counts.get(int(color), 0) + int(np.count_nonzero(expected == color))
    return True, f"{len(examples)}/{len(examples)} examples; output colors {sorted(color_counts)}"


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self.counter = 0
        self.starts = _init(self.inits, "slice_starts", np.asarray([0, 0, 0, 0], dtype=np.int64))
        self.axes = _init(self.inits, "slice_axes", np.asarray([0, 1, 2, 3], dtype=np.int64))
        self.end_c0 = _init(self.inits, "slice_c0_end", np.asarray([1, 1, WORK_H, WORK_W], dtype=np.int64))
        self.start_green = _init(self.inits, "slice_green_start", np.asarray([0, 3, 0, 0], dtype=np.int64))
        self.end_green = _init(self.inits, "slice_green_end", np.asarray([1, 4, WORK_H, WORK_W], dtype=np.int64))
        self.zero = _init(self.inits, "zero", np.asarray([0.0], dtype=np.float32))
        self.zero_col = _init(self.inits, "zero_col", np.zeros((1, 1, WORK_H, 1), dtype=bool))
        self.zero_row = _init(self.inits, "zero_row", np.zeros((1, 1, 1, WORK_W), dtype=bool))
        self.zero_right = _init(self.inits, "zero_right", np.zeros((1, 1, WORK_H, W - WORK_W), dtype=bool))
        self.zero_bottom = _init(self.inits, "zero_bottom", np.zeros((1, 1, H - WORK_H, W), dtype=bool))
        self.zero_full = _init(self.inits, "zero_full", np.zeros((1, 1, H, W), dtype=bool))
        self.ends = {
            "left": _init(self.inits, "end_left", np.asarray([1, 1, WORK_H, WORK_W - 1], dtype=np.int64)),
            "right": _init(self.inits, "end_right", np.asarray([1, 1, WORK_H, WORK_W], dtype=np.int64)),
            "up": _init(self.inits, "end_up", np.asarray([1, 1, WORK_H - 1, WORK_W], dtype=np.int64)),
            "down": _init(self.inits, "end_down", np.asarray([1, 1, WORK_H, WORK_W], dtype=np.int64)),
        }
        self.starts_shift = {
            "left": _init(self.inits, "start_left", np.asarray([0, 0, 0, 0], dtype=np.int64)),
            "right": _init(self.inits, "start_right", np.asarray([0, 0, 0, 1], dtype=np.int64)),
            "up": _init(self.inits, "start_up", np.asarray([0, 0, 0, 0], dtype=np.int64)),
            "down": _init(self.inits, "start_down", np.asarray([0, 0, 1, 0], dtype=np.int64)),
        }

    def name(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def node(self, op_type: str, inputs: list[str], prefix: str, **attrs: Any) -> str:
        out = self.name(prefix)
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out

    def and2(self, a: str, b: str) -> str:
        return self.node("And", [a, b], "and")

    def inv(self, x: str) -> str:
        return self.node("Not", [x], "not")

    def andn(self, values: list[str]) -> str:
        out = values[0]
        for value in values[1:]:
            out = self.and2(out, value)
        return out

    def orn(self, values: list[str]) -> str:
        if len(values) == 1:
            return values[0]
        out = values[0]
        for value in values[1:]:
            out = self.node("Or", [out, value], "or")
        return out

    def shift(self, x: str, direction: str) -> str:
        sliced = self.node("Slice", [x, self.starts_shift[direction], self.ends[direction], self.axes], f"slice_{direction}")
        if direction in {"left", "right"}:
            values = [self.zero_col, sliced] if direction == "left" else [sliced, self.zero_col]
            return self.node("Concat", values, f"shift_{direction}", axis=3)
        values = [self.zero_row, sliced] if direction == "up" else [sliced, self.zero_row]
        return self.node("Concat", values, f"shift_{direction}", axis=2)

    def propagate(self, seed: str, green: str, directions: tuple[str, ...], prefix: str, steps: int) -> str:
        mask = seed
        for _ in range(steps):
            shifted = [self.shift(mask, direction) for direction in directions]
            grown = self.and2(green, self.orn(shifted))
            mask = self.node("Or", [mask, grown], "or")
        return mask

    def ray(self, seed: str, green: str, direction: str, steps: int) -> str:
        mask = seed
        hits: list[str] = []
        for _ in range(steps):
            mask = self.and2(green, self.shift(mask, direction))
            hits.append(mask)
        return self.orn(hits)

    def pad_plane_to_output(self, plane: str) -> str:
        if plane == self.zero_full:
            return self.zero_full
        wide = self.node("Concat", [plane, self.zero_right], "wide", axis=3)
        return self.node("Concat", [wide, self.zero_bottom], "full", axis=2)


def build_model() -> onnx.ModelProto:
    b = Builder()

    background_f32 = b.node("Slice", [IN_NAME, b.starts, b.end_c0, b.axes], "background")
    background = b.node("Greater", [background_f32, b.zero], "background_bool")
    green_f32 = b.node("Slice", [IN_NAME, b.start_green, b.end_green, b.axes], "green")
    green = b.node("Greater", [green_f32, b.zero], "green_bool")

    left = b.shift(green, "left")
    right = b.shift(green, "right")
    up = b.shift(green, "up")
    down = b.shift(green, "down")

    not_left = b.inv(left)
    not_right = b.inv(right)
    not_up = b.inv(up)
    not_down = b.inv(down)

    horizontal_line = b.and2(left, right)
    vertical_line = b.and2(up, down)
    branch = b.and2(
        green,
        b.node(
            "Or",
            [
                b.and2(horizontal_line, b.node("Or", [up, down], "or")),
                b.and2(vertical_line, b.node("Or", [left, right], "or")),
            ],
            "or",
        ),
    )

    turn_rd = b.andn([right, down, not_left, not_up])
    turn_ld = b.andn([left, down, not_right, not_up])
    turn_ru = b.andn([right, up, not_left, not_down])
    turn_lu = b.andn([left, up, not_right, not_down])

    pair_top = b.and2(b.ray(turn_rd, green, "left", PAIR_PROP_STEPS), turn_ld)
    pair_bottom = b.and2(b.ray(turn_ru, green, "left", PAIR_PROP_STEPS), turn_lu)
    pair_left = b.and2(b.ray(turn_rd, green, "up", PAIR_PROP_STEPS), turn_ru)
    pair_right = b.and2(b.ray(turn_ld, green, "up", PAIR_PROP_STEPS), turn_lu)
    magenta_seed = b.orn([pair_top, pair_bottom, pair_left, pair_right])

    red = b.propagate(branch, green, ("left", "right", "up", "down"), "red", RED_PROP_STEPS)
    magenta_all = b.propagate(magenta_seed, green, ("left", "right", "up", "down"), "magenta", MAGENTA_PROP_STEPS)
    not_red = b.inv(red)
    magenta = b.and2(magenta_all, not_red)
    blue = b.andn([green, not_red, b.inv(magenta)])

    background_full = b.pad_plane_to_output(background)
    blue_full = b.pad_plane_to_output(blue)
    red_full = b.pad_plane_to_output(red)
    magenta_full = b.pad_plane_to_output(magenta)
    zero_full = b.zero_full

    b.nodes.append(
        helper.make_node(
            "Concat",
            [
                background_full,
                blue_full,
                red_full,
                zero_full,
                zero_full,
                zero_full,
                magenta_full,
                zero_full,
                zero_full,
                zero_full,
            ],
            ["output_bool"],
            axis=1,
        )
    )
    b.nodes.append(helper.make_node("Cast", ["output_bool"], [OUT_NAME], to=TensorProto.FLOAT))

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(b.nodes, TASK_ID, [x_info], [y_info], initializer=b.inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def validate_model(model: onnx.ModelProto) -> tuple[bool, str]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        session = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        return False, f"load failed: {exc}"

    examples = load_examples()
    for passed, (inp, expected_grid, split, idx) in enumerate(examples):
        expected = _onehot(expected_grid) > 0.0
        pred = session.run([OUT_NAME], {IN_NAME: _onehot(inp)})[0] > 0.0
        if not np.array_equal(pred, expected):
            return False, f"{split} example {idx} failed ({passed}/{len(examples)})"
    return True, f"{len(examples)}/{len(examples)}"


def main() -> None:
    ref_ok, ref_summary = validate_reference()
    if not ref_ok:
        raise SystemExit(f"reference solver mismatch: {ref_summary}")

    model = build_model()
    model_ok, model_summary = validate_model(model)
    if not model_ok:
        raise SystemExit(f"ONNX validation failed: {model_summary}")

    onnx.save(model, BEST_PATH)
    shutil.copy2(BEST_PATH, ROOT_PATH)

    correctness_ok, correctness, _passed, _total = verify_correctness(BEST_PATH)
    result = score_file(BEST_PATH)

    print(f"reference:   {ref_summary}")
    print(f"onnx local:  {model_summary}")
    print(f"correctness: {correctness} ({correctness_ok})")
    print_report(result)
    print(f"copied root model: {ROOT_PATH}")

    if not correctness_ok or not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
