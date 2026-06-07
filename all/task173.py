"""ONNX solution for ARC task173: complete repeated two-color 3x3 motifs.

Task rule: each input contains one or more prototype motifs made from a
non-zero center color and a second non-zero color occupying one of four
neighbor masks around it: horizontal endpoints, vertical endpoints, cardinal
cross arms, or diagonal corners.  Every partial copy of an inferred prototype
is completed: center-only copies receive the missing neighbor mask, and
neighbor-only copies receive the missing center.  Existing cells are preserved.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task173"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
FG = 9
IO_H = IO_W = 30
H = W = 25
HW = H * W
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, IO_H, IO_W]
FG_SHAPE = [1, FG, H, W]
FLAT_SHAPE = [1, FG, HW]
OPSET = 10
IR_VERSION = 10

MASKS: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "horizontal": ((0, -1), (0, 1)),
    "vertical": ((-1, 0), (1, 0)),
    "cardinal_cross": ((-1, 0), (1, 0), (0, -1), (0, 1)),
    "diagonal_corners": ((-1, -1), (-1, 1), (1, -1), (1, 1)),
}


def _load_examples() -> List[Tuple[str, int, np.ndarray, np.ndarray]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    out: List[Tuple[str, int, np.ndarray, np.ndarray]] = []
    for split in ("train", "test", "arc-gen"):
        for index, ex in enumerate(data.get(split, [])):
            out.append(
                (
                    split,
                    index,
                    np.asarray(ex["input"], dtype=np.int64),
                    np.asarray(ex["output"], dtype=np.int64),
                )
            )
    return out


def solve_with_masks(grid: np.ndarray, masks: Iterable[Sequence[Tuple[int, int]]]) -> np.ndarray:
    """Reference implementation for a selected set of neighbor masks."""
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    out = g.copy()

    for offsets in masks:
        offsets = tuple(offsets)
        prototypes: set[Tuple[int, int, Tuple[Tuple[int, int], ...]]] = set()
        for row in range(h):
            for col in range(w):
                center = int(g[row, col])
                if center == 0:
                    continue
                values: List[int] = []
                ok = True
                for dy, dx in offsets:
                    rr, cc = row + dy, col + dx
                    if not (0 <= rr < h and 0 <= cc < w):
                        ok = False
                        break
                    values.append(int(g[rr, cc]))
                if ok and values[0] != 0 and values[0] != center and all(v == values[0] for v in values):
                    prototypes.add((center, values[0], offsets))

        for center, outer, offsets in prototypes:
            for row in range(h):
                for col in range(w):
                    if all(
                        0 <= row + dy < h
                        and 0 <= col + dx < w
                        and g[row + dy, col + dx] == outer
                        for dy, dx in offsets
                    ):
                        if out[row, col] == 0:
                            out[row, col] = center

            for row, col in np.argwhere(g == center):
                for dy, dx in offsets:
                    rr, cc = int(row) + dy, int(col) + dx
                    if 0 <= rr < h and 0 <= cc < w and out[rr, cc] == 0:
                        out[rr, cc] = outer
    return out


def solve_reference(grid: np.ndarray) -> np.ndarray:
    return solve_with_masks(grid, MASKS.values())


def _bbox_horizontal_symmetry(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    coords = np.argwhere(grid)
    if len(coords) == 0:
        return out
    lo, hi = int(coords[:, 1].min()), int(coords[:, 1].max())
    for row, col in coords:
        cc = lo + hi - int(col)
        if 0 <= cc < grid.shape[1] and out[int(row), cc] == 0:
            out[int(row), cc] = grid[int(row), int(col)]
    return out


def _bbox_vertical_symmetry(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    coords = np.argwhere(grid)
    if len(coords) == 0:
        return out
    lo, hi = int(coords[:, 0].min()), int(coords[:, 0].max())
    for row, col in coords:
        rr = lo + hi - int(row)
        if 0 <= rr < grid.shape[0] and out[rr, int(col)] == 0:
            out[rr, int(col)] = grid[int(row), int(col)]
    return out


def _main_diagonal_symmetry(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for row, col in np.argwhere(grid):
        if col < grid.shape[0] and row < grid.shape[1] and out[int(col), int(row)] == 0:
            out[int(col), int(row)] = grid[int(row), int(col)]
    return out


def evaluate_candidates() -> List[Tuple[str, int, int]]:
    examples = _load_examples()
    candidates: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
        ("horizontal reflection", lambda g: solve_with_masks(g, [MASKS["horizontal"]])),
        ("vertical reflection", lambda g: solve_with_masks(g, [MASKS["vertical"]])),
        ("diagonal reflection", lambda g: solve_with_masks(g, [MASKS["diagonal_corners"]])),
        ("quadrant/cross completion", lambda g: solve_with_masks(g, [MASKS["cardinal_cross"]])),
        ("bbox horizontal symmetry", _bbox_horizontal_symmetry),
        ("bbox vertical symmetry", _bbox_vertical_symmetry),
        ("main diagonal symmetry", _main_diagonal_symmetry),
        ("motif completion", solve_reference),
    ]
    scores: List[Tuple[str, int, int]] = []
    for name, fn in candidates:
        train_ok = 0
        all_ok = 0
        for split, _, inp, expected in examples:
            ok = np.array_equal(fn(inp), expected)
            train_ok += int(split == "train" and ok)
            all_ok += int(ok)
        scores.append((name, train_ok, all_ok))
    return scores


class Builder:
    def __init__(self) -> None:
        self.nodes: List[onnx.NodeProto] = []
        self.inits: List[onnx.TensorProto] = []
        self._init_cache: Dict[Tuple[str, Tuple[int, ...], bytes], str] = {}
        self._counter = 0

    def name(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    def init(self, array: np.ndarray | Sequence[int] | Sequence[float], name: str | None = None) -> str:
        arr = np.asarray(array)
        key = (arr.dtype.str, tuple(arr.shape), arr.tobytes())
        cached = self._init_cache.get(key)
        if cached is not None:
            return cached
        name = name or self.name("k")
        self.inits.append(numpy_helper.from_array(arr, name=name))
        self._init_cache[key] = name
        return name

    def node(self, op_type: str, inputs: Sequence[str], prefix: str, **attrs: object) -> str:
        out = self.name(prefix)
        self.nodes.append(helper.make_node(op_type, list(inputs), [out], **attrs))
        return out

    def slice(self, x: str, starts: Sequence[int], ends: Sequence[int], axes: Sequence[int], prefix: str) -> str:
        return self.node(
            "Slice",
            [
                x,
                self.init(np.asarray(starts, dtype=np.int64)),
                self.init(np.asarray(ends, dtype=np.int64)),
                self.init(np.asarray(axes, dtype=np.int64)),
            ],
            prefix,
        )

    def reshape(self, x: str, shape: Sequence[int], prefix: str) -> str:
        return self.node("Reshape", [x, self.init(np.asarray(shape, dtype=np.int64))], prefix)

    def conv_mask(self, x: str, kernel: str, prefix: str) -> str:
        return self.node("Conv", [x, kernel], prefix, pads=[1, 1, 1, 1], group=FG)

    def conv_transpose_mask(self, x: str, kernel: str, prefix: str) -> str:
        return self.node("ConvTranspose", [x, kernel], prefix, pads=[1, 1, 1, 1], group=FG)

    def shift(self, x: str, dy: int, dx: int, channels: int, prefix: str) -> str:
        rs, re = max(0, dy), H + min(0, dy)
        cs, ce = max(0, dx), W + min(0, dx)
        pt, pb = max(0, -dy), max(0, dy)
        pl, pr = max(0, -dx), max(0, dx)
        cut = self.slice(x, [0, 0, rs, cs], [1, channels, re, ce], [0, 1, 2, 3], f"{prefix}s")
        return self.node("Pad", [cut], prefix, pads=[0, 0, pt, pl, 0, 0, pb, pr])


def build_onnx_model() -> onnx.ModelProto:
    b = Builder()
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    zero = b.init(np.asarray([0.0], dtype=np.float16), "zero")
    input_h = b.node("Cast", [IN_NAME], "inputh", to=TensorProto.FLOAT16)
    bg_in = b.slice(input_h, [0, 0, 0, 0], [1, 1, H, W], [0, 1, 2, 3], "bg")
    fg = b.slice(input_h, [0, 1, 0, 0], [1, C, H, W], [0, 1, 2, 3], "fg")
    fg_flat = b.reshape(fg, FLAT_SHAPE, "fgflat")
    fg_t = b.node("Transpose", [fg_flat], "fgtr", perm=[0, 2, 1])

    additions: List[str] = [fg]
    center_flats: List[str] = []
    for mask_name, offsets in MASKS.items():
        kernel_arr = np.zeros((FG, 1, 3, 3), dtype=np.float16)
        for channel in range(FG):
            for dy, dx in offsets:
                kernel_arr[channel, 0, dy + 1, dx + 1] = 1.0
        kernel = b.init(kernel_arr, f"{mask_name}_kernel")
        threshold = b.init(np.asarray([len(offsets) - 0.5], dtype=np.float16), f"{mask_name}_threshold")

        outer_count = b.conv_mask(fg, kernel, f"{mask_name}_count")
        outer_bool = b.node("Greater", [outer_count, threshold], f"{mask_name}_outerb")
        outer = b.node("Cast", [outer_bool], f"{mask_name}_outer", to=TensorProto.FLOAT16)
        outer_flat = b.reshape(outer, FLAT_SHAPE, f"{mask_name}_flat")
        pair_counts = b.node("MatMul", [outer_flat, fg_t], f"{mask_name}_pairs")
        pair_any = b.node("Greater", [pair_counts, zero], f"{mask_name}_any")
        pair_proto = b.node("Cast", [pair_any], f"{mask_name}_proto", to=TensorProto.FLOAT16)

        pair_proto_t = b.node("Transpose", [pair_proto], f"{mask_name}_ptr", perm=[0, 2, 1])
        add_center_flat = b.node("MatMul", [pair_proto_t, outer_flat], f"{mask_name}_centerflat")
        center_flats.append(add_center_flat)

        outer_from_centers_flat = b.node("MatMul", [pair_proto, fg_flat], f"{mask_name}_outerflat")
        outer_from_centers = b.reshape(outer_from_centers_flat, FG_SHAPE, f"{mask_name}_outercenter")
        additions.append(b.conv_transpose_mask(outer_from_centers, kernel, f"{mask_name}_stamp"))

    center_sum_flat = b.node("Sum", center_flats, "centersumflat")
    additions.append(b.reshape(center_sum_flat, FG_SHAPE, "centersum"))
    fg_sum = b.node("Sum", additions, "fgsum")
    fg_out = fg_sum

    active_sum = b.node("ReduceSum", [fg_out], "activesum", axes=[1], keepdims=1)
    bg_out = b.node("Sub", [bg_in, active_sum], "bgout")

    small_out_h = b.name("smallouth")
    b.nodes.append(helper.make_node("Concat", [bg_out, fg_out], [small_out_h], axis=1))
    padded_h = b.name("paddedh")
    b.nodes.append(
        helper.make_node(
            "Pad",
            [small_out_h],
            [padded_h],
            pads=[0, 0, 0, 0, 0, 0, IO_H - H, IO_W - W],
        )
    )
    b.nodes.append(helper.make_node("Cast", [padded_h], [OUT_NAME], to=TensorProto.FLOAT))

    graph = helper.make_graph(b.nodes, "task173_motif_completion", [x_info], [y_info], initializer=b.inits)
    model = helper.make_model(
        graph,
        producer_name="task173",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _grid_to_onehot(grid: np.ndarray) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            out[0, int(color), row, col] = 1.0
    return out


def _verify_onnx(path: Path) -> Tuple[int, int]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ok = 0
    examples = _load_examples()
    for _, _, inp, expected_grid in examples:
        pred = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(inp)})[0]
        expected = _grid_to_onehot(expected_grid)
        ok += int(np.array_equal(pred > 0.0, expected > 0.0))
    return ok, len(examples)


def _realized_tensor_count(model: onnx.ModelProto) -> int:
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    init_names = {init.name for init in graph.initializer}
    io_names = {IN_NAME, OUT_NAME}
    names = set()
    for node in graph.node:
        names.update(name for name in node.output if name)
    return len(names - init_names - io_names)


def main() -> None:
    scores = evaluate_candidates()
    for name, train_ok, all_ok in scores:
        print(f"{name}: train {train_ok}/3, all visible {all_ok}/{len(_load_examples())}")

    model = build_onnx_model()
    onnx.save(model, BEST_PATH)

    ok, total = _verify_onnx(BEST_PATH)
    result = score_file(BEST_PATH)
    print("chosen hypothesis: motif completion")
    print(f"train score: {next(score for name, score, _ in scores if name == 'motif completion')}/3")
    print(f"visible verification: {ok}/{total}")
    print(f"node count: {len(model.graph.node)}")
    print(f"realized tensor count: {_realized_tensor_count(model)}")
    if result["valid"]:
        print(
            "score_model: "
            f"memory={result['memory']} params={result['params']} "
            f"cost={result['cost']} score={result['score']:.6f}"
        )
    else:
        print(f"score_model: INVALID {result['error']}")
    print("estimated hidden-test robustness: high for same-color 3x3 motif completion with the four observed masks")


if __name__ == "__main__":
    main()
