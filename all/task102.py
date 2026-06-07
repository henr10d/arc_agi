"""Fill square black holes enclosed by gray objects with red.

Task rule: inputs are 12x12 grids containing black background and gray
objects.  Black connected components that are completely enclosed by gray are
holes.  If an enclosed hole's bounding box is solid black and has equal height
and width, recolor exactly that hole red; rectangular, non-solid, or exterior
black regions stay unchanged.

ONNX approach: because the task inputs use only black and gray, slice only the
gray 12x12 crop and derive black as Not(gray). Detect every solid k by k black
interior for k=1..4 whose one-cell bounding border is all gray, expand
detections back over their interior cells, and rebuild the one-hot output on the
compact 12x12 crop before the final pad to the NeuroGolf 30x30 tensor.
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task102"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task102.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

IN_NAME = "input"
OUT_NAME = "output"
C = 10
H = W = 30
G = 12
MAX_SQUARE = 4
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _border_kernel(size: int) -> np.ndarray:
    kernel = np.ones((size + 2, size + 2), dtype=np.float32)
    kernel[1:-1, 1:-1] = 0.0
    return kernel


def _scale_positive_weights_to_sum(weights: np.ndarray, target_sum: int) -> np.ndarray:
    scaled = (weights > 0).astype(np.uint8)
    extra = target_sum - int(scaled.sum())
    if extra < 0:
        raise ValueError("target sum is smaller than the positive weight count")
    flat = scaled.reshape(-1)
    positive = np.flatnonzero(flat)
    for idx in np.resize(positive, extra):
        flat[idx] += 1
    return scaled


def solve_border_scan(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    """Candidate C: local all-black square plus all-gray border detection."""
    x = np.asarray(grid, dtype=np.int64)
    out = x.copy()
    paint = np.zeros_like(x, dtype=bool)
    for size in range(1, MAX_SQUARE + 1):
        border = _border_kernel(size).astype(bool)
        for r in range(1, G - size):
            for c in range(1, G - size):
                if not np.all(x[r : r + size, c : c + size] == 0):
                    continue
                patch = x[r - 1 : r + size + 1, c - 1 : c + size + 1]
                if np.all(patch[border] == 5):
                    paint[r : r + size, c : c + size] = True
    out[paint] = 2
    return out


def solve_flood_fill(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    """Candidate A: border flood-fill exterior black, then test hole bboxes."""
    x = np.asarray(grid, dtype=np.int64)
    rows, cols = x.shape
    exterior = np.zeros((rows, cols), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def seed(r: int, c: int) -> None:
        if x[r, c] == 0 and not exterior[r, c]:
            exterior[r, c] = True
            queue.append((r, c))

    for r in range(rows):
        seed(r, 0)
        seed(r, cols - 1)
    for c in range(cols):
        seed(0, c)
        seed(rows - 1, c)

    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                seed(rr, cc)

    return _fill_square_components(x, (x == 0) & ~exterior)


def solve_background_components(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    """Candidate B: connected components over background, excluding border CCs."""
    x = np.asarray(grid, dtype=np.int64)
    rows, cols = x.shape
    seen = np.zeros((rows, cols), dtype=bool)
    enclosed = np.zeros((rows, cols), dtype=bool)

    for start_r in range(rows):
        for start_c in range(cols):
            if x[start_r, start_c] != 0 or seen[start_r, start_c]:
                continue
            queue: deque[tuple[int, int]] = deque([(start_r, start_c)])
            seen[start_r, start_c] = True
            cells: list[tuple[int, int]] = []
            touches_border = False
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                touches_border |= r in {0, rows - 1} or c in {0, cols - 1}
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols and x[rr, cc] == 0 and not seen[rr, cc]:
                        seen[rr, cc] = True
                        queue.append((rr, cc))
            if not touches_border:
                for r, c in cells:
                    enclosed[r, c] = True

    return _fill_square_components(x, enclosed)


def _fill_square_components(grid: np.ndarray, enclosed: np.ndarray) -> np.ndarray:
    out = grid.copy()
    rows, cols = grid.shape
    seen = np.zeros((rows, cols), dtype=bool)
    for start_r in range(rows):
        for start_c in range(cols):
            if not enclosed[start_r, start_c] or seen[start_r, start_c]:
                continue
            queue: deque[tuple[int, int]] = deque([(start_r, start_c)])
            seen[start_r, start_c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols and enclosed[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        queue.append((rr, cc))
            rr = [r for r, _ in cells]
            cc = [c for _, c in cells]
            r0, r1 = min(rr), max(rr)
            c0, c1 = min(cc), max(cc)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height == width and len(cells) == height * width:
                out[r0 : r1 + 1, c0 : c1 + 1] = 2
    return out


def _init(inits: list[onnx.TensorProto], arr: np.ndarray | Iterable[int] | Iterable[float], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Iterable[int], name: str) -> str:
    return _init(inits, np.asarray(list(vals), dtype=np.int64), name)


def _f32(inits: list[onnx.TensorProto], arr: np.ndarray | Iterable[float], name: str) -> str:
    return _init(inits, np.asarray(arr, dtype=np.float32), name)


def _u8(inits: list[onnx.TensorProto], arr: np.ndarray | Iterable[int], name: str) -> str:
    return _init(inits, np.asarray(arr, dtype=np.uint8), name)


def _grid_to_onehot(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    x = np.asarray(grid, dtype=np.int64)
    for r in range(x.shape[0]):
        for c in range(x.shape[1]):
            out[0, int(x[r, c]), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)[:G, :G]


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes_chw = _i64(inits, [1, 2, 3], "axes_chw")
    ch5_st = _i64(inits, [5, 0, 0], "ch5_st")
    ch5_en = _i64(inits, [6, G, G], "ch5_en")
    q_scale = _f32(inits, [1.0], "q_scale")
    q_zero = _u8(inits, [0], "q_zero")
    detector_y_scale = _f32(inits, [71.0], "detector_y_scale")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, ch5_st, ch5_en, axes_chw], ["gray_f"]),
            helper.make_node("Cast", ["gray_f"], ["gray_b"], to=TensorProto.BOOL),
            helper.make_node("Not", ["gray_b"], ["black_b"]),
            helper.make_node("Cast", ["black_b"], ["black_u"], to=TensorProto.UINT8),
            helper.make_node("Cast", ["gray_f"], ["gray_u"], to=TensorProto.UINT8),
            helper.make_node("Concat", ["black_u", "gray_u"], ["black_gray"], axis=1),
        ]
    )

    cover_parts: list[str] = []
    for size in range(1, MAX_SQUARE + 1):
        suffix = f"s{size}"
        cand_extent = G - size - 1
        detector = np.zeros((1, 2, size + 2, size + 2), dtype=np.float32)
        detector[0, 0, 1:-1, 1:-1] = 1.0
        detector[0, 1] = _border_kernel(size)
        detector_w = _u8(inits, _scale_positive_weights_to_sum(detector, 36), f"dw_{suffix}")
        # Each valid detector sums to 36, while any invalid window misses at
        # least one positive weight and sums to at most 35. With y_scale=71,
        # QLinearConv rounds valid windows to 1 and invalid windows to 0.

        nodes.extend(
            [
                helper.make_node(
                    "QLinearConv",
                    ["black_gray", q_scale, q_zero, detector_w, q_scale, q_zero, detector_y_scale, q_zero],
                    [f"score_{suffix}"],
                ),
                helper.make_node("Cast", [f"score_{suffix}"], [f"candf_{suffix}"], to=TensorProto.FLOAT),
            ]
        )
        if size == 1:
            cover_parts.append(f"candf_{suffix}")
        else:
            paint_w = _f32(inits, np.ones((1, 1, size, size), dtype=np.float32), f"pw_{suffix}")
            nodes.append(helper.make_node("ConvTranspose", [f"candf_{suffix}", paint_w], [f"cover_{suffix}"]))
            cover_parts.append(f"cover_{suffix}")
        assert cand_extent == G - size - 1

    nodes.extend(
        [
            helper.make_node("Sum", cover_parts, ["paint10_f"]),
            helper.make_node(
                "Pad",
                ["paint10_f"],
                ["paint12_f"],
                pads=[0, 0, 1, 1, 0, 0, 1, 1],
            ),
            helper.make_node("Cast", ["paint12_f"], ["paint"], to=TensorProto.BOOL),
            helper.make_node("Not", ["paint"], ["not_paint"]),
            helper.make_node("And", ["black_b", "not_paint"], ["c0"]),
            helper.make_node("And", ["gray_b", "black_b"], ["empty"]),
            helper.make_node("Concat", ["c0", "empty", "paint", "empty", "empty", "gray_b"], ["out12_b"], axis=1),
            helper.make_node("Cast", ["out12_b"], ["out12"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out12"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, C - 6, H - G, W - G],
            ),
        ]
    )

    graph = helper.make_graph(nodes, "task102", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_reference_candidates(data: dict[str, list[dict[str, list[list[int]]]]]) -> None:
    candidates = {
        "border_scan": solve_border_scan,
        "flood_fill": solve_flood_fill,
        "background_components": solve_background_components,
    }
    for name, fn in candidates.items():
        for split, examples in data.items():
            for idx, example in enumerate(examples):
                got = fn(example["input"])
                expected = np.asarray(example["output"], dtype=np.int64)
                if not np.array_equal(got, expected):
                    raise AssertionError(f"{name} failed {split}[{idx}]")


def validate_onnx(model: onnx.ModelProto, data: dict[str, list[dict[str, list[list[int]]]]]) -> None:
    for split, examples in data.items():
        for idx, example in enumerate(examples):
            x = _grid_to_onehot(example["input"])
            got = _run_onnx(model, x)
            expected = _grid_to_onehot(example["output"])
            if not np.array_equal(got > 0.0, expected > 0.0):
                pred_grid = _onehot_to_grid(got)
                raise AssertionError(f"ONNX failed {split}[{idx}]\n{pred_grid}")


def main() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    validate_reference_candidates(data)
    model = build_model()
    validate_onnx(model, data)
    BEST_PATH.write_bytes(model.SerializeToString())

    result = score_file(BEST_PATH)
    if not result["valid"]:
        raise RuntimeError(result["error"])
    print(
        f"{BEST_PATH} memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']:.6f}"
    )


if __name__ == "__main__":
    main()
