"""ONNX for NeuroGolf task374: recolor gray line segments by length rank.

Task rule: each 10x10 input contains exactly three disconnected gray (5)
straight line segments on black background. Segments are axis-aligned and have
distinct lengths. The output preserves every segment's position and geometry,
recoloring the longest segment blue (1), the middle-length segment yellow (4),
and the shortest segment red (2); black cells remain black.

ONNX: crop the gray plane, detect exact horizontal and vertical runs of each
length 2..9 with float16 convolutions, spread run anchors back to cell masks
with float16 MaxPool, gather the shortest/longest masks by scalar length rank,
and build the compact 10x10 one-hot output before padding to the required
30x30 tensor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_NUM = "374"
TASK_ID = f"task{TASK_NUM}"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
GRID = 10
GRAY = 5
IR_VERSION = 10
OPSET = 10
LENGTHS = tuple(range(2, 10))


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self._counter = 0
        self._i64_cache: dict[tuple[int, ...], str] = {}
        self._f16_cache: dict[tuple[float, ...], str] = {}

    def name(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def init(self, name: str, values: Any, dtype: Any) -> str:
        self.initializers.append(numpy_helper.from_array(np.asarray(values, dtype=dtype), name))
        return name

    def i64(self, values: Iterable[int]) -> str:
        key = tuple(int(v) for v in values)
        if key not in self._i64_cache:
            self._i64_cache[key] = self.init(self.name("i"), key, np.int64)
        return self._i64_cache[key]

    def f16(self, values: Iterable[float]) -> str:
        key = tuple(float(v) for v in values)
        if key not in self._f16_cache:
            self._f16_cache[key] = self.init(self.name("h"), key, np.float16)
        return self._f16_cache[key]

    def node(self, op_type: str, inputs: Sequence[str], prefix: str, **attrs: Any) -> str:
        output = self.name(prefix)
        self.nodes.append(helper.make_node(op_type, list(inputs), [output], **attrs))
        return output


def make_model(nodes: list[onnx.NodeProto], initializers: list[onnx.TensorProto]) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        initializer=initializers,
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


def kernel(length: int, vertical: bool) -> np.ndarray:
    shape = (1, 1, length, 1) if vertical else (1, 1, 1, length)
    return np.ones(shape, dtype=np.float32)


def detect_runs(
    b: Builder,
    gray_f16: str,
    length: int,
    vertical: bool,
) -> tuple[str, str]:
    tag = f"{'v' if vertical else 'h'}{length}"
    run_kernel = b.init(f"k_{tag}", kernel(length, vertical), np.float16)

    inner = b.node("Conv", [gray_f16, run_kernel], f"{tag}_inner")
    anchor_b = b.node("Greater", [inner, b.f16([float(length) - 0.5])], f"{tag}_anchor_b")
    anchor_f16 = b.node("Cast", [anchor_b], f"{tag}_anchor_f16", to=TensorProto.FLOAT16)
    if vertical:
        spread = b.node(
            "MaxPool",
            [anchor_f16],
            f"{tag}_spread",
            kernel_shape=[length, 1],
            pads=[length - 1, 0, length - 1, 0],
            strides=[1, 1],
        )
    else:
        spread = b.node(
            "MaxPool",
            [anchor_f16],
            f"{tag}_spread",
            kernel_shape=[1, length],
            pads=[0, length - 1, 0, length - 1],
            strides=[1, 1],
        )
    return b.node("Cast", [spread], f"{tag}_mask", to=TensorProto.BOOL), anchor_f16


def build_model() -> onnx.ModelProto:
    b = Builder()
    gray = b.node(
        "Slice",
        [IN_NAME, b.i64([0, GRAY, 0, 0]), b.i64([1, GRAY + 1, GRID, GRID])],
        "gray",
    )
    gray_b = b.node("Cast", [gray], "gray_b", to=TensorProto.BOOL)
    gray_f16 = b.node("Cast", [gray], "gray_f16", to=TensorProto.FLOAT16)
    bg = b.node("Not", [gray_b], "bg")
    zero_b = b.node("And", [gray_b, bg], "zero_b")

    h_any: dict[int, str] = {}
    v_any: dict[int, str] = {}
    run_counts: dict[int, str] = {}
    for length in LENGTHS:
        h_any[length], hanchor = detect_runs(b, gray_f16, length, vertical=False)
        v_any[length], vanchor = detect_runs(b, gray_f16, length, vertical=True)
        hcount = b.node("ReduceSum", [hanchor], f"len{length}_hcount", axes=[0, 1, 2, 3], keepdims=1)
        vcount = b.node("ReduceSum", [vanchor], f"len{length}_vcount", axes=[0, 1, 2, 3], keepdims=1)
        run_counts[length] = b.node("Add", [hcount, vcount], f"len{length}_runs")

    length_masks: dict[int, str] = {}
    present: dict[int, str] = {}
    half_f16 = b.f16([0.5])
    for length in LENGTHS:
        if length == LENGTHS[-1]:
            hmask = h_any[length]
            vmask = v_any[length]
        else:
            hmask = b.node("Xor", [h_any[length], h_any[length + 1]], f"h{length}_exact")
            vmask = b.node("Xor", [v_any[length], v_any[length + 1]], f"v{length}_exact")
        mask = b.node("Or", [hmask, vmask], f"len{length}_mask")
        length_masks[length] = mask
        if length == LENGTHS[-1]:
            exact_runs = run_counts[length]
        elif length == LENGTHS[-2]:
            twice_next = b.node("Add", [run_counts[length + 1], run_counts[length + 1]], f"len{length}_twice_next")
            exact_runs = b.node("Sub", [run_counts[length], twice_next], f"len{length}_exact_runs")
        else:
            outer = b.node("Add", [run_counts[length], run_counts[length + 2]], f"len{length}_outer_runs")
            twice_next = b.node("Add", [run_counts[length + 1], run_counts[length + 1]], f"len{length}_twice_next")
            exact_runs = b.node("Sub", [outer, twice_next], f"len{length}_exact_runs")
        present[length] = b.node("Greater", [exact_runs, half_f16], f"len{length}_present")

    masks_stack = b.node("Concat", [length_masks[length] for length in LENGTHS], "masks_stack", axis=0)
    present_1d = []
    for length in LENGTHS:
        present_u8 = b.node("Cast", [present[length]], f"len{length}_present_u8", to=TensorProto.UINT8)
        present_1d.append(b.node("Reshape", [present_u8, b.i64([1])], f"len{length}_present_1d"))
    present_vec = b.node("Concat", present_1d, "present_vec", axis=0)
    present_rev = b.node("Concat", list(reversed(present_1d)), "present_rev", axis=0)
    red_idx = b.node("ArgMax", [present_vec], "red_idx", axis=0, keepdims=1)
    blue_rev_idx = b.node("ArgMax", [present_rev], "blue_rev_idx", axis=0, keepdims=1)
    blue_idx = b.node("Sub", [b.i64([len(LENGTHS) - 1]), blue_rev_idx], "blue_idx")
    blue = b.node("Gather", [masks_stack, blue_idx], "blue", axis=0)
    red = b.node("Gather", [masks_stack, red_idx], "red", axis=0)
    blue_or_red = b.node("Or", [blue, red], "blue_or_red")
    yellow = b.node("Xor", [gray_b, blue_or_red], "yellow")
    out_b = b.node(
        "Concat",
        [bg, blue, red, zero_b, yellow],
        "out_b",
        axis=1,
    )
    out10 = b.node("Cast", [out_b], "out10", to=TensorProto.FLOAT)
    b.nodes.append(
        helper.make_node(
            "Pad",
            [out10],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 5, 20, 20],
            value=0.0,
        )
    )
    return make_model(b.nodes, b.initializers)


def solve(grid: Sequence[Sequence[int]]) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(g)
    seen = np.zeros(g.shape, dtype=np.bool_)
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    h, w = g.shape
    for row in range(h):
        for col in range(w):
            if g[row, col] == 0 or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != 0 and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            comps.append((max(max(rows) - min(rows) + 1, max(cols) - min(cols) + 1), cells))

    ranked = sorted(range(len(comps)), key=lambda idx: -comps[idx][0])
    colors = {ranked[0]: 1, ranked[1]: 4, ranked[2]: 2}
    for idx, (_, cells) in enumerate(comps):
        for r, c in cells:
            out[r, c] = colors[idx]
    return out


def _onehot(grid: list[list[int]]) -> np.ndarray:
    arr = convert_to_numpy({"input": grid}, "input")
    assert arr is not None
    return arr


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> tuple[int, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    bad = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            expected = _onehot(ex["output"])
            total += 1
            pred = _run_onnx(model, _onehot(ex["input"]))
            if not np.array_equal(pred > 0.0, expected > 0.0):
                bad += 1
                got = (pred[0, :, :GRID, :GRID] > 0.0).argmax(axis=0)
                want = np.asarray(ex["output"], dtype=np.int64)
                print(f"mismatch {split} #{idx}")
                print(got)
                print(want)
                break
    return bad, total


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    bad, total = validate_json(model)
    assert bad == 0, f"{bad} mismatches across {total} examples"
    print(f"verified {total} {TASK_ID} examples")

    result = score_file(BEST_PATH)
    print(result)


if __name__ == "__main__":
    main()
