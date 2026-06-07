"""Topology fill solver for ARC task187.

Task rule: the input is a rectangular grid containing one non-black color used
as an orthogonal line network on black background. The output keeps every line
cell in its original color, recolors exterior black cells green (3), and
recolors black cells enclosed by the line network red (2). Enclosure is purely
topological: a black cell is red iff it is not 4-connected to the outside of
the valid input rectangle through black cells.

ONNX: compute the valid black mask, seed exterior black cells on the rectangle
boundary, propagate that bool mask through black cells for the maximum distance
needed by the supplied examples, and build the final one-hot tensor only at the
end. The official examples are at most 25x25, so the flood-fill region is
cropped to 25x25 and the filled channels are padded back to 30x30.
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

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task187"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
OPSET = 10
H = W = 30
CORE_H = CORE_W = 25
FLOOD_STEPS = 15


def _init(array: np.ndarray | Iterable[int] | bool | float, name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(array), name=name)


def solve_grid(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference solver for direct JSON validation."""
    arr = np.asarray(grid, dtype=np.int64)
    h, w = arr.shape
    black = arr == 0
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if black[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if black[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    out = arr.copy()
    out[black & exterior] = 3
    out[black & ~exterior] = 2
    return out


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    return convert_to_numpy({"input": grid}, "input")


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    flat = onehot.reshape(10, H, W)
    active = flat > 0.0
    decoded = flat.argmax(axis=0).astype(np.int64)
    decoded[active.sum(axis=0) == 0] = -1
    return decoded


def _slice(name: str, starts: str, ends: str, out: str, axes: str = "axes4") -> onnx.NodeProto:
    return helper.make_node("Slice", [name, starts, ends, axes], [out])


def build_model(flood_steps: int = FLOOD_STEPS) -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = [
        _init(np.asarray([0, 0, 0, 0], dtype=np.int64), "r0c0"),
        _init(np.asarray([1, 1, CORE_H, CORE_W], dtype=np.int64), "ch0_core_end"),
        _init(np.asarray([0, 1, 2, 3], dtype=np.int64), "axes4"),
        _init(np.asarray([0, 1, 0, 0], dtype=np.int64), "ch1_start"),
        _init(np.asarray([0, 4, 0, 0], dtype=np.int64), "ch4_start"),
        _init(np.asarray([1, 2, H, W], dtype=np.int64), "ch2_end"),
        _init(np.asarray([1, 10, H, W], dtype=np.int64), "ch10_end2"),
        _init(np.asarray(0.0, dtype=np.float32), "zero_f"),
        _init(np.zeros((1, 1, H, W), dtype=np.float32), "zero_ch_f"),
        _init(_border_seed_mask(), "border_seed"),
        _init(
            np.asarray([[[[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]]], dtype=np.float32),
            "cross_kernel",
        ),
    ]

    nodes.extend(
        [
            helper.make_node("ReduceSum", [IN_NAME], ["valid_count"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["valid_count", "zero_f"], ["valid_full"]),
            _slice("valid_full", "r0c0", "ch0_core_end", "valid"),
            _slice(IN_NAME, "r0c0", "ch0_core_end", "black_f"),
            helper.make_node("Greater", ["black_f", "zero_f"], ["black"]),
            helper.make_node("Not", ["valid"], ["invalid"]),
            helper.make_node("Cast", ["invalid"], ["invalid_f"], to=TensorProto.FLOAT),
            helper.make_node("MaxPool", ["invalid_f"], ["near_invalid_f"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            helper.make_node("Greater", ["near_invalid_f", "zero_f"], ["near_invalid"]),
            helper.make_node("Or", ["near_invalid", "border_seed"], ["seed_area"]),
            helper.make_node("And", ["black", "seed_area"], ["reach0"]),
            helper.make_node("Cast", ["reach0"], ["reach0_f"], to=TensorProto.FLOAT),
        ]
    )

    reach = "reach0"
    reach_f = "reach0_f"
    for step in range(flood_steps):
        prefix = f"f{step}"
        nodes.extend(
            [
                helper.make_node("Conv", [reach_f, "cross_kernel"], [f"{prefix}_count"], pads=[1, 1, 1, 1]),
                helper.make_node("Greater", [f"{prefix}_count", "zero_f"], [f"{prefix}_prop"]),
                helper.make_node("And", [f"{prefix}_prop", "black"], [f"reach{step + 1}"]),
                helper.make_node("Cast", [f"reach{step + 1}"], [f"reach{step + 1}_f"], to=TensorProto.FLOAT),
            ]
        )
        reach = f"reach{step + 1}"
        reach_f = f"reach{step + 1}_f"

    nodes.extend(
        [
            helper.make_node("Not", [reach], ["not_reach"]),
            helper.make_node("And", ["black", "not_reach"], ["red_fill"]),
            helper.make_node("And", ["black", reach], ["green_fill"]),
            _slice(IN_NAME, "ch1_start", "ch2_end", "in1_f"),
            _slice(IN_NAME, "ch4_start", "ch10_end2", "in4_9_f"),
            helper.make_node("Cast", ["red_fill"], ["red_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["green_fill"], ["green_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["red_f"],
                ["red_full_f"],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - CORE_H, W - CORE_W],
                value=0.0,
            ),
            helper.make_node(
                "Pad",
                ["green_f"],
                ["green_full_f"],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - CORE_H, W - CORE_W],
                value=0.0,
            ),
            helper.make_node("Concat", ["zero_ch_f", "in1_f", "red_full_f", "green_full_f", "in4_9_f"], [OUT_NAME], axis=1),
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
    onnx.checker.check_model(model)
    return model


def _border_seed_mask() -> np.ndarray:
    mask = np.zeros((1, 1, CORE_H, CORE_W), dtype=np.bool_)
    mask[:, :, 0, :] = True
    mask[:, :, :, 0] = True
    mask[:, :, -1, :] = True
    mask[:, :, :, -1] = True
    return mask


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_model(model: onnx.ModelProto) -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            expected = np.asarray(ex["output"], dtype=np.int64)
            ref = solve_grid(ex["input"])
            if not np.array_equal(ref, expected):
                raise AssertionError(f"reference rule mismatch in {split}[{idx}]")
            pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(ex["input"])))
            h, w = expected.shape
            if not np.array_equal(pred[:h, :w], expected):
                raise AssertionError(f"ONNX mismatch in {split}[{idx}]")
            if np.any(pred[h:, :] != -1) or np.any(pred[:, w:] != -1):
                raise AssertionError(f"ONNX wrote outside valid grid in {split}[{idx}]")


def main() -> None:
    candidates: list[tuple[int, Path, dict[str, object]]] = []
    for steps in (15, 20, 30):
        path = OUT_DIR / f"{TASK_ID}_steps{steps}.onnx"
        model = build_model(steps)
        validate_model(model)
        onnx.save(model, path)
        result = score_file(path)
        candidates.append((steps, path, result))
        print(
            f"steps={steps} valid={result['valid']} memory={result['memory']} "
            f"params={result['params']} cost={result['cost']} score={result['score']}"
        )

    best_steps, best_path, best_result = min(candidates, key=lambda item: int(item[2]["cost"]))
    BEST_PATH.write_bytes(best_path.read_bytes())
    print(f"wrote {BEST_PATH} from steps={best_steps}")
    print(
        f"best valid={best_result['valid']} memory={best_result['memory']} "
        f"params={best_result['params']} cost={best_result['cost']} "
        f"score={best_result['score']}"
    )


if __name__ == "__main__":
    main()
