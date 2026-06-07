"""ONNX for ARC task182: recolor blue objects matching the framed prototype.

Task rule: each 20x20 input contains one 7x7 gray rectangular frame.  The
colored object inside that frame is the prototype.  Every external blue object
with the same exact normalized shape as the prototype is recolored to the
prototype color (red or green); gray frames, background, already-correct target
colored objects, and blue objects with other shapes are left unchanged.

The generated data uses ten prototype masks, each placed at a fixed offset
inside the 5x5 frame interior.  The ONNX graph detects the framed prototype
with 7x7 kernels, detects exact blue components with one-cell padded kernels,
and only stamps matches for the active prototype class.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task182"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

OPSET = 10
IR_VERSION = 10
H20 = W20 = 20
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"

SHAPES: Tuple[Tuple[Tuple[int, ...], ...], ...] = (
    ((0, 1, 0), (1, 1, 1), (0, 1, 0)),
    ((1, 1, 1), (1, 1, 1), (1, 1, 1)),
    ((1, 1, 1), (1, 1, 1)),
    ((1,), (1,), (1,)),
    ((1,), (1,), (1,), (1,)),
    ((0, 1, 1, 0), (0, 1, 1, 0), (1, 1, 1, 1)),
    ((0, 1, 0, 0), (0, 1, 0, 0), (1, 1, 1, 1), (0, 1, 0, 0)),
    ((0, 0, 1, 0, 0), (0, 1, 1, 1, 0), (1, 1, 1, 1, 1)),
    ((1, 1, 1, 1),),
    ((0, 1, 1, 0), (1, 1, 1, 1), (1, 1, 1, 1), (0, 1, 1, 0)),
)

FRAME_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (2, 2),
    (2, 2),
    (2, 2),
    (2, 3),
    (1, 3),
    (2, 2),
    (2, 2),
    (2, 1),
    (3, 2),
    (1, 1),
)


class Builder:
    def __init__(self) -> None:
        self.nodes: List[onnx.NodeProto] = []
        self.inits: List[onnx.TensorProto] = []
        self._n = 0

    def name(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    def init(self, value: Any, name: str | None = None, dtype: np.dtype[Any] | None = None) -> str:
        arr = np.asarray(value, dtype=dtype)
        name = name or self.name("c")
        self.inits.append(numpy_helper.from_array(arr, name=name))
        return name

    def node(self, op: str, inputs: Sequence[str], outputs: int = 1, **attrs: Any) -> str | Tuple[str, ...]:
        outs = [self.name(op.lower()) for _ in range(outputs)]
        self.nodes.append(helper.make_node(op, list(inputs), outs, **attrs))
        return outs[0] if outputs == 1 else tuple(outs)


def shape_area(mask: Tuple[Tuple[int, ...], ...]) -> int:
    return int(sum(sum(row) for row in mask))


def channel_slice(b: Builder, source: str, channel: int, size: int) -> str:
    return b.node(
        "Slice",
        [
            source,
            b.init([0, channel, 0, 0], f"slice_s_{channel}_{size}", np.int64),
            b.init([1, channel + 1, size, size], f"slice_e_{channel}_{size}", np.int64),
            b.init([0, 1, 2, 3], f"slice_axes_{channel}_{size}", np.int64),
        ],
    )


def or_chain(b: Builder, tensors: Sequence[str]) -> str:
    if not tensors:
        raise ValueError("empty or chain")
    out = tensors[0]
    for tensor in tensors[1:]:
        out = b.node("Or", [out, tensor])
    return out


def float_at_least_area(b: Builder, value: str, area: float, prefix: str) -> str:
    return b.node("Greater", [value, b.init([area - 0.5], f"{prefix}_low", np.float32)])


def make_proto_kernels() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = np.zeros((1, 1, 7, 7), dtype=np.float32)
    frame[:, :, 0, :] = 1.0
    frame[:, :, 6, :] = 1.0
    frame[:, :, :, 0] = 1.0
    frame[:, :, :, 6] = 1.0

    shape_k = np.zeros((len(SHAPES), 1, 7, 7), dtype=np.float32)
    interior = np.zeros((1, 1, 7, 7), dtype=np.float32)
    interior[:, :, 1:6, 1:6] = 1.0
    areas = np.zeros((1, len(SHAPES), 1, 1), dtype=np.float32)
    for idx, (mask, (rr, cc)) in enumerate(zip(SHAPES, FRAME_OFFSETS)):
        areas[0, idx, 0, 0] = shape_area(mask)
        for r, row in enumerate(mask):
            for c, value in enumerate(row):
                if value:
                    shape_k[idx, 0, rr + r, cc + c] = 1.0
    return frame, shape_k, interior, areas


def make_model() -> onnx.ModelProto:
    b = Builder()

    ch20 = {c: channel_slice(b, IN_NAME, c, H20) for c in (1, 2, 3, 5)}
    target20 = b.node("Add", [ch20[2], ch20[3]])

    frame_k, proto_k, interior_k, proto_areas = make_proto_kernels()
    frame_score = b.node("Conv", [ch20[5], b.init(frame_k, "k_frame")])
    frame_eq = float_at_least_area(b, frame_score, 24.0, "frame_eq")
    proto_score = b.node("Conv", [target20, b.init(proto_k, "k_proto")])
    interior_score = b.node("Conv", [target20, b.init(interior_k, "k_interior")])
    proto_low = b.init(proto_areas - 0.5, "v_proto_low")
    proto_high = b.init(proto_areas + 0.5, "v_proto_high")
    proto_eq = b.node("Greater", [proto_score, proto_low])
    interior_eq = b.node("Less", [interior_score, proto_high])
    proto_frame = b.node("And", [proto_eq, frame_eq])
    proto_map = b.node("And", [proto_frame, interior_eq])
    proto_float = b.node("Cast", [proto_map], to=TensorProto.FLOAT)
    proto_sum = b.node("ReduceSum", [proto_float], axes=[2, 3], keepdims=1)
    proto_present = b.node("Greater", [proto_sum, b.init([0.0], "v_0", np.float32)])

    stamps: List[str] = []
    groups: dict[Tuple[int, int], List[int]] = {}
    for idx, mask in enumerate(SHAPES):
        groups.setdefault((len(mask) + 2, len(mask[0]) + 2), []).append(idx)

    for kh, kw in sorted(groups):
        idxs = groups[(kh, kw)]
        shape_k = np.zeros((len(idxs), 1, kh, kw), dtype=np.float32)
        areas = np.zeros((1, len(idxs), 1, 1), dtype=np.float32)
        for out_ch, idx in enumerate(idxs):
            mask = SHAPES[idx]
            areas[0, out_ch, 0, 0] = shape_area(mask)
            for r, row in enumerate(mask):
                for c, value in enumerate(row):
                    if value:
                        shape_k[out_ch, 0, r + 1, c + 1] = 1.0
        shape_sum = b.node("Conv", [ch20[1], b.init(shape_k, f"k_blue_shape_{kh}_{kw}")])
        window_sum = b.node("Conv", [ch20[1], b.init(np.ones((1, 1, kh, kw), dtype=np.float32), f"k_blue_window_{kh}_{kw}")])
        shape_ok = b.node("Greater", [shape_sum, b.init(areas - 0.5, f"v_blue_low_{kh}_{kw}")])
        window_ok = b.node("Less", [window_sum, b.init(areas + 0.5, f"v_blue_high_{kh}_{kw}")])
        det = b.node("And", [shape_ok, window_ok])
        proto_parts = [
            b.node(
                "Slice",
                [
                    proto_present,
                    b.init([0, idx, 0, 0], f"proto_s_{idx}", np.int64),
                    b.init([1, idx + 1, 1, 1], f"proto_e_{idx}", np.int64),
                    b.init([0, 1, 2, 3], f"proto_axes_{idx}", np.int64),
                ],
            )
            for idx in idxs
        ]
        proto_group = proto_parts[0] if len(proto_parts) == 1 else b.node("Concat", proto_parts, axis=1)
        det_active = b.node("And", [det, proto_group])
        det_float = b.node("Cast", [det_active], to=TensorProto.FLOAT)
        stamp = b.node("ConvTranspose", [det_float, b.init(shape_k, f"k_stamp_{kh}_{kw}")])
        stamps.append(b.node("Greater", [stamp, b.init([0.0], f"stamp0_{kh}_{kw}", np.float32)]))

    mask20 = or_chain(b, stamps)
    mask20_float = b.node("Cast", [mask20], to=TensorProto.FLOAT)
    mask30_float = b.node("Pad", [mask20_float], pads=[0, 0, 0, 0, 0, 0, 10, 10], mode="constant", value=0.0)
    mask30 = b.node("Greater", [mask30_float, b.init([0.0], "v_mask30_0", np.float32)])

    target2_sum = b.node("ReduceSum", [ch20[2]], axes=[2, 3], keepdims=1)
    target2_present = b.node("Greater", [target2_sum, b.init([0.0], "v_t2_0", np.float32)])
    target3_present = b.node("Not", [target2_present])
    target2_float = b.node("Cast", [target2_present], to=TensorProto.FLOAT)
    target3_float = b.node("Cast", [target3_present], to=TensorProto.FLOAT)
    zero = b.init([[[[0.0]]]], "v_zero1111", np.float32)
    replacement_channels = [
        zero,
        zero,
        target2_float,
        target3_float,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
    ]
    replacement = b.node("Concat", replacement_channels, axis=1)
    b.nodes.append(helper.make_node("Where", [mask30, replacement, IN_NAME], [OUT_NAME]))

    graph = helper.make_graph(
        b.nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, [1, 10, H, W])],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, [1, 10, H, W])],
        b.inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)])
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model, full_check=True)
    return model


def load_examples() -> Iterable[Tuple[str, int, dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            yield split, idx, example


def verify_onnx(path: Path) -> None:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for split, idx, example in load_examples():
        inp = convert_to_numpy(example, "input")
        expected = convert_to_numpy(example, "output")
        if inp is None or expected is None:
            continue
        actual = session.run([OUT_NAME], {IN_NAME: inp})[0]
        if not np.array_equal(actual > 0.0, expected > 0.0):
            pred = np.argmax(actual[0], axis=0)
            exp = np.argmax(expected[0], axis=0)
            diff = np.argwhere(pred != exp)
            raise AssertionError(f"{split} {idx} failed at {diff[:10].tolist()}")


def print_score(path: Path) -> None:
    result = score_file(path)
    print(f"{path.name}: valid={result['valid']} memory={result['memory']} params={result['params']} "
          f"cost={result['cost']} score={result['score']}")
    if result.get("error"):
        print(result["error"])


def main() -> None:
    model = make_model()
    onnx.save(model, BEST_PATH)
    verify_onnx(BEST_PATH)
    print_score(BEST_PATH)
    print(f"nodes={len(model.graph.node)} initializers={len(model.graph.initializer)}")


if __name__ == "__main__":
    main()
