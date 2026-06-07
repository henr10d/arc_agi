"""Build a compact ONNX model for NeuroGolf task043.

Task rule: in a 10x10 grid containing only background 0 and gray 5,
gray cells on the top border select columns and gray cells on the right
border select rows. The output preserves all gray markers and places red
2 at each selected row/column intersection, but only on background cells.

The graph keeps the computation as a single 10x10 bool gray mask, builds
only output channels 0..5 from that mask, then casts and pads to the
NeuroGolf float one-hot 30x30 layout at the final step. The final pad
creates the unused channels 6..9 without realizing them first.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402


TASK_NUM = "043"
TASK_NAME = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_NAME}.json"
OUT_PATH = ROOT / "all" / f"{TASK_NAME}.onnx"


def vi(name: str, dtype: int, shape: tuple[int, ...]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, list(shape))


def node(op_type: str, inputs: list[str], output: str, **attrs: object) -> onnx.NodeProto:
    return helper.make_node(op_type, inputs, [output], **attrs)


def build_model() -> onnx.ModelProto:
    value_infos = [
        vi("g5_f", TensorProto.FLOAT, (1, 1, 10, 10)),
        vi("g5", TensorProto.BOOL, (1, 1, 10, 10)),
        vi("top", TensorProto.BOOL, (1, 1, 1, 10)),
        vi("right", TensorProto.BOOL, (1, 1, 10, 1)),
        vi("red", TensorProto.BOOL, (1, 1, 10, 10)),
        vi("occupied", TensorProto.BOOL, (1, 1, 10, 10)),
        vi("out0", TensorProto.BOOL, (1, 1, 10, 10)),
        vi("false", TensorProto.BOOL, (1, 1, 10, 10)),
        vi("small_bool", TensorProto.BOOL, (1, 6, 10, 10)),
        vi("small_float", TensorProto.FLOAT, (1, 6, 10, 10)),
    ]

    nodes = [
        node("Slice", ["input"], "g5_f", axes=[0, 1, 2, 3], starts=[0, 5, 0, 0], ends=[1, 6, 10, 10]),
        node("Cast", ["g5_f"], "g5", to=TensorProto.BOOL),
        node("Slice", ["g5"], "top", axes=[2], starts=[0], ends=[1]),
        node("Slice", ["g5"], "right", axes=[3], starts=[9], ends=[10]),
        node("And", ["top", "right"], "red"),
        node("Or", ["red", "g5"], "occupied"),
        node("Not", ["occupied"], "out0"),
        node("And", ["red", "g5"], "false"),
        node(
            "Concat",
            ["out0", "false", "red", "false", "false", "g5"],
            "small_bool",
            axis=1,
        ),
        node("Cast", ["small_bool"], "small_float", to=TensorProto.FLOAT),
        node(
            "Pad",
            ["small_float"],
            "output",
            mode="constant",
            pads=[0, 0, 0, 0, 0, 4, 20, 20],
            value=0.0,
        ),
    ]

    graph = helper.make_graph(
        nodes,
        TASK_NAME,
        [vi("input", TensorProto.FLOAT, (1, 10, 30, 30))],
        [vi("output", TensorProto.FLOAT, (1, 10, 30, 30))],
        [],
        value_info=value_infos,
    )
    return helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 9)])


def grid_from_onehot(arr: np.ndarray) -> np.ndarray:
    return arr[0, :, :10, :10].argmax(axis=0).astype(np.int64)


def load_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def validate(model_path: Path, data: dict[str, list[dict[str, list[list[int]]]]]) -> bool:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    ok = True
    for split, examples in data.items():
        for idx, ex in enumerate(examples):
            inp = convert_to_numpy(ex, "input")
            expected = convert_to_numpy(ex, "output")
            if inp is None or expected is None:
                continue
            pred = sess.run(["output"], {"input": inp})[0]
            match = np.array_equal(pred > 0.0, expected > 0.0)
            ok = ok and match
            if split == "train":
                print(f"train[{idx}] match={match}")
                if not match:
                    print("predicted:")
                    print(grid_from_onehot(pred))
                    print("expected:")
                    print(np.array(ex["output"], dtype=np.int64))
            elif not match:
                print(f"{split}[{idx}] failed")
    return ok


def main() -> None:
    data = load_data()
    model = build_model()
    onnx.checker.check_model(model, full_check=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUT_PATH)
    valid = validate(OUT_PATH, data)
    with tempfile.TemporaryDirectory() as tmpdir:
        score_path = Path(tmpdir) / f"{TASK_NAME}.onnx"
        onnx.save(model, score_path)
        stats = score_file(score_path)
    print(f"valid_examples={valid}")
    print(
        "score "
        f"memory={stats['memory']} params={stats['params']} cost={stats['cost']} "
        f"points={stats['score'] if stats['score'] is not None else None}"
    )
    if stats["cost"] is not None:
        print(f"manual_points={max(1.0, 25.0 - math.log(max(1.0, stats['cost']))):.6f}")


if __name__ == "__main__":
    main()
