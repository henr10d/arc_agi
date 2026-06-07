"""Minimal ONNX for ARC task258: fill one-cell horizontal blue gaps red.

Task rule: each row may contain blue cells (color 1) spaced two columns apart,
such as B . B or B . B . B.  The output keeps the blue cells unchanged and
paints the background cell between every horizontal blue pair red (color 2).
Rows are handled independently; no vertical or cross-row connections are made.

ONNX: the best candidate is a single grouped 1x3 Conv that writes the final
10-channel one-hot tensor directly.  With group=2, output colors 0-4 only see
input colors 0-4, which is enough for this task and halves the dense kernel
parameter count versus an ungrouped Conv.  Its linear scores are chosen so
thresholding at zero preserves background/blue and activates red only for
background cells with blue neighbors on both horizontal sides, leaving padded
cells unactivated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task258"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

IN_NAME = "input"
OUT_NAME = "output"
C = 10
H = W = 30
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _vi(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def _init(inits: list[onnx.TensorProto], name: str, arr: np.ndarray) -> str:
    inits.append(numpy_helper.from_array(arr, name))
    return name


def _make_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        name,
        [_vi(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [_vi(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        inits,
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


def build_model() -> onnx.ModelProto:
    """One final group=2 Conv; colors 0-2 only need input channels 0 and 1."""
    weights = np.zeros((C, C // 2, 1, 3), dtype=np.float32)
    bias = np.zeros((C,), dtype=np.float32)

    # In group 0, local input channel 0 is background and 1 is blue.
    weights[0, 0, 0, 1] = 1.0
    weights[0, 1, 0, 0] = -0.6
    weights[0, 1, 0, 2] = -0.6

    weights[1, 1, 0, 1] = 1.0

    weights[2, 1, 0, 0] = 1.0
    weights[2, 0, 0, 1] = 1.0
    weights[2, 1, 0, 2] = 1.0
    bias[2] = -2.5

    inits: list[onnx.TensorProto] = []
    _init(inits, "conv_w", weights)
    _init(inits, "conv_b", bias)
    nodes = [
        helper.make_node(
            "Conv",
            [IN_NAME, "conv_w", "conv_b"],
            [OUT_NAME],
            kernel_shape=[1, 3],
            pads=[0, 1, 0, 1],
            group=2,
        )
    ]
    return _make_model(nodes, inits, "task258_grouped_conv")


def _to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def _examples() -> list[dict[str, list[list[int]]]]:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return [ex for split in ("train", "test", "arc-gen") for ex in data[split]]


def verify_model(model: onnx.ModelProto) -> tuple[int, int]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    passed = failed = 0
    for ex in _examples():
        actual = session.run([OUT_NAME], {IN_NAME: _to_onehot(ex["input"])})[0] > 0.0
        expected = _to_onehot(ex["output"]) > 0.0
        if np.array_equal(actual, expected):
            passed += 1
        else:
            failed += 1
    return passed, failed


def save_and_score(model: onnx.ModelProto) -> None:
    passed, failed = verify_model(model)
    if failed:
        raise AssertionError(f"model failed {failed} examples after passing {passed}")

    onnx.save(model, BEST_PATH)
    result = score_file(BEST_PATH)
    if not result["valid"]:
        raise AssertionError(f"score_model invalid: {result['error']}")

    print(f"pass:    {passed}")
    print(f"wrote {BEST_PATH}")
    print(f"valid:   {result['valid']}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


def main() -> None:
    save_and_score(build_model())


if __name__ == "__main__":
    main()
