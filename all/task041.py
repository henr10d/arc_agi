"""Build an ONNX solver for NeuroGolf task041.

Task rule: each non-black color marks paired boundary cells in rows of a
downward triangle/trapezoid inside the fixed 10x10 grid. For every color and
row independently, fill the horizontal segment between that row's leftmost and
rightmost cells of the same color. Rows without that color stay black. In the
provided train/test/arc-gen examples these inferred spans do not overlap across
colors, so the completed masks directly form the output foreground.

The ONNX graph slices the ten foreground columns as compact [1,9,10,1] tensors,
uses boolean prefix/suffix scans to mark cells with a same-color endpoint on
both sides, concatenates the completed columns, then casts to float and pads
only for the final 30x30 competition output.
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
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402


TASK_NUM = "041"
TASK_NAME = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_NAME}.json"
OUT_PATH = Path(__file__).resolve().parent / f"{TASK_NAME}.onnx"


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.value_infos: list[onnx.ValueInfoProto] = []
        self.counter = 0

    def name(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def init(self, name: str, array: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(array, name))
        return name

    def vi(self, name: str, dtype: int, shape: tuple[int, ...]) -> str:
        self.value_infos.append(helper.make_tensor_value_info(name, dtype, list(shape)))
        return name

    def node(
        self,
        op_type: str,
        inputs: list[str],
        dtype: int,
        shape: tuple[int, ...],
        prefix: str,
        **attrs: object,
    ) -> str:
        out = self.vi(self.name(prefix), dtype, shape)
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out


def grid_from_onehot(arr: np.ndarray) -> np.ndarray:
    crop = arr[0, :, :10, :10]
    active = crop > 0
    grid = active.argmax(axis=0).astype(np.int64)
    grid[~active.any(axis=0)] = 0
    return grid


def add_initializers(b: Builder) -> None:
    b.init("zero_f", np.array(0.0, dtype=np.float32))
    b.init("col_axes", np.array([0, 1, 2, 3], dtype=np.int64))
    for col in range(10):
        b.init(f"col{col}_starts", np.array([0, 1, 0, col], dtype=np.int64))
        b.init(f"col{col}_ends", np.array([1, 10, 10, col + 1], dtype=np.int64))


def build_model() -> onnx.ModelProto:
    b = Builder()
    add_initializers(b)

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])

    input_cols: list[str] = []
    for col in range(10):
        col_float = b.node(
            "Slice",
            ["input", f"col{col}_starts", f"col{col}_ends", "col_axes"],
            TensorProto.FLOAT,
            (1, 9, 10, 1),
            f"col{col}_float",
        )
        col_bool = b.node("Greater", [col_float, "zero_f"], TensorProto.BOOL, (1, 9, 10, 1), f"col{col}_bool")
        input_cols.append(col_bool)

    prefixes: list[str] = [input_cols[0]]
    for col in range(1, 10):
        prefixes.append(b.node("Or", [prefixes[-1], input_cols[col]], TensorProto.BOOL, (1, 9, 10, 1), "prefix"))

    suffixes: list[str] = [""] * 10
    suffixes[9] = input_cols[9]
    for col in range(8, -1, -1):
        suffixes[col] = b.node("Or", [input_cols[col], suffixes[col + 1]], TensorProto.BOOL, (1, 9, 10, 1), "suffix")

    out_cols = [
        b.node("And", [prefixes[col], suffixes[col]], TensorProto.BOOL, (1, 9, 10, 1), "out_col")
        for col in range(10)
    ]
    wedges = b.node("Concat", out_cols, TensorProto.BOOL, (1, 9, 10, 10), "wedges", axis=3)

    wedge_channels = [b.vi(b.name("wedge_ch"), TensorProto.BOOL, (1, 1, 10, 10)) for _ in range(9)]
    b.nodes.append(helper.make_node("Split", [wedges], wedge_channels, axis=1, split=[1] * 9))
    any_out_bool = wedge_channels[0]
    for channel in wedge_channels[1:]:
        any_out_bool = b.node("Or", [any_out_bool, channel], TensorProto.BOOL, (1, 1, 10, 10), "any_out_bool")
    bg_bool = b.node("Not", [any_out_bool], TensorProto.BOOL, (1, 1, 10, 10), "bg_bool")
    solved_bool = b.node("Concat", [bg_bool, wedges], TensorProto.BOOL, (1, 10, 10, 10), "solved_bool", axis=1)
    final_float = b.node("Cast", [solved_bool], TensorProto.FLOAT, (1, 10, 10, 10), "final_float", to=TensorProto.FLOAT)
    b.nodes.append(
        helper.make_node(
            "Pad",
            [final_float],
            ["output"],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 20, 20],
            value=0.0,
        )
    )

    graph = helper.make_graph(b.nodes, TASK_NAME, [inp], [out], b.initializers, value_info=b.value_infos)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 10)])
    model.ir_version = 10
    return model


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
            match = np.array_equal(pred > 0, expected > 0)
            ok = ok and match
            if not match:
                print(f"{split}[{idx}] failed")
                print("predicted:")
                print(grid_from_onehot(pred))
                print("expected:")
                print(np.array(ex["output"], dtype=np.int64))
                return False
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
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
