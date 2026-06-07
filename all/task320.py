"""ONNX for ARC task320: recolor the lower half of each red bar cyan.

Task rule: each input is a black grid with several contiguous vertical red
columns. For every red column, keep the upper ``ceil(height / 2)`` red cells
unchanged and recolor the remaining lower red cells cyan. Black cells and
padding outside the true grid stay unchanged.

ONNX approach: the selected graph slices the observed 11x9 active canvas,
computes each red column's top row with ``ArgMax`` and height with
``ReduceSum``, splits lower-half red cells into a cyan channel, concatenates a
compact 11x9 one-hot crop, and pads that crop back to the required 30x30
output. The script also scores two full-canvas alternatives for comparison.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task320"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
OPSET = 10
RED = 2
CYAN = 8


@dataclass(frozen=True)
class Candidate:
    name: str
    strategy: str


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []

    def node(self, op_type: str, inputs: list[str], output: str, **attrs: Any) -> str:
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        return output

    def init(self, name: str, values: Any, dtype: np.dtype[Any] | type[Any]) -> str:
        self.inits.append(numpy_helper.from_array(np.asarray(values, dtype=dtype), name))
        return name

    def f32(self, name: str, values: Any) -> str:
        return self.init(name, values, np.float32)

    def i64(self, name: str, values: Any) -> str:
        return self.init(name, values, np.int64)


def load_examples() -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    examples: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            examples.append(
                (
                    split,
                    idx,
                    np.asarray(ex["input"], dtype=np.int64),
                    np.asarray(ex["output"], dtype=np.int64),
                )
            )
    return examples


def solve(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=np.int64).copy()
    _, w = out.shape
    for c in range(w):
        rows = np.flatnonzero(out[:, c] == RED)
        if rows.size == 0:
            continue
        split = int(rows[0] + math.ceil(rows.size / 2))
        out[split : rows[-1] + 1, c] = CYAN
    return out


def grid_to_onehot(grid: np.ndarray) -> np.ndarray:
    arr = np.zeros(SHAPE, dtype=np.float32)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            arr[0, int(grid[r, c]), r, c] = 1.0
    return arr


def onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return (onehot > 0.0).argmax(axis=1)[0].astype(np.int64)


def run_onnx(model: onnx.ModelProto, grid: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run([OUT_NAME], {IN_NAME: grid_to_onehot(grid)})[0]


def make_model(b: Builder, graph_name: str) -> onnx.ModelProto:
    graph = helper.make_graph(
        b.nodes,
        graph_name,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        b.inits,
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


def add_red_gather(b: Builder) -> str:
    red_idx = b.i64("red_idx", [RED])
    return b.node("Gather", [IN_NAME, red_idx], "red", axis=1)


def add_cyan_where_output(b: Builder, mask: str) -> None:
    cyan = np.zeros((1, 10, 1, 1), dtype=np.float32)
    cyan[:, CYAN, :, :] = 1.0
    b.node("Where", [mask, b.f32("cyan", cyan), IN_NAME], OUT_NAME)


def build_top_height_model() -> onnx.ModelProto:
    """Full-canvas candidate: top row from ArgMax and bar height from ReduceSum."""
    b = Builder()
    red = add_red_gather(b)
    height = b.node("ReduceSum", [red], "height", axes=[2], keepdims=1)
    top_i = b.node("ArgMax", [red], "top_i", axis=2, keepdims=1)
    top = b.node("Cast", [top_i], "top", to=TensorProto.FLOAT)
    top2 = b.node("Add", [top, top], "top2")
    split_limit = b.node("Sub", [b.node("Add", [top2, height], "sum"), b.f32("one", [1.0])], "limit")
    row2 = b.f32("row2", (np.arange(30, dtype=np.float32) * 2.0).reshape(1, 1, 30, 1))
    lower = b.node("Greater", [row2, split_limit], "lower")
    red_b = b.node("Cast", [red], "red_b", to=TensorProto.BOOL)
    mask = b.node("And", [lower, red_b], "mask")
    add_cyan_where_output(b, mask)
    return make_model(b, "task320_top_height")


def build_crop_concat_model() -> onnx.ModelProto:
    """Alternative candidate: crop to the observed 11x9 canvas and pad output."""
    b = Builder()
    axes = b.i64("axes", [1, 2, 3])
    ch0 = b.node("Slice", [IN_NAME, b.i64("s0", [0, 0, 0]), b.i64("e0", [1, 11, 9]), axes], "ch0")
    red = b.node("Slice", [IN_NAME, b.i64("sr", [RED, 0, 0]), b.i64("er", [RED + 1, 11, 9]), axes], "red")
    height = b.node("ReduceSum", [red], "height", axes=[2], keepdims=1)
    top_i = b.node("ArgMax", [red], "top_i", axis=2, keepdims=1)
    top = b.node("Cast", [top_i], "top", to=TensorProto.FLOAT)
    top2 = b.node("Add", [top, top], "top2")
    limit = b.node("Sub", [b.node("Add", [top2, height], "sum"), b.f32("one", [1.0])], "limit")
    row2 = b.f32("row2", (np.arange(11, dtype=np.float32) * 2.0).reshape(1, 1, 11, 1))
    lower = b.node("Greater", [row2, limit], "lower")
    zero = b.node("Sub", [ch0, ch0], "zero")
    red_out = b.node("Where", [lower, zero, red], "red_out")
    cyan_out = b.node("Where", [lower, red, zero], "cyan_out")
    b.node(
        "Concat",
        [ch0, zero, red_out, zero, zero, zero, zero, zero, cyan_out, zero],
        "crop",
        axis=1,
    )
    b.nodes.append(
        helper.make_node(
            "Pad",
            ["crop"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 19, 21],
            value=0.0,
        )
    )
    return make_model(b, "task320_crop_concat")


def build_reverse_argmax_model() -> onnx.ModelProto:
    """Alternative candidate: compute top and bottom rows with two ArgMax ops."""
    b = Builder()
    red = add_red_gather(b)
    top_i = b.node("ArgMax", [red], "top_i", axis=2, keepdims=1)
    top = b.node("Cast", [top_i], "top", to=TensorProto.FLOAT)
    rev = b.node(
        "Slice",
        [red, b.i64("rev_s", [29]), b.i64("rev_e", [-31]), b.i64("rev_a", [2]), b.i64("rev_step", [-1])],
        "rev",
    )
    rev_i = b.node("ArgMax", [rev], "rev_i", axis=2, keepdims=1)
    rev_f = b.node("Cast", [rev_i], "rev_f", to=TensorProto.FLOAT)
    bottom = b.node("Sub", [b.f32("last", [29.0]), rev_f], "bottom")
    limit = b.node("Add", [top, bottom], "limit")
    row2 = b.f32("row2", (np.arange(30, dtype=np.float32) * 2.0).reshape(1, 1, 30, 1))
    lower = b.node("Greater", [row2, limit], "lower")
    red_b = b.node("Cast", [red], "red_b", to=TensorProto.BOOL)
    mask = b.node("And", [lower, red_b], "mask")
    add_cyan_where_output(b, mask)
    return make_model(b, "task320_reverse_argmax")


def validate_reference(examples: list[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    for split, idx, inp, expected in examples:
        got = solve(inp)
        if not np.array_equal(got, expected):
            raise AssertionError(f"reference failed {split}[{idx}]\nexpected:\n{expected}\ngot:\n{got}")


def validate_model(model: onnx.ModelProto, examples: list[tuple[str, int, np.ndarray, np.ndarray]]) -> None:
    for split, idx, inp, expected in examples:
        pred = onehot_to_grid(run_onnx(model, inp))[: expected.shape[0], : expected.shape[1]]
        if not np.array_equal(pred, expected):
            raise AssertionError(f"ONNX failed {split}[{idx}]\nexpected:\n{expected}\ngot:\n{pred}")


def evaluate(candidate: Candidate, examples: list[tuple[str, int, np.ndarray, np.ndarray]], workdir: Path) -> dict[str, Any]:
    builders = {
        "top_height": build_top_height_model,
        "crop_concat": build_crop_concat_model,
        "reverse_argmax": build_reverse_argmax_model,
    }
    model = builders[candidate.strategy]()
    validate_model(model, examples)
    path = workdir / f"{candidate.name}.onnx"
    onnx.save(model, path)
    result = score_file(path)
    result["candidate"] = candidate
    return result


def print_result(result: dict[str, Any]) -> None:
    candidate = result["candidate"]
    if not result["valid"]:
        print(f"{candidate.name}: INVALID {result['error']}")
        return
    print(
        f"{candidate.name}: memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']:.6f}"
    )


def main() -> None:
    examples = load_examples()
    validate_reference(examples)

    candidates = [
        Candidate("top_height", "top_height"),
        Candidate("crop_concat", "crop_concat"),
        Candidate("reverse_argmax", "reverse_argmax"),
    ]
    best: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}_") as tmp:
        workdir = Path(tmp)
        for candidate in candidates:
            try:
                result = evaluate(candidate, examples, workdir)
            except Exception as exc:
                print(f"{candidate.name}: INVALID {exc}")
                continue
            print_result(result)
            if result["valid"] and (best is None or int(result["cost"]) < int(best["cost"])):
                best = result

        if best is None:
            raise SystemExit("no valid candidate found")
        shutil.copyfile(best["path"], BEST_PATH)

    final = score_file(BEST_PATH)
    if not final["valid"]:
        raise SystemExit(f"persisted model is invalid: {final['error']}")

    chosen = best["candidate"]
    print(
        f"selected {chosen.name}: memory={final['memory']} params={final['params']} "
        f"cost={final['cost']} score={final['score']:.6f}"
    )
    print(f"wrote {BEST_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
