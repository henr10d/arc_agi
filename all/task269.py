"""Minimal ONNX for ARC task269: expand colored 3x3 cells by foreground count.

Task rule: the input is a 3x3 grid.  Let k be the number of non-black cells.
The output has logical size 3*k by 3*k.  Each colored input cell at (r, c)
becomes a solid k by k block of the same color at rows r*k:(r+1)*k and columns
c*k:(c+1)*k.  Black input cells become black blocks inside the logical output;
competition padding outside 3*k by 3*k remains all-zero.

ONNX: crop the 3x3 one-hot core, compute k as 9 minus the black-cell count,
build a compact 30x3 row/column selector from coord // k, then apply the
selectors with two MatMul nodes.  Rows/columns beyond 3*k map to indices >= 3
and therefore have no selector entries, so padding is emitted as all-zero
without a final Pad.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task269"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task269.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
N = 3
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver for the count-controlled 3x3 block expansion."""
    core = np.asarray(grid, dtype=np.int64)[:N, :N]
    k = int(np.count_nonzero(core))
    out = np.zeros((N * k, N * k), dtype=np.int64)
    for r in range(N):
        for c in range(N):
            color = int(core[r, c])
            out[r * k : (r + 1) * k, c * k : (c + 1) * k] = color
    return out


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _i32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int32), name)


def _make_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        name,
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
    onnx.checker.check_model(model, full_check=True)
    return model


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    core_starts = _i64(inits, [0, 0], "core_starts")
    core_ends = _i64(inits, [N, N], "core_ends")
    core_axes = _i64(inits, [2, 3], "core_axes")
    black_index = _i64(inits, 0, "black_index")
    coords = _i32(inits, np.arange(H, dtype=np.int32).reshape(H, 1), "coords")
    src = _i32(inits, np.arange(N, dtype=np.int32).reshape(1, N), "src")
    nine = _i32(inits, 9, "nine")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, core_starts, core_ends, core_axes], ["core"]),
            helper.make_node("Gather", ["core", black_index], ["black"], axis=1),
            helper.make_node("ReduceSum", ["black"], ["black_count"], axes=[1, 2], keepdims=0),
            helper.make_node("Cast", ["black_count"], ["black_count_i"], to=TensorProto.INT32),
            helper.make_node("Sub", [nine, "black_count_i"], ["ki"]),
            helper.make_node("Div", [coords, "ki"], ["idx"]),
            helper.make_node("Equal", ["idx", src], ["selector_b"]),
            helper.make_node("Cast", ["selector_b"], ["selector"], to=TensorProto.FLOAT),
            helper.make_node("Transpose", ["selector"], ["selector_t"], perm=[1, 0]),
            helper.make_node("MatMul", ["selector", "core"], ["rows"]),
            helper.make_node("MatMul", ["rows", "selector_t"], [OUT_NAME]),
        ]
    )
    return _make_model(nodes, inits, "task269_dynamic_count_expansion")


def validate_json(model: onnx.ModelProto) -> dict[str, int]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    counts: dict[str, int] = {}
    for split in ("train", "test", "arc-gen"):
        counts[split] = 0
        for idx, ex in enumerate(data.get(split, [])):
            expected_grid = np.asarray(ex["output"], dtype=np.int64)
            ref = solve(np.asarray(ex["input"], dtype=np.int64))
            if not np.array_equal(ref, expected_grid):
                raise AssertionError(f"reference mismatch in {split} example {idx}")

            inp = convert_to_numpy(ex, "input")
            expected = convert_to_numpy(ex, "output")
            if inp is None or expected is None:
                continue
            pred = session.run([OUT_NAME], {IN_NAME: inp})[0]
            if not np.array_equal(pred > 0.0, expected > 0.0):
                raise AssertionError(f"ONNX mismatch in {split} example {idx}")
            counts[split] += 1
    return counts


def tensor_count(model: onnx.ModelProto) -> int:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return len(inferred.graph.value_info)


def main() -> None:
    model = build_model()
    counts = validate_json(model)
    with tempfile.NamedTemporaryFile(suffix=f"_{TASK_ID}.onnx", dir=OUT_DIR, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        onnx.save(model, tmp_path)
        result = score_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not result["valid"]:
        raise AssertionError(f"invalid model: {result['error']}")

    onnx.save(model, BEST_PATH)
    print(f"validated {counts}")
    print(
        f"wrote {BEST_PATH} nodes={len(model.graph.node)} tensors={tensor_count(model)} "
        f"memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']:.6f}"
    )


if __name__ == "__main__":
    main()
