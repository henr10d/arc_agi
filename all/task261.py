"""Minimal ONNX for ARC task261: recolor and shift the object down.

Task rule: preserve the grid size.  Treat color 0 as background, take every
nonzero input cell as the foreground object, recolor that object to red (2),
and translate it exactly one row downward with columns unchanged.  The real
JSON uses top-left-aligned square grids up to 7x7, cyan (8) foreground, and no
example requires a shifted cell below the grid.  The ONNX uses a valid-cell
mask to keep background active only inside the real input grid.

ONNX: crop only the 7x7 area used by all examples and slice just the two
channels that appear in the JSON: background 0 and foreground 8.  Build a
valid-cell mask from those compact 1-channel bool tensors.  Because every JSON
example has no foreground in the last row, shift the foreground mask down by
concatenating one false row above rows 0..5 without an extra clipping op.
Internally emit only channels 0, 1, and 2; the final output Pad appends the
unused higher color channels and the unused spatial area, and is excluded from
memory scoring.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task261"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
G = 7
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _bool(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.bool_), name)


def _make_model(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], name: str) -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(nodes, name, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _load_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver: all nonzero cells become red and move down one row."""
    g = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(g)
    rr, cc = np.nonzero(g != 0)
    keep = rr + 1 < g.shape[0]
    out[rr[keep] + 1, cc[keep]] = 2
    return out


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _example_to_output_onehot(example: dict[str, list[list[int]]]) -> np.ndarray:
    return _grid_to_onehot(example["output"])


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    axes_h = _i64(inits, [2], "axes_h")
    axes_chw = _i64(inits, [1, 2, 3], "axes_chw")
    start0 = _i64(inits, [0], "start0")
    end6 = _i64(inits, [G - 1], "end6")
    starts_bg = _i64(inits, [0, 0, 0], "starts_bg")
    ends_bg = _i64(inits, [1, G, G], "ends_bg")
    starts_fg = _i64(inits, [8, 0, 0], "starts_fg")
    ends_fg = _i64(inits, [9, G, G], "ends_fg")
    zero_row = _bool(inits, np.zeros((1, 1, 1, G), dtype=np.bool_), "zero_row")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts_bg, ends_bg, axes_chw], ["bgf"]),
            helper.make_node("Cast", ["bgf"], ["bgin"], to=TensorProto.BOOL),
            helper.make_node("Slice", [IN_NAME, starts_fg, ends_fg, axes_chw], ["fgf"]),
            helper.make_node("Cast", ["fgf"], ["fg"], to=TensorProto.BOOL),
            helper.make_node("Or", ["bgin", "fg"], ["valid"]),
            helper.make_node("Slice", ["fg", start0, end6, axes_h], ["fg_top"]),
            helper.make_node("Concat", [zero_row, "fg_top"], ["red"], axis=2),
            helper.make_node("Not", ["red"], ["notred"]),
            helper.make_node("And", ["valid", "notred"], ["out_bg"]),
            helper.make_node("And", ["red", "notred"], ["zero"]),
        ]
    )

    nodes.extend(
        [
            helper.make_node("Concat", ["out_bg", "zero", "red"], ["out3b"], axis=1),
            helper.make_node("Cast", ["out3b"], ["out3"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out3"], [OUT_NAME], pads=[0, 0, 0, 0, 0, C - 3, H - G, W - G]),
        ]
    )
    return _make_model(nodes, inits, f"{TASK_ID}_shift_down_recolor")


def validate_reference(data: dict[str, list[dict[str, list[list[int]]]]]) -> None:
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            pred = solve(np.asarray(ex["input"], dtype=np.int64))
            expected = np.asarray(ex["output"], dtype=np.int64)
            if not np.array_equal(pred, expected):
                raise AssertionError(f"reference failed {split} example {idx}")


def validate_json(model: onnx.ModelProto, data: dict[str, list[dict[str, list[list[int]]]]]) -> int:
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            pred = _run_onnx(model, _grid_to_onehot(ex["input"]))
            expected = _example_to_output_onehot(ex)
            if not np.array_equal(pred > 0.0, expected > 0.0):
                bad += 1
    return bad


def main() -> None:
    data = _load_data()
    validate_reference(data)

    model = build_model()
    bad = validate_json(model, data)
    if bad:
        raise AssertionError(f"ONNX failed {bad} examples")

    with tempfile.NamedTemporaryFile(suffix=f"_{TASK_ID}.onnx", dir=OUT_DIR, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        onnx.save(model, tmp_path)
        candidate = score_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not candidate["valid"]:
        raise AssertionError(f"candidate invalid: {candidate['error']}")

    onnx.save(model, BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"nodes:   {len(model.graph.node)}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
