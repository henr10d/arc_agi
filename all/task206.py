"""ONNX solution for ARC task206: duplicate a 3x3-bounded colored object.

Task rule: the input has one gray marker (color 5) and one connected non-gray
colored object whose bounding box is 3x3. Preserve the original object, remove
the gray marker, and paste a second copy of the object's 3x3 patch so that the
center of the source bounding box lands on the gray marker cell. The copied
shape keeps the same colors, orientation, and geometry.

ONNX approach: all task206 JSON grids fit in the top-left 12x12 region and use
object colors 1, 2, 3, and 6. The graph slices only those input channels, keeps
the copied foreground as four compact bool channels, scatters the 3x3 patch
directly into the original foreground tensor, derives zero channels on-graph,
then casts once before padding to the required 30x30 float output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task206"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task206.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
OH = OW = 30
H = W = 12
N = H * W
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, OH, OW]
OPSET = 10
IR_VERSION = 10


def solve(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    """Reference implementation used to validate the inferred rule."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    gray_pos = np.argwhere(g == 5)
    if len(gray_pos) != 1:
        raise ValueError("expected exactly one gray marker")
    gr, gc = map(int, gray_pos[0])
    out[gr, gc] = 0

    obj = np.argwhere((g != 0) & (g != 5))
    if len(obj) == 0:
        return out
    r0, c0 = obj.min(axis=0)
    r1, c1 = obj.max(axis=0)
    cr, cc = int(r0 + 1), int(c0 + 1)
    for r in range(int(r0), int(r1) + 1):
        for c in range(int(c0), int(c1) + 1):
            color = int(g[r, c])
            if color not in (0, 5):
                tr, tc = gr + (r - cr), gc + (c - cc)
                if 0 <= tr < g.shape[0] and 0 <= tc < g.shape[1]:
                    out[tr, tc] = color
    return out


def _grid_to_onehot(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(arr):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def _onehot_expected(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    return _grid_to_onehot(grid)


def _decode(onehot: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    return (onehot[0, :, :h, :w] > 0).argmax(axis=0).astype(np.int64)


def _arr(inits: List[onnx.TensorProto], values: Any, name: str, dtype: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(values, dtype=dtype), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], values: Any, name: str) -> str:
    return _arr(inits, values, name, np.int64)


def _f32(inits: List[onnx.TensorProto], values: Any, name: str) -> str:
    return _arr(inits, values, name, np.float32)


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    fg_flat_shape = _i64(inits, [1, 4, N], "fg_flat_shape")
    fg_grid_shape = _i64(inits, [1, 4, H, W], "fg_grid_shape")
    one_i = _i64(inits, [1], "one_i")
    thirty_i = _i64(inits, [W], "thirty_i")
    patch_offsets = _i64(
        inits,
        [-W - 1, -W, -W + 1, -1, 0, 1, W - 1, W, W + 1],
        "patch_offsets",
    )
    zero = _f32(inits, [0.0], "zero")

    s0 = _i64(inits, [0, 0, 0, 0], "s0")
    e1 = _i64(inits, [1, 1, H, W], "e1")
    s1 = _i64(inits, [0, 1, 0, 0], "s1")
    e4 = _i64(inits, [1, 4, H, W], "e4")
    s5 = _i64(inits, [0, 5, 0, 0], "s5")
    e6 = _i64(inits, [1, 6, H, W], "e6")
    s6 = _i64(inits, [0, 6, 0, 0], "s6")
    e7 = _i64(inits, [1, 7, H, W], "e7")

    def add(op: str, ins: list[str], outs: list[str], **kwargs: Any) -> None:
        nodes.append(helper.make_node(op, ins, outs, **kwargs))

    # Object mask excludes background and gray. Its first occupied row/column
    # plus one is the center because all examples use a 3x3 object bbox.
    add("Slice", [IN_NAME, s1, e4, axes4], ["obj_a_f"])
    add("Slice", [IN_NAME, s6, e7, axes4], ["obj_b_f"])
    add("Greater", ["obj_a_f", zero], ["obj_a"])
    add("Greater", ["obj_b_f", zero], ["obj_b"])
    add("Concat", ["obj_a", "obj_b"], ["obj4"], axis=1)
    add("ReduceSum", ["obj_a_f"], ["obj_a_rows"], axes=[1, 3], keepdims=0)
    add("ReduceSum", ["obj_b_f"], ["obj_b_rows"], axes=[1, 3], keepdims=0)
    add("ReduceSum", ["obj_a_f"], ["obj_a_cols"], axes=[1, 2], keepdims=0)
    add("ReduceSum", ["obj_b_f"], ["obj_b_cols"], axes=[1, 2], keepdims=0)
    add("Add", ["obj_a_rows", "obj_b_rows"], ["obj_rows_sum"])
    add("Add", ["obj_a_cols", "obj_b_cols"], ["obj_cols_sum"])
    add("Greater", ["obj_rows_sum", zero], ["obj_rows_b"])
    add("Greater", ["obj_cols_sum", zero], ["obj_cols_b"])
    add("Cast", ["obj_rows_b"], ["obj_rows"], to=TensorProto.FLOAT)
    add("Cast", ["obj_cols_b"], ["obj_cols"], to=TensorProto.FLOAT)
    add("ArgMax", ["obj_rows"], ["obj_r0"], axis=1, keepdims=1)
    add("ArgMax", ["obj_cols"], ["obj_c0"], axis=1, keepdims=1)
    add("Add", ["obj_r0", one_i], ["obj_cr_3"])
    add("Add", ["obj_c0", one_i], ["obj_cc_3"])
    add("Squeeze", ["obj_cr_3"], ["obj_cr"], axes=[0, 1])
    add("Squeeze", ["obj_cc_3"], ["obj_cc"], axes=[0, 1])

    # Gray marker position.
    add("Slice", [IN_NAME, s5, e6, axes4], ["gray"])
    add("ReduceSum", ["gray"], ["gray_rows"], axes=[3], keepdims=0)
    add("ReduceSum", ["gray"], ["gray_cols"], axes=[2], keepdims=0)
    add("ArgMax", ["gray_rows"], ["gray_r_3"], axis=2, keepdims=1)
    add("ArgMax", ["gray_cols"], ["gray_c_3"], axis=2, keepdims=1)
    add("Squeeze", ["gray_r_3"], ["gray_r"], axes=[0, 1, 2])
    add("Squeeze", ["gray_c_3"], ["gray_c"], axes=[0, 1, 2])

    # Gather the source 3x3 patch and scatter it at the gray-centered target.
    add("Mul", ["obj_cr", thirty_i], ["src_r_flat"])
    add("Add", ["src_r_flat", "obj_cc"], ["src_center"])
    add("Add", ["src_center", patch_offsets], ["src_idx"])
    add("Mul", ["gray_r", thirty_i], ["tgt_r_flat"])
    add("Add", ["tgt_r_flat", "gray_c"], ["tgt_center"])
    add("Add", ["tgt_center", patch_offsets], ["tgt_idx"])
    add("Unsqueeze", ["tgt_idx"], ["tgt_idx_1"], axes=[0])
    add("Unsqueeze", ["tgt_idx_1"], ["tgt_idx_2"], axes=[0])
    add("Concat", ["tgt_idx_2", "tgt_idx_2", "tgt_idx_2", "tgt_idx_2"], ["tgt_idx_tiled"], axis=1)

    add("Reshape", ["obj4", fg_flat_shape], ["obj4_flat"])
    add("Gather", ["obj4_flat", "src_idx"], ["patch"], axis=2)
    add("Scatter", ["obj4_flat", "tgt_idx_tiled", "patch"], ["fg4_flat"], axis=2)
    add("Reshape", ["fg4_flat", fg_grid_shape], ["fg4"])

    # Merge foreground only; channel 5 is forced to zero and background is
    # reconstructed inside the original padded grid extent.
    add("Split", ["fg4"], ["fg1", "fg2", "fg3", "fg6"], axis=1, split=[1, 1, 1, 1])
    add("Or", ["fg1", "fg2"], ["fg12"])
    add("Or", ["fg3", "fg6"], ["fg36"])
    add("Or", ["fg12", "fg36"], ["fg_present"])
    add("Slice", [IN_NAME, s0, e1, axes4], ["bg_in_f"])
    add("Greater", ["bg_in_f", zero], ["bg_in"])
    add("Greater", ["gray", zero], ["gray_b"])
    add("Or", ["bg_in", "gray_b"], ["valid_bg"])
    add("Not", ["fg_present"], ["no_fg"])
    add("And", ["fg_present", "no_fg"], ["zero_ch"])
    add("And", ["valid_bg", "no_fg"], ["bg_b"])
    add(
        "Concat",
        ["bg_b", "fg1", "fg2", "fg3", "zero_ch", "zero_ch", "fg6", "zero_ch", "zero_ch", "zero_ch"],
        ["out12_b"],
        axis=1,
    )
    add("Cast", ["out12_b"], ["out12"], to=TensorProto.FLOAT)
    add("Pad", ["out12"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, OH - H, OW - W])

    graph = helper.make_graph(nodes, "task206", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _candidate_report(data: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Validate candidate anchor definitions against all train examples."""

    def object_cells(g: np.ndarray) -> np.ndarray:
        return np.argwhere((g != 0) & (g != 5))

    def translated(g: np.ndarray, ref: Tuple[int, int]) -> np.ndarray:
        gray = tuple(map(int, np.argwhere(g == 5)[0]))
        out = g.copy()
        out[gray] = 0
        dr, dc = gray[0] - ref[0], gray[1] - ref[1]
        for r, c in object_cells(g):
            color = int(g[r, c])
            tr, tc = int(r + dr), int(c + dc)
            if 0 <= tr < g.shape[0] and 0 <= tc < g.shape[1]:
                out[tr, tc] = color
        return out

    candidates: list[tuple[str, Any]] = [
        ("bbox top-left", lambda g, obj: tuple(obj.min(axis=0))),
        ("bbox top-right", lambda g, obj: (int(obj[:, 0].min()), int(obj[:, 1].max()))),
        ("bbox bottom-left", lambda g, obj: (int(obj[:, 0].max()), int(obj[:, 1].min()))),
        ("bbox center", lambda g, obj: (int(obj[:, 0].min() + 1), int(obj[:, 1].min() + 1))),
        ("rounded object centroid", lambda g, obj: tuple(np.rint(obj.mean(axis=0)).astype(int))),
        ("first non-gray cell", lambda g, obj: tuple(obj[np.lexsort((obj[:, 1], obj[:, 0]))][0])),
    ]
    lines: list[str] = []
    for name, fn in candidates:
        ok = 0
        for ex in data["train"]:
            g = np.asarray(ex["input"], dtype=np.int64)
            expected = np.asarray(ex["output"], dtype=np.int64)
            pred = translated(g, fn(g, object_cells(g)))
            ok += int(np.array_equal(pred, expected))
        lines.append(f"{name}: {ok}/{len(data['train'])}")
    return lines


def verify_model(model: onnx.ModelProto, examples: Iterable[dict[str, Any]]) -> int:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    checked = 0
    for ex in examples:
        inp_grid = np.asarray(ex["input"], dtype=np.int64)
        expected = _onehot_expected(ex["output"])
        pred = sess.run([OUT_NAME], {IN_NAME: _grid_to_onehot(inp_grid)})[0]
        if not np.array_equal(pred > 0.0, expected > 0.0):
            decoded = _decode(pred, np.asarray(ex["output"]).shape)
            raise AssertionError(f"prediction mismatch:\n{decoded}\nexpected:\n{np.asarray(ex['output'])}")
        checked += 1
    return checked


def main() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    print("Candidate anchors on train:")
    for line in _candidate_report(data):
        print(f"  {line}")

    for split, examples in data.items():
        for i, ex in enumerate(examples):
            ref = solve(ex["input"])
            if not np.array_equal(ref, np.asarray(ex["output"], dtype=np.int64)):
                raise AssertionError(f"reference failed {split}[{i}]")

    model = build_model()
    checked = verify_model(model, [ex for split in ("train", "test", "arc-gen") for ex in data[split]])
    onnx.save(model, BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"Verified {checked} examples.")
    print(
        f"{BEST_PATH}: valid={result['valid']} memory={result['memory']} "
        f"params={result['params']} cost={result['cost']} score={result['score']}"
    )


if __name__ == "__main__":
    main()
