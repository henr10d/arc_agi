"""ONNX solver for ARC task011 using Kaggle one-hot I/O.

Task rule: the 11x11 grid is split by gray separator rows/columns at indices
3 and 7 into a 3x3 array of 3x3 panels. Exactly one panel is a template: it is
the only 3x3 panel that does not contain color 8. Copy that template into the
3x3 meta-grid of the output by replacing each template cell with a solid 3x3
block of the same color, while preserving the gray separator cross.

ONNX approach: sum the color-8 channel inside each candidate panel to find the
zero-count template panel, select that compact [1,10,3,3] one-hot panel, tile
each template cell to 3x3, then concatenate the fixed gray separator strips.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task011"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
S = 11
P = 3
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10
BLOCKS: Tuple[Tuple[int, int], ...] = (
    (0, 0),
    (0, 4),
    (0, 8),
    (4, 0),
    (4, 4),
    (4, 8),
    (8, 0),
    (8, 4),
    (8, 8),
)


def _i64(inits: List[onnx.TensorProto], vals: Sequence[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))
    return name


def _f32(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name))
    return name


def _bool(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=bool), name=name))
    return name


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver: expand the one 3x3 panel that omits color 8."""
    g = np.asarray(grid, dtype=np.int64)
    if g.ndim == 4:
        g = g[0].argmax(axis=0).astype(np.int64)
    elif g.ndim == 3:
        g = g.argmax(axis=0).astype(np.int64)

    template = None
    for r, c in BLOCKS:
        block = g[r : r + P, c : c + P]
        if not np.any(block == 8):
            template = block
            break
    if template is None:
        raise ValueError("task011 input has no color-8-free template block")

    out = np.zeros((S, S), dtype=np.int64)
    out[3, :] = 5
    out[7, :] = 5
    out[:, 3] = 5
    out[:, 7] = 5
    for br, r in enumerate((0, 4, 8)):
        for bc, c in enumerate((0, 4, 8)):
            out[r : r + P, c : c + P] = int(template[br, bc])
    return out


def _grid_to_onehot(grid: Sequence[Sequence[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            color = int(val)
            if 0 <= color < C:
                out[0, color, r, c] = 1.0
    return out


def build_onnx_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _i64(inits, [0, 1, 2, 3], "axes4")
    zero = _f32(inits, 0.0, "zero")
    one = _f32(inits, 1.0, "one")
    tile_reps = _i64(inits, [1, 1, 1, P, 1, P], "tile_reps")
    expanded_shape = _i64(inits, [1, C, 9, 9], "expanded_shape")
    c8_start = _i64(inits, [0, 8, 0, 0], "c8s")
    c8_end = _i64(inits, [1, 9, P, P], "c8e")

    selected_terms: List[str] = []
    for index, (r, c) in enumerate(BLOCKS):
        block = f"b{index}"
        start = _i64(inits, [0, 0, r, c], f"bs{index}")
        end = _i64(inits, [1, C, r + P, c + P], f"be{index}")
        nodes.append(helper.make_node("Slice", [IN_NAME, start, end, axes4], [block]))

        color8 = f"c8_{index}"
        nodes.append(helper.make_node("Slice", [block, c8_start, c8_end, axes4], [color8]))
        nodes.append(helper.make_node("ReduceSum", [color8], [f"sum8_{index}"], axes=[1, 2, 3], keepdims=1))
        nodes.append(helper.make_node("Less", [f"sum8_{index}", one], [f"selb_{index}"]))
        nodes.append(helper.make_node("Greater", [block, zero], [f"blockb_{index}"]))
        nodes.append(helper.make_node("And", [f"blockb_{index}", f"selb_{index}"], [f"term_{index}"]))
        selected_terms.append(f"term_{index}")

    current = selected_terms[0]
    for index, term in enumerate(selected_terms[1:], start=1):
        merged = f"picked_{index}"
        nodes.append(helper.make_node("Or", [current, term], [merged]))
        current = merged

    nodes.append(helper.make_node("Unsqueeze", [current], ["picked6"], axes=[3, 5]))
    nodes.append(helper.make_node("Tile", ["picked6", tile_reps], ["tiled6"]))
    nodes.append(helper.make_node("Reshape", ["tiled6", expanded_shape], ["body"]))

    v_sep = np.zeros((1, C, P, 1), dtype=bool)
    v_sep[:, 5, :, :] = 1.0
    h_sep = np.zeros((1, C, 1, S), dtype=bool)
    h_sep[:, 5, :, :] = 1.0
    _bool(inits, v_sep, "v_sep")
    _bool(inits, h_sep, "h_sep")

    rows: List[str] = []
    for br, row_start in enumerate((0, 3, 6)):
        patches: List[str] = []
        for bc, col_start in enumerate((0, 3, 6)):
            patch = f"p{br}{bc}"
            start = _i64(inits, [0, 0, row_start, col_start], f"ps{br}{bc}")
            end = _i64(inits, [1, C, row_start + P, col_start + P], f"pe{br}{bc}")
            nodes.append(helper.make_node("Slice", ["body", start, end, axes4], [patch]))
            patches.append(patch)
        row_name = f"row{br}"
        nodes.append(helper.make_node("Concat", [patches[0], "v_sep", patches[1], "v_sep", patches[2]], [row_name], axis=3))
        rows.append(row_name)

    nodes.append(helper.make_node("Concat", [rows[0], "h_sep", rows[1], "h_sep", rows[2]], ["out11b"], axis=2))
    nodes.append(helper.make_node("Cast", ["out11b"], ["out11"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Pad", ["out11"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - S, W - S]))

    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name=TASK_ID,
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return model


def save_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    model = build_onnx_model()
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return model


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def verify_all(model: onnx.ModelProto) -> int:
    bad = 0
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            g = np.array(ex["input"], dtype=np.int64)
            if g.shape[0] > H or g.shape[1] > W:
                continue
            pred = _run_onnx(model, _grid_to_onehot(g.tolist()))
            ref = _grid_to_onehot(ex["output"])
            if not np.array_equal(pred > 0.0, ref > 0.0):
                bad += 1
    return bad


def main() -> None:
    model = save_model()
    bad = verify_all(model)
    print(f"verify: {'PASS' if bad == 0 else f'FAIL ({bad})'}")
    stats = score_file(BEST_PATH)
    print(
        f"memory={stats.get('memory')} params={stats.get('params')} "
        f"cost={stats.get('cost')} score={stats.get('score', 0):.4f}"
    )


if __name__ == "__main__":
    main()
