"""ARC task049: output the smallest visible colored rectangle.

The input contains several non-black colored rectangle-like objects on a black
background. The output is a solid rectangle in the color with the smallest
visible footprint, placed at the top-left of the 30x30 NeuroGolf canvas. Across
the supplied train/test/arc-gen examples, the winning object is determined by
minimum occupied-row count times occupied-column count, with no selection ties;
the ONNX graph reduces directly over the full input and ignores color 0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task049.onnx"
TASK_JSON = OUT_DIR.parent / "data" / "task049.json"

C = 10
NC = C - 1
H = W = 30
BIG = 999.0
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def solve(grid: Sequence[Sequence[int]]) -> List[List[int]]:
    """Return the smallest visible colored rectangle as a compact grid."""
    g = np.asarray(grid, dtype=np.int64)
    best_key: Tuple[int, int] | None = None
    best_color: int | None = None
    best_hw: Tuple[int, int] | None = None

    for color in sorted(set(g.ravel()) - {0}):
        ys, xs = np.where(g == color)
        height = len(np.unique(ys))
        width = len(np.unique(xs))
        key = (height * width, int(color))
        if best_key is None or key < best_key:
            best_key = key
            best_color = int(color)
            best_hw = (height, width)

    if best_color is None or best_hw is None:
        return []
    height, width = best_hw
    return np.full((height, width), best_color, dtype=np.int64).tolist()


def build_reference_solution(grid: np.ndarray) -> np.ndarray:
    """Pad solve() to the competition 30x30 canvas."""
    if grid.ndim == 4:
        g = grid[0].argmax(axis=0).astype(np.int64)
    elif grid.ndim == 3:
        g = grid.argmax(axis=0).astype(np.int64)
    else:
        g = np.asarray(grid, dtype=np.int64)

    crop = np.asarray(solve(g.tolist()), dtype=np.int64)
    out = np.zeros((H, W), dtype=np.int64)
    if crop.size:
        ch, cw = crop.shape
        out[:ch, :cw] = crop
    return out


def _grid_to_onehot(grid: Sequence[Sequence[int]]) -> np.ndarray:
    out = np.zeros((1, C, H, W), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    flat = onehot.reshape(C, H, W)
    active = flat > 0.0
    out = flat.argmax(axis=0).astype(np.int64)
    out[~active.any(axis=0)] = 0
    return out


def _init(inits: List[onnx.TensorProto], arr: np.ndarray, name: str) -> str:
    inits.append(numpy_helper.from_array(arr, name=name))
    return name


def _f32(inits: List[onnx.TensorProto], vals: Sequence[float], name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def build_onnx_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    half = _f32(inits, [0.5], "half")
    big = _f32(inits, [BIG], "big")
    valid_color = _init(
        inits,
        np.asarray([False] + [True] * NC, dtype=np.bool_).reshape(1, C, 1, 1),
        "valid_color",
    )

    nodes.extend(
        [
            helper.make_node("ReduceMax", [IN_NAME], ["rowocc"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", [IN_NAME], ["colocc"], axes=[2], keepdims=1),
            helper.make_node("ReduceSum", ["rowocc"], ["height_all"], axes=[2], keepdims=1),
            helper.make_node("ReduceSum", ["colocc"], ["width_all"], axes=[3], keepdims=1),
            helper.make_node("Mul", ["height_all", "width_all"], ["area"]),
            helper.make_node("Greater", ["height_all", half], ["has_any"]),
            helper.make_node("And", ["has_any", valid_color], ["has"]),
            helper.make_node("Where", ["has", "area", big], ["key"]),
            helper.make_node("ArgMin", ["key"], ["win"], axis=1, keepdims=1),
            helper.make_node("Gather", ["height_all", "win"], ["height"], axis=1),
            helper.make_node("Gather", ["width_all", "win"], ["width"], axis=1),
        ]
    )

    out_rows = _init(
        inits, np.arange(H, dtype=np.float32).reshape(1, 1, H, 1), "out_rows"
    )
    out_cols = _init(
        inits, np.arange(W, dtype=np.float32).reshape(1, 1, 1, W), "out_cols"
    )
    colors = _init(
        inits, np.arange(C, dtype=np.int64).reshape(1, C, 1, 1), "colors"
    )
    sq_all = [0, 1, 2, 3]

    nodes.extend(
        [
            helper.make_node("Squeeze", ["height"], ["height_s"], axes=sq_all),
            helper.make_node("Squeeze", ["width"], ["width_s"], axes=sq_all),
            helper.make_node("Less", [out_rows, "height_s"], ["inside_y"]),
            helper.make_node("Less", [out_cols, "width_s"], ["inside_x"]),
            helper.make_node("And", ["inside_y", "inside_x"], ["inside"]),
            helper.make_node("Cast", ["inside"], ["inside_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["win"], ["win_i"], to=TensorProto.INT64),
            helper.make_node("Squeeze", ["win_i"], ["win_s"], axes=sq_all),
            helper.make_node("Equal", [colors, "win_s"], ["color_match"]),
            helper.make_node("Cast", ["color_match"], ["color_match_f"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["color_match_f", "inside_f"], [OUT_NAME]),
        ]
    )

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, [1, C, H, W])
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, [1, C, H, W])
    graph = helper.make_graph(nodes, "task049_occ_footprint", [x_info], [y_info], inits)
    model = helper.make_model(
        graph,
        producer_name="task049",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = "Smallest occupied-row/column footprint rectangle"
    onnx.checker.check_model(model)
    return model


def save_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    model = build_onnx_model()
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return model


def model_stats(model: onnx.ModelProto) -> Dict[str, int]:
    params = sum(int(np.prod(list(t.dims))) for t in model.graph.initializer)
    return {
        "nodes": len(model.graph.node),
        "params": params,
        "inits": len(model.graph.initializer),
    }


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        model.SerializeToString(), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def test() -> None:
    model = build_onnx_model()
    stats = model_stats(model)
    print(f"nodes={stats['nodes']} params={stats['params']} inits={stats['inits']}")

    if TASK_JSON.is_file():
        with TASK_JSON.open(encoding="utf-8") as fh:
            data = json.load(fh)
        bad = 0
        for split in ("train", "test", "arc-gen"):
            passed = 0
            total = 0
            for ex in data[split]:
                total += 1
                onehot = _grid_to_onehot(ex["input"])
                pred_g = _onehot_to_grid(_run_onnx(model, onehot)[0])
                exp = np.asarray(ex["output"], dtype=np.int64)
                if np.array_equal(pred_g[: exp.shape[0], : exp.shape[1]], exp):
                    passed += 1
                else:
                    bad += 1
            print(f"{split}: {passed}/{total}")
        print(f"task049.json: {'PASS' if bad == 0 else f'FAIL ({bad} wrong)'}")


def main() -> None:
    save_model()
    test()
    print(f"saved {BEST_PATH}")


if __name__ == "__main__":
    main()
