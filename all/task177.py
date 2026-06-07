"""Minimal ONNX for ARC task177: crop the object and mirror it horizontally.

Task rule: find the bounding box of all non-black pixels, crop that rectangle,
then output the crop flipped left-to-right at the top-left of the output grid.
The object is a filled rectangle of two non-background colors; both colors are
preserved exactly by the mirror. Cells outside the cropped output are padding.

ONNX: compute a compact foreground mask with a 1x1 channel convolution, locate
the top row and rightmost column, gather an 8x8 int64 color crop from ArgMax
colors, rebuild the small one-hot crop, mask cells outside the original bbox,
and pad the compact result to the required 30x30 output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task177"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
OH = OW = 8
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver: nonzero bbox crop, mirrored horizontally."""
    g = np.asarray(grid, dtype=np.int64)
    ys, xs = np.where(g != 0)
    if len(ys) == 0:
        return np.zeros((0, 0), dtype=np.int64)
    crop = g[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return np.fliplr(crop)


def _bbox_crop(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    ys, xs = np.where(g != 0)
    return g[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _fill_holes(grid: np.ndarray) -> np.ndarray:
    crop = _bbox_crop(grid).copy()
    colors, counts = np.unique(crop[crop != 0], return_counts=True)
    dominant = int(colors[np.argmax(counts)])
    crop[crop == 0] = dominant
    return crop


def _rasterize(grid: np.ndarray) -> np.ndarray:
    crop = _bbox_crop(grid).copy()
    colors, counts = np.unique(crop[crop != 0], return_counts=True)
    dominant = int(colors[np.argmax(counts)])
    accent = int(colors[np.argmin(counts)])
    return np.where(crop == accent, accent, dominant).astype(np.int64)


def _accent_projection(grid: np.ndarray) -> np.ndarray:
    rast = _rasterize(grid)
    colors, counts = np.unique(rast, return_counts=True)
    dominant = int(colors[np.argmax(counts)])
    accent = int(colors[np.argmin(counts)])
    out = np.full_like(rast, dominant)
    out[np.any(rast == accent, axis=1), :] = dominant
    out[:, np.any(rast == accent, axis=0)] = accent
    return out


def _filled_hull(grid: np.ndarray) -> np.ndarray:
    return _rasterize(grid)


def evaluate_hypotheses() -> Dict[str, int]:
    """Return train matches for the requested alternatives plus the selected rule."""
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    hypotheses: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "A_pure_crop": _bbox_crop,
        "B_crop_fill_background_holes": _fill_holes,
        "C_object_rasterization": _rasterize,
        "D_accent_projection": _accent_projection,
        "E_filled_hull": _filled_hull,
        "selected_crop_then_horizontal_flip": solve,
    }
    scores: Dict[str, int] = {}
    for name, fn in hypotheses.items():
        scores[name] = sum(
            int(np.array_equal(fn(np.array(ex["input"], dtype=np.int64)), np.array(ex["output"], dtype=np.int64)))
            for ex in data["train"]
        )
    return scores


def _grid_to_onehot(grid: List[List[int]], mode: str = "input") -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    fg_weights = _f32(inits, np.r_[0.0, np.ones(9, dtype=np.float32)].reshape(1, C, 1, 1), "fg_weights")
    row_offsets = _i64(inits, np.arange(OH, dtype=np.int64), "row_offsets")
    col_offsets = _i64(inits, np.arange(OW, dtype=np.int64), "col_offsets")
    col_weights = _f32(inits, np.arange(W, dtype=np.float32).reshape(1, 1, 1, W), "col_weights")
    colors = _i64(inits, np.arange(C, dtype=np.int64).reshape(1, C, 1, 1), "colors")
    out_pads = [0, 0, 0, 0, 0, 0, H - OH, W - OW]

    nodes.extend(
        [
            helper.make_node("Conv", [IN_NAME, fg_weights], ["fg_any"]),
            helper.make_node("ReduceMax", ["fg_any"], ["row_any"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["fg_any"], ["col_any"], axes=[2], keepdims=1),
            helper.make_node("ArgMax", ["row_any"], ["minr_4d"], axis=2, keepdims=1),
            helper.make_node("Squeeze", ["minr_4d"], ["minr"], axes=[0, 1, 2, 3]),
            helper.make_node("Mul", ["col_any", col_weights], ["col_weighted"]),
            helper.make_node("ArgMax", ["col_weighted"], ["maxc_4d"], axis=3, keepdims=1),
            helper.make_node("Squeeze", ["maxc_4d"], ["maxc"], axes=[0, 1, 2, 3]),
            helper.make_node("Add", ["minr", row_offsets], ["row_idx"]),
            helper.make_node("Sub", ["maxc", col_offsets], ["col_idx"]),
            helper.make_node("ArgMax", [IN_NAME], ["color_grid"], axis=1, keepdims=0),
            helper.make_node("Gather", ["color_grid", "row_idx"], ["row_gather"], axis=1),
            helper.make_node("Gather", ["row_gather", "col_idx"], ["crop8"], axis=2),
            helper.make_node("Equal", ["crop8", colors], ["crop8_onehot"]),
            helper.make_node("Gather", ["row_any", "row_idx"], ["row_valid"], axis=2),
            helper.make_node("Gather", ["col_any", "col_idx"], ["col_valid"], axis=3),
            helper.make_node("Cast", ["row_valid"], ["row_valid_bool"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["col_valid"], ["col_valid_bool"], to=TensorProto.BOOL),
            helper.make_node("And", ["row_valid_bool", "col_valid_bool"], ["valid"]),
            helper.make_node("And", ["crop8_onehot", "valid"], ["out8_bool"]),
            helper.make_node("Cast", ["out8_bool"], ["out8"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out8"], [OUT_NAME], pads=out_pads),
        ]
    )

    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for i, ex in enumerate(data[split]):
            out = np.array(ex["output"], dtype=np.int64)
            oh = _grid_to_onehot(ex["input"])
            pred = _onehot_to_grid(_run_onnx(model, oh))[: out.shape[0], : out.shape[1]]
            if not np.array_equal(pred, out):
                print(f"mismatch {split} #{i}")
                print(pred)
                print(out)
                bad += 1
    return bad


def main() -> None:
    scores = evaluate_hypotheses()
    for name, count in scores.items():
        print(f"{name}: {count}/3 train")
    assert scores["selected_crop_then_horizontal_flip"] == 3

    model = build_model()
    onnx.save(model, BEST_PATH)

    bad = validate_json(model)
    print(f"task177.json: {'PASS' if bad == 0 else f'FAIL ({bad} wrong)'}")

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
