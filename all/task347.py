"""Minimal ONNX for ARC task347: overlay two colored 3x3 halves.

Task rule: the input is a 3x6 grid made from two 3x3 panels. A cell in the
3x3 output is magenta color 6 when the corresponding left-panel cell is yellow
color 4, or the corresponding right-panel cell is green color 3. Otherwise the
output cell is black color 0.

The ONNX graph reads only those two color channels, ORs their compact 3x3 bool
masks, uses a broadcast Where against two tiny 7-channel float templates, and
pads the remaining channels plus spatial extent to the required [1, 10, 30, 30]
output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
TASK_ID = "task347"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
CORE = 3
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10

BLACK = 0
GREEN = 3
YELLOW = 4
MAGENTA = 6


def _i64(inits: List[onnx.TensorProto], vals: Iterable[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(list(vals), dtype=np.int64), name=name))
    return name


def solve(grid: np.ndarray | List[List[int]]) -> np.ndarray:
    """Reference implementation for decoded ARC grids."""
    x = np.asarray(grid, dtype=np.int64)
    left_yellow = x[:CORE, :CORE] == YELLOW
    right_green = x[:CORE, CORE : CORE * 2] == GREEN
    return np.where(left_yellow | right_green, MAGENTA, BLACK).astype(np.int64)


def _grid_to_onehot(grid: np.ndarray | List[List[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            color = int(arr[r, c])
            if 0 <= color < C:
                out[0, color, r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return np.asarray(onehot).reshape(C, H, W).argmax(axis=0).astype(np.int64)


def build_onnx_model() -> onnx.ModelProto:
    """Build the compact strict-color projection graph."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, [1, 2, 3], "axes")
    yellow_st = _i64(inits, [YELLOW, 0, 0], "yellow_st")
    yellow_en = _i64(inits, [YELLOW + 1, CORE, CORE], "yellow_en")
    green_st = _i64(inits, [GREEN, 0, CORE], "green_st")
    green_en = _i64(inits, [GREEN + 1, CORE, CORE * 2], "green_en")

    fg_template_arr = np.zeros((7, 1, 1), dtype=np.float32)
    fg_template_arr[MAGENTA, 0, 0] = 1.0
    bg_template_arr = np.zeros((7, 1, 1), dtype=np.float32)
    bg_template_arr[BLACK, 0, 0] = 1.0
    inits.append(numpy_helper.from_array(fg_template_arr, name="fg_template"))
    inits.append(numpy_helper.from_array(bg_template_arr, name="bg_template"))

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, yellow_st, yellow_en, axes], ["left_yellow"]),
            helper.make_node("Slice", [IN_NAME, green_st, green_en, axes], ["right_green"]),
            helper.make_node("Cast", ["left_yellow"], ["left_mask"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["right_green"], ["right_mask"], to=TensorProto.BOOL),
            helper.make_node("Or", ["left_mask", "right_mask"], ["fg"]),
            helper.make_node("Where", ["fg", "fg_template", "bg_template"], ["out7"]),
            helper.make_node("Pad", ["out7"], [OUT_NAME], pads=[0, 0, 0, 0, 0, C - 7, H - CORE, W - CORE]),
        ]
    )

    graph = helper.make_graph(nodes, "task347", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def save_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    model = build_onnx_model()
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return model


def model_stats(model: onnx.ModelProto) -> Dict[str, int]:
    params = sum(int(np.prod(list(t.dims))) for t in model.graph.initializer)
    return {"nodes": len(model.graph.node), "params": params, "inits": len(model.graph.initializer)}


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]
    except ImportError:
        from onnx import reference

        sess = reference.ReferenceEvaluator(model)
        return sess.run(None, {IN_NAME: x.astype(np.float32)})[0]


def _strict_rule(grid: np.ndarray) -> np.ndarray:
    return solve(grid)


def _any_nonblack_rule(grid: np.ndarray) -> np.ndarray:
    x = np.asarray(grid, dtype=np.int64)
    mask = (x[:CORE, :CORE] != BLACK) | (x[:CORE, CORE : CORE * 2] != BLACK)
    return np.where(mask, MAGENTA, BLACK).astype(np.int64)


def _swapped_rule(grid: np.ndarray) -> np.ndarray:
    x = np.asarray(grid, dtype=np.int64)
    mask = (x[:CORE, :CORE] == GREEN) | (x[:CORE, CORE : CORE * 2] == YELLOW)
    return np.where(mask, MAGENTA, BLACK).astype(np.int64)


def _check_rule_variants(data: dict) -> Dict[str, bool]:
    variants = {
        "strict yellow-left OR green-right": _strict_rule,
        "any nonblack in either half": _any_nonblack_rule,
        "swapped green-left OR yellow-right": _swapped_rule,
    }
    result: Dict[str, bool] = {}
    train = data.get("train", [])
    for name, fn in variants.items():
        result[name] = all(
            np.array_equal(fn(np.asarray(ex["input"], dtype=np.int64)), np.asarray(ex["output"], dtype=np.int64))
            for ex in train
        )
    return result


def test() -> None:
    model = build_onnx_model()
    stats = model_stats(model)
    print(f"nodes={stats['nodes']} params={stats['params']} inits={stats['inits']}")

    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    print("train rule variants:")
    for name, ok in _check_rule_variants(data).items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    bad = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data.get(split, []):
            total += 1
            inp = np.asarray(ex["input"], dtype=np.int64)
            exp = np.asarray(ex["output"], dtype=np.int64)
            ref = solve(inp)
            pred_onehot = _run_onnx(model, _grid_to_onehot(inp)) > 0
            exp_onehot = _grid_to_onehot(exp) > 0
            pred = _onehot_to_grid(pred_onehot[0])[:CORE, :CORE]
            if not np.array_equal(ref, exp) or not np.array_equal(pred_onehot, exp_onehot):
                bad += 1
                print(f"{split} example failed")
                print("expected:")
                print(exp)
                print("pred:")
                print(pred)
    print(f"{TASK_ID}.json: {'PASS' if bad == 0 else f'FAIL ({bad}/{total} wrong)'}")
    if bad:
        raise SystemExit(1)


def main() -> None:
    save_model()
    test()
    print(f"saved {BEST_PATH}")


if __name__ == "__main__":
    main()
