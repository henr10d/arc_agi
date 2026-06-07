"""Minimal ONNX for ARC task045: fill rows when left/right borders match.

Task rule: grids are 10×10. Colored cells appear only on the left (col 0) and
right (col 9) borders, on odd rows 1/3/5/7/9 in all provided splits. For each
row r, if border colors match and are nonzero, fill the entire row with that
color; otherwise leave the row unchanged.

ONNX: slice only the two border columns, compare nonzero one-hot channels
directly, expand matching colors across the eight interior columns, assemble the
odd-row bool one-hot result with Concat, gather in fixed all-background even
rows, cast once to float, then pad to 30×30.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

BEST_PATH = OUT_DIR / "task045.onnx"
DATA_PATH = ROOT / "data" / "task045.json"

C = 10
G = 10
RG = 5
H = W = 30
PAD = H - G
PAD_ATTR = [0, 0, 0, 0, 0, 0, PAD, PAD]
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference: fill row when left and right border colors match and are nonzero."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    w = g.shape[1]
    for r in range(g.shape[0]):
        left, right = int(g[r, 0]), int(g[r, w - 1])
        if left != 0 and left == right:
            out[r, :] = left
    return out


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> None:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> None:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.float32), name=name))


def _slice4(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    source: str,
    output: str,
    starts: List[int],
    ends: List[int],
    steps: List[int] | None = None,
) -> None:
    _i64(inits, starts, f"{output}_s")
    _i64(inits, ends, f"{output}_e")
    inputs = [source, f"{output}_s", f"{output}_e", "ax4"]
    if steps is not None:
        _i64(inits, steps, f"{output}_step")
        inputs.append(f"{output}_step")
    nodes.append(helper.make_node("Slice", inputs, [output]))


def _or_reduce_channels(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    source: str,
    prefix: str,
    channels: int,
    rows: int,
) -> str:
    del inits, rows
    names = [f"{prefix}_ch{idx}" for idx in range(channels)]
    nodes.append(helper.make_node("Split", [source], names, axis=1, split=[1] * channels))
    prev = names[0]
    for idx in range(channels):
        if idx:
            out = f"{prefix}_or{idx}"
            nodes.append(helper.make_node("Or", [prev, names[idx]], [out]))
            prev = out
    return prev


def _grid_to_onehot(grid: np.ndarray | List[List[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            color = int(arr[r, c])
            if 0 <= color < C:
                out[0, color, r, c] = 1.0
    return out


def _expected_onehot(grid: np.ndarray | List[List[int]]) -> np.ndarray:
    return _grid_to_onehot(np.asarray(grid, dtype=np.int64))


def build_model(opset: int = OPSET) -> onnx.ModelProto:
    """Best-cost graph: border-only bool assembly + single final Cast."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    _i64(inits, [0, 1, 2, 3], "ax4")
    _f32(inits, [0.0], "zero")
    _i64(inits, [1, 9, RG, 8], "mid9_sh")
    _i64(inits, [1, 1, RG, 8], "mid1_sh")
    _i64(inits, [0, 1, 0, 2, 0, 3, 0, 4, 0, 5], "row_idx")
    zero_row = np.zeros((1, C, 1, G), dtype=bool)
    zero_row[:, 0, :, :] = True
    inits.append(numpy_helper.from_array(zero_row, name="zero_row"))

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    odd_steps = [1, 1, 2, 1]
    _slice4(nodes, inits, IN_NAME, "left_nz", [0, 1, 1, 0], [1, C, G, 1], odd_steps)
    _slice4(nodes, inits, IN_NAME, "right_nz", [0, 1, 1, 9], [1, C, G, G], odd_steps)
    _slice4(nodes, inits, IN_NAME, "left_zero_f", [0, 0, 1, 0], [1, 1, G, 1], odd_steps)
    _slice4(nodes, inits, IN_NAME, "right_zero_f", [0, 0, 1, 9], [1, 1, G, G], odd_steps)
    nodes.extend(
        [
            helper.make_node("Greater", ["left_nz", "zero"], ["left"]),
            helper.make_node("Greater", ["right_nz", "zero"], ["right"]),
            helper.make_node("And", ["left", "right"], ["fill_color_col"]),
            helper.make_node("Expand", ["fill_color_col", "mid9_sh"], ["fill_color_mid"]),
            helper.make_node("Concat", ["left", "fill_color_mid", "right"], ["out_nz"], axis=3),
        ]
    )

    fill_any_col = _or_reduce_channels(nodes, inits, "fill_color_col", "fill_any", 9, RG)
    nodes.extend(
        [
            helper.make_node("Expand", [fill_any_col, "mid1_sh"], ["fill_any_mid"]),
            helper.make_node("Not", ["fill_any_mid"], ["zero_mid"]),
            helper.make_node("Greater", ["left_zero_f", "zero"], ["zero_left"]),
            helper.make_node("Greater", ["right_zero_f", "zero"], ["zero_right"]),
            helper.make_node("Concat", ["zero_left", "zero_mid", "zero_right"], ["out_zero"], axis=3),
            helper.make_node("Concat", ["out_zero", "out_nz"], ["odd_rows"], axis=1),
            helper.make_node("Concat", ["zero_row", "odd_rows"], ["row_bank"], axis=2),
            helper.make_node("Gather", ["row_bank", "row_idx"], ["out10b"], axis=2),
            helper.make_node("Cast", ["out10b"], ["out10"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out10"], [OUT_NAME], mode="constant", pads=PAD_ATTR),
        ]
    )

    graph = helper.make_graph(nodes, "task045", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    onnx.checker.check_model(model)
    return model


def build_argmax_onehot(opset: int = OPSET) -> onnx.ModelProto:
    """Alternate: OneHot decode (higher memory, kept for experiments)."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    _i64(inits, [0, 1, 2, 3], "ax4")
    _i64(inits, [0, 0, 0, 0], "z0")
    _i64(inits, [0, 0, 0, 0], "st0")
    _i64(inits, [1, C, G, G], "e_core")
    _i64(inits, [1, C, G, 1], "e_col1")
    _i64(inits, [0, 0, 0, 9], "c9")
    _i64(inits, [0], "iz")
    _i64(inits, [C], "depth")
    _i64(inits, [1, 1, G, 1], "r1")
    _i64(inits, [1, 1, G, G], "id_sh")
    inits.append(numpy_helper.from_array(np.array([0.0, 1.0], dtype=np.float32), name="oh_vals"))

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, "z0", "e_core", "ax4"], ["core"]),
            helper.make_node("ArgMax", ["core"], ["ids"], axis=1, keepdims=1),
            helper.make_node("Slice", ["ids", "st0", "e_col1", "ax4"], ["lid"]),
            helper.make_node("Slice", ["ids", "c9", "e_core", "ax4"], ["rid"]),
            helper.make_node("Equal", ["lid", "rid"], ["same"]),
            helper.make_node("Greater", ["lid", "iz"], ["lnz"]),
            helper.make_node("And", ["same", "lnz"], ["fill"]),
            helper.make_node("Reshape", ["fill", "r1"], ["fill1"]),
            helper.make_node("Expand", ["fill1", "id_sh"], ["fillw"]),
            helper.make_node("Expand", ["lid", "id_sh"], ["body"]),
            helper.make_node("Where", ["fillw", "body", "ids"], ["yids"]),
            helper.make_node("OneHot", ["yids", "depth", "oh_vals"], ["oh5"], axis=-1),
            helper.make_node("Squeeze", ["oh5"], ["oh4"], axes=[1]),
            helper.make_node("Transpose", ["oh4"], ["out10"], perm=[0, 3, 1, 2]),
            helper.make_node("Pad", ["out10"], [OUT_NAME], mode="constant", pads=PAD_ATTR),
        ]
    )

    graph = helper.make_graph(nodes, "argmax_onehot", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    onnx.checker.check_model(model)
    return model


def _strict_onehot_matches(pred: np.ndarray, expected: np.ndarray) -> bool:
    return pred.shape == expected.shape and np.array_equal(pred > 0, expected > 0)


def _random_border_grid(rng: np.random.Generator) -> np.ndarray:
    g = np.zeros((G, G), dtype=np.int64)
    for r in range(1, G, 2):
        if rng.random() < 0.65:
            g[r, 0] = int(rng.integers(0, C))
        if rng.random() < 0.65:
            g[r, G - 1] = int(rng.integers(0, C))
    return g


def validate_model(model: onnx.ModelProto, *, random_cases: int = 300) -> Tuple[bool, str]:
    try:
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    except Exception as exc:
        return False, f"ORT load failed: {exc}"

    rng = np.random.default_rng(45)
    for _ in range(random_cases):
        x = _random_border_grid(rng)
        ref = solve(x)
        pred = sess.run([OUT_NAME], {IN_NAME: _grid_to_onehot(x)})[0]
        if not _strict_onehot_matches(pred, _expected_onehot(ref)):
            return False, f"random mismatch:\n{x}\nref:\n{ref}"

    if DATA_PATH.is_file():
        with DATA_PATH.open(encoding="utf-8") as fh:
            data = json.load(fh)
        for split in ("train", "test", "arc-gen"):
            for idx, ex in enumerate(data[split]):
                inp = np.asarray(ex["input"], dtype=np.int64)
                exp = np.asarray(ex["output"], dtype=np.int64)
                pred = sess.run([OUT_NAME], {IN_NAME: _grid_to_onehot(inp)})[0]
                if not _strict_onehot_matches(pred, _expected_onehot(exp)):
                    return False, f"{split}#{idx} mismatch"

    return True, "PASS"


def run_experiments() -> List[Dict[str, Any]]:
    builders: Dict[str, Callable[[], onnx.ModelProto]] = {
        "ids_concat": build_model,
        "argmax_onehot": build_argmax_onehot,
    }
    results: List[Dict[str, Any]] = []
    for name, builder in builders.items():
        path = OUT_DIR / f"task045_variant_{name}.onnx"
        row: Dict[str, Any] = {"name": name, "path": path, "valid": False}
        try:
            model = builder()
            onnx.save(model, str(path))
            ok, msg = validate_model(model)
            row.update(
                {
                    "valid": ok,
                    "validation": msg,
                    "opset": model.opset_import[0].version,
                    "nodes": len(model.graph.node),
                }
            )
            scored = score_file(path)
            row["memory"] = scored.get("memory")
            row["params"] = scored.get("params")
            row["cost"] = scored.get("cost")
            row["score"] = scored.get("score")
        except Exception as exc:
            row["validation"] = f"failed: {exc}"
        results.append(row)
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", action="store_true", help="compare ONNX variants")
    args = parser.parse_args()

    if args.experiment:
        results = run_experiments()
        print(f"\n{'variant':<16} {'pass':<5} {'memory':>8} {'params':>6} {'cost':>6} {'score':>8}")
        for row in results:
            sc = row.get("score")
            sc_s = f"{sc:.4f}" if isinstance(sc, float) else "INVALID"
            print(
                f"{row['name']:<16} {str(row.get('valid')):<5} "
                f"{str(row.get('memory', '-')):>8} {str(row.get('params', '-')):>6} "
                f"{str(row.get('cost', '-')):>6} {sc_s:>8}"
            )
        valid = [r for r in results if r.get("valid") and r.get("cost") is not None]
        if valid:
            best = min(valid, key=lambda r: int(r["cost"]))
            onnx.save(onnx.load(str(best["path"])), str(BEST_PATH))
            print(f"\nBest variant {best['name']} -> {BEST_PATH}")
        return

    model = build_model()
    onnx.save(model, str(BEST_PATH))
    ok, msg = validate_model(model)
    scored = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"validation: {msg}")
    print(f"nodes:    {len(model.graph.node)}")
    print(f"memory:   {scored['memory']}")
    print(f"params:   {scored['params']}")
    print(f"cost:     {scored['cost']}")
    print(f"score:    {scored['score']:.6f}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
