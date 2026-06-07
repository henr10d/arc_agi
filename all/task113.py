"""Minimal ONNX for ARC task113 using Kaggle one-hot I/O.

Task rule (input and output are always 10 rows tall, 2 to 10 columns wide):
- The top of the grid holds a horizontal colored stripe pattern that occupies at
  most the first four rows; everything below is background color 0. In the
  provided splits the stripe height is 2, 3, or 4 rows.
- The output keeps the input unchanged and additionally mirrors the nonzero top
  stripe rows to the bottom of the grid in reversed vertical order.
  Example: input row0=red, row1=cyan -> output row8=cyan, row9=red.

ONNX approach (cost = internal tensor bytes + params, lower is better):
The transform is a pure permutation of the 30 height rows, so it is a single
Gather along axis 2 (height):
    rows 0..4   -> input rows 0..4    (top kept unchanged)
    rows 5..9   -> input rows 4..0    (stripe mirrored to the bottom)
    rows 10..29 -> input rows 10..29  (already all-zero padding -> stays zero)
Because the stripe never reaches past row 3, the 5+5 split reproduces the mirror
rule exactly, and the padding rows simply gather already-zero rows.

The single Gather node writes straight into ``output``. NeuroGolf excludes the
``input``/``output`` tensors from memory scoring, so this graph materializes no
internal tensor at all: memory = 0. The only cost is the 30-element index
initializer (30 params), giving cost = 30. Slice/Concat/Pad and dynamic-index
variants were tested, but their internal tensors cost more than this initializer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))
BEST_PATH = OUT_DIR / "task113.onnx"
DATA_PATH = ROOT / "data" / "task113.json"

C = 10           # color channels
H = W = 30       # competition I/O grid
GRID_H = 10      # actual task height
GRID_W = 10      # max actual task width across all splits
HALF = 5         # symmetric height split (stripes never pass row 3)
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
IR_VERSION = 10
OPSET = 10


def _i64(inits: List[onnx.TensorProto], vals: List[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))
    return name


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference transform on a raw 10xW integer grid."""
    grid = np.asarray(grid)
    top5 = grid[0:HALF, :]
    return np.concatenate([top5, top5[::-1, :]], axis=0)


def build_gather_model() -> onnx.ModelProto:
    """Best variant: a single Gather along the height axis -> zero internal memory.

    The transform is a fixed permutation of the 30 height rows:
        rows 0..4  -> input rows 0..4   (top kept unchanged)
        rows 5..9  -> input rows 4..0   (stripe mirrored to the bottom)
        rows 10..29-> input rows 10..29 (already all-zero padding -> stays zero)
    Because the single Gather node writes directly into ``output`` (and ``input``
    /``output`` are excluded from memory scoring), no internal tensor is ever
    materialized. The only cost is the 30-element index initializer (30 params).
    """
    inits: List[onnx.TensorProto] = []
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    perm = list(range(HALF)) + list(range(HALF - 1, -1, -1)) + list(range(GRID_H, H))
    idx = _i64(inits, perm, "i")

    nodes = [helper.make_node("Gather", [IN_NAME, idx], [OUT_NAME], axis=2)]

    graph = helper.make_graph(nodes, "g_gather", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_float_concat_model() -> onnx.ModelProto:
    """Opset-10 float variant: same logic but float Slice/Concat + float Pad.

    Kept for comparison; float intermediates make this strictly costlier than the
    bool variant, but it stays at opset 10 with a float output for stricter setups.
    """
    inits: List[onnx.TensorProto] = []
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    top_s = _i64(inits, [0, 0], "ts")
    top_e = _i64(inits, [HALF, GRID_W], "te")
    hw_ax = _i64(inits, [2, 3], "hw")
    rev_s = _i64(inits, [HALF - 1], "rs")
    rev_e = _i64(inits, [-HALF - 1], "re")
    rev_a = _i64(inits, [2], "ra")
    rev_t = _i64(inits, [-1], "rt")

    nodes = [
        helper.make_node("Slice", [IN_NAME, top_s, top_e, hw_ax], ["top5"]),
        helper.make_node("Slice", ["top5", rev_s, rev_e, rev_a, rev_t], ["flip5"]),
        helper.make_node("Concat", ["top5", "flip5"], ["grid"], axis=2),
        helper.make_node(
            "Pad",
            ["grid"],
            [OUT_NAME],
            pads=[0, 0, 0, 0, 0, 0, H - GRID_H, W - GRID_W],
        ),
    ]

    graph = helper.make_graph(nodes, "g_float_concat", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_onnx_model() -> onnx.ModelProto:
    return build_gather_model()


# ---------------------------------------------------------------------------
# Validation + scoring helpers
# ---------------------------------------------------------------------------

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
    return _grid_to_onehot(grid)


def _strict_onehot_matches(pred: np.ndarray, expected: np.ndarray) -> bool:
    return pred.shape == expected.shape and np.array_equal(pred > 0, expected > 0)


def validate_model(model: onnx.ModelProto) -> tuple[bool, str]:
    try:
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        return False, f"ORT load failed: {exc}"

    if DATA_PATH.is_file():
        with DATA_PATH.open(encoding="utf-8") as fh:
            data = json.load(fh)
        for split in ("train", "test", "arc-gen"):
            for idx, ex in enumerate(data.get(split, [])):
                x = np.asarray(ex["input"], dtype=np.int64)
                exp = np.asarray(ex["output"], dtype=np.int64)
                if max(x.shape) > 30:
                    continue
                pred = sess.run([OUT_NAME], {IN_NAME: _grid_to_onehot(x)})[0]
                if not _strict_onehot_matches(pred, _expected_onehot(exp)):
                    return False, f"{split}#{idx} mismatch"

    rng = np.random.default_rng(0)
    for _ in range(200):
        h, w = 10, int(rng.integers(3, GRID_W + 1))
        k = int(rng.integers(1, HALF))
        g = np.zeros((h, w), dtype=np.int64)
        for r in range(k):
            g[r, :] = int(rng.integers(1, C))
        pred = sess.run([OUT_NAME], {IN_NAME: _grid_to_onehot(g)})[0]
        if not _strict_onehot_matches(pred, _expected_onehot(solve(g))):
            return False, f"random mismatch for {g.tolist()}"
    return True, "PASS"


def official_score(path: Path) -> Dict[str, Any]:
    from score_model import score_file

    return score_file(path)


def save_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    model = build_onnx_model()
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return model


def variant_builders() -> Dict[str, Callable[[], onnx.ModelProto]]:
    return {
        "gather": build_gather_model,
        "float_concat": build_float_concat_model,
    }


def main() -> None:
    results: List[Dict[str, Any]] = []
    for name, builder in variant_builders().items():
        path = OUT_DIR / f"task113_variant_{name}.onnx"
        row: Dict[str, Any] = {"name": name, "path": path}
        model = builder()
        onnx.save(model, str(path))
        ok, msg = validate_model(model)
        row["valid"] = ok
        row["validation"] = msg
        scored = official_score(path)
        row["memory"] = scored.get("memory")
        row["params"] = scored.get("params")
        row["cost"] = scored.get("cost")
        row["score"] = scored.get("score")
        row["error"] = scored.get("error")
        results.append(row)

    print(f"{'variant':<14}{'pass':<6}{'memory':>8}{'params':>8}{'cost':>8}{'score':>10}")
    for row in results:
        score = row.get("score")
        score_str = f"{score:.6f}" if isinstance(score, float) else "INVALID"
        print(
            f"{row['name']:<14}{str(row.get('valid')):<6}"
            f"{str(row.get('memory')):>8}{str(row.get('params')):>8}"
            f"{str(row.get('cost')):>8}{score_str:>10}"
        )
        if not row.get("valid") or row.get("error"):
            print(f"  note: {row.get('validation')} {row.get('error') or ''}".rstrip())

    valid = [r for r in results if r.get("valid") and r.get("cost") is not None]
    best = min(valid, key=lambda r: int(r["cost"]))
    onnx.save(onnx.load(str(best["path"])), str(BEST_PATH))
    print(f"\nsaved best variant '{best['name']}' (cost={best['cost']}, "
          f"score={best['score']:.6f}) to {BEST_PATH}")


if __name__ == "__main__":
    main()
