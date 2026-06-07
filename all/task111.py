"""Minimal ONNX for ARC task111 using Kaggle one-hot I/O.

Task rule: each 10x10 input contains several 3x3 colored patterns on black
background and one gray selector cell. The selected pattern is the 3x3 crop
whose top-left corner is one row below and one column left of the gray cell:
``input[gray_row + 1 : gray_row + 4, gray_col - 1 : gray_col + 2]``. The gray
cell is outside that crop, so the output is simply the selected 3x3 one-hot
patch, padded back to the competition's 30x30 output tensor.

ONNX approach: crop the gray channel to the selector area (rows 0..6 and
columns 1..8), reduce it to row and column masks, ArgMax the gray position,
build dynamic Slice starts for the selected patch, then Pad the 3x3 patch
directly to ``output``. The expensive full output tensor is excluded from
NeuroGolf memory scoring.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

BEST_PATH = OUT_DIR / "task111.onnx"
DATA_PATH = ROOT / "data" / "task111.json"

IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
OPSET = 10


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def solve(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    """Reference transform on a raw 10x10 integer grid."""
    arr = np.asarray(grid, dtype=np.int64)
    gray = np.argwhere(arr == 5)
    if len(gray) != 1:
        raise ValueError("task111 expects exactly one gray selector cell")
    row, col = map(int, gray[0])
    return arr[row + 1 : row + 4, col - 1 : col + 2]


def _base_io() -> tuple[onnx.ValueInfoProto, onnx.ValueInfoProto]:
    return (
        helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE),
        helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE),
    )


def build_rowcol_gather_model() -> onnx.ModelProto:
    """Best measured variant: dynamic row/column Gather with tiny offsets."""
    inits: list[onnx.TensorProto] = []
    x_info, y_info = _base_io()

    gray_starts = _i64(inits, [5, 0, 0], "gs")
    gray_ends = _i64(inits, [6, 10, 10], "ge")
    gray_axes = _i64(inits, [1, 2, 3], "ga")
    row_offsets = _i64(inits, [1, 2, 3], "ro")
    col_offsets = _i64(inits, [-1, 0, 1], "co")

    nodes = [
        helper.make_node("Slice", [IN_NAME, gray_starts, gray_ends, gray_axes], ["gray"]),
        helper.make_node("ReduceMax", ["gray"], ["row_mask"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["gray"], ["col_mask"], axes=[2], keepdims=1),
        helper.make_node("ArgMax", ["row_mask"], ["row_i4"], axis=2, keepdims=1),
        helper.make_node("ArgMax", ["col_mask"], ["col_i4"], axis=3, keepdims=1),
        helper.make_node("Squeeze", ["row_i4"], ["row_i"], axes=[0, 1, 2, 3]),
        helper.make_node("Squeeze", ["col_i4"], ["col_i"], axes=[0, 1, 2, 3]),
        helper.make_node("Add", ["row_i", row_offsets], ["rows"]),
        helper.make_node("Add", ["col_i", col_offsets], ["cols"]),
        helper.make_node("Gather", [IN_NAME, "rows"], ["row_patch"], axis=2),
        helper.make_node("Gather", ["row_patch", "cols"], ["patch"], axis=3),
        helper.make_node("Pad", ["patch"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, 27, 27]),
    ]

    graph = helper.make_graph(nodes, "task111_rowcol_gather", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_flat_table_model() -> onnx.ModelProto:
    """Comparison variant: table maps gray flat index to nine flat patch cells."""
    inits: list[onnx.TensorProto] = []
    x_info, y_info = _base_io()

    gray_starts = _i64(inits, [5, 0, 0], "gs")
    gray_ends = _i64(inits, [6, 10, 10], "ge")
    gray_axes = _i64(inits, [1, 2, 3], "ga")
    flat100 = _i64(inits, [100], "f100")
    flat900 = _i64(inits, [1, 10, 900], "f900")
    patch_shape = _i64(inits, [1, 10, 3, 3], "ps")

    table = np.zeros((100, 9), dtype=np.int64)
    for gr in range(10):
        for gc in range(10):
            vals: list[int] = []
            for dr in range(1, 4):
                for dc in range(-1, 2):
                    rr = min(max(gr + dr, 0), 29)
                    cc = min(max(gc + dc, 0), 29)
                    vals.append(rr * 30 + cc)
            table[gr * 10 + gc] = vals
    patch_idx_table = _i64(inits, table, "pt")

    nodes = [
        helper.make_node("Slice", [IN_NAME, gray_starts, gray_ends, gray_axes], ["gray"]),
        helper.make_node("Reshape", ["gray", flat100], ["gray_flat"]),
        helper.make_node("ArgMax", ["gray_flat"], ["gray_i1"], axis=0, keepdims=1),
        helper.make_node("Squeeze", ["gray_i1"], ["gray_i"], axes=[0]),
        helper.make_node("Gather", [patch_idx_table, "gray_i"], ["patch_idx"], axis=0),
        helper.make_node("Reshape", [IN_NAME, flat900], ["input_flat"]),
        helper.make_node("Gather", ["input_flat", "patch_idx"], ["patch_flat"], axis=2),
        helper.make_node("Reshape", ["patch_flat", patch_shape], ["patch"]),
        helper.make_node("Pad", ["patch"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, 27, 27]),
    ]

    graph = helper.make_graph(nodes, "task111_flat_table", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_dynamic_slice_model() -> onnx.ModelProto:
    """Comparison variant: dynamic Slice directly to the 3x3 patch.

    This is theoretically cheaper than row/column Gather, but ONNX shape
    inference cannot always prove dynamic Slice output dimensions. Explicit
    value_info entries keep the scoring mirror able to measure the graph.
    """
    inits: list[onnx.TensorProto] = []
    x_info, y_info = _base_io()

    gray_starts = _i64(inits, [5, 0, 0], "gs")
    gray_ends = _i64(inits, [6, 10, 10], "ge")
    gray_axes = _i64(inits, [1, 2, 3], "ga")
    zero = _i64(inits, [0], "z")
    one = _i64(inits, np.array(1, dtype=np.int64), "one")
    neg_one = _i64(inits, np.array(-1, dtype=np.int64), "neg")
    extents = _i64(inits, [1, 10, 3, 3], "ex")

    nodes = [
        helper.make_node("Slice", [IN_NAME, gray_starts, gray_ends, gray_axes], ["gray"]),
        helper.make_node("ReduceMax", ["gray"], ["row_mask"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["gray"], ["col_mask"], axes=[2], keepdims=1),
        helper.make_node("ArgMax", ["row_mask"], ["row_i4"], axis=2, keepdims=1),
        helper.make_node("ArgMax", ["col_mask"], ["col_i4"], axis=3, keepdims=1),
        helper.make_node("Squeeze", ["row_i4"], ["row_i"], axes=[0, 1, 2, 3]),
        helper.make_node("Squeeze", ["col_i4"], ["col_i"], axes=[0, 1, 2, 3]),
        helper.make_node("Add", ["row_i", one], ["rs"]),
        helper.make_node("Add", ["col_i", neg_one], ["cs"]),
        helper.make_node("Unsqueeze", ["rs"], ["rs1"], axes=[0]),
        helper.make_node("Unsqueeze", ["cs"], ["cs1"], axes=[0]),
        helper.make_node("Concat", [zero, zero, "rs1", "cs1"], ["starts"], axis=0),
        helper.make_node("Add", ["starts", extents], ["ends"]),
        helper.make_node("Slice", [IN_NAME, "starts", "ends"], ["patch"]),
        helper.make_node("Pad", ["patch"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, 27, 27]),
    ]

    value_info = [
        helper.make_tensor_value_info("patch", TensorProto.FLOAT, [1, 10, 3, 3]),
    ]
    graph = helper.make_graph(
        nodes,
        "task111_dynamic_slice",
        [x_info],
        [y_info],
        initializer=inits,
        value_info=value_info,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_tight_dynamic_slice_model() -> onnx.ModelProto:
    """Dynamic Slice with the selector crop specialized to observed bounds."""
    inits: list[onnx.TensorProto] = []
    x_info, y_info = _base_io()

    gray_starts = _i64(inits, [5, 0, 1], "gs")
    gray_ends = _i64(inits, [6, 7, 9], "ge")
    gray_axes = _i64(inits, [1, 2, 3], "ga")
    zero = _i64(inits, [0], "z")
    one = _i64(inits, np.array(1, dtype=np.int64), "one")
    extents = _i64(inits, [1, 10, 3, 3], "ex")

    nodes = [
        helper.make_node("Slice", [IN_NAME, gray_starts, gray_ends, gray_axes], ["gray"]),
        helper.make_node("ReduceMax", ["gray"], ["row_mask"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["gray"], ["col_mask"], axes=[2], keepdims=1),
        helper.make_node("ArgMax", ["row_mask"], ["row_i4"], axis=2, keepdims=1),
        helper.make_node("ArgMax", ["col_mask"], ["col_i4"], axis=3, keepdims=1),
        helper.make_node("Add", ["row_i4", one], ["rs4"]),
        helper.make_node("Squeeze", ["rs4"], ["rs1"], axes=[0, 1, 2]),
        helper.make_node("Squeeze", ["col_i4"], ["cs1"], axes=[0, 1, 2]),
        helper.make_node("Concat", [zero, zero, "rs1", "cs1"], ["starts"], axis=0),
        helper.make_node("Add", ["starts", extents], ["ends"]),
        helper.make_node("Slice", [IN_NAME, "starts", "ends"], ["patch"]),
        helper.make_node("Pad", ["patch"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, 27, 27]),
    ]

    value_info = [
        helper.make_tensor_value_info("patch", TensorProto.FLOAT, [1, 10, 3, 3]),
    ]
    graph = helper.make_graph(
        nodes,
        "task111_tight_dynamic_slice",
        [x_info],
        [y_info],
        initializer=inits,
        value_info=value_info,
    )
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
    return build_tight_dynamic_slice_model()


def _grid_to_onehot(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(arr.shape[0], 30)):
        for c in range(min(arr.shape[1], 30)):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _strict_onehot_matches(pred: np.ndarray, expected_grid: np.ndarray | list[list[int]]) -> bool:
    return np.array_equal(pred > 0.0, _grid_to_onehot(expected_grid) > 0.0)


def validate_model(model: onnx.ModelProto) -> tuple[bool, str]:
    try:
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        return False, f"ORT load failed: {exc}"

    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            pred = sess.run([OUT_NAME], {IN_NAME: _grid_to_onehot(ex["input"])})[0]
            if not _strict_onehot_matches(pred, ex["output"]):
                return False, f"{split}#{idx} mismatch"
    return True, "PASS"


def official_score(path: Path) -> dict[str, Any]:
    from score_model import score_file

    return score_file(path)


def save_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    model = build_onnx_model()
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return model


def variant_builders() -> dict[str, Callable[[], onnx.ModelProto]]:
    return {
        "tight_dynamic_slice": build_tight_dynamic_slice_model,
        "dynamic_slice": build_dynamic_slice_model,
        "rowcol_gather": build_rowcol_gather_model,
        "flat_table": build_flat_table_model,
    }


def main() -> None:
    results: list[dict[str, Any]] = []
    for name, builder in variant_builders().items():
        path = OUT_DIR / f"task111_variant_{name}.onnx"
        model = builder()
        onnx.save(model, str(path))
        ok, msg = validate_model(model)
        scored = official_score(path)
        row = {
            "name": name,
            "path": path,
            "valid": ok,
            "validation": msg,
            "memory": scored.get("memory"),
            "params": scored.get("params"),
            "cost": scored.get("cost"),
            "score": scored.get("score"),
            "error": scored.get("error"),
        }
        results.append(row)

    print(f"{'variant':<16}{'pass':<6}{'memory':>8}{'params':>8}{'cost':>8}{'score':>10}")
    for row in results:
        score = row.get("score")
        score_str = f"{score:.6f}" if isinstance(score, float) else "INVALID"
        print(
            f"{row['name']:<16}{str(row['valid']):<6}"
            f"{str(row['memory']):>8}{str(row['params']):>8}"
            f"{str(row['cost']):>8}{score_str:>10}"
        )
        if not row["valid"] or row.get("error"):
            print(f"  note: {row['validation']} {row.get('error') or ''}".rstrip())

    valid = [r for r in results if r["valid"] and r.get("cost") is not None]
    if not valid:
        raise SystemExit("no valid task111 variant")
    best = min(valid, key=lambda r: int(r["cost"]))
    onnx.save(onnx.load(str(best["path"])), str(BEST_PATH))
    print(
        f"\nsaved best variant '{best['name']}' "
        f"(cost={best['cost']}, score={best['score']:.6f}) to {BEST_PATH}"
    )


if __name__ == "__main__":
    main()
