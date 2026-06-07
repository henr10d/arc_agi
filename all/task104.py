"""Minimal ONNX for ARC task104: L-shaped green corner around red center.

Task rule: the 3x3 input has red (2) at center (1,1) and three green (3)
cells forming a 2x2 L-shape in one corner quadrant. Output is 9x9 with two
4x4 green blocks placed by which L-shape is present (four cases A-D):

  A (TL L): blocks at (0,0) and (4,4)
  B (TR L): blocks at (0,5) and (4,1)
  C (BL L): blocks at (1,4) and (5,0)
  D (BR L): blocks at (1,1) and (5,5)

ONNX: read only the east and south green-channel cells, compute the case as
east + 2*south, gather a compact bool 9x9 mask, build only channels 0..3,
then Cast and Pad once to [1,10,30,30].
"""

from __future__ import annotations

import json
import math
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

TASK_ID = "task104"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
CORE = 3
OUT = 9
PAD = H - OUT
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10

BLOCKS: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]] = {
    "A": ((0, 0), (4, 4)),
    "B": ((0, 5), (4, 1)),
    "C": ((1, 4), (5, 0)),
    "D": ((1, 1), (5, 5)),
}


def _mask_for_case(case: str) -> np.ndarray:
    m = np.zeros((OUT, OUT), dtype=np.bool_)
    for r0, c0 in BLOCKS[case]:
        m[r0 : r0 + 4, c0 : c0 + 4] = True
    return m


def solve(grid: np.ndarray | List[List[int]]) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    out = np.zeros((OUT, OUT), dtype=np.int64)

    def isg(r: int, c: int) -> bool:
        return int(g[r, c]) == 3

    if isg(0, 0) and isg(0, 1) and isg(1, 0):
        case = "A"
    elif isg(0, 1) and isg(0, 2) and isg(1, 2):
        case = "B"
    elif isg(1, 0) and isg(2, 0) and isg(2, 1):
        case = "C"
    elif isg(1, 2) and isg(2, 1) and isg(2, 2):
        case = "D"
    else:
        raise ValueError(f"unknown L-shape:\n{g}")
    for r0, c0 in BLOCKS[case]:
        out[r0 : r0 + 4, c0 : c0 + 4] = 3
    return out


def _i64(inits: List[onnx.TensorProto], vals: List[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))
    return name


def _f32(inits: List[onnx.TensorProto], vals: float | List[float], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.float32), name=name))
    return name


def _bool(inits: List[onnx.TensorProto], arr: np.ndarray, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.bool_), name=name))
    return name


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
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _append_case_detection(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    green3: str,
    half: str,
) -> Tuple[str, str, str, str]:
    """Return scalar bool node names A, B, C, D."""

    def cell(r: int, c: int) -> str:
        name = f"gc{r}{c}"
        st = _i64(inits, [0, 0, r, c], f"s{r}{c}")
        en = _i64(inits, [1, 1, r + 1, c + 1], f"e{r}{c}")
        nodes.append(helper.make_node("Slice", [green3, st, en], [name]))
        nodes.append(helper.make_node("Greater", [name, half], [f"{name}b"]))
        return f"{name}b"

    g00 = cell(0, 0)
    g01 = cell(0, 1)
    g02 = cell(0, 2)
    g10 = cell(1, 0)
    g12 = cell(1, 2)
    g20 = cell(2, 0)
    g21 = cell(2, 1)
    g22 = cell(2, 2)

    def and3(a: str, b: str, c: str, out: str) -> None:
        t = f"{out}_t"
        nodes.append(helper.make_node("And", [a, b], [t]))
        nodes.append(helper.make_node("And", [t, c], [out]))

    and3(g00, g01, g10, "caseA")
    and3(g01, g02, g12, "caseB")
    and3(g10, g20, g21, "caseC")
    and3(g12, g21, g22, "caseD")
    return "caseA", "caseB", "caseC", "caseD"


def _append_onehot_output(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    green9: str,
) -> None:
    """Build bool [1,4,9,9], Cast to float, Pad channels/spatial to I/O."""
    nodes.append(helper.make_node("Not", [green9], ["bg9"]))
    nodes.append(helper.make_node("And", [green9, "bg9"], ["false9"]))
    nodes.append(
        helper.make_node(
            "Concat",
            ["bg9", "false9", "false9", green9],
            ["out9b"],
            axis=1,
        )
    )
    nodes.append(helper.make_node("Cast", ["out9b"], ["out9f"], to=TensorProto.FLOAT))
    nodes.append(
        helper.make_node(
            "Pad",
            ["out9f"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 6, PAD, PAD],
            value=0.0,
        )
    )


def _green_core_nodes(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
) -> Tuple[str, str]:
    half = _f32(inits, 0.5, "half")
    core_st = _i64(inits, [0, 3, 0, 0], "core_st")
    core_en = _i64(inits, [1, 4, CORE, CORE], "core_en")
    nodes.append(helper.make_node("Slice", [IN_NAME, core_st, core_en], ["green3"]))
    return "green3", "half"


def _append_edge_case_detection(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    green3: str,
    half: str,
) -> Tuple[str, str, str, str]:
    """Return bool [1] case flags A, B, C, D from N/W/E/S green cells."""

    shape9 = _i64(inits, [9], "shape9")
    edge_idx = _i64(inits, [1, 3, 5, 7], "edge_idx")  # N, W, E, S in row-major order.
    nodes.append(helper.make_node("Reshape", [green3, shape9], ["green9flat"]))
    nodes.append(helper.make_node("Gather", ["green9flat", edge_idx], ["edge_vals"], axis=0))
    nodes.append(helper.make_node("Greater", ["edge_vals", half], ["edge_green"]))

    for idx, name in enumerate(("n", "w", "e", "s")):
        nodes.append(helper.make_node("Gather", ["edge_green", _i64(inits, [idx], f"{name}_idx")], [name], axis=0))

    nodes.append(helper.make_node("And", ["n", "w"], ["caseA"]))
    nodes.append(helper.make_node("And", ["n", "e"], ["caseB"]))
    nodes.append(helper.make_node("And", ["w", "s"], ["caseC"]))
    nodes.append(helper.make_node("And", ["e", "s"], ["caseD"]))
    return "caseA", "caseB", "caseC", "caseD"


def build_edge_gather_mask_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    green3, half = _green_core_nodes(nodes, inits)
    caseA, caseB, caseC, caseD = _append_edge_case_detection(nodes, inits, green3, half)

    masks = np.stack([_mask_for_case(k) for k in ("A", "B", "C", "D")], axis=0)[:, np.newaxis, :, :]
    _bool(inits, masks, "masks4")

    for case, out in zip((caseA, caseB, caseC, caseD), ("cAf", "cBf", "cCf", "cDf")):
        nodes.append(helper.make_node("Cast", [case], [out], to=TensorProto.FLOAT))

    shape14 = _i64(inits, [1, 4], "shape14_edge")
    nodes.append(helper.make_node("Concat", ["cAf", "cBf", "cCf", "cDf"], ["cases4"], axis=0))
    nodes.append(helper.make_node("Reshape", ["cases4", shape14], ["cases14"]))
    nodes.append(helper.make_node("ArgMax", ["cases14"], ["case_idx"], axis=1, keepdims=0))
    nodes.append(helper.make_node("Gather", ["masks4", "case_idx"], ["green9"], axis=0))
    _append_onehot_output(nodes, inits, "green9")

    graph = helper.make_graph(nodes, "edge_gather_mask", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return model


def build_direct_es_model() -> onnx.ModelProto:
    """Classify by east/south green cells: case index = east + 2*south."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    east_st = _i64(inits, [0, 3, 1, 2], "east_st")
    east_en = _i64(inits, [1, 4, 2, 3], "east_en")
    south_st = _i64(inits, [0, 3, 2, 1], "south_st")
    south_en = _i64(inits, [1, 4, 3, 2], "south_en")
    nodes.append(helper.make_node("Slice", [IN_NAME, east_st, east_en], ["east_f"]))
    nodes.append(helper.make_node("Slice", [IN_NAME, south_st, south_en], ["south_f"]))
    nodes.append(helper.make_node("Cast", ["east_f"], ["east_i"], to=TensorProto.INT32))
    nodes.append(helper.make_node("Cast", ["south_f"], ["south_i"], to=TensorProto.INT32))
    nodes.append(helper.make_node("Add", ["south_i", "south_i"], ["south2_i"]))
    nodes.append(helper.make_node("Add", ["east_i", "south2_i"], ["case_idx_raw"]))
    nodes.append(helper.make_node("Squeeze", ["case_idx_raw"], ["case_idx"], axes=[1, 2, 3]))

    masks = np.stack([_mask_for_case(k) for k in ("A", "B", "C", "D")], axis=0)[:, np.newaxis, :, :]
    _bool(inits, masks, "masks4")
    nodes.append(helper.make_node("Gather", ["masks4", "case_idx"], ["green9"], axis=0))
    _append_onehot_output(nodes, inits, "green9")

    graph = helper.make_graph(nodes, "direct_es", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return model


def build_core_es_model() -> onnx.ModelProto:
    """Slice the 3x3 green core, gather east/south, and compute east + 2*south."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    core_st = _i64(inits, [0, 3, 0, 0], "core_st_es")
    core_en = _i64(inits, [1, 4, CORE, CORE], "core_en_es")
    nodes.append(helper.make_node("Slice", [IN_NAME, core_st, core_en], ["green3"]))
    shape9 = _i64(inits, [9], "shape9_es")
    es_idx = _i64(inits, [5, 7], "es_idx")  # east, south in row-major 3x3 core.
    nodes.append(helper.make_node("Reshape", ["green3", shape9], ["green9flat"]))
    nodes.append(helper.make_node("Gather", ["green9flat", es_idx], ["es_f"], axis=0))
    nodes.append(helper.make_node("Cast", ["es_f"], ["es_i"], to=TensorProto.INT32))
    nodes.append(helper.make_node("Gather", ["es_i", _i64(inits, [0], "e_idx")], ["east_i"], axis=0))
    nodes.append(helper.make_node("Gather", ["es_i", _i64(inits, [1], "s_idx")], ["south_i"], axis=0))
    nodes.append(helper.make_node("Add", ["south_i", "south_i"], ["south2_i"]))
    nodes.append(helper.make_node("Add", ["east_i", "south2_i"], ["case_idx"]))

    masks = np.stack([_mask_for_case(k) for k in ("A", "B", "C", "D")], axis=0)[:, np.newaxis, :, :]
    _bool(inits, masks, "masks4")
    nodes.append(helper.make_node("Gather", ["masks4", "case_idx"], ["green9"], axis=0))
    _append_onehot_output(nodes, inits, "green9")

    graph = helper.make_graph(nodes, "core_es", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return model


def build_masks_or_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    green3, half = _green_core_nodes(nodes, inits)
    caseA, caseB, caseC, caseD = _append_case_detection(nodes, inits, green3, half)

    mask_names: List[str] = []
    for tag in ("A", "B", "C", "D"):
        m = _bool(inits, _mask_for_case(tag)[np.newaxis, np.newaxis, :, :], f"mask_{tag}")
        mask_names.append(m)

    parts: List[str] = []
    for case, m in zip((caseA, caseB, caseC, caseD), mask_names):
        p = f"sel_{m}"
        nodes.append(helper.make_node("And", [case, m], [p]))
        parts.append(p)

    nodes.append(helper.make_node("Or", [parts[0], parts[1]], ["or01"]))
    nodes.append(helper.make_node("Or", [parts[2], parts[3]], ["or23"]))
    nodes.append(helper.make_node("Or", ["or01", "or23"], ["green9"]))
    _append_onehot_output(nodes, inits, "green9")

    graph = helper.make_graph(nodes, "masks_or", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return model


def build_gather_add_model() -> onnx.ModelProto:
    """Gather mask via weighted case sum (no ArgMax/Reshape)."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    green3, half = _green_core_nodes(nodes, inits)
    caseA, caseB, caseC, caseD = _append_case_detection(nodes, inits, green3, half)

    masks = np.stack([_mask_for_case(k) for k in ("A", "B", "C", "D")], axis=0)[:, np.newaxis, :, :]
    _bool(inits, masks, "masks4")

    zero = _f32(inits, 0.0, "zero")
    one = _f32(inits, 1.0, "one")
    two = _f32(inits, 2.0, "two")
    three = _f32(inits, 3.0, "three")

    for case, out, weight in zip(
        (caseA, caseB, caseC, caseD),
        ("wA", "wB", "wC", "wD"),
        (zero, one, two, three),
    ):
        nodes.append(helper.make_node("Cast", [case], [f"{out}b"], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Mul", [f"{out}b", weight], [out]))

    nodes.append(helper.make_node("Add", ["wA", "wB"], ["idx01"]))
    nodes.append(helper.make_node("Add", ["idx01", "wC"], ["idx012"]))
    nodes.append(helper.make_node("Add", ["idx012", "wD"], ["case_idx_f"]))
    nodes.append(
        helper.make_node("ReduceSum", ["case_idx_f"], ["case_idx_fsum"], axes=[0, 1, 2, 3], keepdims=0)
    )
    nodes.append(helper.make_node("Cast", ["case_idx_fsum"], ["case_idx"], to=TensorProto.INT64))
    nodes.append(helper.make_node("Gather", ["masks4", "case_idx"], ["green9_raw"], axis=0))
    nodes.append(helper.make_node("Unsqueeze", ["green9_raw"], ["green9"], axes=[0]))
    _append_onehot_output(nodes, inits, "green9")

    graph = helper.make_graph(nodes, "gather_add", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return model


def build_gather_mask_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    green3, half = _green_core_nodes(nodes, inits)
    caseA, caseB, caseC, caseD = _append_case_detection(nodes, inits, green3, half)

    masks = np.stack([_mask_for_case(k) for k in ("A", "B", "C", "D")], axis=0)[:, np.newaxis, :, :]
    _bool(inits, masks, "masks4")

    for case, out in zip((caseA, caseB, caseC, caseD), ("cAf", "cBf", "cCf", "cDf")):
        nodes.append(helper.make_node("Cast", [case], [out], to=TensorProto.FLOAT))

    shape4 = _i64(inits, [4], "shape4")
    shape14 = _i64(inits, [1, 4], "shape14")
    nodes.append(helper.make_node("Concat", ["cAf", "cBf", "cCf", "cDf"], ["cases4"], axis=0))
    nodes.append(helper.make_node("Reshape", ["cases4", shape4], ["cases_vec"]))
    nodes.append(helper.make_node("Reshape", ["cases_vec", shape14], ["cases14"]))
    nodes.append(helper.make_node("ArgMax", ["cases14"], ["case_idx"], axis=1, keepdims=0))
    nodes.append(helper.make_node("Gather", ["masks4", "case_idx"], ["green9"], axis=0))
    _append_onehot_output(nodes, inits, "green9")

    graph = helper.make_graph(nodes, "gather_mask", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return model


def build_model() -> onnx.ModelProto:
    """Default export: lowest official cost in local benchmarks."""
    return build_direct_es_model()


def variant_builders() -> Dict[str, Callable[[], onnx.ModelProto]]:
    return {
        "direct_es": build_direct_es_model,
        "core_es": build_core_es_model,
        "edge_gather": build_edge_gather_mask_model,
        "gather_add": build_gather_add_model,
        "gather_mask": build_gather_mask_model,
        "masks_or": build_masks_or_model,
    }


def model_stats(model: onnx.ModelProto, path: Path | None = None) -> Dict[str, Any]:
    params = sum(int(np.prod(list(t.dims))) for t in model.graph.initializer)
    stats: Dict[str, Any] = {
        "nodes": len(model.graph.node),
        "params": params,
        "inits": len(model.graph.initializer),
        "opset": model.opset_import[0].version,
        "ir": model.ir_version,
    }
    if path is not None and path.is_file():
        stats["bytes"] = path.stat().st_size
    return stats


def official_score(path: Path) -> Dict[str, Any]:
    from score_model import score_file

    return score_file(path)


def _strict_onehot_matches(pred: np.ndarray, expected: np.ndarray) -> bool:
    return pred.shape == expected.shape and np.array_equal(pred > 0, expected > 0)


def validate_model(model: onnx.ModelProto, path: Path | None = None) -> Tuple[bool, str]:
    from score_model import convert_to_numpy

    try:
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    except Exception as exc:
        return False, f"ORT load failed: {exc}"

    if not DATA_PATH.is_file():
        return False, "missing task data"

    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            inp = convert_to_numpy(ex, "input")
            exp = convert_to_numpy(ex, "output")
            if inp is None or exp is None:
                return False, f"{split}#{idx} invalid example size"
            pred = sess.run([OUT_NAME], {IN_NAME: inp})[0]
            if not _strict_onehot_matches(pred, exp):
                return False, f"{split}#{idx} strict one-hot mismatch"

    if path is not None:
        try:
            from train_arc import validate_task_onnx

            ok, _ = validate_task_onnx(TASK_ID, path, save_viz=False)
            if not ok:
                return False, "train_arc validation failed"
        except Exception as exc:
            return False, f"train_arc validation error: {exc}"

    return True, "PASS"


def run_experiments() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for name, builder in variant_builders().items():
        path = OUT_DIR / f"{TASK_ID}_variant_{name}.onnx"
        row: Dict[str, Any] = {"name": name, "path": path, "valid": False}
        try:
            model = builder()
            path.parent.mkdir(parents=True, exist_ok=True)
            onnx.save(model, str(path))
            row.update(model_stats(model, path))
            ok, message = validate_model(model, path)
            row["valid"] = ok
            row["validation"] = message
            scored = official_score(path)
            row["official_memory"] = scored.get("memory")
            row["official_params"] = scored.get("params")
            row["official_cost"] = scored.get("cost")
            row["official_score"] = scored.get("score")
            row["score_error"] = scored.get("error")
        except Exception as exc:
            row["validation"] = f"build/check failed: {exc}"
        results.append(row)

    passing = [
        r
        for r in results
        if r.get("valid") and r.get("official_cost") is not None and r.get("official_score") is not None
    ]
    if passing:
        best = min(passing, key=lambda r: int(r["official_cost"]))
        onnx.save(onnx.load(str(best["path"])), str(BEST_PATH))
    return results


def print_report(results: List[Dict[str, Any]]) -> None:
    print("\nVariant comparison (task104)")
    print(
        f"{'variant':<14} {'pass':<5} {'bytes':>6} {'nodes':>5} {'params':>6} "
        f"{'memory':>8} {'cost':>6} {'score':>9}"
    )
    for row in results:
        score = row.get("official_score")
        score_s = f"{score:.6f}" if isinstance(score, float) else "INVALID"
        print(
            f"{row['name']:<14} {str(row.get('valid')):<5} "
            f"{str(row.get('bytes', '-')):>6} {str(row.get('nodes', '-')):>5} "
            f"{str(row.get('official_params', row.get('params', '-'))):>6} "
            f"{str(row.get('official_memory', '-')):>8} "
            f"{str(row.get('official_cost', '-')):>6} {score_s:>9}"
        )
        if not row.get("valid") or row.get("score_error"):
            print(f"  note: {row.get('validation')} {row.get('score_error') or ''}".rstrip())

    passing = [r for r in results if r.get("valid") and r.get("official_cost") is not None]
    if passing:
        best = min(passing, key=lambda r: int(r["official_cost"]))
        print(
            f"\nBest: {best['name']} cost={best['official_cost']} "
            f"score={best['official_score']:.6f} -> {BEST_PATH}"
        )


def main() -> None:
    results = run_experiments()
    print_report(results)


if __name__ == "__main__":
    main()
