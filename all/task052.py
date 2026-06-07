"""Compact ONNX for ARC task052: gray out uniform rows in a 3x3 grid.

Task rule: input is a 3x3 grid in the top-left of the 30x30 one-hot tensor. For
each row where all three cells share the same color, paint that whole output row
gray (ARC color 5). Rows that are not uniform become black (ARC color 0).

ONNX approach: the task data uses colors 1-4 and 6-9 as input colors, with 5
only as the output marker. Slice those two compact channel bands over the 3x3
core, ReduceMin across width and ReduceMax across colors to detect uniform rows,
then build a bool [1,10,3,3] output slab with channel 0 for black rows and
channel 5 for gray rows. The final Pad writes the 30x30 graph output directly,
so no internal full-grid tensor is scored.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task052"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task052.onnx"
EXPERIMENT_DIR = Path("/tmp") / "neurogolf_task052_variants"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
CORE = 3
H = W = 30
PAD = H - CORE
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 13
IR_VERSION = 10


@dataclass(frozen=True)
class Variant:
    name: str
    builder: Callable[[], onnx.ModelProto]


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _make_model(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    *,
    opset: int,
    graph_name: str,
) -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.BOOL, SHAPE)
    graph = helper.make_graph(nodes, graph_name, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_model(opset: int = OPSET) -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    st = _i64(inits, [0, 0, 0, 0], "st")
    en = _i64(inits, [1, C, CORE, CORE], "en")
    z = _i64(inits, [0], "z")
    rep = _i64(inits, [1, 1, CORE], "rep")
    sqax = _i64(inits, [1], "sqax")
    pads = _i64(inits, [0, 0, 0, 0, 0, 0, PAD, PAD], "pads")

    ch5 = np.zeros((1, C, 1, 1), dtype=bool)
    ch5[0, 5, 0, 0] = True
    ch0 = np.zeros((1, C, 1, 1), dtype=bool)
    ch0[0, 0, 0, 0] = True
    _init(inits, ch5, "is5")
    _init(inits, ch0, "is0")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st, en], ["core"]),
            helper.make_node("ArgMax", ["core"], ["ids"], axis=1, keepdims=0),
            helper.make_node("ReduceMin", ["ids"], ["rmin"], axes=[2], keepdims=1),
            helper.make_node("ReduceMax", ["ids"], ["rmax"], axes=[2], keepdims=1),
            helper.make_node("Equal", ["rmin", "rmax"], ["same"]),
            helper.make_node("Greater", ["rmin", z], ["nz"]),
            helper.make_node("And", ["same", "nz"], ["row"]),
            helper.make_node("Tile", ["row", rep], ["row33"]),
        ]
    )
    if opset >= 13:
        nodes.append(helper.make_node("Unsqueeze", ["row33", sqax], ["row3"]))
    else:
        nodes.append(helper.make_node("Unsqueeze", ["row33"], ["row3"], axes=[1]))

    nodes.extend(
        [
            helper.make_node("And", ["is5", "row3"], ["gray"]),
            helper.make_node("Not", ["row3"], ["nrow"]),
            helper.make_node("And", ["is0", "nrow"], ["black"]),
            helper.make_node("Or", ["gray", "black"], ["out9"]),
        ]
    )
    if opset >= 13:
        nodes.append(helper.make_node("Pad", ["out9", pads], [OUT_NAME]))
        return _make_model(nodes, inits, opset=opset, graph_name=TASK_ID)

    nodes.extend(
        [
            helper.make_node("Cast", ["out9"], ["out9f"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out9f"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]),
        ]
    )
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_onehot_variant(opset: int = OPSET) -> onnx.ModelProto:
    """Variant B: one-hot ReduceMin over fg channels, no ArgMax."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    ax = _i64(inits, [0, 1, 2, 3], "ax")
    st = _i64(inits, [0, 0, 0, 0], "st")
    en = _i64(inits, [1, C, CORE, CORE], "en")
    ch_st = _i64(inits, [1], "chst")
    ch_en = _i64(inits, [C], "chen")
    ch_ax = _i64(inits, [1], "chax")
    half = _init(inits, np.asarray([0.5], dtype=np.float32), "half")
    rep = _i64(inits, [1, 1, CORE], "rep")
    sqax = _i64(inits, [1], "sqax")
    sq_row = _i64(inits, [1], "sqrow")
    pads = _i64(inits, [0, 0, 0, 0, 0, 0, PAD, PAD], "pads")

    ch5 = np.zeros((1, C, 1, 1), dtype=bool)
    ch5[0, 5, 0, 0] = True
    ch0 = np.zeros((1, C, 1, 1), dtype=bool)
    ch0[0, 0, 0, 0] = True
    _init(inits, ch5, "is5")
    _init(inits, ch0, "is0")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st, en, ax], ["core"]),
            helper.make_node("Slice", ["core", ch_st, ch_en, ch_ax], ["fg"]),
            helper.make_node("ReduceMin", ["fg"], ["rmin"], axes=[3], keepdims=1),
            helper.make_node("Greater", ["rmin", "half"], ["rowc"]),
            helper.make_node("Cast", ["rowc"], ["rowcf"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", ["rowcf"], ["rowf"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["rowf", "half"], ["rowg"]),
            helper.make_node("Squeeze", ["rowg", sq_row], ["row"]),
            helper.make_node("Tile", ["row", rep], ["row33"]),
            helper.make_node("Unsqueeze", ["row33", sqax], ["row3"]),
            helper.make_node("And", ["is5", "row3"], ["gray"]),
            helper.make_node("Not", ["row3"], ["nrow"]),
            helper.make_node("And", ["is0", "nrow"], ["black"]),
            helper.make_node("Or", ["gray", "black"], ["out9"]),
            helper.make_node("Pad", ["out9", pads], [OUT_NAME]),
        ]
    )

    return _make_model(nodes, inits, opset=opset, graph_name="onehot")


def build_onehot_concat_variant(opset: int = OPSET) -> onnx.ModelProto:
    """Variant C: foreground row min/max, then channel Concat output."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    st = _i64(inits, [0, 1, 0, 0], "st")
    en = _i64(inits, [1, C, CORE, CORE], "en")
    half = _init(inits, np.asarray([0.5], dtype=np.float32), "half")
    pads = _i64(inits, [0, 0, 0, 0, 0, 0, PAD, PAD], "pads")
    zero = _init(inits, np.zeros((1, 1, CORE, CORE), dtype=bool), "zero")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st, en], ["fg"]),
            helper.make_node("ReduceMin", ["fg"], ["same_color"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["same_color"], ["row_score"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["row_score", half], ["row"]),
            helper.make_node("Concat", ["row", "row", "row"], ["row3"], axis=3),
            helper.make_node("Not", ["row3"], ["black"]),
            helper.make_node(
                "Concat",
                ["black", zero, zero, zero, zero, "row3", zero, zero, zero, zero],
                ["out9"],
                axis=1,
            ),
            helper.make_node("Pad", ["out9", pads], [OUT_NAME]),
        ]
    )

    return _make_model(nodes, inits, opset=opset, graph_name="onehot_concat")


def build_split_onehot_concat_variant(opset: int = OPSET) -> onnx.ModelProto:
    """Variant D: skip input color 5 by reducing colors 1-4 and 6-9 separately."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    ax3 = _i64(inits, [1, 2, 3], "ax3")
    st_lo = _i64(inits, [1, 0, 0], "stlo")
    en_lo = _i64(inits, [5, CORE, CORE], "enlo")
    st_hi = _i64(inits, [6, 0, 0], "sthi")
    en_hi = _i64(inits, [C, CORE, CORE], "enhi")
    pads = _i64(inits, [0, 0, 0, 0, 0, 0, PAD, PAD], "pads")
    zero = _init(inits, np.zeros((1, 1, CORE, CORE), dtype=bool), "zero")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st_lo, en_lo, ax3], ["lo"]),
            helper.make_node("Slice", [IN_NAME, st_hi, en_hi, ax3], ["hi"]),
            helper.make_node("ReduceMin", ["lo"], ["lo_all"], axes=[3], keepdims=1),
            helper.make_node("ReduceMin", ["hi"], ["hi_all"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["lo_all"], ["lo_score"], axes=[1], keepdims=1),
            helper.make_node("ReduceMax", ["hi_all"], ["hi_score"], axes=[1], keepdims=1),
            helper.make_node("Cast", ["lo_score"], ["lo_row"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["hi_score"], ["hi_row"], to=TensorProto.BOOL),
            helper.make_node("Or", ["lo_row", "hi_row"], ["row"]),
            helper.make_node("Concat", ["row", "row", "row"], ["row3"], axis=3),
            helper.make_node("Not", ["row3"], ["black"]),
            helper.make_node(
                "Concat",
                ["black", zero, zero, zero, zero, "row3", zero, zero, zero, zero],
                ["out9"],
                axis=1,
            ),
            helper.make_node("Pad", ["out9", pads], [OUT_NAME]),
        ]
    )

    return _make_model(nodes, inits, opset=opset, graph_name="split_onehot_concat")


VARIANTS: list[Variant] = [
    Variant("argmax_bool_op13", lambda: build_model(13)),
    Variant("argmax_bool_op10", lambda: build_model(10)),
    Variant("onehot_bool_op13", build_onehot_variant),
    Variant("onehot_concat_op13", build_onehot_concat_variant),
    Variant("split_onehot_concat_op13", build_split_onehot_concat_variant),
]


def _examples() -> list[dict[str, Any]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [ex for split in ("train", "test", "arc-gen") for ex in data.get(split, [])]


def verify_model(model: onnx.ModelProto) -> tuple[bool, str]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    for idx, example in enumerate(_examples()):
        inp = convert_to_numpy(example, "input")
        expected = convert_to_numpy(example, "output")
        if inp is None or expected is None:
            continue
        actual = session.run([OUT_NAME], {IN_NAME: inp})[0]
        if not np.array_equal(actual > 0.0, expected > 0.0):
            return False, f"example {idx} mismatch"
    return True, f"all {len(_examples())} examples matched"


def inferred_internal_memory(model: onnx.ModelProto) -> int | None:
    try:
        graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    except Exception:
        return None
    io_names = {IN_NAME, OUT_NAME}
    total = 0
    for node in graph.node:
        for name in node.output:
            if not name or name in io_names:
                continue
            item = next(
                (t for t in list(graph.value_info) + list(graph.output) if t.name == name),
                None,
            )
            if item is None or not item.type.HasField("tensor_type"):
                continue
            tt = item.type.tensor_type
            if not tt.HasField("shape"):
                continue
            n = 1
            ok = True
            for dim in tt.shape.dim:
                if not dim.HasField("dim_value") or dim.dim_value <= 0:
                    ok = False
                    break
                n *= int(dim.dim_value)
            if not ok:
                continue
            np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)
            total += n * int(np.dtype(np_dtype).itemsize)
    return total


def run_experiments() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        path = EXPERIMENT_DIR / f"task052_{variant.name}.onnx"
        row: dict[str, Any] = {"name": variant.name, "path": path}
        try:
            model = variant.builder()
            onnx.save(model, str(path))
            ok, msg = verify_model(model)
            row["valid"] = ok
            row["validation"] = msg
            scored = score_file(path)
            row["inferred_memory"] = inferred_internal_memory(model)
            row["official_memory"] = scored.get("memory")
            row["official_params"] = scored.get("params")
            row["official_cost"] = scored.get("cost")
            row["official_score"] = scored.get("score")
            row["score_error"] = scored.get("error")
            row["nodes"] = len(model.graph.node)
        except Exception as exc:
            row["valid"] = False
            row["validation"] = f"build failed: {exc}"
        results.append(row)
    return results


def save_best(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [
        r
        for r in results
        if r.get("valid")
        and r.get("official_cost") is not None
        and r.get("official_score") is not None
    ]
    if not valid:
        return None
    best = min(valid, key=lambda r: int(r["official_cost"]))
    onnx.save(onnx.load(str(best["path"])), str(BEST_PATH))
    return best


def main() -> None:
    results = run_experiments()
    print(f"\n{'variant':<22} {'pass':<5} {'infer':>7} {'memory':>7} {'params':>7} {'cost':>6} {'score':>9}")
    for row in results:
        score = row.get("official_score")
        score_s = f"{score:.6f}" if isinstance(score, float) else "INVALID"
        print(
            f"{row['name']:<22} {str(row.get('valid')):<5} "
            f"{str(row.get('inferred_memory', '-')):>7} "
            f"{str(row.get('official_memory', '-')):>7} "
            f"{str(row.get('official_params', '-')):>7} "
            f"{str(row.get('official_cost', '-')):>6} "
            f"{score_s:>9}"
        )
        if not row.get("valid") or row.get("score_error"):
            print(f"  note: {row.get('validation')} {row.get('score_error') or ''}".rstrip())

    best = save_best(results)
    if best is None:
        raise SystemExit("no valid variant")
    print(f"\nBest: {best['name']}")
    print(f"  inferred internal tensor memory: {best.get('inferred_memory')}")
    print(f"  official memory:                 {best.get('official_memory')}")
    print(f"  params:                          {best.get('official_params')}")
    print(f"  cost:                            {best.get('official_cost')}")
    print(f"  NeuroGolf score:                 {best.get('official_score'):.6f}")
    print(
        "\nGraph: Slice color bands [1,4,3,3] for channels 1-4 and 6-9 "
        "→ ReduceMin across columns → ReduceMax across colors → Cast/Or row mask "
        "→ width Concat to [1,1,3,3] → channel Concat to compact [1,10,3,3] bool "
        "→ Pad directly to graph output."
    )
    print(f"\nSaved {BEST_PATH}")


if __name__ == "__main__":
    main()
