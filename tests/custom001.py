"""Minimal-size NeuroGolf task001 ONNX (opset-10 compatible hand-built graph)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

C = 10
H = W = 30
CORE = 3
OUT = CORE * CORE
PAD = H - OUT
IN_NAME = "input"
OUT_NAME = "output"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task001.onnx"

# Competition requires opset 10 + IR 10 (see NeuroGolf 2026 rules).
OPSET = 10
IR_VERSION = 10


def _i64(inits: List[onnx.TensorProto], vals: List[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.array(vals, dtype=np.int64), name=name))
    return name


def _f32(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name))
    return name


def _strip(model: onnx.ModelProto) -> onnx.ModelProto:
    model.doc_string = ""
    model.producer_name = ""
    model.producer_version = ""
    model.domain = ""
    model.model_version = 0
    del model.metadata_props[:]
    return model


def _simplify(model: onnx.ModelProto) -> Tuple[onnx.ModelProto, bool]:
    try:
        import onnxsim
    except ImportError:
        return model, False
    try:
        slim, ok = onnxsim.simplify(model)
        return (slim if ok else model), ok
    except Exception:
        return model, False


def _stats(path: Path, model: onnx.ModelProto | None = None) -> Dict:
    model = model or onnx.load(str(path))
    ops: Dict[str, int] = {}
    for n in model.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    params = sum(int(np.prod(list(t.dims))) for t in model.graph.initializer)
    return {
        "path": str(path),
        "bytes": path.stat().st_size if path.is_file() else len(model.SerializeToString()),
        "nodes": len(model.graph.node),
        "inits": len(model.graph.initializer),
        "params": params,
        "ops": ops,
        "opset": model.opset_import[0].version if model.opset_import else None,
        "ir": model.ir_version,
    }


def build_best() -> onnx.ModelProto:
    """
    Minimal opset-10 stencil + ch0 background fix for official grader.

    Kaggle compares (output > 0) to one-hot tensors, not argmax grids.
    Empty cells must have channel 0 = 1.0, not all zeros.
    """
    shape = [1, C, H, W]
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, shape)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, shape)

    ax = _i64(inits, [0, 1, 2, 3], "a")
    st = _i64(inits, [0, 0, 0, 0], "s")
    en = _i64(inits, [1, C, CORE, CORE], "e")
    e0 = _i64(inits, [1, 1, CORE, CORE], "f")
    efg = _i64(inits, [1, C, H, W], "g")
    ec0 = _i64(inits, [1, 1, H, W], "h")
    one = _f32(inits, [1.0], "o")
    zero = _f32(inits, [0.0], "z")
    rep4 = _i64(inits, [1, 1, CORE, CORE], "r")
    r6m = _i64(inits, [1, 1, CORE, 1, CORE, 1], "b")
    rep6 = _i64(inits, [1, 1, 1, CORE, 1, CORE], "c")
    s9m = _i64(inits, [1, 1, OUT, OUT], "d")
    _f32(
        inits,
        np.pad(np.ones((1, 1, OUT, 1), dtype=np.float32), ((0, 0), (0, 0), (0, PAD), (0, 0))),
        "rm",
    )
    _f32(
        inits,
        np.pad(np.ones((1, 1, 1, OUT), dtype=np.float32), ((0, 0), (0, 0), (0, 0), (0, PAD))),
        "cm",
    )
    st1 = _i64(inits, [0, 1, 0, 0], "t")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st, en, ax], ["core"]),
            helper.make_node("Slice", ["core", st, e0, ax], ["ch0"]),
            helper.make_node("Sub", [one, "ch0"], ["mask"]),
            helper.make_node("Tile", ["core", rep4], ["body"]),
            helper.make_node("Reshape", ["mask", r6m], ["m6"]),
            helper.make_node("Tile", ["m6", rep6], ["mt"]),
            helper.make_node("Reshape", ["mt", s9m], ["m9"]),
            helper.make_node("Mul", ["body", "m9"], ["o9"]),
            helper.make_node("Pad", ["o9"], ["pad"], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]),
            helper.make_node("Slice", ["pad", st, efg, ax], ["fg"]),
            helper.make_node("ReduceMax", ["fg"], ["mx"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["mx", zero], ["fa"]),
            helper.make_node("Not", ["fa"], ["nf"]),
            helper.make_node("Mul", ["rm", "cm"], ["rg"]),
            helper.make_node("Greater", ["rg", zero], ["ir"]),
            helper.make_node("And", ["nf", "ir"], ["bg"]),
            helper.make_node("Slice", ["pad", st, ec0, ax], ["c0"]),
            helper.make_node("Where", ["bg", one, "c0"], ["c0f"]),
            helper.make_node("Slice", ["pad", st1, efg, ax], ["tl"]),
            helper.make_node("Concat", ["c0f", "tl"], [OUT_NAME], axis=1),
        ]
    )

    graph = helper.make_graph(nodes, "g", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_reference_opset10() -> onnx.ModelProto:
    """Known-good task001 graph from build_tile_mask_model (larger, for A/B)."""
    from build_tile_mask_model import build_optimized_model

    model = build_optimized_model()
    model.producer_name = ""
    model.doc_string = ""
    return model


def _save(model: onnx.ModelProto, path: Path) -> Dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    slim, did = _simplify(_strip(model))
    onnx.save(slim, str(path))
    st = _stats(path, slim)
    st["simplified"] = did
    return st


def _valid(path: Path) -> bool:
    try:
        import json
        import onnxruntime as ort

        sys.path.insert(0, str(ROOT / "data" / "neurogolf_utils"))
        from neurogolf_utils import verify_subset

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        with (ROOT / "data" / "task001.json").open(encoding="utf-8") as fh:
            examples = json.load(fh)
        for split in ("train", "test", "arc-gen"):
            right, wrong, _ = verify_subset(sess, examples[split])
            if wrong:
                print(f"  neurogolf {split}: FAIL ({wrong} wrong)")
                return False
            print(f"  neurogolf {split}: PASS ({right})")

        from train_arc import validate_task_onnx

        ok, _ = validate_task_onnx("task001", path, save_viz=False)
        return ok
    except Exception as exc:
        print(f"  validate error: {exc}")
        return False


def _print_stats(label: str, st: Dict) -> None:
    sim = " sim" if st.get("simplified") else ""
    print(
        f"  {label}: {st['bytes']} B | nodes={st['nodes']} inits={st['inits']} "
        f"params={st['params']} opset={st['opset']} ir={st['ir']} ops={st['ops']}{sim}"
    )


def main() -> None:
    rows: List[Tuple[str, Dict, bool]] = []

    print("=== competition-compatible minimal (opset 10) ===")
    path = OUT_DIR / ".best_o10.onnx"
    st = _save(build_best(), path)
    ok = _valid(path)
    st["source"] = "best@10"
    _print_stats("best@10", st)
    rows.append(("best@10", st, ok))

    print("\n=== reference build_tile_mask optimized ===")
    ref_path = OUT_DIR / ".ref_o10.onnx"
    try:
        st_ref = _save(build_reference_opset10(), ref_path)
        ok_ref = _valid(ref_path)
        st_ref["source"] = "ref@10"
        _print_stats("ref@10", st_ref)
        rows.append(("ref@10", st_ref, ok_ref))
    except Exception as exc:
        print(f"  ref@10: FAIL ({exc})")

    valid = [(n, s) for n, s, ok in rows if ok]
    if not valid:
        print("\nNo valid exports.")
        raise SystemExit(1)

    best_name, best = min(valid, key=lambda x: x[1]["bytes"])
    src = Path(best["path"])
    if src != BEST_PATH:
        src.replace(BEST_PATH)
    best["path"] = str(BEST_PATH)

    print("\n=== comparison (valid only) ===")
    for name, st, ok in sorted(rows, key=lambda x: x[1]["bytes"]):
        flag = "PASS" if ok else "FAIL"
        print(
            f"  [{flag}] {name:12s} {st['bytes']:5d} B  nodes={st['nodes']:2d}  "
            f"inits={st['inits']:2d}  opset={st['opset']}  {st['ops']}"
        )

    print(f"\nSMALLEST VALID: {best_name} -> {BEST_PATH}")
    _print_stats("winner", best)

    for p in OUT_DIR.glob(".*.onnx"):
        p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
