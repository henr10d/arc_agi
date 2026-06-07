"""Measure NeuroGolf points from ONNX cost only (no correctness check).

Scores use the official formula on memory + params. The model does not need
to solve any ARC task or match expected outputs — only be valid enough to
measure (or expose params when runtime profiling fails).

Usage:
  python measure_onnx_score.py path/to/model.onnx
  python measure_onnx_score.py --generate 5
  python measure_onnx_score.py --generate 10 --seed 42 --out-dir /tmp/random_onnx
"""

from __future__ import annotations

import argparse
import copy
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from score_model import (
    EXCLUDED_OP_TYPES,
    FILESIZE_LIMIT_IN_BYTES,
    GRID_SHAPE,
    calculate_memory,
    calculate_params,
    sanitize_model,
    score,
)

IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, 10, 30, 30]
OPSET = 10
IR_VERSION = 10

DUMMY_INPUT = np.zeros(GRID_SHAPE, dtype=np.float32)


def _run_profile(model: onnx.ModelProto, stem: str) -> tuple[str | None, str | None]:
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(Path(tempfile.gettempdir()) / f"ng_measure_{stem}")
    try:
        session = ort.InferenceSession(
            model.SerializeToString(),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        session.run([OUT_NAME], {IN_NAME: DUMMY_INPUT})
        return session.end_profiling(), None
    except Exception:
        return None, traceback.format_exc()


def measure_model(model: onnx.ModelProto, *, label: str = "model") -> dict[str, Any]:
    """Return cost/score metrics without checking task correctness."""
    serialized = model.SerializeToString()
    filesize = len(serialized)
    result: dict[str, Any] = {
        "label": label,
        "filesize": filesize,
        "memory": None,
        "params": None,
        "cost": None,
        "score": None,
        "measurable": False,
        "params_only": False,
        "error": None,
    }
    if filesize > FILESIZE_LIMIT_IN_BYTES:
        result["error"] = f"filesize {filesize} exceeds {FILESIZE_LIMIT_IN_BYTES:.0f}"
        return result

    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        result["error"] = "model failed NeuroGolf name sanitization"
        return result

    for node in sanitized.graph.node:
        if node.op_type.upper() in EXCLUDED_OP_TYPES or "Sequence" in node.op_type:
            result["error"] = f"op type {node.op_type} is not permitted"
            return result

    params = calculate_params(sanitized)
    result["params"] = params
    if params is None or params < 0:
        result["error"] = "params could not be measured"
        return result

    trace_path, runtime_error = _run_profile(sanitized, label)
    if trace_path is None:
        result["params_only"] = True
        result["error"] = runtime_error
        cost = int(params)
        result["cost"] = cost
        result["score"] = score(cost)
        result["measurable"] = True
        return result

    memory = calculate_memory(sanitized, trace_path)
    result["memory"] = memory
    if memory is None or memory < 0:
        result["error"] = "memory could not be measured"
        return result

    cost = int(memory + params)
    result["cost"] = cost
    result["score"] = score(cost)
    result["measurable"] = True
    return result


def measure_file(path: Path) -> dict[str, Any]:
    try:
        model = onnx.load(str(path))
    except Exception:
        return {
            "label": path.name,
            "filesize": path.stat().st_size if path.is_file() else 0,
            "memory": None,
            "params": None,
            "cost": None,
            "score": None,
            "measurable": False,
            "params_only": False,
            "error": traceback.format_exc(),
        }
    result = measure_model(model, label=path.stem)
    result["path"] = path
    result["filesize"] = path.stat().st_size
    return result


def _i64(inits: list[onnx.TensorProto], vals: list[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))
    return name


def _f32_tensor(inits: list[onnx.TensorProto], arr: np.ndarray, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name))
    return name


def _make_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(nodes, name, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _identity_graph() -> onnx.ModelProto:
    node = helper.make_node("Identity", [IN_NAME], [OUT_NAME])
    return _make_model([node], [], "identity")


def _cast_roundtrip_graph() -> onnx.ModelProto:
    nodes = [
        helper.make_node("Cast", [IN_NAME], ["b"], to=TensorProto.BOOL),
        helper.make_node("Cast", ["b"], [OUT_NAME], to=TensorProto.FLOAT),
    ]
    return _make_model(nodes, [], "cast_roundtrip")


def _slice_pad_graph(rng: np.random.Generator) -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = []
    h = int(rng.integers(3, 16))
    w = int(rng.integers(3, 16))
    r0 = int(rng.integers(0, 30 - h + 1))
    c0 = int(rng.integers(0, 30 - w + 1))
    ax = _i64(inits, [0, 1, 2, 3], "ax")
    st = _i64(inits, [0, 0, r0, c0], "st")
    en = _i64(inits, [1, 10, r0 + h, c0 + w], "en")
    pads = [0, 0, r0, c0, 0, 0, 30 - r0 - h, 30 - c0 - w]
    nodes = [
        helper.make_node("Slice", [IN_NAME, st, en, ax], ["core"]),
        helper.make_node("Cast", ["core"], ["core_b"], to=TensorProto.BOOL),
        helper.make_node("Cast", ["core_b"], ["core_f"], to=TensorProto.FLOAT),
        helper.make_node("Pad", ["core_f"], [OUT_NAME], pads=pads),
    ]
    return _make_model(nodes, inits, "slice_pad")


def _constant_add_graph(rng: np.random.Generator) -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = []
    side = int(rng.integers(1, 31))
    arr = rng.random((1, 10, side, side), dtype=np.float32)
    const = _f32_tensor(inits, arr, "k")
    if side == 30:
        nodes = [helper.make_node("Add", [IN_NAME, const], [OUT_NAME])]
    else:
        ax = _i64(inits, [0, 1, 2, 3], "ax")
        st = _i64(inits, [0, 0, 0, 0], "st")
        en = _i64(inits, [1, 10, side, side], "en")
        pads = [0, 0, 0, 0, 0, 0, 30 - side, 30 - side]
        nodes = [
            helper.make_node("Add", [IN_NAME, const], ["partial"]),
            helper.make_node("Slice", ["partial", st, en, ax], ["crop"]),
            helper.make_node("Pad", ["crop"], [OUT_NAME], pads=pads),
        ]
    return _make_model(nodes, inits, "constant_add")


def _identity_chain_graph(rng: np.random.Generator) -> onnx.ModelProto:
    depth = int(rng.integers(1, 12))
    nodes: list[onnx.NodeProto] = []
    prev = IN_NAME
    for i in range(depth - 1):
        nxt = f"t{i}"
        nodes.append(helper.make_node("Identity", [prev], [nxt]))
        prev = nxt
    nodes.append(helper.make_node("Identity", [prev], [OUT_NAME]))
    return _make_model(nodes, [], "identity_chain")


def generate_random_model(rng: np.random.Generator) -> onnx.ModelProto:
    builders = [
        _identity_graph,
        _cast_roundtrip_graph,
        lambda: _slice_pad_graph(rng),
        lambda: _constant_add_graph(rng),
        lambda: _identity_chain_graph(rng),
    ]
    return builders[int(rng.integers(0, len(builders)))]()


def print_report(result: dict[str, Any]) -> None:
    label = result.get("path", Path(result["label"])).name if "path" in result else result["label"]
    print(label)
    print("----------------------------------------")
    print("correctness:       NOT CHECKED")
    print(f"filesize:          {result['filesize']}")
    if not result["measurable"]:
        print("score:             UNMEASURABLE")
        if result["error"]:
            print(f"error:             {str(result['error']).strip()}")
        return
    if result.get("params_only"):
        print("memory:            UNMEASURABLE (runtime failed; params-only estimate)")
    else:
        print(f"memory:            {result['memory']}")
    print(f"params:            {result['params']}")
    print(f"cost:              {result['cost']}")
    print(f"potential score:   {result['score']:.6f}")
    if result.get("params_only"):
        print("note:              score assumes memory=0; actual score may be lower")


def collect_onnx_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*.onnx") if p.is_file())
    raise FileNotFoundError(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure NeuroGolf points from ONNX cost only (no task correctness)."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="ONNX file or directory to score",
    )
    parser.add_argument(
        "--generate",
        type=int,
        metavar="N",
        help="generate and score N random valid ONNX models",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for --generate")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("random_onnx"),
        help="where to save generated models (default: random_onnx/)",
    )
    args = parser.parse_args()

    if args.generate is not None:
        if args.generate <= 0:
            raise SystemExit("--generate must be a positive integer")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(args.seed)
        results: list[dict[str, Any]] = []
        for i in range(args.generate):
            model = generate_random_model(rng)
            out_path = args.out_dir / f"random_{i:04d}.onnx"
            onnx.save(model, str(out_path))
            result = measure_file(out_path)
            results.append(result)
        print(f"Generated {args.generate} models in {args.out_dir.resolve()}\n")
        for index, result in enumerate(results):
            if index:
                print()
            print_report(result)
        measurable = [r for r in results if r["measurable"]]
        if measurable:
            scores = [float(r["score"]) for r in measurable]
            costs = [int(r["cost"]) for r in measurable]
            print()
            print(f"MEASURABLE: {len(measurable)}/{len(results)}")
            print(f"score range: {min(scores):.6f} .. {max(scores):.6f}")
            print(f"cost range:  {min(costs)} .. {max(costs)}")
        return

    if args.path is None:
        parser.error("provide a path or use --generate N")

    paths = collect_onnx_paths(args.path)
    if not paths:
        raise SystemExit(f"no ONNX files found under {args.path}")

    results = [measure_file(path) for path in paths]
    for index, result in enumerate(results):
        if index:
            print()
        print_report(result)

    if len(results) > 1:
        measurable = [r for r in results if r["measurable"]]
        total = sum(float(r["score"]) for r in measurable)
        print()
        print(f"MEASURABLE: {len(measurable)}/{len(results)}")
        print(f"TOTAL POTENTIAL SCORE (if all correct): {total:.6f}")


if __name__ == "__main__":
    main()
