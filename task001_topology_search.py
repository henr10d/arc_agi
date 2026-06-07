"""Experimental topology search for task001.

This script benchmarks radical final-node ideas against the current F
fallback. Several candidates intentionally produce the large
[1,10,30,30] tensor directly as graph output to test whether it is
excluded from official memory; correctness is checked separately.
"""

from __future__ import annotations

import json
import math
import shutil
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from graph_onnx_memory import analyze
from onnx1 import IN_NAME, OUT_NAME, build_model
from score_model import convert_to_numpy, score_file


ROOT = Path(__file__).resolve().parent
TMP_ROOT = Path("/tmp/ng_task001_search")


def init(name: str, arr, dtype=np.int64):
    return numpy_helper.from_array(np.asarray(arr, dtype=dtype), name)


def make_model(nodes, inits, *, name: str, opset: int = 14) -> onnx.ModelProto:
    input_vi = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, [1, 10, 30, 30])
    output_vi = helper.make_tensor_value_info(OUT_NAME, TensorProto.UINT8, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, name, [input_vi], [output_vi], initializer=inits)
    return helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", opset)],
        ir_version=7,
    )


def build_expand_scalar() -> onnx.ModelProto:
    inits = [
        init("starts", [0, 0, 0, 0]),
        init("ends", [1, 1, 1, 1]),
        init("shape", [1, 10, 30, 30]),
    ]
    nodes = [
        helper.make_node("Slice", [IN_NAME, "starts", "ends"], ["cell"], name="slice_one_cell"),
        helper.make_node("Cast", ["cell"], ["cell_u8"], name="cast_cell_u8", to=TensorProto.UINT8),
        helper.make_node("Expand", ["cell_u8", "shape"], [OUT_NAME], name="expand_direct_output"),
    ]
    return make_model(nodes, inits, name="task001_expand_scalar_direct")


def build_tile_core() -> onnx.ModelProto:
    inits = [
        init("starts", [0, 0, 0, 0]),
        init("ends", [1, 10, 3, 3]),
        init("repeats", [1, 1, 10, 10]),
    ]
    nodes = [
        helper.make_node("Slice", [IN_NAME, "starts", "ends"], ["core"], name="slice_full_core"),
        helper.make_node("Cast", ["core"], ["core_u8"], name="cast_core_u8", to=TensorProto.UINT8),
        helper.make_node("Tile", ["core_u8", "repeats"], [OUT_NAME], name="tile_direct_output"),
    ]
    return make_model(nodes, inits, name="task001_tile_core_direct")


def build_resize_core() -> onnx.ModelProto:
    inits = [
        init("starts", [0, 0, 0, 0]),
        init("ends", [1, 10, 3, 3]),
        init("scales", [1.0, 1.0, 10.0, 10.0], dtype=np.float32),
    ]
    nodes = [
        helper.make_node("Slice", [IN_NAME, "starts", "ends"], ["core"], name="slice_full_core"),
        helper.make_node("Cast", ["core"], ["core_u8"], name="cast_core_u8", to=TensorProto.UINT8),
        helper.make_node(
            "Resize",
            ["core_u8", "", "scales"],
            [OUT_NAME],
            name="resize_direct_output",
            mode="nearest",
        ),
    ]
    return make_model(nodes, inits, name="task001_resize_core_direct")


def build_pad_bg_core() -> onnx.ModelProto:
    inits = [
        init("starts", [0, 0, 0, 0]),
        init("ends", [1, 1, 3, 3]),
        init("pads", [0, 0, 0, 0, 0, 9, 27, 27]),
    ]
    nodes = [
        helper.make_node("Slice", [IN_NAME, "starts", "ends"], ["bg_core"], name="slice_bg_core"),
        helper.make_node("Cast", ["bg_core"], ["bg_core_u8"], name="cast_bg_core_u8", to=TensorProto.UINT8),
        helper.make_node("Pad", ["bg_core_u8", "pads"], [OUT_NAME], name="pad_direct_output", mode="constant"),
    ]
    return make_model(nodes, inits, name="task001_pad_bg_core_direct")


def correctness(path: Path) -> str:
    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        data = json.loads((ROOT / "data" / "task001.json").read_text(encoding="utf-8"))
        parts = []
        for split in ("train", "test", "arc-gen"):
            ok = 0
            total = 0
            for example in data.get(split, []):
                inp = convert_to_numpy(example, "input")
                expected = convert_to_numpy(example, "output")
                if inp is None or expected is None:
                    continue
                pred = session.run([OUT_NAME], {IN_NAME: inp})[0]
                ok += bool(np.array_equal(pred > 0, expected > 0))
                total += 1
            parts.append(f"{split} {ok}/{total}")
        return ", ".join(parts)
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}"


def largest_internal(path: Path) -> str:
    try:
        _, tensor_memory, _, _ = analyze(path, fast=True)
        scored = [t for t in tensor_memory.values() if t.scored]
        if not scored:
            return "none"
        largest = max(scored, key=lambda t: t.bytes)
        return f"{largest.name} {largest.bytes} B {largest.dtype} {largest.shape}"
    except Exception:
        return "n/a"


def benchmark(name: str, builder: Callable[[], onnx.ModelProto], idea: str) -> dict[str, object]:
    path = TMP_ROOT / name / "task001.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "variant": name,
        "idea": idea,
        "valid": False,
        "correctness": "not run",
        "memory": None,
        "params": None,
        "cost": None,
        "score": None,
        "largest": "n/a",
        "note": "",
    }
    try:
        model = builder()
        onnx.checker.check_model(model)
        onnx.save(model, path)
        score = score_file(path)
        result.update(
            valid=score["valid"],
            memory=score["memory"],
            params=score["params"],
            cost=score["cost"],
            score=f"{score['score']:.6f}" if score["score"] is not None else None,
            note="" if score["valid"] else str(score["error"]).splitlines()[-1],
        )
        result["correctness"] = correctness(path)
        result["largest"] = largest_internal(path) if score["valid"] else "n/a"
    except Exception:
        result["note"] = traceback.format_exc().splitlines()[-1]
    return result


def main() -> None:
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)
    rows = [
        benchmark("F", lambda: build_model("F"), "current correct fallback"),
        benchmark("G_einsum", lambda: build_model("G"), "direct 30x30 Einsum output"),
        benchmark("expand_scalar", build_expand_scalar, "Expand directly to graph output"),
        benchmark("tile_core", build_tile_core, "Tile 3x3 core directly to graph output"),
        benchmark("resize_core", build_resize_core, "Resize 3x3 core directly to graph output"),
        benchmark("pad_bg_core", build_pad_bg_core, "Pad tiny bg core directly to graph output"),
    ]

    headers = ["variant", "idea", "valid", "correctness", "memory", "params", "cost", "score", "largest", "note"]
    print(" | ".join(headers))
    print(" | ".join("---" for _ in headers))
    for row in rows:
        print(" | ".join(str(row[h]) for h in headers))

    print()
    print(f"score 19 requires cost <= {math.exp(6):.3f}")


if __name__ == "__main__":
    main()
