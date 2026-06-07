"""Compact ONNX for ARC task106: quadrant rotation tiling.

Task rule: the input is a square N×N grid (N is 2 or 3 in this dataset). The
output is a 2N×2N grid formed by four N×N blocks: top-left is the original,
top-right is rotated 90° clockwise, bottom-left is 90° counterclockwise, and
bottom-right is 180°. Everything outside the logical output stays unactivated
on the 30×30 competition canvas.

ONNX: ArgMax a 3×3 crop to compact color ids, cast ids to int32, Gather with
precomputed rotation indices for N=2 and N=3, broadcast Equal for one-hot decode
with 255 padding on inactive slots, then Pad once to float 30×30 output.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_NUM = "106"
TASK_ID = f"task{TASK_NUM}"
TASK_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
FULL_SHAPE = [1, 10, 30, 30]
C = 10
MAX_N = 3
OUT_MAX = 2 * MAX_N
IR_VERSION = 10


def _solve(grid: np.ndarray) -> np.ndarray:
    m = np.asarray(grid)
    tl = m
    tr = np.rot90(m, k=-1)
    bl = np.rot90(m, k=1)
    br = np.rot90(m, k=2)
    top = np.concatenate([tl, tr], axis=1)
    bot = np.concatenate([bl, br], axis=1)
    return np.concatenate([top, bot], axis=0)


def _gather_indices(n: int) -> list[int]:
    m = np.arange(n * n, dtype=np.int64).reshape(n, n)
    return [int(v) for v in _solve(m).reshape(-1)]


def _remap_indices(idxs: list[int], n: int, max_n: int = MAX_N) -> list[int]:
    out: list[int] = []
    for idx in idxs:
        r, c = divmod(idx, n)
        out.append(r * max_n + c)
    return out


IDX2 = _remap_indices(_gather_indices(2), 2)
IDX3 = _gather_indices(3)
SENTINEL_INDEX = MAX_N * MAX_N


def _idx2_padded() -> list[int]:
    out = np.full((OUT_MAX, OUT_MAX), SENTINEL_INDEX, dtype=np.int64)
    out[:4, :4] = np.asarray(IDX2, dtype=np.int64).reshape(4, 4)
    return [int(v) for v in out.reshape(-1)]


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.counter = 0

    def init_i64(self, name: str, values: list[int] | np.ndarray) -> str:
        self.initializers.append(
            numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)
        )
        return name

    def init_f32(self, name: str, values: list[float] | np.ndarray) -> str:
        self.initializers.append(
            numpy_helper.from_array(np.asarray(values, dtype=np.float32), name=name)
        )
        return name

    def init_i32(self, name: str, values: list[int] | np.ndarray) -> str:
        self.initializers.append(
            numpy_helper.from_array(np.asarray(values, dtype=np.int32), name=name)
        )
        return name

    def name(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def node(self, op_type: str, inputs: list[str], prefix: str, **attrs: Any) -> str:
        out = self.name(prefix)
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out


def make_model(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    *,
    opset: int,
) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, FULL_SHAPE)],
        initializers,
    )
    model = helper.make_model(
        graph,
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def add_output_pad(b: Builder, x: str, *, opset: int) -> None:
    pads = [0, 0, 0, 0, 0, 0, 30 - OUT_MAX, 30 - OUT_MAX]
    if opset >= 11:
        b.init_i64("pad_to_30", pads)
        b.nodes.append(helper.make_node("Pad", [x, "pad_to_30"], [OUT_NAME], mode="constant"))
    else:
        b.nodes.append(
            helper.make_node("Pad", [x], [OUT_NAME], mode="constant", pads=pads, value=0.0)
        )


def _detect_is3(b: Builder, crop: str) -> str:
    b.init_i64("cell22_st", [0, 0, 2, 2])
    cell22 = b.node("Slice", [crop, "cell22_st", "crop_en", "axes4"], "cell22")
    cell22_max = b.node("ReduceMax", [cell22], "cell22_max", axes=[1, 2, 3], keepdims=0)
    return b.node("Cast", [cell22_max], "is3", to=TensorProto.BOOL)


def build_model(*, opset: int = 10) -> onnx.ModelProto:
    """ArgMax 3×3 crop, int32 dual Gather for N=2/3, broadcast Equal one-hot."""
    b = Builder()
    b.init_i64("crop_st", [0, 0, 0, 0])
    b.init_i64("crop_en", [1, C, MAX_N, MAX_N])
    b.init_i64("axes4", [0, 1, 2, 3])
    b.init_i64("flat9", [MAX_N * MAX_N])
    b.init_i64("idx2", np.asarray(_idx2_padded(), dtype=np.int64).reshape(1, 1, OUT_MAX, OUT_MAX))
    b.init_i64("idx3", np.asarray(IDX3, dtype=np.int64).reshape(1, 1, OUT_MAX, OUT_MAX))
    b.init_i32("sentinel", [255])
    b.init_i32("ch4", np.arange(C, dtype=np.int32).reshape(1, C, 1, 1))

    crop = b.node("Slice", [IN_NAME, "crop_st", "crop_en", "axes4"], "crop")
    is3 = _detect_is3(b, crop)
    ids = b.node("ArgMax", [crop], "ids", axis=1, keepdims=1)
    ids_i32 = b.node("Cast", [ids], "ids_i32", to=TensorProto.INT32)
    flat = b.node("Reshape", [ids_i32, "flat9"], "flat")
    flat10 = b.node("Concat", [flat, "sentinel"], "flat10", axis=0)

    c4p = b.node("Gather", [flat10, "idx2"], "c4p", axis=0)
    c6 = b.node("Gather", [flat, "idx3"], "c6", axis=0)
    colors = b.node("Where", [is3, c6, c4p], "colors")

    onehot = b.node("Equal", ["ch4", colors], "onehot")
    out_f = b.node("Cast", [onehot], "out_f", to=TensorProto.FLOAT)
    add_output_pad(b, out_f, opset=opset)
    return make_model(b.nodes, b.initializers, opset=opset)


def build_variants() -> dict[str, onnx.ModelProto]:
    return {
        "gather_op10": build_model(opset=10),
        "gather_op11": build_model(opset=11),
    }


def load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with TASK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def verify_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    split_counts: dict[str, tuple[int, int]] = {}
    all_ok = True
    for split, examples in load_task_data().items():
        passed = 0
        checked = 0
        for example in examples:
            inp = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if inp is None or expected is None:
                continue
            pred = session.run([OUT_NAME], {IN_NAME: inp})[0]
            checked += 1
            if np.array_equal(pred > 0.0, expected > 0.0):
                passed += 1
            else:
                all_ok = False
        split_counts[split] = (passed, checked)
    return all_ok, split_counts


def write_model(model: onnx.ModelProto, path: Path) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(path))


def benchmark_variants() -> tuple[dict[str, dict[str, Any]], str, dict[str, onnx.ModelProto]]:
    variants = build_variants()
    results: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}_") as tmp:
        tmp_path = Path(tmp)
        for name, model in variants.items():
            ok, splits = verify_correct(model)
            path = tmp_path / f"{TASK_ID}.onnx"
            write_model(model, path)
            result = score_file(path)
            result["correct"] = ok
            result["splits"] = splits
            results[name] = result

    def sort_key(name: str) -> int:
        result = results[name]
        if not result["valid"] or not result["correct"]:
            return 10**18
        return int(result["cost"])

    best_name = min(results, key=sort_key)
    if sort_key(best_name) >= 10**18:
        raise RuntimeError(f"no valid correct variant: {results}")
    return results, best_name, variants


def print_benchmark(results: dict[str, dict[str, Any]], best_name: str) -> None:
    print(f"{'variant':<16} {'ok':<5} {'valid':<6} {'memory':>8} {'params':>7} {'cost':>8} {'score':>10}")
    for name, result in sorted(results.items(), key=lambda item: (item[0] != best_name, item[0])):
        score = result["score"]
        score_text = f"{score:.6f}" if isinstance(score, float) else "None"
        print(
            f"{name:<16} {str(result['correct']):<5} {str(result['valid']):<6} "
            f"{str(result['memory']):>8} {str(result['params']):>7} "
            f"{str(result['cost']):>8} {score_text:>10}"
        )
    best = results[best_name]
    print(f"\nbest={best_name} cost={best['cost']} score={best['score']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Build and score {TASK_ID}")
    parser.add_argument("--benchmark", action="store_true", help="Compare all variants")
    _ = parser.parse_args()

    results, best_name, variants = benchmark_variants()
    print_benchmark(results, best_name)
    write_model(variants[best_name], BEST_PATH)
    print(f"wrote {BEST_PATH}")


if __name__ == "__main__":
    main()
