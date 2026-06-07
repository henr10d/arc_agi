"""Dynamic ONNX solver for ARC task110 periodic hole filling.

Task rule: the 29x29 input is a nonzero repeating color pattern with black
holes.  Detect the smallest visible same-column and same-row periods, then
fill each black cell from a nonzero cell one or two periods away while leaving
all existing nonzero cells unchanged.  The final output is padded back to the
NeuroGolf 30x30 one-hot contract.
"""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "task110"
TASK_NUM = 110
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
TMP_DIR = OUT_DIR / "_task110_variants"

H = W = 29
FULL = 30
Y_PERIODS = (4, 5, 6, 7, 8, 9)
X_PERIODS = (5, 6, 9)
Y_SOURCES = {
    4: ((1, -1),),
    5: ((1, -1), (1, 1), (2, -1), (2, 1)),
    6: ((1, -1), (1, 1), (2, -1), (2, 1)),
    7: ((1, -1), (1, 1), (2, -1), (2, 1)),
    8: ((1, -1), (1, 1), (2, -1), (2, 1)),
    9: ((1, -1), (1, 1), (2, -1), (2, 1)),
}
X_SOURCES = {5: ((1, 1),), 6: ((1, 1),), 9: ((1, 1),)}


def load_score_model() -> Any:
    spec = importlib.util.spec_from_file_location("score_model", ROOT / "score_model.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import score_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_task() -> dict[str, Any]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def one_hot(grid: list[list[int]]) -> np.ndarray:
    arr = np.zeros((1, 10, FULL, FULL), dtype=np.float32)
    for y, row in enumerate(grid):
        for x, color in enumerate(row):
            arr[0, int(color), y, x] = 1.0
    return arr


def examples() -> list[dict[str, list[list[int]]]]:
    data = load_task()
    out: list[dict[str, list[list[int]]]] = []
    for split in ("train", "test", "arc-gen"):
        out.extend(data.get(split, []))
    return out


class Builder:
    def __init__(self, name: str) -> None:
        self.name = name
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self.seen_init: set[str] = set()
        self.counter = 0

    def n(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def init(self, name: str, array: np.ndarray) -> str:
        if name not in self.seen_init:
            self.inits.append(numpy_helper.from_array(array, name))
            self.seen_init.add(name)
        return name

    def scalar_u8(self, value: int) -> str:
        return self.init(f"u8_{value}", np.array(value, dtype=np.uint8))

    def scalar_i64(self, value: int) -> str:
        return self.init(f"i64_{value}", np.array(value, dtype=np.int64))

    def scalar_f32(self, value: float) -> str:
        safe = str(value).replace(".", "_").replace("-", "m")
        return self.init(f"f32_{safe}", np.array(value, dtype=np.float32))

    def scalar_f16(self, value: float) -> str:
        safe = str(value).replace(".", "_").replace("-", "m")
        return self.init(f"f16_{safe}", np.array(value, dtype=np.float16))

    def equal_value(self, a: str, b: str, prefix: str) -> str:
        gt = self.op("Greater", [a, b], f"{prefix}_gt")
        lt = self.op("Less", [a, b], f"{prefix}_lt")
        diff = self.op("Or", [gt, lt], f"{prefix}_diff")
        return self.op("Not", [diff], f"{prefix}_eq")

    def slice4(self, data: str, y0: int, y1: int, x0: int, x1: int, prefix: str) -> str:
        out = self.n(prefix)
        starts = self.init(f"starts4_{y0}_{x0}", np.array([0, 0, y0, x0], dtype=np.int64))
        ends = self.init(f"ends4_1_10_{y1}_{x1}", np.array([1, 10, y1, x1], dtype=np.int64))
        axes = self.init("axes4", np.array([0, 1, 2, 3], dtype=np.int64))
        self.nodes.append(helper.make_node("Slice", [data, starts, ends, axes], [out], name=out))
        return out

    def slice3(self, data: str, y0: int, y1: int, x0: int, x1: int, prefix: str) -> str:
        out = self.n(prefix)
        starts = self.init(f"starts3_{y0}_{x0}", np.array([0, y0, x0], dtype=np.int64))
        ends = self.init(f"ends3_1_{y1}_{x1}", np.array([1, y1, x1], dtype=np.int64))
        axes = self.init("axes3", np.array([0, 1, 2], dtype=np.int64))
        self.nodes.append(helper.make_node("Slice", [data, starts, ends, axes], [out], name=out))
        return out

    def pad3(self, data: str, top: int, left: int, bottom: int, right: int, prefix: str) -> str:
        out = self.n(prefix)
        pads = [0, top, left, 0, bottom, right]
        self.nodes.append(helper.make_node("Pad", [data], [out], name=out, mode="constant", pads=pads, value=0.0))
        return out

    def op(self, op_type: str, inputs: list[str], prefix: str, **attrs: Any) -> str:
        out = self.n(prefix)
        self.nodes.append(helper.make_node(op_type, inputs, [out], name=out, **attrs))
        return out

    def same_y_ok(self, grid: str, p: int) -> str:
        a = self.slice3(grid, 8, 18, 27, 28, f"ya{p}")
        b = self.slice3(grid, p + 8, p + 18, 27, 28, f"yb{p}")
        return self.period_ok(a, b, 10, f"yok{p}")

    def same_x_ok(self, grid: str, p: int) -> str:
        a = self.slice3(grid, 0, H, 0, W - p, f"xa{p}")
        b = self.slice3(grid, 0, H, p, W, f"xb{p}")
        return self.period_ok(a, b, H * (W - p), f"xok{p}")

    def period_ok(self, a: str, b: str, size: int, prefix: str) -> str:
        zero = self.scalar_f16(0.0)
        za = self.op("Greater", [a, zero], f"{prefix}_za")
        zb = self.op("Greater", [b, zero], f"{prefix}_zb")
        both = self.op("And", [za, zb], f"{prefix}_both")
        neq = self.op("Or", [self.op("Greater", [a, b], f"{prefix}_gt"), self.op("Less", [a, b], f"{prefix}_lt")], f"{prefix}_neq")
        bad = self.op("And", [both, neq], f"{prefix}_bad")
        bad_i = self.op("Cast", [bad], f"{prefix}_badf", to=TensorProto.FLOAT16)
        bad_count = self.op("ReduceSum", [bad_i], f"{prefix}_sum", axes=[0, 1, 2], keepdims=0)
        has_bad = self.op("Greater", [bad_count, self.scalar_f16(0.0)], f"{prefix}_hasbad")
        return self.op("Not", [has_bad], f"{prefix}_ok")

    def shifted_source(self, grid: str, axis: str, delta: int, direction: int) -> str:
        if axis == "y" and direction < 0:
            return self.pad3(self.slice3(grid, 0, H - delta, 0, W, f"src_y_up_s{delta}"), delta, 0, 0, 0, f"src_y_up_p{delta}")
        if axis == "y":
            return self.pad3(self.slice3(grid, delta, H, 0, W, f"src_y_dn_s{delta}"), 0, 0, delta, 0, f"src_y_dn_p{delta}")
        if direction < 0:
            return self.pad3(self.slice3(grid, 0, H, 0, W - delta, f"src_x_l_s{delta}"), 0, delta, 0, 0, f"src_x_l_p{delta}")
        return self.pad3(self.slice3(grid, 0, H, delta, W, f"src_x_r_s{delta}"), 0, 0, 0, delta, f"src_x_r_p{delta}")

    def fill_from(self, current: str, source: str, period_ok: str, prefix: str) -> str:
        zero = self.scalar_f16(0.0)
        current_zero = self.op("Not", [self.op("Greater", [current, zero], f"{prefix}_cpos")], f"{prefix}_cz")
        source_nonzero = self.op("Greater", [source, zero], f"{prefix}_snz")
        usable = self.op("And", [current_zero, source_nonzero], f"{prefix}_usable")
        gated = self.op("And", [usable, period_ok], f"{prefix}_gated")
        return self.op("Where", [gated, source, current], f"{prefix}_where")

    def first_nonzero_source(self, sources: list[str], prefix: str) -> str:
        zero = self.scalar_f16(0.0)
        current = sources[0]
        for index, source in enumerate(sources[1:], start=1):
            current_zero = self.op("Not", [self.op("Greater", [current, zero], f"{prefix}_{index}_cpos")], f"{prefix}_{index}_cz")
            source_nonzero = self.op("Greater", [source, zero], f"{prefix}_{index}_snz")
            take = self.op("And", [current_zero, source_nonzero], f"{prefix}_{index}_take")
            current = self.op("Where", [take, source, current], f"{prefix}_{index}_where")
        return current

    def fill_period(self, current: str, grid: str, axis: str, p: int, period_ok: str) -> str:
        sources: list[str] = []
        source_map = Y_SOURCES if axis == "y" else X_SOURCES
        for mul, direction in source_map[p]:
            delta = p * mul
            if delta < H:
                sources.append(self.shifted_source(grid, axis, delta, direction))
        candidate = self.first_nonzero_source(sources, f"cand_{axis}{p}")
        return self.fill_from(current, candidate, period_ok, f"fill_{axis}{p}")

    def finalize(self, grid: str) -> None:
        self.nodes.append(
            helper.make_node(
                "Pad",
                [grid],
                ["grid30h"],
                name="grid30h",
                mode="constant",
                pads=[0, 0, 0, 0, 1, 1],
                value=-1.0,
            )
        )
        grid_i = self.op("Cast", ["grid30h"], "grid30i", to=TensorProto.INT32)
        unsq = self.op("Unsqueeze", [grid_i], "unsq", axes=[1])
        colors = self.init("colors", np.arange(10, dtype=np.int32).reshape(1, 10, 1, 1))
        hot = self.op("Equal", [unsq, colors], "hot")
        self.nodes.append(helper.make_node("Cast", [hot], ["output"], name="output", to=TensorProto.FLOAT))

    def model(self, fill_order: tuple[str, ...]) -> onnx.ModelProto:
        input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, FULL, FULL])
        output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, FULL, FULL])

        arg_full = self.op("ArgMax", ["input"], "arg", axis=1, keepdims=0)
        arg = self.slice3(arg_full, 0, H, 0, W, "arg_core")
        grid = self.op("Cast", [arg], "grid", to=TensorProto.FLOAT16)

        y_ok = {p: self.same_y_ok(grid, p) for p in Y_PERIODS}
        x_ok = {p: y_ok[p] for p in X_PERIODS}

        current = grid
        for axis in fill_order:
            periods = Y_PERIODS if axis == "y" else X_PERIODS
            ok_map = y_ok if axis == "y" else x_ok
            for p in periods:
                current = self.fill_period(current, grid, axis, p, ok_map[p])

        self.finalize(current)
        graph = helper.make_graph(self.nodes, self.name, [input_info], [output_info], self.inits)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
        model.ir_version = 10
        return model


def build_model(path: Path, fill_order: tuple[str, ...]) -> None:
    builder = Builder(path.stem)
    model = builder.model(fill_order)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(inferred, path)


def validate(path: Path) -> tuple[bool, str]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for idx, example in enumerate(examples()):
        pred = session.run(["output"], {"input": one_hot(example["input"])})[0]
        expected = one_hot(example["output"])
        if not np.array_equal(pred > 0, expected > 0):
            return False, f"example {idx} mismatch"
    return True, "ok"


def initializer_scalars(path: Path) -> int:
    model = onnx.load(str(path))
    return int(sum(math.prod(init.dims) for init in model.graph.initializer))


def main() -> None:
    TMP_DIR.mkdir(exist_ok=True)
    score_model = load_score_model()
    candidates = [
        ("dynamic_yx", ("y", "x")),
    ]

    results: list[dict[str, Any]] = []
    for name, order in candidates:
        path = TMP_DIR / f"{TASK_ID}_{name}.onnx"
        build_model(path, order)
        ok, message = validate(path)
        result = score_model.score_file(path)
        result["candidate"] = name
        result["validation"] = message
        result["init_scalars"] = initializer_scalars(path)
        result["correct"] = ok
        results.append(result)
        status = "OK" if ok and result["valid"] else "BAD"
        print(
            f"{name}: {status} validation={message} "
            f"init_scalars={result['init_scalars']} memory={result['memory']} "
            f"params={result['params']} cost={result['cost']} score={result['score']}"
        )

    valid = [r for r in results if r["correct"] and r["valid"]]
    if not valid:
        raise SystemExit("no valid correct task110 candidate")
    best = min(valid, key=lambda r: int(r["cost"]))
    best_path = Path(best["path"])
    shutil.copyfile(best_path, BEST_PATH)
    print(f"kept {best['candidate']} -> {BEST_PATH}")
    print(
        f"best: init_scalars={best['init_scalars']} memory={best['memory']} "
        f"params={best['params']} cost={best['cost']} score={best['score']:.6f}"
    )


if __name__ == "__main__":
    main()
