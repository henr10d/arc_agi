"""Build and score ONNX variants for NeuroGolf ARC task051.

Task rule: input and output share the same H×W. A small arrow/triangle object
(dominant color) contains a unique marker cell (count==1). Keep the input and
draw a straight ray from the marker side outward to the grid boundary using the
marker color: right from a left-side marker, left from a right-side marker,
down from a top marker, otherwise up.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "task051.json"
BEST_PATH = OUT_DIR / "task051.onnx"
MODEL_PATH = OUT_DIR / "model.onnx"

C, H, W = 10, 30, 30
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
IR_VERSION = 10
NC = 9


def load_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def solve_grid(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    counts: dict[int, int] = {}
    for v in g.flat:
        if v:
            counts[int(v)] = counts.get(int(v), 0) + 1
    marker = min(counts, key=counts.get)
    mr, mc = (int(x) for x in np.argwhere(g == marker)[0])
    h, w = g.shape
    obj_mask = g == max(counts, key=counts.get)
    r0 = int(obj_mask.any(1).argmax())
    r1 = int(len(g) - 1 - obj_mask.any(1)[::-1].argmax())
    c0 = int(obj_mask.any(0).argmax())
    c1 = int(len(g[0]) - 1 - obj_mask.any(0)[::-1].argmax())
    first_row = 0
    last_row = h - 1
    first_col = 0
    last_col = w - 1
    bg = g == 0
    if mc == c0:
        sl = out[mr, c1 + 1 : last_col + 1]
        out[mr, c1 + 1 : last_col + 1] = np.where(bg[mr, c1 + 1 : last_col + 1], marker, sl)
    elif mc == c1:
        sl = out[mr, first_col:mc]
        out[mr, first_col:mc] = np.where(bg[mr, first_col:mc], marker, sl)
    elif mr == r0:
        sl = out[r1 + 1 : last_row + 1, mc]
        out[r1 + 1 : last_row + 1, mc] = np.where(bg[r1 + 1 : last_row + 1, mc], marker, sl)
    else:
        sl = out[first_row:r0, mc]
        out[first_row:r0, mc] = np.where(bg[first_row:r0, mc], marker, sl)
    return out


def grid_to_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            color = int(arr[r, c])
            if 0 <= color < C:
                out[0, color, r, c] = 1.0
    return out


def expected_onehot(example: dict[str, list[list[int]]]) -> np.ndarray:
    return grid_to_onehot(solve_grid(example["input"]))


def verify_reference(data: dict[str, list[dict[str, list[list[int]]]]]) -> None:
    for split, examples in data.items():
        for i, ex in enumerate(examples):
            got = solve_grid(ex["input"])
            exp = np.asarray(ex["output"], dtype=np.int64)
            if not np.array_equal(got, exp):
                raise RuntimeError(f"reference solver failed on {split}[{i}]")


def run_pass_rate(model_path: Path, data: dict[str, list[dict[str, list[list[int]]]]]) -> dict[str, tuple[int, int]]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    out: dict[str, tuple[int, int]] = {}
    for split in ("train", "test", "arc-gen"):
        examples = data.get(split, [])
        passed = 0
        for ex in examples:
            pred = session.run([OUT_NAME], {IN_NAME: grid_to_onehot(ex["input"])})[0]
            if np.array_equal((pred > 0).astype(np.float32), expected_onehot(ex)):
                passed += 1
        out[split] = (passed, len(examples))
    return out


@dataclass(frozen=True)
class Variant:
    name: str
    opset: int


class Builder:
    def __init__(self, opset: int) -> None:
        self.opset = opset
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self._n = 0

    def uniq(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    def const(self, name: str, arr: np.ndarray) -> str:
        self.inits.append(numpy_helper.from_array(arr, name=name))
        return name

    def i64(self, name: str, vals: list[int] | np.ndarray) -> str:
        return self.const(name, np.asarray(vals, dtype=np.int64))

    def f32(self, name: str, vals: list[float] | np.ndarray) -> str:
        return self.const(name, np.asarray(vals, dtype=np.float32))

    def add(self, op: str, inputs: list[str], **kwargs: Any) -> str:
        out = self.uniq("n")
        self.nodes.append(helper.make_node(op, inputs, [out], **kwargs))
        return out

    def reduce_sum(self, data: str, axes: list[int], keepdims: int = 1) -> str:
        if self.opset >= 13:
            ax_name = self.uniq("rsax")
            self.inits.append(numpy_helper.from_array(np.asarray(axes, dtype=np.int64), name=ax_name))
            return self.add("ReduceSum", [data, ax_name], keepdims=keepdims)
        return self.add("ReduceSum", [data], axes=axes, keepdims=keepdims)

    def logic_and(self, inputs: list[str]) -> str:
        cur = inputs[0]
        for nxt in inputs[1:]:
            cur = self.add("And", [cur, nxt])
        return cur

    def logic_or(self, inputs: list[str]) -> str:
        cur = inputs[0]
        for nxt in inputs[1:]:
            cur = self.add("Or", [cur, nxt])
        return cur


def build_model(variant: Variant) -> onnx.ModelProto:
    b = Builder(variant.opset)
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    zero_f = b.f32("zero_f", [0.0])
    zero_i = b.i64("zero_i", [0])
    one_f = b.f32("one_f", [1.0])
    one_i = b.i64("one_i", [1])
    big_f = b.f32("big_f", [999.0])
    neg_f = b.f32("neg_f", [-1.0])
    chan_ids = b.i64("chan_ids", np.arange(C, dtype=np.int64).reshape(1, C, 1, 1))

    rows = b.i64("rows", np.arange(H, dtype=np.int64).reshape(1, 1, H, 1))
    cols = b.i64("cols", np.arange(W, dtype=np.int64).reshape(1, 1, 1, W))
    rows_f = b.f32("rows_f", np.arange(H, dtype=np.float32).reshape(1, 1, H, 1))
    cols_f = b.f32("cols_f", np.arange(W, dtype=np.float32).reshape(1, 1, 1, W))

    counts = b.reduce_sum(IN_NAME, [2, 3])
    marker_chan = b.add("Equal", [b.add("Cast", [counts], to=TensorProto.INT64), one_i])
    marker_chan_f = b.add("Cast", [marker_chan], to=TensorProto.FLOAT)
    marker_chan_i = b.add("Cast", [marker_chan], to=TensorProto.INT64)
    marker_color = b.reduce_sum(b.add("Mul", [marker_chan_i, chan_ids]), [1])
    color_grid = b.add("ArgMax", [IN_NAME], axis=1, keepdims=1)
    marker_spatial = b.add("Cast", [b.add("Equal", [color_grid, marker_color])], to=TensorProto.FLOAT)

    mr = b.reduce_sum(b.add("Mul", [marker_spatial, rows_f]), [2, 3])
    mc = b.reduce_sum(b.add("Mul", [marker_spatial, cols_f]), [2, 3])

    fg = b.add("Greater", [color_grid, zero_i])
    fg_f = b.add("Cast", [fg], to=TensorProto.FLOAT)
    row_has_fg = b.add("Greater", [b.add("ReduceMax", [fg_f], axes=[3], keepdims=1), zero_f])
    col_has_fg = b.add("Greater", [b.add("ReduceMax", [fg_f], axes=[2], keepdims=1), zero_f])

    r0 = b.add("ReduceMin", [b.add("Where", [row_has_fg, rows_f, big_f])], axes=[2, 3], keepdims=1)
    r1 = b.add("ReduceMax", [b.add("Where", [row_has_fg, rows_f, neg_f])], axes=[2, 3], keepdims=1)
    c0 = b.add("ReduceMin", [b.add("Where", [col_has_fg, cols_f, big_f])], axes=[2, 3], keepdims=1)
    c1 = b.add("ReduceMax", [b.add("Where", [col_has_fg, cols_f, neg_f])], axes=[2, 3], keepdims=1)

    mr_i = b.add("Cast", [mr], to=TensorProto.INT64)
    mc_i = b.add("Cast", [mc], to=TensorProto.INT64)
    r0_i = b.add("Cast", [r0], to=TensorProto.INT64)
    r1_i = b.add("Cast", [r1], to=TensorProto.INT64)
    c0_i = b.add("Cast", [c0], to=TensorProto.INT64)
    c1_i = b.add("Cast", [c1], to=TensorProto.INT64)

    col_grid = b.add("Greater", [b.reduce_sum(IN_NAME, [1, 2]), zero_f])
    row_grid = b.add("Greater", [b.reduce_sum(IN_NAME, [1, 3]), zero_f])
    last_col = b.add(
        "Cast",
        [b.add("Sub", [b.reduce_sum(b.add("Cast", [col_grid], to=TensorProto.FLOAT), [3]), one_f])],
        to=TensorProto.INT64,
    )
    last_row = b.add(
        "Cast",
        [b.add("Sub", [b.reduce_sum(b.add("Cast", [row_grid], to=TensorProto.FLOAT), [2]), one_f])],
        to=TensorProto.INT64,
    )

    go_right = b.add("Equal", [mc_i, c0_i])
    go_left = b.add("Equal", [mc_i, c1_i])
    go_down = b.add("Equal", [mr_i, r0_i])
    not_side = b.add("Not", [b.logic_or([go_right, go_left])])
    go_down = b.add("And", [go_down, not_side])
    go_up = b.add("Not", [b.logic_or([go_right, go_left, go_down])])

    same_row = b.add("Equal", [rows, mr_i])
    same_col = b.add("Equal", [cols, mc_i])
    gt_c1 = b.add("Greater", [cols, c1_i])
    lt_c0 = b.add("Less", [cols, c0_i])
    le_last_col = b.add("Not", [b.add("Greater", [cols, last_col])])
    gt_r1 = b.add("Greater", [rows, r1_i])
    lt_r0 = b.add("Less", [rows, r0_i])
    le_last_row = b.add("Not", [b.add("Greater", [rows, last_row])])

    line_r = b.add("And", [b.logic_and([go_right, gt_c1, le_last_col]), same_row])
    line_l = b.add("And", [b.add("And", [go_left, lt_c0]), same_row])
    line_d = b.add("And", [b.logic_and([go_down, gt_r1, le_last_row]), same_col])
    line_u = b.add("And", [b.add("And", [go_up, lt_r0]), same_col])
    line = b.logic_or([line_r, line_l, line_d, line_u])
    b.nodes.append(helper.make_node("Where", [line, marker_chan_f, IN_NAME], [OUT_NAME]))

    graph = helper.make_graph(b.nodes, f"task051_{variant.name}", [x_info], [y_info], initializer=b.inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", variant.opset)],
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def evaluate_variant(variant: Variant, data: dict[str, list[dict[str, list[list[int]]]]], workdir: Path) -> dict[str, Any]:
    path = workdir / f"task051_{variant.name}_opset{variant.opset}.onnx"
    result: dict[str, Any] = {"variant": variant, "path": path}
    try:
        model = build_model(variant)
        onnx.save(model, path)
        passes = run_pass_rate(path, data)
        scored = score_file(path)
        result.update(scored)
        result["passes"] = passes
        result["all_pass"] = all(passes[s][0] == passes[s][1] for s in passes)
    except Exception as exc:
        result["valid"] = False
        result["error"] = repr(exc)
        result["all_pass"] = False
    return result


def print_result(result: dict[str, Any]) -> None:
    v = result["variant"]
    passes = result.get("passes", {})
    train = passes.get("train", (0, 0))
    test = passes.get("test", (0, 0))
    arc = passes.get("arc-gen", (0, 0))
    score_txt = result.get("score")
    if isinstance(score_txt, float):
        score_txt = f"{score_txt:.6f}"
    print(
        f"{v.name:<12} opset={v.opset:<2} valid={result.get('valid', False)} "
        f"train={train[0]}/{train[1]} test={test[0]}/{test[1]} arc-gen={arc[0]}/{arc[1]} "
        f"params={result.get('params')} memory={result.get('memory')} "
        f"cost={result.get('cost')} score={score_txt}"
    )
    if result.get("error"):
        print(f"  error: {str(result['error']).splitlines()[-1]}")


def main() -> None:
    data = load_data()
    verify_reference(data)
    print("reference solver: OK on all bundled examples")

    variants = [
        Variant("bbox_rule_a", 10),
        Variant("bbox_rule_b", 11),
        Variant("bbox_rule_c", 12),
        Variant("bbox_rule_d", 13),
    ]

    best: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="task051_") as tmp:
        workdir = Path(tmp)
        for variant in variants:
            result = evaluate_variant(variant, data, workdir)
            print_result(result)
            if result.get("valid") and result.get("all_pass"):
                if best is None or int(result["cost"]) < int(best["cost"]):
                    best = result

        if best is None:
            raise SystemExit("no valid fully-correct candidate found")

        shutil.copyfile(best["path"], BEST_PATH)
        shutil.copyfile(best["path"], MODEL_PATH)
        v = best["variant"]
        print()
        print(
            f"selected {v.name} opset={v.opset}: memory={best['memory']} params={best['params']} "
            f"cost={best['cost']} score={best['score']:.6f}"
        )
        print(f"wrote {BEST_PATH.relative_to(ROOT)} and {MODEL_PATH.relative_to(ROOT)}")

    final = score_file(BEST_PATH)
    print(
        f"persisted: memory={final['memory']} params={final['params']} "
        f"cost={final['cost']} score={final['score']:.6f}"
    )


if __name__ == "__main__":
    main()
