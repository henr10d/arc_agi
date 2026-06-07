"""Build optimized ONNX solvers for NeuroGolf task097.

Task rule: for each variable-size grid padded inside the ARC one-hot tensor,
remove every nonzero cell that has no nonzero neighbor in its surrounding 3x3
window. Adjacent horizontal, vertical, or diagonal foreground pixels keep their
original positions and colors; background remains black inside the task grid
and padded cells remain all-zero.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from score_all_onnx import verify_correctness  # noqa: E402
from score_model import print_report, score_file  # noqa: E402

TASK_JSON = ROOT / "data" / "task097.json"
OUT_PATH = ROOT / "task097.onnx"

C = 10
H = W = 30
GH = GW = 20
PAD_H = H - GH
PAD_W = W - GW
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def init(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def scalar_f(inits: list[onnx.TensorProto], name: str, value: float) -> str:
    return init(inits, name, np.asarray([value], dtype=np.float32))


def make_value_infos() -> tuple[onnx.ValueInfoProto, onnx.ValueInfoProto]:
    x = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    return x, y


def finish_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
    x, y = make_value_infos()
    graph = helper.make_graph(nodes, name, [x], [y], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def add_common_output(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    dense: str,
    fg_mask: str,
    active: str,
    zero: str,
    *,
    prefix: str,
) -> None:
    """Apply compact keep mask to one-hot colors and pad to competition shape."""
    starts_bg = init(inits, f"{prefix}_out_s_bg", np.asarray([0, 0, 0, 0], dtype=np.int64))
    ends_bg = init(inits, f"{prefix}_out_e_bg", np.asarray([1, 1, GH, GW], dtype=np.int64))

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts_bg, ends_bg], [f"{prefix}_ch0"]),
            helper.make_node("Greater", [f"{prefix}_ch0", zero], [f"{prefix}_out_ch0b"]),
            helper.make_node("Greater", [active, zero], [f"{prefix}_out_activeb"]),
            helper.make_node("Not", [dense], [f"{prefix}_out_not_dense"]),
            helper.make_node("And", [fg_mask, f"{prefix}_out_not_dense"], [f"{prefix}_out_removed"]),
            helper.make_node("Or", [f"{prefix}_out_ch0b", f"{prefix}_out_removed"], [f"{prefix}_out_bg"]),
            helper.make_node("And", [f"{prefix}_out_activeb", dense], [f"{prefix}_out_fg"]),
            helper.make_node("Concat", [f"{prefix}_out_bg", f"{prefix}_out_fg"], [f"{prefix}_outb"], axis=1),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Cast", [f"{prefix}_outb"], [f"{prefix}_out20"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                [f"{prefix}_out20"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, PAD_H, PAD_W],
            ),
        ]
    )


def add_common_output_from_argmax(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    color_idx: str,
    dense: str,
    fg_mask: str,
    zero_f: str,
    *,
    prefix: str,
) -> None:
    """Build compact one-hot output from int64 color indices and a keep mask."""
    starts_bg = init(inits, f"{prefix}_out_s_bg", np.asarray([0, 0, 0, 0], dtype=np.int64))
    ends_bg = init(inits, f"{prefix}_out_e_bg", np.asarray([1, 1, GH, GW], dtype=np.int64))
    color_values = init(inits, f"{prefix}_colors", np.arange(1, C, dtype=np.int64).reshape(1, C - 1, 1, 1))

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts_bg, ends_bg], [f"{prefix}_ch0"]),
            helper.make_node("Greater", [f"{prefix}_ch0", zero_f], [f"{prefix}_out_ch0b"]),
            helper.make_node("Equal", [color_idx, color_values], [f"{prefix}_out_activeb"]),
            helper.make_node("Not", [dense], [f"{prefix}_out_not_dense"]),
            helper.make_node("And", [fg_mask, f"{prefix}_out_not_dense"], [f"{prefix}_out_removed"]),
            helper.make_node("Or", [f"{prefix}_out_ch0b", f"{prefix}_out_removed"], [f"{prefix}_out_bg"]),
            helper.make_node("And", [f"{prefix}_out_activeb", dense], [f"{prefix}_out_fg"]),
            helper.make_node("Concat", [f"{prefix}_out_bg", f"{prefix}_out_fg"], [f"{prefix}_outb"], axis=1),
            helper.make_node("Cast", [f"{prefix}_outb"], [f"{prefix}_out20"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                [f"{prefix}_out20"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, PAD_H, PAD_W],
            ),
        ]
    )


def build_argmax_model() -> onnx.ModelProto:
    """Use compact int64 color indices instead of a 9-channel active-color slice."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    s = init(inits, "arg_s", np.asarray([0, 0, 0, 0], dtype=np.int64))
    e = init(inits, "arg_e", np.asarray([1, 1, GH, GW], dtype=np.int64))
    zero_i = init(inits, "arg_zero_i", np.asarray([0], dtype=np.int64))
    zero_f = scalar_f(inits, "arg_zero_f", 0.0)
    threshold = scalar_f(inits, "arg_threshold", 0.15)
    nodes.extend(
        [
            helper.make_node("ArgMax", [IN_NAME], ["arg_full"], axis=1, keepdims=1),
            helper.make_node("Slice", ["arg_full", s, e], ["arg_idx"]),
            helper.make_node("Greater", ["arg_idx", zero_i], ["arg_fgb"]),
            helper.make_node("Cast", ["arg_fgb"], ["arg_fg"], to=TensorProto.FLOAT),
            helper.make_node(
                "AveragePool",
                ["arg_fg"],
                ["arg_pool"],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                strides=[1, 1],
                count_include_pad=1,
            ),
            helper.make_node("Greater", ["arg_pool", threshold], ["arg_dense"]),
        ]
    )
    add_common_output_from_argmax(nodes, inits, "arg_idx", "arg_dense", "arg_fgb", zero_f, prefix="arg")
    return finish_model(nodes, inits, "task097_argmax")


def add_common_full_output_from_argmax(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    color_idx: str,
    dense: str,
    fg_mask: str,
    zero_f: str,
    *,
    prefix: str,
) -> None:
    """Build full-size bool one-hot output and cast it directly to graph output."""
    starts_bg = init(inits, f"{prefix}_out_s_bg", np.asarray([0, 0, 0, 0], dtype=np.int64))
    ends_bg = init(inits, f"{prefix}_out_e_bg", np.asarray([1, 1, H, W], dtype=np.int64))
    color_values = init(inits, f"{prefix}_colors", np.arange(1, C, dtype=np.int64).reshape(1, C - 1, 1, 1))

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts_bg, ends_bg], [f"{prefix}_ch0"]),
            helper.make_node("Greater", [f"{prefix}_ch0", zero_f], [f"{prefix}_out_ch0b"]),
            helper.make_node("Equal", [color_idx, color_values], [f"{prefix}_out_activeb"]),
            helper.make_node("Not", [dense], [f"{prefix}_out_not_dense"]),
            helper.make_node("And", [fg_mask, f"{prefix}_out_not_dense"], [f"{prefix}_out_removed"]),
            helper.make_node("Or", [f"{prefix}_out_ch0b", f"{prefix}_out_removed"], [f"{prefix}_out_bg"]),
            helper.make_node("And", [f"{prefix}_out_activeb", dense], [f"{prefix}_out_fg"]),
            helper.make_node("Concat", [f"{prefix}_out_bg", f"{prefix}_out_fg"], [f"{prefix}_outb"], axis=1),
            helper.make_node("Cast", [f"{prefix}_outb"], [OUT_NAME], to=TensorProto.FLOAT),
        ]
    )


def build_argmax_full_model() -> onnx.ModelProto:
    """Use full-grid color indices so the final cast writes directly to output."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    zero_i = init(inits, "argfull_zero_i", np.asarray([0], dtype=np.int64))
    zero_f = scalar_f(inits, "argfull_zero_f", 0.0)
    threshold = scalar_f(inits, "argfull_threshold", 0.15)
    nodes.extend(
        [
            helper.make_node("ArgMax", [IN_NAME], ["argfull_idx"], axis=1, keepdims=1),
            helper.make_node("Greater", ["argfull_idx", zero_i], ["argfull_fgb"]),
            helper.make_node("Cast", ["argfull_fgb"], ["argfull_fg"], to=TensorProto.FLOAT),
            helper.make_node(
                "AveragePool",
                ["argfull_fg"],
                ["argfull_pool"],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                strides=[1, 1],
                count_include_pad=1,
            ),
            helper.make_node("Greater", ["argfull_pool", threshold], ["argfull_dense"]),
        ]
    )
    add_common_full_output_from_argmax(
        nodes,
        inits,
        "argfull_idx",
        "argfull_dense",
        "argfull_fgb",
        zero_f,
        prefix="argfull",
    )
    return finish_model(nodes, inits, "task097_argmax_full")


def add_foreground_sum(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    *,
    prefix: str,
) -> tuple[str, str, str, str]:
    starts = init(inits, f"{prefix}_s_fg", np.asarray([0, 1, 0, 0], dtype=np.int64))
    ends = init(inits, f"{prefix}_e_fg", np.asarray([1, C, GH, GW], dtype=np.int64))
    zero = scalar_f(inits, f"{prefix}_zero", 0.0)
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts, ends], [f"{prefix}_active_for_mask"]),
            helper.make_node("ReduceSum", [f"{prefix}_active_for_mask"], [f"{prefix}_fg"], axes=[1], keepdims=1),
            helper.make_node("Greater", [f"{prefix}_fg", zero], [f"{prefix}_fgb"]),
        ]
    )
    return f"{prefix}_fg", f"{prefix}_fgb", f"{prefix}_active_for_mask", zero


def build_avgpool_model() -> onnx.ModelProto:
    """Use AveragePool as a parameter-free 3x3 foreground counter."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    fg, fgb, active, zero = add_foreground_sum(nodes, inits, prefix="avg")
    threshold = scalar_f(inits, "avg_threshold", 0.15)
    nodes.extend(
        [
            helper.make_node(
                "AveragePool",
                [fg],
                ["avg_pool"],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                strides=[1, 1],
                count_include_pad=1,
            ),
            helper.make_node("Greater", ["avg_pool", threshold], ["avg_dense"]),
        ]
    )
    add_common_output(nodes, inits, "avg_dense", fgb, active, zero, prefix="avg")
    return finish_model(nodes, inits, "task097_avgpool")


def build_conv_model() -> onnx.ModelProto:
    """Use a 3x3 Conv with a zero center to count neighboring foreground cells."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    fg, fgb, active, zero0 = add_foreground_sum(nodes, inits, prefix="conv")
    kernel = np.ones((1, 1, 3, 3), dtype=np.float32)
    kernel[0, 0, 1, 1] = 0.0
    init(inits, "conv_kernel", kernel)
    zero = scalar_f(inits, "conv_count_zero", 0.0)
    nodes.extend(
        [
            helper.make_node(
                "Conv",
                [fg, "conv_kernel"],
                ["conv_count"],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                strides=[1, 1],
            ),
            helper.make_node("Greater", ["conv_count", zero], ["conv_neighbor"]),
        ]
    )
    add_common_output(nodes, inits, "conv_neighbor", fgb, active, zero0, prefix="conv")
    return finish_model(nodes, inits, "task097_conv")


def add_slice(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    source: str,
    out: str,
    starts: list[int],
    ends: list[int],
    axes: list[int],
) -> str:
    s = init(inits, f"{out}_s", np.asarray(starts, dtype=np.int64))
    e = init(inits, f"{out}_e", np.asarray(ends, dtype=np.int64))
    a = init(inits, f"{out}_a", np.asarray(axes, dtype=np.int64))
    nodes.append(helper.make_node("Slice", [source, s, e, a], [out]))
    return out


def shifted_mask(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    source: str,
    name: str,
    dr: int,
    dc: int,
) -> str:
    r0 = max(0, -dr)
    r1 = GH - max(0, dr)
    c0 = max(0, -dc)
    c1 = GW - max(0, dc)
    inner_h = r1 - r0
    inner_w = c1 - c0
    sliced = add_slice(nodes, inits, source, f"{name}_sl", [r0, c0], [r1, c1], [2, 3])
    cur = sliced
    cur_h, cur_w = inner_h, inner_w

    if dc > 0:
        z = init(inits, f"{name}_zl", np.zeros((1, 1, cur_h, dc), dtype=bool))
        nodes.append(helper.make_node("Concat", [z, cur], [f"{name}_pc"], axis=3))
        cur = f"{name}_pc"
        cur_w += dc
    elif dc < 0:
        z = init(inits, f"{name}_zr", np.zeros((1, 1, cur_h, -dc), dtype=bool))
        nodes.append(helper.make_node("Concat", [cur, z], [f"{name}_pc"], axis=3))
        cur = f"{name}_pc"
        cur_w += -dc

    if dr > 0:
        z = init(inits, f"{name}_zt", np.zeros((1, 1, dr, cur_w), dtype=bool))
        nodes.append(helper.make_node("Concat", [z, cur], [f"{name}_pr"], axis=2))
        cur = f"{name}_pr"
        cur_h += dr
    elif dr < 0:
        z = init(inits, f"{name}_zb", np.zeros((1, 1, -dr, cur_w), dtype=bool))
        nodes.append(helper.make_node("Concat", [cur, z], [f"{name}_pr"], axis=2))
        cur = f"{name}_pr"
        cur_h += -dr

    assert cur_h == GH and cur_w == GW
    return cur


def build_shift_model() -> onnx.ModelProto:
    """Use eight sliced/concatenated bool shifts and OR them together."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    _, fgb, active, zero = add_foreground_sum(nodes, inits, prefix="shift")
    shifts = [
        shifted_mask(nodes, inits, fgb, f"shift_{idx}", dr, dc)
        for idx, (dr, dc) in enumerate(
            [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        )
    ]
    acc = shifts[0]
    for idx, item in enumerate(shifts[1:], start=1):
        out = f"shift_or_{idx}"
        nodes.append(helper.make_node("Or", [acc, item], [out]))
        acc = out
    nodes.extend(
        [
        ]
    )
    add_common_output(nodes, inits, acc, fgb, active, zero, prefix="shift")
    return finish_model(nodes, inits, "task097_shift")


def load_examples() -> list[dict[str, Any]]:
    data = json.loads(TASK_JSON.read_text(encoding="utf-8"))
    examples: list[dict[str, Any]] = []
    for split in ("train", "test", "arc-gen"):
        examples.extend(data.get(split, []))
    return examples


def verify_rule() -> None:
    """Sanity-check the stated rule against the bundled examples."""
    bad = 0
    for ex in load_examples():
        g = np.asarray(ex["input"], dtype=np.int64)
        expected = np.asarray(ex["output"], dtype=np.int64)
        out = np.zeros_like(g)
        fg = g != 0
        for r in range(g.shape[0]):
            for c in range(g.shape[1]):
                if not fg[r, c]:
                    continue
                r0, r1 = max(0, r - 1), min(g.shape[0], r + 2)
                c0, c1 = max(0, c - 1), min(g.shape[1], c + 2)
                if int(fg[r0:r1, c0:c1].sum()) > 1:
                    out[r, c] = g[r, c]
        if not np.array_equal(out, expected):
            bad += 1
    if bad:
        raise RuntimeError(f"local rule check failed on {bad} examples")


def measure_variant(name: str, model: onnx.ModelProto) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"task097_{name}_") as tmp:
        path = Path(tmp) / "task097.onnx"
        onnx.save(model, str(path))
        ok, summary, _, _ = verify_correctness(path)
        stats = score_file(path)
        return {
            "name": name,
            "model": model,
            "correct": ok,
            "summary": summary,
            "stats": stats,
        }


def main() -> None:
    verify_rule()
    variants = [
        ("argmax-full-grid", build_argmax_full_model()),
        ("argmax-color-index", build_argmax_model()),
        ("8-shift-slice-concat", build_shift_model()),
        ("3x3-conv-neighbor-count", build_conv_model()),
        ("avgpool-3x3-count", build_avgpool_model()),
    ]
    results = [measure_variant(name, model) for name, model in variants]

    print("task097 variants")
    print("----------------------------------------")
    for item in results:
        stats = item["stats"]
        valid = bool(stats["valid"])
        cost = stats["cost"] if valid else None
        score = f"{float(stats['score']):.6f}" if valid else "INVALID"
        print(
            f"{item['name']:<24} correct={item['summary']:<7} "
            f"valid={valid!s:<5} memory={stats['memory']} params={stats['params']} "
            f"cost={cost} score={score}"
        )
        if not valid and stats.get("error"):
            print(f"  error: {str(stats['error']).strip().splitlines()[0]}")

    passing = [
        item
        for item in results
        if item["correct"] and item["stats"]["valid"] and item["stats"]["cost"] is not None
    ]
    if not passing:
        raise SystemExit("no correct valid variant")

    best = min(passing, key=lambda item: int(item["stats"]["cost"]))
    onnx.save(best["model"], str(OUT_PATH))
    print()
    print(f"selected: {best['name']} -> {OUT_PATH}")
    print_report(score_file(OUT_PATH))
    ok, summary, _, _ = verify_correctness(OUT_PATH)
    if not ok:
        raise SystemExit(f"final correctness failed: {summary}")
    print(f"correctness:       {summary}")

    all_dir = ROOT / "all"
    if all_dir.is_dir():
        shutil.copy2(OUT_PATH, all_dir / "task097.onnx")
        print(f"copied:            {all_dir / 'task097.onnx'}")


if __name__ == "__main__":
    main()
