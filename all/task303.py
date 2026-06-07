"""ONNX generator for ARC task303 using row/column emptiness masks.

Task rule: within the active top-left grid, detect rows and columns containing
only background color 0. The output preserves every existing cell, but paints
those fully empty rows and fully empty columns red (color 2). Padded cells
outside the active grid remain all-zero in the NeuroGolf one-hot tensor.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graph_onnx_memory import analyze  # noqa: E402
from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task303"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task303.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


@dataclass(frozen=True)
class Variant:
    name: str
    use_bg_for_foreground: bool
    cast_input_once: bool
    arithmetic_fg: bool = False
    concat_output: bool = False
    where_output: bool = False
    bg_sum_empty_lines: bool = False
    gather_bg: bool = False


VARIANTS = [
    Variant("bg_channel_float", use_bg_for_foreground=True, cast_input_once=False),
    Variant("fg_slice_float", use_bg_for_foreground=False, cast_input_once=False),
    Variant("bg_channel_bool_input", use_bg_for_foreground=True, cast_input_once=True),
    Variant("bg_arithmetic_full", use_bg_for_foreground=True, cast_input_once=False, arithmetic_fg=True),
    Variant("bg_arithmetic_concat", use_bg_for_foreground=True, cast_input_once=False, arithmetic_fg=True, concat_output=True),
    Variant("bg_arithmetic_where", use_bg_for_foreground=True, cast_input_once=False, arithmetic_fg=True, where_output=True),
    Variant(
        "bg_sum_where",
        use_bg_for_foreground=True,
        cast_input_once=False,
        where_output=True,
        bg_sum_empty_lines=True,
        gather_bg=True,
    ),
]


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _slice(nodes: List[onnx.NodeProto], data: str, out: str, starts: str, ends: str, axes: str) -> str:
    nodes.append(helper.make_node("Slice", [data, starts, ends, axes], [out]))
    return out


def build_model(variant: Variant) -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    half = _f32(inits, [0.5], "half")
    if not (variant.bg_sum_empty_lines and variant.gather_bg):
        st_bg = _i64(inits, [0, 0, 0, 0], "st_bg")
        en_bg = _i64(inits, [1, 1, H, W], "en_bg")
        axes4 = _i64(inits, [0, 1, 2, 3], "axes4")

    if variant.bg_sum_empty_lines:
        nodes.append(helper.make_node("ReduceMax", [IN_NAME], ["row_active_f"], axes=[1, 3], keepdims=1))
        nodes.append(helper.make_node("ReduceMax", [IN_NAME], ["col_active_f"], axes=[1, 2], keepdims=1))
        nodes.append(helper.make_node("Greater", ["row_active_f", half], ["row_active"]))
        nodes.append(helper.make_node("Greater", ["col_active_f", half], ["col_active"]))
        if variant.gather_bg:
            bg_index = _i64(inits, [0], "bg_index")
            nodes.append(helper.make_node("Gather", [IN_NAME, bg_index], ["bg_f"], axis=1))
        else:
            _slice(nodes, IN_NAME, "bg_f", st_bg, en_bg, axes4)
        nodes.append(helper.make_node("ReduceSum", ["bg_f"], ["row_bg_sum"], axes=[3], keepdims=1))
        nodes.append(helper.make_node("ReduceSum", ["bg_f"], ["col_bg_sum"], axes=[2], keepdims=1))
        nodes.append(helper.make_node("ReduceSum", ["col_active_f"], ["active_cols"], axes=[3], keepdims=1))
        nodes.append(helper.make_node("ReduceSum", ["row_active_f"], ["active_rows"], axes=[2], keepdims=1))
        nodes.append(helper.make_node("Sub", ["active_cols", half], ["active_cols_min"]))
        nodes.append(helper.make_node("Sub", ["active_rows", half], ["active_rows_min"]))
        nodes.append(helper.make_node("Greater", ["row_bg_sum", "active_cols_min"], ["empty_row"]))
        nodes.append(helper.make_node("Greater", ["col_bg_sum", "active_rows_min"], ["empty_col"]))
        nodes.append(helper.make_node("And", ["empty_col", "row_active"], ["empty_col_area"]))
        nodes.append(helper.make_node("And", ["empty_row", "col_active"], ["empty_row_area"]))
        nodes.append(helper.make_node("Or", ["empty_row_area", "empty_col_area"], ["red_mask"]))
    else:
        if variant.cast_input_once:
            nodes.append(helper.make_node("Greater", [IN_NAME, half], ["in_bool"]))
            source_for_keep = "in_bool"
            nodes.append(helper.make_node("Cast", ["in_bool"], ["in_float"], to=TensorProto.FLOAT))
            source_for_reductions = "in_float"
        else:
            source_for_keep = None
            source_for_reductions = IN_NAME

        nodes.append(helper.make_node("ReduceMax", [source_for_reductions], ["active_f"], axes=[1], keepdims=1))
        if not variant.arithmetic_fg:
            nodes.append(helper.make_node("Greater", ["active_f", half], ["active"]))
        nodes.append(helper.make_node("ReduceMax", ["active_f"], ["row_active_f"], axes=[3], keepdims=1))
        nodes.append(helper.make_node("ReduceMax", ["active_f"], ["col_active_f"], axes=[2], keepdims=1))
        nodes.append(helper.make_node("Greater", ["row_active_f", half], ["row_active"]))
        nodes.append(helper.make_node("Greater", ["col_active_f", half], ["col_active"]))

        if variant.arithmetic_fg:
            _slice(nodes, source_for_reductions, "bg_f", st_bg, en_bg, axes4)
            nodes.append(helper.make_node("Sub", ["active_f", "bg_f"], ["fg_reducible"]))
        elif variant.use_bg_for_foreground:
            _slice(nodes, source_for_reductions, "bg_f", st_bg, en_bg, axes4)
            nodes.append(helper.make_node("Less", ["bg_f", half], ["not_bg"]))
            nodes.append(helper.make_node("And", ["active", "not_bg"], ["fg"]))
            nodes.append(helper.make_node("Cast", ["fg"], ["fg_reducible"], to=TensorProto.FLOAT))
        else:
            st_fg = _i64(inits, [0, 1, 0, 0], "st_fg")
            en_fg = _i64(inits, [1, C, H, W], "en_fg")
            _slice(nodes, source_for_reductions, "fg_slice", st_fg, en_fg, axes4)
            nodes.append(helper.make_node("ReduceMax", ["fg_slice"], ["fg_f"], axes=[1], keepdims=1))
            nodes.append(helper.make_node("Greater", ["fg_f", half], ["fg"]))
            nodes.append(helper.make_node("Cast", ["fg"], ["fg_reducible"], to=TensorProto.FLOAT))

        nodes.append(helper.make_node("ReduceMax", ["fg_reducible"], ["row_fg_f"], axes=[3], keepdims=1))
        nodes.append(helper.make_node("ReduceMax", ["fg_reducible"], ["col_fg_f"], axes=[2], keepdims=1))
        nodes.append(helper.make_node("Greater", ["row_fg_f", half], ["row_fg"]))
        nodes.append(helper.make_node("Greater", ["col_fg_f", half], ["col_fg"]))
        nodes.append(helper.make_node("Not", ["row_fg"], ["row_no_fg"]))
        nodes.append(helper.make_node("Not", ["col_fg"], ["col_no_fg"]))
        nodes.append(helper.make_node("And", ["row_active", "row_no_fg"], ["empty_row"]))
        nodes.append(helper.make_node("And", ["col_active", "col_no_fg"], ["empty_col"]))
        nodes.append(helper.make_node("Or", ["empty_row", "empty_col"], ["empty_line"]))
        nodes.append(helper.make_node("And", ["row_active", "col_active"], ["active_area"]))
        nodes.append(helper.make_node("And", ["empty_line", "active_area"], ["red_mask"]))

    if variant.where_output:
        red_vec = np.zeros((1, C, 1, 1), dtype=np.float32)
        red_vec[0, 2, 0, 0] = 1.0
        red_onehot = _f32(inits, red_vec, "red_onehot")
        nodes.append(helper.make_node("Where", ["red_mask", red_onehot, IN_NAME], [OUT_NAME]))
    elif variant.concat_output:
        st_ch1 = _i64(inits, [0, 1, 0, 0], "st_ch1")
        en_ch1 = _i64(inits, [1, 2, H, W], "en_ch1")
        st_ch3 = _i64(inits, [0, 3, 0, 0], "st_ch3")
        en_ch3 = _i64(inits, [1, C, H, W], "en_ch3")
        zeros_1 = _f32(inits, np.zeros((1, 1, H, W), dtype=np.float32), "zeros_1")
        ones_1 = _f32(inits, np.ones((1, 1, H, W), dtype=np.float32), "ones_1")
        _slice(nodes, IN_NAME, "bg_in", st_bg, en_bg, axes4)
        _slice(nodes, IN_NAME, "ch1", st_ch1, en_ch1, axes4)
        _slice(nodes, IN_NAME, "red_in", _i64(inits, [0, 2, 0, 0], "st_red"), _i64(inits, [1, 3, H, W], "en_red"), axes4)
        _slice(nodes, IN_NAME, "ch3_9", st_ch3, en_ch3, axes4)
        nodes.append(helper.make_node("Where", ["red_mask", zeros_1, "bg_in"], ["bg_out"]))
        nodes.append(helper.make_node("Where", ["red_mask", ones_1, "red_in"], ["red_out"]))
        nodes.append(helper.make_node("Concat", ["bg_out", "ch1", "red_out", "ch3_9"], [OUT_NAME], axis=1))
    else:
        nodes.append(helper.make_node("Not", ["red_mask"], ["not_red_mask"]))
        if source_for_keep is None:
            nodes.append(helper.make_node("Greater", [IN_NAME, half], ["in_bool"]))
            source_for_keep = "in_bool"
        nodes.append(helper.make_node("And", [source_for_keep, "not_red_mask"], ["kept"]))
        red = _i64(inits, [2], "red")
        channels = _i64(inits, np.arange(C, dtype=np.int64).reshape(1, C, 1, 1), "channels")
        nodes.append(helper.make_node("Equal", [channels, red], ["is_red_channel"]))
        nodes.append(helper.make_node("And", ["red_mask", "is_red_channel"], ["red_onehot"]))
        nodes.append(helper.make_node("Or", ["kept", "red_onehot"], ["out_bool"]))
        nodes.append(helper.make_node("Cast", ["out_bool"], [OUT_NAME], to=TensorProto.FLOAT))

    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _examples() -> list[dict[str, Any]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [ex for split in ("train", "test", "arc-gen") for ex in data.get(split, [])]


def verify_correct(model_path: Path) -> tuple[bool, str]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    for idx, example in enumerate(_examples()):
        inp = convert_to_numpy(example, "input")
        expected = convert_to_numpy(example, "output")
        if inp is None or expected is None:
            continue
        actual = session.run([OUT_NAME], {IN_NAME: inp})[0]
        if not np.array_equal(actual > 0.0, expected > 0.0):
            return False, f"example {idx} mismatch"
    return True, "all examples matched"


def largest_internal(path: Path) -> str:
    _, tensor_memory, _, _ = analyze(path, fast=False)
    scored = [item for item in tensor_memory.values() if item.scored]
    if not scored:
        return "n/a"
    largest = max(scored, key=lambda item: item.bytes)
    return f"{largest.name} {largest.bytes} B {largest.dtype} {largest.shape}"


def main() -> None:
    results = []
    with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}_") as tmp:
        tmp_dir = Path(tmp)
        for variant in VARIANTS:
            path = tmp_dir / f"{TASK_ID}_{variant.name}.onnx"
            onnx.save(build_model(variant), path)
            ok, note = verify_correct(path)
            score = score_file(path)
            largest = largest_internal(path) if ok and score["valid"] else "n/a"
            results.append((variant, path, ok, note, score, largest))

        valid = [item for item in results if item[2] and item[4]["valid"]]
        if not valid:
            for variant, path, ok, note, score, _ in results:
                print(f"{variant.name}: correct={ok} note={note} valid={score['valid']} error={score['error']}")
            raise SystemExit("no valid task303 variant")

        best = min(valid, key=lambda item: int(item[4]["cost"]))
        onnx.save(onnx.load(str(best[1])), BEST_PATH)

        print("variant comparison")
        for variant, path, ok, note, score, largest in sorted(
            results, key=lambda item: (not item[4]["valid"], item[4]["cost"] or 10**18)
        ):
            score_text = "INVALID" if not score["valid"] else f"{score['score']:.6f}"
            print(
                f"{variant.name:<22} correct={ok} valid={score['valid']} "
                f"memory={score['memory']} params={score['params']} cost={score['cost']} "
                f"score={score_text} largest={largest} note={note}"
            )
        print(f"kept {best[0].name} -> {BEST_PATH}")


if __name__ == "__main__":
    main()
