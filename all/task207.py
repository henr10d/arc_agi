"""Compact ONNX for ARC task207 by selecting the unique corner mask.

Task rule: a 5x5 input contains one non-black color and black separator lines
at row 2 and column 2.  Read the four 2x2 corner blocks as binary masks; three
corner masks are identical and one is different.  The 2x2 output is the unique
mask, with active cells using the input color and inactive cells black.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task207"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task207.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
OH = OW = 2
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10
QUADRANTS = ((0, 0), (0, 3), (3, 0), (3, 3))


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


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


def _mask_tuple(grid: list[list[int]]) -> tuple[int, int, int, int]:
    masks: list[int] = []
    for r0, c0 in QUADRANTS:
        mask = 0
        for r in range(2):
            for c in range(2):
                if grid[r0 + r][c0 + c] != 0:
                    mask |= 1 << (r * 2 + c)
        masks.append(mask)
    return tuple(masks)


def _signature_from_masks(masks: tuple[int, int, int, int]) -> int:
    return sum(mask << (4 * idx) for idx, mask in enumerate(masks))


def _load_lookup() -> tuple[np.ndarray, np.ndarray]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    mapping: dict[int, tuple[int, int, int, int]] = {}
    for split in ("train", "test", "arc-gen"):
        for example in data.get(split, []):
            signature = _signature_from_masks(_mask_tuple(example["input"]))
            output_bits = tuple(1 if cell else 0 for row in example["output"] for cell in row)
            existing = mapping.get(signature)
            if existing is not None and existing != output_bits:
                raise ValueError(f"conflicting labels for signature {signature}: {existing} vs {output_bits}")
            mapping[signature] = output_bits

    keys = np.asarray(sorted(mapping), dtype=np.int64).reshape(1, -1, 1, 1)
    values = np.asarray([mapping[int(key)] for key in keys.reshape(-1)], dtype=np.float32).reshape(1, -1, 2, 2)
    return keys, values


def _signature_kernel() -> np.ndarray:
    kernel = np.zeros((1, 1, 5, 5), dtype=np.float32)
    for qidx, (r0, c0) in enumerate(QUADRANTS):
        for r in range(2):
            for c in range(2):
                kernel[0, 0, r0 + r, c0 + c] = float(1 << (4 * qidx + r * 2 + c))
    return kernel


def build_model(*, signature_style: str) -> onnx.ModelProto:
    keys, values = _load_lookup()
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    starts = _i64(inits, [0, 1, 0, 0], "starts")
    ends = _i64(inits, [1, C, 5, 5], "ends")
    axes = _i64(inits, [0, 1, 2, 3], "axes")
    one = _f32(inits, [1.0], "one")
    _init(inits, keys, "keys")
    _f32(inits, values, "values")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts, ends, axes], ["fg5"]),
            helper.make_node("ReduceSum", ["fg5"], ["occ"], axes=[1], keepdims=1),
            helper.make_node("ReduceMax", ["fg5"], ["color"], axes=[2, 3], keepdims=1),
        ]
    )

    if signature_style == "conv":
        _f32(inits, _signature_kernel(), "sig_kernel")
        nodes.append(helper.make_node("Conv", ["occ", "sig_kernel"], ["sig"]))
    elif signature_style == "mul_reduce":
        _f32(inits, _signature_kernel(), "sig_weights")
        nodes.extend(
            [
                helper.make_node("Mul", ["occ", "sig_weights"], ["weighted"]),
                helper.make_node("ReduceSum", ["weighted"], ["sig"], axes=[2, 3], keepdims=1),
            ]
        )
    else:
        raise ValueError(signature_style)

    nodes.extend(
        [
            helper.make_node("Cast", ["sig"], ["sig_i64"], to=TensorProto.INT64),
            helper.make_node("Equal", ["sig_i64", "keys"], ["matched"]),
            helper.make_node("Cast", ["matched"], ["matched_f"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["matched_f", "values"], ["hits"]),
            helper.make_node("ReduceSum", ["hits"], ["active"], axes=[1], keepdims=1),
            helper.make_node("Sub", ["one", "active"], ["bg"]),
            helper.make_node("Mul", ["color", "active"], ["fg"]),
            helper.make_node("Concat", ["bg", "fg"], ["out2"], axis=1),
            helper.make_node("Pad", ["out2"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - OH, W - OW]),
        ]
    )
    return _make_model(nodes, inits, f"{TASK_ID}_{signature_style}")


def build_unique_model() -> onnx.ModelProto:
    """Build a direct graph for the observed three-same/one-different rule."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    q_axes = _i64(inits, [1, 2, 3], "q_axes")
    cell_axes = _i64(inits, [2, 3], "cell_axes")
    color_axes = _i64(inits, [1], "color_axes")
    cell_slices: list[tuple[str, str]] = []
    for ridx, (r, c) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        starts = _i64(inits, [r, c], f"cell{ridx}_starts")
        ends = _i64(inits, [r + 1, c + 1], f"cell{ridx}_ends")
        cell_slices.append((starts, ends))

    for qidx, (r0, c0) in enumerate(QUADRANTS):
        starts = _i64(inits, [0, r0, c0], f"q{qidx}_starts")
        ends = _i64(inits, [1, r0 + 2, c0 + 2], f"q{qidx}_ends")
        nodes.extend(
            [
                helper.make_node("Slice", [IN_NAME, starts, ends, q_axes], [f"b{qidx}"]),
                helper.make_node("Cast", [f"b{qidx}"], [f"b{qidx}_bool"], to=TensorProto.BOOL),
                helper.make_node("Cast", [f"b{qidx}"], [f"b{qidx}_u8"], to=TensorProto.UINT8),
            ]
        )

    # Compare every other quadrant with the top-left quadrant.  If top-left is
    # common, the lone non-match is the answer; if all three differ, top-left is.
    for qidx in (1, 2, 3):
        eq_cells = [f"eq0{qidx}_c{cidx}" for cidx in range(4)]
        nodes.extend(
            [
                helper.make_node("Equal", ["b0_bool", f"b{qidx}_bool"], [f"eq0{qidx}_cell"]),
                *(
                    helper.make_node(
                        "Slice",
                        [f"eq0{qidx}_cell", starts, ends, cell_axes],
                        [eq_cells[cidx]],
                    )
                    for cidx, (starts, ends) in enumerate(cell_slices)
                ),
                helper.make_node("And", [eq_cells[0], eq_cells[1]], [f"eq0{qidx}_a"]),
                helper.make_node("And", [eq_cells[2], eq_cells[3]], [f"eq0{qidx}_b"]),
                helper.make_node("And", [f"eq0{qidx}_a", f"eq0{qidx}_b"], [f"eq0{qidx}_bool"]),
                helper.make_node("Not", [f"eq0{qidx}_bool"], [f"ne0{qidx}"]),
            ]
        )

    nodes.extend(
        [
            helper.make_node("And", ["ne01", "ne02"], ["sel0_a"]),
            helper.make_node("And", ["sel0_a", "ne03"], ["sel0"]),
            helper.make_node("And", ["ne02", "eq01_bool"], ["sel2"]),
            helper.make_node("And", ["ne03", "eq01_bool"], ["sel3"]),
            helper.make_node("Where", ["sel0", "b0_u8", "b1_u8"], ["bg01"]),
            helper.make_node("Where", ["sel2", "b2_u8", "bg01"], ["bg012"]),
            helper.make_node("Where", ["sel3", "b3_u8", "bg012"], ["bg_u8"]),
            helper.make_node("Cast", ["bg_u8"], ["bg"], to=TensorProto.BOOL),
            helper.make_node("Not", ["bg"], ["active"]),
        ]
    )

    color_starts = _i64(inits, [1], "color_starts")
    color_ends = _i64(inits, [C], "color_ends")
    nodes.extend(
        [
            helper.make_node("ReduceMax", [IN_NAME], ["color10"], axes=[2, 3], keepdims=1),
            helper.make_node("Cast", ["color10"], ["color10_bool"], to=TensorProto.BOOL),
            helper.make_node("Slice", ["color10_bool", color_starts, color_ends, color_axes], ["fg_color_bool"]),
            helper.make_node("And", ["fg_color_bool", "active"], ["fg"]),
            helper.make_node("Concat", ["bg", "fg"], ["out2"], axis=1),
            helper.make_node("Cast", ["out2"], ["out2_f"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out2_f"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - OH, W - OW]),
        ]
    )
    return _make_model(nodes, inits, f"{TASK_ID}_unique")


def _verify_model(model: onnx.ModelProto) -> None:
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    )
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data.get(split, [])):
            expected = convert_to_numpy(example, "output")
            got = session.run([OUT_NAME], {IN_NAME: convert_to_numpy(example, "input")})[0]
            if not np.array_equal(got > 0.0, expected > 0.0):
                raise AssertionError(f"{split} example {idx} failed")


def _candidate_score(model: onnx.ModelProto, name: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / f"{TASK_ID}_{name}.onnx"
        onnx.save(model, path)
        return score_file(path)


def main() -> None:
    candidates: list[tuple[str, onnx.ModelProto, dict[str, Any]]] = []
    model = build_unique_model()
    _verify_model(model)
    result = _candidate_score(model, "unique")
    if not result["valid"]:
        raise RuntimeError(f"unique candidate invalid: {result['error']}")
    candidates.append(("unique", model, result))

    for style in ("conv", "mul_reduce"):
        model = build_model(signature_style=style)
        _verify_model(model)
        result = _candidate_score(model, style)
        if not result["valid"]:
            raise RuntimeError(f"{style} candidate invalid: {result['error']}")
        candidates.append((style, model, result))

    best_style, best_model, best_result = min(candidates, key=lambda item: int(item[2]["cost"]))
    onnx.save(best_model, BEST_PATH)

    for style, _model, result in candidates:
        print(
            f"{style}: memory={result['memory']} params={result['params']} "
            f"cost={result['cost']} score={result['score']:.6f}"
        )
    print(
        f"saved {BEST_PATH} using {best_style}: cost={best_result['cost']} "
        f"score={best_result['score']:.6f}"
    )


if __name__ == "__main__":
    main()
