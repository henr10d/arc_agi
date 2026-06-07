"""ONNX generator for ARC task200: expand one bottom marker into a striped 10x10 panel.

Task rule: the input is a 10x10 grid with one non-gray colored cell on the
bottom row at column x. The output keeps only a 10x10 active region. It draws
the input color in every row at columns x, x+2, x+4, ... through the right
edge. The intervening gap columns contain gray markers on the top row for
offsets 1 mod 4 and on the bottom row for offsets 3 mod 4. Background cells
inside the 10x10 panel are color 0; padding outside the panel remains all-zero
for the NeuroGolf 30x30 one-hot interface. The ONNX graph specializes to the
provided examples, where color 5 is reserved for gray markers and never appears
as the input marker color.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task200"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

IN_NAME = "input"
OUT_NAME = "output"
C = 10
H = W = 30
N = 10
PAD = H - N
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _i64(inits: list[onnx.TensorProto], vals: list[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))
    return name


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.float32), name=name))
    return name


def _bool(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.bool_), name=name))
    return name


def _value_info() -> tuple[onnx.ValueInfoProto, onnx.ValueInfoProto]:
    return (
        helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE),
        helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE),
    )


def solve_grid(grid: list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    markers = np.argwhere(arr != 0)
    if markers.shape[0] != 1:
        raise ValueError(f"expected one colored input cell, got {markers.shape[0]}")
    row, x = map(int, markers[0])
    if row != N - 1:
        raise ValueError(f"expected marker on bottom row, got row {row}")
    color = int(arr[row, x])

    out = np.zeros((N, N), dtype=np.int64)
    for offset in range(0, N - x, 2):
        out[:, x + offset] = color
    for offset in range(1, N - x, 2):
        target_row = 0 if offset % 4 == 1 else N - 1
        out[target_row, x + offset] = 5
    return out


def grid_to_onehot(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    active = onehot[0, :, :N, :N] > 0.0
    bad = active.sum(axis=0) != 1
    if np.any(bad):
        raise AssertionError("model produced invalid one-hot cells in the 10x10 panel")
    if np.any(onehot[0, :, N:, :] > 0.0) or np.any(onehot[0, :, :, N:] > 0.0):
        raise AssertionError("model activated padding outside the 10x10 panel")
    return active.argmax(axis=0).astype(np.int64)


def load_examples() -> list[dict[str, list[list[int]]]]:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return [ex for split in ("train", "test", "arc-gen") for ex in data.get(split, [])]


def validate_rule(examples: list[dict[str, list[list[int]]]]) -> None:
    alternatives: dict[str, int] = {
        "x_anchored_every_other_columns": 0,
        "remaining_width_controls_stop": 0,
        "last_column_as_independent_border": 0,
        "top_gray_offsets_1_mod_4_bottom_3_mod_4": 0,
    }
    for ex in examples:
        expected = np.asarray(ex["output"], dtype=np.int64)
        actual = solve_grid(ex["input"])
        if not np.array_equal(actual, expected):
            raise AssertionError(f"rule mismatch for input {ex['input']}")
        alternatives["x_anchored_every_other_columns"] += 1
        alternatives["remaining_width_controls_stop"] += 1
        alternatives["top_gray_offsets_1_mod_4_bottom_3_mod_4"] += 1

        markers = np.argwhere(np.asarray(ex["input"]) != 0)
        x = int(markers[0, 1])
        color = int(np.asarray(ex["input"])[N - 1, x])
        border = np.zeros((N, N), dtype=np.int64)
        border[:, 9] = color
        for col in range(x, 9, 2):
            border[:, col] = color
        for offset in range(1, N - x, 2):
            border[0 if offset % 4 == 1 else 9, x + offset] = 5
        if np.array_equal(border, expected):
            alternatives["last_column_as_independent_border"] += 1

    print("Validated rule variants against available examples:")
    for name, count in alternatives.items():
        print(f"  {name}: {count}/{len(examples)}")


def _slice(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    src: str,
    dst: str,
    starts: list[int],
    ends: list[int],
    axes: list[int],
) -> str:
    s = _i64(inits, starts, f"{dst}_s")
    e = _i64(inits, ends, f"{dst}_e")
    a = _i64(inits, axes, f"{dst}_a")
    nodes.append(helper.make_node("Slice", [src, s, e, a], [dst]))
    return dst


def build_structural_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    x_info, y_info = _value_info()

    one = _f32(inits, [1.0], "one")
    tile_h = _i64(inits, [1, 1, N, 1], "tile_h")

    bottom = _slice(nodes, inits, IN_NAME, "bottom", [0, 1, 9, 0], [1, C, 10, N], [0, 1, 2, 3])
    nodes.append(helper.make_node("ReduceSum", [bottom], ["pos"], axes=[1], keepdims=1))
    nodes.append(helper.make_node("ReduceSum", [bottom], ["color9"], axes=[2, 3], keepdims=1))

    def shifted(k: int) -> str:
        if k == 0:
            nodes.append(helper.make_node("Identity", ["pos"], ["sh0"]))
            return "sh0"
        cut = _slice(nodes, inits, "pos", f"cut{k}", [0, 0, 0, 0], [1, 1, 1, N - k], [0, 1, 2, 3])
        out = f"sh{k}"
        nodes.append(helper.make_node("Pad", [cut], [out], pads=[0, 0, 0, k, 0, 0, 0, 0]))
        return out

    shifts = {k: shifted(k) for k in range(10)}

    def add_chain(names: list[str], prefix: str) -> str:
        current = names[0]
        for idx, nxt in enumerate(names[1:], 1):
            out = f"{prefix}{idx}"
            nodes.append(helper.make_node("Add", [current, nxt], [out]))
            current = out
        return current

    stripe_row = add_chain([shifts[k] for k in (0, 2, 4, 6, 8)], "stripe_row")
    top_row = add_chain([shifts[k] for k in (1, 5, 9)], "top_row")
    bottom_row = add_chain([shifts[k] for k in (3, 7)], "bottom_row")
    nodes.append(helper.make_node("Tile", [stripe_row, tile_h], ["stripe"]))
    nodes.append(helper.make_node("Pad", [top_row], ["gray_top"], pads=[0, 0, 0, 0, 0, 0, N - 1, 0]))
    nodes.append(helper.make_node("Pad", [bottom_row], ["gray_bottom"], pads=[0, 0, N - 1, 0, 0, 0, 0, 0]))
    nodes.append(helper.make_node("Add", ["gray_top", "gray_bottom"], ["gray"]))
    nodes.append(helper.make_node("Add", ["stripe", "gray"], ["occupied"]))
    nodes.append(helper.make_node("Sub", [one, "occupied"], ["bg"]))
    channels = ["bg"]
    for idx in range(9):
        scalar = _slice(nodes, inits, "color9", f"color_scalar_{idx + 1}", [0, idx, 0, 0], [1, idx + 1, 1, 1], [0, 1, 2, 3])
        color_channel = f"color_ch_{idx + 1}"
        nodes.append(helper.make_node("Mul", [scalar, "stripe"], [color_channel]))
        if idx == 4:
            nodes.append(helper.make_node("Add", [color_channel, "gray"], ["c5"]))
            channels.append("c5")
        else:
            channels.append(color_channel)
    nodes.append(helper.make_node("Concat", channels, ["out10"], axis=1))
    nodes.append(helper.make_node("Pad", ["out10"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]))

    graph = helper.make_graph(nodes, "task200_structural", [x_info], [y_info], initializer=inits)
    model = helper.make_model(graph, producer_name="", ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", OPSET)])
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_lookup_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    x_info, y_info = _value_info()

    masks = np.zeros((N, N * N), dtype=np.float32)
    grays = np.zeros((N, N * N), dtype=np.float32)
    for x in range(N):
        toy = np.zeros((N, N), dtype=np.int64)
        toy[N - 1, x] = 1
        solved = solve_grid(toy)
        masks[x] = (solved == 1).reshape(-1)
        grays[x] = (solved == 5).reshape(-1)
    _f32(inits, masks, "stripe_table")
    _f32(inits, grays, "gray_table")
    one = _f32(inits, [1.0], "one")

    bottom = _slice(nodes, inits, IN_NAME, "bottom", [0, 1, 9, 0], [1, C, 10, N], [0, 1, 2, 3])
    nodes.append(helper.make_node("ReduceSum", [bottom], ["pos4"], axes=[1, 2], keepdims=0))
    nodes.append(helper.make_node("MatMul", ["pos4", "stripe_table"], ["stripe_flat"]))
    nodes.append(helper.make_node("MatMul", ["pos4", "gray_table"], ["gray_flat"]))
    shape = _i64(inits, [1, 1, N, N], "mask_shape")
    nodes.append(helper.make_node("Reshape", ["stripe_flat", shape], ["stripe"]))
    nodes.append(helper.make_node("Reshape", ["gray_flat", shape], ["gray"]))
    nodes.append(helper.make_node("ReduceSum", [bottom], ["color9"], axes=[2, 3], keepdims=1))
    nodes.append(helper.make_node("Add", ["stripe", "gray"], ["occupied"]))
    nodes.append(helper.make_node("Sub", [one, "occupied"], ["bg"]))
    channels = ["bg"]
    for idx in range(9):
        scalar = _slice(nodes, inits, "color9", f"color_scalar_{idx + 1}", [0, idx, 0, 0], [1, idx + 1, 1, 1], [0, 1, 2, 3])
        color_channel = f"color_ch_{idx + 1}"
        nodes.append(helper.make_node("Mul", [scalar, "stripe"], [color_channel]))
        if idx == 4:
            nodes.append(helper.make_node("Add", [color_channel, "gray"], ["c5"]))
            channels.append("c5")
        else:
            channels.append(color_channel)
    nodes.append(helper.make_node("Concat", channels, ["out10"], axis=1))
    nodes.append(helper.make_node("Pad", ["out10"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]))

    graph = helper.make_graph(nodes, "task200_lookup", [x_info], [y_info], initializer=inits)
    model = helper.make_model(graph, producer_name="", ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", OPSET)])
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_bool_mask_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    x_info, y_info = _value_info()

    zero = _f32(inits, [0.0], "zero")
    tile_h = _i64(inits, [1, 1, N, 1], "bool_tile_h")

    bottom = _slice(nodes, inits, IN_NAME, "b_bottom", [0, 1, 9, 0], [1, C, 10, N], [0, 1, 2, 3])
    nodes.append(helper.make_node("ReduceSum", [bottom], ["b_pos_f"], axes=[1], keepdims=1))
    nodes.append(helper.make_node("ReduceSum", [bottom], ["b_color9"], axes=[2, 3], keepdims=1))
    nodes.append(helper.make_node("Greater", ["b_pos_f", zero], ["b_pos"]))

    def shifted(k: int) -> str:
        if k == 0:
            nodes.append(helper.make_node("Identity", ["b_pos"], ["b_sh0"]))
            return "b_sh0"
        cut = _slice(nodes, inits, "b_pos", f"b_cut{k}", [0, 0, 0, 0], [1, 1, 1, N - k], [0, 1, 2, 3])
        out = f"b_sh{k}"
        left = _bool(inits, np.zeros((1, 1, 1, k), dtype=np.bool_), f"b_left{k}")
        nodes.append(helper.make_node("Concat", [left, cut], [out], axis=3))
        return out

    shifts = {k: shifted(k) for k in range(10)}

    def or_chain(names: list[str], prefix: str) -> str:
        current = names[0]
        for idx, nxt in enumerate(names[1:], 1):
            out = f"{prefix}{idx}"
            nodes.append(helper.make_node("Or", [current, nxt], [out]))
            current = out
        return current

    stripe_row = or_chain([shifts[k] for k in (0, 2, 4, 6, 8)], "b_stripe_row")
    top_row = or_chain([shifts[k] for k in (1, 5, 9)], "b_top_row")
    bottom_row = or_chain([shifts[k] for k in (3, 7)], "b_bottom_row")
    nodes.append(helper.make_node("Tile", [stripe_row, tile_h], ["b_stripe"]))
    zero_mid = _bool(inits, np.zeros((1, 1, N - 2, N), dtype=np.bool_), "b_zero_mid")
    nodes.append(helper.make_node("Concat", [top_row, zero_mid, bottom_row], ["b_gray"], axis=2))
    nodes.append(helper.make_node("Or", ["b_stripe", "b_gray"], ["b_occupied"]))
    nodes.append(helper.make_node("Not", ["b_occupied"], ["b_bg_bool"]))
    nodes.append(helper.make_node("Cast", ["b_bg_bool"], ["b_bg"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Cast", ["b_gray"], ["b_gray_f"], to=TensorProto.FLOAT))

    channels = ["b_bg"]
    for idx in range(9):
        scalar = _slice(nodes, inits, "b_color9", f"b_color_scalar_{idx + 1}", [0, idx, 0, 0], [1, idx + 1, 1, 1], [0, 1, 2, 3])
        color_channel = f"b_color_ch_{idx + 1}"
        nodes.append(helper.make_node("Where", ["b_stripe", scalar, zero], [color_channel]))
        if idx == 4:
            nodes.append(helper.make_node("Add", [color_channel, "b_gray_f"], ["b_c5"]))
            channels.append("b_c5")
        else:
            channels.append(color_channel)
    nodes.append(helper.make_node("Concat", channels, ["b_out10"], axis=1))
    nodes.append(helper.make_node("Pad", ["b_out10"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]))

    graph = helper.make_graph(nodes, "task200_bool_mask", [x_info], [y_info], initializer=inits)
    model = helper.make_model(graph, producer_name="", ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", OPSET)])
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_bool_onehot_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    x_info, y_info = _value_info()

    zero = _f32(inits, [0.0], "oh_zero")
    tile_h = _i64(inits, [1, 1, N, 1], "oh_tile_h")

    bottom = _slice(nodes, inits, IN_NAME, "oh_bottom", [0, 1, 9, 0], [1, C, 10, N], [0, 1, 2, 3])
    nodes.append(helper.make_node("ReduceSum", [bottom], ["oh_pos_f"], axes=[1], keepdims=1))
    nodes.append(helper.make_node("ReduceSum", [bottom], ["oh_color9"], axes=[2, 3], keepdims=1))
    nodes.append(helper.make_node("Greater", ["oh_pos_f", zero], ["oh_pos"]))
    nodes.append(helper.make_node("Greater", ["oh_color9", zero], ["oh_color_active"]))

    def shifted(k: int) -> str:
        if k == 0:
            nodes.append(helper.make_node("Identity", ["oh_pos"], ["oh_sh0"]))
            return "oh_sh0"
        cut = _slice(nodes, inits, "oh_pos", f"oh_cut{k}", [0, 0, 0, 0], [1, 1, 1, N - k], [0, 1, 2, 3])
        left = _bool(inits, np.zeros((1, 1, 1, k), dtype=np.bool_), f"oh_left{k}")
        out = f"oh_sh{k}"
        nodes.append(helper.make_node("Concat", [left, cut], [out], axis=3))
        return out

    shifts = {k: shifted(k) for k in range(10)}

    def or_chain(names: list[str], prefix: str) -> str:
        current = names[0]
        for idx, nxt in enumerate(names[1:], 1):
            out = f"{prefix}{idx}"
            nodes.append(helper.make_node("Or", [current, nxt], [out]))
            current = out
        return current

    stripe_row = or_chain([shifts[k] for k in (0, 2, 4, 6, 8)], "oh_stripe_row")
    top_row = or_chain([shifts[k] for k in (1, 5, 9)], "oh_top_row")
    bottom_row = or_chain([shifts[k] for k in (3, 7)], "oh_bottom_row")
    nodes.append(helper.make_node("Tile", [stripe_row, tile_h], ["oh_stripe"]))
    zero_mid = _bool(inits, np.zeros((1, 1, N - 2, N), dtype=np.bool_), "oh_zero_mid")
    nodes.append(helper.make_node("Concat", [top_row, zero_mid, bottom_row], ["oh_gray"], axis=2))
    nodes.append(helper.make_node("Or", ["oh_stripe", "oh_gray"], ["oh_occupied"]))
    nodes.append(helper.make_node("Not", ["oh_occupied"], ["oh_bg"]))

    channels = ["oh_bg"]
    for idx in range(9):
        if idx == 4:
            channels.append("oh_gray")
            continue
        active = _slice(nodes, inits, "oh_color_active", f"oh_color_active_{idx + 1}", [0, idx, 0, 0], [1, idx + 1, 1, 1], [0, 1, 2, 3])
        color_channel = f"oh_color_ch_{idx + 1}"
        nodes.append(helper.make_node("And", ["oh_stripe", active], [color_channel]))
        channels.append(color_channel)
    nodes.append(helper.make_node("Concat", channels, ["oh_out10_bool"], axis=1))
    nodes.append(helper.make_node("Cast", ["oh_out10_bool"], ["oh_out10"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Pad", ["oh_out10"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]))

    graph = helper.make_graph(nodes, "task200_bool_onehot", [x_info], [y_info], initializer=inits)
    model = helper.make_model(graph, producer_name="", ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", OPSET)])
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def verify_model(model: onnx.ModelProto, examples: list[dict[str, list[list[int]]]]) -> None:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    for idx, ex in enumerate(examples):
        got = session.run([OUT_NAME], {IN_NAME: grid_to_onehot(ex["input"])})[0]
        expected = np.asarray(ex["output"], dtype=np.int64)
        decoded = onehot_to_grid(got)
        if not np.array_equal(decoded, expected):
            raise AssertionError(f"candidate failed example {idx}: got\n{decoded}\nexpected\n{expected}")
        expected_onehot = grid_to_onehot(expected)
        if not np.array_equal(got > 0.0, expected_onehot > 0.0):
            raise AssertionError(f"candidate failed one-hot threshold check on example {idx}")


def candidate_stats(path: Path, model: onnx.ModelProto, result: dict[str, Any]) -> dict[str, Any]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    realized = {out for node in inferred.graph.node for out in node.output if out and out != OUT_NAME}
    return {
        "path": path,
        "score": result["score"],
        "cost": result["cost"],
        "memory": result["memory"],
        "params": result["params"],
        "nodes": len(model.graph.node),
        "realized_tensors": len(realized),
        "filesize": path.stat().st_size,
    }


def evaluate_candidates(builders: dict[str, Callable[[], onnx.ModelProto]], examples: list[dict[str, list[list[int]]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="task200_candidates_") as tmp:
        tmpdir = Path(tmp)
        for name, builder in builders.items():
            model = builder()
            verify_model(model, examples)
            path = tmpdir / f"{TASK_ID}_{name}.onnx"
            onnx.save(model, path)
            result = score_file(path)
            if not result["valid"]:
                raise AssertionError(f"{name} invalid: {result['error']}")
            rows.append({"name": name, "model": model, **candidate_stats(path, model, result)})

        rows.sort(key=lambda row: (float(row["score"]), -int(row["nodes"]), -int(row["realized_tensors"])), reverse=True)
        best = rows[0]
        onnx.save(best["model"], BEST_PATH)
        best["filesize"] = BEST_PATH.stat().st_size
    return rows


def main() -> None:
    examples = load_examples()
    validate_rule(examples)
    rows = evaluate_candidates(
        {
            "structural": build_structural_model,
            "lookup": build_lookup_model,
            "bool_mask": build_bool_mask_model,
            "bool_onehot": build_bool_onehot_model,
        },
        examples,
    )
    print("\nCandidate comparison:")
    for row in rows:
        print(
            f"  {row['name']:<10} score={row['score']:.6f} cost={row['cost']} "
            f"memory={row['memory']} params={row['params']} nodes={row['nodes']} "
            f"realized={row['realized_tensors']} size={row['filesize']}"
        )
    print(f"\nWrote best model: {BEST_PATH}")


if __name__ == "__main__":
    main()
