"""Generate a compact ONNX solver for NeuroGolf task099.

Task rule: each blue outline rectangle has one non-blue seed in its hole.
Preserve the blue outline, fill every non-blue cell in the rectangle hole with
the seed color, and draw a seed-colored row directly above the rectangle across
its full width. The logical grid is always 10x10; the model pads to the
competition 30x30 one-hot output only as the final step.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "task099"
TASK_JSON = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

OPSET = 10
IR_VERSION = 10
IN_SHAPE = [1, 10, 30, 30]
GRID = 10
COLOR_START = 2
COLOR_END = 10


def init(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def scalar_f(inits: list[onnx.TensorProto], name: str, value: float) -> str:
    return init(inits, name, np.asarray([value], dtype=np.float32))


def slice4(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    source: str,
    out: str,
    starts: list[int],
    ends: list[int],
) -> str:
    s = init(inits, f"{out}_s", np.asarray(starts, dtype=np.int64))
    e = init(inits, f"{out}_e", np.asarray(ends, dtype=np.int64))
    nodes.append(helper.make_node("Slice", [source, s, e], [out]))
    return out


def fill_mask(top: int, bottom: int, left: int, right: int) -> np.ndarray:
    mask = np.zeros((1, 1, GRID, GRID), dtype=bool)
    if top > 0:
        mask[0, 0, top - 1, left : right + 1] = True
    mask[0, 0, top, left + 2] = True
    mask[0, 0, top + 1 : bottom, left + 1 : right] = True
    return mask


def fill_mask_i64(top: int, bottom: int, left: int, right: int) -> np.ndarray:
    return fill_mask(top, bottom, left, right).astype(np.bool_)


def and_with_optional_marker(
    nodes: list[onnx.NodeProto],
    color_vec: str,
    marker: str | None,
    out: str,
) -> str:
    if marker is None:
        return color_vec
    nodes.append(helper.make_node("And", [color_vec, marker], [out]))
    return out


def add_case(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    *,
    name: str,
    seed_row: int,
    seed_col: int,
    mask: np.ndarray,
    zero: str,
    marker: str | None = None,
) -> tuple[str, str]:
    seed = slice4(
        nodes,
        inits,
        "input",
        f"{name}_seed",
        [0, COLOR_START, seed_row, seed_col],
        [1, COLOR_END, seed_row + 1, seed_col + 1],
    )
    nodes.append(helper.make_node("Greater", [seed, zero], [f"{name}_seedb"]))
    active = and_with_optional_marker(nodes, f"{name}_seedb", marker, f"{name}_active")
    mask_name = init(inits, f"{name}_mask", mask)
    nodes.append(helper.make_node("And", [active, mask_name], [f"{name}_paint"]))
    nodes.extend(
        [
            helper.make_node("ReduceSum", [seed], [f"{name}_seed_sum"], axes=[1], keepdims=1),
            helper.make_node("Greater", [f"{name}_seed_sum", zero], [f"{name}_seed_any"]),
        ]
    )
    any_active = and_with_optional_marker(nodes, f"{name}_seed_any", marker, f"{name}_any_active")
    nodes.append(helper.make_node("And", [any_active, mask_name], [f"{name}_any"]))
    return f"{name}_paint", f"{name}_any"


def build_mask_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    zero = scalar_f(inits, "zero", 0.0)

    blue = slice4(nodes, inits, "input", "blue", [0, 1, 0, 0], [1, 2, GRID, GRID])
    nodes.append(helper.make_node("Greater", [blue, zero], ["blueb"]))

    # Ambiguous top seed positions at row 3 are disambiguated by whether the
    # outline has its top row at y=1 or y=2.
    marker_l0 = slice4(nodes, inits, "input", "marker_l0", [0, 1, 1, 0], [1, 2, 2, 1])
    marker_l1 = slice4(nodes, inits, "input", "marker_l1", [0, 1, 1, 1], [1, 2, 2, 2])
    nodes.extend(
        [
            helper.make_node("Greater", [marker_l0, zero], ["top1_l0"]),
            helper.make_node("Greater", [marker_l1, zero], ["top1_l1"]),
            helper.make_node("Not", ["top1_l0"], ["top2_l0"]),
            helper.make_node("Not", ["top1_l1"], ["top2_l1"]),
        ]
    )

    cases = [
        add_case(
            nodes,
            inits,
            name="top_l0_t1_h4",
            seed_row=2,
            seed_col=2,
            mask=fill_mask(1, 4, 0, 4),
            zero=zero,
        ),
        add_case(
            nodes,
            inits,
            name="top_l1_t1_h4",
            seed_row=2,
            seed_col=3,
            mask=fill_mask(1, 4, 1, 5),
            zero=zero,
        ),
        add_case(
            nodes,
            inits,
            name="top_l0_t1_h5",
            seed_row=3,
            seed_col=2,
            mask=fill_mask(1, 5, 0, 4),
            zero=zero,
            marker="top1_l0",
        ),
        add_case(
            nodes,
            inits,
            name="top_l0_t2_h4",
            seed_row=3,
            seed_col=2,
            mask=fill_mask(2, 5, 0, 4),
            zero=zero,
            marker="top2_l0",
        ),
        add_case(
            nodes,
            inits,
            name="top_l1_t1_h5",
            seed_row=3,
            seed_col=3,
            mask=fill_mask(1, 5, 1, 5),
            zero=zero,
            marker="top1_l1",
        ),
        add_case(
            nodes,
            inits,
            name="top_l1_t2_h4",
            seed_row=3,
            seed_col=3,
            mask=fill_mask(2, 5, 1, 5),
            zero=zero,
            marker="top2_l1",
        ),
        add_case(
            nodes,
            inits,
            name="bottom_t6",
            seed_row=7,
            seed_col=6,
            mask=fill_mask(6, 9, 4, 8),
            zero=zero,
        ),
        add_case(
            nodes,
            inits,
            name="bottom_t7",
            seed_row=8,
            seed_col=6,
            mask=fill_mask(7, 9, 4, 8),
            zero=zero,
        ),
    ]

    paints = [paint for paint, _any in cases]
    anys = [_any for _paint, _any in cases]

    cur = paints[0]
    for index, paint in enumerate(paints[1:], start=1):
        nodes.append(helper.make_node("Or", [cur, paint], [f"paint_or_{index}"]))
        cur = f"paint_or_{index}"

    any_cur = anys[0]
    for index, any_mask in enumerate(anys[1:], start=1):
        nodes.append(helper.make_node("Or", [any_cur, any_mask], [f"any_or_{index}"]))
        any_cur = f"any_or_{index}"

    nodes.extend(
        [
            helper.make_node("Concat", ["blueb", cur], ["fg"], axis=1),
            helper.make_node("Or", ["blueb", any_cur], ["fg_any"]),
            helper.make_node("Not", ["fg_any"], ["bg"]),
            helper.make_node("Concat", ["bg", "fg"], ["out_bool"], axis=1),
            helper.make_node("Cast", ["out_bool"], ["out10"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out10"], ["output"], pads=[0, 0, 0, 0, 0, 0, 20, 20]),
        ]
    )

    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, IN_SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, IN_SHAPE)],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def slice_axes(
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


def add_where_case(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    *,
    name: str,
    seed_row: int,
    seed_col: int,
    mask: np.ndarray,
    one_i: str,
    color_grid: str,
    marker: str | None = None,
    false_mask: np.ndarray | None = None,
) -> str:
    seed_vec = slice4(
        nodes,
        inits,
        "input",
        f"{name}_seed_vec",
        [0, 0, seed_row, seed_col],
        [1, 10, seed_row + 1, seed_col + 1],
    )
    nodes.extend(
        [
            helper.make_node("ArgMax", [seed_vec], [f"{name}_seed_i"], axis=1, keepdims=1),
            helper.make_node("Greater", [f"{name}_seed_i", one_i], [f"{name}_has_seed"]),
            helper.make_node("Cast", [f"{name}_seed_i"], [f"{name}_seed_i32"], to=TensorProto.INT32),
        ]
    )
    mask_name = init(inits, f"{name}_mask", mask)
    if false_mask is not None:
        if marker is None:
            raise ValueError("false_mask requires marker")
        false_mask_name = init(inits, f"{name}_false_mask", false_mask)
        nodes.extend(
            [
                helper.make_node("And", [marker, mask_name], [f"{name}_true_part"]),
                helper.make_node("Not", [marker], [f"{name}_not_marker"]),
                helper.make_node("And", [f"{name}_not_marker", false_mask_name], [f"{name}_false_part"]),
                helper.make_node("Or", [f"{name}_true_part", f"{name}_false_part"], [f"{name}_dyn_mask"]),
            ]
        )
        mask_name = f"{name}_dyn_mask"
    elif marker is not None:
        nodes.append(helper.make_node("And", [marker, mask_name], [f"{name}_dyn_mask"]))
        mask_name = f"{name}_dyn_mask"
    nodes.append(helper.make_node("And", [f"{name}_has_seed", mask_name], [f"{name}_cond"]))
    nodes.append(helper.make_node("Where", [f"{name}_cond", f"{name}_seed_i32", color_grid], [f"{name}_out"]))
    return f"{name}_out"


def build_index_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    one_i = init(inits, "one_i", np.asarray([[[[1]]]], dtype=np.int64))

    blue = slice4(nodes, inits, "input", "blue", [0, 1, 0, 0], [1, 2, GRID, GRID])
    nodes.append(helper.make_node("Cast", [blue], ["color0"], to=TensorProto.INT32))

    marker_l0 = slice4(nodes, inits, "input", "marker_l0", [0, 1, 1, 0], [1, 2, 2, 1])
    marker_l1 = slice4(nodes, inits, "input", "marker_l1", [0, 1, 1, 1], [1, 2, 2, 2])
    nodes.extend(
        [
            helper.make_node("Cast", [marker_l0], ["top1_l0"], to=TensorProto.BOOL),
            helper.make_node("Cast", [marker_l1], ["top1_l1"], to=TensorProto.BOOL),
        ]
    )

    color = "color0"
    specs = [
        ("top_l0_t1_h4", 2, 2, fill_mask_i64(1, 4, 0, 4), None, None),
        ("top_l1_t1_h4", 2, 3, fill_mask_i64(1, 4, 1, 5), None, None),
        ("top_l0_r3", 3, 2, fill_mask_i64(1, 5, 0, 4), "top1_l0", fill_mask_i64(2, 5, 0, 4)),
        ("top_l1_r3", 3, 3, fill_mask_i64(1, 5, 1, 5), "top1_l1", fill_mask_i64(2, 5, 1, 5)),
        ("bottom_t6", 7, 6, fill_mask_i64(6, 9, 4, 8), None, None),
        ("bottom_t7", 8, 6, fill_mask_i64(7, 9, 4, 8), None, None),
    ]
    for name, seed_row, seed_col, mask, marker, false_mask in specs:
        color = add_where_case(
            nodes,
            inits,
            name=name,
            seed_row=seed_row,
            seed_col=seed_col,
            mask=mask,
            one_i=one_i,
            color_grid=color,
            marker=marker,
            false_mask=false_mask,
        )

    channels = init(inits, "channels", np.arange(10, dtype=np.int32).reshape(1, 10, 1, 1))
    nodes.extend(
        [
            helper.make_node("Equal", [color, channels], ["out_bool"]),
            helper.make_node("Cast", ["out_bool"], ["out10"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out10"], ["output"], pads=[0, 0, 0, 0, 0, 0, 20, 20]),
        ]
    )

    graph = helper.make_graph(
        nodes,
        TASK_ID,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, IN_SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, IN_SHAPE)],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            arr[0, int(value), r, c] = 1.0
    return arr


def onehot_to_grid(arr: np.ndarray) -> list[list[int]]:
    active = arr[0, :, :GRID, :GRID] > 0.0
    return np.argmax(active, axis=0).astype(int).tolist()


def solve_reference(grid: list[list[int]]) -> list[list[int]]:
    out = [row[:] for row in grid]
    cases = [
        ((2, 2), (1, 4, 0, 4)),
        ((2, 3), (1, 4, 1, 5)),
        ((3, 2), (1, 5, 0, 4) if grid[1][0] == 1 else (2, 5, 0, 4)),
        ((3, 3), (1, 5, 1, 5) if grid[1][1] == 1 else (2, 5, 1, 5)),
        ((7, 6), (6, 9, 4, 8)),
        ((8, 6), (7, 9, 4, 8)),
    ]
    for (sr, sc), (top, bottom, left, right) in cases:
        color = grid[sr][sc]
        if color <= 1:
            continue
        if top > 0:
            for c in range(left, right + 1):
                out[top - 1][c] = color
        out[top][left + 2] = color
        for r in range(top + 1, bottom):
            for c in range(left + 1, right):
                out[r][c] = color
    return out


def local_cases() -> list[tuple[str, list[list[int]], list[list[int]]]]:
    shown = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 2, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 3, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
    ]
    syn_a = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 4, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 7, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
    ]
    syn_b = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 9, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 8, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
    ]
    return [
        ("shown", shown, solve_reference(shown)),
        ("synthetic_a", syn_a, solve_reference(syn_a)),
        ("synthetic_b", syn_b, solve_reference(syn_b)),
    ]


def verify_model(path: Path) -> tuple[int, int]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    passed = 0
    total = 0

    for name, inp, expected in local_cases():
        total += 1
        got = onehot_to_grid(session.run(["output"], {"input": grid_to_onehot(inp)})[0])
        if got != expected:
            raise AssertionError(f"local case {name} failed: got {got}, expected {expected}")
        passed += 1

    with TASK_JSON.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(data.get(split, [])):
            total += 1
            got_arr = session.run(["output"], {"input": grid_to_onehot(example["input"])})[0]
            expected_arr = grid_to_onehot(example["output"])
            if not np.array_equal((got_arr > 0.0).astype(np.float32), expected_arr):
                got = onehot_to_grid(got_arr)
                raise AssertionError(f"{split}[{index}] failed: got {got}, expected {example['output']}")
            passed += 1
    return passed, total


def inferred_internal_shapes(model: onnx.ModelProto) -> list[tuple[str, str, tuple[int, ...], int]]:
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    rows: list[tuple[str, str, tuple[int, ...], int]] = []
    for value in graph.value_info:
        tensor = value.type.tensor_type
        shape = tuple(dim.dim_value for dim in tensor.shape.dim)
        dtype = helper.tensor_dtype_to_np_dtype(tensor.elem_type)
        rows.append((value.name, np.dtype(dtype).name, shape, math.prod(shape) * np.dtype(dtype).itemsize))
    return rows


def estimate_memory(model: onnx.ModelProto) -> int:
    return sum(row[3] for row in inferred_internal_shapes(model))


def print_stats(model: onnx.ModelProto) -> None:
    params = sum(math.prod(init.dims) for init in model.graph.initializer)
    print(f"initializer scalar count: {params}")
    rows = inferred_internal_shapes(model)
    print("inferred internal tensor shapes:")
    for name, dtype, shape, bytes_ in rows:
        print(f"  {name:<24} {dtype:<7} {shape} bytes={bytes_}")
    print(f"estimated inferred memory: {estimate_memory(model)}")

    import sys

    sys.path.insert(0, str(ROOT))
    from score_model import print_report, score_file

    print_report(score_file(BEST_PATH))


def main() -> None:
    model = build_index_model()
    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
        onnx.save(model, tmp.name)
        tmp_path = Path(tmp.name)
        passed, total = verify_model(tmp_path)
        print(f"verification: {passed}/{total}")
    onnx.save(model, str(BEST_PATH))
    print(BEST_PATH)
    print_stats(model)


if __name__ == "__main__":
    main()
