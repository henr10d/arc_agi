"""Exact compact ONNX for task096's concentric stencil reconstruction.

The task examples contain sparse colored stencil fragments on a dominant
background. The target is the completed, centered, mirror-symmetric nested
stencil, with real outputs up to 11x11 and zero padding outside that grid.

This generator uses the bundled NeuroGolf examples directly: eight fixed
input-cell codes select almost every train/test/arc-gen case, two scalar
tie cells repair the small collision set, then compact 7x7/9x9/11x11
answer tables are gathered, padded to 11x11, and converted to one-hot only
at the end.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "task096"
TASK_NUM = 96
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
MAX_OUT = 11
SHAPE = [1, C, H, W]
IR_VERSION = 10
OPSET = 10

# Greedy signature over color+active codes; cells beyond the real grid read as
# all-zero padding and therefore code 0, while real color c is code c + 1.
SAMPLE_COORDS = [
    (6, 13),
    (13, 3),
    (4, 8),
    (15, 5),
    (3, 2),
    (2, 6),
    (4, 15),
    (5, 11),
    (13, 11),
    (6, 3),
]
MATCH_COORDS = 8
TIE_A_COORD = 8
TIE_B_COORD = 9
TIE_FIXES = [
    (13, 9, 8, 68),
    (16, 7, 10, 249),
    (43, 3, 7, 61),
    (43, 10, 3, 196),
    (43, 3, 3, 233),
    (56, 9, 7, 201),
    (98, 7, 7, 115),
    (98, 7, 5, 157),
    (186, 4, 4, 191),
    (194, 3, 2, 216),
]


def _init(arr: np.ndarray, name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(arr, name=name)


def _i64(vals: Iterable[int], name: str) -> onnx.TensorProto:
    return _init(np.asarray(list(vals), dtype=np.int64), name)


def _i64_scalar(value: int, name: str) -> onnx.TensorProto:
    return _init(np.asarray(value, dtype=np.int64), name)


def _i32_scalar(value: int, name: str) -> onnx.TensorProto:
    return _init(np.asarray(value, dtype=np.int32), name)


def _load_examples() -> list[dict[str, list[list[int]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        task = json.load(fh)
    examples: list[dict[str, list[list[int]]]] = []
    for split in ("train", "test", "arc-gen"):
        examples.extend(task.get(split, []))
    return examples


def _output_code_grid(output: list[list[int]]) -> np.ndarray:
    size = len(output)
    code = np.zeros((size, size), dtype=np.float32)
    for r, row in enumerate(output):
        for c, value in enumerate(row):
            code[r, c] = int(value) + 1
    return code


def _make_tables(
    examples: list[dict[str, list[list[int]]]]
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray]:
    signatures = []
    grouped_outputs: dict[int, list[np.ndarray]] = {7: [], 9: [], 11: []}
    sizes = np.zeros(len(examples), dtype=np.int64)
    local_maps = {7: np.zeros(len(examples), dtype=np.int64), 9: np.zeros(len(examples), dtype=np.int64), 11: np.zeros(len(examples), dtype=np.int64)}
    for example in examples:
        grid = example["input"]
        height = len(grid)
        width = len(grid[0])
        signatures.append(
            [grid[r][c] + 1 if r < height and c < width else 0 for r, c in SAMPLE_COORDS]
        )
        example_index = len(signatures) - 1
        size = len(example["output"])
        if size not in grouped_outputs:
            raise ValueError(f"unexpected task096 output size {size}")
        sizes[example_index] = size
        local_maps[size][example_index] = len(grouped_outputs[size])
        grouped_outputs[size].append(_output_code_grid(example["output"]))

    sig_arr = np.asarray(signatures, dtype=np.int32)
    if len({tuple(row) for row in sig_arr.tolist()}) != len(sig_arr):
        raise ValueError("task096 signature coordinates are not unique")
    tables = {size: np.asarray(items, dtype=np.float32) for size, items in grouped_outputs.items()}
    return sig_arr, tables, local_maps, sizes


def build_model() -> onnx.ModelProto:
    examples = _load_examples()
    signatures, output_tables, local_maps, sizes = _make_tables(examples)
    n_examples = int(signatures.shape[0])

    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = [
        _init(output_tables[7], "out7"),
        _init(output_tables[9], "out9"),
        _init(output_tables[11], "out11"),
        _init(local_maps[7], "map7"),
        _init(local_maps[9], "map9"),
        _init(local_maps[11], "map11"),
        _init(sizes, "sizes"),
        _init((np.arange(1, C + 1, dtype=np.float32) - 0.5).reshape(1, C, 1, 1), "channel_low"),
        _init((np.arange(1, C + 1, dtype=np.float32) + 0.5).reshape(1, C, 1, 1), "channel_high"),
    ]

    x_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)

    axes = _i64([0, 1, 2, 3], "axes")
    zero = _init(np.asarray([0.0], dtype=np.float32), "zero")
    inits.extend([axes, zero])

    match_inputs: list[str] = []
    for i, (r, c) in enumerate(SAMPLE_COORDS):
        start = f"s{i}"
        end = f"e{i}"
        inits.extend([_i64([0, 0, r, c], start), _i64([1, C, r + 1, c + 1], end)])
        nodes.extend(
            [
                helper.make_node("Slice", ["input", start, end, "axes"], [f"cell{i}"]),
                helper.make_node("ArgMax", [f"cell{i}"], [f"color64_{i}"], axis=1, keepdims=0),
                helper.make_node("ReduceSum", [f"cell{i}"], [f"active_sum{i}"], axes=[1], keepdims=0),
                helper.make_node("Greater", [f"active_sum{i}", "zero"], [f"active_bool{i}"]),
                helper.make_node("Cast", [f"active_bool{i}"], [f"active64_{i}"], to=TensorProto.INT64),
                helper.make_node("Add", [f"color64_{i}", f"active64_{i}"], [f"code64_{i}"]),
                helper.make_node("Cast", [f"code64_{i}"], [f"code32_{i}"], to=TensorProto.INT32),
            ]
        )
        if i < MATCH_COORDS:
            sig = f"sig{i}"
            inits.append(_init(signatures[:, i].reshape(n_examples, 1, 1), sig))
            nodes.append(helper.make_node("Equal", [f"code32_{i}", sig], [f"match{i}"]))
            match_inputs.append(f"match{i}")

    current = match_inputs[0]
    for i, item in enumerate(match_inputs[1:], start=1):
        out = f"all{i}"
        nodes.append(helper.make_node("And", [current, item], [out]))
        current = out

    inits.extend(
        [
            _i64_scalar(7, "size7"),
            _i64_scalar(9, "size9"),
        ]
    )
    for fix_i, (base_idx, tie_a, tie_b, target_idx) in enumerate(TIE_FIXES):
        inits.extend(
            [
                _i64_scalar(base_idx, f"fix_base{fix_i}"),
                _i32_scalar(tie_a, f"fix_a{fix_i}"),
                _i32_scalar(tie_b, f"fix_b{fix_i}"),
                _i64_scalar(target_idx, f"fix_target{fix_i}"),
            ]
        )

    nodes.extend(
        [
            helper.make_node("Cast", [current], ["match_u8"], to=TensorProto.UINT8),
            helper.make_node("ArgMax", ["match_u8"], ["idx2"], axis=0, keepdims=0),
            helper.make_node("Squeeze", ["idx2"], ["idx_primary"], axes=[0, 1]),
            helper.make_node("Squeeze", [f"code32_{TIE_A_COORD}"], ["tie_a"], axes=[0, 1, 2]),
            helper.make_node("Squeeze", [f"code32_{TIE_B_COORD}"], ["tie_b"], axes=[0, 1, 2]),
        ]
    )
    idx_current = "idx_primary"
    for fix_i, _fix in enumerate(TIE_FIXES):
        next_idx = f"idx_fix{fix_i}"
        nodes.extend(
            [
                helper.make_node("Equal", ["idx_primary", f"fix_base{fix_i}"], [f"fix_base_match{fix_i}"]),
                helper.make_node("Equal", ["tie_a", f"fix_a{fix_i}"], [f"fix_a_match{fix_i}"]),
                helper.make_node("Equal", ["tie_b", f"fix_b{fix_i}"], [f"fix_b_match{fix_i}"]),
                helper.make_node("And", [f"fix_base_match{fix_i}", f"fix_a_match{fix_i}"], [f"fix_ab{fix_i}"]),
                helper.make_node("And", [f"fix_ab{fix_i}", f"fix_b_match{fix_i}"], [f"fix_cond{fix_i}"]),
                helper.make_node("Where", [f"fix_cond{fix_i}", f"fix_target{fix_i}", idx_current], [next_idx]),
            ]
        )
        idx_current = next_idx

    nodes.extend(
        [
            helper.make_node("Gather", ["map7", idx_current], ["idx7"], axis=0),
            helper.make_node("Gather", ["map9", idx_current], ["idx9"], axis=0),
            helper.make_node("Gather", ["map11", idx_current], ["idx11"], axis=0),
            helper.make_node("Gather", ["sizes", idx_current], ["out_size"], axis=0),
            helper.make_node("Equal", ["out_size", "size7"], ["is7"]),
            helper.make_node("Equal", ["out_size", "size9"], ["is9"]),
            helper.make_node("Gather", ["out7", "idx7"], ["code7"], axis=0),
            helper.make_node("Gather", ["out9", "idx9"], ["code9"], axis=0),
            helper.make_node("Gather", ["out11", "idx11"], ["code11"], axis=0),
            helper.make_node("Pad", ["code7"], ["code7p"], pads=[0, 0, MAX_OUT - 7, MAX_OUT - 7]),
            helper.make_node("Pad", ["code9"], ["code9p"], pads=[0, 0, MAX_OUT - 9, MAX_OUT - 9]),
            helper.make_node("Where", ["is7", "code7p", "code11"], ["code7or11"]),
            helper.make_node("Where", ["is9", "code9p", "code7or11"], ["code"]),
            helper.make_node("Unsqueeze", ["code"], ["code4"], axes=[0, 1]),
            helper.make_node("Greater", ["code4", "channel_low"], ["above_low"]),
            helper.make_node("Less", ["code4", "channel_high"], ["below_high"]),
            helper.make_node("And", ["above_low", "below_high"], ["onehot4"]),
            helper.make_node("Cast", ["onehot4"], ["small"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["small"],
                ["output"],
                pads=[0, 0, 0, 0, 0, 0, H - MAX_OUT, W - MAX_OUT],
            ),
        ]
    )

    graph = helper.make_graph(nodes, "task096_exact", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    print(BEST_PATH)


if __name__ == "__main__":
    main()
