"""Build a specialized ONNX solver for NeuroGolf task079.

Task rule: the 14x14 input contains several colored 3x3 pattern instances on
black. Each valid object window uses exactly one nonzero color, has at least
four filled cells, and touches every row and column of its 3x3 bounding window.
The output is the 3x3 binary pattern that appears most often, rendered in the
color of one winning occurrence and black elsewhere.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from score_model import calculate_params, convert_to_numpy, score_file  # noqa: E402

TASK_JSON = ROOT / "data" / "task079.json"
OUT_PATH = ROOT / "solution.onnx"
TASK_COPY_PATH = ROOT / "task079.onnx"

C = 10
NC = 9
H = W = 30
GH = GW = 14
WH = WW = 12
N = NC * WH * WW
IN_NAME = "input"
OUT_NAME = "output"


def init(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def scalar_f(inits: list[onnx.TensorProto], name: str, value: float) -> str:
    return init(inits, name, np.asarray([value], dtype=np.float32))


def scalar_i(inits: list[onnx.TensorProto], name: str, value: int) -> str:
    return init(inits, name, np.asarray([value], dtype=np.int64))


def arc_grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros((1, C, H, W), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def decode_onehot(arr: np.ndarray, rows: int = 3, cols: int = 3) -> np.ndarray:
    return np.argmax(arr[0, :, :rows, :cols], axis=0).astype(np.int64)


def reference(grid: list[list[int]]) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    weights = (2 ** np.arange(9, dtype=np.int64)).reshape(3, 3)
    counts: dict[int, int] = {}
    colors: dict[int, int] = {}
    for r in range(WH):
        for c in range(WW):
            w = g[r : r + 3, c : c + 3]
            nonzero_colors = sorted(set(int(x) for x in w.ravel()) - {0})
            if len(nonzero_colors) != 1:
                continue
            mask = w > 0
            if int(mask.sum()) < 4 or not mask.any(axis=0).all() or not mask.any(axis=1).all():
                continue
            code = int((mask * weights).sum())
            counts[code] = counts.get(code, 0) + 1
            colors.setdefault(code, nonzero_colors[0])
    best_code = max(counts, key=lambda code: counts[code])
    mask = np.asarray([(best_code >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    out = np.zeros((3, 3), dtype=np.int64)
    out[mask] = colors[best_code]
    return out


def make_model(name: str, *, variant: str) -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    starts = init(inits, "starts", np.asarray([1, 0, 0], dtype=np.int64))
    ends = init(inits, "ends", np.asarray([10, GH, GW], dtype=np.int64))
    axes = init(inits, "axes", np.asarray([1, 2, 3], dtype=np.int64))
    flat_shape = init(inits, "flat_shape", np.asarray([N], dtype=np.int64))
    one_n_shape = init(inits, "one_n_shape", np.asarray([1, N], dtype=np.int64))
    tiny_shape = init(inits, "tiny_shape", np.asarray([1, C, 3, 3], dtype=np.int64))
    colors = init(inits, "colors", np.arange(C, dtype=np.int64).reshape(C, 1))
    bg_channel = init(inits, "bg_channel", np.asarray([[1.0]] + [[0.0]] * 9, dtype=np.float32))
    one = scalar_f(inits, "one", 1.0)
    half = scalar_f(inits, "half", 0.5)
    three_half = scalar_f(inits, "three_half", 3.5)

    nodes.append(helper.make_node("Slice", [IN_NAME, starts, ends, axes], ["crop"]))
    nodes.append(helper.make_node("Squeeze", ["crop"], ["x"], axes=[0]))

    cells: list[str] = []
    for dr in range(3):
        for dc in range(3):
            s = init(inits, f"s_{dr}_{dc}", np.asarray([0, dr, dc], dtype=np.int64))
            e = init(inits, f"e_{dr}_{dc}", np.asarray([NC, dr + WH, dc + WW], dtype=np.int64))
            a = init(inits, f"a_{dr}_{dc}", np.asarray([0, 1, 2], dtype=np.int64))
            out = f"cell_{dr}_{dc}"
            nodes.append(helper.make_node("Slice", ["x", s, e, a], [out]))
            cells.append(out)

    def add_many(inputs: list[str], prefix: str) -> str:
        acc = inputs[0]
        for idx, name in enumerate(inputs[1:], start=1):
            out = f"{prefix}_{idx}"
            nodes.append(helper.make_node("Add", [acc, name], [out]))
            acc = out
        return acc

    color_count = add_many(cells, "color_count")
    code_sum: str | None = None
    masks: str | None = None
    if variant != "color_window_count":
        masks = init(
            inits,
            "masks",
            (((np.arange(512, dtype=np.int64)[:, None] >> np.arange(9, dtype=np.int64)) & 1).astype(np.float32)),
        )
        weighted_terms = []
        for idx, cell in enumerate(cells):
            if idx == 0:
                weighted_terms.append(cell)
                continue
            weight = scalar_f(inits, f"w_{idx}", float(2**idx))
            out = f"weighted_{idx}"
            nodes.append(helper.make_node("Mul", [cell, weight], [out]))
            weighted_terms.append(out)
        code_sum = add_many(weighted_terms, "code_sum")

    nodes.extend(
        [
            helper.make_node("ReduceSum", [color_count], ["total_count"], axes=[0], keepdims=0),
            helper.make_node("Sub", ["total_count", color_count], ["other_count"]),
            helper.make_node("Less", ["other_count", half], ["one_color"]),
            helper.make_node("Greater", [color_count, three_half], ["enough"]),
        ]
    )

    row_masks = []
    col_masks = []
    for i, idxs in enumerate(((0, 1, 2), (3, 4, 5), (6, 7, 8))):
        nodes.append(helper.make_node("Add", [cells[idxs[0]], cells[idxs[1]]], [f"r{i}_a"]))
        nodes.append(helper.make_node("Add", [f"r{i}_a", cells[idxs[2]]], [f"r{i}_s"]))
        nodes.append(helper.make_node("Greater", [f"r{i}_s", half], [f"r{i}_b"]))
        row_masks.append(f"r{i}_b")
    for i, idxs in enumerate(((0, 3, 6), (1, 4, 7), (2, 5, 8))):
        nodes.append(helper.make_node("Add", [cells[idxs[0]], cells[idxs[1]]], [f"c{i}_a"]))
        nodes.append(helper.make_node("Add", [f"c{i}_a", cells[idxs[2]]], [f"c{i}_s"]))
        nodes.append(helper.make_node("Greater", [f"c{i}_s", half], [f"c{i}_b"]))
        col_masks.append(f"c{i}_b")

    nodes.extend(
        [
            helper.make_node("And", [row_masks[0], row_masks[1]], ["rows01"]),
            helper.make_node("And", ["rows01", row_masks[2]], ["rows_ok"]),
            helper.make_node("And", [col_masks[0], col_masks[1]], ["cols01"]),
            helper.make_node("And", ["cols01", col_masks[2]], ["cols_ok"]),
            helper.make_node("And", ["one_color", "enough"], ["v0"]),
            helper.make_node("And", ["v0", "rows_ok"], ["v1"]),
            helper.make_node("And", ["v1", "cols_ok"], ["valid_9_12_12"]),
            helper.make_node("Reshape", ["valid_9_12_12", flat_shape], ["valid_flat"]),
        ]
    )

    if variant == "color_window_count":
        one_i64 = scalar_i(inits, "one_i64", 1)
        valid_9_144 = init(inits, "valid_9_144_shape", np.asarray([NC, WH * WW], dtype=np.int64))
        flat_144 = init(inits, "flat_144_shape", np.asarray([WH * WW], dtype=np.int64))
        nodes.extend(
            [
                helper.make_node("Cast", ["valid_9_12_12"], ["valid_f"], to=TensorProto.FLOAT),
                helper.make_node("ReduceSum", ["valid_f"], ["counts"], axes=[1, 2], keepdims=0),
                helper.make_node("ArgMax", ["counts"], ["win_color0"], axis=0, keepdims=0),
                helper.make_node("Add", ["win_color0", one_i64], ["win_color"]),
                helper.make_node("Reshape", ["valid_f", valid_9_144], ["valid_by_color"]),
                helper.make_node("Gather", ["valid_by_color", "win_color0"], ["winning_color_valid"], axis=0),
                helper.make_node("ArgMax", ["winning_color_valid"], ["win_pos"], axis=0, keepdims=0),
            ]
        )
        mask_parts = []
        for idx, cell in enumerate(cells):
            color_row = f"out_cell_{idx}_color"
            flat_color = f"out_cell_{idx}_flat_color"
            bit = f"out_cell_{idx}_bit"
            bit_u = f"out_cell_{idx}_u"
            nodes.extend(
                [
                    helper.make_node("Gather", [cell, "win_color0"], [color_row], axis=0),
                    helper.make_node("Reshape", [color_row, flat_144], [flat_color]),
                    helper.make_node("Gather", [flat_color, "win_pos"], [bit], axis=0),
                    helper.make_node("Unsqueeze", [bit], [bit_u], axes=[0]),
                ]
            )
            mask_parts.append(bit_u)
        nodes.append(helper.make_node("Concat", mask_parts, ["win_mask"], axis=0))
    elif variant == "codebook":
        assert code_sum is not None and masks is not None
        all_codes = init(inits, "all_codes", np.arange(512, dtype=np.int64).reshape(512, 1))
        candidate_colors = init(
            inits,
            "candidate_colors",
            np.repeat(np.arange(1, 10, dtype=np.int64), WH * WW),
        )
        nodes.extend(
            [
                helper.make_node("Reshape", [code_sum, flat_shape], ["code_flat"]),
                helper.make_node("Cast", ["code_flat"], ["code_flat_i64"], to=TensorProto.INT64),
                helper.make_node("Reshape", ["code_flat_i64", one_n_shape], ["code_row"]),
                helper.make_node("Equal", [all_codes, "code_row"], ["eq_code"]),
                helper.make_node("Reshape", ["valid_flat", one_n_shape], ["valid_row"]),
                helper.make_node("And", ["eq_code", "valid_row"], ["eq_valid"]),
                helper.make_node("Cast", ["eq_valid"], ["eq_valid_f"], to=TensorProto.FLOAT),
                helper.make_node("ReduceSum", ["eq_valid_f"], ["counts"], axes=[1], keepdims=0),
                helper.make_node("ArgMax", ["counts"], ["win_code_i64"], axis=0, keepdims=0),
                helper.make_node("Equal", ["code_flat_i64", "win_code_i64"], ["code_match"]),
                helper.make_node("And", ["code_match", "valid_flat"], ["winning_candidates"]),
                helper.make_node("Cast", ["winning_candidates"], ["winning_candidates_f"], to=TensorProto.FLOAT),
                helper.make_node("ArgMax", ["winning_candidates_f"], ["win_candidate"], axis=0, keepdims=0),
                helper.make_node("Gather", [candidate_colors, "win_candidate"], ["win_color"], axis=0),
                helper.make_node("Gather", [masks, "win_code_i64"], ["win_mask"], axis=0),
            ]
        )
    elif variant == "pairwise":
        assert code_sum is not None and masks is not None
        candidate_colors = init(
            inits,
            "candidate_colors",
            np.repeat(np.arange(1, 10, dtype=np.int64), WH * WW),
        )
        nodes.extend(
            [
                helper.make_node("Reshape", [code_sum, flat_shape], ["code_flat"]),
                helper.make_node("Cast", ["code_flat"], ["code_flat_i64"], to=TensorProto.INT64),
                helper.make_node("Reshape", ["code_flat_i64", one_n_shape], ["code_row"]),
                helper.make_node("Reshape", ["code_flat_i64", np_name(inits, "n_one_shape", [N, 1])], ["code_col"]),
                helper.make_node("Equal", ["code_col", "code_row"], ["eq_code_nn"]),
                helper.make_node("Reshape", ["valid_flat", one_n_shape], ["valid_row"]),
                helper.make_node("And", ["eq_code_nn", "valid_row"], ["eq_valid_nn"]),
                helper.make_node("Cast", ["eq_valid_nn"], ["eq_valid_nn_f"], to=TensorProto.FLOAT),
                helper.make_node("ReduceSum", ["eq_valid_nn_f"], ["candidate_counts"], axes=[1], keepdims=0),
                helper.make_node("Cast", ["valid_flat"], ["valid_flat_f"], to=TensorProto.FLOAT),
                helper.make_node("Mul", ["candidate_counts", "valid_flat_f"], ["candidate_scores"]),
                helper.make_node("ArgMax", ["candidate_scores"], ["win_candidate"], axis=0, keepdims=0),
                helper.make_node("Gather", [candidate_colors, "win_candidate"], ["win_color"], axis=0),
                helper.make_node("Gather", ["code_flat_i64", "win_candidate"], ["win_code_i64"], axis=0),
                helper.make_node("Gather", [masks, "win_code_i64"], ["win_mask"], axis=0),
            ]
        )
    else:
        raise ValueError(variant)

    nodes.extend(
        [
            helper.make_node("Sub", [one, "win_mask"], ["inv_mask"]),
            helper.make_node("Equal", [colors, "win_color"], ["color_eq"]),
            helper.make_node("Cast", ["color_eq"], ["color_eq_f"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["color_eq_f", "win_mask"], ["fg_out"]),
            helper.make_node("Mul", [bg_channel, "inv_mask"], ["bg_out"]),
            helper.make_node("Add", ["fg_out", "bg_out"], ["tiny_10_9"]),
            helper.make_node("Reshape", ["tiny_10_9", tiny_shape], ["tiny"]),
            helper.make_node(
                "Pad",
                ["tiny"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 0, H - 3, W - 3],
                value=0.0,
            ),
        ]
    )

    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, [1, C, H, W])],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, [1, C, H, W])],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="task079",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 10)],
    )
    onnx.checker.check_model(model)
    return model


def np_name(inits: list[onnx.TensorProto], name: str, values: Any) -> str:
    return init(inits, name, np.asarray(values, dtype=np.int64))


def validate_model(model: onnx.ModelProto) -> tuple[int, int, int]:
    onnx.checker.check_model(model, full_check=True)
    ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])

    with TASK_JSON.open(encoding="utf-8") as fh:
        task = json.load(fh)

    total = 0
    correct = 0
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(task.get(split, [])):
            total += 1
            pred = session.run([OUT_NAME], {IN_NAME: arc_grid_to_onehot(ex["input"])})[0]
            pred_grid = decode_onehot(pred, len(ex["output"]), len(ex["output"][0]))
            expected = np.asarray(ex["output"], dtype=np.int64)
            ref = reference(ex["input"])
            if not np.array_equal(ref, expected):
                raise AssertionError(f"reference mismatch on {split} {idx}: {ref.tolist()} != {expected.tolist()}")
            if not np.array_equal(pred_grid, expected):
                raise AssertionError(f"ONNX mismatch on {split} {idx}: {pred_grid.tolist()} != {expected.tolist()}")
            correct += 1
            if split == "train":
                print(f"{split} {idx} output:")
                for row in pred_grid.tolist():
                    print("".join(str(int(v)) for v in row))
    params = calculate_params(model)
    if params is None:
        raise AssertionError("parameter count failed")
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    internal_memory = 0
    for value in inferred.graph.value_info:
        tensor_type = value.type.tensor_type
        shape = [dim.dim_value for dim in tensor_type.shape.dim]
        if not shape or any(dim <= 0 for dim in shape):
            continue
        itemsize = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)).itemsize
        internal_memory += int(math.prod(shape) * itemsize)
    return correct, total, params, internal_memory


def main() -> None:
    attempts = [
        ("color_window_count", "color_window_count"),
        ("codebook_512x1296", "codebook"),
        ("candidate_pairwise_1296x1296", "pairwise"),
    ]
    results = []
    for name, variant in attempts:
        print(f"\nBuilding {name}")
        model = make_model(name, variant=variant)
        tmp_path = ROOT / f"task079_{name}.onnx"
        onnx.save(model, tmp_path)
        correct, total, params, inferred_memory = validate_model(model)
        score = score_file(tmp_path)
        print(f"validated examples: {correct}/{total}")
        print(f"initializer/constant params: {params}")
        print(f"inferred internal memory sum: {inferred_memory}")
        print(
            "measured:",
            f"valid={score['valid']}",
            f"memory={score['memory']}",
            f"params={score['params']}",
            f"cost={score['cost']}",
            f"score={score['score']}",
        )
        results.append((score["cost"] if score["valid"] else 10**18, name, model, tmp_path, score))

    results.sort(key=lambda item: item[0])
    _, best_name, best_model, _, best_score = results[0]
    onnx.save(best_model, OUT_PATH)
    onnx.save(best_model, TASK_COPY_PATH)
    print("\nAttempt comparison:")
    for cost, name, _, tmp_path, score in results:
        print(f"{name}: cost={cost} score={score['score']} path={tmp_path.name}")
    print(f"\nKept {best_name} -> {OUT_PATH.name} and {TASK_COPY_PATH.name}")
    print(
        f"Final measured cost={best_score['cost']} memory={best_score['memory']} "
        f"params={best_score['params']} score={best_score['score']}"
    )


if __name__ == "__main__":
    main()
