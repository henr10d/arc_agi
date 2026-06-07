"""Minimal ONNX for ARC task010: recolor four fixed vertical bars by height.

Task rule: the 9x9 input contains four color-5 vertical bars in columns
1, 3, 5, and 7. Each bar is bottom-aligned and extends to row 8. The output
keeps the same bar cells but recolors by relative height among the four bars:
longest -> color 1, second -> color 2, third -> color 3, shortest -> color 4.
All other cells are background color 0.

ONNX approach: read only the four bar columns from background channel 0, infer
bar masks and scalar heights, rank the four heights with scalar comparisons,
build a compact bool [1, 5, 9, 9] one-hot grid, then Pad once to the required
[1, 10, 30, 30] output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "task010.json"
BEST_PATH = OUT_DIR / "task010.onnx"

C = 10
H = W = 30
CORE = 9
BAR_COLS = [1, 3, 5, 7]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def _i64(inits: List[onnx.TensorProto], vals: List[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))
    return name


def _f32(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name))
    return name


def _bool(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.bool_), name=name))
    return name


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solution for task010."""
    x = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(x)
    heights: Dict[int, int] = {}
    for col in BAR_COLS:
        heights[col] = int(np.count_nonzero(x[:, col]))
    ranked = sorted(heights.items(), key=lambda item: item[1], reverse=True)
    for color, (col, _) in enumerate(ranked, start=1):
        out[x[:, col] != 0, col] = color
    return out


def _grid_to_onehot(grid: np.ndarray) -> np.ndarray:
    out = np.zeros((1, C, H, W), dtype=np.float32)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            out[0, int(grid[r, c]), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return np.asarray(onehot).reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _and3(nodes: List[onnx.NodeProto], a: str, b: str, c: str, out: str) -> None:
    tmp = f"{out}_ab"
    nodes.append(helper.make_node("And", [a, b], [tmp]))
    nodes.append(helper.make_node("And", [tmp, c], [out]))


def build_onnx_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, [1, C, H, W])
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, [1, C, H, W])

    half = _f32(inits, 0.5, "half")
    zero_i = _i64(inits, [0], "zero_i")
    one_i = _i64(inits, [1], "one_i")
    two_i = _i64(inits, [2], "two_i")
    three_i = _i64(inits, [3], "three_i")
    false_scalar = _bool(inits, np.zeros((1, 1, 1, 1), dtype=np.bool_), "false_scalar")
    false_col = _bool(inits, np.zeros((1, 1, CORE, 1), dtype=np.bool_), "false_col")

    masks: List[str] = []
    heights: List[str] = []
    for i, col in enumerate(BAR_COLS):
        start = _i64(inits, [0, 0, 0, col], f"s{i}")
        end = _i64(inits, [1, 1, CORE, col + 1], f"e{i}")
        nodes.extend(
            [
                helper.make_node("Slice", [IN_NAME, start, end], [f"ch0_{i}"]),
                helper.make_node("Less", [f"ch0_{i}", half], [f"bar{i}"]),
                helper.make_node("Cast", [f"bar{i}"], [f"bar{i}_f"], to=TensorProto.FLOAT),
                helper.make_node("ReduceSum", [f"bar{i}_f"], [f"h{i}"], axes=[2, 3], keepdims=1),
            ]
        )
        masks.append(f"bar{i}")
        heights.append(f"h{i}")

    rank_counts: List[str] = []
    for i in range(4):
        gt_terms: List[str] = []
        for j in range(4):
            if i == j:
                continue
            gt = f"gt{i}{j}"
            gt_i = f"{gt}_i"
            nodes.extend(
                [
                    helper.make_node("Greater", [heights[i], heights[j]], [gt]),
                    helper.make_node("Cast", [gt], [gt_i], to=TensorProto.INT64),
                ]
            )
            gt_terms.append(gt_i)
        nodes.append(helper.make_node("Add", [gt_terms[0], gt_terms[1]], [f"rank{i}_a"]))
        nodes.append(helper.make_node("Add", [f"rank{i}_a", gt_terms[2]], [f"rank{i}"]))
        rank_counts.append(f"rank{i}")

    color_scalars: Dict[int, List[str]] = {1: [], 2: [], 3: [], 4: []}
    rank_to_const = {3: three_i, 2: two_i, 1: one_i, 0: zero_i}
    for i, rank_name in enumerate(rank_counts):
        for rank_count, color in ((3, 1), (2, 2), (1, 3), (0, 4)):
            name = f"bar{i}_is_c{color}"
            nodes.append(helper.make_node("Equal", [rank_name, rank_to_const[rank_count]], [name]))
            color_scalars[color].append(name)

    nodes.append(
        helper.make_node(
            "Concat",
            [
                false_col,
                masks[0],
                false_col,
                masks[1],
                false_col,
                masks[2],
                false_col,
                masks[3],
                false_col,
            ],
            ["object9"],
            axis=3,
        )
    )

    color_channels: List[str] = []
    for color in (1, 2, 3, 4):
        colmask = f"colmask_c{color}"
        nodes.append(
            helper.make_node(
                "Concat",
                [
                    false_scalar,
                    color_scalars[color][0],
                    false_scalar,
                    color_scalars[color][1],
                    false_scalar,
                    color_scalars[color][2],
                    false_scalar,
                    color_scalars[color][3],
                    false_scalar,
                ],
                [colmask],
                axis=3,
            )
        )
        out_ch = f"out_c{color}"
        nodes.append(helper.make_node("And", ["object9", colmask], [out_ch]))
        color_channels.append(out_ch)

    nodes.extend(
        [
            helper.make_node("Not", ["object9"], ["bg9"]),
            helper.make_node("Concat", ["bg9", *color_channels], ["out5"], axis=1),
            helper.make_node("Cast", ["out5"], ["out5_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out5_f"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 0, 0, 0, 0, 5, H - CORE, W - CORE],
            ),
        ]
    )

    graph = helper.make_graph(nodes, "task010", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def save_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    model = build_onnx_model()
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return model


def model_stats(model: onnx.ModelProto) -> Dict[str, int]:
    params = sum(int(np.prod(list(t.dims))) for t in model.graph.initializer)
    return {"nodes": len(model.graph.node), "params": params, "inits": len(model.graph.initializer)}


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def inspect_task_data() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    rank_colors: Dict[int, set[int]] = {0: set(), 1: set(), 2: set(), 3: set()}
    print("task010 data")
    for split in ("train", "test", "arc-gen"):
        examples = data.get(split, [])
        print(f"{split}: {len(examples)} examples")
        for idx, ex in enumerate(examples):
            inp = np.asarray(ex["input"], dtype=np.int64)
            out = np.asarray(ex["output"], dtype=np.int64)
            nz = np.argwhere(inp != 0)
            cols = sorted(set(nz[:, 1].tolist())) if nz.size else []
            heights = [(col, int(np.count_nonzero(inp[:, col]))) for col in cols]
            for rank, (col, height) in enumerate(sorted(heights, key=lambda item: item[1])):
                vals = sorted(set(out[inp[:, col] != 0, col].tolist()))
                if len(vals) == 1:
                    rank_colors[rank].add(vals[0])
            if idx < 3:
                print(
                    f"  {idx}: in={inp.shape} out={out.shape} cols={cols} "
                    f"in_colors={sorted(set(inp.ravel()))} out_colors={sorted(set(out.ravel()))} "
                    f"heights={heights}"
                )
    print("rank colors shortest..longest:", {k: sorted(v) for k, v in rank_colors.items()})


def test() -> None:
    model = build_onnx_model()
    stats = model_stats(model)
    print(f"nodes={stats['nodes']} params={stats['params']} inits={stats['inits']}")

    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    bad = 0
    split_totals: Dict[str, int] = {}
    for split in ("train", "test", "arc-gen"):
        split_bad = 0
        examples = data.get(split, [])
        for ex in examples:
            inp = np.asarray(ex["input"], dtype=np.int64)
            exp = np.asarray(ex["output"], dtype=np.int64)
            ref = solve(inp)
            if not np.array_equal(ref, exp):
                raise AssertionError(f"reference mismatch in {split}")
            pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(inp))[0])
            expected_padded = np.zeros((H, W), dtype=np.int64)
            expected_padded[: exp.shape[0], : exp.shape[1]] = exp
            if not np.array_equal(pred, expected_padded):
                split_bad += 1
                bad += 1
        split_totals[split] = len(examples) - split_bad
        print(f"task010 {split}: {'PASS' if split_bad == 0 else f'FAIL ({split_bad} wrong)'}")
    if bad:
        raise SystemExit(1)


def main() -> None:
    inspect_task_data()
    save_model()
    test()
    print(f"saved {BEST_PATH}")


if __name__ == "__main__":
    main()
