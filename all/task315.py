"""ONNX solution for ARC task315: copy the 3x3 input at every color-2 cell.

Task rule: the input uses colors 0, 1, and 2.  Color 0 is background and
color 1 is inert.  For each input cell whose value is 2, paste a full copy of
the input grid into the corresponding block of the output; all other blocks are
background.  In the provided data the input is 3x3, so the output is a 3x3
arrangement of 3x3 copies, padded to the NeuroGolf 30x30 one-hot tensor.

The best graph below builds the 9 possible blocks directly from a compact
boolean 3x3 core, concatenates them into the 9x9 result, casts that compact
result to float, and pads only at the graph output.
"""

from __future__ import annotations

import json
import sys
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

TASK_ID = "task315"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
ACTIVE_C = 3
H = W = 30
CORE = 3
OUT = CORE * CORE
PAD = H - OUT
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference solver: paste the whole grid at every position containing 2."""
    arr = np.asarray(grid, dtype=np.int64)
    rows, cols = arr.shape
    out = np.zeros((rows * rows, cols * cols), dtype=np.int64)
    for r in range(rows):
        for c in range(cols):
            if arr[r, c] == 2:
                out[r * rows : (r + 1) * rows, c * cols : (c + 1) * cols] = arr
    return out


def diagonal_hypothesis(grid: list[list[int]] | np.ndarray, *, swap: bool = False) -> np.ndarray:
    """The rejected screenshot hypothesis, kept as a regression check."""
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros((arr.shape[0] * 3, arr.shape[1] * 3), dtype=np.int64)
    main_color, anti_color = (1, 2) if swap else (2, 1)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            color = int(arr[r, c])
            if color == main_color:
                for k in range(3):
                    out[r * 3 + k, c * 3 + k] = color
            elif color == anti_color:
                for k in range(3):
                    out[r * 3 + k, c * 3 + 2 - k] = color
    return out


def _init(inits: list[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _bool(inits: list[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=bool), name)


def _grid_to_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(arr):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def _expected_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    return _grid_to_onehot(grid)


def _strict_onehot_matches(pred: np.ndarray, expected: np.ndarray) -> bool:
    return pred.shape == expected.shape and np.array_equal(pred > 0.0, expected > 0.0)


def _load_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def verify_hypotheses() -> dict[str, int]:
    """Return mismatch counts for the requested hypotheses and confirmed rule."""
    data = _load_data()
    results: dict[str, int] = {}
    for name, fn in {
        "diagonal": diagonal_hypothesis,
        "diagonal_swapped": lambda x: diagonal_hypothesis(x, swap=True),
        "copy_on_2": solve,
    }.items():
        mismatches = 0
        for split in ("train", "test", "arc-gen"):
            for ex in data.get(split, []):
                expected = np.asarray(ex["output"], dtype=np.int64)
                actual = fn(ex["input"])
                if actual.shape != expected.shape or not np.array_equal(actual, expected):
                    mismatches += 1
        results[name] = mismatches
    return results


def colors_in_data() -> list[int]:
    data = _load_data()
    colors: set[int] = set()
    for examples in data.values():
        for ex in examples:
            for key in ("input", "output"):
                colors.update(int(v) for row in ex[key] for v in row)
    return sorted(colors)


def build_concat_blocks_model() -> onnx.ModelProto:
    """Lowest-cost model found locally: direct logical blocks plus Concat."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, [0, 1, 2, 3], "axes")
    core_st = _i64(inits, [0, 0, 0, 0], "core_st")
    core_en = _i64(inits, [1, ACTIVE_C, CORE, CORE], "core_en")
    ch2_st = _i64(inits, [0, 2, 0, 0], "ch2_st")
    ch2_en = _i64(inits, [1, 3, CORE, CORE], "ch2_en")
    flat_shape = _i64(inits, [CORE * CORE], "flat_shape")
    zero = _f32(inits, [0.0], "zero")
    bg = np.zeros((1, ACTIVE_C, 1, 1), dtype=bool)
    bg[:, 0, :, :] = True
    bg_name = _bool(inits, bg, "bg")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, core_st, core_en, axes], ["core"]),
            helper.make_node("Greater", ["core", zero], ["coreb"]),
            helper.make_node("Slice", ["coreb", ch2_st, ch2_en, axes], ["ch2"]),
            helper.make_node("Reshape", ["ch2", flat_shape], ["mask_flat"]),
            helper.make_node(
                "Split",
                ["mask_flat"],
                [f"m{i}" for i in range(CORE * CORE)],
                axis=0,
                split=[1] * (CORE * CORE),
            ),
        ]
    )

    for i in range(CORE * CORE):
        nodes.extend(
            [
                helper.make_node("And", ["coreb", f"m{i}"], [f"active{i}"]),
                helper.make_node("Not", [f"m{i}"], [f"not{i}"]),
                helper.make_node("And", [bg_name, f"not{i}"], [f"bg{i}"]),
                helper.make_node("Or", [f"active{i}", f"bg{i}"], [f"b{i}"]),
            ]
        )

    for r in range(CORE):
        blocks = [f"b{r * CORE + c}" for c in range(CORE)]
        nodes.append(helper.make_node("Concat", blocks, [f"row{r}"], axis=3))

    nodes.extend(
        [
            helper.make_node("Concat", [f"row{r}" for r in range(CORE)], ["out9b"], axis=2),
            helper.make_node("Cast", ["out9b"], ["out9"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out9"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, C - ACTIVE_C, PAD, PAD],
            ),
        ]
    )

    graph = helper.make_graph(nodes, "task315_concat_blocks", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_foreground_blocks_model() -> onnx.ModelProto:
    """Fewest realized tensors: build only colors 1/2, then derive background."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, [0, 1, 2, 3], "axes")
    core_st = _i64(inits, [0, 1, 0, 0], "core_st")
    core_en = _i64(inits, [1, ACTIVE_C, CORE, CORE], "core_en")
    ch2_st = _i64(inits, [0, 1, 0, 0], "ch2_st")
    ch2_en = _i64(inits, [1, 2, CORE, CORE], "ch2_en")
    flat_shape = _i64(inits, [CORE * CORE], "flat_shape")
    zero = _f32(inits, [0.0], "zero")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, core_st, core_en, axes], ["core_fg"]),
            helper.make_node("Greater", ["core_fg", zero], ["core_fgb"]),
            helper.make_node("Slice", ["core_fgb", ch2_st, ch2_en, axes], ["ch2"]),
            helper.make_node("Reshape", ["ch2", flat_shape], ["mask_flat"]),
            helper.make_node(
                "Split",
                ["mask_flat"],
                [f"m{i}" for i in range(CORE * CORE)],
                axis=0,
                split=[1] * (CORE * CORE),
            ),
        ]
    )

    for i in range(CORE * CORE):
        nodes.append(helper.make_node("And", ["core_fgb", f"m{i}"], [f"b{i}"]))

    for r in range(CORE):
        blocks = [f"b{r * CORE + c}" for c in range(CORE)]
        nodes.append(helper.make_node("Concat", blocks, [f"row{r}"], axis=3))

    nodes.extend(
        [
            helper.make_node("Concat", [f"row{r}" for r in range(CORE)], ["fg9"], axis=2),
            helper.make_node("Split", ["fg9"], ["fg1", "fg2"], axis=1, split=[1, 1]),
            helper.make_node("Or", ["fg1", "fg2"], ["any_fg"]),
            helper.make_node("Not", ["any_fg"], ["bg9"]),
            helper.make_node("Concat", ["bg9", "fg9"], ["out9b"], axis=1),
            helper.make_node("Cast", ["out9b"], ["out9"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out9"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, C - ACTIVE_C, PAD, PAD],
            ),
        ]
    )

    graph = helper.make_graph(nodes, "task315_foreground_blocks", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_tiled_mask_model() -> onnx.ModelProto:
    """Smaller graph, slightly higher memory: tile core and color-2 mask."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, [0, 1, 2, 3], "axes")
    st = _i64(inits, [0, 0, 0, 0], "st")
    en = _i64(inits, [1, ACTIVE_C, CORE, CORE], "en")
    ch2_st = _i64(inits, [0, 2, 0, 0], "ch2_st")
    ch2_en = _i64(inits, [1, 3, CORE, CORE], "ch2_en")
    st1 = _i64(inits, [0, 1, 0, 0], "st1")
    en9 = _i64(inits, [1, ACTIVE_C, OUT, OUT], "en9")
    en0 = _i64(inits, [1, 1, OUT, OUT], "en0")
    zero = _f32(inits, [0.0], "zero")
    reps_body = _i64(inits, [1, 1, CORE, CORE], "reps_body")
    mask6_shape = _i64(inits, [1, 1, CORE, 1, CORE, 1], "mask6_shape")
    reps_mask = _i64(inits, [1, 1, 1, CORE, 1, CORE], "reps_mask")
    out9_shape = _i64(inits, [1, 1, OUT, OUT], "out9_shape")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st, en, axes], ["core"]),
            helper.make_node("Greater", ["core", zero], ["coreb"]),
            helper.make_node("Slice", ["coreb", ch2_st, ch2_en, axes], ["mask"]),
            helper.make_node("Tile", ["coreb", reps_body], ["body"]),
            helper.make_node("Reshape", ["mask", mask6_shape], ["mask6"]),
            helper.make_node("Tile", ["mask6", reps_mask], ["maskt"]),
            helper.make_node("Reshape", ["maskt", out9_shape], ["mask9"]),
            helper.make_node("Slice", ["body", st1, en9, axes], ["tail_body"]),
            helper.make_node("And", ["tail_body", "mask9"], ["tail"]),
            helper.make_node("Slice", ["body", st, en0, axes], ["body0"]),
            helper.make_node("Not", ["mask9"], ["not_mask"]),
            helper.make_node("Or", ["body0", "not_mask"], ["out0"]),
            helper.make_node("Concat", ["out0", "tail"], ["out9b"], axis=1),
            helper.make_node("Cast", ["out9b"], ["out9"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out9"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, C - ACTIVE_C, PAD, PAD],
            ),
        ]
    )

    graph = helper.make_graph(nodes, "task315_tiled_mask", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def validate_model(model: onnx.ModelProto) -> tuple[bool, str]:
    try:
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    except Exception as exc:
        return False, f"ORT load failed: {exc}"

    data = _load_data()
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            expected = _expected_onehot(ex["output"])
            pred = sess.run([OUT_NAME], {IN_NAME: _grid_to_onehot(ex["input"])})[0]
            if not _strict_onehot_matches(pred, expected):
                return False, f"{split}#{idx} strict one-hot mismatch"
    return True, "PASS"


def model_stats(model: onnx.ModelProto, path: Path) -> dict[str, int]:
    return {
        "bytes": path.stat().st_size,
        "nodes": len(model.graph.node),
        "inits": len(model.graph.initializer),
        "params": sum(int(np.prod(init.dims)) for init in model.graph.initializer),
    }


def variant_builders() -> dict[str, Callable[[], onnx.ModelProto]]:
    return {
        "foreground": build_foreground_blocks_model,
        "concat_blocks": build_concat_blocks_model,
        "tiled_mask": build_tiled_mask_model,
    }


def run_experiments() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for name, builder in variant_builders().items():
        path = OUT_DIR / f"{TASK_ID}_{name}.onnx"
        row: dict[str, Any] = {"name": name, "path": path}
        model = builder()
        onnx.save(model, str(path))
        row.update(model_stats(model, path))
        ok, message = validate_model(model)
        row["valid"] = ok
        row["validation"] = message
        scored = score_file(path)
        row["official_memory"] = scored.get("memory")
        row["official_params"] = scored.get("params")
        row["official_cost"] = scored.get("cost")
        row["official_score"] = scored.get("score")
        row["score_error"] = scored.get("error")
        results.append(row)

    valid = [
        row
        for row in results
        if row["valid"] and row["official_cost"] is not None and row["official_score"] is not None
    ]
    if valid:
        best = min(valid, key=lambda row: (int(row["official_cost"]), int(row["nodes"])))
        onnx.save(onnx.load(str(best["path"])), str(BEST_PATH))
    return results


def save_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    model = build_concat_blocks_model()
    onnx.save(model, str(path))
    return model


def main() -> None:
    print(f"colors: {colors_in_data()}")
    print(f"hypothesis mismatches: {verify_hypotheses()}")
    results = run_experiments()
    print(
        f"{'variant':<14} {'valid':<5} {'nodes':>5} {'params':>6} "
        f"{'memory':>8} {'cost':>6} {'score':>9}"
    )
    for row in results:
        score = row.get("official_score")
        score_text = f"{score:.6f}" if isinstance(score, float) else "INVALID"
        print(
            f"{row['name']:<14} {str(row['valid']):<5} {row['nodes']:>5} "
            f"{str(row.get('official_params')):>6} {str(row.get('official_memory')):>8} "
            f"{str(row.get('official_cost')):>6} {score_text:>9}"
        )
        if not row["valid"] or row.get("score_error"):
            print(f"  note: {row['validation']} {row.get('score_error') or ''}".rstrip())

    best_score = score_file(BEST_PATH)
    print(f"saved: {BEST_PATH}")
    print(
        f"best cost={best_score.get('cost')} memory={best_score.get('memory')} "
        f"params={best_score.get('params')} score={best_score.get('score'):.6f}"
    )


if __name__ == "__main__":
    main()
