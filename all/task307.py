"""ONNX generator for ARC task307 using Kaggle one-hot I/O.

Task rule: expand every input cell into a 2 by 2 block of the same color.
For an H by W grid, the expected output is
np.repeat(np.repeat(input_grid, 2, axis=0), 2, axis=1). The examples use
2x2 through 5x5 input grids, so the visible output is at most 10x10 and the
remaining competition tensor stays zero-padded.

The selected ONNX model implements the nearest-neighbor upscale directly over
the fixed 30 by 30 competition tensor with one grouped ConvTranspose node. It
writes that node straight to the graph output, so the only produced activation
is excluded from official memory. Lower-parameter formulations need cropped or
generated intermediate tensors, which score worse than this 0-memory/40-param
graph.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, sanitize_model, score_file  # noqa: E402

TASK_ID = "task307"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
MAX_ACTIVE = 5
UPSCALED = MAX_ACTIVE * 2
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10
IN_NAME = "input"
OUT_NAME = "output"


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _model(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], name: str) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
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


def build_direct_convtranspose() -> onnx.ModelProto:
    """Selected model: one node, no scored internal activation tensors."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    weight = np.ones((C, 1, 2, 2), dtype=np.float32)
    w_name = _f32(inits, weight, "w")
    nodes.append(
        helper.make_node(
            "ConvTranspose",
            [IN_NAME, w_name],
            [OUT_NAME],
            group=C,
            pads=[0, 0, H, W],
            strides=[2, 2],
        )
    )
    return _model(nodes, inits, "task307_direct_convtranspose")


def build_crop_convtranspose() -> onnx.ModelProto:
    nodes, inits = _cropped_input()
    weight = np.ones((C, 1, 2, 2), dtype=np.float32)
    w_name = _f32(inits, weight, "w")
    nodes.append(helper.make_node("ConvTranspose", ["x5", w_name], ["up10"], group=C, strides=[2, 2]))
    _pad_to_output(nodes, "up10")
    return _model(nodes, inits, "task307_crop_convtranspose")


def build_gather_duplication() -> onnx.ModelProto:
    nodes, inits = _cropped_input()
    idx = _i64(inits, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4], "idx")
    nodes.append(helper.make_node("Gather", ["x5", idx], ["rows"], axis=2))
    nodes.append(helper.make_node("Gather", ["rows", idx], ["up10"], axis=3))
    _pad_to_output(nodes, "up10")
    return _model(nodes, inits, "task307_gather_duplication")


def build_tile_reshape() -> onnx.ModelProto:
    nodes, inits = _cropped_input()
    rep_cols = _i64(inits, [1, 1, 1, 1, 2], "rep_cols")
    rep_rows = _i64(inits, [1, 1, 1, 2, 1], "rep_rows")
    shape_cols = _i64(inits, [1, C, MAX_ACTIVE, UPSCALED], "shape_cols")
    shape_up = _i64(inits, [1, C, UPSCALED, UPSCALED], "shape_up")
    nodes.append(helper.make_node("Unsqueeze", ["x5"], ["x5u"], axes=[4]))
    nodes.append(helper.make_node("Tile", ["x5u", rep_cols], ["cols_tiled"]))
    nodes.append(helper.make_node("Reshape", ["cols_tiled", shape_cols], ["cols10"]))
    nodes.append(helper.make_node("Unsqueeze", ["cols10"], ["cols10u"], axes=[3]))
    nodes.append(helper.make_node("Tile", ["cols10u", rep_rows], ["rows_tiled"]))
    nodes.append(helper.make_node("Reshape", ["rows_tiled", shape_up], ["up10"]))
    _pad_to_output(nodes, "up10")
    return _model(nodes, inits, "task307_tile_reshape")


def build_expand_reshape() -> onnx.ModelProto:
    nodes, inits = _cropped_input()
    expand_cols = _i64(inits, [1, C, MAX_ACTIVE, MAX_ACTIVE, 2], "expand_cols")
    expand_rows = _i64(inits, [1, C, MAX_ACTIVE, 2, UPSCALED], "expand_rows")
    shape_cols = _i64(inits, [1, C, MAX_ACTIVE, UPSCALED], "shape_cols")
    shape_up = _i64(inits, [1, C, UPSCALED, UPSCALED], "shape_up")
    nodes.append(helper.make_node("Unsqueeze", ["x5"], ["x5u"], axes=[4]))
    nodes.append(helper.make_node("Expand", ["x5u", expand_cols], ["cols_expanded"]))
    nodes.append(helper.make_node("Reshape", ["cols_expanded", shape_cols], ["cols10"]))
    nodes.append(helper.make_node("Unsqueeze", ["cols10"], ["cols10u"], axes=[3]))
    nodes.append(helper.make_node("Expand", ["cols10u", expand_rows], ["rows_expanded"]))
    nodes.append(helper.make_node("Reshape", ["rows_expanded", shape_up], ["up10"]))
    _pad_to_output(nodes, "up10")
    return _model(nodes, inits, "task307_expand_reshape")


def build_concat_repeat() -> onnx.ModelProto:
    nodes, inits = _cropped_input()
    axis3 = _i64(inits, [3], "axis3")
    axis2 = _i64(inits, [2], "axis2")

    col_parts: List[str] = []
    for col in range(MAX_ACTIVE):
        st = _i64(inits, [col], f"c{col}_st")
        en = _i64(inits, [col + 1], f"c{col}_en")
        out = f"col{col}"
        nodes.append(helper.make_node("Slice", ["x5", st, en, axis3], [out]))
        col_parts.extend([out, out])
    nodes.append(helper.make_node("Concat", col_parts, ["wide5"], axis=3))

    row_parts: List[str] = []
    for row in range(MAX_ACTIVE):
        st = _i64(inits, [row], f"r{row}_st")
        en = _i64(inits, [row + 1], f"r{row}_en")
        out = f"row{row}"
        nodes.append(helper.make_node("Slice", ["wide5", st, en, axis2], [out]))
        row_parts.extend([out, out])
    nodes.append(helper.make_node("Concat", row_parts, ["up10"], axis=2))
    _pad_to_output(nodes, "up10")
    return _model(nodes, inits, "task307_concat_repeat")


def _cropped_input() -> tuple[List[onnx.NodeProto], List[onnx.TensorProto]]:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    starts = _i64(inits, [0, 0, 0, 0], "crop_starts")
    ends = _i64(inits, [1, C, MAX_ACTIVE, MAX_ACTIVE], "crop_ends")
    axes = _i64(inits, [0, 1, 2, 3], "crop_axes")
    nodes.append(helper.make_node("Slice", [IN_NAME, starts, ends, axes], ["x5"]))
    return nodes, inits


def _pad_to_output(nodes: List[onnx.NodeProto], data: str) -> None:
    nodes.append(
        helper.make_node(
            "Pad",
            [data],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - UPSCALED, W - UPSCALED],
        )
    )


def _examples() -> list[dict[str, list[list[int]]]]:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    examples: list[dict[str, list[list[int]]]] = []
    for split in ("train", "test", "arc-gen"):
        examples.extend(data.get(split, []))
    return examples


def verify_repeat_rule() -> bool:
    for example in _examples():
        inp = np.asarray(example["input"], dtype=np.int64)
        expected = np.asarray(example["output"], dtype=np.int64)
        got = np.repeat(np.repeat(inp, 2, axis=0), 2, axis=1)
        if not np.array_equal(got, expected):
            return False
    return True


def verify_correct(model: onnx.ModelProto) -> bool:
    sanitized = sanitize_model(model)
    if sanitized is None:
        return False
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(sanitized.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    for example in _examples():
        inp = convert_to_numpy(example, "input")
        expected = convert_to_numpy(example, "output")
        if inp is None or expected is None:
            continue
        got = session.run([OUT_NAME], {IN_NAME: inp})[0]
        if not np.array_equal(got > 0.0, expected > 0.0):
            return False
    return True


def internal_tensor_stats(model: onnx.ModelProto) -> tuple[int, int] | None:
    sanitized = sanitize_model(model)
    if sanitized is None:
        return None
    try:
        graph = onnx.shape_inference.infer_shapes(sanitized, strict_mode=True).graph
    except Exception:
        return None

    value_map = {item.name: item for item in list(graph.value_info) + list(graph.output) + list(graph.input)}
    count = 0
    largest = 0
    for node in graph.node:
        for name in node.output:
            if name in {IN_NAME, OUT_NAME}:
                continue
            item = value_map.get(name)
            if item is None or not item.type.HasField("tensor_type"):
                return None
            tensor_type = item.type.tensor_type
            elems = 1
            for dim in tensor_type.shape.dim:
                if not dim.HasField("dim_value") or dim.dim_value <= 0:
                    return None
                elems *= dim.dim_value
            dtype = helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
            count += 1
            largest = max(largest, int(elems * np.dtype(dtype).itemsize))
    return count, largest


def score_variant(name: str, model: onnx.ModelProto, tmp_root: Path) -> Dict[str, object]:
    variant_dir = tmp_root / name
    variant_dir.mkdir()
    path = variant_dir / f"{TASK_ID}.onnx"
    onnx.save(model, path)
    result = score_file(path)
    stats = internal_tensor_stats(model)
    count, largest = stats if stats is not None else (None, None)
    return {
        "variant": name,
        "model": model,
        "valid": bool(result["valid"]) and verify_correct(model),
        "memory": result["memory"],
        "params": result["params"],
        "cost": result["cost"],
        "score": result["score"],
        "internal_count": count,
        "largest": largest,
        "error": result["error"],
    }


def main() -> None:
    if not verify_repeat_rule():
        raise SystemExit("task307 examples do not match the 2x repeat rule")

    builders: list[tuple[str, Callable[[], onnx.ModelProto]]] = [
        ("direct_convtranspose", build_direct_convtranspose),
        ("crop_convtranspose", build_crop_convtranspose),
        ("gather_duplication", build_gather_duplication),
        ("tile_reshape", build_tile_reshape),
        ("expand_reshape", build_expand_reshape),
        ("concat_repeat", build_concat_repeat),
    ]

    with tempfile.TemporaryDirectory(prefix="task307_variants_") as tmp:
        rows = [score_variant(name, build(), Path(tmp)) for name, build in builders]

    print("variant              valid  memory  params  cost   score      tensors  largest")
    print("-------------------  -----  ------  ------  -----  ---------  -------  -------")
    for row in rows:
        score = row["score"]
        print(
            f"{row['variant']:<19}  {str(row['valid']):<5}  "
            f"{str(row['memory']):>6}  {str(row['params']):>6}  {str(row['cost']):>5}  "
            f"{score if score is None else f'{score:.6f}':>9}  "
            f"{str(row['internal_count']):>7}  {str(row['largest']):>7}"
        )
        if not row["valid"] and row["error"]:
            print(f"  error: {str(row['error']).strip().splitlines()[-1]}")

    valid_rows = [row for row in rows if row["valid"] and row["cost"] is not None]
    if not valid_rows:
        raise SystemExit("no valid task307 model variant")
    best = min(valid_rows, key=lambda row: (int(row["cost"]), int(row["internal_count"] or 0)))
    onnx.save(best["model"], BEST_PATH)
    print()
    print(f"best: {best['variant']} -> {BEST_PATH}")


if __name__ == "__main__":
    main()
