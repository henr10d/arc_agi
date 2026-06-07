"""Minimal ONNX for ARC task060: fill row halves from edge endpoint colors.

Task rule: 5×11 grids have background 0. Rows with a nonzero color on column 0
fill columns 0–4 with that color; rows with a nonzero on column 10 fill columns
6–10 with that color. Column 5 is grey (color 5) when either endpoint is present;
rows without edge colors stay background-only.

ONNX: slice the two full one-hot edge columns; these are already the repeated
left/right column templates. The left background bit identifies inactive rows,
so the separator column is built by placing 1 - background into grey channel 5.
The three compact columns are concatenated across width, then padded.
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

TASK_ID = "task060"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task060.onnx"
DATA_PATH = ROOT / "data" / "task060.json"

C = 10
H = W = 30
OH = 5
OW = 11
HALF = 5
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


def _shared_constants(inits: List[onnx.TensorProto]) -> Dict[str, str]:
    return {
        "ax4": _i64(inits, [0, 1, 2, 3], "ax4"),
        "zero": _f32(inits, [0.0], "zero"),
        "crop_st": _i64(inits, [0, 0, 0, 0], "crop_st"),
        "crop_en": _i64(inits, [1, C, OH, OW], "crop_en"),
        "left_st": _i64(inits, [0, 0, 0, 0], "left_st"),
        "left_en": _i64(inits, [1, C, OH, 1], "left_en"),
        "right_st": _i64(inits, [0, 0, 0, 10], "right_st"),
        "right_en": _i64(inits, [1, C, OH, OW], "right_en"),
        "ch0_st": _i64(inits, [0, 0, 0, 0], "ch0_st"),
        "ch0_en": _i64(inits, [1, 1, OH, 1], "ch0_en"),
        "fg_st": _i64(inits, [0, 1, 0, 0], "fg_st"),
        "fg_en_col": _i64(inits, [1, C, OH, 1], "fg_en_col"),
        "fg_en_full": _i64(inits, [1, C, OH, OW], "fg_en_full"),
    }


def _strip_background(
    nodes: List[onnx.NodeProto],
    const: Dict[str, str],
    col: str,
    prefix: str,
) -> str:
    out = f"{prefix}_no_bg"
    nodes.extend(
        [
            helper.make_node("Slice", [col, const["ch0_st"], const["ch0_en"], const["ax4"]], [f"{prefix}_ch0"]),
            helper.make_node("Sub", [col, f"{prefix}_ch0"], [out]),
        ]
    )
    return out


def _repeat_width_concat(
    nodes: List[onnx.NodeProto],
    const: Dict[str, str],
    col: str,
    prefix: str,
) -> str:
    no_bg = _strip_background(nodes, const, col, prefix)
    out = f"{prefix}_half"
    nodes.append(helper.make_node("Concat", [no_bg] * HALF, [out], axis=3))
    return out


def _grey_separator(
    nodes: List[onnx.NodeProto],
    const: Dict[str, str],
    left_col: str,
    right_col: str,
) -> str:
    nodes.extend(
        [
            helper.make_node("Slice", [left_col, const["fg_st"], const["fg_en_col"], const["ax4"]], ["sep_left_fg"]),
            helper.make_node("Slice", [right_col, const["fg_st"], const["fg_en_col"], const["ax4"]], ["sep_right_fg"]),
            helper.make_node("ReduceSum", ["sep_left_fg"], ["left_sum"], axes=[1], keepdims=1),
            helper.make_node("ReduceSum", ["sep_right_fg"], ["right_sum"], axes=[1], keepdims=1),
            helper.make_node("Add", ["left_sum", "right_sum"], ["edge_sum"]),
            helper.make_node("Greater", ["edge_sum", const["zero"]], ["row_active"]),
            helper.make_node("Cast", ["row_active"], ["row_active_f"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["row_active_f"],
                ["sep_col"],
                mode="constant",
                pads=[0, 5, 0, 0, 0, 4, 0, 0],
            ),
        ]
    )
    return "sep_col"


def _add_background_channel_nodes(const: Dict[str, str]) -> List[onnx.NodeProto]:
    return [
        helper.make_node("Slice", ["core_out", const["fg_st"], const["fg_en_full"], const["ax4"]], ["core_fg"]),
        helper.make_node("ReduceSum", ["core_fg"], ["fg_sum"], axes=[1], keepdims=1),
        helper.make_node("Greater", ["fg_sum", const["zero"]], ["has_fg"]),
        helper.make_node("Not", ["has_fg"], ["is_bg"]),
        helper.make_node("Cast", ["is_bg"], ["bg_ch"], to=TensorProto.FLOAT),
        helper.make_node("Concat", ["bg_ch", "core_fg"], ["core_onehot"], axis=1),
    ]


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


def build_onnx_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    ax4 = _i64(inits, [0, 1, 2, 3], "ax4")
    one = _f32(inits, [1.0], "one")
    left_st = _i64(inits, [0, 0, 0, 0], "left_st")
    left_en = _i64(inits, [1, C, OH, 1], "left_en")
    right_st = _i64(inits, [0, 0, 0, 10], "right_st")
    right_en = _i64(inits, [1, C, OH, OW], "right_en")
    bg_en = _i64(inits, [1, 1, OH, 1], "bg_en")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, left_st, left_en, ax4], ["left_col_out"]),
            helper.make_node("Slice", [IN_NAME, right_st, right_en, ax4], ["right_col_out"]),
            helper.make_node("Slice", ["left_col_out", left_st, bg_en, ax4], ["row_inactive_f"]),
            helper.make_node("Sub", [one, "row_inactive_f"], ["row_active_f"]),
            helper.make_node(
                "Pad",
                ["row_active_f"],
                ["sep_fg"],
                mode="constant",
                pads=[0, 4, 0, 0, 0, 4, 0, 0],
            ),
            helper.make_node("Concat", ["row_inactive_f", "sep_fg"], ["sep_col_out"], axis=1),
            helper.make_node(
                "Concat",
                ["left_col_out"] * HALF + ["sep_col_out"] + ["right_col_out"] * HALF,
                ["core_onehot"],
                axis=3,
            ),
        ]
    )
    nodes.append(
        helper.make_node(
            "Pad",
            ["core_onehot"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - OH, W - OW],
        )
    )
    return _model(nodes, inits, "task060")


def build_float_foreground_variant() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    ax4 = _i64(inits, [0, 1, 2, 3], "ax4")
    zero = _f32(inits, [0.0], "zero")
    left_st = _i64(inits, [0, 1, 0, 0], "left_st")
    left_en = _i64(inits, [1, C, OH, 1], "left_en")
    right_st = _i64(inits, [0, 1, 0, 10], "right_st")
    right_en = _i64(inits, [1, C, OH, OW], "right_en")
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, left_st, left_en, ax4], ["left_fg"]),
            helper.make_node("Slice", [IN_NAME, right_st, right_en, ax4], ["right_fg"]),
            helper.make_node("ReduceSum", ["left_fg"], ["left_row_sum"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["left_row_sum", zero], ["row_active"]),
            helper.make_node("Cast", ["row_active"], ["row_active_f"], to=TensorProto.FLOAT),
            helper.make_node("Not", ["row_active"], ["row_inactive"]),
            helper.make_node("Cast", ["row_inactive"], ["row_inactive_f"], to=TensorProto.FLOAT),
            helper.make_node("Concat", ["left_fg"] * HALF, ["left_half"], axis=3),
            helper.make_node("Concat", ["right_fg"] * HALF, ["right_half"], axis=3),
            helper.make_node("Concat", ["row_inactive_f"] * OW, ["bg_ch"], axis=3),
            helper.make_node(
                "Pad",
                ["row_active_f"],
                ["sep_col"],
                mode="constant",
                pads=[0, 4, 0, 0, 0, 4, 0, 0],
            ),
            helper.make_node("Concat", ["left_half", "sep_col", "right_half"], ["core_fg"], axis=3),
            helper.make_node("Concat", ["bg_ch", "core_fg"], ["core_onehot"], axis=1),
        ]
    )
    nodes.append(
        helper.make_node(
            "Pad",
            ["core_onehot"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - OH, W - OW],
        )
    )
    return _model(nodes, inits, "task060_float_fg")


def build_float_column_variant() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    ax4 = _i64(inits, [0, 1, 2, 3], "ax4")
    zero = _f32(inits, [0.0], "zero")
    left_st = _i64(inits, [0, 1, 0, 0], "left_st")
    left_en = _i64(inits, [1, C, OH, 1], "left_en")
    right_st = _i64(inits, [0, 1, 0, 10], "right_st")
    right_en = _i64(inits, [1, C, OH, OW], "right_en")
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, left_st, left_en, ax4], ["left_fg"]),
            helper.make_node("Slice", [IN_NAME, right_st, right_en, ax4], ["right_fg"]),
            helper.make_node("ReduceSum", ["left_fg"], ["left_row_sum"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["left_row_sum", zero], ["row_active"]),
            helper.make_node("Not", ["row_active"], ["row_inactive"]),
            helper.make_node("Cast", ["row_active"], ["row_active_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["row_inactive"], ["row_inactive_f"], to=TensorProto.FLOAT),
            helper.make_node("Concat", ["row_inactive_f", "left_fg"], ["left_col_out"], axis=1),
            helper.make_node("Concat", ["row_inactive_f", "right_fg"], ["right_col_out"], axis=1),
            helper.make_node(
                "Pad",
                ["row_active_f"],
                ["sep_fg"],
                mode="constant",
                pads=[0, 4, 0, 0, 0, 4, 0, 0],
            ),
            helper.make_node("Concat", ["row_inactive_f", "sep_fg"], ["sep_col_out"], axis=1),
            helper.make_node(
                "Concat",
                ["left_col_out"] * HALF + ["sep_col_out"] + ["right_col_out"] * HALF,
                ["core_onehot"],
                axis=3,
            ),
        ]
    )
    nodes.append(
        helper.make_node(
            "Pad",
            ["core_onehot"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - OH, W - OW],
        )
    )
    return _model(nodes, inits, "task060_float_columns")


def build_tile_variant() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    const = _shared_constants(inits)
    rep = _i64(inits, [1, 1, 1, HALF], "tile_rep")
    nodes.append(helper.make_node("Slice", [IN_NAME, const["crop_st"], const["crop_en"], const["ax4"]], ["core"]))
    nodes.append(helper.make_node("Slice", ["core", const["left_st"], const["left_en"], const["ax4"]], ["left_col"]))
    nodes.append(helper.make_node("Slice", ["core", const["right_st"], const["right_en"], const["ax4"]], ["right_col"]))
    left_no_bg = _strip_background(nodes, const, "left_col", "left")
    right_no_bg = _strip_background(nodes, const, "right_col", "right")
    nodes.append(helper.make_node("Tile", [left_no_bg, rep], ["left_half"]))
    nodes.append(helper.make_node("Tile", [right_no_bg, rep], ["right_half"]))
    sep_col = _grey_separator(nodes, const, "left_col", "right_col")
    nodes.append(helper.make_node("Concat", ["left_half", sep_col, "right_half"], ["core_out"], axis=3))
    nodes.extend(_add_background_channel_nodes(const))
    nodes.append(
        helper.make_node(
            "Pad",
            ["core_onehot"],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - OH, W - OW],
        )
    )
    return _model(nodes, inits, "task060_tile")


def _examples() -> list[dict[str, list[list[int]]]]:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    out: list[dict[str, list[list[int]]]] = []
    for split in ("train", "test", "arc-gen"):
        out.extend(data.get(split, []))
    return out


def verify_correct(model: onnx.ModelProto) -> bool:
    sanitized = sanitize_model(model)
    if sanitized is None:
        return False
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        sanitized.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    for example in _examples():
        inp = convert_to_numpy(example, "input")
        expected = convert_to_numpy(example, "output")
        if inp is None or expected is None:
            continue
        got = session.run([OUT_NAME], {IN_NAME: inp})[0]
        if not np.array_equal(got > 0.0, expected > 0.0):
            return False
    return True


def largest_internal_tensor(model: onnx.ModelProto) -> int | None:
    sanitized = sanitize_model(model)
    if sanitized is None:
        return None
    try:
        graph = onnx.shape_inference.infer_shapes(sanitized, strict_mode=True).graph
    except Exception:
        return None
    value_map = {item.name: item for item in list(graph.value_info) + list(graph.output) + list(graph.input)}
    best = 0
    for node in graph.node:
        for name in node.output:
            if name in {IN_NAME, OUT_NAME}:
                continue
            item = value_map.get(name)
            if item is None or not item.type.HasField("tensor_type"):
                continue
            tensor_type = item.type.tensor_type
            elems = 1
            for dim in tensor_type.shape.dim:
                if not dim.HasField("dim_value") or dim.dim_value <= 0:
                    return None
                elems *= dim.dim_value
            dtype = helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
            best = max(best, int(elems * np.dtype(dtype).itemsize))
    return best


def score_variant(name: str, model: onnx.ModelProto, tmp_root: Path) -> Dict[str, object]:
    variant_dir = tmp_root / name
    variant_dir.mkdir()
    path = variant_dir / f"{TASK_ID}.onnx"
    onnx.save(model, path)
    result = score_file(path)
    return {
        "variant": name,
        "model": model,
        "valid": bool(result["valid"]) and verify_correct(model),
        "memory": result["memory"],
        "params": result["params"],
        "cost": result["cost"],
        "score": result["score"],
        "largest": largest_internal_tensor(model),
        "error": result["error"],
    }


def main() -> None:
    builders: list[tuple[str, Callable[[], onnx.ModelProto]]] = [
        ("edge_cols", build_onnx_model),
    ]

    with tempfile.TemporaryDirectory(prefix="task060_variants_") as tmp:
        rows = [score_variant(name, build(), Path(tmp)) for name, build in builders]

    print("variant       valid  memory  params  cost   score      largest_internal_tensor")
    print("------------  -----  ------  ------  -----  ---------  -----------------------")
    for row in rows:
        score_val = row["score"]
        score_txt = "     None" if score_val is None else f"{score_val:.6f}"
        print(
            f"{row['variant']:<12}  {str(row['valid']):<5}  "
            f"{str(row['memory']):>6}  {str(row['params']):>6}  {str(row['cost']):>5}  "
            f"{score_txt:>9}  {str(row['largest']):>23}"
        )
        if not row["valid"] and row["error"]:
            print(f"  error: {str(row['error']).strip().splitlines()[-1]}")

    valid_rows = [row for row in rows if row["valid"] and row["cost"] is not None]
    if not valid_rows:
        raise SystemExit("no valid task060 model variant")
    best = min(valid_rows, key=lambda row: int(row["cost"]))
    onnx.save(best["model"], BEST_PATH)
    print()
    print(f"best: {best['variant']} -> {BEST_PATH}  cost={best['cost']} score={best['score']:.6f}")


if __name__ == "__main__":
    main()
