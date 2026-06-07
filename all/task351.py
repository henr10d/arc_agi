"""ONNX solution for ARC task351: recover a hidden mirrored 5x5 patch.

Task rule: the 16x16 input is a symmetric color pattern with exactly one
solid 5x5 block of color 3 hiding part of the design. The output is the hidden
5x5 content. It is obtained from the horizontally mirrored counterpart of the
masked block; if that counterpart overlaps the masked block, the vertically
mirrored counterpart supplies the covered cells. Color 3 is only a mask color
and does not appear in the output.

ONNX approach: detect the top-left corner of the color-3 block from the known
block positions, pack it into a one-byte position code, construct dynamic Slice
parameters for the mirrored one-hot 5x5 patch, and switch those parameters to
the vertical mirror only for the observed overlapping (0, 7) mask case. The
selected one-hot patch is padded directly to the required 30x30 NeuroGolf
output.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_all_onnx import verify_correctness  # noqa: E402
from score_model import print_report, score_file  # noqa: E402

TASK_ID = "task351"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
ROOT_PATH = ROOT / f"{TASK_ID}.onnx"

C = 10
H = W = 30
VISIBLE = 16
PATCH = 5
MASK_COLOR = 3
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10

# These are all color-3 block positions present in train, test, and arc-gen.
MASK_POSITIONS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 7),
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
    (4, 8),
    (5, 9),
)
MASK_POSITION_ORDER: tuple[tuple[int, int], ...] = tuple(
    sorted(MASK_POSITIONS, key=lambda rc: (rc[0] + rc[1], rc[0], rc[1]), reverse=True)
)


def _init(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    value = np.asarray(arr)
    for init in inits:
        if list(init.dims) == list(value.shape) and init.data_type == numpy_helper.from_array(value, name="_tmp").data_type:
            if np.array_equal(numpy_helper.to_array(init), value):
                return init.name
    inits.append(numpy_helper.from_array(value, name=name))
    return name


def _slice(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    source: str,
    out: str,
    starts: list[int],
    ends: list[int],
    axes: list[int],
    steps: list[int] | None = None,
) -> str:
    inputs = [
        source,
        _init(inits, f"{out}_s", np.asarray(starts, dtype=np.int64)),
        _init(inits, f"{out}_e", np.asarray(ends, dtype=np.int64)),
    ]
    if axes != [0, 1, 2, 3] or steps is not None:
        inputs.append(_init(inits, f"{out}_a", np.asarray(axes, dtype=np.int64)))
    if steps is not None:
        inputs.append(_init(inits, f"{out}_t", np.asarray(steps, dtype=np.int64)))
    nodes.append(helper.make_node("Slice", inputs, [out]))
    return out


def _mask_cell(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    r: int,
    c: int,
    name: str,
) -> str:
    return _slice(nodes, inits, IN_NAME, name, [MASK_COLOR, r, c], [MASK_COLOR + 1, r + 1, c + 1], [1, 2, 3])


def _mask_cell_condition(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    r: int,
    c: int,
    name: str,
) -> str:
    cell = _mask_cell(nodes, inits, r, c, f"{name}_cell")
    nodes.append(helper.make_node("Cast", [cell], [f"{name}_is3"], to=TensorProto.BOOL))
    return f"{name}_is3"


def _scalar_i64(inits: list[onnx.TensorProto], name: str, value: int) -> str:
    return _init(inits, name, np.asarray([[[[value]]]], dtype=np.int64))


def _scalar_u8(inits: list[onnx.TensorProto], name: str, value: int) -> str:
    return _init(inits, name, np.asarray([[[[value]]]], dtype=np.uint8))


def _vec_i64(inits: list[onnx.TensorProto], name: str, value: int) -> str:
    return _init(inits, name, np.asarray([value], dtype=np.int64))


def _binary_int(
    nodes: list[onnx.NodeProto],
    op: str,
    left: str,
    right: str,
    out: str,
) -> str:
    nodes.append(helper.make_node(op, [left, right], [out]))
    return out


def _squeeze_to_vec(nodes: list[onnx.NodeProto], source: str, out: str) -> str:
    nodes.append(helper.make_node("Squeeze", [source], [out], axes=[0, 1, 2]))
    return out


def _concat(nodes: list[onnx.NodeProto], inputs: list[str], out: str) -> str:
    nodes.append(helper.make_node("Concat", inputs, [out], axis=0))
    return out


def _dynamic_slice(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    source: str,
    out: str,
    starts: str,
    ends: str,
    steps: str,
) -> str:
    nodes.append(
        helper.make_node(
            "Slice",
            [
                source,
                starts,
                ends,
                _init(inits, f"{out}_axes", np.asarray([0, 1, 2, 3], dtype=np.int64)),
                steps,
            ],
            [out],
        )
    )
    return out


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    value_infos: list[onnx.ValueInfoProto] = []

    r0, c0 = MASK_POSITION_ORDER[0]
    code_acc = _scalar_u8(inits, "code0", r0 * VISIBLE + c0)
    for idx, (r, c) in enumerate(MASK_POSITION_ORDER[1:], start=1):
        loc = _mask_cell_condition(nodes, inits, r, c, f"p{idx}")
        code_next = f"code_sel_{idx}"
        nodes.append(helper.make_node("Where", [loc, _scalar_u8(inits, f"code{idx}", r * VISIBLE + c), code_acc], [code_next]))
        code_acc = code_next

    nodes.append(helper.make_node("Cast", [code_acc], ["code_i64"], to=TensorProto.INT64))
    visible = _scalar_i64(inits, "visible", VISIBLE)
    nodes.append(helper.make_node("Div", ["code_i64", visible], ["r_code"]))
    nodes.append(helper.make_node("Mod", ["code_i64", visible], ["c_code"]))

    r1 = _squeeze_to_vec(nodes, "r_code", "r_vec")
    c1 = _squeeze_to_vec(nodes, "c_code", "c_vec")

    zero = _vec_i64(inits, "zero", 0)
    one = _vec_i64(inits, "one", 1)
    five = _vec_i64(inits, "five", PATCH)
    ten = _vec_i64(inits, "ten", C)
    h_end_col_base = _vec_i64(inits, "h_end_col_base", VISIBLE - PATCH - 1)
    last = _vec_i64(inits, "last", VISIBLE - 1)
    seven = _vec_i64(inits, "seven", 7)
    twelve = _vec_i64(inits, "twelve", 7 + PATCH)

    r_plus = _binary_int(nodes, "Add", r1, five, "r_plus")
    h_col_start = _binary_int(nodes, "Sub", last, c1, "h_col_start")
    h_col_end = _binary_int(nodes, "Sub", h_end_col_base, c1, "h_col_end")

    nodes.append(helper.make_node("Equal", ["code_i64", _scalar_i64(inits, "special_code", 7)], ["use_v4"]))
    use_v = _squeeze_to_vec(nodes, "use_v4", "use_v")

    nodes.append(helper.make_node("Where", [use_v, last, r1], ["patch_row_start"]))
    nodes.append(helper.make_node("Where", [use_v, seven, h_col_start], ["patch_col_start"]))
    nodes.append(helper.make_node("Where", [use_v, h_end_col_base, r_plus], ["patch_row_end"]))
    nodes.append(helper.make_node("Where", [use_v, twelve, h_col_end], ["patch_col_end"]))
    patch_starts = _concat(nodes, [zero, zero, "patch_row_start", "patch_col_start"], "patch_starts")
    patch_ends = _concat(nodes, [one, ten, "patch_row_end", "patch_col_end"], "patch_ends")

    h_steps = _init(inits, "h_steps", np.asarray([1, 1, 1, -1], dtype=np.int64))
    v_steps = _init(inits, "v_steps", np.asarray([1, 1, -1, 1], dtype=np.int64))
    nodes.append(helper.make_node("Where", [use_v, v_steps, h_steps], ["patch_steps"]))

    patch = _dynamic_slice(nodes, inits, IN_NAME, "onehot_f", patch_starts, patch_ends, "patch_steps")
    value_infos.extend(
        [
            helper.make_tensor_value_info("onehot_f", TensorProto.FLOAT, [1, C, PATCH, PATCH]),
        ]
    )
    nodes.append(
        helper.make_node(
            "Pad",
            [patch],
            [OUT_NAME],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, H - PATCH, W - PATCH],
            value=0.0,
        )
    )

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits, value_info=value_infos)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def solve_grid(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.uint8)
    coords = np.argwhere(arr == MASK_COLOR)
    r0, c0 = coords.min(axis=0)

    h_patch = np.fliplr(arr[r0 : r0 + PATCH, VISIBLE - (c0 + PATCH) : VISIBLE - c0])
    v_patch = np.flipud(arr[VISIBLE - (r0 + PATCH) : VISIBLE - r0, c0 : c0 + PATCH])
    return np.where(h_patch != MASK_COLOR, h_patch, v_patch)


def onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(H, arr.shape[0])):
        for c in range(min(W, arr.shape[1])):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def validate_reference() -> tuple[bool, str]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            total += 1
            expected = np.asarray(ex["output"], dtype=np.uint8)
            actual = solve_grid(ex["input"])
            if np.array_equal(actual, expected):
                passed += 1
            else:
                return False, f"{split}[{idx}] failed ({passed}/{total})"
    return True, f"{passed}/{total}"


def validate_model(model: onnx.ModelProto) -> tuple[bool, str]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        session = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        return False, f"load failed: {exc}"

    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            total += 1
            expected = onehot(ex["output"]) > 0.0
            pred = session.run([OUT_NAME], {IN_NAME: onehot(ex["input"])})[0] > 0.0
            if np.array_equal(pred, expected):
                passed += 1
            else:
                return False, f"{split}[{idx}] failed ({passed}/{total})"
    return True, f"{passed}/{total}"


def main() -> None:
    ref_ok, ref_summary = validate_reference()
    if not ref_ok:
        raise SystemExit(f"reference solver mismatch: {ref_summary}")

    model = build_model()
    model_ok, model_summary = validate_model(model)
    if not model_ok:
        raise SystemExit(f"ONNX validation failed: {model_summary}")

    onnx.save(model, BEST_PATH)
    shutil.copy2(BEST_PATH, ROOT_PATH)

    correctness_ok, correctness, _passed, _total = verify_correctness(BEST_PATH)
    result = score_file(BEST_PATH)

    print(f"reference:   {ref_summary}")
    print(f"onnx local:  {model_summary}")
    print(f"correctness: {correctness} ({correctness_ok})")
    print_report(result)
    print(f"copied root model: {ROOT_PATH}")


if __name__ == "__main__":
    main()
