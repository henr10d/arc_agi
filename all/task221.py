"""Minimal ONNX for ARC task221: counted row-major copies of a 3x3 motif.

Task rule: the input is a 3x3 grid containing black plus one foreground color.
Let n be the number of non-black cells. The output square has side
3 * (9 - n), and contains n copies of the full 3x3 input pattern placed in
row-major order over that block grid. All remaining cells inside the output
square are black, and cells outside the square are left as zero padding for the
NeuroGolf 30x30 one-hot I/O contract.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task221"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task221.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
CORE = 3
MAX_K = 7
MAX_SIDE = CORE * MAX_K
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10
COUNTS = (2, 3, 4, 5, 6)


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver matching the local task221 JSON examples."""
    g = np.asarray(grid, dtype=np.int64)[:CORE, :CORE]
    count = int((g != 0).sum())
    k = 9 - count
    out = np.zeros((CORE * k, CORE * k), dtype=np.int64)
    for tile in range(count):
        tr, tc = divmod(tile, k)
        out[CORE * tr : CORE * (tr + 1), CORE * tc : CORE * (tc + 1)] = g
    return out


def _copy_mask(count: int) -> np.ndarray:
    """[1,1,21,21] mask for cells belonging to copied 3x3 blocks."""
    k = 9 - count
    mask = np.zeros((1, 1, MAX_SIDE, MAX_SIDE), dtype=np.float32)
    for tile in range(count):
        tr, tc = divmod(tile, k)
        mask[:, :, CORE * tr : CORE * (tr + 1), CORE * tc : CORE * (tc + 1)] = 1.0
    return mask


def _active_mask(count: int) -> np.ndarray:
    """[1,1,21,21] mask for the exact output square."""
    side = CORE * (9 - count)
    mask = np.zeros((1, 1, MAX_SIDE, MAX_SIDE), dtype=np.float32)
    mask[:, :, :side, :side] = 1.0
    return mask


def _block_mask(count: int) -> np.ndarray:
    """[1,1,7,7] mask for copied 3x3 block positions."""
    k = 9 - count
    mask = np.zeros((1, 1, MAX_K, MAX_K), dtype=np.float32)
    for tile in range(count):
        tr, tc = divmod(tile, k)
        mask[:, :, tr, tc] = 1.0
    return mask


def _make_model(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto], name: str) -> onnx.ModelProto:
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


def _selected_mask(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    cnt: str,
    *,
    name: str,
    factory: Callable[[int], np.ndarray],
) -> str:
    masks = {count: _init(inits, factory(count), f"{name}{count}") for count in COUNTS}
    current = masks[COUNTS[0]]
    for count in COUNTS[1:]:
        eq = f"{name}_is{count}"
        out = f"{name}_sel{count}"
        const = _i64(inits, np.asarray([[[[count]]]], dtype=np.int64), f"{name}_c{count}")
        nodes.append(helper.make_node("Equal", [cnt, const], [eq]))
        nodes.append(helper.make_node("Where", [eq, masks[count], current], [out]))
        current = out
    return current


def _common_prefix(nodes: List[onnx.NodeProto], inits: List[onnx.TensorProto]) -> tuple[str, str]:
    axes = _i64(inits, [0, 1, 2, 3], "axes")
    fg_start = _i64(inits, [0, 1, 0, 0], "fg_start")
    fg_end = _i64(inits, [1, C, CORE, CORE], "fg_end")
    rep = _i64(inits, [1, 1, MAX_K, MAX_K], "rep")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, fg_start, fg_end, axes], ["fg_core"]),
            helper.make_node("ReduceSum", ["fg_core"], ["cnt_f"], axes=[1, 2, 3], keepdims=1),
            helper.make_node("Cast", ["cnt_f"], ["cnt"], to=TensorProto.INT64),
            helper.make_node("Cast", ["fg_core"], ["fg_bool"], to=TensorProto.BOOL),
            helper.make_node("Tile", ["fg_bool", rep], ["fg_tile"]),
        ]
    )
    return "cnt", "fg_tile"


def _common_suffix(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    cnt: str,
    fg_tile: str,
    copy_mask: str,
) -> None:
    active = _selected_mask(nodes, inits, cnt, name="act", factory=_active_mask)
    zero = _f32(inits, [0.0], "zero")
    nodes.extend(
        [
            helper.make_node("Greater", [copy_mask, zero], ["copy_mask_b"]),
            helper.make_node("And", [fg_tile, "copy_mask_b"], ["out_fg_b"]),
            helper.make_node("Cast", ["out_fg_b"], ["out_fg"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", ["out_fg"], ["any_fg"], axes=[1], keepdims=1),
            helper.make_node("Sub", [active, "any_fg"], ["bg"]),
            helper.make_node("Concat", ["bg", "out_fg"], ["out21"], axis=1),
            helper.make_node("Pad", ["out21"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - MAX_SIDE, W - MAX_SIDE]),
        ]
    )


def build_precomputed_model() -> onnx.ModelProto:
    """Directly select a full 21x21 copy mask."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    cnt, fg_tile = _common_prefix(nodes, inits)
    copy_mask = _selected_mask(nodes, inits, cnt, name="copy", factory=_copy_mask)
    _common_suffix(nodes, inits, cnt, fg_tile, copy_mask)
    return _make_model(nodes, inits, "task221_precomputed")


def build_kron_model() -> onnx.ModelProto:
    """Select a compact 7x7 block mask, then expand it Kronecker-style to 21x21."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    cnt, fg_tile = _common_prefix(nodes, inits)
    block = _selected_mask(nodes, inits, cnt, name="block", factory=_block_mask)
    r6 = _i64(inits, [1, 1, MAX_K, 1, MAX_K, 1], "r6")
    rep6 = _i64(inits, [1, 1, 1, CORE, 1, CORE], "rep6")
    shape21 = _i64(inits, [1, 1, MAX_SIDE, MAX_SIDE], "shape21")
    nodes.extend(
        [
            helper.make_node("Reshape", [block, r6], ["block_rank6"]),
            helper.make_node("Tile", ["block_rank6", rep6], ["copy_rank6"]),
            helper.make_node("Reshape", ["copy_rank6", shape21], ["copy_mask"]),
        ]
    )
    _common_suffix(nodes, inits, cnt, fg_tile, "copy_mask")
    return _make_model(nodes, inits, "task221_kron")


def build_bool_kron_model() -> onnx.ModelProto:
    """Bool-first graph with arithmetic 7x7 masks and one compact final float cast."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    axes = _i64(inits, [0, 1, 2, 3], "b_axes")
    fg_start = _i64(inits, [0, 1, 0, 0], "b_fg_start")
    fg_end = _i64(inits, [1, C, CORE, CORE], "b_fg_end")
    bg_start = _i64(inits, [0, 0, 0, 0], "b_bg_start")
    bg_end = _i64(inits, [1, 1, CORE, CORE], "b_bg_end")
    rep_fg = _i64(inits, [1, 1, MAX_K, MAX_K], "b_rep_fg")
    rep_bg = _i64(inits, [1, 1, MAX_K, MAX_K], "b_rep_bg")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, fg_start, fg_end, axes], ["b_fg_core"]),
            helper.make_node("ReduceSum", ["b_fg_core"], ["b_cnt_f"], axes=[1, 2, 3], keepdims=1),
            helper.make_node("Cast", ["b_fg_core"], ["b_fg_bool"], to=TensorProto.BOOL),
            helper.make_node("Tile", ["b_fg_bool", rep_fg], ["b_fg_tile"]),
            helper.make_node("Slice", [IN_NAME, bg_start, bg_end, axes], ["b_bg_core"]),
            helper.make_node("Cast", ["b_bg_core"], ["b_bg_bool"], to=TensorProto.BOOL),
            helper.make_node("Tile", ["b_bg_bool", rep_bg], ["b_bg_tile"]),
        ]
    )

    nine = _f32(inits, np.asarray([[[[9.0]]]], dtype=np.float32), "b_nine")
    rows = _f32(inits, np.arange(MAX_K, dtype=np.float32).reshape(1, 1, MAX_K, 1), "b_rows")
    cols = _f32(inits, np.arange(MAX_K, dtype=np.float32).reshape(1, 1, 1, MAX_K), "b_cols")
    r6 = _i64(inits, [1, 1, MAX_K, 1, MAX_K, 1], "b_r6")
    rep6 = _i64(inits, [1, 1, 1, CORE, 1, CORE], "b_rep6")
    shape21 = _i64(inits, [1, 1, MAX_SIDE, MAX_SIDE], "b_shape21")
    nodes.extend(
        [
            helper.make_node("Sub", [nine, "b_cnt_f"], ["b_k"]),
            helper.make_node("Less", [rows, "b_k"], ["b_row_active"]),
            helper.make_node("Less", [cols, "b_k"], ["b_col_active"]),
            helper.make_node("And", ["b_row_active", "b_col_active"], ["b_active_block"]),
            helper.make_node("Mul", [rows, "b_k"], ["b_row_base"]),
            helper.make_node("Add", ["b_row_base", cols], ["b_block_index"]),
            helper.make_node("Less", ["b_block_index", "b_cnt_f"], ["b_index_copy"]),
            helper.make_node("And", ["b_active_block", "b_index_copy"], ["b_block"]),
            helper.make_node("Not", ["b_block"], ["b_not_block"]),
            helper.make_node("And", ["b_active_block", "b_not_block"], ["b_no_copy_block"]),
            helper.make_node("Reshape", ["b_block", r6], ["b_block_rank6"]),
            helper.make_node("Tile", ["b_block_rank6", rep6], ["b_copy_rank6"]),
            helper.make_node("Reshape", ["b_copy_rank6", shape21], ["b_copy"]),
            helper.make_node("Reshape", ["b_no_copy_block", r6], ["b_no_copy_rank6"]),
            helper.make_node("Tile", ["b_no_copy_rank6", rep6], ["b_no_copy_copy_rank6"]),
            helper.make_node("Reshape", ["b_no_copy_copy_rank6", shape21], ["b_no_copy"]),
            helper.make_node("And", ["b_fg_tile", "b_copy"], ["b_out_fg"]),
            helper.make_node("And", ["b_bg_tile", "b_copy"], ["b_copy_bg"]),
            helper.make_node("Or", ["b_copy_bg", "b_no_copy"], ["b_bg"]),
            helper.make_node("Concat", ["b_bg", "b_out_fg"], ["b_out21"], axis=1),
            helper.make_node("Cast", ["b_out21"], ["b_out21_f"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["b_out21_f"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - MAX_SIDE, W - MAX_SIDE]),
        ]
    )
    return _make_model(nodes, inits, "task221_bool_kron")


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def validate_json(model: onnx.ModelProto, *, splits: tuple[str, ...] = ("train", "test", "arc-gen")) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    bad = 0
    for split in splits:
        for ex in data.get(split, []):
            x = convert_to_numpy(ex, "input")
            if x is None:
                continue
            expected = np.asarray(ex["output"], dtype=np.int64)
            pred = _onehot_to_grid(_run_onnx(model, x))[: expected.shape[0], : expected.shape[1]]
            if not np.array_equal(pred, expected):
                bad += 1
    return bad


def _score_temp(model: onnx.ModelProto, label: str) -> dict[str, Any]:
    path = Path(tempfile.gettempdir()) / f"{TASK_ID}_{label}.onnx"
    onnx.save(model, path)
    return score_file(path)


def realized_tensors(model: onnx.ModelProto) -> int:
    return sum(1 for node in model.graph.node for output in node.output if output)


def main() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        first = json.load(fh)["train"][0]
    assert np.array_equal(solve(np.asarray(first["input"])), np.asarray(first["output"]))

    candidates = {
        "precomputed": build_precomputed_model(),
        "kron": build_kron_model(),
        "bool_kron": build_bool_kron_model(),
    }
    rows: list[tuple[str, onnx.ModelProto, int, dict[str, Any], int]] = []
    for label, model in candidates.items():
        bad_train = validate_json(model, splits=("train",))
        result = _score_temp(model, label)
        rows.append((label, model, bad_train, result, realized_tensors(model)))

    passing = [row for row in rows if row[2] == 0 and row[3]["valid"]]
    if not passing:
        raise RuntimeError("no valid candidate passed train examples")
    label, model, bad_train, result, tensors = min(passing, key=lambda row: int(row[3]["cost"] or 10**9))

    onnx.save(model, BEST_PATH)
    bad_all = validate_json(model)
    final = score_file(BEST_PATH)

    print("candidate comparison:")
    for cand_label, _model, cand_bad, cand_result, cand_tensors in rows:
        print(
            f"  {cand_label:<11} train_bad={cand_bad} tensors={cand_tensors} "
            f"valid={cand_result['valid']} cost={cand_result['cost']}"
        )
    print(f"selected: {label}")
    print(f"json:     {'PASS' if bad_all == 0 else f'FAIL ({bad_all} wrong)'}")
    print(f"wrote:    {BEST_PATH}")
    print(f"valid:    {final['valid']}")
    if final["error"]:
        print(f"error:    {str(final['error']).strip()}")
    print(f"nodes:    {len(model.graph.node)}")
    print(f"tensors:  {tensors}")
    print(f"memory:   {final['memory']}")
    print(f"params:   {final['params']}")
    print(f"cost:     {final['cost']}")
    if final["score"] is not None:
        print(f"score:    {final['score']:.6f}")


if __name__ == "__main__":
    main()
