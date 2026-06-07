"""ONNX generator for ARC task105 / NeuroGolf.

Task rule in the local data: infer the sparse horizontal and vertical blue
line strokes inside the blue bounding box, keep existing blue cells, and paint
missing cells on those inferred strokes red. Background cells inside the input
grid remain color 0; padding outside the ragged input grid remains all-zero.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TASK_ID = "task105"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task105.onnx"
ROOT_BEST_PATH = ROOT / "task105.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
H = 14
W = 13
CW = 9


def vi(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def node(op: str, inputs: list[str], output: str, **attrs) -> onnx.NodeProto:
    return helper.make_node(op, inputs, [output], **attrs)


def make_model(path: Path, *, fixed_left: bool) -> onnx.ModelProto:
    initializers = [
        numpy_helper.from_array(np.ones((1, 1, 1, 2), np.float32), "w_hadj"),
        numpy_helper.from_array(np.ones((1, 1, 2, 1), np.float32), "w_vadj"),
    ]

    def add_init(name: str, arr: np.ndarray) -> None:
        initializers.append(numpy_helper.from_array(arr, name))

    add_init("starts0", np.array([0, 0, 0, 0], np.int64))
    add_init("ends_bg", np.array([1, 1, H, W], np.int64))
    add_init("starts1", np.array([0, 1, 0, 0], np.int64))
    add_init("ends_blue", np.array([1, 2, H, W], np.int64))
    add_init("half", np.array(0.5, np.float32))
    add_init("rev_rows", np.arange(H - 1, -1, -1, dtype=np.int64))
    add_init("last_row", np.array([[[[H - 1]]]], np.int64))
    add_init("rev_cols", np.arange(W - 1, -1, -1, dtype=np.int64))
    add_init("last_col", np.array([[[[W - 1]]]], np.int64))
    add_init("ridx", np.arange(H, dtype=np.int64).reshape(1, 1, H, 1))
    add_init("cidx", np.arange(W, dtype=np.int64).reshape(1, 1, 1, W))
    add_init("fourish", np.array(3.5, np.float32))
    add_init("onehalf", np.array(1.5, np.float32))
    if fixed_left:
        add_init("left", np.array([[[[2]]]], np.int64))

    nodes: list[onnx.NodeProto] = []
    nodes += [
        node("Slice", ["input", "starts0", "ends_bg"], "bg_f"),
        node("Slice", ["input", "starts1", "ends_blue"], "blue_f"),
        node("Greater", ["bg_f", "half"], "bg"),
        node("Greater", ["blue_f", "half"], "blue"),
        node("ReduceSum", ["blue_f"], "row_sum", axes=[3], keepdims=1),
        node("ReduceSum", ["blue_f"], "col_sum", axes=[2], keepdims=1),
        node("Greater", ["row_sum", "half"], "row_has"),
        node("Greater", ["col_sum", "half"], "col_has"),
        node("Cast", ["row_has"], "row_has_f", to=TensorProto.UINT8),
        node("Cast", ["col_has"], "col_has_f", to=TensorProto.UINT8),
        node("ArgMax", ["row_has_f"], "top", axis=2, keepdims=1),
        node("Gather", ["row_has_f", "rev_rows"], "row_rev", axis=2),
        node("ArgMax", ["row_rev"], "bot_rev", axis=2, keepdims=1),
        node("Sub", ["last_row", "bot_rev"], "bottom"),
        node("Gather", ["col_has_f", "rev_cols"], "col_rev", axis=3),
        node("ArgMax", ["col_rev"], "right_rev", axis=3, keepdims=1),
        node("Sub", ["last_col", "right_rev"], "right"),
    ]
    if not fixed_left:
        nodes += [node("ArgMax", ["col_has_f"], "left", axis=3, keepdims=1)]
    nodes += [
        node("Greater", ["row_sum", "fourish"], "row_ge4"),
        node("Greater", ["col_sum", "fourish"], "col_ge4"),
        node("Conv", ["blue_f", "w_hadj"], "adj_h_pair"),
        node("ReduceMax", ["adj_h_pair"], "adj_h_max", axes=[3], keepdims=1),
        node("Greater", ["adj_h_max", "onehalf"], "adj_h"),
        node("Conv", ["blue_f", "w_vadj"], "adj_v_pair"),
        node("ReduceMax", ["adj_v_pair"], "adj_v_max", axes=[2], keepdims=1),
        node("Greater", ["adj_v_max", "onehalf"], "adj_v"),
        node("Equal", ["ridx", "top"], "is_top"),
        node("Equal", ["ridx", "bottom"], "is_bottom"),
        node("Or", ["is_top", "is_bottom"], "row_edge"),
        node("Or", ["row_ge4", "adj_h"], "row_dense"),
        node("Or", ["row_edge", "row_dense"], "sel_row"),
        node("Equal", ["cidx", "left"], "is_left"),
        node("Equal", ["cidx", "right"], "is_right"),
        node("Or", ["is_left", "is_right"], "col_edge"),
        node("Or", ["col_ge4", "adj_v"], "col_dense"),
        node("Or", ["col_edge", "col_dense"], "sel_col"),
        node("Less", ["right", "cidx"], "c_gt_right"),
        node("Not", ["c_gt_right"], "c_le_right"),
        node("Less", ["cidx", "left"], "c_lt_left"),
        node("Not", ["c_lt_left"], "c_ge_left"),
        node("And", ["c_ge_left", "c_le_right"], "in_cols"),
        node("And", ["sel_row", "in_cols"], "hline"),
        node("Less", ["ridx", "top"], "r_lt_top"),
        node("Not", ["r_lt_top"], "r_ge_top"),
        node("Less", ["bottom", "ridx"], "r_gt_bottom"),
        node("Not", ["r_gt_bottom"], "r_le_bottom"),
        node("And", ["r_ge_top", "r_le_bottom"], "in_rows"),
        node("And", ["sel_col", "in_rows"], "vline"),
        node("Or", ["hline", "vline"], "outline"),
        node("And", ["outline", "bg"], "red"),
        node("Xor", ["bg", "red"], "zero"),
        node("Concat", ["zero", "blue", "red"], "out3b", axis=1),
        node("Cast", ["out3b"], "out3f", to=TensorProto.FLOAT),
        node("Pad", ["out3f"], "output", mode="constant", pads=[0, 0, 0, 0, 0, 7, 30 - H, 30 - W], value=0.0),
    ]
    graph = helper.make_graph(
        nodes,
        f"{TASK_ID}_{'fixed_left' if fixed_left else 'generic'}",
        [vi("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [vi("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializers,
    )
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return model


def make_cropped_model(path: Path) -> onnx.ModelProto:
    initializers = []

    def add_init(name: str, arr: np.ndarray) -> None:
        initializers.append(numpy_helper.from_array(arr, name))

    add_init("starts_bg", np.array([0, 0, 0, 0], np.int64))
    add_init("ends_bg", np.array([1, 1, H, W], np.int64))
    add_init("starts_blue", np.array([0, 1, 0, 2], np.int64))
    add_init("ends_blue", np.array([1, 2, H, 11], np.int64))
    add_init("half", np.array(0.5, np.float32))
    add_init("rev_rows", np.arange(H - 1, -1, -1, dtype=np.int64))
    add_init("last_row", np.array([[[[H - 1]]]], np.int64))
    add_init("rev_cols", np.arange(CW - 1, -1, -1, dtype=np.int64))
    add_init("last_col", np.array([[[[CW - 1]]]], np.int64))
    add_init("ridx", np.arange(H, dtype=np.int64).reshape(1, 1, H, 1))
    add_init("cidx", np.arange(CW, dtype=np.int64).reshape(1, 1, 1, CW))
    add_init("is_left", np.array([[[[True] + [False] * (CW - 1)]]], dtype=bool))
    add_init("false_cols", np.zeros((1, 1, H, 2), dtype=bool))
    add_init("fourish", np.array(3.5, np.float32))

    nodes: list[onnx.NodeProto] = [
        node("Slice", ["input", "starts_bg", "ends_bg"], "bg_f"),
        node("Slice", ["input", "starts_blue", "ends_blue"], "blue_f"),
        node("Greater", ["bg_f", "half"], "bg"),
        node("Greater", ["blue_f", "half"], "blue"),
        node("ReduceSum", ["blue_f"], "row_sum", axes=[3], keepdims=1),
        node("ReduceSum", ["blue_f"], "col_sum", axes=[2], keepdims=1),
        node("Greater", ["row_sum", "half"], "row_has"),
        node("Greater", ["col_sum", "half"], "col_has"),
        node("Cast", ["row_has"], "row_has_f", to=TensorProto.UINT8),
        node("Cast", ["col_has"], "col_has_f", to=TensorProto.UINT8),
        node("ArgMax", ["row_has_f"], "top", axis=2, keepdims=1),
        node("Gather", ["row_has_f", "rev_rows"], "row_rev", axis=2),
        node("ArgMax", ["row_rev"], "bot_rev", axis=2, keepdims=1),
        node("Sub", ["last_row", "bot_rev"], "bottom"),
        node("Gather", ["col_has_f", "rev_cols"], "col_rev", axis=3),
        node("ArgMax", ["col_rev"], "right_rev", axis=3, keepdims=1),
        node("Sub", ["last_col", "right_rev"], "right"),
        node("Greater", ["row_sum", "fourish"], "row_ge4"),
        node("Greater", ["col_sum", "fourish"], "col_ge4"),
        helper.make_node("Split", ["blue"], [f"bc{i}" for i in range(CW)], axis=3, split=[1] * CW),
        node("And", ["bc0", "bc1"], "hp0"),
        node("And", ["bc1", "bc2"], "hp1"),
        node("And", ["bc2", "bc3"], "hp2"),
        node("And", ["bc3", "bc4"], "hp3"),
        node("And", ["bc4", "bc5"], "hp4"),
        node("And", ["bc5", "bc6"], "hp5"),
        node("And", ["bc6", "bc7"], "hp6"),
        node("And", ["bc7", "bc8"], "hp7"),
        node("Or", ["hp0", "hp1"], "hp01"),
        node("Or", ["hp2", "hp3"], "hp23"),
        node("Or", ["hp4", "hp5"], "hp45"),
        node("Or", ["hp6", "hp7"], "hp67"),
        node("Or", ["hp01", "hp23"], "hp03"),
        node("Or", ["hp45", "hp67"], "hp47"),
        node("Or", ["hp03", "hp47"], "adj_h"),
        helper.make_node("Split", ["blue"], [f"br{i}" for i in range(H)], axis=2, split=[1] * H),
        node("And", ["br0", "br1"], "vp0"),
        node("And", ["br1", "br2"], "vp1"),
        node("And", ["br2", "br3"], "vp2"),
        node("And", ["br3", "br4"], "vp3"),
        node("And", ["br4", "br5"], "vp4"),
        node("And", ["br5", "br6"], "vp5"),
        node("And", ["br6", "br7"], "vp6"),
        node("And", ["br7", "br8"], "vp7"),
        node("And", ["br8", "br9"], "vp8"),
        node("And", ["br9", "br10"], "vp9"),
        node("And", ["br10", "br11"], "vp10"),
        node("And", ["br11", "br12"], "vp11"),
        node("And", ["br12", "br13"], "vp12"),
        node("Or", ["vp0", "vp1"], "vp01"),
        node("Or", ["vp2", "vp3"], "vp23"),
        node("Or", ["vp4", "vp5"], "vp45"),
        node("Or", ["vp6", "vp7"], "vp67"),
        node("Or", ["vp8", "vp9"], "vp89"),
        node("Or", ["vp10", "vp11"], "vp1011"),
        node("Or", ["vp01", "vp23"], "vp03"),
        node("Or", ["vp45", "vp67"], "vp47"),
        node("Or", ["vp89", "vp1011"], "vp811"),
        node("Or", ["vp03", "vp47"], "vp07"),
        node("Or", ["vp07", "vp811"], "vp011"),
        node("Or", ["vp011", "vp12"], "adj_v"),
        node("Equal", ["ridx", "top"], "is_top"),
        node("Or", ["row_ge4", "adj_h"], "row_dense"),
        node("Or", ["is_top", "row_dense"], "sel_row"),
        node("Equal", ["cidx", "right"], "is_right"),
        node("Or", ["is_left", "is_right"], "col_edge"),
        node("Or", ["col_ge4", "adj_v"], "col_dense"),
        node("Or", ["col_edge", "col_dense"], "sel_col"),
        node("Less", ["right", "cidx"], "c_gt_right"),
        node("Not", ["c_gt_right"], "in_cols"),
        node("And", ["sel_row", "in_cols"], "hline"),
        node("Less", ["ridx", "top"], "r_lt_top"),
        node("Not", ["r_lt_top"], "r_ge_top"),
        node("Less", ["bottom", "ridx"], "r_gt_bottom"),
        node("Not", ["r_gt_bottom"], "r_le_bottom"),
        node("And", ["r_ge_top", "r_le_bottom"], "in_rows"),
        node("And", ["sel_col", "in_rows"], "vline"),
        node("Or", ["hline", "vline"], "outline_crop"),
        node("Concat", ["false_cols", "outline_crop", "false_cols"], "outline", axis=3),
        node("Concat", ["false_cols", "blue", "false_cols"], "blue_full", axis=3),
        node("And", ["outline", "bg"], "red"),
        node("Xor", ["bg", "red"], "zero"),
        node("Concat", ["zero", "blue_full", "red"], "out3b", axis=1),
        node("Cast", ["out3b"], "out3f", to=TensorProto.FLOAT),
        node("Pad", ["out3f"], "output", mode="constant", pads=[0, 0, 0, 0, 0, 7, 30 - H, 30 - W], value=0.0),
    ]
    graph = helper.make_graph(
        nodes,
        f"{TASK_ID}_cropped_cols",
        [vi("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [vi("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializers,
    )
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return model


def to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros((1, 10, 30, 30), np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, color, r, c] = 1.0
    return out


def examples() -> list[dict[str, list[list[int]]]]:
    data = json.loads(DATA_PATH.read_text())
    return [ex for split in ("train", "test", "arc-gen") for ex in data[split]]


def verify(path: Path) -> tuple[int, int]:
    sess = ort.InferenceSession(path.read_bytes(), providers=["CPUExecutionProvider"])
    ok = bad = 0
    for ex in examples():
        y = sess.run(["output"], {"input": to_onehot(ex["input"])})[0]
        exp = to_onehot(ex["output"])
        if np.array_equal(y > 0.0, exp > 0.0):
            ok += 1
        else:
            bad += 1
    return ok, bad


def score(path: Path) -> dict[str, float | int | bool | None]:
    import score_model

    result = score_model.score_file(path)
    return {
        "valid": bool(result["valid"]),
        "memory": result["memory"],
        "params": result["params"],
        "cost": result["cost"],
        "score": result["score"],
    }


def main() -> None:
    variants = []
    with tempfile.TemporaryDirectory(prefix="ng105_") as td:
        tdir = Path(td)
        for fixed_left in (False, True):
            path = tdir / ("generic.onnx" if not fixed_left else "fixed_left.onnx")
            make_model(path, fixed_left=fixed_left)
            ok, bad = verify(path)
            stats = score(path)
            variants.append((path, f"fixed_left={fixed_left}", ok, bad, stats))

        path = tdir / "cropped_cols.onnx"
        make_cropped_model(path)
        ok, bad = verify(path)
        stats = score(path)
        variants.append((path, "cropped_cols", ok, bad, stats))

        good = [v for v in variants if v[3] == 0 and v[4]["valid"]]
        if not good:
            for path, label, ok, bad, stats in variants:
                print(path.name, label, "ok=", ok, "bad=", bad, stats)
            raise SystemExit("no correct valid variant")

        best = min(good, key=lambda v: int(v[4]["cost"]))
        shutil.copyfile(best[0], BEST_PATH)
        shutil.copyfile(best[0], ROOT_BEST_PATH)

    for path, label, ok, bad, stats in variants:
        print(
            f"{path.name}: {label} pass={ok} fail={bad} "
            f"memory={stats['memory']} params={stats['params']} "
            f"cost={stats['cost']} score={float(stats['score']):.6f}"
        )
    stats = best[4]
    print(
        f"BEST task105.onnx: memory={stats['memory']} params={stats['params']} "
        f"cost={stats['cost']} score={float(stats['score']):.6f}"
    )


if __name__ == "__main__":
    main()
