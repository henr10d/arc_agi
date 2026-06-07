"""Minimal ONNX for ARC task006: AND the left and right halves.

Task rule: the input is a 3x7 grid with a separator column in the middle.
Compare the 3x3 block at columns 0:3 with the 3x3 block at columns 4:7.
The 3x3 output is color 2 exactly where both corresponding input cells are
non-background; all other output cells are background color 0. In this task's
data the non-background marks inside the compared blocks are color 1.

The ONNX graph only reads color channel 1 from the two 3x3 blocks, casts those
0/1 floats to bool, and builds a compact bool one-hot result for channels 0, 1,
and 2. The final Pad fills channels 3:10 and the 30x30 canvas directly in bool.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task006.onnx"

C = 10
H = W = 30
OUT_COLOR = 2
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 14
IR_VERSION = 10

# train[0] from task006.json
TOY_INPUT = [
    [1, 0, 0, 5, 0, 1, 0],
    [0, 1, 0, 5, 1, 1, 1],
    [1, 0, 0, 5, 0, 0, 0],
]
TOY_OUTPUT = [
    [0, 0, 0],
    [0, 2, 0],
    [0, 0, 0],
]


def _i64(inits: List[onnx.TensorProto], vals: List[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.array(vals, dtype=np.int64), name=name))
    return name


def _bool(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.bool_), name=name))
    return name


def build_reference_numpy(grid: np.ndarray, out_color: int = OUT_COLOR) -> np.ndarray:
    """Split halves, crop to common shape, emit out_color where both sides are non-zero."""
    x = np.asarray(grid, dtype=np.int64)
    if x.ndim == 4:
        x = x[0].argmax(axis=0)
    elif x.ndim == 3:
        x = x.argmax(axis=0)

    h, w = x.shape
    if w % 2 == 0:
        mid = w // 2
        a, b = x[:, :mid], x[:, mid:]
    elif h % 2 == 0:
        mid = h // 2
        a, b = x[:mid, :], x[mid:, :]
    else:
        mid = w // 2
        a, b = x[:, :mid], x[:, mid + 1 :]

    mh, mw = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a, b = a[:mh, :mw], b[:mh, :mw]
    return np.where((a > 0) & (b > 0), out_color, 0).astype(np.int64)


def _grid_to_onehot(grid: List[List[int]]) -> np.ndarray:
    out = np.zeros((1, C, H, W), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            color = int(val)
            if 0 <= color < C:
                out[0, color, r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def build_onnx_model() -> onnx.ModelProto:
    """Build the compact color-1 AND graph."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    shape = [1, C, H, W]
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, shape)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.BOOL, shape)

    axes = _i64(inits, [1, 2, 3], "axes")
    sl = _i64(inits, [1, 0, 0], "sl")
    el = _i64(inits, [2, 3, 3], "el")
    sr = _i64(inits, [1, 0, 4], "sr")
    er = _i64(inits, [2, 3, 7], "er")
    pads = _i64(inits, [0, 0, 0, 0, 0, 7, 27, 27], "pads")
    _bool(inits, np.zeros((1, 1, 3, 3), dtype=np.bool_), "zero_ch_bool")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, sl, el, axes], ["left1"]),
            helper.make_node("Slice", [IN_NAME, sr, er, axes], ["right1"]),
            helper.make_node("Cast", ["left1"], ["left_nonzero"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["right1"], ["right_nonzero"], to=TensorProto.BOOL),
            helper.make_node("And", ["left_nonzero", "right_nonzero"], ["match_bool"]),
            helper.make_node("Not", ["match_bool"], ["bg_bool"]),
            helper.make_node(
                "Concat",
                [
                    "bg_bool",
                    "zero_ch_bool",
                    "match_bool",
                ],
                ["out3"],
                axis=1,
            ),
            helper.make_node("Pad", ["out3", pads], [OUT_NAME]),
        ]
    )

    graph = helper.make_graph(nodes, "g", [x_info], [y_info], initializer=inits)
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
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]
    except ImportError:
        pass

    from onnx import reference

    sess = reference.ReferenceEvaluator(model)
    return sess.run(None, {IN_NAME: x.astype(np.float32)})[0]


def test() -> None:
    model = build_onnx_model()
    stats = model_stats(model)
    print(f"nodes={stats['nodes']} params={stats['params']} inits={stats['inits']}")

    inp = np.array(TOY_INPUT, dtype=np.int64)
    exp = np.array(TOY_OUTPUT, dtype=np.int64)
    ref = build_reference_numpy(inp)
    assert np.array_equal(ref, exp)

    toy_in = _grid_to_onehot(TOY_INPUT)
    toy_out = _run_onnx(model, toy_in)
    toy_expected = _grid_to_onehot(TOY_OUTPUT) > 0
    pred = _onehot_to_grid(toy_out[0])[:3, :3]
    ok = np.array_equal(pred, exp) and np.array_equal(toy_out > 0, toy_expected)
    print(f"toy example: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("expected:\n", exp)
        print("got:\n", pred)
        raise SystemExit(1)

    task_json = OUT_DIR.parent / "data" / "task006.json"
    if task_json.is_file():
        with task_json.open(encoding="utf-8") as fh:
            data = json.load(fh)
        bad = 0
        for split in ("train", "test", "arc-gen"):
            for ex in data[split]:
                inp = np.array(ex["input"], dtype=np.int64)
                exp = np.array(ex["output"], dtype=np.int64)
                onehot = np.zeros((1, C, H, W), dtype=np.float32)
                for r in range(inp.shape[0]):
                    for c in range(inp.shape[1]):
                        onehot[0, inp[r, c], r, c] = 1.0
                pred_onehot = _run_onnx(model, onehot) > 0
                exp_onehot = _grid_to_onehot(exp.tolist()) > 0
                if not np.array_equal(pred_onehot, exp_onehot):
                    bad += 1
        print(f"task006.json: {'PASS' if bad == 0 else f'FAIL ({bad} wrong)'}")


def main() -> None:
    inp = np.array(TOY_INPUT, dtype=np.int64)
    out = build_reference_numpy(inp)
    print("input:")
    print(inp)
    print("output:")
    print(out)
    save_model()
    test()
    print(f"saved {BEST_PATH}")


if __name__ == "__main__":
    main()
