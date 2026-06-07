"""ARC task047: two-dot row/column crosses; diagonal intersections become red (2).

Task rule: exactly two nonzero pixels (colors 7 and 8) on a 9×9 grid. Each dot
expands to a full row+column cross in its color. Where one dot's row meets the
other dot's column (and vice versa), output color 2. Background stays 0.

ONNX: slice 9×9 ch7/ch8, reduce each dot channel directly to compact row/col
masks, broadcast those masks into crosses, reuse one false plane for inactive
channels, bool Pad to 30×30.
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

from score_model import score_file  # noqa: E402

BEST_PATH = OUT_DIR / "task047.onnx"
DATA_PATH = ROOT / "data" / "task047.json"

C = 10
G = 9  # all task047 examples are 9×9
H = W = 30
COLOR_A = 7
COLOR_B = 8
OVERLAP = 2
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 14
IR_VERSION = 10

TOY_INPUT = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 7, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]
TOY_OUTPUT = [
    [0, 0, 8, 0, 0, 0, 7, 0, 0],
    [0, 0, 8, 0, 0, 0, 7, 0, 0],
    [8, 8, 8, 8, 8, 8, 2, 8, 8],
    [0, 0, 8, 0, 0, 0, 7, 0, 0],
    [0, 0, 8, 0, 0, 0, 7, 0, 0],
    [0, 0, 8, 0, 0, 0, 7, 0, 0],
    [7, 7, 2, 7, 7, 7, 7, 7, 7],
    [0, 0, 8, 0, 0, 0, 7, 0, 0],
    [0, 0, 8, 0, 0, 0, 7, 0, 0],
]


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))
    return name


def _f32(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name))
    return name


def build_reference_solution(grid: np.ndarray) -> np.ndarray:
    """Expand each dot across its row and column; diagonal intersections → color 2."""
    g = np.asarray(grid, dtype=np.int64)
    if g.ndim == 4:
        g = g[0].argmax(axis=0)
    elif g.ndim == 3:
        g = g.argmax(axis=0)

    dots = [(r, c, int(g[r, c])) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c]]
    if len(dots) != 2:
        raise ValueError(f"expected exactly 2 dots, got {len(dots)}")

    (y1, x1, _), (y2, x2, _) = dots
    h, w = g.shape
    out = np.zeros((h, w), dtype=np.int64)
    for r in range(h):
        for c in range(w):
            on1 = r == y1 or c == x1
            on2 = r == y2 or c == x2
            if not (on1 or on2):
                continue
            cross = (r == y1 and c == x2) or (r == y2 and c == x1)
            if cross:
                out[r, c] = OVERLAP
            elif on1:
                out[r, c] = int(g[y1, x1])
            else:
                out[r, c] = int(g[y2, x2])
    return out


def _grid_to_onehot(grid: List[List[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def build_onnx_model() -> onnx.ModelProto:
    """9×9 bool crosses from reduced row/column occupancy masks."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.BOOL, SHAPE)

    sa = _i64(inits, [0, COLOR_A, 0, 0], "sa")
    ea = _i64(inits, [1, COLOR_A + 1, G, G], "ea")
    sb = _i64(inits, [0, COLOR_B, 0, 0], "sb")
    eb = _i64(inits, [1, COLOR_B + 1, G, G], "eb")
    pads = _i64(inits, [0, 0, 0, 0, 0, 0, H - G, W - G], "pads")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, sa, ea], ["cha"]),
            helper.make_node("Slice", [IN_NAME, sb, eb], ["chb"]),
            helper.make_node("ReduceMax", ["cha"], ["arowf"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["cha"], ["acolf"], axes=[2], keepdims=1),
            helper.make_node("ReduceMax", ["chb"], ["browf"], axes=[3], keepdims=1),
            helper.make_node("ReduceMax", ["chb"], ["bcolf"], axes=[2], keepdims=1),
            helper.make_node("Cast", ["arowf"], ["rya"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["acolf"], ["cxa"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["browf"], ["ryb"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["bcolf"], ["cxb"], to=TensorProto.BOOL),
            helper.make_node("Or", ["rya", "cxa"], ["expa"]),
            helper.make_node("Or", ["ryb", "cxb"], ["expb"]),
            helper.make_node("Or", ["expa", "expb"], ["exp"]),
            helper.make_node("And", ["rya", "cxb"], ["xab"]),
            helper.make_node("And", ["ryb", "cxa"], ["xba"]),
            helper.make_node("Or", ["xab", "xba"], ["cross"]),
            helper.make_node("Not", ["cross"], ["ncross"]),
            helper.make_node("And", ["expa", "ncross"], ["ma"]),
            helper.make_node("And", ["expb", "ncross"], ["mb"]),
            helper.make_node("Not", ["exp"], ["ch0"]),
            helper.make_node("And", ["ch0", "cross"], ["zf"]),
            helper.make_node(
                "Concat",
                ["ch0", "zf", "cross", "zf", "zf", "zf", "zf", "ma", "mb", "zf"],
                ["core9"],
                axis=1,
            ),
            helper.make_node("Pad", ["core9", "pads"], [OUT_NAME], mode="constant"),
        ]
    )

    graph = helper.make_graph(nodes, "task047", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="task047",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return model


def save_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    model = build_onnx_model()
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return model


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_all(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data[split]:
            g = np.array(ex["input"], dtype=np.int64)
            if g.shape[0] > H or g.shape[1] > W:
                continue
            oh = _grid_to_onehot(ex["input"])
            pred = _onehot_to_grid(_run_onnx(model, oh)[0])[: g.shape[0], : g.shape[1]]
            ref = build_reference_solution(g)
            if not np.array_equal(pred, ref):
                bad += 1
    return bad


def main() -> None:
    model = build_onnx_model()
    bad = validate_all(model)
    print(f"validation: {'PASS' if bad == 0 else f'FAIL ({bad})'}")

    toy = _grid_to_onehot(TOY_INPUT)
    pred = _onehot_to_grid(_run_onnx(model, toy)[0])[:G, :G]
    assert np.array_equal(pred, TOY_OUTPUT), (pred, TOY_OUTPUT)

    tmp = OUT_DIR / "_task047_tmp.onnx"
    onnx.save(model, str(tmp))
    rep = score_file(tmp)
    if rep["valid"]:
        print(
            f"memory={rep['memory']} params={rep['params']} "
            f"cost={rep['cost']} score={rep['score']:.6f}"
        )
    else:
        print("score error:", rep.get("error"))
    tmp.unlink(missing_ok=True)

    save_model()
    rep2 = score_file(BEST_PATH)
    if rep2["valid"]:
        print(f"saved {BEST_PATH} cost={rep2['cost']} score={rep2['score']:.6f}")


if __name__ == "__main__":
    main()
