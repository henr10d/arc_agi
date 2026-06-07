"""Compact ONNX for ARC task168: complete diagonal rays from L trominoes.

Task rule: the 10x10 grid contains two or three same-colored L-shaped
trominoes, each occupying three cells of a 2x2 block. Preserve the trominoes.
For each 2x2 block, find the missing corner and extend the diagonal from the
opposite occupied cell through that missing corner out to the grid boundary,
leaving the missing corner itself black. All drawn cells use the input color;
background stays black.

ONNX: detect the four possible missing-corner patterns on a single object mask
with a 2x2 convolution, then use one 4-channel ConvTranspose kernel to write
all diagonal rays. The ray and input-object masks are ORed as bool tensors and
recolored by the one foreground color present in the input.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task168"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
N = 10
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def _f16(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float16), name)


def solve(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    """Reference solver using the missing corner of every 2x2 L."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    h, w = g.shape
    for r in range(h - 1):
        for c in range(w - 1):
            block = g[r : r + 2, c : c + 2]
            vals = block[block != 0]
            if vals.size != 3 or len(set(vals.tolist())) != 1:
                continue
            missing = np.argwhere(block == 0)
            if missing.shape[0] != 1:
                continue
            mr, mc = (int(missing[0, 0]), int(missing[0, 1]))
            dr = mr - (1 - mr)
            dc = mc - (1 - mc)
            rr = r + mr + dr
            cc = c + mc + dc
            while 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(vals[0])
                rr += dr
                cc += dc
    return out


def _detect_kernel() -> np.ndarray:
    """Four 2x2 background filters for missing TL, TR, BL, BR respectively."""
    patterns = [
        [[3, -1], [-1, -1]],
        [[-1, 3], [-1, -1]],
        [[-1, -1], [3, -1]],
        [[-1, -1], [-1, 3]],
    ]
    return np.asarray(patterns, dtype=np.float32).reshape(4, 1, 2, 2)


def _ray_kernel() -> np.ndarray:
    """ConvTranspose weights [4, 1, 18, 18] that paint rays into 10x10."""
    missing_corners = [(0, 0), (0, 1), (1, 0), (1, 1)]
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    weight = np.zeros((4, 1, 18, 18), dtype=np.float32)
    for channel, ((mr, mc), (dr, dc)) in enumerate(zip(missing_corners, directions)):
        for step in range(1, N):
            row = 8 + mr + dr * step
            col = 8 + mc + dc * step
            if 0 <= row < 18 and 0 <= col < 18:
                weight[channel, 0, row, col] = 1.0
    return weight


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    s_bg = _i64(inits, [0, 0, 0, 0], "s_bg")
    e_bg = _i64(inits, [1, 1, N, N], "e_bg")
    s_color = _i64(inits, [1], "s_color")
    e_color = _i64(inits, [C], "e_color")
    axis_color = _i64(inits, [1], "axis_color")
    hit_threshold = _f16(inits, [2.5], "hit_threshold")
    det_kernel = _f16(inits, _detect_kernel(), "det_kernel")
    ray_kernel = _f16(inits, _ray_kernel(), "ray_kernel")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, s_bg, e_bg], ["bg_in"]),
            helper.make_node("Cast", ["bg_in"], ["bg16"], to=TensorProto.FLOAT16),
            helper.make_node("Conv", ["bg16", det_kernel], ["score"]),
            helper.make_node("Greater", ["score", hit_threshold], ["hit"]),
            helper.make_node("Cast", ["hit"], ["hitf"], to=TensorProto.FLOAT16),
            helper.make_node(
                "ConvTranspose",
                ["hitf", ray_kernel],
                ["line"],
                kernel_shape=[18, 18],
                pads=[8, 8, 8, 8],
            ),
            helper.make_node("Cast", ["bg16"], ["bginb"], to=TensorProto.BOOL),
            helper.make_node("Not", ["bginb"], ["objb"]),
            helper.make_node("Cast", ["line"], ["lineb"], to=TensorProto.BOOL),
            helper.make_node("Or", ["objb", "lineb"], ["maskb"]),
            helper.make_node("Not", ["maskb"], ["bgb"]),
            helper.make_node("ReduceMax", [IN_NAME], ["color10f"], axes=[2, 3], keepdims=1),
            helper.make_node("Cast", ["color10f"], ["color10b"], to=TensorProto.BOOL),
            helper.make_node("Slice", ["color10b", s_color, e_color, axis_color], ["colorb"]),
            helper.make_node("And", ["colorb", "maskb"], ["fgb"]),
            helper.make_node("Concat", ["bgb", "fgb"], ["out10b"], axis=1),
            helper.make_node("Cast", ["out10b"], ["out10"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out10"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - N, W - N]),
        ]
    )

    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    bad = 0
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data[split]):
            expected_grid = solve(example["input"])
            if not np.array_equal(expected_grid, np.asarray(example["output"], dtype=np.int64)):
                raise AssertionError(f"reference mismatch in {split}[{idx}]")
            inp = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if inp is None or expected is None:
                continue
            pred = session.run([OUT_NAME], {IN_NAME: inp})[0]
            if not np.array_equal(pred > 0.0, expected > 0.0):
                bad += 1
    return bad


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    bad = validate_json(model)
    if bad:
        raise AssertionError(f"{bad} examples failed")
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(
        f"score={result['score']:.6f} cost={result['cost']} "
        f"memory={result['memory']} params={result['params']}"
    )


if __name__ == "__main__":
    main()
