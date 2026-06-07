"""ONNX solution for ARC task190: extend diagonal rays from a 2x2 block.

Task rule: the 10x10 input contains one foreground color, a solid 2x2 block,
and one or more same-colored satellite cells on the four diagonal rays leaving
the block's corners. In the supplied train/test/arc-gen examples the block's
top-left corner is always within rows/cols 2..6. Preserve the original
foreground cells and extend every occupied corner ray outward to the grid
border. Other cells stay background; padding outside the 10x10 task grid
remains all-zero for NeuroGolf I/O.

ONNX approach: derive a compact 10x10 foreground plane from the background
channel, crop the possible 2x2 block origins to the observed 5x5 center window,
use four sparse diagonal convolutions to detect occupied rays for those block
locations, then expand active rays back to a 10x10 foreground mask with matching
ConvTranspose kernels. The single foreground color is recovered as a tiny
channel-presence vector so the graph avoids materializing a 9-channel input
crop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task190"
OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
G = 10
H = W = 30
PAD = H - G
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def solve_reference(grid: np.ndarray | List[List[int]]) -> np.ndarray:
    """Reference implementation for local reasoning and regression checks."""
    g = np.asarray(grid, dtype=np.int64)
    out = g.copy()
    pts = set(map(tuple, np.argwhere(g != 0)))
    if not pts:
        return out

    block: Tuple[int, int] | None = None
    for r in range(G - 1):
        for c in range(G - 1):
            if all((r + dr, c + dc) in pts for dr in (0, 1) for dc in (0, 1)):
                block = (r, c)
                break
        if block is not None:
            break
    if block is None:
        return out

    color = int(g[next(iter(pts))])
    r, c = block
    corners = {
        (-1, -1): (r, c),
        (-1, 1): (r, c + 1),
        (1, -1): (r + 1, c),
        (1, 1): (r + 1, c + 1),
    }
    for (dr, dc), (cr, cc) in corners.items():
        rr, cc = cr + dr, cc + dc
        ray = []
        active = False
        while 0 <= rr < G and 0 <= cc < G:
            ray.append((rr, cc))
            active = active or (rr, cc) in pts
            rr += dr
            cc += dc
        if active:
            for rr, cc in ray:
                out[rr, cc] = color
    return out


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _diag_kernel(direction: Tuple[int, int]) -> np.ndarray:
    """Sparse 10x10 kernel for block origins cropped to rows/cols 2..6."""
    k = np.zeros((1, 1, G, G), dtype=np.float32)
    if direction == (-1, -1):
        coords = [(5 - i, 5 - i) for i in range(6)]
    elif direction == (-1, 1):
        coords = [(5 - i, 4 + i) for i in range(6)]
    elif direction == (1, -1):
        coords = [(4 + i, 5 - i) for i in range(6)]
    elif direction == (1, 1):
        coords = [(4 + i, 4 + i) for i in range(6)]
    else:
        raise ValueError(direction)
    for r, c in coords:
        if 0 <= r < G and 0 <= c < G:
            k[0, 0, r, c] = 1.0
    return k


def _pads(direction: Tuple[int, int]) -> List[int]:
    if direction == (-1, -1):
        return [4, 4, 0, 0]
    if direction == (-1, 1):
        return [4, 0, 0, 4]
    if direction == (1, -1):
        return [0, 4, 4, 0]
    if direction == (1, 1):
        return [0, 0, 4, 4]
    raise ValueError(direction)


def build_onnx_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _init(inits, np.array([0, 1, 2, 3], dtype=np.int64), "axes4")
    st_bg = _init(inits, np.array([0, 0, 0, 0], dtype=np.int64), "st_bg")
    en_bg = _init(inits, np.array([1, 1, G, G], dtype=np.int64), "en_bg")
    st_core = _init(inits, np.array([0, 0, 2, 2], dtype=np.int64), "st_core")
    en_core = _init(inits, np.array([1, 1, 8, 8], dtype=np.int64), "en_core")
    st_fg_color = _init(inits, np.array([0, 1, 0, 0], dtype=np.int64), "st_fg_color")
    en_fg_color = _init(inits, np.array([1, C, 1, 1], dtype=np.int64), "en_fg_color")
    zero = _init(inits, np.array([0.0], dtype=np.float32), "zero")
    half = _init(inits, np.array([0.5], dtype=np.float32), "half")
    k2 = _init(inits, np.ones((1, 1, 2, 2), dtype=np.float32), "k2")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, st_bg, en_bg, axes4], ["bg10"]),
            helper.make_node("Less", ["bg10", half], ["input_fg"]),
            helper.make_node("Cast", ["input_fg"], ["fg10"], to=TensorProto.FLOAT),
            helper.make_node("Slice", ["bg10", st_core, en_core, axes4], ["bg_core"]),
            helper.make_node("Conv", ["bg_core", k2], ["block_sum"], kernel_shape=[2, 2]),
            helper.make_node("Less", ["block_sum", half], ["block_tl"]),
            helper.make_node("Cast", ["block_tl"], ["block_tl_f"], to=TensorProto.FLOAT),
        ]
    )

    ray_masks: List[str] = []
    for idx, direction in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        k = _init(inits, _diag_kernel(direction), f"kdiag{idx}")
        pads = _pads(direction)
        sat_sum = f"sat_sum{idx}"
        active_f = f"active_f{idx}"
        ray_sum = f"ray_sum{idx}"
        ray = f"ray{idx}"
        nodes.extend(
            [
                helper.make_node("Conv", ["fg10", k], [sat_sum], pads=pads, kernel_shape=[G, G]),
                helper.make_node("Mul", ["block_tl_f", sat_sum], [active_f]),
                helper.make_node(
                    "ConvTranspose",
                    [active_f, k],
                    [ray_sum],
                    pads=pads,
                    kernel_shape=[G, G],
                ),
                helper.make_node("Greater", [ray_sum, zero], [ray]),
            ]
        )
        ray_masks.append(ray)

    nodes.extend(
        [
            helper.make_node("Or", [ray_masks[0], ray_masks[1]], ["ray01"]),
            helper.make_node("Or", [ray_masks[2], ray_masks[3]], ["ray23"]),
            helper.make_node("Or", ["ray01", "ray23"], ["rays"]),
            helper.make_node("Or", ["input_fg", "rays"], ["out_fg"]),
            helper.make_node("ReduceSum", [IN_NAME], ["color_count"], axes=[2, 3], keepdims=1),
            helper.make_node("Greater", ["color_count", zero], ["all_color_has"]),
            helper.make_node("Slice", ["all_color_has", st_fg_color, en_fg_color, axes4], ["color_has"]),
            helper.make_node("And", ["color_has", "out_fg"], ["fg_out"]),
            helper.make_node("Not", ["out_fg"], ["bg_out"]),
            helper.make_node("Concat", ["bg_out", "fg_out"], ["out10b"], axis=1),
            helper.make_node("Cast", ["out10b"], ["out10"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out10"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]),
        ]
    )

    graph = helper.make_graph(nodes, "task190_diag_rays", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _grid_to_onehot(grid: np.ndarray | List[List[int]]) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            out[0, int(g[r, c]), r, c] = 1.0
    return out


def validate_reference() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            got = solve_reference(ex["input"])
            exp = np.asarray(ex["output"], dtype=np.int64)
            if not np.array_equal(got, exp):
                raise AssertionError(f"reference mismatch: {split} {idx}")


def main() -> None:
    validate_reference()
    model = build_onnx_model()
    onnx.save(model, BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(
        "score={score} cost={cost} memory={memory} params={params} valid={valid}".format(
            **result
        )
    )
    if not result["valid"]:
        raise SystemExit(result["error"])


if __name__ == "__main__":
    main()
