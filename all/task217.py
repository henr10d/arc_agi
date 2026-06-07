"""ONNX for ARC task217: expand a 3x3 motif into a 9x9 self-template.

Task rule: the input contains one colored motif inside a single 3x3 block of
the 9x9 grid. Ignore that block's absolute position, read the motif's occupied
cells, and place a full copy of the motif into each 3x3 output block whose
corresponding motif cell is occupied. The original color is preserved; cells
outside the 9x9 task grid remain unactivated in the NeuroGolf 30x30 tensor.

ONNX approach: use stepped slices to read the nine motif-relative positions
directly across the 3x3 block grid, avoiding a realized 9x9-to-6D reshape.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task217"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task217.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
FG = 9
H = W = 30
G = 9
CORE = 3
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10

Grid = List[List[int]]
Vec = Tuple[int, int]


def _arr(vals: Sequence[int], name: str, inits: List[onnx.TensorProto]) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))
    return name


def _f32(vals: Sequence[float], name: str, inits: List[onnx.TensorProto]) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.float32), name=name))
    return name


def load_examples() -> Dict[str, List[dict]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def foreground(grid: np.ndarray) -> Tuple[int, np.ndarray]:
    coords = np.argwhere(grid != 0)
    if coords.size == 0:
        return 0, coords
    return int(grid[tuple(coords[0])]), coords


def extract_core(grid: np.ndarray) -> Tuple[int, np.ndarray, Tuple[int, int]]:
    color, coords = foreground(grid)
    r0, c0 = coords.min(axis=0)
    r0 = int(r0 // CORE * CORE)
    c0 = int(c0 // CORE * CORE)
    return color, (grid[r0 : r0 + CORE, c0 : c0 + CORE] != 0), (r0, c0)


def solve_reference(grid: np.ndarray | Grid) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    color, core, _ = extract_core(g)
    out = np.zeros((G, G), dtype=np.int64)
    for br in range(CORE):
        for bc in range(CORE):
            if core[br, bc]:
                out[br * CORE : (br + 1) * CORE, bc * CORE : (bc + 1) * CORE][core] = color
    return out


def _valid_copy(cells: Iterable[Tuple[int, int]], h: int, w: int, dr: int, dc: int) -> bool:
    return all(0 <= r + dr < h and 0 <= c + dc < w for r, c in cells)


def _paint_copy(out: np.ndarray, cells: Iterable[Tuple[int, int]], color: int, dr: int, dc: int) -> None:
    for r, c in cells:
        out[r + dr, c + dc] = color


def infer_translation(inp: np.ndarray, out: np.ndarray) -> Tuple[Vec, int]:
    color, coords = foreground(inp)
    if not color:
        return (0, 0), 0
    cells = {tuple(map(int, rc)) for rc in coords}
    matches: List[Vec] = []
    for dr in range(-G, G + 1):
        for dc in range(-G, G + 1):
            shifted = {(r + dr, c + dc) for r, c in cells}
            if shifted and all(0 <= r < G and 0 <= c < G and out[r, c] == color for r, c in shifted):
                matches.append((dr, dc))
    nonzero = [v for v in matches if v != (0, 0)]
    if not nonzero:
        return (0, 0), len(matches)
    return min(nonzero, key=lambda v: (abs(v[0]) + abs(v[1]), v[0], v[1])), len(matches)


def propagate_whole(grid: np.ndarray, directions: Sequence[Vec], step: Vec | None = None) -> np.ndarray:
    color, coords = foreground(grid)
    out = grid.copy()
    cells = [tuple(map(int, rc)) for rc in coords]
    if not cells:
        return out
    if step is None:
        _, _, (r0, c0) = extract_core(grid)
        step = (CORE if r0 <= CORE else -CORE, CORE if c0 <= CORE else -CORE)
    for sdr, sdc in directions:
        dr, dc = step[0] * sdr, step[1] * sdc
        cur_r = cur_c = 0
        while True:
            cur_r += dr
            cur_c += dc
            if not _valid_copy(cells, *grid.shape, cur_r, cur_c):
                break
            _paint_copy(out, cells, color, cur_r, cur_c)
    return out


def propagate_from_pixels(grid: np.ndarray) -> np.ndarray:
    color, coords = foreground(grid)
    out = grid.copy()
    for r, c in coords:
        for dr, dc in ((CORE, CORE), (CORE, -CORE), (-CORE, CORE), (-CORE, -CORE)):
            rr, cc = int(r), int(c)
            while 0 <= rr + dr < G and 0 <= cc + dc < G:
                rr += dr
                cc += dc
                out[rr, cc] = color
    return out


def bbox_step(grid: np.ndarray) -> np.ndarray:
    color, coords = foreground(grid)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    step = (int(r1 - r0), int(c1 - c0))
    return propagate_whole(grid, ((1, 1),), step)


def nearest_neighbor_step(grid: np.ndarray) -> np.ndarray:
    _, coords = foreground(grid)
    best = (CORE, CORE)
    best_dist = 10**9
    pts = [tuple(map(int, rc)) for rc in coords]
    for i, (r0, c0) in enumerate(pts):
        for r1, c1 in pts[i + 1 :]:
            dr, dc = r1 - r0, c1 - c0
            if dr and dc and abs(dr) == abs(dc):
                dist = abs(dr) + abs(dc)
                if dist < best_dist:
                    best = (dr * CORE, dc * CORE)
                    best_dist = dist
    return propagate_whole(grid, ((1, 1),), best)


def candidate_diagnostics(data: Dict[str, List[dict]]) -> None:
    candidates: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "A whole-object one-way": lambda g: propagate_whole(g, ((1, 1),)),
        "B whole-object both-ways": lambda g: propagate_whole(g, ((1, 1), (-1, -1))),
        "C per-pixel diagonals": propagate_from_pixels,
        "D bbox-size step": bbox_step,
        "E nearest-neighbor diagonal step": nearest_neighbor_step,
        "F 3x3 motif self-template": solve_reference,
    }

    train = data["train"]
    print("Diagnostics for task217")
    for idx, ex in enumerate(train):
        inp = np.asarray(ex["input"], dtype=np.int64)
        out = np.asarray(ex["output"], dtype=np.int64)
        vec, copies = infer_translation(inp, out)
        _, core, top_left = extract_core(inp)
        print(f"train[{idx}] top_left={top_left} inferred_translation={vec} generated_copies={copies}")
        print(core.astype(int))
    for name, fn in candidates.items():
        ok = 0
        for ex in train:
            pred = fn(np.asarray(ex["input"], dtype=np.int64))
            ok += int(np.array_equal(pred, np.asarray(ex["output"], dtype=np.int64)))
        print(f"{name}: train_accuracy={ok}/{len(train)}")


def build_onnx_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes4 = _arr([0, 1, 2, 3], "axes4", inits)
    end_fg9 = _arr([1, C, G, G], "end_fg9", inits)
    steps3 = _arr([1, 1, CORE, CORE], "steps3", inits)
    zero = _f32([0.0], "zero", inits)

    cell_colors: List[str] = []
    cell_masks: List[str] = []
    for r in range(CORE):
        for c in range(CORE):
            start = _arr([0, 1, r, c], f"start_{r}{c}", inits)
            blocks = f"blocks_{r}{c}"
            color = f"cell_color_{r}{c}"
            maskf = f"cell_maskf_{r}{c}"
            mask = f"cell_mask_{r}{c}"
            nodes.extend(
                [
                    helper.make_node("Slice", [IN_NAME, start, end_fg9, axes4, steps3], [blocks]),
                    helper.make_node("ReduceMax", [blocks], [color], axes=[2, 3], keepdims=1),
                    helper.make_node("ReduceMax", [color], [maskf], axes=[1], keepdims=1),
                    helper.make_node("Greater", [maskf, zero], [mask]),
                ]
            )
            cell_colors.append(color)
            cell_masks.append(mask)

    row_masks: List[str] = []
    for r in range(CORE):
        row = f"mask_row_{r}"
        nodes.append(helper.make_node("Concat", cell_masks[r * CORE : (r + 1) * CORE], [row], axis=3))
        row_masks.append(row)

    nodes.extend(
        [
            helper.make_node("Max", cell_colors, ["colorf"]),
            helper.make_node("Greater", ["colorf", zero], ["colorb"]),
            helper.make_node("Concat", row_masks, ["mask3"], axis=2),
        ]
    )

    block_rows: List[str] = []
    for r in range(CORE):
        blocks: List[str] = []
        for c in range(CORE):
            block = f"filled_block_{r}{c}"
            nodes.append(helper.make_node("And", [cell_masks[r * CORE + c], "mask3"], [block]))
            blocks.append(block)
        row = f"filled_row_{r}"
        nodes.append(helper.make_node("Concat", blocks, [row], axis=3))
        block_rows.append(row)

    nodes.extend(
        [
            helper.make_node("Concat", block_rows, ["filled"], axis=2),
            helper.make_node("And", ["colorb", "filled"], ["fg_out"]),
            helper.make_node("Not", ["filled"], ["bg9"]),
            helper.make_node("Concat", ["bg9", "fg_out"], ["out9b"], axis=1),
            helper.make_node("Cast", ["out9b"], ["out9"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out9"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - G, W - G]),
        ]
    )

    graph = helper.make_graph(nodes, "task217", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="task217",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def grid_to_onehot(grid: np.ndarray | Grid) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros((1, C, H, W), dtype=np.float32)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def verify_model(path: Path, data: Dict[str, List[dict]]) -> None:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    total = 0
    for split in ("train", "test", "arc-gen"):
        ok = 0
        for ex in data.get(split, []):
            expected = grid_to_onehot(ex["output"])
            actual = session.run([OUT_NAME], {IN_NAME: grid_to_onehot(ex["input"])})[0]
            passed = np.array_equal(actual > 0.0, expected > 0.0)
            ok += int(passed)
            total += 1
            if not passed:
                raise AssertionError(f"{split} example {ok} failed")
        print(f"{split}: {ok}/{len(data.get(split, []))} correct")
    print(f"all examples: {total}/{total} correct")


def main() -> None:
    data = load_examples()
    candidate_diagnostics(data)
    model = build_onnx_model()
    onnx.save(model, BEST_PATH)
    verify_model(BEST_PATH, data)
    result = score_file(BEST_PATH)
    print(
        "score: "
        f"valid={result['valid']} memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} points={result['score']}"
    )


if __name__ == "__main__":
    main()
