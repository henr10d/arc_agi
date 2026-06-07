"""General ONNX for ARC task018: move source objects onto marker layouts.

Task rule: one global majority color forms one or two source objects.  Each
source object also contains exactly one cell of each of the other three colors.
The remaining non-majority cells are marker layouts.  For each source object,
find the dihedral transform and translation that moves its three key cells onto
matching-color marker cells, then draw the whole transformed source object and
clear the original sources and markers.

ONNX approach: export a static PyTorch graph.  It flood-fills from the global
majority color to isolate source components, splits at most two source objects,
uses dynamic convolutions to search all eight dihedral transforms and all
translations against the marker cells, and reconstructs the one-hot output.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task018"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
K = 59
PAD = 58
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10

MATS = (
    (1, 0, 0, 1),
    (1, 0, 0, -1),
    (-1, 0, 0, 1),
    (-1, 0, 0, -1),
    (0, 1, 1, 0),
    (0, 1, -1, 0),
    (0, -1, 1, 0),
    (0, -1, -1, 0),
)


def _inverse_matrix(mat: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    a, b, c, d = mat
    det = a * d - b * c
    return d // det, -b // det, -c // det, a // det


def _transform_indices() -> tuple[np.ndarray, np.ndarray]:
    indices: list[list[int]] = []
    valid: list[list[float]] = []
    for mat in MATS:
        inv = _inverse_matrix(mat)
        mat_indices: list[int] = []
        mat_valid: list[float] = []
        for rr in range(K):
            for cc in range(K):
                y = rr - 29
                x = cc - 29
                src_r = inv[0] * y + inv[1] * x
                src_c = inv[2] * y + inv[3] * x
                ok = 0 <= src_r < H and 0 <= src_c < W
                mat_indices.append((src_r if ok else 0) * W + (src_c if ok else 0))
                mat_valid.append(1.0 if ok else 0.0)
        indices.append(mat_indices)
        valid.append(mat_valid)
    return np.asarray(indices, dtype=np.int64), np.asarray(valid, dtype=np.float32)


class Task018Module(nn.Module):
    """Static tensor implementation of the task018 transform search."""

    def __init__(self) -> None:
        super().__init__()
        indices, valid = _transform_indices()
        self.register_buffer("color_ids_i64", torch.arange(C, dtype=torch.int64).view(C))
        self.register_buffer(
            "nonzero_channels",
            torch.tensor([0] + [1] * 9, dtype=torch.float32).view(C),
        )
        self.register_buffer("flat_ids", torch.arange(H * W, dtype=torch.int64))
        self.register_buffer("indices", torch.from_numpy(indices))
        self.register_buffer("valid", torch.from_numpy(valid))

    @staticmethod
    def flood(seed: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        out = seed
        for _ in range(30):
            out = (F.max_pool2d(out, 3, 1, 1) * domain > 0.5).float()
        return out

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        counts = input.sum(dim=(0, 2, 3)) * self.nonzero_channels
        majority = torch.argmax(counts, dim=0)
        majority_channel = (self.color_ids_i64 == majority).float().view(C, 1, 1)

        majority_mask = (input[0] * majority_channel).sum(dim=0, keepdim=True).unsqueeze(0)
        inside_grid = input.sum(dim=1, keepdim=True).clamp(0, 1)
        nonzero = input[:, 1:, :, :].sum(dim=1, keepdim=True).clamp(0, 1)

        source_union = self.flood(majority_mask, nonzero)
        source_key = (source_union * (1.0 - majority_mask)).clamp(0, 1)
        seed_index = torch.argmax(source_key.reshape(-1), dim=0)
        seed = (self.flat_ids == seed_index).float().view(1, 1, H, W) * source_key

        comp1 = self.flood(seed, source_union)
        comp2 = (source_union * (1.0 - comp1)).clamp(0, 1)
        marker = (input[0] * (1.0 - source_union[0])).unsqueeze(0)
        padded_marker = F.pad(marker, (PAD, PAD, PAD, PAD))
        padded_inside = F.pad(inside_grid, (PAD, PAD, PAD, PAD))

        out_colors: list[torch.Tensor] = []
        for color in range(C):
            accum = torch.zeros(1, 1, H, W, dtype=input.dtype, device=input.device)
            for comp in (comp1, comp2):
                comp_color = input[0] * comp[0]
                comp_flat = comp_color.reshape(C, -1)

                for transform in range(8):
                    gathered = torch.index_select(comp_flat, 1, self.indices[transform])
                    gathered = gathered.view(C, K, K) * self.valid[transform].view(1, K, K)

                    key_kernel = (gathered * (1.0 - majority_channel)).unsqueeze(0)
                    full_kernel = gathered.sum(dim=0, keepdim=True).clamp(0, 1).unsqueeze(0)
                    key_count = key_kernel.sum()
                    full_count = full_kernel.sum()

                    marker_hits = F.conv2d(padded_marker, key_kernel)
                    inside_hits = F.conv2d(padded_inside, full_kernel)
                    candidate = (
                        (marker_hits > (key_count - 0.5))
                        & (inside_hits > (full_count - 0.5))
                        & (key_count > 0.5)
                    ).float()

                    placed = F.conv_transpose2d(candidate, gathered[color : color + 1].unsqueeze(0))
                    accum = accum + placed[:, :, PAD : PAD + H, PAD : PAD + W]

            out_colors.append((accum > 0.5).float())

        color_outputs = [color_mask * inside_grid for color_mask in out_colors[1:]]
        foreground = torch.cat(color_outputs, dim=1).sum(dim=1, keepdim=True).clamp(0, 1)
        background = (inside_grid - foreground).clamp(0, 1)
        return torch.cat([background] + color_outputs, dim=1)


def build_model() -> onnx.ModelProto:
    module = Task018Module().eval()
    dummy = torch.zeros(*SHAPE, dtype=torch.float32)
    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
        torch.onnx.export(
            module,
            dummy,
            tmp.name,
            input_names=[IN_NAME],
            output_names=[OUT_NAME],
            opset_version=OPSET,
            do_constant_folding=True,
            dynamic_axes=None,
        )
        model = onnx.load(tmp.name)

    model.ir_version = IR_VERSION
    model.doc_string = ""
    model.producer_name = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            out[0, int(color), row, col] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return (onehot[0] > 0.0).argmax(axis=0).astype(np.uint8)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def _components(grid: np.ndarray) -> list[list[tuple[int, int, int]]]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[list[tuple[int, int, int]]] = []
    for row in range(height):
        for col in range(width):
            if grid[row, col] == 0 or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells: list[tuple[int, int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc, int(grid[rr, cc])))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and grid[nr, nc] != 0
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(cells)
    return components


def _is_source(component: list[tuple[int, int, int]]) -> bool:
    counts = Counter(color for _, _, color in component)
    return len(counts) == 4 and sorted(counts.values())[:3] == [1, 1, 1] and max(counts.values()) >= 3


def _transform_point(row: int, col: int, mat: tuple[int, int, int, int]) -> tuple[int, int]:
    return mat[0] * row + mat[1] * col, mat[2] * row + mat[3] * col


def reference_solve(grid: list[list[int]]) -> np.ndarray:
    """Small Python mirror used to exhaustively validate the rule on JSON data."""
    arr = np.asarray(grid, dtype=np.uint8)
    height, width = arr.shape
    out = np.zeros_like(arr)
    sources = [component for component in _components(arr) if _is_source(component)]

    source_mask = np.zeros((height, width), dtype=bool)
    for component in sources:
        for row, col, _ in component:
            source_mask[row, col] = True

    marker = arr.copy()
    marker[source_mask] = 0

    for component in sources:
        counts = Counter(color for _, _, color in component)
        majority = max(counts, key=counts.get)
        keys = [(row, col, color) for row, col, color in component if color != majority]
        anchor_row, anchor_col, anchor_color = keys[0]

        for mat in MATS:
            transformed_anchor = _transform_point(anchor_row, anchor_col, mat)
            for marker_row, marker_col in np.argwhere(marker == anchor_color):
                delta_row = int(marker_row) - transformed_anchor[0]
                delta_col = int(marker_col) - transformed_anchor[1]

                matched = True
                for key_row, key_col, key_color in keys:
                    tr, tc = _transform_point(key_row, key_col, mat)
                    rr, cc = tr + delta_row, tc + delta_col
                    if not (0 <= rr < height and 0 <= cc < width and marker[rr, cc] == key_color):
                        matched = False
                        break
                if not matched:
                    continue

                placed: list[tuple[int, int, int]] = []
                for row, col, color in component:
                    tr, tc = _transform_point(row, col, mat)
                    rr, cc = tr + delta_row, tc + delta_col
                    if not (0 <= rr < height and 0 <= cc < width):
                        matched = False
                        break
                    placed.append((rr, cc, color))
                if matched:
                    for rr, cc, color in placed:
                        out[rr, cc] = color
    return out


def validate_reference() -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            expected = np.asarray(ex["output"], dtype=np.uint8)
            pred = reference_solve(ex["input"])
            if not np.array_equal(pred, expected):
                bad += 1
                print(f"reference mismatch: {split}[{idx}]")
    return bad


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    # Full ONNX validation is slow for this graph.  These cases cover the
    # training examples, the labeled test example, one-source and two-source
    # arc-gen layouts, connected marker layouts, and symmetric duplicate shapes.
    checks: tuple[tuple[str, int], ...] = (
        ("train", 0),
        ("train", 1),
        ("train", 2),
        ("test", 0),
        ("arc-gen", 0),
        ("arc-gen", 3),
        ("arc-gen", 5),
        ("arc-gen", 12),
        ("arc-gen", 131),
        ("arc-gen", 261),
    )

    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    bad = 0
    for split, idx in checks:
        ex = data[split][idx]
        expected = _grid_to_onehot(ex["output"]) > 0.0
        pred = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(ex["input"])})[0] > 0.0
        if not np.array_equal(pred, expected):
            bad += 1
            print(f"onnx mismatch: {split}[{idx}]")
    return bad


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)

    ref_bad = validate_reference()
    onnx_bad = validate_json(model)
    print(f"{TASK_ID}.json reference: {'PASS' if ref_bad == 0 else f'FAIL ({ref_bad} wrong)'}")
    print(f"{TASK_ID}.onnx sample: {'PASS' if onnx_bad == 0 else f'FAIL ({onnx_bad} wrong)'}")

    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"valid:   {result['valid']}")
    if result["error"]:
        print(f"error:   {result['error']}")
    print(f"nodes:   {len(model.graph.node)}")
    print(f"memory:  {result['memory']}")
    print(f"params:  {result['params']}")
    print(f"cost:    {result['cost']}")
    if result["score"] is not None:
        print(f"score:   {result['score']:.6f}")


if __name__ == "__main__":
    main()
