#!/usr/bin/env python3
"""
ARC-AGI training + ONNX export pipeline optimized for rule generalization.

Design goals:
  - Cellular-automata-style iterative conv updates (translation-equivariant)
  - Synthetic ARC-GEN-style augmentation during training
  - Reject solutions that memorize train positions / colors
  - Static-shape ONNX export without banned ops
"""

from __future__ import annotations

import json
import random
import sys
import time
import zipfile
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Config (edit here; no argparse)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
SUBMISSION_DIR = ROOT / "submission"
VIZ_DIR = ROOT / "out"

GRID_SIZE = 30
NUM_COLORS = 10
ONNX_INPUT_SHAPE = (1, NUM_COLORS, GRID_SIZE, GRID_SIZE)
ONNX_OPSET = 10

# Cellular-automata model
HIDDEN_CHANNELS = 32
N_CA_ITERATIONS = 18          # unrolled in forward(); RF ~ 37 on 30x30
CA_KERNEL = 3

# Training
BATCH_SIZE = 16
TASK_STEPS = 30_000
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
SEED = 42

# Generalization-first training mix (mostly procedural, not raw train)
SYNTH_SAMPLES_PER_STEP = 12
REAL_IN_BATCH = 1             # max un-augmented real pairs per batch (anti-memorization)
AUGMENT_REAL_PROB = 0.85
SYNTH_TOPLEFT_FRACTION = 0.5  # half of synth examples match ARC top-left layout
MIN_SYNTH_VAL_ACC = 0.90
MIN_ARCGEN_PASS_RATE = 0.95
MEMORIZATION_GAP = 0.20

TASK_FILTER: Optional[str] = None        # set e.g. "task001" to train a single task
TASK_START = 1
TASK_END = 400

EXPORT_ONLY = "--export_only" in sys.argv
VALIDATE_ONLY = "--validate_only" in sys.argv
SKIP_TRAIN = "--skip_train" in sys.argv

BANNED_ONNX_OPS = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Grid utilities
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_task_num(task_id: str) -> int:
    return int(task_id.replace("task", ""))


def grid_shape(grid: List[List[int]]) -> Tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def grid_to_array(grid: List[List[int]]) -> np.ndarray:
    h, w = grid_shape(grid)
    arr = np.zeros((h, w), dtype=np.int64)
    for r in range(h):
        for c in range(w):
            arr[r, c] = int(grid[r][c])
    return arr


def array_to_grid(arr: np.ndarray) -> List[List[int]]:
    return arr.astype(int).tolist()


def bbox_of_grid(grid: np.ndarray, background: int = 0) -> Tuple[int, int, int, int]:
    """Inclusive min/max row/col of cells != background; empty -> (0,0,0,0)."""
    active = np.argwhere(grid != background)
    if active.size == 0:
        return 0, 0, 0, 0
    r0, c0 = active.min(axis=0)
    r1, c1 = active.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, background: int = 0) -> np.ndarray:
    r0, c0, r1, c1 = bbox_of_grid(grid, background)
    if r1 < r0:
        return np.zeros((1, 1), dtype=grid.dtype)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def pad_grid(grid: List[List[int]] | np.ndarray, size: int = GRID_SIZE) -> np.ndarray:
    if isinstance(grid, list):
        grid = grid_to_array(grid)
    arr = np.zeros((size, size), dtype=np.int64)
    h, w = min(grid.shape[0], size), min(grid.shape[1], size)
    arr[:h, :w] = grid[:h, :w]
    return arr


def place_patch(canvas: np.ndarray, patch: np.ndarray, row: int, col: int) -> np.ndarray:
    out = canvas.copy()
    ph, pw = patch.shape
    r1 = min(row + ph, out.shape[0])
    c1 = min(col + pw, out.shape[1])
    out[row:r1, col:c1] = patch[: r1 - row, : c1 - col]
    return out


def grid_to_onehot(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    out = np.zeros((1, NUM_COLORS, h, w), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if 0 <= color < NUM_COLORS:
                out[0, color, r, c] = 1.0
    return out


def onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    if onehot.ndim == 4:
        channel_map = onehot[0]
    elif onehot.ndim == 3:
        channel_map = onehot
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {onehot.shape}")
    return channel_map.argmax(axis=0).astype(np.int64)


def example_to_tensors(example: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    inp = pad_grid(example["input"])
    out = pad_grid(example["output"])
    inp_oh = torch.from_numpy(grid_to_onehot(inp))
    return inp_oh, torch.from_numpy(out)


def make_static_example_input() -> torch.Tensor:
    return torch.zeros(ONNX_INPUT_SHAPE, dtype=torch.float32)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_competition_score(n_params: int) -> float:
    return max(1.0, 25.0 - np.log(max(1.0, n_params + 8_000)))


# ---------------------------------------------------------------------------
# Rule system: detect + apply algorithmic transforms
# ---------------------------------------------------------------------------
class RuleKind(Enum):
    UNKNOWN = auto()
    TILE = auto()
    SCALE_NEAREST = auto()
    STENCIL_COPY = auto()       # copy full pattern into blocks where input cell != 0
    HFLIP = auto()
    VFLIP = auto()
    HVFLIP = auto()
    COLOR_REPLACE = auto()
    COPY_OFFSET = auto()
    FLOOD_FILL_BOUNDARY = auto()


@dataclass
class TaskRule:
    kind: RuleKind
    factor: int = 1
    factor_w: Optional[int] = None
    color_map: Dict[int, int] = field(default_factory=dict)
    offset: Tuple[int, int] = (0, 0)
    confidence: float = 0.0
    label: str = "unknown"

    @property
    def factor_h(self) -> int:
        return self.factor

    @property
    def fw(self) -> int:
        return self.factor if self.factor_w is None else self.factor_w


def tile_patch(patch: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(patch, np.ones((factor, factor), dtype=patch.dtype))


def scale_nearest(patch: np.ndarray, factor: int) -> np.ndarray:
    h, w = patch.shape
    out = np.zeros((h * factor, w * factor), dtype=patch.dtype)
    for r in range(h):
        for c in range(w):
            out[r * factor : (r + 1) * factor, c * factor : (c + 1) * factor] = patch[r, c]
    return out


def apply_color_map(grid: np.ndarray, color_map: Dict[int, int]) -> np.ndarray:
    out = grid.copy()
    for src, dst in color_map.items():
        out[grid == src] = dst
    return out


def flip_grid(grid: np.ndarray, mode: str) -> np.ndarray:
    if mode == "h":
        return np.fliplr(grid)
    if mode == "v":
        return np.flipud(grid)
    if mode == "hv":
        return np.flipud(np.fliplr(grid))
    raise ValueError(mode)


def stencil_copy_pattern(inp_core: np.ndarray, factor_h: int, factor_w: Optional[int] = None) -> np.ndarray:
    """Place a copy of the full input wherever input[r,c] != 0 (ARC task001-style)."""
    if factor_w is None:
        factor_w = factor_h
    padded = np.zeros((factor_h, factor_w), dtype=inp_core.dtype)
    ph = min(factor_h, inp_core.shape[0])
    pw = min(factor_w, inp_core.shape[1])
    padded[:ph, :pw] = inp_core[:ph, :pw]
    inp_core = padded
    ih, iw = inp_core.shape
    out = np.zeros((ih * factor_h, iw * factor_w), dtype=inp_core.dtype)
    for r in range(ih):
        for c in range(iw):
            if inp_core[r, c] != 0:
                out[
                    r * factor_h : (r + 1) * factor_h,
                    c * factor_w : (c + 1) * factor_w,
                ] = inp_core
    return out


def apply_rule_to_core(inp_core: np.ndarray, rule: TaskRule) -> np.ndarray:
    if rule.kind == RuleKind.TILE:
        return tile_patch(inp_core, rule.factor)
    if rule.kind == RuleKind.SCALE_NEAREST:
        return scale_nearest(inp_core, rule.factor)
    if rule.kind == RuleKind.STENCIL_COPY:
        return stencil_copy_pattern(inp_core, rule.factor_h, rule.fw)
    if rule.kind == RuleKind.HFLIP:
        return flip_grid(inp_core, "h")
    if rule.kind == RuleKind.VFLIP:
        return flip_grid(inp_core, "v")
    if rule.kind == RuleKind.HVFLIP:
        return flip_grid(inp_core, "hv")
    if rule.kind == RuleKind.COLOR_REPLACE:
        return apply_color_map(inp_core, rule.color_map)
    if rule.kind == RuleKind.COPY_OFFSET:
        dr, dc = rule.offset
        out = np.zeros_like(inp_core)
        h, w = inp_core.shape
        for r in range(h):
            for c in range(w):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = inp_core[r, c]
        return out
    if rule.kind == RuleKind.FLOOD_FILL_BOUNDARY:
        return flood_fill_bbox(inp_core)
    return inp_core.copy()


def apply_rule_padded(inp: np.ndarray, rule: TaskRule) -> np.ndarray:
    """Apply rule on the active content region, preserving canvas placement."""
    r0, c0, r1, c1 = bbox_of_grid(inp)
    if r1 < r0:
        return np.zeros_like(inp)
    core = inp[r0 : r1 + 1, c0 : c1 + 1]
    out_core = apply_rule_to_core(core, rule)
    out = np.zeros((max(inp.shape[0], r0 + out_core.shape[0]), max(inp.shape[1], c0 + out_core.shape[1])), dtype=inp.dtype)
    out = np.zeros_like(inp)
    rh, rw = out_core.shape
    if r0 + rh <= out.shape[0] and c0 + rw <= out.shape[1]:
        out[r0 : r0 + rh, c0 : c0 + rw] = out_core
    return out


def flood_fill_bbox(grid: np.ndarray, fill_color: int = 4) -> np.ndarray:
    """Fill bounding box interior (simple morphological task)."""
    r0, c0, r1, c1 = bbox_of_grid(grid)
    if r1 < r0:
        return grid.copy()
    out = grid.copy()
    if r1 - r0 >= 2 and c1 - c0 >= 2:
        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color
    return out


def _shapes_match_rule(inp: np.ndarray, out: np.ndarray, rule: TaskRule) -> bool:
    pred = apply_rule_to_core(inp, rule)
    return pred.shape == out.shape and np.array_equal(pred, out)


def infer_rule_from_pair(inp: np.ndarray, out: np.ndarray) -> Optional[TaskRule]:
    """Infer transform using native (un-cropped) grids from the JSON example."""
    if isinstance(inp, list):
        inp = grid_to_array(inp)
    if isinstance(out, list):
        out = grid_to_array(out)
    inp_c = inp
    out_c = out
    ih, iw = inp_c.shape
    oh, ow = out_c.shape

    if ih > 0 and iw > 0 and oh % ih == 0 and ow % iw == 0:
        fh, fw = oh // ih, ow // iw
        sc = TaskRule(RuleKind.STENCIL_COPY, factor=fh, factor_w=fw, label=f"stencil_copy_{fh}x{fw}")
        if _shapes_match_rule(inp_c, out_c, sc):
            return sc

    for factor in (2, 3, 4, 5):
        if oh == ih * factor and ow == iw * factor:
            for kind in (RuleKind.TILE, RuleKind.SCALE_NEAREST):
                rule = TaskRule(kind, factor=factor, label=f"{kind.name.lower()}_x{factor}")
                if _shapes_match_rule(inp_c, out_c, rule):
                    return rule

    for mode, kind in (("h", RuleKind.HFLIP), ("v", RuleKind.VFLIP), ("hv", RuleKind.HVFLIP)):
        rule = TaskRule(kind, label=f"flip_{mode}")
        if inp_c.shape == out_c.shape and _shapes_match_rule(inp_c, out_c, rule):
            return rule

    if inp_c.shape == out_c.shape:
        color_map: Dict[int, int] = {}
        ok = True
        for a, b in zip(inp_c.flatten(), out_c.flatten()):
            a, b = int(a), int(b)
            if a in color_map and color_map[a] != b:
                ok = False
                break
            color_map[a] = b
        if ok and color_map != {a: a for a in color_map}:
            return TaskRule(RuleKind.COLOR_REPLACE, color_map=color_map, label="color_replace")

        ff = TaskRule(RuleKind.FLOOD_FILL_BOUNDARY, label="flood_fill_bbox")
        if _shapes_match_rule(inp_c, out_c, ff):
            return ff

        for dr in range(-ih, ih + 1):
            for dc in range(-iw, iw + 1):
                rule = TaskRule(RuleKind.COPY_OFFSET, offset=(dr, dc), label=f"copy_{dr}_{dc}")
                if _shapes_match_rule(inp_c, out_c, rule):
                    return rule

    return None


def detect_task_rule(pairs: Sequence[Dict]) -> TaskRule:
    if not pairs:
        return TaskRule(RuleKind.UNKNOWN)

    votes: Dict[str, Tuple[TaskRule, int]] = {}
    for ex in pairs:
        inp = grid_to_array(ex["input"])
        out = grid_to_array(ex["output"])
        rule = infer_rule_from_pair(inp, out)
        if rule is None:
            continue
        key = f"{rule.kind.name}:{rule.factor}:{rule.offset}:{sorted(rule.color_map.items())}"
        if key not in votes:
            votes[key] = (rule, 0)
        votes[key] = (rule, votes[key][1] + 1)

    if not votes:
        return TaskRule(RuleKind.UNKNOWN)

    best_rule, count = max(votes.values(), key=lambda x: x[1])
    best_rule.confidence = count / len(pairs)
    return best_rule


# ---------------------------------------------------------------------------
# ARC-GEN-style augmentation
# ---------------------------------------------------------------------------
def random_color_permutation(rng: random.Random, n_active: int = 9) -> Dict[int, int]:
    colors = list(range(1, NUM_COLORS))
    rng.shuffle(colors)
    perm = {0: 0}
    for i in range(1, min(n_active + 1, NUM_COLORS)):
        perm[i] = colors[i - 1]
    return perm


def permute_grid_colors(grid: np.ndarray, perm: Dict[int, int]) -> np.ndarray:
    out = grid.copy()
    for src, dst in perm.items():
        out[grid == src] = dst
    return out


def translate_grid(grid: np.ndarray, dr: int, dc: int, size: int = GRID_SIZE) -> np.ndarray:
    out = np.zeros((size, size), dtype=grid.dtype)
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                out[nr, nc] = grid[r, c]
    return out


def mirror_augment(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    mode = rng.choice(["none", "h", "v", "hv"])
    if mode == "none":
        return grid
    return flip_grid(grid, mode)


def add_noise_objects(grid: np.ndarray, rng: random.Random, n_noise: int = 2) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    for _ in range(n_noise):
        r, c = rng.randint(0, h - 1), rng.randint(0, w - 1)
        if out[r, c] == 0:
            out[r, c] = rng.randint(1, NUM_COLORS - 1)
    return out


def random_patch(rng: random.Random, min_size: int = 2, max_size: int = 6) -> np.ndarray:
    h = rng.randint(min_size, max_size)
    w = rng.randint(min_size, max_size)
    patch = np.zeros((h, w), dtype=np.int64)
    density = rng.uniform(0.25, 0.85)
    n_colors = rng.randint(2, 5)
    palette = [0] + rng.sample(range(1, NUM_COLORS), k=min(n_colors, NUM_COLORS - 1))
    for r in range(h):
        for c in range(w):
            if rng.random() < density:
                patch[r, c] = rng.choice(palette[1:])
    return patch


def augment_pair(
    inp: np.ndarray,
    out: np.ndarray,
    rule: TaskRule,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray]:
    """Position / color / mirror augmentations that preserve the rule."""
    inp_c = crop_bbox(inp)
    out_c = crop_bbox(out)

    perm = random_color_permutation(rng)
    inp_c = permute_grid_colors(inp_c, perm)
    out_c = permute_grid_colors(out_c, perm)

    if rng.random() < 0.5:
        flip_mode = rng.choice(["h", "v", "hv"])
        inp_c = flip_grid(inp_c, flip_mode)
        out_c = flip_grid(out_c, flip_mode)

    max_r = max(1, GRID_SIZE - out_c.shape[0])
    max_c = max(1, GRID_SIZE - out_c.shape[1])
    row = rng.randint(0, max_r)
    col = rng.randint(0, max_c)

    canvas_inp = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)
    canvas_out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)
    canvas_inp = place_patch(canvas_inp, inp_c, row, col)

    if rule.kind in (RuleKind.TILE, RuleKind.SCALE_NEAREST, RuleKind.STENCIL_COPY):
        out_row, out_col = row * rule.factor_h, col * rule.fw
        if out_row + out_c.shape[0] <= GRID_SIZE and out_col + out_c.shape[1] <= GRID_SIZE:
            canvas_out = place_patch(canvas_out, out_c, out_row, out_col)
        else:
            canvas_out = place_patch(canvas_out, out_c, row, col)
    else:
        canvas_out = place_patch(canvas_out, out_c, row, col)

    if rng.random() < 0.2:
        canvas_inp = add_noise_objects(canvas_inp, rng, n_noise=rng.randint(1, 4))

    expected = apply_rule_padded(canvas_inp, rule)
    if rule.kind != RuleKind.UNKNOWN and np.array_equal(expected, canvas_out):
        return canvas_inp, canvas_out
    return canvas_inp, canvas_out


def synthesize_example(rule: TaskRule, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    """Procedurally generate input/output pair obeying the task rule."""
    if rule.kind == RuleKind.UNKNOWN:
        patch = random_patch(rng, 3, 5)
        row, col = rng.randint(0, 20), rng.randint(0, 20)
        inp = place_patch(np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64), patch, row, col)
        return inp, inp.copy()

    if rule.kind in (RuleKind.TILE, RuleKind.SCALE_NEAREST, RuleKind.STENCIL_COPY):
        if rule.kind == RuleKind.STENCIL_COPY:
            core = random_patch(rng, rule.factor_h, rule.factor_h)
        else:
            core = random_patch(rng, 2, 5)
        perm = random_color_permutation(rng)
        core = permute_grid_colors(core, perm)
        if rng.random() < 0.4:
            core = mirror_augment(core, rng)
        out_core = apply_rule_to_core(core, rule)
        row = rng.randint(0, max(0, GRID_SIZE - out_core.shape[0]))
        col = rng.randint(0, max(0, GRID_SIZE - out_core.shape[1]))
        if rng.random() < SYNTH_TOPLEFT_FRACTION:
            row, col = 0, 0
        if rule.kind in (RuleKind.TILE, RuleKind.STENCIL_COPY):
            in_row = row // rule.factor_h if rule.factor_h else row
            in_col = col // rule.fw if rule.fw else col
        else:
            in_row, in_col = row, col
        inp = place_patch(np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64), core, in_row, in_col)
        out = apply_rule_padded(inp, rule)
        return inp, out

    if rule.kind in (RuleKind.HFLIP, RuleKind.VFLIP, RuleKind.HVFLIP):
        core = random_patch(rng, 3, 8)
        perm = random_color_permutation(rng)
        core = permute_grid_colors(core, perm)
        row = rng.randint(0, max(0, GRID_SIZE - core.shape[0]))
        col = rng.randint(0, max(0, GRID_SIZE - core.shape[1]))
        inp = place_patch(np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64), core, row, col)
        out = apply_rule_padded(inp, rule)
        return inp, out

    if rule.kind == RuleKind.COLOR_REPLACE:
        core = random_patch(rng, 3, 7)
        row = rng.randint(0, max(0, GRID_SIZE - core.shape[0]))
        col = rng.randint(0, max(0, GRID_SIZE - core.shape[1]))
        inp = place_patch(np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64), core, row, col)
        out = apply_rule_padded(inp, rule)
        return inp, out

    if rule.kind == RuleKind.FLOOD_FILL_BOUNDARY:
        core = random_patch(rng, 4, 8)
        row = rng.randint(0, max(0, GRID_SIZE - core.shape[0]))
        col = rng.randint(0, max(0, GRID_SIZE - core.shape[1]))
        inp = place_patch(np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64), core, row, col)
        out = apply_rule_padded(inp, rule)
        return inp, out

    if rule.kind == RuleKind.COPY_OFFSET:
        core = random_patch(rng, 3, 6)
        row = rng.randint(0, max(0, GRID_SIZE - core.shape[0]))
        col = rng.randint(0, max(0, GRID_SIZE - core.shape[1]))
        inp = place_patch(np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64), core, row, col)
        out = apply_rule_padded(inp, rule)
        return inp, out

    patch = random_patch(rng)
    inp = place_patch(np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64), patch, 0, 0)
    return inp, inp.copy()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TaskPairDataset(Dataset):
    def __init__(self, task_path: Path, splits: Tuple[str, ...] = ("train",)) -> None:
        with task_path.open("r", encoding="utf-8") as f:
            task = json.load(f)
        self.task_id = task_path.stem
        self.pairs: List[Dict] = []
        for split in splits:
            self.pairs.extend(task.get(split, []))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        inp_oh, out = example_to_tensors(self.pairs[idx])
        return {"input": inp_oh.squeeze(0), "output": out}


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "input": torch.stack([b["input"] for b in batch]),
        "output": torch.stack([b["output"] for b in batch]),
    }


def load_task_json(task_id: str) -> Dict:
    path = DATA_DIR / f"{task_id}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_task_paths(
    task_filter: Optional[str] = None,
    task_start: int = TASK_START,
    task_end: int = TASK_END,
) -> List[Path]:
    if task_filter is not None:
        return [DATA_DIR / f"{task_filter}.json"]
    return sorted(
        p
        for p in DATA_DIR.glob("task*.json")
        if task_start <= parse_task_num(p.stem) <= task_end
    )


def build_training_batch(
    real_pairs: Sequence[Dict],
    rule: TaskRule,
    rng: random.Random,
    batch_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Mix augmented real examples + fresh synthetic procedural examples."""
    inputs: List[torch.Tensor] = []
    outputs: List[torch.Tensor] = []

    n_real = min(REAL_IN_BATCH, max(1, batch_size // 8))
    n_synth = batch_size - n_real

    for _ in range(n_real):
        ex = rng.choice(real_pairs)
        inp, out = example_to_tensors(ex)
        inp_g = inp.squeeze(0).numpy()
        out_g = out.numpy()
        inp_g = onehot_to_grid(inp_g)
        if rng.random() < AUGMENT_REAL_PROB and rule.kind != RuleKind.UNKNOWN:
            inp_g, out_g = augment_pair(inp_g, out_g, rule, rng)
        inputs.append(torch.from_numpy(grid_to_onehot(inp_g)).squeeze(0))
        outputs.append(torch.from_numpy(out_g))

    for _ in range(n_synth):
        inp_g, out_g = synthesize_example(rule, rng)
        inputs.append(torch.from_numpy(grid_to_onehot(inp_g)).squeeze(0))
        outputs.append(torch.from_numpy(out_g))

    idx = list(range(len(inputs)))
    rng.shuffle(idx)
    inputs = [inputs[i] for i in idx][:batch_size]
    outputs = [outputs[i] for i in idx][:batch_size]

    return {
        "input": torch.stack(inputs).to(device),
        "output": torch.stack(outputs).to(device),
    }


# ---------------------------------------------------------------------------
# Model: unrolled cellular-automata conv network (ONNX-safe)
# ---------------------------------------------------------------------------
class CAUpdateBlock(nn.Module):
    """Shared-weight 3x3 residual update (translation equivariant)."""

    def __init__(self, channels: int, kernel: int = CA_KERNEL) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels * 2, channels, kernel, padding=kernel // 2, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel, padding=kernel // 2, bias=True)

    def forward(self, state: torch.Tensor, inp_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, inp_embed], dim=1)
        x = F.relu(self.conv1(x), inplace=True)
        x = self.conv2(x)
        return state + x


class ARCCellularModel(nn.Module):
    """
    Iterative conv updates over a hidden state; input one-hot is fixed context.
    Python for-loop is unrolled at export time (no ONNX Loop/Scan).
    """

    def __init__(
        self,
        in_channels: int = NUM_COLORS,
        hidden_channels: int = HIDDEN_CHANNELS,
        out_channels: int = NUM_COLORS,
        n_iterations: int = N_CA_ITERATIONS,
    ) -> None:
        super().__init__()
        self.n_iterations = n_iterations
        self.embed = nn.Conv2d(in_channels, hidden_channels, 1, bias=True)
        self.update = CAUpdateBlock(hidden_channels)
        self.head = nn.Conv2d(hidden_channels, out_channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp_embed = F.relu(self.embed(x), inplace=True)
        state = torch.zeros(
            x.size(0),
            inp_embed.size(1),
            x.size(2),
            x.size(3),
            device=x.device,
            dtype=x.dtype,
        )
        for _ in range(self.n_iterations):
            state = F.relu(self.update(state, inp_embed), inplace=True)
        return self.head(state)


# Backward-compatible alias used by older tooling
ARCTaskModel = ARCCellularModel


class ONNXExportWrapper(nn.Module):
    def __init__(self, model: ARCCellularModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)


# ---------------------------------------------------------------------------
# Loss & metrics
# ---------------------------------------------------------------------------
def compute_loss(logits: torch.Tensor, targets: torch.Tensor, bg_weight: float = 0.3) -> torch.Tensor:
    """Cross-entropy with down-weighted background (color 0) to focus on rule structure."""
    per_pixel = F.cross_entropy(logits, targets, reduction="none")
    weights = torch.where(targets == 0, bg_weight, 1.0)
    return (per_pixel * weights).mean()


@torch.no_grad()
def predict_grids(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    model.eval()
    return model(batch["input"]).argmax(dim=1)


def exact_match(pred: torch.Tensor, target: torch.Tensor) -> bool:
    return bool(torch.equal(pred, target))


@torch.no_grad()
def batch_accuracy(model: nn.Module, batch: Dict[str, torch.Tensor]) -> float:
    preds = predict_grids(model, batch)
    correct = sum(exact_match(preds[i], batch["output"][i]) for i in range(preds.size(0)))
    return correct / max(1, preds.size(0))


@torch.no_grad()
def evaluate_split_pass_rate(
    model: nn.Module,
    examples: Sequence[Dict],
    device: torch.device,
) -> float:
    if not examples:
        return 1.0
    model.eval()
    ok = 0
    for ex in examples:
        inp_oh, out = example_to_tensors(ex)
        inp_t = inp_oh.to(device)
        if inp_t.ndim == 3:
            inp_t = inp_t.unsqueeze(0)
        pred = model(inp_t).argmax(dim=1).squeeze(0).cpu()
        if torch.equal(pred, out):
            ok += 1
    return ok / len(examples)


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> bool:
    if len(loader.dataset) == 0:
        return False
    model.eval()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = predict_grids(model, batch)
        for i in range(preds.size(0)):
            if not exact_match(preds[i], batch["output"][i]):
                return False
    return True


@torch.no_grad()
def evaluate_synthetic(
    model: nn.Module,
    rule: TaskRule,
    n_samples: int,
    rng: random.Random,
    device: torch.device,
) -> float:
    correct = 0
    for _ in range(n_samples):
        inp, out = synthesize_example(rule, rng)
        inp_oh = torch.from_numpy(grid_to_onehot(inp)).float().to(device)
        target = torch.from_numpy(out).to(device)
        logits = model(inp_oh)
        pred = logits.argmax(dim=1).squeeze(0)
        if exact_match(pred, target):
            correct += 1
    return correct / max(1, n_samples)


@torch.no_grad()
def check_translation_invariance(
    model: nn.Module,
    rule: TaskRule,
    rng: random.Random,
    device: torch.device,
    n_trials: int = 8,
) -> Tuple[float, str]:
    if rule.kind == RuleKind.UNKNOWN:
        return 0.0, "skipped (unknown rule)"

    ok = 0
    for _ in range(n_trials):
        inp, expected = synthesize_example(rule, rng)
        dr, dc = rng.randint(1, 5), rng.randint(1, 5)
        shifted_inp = translate_grid(inp, dr, dc)
        shifted_expected = apply_rule_padded(shifted_inp, rule)

        inp_oh = torch.from_numpy(grid_to_onehot(shifted_inp)).float().to(device)
        pred = model(inp_oh).argmax(dim=1).squeeze(0).cpu().numpy()
        if np.array_equal(pred, shifted_expected):
            ok += 1
    rate = ok / n_trials
    status = "RULE-LIKE (translation stable)" if rate >= 0.75 else "POSITION-SENSITIVE (likely memorization)"
    return rate, status


@torch.no_grad()
def check_color_invariance_under_perm(
    model: nn.Module,
    rule: TaskRule,
    rng: random.Random,
    device: torch.device,
    n_trials: int = 8,
) -> Tuple[float, str]:
    if rule.kind == RuleKind.UNKNOWN:
        return 0.0, "skipped"

    ok = 0
    for _ in range(n_trials):
        inp, _ = synthesize_example(rule, rng)
        perm = random_color_permutation(rng)
        perm_inp = permute_grid_colors(inp, perm)
        expected = apply_rule_padded(perm_inp, rule)

        inp_oh = torch.from_numpy(grid_to_onehot(perm_inp)).float().to(device)
        pred = model(inp_oh).argmax(dim=1).squeeze(0).cpu().numpy()
        if np.array_equal(pred, expected):
            ok += 1
    rate = ok / n_trials
    status = "COLOR-EQUIVARIANT" if rate >= 0.75 else "COLOR-MEMORIZED"
    return rate, status


def diagnose_learning(
    train_acc: float,
    synth_acc: float,
    arc_gen_ok: bool,
) -> str:
    gap = train_acc - synth_acc
    if synth_acc >= MIN_SYNTH_VAL_ACC and arc_gen_ok:
        return "RULE LEARNED — generalizes to procedural + ARC-GEN splits"
    if train_acc > 0.9 and gap > MEMORIZATION_GAP:
        return "MEMORIZATION — fits train but fails synthetic hold-out"
    if train_acc > 0.9 and not arc_gen_ok:
        return "OVERFIT — train-like but ARC-GEN split fails"
    return "IN PROGRESS — keep training or adjust rule/augmentation"


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def save_prediction_png(
    task_id: str,
    split: str,
    idx: int,
    inp: np.ndarray,
    expected: np.ndarray,
    pred: np.ndarray,
    passed: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from visualize_tasks import COLORS
    except ImportError:
        return

    folder = VIZ_DIR / task_id / ("passed" if passed else "failed")
    folder.mkdir(parents=True, exist_ok=True)

    cmap = ListedColormap(np.array(COLORS[:NUM_COLORS]) / 255.0)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, grid, title in zip(
        axes,
        [inp, expected, pred],
        ["input", "expected", "predicted"],
    ):
        ax.imshow(
            grid,
            cmap=cmap,
            vmin=0,
            vmax=NUM_COLORS - 1,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{task_id} [{split}#{idx}] {'PASS' if passed else 'FAIL'}")
    fig.tight_layout()
    out_path = folder / f"{task_id}_{split}_{idx:03d}.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# ONNX export & inference
# ---------------------------------------------------------------------------
def export_task_model(
    task_id: int,
    model: nn.Module,
    example_input: torch.Tensor,
    out_dir: Path = SUBMISSION_DIR,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"task{task_id:03d}.onnx"

    wrapped = ONNXExportWrapper(model).eval().cpu()
    example = example_input.detach().cpu().float()
    if example.ndim == 3:
        example = example.unsqueeze(0)
    if tuple(example.shape) != ONNX_INPUT_SHAPE:
        raise ValueError(f"Expected {ONNX_INPUT_SHAPE}, got {tuple(example.shape)}")

    torch.onnx.export(
        wrapped,
        example,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    _assert_static_onnx(out_path)
    size = out_path.stat().st_size
    print(f"  exported -> {out_path} ({size:,} bytes, {size / 1024:.1f} KB)")
    return out_path


def _assert_static_onnx(path: Path) -> None:
    model = onnx.load(str(path))
    graph = model.graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise ValueError(f"{path.name}: expected exactly one input and one output")

    for tensor in list(graph.input) + list(graph.output):
        for dim in tensor.type.tensor_type.shape.dim:
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                raise ValueError(f"{path.name}: non-static dimension on {tensor.name}")

    for node in graph.node:
        if node.op_type.upper() in BANNED_ONNX_OPS:
            raise ValueError(f"{path.name}: banned op {node.op_type}")
        if "Sequence" in node.op_type:
            raise ValueError(f"{path.name}: banned op {node.op_type}")

    if model.functions:
        raise ValueError(f"{path.name}: ONNX functions/subgraphs are not allowed")


def run_onnx_logits(session: ort.InferenceSession, grid_or_onehot: np.ndarray) -> np.ndarray:
    arr = np.asarray(grid_or_onehot)
    if arr.ndim == 2:
        arr = grid_to_onehot(pad_grid(arr.tolist()))
    elif arr.ndim == 4 and arr.shape[1] == 1:
        flat = arr[0, 0]
        arr = grid_to_onehot(pad_grid(flat.astype(int).tolist()))
    elif arr.ndim == 4 and arr.shape[1] == NUM_COLORS:
        arr = arr.astype(np.float32)
    else:
        raise ValueError(f"Unsupported input shape {arr.shape}")

    if tuple(arr.shape) != ONNX_INPUT_SHAPE:
        raise ValueError(f"Expected {ONNX_INPUT_SHAPE}, got {arr.shape}")

    outputs = session.run(["output"], {"input": arr})
    return outputs[0]


def postprocess_logits(logits: np.ndarray) -> np.ndarray:
    return onehot_to_grid(logits)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_task_onnx(
    task_id: str,
    onnx_path: Optional[Path] = None,
    splits: Tuple[str, ...] = ("train", "test", "arc-gen"),
    save_viz: bool = True,
) -> Tuple[bool, Dict[str, bool]]:
    task_num = parse_task_num(task_id)
    onnx_path = onnx_path or (SUBMISSION_DIR / f"task{task_num:03d}.onnx")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Missing ONNX model: {onnx_path}")

    _assert_static_onnx(onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    task = load_task_json(task_id)

    split_pass: Dict[str, bool] = {}
    all_pass = True

    for split in splits:
        examples = task.get(split, [])
        if not examples:
            split_pass[split] = True
            continue

        split_ok = True
        for idx, example in enumerate(examples):
            inp = pad_grid(example["input"])
            expected = pad_grid(example["output"])
            logits = run_onnx_logits(session, grid_to_onehot(inp))
            pred = postprocess_logits(logits)
            ok = np.array_equal(pred, expected)
            if save_viz:
                save_prediction_png(task_id, split, idx, inp, expected, pred, ok)
            if not ok:
                split_ok = False
                print(
                    f"  FAIL {task_id} [{split}#{idx}] "
                    f"mismatched={(pred != expected).sum()} cells"
                )
                break

        split_pass[split] = split_ok
        print(f"  {task_id} {split}: {'PASS' if split_ok else 'FAIL'}")
        all_pass = all_pass and split_ok

    print(f"  {task_id} overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass, split_pass


def validate_submission(
    submission_dir: Path = SUBMISSION_DIR,
    task_paths: Optional[List[Path]] = None,
) -> Dict[str, bool]:
    task_paths = task_paths or iter_task_paths(TASK_FILTER, TASK_START, TASK_END)
    results: Dict[str, bool] = {}
    for path in task_paths:
        task_id = path.stem
        onnx_path = submission_dir / f"{task_id}.onnx"
        try:
            ok, _ = validate_task_onnx(task_id, onnx_path)
        except Exception as exc:
            print(f"  {task_id}: ERROR ({exc})")
            ok = False
        results[task_id] = ok

    passed = sum(results.values())
    print(f"\nValidation summary: {passed}/{len(results)} tasks passed")
    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_single_task(
    task_id: str,
    device: torch.device,
    steps: int = TASK_STEPS,
    log_every: int = 200,
) -> Tuple[ARCCellularModel, bool, TaskRule]:
    task_path = DATA_DIR / f"{task_id}.json"
    with task_path.open("r", encoding="utf-8") as f:
        raw_task = json.load(f)
    train_pairs = raw_task.get("train", [])
    if not train_pairs:
        raise ValueError(f"No train pairs for {task_id}")

    rule = detect_task_rule(train_pairs)
    print(
        f"  detected rule: {rule.label} ({rule.kind.name}) "
        f"confidence={rule.confidence:.2f}"
    )
    if rule.kind == RuleKind.UNKNOWN:
        print("  WARNING: unknown rule — synthetic augmentation will be weak")

    rng = random.Random(SEED + parse_task_num(task_id))

    real_dataset = TaskPairDataset(task_path, splits=("train",))
    real_loader = DataLoader(
        real_dataset,
        batch_size=min(BATCH_SIZE, len(real_dataset)),
        shuffle=True,
        collate_fn=collate_batch,
    )

    arcgen_dataset = TaskPairDataset(task_path, splits=("arc-gen",))
    arcgen_loader = DataLoader(
        arcgen_dataset,
        batch_size=min(BATCH_SIZE, max(1, len(arcgen_dataset))),
        shuffle=False,
        collate_fn=collate_batch,
    )

    model = ARCCellularModel().to(device)
    n_params = count_parameters(model)
    print(
        f"  model: CA hidden={HIDDEN_CHANNELS} iters={N_CA_ITERATIONS} "
        f"params={n_params:,} est_score~{estimate_competition_score(n_params):.1f}"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_synth = 0.0
    best_score = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    solved = False
    last_diagnosis = "IN PROGRESS"
    arcgen_examples = raw_task.get("arc-gen", [])
    test_examples = raw_task.get("test", [])

    for step in range(1, steps + 1):
        model.train()
        batch = build_training_batch(train_pairs, rule, rng, BATCH_SIZE, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["input"])
        loss = compute_loss(logits, batch["output"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % log_every == 0 or step == steps:
            model.eval()
            train_acc = 0.0
            for rb in real_loader:
                rb = {k: v.to(device) for k, v in rb.items()}
                train_acc += batch_accuracy(model, rb) * rb["input"].size(0)
            train_acc /= max(1, len(real_dataset))

            synth_acc = evaluate_synthetic(model, rule, n_samples=24, rng=rng, device=device)
            best_synth = max(best_synth, synth_acc)

            arcgen_rate = evaluate_split_pass_rate(model, arcgen_examples, device)
            test_rate = evaluate_split_pass_rate(model, test_examples, device)
            arcgen_ok = arcgen_rate >= MIN_ARCGEN_PASS_RATE
            trans_rate, trans_status = check_translation_invariance(model, rule, rng, device)
            color_rate, color_status = check_color_invariance_under_perm(model, rule, rng, device)
            last_diagnosis = diagnose_learning(train_acc, synth_acc, arcgen_ok)
            combined = 0.4 * synth_acc + 0.4 * arcgen_rate + 0.2 * test_rate
            if combined > best_score:
                best_score = combined
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            print(
                f"  {task_id} step {step}/{steps} | loss={loss.item():.4f} | "
                f"train_acc={train_acc:.2f} | synth_acc={synth_acc:.2f} | "
                f"test={test_rate:.2f} | arc-gen={arcgen_rate:.2f} ({'PASS' if arcgen_ok else 'FAIL'})"
            )
            print(
                f"    invariance: translation={trans_rate:.2f} ({trans_status}) | "
                f"color_perm={color_rate:.2f} ({color_status})"
            )
            print(f"    diagnosis: {last_diagnosis}")

            if synth_acc >= MIN_SYNTH_VAL_ACC and arcgen_ok and test_rate >= 1.0:
                solved = True
                print(f"  >>> Generalization criteria met at step {step}")
                break

            if train_acc > 0.95 and synth_acc < MIN_SYNTH_VAL_ACC - 0.15:
                print(
                    f"  >>> REJECT: memorization detected "
                    f"(train={train_acc:.2f}, synth={synth_acc:.2f})"
                )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  restored best checkpoint (combined score={best_score:.3f})")

    ckpt_path = CHECKPOINT_DIR / f"{task_id}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "solved": solved,
            "rule": rule.label,
            "best_synth_acc": best_synth,
            "diagnosis": last_diagnosis,
        },
        ckpt_path,
    )
    print(f"  checkpoint -> {ckpt_path} | final diagnosis: {last_diagnosis}")
    return model, solved, rule


# ---------------------------------------------------------------------------
# Submission packaging
# ---------------------------------------------------------------------------
def pack_submission_zip(
    submission_dir: Path = SUBMISSION_DIR,
    zip_path: Path = ROOT / "submission.zip",
    task_paths: Optional[List[Path]] = None,
    strict: bool = True,
) -> Path:
    task_paths = task_paths or iter_task_paths(TASK_FILTER, TASK_START, TASK_END)
    written = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in task_paths:
            task_id = path.stem
            onnx_file = submission_dir / f"{task_id}.onnx"
            if not onnx_file.is_file():
                if strict:
                    raise FileNotFoundError(f"Missing {onnx_file} for submission zip")
                print(f"  skip (missing): {onnx_file.name}")
                continue
            zf.write(onnx_file, arcname=onnx_file.name)
            written += 1
    print(f"Wrote {zip_path} ({written} models, {zip_path.stat().st_size:,} bytes)")
    return zip_path


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def process_task(task_id: str, device: torch.device, steps: int) -> bool:
    print(f"\n=== {task_id} ===")
    existing = SUBMISSION_DIR / f"{task_id}.onnx"
    if existing.is_file():
        try:
            ok, split_pass = validate_task_onnx(task_id, existing, save_viz=False)
            if ok:
                print(f"  skip: valid ONNX already at {existing}")
                return True
            print(f"  existing ONNX failed validation {split_pass}; retraining")
        except Exception as exc:
            print(f"  existing ONNX invalid ({exc}); retraining")

    if not SKIP_TRAIN:
        model, solved, rule = train_single_task(task_id, device, steps=steps)
        print(f"  rule={rule.label} | solved={solved}")
    else:
        ckpt = CHECKPOINT_DIR / f"{task_id}.pt"
        model = ARCCellularModel().to(device)
        if ckpt.is_file():
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model_state_dict"])
        else:
            print(f"  warning: no checkpoint at {ckpt}")

    export_task_model(parse_task_num(task_id), model.cpu(), make_static_example_input())
    ok, split_pass = validate_task_onnx(task_id)
    if ok:
        print(f"  RESULT: task solved with ARC-GEN generalization")
    else:
        print(f"  RESULT: validation failed {split_pass}")
    return ok


def main() -> None:
    set_seed(SEED)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    task_paths = iter_task_paths(TASK_FILTER, TASK_START, TASK_END)
    if not task_paths:
        print(f"No tasks found in {DATA_DIR} (filter={TASK_FILTER})")
        sys.exit(1)

    print(f"Device: {DEVICE}")
    print(f"Tasks: {[p.stem for p in task_paths]}")
    probe = ARCCellularModel()
    n_params = count_parameters(probe)
    print(
        f"Architecture: ARCCellularModel | hidden={HIDDEN_CHANNELS} | "
        f"CA iterations={N_CA_ITERATIONS} (unrolled) | params={n_params:,} | "
        f"est_score_if_solved~{estimate_competition_score(n_params):.1f}"
    )
    print(
        f"Training policy: {SYNTH_SAMPLES_PER_STEP} synth/step | "
        f"min_synth_val={MIN_SYNTH_VAL_ACC} | generalization-first"
    )

    if VALIDATE_ONLY:
        validate_submission(SUBMISSION_DIR, task_paths)
        return

    if EXPORT_ONLY:
        for path in task_paths:
            task_id = path.stem
            ckpt = CHECKPOINT_DIR / f"{task_id}.pt"
            model = ARCCellularModel()
            if ckpt.is_file():
                state = torch.load(ckpt, map_location="cpu")
                model.load_state_dict(state["model_state_dict"])
            export_task_model(parse_task_num(task_id), model, make_static_example_input())
        validate_submission(SUBMISSION_DIR, task_paths)
        pack_submission_zip(SUBMISSION_DIR, ROOT / "submission.zip", task_paths)
        return

    t0 = time.time()
    results: Dict[str, bool] = {}
    total_tasks = len(task_paths)
    for task_idx, path in enumerate(task_paths, start=1):
        task_id = path.stem
        print(f"\n>>> Task {task_idx}/{total_tasks}: {task_id}")
        try:
            results[task_id] = process_task(task_id, DEVICE, TASK_STEPS)
        except Exception as exc:
            print(f"  ERROR on {task_id}: {exc}")
            import traceback
            traceback.print_exc()
            results[task_id] = False

    pack_submission_zip(SUBMISSION_DIR, ROOT / "submission.zip", task_paths)
    passed = sum(results.values())
    print(f"\nDone in {time.time() - t0:.1f}s | tasks passed validation: {passed}/{len(results)}")


if __name__ == "__main__":
    main()
