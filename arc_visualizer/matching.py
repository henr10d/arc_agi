"""Heuristic object matching between input and output grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from objects import GridObject


@dataclass
class MatchScore:
    iou: float
    centroid_distance: float
    color_overlap: float
    total: float


@dataclass
class MatchResult:
    """input_object_id -> output_object_id (or None if unmatched)."""

    mapping: dict[int, int | None]
    scores: dict[tuple[int, int], MatchScore]
    warnings: list[str]


def _resize_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Nearest-neighbor resize of a boolean mask."""
    if mask.size == 0:
        return np.zeros((target_h, target_w), dtype=bool)
    src_h, src_w = mask.shape
    row_idx = (np.arange(target_h) * src_h / target_h).astype(int)
    col_idx = (np.arange(target_w) * src_w / target_w).astype(int)
    return mask[np.ix_(row_idx, col_idx)]


def shape_iou(a: GridObject, b: GridObject) -> float:
    """IoU of object shapes after cropping to bbox and normalizing size."""
    ma = a.cropped_mask()
    mb = b.cropped_mask()
    target_h = max(ma.shape[0], mb.shape[0], 1)
    target_w = max(ma.shape[1], mb.shape[1], 1)
    ra = _resize_mask(ma, target_h, target_w)
    rb = _resize_mask(mb, target_h, target_w)
    inter = np.logical_and(ra, rb).sum()
    union = np.logical_or(ra, rb).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def normalized_centroid_distance(
    a: GridObject,
    b: GridObject,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
) -> float:
    """Euclidean distance between centroids in normalized [0, 1] coordinates."""
    in_h, in_w = input_shape
    out_h, out_w = output_shape
    ar, ac = a.centroid
    br, bc = b.centroid
    norm_a = (ar / max(in_h - 1, 1), ac / max(in_w - 1, 1))
    norm_b = (br / max(out_h - 1, 1), bc / max(out_w - 1, 1))
    return float(np.hypot(norm_a[0] - norm_b[0], norm_a[1] - norm_b[1]))


def color_overlap(a: GridObject, b: GridObject) -> float:
    """Jaccard overlap of color sets."""
    inter = len(a.colors & b.colors)
    union = len(a.colors | b.colors)
    if union == 0:
        return 0.0
    return inter / union


def score_pair(
    inp: GridObject,
    out: GridObject,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
) -> MatchScore:
    iou = shape_iou(inp, out)
    dist = normalized_centroid_distance(inp, out, input_shape, output_shape)
    colors = color_overlap(inp, out)

    # Higher is better; centroid distance inverted (max ~sqrt(2) ≈ 1.41)
    dist_score = max(0.0, 1.0 - dist)
    total = 0.45 * iou + 0.30 * dist_score + 0.25 * colors

    return MatchScore(
        iou=iou,
        centroid_distance=dist,
        color_overlap=colors,
        total=total,
    )


def match_objects(
    input_objects: list[GridObject],
    output_objects: list[GridObject],
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    *,
    min_score: float = 0.05,
) -> MatchResult:
    """Greedy one-to-one matching: each input picks best available output."""
    scores: dict[tuple[int, int], MatchScore] = {}
    candidates: list[tuple[float, int, int]] = []

    for inp in input_objects:
        for out in output_objects:
            s = score_pair(inp, out, input_shape, output_shape)
            scores[(inp.object_id, out.object_id)] = s
            candidates.append((s.total, inp.object_id, out.object_id))

    candidates.sort(reverse=True)
    mapping: dict[int, int | None] = {obj.object_id: None for obj in input_objects}
    used_outputs: set[int] = set()
    warnings: list[str] = []

    # Track which outputs each input would prefer (for ambiguity warnings)
    best_per_input: dict[int, list[tuple[float, int]]] = {
        obj.object_id: [] for obj in input_objects
    }
    for inp in input_objects:
        ranked = sorted(
            ((scores[(inp.object_id, out.object_id)].total, out.object_id) for out in output_objects),
            reverse=True,
        )
        best_per_input[inp.object_id] = ranked[:3]

    for total, inp_id, out_id in candidates:
        if mapping[inp_id] is not None:
            continue
        if out_id in used_outputs:
            continue
        if total < min_score:
            continue
        mapping[inp_id] = out_id
        used_outputs.add(out_id)

    for inp_id, ranked in best_per_input.items():
        if len(ranked) < 2:
            continue
        top_score, top_out = ranked[0]
        second_score, second_out = ranked[1]
        if top_score - second_score < 0.08 and mapping.get(inp_id) == top_out:
            warnings.append(
                f"Input object {inp_id}: ambiguous match "
                f"(out {top_out} score={top_score:.3f} vs out {second_out} score={second_score:.3f})"
            )

    output_claims: dict[int, list[int]] = {}
    for inp_id, out_id in mapping.items():
        if out_id is not None:
            output_claims.setdefault(out_id, []).append(inp_id)

    for out_id, inp_ids in output_claims.items():
        if len(inp_ids) > 1:
            warnings.append(
                f"Output object {out_id} matched by multiple inputs: {inp_ids}"
            )

    return MatchResult(mapping=mapping, scores=scores, warnings=warnings)
