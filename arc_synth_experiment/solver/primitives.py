"""Deterministic grid transformation primitives for ARC program synthesis."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

Grid = np.ndarray


@dataclass(frozen=True)
class Component:
    """A connected non-background component."""

    label: int
    cells: Tuple[Tuple[int, int], ...]
    colors: Tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.cells)


def _as_grid(grid: Union[Grid, Sequence[Sequence[int]]]) -> Grid:
    arr = np.asarray(grid, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grid, got shape {arr.shape}")
    return arr


def find_connected_components(
    grid: Union[Grid, Sequence[Sequence[int]]], background: int = 0
) -> List[Component]:
    """Return 4-connected components of non-background cells."""
    g = _as_grid(grid)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: List[Component] = []
    label = 0
    for r in range(h):
        for c in range(w):
            if seen[r, c] or g[r, c] == background:
                continue
            label += 1
            color = int(g[r, c])
            cells: List[Tuple[int, int]] = []
            q: deque[Tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] != background:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(
                Component(label=label, cells=tuple(cells), colors=tuple([color] * len(cells)))
            )
    return components


def largest_component(
    grid: Union[Grid, Sequence[Sequence[int]]], background: int = 0
) -> Optional[Component]:
    comps = find_connected_components(grid, background=background)
    if not comps:
        return None
    return max(comps, key=lambda c: c.size)


def bounding_box(component: Component) -> Tuple[int, int, int, int]:
    """Return (min_row, min_col, max_row_exclusive, max_col_exclusive)."""
    rows = [r for r, _ in component.cells]
    cols = [c for _, c in component.cells]
    return min(rows), min(cols), max(rows) + 1, max(cols) + 1


def mask_component(component: Component, shape: Tuple[int, int], color: Optional[int] = None) -> Grid:
    """Rasterize a component mask; optional solid color, else original colors."""
    h, w = shape
    out = np.zeros((h, w), dtype=np.int64)
    if color is None:
        for (r, c), col in zip(component.cells, component.colors):
            out[r, c] = col
    else:
        for r, c in component.cells:
            out[r, c] = color
    return out


def translate(grid: Union[Grid, Sequence[Sequence[int]]], dx: int, dy: int) -> Grid:
    g = _as_grid(grid)
    h, w = g.shape
    out = np.zeros_like(g)
    for r in range(h):
        for c in range(w):
            nr, nc = r + dx, c + dy
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = g[r, c]
    return out


def mirror_x(grid: Union[Grid, Sequence[Sequence[int]]]) -> Grid:
    return np.fliplr(_as_grid(grid))


def mirror_y(grid: Union[Grid, Sequence[Sequence[int]]]) -> Grid:
    return np.flipud(_as_grid(grid))


def rotate_90(grid: Union[Grid, Sequence[Sequence[int]]]) -> Grid:
    return np.rot90(_as_grid(grid), k=-1)


def crop_bbox(grid: Union[Grid, Sequence[Sequence[int]]], box: Tuple[int, int, int, int]) -> Grid:
    r0, c0, r1, c1 = box
    return _as_grid(grid)[r0:r1, c0:c1].copy()


def paste_at(
    canvas: Union[Grid, Sequence[Sequence[int]]],
    patch: Union[Grid, Sequence[Sequence[int]]],
    dx: int,
    dy: int,
) -> Grid:
    """Paste patch onto canvas at offset (dx, dy); non-zero patch cells overwrite."""
    out = _as_grid(canvas).copy()
    patch_arr = _as_grid(patch)
    ph, pw = patch_arr.shape
    ch, cw = out.shape
    for r in range(ph):
        for c in range(pw):
            val = int(patch_arr[r, c])
            if val == 0:
                continue
            nr, nc = r + dx, c + dy
            if 0 <= nr < ch and 0 <= nc < cw:
                out[nr, nc] = val
    return out


def paste_stencil(
    grid: Union[Grid, Sequence[Sequence[int]]], factor: int
) -> Grid:
    """ARC task001-style stencil: copy full input wherever input[r,c] != 0."""
    g = _as_grid(grid)
    h, w = g.shape
    out = np.zeros((h * factor, w * factor), dtype=g.dtype)
    for r in range(h):
        for c in range(w):
            if g[r, c] != 0:
                out = paste_at(out, g, r * factor, c * factor)
    return out


def tile_grid(grid: Union[Grid, Sequence[Sequence[int]]], factor: int) -> Grid:
    g = _as_grid(grid)
    block = np.ones((factor, factor), dtype=g.dtype)
    return np.kron(g, block)


def scale_nearest(grid: Union[Grid, Sequence[Sequence[int]]], factor: int) -> Grid:
    g = _as_grid(grid)
    h, w = g.shape
    out = np.zeros((h * factor, w * factor), dtype=g.dtype)
    for r in range(h):
        for c in range(w):
            out[r * factor : (r + 1) * factor, c * factor : (c + 1) * factor] = g[r, c]
    return out


def flood_fill_boundary(grid: Union[Grid, Sequence[Sequence[int]]], fill_color: int = 4) -> Grid:
    g = _as_grid(grid)
    active = np.argwhere(g != 0)
    if active.size == 0:
        return g.copy()
    r0, c0 = active.min(axis=0)
    r1, c1 = active.max(axis=0)
    out = g.copy()
    if r1 - r0 >= 2 and c1 - c0 >= 2:
        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color
    return out


def color_map(
    grid: Union[Grid, Sequence[Sequence[int]]], mapping: dict
) -> Grid:
    g = _as_grid(grid)
    out = g.copy()
    for old, new in mapping.items():
        out[g == int(old)] = int(new)
    return out


def fill_background(
    height: int, width: int, color: int = 0, dtype=np.int64
) -> Grid:
    return np.full((height, width), int(color), dtype=dtype)


def replace_color(
    grid: Union[Grid, Sequence[Sequence[int]]], old_color: int, new_color: int
) -> Grid:
    g = _as_grid(grid)
    out = g.copy()
    out[g == int(old_color)] = int(new_color)
    return out


Transform = Callable[[Grid], Grid]


def compose(f: Transform, g: Transform) -> Transform:
    def composed(x: Grid) -> Grid:
        return f(g(x))

    return composed


def conditional_if(condition: Grid, a: Transform, b: Transform) -> Transform:
    """Per-pixel choose a(x) or b(x) where condition != 0."""

    def wrapped(x: Grid) -> Grid:
        cond = _as_grid(condition)
        out_a = a(x)
        out_b = b(x)
        mask = cond != 0
        out = out_b.copy()
        out[mask] = out_a[mask]
        return out

    return wrapped


def cell_mask(grid: Grid, row: int, col: int) -> Grid:
    """Binary mask for one input cell being non-zero."""
    g = _as_grid(grid)
    mask = np.zeros_like(g)
    if 0 <= row < g.shape[0] and 0 <= col < g.shape[1] and g[row, col] != 0:
        mask[row, col] = 1
    return mask


def grids_equal(a: Grid, b: Grid) -> bool:
    aa = _as_grid(a)
    bb = _as_grid(b)
    return aa.shape == bb.shape and np.array_equal(aa, bb)
