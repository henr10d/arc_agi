"""Connected-component object extraction from ARC grids.

Neighborhood: 4-connected (orthogonal only). Diagonal pixels of the same color
are treated as separate objects unless they share an edge. This matches common
ARC object semantics where corner-touching regions are distinct.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Grid = list[list[int]]

# 4-neighborhood offsets: up, down, left, right
_NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class GridObject:
    """One connected component (same color, 4-neighborhood)."""

    object_id: int
    mask: np.ndarray  # bool (H, W), True where object pixels lie
    min_x: int
    max_x: int
    min_y: int
    max_y: int
    colors: frozenset[int]
    centroid: tuple[float, float]  # (row, col)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.min_x, self.max_x, self.min_y, self.max_y)

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    @property
    def width(self) -> int:
        return self.max_x - self.min_x + 1

    @property
    def height(self) -> int:
        return self.max_y - self.min_y + 1

    def cropped_mask(self) -> np.ndarray:
        return self.mask[self.min_y : self.max_y + 1, self.min_x : self.max_x + 1]


def extract_objects(grid: Grid, *, background: int = 0) -> list[GridObject]:
    """Label connected components and return object records."""
    arr = np.asarray(grid, dtype=np.int32)
    height, width = arr.shape
    visited = np.zeros((height, width), dtype=bool)
    objects: list[GridObject] = []
    next_id = 0

    for row in range(height):
        for col in range(width):
            color = int(arr[row, col])
            if color == background or visited[row, col]:
                continue

            stack = [(row, col)]
            visited[row, col] = True
            pixels: list[tuple[int, int]] = []

            while stack:
                r, c = stack.pop()
                pixels.append((r, c))
                for dr, dc in _NEIGHBORS_4:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and not visited[nr, nc]
                        and int(arr[nr, nc]) == color
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            rows = [p[0] for p in pixels]
            cols = [p[1] for p in pixels]
            min_y, max_y = min(rows), max(rows)
            min_x, max_x = min(cols), max(cols)

            mask = np.zeros((height, width), dtype=bool)
            for r, c in pixels:
                mask[r, c] = True

            centroid_row = sum(rows) / len(rows)
            centroid_col = sum(cols) / len(cols)

            objects.append(
                GridObject(
                    object_id=next_id,
                    mask=mask,
                    min_x=min_x,
                    max_x=max_x,
                    min_y=min_y,
                    max_y=max_y,
                    colors=frozenset({color}),
                    centroid=(centroid_row, centroid_col),
                )
            )
            next_id += 1

    return objects
