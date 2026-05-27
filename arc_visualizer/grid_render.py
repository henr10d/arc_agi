"""Matplotlib grid rendering for ARC color grids."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

Grid = list[list[int]]

# Standard ARC palette (colors 0–9)
ARC_COLORS = [
    "#000000",  # 0 black
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 gray
    "#F012BE",  # 6 magenta
    "#FF851B",  # 7 orange
    "#7FDBFF",  # 8 cyan
    "#870C25",  # 9 maroon
]

CMAP = ListedColormap(ARC_COLORS)


def grid_to_array(grid: Grid) -> np.ndarray:
    return np.asarray(grid, dtype=np.int32)


def render_grid(
    ax: Axes,
    grid: Grid,
    *,
    title: str = "",
    cell_size: float = 1.0,
    gap: float = 0.08,
) -> None:
    """Draw a single ARC grid with clean cell spacing."""
    arr = grid_to_array(grid)
    height, width = arr.shape

    ax.set_xlim(-gap, width * cell_size + gap)
    ax.set_ylim(height * cell_size + gap, -gap)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11, pad=8)

    for row in range(height):
        for col in range(width):
            color_idx = int(arr[row, col]) % len(ARC_COLORS)
            rect = Rectangle(
                (col * cell_size, row * cell_size),
                cell_size * (1 - gap * 0.5),
                cell_size * (1 - gap * 0.5),
                facecolor=ARC_COLORS[color_idx],
                edgecolor="#333333",
                linewidth=0.6,
            )
            ax.add_patch(rect)


def render_grids_side_by_side(
    input_grid: Grid,
    output_grid: Grid,
    *,
    input_title: str = "Input",
    output_title: str = "Output",
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, Axes, Axes]:
    """Render input and output grids side-by-side with spacing between them."""
    in_arr = grid_to_array(input_grid)
    out_arr = grid_to_array(output_grid)
    in_h, in_w = in_arr.shape
    out_h, out_w = out_arr.shape
    spacer = 2.0

    if figsize is None:
        figsize = (max(6, (in_w + out_w) * 0.55 + 2), max(4, max(in_h, out_h) * 0.55))

    fig, (ax_in, ax_out) = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(wspace=0.35)

    render_grid(ax_in, input_grid, title=input_title)
    render_grid(ax_out, output_grid, title=output_title)

    return fig, ax_in, ax_out


def object_outline_colors(n: int) -> list[str]:
    """Distinct outline colors for up to n objects."""
    base = plt.cm.tab10(np.linspace(0, 1, max(n, 1)))
    return [plt.matplotlib.colors.rgb2hex(c[:3]) for c in base]
