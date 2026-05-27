"""Interactive ARC object visualization UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.widgets import CheckButtons

from grid_render import object_outline_colors, render_grid
from loader import ArcTask, GridPair, load_task_json, list_task_ids, parse_task_id
from matching import MatchResult, match_objects
from objects import GridObject, extract_objects

Grid = list[list[int]]

SPLITS = ("train", "test", "arc-gen")


@dataclass
class ViewOptions:
    show_raw: bool = True
    show_masks: bool = True
    show_bboxes_only: bool = False


class ArcVisualizerUI:
    """Matplotlib UI: input/output overlays + matching arrows + toggles."""

    def __init__(
        self,
        *,
        title: str = "ARC Object Visualizer",
        status: str = "",
        on_key: Callable | None = None,
    ) -> None:
        self.options = ViewOptions()
        self.cell = 1.0
        self.gap = 2.5

        self.input_grid: Grid = [[0]]
        self.output_grid: Grid = [[0]]
        self.input_objects: list[GridObject] = []
        self.output_objects: list[GridObject] = []
        self.match_result = MatchResult(mapping={}, scores={}, warnings=[])
        self.outline_colors: list[str] = object_outline_colors(1)

        self.in_h, self.in_w = 1, 1
        self.out_h, self.out_w = 1, 1

        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle(title, fontsize=13, fontweight="bold")

        self.ax_in = self.fig.add_axes([0.05, 0.44, 0.40, 0.48])
        self.ax_out = self.fig.add_axes([0.55, 0.44, 0.40, 0.48])
        self.ax_match = self.fig.add_axes([0.05, 0.14, 0.90, 0.26])
        self.ax_toggle = self.fig.add_axes([0.05, 0.07, 0.35, 0.05])
        self.ax_help = self.fig.add_axes([0.42, 0.02, 0.56, 0.10])
        self.ax_help.axis("off")

        labels = ["Raw grid", "Object masks", "BBoxes only"]
        self.check = CheckButtons(
            self.ax_toggle,
            labels,
            [self.options.show_raw, self.options.show_masks, self.options.show_bboxes_only],
        )
        self.check.on_clicked(self._on_toggle)

        self._status_text = self.ax_help.text(
            0.0,
            0.95,
            status,
            transform=self.ax_help.transAxes,
            fontsize=9,
            verticalalignment="top",
            family="monospace",
        )

        if on_key is not None:
            self.fig.canvas.mpl_connect("key_press_event", on_key)

    def set_status(self, status: str) -> None:
        self._status_text.set_text(status)

    def set_title(self, title: str) -> None:
        self.fig.suptitle(title, fontsize=13, fontweight="bold")

    def update_data(
        self,
        input_grid: Grid,
        output_grid: Grid,
        input_objects: list[GridObject],
        output_objects: list[GridObject],
        match_result: MatchResult,
    ) -> None:
        self.input_grid = input_grid
        self.output_grid = output_grid
        self.input_objects = input_objects
        self.output_objects = output_objects
        self.match_result = match_result

        self.in_h, self.in_w = len(input_grid), len(input_grid[0])
        self.out_h, self.out_w = len(output_grid), len(output_grid[0])
        self.outline_colors = object_outline_colors(
            max(len(input_objects), len(output_objects), 1)
        )
        self._redraw()

    def _on_toggle(self, label: str) -> None:
        if label == "Raw grid":
            self.options.show_raw = not self.options.show_raw
        elif label == "Object masks":
            self.options.show_masks = not self.options.show_masks
        elif label == "BBoxes only":
            self.options.show_bboxes_only = not self.options.show_bboxes_only
        self._redraw()

    def _redraw(self) -> None:
        for ax in (self.ax_in, self.ax_out, self.ax_match):
            ax.clear()

        self._draw_panel(
            self.ax_in,
            self.input_grid,
            self.input_objects,
            "Input",
            side="input",
        )
        self._draw_panel(
            self.ax_out,
            self.output_grid,
            self.output_objects,
            "Output",
            side="output",
        )
        self._draw_matching_view()
        self.fig.canvas.draw_idle()

    def _draw_panel(
        self,
        ax,
        grid: Grid,
        objects: list[GridObject],
        title: str,
        *,
        side: str,
    ) -> None:
        if self.options.show_raw:
            render_grid(ax, grid, title=title, cell_size=self.cell)
        else:
            height, width = len(grid), len(grid[0])
            ax.set_xlim(-0.1, width)
            ax.set_ylim(height, -0.1)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=11, pad=8)
            ax.set_facecolor("#f5f5f5")

        matched_ids = set(self.match_result.mapping.values())
        matched_ids.discard(None)

        for obj in objects:
            color = self.outline_colors[obj.object_id % len(self.outline_colors)]
            is_unmatched = self._is_unmatched(obj, side, matched_ids)
            edge_color = "#CC0000" if is_unmatched else color
            lw = 2.8 if is_unmatched else 2.0

            if self.options.show_bboxes_only or not self.options.show_masks:
                rect = Rectangle(
                    (obj.min_x * self.cell, obj.min_y * self.cell),
                    obj.width * self.cell * 0.98,
                    obj.height * self.cell * 0.98,
                    fill=False,
                    edgecolor=edge_color,
                    linewidth=lw,
                    linestyle="-" if not is_unmatched else "--",
                )
                ax.add_patch(rect)
            else:
                rows, cols = np.where(obj.mask)
                for r, c in zip(rows, cols, strict=False):
                    patch = Rectangle(
                        (c * self.cell, r * self.cell),
                        self.cell * 0.92,
                        self.cell * 0.92,
                        facecolor=(*self._hex_to_rgb01(color), 0.35),
                        edgecolor=edge_color,
                        linewidth=0.8,
                    )
                    ax.add_patch(patch)

            cr, cc = obj.centroid
            ax.text(
                cc * self.cell + 0.15,
                cr * self.cell + 0.15,
                str(obj.object_id),
                color="white" if self.options.show_raw else "black",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=edge_color, alpha=0.85),
            )

    def _is_unmatched(
        self, obj: GridObject, side: str, matched_output_ids: set[int]
    ) -> bool:
        if side == "input":
            return self.match_result.mapping.get(obj.object_id) is None
        mapped_from = [
            i for i, o in self.match_result.mapping.items() if o == obj.object_id
        ]
        return len(mapped_from) == 0

    @staticmethod
    def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16) / 255
        g = int(hex_color[2:4], 16) / 255
        b = int(hex_color[4:6], 16) / 255
        return (r, g, b)

    def _draw_matching_view(self) -> None:
        ax = self.ax_match
        ax.set_title("Matching (input → output)", fontsize=11, pad=6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        in_offset_x = 0.0
        out_offset_x = self.in_w * self.cell + self.gap
        total_w = out_offset_x + self.out_w * self.cell
        max_h = max(self.in_h, self.out_h) * self.cell

        ax.set_xlim(-0.5, total_w + 0.5)
        ax.set_ylim(max_h + 0.5, -0.5)
        ax.set_facecolor("#fafafa")

        ax.add_patch(
            Rectangle(
                (in_offset_x, 0),
                self.in_w * self.cell,
                self.in_h * self.cell,
                facecolor="#eeeeee",
                edgecolor="#999999",
                linewidth=1,
            )
        )
        ax.add_patch(
            Rectangle(
                (out_offset_x, 0),
                self.out_w * self.cell,
                self.out_h * self.cell,
                facecolor="#eeeeee",
                edgecolor="#999999",
                linewidth=1,
            )
        )
        ax.text(in_offset_x + 0.2, -0.4, "Input", fontsize=9, color="#444")
        ax.text(out_offset_x + 0.2, -0.4, "Output", fontsize=9, color="#444")

        obj_by_in = {o.object_id: o for o in self.input_objects}
        obj_by_out = {o.object_id: o for o in self.output_objects}

        for inp_id, out_id in self.match_result.mapping.items():
            inp = obj_by_in[inp_id]
            icr, icc = inp.centroid
            x1 = in_offset_x + icc * self.cell + self.cell * 0.5
            y1 = icr * self.cell + self.cell * 0.5

            if out_id is None:
                ax.plot(x1, y1, "x", color="#CC0000", markersize=12, markeredgewidth=2)
                ax.text(x1 + 0.2, y1, f"#{inp_id} → ∅", fontsize=8, color="#CC0000")
                continue

            out = obj_by_out[out_id]
            ocr, occ = out.centroid
            x2 = out_offset_x + occ * self.cell + self.cell * 0.5
            y2 = ocr * self.cell + self.cell * 0.5

            color = self.outline_colors[inp_id % len(self.outline_colors)]
            arrow = FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.8,
                color=color,
                alpha=0.85,
                connectionstyle="arc3,rad=0.08",
            )
            ax.add_patch(arrow)

            score = self.match_result.scores.get((inp_id, out_id))
            label = f"{inp_id}→{out_id}"
            if score:
                label += f" ({score.total:.2f})"
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y - 0.25, label, fontsize=7, ha="center", color="#333")

        matched_out = {v for v in self.match_result.mapping.values() if v is not None}
        for out in self.output_objects:
            if out.object_id not in matched_out:
                ocr, occ = out.centroid
                x = out_offset_x + occ * self.cell + self.cell * 0.5
                y = ocr * self.cell + self.cell * 0.5
                ax.plot(x, y, "x", color="#CC0000", markersize=12, markeredgewidth=2)
                ax.text(x + 0.2, y, f"out #{out.object_id} (new?)", fontsize=8, color="#CC0000")

    def show(self) -> None:
        plt.show()


class ArcExplorerApp:
    """Browse tasks and examples with keyboard shortcuts."""

    HELP = (
        "←/→ or a/d : prev/next example\n"
        "[/]        : prev/next task\n"
        "↑/↓        : prev/next split (train/test/arc-gen)\n"
        "Click window first so keys register"
    )

    def __init__(
        self,
        dataset_dir: Path,
        task_id: str,
        *,
        split: str = "train",
        example_index: int = 0,
        on_example_change: Callable[..., None] | None = None,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.task_ids = list_task_ids(dataset_dir)
        if not self.task_ids:
            raise FileNotFoundError(f"No task*.json files in {dataset_dir}")

        self.task_id = parse_task_id(task_id)
        if self.task_id not in self.task_ids:
            raise FileNotFoundError(
                f"Task {self.task_id} not found in {dataset_dir}"
            )

        self.split = split if split in SPLITS else "train"
        self.example_index = example_index
        self.on_example_change = on_example_change

        self.task: ArcTask | None = None
        self.ui = ArcVisualizerUI(
            title="Loading…",
            status=self.HELP,
            on_key=self._on_key,
        )
        self._load_task(self.task_id)
        self._apply_current_example(print_debug=True)

    def _pairs(self) -> list[GridPair]:
        assert self.task is not None
        return {
            "train": self.task.train,
            "test": self.task.test,
            "arc-gen": self.task.arc_gen,
        }[self.split]

    def _load_task(self, task_id: str) -> None:
        path = self.dataset_dir / f"{task_id}.json"
        self.task = load_task_json(path)
        self.task_id = task_id
        pairs = self._pairs()
        if not pairs:
            self.example_index = 0
        else:
            self.example_index = min(self.example_index, len(pairs) - 1)

    def _status_line(self) -> str:
        pairs = self._pairs()
        n = len(pairs)
        idx = self.example_index + 1 if n else 0
        task_pos = self.task_ids.index(self.task_id) + 1
        n_tasks = len(self.task_ids)
        return (
            f"{self.task_id}  ({task_pos}/{n_tasks})  "
            f"split={self.split}  example {idx}/{n}  "
            f"in={self.ui.in_h}x{self.ui.in_w}  out={self.ui.out_h}x{self.ui.out_w}"
        )

    def _apply_current_example(self, *, print_debug: bool = False) -> None:
        pairs = self._pairs()
        if not pairs:
            print(f"No examples in split {self.split!r} for {self.task_id}")
            return

        pair = pairs[self.example_index]
        input_objects = extract_objects(pair.input)
        output_objects = extract_objects(pair.output)
        in_shape = (len(pair.input), len(pair.input[0]))
        out_shape = (len(pair.output), len(pair.output[0]))
        match_result = match_objects(
            input_objects, output_objects, in_shape, out_shape
        )

        title = f"{self.task_id} | {self.split}[{self.example_index}]"
        self.ui.set_title(title)
        self.ui.update_data(
            pair.input,
            pair.output,
            input_objects,
            output_objects,
            match_result,
        )
        self.ui.set_status(self._status_line() + "\n\n" + self.HELP)

        if print_debug and self.on_example_change:
            self.on_example_change(
                task_id=self.task_id,
                split=self.split,
                example_index=self.example_index,
                pair=pair,
                input_objects=input_objects,
                output_objects=output_objects,
                match_result=match_result,
            )

    def _on_key(self, event) -> None:
        key = event.key
        if key in ("right", "d"):
            pairs = self._pairs()
            if pairs:
                self.example_index = (self.example_index + 1) % len(pairs)
                self._apply_current_example(print_debug=True)
        elif key in ("left", "a"):
            pairs = self._pairs()
            if pairs:
                self.example_index = (self.example_index - 1) % len(pairs)
                self._apply_current_example(print_debug=True)
        elif key == "]":
            i = self.task_ids.index(self.task_id)
            self.task_id = self.task_ids[(i + 1) % len(self.task_ids)]
            self._load_task(self.task_id)
            self._apply_current_example(print_debug=True)
        elif key == "[":
            i = self.task_ids.index(self.task_id)
            self.task_id = self.task_ids[(i - 1) % len(self.task_ids)]
            self._load_task(self.task_id)
            self._apply_current_example(print_debug=True)
        elif key == "up":
            si = SPLITS.index(self.split)
            self.split = SPLITS[(si - 1) % len(SPLITS)]
            self.example_index = 0
            self._apply_current_example(print_debug=True)
        elif key == "down":
            si = SPLITS.index(self.split)
            self.split = SPLITS[(si + 1) % len(SPLITS)]
            self.example_index = 0
            self._apply_current_example(print_debug=True)

    def show(self) -> None:
        self.ui.show()


def open_explorer(
    dataset_dir: Path,
    task_id: str,
    *,
    split: str = "train",
    example_index: int = 0,
    on_example_change: Callable[..., None] | None = None,
) -> ArcExplorerApp:
    return ArcExplorerApp(
        dataset_dir,
        task_id,
        split=split,
        example_index=example_index,
        on_example_change=on_example_change,
    )
