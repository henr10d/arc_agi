#!/usr/bin/env python3
"""Visualize NeuroGolf / ARC task grids from JSON files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Same palette as data/neurogolf_utils/neurogolf_utils.py
COLORS = [
    (0, 0, 0),
    (30, 147, 255),
    (250, 61, 49),
    (78, 204, 48),
    (255, 221, 0),
    (153, 153, 153),
    (229, 59, 163),
    (255, 133, 28),
    (136, 216, 241),
    (147, 17, 49),
    (240, 240, 240),
    (146, 117, 86),
]

SPLITS = ("train", "test", "arc-gen")
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def parse_task_id(raw: str) -> str:
    raw = raw.strip().lower()
    if raw.isdigit():
        return f"task{int(raw):03d}"
    match = re.fullmatch(r"task(\d+)", raw)
    if match:
        return f"task{int(match.group(1)):03d}"
    raise argparse.ArgumentTypeError(f"Invalid task id: {raw!r} (use 1, 001, or task001)")


def load_task(data_dir: Path, task_id: str) -> dict:
    path = data_dir / f"{task_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Task file not found: {path}")
    with path.open() as f:
        return json.load(f)


def list_tasks(data_dir: Path) -> list[str]:
    return sorted(p.stem for p in data_dir.glob("task*.json"))


def render_examples(
    examples: list[dict],
    *,
    bgcolor: tuple[int, int, int] = (255, 255, 255),
    title: str | None = None,
):
    """Draw input -> output pairs side by side, matching competition styling."""
    if not examples:
        raise ValueError("No examples to render")

    width, height, offset = 0, 0, 1
    for example in examples:
        grid, output = example["input"], example["output"]
        width += len(grid[0]) + 1 + len(output[0]) + 4
        height = max(height, max(len(grid), len(output)) + 4)

    image = [[bgcolor for _ in range(width)] for _ in range(height)]
    offsets: list[tuple[int, int, int, int, int, int]] = []

    offset = 1
    for example in examples:
        grid, output = example["input"], example["output"]
        grid_width, grid_height = len(grid[0]), len(grid)
        output_width, output_height = len(output[0]), len(output)

        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                image[r + 2][offset + c + 1] = COLORS[cell]

        input_offset = offset
        offset += grid_width + 1

        for r, row in enumerate(output):
            for c, cell in enumerate(row):
                image[r + 2][offset + c + 1] = COLORS[cell]

        output_offset = offset
        offsets.append(
            (input_offset, grid_width, grid_height, output_offset, output_width, output_height)
        )
        offset += output_width + 4

    fig = plt.figure(figsize=(max(10, width * 0.35), max(5, height * 0.45)))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.array(image))

    for input_offset, grid_width, grid_height, output_offset, output_width, output_height in offsets:
        ax.hlines(
            [r + 1.5 for r in range(grid_height + 1)],
            xmin=input_offset + 0.5,
            xmax=input_offset + grid_width + 0.5,
            color="black",
        )
        ax.vlines(
            [input_offset + c + 0.5 for c in range(grid_width + 1)],
            ymin=1.5,
            ymax=grid_height + 1.5,
            color="black",
        )
        ax.hlines(
            [r + 1.5 for r in range(output_height + 1)],
            xmin=output_offset + 0.5,
            xmax=output_offset + output_width + 0.5,
            color="black",
        )
        ax.vlines(
            [output_offset + c + 0.5 for c in range(output_width + 1)],
            ymin=1.5,
            ymax=output_height + 1.5,
            color="black",
        )
        sep = output_offset + output_width + 2
        ax.vlines([sep + 0.5], ymin=-0.5, ymax=height - 0.5, color="black")

    if title:
        ax.set_title(title, fontsize=12, pad=8)

    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def render_legend():
    image = [[(255, 255, 255) for _ in range(21)] for _ in range(5)]
    for idx, color in enumerate(COLORS[:10]):
        image[1][2 * idx + 1] = color
    for idx, color in enumerate(COLORS[10:]):
        for col in range(3):
            image[3][12 * idx + col + 3] = color

    fig = plt.figure(figsize=(10, 2.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.array(image))
    for idx, _ in enumerate(COLORS[:10]):
        color = "white" if idx in (0, 9) else "black"
        ax.text(2 * idx + 0.9, 1.1, str(idx), color=color)
    ax.text(3.4, 3.1, "no color", color="black")
    ax.text(5.75, 3.1, "<--- special colors to indicate one-hot encoding errors --->", color="black")
    ax.text(14.85, 3.1, "too many colors", color="white")
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def summarize_task(task_id: str, task: dict) -> str:
    parts = []
    for split in SPLITS:
        count = len(task.get(split, []))
        if count:
            parts.append(f"{split}={count}")
    return f"{task_id}: " + ", ".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize NeuroGolf ARC tasks (input -> output grids)."
    )
    parser.add_argument(
        "tasks",
        nargs="*",
        type=parse_task_id,
        help="Task ids like 1, 42, task001. Omit to list available tasks.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing task*.json (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--split",
        choices=SPLITS,
        action="append",
        dest="splits",
        help="Split to show (repeatable). Default: train, test, arc-gen",
    )
    parser.add_argument(
        "--example",
        type=int,
        default=None,
        help="Show only this example index within each selected split",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save PNG(s) to this file or directory instead of showing a window",
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        help="Also show the color legend",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available tasks and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    data_dir = args.data_dir.expanduser().resolve()

    if not data_dir.is_dir():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        return 1

    if args.list or not args.tasks:
        tasks = list_tasks(data_dir)
        print(f"Found {len(tasks)} tasks in {data_dir}")
        for task_id in tasks:
            try:
                task = load_task(data_dir, task_id)
                print(" ", summarize_task(task_id, task))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"  {task_id}: error loading ({exc})")
        if not args.tasks:
            print("\nUsage examples:")
            print("  python visualize_tasks.py 1")
            print("  python visualize_tasks.py task001 task042 --split train")
            print("  python visualize_tasks.py 1 --save out/task001.png")
        return 0

    splits = args.splits or list(SPLITS)
    figures: list[tuple[str, plt.Figure]] = []

    for task_id in args.tasks:
        task = load_task(data_dir, task_id)
        for split in splits:
            examples = task.get(split, [])
            if args.example is not None:
                if args.example < 0 or args.example >= len(examples):
                    print(
                        f"Skipping {task_id} {split}: example {args.example} "
                        f"out of range (0..{max(len(examples) - 1, 0)})",
                        file=sys.stderr,
                    )
                    continue
                examples = [examples[args.example]]
            if not examples:
                print(f"No examples for {task_id} split={split}", file=sys.stderr)
                continue

            title = f"{task_id} [{split}]"
            if args.example is not None:
                title += f" example {args.example}"
            fig = render_examples(examples, title=title)
            figures.append((f"{task_id}_{split}.png", fig))

    if args.legend:
        figures.append(("legend.png", render_legend()))

    if not figures:
        print("Nothing to display.", file=sys.stderr)
        return 1

    save_target = args.save
    if save_target is None:
        plt.show()
        return 0

    save_target = save_target.expanduser().resolve()
    if len(figures) == 1 and save_target.suffix.lower() == ".png":
        figures[0][1].savefig(save_target, bbox_inches="tight", dpi=150)
        print(f"Saved {save_target}")
        return 0

    save_target.mkdir(parents=True, exist_ok=True)
    for filename, fig in figures:
        out_path = save_target / filename
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
