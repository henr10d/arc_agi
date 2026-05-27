#!/usr/bin/env python3
"""Visualize ONNX predictions vs ground truth for one ARC task."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

from train_arc import (
    SUBMISSION_DIR,
    grid_to_onehot,
    load_task_json,
    pad_grid,
    parse_task_num,
    postprocess_logits,
    run_onnx_logits,
)
from visualize_tasks import COLORS

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "test", "arc-gen")
MAX_TASK = 400


def parse_task_num_arg(raw: str) -> int:
    raw = raw.strip().lower()
    if raw.isdigit():
        num = int(raw)
    else:
        match = re.fullmatch(r"task(\d+)", raw)
        if not match:
            raise argparse.ArgumentTypeError(
                f"Invalid task: {raw!r} (use a number 1-{MAX_TASK}, e.g. 2 or task002)"
            )
        num = int(match.group(1))

    if not 1 <= num <= MAX_TASK:
        raise argparse.ArgumentTypeError(f"Task must be between 1 and {MAX_TASK}, got {num}")
    return num


def task_id_from_num(task_num: int) -> str:
    return f"task{task_num:03d}"


def grid_bounds(grid: np.ndarray) -> tuple[int, int]:
    """Last row/col index (inclusive) with any non-zero cell; at least 0."""
    active = np.argwhere(grid != 0)
    if active.size == 0:
        return 0, 0
    return int(active[:, 0].max()), int(active[:, 1].max())


def crop_for_display(*grids: np.ndarray) -> list[np.ndarray]:
    """Crop padded grids to the union of their non-zero content."""
    max_r = max(c for r, _ in (grid_bounds(g) for g in grids) for c in [r])
    max_c = max(c for _, c in (grid_bounds(g) for g in grids) for c in [c])
    return [g[: max_r + 1, : max_c + 1].copy() for g in grids]


def grid_to_rgb(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            color_idx = int(grid[r, c])
            if 0 <= color_idx < len(COLORS):
                image[r, c] = COLORS[color_idx]
    return image


def render_comparison(
    inp: np.ndarray,
    predicted: np.ndarray,
    expected: np.ndarray,
    *,
    title: str,
    match: bool,
) -> plt.Figure:
    """Draw input | predicted | expected side by side."""
    inp, predicted, expected = crop_for_display(inp, predicted, expected)
    panels = [
        ("input", inp),
        ("predicted", predicted),
        ("expected", expected),
    ]

    gap = 2
    total_w = sum(g.shape[1] for _, g in panels) + gap * (len(panels) - 1) + 4
    max_h = max(g.shape[0] for _, g in panels) + 4

    image = np.full((max_h, total_w, 3), 255, dtype=np.uint8)
    offsets: list[tuple[str, int, np.ndarray, bool]] = []

    x = 2
    for label, grid in panels:
        rgb = grid_to_rgb(grid)
        h, w = rgb.shape[:2]
        image[2 : 2 + h, x : x + w] = rgb
        highlight = label == "predicted" and not match
        offsets.append((label, x, grid, highlight))
        x += w + gap

    status = "MATCH" if match else "MISMATCH"
    fig = plt.figure(figsize=(max(12, total_w * 0.35), max(4, max_h * 0.45)))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)

    for label, x_off, grid, highlight in offsets:
        h, w = grid.shape
        color = "limegreen" if label == "predicted" and match else "red" if highlight else "black"
        ax.hlines(
            [r + 1.5 for r in range(h + 1)],
            xmin=x_off - 0.5,
            xmax=x_off + w - 0.5,
            color=color,
            linewidth=1.5 if highlight else 1.0,
        )
        ax.vlines(
            [x_off + c - 0.5 for c in range(w + 1)],
            ymin=1.5,
            ymax=h + 1.5,
            color=color,
            linewidth=1.5 if highlight else 1.0,
        )
        ax.text(
            x_off + w / 2 - 0.5,
            0.3,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold" if label == "predicted" else "normal",
        )

    ax.set_title(f"{title} — {status}", fontsize=12, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def predict_example(session: ort.InferenceSession, example: dict) -> tuple[np.ndarray, np.ndarray]:
    inp_oh = grid_to_onehot(pad_grid(example["input"]))
    expected = pad_grid(example["output"])
    logits = run_onnx_logits(session, inp_oh)
    predicted = postprocess_logits(logits)
    return predicted, expected


def iter_task_examples(task: dict) -> list[tuple[str, int, dict]]:
    """All input/output pairs from train, test, and arc-gen splits."""
    items: list[tuple[str, int, dict]] = []
    for split in SPLITS:
        for idx, example in enumerate(task.get(split, [])):
            items.append((split, idx, example))
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one task's ONNX model on every example in its JSON "
            f"(train + test + arc-gen) and save comparisons to out/taskNNN/passed|failed."
        )
    )
    parser.add_argument(
        "task",
        type=parse_task_num_arg,
        help=f"Task number 1-{MAX_TASK} (e.g. 2)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    task_num = args.task
    task_id = task_id_from_num(task_num)

    onnx_path = SUBMISSION_DIR / f"task{task_num:03d}.onnx"
    if not onnx_path.is_file():
        print(f"ONNX model not found: {onnx_path}", file=sys.stderr)
        return 1

    task = load_task_json(task_id)
    examples = iter_task_examples(task)
    if not examples:
        print(f"No examples found in {task_id}.json", file=sys.stderr)
        return 1

    out_dir = ROOT / "out" / task_id
    passed_dir = out_dir / "passed"
    failed_dir = out_dir / "failed"
    passed_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    passed_count = 0
    failed_count = 0
    split_totals: dict[str, tuple[int, int]] = {}

    print(f"Running {task_id} on {len(examples)} examples from {task_id}.json")
    print(f"Output: {out_dir}/passed and {out_dir}/failed\n")

    for split, idx, example in examples:
        predicted, expected = predict_example(session, example)
        match = np.array_equal(predicted, expected)
        mismatched = int((predicted != expected).sum())
        title = f"{task_id} [{split}#{idx}]"
        if not match:
            title += f" ({mismatched} cell mismatches on 30x30 canvas)"

        inp = pad_grid(example["input"])
        fig = render_comparison(inp, predicted, expected, title=title, match=match)

        dest_dir = passed_dir if match else failed_dir
        filename = f"{task_id}_{split}_{idx:03d}.png"
        out_path = dest_dir / filename
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

        if match:
            passed_count += 1
        else:
            failed_count += 1

        split_pass, split_fail = split_totals.get(split, (0, 0))
        if match:
            split_totals[split] = (split_pass + 1, split_fail)
        else:
            split_totals[split] = (split_pass, split_fail + 1)

        print(f"  {split}#{idx}: {'PASS' if match else 'FAIL'}")

    print(f"\n{task_id} summary: {passed_count} passed, {failed_count} failed")
    for split in SPLITS:
        if split not in split_totals:
            continue
        ok, bad = split_totals[split]
        print(f"  {split}: {ok} passed, {bad} failed")
    print(f"  passed -> {passed_dir}")
    print(f"  failed -> {failed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
