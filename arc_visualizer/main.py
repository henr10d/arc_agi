#!/usr/bin/env python3
"""Standalone ARC grid visualizer for manual task analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure local imports work when run as script
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loader import parse_task_id
from objects import GridObject
from ui import open_explorer

# --- Default dataset path (override with --data-dir) ---
DATASET_DIR = Path("/home/filip/Desktop/neuro_golf/data")


def _print_object_debug(label: str, objects: list[GridObject]) -> None:
    print(f"\n{label} objects ({len(objects)}):")
    for obj in objects:
        print(
            f"  id={obj.object_id}  bbox=(x:{obj.min_x}-{obj.max_x}, y:{obj.min_y}-{obj.max_y})  "
            f"area={obj.area}  centroid=({obj.centroid[0]:.2f}, {obj.centroid[1]:.2f})  "
            f"colors={sorted(obj.colors)}"
        )


def _print_matching_debug(input_objects, output_objects, match_result) -> None:
    print("\nMatching scores (input → output):")
    for inp in input_objects:
        for out in output_objects:
            key = (inp.object_id, out.object_id)
            if key not in match_result.scores:
                continue
            s = match_result.scores[key]
            print(
                f"  {inp.object_id}→{out.object_id}: total={s.total:.3f}  "
                f"iou={s.iou:.3f}  centroid_dist={s.centroid_distance:.3f}  "
                f"color={s.color_overlap:.3f}"
            )

    print("\nAssigned pairs:")
    for inp_id, out_id in sorted(match_result.mapping.items()):
        if out_id is None:
            print(f"  input {inp_id} → UNMATCHED")
        else:
            s = match_result.scores[(inp_id, out_id)]
            print(f"  input {inp_id} → output {out_id}  (score={s.total:.3f})")

    if match_result.warnings:
        print("\nWarnings:")
        for w in match_result.warnings:
            print(f"  ⚠ {w}")


def _print_summary(input_objects, output_objects, match_result) -> None:
    matched = sum(1 for v in match_result.mapping.values() if v is not None)
    unmatched_in = sum(1 for v in match_result.mapping.values() if v is None)
    matched_out_ids = {v for v in match_result.mapping.values() if v is not None}
    unmatched_out = len(output_objects) - len(matched_out_ids)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Input objects:  {len(input_objects)}")
    print(f"Output objects: {len(output_objects)}")
    print(f"Matched pairs:  {matched}")
    print(f"Unmatched input objects:  {unmatched_in}")
    print(f"Unmatched output objects: {unmatched_out}")
    print("=" * 60)


def _on_example_change(**kwargs) -> None:
    task_id = kwargs["task_id"]
    split = kwargs["split"]
    example_index = kwargs["example_index"]
    pair = kwargs["pair"]
    input_objects = kwargs["input_objects"]
    output_objects = kwargs["output_objects"]
    match_result = kwargs["match_result"]

    print("\n" + "#" * 60)
    print(f"{task_id}  split={split}  example={example_index}")
    print(f"Input shape:  {len(pair.input)}x{len(pair.input[0])}")
    print(f"Output shape: {len(pair.output)}x{len(pair.output[0])}")
    _print_object_debug("Input", input_objects)
    _print_object_debug("Output", output_objects)
    _print_matching_debug(input_objects, output_objects, match_result)
    _print_summary(input_objects, output_objects, match_result)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive ARC object visualizer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Keyboard shortcuts (click the plot window first):
  ← / →  or  a / d     previous / next example
  [ / ]                previous / next task
  ↑ / ↓                previous / next split (train / test / arc-gen)

Examples:
  python main.py                  # start on task001
  python main.py 42               # start on task042
  python main.py task002 --split test
        """,
    )
    parser.add_argument(
        "task",
        nargs="?",
        default="task001",
        help="Task id: 1, 001, or task001 (default: task001)",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "arc-gen"),
        default="train",
        help="Which example split to open (default: train)",
    )
    parser.add_argument(
        "--example",
        type=int,
        default=0,
        metavar="N",
        help="Starting example index (default: 0)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATASET_DIR,
        help=f"Folder with task*.json files (default: {DATASET_DIR})",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        task_id = parse_task_id(args.task)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if not args.data_dir.is_dir():
        print(f"ERROR: Data directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        app = open_explorer(
            args.data_dir,
            task_id,
            split=args.split,
            example_index=args.example,
            on_example_change=_on_example_change,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    app.show()


if __name__ == "__main__":
    main()
