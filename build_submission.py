#!/usr/bin/env python3
"""
Build NeuroGolf submission.zip with task001.onnx ... task400.onnx.

Pipeline:
  1. Analytical program synthesis (arc_synth_experiment) — fast, exact for simple rules
  2. Neural CA training (train_arc.py) — remaining tasks
  3. Validate + pack submission.zip
"""

from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
SUBMISSION_DIR = ROOT / "submission"
SYNTH_ROOT = ROOT / "arc_synth_experiment"
DATA_DIR = ROOT / "data"

TASK_START = 1
TASK_END = 400


def list_task_ids(start: int = TASK_START, end: int = TASK_END) -> List[str]:
    return [
        f"task{num:03d}"
        for num in range(start, end + 1)
        if (DATA_DIR / f"task{num:03d}.json").is_file()
    ]


def submission_onnx_path(task_id: str) -> Path:
    return SUBMISSION_DIR / f"{task_id}.onnx"


def count_submission_files(task_ids: List[str]) -> int:
    return sum(1 for tid in task_ids if submission_onnx_path(tid).is_file())


def validate_submission_onnx(task_id: str, onnx_path: Optional[Path] = None) -> bool:
    onnx_path = onnx_path or submission_onnx_path(task_id)
    if not onnx_path.is_file():
        return False
    try:
        from train_arc import validate_task_onnx

        ok, _ = validate_task_onnx(task_id, onnx_path, save_viz=False)
        return ok
    except Exception as exc:  # noqa: BLE001
        print(f"  validate error {task_id}: {exc}")
        return False


def run_analytical_synthesis(
    start: int = TASK_START,
    end: int = TASK_END,
    task_filter: Optional[str] = None,
) -> Dict[str, dict]:
    sys.path.insert(0, str(SYNTH_ROOT))
    from main import run_single_task
    from tasks.loader import load_tasks

    tasks = load_tasks(start=start, end=end, task_filter=task_filter)
    results: Dict[str, dict] = {}
    for task in tasks:
        row = run_single_task(task, verbose_search=False)
        results[task.task_id] = row
    return results


def run_neural_training(
    task_ids: List[str],
    steps: Optional[int] = None,
    skip_valid: bool = True,
) -> Dict[str, bool]:
    from train_arc import DEVICE, TASK_STEPS, process_task

    steps = steps or TASK_STEPS
    outcomes: Dict[str, bool] = {}
    missing = [tid for tid in task_ids if not submission_onnx_path(tid).is_file()]
    if skip_valid:
        missing = [
            tid
            for tid in missing
            if not (
                submission_onnx_path(tid).is_file() and validate_submission_onnx(tid)
            )
        ]

    total = len(missing)
    print(f"\nNeural training for {total} task(s)...")
    for idx, task_id in enumerate(missing, start=1):
        print(f"\n>>> Neural {idx}/{total}: {task_id}")
        try:
            outcomes[task_id] = process_task(task_id, DEVICE, steps)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            outcomes[task_id] = False
    return outcomes


def pack_submission_zip(
    zip_path: Path = ROOT / "submission.zip",
    task_ids: Optional[List[str]] = None,
    strict: bool = True,
) -> Path:
    task_ids = task_ids or list_task_ids()
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    missing = [tid for tid in task_ids if not submission_onnx_path(tid).is_file()]

    if missing:
        msg = f"Missing {len(missing)} ONNX file(s): {missing[:5]}{'...' if len(missing) > 5 else ''}"
        if strict:
            raise FileNotFoundError(msg)
        print(f"WARNING: {msg}")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        written = 0
        for task_id in task_ids:
            onnx_file = submission_onnx_path(task_id)
            if onnx_file.is_file():
                zf.write(onnx_file, arcname=onnx_file.name)
                written += 1
    print(f"Wrote {zip_path} ({written} models, {zip_path.stat().st_size:,} bytes)")
    return zip_path


def write_status_report(
    task_ids: List[str],
    synth_results: Dict[str, dict],
    neural_results: Dict[str, bool],
    path: Path = SUBMISSION_DIR / "build_status.json",
) -> None:
    rows = []
    for task_id in task_ids:
        onnx_path = submission_onnx_path(task_id)
        rows.append(
            {
                "task_id": task_id,
                "onnx_exists": onnx_path.is_file(),
                "onnx_valid": validate_submission_onnx(task_id) if onnx_path.is_file() else False,
                "synth_status": synth_results.get(task_id, {}).get("status"),
                "neural_pass": neural_results.get(task_id),
            }
        )
    payload = {
        "tasks_total": len(task_ids),
        "onnx_present": sum(1 for r in rows if r["onnx_exists"]),
        "onnx_valid": sum(1 for r in rows if r["onnx_valid"]),
        "tasks": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote {path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build full NeuroGolf submission.zip")
    parser.add_argument("--start", type=int, default=TASK_START)
    parser.add_argument("--end", type=int, default=TASK_END)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--synth-only", action="store_true", help="Only run analytical synthesis")
    parser.add_argument("--train-only", action="store_true", help="Skip synthesis, train missing tasks")
    parser.add_argument("--no-train", action="store_true", help="Skip neural training")
    parser.add_argument("--steps", type=int, default=None, help="Override train_arc TASK_STEPS")
    parser.add_argument("--pack", action="store_true", help="Pack submission.zip at end")
    parser.add_argument(
        "--allow-partial-zip",
        action="store_true",
        help="Pack zip even if some tasks are missing ONNX",
    )
    args = parser.parse_args()

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    task_ids = [args.task] if args.task else list_task_ids(args.start, args.end)
    if not task_ids:
        raise SystemExit("No tasks found.")

    print("=== NeuroGolf Submission Builder ===")
    print(f"Output dir: {SUBMISSION_DIR}")
    print(f"Tasks: {len(task_ids)} ({task_ids[0]} .. {task_ids[-1]})")
    t0 = time.time()

    synth_results: Dict[str, dict] = {}
    if not args.train_only:
        print("\n--- Phase 1: Analytical synthesis ---")
        synth_results = run_analytical_synthesis(
            start=args.start,
            end=args.end,
            task_filter=args.task,
        )
        n_synth = sum(1 for r in synth_results.values() if r.get("onnx_validation_pass"))
        print(f"Analytical ONNX validated: {n_synth}/{len(synth_results)}")

    neural_results: Dict[str, bool] = {}
    if not args.synth_only and not args.no_train:
        print("\n--- Phase 2: Neural training (missing tasks) ---")
        neural_results = run_neural_training(task_ids, steps=args.steps)

    present = count_submission_files(task_ids)
    valid = sum(1 for tid in task_ids if validate_submission_onnx(tid))
    print(f"\nSubmission dir: {present}/{len(task_ids)} ONNX files, {valid} fully validated")

    write_status_report(task_ids, synth_results, neural_results)

    if args.pack:
        pack_submission_zip(strict=not args.allow_partial_zip)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    if present < len(task_ids):
        print(
            "Note: competition requires functionally correct ONNX for each task. "
            "Run without --no-train to train remaining tasks via train_arc.py."
        )


if __name__ == "__main__":
    main()
