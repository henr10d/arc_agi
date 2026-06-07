#!/usr/bin/env python3
"""Score all task ONNX files in a directory and print a points table.

Uses official NeuroGolf rules: points apply only when the model is valid,
measurable, and 100% correct on train + test + arc-gen. Otherwise points = 0.

Usage:
  python score_all_onnx.py
  python score_all_onnx.py /path/to/onnx/dir
  python score_all_onnx.py --start 1 --end 10
  python score_all_onnx.py --csv scores.csv
  python score_all_onnx.py --jobs 4 --timeout 14
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import time
import traceback
from pathlib import Path

import numpy as np
import onnxruntime as ort

from score_model import DATA_DIR, collect_onnx_paths, convert_to_numpy, score_file

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - keeps the scorer usable without tqdm installed.
    tqdm = None

ROOT = Path(__file__).resolve().parent
DEFAULT_ONNX_DIR = Path("/home/filip/Desktop/neuro_golf/all/submission/")
# DEFAULT_ONNX_DIR = ROOT / "all"
TASK_RE = re.compile(r"^task(\d+)$")


def default_jobs() -> int:
    """Conservative worker count; ONNX Runtime also uses CPU threads internally."""
    return max(1, min(4, os.cpu_count() or 1))


def _task_json_path(onnx_path: Path) -> Path:
    return DATA_DIR / f"{onnx_path.stem}.json"


def task_number(onnx_path: Path) -> int | None:
    """Return numeric task id from names like task001.onnx, else None."""
    match = TASK_RE.match(onnx_path.stem)
    if match is None:
        return None
    return int(match.group(1))


def filter_paths_by_task_range(
    paths: list[Path], start: int | None, end: int | None
) -> list[Path]:
    """Filter ONNX paths by inclusive task number bounds."""
    if start is None and end is None:
        return paths

    filtered: list[Path] = []
    for path in paths:
        number = task_number(path)
        if number is None:
            continue
        if start is not None and number < start:
            continue
        if end is not None and number > end:
            continue
        filtered.append(path)
    return filtered


def verify_correctness(onnx_path: Path) -> tuple[bool, str, int, int]:
    """Return (all_pass, summary, passed, total) for train+test+arc-gen."""
    task_path = _task_json_path(onnx_path)
    if not task_path.is_file():
        return False, "no task JSON", 0, 0

    with task_path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    examples: list[dict] = []
    for split in ("train", "test", "arc-gen"):
        examples.extend(data.get(split, []))

    if not examples:
        return False, "no examples", 0, 0

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        return False, f"load failed: {exc}", 0, len(examples)

    passed = 0
    total = len(examples)
    for example in examples:
        benchmark = convert_to_numpy(example, "input")
        expected = convert_to_numpy(example, "output")
        if benchmark is None or expected is None:
            total -= 1
            continue
        try:
            out = session.run(["output"], {"input": benchmark})[0]
        except Exception as exc:
            return False, f"run failed: {exc}", passed, total
        pred = (out > 0.0).astype(np.float32)
        if np.array_equal(pred, expected):
            passed += 1

    summary = f"{passed}/{total}"
    return passed == total, summary, passed, total


def evaluate_onnx(onnx_path: Path) -> dict:
    correctness_ok, correctness, passed, total = verify_correctness(onnx_path)
    metrics = score_file(onnx_path)

    points = 0.0
    status = "invalid"
    if not metrics["valid"]:
        status = "invalid"
        detail = str(metrics.get("error") or "measurement failed").strip().splitlines()[0]
    elif not correctness_ok:
        status = "wrong"
        detail = correctness
        points = 0.0
    else:
        status = "ok"
        detail = correctness
        points = float(metrics["score"])

    return {
        "task": onnx_path.stem,
        "path": onnx_path,
        "status": status,
        "points": points,
        "cost": metrics.get("cost"),
        "memory": metrics.get("memory"),
        "params": metrics.get("params"),
        "correctness": correctness,
        "passed": passed,
        "total": total,
        "detail": detail,
    }


def evaluate_onnx_safe(onnx_path: Path) -> dict:
    """Worker-safe wrapper that turns unexpected crashes into table rows."""
    try:
        return evaluate_onnx(onnx_path)
    except Exception:
        return {
            "task": onnx_path.stem,
            "path": onnx_path,
            "status": "invalid",
            "points": 0.0,
            "cost": None,
            "memory": None,
            "params": None,
            "correctness": "-",
            "passed": 0,
            "total": 0,
            "detail": traceback.format_exc().strip().splitlines()[-1],
        }


def timeout_row(onnx_path: Path, timeout_seconds: float) -> dict:
    return {
        "task": onnx_path.stem,
        "path": onnx_path,
        "status": "invalid",
        "points": 0.0,
        "cost": None,
        "memory": None,
        "params": None,
        "correctness": "-",
        "passed": 0,
        "total": 0,
        "detail": f"timed out after {timeout_seconds:g}s",
    }


def _worker_evaluate(onnx_path: Path, queue: mp.Queue) -> None:
    queue.put(evaluate_onnx_safe(onnx_path))


def evaluate_paths(paths: list[Path], jobs: int, timeout_seconds: float) -> list[dict]:
    """Evaluate ONNX paths, optionally in parallel, while keeping final output sortable."""
    rows = []
    pending = iter(paths)
    active: dict[int, tuple[Path, mp.Process, mp.Queue, float]] = {}
    max_jobs = max(1, jobs)

    progress = tqdm(total=len(paths), desc="Scoring ONNX", unit="model") if tqdm is not None else None

    def start_next() -> bool:
        try:
            path = next(pending)
        except StopIteration:
            return False
        queue: mp.Queue = mp.Queue(maxsize=1)
        process = mp.Process(target=_worker_evaluate, args=(path, queue))
        process.start()
        active[process.pid] = (path, process, queue, time.monotonic())
        if progress is not None:
            progress.set_postfix_str(path.name, refresh=True)
        return True

    try:
        while len(active) < max_jobs and start_next():
            pass

        while active:
            now = time.monotonic()
            for pid, (path, process, queue, started) in list(active.items()):
                if process.is_alive() and timeout_seconds > 0 and now - started > timeout_seconds:
                    process.terminate()
                    process.join(timeout=1)
                    if process.is_alive():
                        process.kill()
                        process.join()
                    rows.append(timeout_row(path, timeout_seconds))
                    active.pop(pid)
                    queue.close()
                    if progress is not None:
                        progress.set_postfix_str(f"{path.name} timeout", refresh=True)
                        progress.update(1)
                    start_next()
                    continue

                if process.is_alive():
                    continue

                process.join()
                try:
                    row = queue.get_nowait()
                except Exception:
                    row = {
                        "task": path.stem,
                        "path": path,
                        "status": "invalid",
                        "points": 0.0,
                        "cost": None,
                        "memory": None,
                        "params": None,
                        "correctness": "-",
                        "passed": 0,
                        "total": 0,
                        "detail": f"worker exited with code {process.exitcode}",
                    }
                rows.append(row)
                active.pop(pid)
                queue.close()
                if progress is not None:
                    progress.set_postfix_str(path.name, refresh=True)
                    progress.update(1)
                start_next()

            time.sleep(0.05)
    finally:
        for path, process, queue, _started in active.values():
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    process.kill()
                    process.join()
            queue.close()
        if progress is not None:
            progress.close()

    return rows


def average_points_correct(rows: list[dict]) -> float | None:
    """Mean points over tasks with status ok; None if none."""
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        return None
    return sum(r["points"] for r in ok) / len(ok)


def print_table(rows: list[dict], *, total_points: float, avg_points: float | None) -> None:
    headers = ("task", "points", "cost", "status", "correct", "note")
    table_rows: list[tuple[str, ...]] = []
    for row in rows:
        cost = row["cost"]
        cost_s = str(cost) if cost is not None else "-"
        note = row["detail"] if row["status"] != "ok" else ""
        table_rows.append(
            (
                row["task"],
                f"{row['points']:.3f}" if row["points"] else "0",
                cost_s,
                row["status"],
                row["correctness"],
                note[:48],
            )
        )

    widths = [len(h) for h in headers]
    for cells in table_rows:
        for i, cell in enumerate(cells):
            widths[i] = max(widths[i], len(cell))

    def fmt_line(cells: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print(fmt_line(headers))
    print(fmt_line(tuple("-" * w for w in widths)))
    for cells in table_rows:
        print(fmt_line(cells))

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_wrong = sum(1 for r in rows if r["status"] == "wrong")
    n_invalid = sum(1 for r in rows if r["status"] == "invalid")
    print()
    print(f"Models:        {len(rows)}")
    print(f"Scoring:       {n_ok} ok, {n_wrong} wrong, {n_invalid} invalid")
    print(f"Total points:  {total_points:.6f}")
    if avg_points is not None:
        print(f"Avg (correct): {avg_points:.6f}  ({n_ok} tasks)")
    else:
        print("Avg (correct): -  (no correct tasks)")


def write_csv(path: Path, rows: list[dict], *, total_points: float, avg_points: float | None) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "task",
                "points",
                "cost",
                "memory",
                "params",
                "status",
                "correctness",
                "note",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "task": row["task"],
                    "points": f"{row['points']:.6f}",
                    "cost": row["cost"] if row["cost"] is not None else "",
                    "memory": row["memory"] if row["memory"] is not None else "",
                    "params": row["params"] if row["params"] is not None else "",
                    "status": row["status"],
                    "correctness": row["correctness"],
                    "note": row["detail"] if row["status"] != "ok" else "",
                }
            )
        writer.writerow({})
        writer.writerow({"task": "TOTAL", "points": f"{total_points:.6f}"})
        if avg_points is not None:
            n_ok = sum(1 for r in rows if r["status"] == "ok")
            writer.writerow(
                {"task": "AVG_CORRECT", "points": f"{avg_points:.6f}", "note": f"{n_ok} tasks"}
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score all ONNX models in a directory (official points if correct)."
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=DEFAULT_ONNX_DIR,
        help=f"Directory with task*.onnx (default: {DEFAULT_ONNX_DIR})",
    )
    parser.add_argument("--csv", type=Path, help="Also write results to this CSV file")
    parser.add_argument("--start", type=int, help="First task number to score, inclusive")
    parser.add_argument("--end", type=int, help="Last task number to score, inclusive")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=default_jobs(),
        help=f"Parallel worker processes (default: {default_jobs()}, use 1 for sequential)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=14.0,
        help="Per-model timeout in seconds before marking invalid and skipping (default: 14)",
    )
    args = parser.parse_args()

    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    if args.timeout < 0:
        raise SystemExit("--timeout must be >= 0")
    if args.start is not None and args.start < 1:
        raise SystemExit("--start must be >= 1")
    if args.end is not None and args.end < 1:
        raise SystemExit("--end must be >= 1")
    if args.start is not None and args.end is not None and args.start > args.end:
        raise SystemExit("--start must be <= --end")

    onnx_dir = args.path.resolve()
    if not onnx_dir.is_dir():
        raise SystemExit(f"not a directory: {onnx_dir}")

    paths = collect_onnx_paths(onnx_dir)
    if not paths:
        raise SystemExit(f"no ONNX files under {onnx_dir}")
    paths = filter_paths_by_task_range(paths, args.start, args.end)
    if not paths:
        bounds = []
        if args.start is not None:
            bounds.append(f"start={args.start}")
        if args.end is not None:
            bounds.append(f"end={args.end}")
        range_note = ", ".join(bounds)
        raise SystemExit(f"no ONNX files under {onnx_dir} match task range ({range_note})")

    rows = evaluate_paths(paths, args.jobs, args.timeout)
    rows.sort(key=lambda r: r["task"])
    total_points = sum(r["points"] for r in rows)
    avg_points = average_points_correct(rows)

    print_table(rows, total_points=total_points, avg_points=avg_points)
    if args.csv:
        write_csv(args.csv.resolve(), rows, total_points=total_points, avg_points=avg_points)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
