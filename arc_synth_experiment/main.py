#!/usr/bin/env python3
"""Algorithmic ARC solver: rule detection + program search -> static ONNX."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path("/home/filip/Desktop/neuro_golf/arc_synth_experiment")
NEURO_GOLF_ROOT = ROOT.parent
SUBMISSION_DIR = NEURO_GOLF_ROOT / "submission"
sys.path.insert(0, str(ROOT))

from export.onnx_export import can_export_program, export_program_to_onnx
from export.rule_export import export_rule_to_onnx, try_algorithmic_task
from solver.program_ir import ProgramNode, execute
from solver.search import search_programs
from tasks.loader import ArcTask, load_tasks

OUTPUT_DIR = ROOT / "outputs"
SOLUTIONS_PATH = OUTPUT_DIR / "solutions.json"
TASK_START = 1
TASK_END = 400


def _validate_and_save_onnx(
    task: ArcTask,
    program: ProgramNode,
    payload: Dict[str, Any],
    *,
    rule=None,
) -> Dict[str, Any]:
    if not can_export_program(program):
        print("  -> solved but ONNX export not implemented for this program")
        payload["status"] = "solved_not_exportable"
        return payload

    onnx_path = SUBMISSION_DIR / f"{task.task_id}.onnx"
    outputs_copy = OUTPUT_DIR / f"{task.task_id}.onnx"
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if rule is not None:
            export_rule_to_onnx(rule, str(onnx_path), task.train)
        else:
            export_program_to_onnx(program, str(onnx_path), examples=task.train)
        print(f"  -> wrote {onnx_path}")
        outputs_copy.write_bytes(onnx_path.read_bytes())
    except Exception as exc:  # noqa: BLE001
        payload["status"] = "export_error"
        payload["error"] = str(exc)
        print(f"  -> ONNX export failed: {exc}")
        return payload

    sys.path.insert(0, str(NEURO_GOLF_ROOT))
    from train_arc import validate_task_onnx

    try:
        onnx_ok, split_pass = validate_task_onnx(task.task_id, onnx_path, save_viz=False)
    except Exception as exc:  # noqa: BLE001
        payload["status"] = "export_validation_failed"
        payload["error"] = str(exc)
        print(f"  -> validation error: {exc}")
        onnx_path.unlink(missing_ok=True)
        outputs_copy.unlink(missing_ok=True)
        return payload
    payload["onnx_path"] = str(onnx_path)
    payload["onnx_validation_pass"] = onnx_ok
    payload["split_pass"] = split_pass
    payload["status"] = "solved_and_exported" if onnx_ok else "export_validation_failed"
    print(f"  -> competition validation: {'PASS' if onnx_ok else 'FAIL'} {split_pass}")

    if not onnx_ok:
        onnx_path.unlink(missing_ok=True)
        outputs_copy.unlink(missing_ok=True)
        payload["onnx_path"] = None
    return payload


def run_single_task(
    task: ArcTask,
    *,
    min_depth: int = 1,
    max_depth: int = 4,
    verbose_search: bool = False,
) -> Dict[str, Any]:
    print(f"\n{'=' * 60}")
    print(f"Task {task.task_id}: {len(task.train)} train, {len(task.test)} test")
    print(f"{'=' * 60}")

    started = time.time()
    payload: Dict[str, Any] = {
        "task_id": task.task_id,
        "best_score": 0.0,
        "program": None,
        "program_description": None,
        "method": None,
        "training_validation_all_pass": False,
        "onnx_path": None,
        "onnx_validation_pass": False,
        "onnx_exportable": False,
        "status": "unsolved",
        "elapsed_seconds": 0.0,
        "pair_results": [],
    }

    # Phase A: algorithmic rule detection (fast, padded-grid semantics).
    rule, rule_program = try_algorithmic_task(task.task_id, task.train, task.test)
    if rule_program is not None:
        print(f"Rule detected: {rule.label} -> {rule_program.describe()}")
        payload["best_score"] = 1.0
        payload["program"] = rule_program.to_dict()
        payload["program_description"] = rule_program.describe()
        payload["method"] = "rule_detection"
        payload["training_validation_all_pass"] = True
        payload["onnx_exportable"] = can_export_program(rule_program)
        payload["status"] = "solved"
        payload["elapsed_seconds"] = round(time.time() - started, 2)
        payload = _validate_and_save_onnx(task, rule_program, payload, rule=rule)
        payload["elapsed_seconds"] = round(time.time() - started, 2)
        return payload

    # Phase B: brute-force program search over DSL.
    result = search_programs(
        task.train,
        min_depth=min_depth,
        max_depth=max_depth,
        stop_on_perfect=True,
        verbose=verbose_search,
    )
    print(f"Search best: {result.program.describe()} (score={result.score:.4f})")

    training_ok = True
    for idx, pair in enumerate(task.train):
        pred = execute(result.program, pair["input"])
        ok = pred.tolist() == pair["output"]
        training_ok = training_ok and ok

    payload.update(
        {
            "best_score": result.score,
            "program": result.program.to_dict(),
            "program_description": result.program.describe(),
            "method": "program_search",
            "training_validation_all_pass": training_ok,
            "onnx_exportable": can_export_program(result.program),
            "pair_results": result.pair_results,
            "elapsed_seconds": round(time.time() - started, 2),
        }
    )

    if result.score < 1.0 - 1e-9:
        print(f"  -> no perfect training fit (best score {result.score:.4f})")
        return payload

    payload["status"] = "solved"
    return _validate_and_save_onnx(task, result.program, payload)


def print_summary(results: List[Dict[str, Any]]) -> None:
    solved = [r for r in results if r["best_score"] >= 1.0 - 1e-9]
    exported = [r for r in results if r.get("onnx_validation_pass")]
    print(f"\n{'=' * 60}")
    print("Summary (algorithmic — no neural training)")
    print(f"{'=' * 60}")
    print(f"Tasks run:           {len(results)}")
    print(f"Perfect train fit:   {len(solved)}")
    print(f"Competition ONNX:    {len(exported)}")
    if exported:
        print("Validated exports:")
        for row in exported:
            print(f"  {row['task_id']} [{row.get('method')}]: {row['program_description']}")
    unsolved = len(results) - len(solved)
    if unsolved:
        print(f"\n{unsolved} tasks: no rule/program found in current algorithm library.")
        print("Adding more DSL primitives or task-specific logic is required (still algorithmic).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Algorithmic ARC solver -> ONNX for all tasks.")
    parser.add_argument("--start", type=int, default=TASK_START)
    parser.add_argument("--end", type=int, default=TASK_END)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--pack-zip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_tasks(start=args.start, end=args.end, task_filter=args.task)
    if not tasks:
        raise SystemExit("No tasks found.")

    print("=== Algorithmic ARC Solver ===")
    print("Detects rules + searches programs. Exports static ONNX. No neural training.")
    print(f"Output: {SUBMISSION_DIR}/taskNNN.onnx")
    print(f"Tasks: {len(tasks)} ({tasks[0].task_id} .. {tasks[-1].task_id})")

    results: List[Dict[str, Any]] = []
    for task in tasks:
        try:
            results.append(
                run_single_task(
                    task,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                    verbose_search=args.verbose or args.task is not None,
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  -> ERROR: {exc}")
            results.append(
                {
                    "task_id": task.task_id,
                    "status": "error",
                    "error": str(exc),
                    "best_score": 0.0,
                    "onnx_validation_pass": False,
                }
            )

    summary = {
        "mode": "algorithmic",
        "task_start": args.start,
        "task_end": args.end,
        "tasks_run": len(results),
        "perfect_train_fit": sum(1 for r in results if r.get("best_score", 0) >= 1.0 - 1e-9),
        "onnx_validated": sum(1 for r in results if r.get("onnx_validation_pass")),
        "results": results,
    }
    with open(SOLUTIONS_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {SOLUTIONS_PATH}")
    print_summary(results)

    if args.pack_zip:
        sys.path.insert(0, str(NEURO_GOLF_ROOT))
        from build_submission import pack_submission_zip

        pack_submission_zip(strict=False)


if __name__ == "__main__":
    main()
