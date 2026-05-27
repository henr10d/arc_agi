"""Load ARC tasks from hardcoded filesystem paths."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


DATA_DIR = Path("/home/filip/Desktop/neuro_golf/data")
TASK001_PATH = DATA_DIR / "task001.json"


@dataclass
class ArcTask:
    task_id: str
    train: List[Dict[str, Any]]
    test: List[Dict[str, Any]]


def list_task_ids(start: int = 1, end: int = 400) -> List[str]:
    """Return task IDs that exist on disk within [start, end]."""
    ids: List[str] = []
    for num in range(start, end + 1):
        path = DATA_DIR / f"task{num:03d}.json"
        if path.is_file():
            ids.append(f"task{num:03d}")
    return ids


def load_task(task_id: str) -> ArcTask:
    path = DATA_DIR / f"{task_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing task file: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return ArcTask(task_id=task_id, train=raw["train"], test=raw.get("test", []))


def load_task001() -> ArcTask:
    return load_task("task001")


def load_tasks(
    start: int = 1,
    end: int = 400,
    task_filter: Optional[str] = None,
) -> List[ArcTask]:
    if task_filter:
        return [load_task(task_filter)]
    return [load_task(task_id) for task_id in list_task_ids(start, end)]
