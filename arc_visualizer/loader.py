"""Load ARC task JSON files (standalone, no project imports)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


Grid = list[list[int]]


@dataclass(frozen=True)
class GridPair:
    input: Grid
    output: Grid


@dataclass(frozen=True)
class ArcTask:
    task_id: str
    train: list[GridPair]
    test: list[GridPair]
    arc_gen: list[GridPair]


def _parse_pairs(raw_pairs: list[dict[str, Any]]) -> list[GridPair]:
    pairs: list[GridPair] = []
    for item in raw_pairs:
        pairs.append(GridPair(input=item["input"], output=item["output"]))
    return pairs


def list_task_ids(data_dir: Path) -> list[str]:
    """Return sorted task ids (task001, task002, ...) available in data_dir."""
    return sorted(p.stem for p in data_dir.glob("task*.json"))


def parse_task_id(raw: str) -> str:
    """Accept '1', '001', or 'task001'."""
    raw = raw.strip().lower()
    if raw.isdigit():
        return f"task{int(raw):03d}"
    if raw.startswith("task") and raw[4:].isdigit():
        return f"task{int(raw[4:]):03d}"
    raise ValueError(f"Invalid task id: {raw!r} (use 1, 001, or task001)")


def load_task_json(path: Path) -> ArcTask:
    """Load one ARC task file."""
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    task_id = path.stem
    return ArcTask(
        task_id=task_id,
        train=_parse_pairs(data.get("train", [])),
        test=_parse_pairs(data.get("test", [])),
        arc_gen=_parse_pairs(data.get("arc-gen", [])),
    )
