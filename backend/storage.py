from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROJECTS_DIR = ROOT / "projects"
ONNX_DIR = ROOT / "onnx"
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def safe_name(name: str) -> str:
    name = name.removesuffix(".json")
    if not SAFE_NAME.match(name):
        raise ValueError("Names may only contain letters, numbers, dot, underscore, and dash")
    return name


def list_tasks() -> list[dict[str, Any]]:
    # Keep task listing cheap; this repo may contain hundreds of large ARC JSON files.
    return [
        {"id": path.stem, "name": path.stem, "train": 0, "test": 0, "extra": 0}
        for path in sorted(DATA_DIR.glob("*.json"))
    ]


def load_task(task_id: str) -> dict[str, Any]:
    name = safe_name(task_id)
    path = DATA_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(name)
    return json.loads(path.read_text())


def list_projects() -> list[str]:
    PROJECTS_DIR.mkdir(exist_ok=True)
    return [p.stem for p in sorted(PROJECTS_DIR.glob("*.json"))]


def project_path(name: str) -> Path:
    return PROJECTS_DIR / f"{safe_name(name)}.json"


def load_project(name: str) -> dict[str, Any]:
    path = project_path(name)
    if not path.exists():
        raise FileNotFoundError(name)
    return json.loads(path.read_text())


def save_project(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    PROJECTS_DIR.mkdir(exist_ok=True)
    path = project_path(name)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return {"name": path.stem, "path": str(path)}


def export_path(task_id: str) -> Path:
    return ONNX_DIR / f"{safe_name(task_id)}.onnx"
