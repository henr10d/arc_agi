"""Validate task076_best.onnx on every local task076 example."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from all.task076 import BEST_PATH, validate  # noqa: E402


def main() -> None:
    if not BEST_PATH.is_file():
        raise SystemExit(f"missing model: {BEST_PATH}")
    if not validate(BEST_PATH):
        raise SystemExit(1)
    print("task076 validation passed")


if __name__ == "__main__":
    main()
