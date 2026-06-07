"""Validate task085.onnx on every local task085 example."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from all.task085 import BEST_PATH, validate_model  # noqa: E402


def main() -> None:
    if not BEST_PATH.is_file():
        raise SystemExit(f"missing model: {BEST_PATH}")
    validate_model(BEST_PATH)
    print("task085 validation passed")


if __name__ == "__main__":
    main()
