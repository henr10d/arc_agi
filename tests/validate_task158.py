"""Validate the generated ONNX model for NeuroGolf task158."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from all.task158 import BEST_PATH, print_result, validate  # noqa: E402


def main() -> None:
    print_result(validate(BEST_PATH))


if __name__ == "__main__":
    main()
