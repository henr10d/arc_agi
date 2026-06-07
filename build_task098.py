"""Build task098.onnx from the optimized task098 generator.

The generator benchmarks several ONNX variants, writes the best model to
``all/task098.onnx`` for submission scoring, and copies it to ``task098.onnx`` at
the repository root for quick direct inspection.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    runpy.run_path(str(Path(__file__).resolve().parent / "all" / "task098.py"), run_name="__main__")


if __name__ == "__main__":
    main()
