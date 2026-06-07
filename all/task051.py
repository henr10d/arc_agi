"""Minimal ONNX for ARC task051: extend marker-color ray from arrow tip.

Task rule: same H×W input/output. A dominant-color triangle/arrow on black
background contains a unique marker cell (count==1). Output keeps the input and
draws a straight line from the marker side to the grid edge in the marker color:
marker on left → horizontal right; on right → horizontal left; on top → vertical
down; on bottom/interior → vertical up. Lines paint only empty (non-foreground)
cells inside the active grid.

Build: ``python all/create_task051_onnx.py`` writes ``task051.onnx``.
"""

from __future__ import annotations

import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT_DIR))

from create_task051_onnx import main as build_main  # noqa: E402

if __name__ == "__main__":
    build_main()
