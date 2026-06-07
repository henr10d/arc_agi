"""Generate the selected ONNX solver for NeuroGolf task097.

Task rule: remove isolated nonzero pixels. A foreground cell is kept only when
at least one of its eight Chebyshev-distance-1 neighbors is also foreground;
kept cells preserve their original color and position, while removed cells
become black inside the variable-size task grid. The ONNX graph uses compact
argmax color indices over the 20x20 maximum task region to avoid realizing the
full 9-channel foreground slice used by the earlier avgpool version.
"""

from __future__ import annotations

import sys
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_task097 import build_argmax_model  # noqa: E402

BEST_PATH = Path(__file__).resolve().with_suffix(".onnx")


def main() -> None:
    onnx.save(build_argmax_model(), str(BEST_PATH))
    print(BEST_PATH)


if __name__ == "__main__":
    main()
