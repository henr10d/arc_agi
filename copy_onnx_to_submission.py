#!/usr/bin/env python3
"""Copy generated ONNX files from all/ into all/submission/."""

from pathlib import Path
import shutil


REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "all"
DEST_DIR = SOURCE_DIR / "submission"


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    moved = 0
    for onnx_path in sorted(SOURCE_DIR.glob("*.onnx")):
        destination = DEST_DIR / onnx_path.name
        shutil.copy2(str(onnx_path), str(destination))
        moved += 1
        print(f"copied {onnx_path.relative_to(REPO_ROOT)} -> {destination.relative_to(REPO_ROOT)}")

    print(f"Copied {moved} ONNX file(s) to {DEST_DIR.relative_to(REPO_ROOT)}.")


if __name__ == "__main__":
    main()
