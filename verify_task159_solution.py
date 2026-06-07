"""Verify task159 ONNX accuracy and NeuroGolf score metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from all.task159 import BEST_PATH, TASK_PATH, build_dynamic_index_selector, write_model  # noqa: E402
from score_model import convert_to_numpy, score_file  # noqa: E402


def ensure_model(path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_model(build_dynamic_index_selector(), path)


def verify_examples(path: Path, splits: tuple[str, ...]) -> tuple[int, int]:
    model = onnx.load(str(path))
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    with TASK_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    passed = 0
    checked = 0
    for split in splits:
        split_passed = 0
        split_checked = 0
        for idx, example in enumerate(data.get(split, [])):
            inp = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if inp is None or expected is None:
                continue
            pred = session.run(["output"], {"input": inp})[0]
            ok = np.array_equal(pred > 0.0, expected > 0.0)
            split_passed += int(ok)
            split_checked += 1
            if not ok:
                print(f"mismatch: {split}[{idx}]")
        passed += split_passed
        checked += split_checked
        print(f"{split}: {split_passed}/{split_checked}")
    return passed, checked


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify task159 exact-match accuracy and score.")
    parser.add_argument("--model", type=Path, default=BEST_PATH, help="ONNX model path")
    parser.add_argument("--all-splits", action="store_true", help="verify train, test, and arc-gen")
    args = parser.parse_args()

    ensure_model(args.model)
    splits = ("train", "test", "arc-gen") if args.all_splits else ("train",)
    passed, checked = verify_examples(args.model, splits)
    accuracy = passed / checked if checked else 0.0

    result = score_file(args.model)
    print(f"exact-match accuracy: {passed}/{checked} ({accuracy:.2%})")
    print(f"valid: {result['valid']}")
    print(f"memory: {result['memory']}")
    print(f"params: {result['params']}")
    print(f"cost: {result['cost']}")
    score = result["score"]
    print(f"score: {score:.6f}" if isinstance(score, float) else "score: None")
    if result["error"]:
        print(f"error: {str(result['error']).strip()}")


if __name__ == "__main__":
    main()
