"""Verify and score the generated task098 ONNX model.

Checks all local train/test/arc-gen examples plus randomized separated solid
rectangle grids against the reference rectangle-outline rule.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx

from all.task098 import BEST_PATH, validate_model
from score_all_onnx import verify_correctness
from score_model import print_report, score_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark task098 ONNX correctness and NeuroGolf cost.")
    parser.add_argument("model", nargs="?", type=Path, default=BEST_PATH)
    parser.add_argument("--random-cases", type=int, default=512)
    args = parser.parse_args()

    model = onnx.load(str(args.model))
    random_ok, random_detail = validate_model(model, random_cases=args.random_cases)
    local_ok, local_detail, _passed, _total = verify_correctness(args.model)
    result = score_file(args.model)

    print(f"model:             {args.model}")
    print(f"local examples:    {local_detail} ({local_ok})")
    print(f"random rectangles: {random_detail} ({random_ok})")
    print_report(result)
    if not local_ok or not random_ok or not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
