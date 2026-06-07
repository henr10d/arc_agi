"""Benchmark NeuroGolf task083 ONNX variants."""

from __future__ import annotations

from all.task083 import benchmark_variants, print_benchmark


def main() -> None:
    results, best_name, _ = benchmark_variants()
    print_benchmark(results, best_name)


if __name__ == "__main__":
    main()
