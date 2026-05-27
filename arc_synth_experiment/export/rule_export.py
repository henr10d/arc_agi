"""Export detected TaskRule objects directly to competition ONNX."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from export.onnx_export import export_program_to_onnx, infer_core_size, infer_output_size
from solver.program_ir import ProgramNode
from solver.rule_search import rule_to_program

_NEURO_GOLF = Path(__file__).resolve().parents[2]
if str(_NEURO_GOLF) not in sys.path:
    sys.path.insert(0, str(_NEURO_GOLF))

from train_arc import RuleKind, apply_rule_padded, grid_to_array, pad_grid  # noqa: E402


def verify_rule_on_examples(rule, examples: Sequence[dict]) -> bool:
    for example in examples:
        inp = pad_grid(example["input"])
        expected = pad_grid(example["output"])
        pred = apply_rule_padded(inp, rule)
        if pred.shape != expected.shape or not np.array_equal(pred, expected):
            return False
    return True


def infer_rule_output_size(rule, examples: Sequence[dict]) -> Tuple[int, int]:
    oh, ow = 0, 0
    for example in examples:
        inp = pad_grid(example["input"])
        out = apply_rule_padded(inp, rule)
        oh = max(oh, out.shape[0])
        ow = max(ow, out.shape[1])
    return oh, ow


def export_rule_to_onnx(rule, output_path: str, examples: List[dict]) -> None:
    """Export a verified TaskRule via program ONNX (all supported rules map to ProgramNode)."""
    program = rule_to_program(rule)
    if program is None:
        raise ValueError(f"Unsupported rule export: {rule.kind}")
    export_program_to_onnx(program, output_path, examples=examples)


def try_algorithmic_task(task_id: str, train: List[dict], test: List[dict]) -> Tuple[object | None, ProgramNode | None]:
    """Detect rule, verify on padded train+test, return (rule, program) or (None, None)."""
    from train_arc import detect_task_rule

    rule = detect_task_rule(train)
    if rule.kind == RuleKind.UNKNOWN:
        return None, None

    program = rule_to_program(rule)
    if program is None:
        return None, None

    if not verify_rule_on_examples(rule, train):
        return None, None
    if test and not verify_rule_on_examples(rule, test):
        return None, None

    return rule, program
