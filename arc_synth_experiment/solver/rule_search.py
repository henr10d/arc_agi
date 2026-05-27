"""Fast rule detection before brute-force DSL search."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from solver.program_ir import ProgramNode, execute

_NEURO_GOLF = Path(__file__).resolve().parents[2]
if str(_NEURO_GOLF) not in sys.path:
    sys.path.insert(0, str(_NEURO_GOLF))

from train_arc import (  # noqa: E402
    RuleKind,
    apply_rule_padded,
    detect_task_rule,
    pad_grid,
)


def rule_to_program(rule) -> Optional[ProgramNode]:
    if rule.kind == RuleKind.STENCIL_COPY:
        return ProgramNode("paste_at", {"mode": "stencil", "factor": rule.factor_h})
    if rule.kind == RuleKind.COPY_OFFSET:
        dr, dc = rule.offset
        return ProgramNode("translate", {"dx": dr, "dy": dc})
    if rule.kind == RuleKind.HFLIP:
        return ProgramNode("mirror_x")
    if rule.kind == RuleKind.VFLIP:
        return ProgramNode("mirror_y")
    if rule.kind == RuleKind.HVFLIP:
        return ProgramNode("compose", children=[ProgramNode("mirror_x"), ProgramNode("mirror_y")])
    if rule.kind == RuleKind.TILE:
        return ProgramNode("tile", {"factor": rule.factor})
    if rule.kind == RuleKind.SCALE_NEAREST:
        return ProgramNode("scale_nearest", {"factor": rule.factor})
    if rule.kind == RuleKind.COLOR_REPLACE:
        mapping = {str(k): int(v) for k, v in rule.color_map.items()}
        return ProgramNode("color_map", {"mapping": mapping})
    if rule.kind == RuleKind.FLOOD_FILL_BOUNDARY:
        return ProgramNode("flood_fill_boundary")
    return None


def try_rule_program(train_pairs: Sequence[dict]) -> Optional[ProgramNode]:
    """Return a program if train_arc rule detection fits all training pairs (padded)."""
    rule = detect_task_rule(list(train_pairs))
    program = rule_to_program(rule)
    if program is None:
        return None

    for pair in train_pairs:
        inp = pad_grid(pair["input"])
        expected = pad_grid(pair["output"])
        try:
            pred = apply_rule_padded(inp, rule)
        except Exception:
            return None
        if pred.shape != expected.shape or not np.array_equal(pred, expected):
            return None
    return program
