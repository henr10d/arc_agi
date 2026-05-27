"""Brute-force program search over the ARC transformation DSL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from solver.primitives import grids_equal
from solver.program_ir import ProgramNode, execute
from solver.rule_search import try_rule_program


@dataclass
class SearchResult:
    program: ProgramNode
    score: float
    pair_results: List[Dict]


def _score_program(program: ProgramNode, train_pairs: Sequence[dict]) -> Tuple[float, List[Dict]]:
    total = 0
    matched = 0
    pair_results: List[Dict] = []
    for idx, pair in enumerate(train_pairs):
        inp = np.asarray(pair["input"], dtype=np.int64)
        expected = np.asarray(pair["output"], dtype=np.int64)
        try:
            pred = execute(program, inp)
        except Exception as exc:  # noqa: BLE001 - search sandbox
            pair_results.append(
                {
                    "pair_index": idx,
                    "exact_match": False,
                    "error": str(exc),
                    "expected_shape": list(expected.shape),
                }
            )
            continue
        exact = grids_equal(pred, expected)
        cells = expected.size
        total += cells
        if exact:
            matched += cells
        elif pred.shape == expected.shape:
            matched += int(np.sum(pred == expected))
        pair_results.append(
            {
                "pair_index": idx,
                "exact_match": exact,
                "predicted_shape": list(pred.shape),
                "expected_shape": list(expected.shape),
                "mismatch_cells": int(np.sum(pred != expected)) if pred.shape == expected.shape else cells,
            }
        )
    score = matched / total if total else 0.0
    return score, pair_results


def _infer_stencil_factor(train_pairs: Sequence[dict]) -> Optional[int]:
    for pair in train_pairs:
        inp = np.asarray(pair["input"])
        out = np.asarray(pair["output"])
        ih, iw = inp.shape
        oh, ow = out.shape
        if oh % ih == 0 and ow % iw == 0:
            fh, fw = oh // ih, ow // iw
            if fh == fw and fh >= 2:
                return fh
    return None


def _single_primitive_candidates(train_pairs: Sequence[dict]) -> List[ProgramNode]:
    atoms: List[ProgramNode] = [ProgramNode("identity")]

    factor = _infer_stencil_factor(train_pairs)
    if factor is not None:
        atoms.append(ProgramNode("paste_at", {"mode": "stencil", "factor": factor}))
        atoms.append(ProgramNode("tile", {"factor": factor}))
        atoms.append(ProgramNode("scale_nearest", {"factor": factor}))

    for f in (2, 3, 4, 5):
        atoms.append(ProgramNode("tile", {"factor": f}))
        atoms.append(ProgramNode("scale_nearest", {"factor": f}))

    if train_pairs:
        inp0 = np.asarray(train_pairs[0]["input"])
        out0 = np.asarray(train_pairs[0]["output"])
        if inp0.shape == out0.shape:
            mapping: dict = {}
            ok = True
            for a, b in zip(inp0.flatten(), out0.flatten()):
                a, b = int(a), int(b)
                if a in mapping and mapping[a] != b:
                    ok = False
                    break
                mapping[a] = b
            if ok and mapping != {k: k for k in mapping}:
                atoms.append(
                    ProgramNode("color_map", {"mapping": {str(k): v for k, v in mapping.items()}})
                )

    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        atoms.append(ProgramNode("translate", {"dx": dx, "dy": dy}))

    for op in ("mirror_x", "mirror_y", "rotate_90"):
        atoms.append(ProgramNode(op))

    if train_pairs:
        out0 = np.asarray(train_pairs[0]["output"])
        atoms.append(
            ProgramNode(
                "fill_background",
                {"height": int(out0.shape[0]), "width": int(out0.shape[1]), "color": 0},
            )
        )

    return atoms


def _compose_pairs(atoms: List[ProgramNode]) -> Iterable[ProgramNode]:
    for i, first in enumerate(atoms):
        for j, second in enumerate(atoms):
            if i == j:
                continue
            yield ProgramNode("compose", children=[first, second])


def _compose_triples(atoms: List[ProgramNode]) -> Iterable[ProgramNode]:
    for i, first in enumerate(atoms):
        for j, second in enumerate(atoms):
            if i == j:
                continue
            for k, third in enumerate(atoms):
                if k in (i, j):
                    continue
                yield ProgramNode("compose", children=[first, second, third])


def enumerate_candidates(
    train_pairs: Sequence[dict], min_depth: int = 1, max_depth: int = 6
) -> Iterable[ProgramNode]:
    atoms = _single_primitive_candidates(train_pairs)

    if min_depth <= 1:
        for node in atoms:
            yield node

    if max_depth >= 2:
        for node in _compose_pairs(atoms):
            yield node

    if max_depth >= 3:
        for node in _compose_triples(atoms):
            yield node


def search_programs(
    train_pairs: Sequence[dict],
    min_depth: int = 1,
    max_depth: int = 6,
    stop_on_perfect: bool = True,
    verbose: bool = True,
) -> SearchResult:
    rule_program = try_rule_program(train_pairs)
    if rule_program is not None:
        score, pair_results = _score_program(rule_program, train_pairs)
        if verbose:
            print(f"Rule detection: {rule_program.describe()} (score={score:.4f})")
        if score >= 1.0 - 1e-9:
            return SearchResult(program=rule_program, score=score, pair_results=pair_results)

    best: Optional[SearchResult] = None

    for program in enumerate_candidates(train_pairs, min_depth=min_depth, max_depth=max_depth):
        combo_desc = program.describe()
        if verbose:
            print(f"Testing program: {combo_desc}")
        score, pair_results = _score_program(program, train_pairs)
        if verbose:
            print(f"  score={score:.4f}")
            for pr in pair_results:
                status = "PASS" if pr.get("exact_match") else "FAIL"
                detail = pr.get("error") or f"mismatches={pr.get('mismatch_cells', '?')}"
                print(f"  pair {pr['pair_index']}: {status} ({detail})")

        if best is None or score > best.score:
            best = SearchResult(program=program, score=score, pair_results=pair_results)

        if stop_on_perfect and score >= 1.0 - 1e-9:
            if verbose:
                print("Perfect training fit found; stopping search.")
            break

    if best is None:
        raise RuntimeError("No candidate programs generated.")
    return best
