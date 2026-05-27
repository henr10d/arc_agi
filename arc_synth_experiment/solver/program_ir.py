"""Simple AST for grid transformation programs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import numpy as np

from solver.primitives import (
    Grid,
    bounding_box,
    color_map,
    crop_bbox,
    fill_background,
    find_connected_components,
    flood_fill_boundary,
    largest_component,
    mask_component,
    mirror_x,
    mirror_y,
    paste_at,
    paste_stencil,
    replace_color,
    rotate_90,
    scale_nearest,
    tile_grid,
    translate,
)


@dataclass
class ProgramNode:
    op: str
    params: Dict[str, Any] = field(default_factory=dict)
    children: List["ProgramNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "params": self.params,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProgramNode":
        return cls(
            op=data["op"],
            params=dict(data.get("params", {})),
            children=[cls.from_dict(c) for c in data.get("children", [])],
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "ProgramNode":
        return cls.from_dict(json.loads(text))

    def describe(self) -> str:
        if self.params:
            param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
            head = f"{self.op}({param_str})"
        else:
            head = self.op
        if not self.children:
            return head
        child_str = " -> ".join(c.describe() for c in self.children)
        return f"{head}[{child_str}]"


def execute(node: ProgramNode, grid: Union[Grid, List[List[int]]]) -> Grid:
    """Evaluate a program node on an input grid."""
    g = np.asarray(grid, dtype=np.int64)

    if node.op == "identity":
        return g.copy()

    if node.op == "compose":
        state = g
        for child in node.children:
            state = execute(child, state)
        return state

    if node.op == "find_connected_components":
        return np.array([len(find_connected_components(g))], dtype=np.int64)

    if node.op == "largest_component":
        comp = largest_component(g)
        if comp is None:
            return np.zeros_like(g)
        return mask_component(comp, g.shape)

    if node.op == "bounding_box":
        comp = largest_component(g)
        if comp is None:
            return np.zeros((1, 4), dtype=np.int64)
        return np.array([bounding_box(comp)], dtype=np.int64)

    if node.op == "mask_component":
        comp = largest_component(g)
        if comp is None:
            return np.zeros_like(g)
        color = node.params.get("color")
        return mask_component(comp, g.shape, color=color)

    if node.op == "translate":
        return translate(g, int(node.params["dx"]), int(node.params["dy"]))

    if node.op == "mirror_x":
        return mirror_x(g)

    if node.op == "mirror_y":
        return mirror_y(g)

    if node.op == "rotate_90":
        return rotate_90(g)

    if node.op == "crop_bbox":
        comp = largest_component(g)
        if comp is None:
            return np.zeros((1, 1), dtype=np.int64)
        return crop_bbox(g, bounding_box(comp))

    if node.op == "paste_at":
        if node.params.get("mode") == "stencil":
            return paste_stencil(g, int(node.params["factor"]))
        if len(node.children) != 2:
            raise ValueError("paste_at requires two children unless mode=stencil")
        canvas = execute(node.children[0], g)
        patch = execute(node.children[1], g)
        return paste_at(canvas, patch, int(node.params["dx"]), int(node.params["dy"]))

    if node.op == "color_map":
        mapping = {int(k): int(v) for k, v in node.params["mapping"].items()}
        return color_map(g, mapping)

    if node.op == "tile":
        return tile_grid(g, int(node.params["factor"]))

    if node.op == "scale_nearest":
        return scale_nearest(g, int(node.params["factor"]))

    if node.op == "flood_fill_boundary":
        return flood_fill_boundary(g, int(node.params.get("fill_color", 4)))

    if node.op == "fill_background":
        return fill_background(
            int(node.params["height"]),
            int(node.params["width"]),
            int(node.params.get("color", 0)),
        )

    if node.op == "replace_color":
        return replace_color(g, int(node.params["old_color"]), int(node.params["new_color"]))

    if node.op == "conditional_if":
        if len(node.children) != 3:
            raise ValueError("conditional_if requires condition, then-branch, else-branch")
        cond = execute(node.children[0], g)
        then_out = execute(node.children[1], g)
        else_out = execute(node.children[2], g)
        if not (then_out.shape == else_out.shape == cond.shape):
            raise ValueError("conditional_if branches must share shape")
        mask = cond != 0
        out = else_out.copy()
        out[mask] = then_out[mask]
        return out

    raise ValueError(f"Unknown op: {node.op}")
