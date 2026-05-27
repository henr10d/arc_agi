"""Export a solved program AST to a static, fully unrolled ONNX graph."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from solver.program_ir import ProgramNode

CANVAS = 30
NUM_COLORS = 10
INPUT_NAME = "input"
OUTPUT_NAME = "output"
ONNX_OPSET = 10
IR_VERSION = 10


def infer_core_size(examples: List[dict]) -> Tuple[int, int]:
    return (
        max(len(ex["input"]) for ex in examples),
        max(len(ex["input"][0]) for ex in examples),
    )


def uniform_input_size(examples: List[dict]) -> bool:
    sizes = {(len(ex["input"]), len(ex["input"][0])) for ex in examples}
    return len(sizes) == 1


def can_export_program(program: ProgramNode) -> bool:
    if program.op == "paste_at":
        return program.params.get("mode") == "stencil"
    if program.op in {
        "identity",
        "mirror_x",
        "mirror_y",
        "rotate_90",
        "translate",
        "tile",
        "scale_nearest",
        "color_map",
        "flood_fill_boundary",
    }:
        return True
    if program.op == "compose":
        return bool(program.children) and all(can_export_program(child) for child in program.children)
    return False


def _resolve_stencil_program(program: ProgramNode) -> Tuple[ProgramNode, int] | None:
    if program.op == "paste_at" and program.params.get("mode") == "stencil":
        return program, int(program.params["factor"])
    if program.op == "compose" and program.children:
        return _resolve_stencil_program(program.children[-1])
    return None


def _core_mask_initializer(core_h: int, core_w: int, name: str = "core_mask") -> onnx.TensorProto:
    mask = np.zeros((1, 1, CANVAS, CANVAS), dtype=np.float32)
    mask[0, 0, :core_h, :core_w] = 1.0
    return numpy_helper.from_array(mask, name=name)


def _masked_input_nodes(
    input_tensor: str,
    core_h: int,
    core_w: int,
    prefix: str = "core",
) -> Tuple[List[onnx.NodeProto], List[onnx.TensorProto], str]:
    mask_name = f"{prefix}_mask"
    inits = [_core_mask_initializer(core_h, core_w, mask_name)]
    masked = f"{prefix}_input"
    nodes = [
        helper.make_node(
            "Mul",
            inputs=[input_tensor, mask_name],
            outputs=[masked],
            name=f"{prefix}_mask_mul",
        )
    ]
    return nodes, inits, masked


def _onehot_from_colors(colors: np.ndarray) -> np.ndarray:
    out = np.zeros((NUM_COLORS, CANVAS, CANVAS), dtype=np.float32)
    for c in range(NUM_COLORS):
        out[c] = (colors[:CANVAS, :CANVAS] == c).astype(np.float32)
    return out


def _colors_from_onehot(onehot: np.ndarray) -> np.ndarray:
    return np.argmax(onehot, axis=0).astype(np.int64)


def _neurogolf_logits_from_grid(grid: List[List[int]] | np.ndarray) -> np.ndarray:
    """Match data/neurogolf_utils.convert_to_numpy encoding."""
    if isinstance(grid, np.ndarray):
        grid = grid.tolist()
    out = np.zeros((1, NUM_COLORS, CANVAS, CANVAS), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def _stencil_onehot(
    onehot: np.ndarray,
    factor: int,
    core_h: int,
    core_w: int,
) -> np.ndarray:
    colors = _colors_from_onehot(onehot)
    masked = np.zeros_like(colors)
    masked[:core_h, :core_w] = colors[:core_h, :core_w]
    colors = masked
    out = np.zeros((CANVAS, CANVAS), dtype=np.int64)
    for r in range(core_h):
        for c in range(core_w):
            if colors[r, c] != 0:
                patch = colors[:core_h, :core_w]
                for pr in range(core_h):
                    for pc in range(core_w):
                        val = patch[pr, pc]
                        if val == 0:
                            continue
                        nr, nc = r * factor + pr, c * factor + pc
                        if nr < CANVAS and nc < CANVAS:
                            out[nr, nc] = val
    return _onehot_from_colors(out)


def evaluate_program_onehot(
    program: ProgramNode,
    onehot: np.ndarray,
    core_h: int,
    core_w: int,
) -> np.ndarray:
    if program.op == "identity":
        return onehot.copy()
    if program.op == "compose":
        state = onehot
        for child in program.children:
            state = evaluate_program_onehot(child, state, core_h, core_w)
        return state
    if program.op == "paste_at" and program.params.get("mode") == "stencil":
        return _stencil_onehot(onehot, int(program.params["factor"]), core_h, core_w)
    raise ValueError(f"Unsupported reference op: {program.op}")


def _make_slice_node(
    name: str,
    input_tensor: str,
    output_tensor: str,
    starts: List[int],
    ends: List[int],
    axes: List[int],
) -> Tuple[onnx.NodeProto, List[onnx.TensorProto]]:
    starts_name = f"{name}_starts"
    ends_name = f"{name}_ends"
    axes_name = f"{name}_axes"
    inits = [
        numpy_helper.from_array(np.array(starts, dtype=np.int64), name=starts_name),
        numpy_helper.from_array(np.array(ends, dtype=np.int64), name=ends_name),
        numpy_helper.from_array(np.array(axes, dtype=np.int64), name=axes_name),
    ]
    node = helper.make_node(
        "Slice",
        inputs=[input_tensor, starts_name, ends_name, axes_name],
        outputs=[output_tensor],
        name=name,
    )
    return node, inits


def _append_slice(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    name: str,
    input_tensor: str,
    output_tensor: str,
    starts: List[int],
    ends: List[int],
    axes: List[int],
) -> None:
    node, slice_inits = _make_slice_node(name, input_tensor, output_tensor, starts, ends, axes)
    nodes.append(node)
    inits.extend(slice_inits)


def _build_translate_subgraph(
    input_tensor: str, dx: int, dy: int, prefix: str
) -> Tuple[List[onnx.NodeProto], List[onnx.TensorProto], str]:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    channel_tensors: List[str] = []

    src_r0 = max(0, -dx)
    src_c0 = max(0, -dy)
    src_r1 = CANVAS - max(0, dx)
    src_c1 = CANVAS - max(0, dy)
    dst_r0 = max(0, dx)
    dst_c0 = max(0, dy)
    dst_r1 = dst_r0 + (src_r1 - src_r0)
    dst_c1 = dst_c0 + (src_c1 - src_c0)
    pads = [0, 0, dst_r0, dst_c0, 0, 0, CANVAS - dst_r1, CANVAS - dst_c1]

    for c in range(NUM_COLORS):
        ch_in = f"{prefix}_in_{c}"
        cropped = f"{prefix}_crop_{c}"
        padded = f"{prefix}_out_{c}"
        _append_slice(
            nodes,
            inits,
            f"{prefix}_slice_{c}",
            input_tensor,
            ch_in,
            [0, c, 0, 0],
            [1, c + 1, CANVAS, CANVAS],
            [0, 1, 2, 3],
        )
        _append_slice(
            nodes,
            inits,
            f"{prefix}_crop_{c}",
            ch_in,
            cropped,
            [0, 0, src_r0, src_c0],
            [1, 1, src_r1, src_c1],
            [0, 1, 2, 3],
        )
        nodes.append(
            helper.make_node(
                "Pad",
                inputs=[cropped],
                outputs=[padded],
                pads=pads,
                name=f"{prefix}_pad_node_{c}",
            )
        )
        channel_tensors.append(padded)

    out = f"{prefix}_translate"
    nodes.append(
        helper.make_node("Concat", inputs=channel_tensors, outputs=[out], axis=1, name=f"{prefix}_concat")
    )
    return nodes, inits, out


def _build_background_fill_subgraph(
    working: str,
    out_h: int,
    out_w: int,
) -> Tuple[List[onnx.NodeProto], List[onnx.TensorProto], str]:
    """Set channel 0 = 1 on background cells inside the output region only."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    region = np.zeros((1, 1, CANVAS, CANVAS), dtype=np.float32)
    region[0, 0, :out_h, :out_w] = 1.0
    inits.append(numpy_helper.from_array(region, name="bg_region"))

    fg_slice = "bg_fg_slice"
    _append_slice(
        nodes,
        inits,
        "bg_fg_slice_node",
        working,
        fg_slice,
        [0, 1, 0, 0],
        [1, NUM_COLORS, CANVAS, CANVAS],
        [0, 1, 2, 3],
    )
    fg_max = "bg_fg_max"
    nodes.append(
        helper.make_node(
            "ReduceMax",
            inputs=[fg_slice],
            outputs=[fg_max],
            axes=[1],
            keepdims=1,
            name="bg_fg_max_node",
        )
    )
    zero_arr = np.zeros((1, 1, CANVAS, CANVAS), dtype=np.float32)
    one_arr = np.ones((1, 1, CANVAS, CANVAS), dtype=np.float32)
    inits.append(numpy_helper.from_array(zero_arr, name="bg_zero"))
    inits.append(numpy_helper.from_array(one_arr, name="bg_one"))
    fg_active = "bg_fg_active"
    nodes.append(
        helper.make_node("Greater", inputs=[fg_max, "bg_zero"], outputs=[fg_active], name="bg_fg_active_node")
    )
    no_fg = "bg_no_fg"
    nodes.append(
        helper.make_node("Not", inputs=[fg_active], outputs=[no_fg], name="bg_not_fg_node")
    )
    in_region = "bg_in_region"
    nodes.append(
        helper.make_node("Greater", inputs=["bg_region", "bg_zero"], outputs=[in_region], name="bg_in_region_node")
    )
    bg_mask = "bg_mask"
    nodes.append(
        helper.make_node("And", inputs=[no_fg, in_region], outputs=[bg_mask], name="bg_mask_node")
    )
    ch0 = "bg_ch0"
    nodes.append(
        helper.make_node("Where", inputs=[bg_mask, "bg_one", "bg_zero"], outputs=[ch0], name="bg_ch0_node")
    )
    tail = "bg_tail"
    _append_slice(
        nodes,
        inits,
        "bg_tail_slice_node",
        working,
        tail,
        [0, 1, 0, 0],
        [1, NUM_COLORS, CANVAS, CANVAS],
        [0, 1, 2, 3],
    )
    out = OUTPUT_NAME
    nodes.append(
        helper.make_node("Concat", inputs=[ch0, tail], outputs=[out], axis=1, name="bg_concat_node")
    )
    return nodes, inits, out


def _build_stencil_graph(
    input_tensor: str,
    factor: int,
    core_h: int,
    core_w: int,
) -> Tuple[List[onnx.NodeProto], List[onnx.TensorProto], str]:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    mask_nodes, mask_inits, masked_input = _masked_input_nodes(input_tensor, core_h, core_w)
    nodes.extend(mask_nodes)
    inits.extend(mask_inits)

    zero_canvas = np.zeros((1, NUM_COLORS, CANVAS, CANVAS), dtype=np.float32)
    canvas_name = "stencil_canvas_init"
    inits.append(numpy_helper.from_array(zero_canvas, name=canvas_name))
    inits.append(numpy_helper.from_array(zero_canvas, name="stencil_float_zero"))
    working = canvas_name

    for r in range(core_h):
        for c in range(core_w):
            slice_nonzero = f"stencil_nz_slice_{r}_{c}"
            _append_slice(
                nodes,
                inits,
                f"stencil_nz_slice_node_{r}_{c}",
                masked_input,
                slice_nonzero,
                [0, 1, r, c],
                [1, NUM_COLORS, r + 1, c + 1],
                [0, 1, 2, 3],
            )
            marker = f"stencil_marker_{r}_{c}"
            nodes.append(
                helper.make_node(
                    "ReduceMax",
                    inputs=[slice_nonzero],
                    outputs=[marker],
                    axes=[1],
                    keepdims=1,
                    name=f"stencil_marker_node_{r}_{c}",
                )
            )

            shift_nodes, shift_inits, shifted = _build_translate_subgraph(
                masked_input, dx=r * factor, dy=c * factor, prefix=f"stencil_{r}_{c}"
            )
            nodes.extend(shift_nodes)
            inits.extend(shift_inits)

            weighted = f"stencil_weighted_{r}_{c}"
            nodes.append(
                helper.make_node(
                    "Mul",
                    inputs=[shifted, marker],
                    outputs=[weighted],
                    name=f"stencil_weighted_node_{r}_{c}",
                )
            )

            gt = f"stencil_gt_{r}_{c}"
            nodes.append(
                helper.make_node(
                    "Greater",
                    inputs=[weighted, "stencil_float_zero"],
                    outputs=[gt],
                    name=f"stencil_gt_node_{r}_{c}",
                )
            )

            merged = f"stencil_merged_{r}_{c}"
            nodes.append(
                helper.make_node(
                    "Where",
                    inputs=[gt, weighted, working],
                    outputs=[merged],
                    name=f"stencil_where_{r}_{c}",
                )
            )
            working = merged

    fill_nodes, fill_inits, _ = _build_background_fill_subgraph(
        working, core_h * factor, core_w * factor
    )
    nodes.extend(fill_nodes)
    inits.extend(fill_inits)
    return nodes, inits, OUTPUT_NAME


def infer_output_size(examples: List[dict]) -> Tuple[int, int]:
    oh = max(len(ex["output"]) for ex in examples)
    ow = max(len(ex["output"][0]) for ex in examples)
    return oh, ow


def infer_output_size_from_program(program: ProgramNode, examples: List[dict]) -> Tuple[int, int]:
    from solver.program_ir import execute

    oh, ow = 0, 0
    for ex in examples:
        pred = execute(program, ex["input"])
        oh = max(oh, pred.shape[0])
        ow = max(ow, pred.shape[1])
    return oh, ow


def _flatten_program(program: ProgramNode) -> List[ProgramNode]:
    if program.op == "compose":
        nodes: List[ProgramNode] = []
        for child in program.children:
            nodes.extend(_flatten_program(child))
        return nodes
    return [program]


def _pad_core_to_canvas(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    tensor: str,
    core_h: int,
    core_w: int,
    prefix: str,
) -> str:
    padded = f"{prefix}_canvas"
    pads = [0, 0, 0, 0, 0, 0, CANVAS - core_h, CANVAS - core_w]
    nodes.append(
        helper.make_node("Pad", inputs=[tensor], outputs=[padded], pads=pads, name=f"{prefix}_pad")
    )
    return padded


def _build_core_slice(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    input_tensor: str,
    core_h: int,
    core_w: int,
    prefix: str,
) -> str:
    core = f"{prefix}_core"
    _append_slice(
        nodes,
        inits,
        f"{prefix}_core_slice",
        input_tensor,
        core,
        [0, 0, 0, 0],
        [1, NUM_COLORS, core_h, core_w],
        [0, 1, 2, 3],
    )
    return core


def _build_gather_flip_core(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    input_tensor: str,
    core_h: int,
    core_w: int,
    flip_h: bool,
    flip_w: bool,
    prefix: str,
) -> Tuple[str, int, int]:
    core = _build_core_slice(nodes, inits, input_tensor, core_h, core_w, prefix)
    working = core
    out_h, out_w = core_h, core_w

    if flip_w:
        idx = np.arange(core_w - 1, -1, -1, dtype=np.int64)
        idx_name = f"{prefix}_flip_w_idx"
        inits.append(numpy_helper.from_array(idx, name=idx_name))
        flipped = f"{prefix}_flip_w"
        nodes.append(
            helper.make_node("Gather", inputs=[working, idx_name], outputs=[flipped], axis=3, name=f"{prefix}_flip_w_node")
        )
        working = flipped

    if flip_h:
        idx = np.arange(core_h - 1, -1, -1, dtype=np.int64)
        idx_name = f"{prefix}_flip_h_idx"
        inits.append(numpy_helper.from_array(idx, name=idx_name))
        flipped = f"{prefix}_flip_h"
        nodes.append(
            helper.make_node("Gather", inputs=[working, idx_name], outputs=[flipped], axis=2, name=f"{prefix}_flip_h_node")
        )
        working = flipped

    if flip_w and flip_h:
        out_h, out_w = core_h, core_w
    elif flip_w or flip_h:
        out_h, out_w = core_h, core_w

    canvas = _pad_core_to_canvas(nodes, inits, working, out_h, out_w, prefix)
    return canvas, out_h, out_w


def _build_rotate90_core(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    input_tensor: str,
    core_h: int,
    core_w: int,
    prefix: str,
) -> Tuple[str, int, int]:
    core = _build_core_slice(nodes, inits, input_tensor, core_h, core_w, prefix)
    transposed = f"{prefix}_transposed"
    nodes.append(
        helper.make_node("Transpose", inputs=[core], outputs=[transposed], perm=[0, 1, 3, 2], name=f"{prefix}_transpose")
    )
    idx = np.arange(core_h - 1, -1, -1, dtype=np.int64)
    idx_name = f"{prefix}_rot_idx"
    inits.append(numpy_helper.from_array(idx, name=idx_name))
    rotated = f"{prefix}_rotated"
    nodes.append(
        helper.make_node("Gather", inputs=[transposed, idx_name], outputs=[rotated], axis=3, name=f"{prefix}_rot_gather")
    )
    out_h, out_w = core_w, core_h
    canvas = _pad_core_to_canvas(nodes, inits, rotated, out_h, out_w, prefix)
    return canvas, out_h, out_w


def _build_translate_core(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    input_tensor: str,
    core_h: int,
    core_w: int,
    dx: int,
    dy: int,
    prefix: str,
) -> Tuple[str, int, int]:
    shift_nodes, shift_inits, shifted = _build_translate_subgraph(
        input_tensor, dx=dx, dy=dy, prefix=prefix
    )
    nodes.extend(shift_nodes)
    inits.extend(shift_inits)
    return shifted, core_h, core_w


def _paste_block_on_canvas(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    block: str,
    row: int,
    col: int,
    block_h: int,
    block_w: int,
    canvas: str,
    zero_name: str,
    prefix: str,
) -> str:
    padded = f"{prefix}_pasted"
    pads = [0, 0, row, col, 0, 0, CANVAS - row - block_h, CANVAS - col - block_w]
    nodes.append(
        helper.make_node("Pad", inputs=[block], outputs=[padded], pads=pads, name=f"{prefix}_paste_pad")
    )
    gt = f"{prefix}_paste_gt"
    nodes.append(
        helper.make_node("Greater", inputs=[padded, zero_name], outputs=[gt], name=f"{prefix}_paste_gt_node")
    )
    merged = f"{prefix}_paste_merged"
    nodes.append(
        helper.make_node("Where", inputs=[gt, padded, canvas], outputs=[merged], name=f"{prefix}_paste_where")
    )
    return merged


def _build_scale_nearest_core(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    input_tensor: str,
    core_h: int,
    core_w: int,
    factor: int,
    prefix: str,
) -> Tuple[str, int, int]:
    """Nearest-neighbor upscale of the top-left core region."""
    core = _build_core_slice(nodes, inits, input_tensor, core_h, core_w, prefix)
    out_h, out_w = core_h * factor, core_w * factor
    zero_canvas = np.zeros((1, NUM_COLORS, CANVAS, CANVAS), dtype=np.float32)
    zero_name = f"{prefix}_scale_zero"
    inits.append(numpy_helper.from_array(zero_canvas, name=zero_name))
    working = zero_name

    for r in range(core_h):
        for c in range(core_w):
            cell = f"{prefix}_cell_{r}_{c}"
            _append_slice(
                nodes,
                inits,
                f"{prefix}_cell_slice_{r}_{c}",
                core,
                cell,
                [0, 0, r, c],
                [1, NUM_COLORS, r + 1, c + 1],
                [0, 1, 2, 3],
            )
            block = f"{prefix}_block_{r}_{c}"
            ones = np.ones((1, NUM_COLORS, factor, factor), dtype=np.float32)
            ones_name = f"{prefix}_ones_{r}_{c}"
            inits.append(numpy_helper.from_array(ones, name=ones_name))
            nodes.append(
                helper.make_node("Mul", inputs=[cell, ones_name], outputs=[block], name=f"{prefix}_block_mul_{r}_{c}")
            )
            working = _paste_block_on_canvas(
                nodes,
                inits,
                block,
                r * factor,
                c * factor,
                factor,
                factor,
                working,
                zero_name,
                f"{prefix}_paste_{r}_{c}",
            )

    return working, out_h, out_w


def _build_color_map_core(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    input_tensor: str,
    mapping: dict,
    core_h: int,
    core_w: int,
    prefix: str,
) -> Tuple[str, int, int]:
    core = _build_core_slice(nodes, inits, input_tensor, core_h, core_w, prefix)
    zero = np.zeros((1, 1, core_h, core_w), dtype=np.float32)
    inits.append(numpy_helper.from_array(zero, name=f"{prefix}_cmap_zero"))
    channel_tensors: List[str] = []

    for dst in range(NUM_COLORS):
        dst_plane = f"{prefix}_cmap_{dst}"
        sources = [int(src) for src, mapped in mapping.items() if int(mapped) == dst]
        if not sources:
            nodes.append(
                helper.make_node("Mul", inputs=[f"{prefix}_cmap_zero", f"{prefix}_cmap_zero"], outputs=[dst_plane], name=f"{prefix}_cmap_empty_{dst}")
            )
        elif len(sources) == 1:
            src = sources[0]
            _append_slice(
                nodes,
                inits,
                f"{prefix}_cmap_slice_{src}",
                core,
                dst_plane,
                [0, src, 0, 0],
                [1, src + 1, core_h, core_w],
                [0, 1, 2, 3],
            )
        else:
            acc = f"{prefix}_cmap_acc_{dst}_0"
            _append_slice(
                nodes,
                inits,
                f"{prefix}_cmap_slice_{dst}_{sources[0]}",
                core,
                acc,
                [0, sources[0], 0, 0],
                [1, sources[0] + 1, core_h, core_w],
                [0, 1, 2, 3],
            )
            working = acc
            for si, src in enumerate(sources[1:], start=1):
                nxt = f"{prefix}_cmap_acc_{dst}_{si}"
                src_t = f"{prefix}_cmap_src_{dst}_{si}"
                _append_slice(
                    nodes,
                    inits,
                    f"{prefix}_cmap_slice_{dst}_{src}_{si}",
                    core,
                    src_t,
                    [0, src, 0, 0],
                    [1, src + 1, core_h, core_w],
                    [0, 1, 2, 3],
                )
                nodes.append(
                    helper.make_node("Add", inputs=[working, src_t], outputs=[nxt], name=f"{prefix}_cmap_add_{dst}_{si}")
                )
                working = nxt
            dst_plane = working
        channel_tensors.append(dst_plane)

    out = f"{prefix}_cmap_out"
    nodes.append(
        helper.make_node("Concat", inputs=channel_tensors, outputs=[out], axis=1, name=f"{prefix}_cmap_concat")
    )
    canvas = _pad_core_to_canvas(nodes, inits, out, core_h, core_w, prefix)
    return canvas, core_h, core_w


def _build_flood_fill_core(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    input_tensor: str,
    core_h: int,
    core_w: int,
    fill_color: int,
    prefix: str,
) -> Tuple[str, int, int]:
    """Fill bbox interior (exclusive border) with fill_color on top-left core."""
    core = _build_core_slice(nodes, inits, input_tensor, core_h, core_w, prefix)
    fg = f"{prefix}_ff_fg"
    _append_slice(
        nodes,
        inits,
        f"{prefix}_ff_fg_slice",
        core,
        fg,
        [0, 1, 0, 0],
        [1, NUM_COLORS, core_h, core_w],
        [0, 1, 2, 3],
    )
    fg_max = f"{prefix}_ff_fg_max"
    nodes.append(
        helper.make_node("ReduceMax", inputs=[fg], outputs=[fg_max], axes=[1], keepdims=1, name=f"{prefix}_ff_max")
    )
    inits.append(numpy_helper.from_array(np.zeros((1, 1, 1, 1), dtype=np.float32), name=f"{prefix}_ff_zero"))
    border_mask = f"{prefix}_ff_border"
    nodes.append(
        helper.make_node("Equal", inputs=[fg_max, f"{prefix}_ff_zero"], outputs=[border_mask], name=f"{prefix}_ff_border_eq")
    )
    interior = np.zeros((1, 1, core_h, core_w), dtype=np.float32)
    if core_h >= 3 and core_w >= 3:
        interior[0, 0, 1 : core_h - 1, 1 : core_w - 1] = 1.0
    inits.append(numpy_helper.from_array(interior, name=f"{prefix}_ff_interior"))
    in_region = f"{prefix}_ff_in"
    nodes.append(
        helper.make_node("Greater", inputs=[f"{prefix}_ff_interior", f"{prefix}_ff_zero"], outputs=[in_region], name=f"{prefix}_ff_in_node")
    )
    fill_mask = f"{prefix}_ff_mask"
    nodes.append(
        helper.make_node("And", inputs=[in_region, border_mask], outputs=[fill_mask], name=f"{prefix}_ff_and")
    )
    fill_one = np.ones((1, 1, core_h, core_w), dtype=np.float32)
    inits.append(numpy_helper.from_array(fill_one, name=f"{prefix}_ff_one"))
    ch_fill = f"{prefix}_ff_ch"
    nodes.append(
        helper.make_node("Where", inputs=[fill_mask, f"{prefix}_ff_one", f"{prefix}_ff_zero"], outputs=[ch_fill], name=f"{prefix}_ff_ch")
    )
    tail = f"{prefix}_ff_tail"
    _append_slice(
        nodes,
        inits,
        f"{prefix}_ff_tail_slice",
        core,
        tail,
        [0, 1, 0, 0],
        [1, NUM_COLORS, core_h, core_w],
        [0, 1, 2, 3],
    )
    remapped = f"{prefix}_ff_remapped"
    nodes.append(
        helper.make_node("Concat", inputs=[ch_fill, tail], outputs=[remapped], axis=1, name=f"{prefix}_ff_cat0")
    )
    # Zero foreground channels inside fill region, then set fill_color channel.
    cleared: List[str] = []
    for c in range(NUM_COLORS):
        ch = f"{prefix}_ff_ch_{c}"
        _append_slice(
            nodes,
            inits,
            f"{prefix}_ff_ch_slice_{c}",
            remapped,
            ch,
            [0, c, 0, 0],
            [1, c + 1, core_h, core_w],
            [0, 1, 2, 3],
        )
        if c == 0:
            cleared_ch = ch_fill  # background channel already handled
        elif c == fill_color:
            filled = f"{prefix}_ff_fill_{c}"
            nodes.append(
                helper.make_node("Where", inputs=[fill_mask, f"{prefix}_ff_one", f"{prefix}_ff_zero"], outputs=[filled], name=f"{prefix}_ff_fillc")
            )
            cleared_ch = filled
        else:
            not_fill = f"{prefix}_ff_nf_{c}"
            nodes.append(
                helper.make_node("Not", inputs=[fill_mask], outputs=[not_fill], name=f"{prefix}_ff_not_{c}")
            )
            kept = f"{prefix}_ff_kept_{c}"
            nodes.append(
                helper.make_node("Mul", inputs=[ch, not_fill], outputs=[kept], name=f"{prefix}_ff_mul_{c}")
            )
            cleared_ch = kept
        cleared.append(cleared_ch)
    merged = f"{prefix}_ff_merged"
    nodes.append(
        helper.make_node("Concat", inputs=cleared, outputs=[merged], axis=1, name=f"{prefix}_ff_merge")
    )
    canvas = _pad_core_to_canvas(nodes, inits, merged, core_h, core_w, prefix)
    return canvas, core_h, core_w


def _apply_program_node(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    input_tensor: str,
    node: ProgramNode,
    core_h: int,
    core_w: int,
    prefix: str,
) -> Tuple[str, int, int]:
    if node.op == "identity":
        canvas = _pad_core_to_canvas(
            nodes, inits, _build_core_slice(nodes, inits, input_tensor, core_h, core_w, prefix), core_h, core_w, prefix
        )
        return canvas, core_h, core_w

    if node.op == "mirror_x":
        return _build_gather_flip_core(nodes, inits, input_tensor, core_h, core_w, False, True, prefix)

    if node.op == "mirror_y":
        return _build_gather_flip_core(nodes, inits, input_tensor, core_h, core_w, True, False, prefix)

    if node.op == "rotate_90":
        return _build_rotate90_core(nodes, inits, input_tensor, core_h, core_w, prefix)

    if node.op == "translate":
        return _build_translate_core(
            nodes, inits, input_tensor, core_h, core_w, int(node.params["dx"]), int(node.params["dy"]), prefix
        )

    if node.op in ("tile", "scale_nearest"):
        factor = int(node.params["factor"])
        return _build_scale_nearest_core(nodes, inits, input_tensor, core_h, core_w, factor, prefix)

    if node.op == "color_map":
        mapping = {int(k): int(v) for k, v in node.params["mapping"].items()}
        return _build_color_map_core(nodes, inits, input_tensor, mapping, core_h, core_w, prefix)

    if node.op == "flood_fill_boundary":
        return _build_flood_fill_core(
            nodes, inits, input_tensor, core_h, core_w, int(node.params.get("fill_color", 4)), prefix
        )

    raise ValueError(f"Unsupported ONNX program op: {node.op}")


def _build_program_graph(
    program: ProgramNode,
    core_h: int,
    core_w: int,
    out_h: int,
    out_w: int,
) -> Tuple[List[onnx.NodeProto], List[onnx.TensorProto], str]:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    working = INPUT_NAME
    h, w = core_h, core_w

    for idx, node in enumerate(_flatten_program(program)):
        working, h, w = _apply_program_node(nodes, inits, working, node, h, w, f"prog_{idx}")

    fill_nodes, fill_inits, _ = _build_background_fill_subgraph(working, out_h, out_w)
    nodes.extend(fill_nodes)
    inits.extend(fill_inits)
    return nodes, inits, OUTPUT_NAME


def _build_identity_graph() -> Tuple[List[onnx.NodeProto], List[onnx.TensorProto], str]:
    nodes = [helper.make_node("Identity", inputs=[INPUT_NAME], outputs=[OUTPUT_NAME], name="identity")]
    return nodes, [], OUTPUT_NAME


def export_program_to_onnx(
    program: ProgramNode,
    output_path: str,
    examples: List[dict] | None = None,
) -> None:
    """Export program to ONNX with fixed 1x10x30x30 one-hot tensors."""
    core_h, core_w = infer_core_size(examples) if examples else (3, 3)
    resolved = _resolve_stencil_program(program)
    if resolved is not None:
        _, factor = resolved
        nodes, inits, _ = _build_stencil_graph(INPUT_NAME, factor, core_h, core_w)
    elif program.op == "identity":
        nodes, inits, _ = _build_identity_graph()
    elif examples is not None:
        out_h, out_w = infer_output_size_from_program(program, examples)
        nodes, inits, _ = _build_program_graph(program, core_h, core_w, out_h, out_w)
    else:
        nodes, inits, _ = _build_identity_graph()

    X = helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, [1, NUM_COLORS, CANVAS, CANVAS])
    Y = helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, [1, NUM_COLORS, CANVAS, CANVAS])
    graph = helper.make_graph(nodes, "arc_program", [X], [Y], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="arc_synth_experiment",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", ONNX_OPSET)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, output_path)


def validate_onnx_matches_program(
    program: ProgramNode, onnx_path: str, examples: List[dict]
) -> bool:
    """Validate using NeuroGolf-style (>0) logits comparison on all examples."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    for example in examples:
        inp = _neurogolf_logits_from_grid(example["input"])
        expected = _neurogolf_logits_from_grid(example["output"])
        result = sess.run([OUTPUT_NAME], {INPUT_NAME: inp})[0]
        pred = (result > 0.0).astype(np.float32)
        if not np.array_equal(pred, expected):
            return False
    return True
