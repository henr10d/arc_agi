#!/usr/bin/env python3
"""Minimal mask-gated 3x3 tiling ONNX graphs (generic + NeuroGolf task001)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parent

# Generic educational model
GENERIC_OPSET = 13
GENERIC_IR = 8
INPUT_X = "X"
INPUT_M = "M"
OUTPUT_Y = "Y"

# NeuroGolf / ARC task001 (stencil paste at nonzero cells)
NEUROGOLF_OPSET = 10
NEUROGOLF_IR = 10
NG_INPUT = "input"
NG_OUTPUT = "output"
NG_CHANNELS = 10
NG_CANVAS = 30
NG_CORE = 3
NG_OUT = NG_CORE * 3


def _int64_tensor(name: str, values: List[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.array(values, dtype=np.int64), name=name)


_HW_AXES = [0, 1, 2, 3]


def _slice_node(
    name: str,
    input_tensor: str,
    output_tensor: str,
    starts: List[int],
    ends: List[int],
    axes: List[int] | None = None,
    *,
    axes_name: str = "slice_axes",
    shared_axes: bool = False,
) -> Tuple[onnx.NodeProto, List[onnx.TensorProto]]:
    if axes is None:
        axes = _HW_AXES
    starts_name = f"{name}_starts"
    ends_name = f"{name}_ends"
    inits = [
        _int64_tensor(starts_name, starts),
        _int64_tensor(ends_name, ends),
    ]
    if not shared_axes:
        inits.append(_int64_tensor(f"{name}_axes", axes))
        axes_tensor = f"{name}_axes"
    else:
        axes_tensor = axes_name
    node = helper.make_node(
        "Slice",
        [input_tensor, starts_name, ends_name, axes_tensor],
        [output_tensor],
        name=name,
    )
    return node, inits


def _graph_stats(model: onnx.ModelProto) -> Tuple[int, int, int]:
    """Return node count, initializer element count, and serialized byte size."""
    param_count = sum(int(np.prod(list(init.dims))) for init in model.graph.initializer)
    raw = model.SerializeToString()
    return len(model.graph.node), param_count, len(raw)


def reference_tile_mask(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Reference: tile X 3x3 and gate each tile by M[i, j]."""
    _, _, H, W = X.shape
    tiled = np.tile(X, (1, 1, 3, 3))
    m6 = M.reshape(1, 1, 3, 1, 3, 1)
    mask = np.broadcast_to(m6, (1, 1, 3, H, 3, W)).reshape(1, 1, 3 * H, 3 * W)
    return tiled * mask


def build_model() -> onnx.ModelProto:
    """Generic 2-input graph with dynamic H/W (NOT NeuroGolf-submittable)."""
    x_info = helper.make_tensor_value_info(
        INPUT_X, TensorProto.FLOAT, [1, "C", "H", "W"]
    )
    m_info = helper.make_tensor_value_info(
        INPUT_M, TensorProto.FLOAT, [1, 1, 3, 3]
    )
    y_info = helper.make_tensor_value_info(
        OUTPUT_Y, TensorProto.FLOAT, [1, "C", None, None]
    )

    inits = [
        _int64_tensor("repeats_x", [1, 1, 3, 3]),
        _int64_tensor("m6_shape", [1, 1, 3, 1, 3, 1]),
        _int64_tensor("repeats_prefix", [1, 1, 1]),
        _int64_tensor("one", [1]),
        _int64_tensor("three", [3]),
        _int64_tensor("out_shape_prefix", [1, 1]),
        _int64_tensor("idx_h", [2]),
        _int64_tensor("idx_w", [3]),
    ]

    nodes = [
        helper.make_node("Tile", [INPUT_X, "repeats_x"], ["tiled_x"], name="tile_x"),
        helper.make_node("Reshape", [INPUT_M, "m6_shape"], ["m6"], name="reshape_m"),
        helper.make_node("Shape", [INPUT_X], ["x_shape"], name="shape_x"),
        helper.make_node("Gather", ["x_shape", "idx_h"], ["h"], axis=0, name="gather_h"),
        helper.make_node("Gather", ["x_shape", "idx_w"], ["w"], axis=0, name="gather_w"),
        helper.make_node(
            "Concat",
            ["repeats_prefix", "h", "one", "w"],
            ["repeats_m6"],
            axis=0,
            name="concat_repeats_m6",
        ),
        helper.make_node("Tile", ["m6", "repeats_m6"], ["mask6"], name="tile_m6"),
        helper.make_node("Mul", ["h", "three"], ["h3"], name="mul_h3"),
        helper.make_node("Mul", ["w", "three"], ["w3"], name="mul_w3"),
        helper.make_node(
            "Concat",
            ["out_shape_prefix", "h3", "w3"],
            ["mask_shape"],
            axis=0,
            name="concat_mask_shape",
        ),
        helper.make_node("Reshape", ["mask6", "mask_shape"], ["mask"], name="reshape_mask"),
        helper.make_node("Mul", ["tiled_x", "mask"], [OUTPUT_Y], name="apply_mask"),
    ]

    graph = helper.make_graph(
        nodes, "tile_mask", [x_info, m_info], [y_info], initializer=inits
    )
    model = helper.make_model(
        graph,
        producer_name="tile_mask",
        ir_version=GENERIC_IR,
        opset_imports=[helper.make_opsetid("", GENERIC_OPSET)],
    )
    onnx.checker.check_model(model)
    return model


def build_baseline_neurogolf_task001_model() -> onnx.ModelProto:
    """Original Expand-based graph kept for equivalence checks."""
    shape = [1, NG_CHANNELS, NG_CANVAS, NG_CANVAS]
    x_info = helper.make_tensor_value_info(NG_INPUT, TensorProto.FLOAT, shape)
    y_info = helper.make_tensor_value_info(NG_OUTPUT, TensorProto.FLOAT, shape)

    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    core_slice, slice_inits = _slice_node(
        "core_slice",
        NG_INPUT,
        "core",
        [0, 0, 0, 0],
        [1, NG_CHANNELS, NG_CORE, NG_CORE],
    )
    nodes.append(core_slice)
    inits.extend(slice_inits)

    fg_slice, fg_inits = _slice_node(
        "fg_slice",
        "core",
        "fg",
        [0, 1, 0, 0],
        [1, NG_CHANNELS, NG_CORE, NG_CORE],
    )
    nodes.append(fg_slice)
    inits.extend(fg_inits)

    nodes.append(
        helper.make_node(
            "ReduceMax",
            ["fg"],
            ["marker"],
            axes=[1],
            keepdims=1,
            name="fg_marker",
        )
    )

    inits.extend(
        [
            _int64_tensor("core6_shape", [1, NG_CHANNELS, 1, NG_CORE, 1, NG_CORE]),
            _int64_tensor(
                "core_exp_shape", [1, NG_CHANNELS, NG_CORE, NG_CORE, NG_CORE, NG_CORE]
            ),
            _int64_tensor("tiled_core_shape", [1, NG_CHANNELS, NG_OUT, NG_OUT]),
            _int64_tensor("m6_shape", [1, 1, NG_CORE, 1, NG_CORE, 1]),
            _int64_tensor("mask_exp_shape", [1, 1, NG_CORE, NG_CORE, NG_CORE, NG_CORE]),
            _int64_tensor("mask_shape", [1, 1, NG_OUT, NG_OUT]),
        ]
    )

    nodes.extend(
        [
            helper.make_node(
                "Reshape", ["core", "core6_shape"], ["core6"], name="reshape_core6"
            ),
            helper.make_node(
                "Expand", ["core6", "core_exp_shape"], ["core_exp"], name="expand_core"
            ),
            helper.make_node(
                "Reshape",
                ["core_exp", "tiled_core_shape"],
                ["tiled_core"],
                name="reshape_tiled_core",
            ),
            helper.make_node("Reshape", ["marker", "m6_shape"], ["m6"], name="reshape_m"),
            helper.make_node(
                "Expand", ["m6", "mask_exp_shape"], ["mask6"], name="expand_mask"
            ),
            helper.make_node(
                "Reshape", ["mask6", "mask_shape"], ["mask"], name="reshape_mask"
            ),
            helper.make_node(
                "Mul", ["tiled_core", "mask"], ["out_core"], name="apply_mask"
            ),
        ]
    )

    pad_bottom = NG_CANVAS - NG_OUT
    nodes.append(
        helper.make_node(
            "Pad",
            ["out_core"],
            ["padded"],
            name="pad_canvas",
            pads=[0, 0, 0, 0, 0, 0, pad_bottom, pad_bottom],
        )
    )

    region = np.zeros((1, 1, NG_CANVAS, NG_CANVAS), dtype=np.float32)
    region[0, 0, :NG_OUT, :NG_OUT] = 1.0
    inits.append(numpy_helper.from_array(region, name="out_region"))
    inits.append(
        numpy_helper.from_array(np.zeros((1, 1, 1, 1), dtype=np.float32), name="zero")
    )
    inits.append(
        numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name="one")
    )

    fg_out_slice, fg_out_inits = _slice_node(
        "fg_out_slice",
        "padded",
        "fg_out",
        [0, 1, 0, 0],
        [1, NG_CHANNELS, NG_CANVAS, NG_CANVAS],
    )
    nodes.append(fg_out_slice)
    inits.extend(fg_out_inits)
    nodes.extend(
        [
            helper.make_node(
                "ReduceMax",
                ["fg_out"],
                ["fg_out_max"],
                axes=[1],
                keepdims=1,
                name="fg_out_max",
            ),
            helper.make_node(
                "Greater", ["fg_out_max", "zero"], ["fg_active"], name="fg_active"
            ),
            helper.make_node("Not", ["fg_active"], ["no_fg"], name="no_fg"),
            helper.make_node(
                "Greater", ["out_region", "zero"], ["in_region"], name="in_region"
            ),
            helper.make_node("And", ["no_fg", "in_region"], ["bg_mask"], name="bg_mask"),
        ]
    )

    ch0_slice, ch0_inits = _slice_node(
        "ch0_slice",
        "padded",
        "ch0",
        [0, 0, 0, 0],
        [1, 1, NG_CANVAS, NG_CANVAS],
    )
    nodes.append(ch0_slice)
    inits.extend(ch0_inits)
    nodes.append(
        helper.make_node("Where", ["bg_mask", "one", "ch0"], ["ch0_out"], name="ch0_fill")
    )

    tail_slice, tail_inits = _slice_node(
        "tail_slice",
        "padded",
        "tail",
        [0, 1, 0, 0],
        [1, NG_CHANNELS, NG_CANVAS, NG_CANVAS],
    )
    nodes.append(tail_slice)
    inits.extend(tail_inits)
    nodes.append(
        helper.make_node(
            "Concat", ["ch0_out", "tail"], [NG_OUTPUT], axis=1, name="merge_output"
        )
    )

    graph = helper.make_graph(
        nodes, "task001_stencil_baseline", [x_info], [y_info], initializer=inits
    )
    model = helper.make_model(
        graph,
        producer_name="tile_mask_neurogolf_baseline",
        ir_version=NEUROGOLF_IR,
        opset_imports=[helper.make_opsetid("", NEUROGOLF_OPSET)],
    )
    onnx.checker.check_model(model)
    return model


def build_optimized_model() -> onnx.ModelProto:
    """
    Minimal NeuroGolf task001 stencil graph:
      - broadcast tiling via Reshape + Tile (no per-cell unrolling)
      - mask/core fused Mul in 6D before a single output Reshape
      - compact 9x9 region mask (18 params vs 900)
      - shared slice axes + fused scalar constants
    """
    shape = [1, NG_CHANNELS, NG_CANVAS, NG_CANVAS]
    x_info = helper.make_tensor_value_info(NG_INPUT, TensorProto.FLOAT, shape)
    y_info = helper.make_tensor_value_info(NG_OUTPUT, TensorProto.FLOAT, shape)

    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = [_int64_tensor("slice_axes", _HW_AXES)]

    core_slice, core_inits = _slice_node(
        "core_slice",
        NG_INPUT,
        "core",
        [0, 0, 0, 0],
        [1, NG_CHANNELS, NG_CORE, NG_CORE],
        shared_axes=True,
    )
    nodes.append(core_slice)
    inits.extend(core_inits)

    fg_slice, fg_inits = _slice_node(
        "fg_slice",
        "core",
        "fg",
        [0, 1, 0, 0],
        [1, NG_CHANNELS, NG_CORE, NG_CORE],
        shared_axes=True,
    )
    nodes.append(fg_slice)
    inits.extend(fg_inits)

    nodes.append(
        helper.make_node(
            "ReduceMax",
            ["fg"],
            ["marker"],
            axes=[1],
            keepdims=1,
            name="fg_marker",
        )
    )

    inits.extend(
        [
            _int64_tensor("core6_shape", [1, NG_CHANNELS, 1, NG_CORE, 1, NG_CORE]),
            _int64_tensor("core_tile_repeats", [1, 1, NG_CORE, 1, NG_CORE, 1]),
            _int64_tensor("m6_shape", [1, 1, NG_CORE, 1, NG_CORE, 1]),
            _int64_tensor("mask_tile_repeats", [1, 1, 1, NG_CORE, 1, NG_CORE]),
            _int64_tensor("out9_shape", [1, NG_CHANNELS, NG_OUT, NG_OUT]),
            _int64_tensor("mask9_shape", [1, 1, NG_OUT, NG_OUT]),
        ]
    )

    nodes.extend(
        [
            helper.make_node(
                "Reshape", ["core", "core6_shape"], ["core6"], name="reshape_core6"
            ),
            helper.make_node(
                "Tile", ["core6", "core_tile_repeats"], ["core_tiled"], name="tile_core"
            ),
            helper.make_node(
                "Reshape", ["core_tiled", "out9_shape"], ["tiled_core"], name="reshape_tiled_core"
            ),
            helper.make_node("Reshape", ["marker", "m6_shape"], ["m6"], name="reshape_m"),
            helper.make_node(
                "Tile", ["m6", "mask_tile_repeats"], ["mask_tiled"], name="tile_mask"
            ),
            helper.make_node(
                "Reshape", ["mask_tiled", "mask9_shape"], ["mask"], name="reshape_mask"
            ),
            helper.make_node(
                "Mul", ["tiled_core", "mask"], ["out_core"], name="apply_mask"
            ),
        ]
    )

    pad_bottom = NG_CANVAS - NG_OUT
    nodes.append(
        helper.make_node(
            "Pad",
            ["out_core"],
            ["padded"],
            name="pad_canvas",
            pads=[0, 0, 0, 0, 0, 0, pad_bottom, pad_bottom],
        )
    )

    inits.extend(
        [
            numpy_helper.from_array(
                np.pad(
                    np.ones((1, 1, NG_OUT, 1), dtype=np.float32),
                    ((0, 0), (0, 0), (0, pad_bottom), (0, 0)),
                ),
                name="row_mask",
            ),
            numpy_helper.from_array(
                np.pad(
                    np.ones((1, 1, 1, NG_OUT), dtype=np.float32),
                    ((0, 0), (0, 0), (0, 0), (0, pad_bottom)),
                ),
                name="col_mask",
            ),
            numpy_helper.from_array(np.zeros((1, 1, 1, 1), dtype=np.float32), name="zero"),
            numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name="one"),
        ]
    )

    fg_out_slice, fg_out_inits = _slice_node(
        "fg_out_slice",
        "padded",
        "fg_out",
        [0, 1, 0, 0],
        [1, NG_CHANNELS, NG_CANVAS, NG_CANVAS],
        shared_axes=True,
    )
    nodes.append(fg_out_slice)
    inits.extend(fg_out_inits)
    nodes.extend(
        [
            helper.make_node(
                "ReduceMax",
                ["fg_out"],
                ["fg_out_max"],
                axes=[1],
                keepdims=1,
                name="fg_out_max",
            ),
            helper.make_node(
                "Greater", ["fg_out_max", "zero"], ["fg_active"], name="fg_active"
            ),
            helper.make_node("Not", ["fg_active"], ["no_fg"], name="no_fg"),
            helper.make_node(
                "Mul", ["row_mask", "col_mask"], ["region_prod"], name="region_prod"
            ),
            helper.make_node(
                "Greater", ["region_prod", "zero"], ["in_region"], name="in_region"
            ),
            helper.make_node("And", ["no_fg", "in_region"], ["bg_mask"], name="bg_mask"),
        ]
    )

    ch0_slice, ch0_inits = _slice_node(
        "ch0_slice",
        "padded",
        "ch0",
        [0, 0, 0, 0],
        [1, 1, NG_CANVAS, NG_CANVAS],
        shared_axes=True,
    )
    nodes.append(ch0_slice)
    inits.extend(ch0_inits)
    nodes.append(
        helper.make_node("Where", ["bg_mask", "one", "ch0"], ["ch0_out"], name="ch0_fill")
    )

    tail_slice, tail_inits = _slice_node(
        "tail_slice",
        "padded",
        "tail",
        [0, 1, 0, 0],
        [1, NG_CHANNELS, NG_CANVAS, NG_CANVAS],
        shared_axes=True,
    )
    nodes.append(tail_slice)
    inits.extend(tail_inits)
    nodes.append(
        helper.make_node(
            "Concat", ["ch0_out", "tail"], [NG_OUTPUT], axis=1, name="merge_output"
        )
    )

    graph = helper.make_graph(
        nodes, "task001_stencil", [x_info], [y_info], initializer=inits
    )
    model = helper.make_model(
        graph,
        producer_name="tile_mask_neurogolf",
        ir_version=NEUROGOLF_IR,
        opset_imports=[helper.make_opsetid("", NEUROGOLF_OPSET)],
    )
    onnx.checker.check_model(model)
    return model


def build_neurogolf_task001_model() -> onnx.ModelProto:
    """NeuroGolf-compatible task001 stencil (optimized builder)."""
    return build_optimized_model()


def save_generic_model(path: str = "tile_mask_model.onnx") -> str:
    onnx.save(build_model(), path)
    return str(path)


def save_model(path: str | Path = ROOT / "submission" / "task001.onnx") -> str:
    """Export the optimized NeuroGolf task001 model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_optimized_model(), str(path))
    return str(path)


def save_neurogolf_task001(path: str | Path = ROOT / "submission" / "task001.onnx") -> str:
    return save_model(path)


def validate_equivalence(
    num_random: int = 32,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    """Compare optimized graph output against the baseline builder."""
    baseline = build_baseline_neurogolf_task001_model()
    optimized = build_optimized_model()

    base_nodes, base_params, base_bytes = _graph_stats(baseline)
    opt_nodes, opt_params, opt_bytes = _graph_stats(optimized)
    node_delta = base_nodes - opt_nodes
    param_delta = base_params - opt_params
    byte_delta = base_bytes - opt_bytes

    print("Graph optimization summary:")
    print(f"  nodes:        {base_nodes} -> {opt_nodes} ({node_delta:+d})")
    print(f"  initializers: {base_params} -> {opt_params} ({param_delta:+d} elements)")
    print(f"  serialized:   {base_bytes} -> {opt_bytes} ({byte_delta:+d} bytes)")
    if base_params:
        print(f"  param reduction: {100.0 * param_delta / base_params:.1f}%")
    if base_bytes:
        print(f"  size reduction:  {100.0 * byte_delta / base_bytes:.1f}%")

    baseline_path = ROOT / ".tmp_baseline_task001.onnx"
    optimized_path = ROOT / ".tmp_optimized_task001.onnx"
    onnx.save(baseline, str(baseline_path))
    onnx.save(optimized, str(optimized_path))

    base_sess = ort.InferenceSession(
        str(baseline_path), providers=["CPUExecutionProvider"]
    )
    opt_sess = ort.InferenceSession(
        str(optimized_path), providers=["CPUExecutionProvider"]
    )

    inputs: List[np.ndarray] = []

    with (ROOT / "data" / "task001.json").open(encoding="utf-8") as fh:
        examples = json.load(fh)
    sys.path.insert(0, str(ROOT / "data" / "neurogolf_utils"))
    from neurogolf_utils import convert_to_numpy

    for split in ("train", "test", "arc-gen"):
        for example in examples[split]:
            encoded = convert_to_numpy(example)
            if encoded is None:
                raise AssertionError(f"failed to encode example in {split}")
            inputs.append(encoded["input"])

    rng = np.random.default_rng(0)
    for _ in range(num_random):
        grid = np.zeros((NG_CANVAS, NG_CANVAS), dtype=np.int64)
        for r in range(NG_CORE):
            for c in range(NG_CORE):
                grid[r, c] = int(rng.integers(0, NG_CHANNELS))
        onehot = np.zeros((1, NG_CHANNELS, NG_CANVAS, NG_CANVAS), dtype=np.float32)
        for r in range(NG_CANVAS):
            for c in range(NG_CANVAS):
                onehot[0, grid[r, c], r, c] = 1.0
        inputs.append(onehot)

    for idx, x in enumerate(inputs):
        expected = base_sess.run([NG_OUTPUT], {NG_INPUT: x})[0]
        actual = opt_sess.run([NG_OUTPUT], {NG_INPUT: x})[0]
        if expected.shape != actual.shape:
            raise AssertionError(f"case {idx}: shape {actual.shape} != {expected.shape}")
        if not np.allclose(actual, expected, rtol=rtol, atol=atol):
            diff = np.max(np.abs(actual - expected))
            raise AssertionError(f"case {idx}: max abs diff {diff}")

    baseline_path.unlink(missing_ok=True)
    optimized_path.unlink(missing_ok=True)
    print(f"validate_equivalence: OK ({len(inputs)} inputs, exact match)")


def verify_model(path: str = "tile_mask_model.onnx") -> None:
    if not os.path.isfile(path):
        save_generic_model(path)

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)

    cases: List[Tuple[int, int, int]] = [(1, 4, 5), (3, 7, 2), (10, 3, 3)]
    for C, H, W in cases:
        X = rng.standard_normal((1, C, H, W), dtype=np.float32)
        M = (rng.random((1, 1, 3, 3)) > 0.5).astype(np.float32)

        expected = reference_tile_mask(X, M)
        actual = sess.run([OUTPUT_Y], {INPUT_X: X, INPUT_M: M})[0]

        if actual.shape != expected.shape:
            raise AssertionError(
                f"shape mismatch for C={C}, H={H}, W={W}: {actual.shape} != {expected.shape}"
            )
        if not np.allclose(actual, expected):
            raise AssertionError(f"value mismatch for C={C}, H={H}, W={W}")

    size_kb = os.path.getsize(path) / 1024
    node_count = len(onnx.load(path).graph.node)
    print(f"verify_model: OK ({node_count} nodes, {size_kb:.2f} KB)")


def verify_neurogolf(path: str | Path) -> None:
    """Run local NeuroGolf + ARC task001 checks."""
    path = Path(path)
    if not path.is_file():
        save_neurogolf_task001(path)

    sys.path.insert(0, str(ROOT / "data" / "neurogolf_utils"))

    import onnxruntime as ort
    from neurogolf_utils import calculate_params, check_network, sanitize_model, verify_subset

    if not check_network(str(path)):
        raise AssertionError(f"check_network failed for {path}")

    model = sanitize_model(onnx.load(str(path)))
    if model is None:
        raise AssertionError("sanitize_model failed")

    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise AssertionError("NeuroGolf requires exactly one input and one output")

    for tensor in list(model.graph.input) + list(model.graph.output):
        for dim in tensor.type.tensor_type.shape.dim:
            if dim.HasField("dim_param") or not dim.HasField("dim_value"):
                raise AssertionError(f"dynamic shape on {tensor.name}")
            if dim.dim_value <= 0:
                raise AssertionError(f"invalid shape on {tensor.name}")

    if model.graph.input[0].name != NG_INPUT or model.graph.output[0].name != NG_OUTPUT:
        raise AssertionError('I/O must be named "input" and "output"')

    params = calculate_params(model)
    if params is None:
        raise AssertionError("calculate_params failed")

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with (ROOT / "data" / "task001.json").open(encoding="utf-8") as fh:
        examples = json.load(fh)

    for split in ("train", "test", "arc-gen"):
        right, wrong, _ = verify_subset(sess, examples[split])
        print(f"  NeuroGolf {split}: {right} pass, {wrong} fail")
        if wrong:
            raise AssertionError(f"NeuroGolf scoring failed on {split} ({wrong} wrong)")

    sys.path.insert(0, str(ROOT))
    from train_arc import validate_task_onnx

    ok, _ = validate_task_onnx("task001", path, save_viz=False)
    if not ok:
        raise AssertionError("task001 grid validation failed")

    size_kb = path.stat().st_size / 1024
    print(
        f"verify_neurogolf: OK ({len(model.graph.node)} nodes, "
        f"{params} params, {size_kb:.2f} KB)"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build tile+mask ONNX models")
    parser.add_argument(
        "--neurogolf",
        action="store_true",
        help="Build NeuroGolf-compatible submission/task001.onnx",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (default: tile_mask_model.onnx or submission/task001.onnx)",
    )
    args = parser.parse_args()

    if args.neurogolf:
        validate_equivalence()
        out = save_neurogolf_task001(
            args.out or str(ROOT / "submission" / "task001.onnx")
        )
        verify_neurogolf(out)
    else:
        out = save_generic_model(args.out or "tile_mask_model.onnx")
        verify_model(out)
        print(
            "\nNote: tile_mask_model.onnx is NOT valid for NeuroGolf submission "
            "(2 inputs, dynamic shapes). Use: python build_tile_mask_model.py --neurogolf"
        )
