"""Minimal ONNX for ARC task002: 4-neighbor outside flood fill.

Task rule: preserve all original non-zero cells. A background cell (color 0)
becomes color 4 iff it is not 4-neighbor connected to the grid border through
other color-0 cells. Padding outside the task grid stays all-zero.

ONNX approach: unroll exterior flood fill from the 30×30 border through zero
cells (25 iterations minimum), then move enclosed zeros from channel 0 to channel 4.
Float variant uses Conv→Mul dilation (center-tap plus kernel). Traversable cells
need color-0 OR all-zero padding (ReduceMax mask); iz=ch0 alone fails task JSON.
Optimized mask uses bool Or + Cast instead of Cast/Max. Bool Pad/Slice variant kept
but excluded from benchmark (higher memory).
"""

from __future__ import annotations

import json
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))
DATA_PATH = ROOT / "data" / "task002.json"
BEST_PATH = OUT_DIR / "task002.onnx"

C = 10
H = W = 30
FILL_COLOR = 4
MAX_ITERS = 62
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def solve(x: np.ndarray, fill: int = FILL_COLOR) -> np.ndarray:
    """Reference: 4-neighbor flood fill from border through zero cells."""
    g = np.asarray(x, dtype=np.int64)
    if g.ndim == 4:
        g = g[0, 0]
    elif g.ndim == 3:
        g = g[0]
    h, w = g.shape
    outside = np.zeros((h, w), dtype=bool)
    stack: List[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if g[r, c] == 0 and (r == 0 or r == h - 1 or c == 0 or c == w - 1):
                outside[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not outside[nr, nc] and g[nr, nc] == 0:
                outside[nr, nc] = True
                stack.append((nr, nc))
    out = g.copy()
    out[(g == 0) & ~outside] = fill
    return out


def _border_mask() -> np.ndarray:
    border = np.zeros((1, 1, H, W), dtype=np.float32)
    border[:, :, 0, :] = 1.0
    border[:, :, H - 1, :] = 1.0
    border[:, :, :, 0] = 1.0
    border[:, :, :, W - 1] = 1.0
    return border


def _border_mask_bool() -> np.ndarray:
    return _border_mask().astype(bool)


def _plus_kernel() -> np.ndarray:
    kernel = np.zeros((1, 1, 3, 3), dtype=np.float32)
    kernel[0, 0, 0, 1] = 1.0
    kernel[0, 0, 1, 0] = 1.0
    kernel[0, 0, 1, 2] = 1.0
    kernel[0, 0, 2, 1] = 1.0
    return kernel


def _plus_kernel_dilate() -> np.ndarray:
    """4-neighbor plus kernel with center tap for monotonic flood retention."""
    kernel = _plus_kernel()
    kernel[0, 0, 1, 1] = 1.0
    return kernel


def _i64(inits: List[onnx.TensorProto], vals: List[int], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))
    return name


def _f32(inits: List[onnx.TensorProto], arr: np.ndarray, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name))
    return name


def _bool_arr(inits: List[onnx.TensorProto], arr: np.ndarray, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=bool), name=name))
    return name


def _slice_constants(inits: List[onnx.TensorProto]) -> Dict[str, str]:
    return {
        "axes": _i64(inits, [0, 1, 2, 3], "axes"),
        "st0": _i64(inits, [0, 0, 0, 0], "st0"),
        "en0": _i64(inits, [1, 1, H, W], "en0"),
        "st1": _i64(inits, [0, 1, 0, 0], "st1"),
        "en4": _i64(inits, [1, FILL_COLOR, H, W], "en4"),
        "st4": _i64(inits, [0, FILL_COLOR, 0, 0], "st4"),
        "en5": _i64(inits, [1, FILL_COLOR + 1, H, W], "en5"),
        "st5": _i64(inits, [0, FILL_COLOR + 1, 0, 0], "st5"),
        "en10": _i64(inits, [1, C, H, W], "en10"),
        "s1": _i64(inits, [0, 0, 1, 0], "s1"),
        "e1": _i64(inits, [1, 1, H + 1, W], "e1"),
        "s2": _i64(inits, [0, 0, 0, 1], "s2"),
        "e2": _i64(inits, [1, 1, H, W + 1], "e2"),
    }


def _append_output_channels(
    nodes: List[onnx.NodeProto],
    ch0: str,
    fill: str,
    const: Dict[str, str],
) -> None:
    nodes.extend(
        [
            helper.make_node("Sub", [ch0, fill], ["out0"]),
            helper.make_node("Slice", [IN_NAME, const["st1"], const["en4"], const["axes"]], ["ch1_3"]),
            helper.make_node("Slice", [IN_NAME, const["st4"], const["en5"], const["axes"]], ["ch4"]),
            helper.make_node("Add", ["ch4", fill], ["out4"]),
            helper.make_node("Slice", [IN_NAME, const["st5"], const["en10"], const["axes"]], ["ch5_9"]),
            helper.make_node("Concat", ["out0", "ch1_3", "out4", "ch5_9"], [OUT_NAME], axis=1),
        ]
    )


def _zero_mask_nodes(
    nodes: List[onnx.NodeProto],
    ch0: str,
    zero: str,
    *,
    as_bool: bool = False,
    legacy_float_max: bool = False,
) -> str:
    """Traversable zeros: explicit color 0 plus all-zero padding cells."""
    nodes.extend(
        [
            helper.make_node("ReduceMax", [IN_NAME], ["mx"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["mx", zero], ["nz"]),
            helper.make_node("Not", ["nz"], ["pad_b"]),
        ]
    )
    if legacy_float_max:
        nodes.extend(
            [
                helper.make_node("Cast", ["pad_b"], ["pad_f"], to=TensorProto.FLOAT),
                helper.make_node("Max", [ch0, "pad_f"], ["iz"]),
            ]
        )
        return "iz"
    nodes.extend(
        [
            helper.make_node("Greater", [ch0, zero], ["ch0_b"]),
            helper.make_node("Or", ["ch0_b", "pad_b"], ["iz_b"]),
        ]
    )
    if as_bool:
        return "iz_b"
    nodes.append(helper.make_node("Cast", ["iz_b"], ["iz"], to=TensorProto.FLOAT))
    return "iz"


def _zero_mask_nodes_legacy(
    nodes: List[onnx.NodeProto],
    ch0: str,
    zero: str,
) -> str:
    """Legacy mask: Cast/Max float merge (higher memory than bool Or)."""
    return _zero_mask_nodes(nodes, ch0, zero, legacy_float_max=True)


def _shift4_or(
    nodes: List[onnx.NodeProto],
    src: str,
    prefix: str,
    const: Dict[str, str],
) -> str:
    """4-neighbor OR via shifted float pads; avoids 8-connected leaks."""
    nodes.extend(
        [
            helper.make_node("Pad", [src], [f"{prefix}pu"], pads=[0, 0, 1, 0, 0, 0, 0, 0]),
            helper.make_node("Pad", [src], [f"{prefix}pd"], pads=[0, 0, 0, 0, 0, 0, 1, 0]),
            helper.make_node("Pad", [src], [f"{prefix}pl"], pads=[0, 0, 0, 1, 0, 0, 0, 0]),
            helper.make_node("Pad", [src], [f"{prefix}pr"], pads=[0, 0, 0, 0, 0, 0, 0, 1]),
            helper.make_node("Slice", [f"{prefix}pu", const["st0"], const["en0"], const["axes"]], [f"{prefix}u"]),
            helper.make_node("Slice", [f"{prefix}pd", const["s1"], const["e1"], const["axes"]], [f"{prefix}d"]),
            helper.make_node("Slice", [f"{prefix}pl", const["st0"], const["en0"], const["axes"]], [f"{prefix}l"]),
            helper.make_node("Slice", [f"{prefix}pr", const["s2"], const["e2"], const["axes"]], [f"{prefix}r"]),
            helper.make_node("Max", [f"{prefix}u", f"{prefix}d"], [f"{prefix}ud"]),
            helper.make_node("Max", [f"{prefix}l", f"{prefix}r"], [f"{prefix}lr"]),
            helper.make_node("Max", [f"{prefix}ud", f"{prefix}lr"], [f"{prefix}n"]),
        ]
    )
    return f"{prefix}n"


def _float_flood_propagation(
    nodes: List[onnx.NodeProto],
    iz: str,
    outside: str,
    n_iters: int,
    *,
    k_clamp: int | None = None,
    zero: str | None = None,
) -> str:
    """Conv→Mul flood steps; optional binary clamp every k_clamp iterations."""
    for i in range(n_iters):
        conv = f"conv{i}"
        outside_n = f"outside{i}"
        nodes.extend(
            [
                helper.make_node(
                    "Conv",
                    [outside, "k_plus"],
                    [conv],
                    kernel_shape=[3, 3],
                    pads=[1, 1, 1, 1],
                    strides=[1, 1],
                ),
                helper.make_node("Mul", [iz, conv], [outside_n]),
            ]
        )
        outside = outside_n
        if k_clamp and zero and (i + 1) % k_clamp == 0 and (i + 1) < n_iters:
            clamp_b = f"clamp_b{i}"
            clamp_f = f"clamp_f{i}"
            nodes.extend(
                [
                    helper.make_node("Greater", [outside, zero], [clamp_b]),
                    helper.make_node("Cast", [clamp_b], [clamp_f], to=TensorProto.FLOAT),
                ]
            )
            outside = clamp_f
    return outside


def build_float_flood_model(n_iters: int) -> onnx.ModelProto:
    """Float Conv dilation with ReduceMax traversable mask (color 0 + padding)."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    const = _slice_constants(inits)

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    zero = _f32(inits, [0.0], "zero")
    _f32(inits, _border_mask(), "border")
    _f32(inits, _plus_kernel_dilate(), "k_plus")

    nodes.append(helper.make_node("Slice", [IN_NAME, const["st0"], const["en0"], const["axes"]], ["ch0"]))
    iz = _zero_mask_nodes(nodes, "ch0", "zero")
    nodes.append(helper.make_node("Mul", [iz, "border"], ["outside"]))

    outside = _float_flood_propagation(nodes, iz, "outside", n_iters)
    _append_fill_from_outside(nodes, outside, "iz_b", zero)
    _append_output_channels(nodes, "ch0", "fill", const)

    graph = helper.make_graph(nodes, f"g_float_ff_{n_iters}", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _append_fill_from_outside(
    nodes: List[onnx.NodeProto],
    outside: str,
    iz_b: str,
    zero: str,
) -> None:
    """Enclosed zeros: iz & ~outside (reuses bool traversable mask)."""
    nodes.extend(
        [
            helper.make_node("Greater", [outside, zero], ["outside_b"]),
            helper.make_node("Not", ["outside_b"], ["not_out_b"]),
            helper.make_node("And", [iz_b, "not_out_b"], ["fill_b"]),
            helper.make_node("Cast", ["fill_b"], ["fill"], to=TensorProto.FLOAT),
        ]
    )


def build_float_flood_model_legacy(n_iters: int) -> onnx.ModelProto:
    """Float flood with legacy Cast/Max zero mask (benchmark baseline)."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    const = _slice_constants(inits)

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    zero = _f32(inits, [0.0], "zero")
    _f32(inits, _border_mask(), "border")
    _f32(inits, _plus_kernel_dilate(), "k_plus")

    nodes.append(helper.make_node("Slice", [IN_NAME, const["st0"], const["en0"], const["axes"]], ["ch0"]))
    iz = _zero_mask_nodes_legacy(nodes, "ch0", "zero")
    nodes.append(helper.make_node("Mul", [iz, "border"], ["outside"]))

    outside = _float_flood_propagation(nodes, iz, "outside", n_iters)
    nodes.extend(
        [
            helper.make_node("Greater", [outside, zero], ["outside_b"]),
            helper.make_node("Greater", [iz, zero], ["iz_b"]),
            helper.make_node("Not", ["outside_b"], ["not_out_b"]),
            helper.make_node("And", ["iz_b", "not_out_b"], ["fill_b"]),
            helper.make_node("Cast", ["fill_b"], ["fill"], to=TensorProto.FLOAT),
        ]
    )
    _append_output_channels(nodes, "ch0", "fill", const)

    graph = helper.make_graph(nodes, f"g_float_legacy_{n_iters}", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_float_flood_model_simple(n_iters: int) -> onnx.ModelProto:
    """Float Conv dilation; traversable cells = channel 0 only (iz = ch0)."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    const = _slice_constants(inits)

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    zero = _f32(inits, [0.0], "zero")
    _f32(inits, _border_mask(), "border")
    _f32(inits, _plus_kernel_dilate(), "k_plus")

    nodes.append(helper.make_node("Slice", [IN_NAME, const["st0"], const["en0"], const["axes"]], ["ch0"]))
    iz = "ch0"
    nodes.append(helper.make_node("Mul", [iz, "border"], ["outside"]))

    outside = _float_flood_propagation(nodes, iz, "outside", n_iters)
    nodes.extend(
        [
            helper.make_node("Greater", [outside, zero], ["outside_b"]),
            helper.make_node("Greater", [iz, zero], ["iz_b"]),
            helper.make_node("Not", ["outside_b"], ["not_out_b"]),
            helper.make_node("And", ["iz_b", "not_out_b"], ["fill_b"]),
            helper.make_node("Cast", ["fill_b"], ["fill"], to=TensorProto.FLOAT),
        ]
    )
    _append_output_channels(nodes, "ch0", "fill", const)

    graph = helper.make_graph(nodes, f"g_float_simple_{n_iters}", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_float_flood_model_periodic(n_iters: int, k_clamp: int) -> onnx.ModelProto:
    """Simple iz=ch0 with binary clamp every k_clamp Conv→Mul steps."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    const = _slice_constants(inits)

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    zero = _f32(inits, [0.0], "zero")
    _f32(inits, _border_mask(), "border")
    _f32(inits, _plus_kernel_dilate(), "k_plus")

    nodes.append(helper.make_node("Slice", [IN_NAME, const["st0"], const["en0"], const["axes"]], ["ch0"]))
    iz = "ch0"
    nodes.append(helper.make_node("Mul", [iz, "border"], ["outside"]))

    outside = _float_flood_propagation(
        nodes, iz, "outside", n_iters, k_clamp=k_clamp, zero="zero"
    )
    nodes.extend(
        [
            helper.make_node("Greater", [outside, zero], ["outside_b"]),
            helper.make_node("Greater", [iz, zero], ["iz_b"]),
            helper.make_node("Not", ["outside_b"], ["not_out_b"]),
            helper.make_node("And", ["iz_b", "not_out_b"], ["fill_b"]),
            helper.make_node("Cast", ["fill_b"], ["fill"], to=TensorProto.FLOAT),
        ]
    )
    _append_output_channels(nodes, "ch0", "fill", const)

    graph = helper.make_graph(
        nodes, f"g_float_periodic_k{k_clamp}_{n_iters}", [x_info], [y_info], initializer=inits
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_bool_flood_model(n_iters: int) -> onnx.ModelProto:
    """Variant B: bool flood fill; Pad/Slice shifts run on cast float masks."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    const = _slice_constants(inits)

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    zero = _f32(inits, [0.0], "zero")
    _bool_arr(inits, _border_mask_bool(), "border_b")

    nodes.append(helper.make_node("Slice", [IN_NAME, const["st0"], const["en0"], const["axes"]], ["ch0"]))
    iz_b = _zero_mask_nodes(nodes, "ch0", "zero", as_bool=True)
    nodes.append(helper.make_node("And", [iz_b, "border_b"], ["outside_b"]))

    outside = "outside_b"
    for i in range(n_iters):
        outside_f = f"outside_f{i}"
        nbr = f"nbr_f{i}"
        has_n = f"has_n{i}"
        grown = f"grown_b{i}"
        outside_n = f"outside_b{i}"
        nodes.extend(
            [
                helper.make_node("Cast", [outside], [outside_f], to=TensorProto.FLOAT),
            ]
        )
        nbr_out = _shift4_or(nodes, outside_f, f"b{i}", const)
        nodes.extend(
            [
                helper.make_node("Greater", [nbr_out, zero], [has_n]),
                helper.make_node("Or", [outside, has_n], [grown]),
                helper.make_node("And", [iz_b, grown], [outside_n]),
            ]
        )
        outside = outside_n

    nodes.extend(
        [
            helper.make_node("Not", [outside], ["not_out_b"]),
            helper.make_node("And", [iz_b, "not_out_b"], ["fill_b"]),
            helper.make_node("Cast", ["fill_b"], ["fill"], to=TensorProto.FLOAT),
        ]
    )
    _append_output_channels(nodes, "ch0", "fill", const)

    graph = helper.make_graph(nodes, f"g_bool_ff_{n_iters}", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_onnx_model(n_iters: int = 25) -> onnx.ModelProto:
    """Default builder: lowest-cost valid flood fill (float, 25 iterations)."""
    return build_float_flood_model(n_iters)


def save_model(path: Path = BEST_PATH, n_iters: int = 25) -> onnx.ModelProto:
    model = build_onnx_model(n_iters)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return model


def model_stats(model: onnx.ModelProto, path: Path | None = None) -> Dict[str, int]:
    params = sum(int(np.prod(list(t.dims))) for t in model.graph.initializer)
    return {
        "bytes": path.stat().st_size if path and path.is_file() else len(model.SerializeToString()),
        "nodes": len(model.graph.node),
        "inits": len(model.graph.initializer),
        "params": params,
        "opset": model.opset_import[0].version,
        "ir": model.ir_version,
    }


def _grid_to_onehot(grid: np.ndarray | List[List[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            color = int(arr[r, c])
            if 0 <= color < C:
                out[0, color, r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    arr = onehot.reshape(C, H, W)
    valid = arr.max(axis=0) > 0
    grid = arr.argmax(axis=0).astype(np.int64)
    grid[~valid] = 0
    return grid


def _pad_grid(grid: np.ndarray) -> np.ndarray:
    out = np.zeros((H, W), dtype=np.int64)
    out[: grid.shape[0], : grid.shape[1]] = grid
    return out


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def _load_task_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _validate_json_examples(model: onnx.ModelProto) -> tuple[bool, Dict[str, bool]]:
    data = _load_task_data()
    split_pass: Dict[str, bool] = {}
    all_ok = True
    for split in ("train", "test", "arc-gen"):
        split_ok = True
        for ex in data[split]:
            pred = _run_onnx(model, _grid_to_onehot(ex["input"]))
            expected = _grid_to_onehot(ex["output"])
            if pred.shape != expected.shape or not np.array_equal(pred > 0.0, expected > 0.0):
                split_ok = False
                break
        split_pass[split] = split_ok
        all_ok = all_ok and split_ok
    return all_ok, split_pass


def _validate_train_arc(path: Path) -> bool:
    from train_arc import validate_task_onnx

    capture = StringIO()
    with redirect_stdout(capture):
        ok, _ = validate_task_onnx("task002", path, save_viz=False)
    return ok


def _score_path(path: Path) -> dict[str, Any]:
    from score_model import score_file

    return score_file(path)


def _solve_ray(x: np.ndarray, fill: int = FILL_COLOR) -> np.ndarray:
    """Ray enclosure (incorrect for corridors); used only in adversarial checks."""
    g = np.asarray(x, dtype=np.int64)
    h, w = g.shape
    wall = g != 0
    out = g.copy()
    for r in range(h):
        for c in range(w):
            if g[r, c] != 0:
                continue
            if (
                wall[:r, c].any()
                and wall[r + 1 :, c].any()
                and wall[r, :c].any()
                and wall[r, c + 1 :].any()
            ):
                out[r, c] = fill
    return out


def _canvas_solve(grid: np.ndarray | List[List[int]], fill: int = FILL_COLOR) -> np.ndarray:
    """Flood fill on the competition 30×30 canvas."""
    canvas = np.zeros((H, W), dtype=np.int64)
    arr = np.asarray(grid, dtype=np.int64)
    canvas[: arr.shape[0], : arr.shape[1]] = arr
    return solve(canvas)


def adversarial_cases() -> List[Tuple[str, List[List[int]], List[List[int]]]]:
    """Cases where ray enclosure diverges from true 4-neighbor flood fill."""
    wall = 3
    cases: List[Tuple[str, List[List[int]], List[List[int]]]] = []

    closed = [
        [wall] * 7,
        [wall, 0, 0, 0, 0, 0, wall],
        [wall, 0, 0, 0, 0, 0, wall],
        [wall, 0, 0, 0, 0, 0, wall],
        [wall, 0, 0, 0, 0, 0, wall],
        [wall] * 7,
        [0] * 7,
    ]
    cases.append(("closed_box", closed, _canvas_solve(closed).tolist()))

    gapped = [row[:] for row in closed]
    gapped[5] = [wall, wall, wall, 0, wall, wall, wall]
    cases.append(("one_cell_gap", gapped, _canvas_solve(gapped).tolist()))

    corridor = (
        [[0] * 9]
        + [[0, wall, wall, wall, wall, wall, wall, wall, 0]]
        + [[0, wall, 0, 0, 0, 0, 0, wall, 0]] * 5
        + [[0, wall, wall, wall, wall, wall, wall, wall, 0]]
        + [[0] * 9]
    )
    cases.append(("winding_corridor", corridor, _canvas_solve(corridor).tolist()))

    mixed = [
        [0] * 13,
        [0, wall, wall, wall, wall, 0, 0, wall, wall, wall, wall, wall, 0],
        [0, wall, 0, 0, wall, 0, 0, wall, 0, 0, 0, wall, 0],
        [0, wall, 0, 0, wall, 0, 0, wall, 0, 0, 0, wall, 0],
        [0, wall, wall, wall, wall, 0, 0, wall, wall, wall, 0, wall, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, wall, 0, wall, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, wall, wall, wall, 0],
        [0] * 13,
    ]
    cases.append(("mixed_open_and_closed", mixed, _canvas_solve(mixed).tolist()))

    return cases


def validate_adversarial(model: onnx.ModelProto) -> tuple[bool, List[str]]:
    failures: List[str] = []
    for name, inp, expected in adversarial_cases():
        pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(inp))[0])
        exp = _pad_grid(np.asarray(expected, dtype=np.int64))
        h, w = np.asarray(expected).shape
        if not np.array_equal(pred[:h, :w], exp[:h, :w]):
            failures.append(name)
    return not failures, failures


@dataclass
class BenchmarkRow:
    variant: str
    n_iters: int
    nodes: int
    params: int
    memory: int | None
    cost: int | None
    score: float | None
    valid: bool
    model: onnx.ModelProto | None = None
    note: str = ""


def _validate_model_full(model: onnx.ModelProto, path: Path) -> tuple[bool, str]:
    json_ok, split_pass = _validate_json_examples(model)
    if not json_ok:
        bad = [split for split, ok in split_pass.items() if not ok]
        return False, f"task002.json failed: {', '.join(bad)}"
    adv_ok, adv_fail = validate_adversarial(model)
    if not adv_ok:
        return False, f"adversarial failed: {', '.join(adv_fail)}"
    if not _validate_train_arc(path):
        return False, "train_arc validation failed"
    scored = _score_path(path)
    if not scored.get("valid"):
        return False, scored.get("error", "score_model invalid")
    return True, "PASS"


def _benchmark_builders() -> dict[str, Callable[[int], onnx.ModelProto]]:
    """Float-only sweep; bool kept in source but excluded from benchmark."""
    builders: dict[str, Callable[[int], onnx.ModelProto]] = {
        "float_zero_mask": build_float_flood_model,
        "float_zero_mask_legacy": build_float_flood_model_legacy,
        "float_simple": build_float_flood_model_simple,
    }
    for k in (3, 5, 8):
        builders[f"float_simple_k{k}"] = lambda n, k=k: build_float_flood_model_periodic(n, k)
    return builders


def benchmark_variants() -> List[BenchmarkRow]:
    builders = _benchmark_builders()
    rows: List[BenchmarkRow] = []

    with tempfile.TemporaryDirectory(prefix="task002_bench_") as tmpdir:
        tmp = Path(tmpdir)
        for variant, builder in builders.items():
            for n_iters in range(1, MAX_ITERS + 1):
                row = BenchmarkRow(
                    variant=variant,
                    n_iters=n_iters,
                    nodes=0,
                    params=0,
                    memory=None,
                    cost=None,
                    score=None,
                    valid=False,
                )
                path = tmp / f"task002_{variant}_{n_iters:02d}.onnx"
                try:
                    model = builder(n_iters)
                    stats = model_stats(model)
                    row.nodes = stats["nodes"]
                    row.params = stats["params"]
                    row.model = model
                    onnx.save(model, str(path))

                    json_ok, _ = _validate_json_examples(model)
                    adv_ok, adv_fail = validate_adversarial(model)
                    if not json_ok:
                        row.note = "json fail"
                    elif not adv_ok:
                        row.note = f"adv fail: {','.join(adv_fail)}"
                    else:
                        train_ok = _validate_train_arc(path)
                        if not train_ok:
                            row.note = "train_arc fail"
                        else:
                            scored = _score_path(path)
                            row.memory = scored.get("memory")
                            row.cost = scored.get("cost")
                            row.score = scored.get("score")
                            row.valid = bool(scored.get("valid"))
                            if not row.valid:
                                row.note = scored.get("error", "invalid score")
                except Exception as exc:
                    row.note = f"build/run fail: {exc}"
                rows.append(row)
                print(
                    f"{row.n_iters:>6} {row.nodes:>5} {row.params:>6} "
                    f"{row.memory if row.memory is not None else '-':>7} "
                    f"{row.cost if row.cost is not None else '-':>6} "
                    f"{row.score if row.score is not None else 0.0:9.6f} "
                    f"{str(row.valid):>5}  {variant} {row.note}"
                )

    return rows


def first_valid_per_variant(rows: List[BenchmarkRow]) -> dict[str, BenchmarkRow]:
    """Earliest n_iters that passes full validation for each variant."""
    picked: dict[str, BenchmarkRow] = {}
    for row in sorted(rows, key=lambda r: (r.variant, r.n_iters)):
        if row.valid and row.variant not in picked:
            picked[row.variant] = row
    return picked


def save_best_model(rows: List[BenchmarkRow]) -> BenchmarkRow | None:
    """Save lowest-cost model among first-valid-per-variant picks."""
    candidates = list(first_valid_per_variant(rows).values())
    if not candidates:
        return None
    best = min(candidates, key=lambda row: int(row.cost or 0))
    if best.model is not None:
        onnx.save(best.model, str(BEST_PATH))
    return best


def print_comparison_summary(rows: List[BenchmarkRow]) -> None:
    picked = first_valid_per_variant(rows)
    print("\n=== Comparison (first valid n_iters per variant) ===")
    print(f"{'variant':<20} {'n_iters':>6} {'nodes':>5} {'params':>6} {'memory':>7} {'cost':>6} {'score':>9} valid")
    for variant in sorted(picked):
        row = picked[variant]
        print(
            f"{variant:<20} {row.n_iters:>6} {row.nodes:>5} {row.params:>6} "
            f"{row.memory if row.memory is not None else '-':>7} "
            f"{row.cost if row.cost is not None else '-':>6} "
            f"{row.score if row.score is not None else 0.0:9.6f} {row.valid}"
        )
    focus = ("float_zero_mask_legacy", "float_zero_mask", "float_simple")
    if "float_zero_mask_legacy" in picked and "float_zero_mask" in picked:
        leg, opt = picked["float_zero_mask_legacy"], picked["float_zero_mask"]
        print("\n--- Legacy vs optimized zero_mask ---")
        print(
            f"  legacy: n={leg.n_iters} cost={leg.cost} memory={leg.memory} score={leg.score:.6f}"
        )
        print(
            f"  optimized: n={opt.n_iters} cost={opt.cost} memory={opt.memory} score={opt.score:.6f}"
        )
        if leg.cost and opt.cost:
            print(f"  cost delta (opt-legacy): {int(opt.cost) - int(leg.cost):+d}")
    if "float_zero_mask" in picked and "float_simple" in picked:
        a, b = picked["float_zero_mask"], picked["float_simple"]
        print("\n--- A vs B: zero_mask vs simple (ch0 only) ---")
        for label, row in (("A zero_mask", a), ("B simple", b)):
            print(
                f"  {label}: n={row.n_iters} cost={row.cost} memory={row.memory} "
                f"score={row.score:.6f} nodes={row.nodes}"
            )
        if a.cost is not None and b.cost is not None:
            delta = int(b.cost) - int(a.cost)
            winner = "B simple" if b.cost < a.cost else "A zero_mask" if a.cost < b.cost else "tie"
            print(f"  cost delta (B-A): {delta:+d}  -> {winner} wins on cost")


def test() -> None:
    print("Adversarial reference checks")
    for name, inp, expected in adversarial_cases():
        canvas = np.zeros((H, W), dtype=np.int64)
        arr = np.asarray(inp)
        canvas[: arr.shape[0], : arr.shape[1]] = arr
        pred = solve(canvas)
        exp = _pad_grid(np.asarray(expected, dtype=np.int64))
        ok = np.array_equal(pred, exp)
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit(1)

    print("\nRay enclosure diverges on corridor case (expected)")
    corridor = np.asarray(adversarial_cases()[2][1], dtype=np.int64)
    canvas = np.zeros((H, W), dtype=np.int64)
    canvas[: corridor.shape[0], : corridor.shape[1]] = corridor
    ray = _solve_ray(canvas)
    flood = solve(canvas)
    print(f"  corridor fills: flood={(flood == FILL_COLOR).sum()} ray={(ray == FILL_COLOR).sum()}")

    print("\nBenchmark sweep (float only, n_iters=1..62)")
    print(f"{'n_iters':>6} {'nodes':>5} {'params':>6} {'memory':>7} {'cost':>6} {'score':>9} {'valid':>5}  variant")
    rows = benchmark_variants()
    print_comparison_summary(rows)
    best = save_best_model(rows)
    if best is None:
        raise SystemExit("no valid benchmark model found")

    print(
        f"\nSaved best (first-valid-per-variant, min cost): variant={best.variant} "
        f"n_iters={best.n_iters} nodes={best.nodes} params={best.params} "
        f"memory={best.memory} cost={best.cost} score={best.score:.6f}"
    )
    print(f"saved {BEST_PATH}")


def main() -> None:
    test()


if __name__ == "__main__":
    main()
