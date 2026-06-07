"""Minimal ONNX for ARC task094: draw magenta plus through frame centers.

Task rule: 15x15 cyan grids contain one or two hollow 5x5 dark-blue (1) frames.
For each frame, draw a magenta (6) plus centered on the frame bbox:
full-width horizontal line on the center row and full-height vertical line on
the center column. Magenta overwrites cyan only; blue frame pixels stay blue.

ONNX: the best variant slices the 15x15 blue plane, uses a 5x5 perimeter Conv
to detect frame top-left corners, reduces those detections to center row/column
masks, paints magenta only on non-blue cells, assembles channels 1/6/8, and pads
the 15x15 float crop to the required 30x30 output.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import calculate_memory, convert_to_numpy, sanitize_model, score_file  # noqa: E402

TASK_ID = "task094"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

G = 15
C = 10
H = W = 30
PAD = H - G
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.value_infos: list[onnx.ValueInfoProto] = []
        self.false_15: str | None = None
        self.counter = 0

    def init(self, name: str, array: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(array, name=name))
        return name

    def vi(self, name: str, dtype: int, shape: tuple[int, ...]) -> str:
        self.value_infos.append(helper.make_tensor_value_info(name, dtype, list(shape)))
        return name

    def name(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def node(
        self,
        op_type: str,
        inputs: list[str],
        dtype: int,
        shape: tuple[int, ...],
        prefix: str,
        **attrs: object,
    ) -> str:
        out = self.vi(self.name(prefix), dtype, shape)
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out

    def false15(self) -> str:
        if self.false_15 is None:
            self.false_15 = self.init("false15", np.zeros((1, 1, G, G), dtype=np.bool_))
        return self.false_15


def _make_model(b: Builder, graph_name: str) -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(b.nodes, graph_name, [x_info], [y_info], initializer=b.initializers, value_info=b.value_infos)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _top_left_corners(b: Builder, blue: str, prefix: str) -> str:
    up_pad = b.node("Concat", ["false1x15", blue], TensorProto.BOOL, (1, 1, G + 1, G), f"{prefix}_up_pad", axis=2)
    up = b.node("Slice", [up_pad, "row0_st", "row15_en", "axes4"], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_up")
    left_pad = b.node("Concat", ["false15x1", blue], TensorProto.BOOL, (1, 1, G, G + 1), f"{prefix}_left_pad", axis=3)
    left = b.node("Slice", [left_pad, "col0_st", "col15_en", "axes4"], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_left")
    blue_br_body = b.node("Concat", [blue, "false5x15"], TensorProto.BOOL, (1, 1, G + 5, G), f"{prefix}_blue_br_body", axis=2)
    blue_br_pad = b.node("Concat", [blue_br_body, "false20x5"], TensorProto.BOOL, (1, 1, G + 5, G + 5), f"{prefix}_blue_br_pad", axis=3)
    br = b.node("Slice", [blue_br_pad, "br_st", "br_en", "axes4"], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_br")
    tr_pad = b.node("Concat", [blue, "false15x4"], TensorProto.BOOL, (1, 1, G, G + 4), f"{prefix}_tr_pad", axis=3)
    tr = b.node("Slice", [tr_pad, "tr_st", "tr_en", "axes4"], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_tr")
    bl_pad = b.node("Concat", [blue, "false4x15"], TensorProto.BOOL, (1, 1, G + 4, G), f"{prefix}_bl_pad", axis=2)
    bl = b.node("Slice", [bl_pad, "bl_st", "bl_en", "axes4"], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_bl")
    not_up = b.node("Not", [up], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_not_up")
    not_left = b.node("Not", [left], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_not_left")
    tl_a = b.node("And", [blue, not_up], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_tl_a")
    tl_b = b.node("And", [tl_a, not_left], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_tl_b")
    tl_c = b.node("And", [tl_b, br], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_tl_c")
    tl_d = b.node("And", [tl_c, tr], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_tl_d")
    return b.node("And", [tl_d, bl], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_tl")


def _two_frame_centers(b: Builder, tl: str, prefix: str) -> tuple[str, str, str, str, str]:
    w = b.node("Where", [tl, "lin_i64", "big_i64"], TensorProto.INT64, (1, 1, G, G), f"{prefix}_w")
    flat = b.node("Reshape", [w, "flat_sh"], TensorProto.INT64, (G * G,), f"{prefix}_flat")
    idx1 = b.node("ArgMin", [flat], TensorProto.INT64, (1,), f"{prefix}_idx1", axis=0, keepdims=1)
    at1 = b.node("Equal", ["lin_i64", idx1], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_at1")
    w2 = b.node("Where", [at1, "big_i64", w], TensorProto.INT64, (1, 1, G, G), f"{prefix}_w2")
    flat2 = b.node("Reshape", [w2, "flat_sh"], TensorProto.INT64, (G * G,), f"{prefix}_flat2")
    idx2 = b.node("ArgMin", [flat2], TensorProto.INT64, (1,), f"{prefix}_idx2", axis=0, keepdims=1)
    r1 = b.node("Div", [idx1, "g_i"], TensorProto.INT64, (1,), f"{prefix}_r1")
    c1 = b.node("Mod", [idx1, "g_i"], TensorProto.INT64, (1,), f"{prefix}_c1")
    r2 = b.node("Div", [idx2, "g_i"], TensorProto.INT64, (1,), f"{prefix}_r2")
    c2 = b.node("Mod", [idx2, "g_i"], TensorProto.INT64, (1,), f"{prefix}_c2")
    cr1 = b.node("Add", [r1, "two_i"], TensorProto.INT64, (1,), f"{prefix}_cr1")
    cc1 = b.node("Add", [c1, "two_i"], TensorProto.INT64, (1,), f"{prefix}_cc1")
    cr2 = b.node("Add", [r2, "two_i"], TensorProto.INT64, (1,), f"{prefix}_cr2")
    cc2 = b.node("Add", [c2, "two_i"], TensorProto.INT64, (1,), f"{prefix}_cc2")
    tl_f = b.node("Cast", [tl], TensorProto.FLOAT, (1, 1, G, G), f"{prefix}_tl_f", to=TensorProto.FLOAT)
    tl_count = b.node("ReduceSum", [tl_f], TensorProto.FLOAT, (1, 1, 1, 1), f"{prefix}_tl_count", axes=[2, 3], keepdims=1)
    has2 = b.node("Greater", [tl_count, "one_f"], TensorProto.BOOL, (1, 1, 1, 1), f"{prefix}_has2")
    return cr1, cc1, cr2, cc2, has2


def _plus_from_centers(
    b: Builder,
    cr1: str,
    cc1: str,
    cr2: str,
    cc2: str,
    has2: str,
    prefix: str,
) -> str:
    eq_r1 = b.node("Equal", ["rows_i", cr1], TensorProto.BOOL, (1, 1, G, 1), f"{prefix}_eq_r1")
    eq_r2 = b.node("Equal", ["rows_i", cr2], TensorProto.BOOL, (1, 1, G, 1), f"{prefix}_eq_r2")
    eq_c1 = b.node("Equal", ["cols_i", cc1], TensorProto.BOOL, (1, 1, 1, G), f"{prefix}_eq_c1")
    eq_c2 = b.node("Equal", ["cols_i", cc2], TensorProto.BOOL, (1, 1, 1, G), f"{prefix}_eq_c2")
    r2_on = b.node("And", [has2, eq_r2], TensorProto.BOOL, (1, 1, G, 1), f"{prefix}_r2_on")
    c2_on = b.node("And", [has2, eq_c2], TensorProto.BOOL, (1, 1, 1, G), f"{prefix}_c2_on")
    plus_h = b.node("Or", [eq_r1, r2_on], TensorProto.BOOL, (1, 1, G, 1), f"{prefix}_plus_h")
    plus_v = b.node("Or", [eq_c1, c2_on], TensorProto.BOOL, (1, 1, 1, G), f"{prefix}_plus_v")
    return b.node("Or", [plus_h, plus_v], TensorProto.BOOL, (1, 1, G, G), f"{prefix}_plus")


def _common_inits(b: Builder) -> None:
    b.init("axes4", np.array([0, 1, 2, 3], dtype=np.int64))
    b.init("half", np.array(0.5, dtype=np.float32))
    b.init("one_f", np.array(1.0, dtype=np.float32))
    b.init("zero_f", np.array(0.0, dtype=np.float32))
    b.init("g_i", np.array(G, dtype=np.int64))
    b.init("two_i", np.array(2, dtype=np.int64))
    b.init("flat_sh", np.array([G * G], dtype=np.int64))
    b.init("rows_i", np.arange(G, dtype=np.int64).reshape(1, 1, G, 1))
    b.init("cols_i", np.arange(G, dtype=np.int64).reshape(1, 1, 1, G))
    b.init("lin_i64", (np.arange(G, dtype=np.int64).reshape(G, 1) * G + np.arange(G, dtype=np.int64)).reshape(1, 1, G, G))
    b.init("big_i64", np.array(9999, dtype=np.int64))

    b.init("blue_st", np.array([0, 1, 0, 0], dtype=np.int64))
    b.init("blue_en", np.array([1, 2, G, G], dtype=np.int64))
    b.init("cyan_st", np.array([0, 8, 0, 0], dtype=np.int64))
    b.init("cyan_en", np.array([1, 9, G, G], dtype=np.int64))

    b.init("row0_st", np.array([0, 0, 0, 0], dtype=np.int64))
    b.init("row15_en", np.array([1, 1, G, G], dtype=np.int64))
    b.init("col0_st", np.array([0, 0, 0, 0], dtype=np.int64))
    b.init("col15_en", np.array([1, 1, G, G], dtype=np.int64))
    b.init("false20x5", np.zeros((1, 1, G + 5, 5), dtype=np.bool_))
    b.init("false5x15", np.zeros((1, 1, 5, G), dtype=np.bool_))
    b.init("false15x4", np.zeros((1, 1, G, 4), dtype=np.bool_))
    b.init("false4x15", np.zeros((1, 1, 4, G), dtype=np.bool_))
    b.init("false1x15", np.zeros((1, 1, 1, G), dtype=np.bool_))
    b.init("false15x1", np.zeros((1, 1, G, 1), dtype=np.bool_))
    b.init("br_st", np.array([0, 0, 4, 4], dtype=np.int64))
    b.init("br_en", np.array([1, 1, 4 + G, 4 + G], dtype=np.int64))
    b.init("tr_st", np.array([0, 0, 0, 4], dtype=np.int64))
    b.init("tr_en", np.array([1, 1, G, 4 + G], dtype=np.int64))
    b.init("bl_st", np.array([0, 0, 4, 0], dtype=np.int64))
    b.init("bl_en", np.array([1, 1, 4 + G, G], dtype=np.int64))


def build_three_plane() -> onnx.ModelProto:
    """Emit only channels 1/6/8 as bool before final cast+pad."""
    b = Builder()
    _common_inits(b)
    blue_f = b.node("Slice", [IN_NAME, "blue_st", "blue_en", "axes4"], TensorProto.FLOAT, (1, 1, G, G), "blue_f")
    cyan_f = b.node("Slice", [IN_NAME, "cyan_st", "cyan_en", "axes4"], TensorProto.FLOAT, (1, 1, G, G), "cyan_f")
    blue = b.node("Greater", [blue_f, "half"], TensorProto.BOOL, (1, 1, G, G), "blue")
    cyan = b.node("Greater", [cyan_f, "half"], TensorProto.BOOL, (1, 1, G, G), "cyan")
    tl = _top_left_corners(b, blue, "tl")
    cr1, cc1, cr2, cc2, has2 = _two_frame_centers(b, tl, "ctr")
    plus = _plus_from_centers(b, cr1, cc1, cr2, cc2, has2, "plus")
    apply = b.node("And", [plus, cyan], TensorProto.BOOL, (1, 1, G, G), "apply")
    not_apply = b.node("Not", [apply], TensorProto.BOOL, (1, 1, G, G), "not_apply")
    cyan_out = b.node("And", [cyan, not_apply], TensorProto.BOOL, (1, 1, G, G), "cyan_out")
    false15 = b.false15()
    crop10_b = b.node(
        "Concat",
        [false15, blue, false15, false15, false15, false15, apply, false15, cyan_out, false15],
        TensorProto.BOOL,
        (1, C, G, G),
        "crop10_b",
        axis=1,
    )
    crop10 = b.node("Cast", [crop10_b], TensorProto.FLOAT, (1, C, G, G), "crop10", to=TensorProto.FLOAT)
    b.nodes.append(
        helper.make_node("Pad", [crop10], [OUT_NAME], mode="constant", pads=[0, 0, 0, 0, 0, 0, PAD, PAD], value=0.0)
    )
    return _make_model(b, "task094_three_plane")


def _conv_inits(b: Builder) -> None:
    b.init("axes4", np.array([0, 1, 2, 3], dtype=np.int64))
    b.init("half", np.array(0.5, dtype=np.float32))
    b.init("frame_threshold", np.array(15.5, dtype=np.float32))
    b.init("blue_st", np.array([0, 1, 0, 0], dtype=np.int64))
    b.init("blue_en", np.array([1, 2, G, G], dtype=np.int64))
    kernel = np.ones((1, 1, 5, 5), dtype=np.float32)
    kernel[:, :, 1:4, 1:4] = 0.0
    b.init("frame_kernel", kernel)
    b.init("false2r", np.zeros((1, 1, 2, 1), dtype=np.bool_))
    b.init("false2c", np.zeros((1, 1, 1, 2), dtype=np.bool_))


def build_conv_rows() -> onnx.ModelProto:
    """Detect 5x5 frame origins with Conv, then expand detected rows/cols."""
    b = Builder()
    _conv_inits(b)
    blue_f = b.node("Slice", [IN_NAME, "blue_st", "blue_en", "axes4"], TensorProto.FLOAT, (1, 1, G, G), "blue_f")
    blue = b.node("Greater", [blue_f, "half"], TensorProto.BOOL, (1, 1, G, G), "blue")
    frame_sum = b.node("Conv", [blue_f, "frame_kernel"], TensorProto.FLOAT, (1, 1, G - 4, G - 4), "frame_sum")
    tl = b.node("Greater", [frame_sum, "frame_threshold"], TensorProto.BOOL, (1, 1, G - 4, G - 4), "tl")
    tl_f = b.node("Cast", [tl], TensorProto.FLOAT, (1, 1, G - 4, G - 4), "tl_f", to=TensorProto.FLOAT)
    row_sum = b.node("ReduceSum", [tl_f], TensorProto.FLOAT, (1, 1, G - 4, 1), "row_sum", axes=[3], keepdims=1)
    col_sum = b.node("ReduceSum", [tl_f], TensorProto.FLOAT, (1, 1, 1, G - 4), "col_sum", axes=[2], keepdims=1)
    row_on = b.node("Greater", [row_sum, "half"], TensorProto.BOOL, (1, 1, G - 4, 1), "row_on")
    col_on = b.node("Greater", [col_sum, "half"], TensorProto.BOOL, (1, 1, 1, G - 4), "col_on")
    plus_h = b.node("Concat", ["false2r", row_on, "false2r"], TensorProto.BOOL, (1, 1, G, 1), "plus_h", axis=2)
    plus_v = b.node("Concat", ["false2c", col_on, "false2c"], TensorProto.BOOL, (1, 1, 1, G), "plus_v", axis=3)
    plus = b.node("Or", [plus_h, plus_v], TensorProto.BOOL, (1, 1, G, G), "plus")
    not_blue = b.node("Not", [blue], TensorProto.BOOL, (1, 1, G, G), "not_blue")
    apply = b.node("And", [plus, not_blue], TensorProto.BOOL, (1, 1, G, G), "apply")
    painted_or_blue = b.node("Or", [blue, apply], TensorProto.BOOL, (1, 1, G, G), "painted_or_blue")
    cyan_out = b.node("Not", [painted_or_blue], TensorProto.BOOL, (1, 1, G, G), "cyan_out")
    false15 = b.false15()
    crop10_b = b.node(
        "Concat",
        [false15, blue, false15, false15, false15, false15, apply, false15, cyan_out, false15],
        TensorProto.BOOL,
        (1, C, G, G),
        "crop10_b",
        axis=1,
    )
    crop10 = b.node("Cast", [crop10_b], TensorProto.FLOAT, (1, C, G, G), "crop10", to=TensorProto.FLOAT)
    b.nodes.append(
        helper.make_node("Pad", [crop10], [OUT_NAME], mode="constant", pads=[0, 0, 0, 0, 0, 0, PAD, PAD], value=0.0)
    )
    return _make_model(b, "task094_conv_rows")


def build_conv_rows_float() -> onnx.ModelProto:
    """Conv detector with direct float crop assembly to avoid bool 10-plane crop."""
    b = Builder()
    _conv_inits(b)
    b.init("zero15f", np.zeros((1, 1, G, G), dtype=np.float32))
    blue_f = b.node("Slice", [IN_NAME, "blue_st", "blue_en", "axes4"], TensorProto.FLOAT, (1, 1, G, G), "blue_f")
    blue = b.node("Greater", [blue_f, "half"], TensorProto.BOOL, (1, 1, G, G), "blue")
    frame_sum = b.node("Conv", [blue_f, "frame_kernel"], TensorProto.FLOAT, (1, 1, G - 4, G - 4), "frame_sum")
    tl = b.node("Greater", [frame_sum, "frame_threshold"], TensorProto.BOOL, (1, 1, G - 4, G - 4), "tl")
    tl_f = b.node("Cast", [tl], TensorProto.FLOAT, (1, 1, G - 4, G - 4), "tl_f", to=TensorProto.FLOAT)
    row_sum = b.node("ReduceSum", [tl_f], TensorProto.FLOAT, (1, 1, G - 4, 1), "row_sum", axes=[3], keepdims=1)
    col_sum = b.node("ReduceSum", [tl_f], TensorProto.FLOAT, (1, 1, 1, G - 4), "col_sum", axes=[2], keepdims=1)
    row_on = b.node("Greater", [row_sum, "half"], TensorProto.BOOL, (1, 1, G - 4, 1), "row_on")
    col_on = b.node("Greater", [col_sum, "half"], TensorProto.BOOL, (1, 1, 1, G - 4), "col_on")
    plus_h = b.node("Concat", ["false2r", row_on, "false2r"], TensorProto.BOOL, (1, 1, G, 1), "plus_h", axis=2)
    plus_v = b.node("Concat", ["false2c", col_on, "false2c"], TensorProto.BOOL, (1, 1, 1, G), "plus_v", axis=3)
    plus = b.node("Or", [plus_h, plus_v], TensorProto.BOOL, (1, 1, G, G), "plus")
    not_blue = b.node("Not", [blue], TensorProto.BOOL, (1, 1, G, G), "not_blue")
    apply = b.node("And", [plus, not_blue], TensorProto.BOOL, (1, 1, G, G), "apply")
    painted_or_blue = b.node("Or", [blue, apply], TensorProto.BOOL, (1, 1, G, G), "painted_or_blue")
    cyan_out = b.node("Not", [painted_or_blue], TensorProto.BOOL, (1, 1, G, G), "cyan_out")
    apply_f = b.node("Cast", [apply], TensorProto.FLOAT, (1, 1, G, G), "apply_f", to=TensorProto.FLOAT)
    cyan_f = b.node("Cast", [cyan_out], TensorProto.FLOAT, (1, 1, G, G), "cyan_f", to=TensorProto.FLOAT)
    z = "zero15f"
    crop10 = b.node(
        "Concat",
        [z, blue_f, z, z, z, z, apply_f, z, cyan_f, z],
        TensorProto.FLOAT,
        (1, C, G, G),
        "crop10",
        axis=1,
    )
    b.nodes.append(
        helper.make_node("Pad", [crop10], [OUT_NAME], mode="constant", pads=[0, 0, 0, 0, 0, 0, PAD, PAD], value=0.0)
    )
    return _make_model(b, "task094_conv_rows_float")


def _load_examples() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _check_correct(model: onnx.ModelProto) -> tuple[bool, dict[str, tuple[int, int]]]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return False, {}
    session = ort.InferenceSession(sanitized.SerializeToString(), providers=["CPUExecutionProvider"])
    task = _load_examples()
    counts: dict[str, tuple[int, int]] = {}
    for split in ("train", "test", "arc-gen"):
        ok_count = 0
        examples = task.get(split, [])
        for example in examples:
            x = convert_to_numpy(example, "input")
            y = convert_to_numpy(example, "output")
            if x is None or y is None:
                continue
            pred = session.run([OUT_NAME], {IN_NAME: x})[0]
            if not np.array_equal(pred > 0.0, y > 0.0):
                return False, {**counts, split: (ok_count, len(examples))}
            ok_count += 1
        counts[split] = (ok_count, len(examples))
    return True, counts


def _profile_largest_internal(model: onnx.ModelProto) -> tuple[int | None, str | None, int | None]:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return None, None, None
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(Path(tempfile.gettempdir()) / TASK_ID)
    session = ort.InferenceSession(sanitized.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    for split_examples in _load_examples().values():
        for example in split_examples:
            x = convert_to_numpy(example, "input")
            if x is not None:
                session.run([OUT_NAME], {IN_NAME: x})
    trace_path = session.end_profiling()
    memory = calculate_memory(sanitized, trace_path)
    graph = onnx.shape_inference.infer_shapes(sanitized, strict_mode=True).graph
    largest_name: str | None = None
    largest_bytes = -1
    for value in graph.value_info:
        if value.name == OUT_NAME or not value.type.HasField("tensor_type"):
            continue
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        n = 1
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                n *= dim.dim_value
        itemsize = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)).itemsize
        size = int(n * itemsize)
        if size > largest_bytes:
            largest_name = value.name
            largest_bytes = size
    return memory, largest_name, largest_bytes


def main() -> None:
    builders: list[tuple[str, Callable[[], onnx.ModelProto]]] = [
        ("three_plane", build_three_plane),
        ("conv_rows", build_conv_rows),
        ("conv_rows_float", build_conv_rows_float),
    ]
    results: list[tuple[int, str, onnx.ModelProto, dict[str, object]]] = []
    for label, builder in builders:
        model = builder()
        tmp_path = OUT_DIR / f"{TASK_ID}_{label}.onnx"
        onnx.save(model, tmp_path)
        correct, counts = _check_correct(model)
        scored = score_file(tmp_path)
        memory, largest_name, largest_bytes = _profile_largest_internal(model)
        score_text = f"{scored['score']:.6f}" if scored.get("score") is not None else "INVALID"
        print(
            f"{label:<14} correct={correct} counts={counts} "
            f"memory={scored['memory']} params={scored['params']} cost={scored['cost']} "
            f"score={score_text} largest={largest_name}:{largest_bytes}"
        )
        if correct and scored["valid"]:
            results.append((int(scored["cost"]), label, model, scored))
        tmp_path.unlink(missing_ok=True)

    if not results:
        raise SystemExit("no valid correct variants")

    _, label, model, scored = min(results, key=lambda item: item[0])
    onnx.save(model, BEST_PATH)
    print()
    print(f"kept:   {label}")
    print(f"wrote:  {BEST_PATH}")
    print(f"memory: {scored['memory']}")
    print(f"params: {scored['params']}")
    print(f"cost:   {scored['cost']}")
    print(f"score:  {float(scored['score']):.6f}")


if __name__ == "__main__":
    main()
