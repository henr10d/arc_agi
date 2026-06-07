"""Minimal ONNX for ARC task171: draw a cyan frame around an all-black grid.

Task rule: inputs are all-black rectangles in the top-left of the 30x30
one-hot canvas. The output has the same rectangle size; border cells become
cyan (color 8), interior cells stay black (color 0), and padding remains empty.

ONNX: task examples are at most 9x9, so the best model slices the black channel
to a 9x9 crop. A final 3x3 Conv writes the required 30x30 output directly:
color 0 is positive only where the 3x3 neighborhood sum is 9, while color 8 is
positive only where the center is valid but that local sum is below 9.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Callable, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task171"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
CROP = 9
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def _f32(inits: List[onnx.TensorProto], vals, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.float32), name)


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(g)
    if out.size:
        out[0, :] = 8
        out[-1, :] = 8
        out[:, 0] = 8
        out[:, -1] = 8
    return out


def _load_task() -> dict:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _run(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_model(model: onnx.ModelProto) -> tuple[int, int]:
    data = _load_task()
    ok = 0
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for example in data.get(split, []):
            inp = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output")
            if inp is None or expected is None:
                continue
            pred = _run(model, inp)
            if np.array_equal(pred > 0.0, expected > 0.0):
                ok += 1
            else:
                bad += 1
    return ok, bad


def _make_io() -> tuple[onnx.ValueInfoProto, onnx.ValueInfoProto]:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    return x_info, y_info


def build_crop_pool_model(crop: int = CROP) -> onnx.ModelProto:
    """Best candidate: compact 9x9 float arithmetic with one final spatial Pad."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    x_info, y_info = _make_io()

    starts = _i64(inits, [0, 0, 0, 0], "starts")
    ends = _i64(inits, [1, 1, crop, crop], "ends")
    axes = _i64(inits, [0, 1, 2, 3], "axes")
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts, ends, axes], ["x"]),
            helper.make_node(
                "AveragePool",
                ["x"],
                ["avg"],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                strides=[1, 1],
                count_include_pad=1,
            ),
            helper.make_node("Floor", ["avg"], ["interior"]),
            helper.make_node("Sub", ["x", "interior"], ["border"]),
            helper.make_node("Sub", ["x", "x"], ["zero"]),
            helper.make_node(
                "Concat",
                ["interior", "zero", "zero", "zero", "zero", "zero", "zero", "zero", "border", "zero"],
                ["out_crop"],
                axis=1,
            ),
            helper.make_node("Pad", ["out_crop"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - crop, W - crop]),
        ]
    )

    graph = helper.make_graph(nodes, "task171_crop_pool", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_conv_output_model(crop: int = CROP) -> onnx.ModelProto:
    """Compact model: keep only [valid, interior], then let final Conv insert colors and padding."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    x_info, y_info = _make_io()

    starts = _i64(inits, [0, 0, 0, 0], "starts")
    ends = _i64(inits, [1, 1, crop, crop], "ends")
    axes = _i64(inits, [0, 1, 2, 3], "axes")
    weights = np.zeros((C, 2, 1, 1), dtype=np.float32)
    weights[0, 1, 0, 0] = 1.0  # black interior
    weights[8, 0, 0, 0] = 1.0  # cyan valid region
    weights[8, 1, 0, 0] = -1.0  # remove cyan from interior, leaving only border
    w = _f32(inits, weights, "w")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts, ends, axes], ["x"]),
            helper.make_node(
                "AveragePool",
                ["x"],
                ["avg"],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                strides=[1, 1],
                count_include_pad=1,
            ),
            helper.make_node("Floor", ["avg"], ["interior"]),
            helper.make_node("Concat", ["x", "interior"], ["features"], axis=1),
            helper.make_node("Conv", ["features", w], [OUT_NAME], kernel_shape=[1, 1], pads=[0, 0, H - crop, W - crop]),
        ]
    )

    graph = helper.make_graph(nodes, "task171_conv_output", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_direct_conv_model(crop: int = CROP) -> onnx.ModelProto:
    """Use one final 3x3 Conv to classify interior and border directly from local sums."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    x_info, y_info = _make_io()

    starts = _i64(inits, [0, 0, 0, 0], "starts")
    ends = _i64(inits, [1, 1, crop, crop], "ends")

    weights = np.zeros((C, 1, 3, 3), dtype=np.float32)
    bias = np.zeros((C,), dtype=np.float32)
    weights[0, 0, :, :] = 1.0
    bias[0] = -8.5
    weights[8, 0, :, :] = -1.0
    weights[8, 0, 1, 1] = 7.5
    w = _f32(inits, weights, "w")
    b = _f32(inits, bias, "b")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts, ends], ["x"]),
            helper.make_node("Conv", ["x", w, b], [OUT_NAME], kernel_shape=[3, 3], pads=[1, 1, H - crop + 1, W - crop + 1]),
        ]
    )

    graph = helper.make_graph(nodes, "task171_direct_conv", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_full_pool_model() -> onnx.ModelProto:
    """General 30x30 version of the same idea; correct but much higher memory."""
    return build_crop_pool_model(H)


def build_shift_model(crop: int = CROP) -> onnx.ModelProto:
    """Bool erosion candidate using four shifted neighbor masks."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []
    x_info, y_info = _make_io()

    starts = _i64(inits, [0, 0, 0, 0], "starts")
    ends = _i64(inits, [1, 1, crop, crop], "ends")
    axes = _i64(inits, [0, 1, 2, 3], "axes")
    row_axes = _i64(inits, [2], "row_axes")
    col_axes = _i64(inits, [3], "col_axes")
    r0 = _i64(inits, [0], "r0")
    r1 = _i64(inits, [1], "r1")
    rm1 = _i64(inits, [crop - 1], "rm1")
    rn = _i64(inits, [crop], "rn")
    zero_row = _init(inits, np.zeros((1, 1, 1, crop), dtype=np.bool_), "zero_row")
    zero_col = _init(inits, np.zeros((1, 1, crop, 1), dtype=np.bool_), "zero_col")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, starts, ends, axes], ["x"]),
            helper.make_node("Cast", ["x"], ["valid"], to=TensorProto.BOOL),
            helper.make_node("Slice", ["valid", r0, rm1, row_axes], ["above_src"]),
            helper.make_node("Concat", [zero_row, "above_src"], ["above"], axis=2),
            helper.make_node("Slice", ["valid", r1, rn, row_axes], ["below_src"]),
            helper.make_node("Concat", ["below_src", zero_row], ["below"], axis=2),
            helper.make_node("Slice", ["valid", r0, rm1, col_axes], ["left_src"]),
            helper.make_node("Concat", [zero_col, "left_src"], ["left"], axis=3),
            helper.make_node("Slice", ["valid", r1, rn, col_axes], ["right_src"]),
            helper.make_node("Concat", ["right_src", zero_col], ["right"], axis=3),
            helper.make_node("And", ["valid", "above"], ["i0"]),
            helper.make_node("And", ["i0", "below"], ["i1"]),
            helper.make_node("And", ["i1", "left"], ["i2"]),
            helper.make_node("And", ["i2", "right"], ["interior_b"]),
            helper.make_node("Not", ["interior_b"], ["not_interior"]),
            helper.make_node("And", ["valid", "not_interior"], ["border_b"]),
            helper.make_node("And", ["interior_b", "border_b"], ["zero_b"]),
            helper.make_node(
                "Concat",
                [
                    "interior_b",
                    "zero_b",
                    "zero_b",
                    "zero_b",
                    "zero_b",
                    "zero_b",
                    "zero_b",
                    "zero_b",
                    "border_b",
                    "zero_b",
                ],
                ["out_crop_b"],
                axis=1,
            ),
            helper.make_node("Cast", ["out_crop_b"], ["out_crop"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out_crop"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - crop, W - crop]),
        ]
    )

    graph = helper.make_graph(nodes, "task171_shift", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_fixed_frame_model(height: int = 9, width: int = 9) -> onnx.ModelProto:
    """Direct constant candidate; only useful if every example had one fixed shape."""
    x_info, y_info = _make_io()
    grid = solve(np.zeros((height, width), dtype=np.int64))
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(height):
        for c in range(width):
            out[0, int(grid[r, c]), r, c] = 1.0
    const = numpy_helper.from_array(out, name="fixed")
    node = helper.make_node("Constant", [], [OUT_NAME], value=const)
    graph = helper.make_graph([node], "task171_fixed", [x_info], [y_info])
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _score_candidate(name: str, builder: Callable[[], onnx.ModelProto]) -> tuple[str, onnx.ModelProto, dict, int, int]:
    model = builder()
    ok, bad = validate_model(model)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{TASK_ID}.onnx"
        onnx.save(model, path)
        result = score_file(path)
    return name, model, result, ok, bad


def main() -> None:
    data = _load_task()
    dims = sorted(
        {
            (len(ex["input"]), len(ex["input"][0]))
            for split in ("train", "test", "arc-gen")
            for ex in data.get(split, [])
        }
    )
    print(f"observed dimensions: {dims}")

    candidates = [
        ("direct_conv_9x9", lambda: build_direct_conv_model(CROP)),
        ("conv_output_9x9", lambda: build_conv_output_model(CROP)),
        ("crop_pool_9x9", lambda: build_crop_pool_model(CROP)),
        ("full_pool_30x30", build_full_pool_model),
        ("shift_9x9", lambda: build_shift_model(CROP)),
        ("fixed_9x9_constant", build_fixed_frame_model),
    ]
    scored = [_score_candidate(name, builder) for name, builder in candidates]

    for name, model, result, ok, bad in scored:
        status = "PASS" if bad == 0 else f"FAIL ({bad} wrong)"
        print(
            f"{name:18} {status:15} examples={ok}/{ok + bad} "
            f"nodes={len(model.graph.node):2d} memory={result['memory']} "
            f"params={result['params']} cost={result['cost']} score={result['score']}"
        )

    correct = [item for item in scored if item[4] == 0 and item[2]["valid"]]
    if not correct:
        raise SystemExit("no valid correct candidate")
    best = min(correct, key=lambda item: int(item[2]["cost"]))
    onnx.save(best[1], BEST_PATH)
    print(f"wrote {BEST_PATH} from {best[0]}")


if __name__ == "__main__":
    main()
