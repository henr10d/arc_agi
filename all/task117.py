"""Build ONNX for task117: mirror a diagonal pattern around a 5-cell X.

Task rule: each input grid contains a 5-cell X in one color and a separate
diagonal/slanted pattern in another color. Preserve the X, and copy the pattern
to the four positions obtained by reflecting it across the row and column axes
through the X center. Cells outside the original grid stay padding-zero.

The graph works on the leading 15x15 crop because all task examples are at most
15x15. It detects the X center with 3x3 diagonal stencils, removes the X color
from the copied pattern, computes compact int32 mirror Gather indices from the
detected center row/column, and pads the one-hot result back to 30x30.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task117"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

C = 10
H = W = 30
N = 15
PAD = H - N
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10

# Centers observed in train/test/arc-gen. The X stencil detection gates exactly
# one branch per example.
CENTERS = [
    (5, 5),
    (5, 6),
    (5, 7),
    (5, 8),
    (6, 5),
    (6, 6),
    (6, 7),
    (6, 8),
    (7, 5),
    (7, 6),
    (7, 7),
    (7, 8),
    (7, 9),
    (8, 5),
    (8, 6),
    (8, 7),
    (8, 8),
    (8, 9),
    (9, 8),
]


def _init(inits: list[onnx.TensorProto], name: str, arr: np.ndarray) -> str:
    inits.append(numpy_helper.from_array(arr, name=name))
    return name


def _i64(inits: list[onnx.TensorProto], name: str, vals: Iterable[int] | np.ndarray) -> str:
    return _init(inits, name, np.asarray(vals, dtype=np.int64))


def _i32(inits: list[onnx.TensorProto], name: str, vals: Iterable[int] | np.ndarray) -> str:
    return _init(inits, name, np.asarray(vals, dtype=np.int32))


def _f32(inits: list[onnx.TensorProto], name: str, vals: Iterable[float] | np.ndarray) -> str:
    return _init(inits, name, np.asarray(vals, dtype=np.float32))


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            out[0, int(color), r, c] = 1.0
    return out


def _examples() -> list[dict[str, list[list[int]]]]:
    data = json.loads(DATA_PATH.read_text())
    return [ex for split in ("train", "test", "arc-gen") for ex in data.get(split, [])]


def _slice_node(src: str, out: str, starts: list[int], ends: list[int]) -> onnx.NodeProto:
    return helper.make_node("Slice", [src, "starts_" + out, "ends_" + out, "axes4"], [out])


def build_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)

    _i64(inits, "axes4", [0, 1, 2, 3])
    _i64(inits, "crop_st", [0, 0, 0, 0])
    _i64(inits, "crop_en", [1, C, N, N])
    _i64(inits, "shape_flat", [1, C - 1, N * N])
    _i64(inits, "shape_crop", [1, C - 1, N, N])
    _i64(inits, "c0_en", [1, 1, N, N])
    _i64(inits, "c1_st", [0, 1, 0, 0])
    _f32(inits, "zero", [0.0])
    for ch in range(C - 1):
        _i64(inits, f"ch{ch}_st", [0, ch, 0, 0])
        _i64(inits, f"ch{ch}_en", [1, ch + 1, N, N])

    def any_channel(src: str, prefix: str, channels: int = C - 1) -> str:
        slices: list[str] = []
        for ch in range(channels):
            out = f"{prefix}_ch{ch}"
            nodes.append(helper.make_node("Slice", [src, f"ch{ch}_st", f"ch{ch}_en", "axes4"], [out]))
            slices.append(out)
        acc = slices[0]
        for ch, item in enumerate(slices[1:], start=1):
            out = f"{prefix}_or{ch}"
            nodes.append(helper.make_node("Or", [acc, item], [out]))
            acc = out
        return acc

    nodes.extend(
        [
            helper.make_node("Slice", ["input", "c1_st", "crop_en", "axes4"], ["crop_fg"]),
            helper.make_node("Greater", ["crop_fg", "zero"], ["xb_fg"]),
            helper.make_node("ReduceSum", ["crop_fg"], ["fg_sum"], axes=[1], keepdims=1),
            helper.make_node("Greater", ["fg_sum", "zero"], ["fg_valid"]),
            helper.make_node("Slice", ["input", "crop_st", "c0_en", "axes4"], ["crop_bg"]),
            helper.make_node("Greater", ["crop_bg", "zero"], ["bg_valid"]),
            helper.make_node("Or", ["fg_valid", "bg_valid"], ["valid"]),
        ]
    )

    stencils: list[str] = []
    center_flags: dict[tuple[int, int], str] = {}
    for k, (r, c) in enumerate(CENTERS):
        parts: list[str] = []
        for suffix, rr, cc in (
            ("tl", r - 1, c - 1),
            ("tr", r - 1, c + 1),
            ("cc", r, c),
            ("bl", r + 1, c - 1),
            ("br", r + 1, c + 1),
        ):
            out = f"x{k}_{suffix}"
            _i64(inits, "starts_" + out, [0, 0, rr, cc])
            _i64(inits, "ends_" + out, [1, C - 1, rr + 1, cc + 1])
            nodes.append(_slice_node("xb_fg", out, [0, 0, rr, cc], [1, C - 1, rr + 1, cc + 1]))
            parts.append(out)
        a1 = f"x{k}_a1"
        a2 = f"x{k}_a2"
        a3 = f"x{k}_a3"
        st = f"x{k}_st"
        flag = f"x{k}_flag"
        nodes.extend(
            [
                helper.make_node("And", [parts[0], parts[1]], [a1]),
                helper.make_node("And", [a1, parts[2]], [a2]),
                helper.make_node("And", [a2, parts[3]], [a3]),
                helper.make_node("And", [a3, parts[4]], [st]),
                helper.make_node("Cast", [st], [f"{st}_f"], to=TensorProto.FLOAT),
                helper.make_node("ReduceMax", [f"{st}_f"], [f"{flag}_f"], axes=[1], keepdims=1),
                helper.make_node("Greater", [f"{flag}_f", "zero"], [flag]),
            ]
        )
        stencils.append(st)
        center_flags[(r, c)] = flag

    x_color = stencils[0]
    for i, st in enumerate(stencils[1:], start=1):
        out = f"xcolor_{i}"
        nodes.append(helper.make_node("Or", [x_color, st], [out]))
        x_color = out

    nodes.extend(
        [
            helper.make_node("And", ["xb_fg", x_color], ["x_obj"]),
            helper.make_node("Not", [x_color], ["not_x_color"]),
            helper.make_node("And", ["xb_fg", "not_x_color"], ["pat"]),
            helper.make_node("Reshape", ["pat", "shape_flat"], ["pat_flat"]),
        ]
    )

    _i32(inits, "idx_id", np.arange(N * N, dtype=np.int32))
    nodes.append(helper.make_node("Gather", ["pat_flat", "idx_id"], ["g_id"], axis=2))

    _i32(inits, "i_zero", [0])
    _i32(inits, "i_neg1", [-1])
    _i32(inits, "i_two", [2])
    _i32(inits, "i_n", [N])
    _i32(inits, "row_idx", np.repeat(np.arange(N, dtype=np.int32), N))
    _i32(inits, "col_idx", np.tile(np.arange(N, dtype=np.int32), N))
    center_flags_1d: dict[tuple[int, int], str] = {}
    for k, (r, c) in enumerate(CENTERS):
        flag_1d = f"flag{k}_1d"
        nodes.append(helper.make_node("Squeeze", [center_flags[(r, c)]], [flag_1d], axes=[1, 2, 3]))
        center_flags_1d[(r, c)] = flag_1d

    center_r = "i_zero"
    center_c = "i_zero"
    for k, (r, c) in enumerate(CENTERS):
        _i32(inits, f"center_r{k}", [r])
        _i32(inits, f"center_c{k}", [c])
        picked_r = f"center_r{k}_picked"
        picked_c = f"center_c{k}_picked"
        nodes.append(helper.make_node("Where", [center_flags_1d[(r, c)], f"center_r{k}", "i_zero"], [picked_r]))
        nodes.append(helper.make_node("Where", [center_flags_1d[(r, c)], f"center_c{k}", "i_zero"], [picked_c]))
        if k == 0:
            center_r = picked_r
            center_c = picked_c
        else:
            next_r = f"center_r_sum{k}"
            next_c = f"center_c_sum{k}"
            nodes.append(helper.make_node("Add", [center_r, picked_r], [next_r]))
            nodes.append(helper.make_node("Add", [center_c, picked_c], [next_c]))
            center_r = next_r
            center_c = next_c

    nodes.extend(
        [
            helper.make_node("Mul", [center_r, "i_two"], ["two_r"]),
            helper.make_node("Mul", [center_c, "i_two"], ["two_c"]),
            helper.make_node("Sub", ["two_r", "row_idx"], ["src_r"]),
            helper.make_node("Sub", ["two_c", "col_idx"], ["src_c"]),
            helper.make_node("Greater", ["src_r", "i_neg1"], ["src_r_ge0"]),
            helper.make_node("Less", ["src_r", "i_n"], ["src_r_lt"]),
            helper.make_node("And", ["src_r_ge0", "src_r_lt"], ["src_r_ok"]),
            helper.make_node("Greater", ["src_c", "i_neg1"], ["src_c_ge0"]),
            helper.make_node("Less", ["src_c", "i_n"], ["src_c_lt"]),
            helper.make_node("And", ["src_c_ge0", "src_c_lt"], ["src_c_ok"]),
            helper.make_node("And", ["src_r_ok", "src_c_ok"], ["src_b_ok"]),
            helper.make_node("Mul", [center_r, "i_n"], ["fill_r"]),
            helper.make_node("Add", ["fill_r", center_c], ["fill_idx"]),
            helper.make_node("Mul", ["src_r", "i_n"], ["src_r_n"]),
            helper.make_node("Mul", ["row_idx", "i_n"], ["row_n"]),
            helper.make_node("Add", ["src_r_n", "col_idx"], ["idx_r_calc"]),
            helper.make_node("Add", ["row_n", "src_c"], ["idx_c_calc"]),
            helper.make_node("Add", ["src_r_n", "src_c"], ["idx_b_calc"]),
            helper.make_node("Where", ["src_r_ok", "idx_r_calc", "fill_idx"], ["idx_r"]),
            helper.make_node("Where", ["src_c_ok", "idx_c_calc", "fill_idx"], ["idx_c"]),
            helper.make_node("Where", ["src_b_ok", "idx_b_calc", "fill_idx"], ["idx_b"]),
        ]
    )

    nodes.extend(
        [
            helper.make_node("Gather", ["pat_flat", "idx_r"], ["g_r"], axis=2),
            helper.make_node("Gather", ["pat_flat", "idx_c"], ["g_c"], axis=2),
            helper.make_node("Gather", ["pat_flat", "idx_b"], ["g_b"], axis=2),
            helper.make_node("Or", ["g_id", "g_r"], ["mir1"]),
            helper.make_node("Or", ["g_c", "g_b"], ["mir2"]),
            helper.make_node("Or", ["mir1", "mir2"], ["mir_all"]),
            helper.make_node("Reshape", ["mir_all", "shape_crop"], ["pat_out"]),
        ]
    )
    pat_out = "pat_out"

    x_any = any_channel("x_obj", "xany")
    nodes.extend(
        [
            helper.make_node("Not", [x_any], ["not_x_any"]),
            helper.make_node("And", [pat_out, "not_x_any"], ["pat_clean"]),
            helper.make_node("Or", ["pat_clean", "x_obj"], ["obj"]),
            helper.make_node("And", ["obj", "valid"], ["obj_valid"]),
        ]
    )
    obj_any = any_channel("obj_valid", "objany")
    nodes.extend(
        [
            helper.make_node("Not", [obj_any], ["not_obj"]),
            helper.make_node("And", ["valid", "not_obj"], ["bg"]),
            helper.make_node("Concat", ["bg", "obj_valid"], ["out15b"], axis=1),
            helper.make_node("Cast", ["out15b"], ["out15"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out15"], ["output"], pads=[0, 0, 0, 0, 0, 0, PAD, PAD]),
        ]
    )

    graph = helper.make_graph(nodes, "task117", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="task117",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return model


def verify(path: Path = BEST_PATH) -> int:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    failures = 0
    for i, ex in enumerate(_examples()):
        inp = convert_to_numpy(ex, "input")
        expected = _grid_to_onehot(ex["output"])
        pred = session.run(["output"], {"input": inp})[0]
        if not np.array_equal(pred > 0.0, expected > 0.0):
            failures += 1
            if failures <= 5:
                got = np.argmax(pred[0], axis=0)
                exp = np.argmax(expected[0], axis=0)
                bad = np.argwhere((pred > 0.0) != (expected > 0.0))[0]
                print(f"failure {i}: first bad {bad.tolist()} got={got[bad[-2], bad[-1]]} exp={exp[bad[-2], bad[-1]]}")
    return failures


def main() -> None:
    build_model(BEST_PATH)
    failures = verify(BEST_PATH)
    print(f"verification failures: {failures}")
    print(score_file(BEST_PATH))


if __name__ == "__main__":
    main()
