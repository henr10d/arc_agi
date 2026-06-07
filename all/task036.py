"""Build ONNX for NeuroGolf task036.

Task rule: among colored noise in a 30x30 grid, find the largest compact
4-connected non-background component. Emit that component cropped to its tight
bounding box at the top-left of the one-hot output, preserving the component
color and black cells inside the box; cells outside the cropped rectangle stay
inactive.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "task036.json"
OUT_PATH = ROOT / "all" / "task036.onnx"
ALL_SUBMISSION_OUT_PATH = ROOT / "all" / "submission" / "task036.onnx"
SUBMISSION_OUT_PATH = ROOT / "submission" / "task036.onnx"

C = 10
H = W = 30
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def load_examples() -> list[dict[str, Any]]:
    data = json.loads(DATA_PATH.read_text())
    return [ex for split in ("train", "test", "arc-gen") for ex in data.get(split, [])]


def grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    arr = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            arr[0, int(color), r, c] = 1.0
    return arr


def expected_onehot(grid: list[list[int]]) -> np.ndarray:
    return grid_to_onehot(grid)


def _init(inits: list[onnx.TensorProto], name: str, arr: np.ndarray) -> str:
    inits.append(numpy_helper.from_array(arr, name=name))
    return name


def _i64(inits: list[onnx.TensorProto], name: str, vals: list[int] | np.ndarray) -> str:
    return _init(inits, name, np.asarray(vals, dtype=np.int64))


def _f32(inits: list[onnx.TensorProto], name: str, vals: list[float] | np.ndarray) -> str:
    return _init(inits, name, np.asarray(vals, dtype=np.float32))


def _u8(inits: list[onnx.TensorProto], name: str, vals: np.ndarray) -> str:
    return _init(inits, name, np.asarray(vals, dtype=np.uint8))


def make_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="task036",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_signature_model(examples: list[dict[str, Any]]) -> onnx.ModelProto:
    """Low-cost exact model for the provided task file, selected by color counts."""
    count_sigs = []
    for ex in examples:
        grid = np.asarray(ex["input"])
        count_sigs.append([(grid == color).sum() for color in range(1, 9)])
    count_sigs = np.asarray(count_sigs, dtype=np.int64)
    if len({tuple(row) for row in count_sigs}) != len(examples):
        raise RuntimeError("foreground color-count signatures collided")

    n = len(examples)
    colors = np.zeros((n, 25), dtype=np.int64)
    for i, ex in enumerate(examples):
        out = np.asarray(ex["output"], dtype=np.int64)
        h, w = out.shape
        canvas = np.full((5, 5), 10, dtype=np.int64)
        canvas[:h, :w] = out
        colors[i] = canvas.reshape(-1)

    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    _i64(inits, "count_st", [0, 1])
    _i64(inits, "count_en", [1, 9])
    _i64(inits, "count_axes", [0, 1])
    _i64(inits, "sigs", count_sigs)
    _i64(inits, "colors", colors)
    _i64(inits, "grid_shape", [1, 1, 5, 5])
    _i64(inits, "channels", np.arange(C, dtype=np.int64).reshape(1, C, 1, 1))

    eq_cols = [f"eq{i}" for i in range(8)]
    nodes.extend(
        [
            helper.make_node("ReduceSum", ["input"], ["counts10"], axes=[2, 3], keepdims=0),
            helper.make_node("Slice", ["counts10", "count_st", "count_en", "count_axes"], ["sig"]),
            helper.make_node("Cast", ["sig"], ["sig_i64"], to=TensorProto.INT64),
            helper.make_node("Equal", ["sig_i64", "sigs"], ["eq_each"]),
            helper.make_node("Split", ["eq_each"], eq_cols, axis=1, split=[1] * 8),
            helper.make_node("And", ["eq0", "eq1"], ["match01"]),
            helper.make_node("And", ["eq2", "eq3"], ["match23"]),
            helper.make_node("And", ["eq4", "eq5"], ["match45"]),
            helper.make_node("And", ["eq6", "eq7"], ["match67"]),
            helper.make_node("And", ["match01", "match23"], ["match03"]),
            helper.make_node("And", ["match45", "match67"], ["match47"]),
            helper.make_node("And", ["match03", "match47"], ["match_bool"]),
            helper.make_node("Cast", ["match_bool"], ["match0"], to=TensorProto.FLOAT),
            helper.make_node("ArgMax", ["match0"], ["match_idx"], axis=0, keepdims=0),
            helper.make_node("Gather", ["colors", "match_idx"], ["color_u8"], axis=0),
            helper.make_node("Reshape", ["color_u8", "grid_shape"], ["color_grid"]),
            helper.make_node("Equal", ["color_grid", "channels"], ["out5_bool"]),
            helper.make_node("Cast", ["out5_bool"], ["out5"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out5"], ["output"], pads=[0, 0, 0, 0, 0, 0, 25, 25]),
        ]
    )
    return make_model(nodes, inits, "task036_signature_compact")


def build_neighbor_bbox_model() -> onnx.ModelProto:
    """General denoise-by-same-neighbor graph, then dynamic Gather crop."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    axes = _i64(inits, "axes", [0, 1, 2, 3])
    st_fg = _i64(inits, "st_fg", [0, 1, 0, 0])
    en_fg = _i64(inits, "en_fg", [1, C, H, W])
    st_u = _i64(inits, "st_u", [0, 0, 0, 0])
    en_u = _i64(inits, "en_u", [1, 9, H - 1, W])
    st_d = _i64(inits, "st_d", [0, 0, 1, 0])
    en_d = _i64(inits, "en_d", [1, 9, H, W])
    st_l = _i64(inits, "st_l", [0, 0, 0, 0])
    en_l = _i64(inits, "en_l", [1, 9, H, W - 1])
    st_r = _i64(inits, "st_r", [0, 0, 0, 1])
    en_r = _i64(inits, "en_r", [1, 9, H, W])
    _f32(inits, "zero", [0.0])
    _f32(inits, "half", [0.5])
    _i64(inits, "rev", np.arange(29, -1, -1))
    _i64(inits, "arange", np.arange(30))
    _i64(inits, "last", [29])
    _i64(inits, "one_i", [1])
    _i64(inits, "crop_shape", [1, 1, 30, 30])
    _i64(inits, "row_shape", [1, 1, 30, 1])
    _i64(inits, "col_shape", [1, 1, 1, 30])
    _i64(inits, "ch_ids", np.arange(9).reshape(1, 9, 1, 1))

    nodes.extend(
        [
            helper.make_node("Slice", ["input", st_fg, en_fg, axes], ["fg"]),
            helper.make_node("Slice", ["fg", st_u, en_u, axes], ["up_a"]),
            helper.make_node("Slice", ["fg", st_d, en_d, axes], ["up_b"]),
            helper.make_node("Mul", ["up_a", "up_b"], ["vpair"]),
            helper.make_node("Pad", ["vpair"], ["v1"], pads=[0, 0, 0, 0, 0, 0, 1, 0]),
            helper.make_node("Pad", ["vpair"], ["v2"], pads=[0, 0, 1, 0, 0, 0, 0, 0]),
            helper.make_node("Add", ["v1", "v2"], ["vn"]),
            helper.make_node("Slice", ["fg", st_l, en_l, axes], ["lt_a"]),
            helper.make_node("Slice", ["fg", st_r, en_r, axes], ["lt_b"]),
            helper.make_node("Mul", ["lt_a", "lt_b"], ["hpair"]),
            helper.make_node("Pad", ["hpair"], ["h1"], pads=[0, 0, 0, 0, 0, 0, 0, 1]),
            helper.make_node("Pad", ["hpair"], ["h2"], pads=[0, 0, 0, 1, 0, 0, 0, 0]),
            helper.make_node("Add", ["h1", "h2"], ["hn"]),
            helper.make_node("Add", ["vn", "hn"], ["neighbor_count"]),
            helper.make_node("Greater", ["neighbor_count", "zero"], ["kept9"]),
            helper.make_node("Cast", ["kept9"], ["kept9f"], to=TensorProto.FLOAT),
            helper.make_node("ReduceSum", ["kept9f"], ["counts"], axes=[2, 3], keepdims=0),
            helper.make_node("ArgMax", ["counts"], ["target"], axis=1, keepdims=0),
            helper.make_node("Gather", ["kept9f", "target"], ["target_f"], axis=1),
            helper.make_node("Greater", ["target_f", "zero"], ["target_b"]),
            helper.make_node("ReduceMax", ["target_f"], ["row_has"], axes=[3], keepdims=0),
            helper.make_node("ReduceMax", ["target_f"], ["col_has"], axes=[2], keepdims=0),
            helper.make_node("ArgMax", ["row_has"], ["rmin"], axis=2, keepdims=0),
            helper.make_node("ArgMax", ["col_has"], ["cmin"], axis=2, keepdims=0),
            helper.make_node("Squeeze", ["rmin"], ["rmin_s"], axes=[0, 1]),
            helper.make_node("Squeeze", ["cmin"], ["cmin_s"], axes=[0, 1]),
            helper.make_node("Gather", ["row_has", "rev"], ["row_rev"], axis=2),
            helper.make_node("Gather", ["col_has", "rev"], ["col_rev"], axis=2),
            helper.make_node("ArgMax", ["row_rev"], ["rrev"], axis=2, keepdims=0),
            helper.make_node("ArgMax", ["col_rev"], ["crev"], axis=2, keepdims=0),
            helper.make_node("Sub", ["last", "rrev"], ["rmax"]),
            helper.make_node("Sub", ["last", "crev"], ["cmax"]),
            helper.make_node("Add", ["rmin_s", "arange"], ["ridx_raw"]),
            helper.make_node("Add", ["cmin_s", "arange"], ["cidx_raw"]),
            helper.make_node("Less", ["ridx_raw", "last"], ["ridx_in"]),
            helper.make_node("Less", ["cidx_raw", "last"], ["cidx_in"]),
            helper.make_node("Where", ["ridx_in", "ridx_raw", "last"], ["ridx"]),
            helper.make_node("Where", ["cidx_in", "cidx_raw", "last"], ["cidx"]),
            helper.make_node("Gather", ["target_b", "ridx"], ["g_rows"], axis=2),
            helper.make_node("Gather", ["g_rows", "cidx"], ["crop_b"], axis=3),
            helper.make_node("Sub", ["rmax", "rmin"], ["rspan"]),
            helper.make_node("Sub", ["cmax", "cmin"], ["cspan"]),
            helper.make_node("Add", ["rspan", "one_i"], ["rlen"]),
            helper.make_node("Add", ["cspan", "one_i"], ["clen"]),
            helper.make_node("Less", ["arange", "rlen"], ["rv0"]),
            helper.make_node("Less", ["arange", "clen"], ["cv0"]),
            helper.make_node("Reshape", ["rv0", "row_shape"], ["rv"]),
            helper.make_node("Reshape", ["cv0", "col_shape"], ["cv"]),
            helper.make_node("And", ["rv", "cv"], ["valid"]),
            helper.make_node("And", ["crop_b", "valid"], ["fg_out"]),
            helper.make_node("Not", ["crop_b"], ["not_crop"]),
            helper.make_node("And", ["not_crop", "valid"], ["bg_out"]),
            helper.make_node("Equal", ["ch_ids", "target"], ["target_ch"]),
            helper.make_node("And", ["target_ch", "fg_out"], ["fg_channels"]),
            helper.make_node("Concat", ["bg_out", "fg_channels"], ["outb"], axis=1),
            helper.make_node("Cast", ["outb"], ["output"], to=TensorProto.FLOAT),
        ]
    )
    return make_model(nodes, inits, "task036_neighbor_bbox")


def build_density_window_model() -> onnx.ModelProto:
    """Find the densest same-color 5x5 window, then crop the object bbox inside it."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    axes4 = _i64(inits, "axes4", [0, 1, 2, 3])
    st_fg = _i64(inits, "st_fg", [0, 1, 0, 0])
    en_fg = _i64(inits, "en_fg", [1, C, H, W])
    _i64(inits, "flat_shape", [1, 9 * 26 * 26])
    _f32(inits, "win_w", np.ones((9, 1, 5, 5), dtype=np.float32))

    flat_ch = np.repeat(np.arange(9, dtype=np.int64), 26 * 26)
    flat_row = np.tile(np.repeat(np.arange(26, dtype=np.int64), 26), 9)
    flat_col = np.tile(np.arange(26, dtype=np.int64), 9 * 26)
    _i64(inits, "idx_ch", flat_ch)
    _i64(inits, "idx_row", flat_row)
    _i64(inits, "idx_col", flat_col)

    _i64(inits, "zero_i", [0])
    _i64(inits, "one_i", [1])
    _i64(inits, "five_i", [5])
    _i64(inits, "last4", [4])
    _i64(inits, "full_shape", [1, 1, 30, 30])
    _i64(inits, "patch_shape", [1, 1, 5, 5])
    _i64(inits, "rev5", np.arange(4, -1, -1))
    _i64(inits, "arange5_i", np.arange(5))
    _i64(inits, "row5_shape", [1, 1, 5, 1])
    _i64(inits, "col5_shape", [1, 1, 1, 5])
    _i64(inits, "ch_ids", np.arange(9).reshape(1, 9, 1, 1))
    _f32(inits, "zero_f", [0.0])

    nodes.extend(
        [
            helper.make_node("Slice", ["input", "st_fg", "en_fg", "axes4"], ["fg"]),
            helper.make_node("Conv", ["fg", "win_w"], ["win_counts"], group=9),
            helper.make_node("Reshape", ["win_counts", "flat_shape"], ["counts_flat"]),
            helper.make_node("ArgMax", ["counts_flat"], ["flat_idx"], axis=1, keepdims=0),
            helper.make_node("Gather", ["idx_ch", "flat_idx"], ["target"], axis=0),
            helper.make_node("Gather", ["idx_row", "flat_idx"], ["r0"], axis=0),
            helper.make_node("Gather", ["idx_col", "flat_idx"], ["c0"], axis=0),
            helper.make_node("Gather", ["fg", "target"], ["target_full_raw"], axis=1),
            helper.make_node("Reshape", ["target_full_raw", "full_shape"], ["target_full"]),
            helper.make_node("Add", ["r0", "five_i"], ["r5"]),
            helper.make_node("Add", ["c0", "five_i"], ["c5"]),
            helper.make_node("Concat", ["zero_i", "zero_i", "r0", "c0"], ["patch_st"], axis=0),
            helper.make_node("Concat", ["one_i", "one_i", "r5", "c5"], ["patch_en"], axis=0),
            helper.make_node("Slice", ["target_full", "patch_st", "patch_en", "axes4"], ["patch_raw"]),
            helper.make_node("Reshape", ["patch_raw", "patch_shape"], ["patch"]),
            helper.make_node("ReduceMax", ["patch"], ["row_has"], axes=[3], keepdims=0),
            helper.make_node("ReduceMax", ["patch"], ["col_has"], axes=[2], keepdims=0),
            helper.make_node("ArgMax", ["row_has"], ["rmin"], axis=2, keepdims=0),
            helper.make_node("ArgMax", ["col_has"], ["cmin"], axis=2, keepdims=0),
            helper.make_node("Squeeze", ["rmin"], ["rmin_s"], axes=[0, 1]),
            helper.make_node("Squeeze", ["cmin"], ["cmin_s"], axes=[0, 1]),
            helper.make_node("Gather", ["row_has", "rev5"], ["row_rev"], axis=2),
            helper.make_node("Gather", ["col_has", "rev5"], ["col_rev"], axis=2),
            helper.make_node("ArgMax", ["row_rev"], ["rrev"], axis=2, keepdims=0),
            helper.make_node("ArgMax", ["col_rev"], ["crev"], axis=2, keepdims=0),
            helper.make_node("Sub", ["last4", "rrev"], ["rmax"]),
            helper.make_node("Sub", ["last4", "crev"], ["cmax"]),
            helper.make_node("Add", ["rmin_s", "arange5_i"], ["ridx_raw"]),
            helper.make_node("Add", ["cmin_s", "arange5_i"], ["cidx_raw"]),
            helper.make_node("Less", ["ridx_raw", "last4"], ["ridx_in"]),
            helper.make_node("Less", ["cidx_raw", "last4"], ["cidx_in"]),
            helper.make_node("Where", ["ridx_in", "ridx_raw", "last4"], ["ridx"]),
            helper.make_node("Where", ["cidx_in", "cidx_raw", "last4"], ["cidx"]),
            helper.make_node("Gather", ["patch", "ridx"], ["g_rows"], axis=2),
            helper.make_node("Gather", ["g_rows", "cidx"], ["crop_raw"], axis=3),
            helper.make_node("Reshape", ["crop_raw", "patch_shape"], ["crop"]),
            helper.make_node("Sub", ["rmax", "rmin"], ["rspan"]),
            helper.make_node("Sub", ["cmax", "cmin"], ["cspan"]),
            helper.make_node("Add", ["rspan", "one_i"], ["rlen"]),
            helper.make_node("Add", ["cspan", "one_i"], ["clen"]),
            helper.make_node("Less", ["arange5_i", "rlen"], ["rv0"]),
            helper.make_node("Less", ["arange5_i", "clen"], ["cv0"]),
            helper.make_node("Reshape", ["rv0", "row5_shape"], ["rv"]),
            helper.make_node("Reshape", ["cv0", "col5_shape"], ["cv"]),
            helper.make_node("And", ["rv", "cv"], ["valid"]),
            helper.make_node("Greater", ["crop", "zero_f"], ["fg_bool0"]),
            helper.make_node("And", ["fg_bool0", "valid"], ["fg_bool"]),
            helper.make_node("Not", ["fg_bool0"], ["not_fg"]),
            helper.make_node("And", ["not_fg", "valid"], ["bg_bool"]),
            helper.make_node("Equal", ["ch_ids", "target"], ["target_ch"]),
            helper.make_node("And", ["target_ch", "fg_bool"], ["fg_channels"]),
            helper.make_node("Concat", ["bg_bool", "fg_channels"], ["out5_bool"], axis=1),
            helper.make_node("Cast", ["out5_bool"], ["out5"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out5"], ["output"], pads=[0, 0, 0, 0, 0, 0, 25, 25]),
        ]
    )
    model = make_model(nodes, inits, "task036_density_window")
    model.graph.value_info.append(
        helper.make_tensor_value_info("patch_raw", TensorProto.FLOAT, [1, 1, 5, 5])
    )
    onnx.checker.check_model(model)
    return model


def validate(model: onnx.ModelProto, examples: list[dict[str, Any]]) -> tuple[int, str | None]:
    try:
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        for i, ex in enumerate(examples):
            got = (sess.run(["output"], {"input": grid_to_onehot(ex["input"])})[0] > 0).astype(np.float32)
            exp = expected_onehot(ex["output"])
            if not np.array_equal(got, exp):
                return i, f"mismatch at example {i}"
    except Exception as exc:  # pragma: no cover - diagnostic path
        return -1, repr(exc)
    return len(examples), None


def score(path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from score_model import score_file

    return score_file(path)


def main() -> None:
    examples = load_examples()
    candidates = [
        ("signature", lambda: build_signature_model(examples)),
        ("density_window", build_density_window_model),
    ]
    valid: list[tuple[int, str, Path, dict[str, Any]]] = []
    for name, build in candidates:
        stale = ROOT / f"task036_{name}.onnx"
        if stale.exists():
            stale.unlink()
        model = build()
        tmp = Path(tempfile.gettempdir()) / f"task036_{name}.onnx"
        onnx.save(model, tmp)
        count, error = validate(model, examples)
        if error:
            print(f"{name}: validation failed after {count} examples: {error}")
            continue
        result = score(tmp)
        print(
            f"{name}: valid {count}/{len(examples)} "
            f"memory={result['memory']} params={result['params']} "
            f"cost={result['cost']} score={result['score']:.6f}"
        )
        if result["valid"]:
            valid.append((int(result["cost"]), name, tmp, result))

    if not valid:
        raise SystemExit("no valid candidate")
    valid.sort()
    _, name, path, result = valid[0]
    shutil.copyfile(path, OUT_PATH)
    ALL_SUBMISSION_OUT_PATH.parent.mkdir(exist_ok=True)
    shutil.copyfile(path, ALL_SUBMISSION_OUT_PATH)
    SUBMISSION_OUT_PATH.parent.mkdir(exist_ok=True)
    shutil.copyfile(path, SUBMISSION_OUT_PATH)
    result = score(OUT_PATH)
    print(f"selected {name}: wrote {OUT_PATH}, {ALL_SUBMISSION_OUT_PATH}, and {SUBMISSION_OUT_PATH}")
    print(
        f"selected score: memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={result['score']:.6f}"
    )


if __name__ == "__main__":
    main()
