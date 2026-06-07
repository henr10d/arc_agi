"""ONNX solver for ARC task127: marker colors fill fixed separator bands.

Task rule: the input is either one 3x11 band or two 3x11 bands separated by a
full gray row.  In each band, gray columns 3 and 7 split the row into three
width-3 segments.  The center row contains one marker in each segment, with
colors 1..4 mapping to output fill colors 6..9.  The output fills all three
rows of each band segment with that mapped color, keeps gray separator cells,
and leaves padding outside the logical grid all-zero for NeuroGolf one-hot I/O.

ONNX: read the marker vectors from channels 1..4 at fixed center columns,
expand those tiny bool vectors into 3x3 segment blocks, synthesize the gray
separator mask from constants gated by the lower-band marker, and build the
compact [1,10,7,11] one-hot tensor before the final Cast+Pad to the required
[1,10,30,30] output.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task127"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
VARIANT_DIR = OUT_DIR / "_task127_variants"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, 10, 30, 30]
IR_VERSION = 10
OPSET = 10


@dataclass(frozen=True)
class Candidate:
    name: str
    build: Callable[[], onnx.ModelProto]
    random_ok: bool = False


def _init(array: object, name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(array), name=name)


def _make_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)],
        initializer=inits,
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


def _finalize(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], color4: str, gray: str, name: str) -> onnx.ModelProto:
    nodes.extend(
        [
            helper.make_node("Concat", [gray, color4], ["out5b"], axis=1),
            helper.make_node("Cast", ["out5b"], ["out5"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["out5"],
                [OUT_NAME],
                mode="constant",
                pads=[0, 5, 0, 0, 0, 0, 23, 19],
            ),
        ]
    )
    return _make_model(nodes, inits, name)


def _add_common_gray(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto]) -> str:
    inits.extend(
        [
            _init(np.asarray([5, 0, 0], dtype=np.int64), "gray_st"),
            _init(np.asarray([6, 7, 11], dtype=np.int64), "gray_en"),
            _init(np.asarray([1, 2, 3], dtype=np.int64), "axes3"),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, "gray_st", "gray_en", "axes3"], ["gray_f"]),
            helper.make_node("Cast", ["gray_f"], ["gray"], to=TensorProto.BOOL),
        ]
    )
    return "gray"


def _add_gated_gray_from_marker(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], marker_float: str) -> str:
    top = np.zeros((1, 1, 3, 11), dtype=np.bool_)
    top[:, :, :, [3, 7]] = True
    bottom = np.zeros((1, 1, 4, 11), dtype=np.bool_)
    bottom[:, :, 0, :] = True
    bottom[:, :, 1:, [3, 7]] = True
    inits.extend(
        [
            _init(top, "top_gray"),
            _init(bottom, "bottom_gray_template"),
        ]
    )
    nodes.extend(
        [
            helper.make_node("ReduceMax", [marker_float], ["h7_f"], axes=[1], keepdims=1),
            helper.make_node("Cast", ["h7_f"], ["h7"], to=TensorProto.BOOL),
            helper.make_node("And", ["bottom_gray_template", "h7"], ["bottom_gray"]),
            helper.make_node("Concat", ["top_gray", "bottom_gray"], ["gray"], axis=2),
        ]
    )
    return "gray"


def _segment_concat(nodes: list[onnx.NodeProto], segs: list[str], prefix: str) -> str:
    nodes.append(helper.make_node("Concat", [segs[0], "zero_col4", segs[1], "zero_col4", segs[2]], [prefix], axis=3))
    return prefix


def _two_band_concat(nodes: list[onnx.NodeProto], top: str, bottom: str, prefix: str) -> str:
    nodes.append(helper.make_node("Concat", [top, "zero_row4", bottom], [prefix], axis=2))
    return prefix


def build_center_slices() -> onnx.ModelProto:
    """Best official-data variant: read fixed center marker cells directly."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = [
        _init(np.zeros((1, 4, 3, 1), dtype=np.bool_), "zero_col4"),
        _init(np.zeros((1, 4, 1, 11), dtype=np.bool_), "zero_row4"),
        _init(np.asarray([1, 4, 3, 3], dtype=np.int64), "seg_shape"),
    ]
    gray = _add_common_gray(nodes, inits)

    for band, row in (("top", 1), ("bot", 5)):
        for idx, col in enumerate((1, 5, 9)):
            inits.extend(
                [
                    _init(np.asarray([1, row, col], dtype=np.int64), f"{band}{idx}_st"),
                    _init(np.asarray([5, row + 1, col + 1], dtype=np.int64), f"{band}{idx}_en"),
                ]
            )
            nodes.extend(
                [
                    helper.make_node("Slice", [IN_NAME, f"{band}{idx}_st", f"{band}{idx}_en", "axes3"], [f"{band}{idx}_f"]),
                    helper.make_node("Cast", [f"{band}{idx}_f"], [f"{band}{idx}_b"], to=TensorProto.BOOL),
                    helper.make_node("Expand", [f"{band}{idx}_b", "seg_shape"], [f"{band}{idx}_seg"]),
                ]
            )

    top = _segment_concat(nodes, ["top0_seg", "top1_seg", "top2_seg"], "top4")
    bottom = _segment_concat(nodes, ["bot0_seg", "bot1_seg", "bot2_seg"], "bot4")
    color4 = _two_band_concat(nodes, top, bottom, "color4")
    return _finalize(nodes, inits, color4, gray, "task127_center_slices")


def build_center_slices_gated_gray() -> onnx.ModelProto:
    """Read fixed center markers and synthesize gray separators from a height gate."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = [
        _init(np.zeros((1, 4, 3, 1), dtype=np.bool_), "zero_col4"),
        _init(np.zeros((1, 4, 1, 11), dtype=np.bool_), "zero_row4"),
        _init(np.asarray([1, 4, 3, 3], dtype=np.int64), "seg_shape"),
        _init(np.asarray([1, 2, 3], dtype=np.int64), "axes3"),
    ]

    for band, row in (("top", 1), ("bot", 5)):
        for idx, col in enumerate((1, 5, 9)):
            inits.extend(
                [
                    _init(np.asarray([1, row, col], dtype=np.int64), f"{band}{idx}_st"),
                    _init(np.asarray([5, row + 1, col + 1], dtype=np.int64), f"{band}{idx}_en"),
                ]
            )
            nodes.extend(
                [
                    helper.make_node("Slice", [IN_NAME, f"{band}{idx}_st", f"{band}{idx}_en", "axes3"], [f"{band}{idx}_f"]),
                    helper.make_node("Cast", [f"{band}{idx}_f"], [f"{band}{idx}_b"], to=TensorProto.BOOL),
                    helper.make_node("Expand", [f"{band}{idx}_b", "seg_shape"], [f"{band}{idx}_seg"]),
                ]
            )

    top = _segment_concat(nodes, ["top0_seg", "top1_seg", "top2_seg"], "top4")
    bottom = _segment_concat(nodes, ["bot0_seg", "bot1_seg", "bot2_seg"], "bot4")
    color4 = _two_band_concat(nodes, top, bottom, "color4")
    gray = _add_gated_gray_from_marker(nodes, inits, "bot0_f")
    return _finalize(nodes, inits, color4, gray, "task127_center_slices_gated_gray")


def build_center_gather() -> onnx.ModelProto:
    """Read each marker row once, gather marker columns, then split segments."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = [
        _init(np.zeros((1, 4, 3, 1), dtype=np.bool_), "zero_col4"),
        _init(np.zeros((1, 4, 1, 11), dtype=np.bool_), "zero_row4"),
        _init(np.asarray([1, 4, 3, 3], dtype=np.int64), "seg_shape"),
        _init(np.asarray([1, 5, 9], dtype=np.int64), "marker_cols"),
    ]
    gray = _add_common_gray(nodes, inits)

    for band, row in (("top", 1), ("bot", 5)):
        inits.extend(
            [
                _init(np.asarray([1, row, 0], dtype=np.int64), f"{band}_st"),
                _init(np.asarray([5, row + 1, 10], dtype=np.int64), f"{band}_en"),
            ]
        )
        nodes.extend(
            [
                helper.make_node("Slice", [IN_NAME, f"{band}_st", f"{band}_en", "axes3"], [f"{band}_row"]),
                helper.make_node("Gather", [f"{band}_row", "marker_cols"], [f"{band}_marks_f"], axis=3),
                helper.make_node("Cast", [f"{band}_marks_f"], [f"{band}_marks"], to=TensorProto.BOOL),
                helper.make_node("Split", [f"{band}_marks"], [f"{band}0", f"{band}1", f"{band}2"], axis=3, split=[1, 1, 1]),
            ]
        )
        for idx in range(3):
            nodes.append(helper.make_node("Expand", [f"{band}{idx}", "seg_shape"], [f"{band}{idx}_seg"]))

    top = _segment_concat(nodes, ["top0_seg", "top1_seg", "top2_seg"], "top4")
    bottom = _segment_concat(nodes, ["bot0_seg", "bot1_seg", "bot2_seg"], "bot4")
    color4 = _two_band_concat(nodes, top, bottom, "color4")
    return _finalize(nodes, inits, color4, gray, "task127_center_gather")


def build_segment_reduce() -> onnx.ModelProto:
    """More general variant: reduce each width-3 segment over columns."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = [
        _init(np.zeros((1, 4, 3, 1), dtype=np.bool_), "zero_col4"),
        _init(np.zeros((1, 4, 1, 11), dtype=np.bool_), "zero_row4"),
        _init(np.asarray([1, 4, 3, 3], dtype=np.int64), "seg_shape"),
    ]
    gray = _add_common_gray(nodes, inits)

    for band, row in (("top", 1), ("bot", 5)):
        for idx, start_col in enumerate((0, 4, 8)):
            inits.extend(
                [
                    _init(np.asarray([1, row, start_col], dtype=np.int64), f"{band}{idx}_st"),
                    _init(np.asarray([5, row + 1, start_col + 3], dtype=np.int64), f"{band}{idx}_en"),
                ]
            )
            nodes.extend(
                [
                    helper.make_node("Slice", [IN_NAME, f"{band}{idx}_st", f"{band}{idx}_en", "axes3"], [f"{band}{idx}_slice"]),
                    helper.make_node("ReduceMax", [f"{band}{idx}_slice"], [f"{band}{idx}_mark_f"], axes=[3], keepdims=1),
                    helper.make_node("Cast", [f"{band}{idx}_mark_f"], [f"{band}{idx}_mark"], to=TensorProto.BOOL),
                    helper.make_node("Expand", [f"{band}{idx}_mark", "seg_shape"], [f"{band}{idx}_seg"]),
                ]
            )

    top = _segment_concat(nodes, ["top0_seg", "top1_seg", "top2_seg"], "top4")
    bottom = _segment_concat(nodes, ["bot0_seg", "bot1_seg", "bot2_seg"], "bot4")
    color4 = _two_band_concat(nodes, top, bottom, "color4")
    return _finalize(nodes, inits, color4, gray, "task127_segment_reduce")


def one_hot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(tuple(SHAPE), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            out[0, int(value), r, c] = 1.0
    return out


def expected_grid(grid: list[list[int]]) -> list[list[int]]:
    height = len(grid)
    out = [[0 for _ in range(11)] for _ in range(height)]
    for base in (0, 4):
        if base + 2 >= height:
            continue
        for seg_idx, marker_col in enumerate((1, 5, 9)):
            fill = grid[base + 1][marker_col] + 5
            for rr in range(base, base + 3):
                for cc in range(seg_idx * 4, seg_idx * 4 + 3):
                    out[rr][cc] = fill
                out[rr][3] = 5
                out[rr][7] = 5
    if height == 7:
        out[3] = [5] * 11
    return out


def expected_grid_random_markers(grid: list[list[int]]) -> list[list[int]]:
    height = len(grid)
    out = [[0 for _ in range(11)] for _ in range(height)]
    for base in (0, 4):
        if base + 2 >= height:
            continue
        for seg_idx, cols in enumerate((range(0, 3), range(4, 7), range(8, 11))):
            marker = max(grid[base + 1][cc] for cc in cols)
            fill = marker + 5
            for rr in range(base, base + 3):
                for cc in cols:
                    out[rr][cc] = fill
                out[rr][3] = 5
                out[rr][7] = 5
    if height == 7:
        out[3] = [5] * 11
    return out


def verify_model(model: onnx.ModelProto) -> None:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(data[split]):
            x = convert_to_numpy(example, "input")
            y = convert_to_numpy(example, "output")
            if x is None or y is None:
                continue
            got = session.run([OUT_NAME], {IN_NAME: x})[0]
            if not np.array_equal(got > 0.0, y > 0.0):
                raise AssertionError(f"{split}#{idx} failed")


def verify_random_marker_model(model: onnx.ModelProto, samples: int = 64) -> None:
    rng = np.random.default_rng(127)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    for _ in range(samples):
        height = int(rng.choice([3, 7]))
        grid = [[0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0] for _ in range(height)]
        if height == 7:
            grid[3] = [5] * 11
        for base in (0, 4):
            if base + 2 >= height:
                continue
            for cols in ((0, 1, 2), (4, 5, 6), (8, 9, 10)):
                col = int(rng.choice(cols))
                grid[base + 1][col] = int(rng.integers(1, 5))
        got = session.run([OUT_NAME], {IN_NAME: one_hot(grid)})[0]
        want = one_hot(expected_grid_random_markers(grid))
        if not np.array_equal(got > 0.0, want > 0.0):
            raise AssertionError("random marker verification failed")


def score_as_task127(path: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="task127_score_") as tmp:
        tmp_path = Path(tmp) / "task127.onnx"
        shutil.copyfile(path, tmp_path)
        return score_file(tmp_path)


def main() -> None:
    candidates = [
        Candidate("center_slices_gated_gray", build_center_slices_gated_gray),
        Candidate("center_slices", build_center_slices),
        Candidate("center_gather", build_center_gather),
        Candidate("segment_reduce", build_segment_reduce, random_ok=True),
    ]
    VARIANT_DIR.mkdir(exist_ok=True)
    rows: list[tuple[int, float, str, Path, onnx.ModelProto, int, int]] = []
    for candidate in candidates:
        model = candidate.build()
        verify_model(model)
        if candidate.random_ok:
            verify_random_marker_model(model)
        path = VARIANT_DIR / f"task127_{candidate.name}.onnx"
        onnx.save(model, path)
        result = score_as_task127(path)
        if not result["valid"]:
            raise RuntimeError(f"{candidate.name} invalid: {result['error']}")
        cost = int(result["cost"])
        score = float(result["score"])
        memory = int(result["memory"])
        params = int(result["params"])
        rows.append((cost, score, candidate.name, path, model, memory, params))
        print(f"{candidate.name}: memory={memory} params={params} cost={cost} score={score:.6f}")

    rows.sort(key=lambda row: row[0])
    cost, score, name, path, model, memory, params = rows[0]
    onnx.save(model, BEST_PATH)
    root_copy = ROOT / "task127.onnx"
    shutil.copyfile(BEST_PATH, root_copy)
    print(f"best={name} memory={memory} params={params} cost={cost} score={score:.6f}")
    print(f"wrote {BEST_PATH}")
    print(f"wrote {root_copy}")


if __name__ == "__main__":
    main()
