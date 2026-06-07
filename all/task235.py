"""ONNX for ARC task235: decode three 4x4 gray/black glyphs into color rows.

Task rule: the 4x14 input is three 4-wide glyphs separated by black columns.
Each glyph maps to one output color, and that color is repeated across a full
width-3 row.  The glyphs are:
all gray -> red (2), middle two rows 0110 -> green (3), bottom two rows 1001
-> olive/yellow-brown (4), and middle two rows 1001 -> cyan/teal (8).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

TASK_ID = "task235"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task235.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr: Any, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Any, name: str) -> str:
    return _init(inits, np.asarray(vals, dtype=np.int64), name)


def solve_grid(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    colors: list[int] = []
    for start in (0, 5, 10):
        p = g[1, start] == 5
        q = g[1, start + 1] == 5
        r = g[2, start + 1] == 5
        if (not p) and q:
            color = 3
        elif p and (not q):
            color = 8
        elif p and q and (not r):
            color = 4
        else:
            color = 2
        colors.append(color)
    return np.asarray([[color] * 3 for color in colors], dtype=np.int64)


def build_model() -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, [1, 2, 3], "axes")
    steps = _i64(inits, [1, 1, 5], "steps")
    zero_ch = _init(inits, np.zeros((1, 1, 3, 3), dtype=bool), "zero_ch")

    samples: dict[str, str] = {}
    for label, row, start_col in (("p", 1, 0), ("q", 1, 1), ("r", 2, 1)):
        st = _i64(inits, [5, row, start_col], f"{label}_st")
        en = _i64(inits, [6, row + 1, start_col + 11], f"{label}_en")
        sliced = f"{label}_cells"
        gray = f"{label}_gray"
        nodes.extend(
            [
                helper.make_node("Slice", [IN_NAME, st, en, axes, steps], [sliced]),
                helper.make_node("Cast", [sliced], [gray], to=TensorProto.BOOL),
            ]
        )
        samples[label] = gray

    p = samples["p"]
    q = samples["q"]
    r = samples["r"]
    not_p = "not_p"
    not_q = "not_q"
    not_r = "not_r"
    pq = "pq"
    is2_w = "is2_w"
    is3_w = "is3_w"
    is4_w = "is4_w"
    is8_w = "is8_w"
    nodes.extend(
        [
            helper.make_node("Not", [p], [not_p]),
            helper.make_node("Not", [q], [not_q]),
            helper.make_node("Not", [r], [not_r]),
            helper.make_node("And", [p, q], [pq]),
            helper.make_node("And", [pq, r], [is2_w]),
            helper.make_node("And", [not_p, q], [is3_w]),
            helper.make_node("And", [pq, not_r], [is4_w]),
            helper.make_node("And", [p, not_q], [is8_w]),
        ]
    )

    by_color = {2: is2_w, 3: is3_w, 4: is4_w, 8: is8_w}
    channel_names: dict[int, str] = {}
    for color in (2, 3, 4, 8):
        rows = f"ch{color}_rows"
        tiled = f"ch{color}"
        nodes.extend(
            [
                helper.make_node("Transpose", [by_color[color]], [rows], perm=[0, 1, 3, 2]),
                helper.make_node("Concat", [rows, rows, rows], [tiled], axis=3),
            ]
        )
        channel_names[color] = tiled

    nodes.extend(
        [
            helper.make_node(
                "Concat",
                [channel_names[2], channel_names[3], channel_names[4], zero_ch, zero_ch, zero_ch, channel_names[8]],
                ["onehot_bool"],
                axis=1,
            ),
            helper.make_node("Cast", ["onehot_bool"], ["onehot3"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["onehot3"], [OUT_NAME], pads=[0, 2, 0, 0, 0, 1, H - 3, W - 3]),
        ]
    )

    graph = helper.make_graph(nodes, "task235", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _verify_reference() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for i, example in enumerate(data[split]):
            got = solve_grid(example["input"])
            expected = np.asarray(example["output"], dtype=np.int64)
            if not np.array_equal(got, expected):
                raise AssertionError(f"reference failed {split}[{i}]: {got.tolist()} != {expected.tolist()}")


def _verify_onnx(path: Path) -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for split in ("train", "test", "arc-gen"):
        for i, example in enumerate(data[split]):
            inp = convert_to_numpy(example, "input")
            expected = convert_to_numpy(example, "output") > 0
            assert inp is not None and expected is not None
            got = session.run([OUT_NAME], {IN_NAME: inp})[0] > 0
            if not np.array_equal(got, expected):
                raise AssertionError(f"ONNX failed {split}[{i}]")


def main() -> None:
    _verify_reference()
    model = build_model()
    onnx.save(model, BEST_PATH)
    _verify_onnx(BEST_PATH)
    result = score_file(BEST_PATH)
    print(f"wrote {BEST_PATH}")
    print(f"valid={result['valid']} memory={result['memory']} params={result['params']} cost={result['cost']} score={result['score']}")


if __name__ == "__main__":
    main()
