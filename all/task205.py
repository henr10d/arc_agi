"""Crop a noisy hidden rectangle and expand its marker cells into crosses.

Task rule: the input contains one axis-aligned rectangular panel, height and
width 6 through 10, whose dominant color fills all but one to three marker
cells. The output is that panel cropped to its rectangle size. Every marker
cell, which is the same non-panel color within an example, expands to a full
horizontal and vertical line of the marker color across the crop; all remaining
cells keep the panel color. Background noise outside the panel is ignored.

ONNX approach: enumerate the possible panel sizes, find high-purity color
windows with grouped convolutions, compute row/column marker hits for the
selected window, then pad the selected crop-shaped one-hot result to 30x30.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
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

TASK_ID = "task205"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"

BATCH = 1
COLORS = 10
HEIGHT = WIDTH = 30
SHAPE = [BATCH, COLORS, HEIGHT, WIDTH]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10
SIZES = tuple(sorted(((h, w) for h in range(6, 11) for w in range(6, 11)), key=lambda item: item[0] * item[1], reverse=True))


def _init(inits: list[onnx.TensorProto], arr: np.ndarray | Iterable[int | float], name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], vals: Iterable[int], name: str) -> str:
    return _init(inits, np.asarray(list(vals), dtype=np.int64), name)


def _f32(inits: list[onnx.TensorProto], vals: Iterable[float], name: str) -> str:
    return _init(inits, np.asarray(list(vals), dtype=np.float32), name)


def _count_kernel(h: int, w: int) -> np.ndarray:
    return np.ones((COLORS, 1, h, w), dtype=np.float32)


def _row_kernel(h: int, w: int) -> np.ndarray:
    k = np.zeros((COLORS * h, 1, h, w), dtype=np.float32)
    for color in range(COLORS):
        for row in range(h):
            k[color * h + row, 0, row, :] = 1.0
    return k


def _col_kernel(h: int, w: int) -> np.ndarray:
    k = np.zeros((COLORS * w, 1, h, w), dtype=np.float32)
    for color in range(COLORS):
        for col in range(w):
            k[color * w + col, 0, :, col] = 1.0
    return k


def solve(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Reference solver: choose the largest exact cross-reconstructing panel."""
    arr = np.asarray(grid, dtype=np.int64)
    best: tuple[int, int, int, int, int, np.ndarray] | None = None
    for h, w in SIZES:
        if h > arr.shape[0] or w > arr.shape[1]:
            continue
        for r0 in range(arr.shape[0] - h + 1):
            for c0 in range(arr.shape[1] - w + 1):
                sub = arr[r0 : r0 + h, c0 : c0 + w]
                color, count = Counter(sub.ravel().tolist()).most_common(1)[0]
                if not (h * w - 3 <= count <= h * w - 1):
                    continue
                out = np.full((h, w), int(color), dtype=np.int64)
                for r, c in np.argwhere(sub != color):
                    out[r, :] = sub[r, c]
                    out[:, c] = sub[r, c]
                score = (h * w, count, -r0, -c0)
                if best is None or score > best[:4]:
                    best = (*score, out)
    if best is None:
        raise ValueError("no high-purity rectangle found")
    return best[4]


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    zero = _f32(inits, [0.0], "zero")
    eps = _f32(inits, [0.5], "eps")
    axes_color = [1]
    axes_pos = [2, 3]
    axes_marker_pos = [3, 4]

    padded_sizes: list[str] = []
    larger_active: str | None = None
    for h, w in SIZES:
        area = h * w
        oh = HEIGHT - h + 1
        ow = WIDTH - w + 1
        tag = f"h{h}w{w}"

        _init(inits, _count_kernel(h, w), f"{tag}_count_k")
        _init(inits, _row_kernel(h, w), f"{tag}_row_k")
        _init(inits, _col_kernel(h, w), f"{tag}_col_k")
        _f32(inits, [float(area - 3) - 0.5], f"{tag}_lo")
        _f32(inits, [float(area) - 0.5], f"{tag}_hi")
        _init(inits, np.ones((1, 1, h, w), dtype=np.bool_), f"{tag}_ones")
        _i64(inits, [1, COLORS, h, oh, ow], f"{tag}_row_shape")
        _i64(inits, [1, COLORS, w, oh, ow], f"{tag}_col_shape")

        nodes.extend(
            [
                helper.make_node("Conv", [IN_NAME, f"{tag}_count_k"], [f"{tag}_counts"], group=COLORS),
                helper.make_node("Greater", [f"{tag}_counts", f"{tag}_lo"], [f"{tag}_enough"]),
                helper.make_node("Less", [f"{tag}_counts", f"{tag}_hi"], [f"{tag}_not_solid"]),
                helper.make_node("And", [f"{tag}_enough", f"{tag}_not_solid"], [f"{tag}_panel_raw"]),
                helper.make_node("Cast", [f"{tag}_panel_raw"], [f"{tag}_panel_raw_f"], to=TensorProto.FLOAT),
                helper.make_node("ReduceSum", [f"{tag}_panel_raw_f"], [f"{tag}_panel_raw_sum"], axes=[1, 2, 3], keepdims=1),
                helper.make_node("Greater", [f"{tag}_panel_raw_sum", zero], [f"{tag}_panel_raw_any"]),
            ]
        )
        if larger_active is None:
            nodes.append(helper.make_node("Identity", [f"{tag}_panel_raw"], [f"{tag}_panel"]))
            larger_active = f"{tag}_panel_raw_any"
        else:
            nodes.extend(
                [
                    helper.make_node("Not", [larger_active], [f"{tag}_no_larger"]),
                    helper.make_node("And", [f"{tag}_panel_raw", f"{tag}_no_larger"], [f"{tag}_panel"]),
                    helper.make_node("Or", [larger_active, f"{tag}_panel_raw_any"], [f"{tag}_larger_next"]),
                ]
            )
            larger_active = f"{tag}_larger_next"

        nodes.extend(
            [
                helper.make_node("Cast", [f"{tag}_panel"], [f"{tag}_panel_f"], to=TensorProto.FLOAT),
                helper.make_node("ReduceSum", [f"{tag}_panel_f"], [f"{tag}_panel_any"], axes=axes_color, keepdims=1),
                helper.make_node("Conv", [IN_NAME, f"{tag}_row_k"], [f"{tag}_row_sum"], group=COLORS),
                helper.make_node("Conv", [IN_NAME, f"{tag}_col_k"], [f"{tag}_col_sum"], group=COLORS),
                helper.make_node("Greater", [f"{tag}_row_sum", zero], [f"{tag}_row_hit_flat"]),
                helper.make_node("Greater", [f"{tag}_col_sum", zero], [f"{tag}_col_hit_flat"]),
                helper.make_node("Reshape", [f"{tag}_row_hit_flat", f"{tag}_row_shape"], [f"{tag}_row_hit"]),
                helper.make_node("Reshape", [f"{tag}_col_hit_flat", f"{tag}_col_shape"], [f"{tag}_col_hit"]),
                helper.make_node("ReduceSum", [f"{tag}_panel_f"], [f"{tag}_panel_scalar"], axes=axes_pos, keepdims=1),
                helper.make_node("Greater", [f"{tag}_panel_scalar", eps], [f"{tag}_panel_scalar_b"]),
                helper.make_node("And", [f"{tag}_panel_scalar_b", f"{tag}_ones"], [f"{tag}_base"]),
            ]
        )

        nodes.extend(
            [
                helper.make_node("Sub", [f"{tag}_panel_any", f"{tag}_panel_f"], [f"{tag}_panel_not_f"]),
                helper.make_node("Greater", [f"{tag}_panel_not_f", eps], [f"{tag}_panel_not"]),
                helper.make_node("Unsqueeze", [f"{tag}_panel_not"], [f"{tag}_panel_not_u"], axes=[2]),
                helper.make_node("And", [f"{tag}_row_hit", f"{tag}_panel_not_u"], [f"{tag}_row_sel"]),
                helper.make_node("And", [f"{tag}_col_hit", f"{tag}_panel_not_u"], [f"{tag}_col_sel"]),
                helper.make_node("Cast", [f"{tag}_row_sel"], [f"{tag}_row_sel_f"], to=TensorProto.FLOAT),
                helper.make_node("Cast", [f"{tag}_col_sel"], [f"{tag}_col_sel_f"], to=TensorProto.FLOAT),
                helper.make_node("ReduceSum", [f"{tag}_row_sel_f"], [f"{tag}_row_mark"], axes=axes_marker_pos, keepdims=0),
                helper.make_node("ReduceSum", [f"{tag}_col_sel_f"], [f"{tag}_col_mark"], axes=axes_marker_pos, keepdims=0),
                helper.make_node("Greater", [f"{tag}_row_mark", zero], [f"{tag}_row_line"]),
                helper.make_node("Greater", [f"{tag}_col_mark", zero], [f"{tag}_col_line"]),
                helper.make_node("Unsqueeze", [f"{tag}_row_line"], [f"{tag}_row_u"], axes=[3]),
                helper.make_node("Unsqueeze", [f"{tag}_col_line"], [f"{tag}_col_u"], axes=[2]),
                helper.make_node("Or", [f"{tag}_row_u", f"{tag}_col_u"], [f"{tag}_lines"]),
                helper.make_node("Cast", [f"{tag}_lines"], [f"{tag}_lines_f"], to=TensorProto.FLOAT),
                helper.make_node("ReduceSum", [f"{tag}_lines_f"], [f"{tag}_line_any_f"], axes=axes_color, keepdims=1),
                helper.make_node("Greater", [f"{tag}_line_any_f", zero], [f"{tag}_line_any"]),
                helper.make_node("Not", [f"{tag}_line_any"], [f"{tag}_not_line"]),
                helper.make_node("And", [f"{tag}_base", f"{tag}_not_line"], [f"{tag}_base_keep"]),
                helper.make_node("Or", [f"{tag}_base_keep", f"{tag}_lines"], [f"{tag}_out_bool"]),
                helper.make_node("Cast", [f"{tag}_out_bool"], [f"{tag}_out_f"], to=TensorProto.FLOAT),
            ]
        )
        pad_h = 10 - h
        pad_w = 10 - w
        nodes.append(
            helper.make_node(
                "Pad",
                [f"{tag}_out_f"],
                [f"{tag}_padded"],
                pads=[0, 0, 0, 0, 0, 0, pad_h, pad_w],
            )
        )
        padded_sizes.append(f"{tag}_padded")

    nodes.append(helper.make_node("Sum", padded_sizes, ["sum10"]))
    nodes.append(
        helper.make_node(
            "Pad",
            ["sum10"],
            [OUT_NAME],
            pads=[0, 0, 0, 0, 0, 0, HEIGHT - 10, WIDTH - 10],
        )
    )

    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="ng_task205",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _run_model(path: Path, arr: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: arr})[0]


def validate_reference() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad: list[tuple[str, int]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            pred = solve(ex["input"])
            if not np.array_equal(pred, np.asarray(ex["output"], dtype=np.int64)):
                bad.append((split, idx))
    if bad:
        raise AssertionError(f"reference rule failed: {bad[:10]}")


def validate_model(path: Path) -> dict[str, int]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    counts: dict[str, int] = {}
    for split in ("train", "test", "arc-gen"):
        ok = 0
        total = 0
        for idx, ex in enumerate(data.get(split, [])):
            inp = convert_to_numpy(ex, "input")
            exp = convert_to_numpy(ex, "output")
            if inp is None or exp is None:
                continue
            got = _run_model(path, inp)
            if np.array_equal(got > 0.0, exp > 0.0):
                ok += 1
            else:
                raise AssertionError(f"model failed {split}[{idx}]")
            total += 1
        counts[f"{split}_ok"] = ok
        counts[f"{split}_total"] = total
    return counts


def main() -> None:
    validate_reference()
    model = build_model()
    onnx.save(model, BEST_PATH)
    counts = validate_model(BEST_PATH)
    result = score_file(BEST_PATH)
    if not result["valid"]:
        raise RuntimeError(result["error"])
    print(f"wrote {BEST_PATH}")
    print(counts)
    print(f"memory={result['memory']} params={result['params']} cost={result['cost']}")
    print(f"score={float(result['score']):.6f}")


if __name__ == "__main__":
    main()
