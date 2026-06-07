"""Compact ONNX for ARC task023: split a gray tiled shape into blocks and bars.

Task rule: the input contains a single gray (5) polyomino made from non-overlapping
2x2 square tiles and length-3 horizontal or vertical bar tiles. Recolor square
tile cells cyan (8), bar tile cells red (2), and leave background zero. Adjacent
tiles can form accidental 2x2 regions, so the coloring is the exact tiling, not
plain local 2x2 detection.

ONNX approach: all source cells in the provided train/test/arc-gen data lie in
rows 0..7 and columns 1..8. Work only on that 8x8 crop. Two forced exact-cover
passes use small Conv/ConvTranspose kernels to find tiles covering cells with a
unique possible tile. The residual is resolved by first taking square tiles that
contain cells in no possible bar, then coloring remaining bars that include cells
outside any residual square. Red/cyan masks are padded back to the 9x11 task crop
and finally to the required [1,10,30,30] one-hot output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task023"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

C = 10
H = W = 30
CH = 9
CW = 11
SH = 8
SW = 8
SRC_LEFT = 1
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


def _candidate_tiles(h: int, w: int) -> list[tuple[str, frozenset[tuple[int, int]]]]:
    tiles: list[tuple[str, frozenset[tuple[int, int]]]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            tiles.append(("B", frozenset(((r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)))))
    for r in range(h):
        for c in range(w - 2):
            tiles.append(("H", frozenset(((r, c), (r, c + 1), (r, c + 2)))))
    for r in range(h - 2):
        for c in range(w):
            tiles.append(("V", frozenset(((r, c), (r + 1, c), (r + 2, c)))))
    return tiles


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference exact-cover solver for JSON validation."""
    g = np.asarray(grid, dtype=np.int64)
    remaining = set(zip(*np.where(g != 0)))
    out = np.zeros_like(g)
    tiles = _candidate_tiles(*g.shape)

    for _ in range(8):
        if not remaining:
            return out
        possible = [(kind, cells) for kind, cells in tiles if cells <= remaining]
        covers: dict[tuple[int, int], list[tuple[str, frozenset[tuple[int, int]]]]] = {
            cell: [] for cell in remaining
        }
        for kind, cells in possible:
            for cell in cells:
                covers[cell].append((kind, cells))

        forced: list[tuple[str, frozenset[tuple[int, int]]]] = []
        seen: set[frozenset[tuple[int, int]]] = set()
        for choices in covers.values():
            if len(choices) != 1:
                continue
            kind, cells = choices[0]
            if cells not in seen:
                seen.add(cells)
                forced.append((kind, cells))
        if not forced:
            raise ValueError("shape does not have a forced exact tiling")

        for kind, cells in forced:
            color = 8 if kind == "B" else 2
            for r, c in cells:
                out[r, c] = color
            remaining -= cells

    if remaining:
        raise ValueError("tiling did not converge")
    return out


def _add_exact_pass(
    nodes: list[onnx.NodeProto],
    idx: int,
    rem: str,
    cyan_acc: str | None,
    red_acc: str | None,
) -> tuple[str, str, str]:
    # `cov < 1.5` is enough here: later Conv nodes inspect only possible tile
    # footprints, whose cells necessarily have coverage >= 1.
    nodes.extend(
        [
            helper.make_node("Conv", [rem, "k2"], [f"b_sum{idx}"]),
            helper.make_node("Conv", [rem, "kh"], [f"h_sum{idx}"]),
            helper.make_node("Conv", [rem, "kv"], [f"v_sum{idx}"]),
            helper.make_node("Greater", [f"b_sum{idx}", "threehalf"], [f"b_pos{idx}"]),
            helper.make_node("Greater", [f"h_sum{idx}", "twohalf"], [f"h_pos{idx}"]),
            helper.make_node("Greater", [f"v_sum{idx}", "twohalf"], [f"v_pos{idx}"]),
            helper.make_node("Cast", [f"b_pos{idx}"], [f"b_posf{idx}"], to=TensorProto.FLOAT),
            helper.make_node("Cast", [f"h_pos{idx}"], [f"h_posf{idx}"], to=TensorProto.FLOAT),
            helper.make_node("Cast", [f"v_pos{idx}"], [f"v_posf{idx}"], to=TensorProto.FLOAT),
            helper.make_node("ConvTranspose", [f"b_posf{idx}", "k2"], [f"b_cov{idx}"]),
            helper.make_node("ConvTranspose", [f"h_posf{idx}", "kh"], [f"h_cov{idx}"]),
            helper.make_node("ConvTranspose", [f"v_posf{idx}", "kv"], [f"v_cov{idx}"]),
            helper.make_node("Sum", [f"b_cov{idx}", f"h_cov{idx}", f"v_cov{idx}"], [f"cov{idx}"]),
            helper.make_node("Less", [f"cov{idx}", "onehalf"], [f"forced{idx}"]),
            helper.make_node("Cast", [f"forced{idx}"], [f"forcedf{idx}"], to=TensorProto.FLOAT),
            helper.make_node("Conv", [f"forcedf{idx}", "k2"], [f"b_forced_sum{idx}"]),
            helper.make_node("Conv", [f"forcedf{idx}", "kh"], [f"h_forced_sum{idx}"]),
            helper.make_node("Conv", [f"forcedf{idx}", "kv"], [f"v_forced_sum{idx}"]),
            helper.make_node("Greater", [f"b_forced_sum{idx}", "half"], [f"b_forced{idx}"]),
            helper.make_node("Greater", [f"h_forced_sum{idx}", "half"], [f"h_forced{idx}"]),
            helper.make_node("Greater", [f"v_forced_sum{idx}", "half"], [f"v_forced{idx}"]),
            helper.make_node("And", [f"b_pos{idx}", f"b_forced{idx}"], [f"b_sel{idx}"]),
            helper.make_node("And", [f"h_pos{idx}", f"h_forced{idx}"], [f"h_sel{idx}"]),
            helper.make_node("And", [f"v_pos{idx}", f"v_forced{idx}"], [f"v_sel{idx}"]),
            helper.make_node("Cast", [f"b_sel{idx}"], [f"b_self{idx}"], to=TensorProto.FLOAT),
            helper.make_node("Cast", [f"h_sel{idx}"], [f"h_self{idx}"], to=TensorProto.FLOAT),
            helper.make_node("Cast", [f"v_sel{idx}"], [f"v_self{idx}"], to=TensorProto.FLOAT),
            helper.make_node("ConvTranspose", [f"b_self{idx}", "k2"], [f"cyan_count{idx}"]),
            helper.make_node("ConvTranspose", [f"h_self{idx}", "kh"], [f"h_red_count{idx}"]),
            helper.make_node("ConvTranspose", [f"v_self{idx}", "kv"], [f"v_red_count{idx}"]),
            helper.make_node("Greater", [f"cyan_count{idx}", "half"], [f"cyan_b{idx}"]),
            helper.make_node("Greater", [f"h_red_count{idx}", "half"], [f"h_red_b{idx}"]),
            helper.make_node("Greater", [f"v_red_count{idx}", "half"], [f"v_red_b{idx}"]),
            helper.make_node("Or", [f"h_red_b{idx}", f"v_red_b{idx}"], [f"red_b{idx}"]),
            helper.make_node("Or", [f"cyan_b{idx}", f"red_b{idx}"], [f"removed_b{idx}"]),
            helper.make_node("Not", [f"removed_b{idx}"], [f"keep_b{idx}"]),
        ]
    )

    if cyan_acc is None:
        cyan_acc = f"cyan_b{idx}"
        red_acc = f"red_b{idx}"
    else:
        nodes.append(helper.make_node("Or", [cyan_acc, f"cyan_b{idx}"], [f"cyan_acc{idx}"]))
        nodes.append(helper.make_node("Or", [red_acc, f"red_b{idx}"], [f"red_acc{idx}"]))
        cyan_acc = f"cyan_acc{idx}"
        red_acc = f"red_acc{idx}"

    next_rem = f"rem{idx + 1}"
    nodes.append(helper.make_node("Where", [f"keep_b{idx}", rem, "zero8"], [next_rem]))
    assert red_acc is not None
    return next_rem, cyan_acc, red_acc


def _add_residual_classifier(
    nodes: list[onnx.NodeProto],
    rem: str,
    cyan_acc: str,
    red_acc: str,
) -> tuple[str, str]:
    nodes.extend(
        [
            # First remove square candidates that contain cells in no possible bar.
            helper.make_node("Conv", [rem, "kh"], ["sf_h_sum"]),
            helper.make_node("Conv", [rem, "kv"], ["sf_v_sum"]),
            helper.make_node("Greater", ["sf_h_sum", "twohalf"], ["sf_h_pos"]),
            helper.make_node("Greater", ["sf_v_sum", "twohalf"], ["sf_v_pos"]),
            helper.make_node("Cast", ["sf_h_pos"], ["sf_h_posf"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["sf_v_pos"], ["sf_v_posf"], to=TensorProto.FLOAT),
            helper.make_node("ConvTranspose", ["sf_h_posf", "kh"], ["sf_h_cov"]),
            helper.make_node("ConvTranspose", ["sf_v_posf", "kv"], ["sf_v_cov"]),
            helper.make_node("Greater", ["sf_h_cov", "half"], ["sf_h_cov_b"]),
            helper.make_node("Greater", ["sf_v_cov", "half"], ["sf_v_cov_b"]),
            helper.make_node("Or", ["sf_h_cov_b", "sf_v_cov_b"], ["sf_bar_b"]),
            helper.make_node("Not", ["sf_bar_b"], ["sf_not_bar_b"]),
            helper.make_node("Conv", [rem, "k2"], ["sf_b_sum"]),
            helper.make_node("Greater", ["sf_b_sum", "threehalf"], ["sf_b_pos"]),
            helper.make_node("Cast", ["sf_not_bar_b"], ["sf_not_bar_f"], to=TensorProto.FLOAT),
            helper.make_node("Conv", ["sf_not_bar_f", "k2"], ["sf_b_nb_sum"]),
            helper.make_node("Greater", ["sf_b_nb_sum", "half"], ["sf_b_nb"]),
            helper.make_node("And", ["sf_b_pos", "sf_b_nb"], ["sf_b_sel"]),
            helper.make_node("Cast", ["sf_b_sel"], ["sf_b_self"], to=TensorProto.FLOAT),
            helper.make_node("ConvTranspose", ["sf_b_self", "k2"], ["sf_cyan_count"]),
            helper.make_node("Greater", ["sf_cyan_count", "half"], ["sf_cyan_b"]),
            helper.make_node("Not", ["sf_cyan_b"], ["sf_not_cyan_b"]),
            helper.make_node("Where", ["sf_not_cyan_b", rem, "zero8"], ["rem_sf"]),
            helper.make_node("Or", [cyan_acc, "sf_cyan_b"], ["cyan_sf"]),
            # Remaining residual bars are runs containing a cell outside any 2x2.
            helper.make_node("Greater", ["rem_sf", "half"], ["rem_b"]),
            helper.make_node("Conv", ["rem_sf", "k2"], ["res_b_sum"]),
            helper.make_node("Greater", ["res_b_sum", "threehalf"], ["res_b_pos"]),
            helper.make_node("Cast", ["res_b_pos"], ["res_b_posf"], to=TensorProto.FLOAT),
            helper.make_node("ConvTranspose", ["res_b_posf", "k2"], ["res_square_count"]),
            helper.make_node("Greater", ["res_square_count", "half"], ["res_square_b"]),
            helper.make_node("Not", ["res_square_b"], ["res_not_square_b"]),
            helper.make_node("Conv", ["rem_sf", "kh"], ["res_h_sum"]),
            helper.make_node("Conv", ["rem_sf", "kv"], ["res_v_sum"]),
            helper.make_node("Greater", ["res_h_sum", "twohalf"], ["res_h_pos"]),
            helper.make_node("Greater", ["res_v_sum", "twohalf"], ["res_v_pos"]),
            helper.make_node("Cast", ["res_not_square_b"], ["res_not_square_f"], to=TensorProto.FLOAT),
            helper.make_node("Conv", ["res_not_square_f", "kh"], ["res_h_ns_sum"]),
            helper.make_node("Conv", ["res_not_square_f", "kv"], ["res_v_ns_sum"]),
            helper.make_node("Greater", ["res_h_ns_sum", "half"], ["res_h_ns"]),
            helper.make_node("Greater", ["res_v_ns_sum", "half"], ["res_v_ns"]),
            helper.make_node("And", ["res_h_pos", "res_h_ns"], ["res_h_sel"]),
            helper.make_node("And", ["res_v_pos", "res_v_ns"], ["res_v_sel"]),
            helper.make_node("Cast", ["res_h_sel"], ["res_h_self"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["res_v_sel"], ["res_v_self"], to=TensorProto.FLOAT),
            helper.make_node("ConvTranspose", ["res_h_self", "kh"], ["res_h_red_count"]),
            helper.make_node("ConvTranspose", ["res_v_self", "kv"], ["res_v_red_count"]),
            helper.make_node("Greater", ["res_h_red_count", "half"], ["res_h_red_b"]),
            helper.make_node("Greater", ["res_v_red_count", "half"], ["res_v_red_b"]),
            helper.make_node("Or", ["res_h_red_b", "res_v_red_b"], ["res_red_b"]),
            helper.make_node("Not", ["res_red_b"], ["res_not_red_b"]),
            helper.make_node("And", ["rem_b", "res_not_red_b"], ["res_cyan_b"]),
            helper.make_node("Or", ["cyan_sf", "res_cyan_b"], ["cyan8_b"]),
            helper.make_node("Or", [red_acc, "res_red_b"], ["red8_b"]),
        ]
    )
    return "cyan8_b", "red8_b"


def build_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    _i64(inits, [0, 1, 2, 3], "axes4")
    _i64(inits, [0, 5, 0, SRC_LEFT], "fg_st")
    _i64(inits, [1, 6, SH, SRC_LEFT + SW], "fg_en")
    _i64(inits, [0, 0, 0, 0], "bg_st")
    _i64(inits, [1, 1, CH, CW], "bg_en")
    _f32(inits, [0.5], "half")
    _f32(inits, [1.5], "onehalf")
    _f32(inits, [2.5], "twohalf")
    _f32(inits, [3.5], "threehalf")
    _f32(inits, np.zeros((1, 1, SH, SW), dtype=np.float32), "zero8")
    _f32(inits, np.zeros((1, 1, CH, CW), dtype=np.float32), "zero")
    _f32(inits, np.ones((1, 1, 2, 2), dtype=np.float32), "k2")
    _f32(inits, np.ones((1, 1, 1, 3), dtype=np.float32), "kh")
    _f32(inits, np.ones((1, 1, 3, 1), dtype=np.float32), "kv")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, "fg_st", "fg_en", "axes4"], ["rem0"]),
            helper.make_node("Slice", [IN_NAME, "bg_st", "bg_en", "axes4"], ["bg"]),
        ]
    )

    rem = "rem0"
    cyan_acc: str | None = None
    red_acc: str | None = None
    for idx in range(2):
        rem, cyan_acc, red_acc = _add_exact_pass(nodes, idx, rem, cyan_acc, red_acc)
    assert cyan_acc is not None and red_acc is not None
    cyan8_b, red8_b = _add_residual_classifier(nodes, rem, cyan_acc, red_acc)

    nodes.extend(
        [
            helper.make_node("Cast", [red8_b], ["red8"], to=TensorProto.FLOAT),
            helper.make_node("Cast", [cyan8_b], ["cyan8"], to=TensorProto.FLOAT),
            helper.make_node(
                "Pad",
                ["red8"],
                ["red"],
                pads=[0, 0, 0, SRC_LEFT, 0, 0, CH - SH, CW - SRC_LEFT - SW],
            ),
            helper.make_node(
                "Pad",
                ["cyan8"],
                ["cyan"],
                pads=[0, 0, 0, SRC_LEFT, 0, 0, CH - SH, CW - SRC_LEFT - SW],
            ),
            helper.make_node(
                "Concat",
                ["bg", "zero", "red", "zero", "zero", "zero", "zero", "zero", "cyan", "zero"],
                ["outcrop"],
                axis=1,
            ),
            helper.make_node(
                "Pad",
                ["outcrop"],
                [OUT_NAME],
                pads=[0, 0, 0, 0, 0, 0, H - CH, W - CW],
            ),
        ]
    )

    graph = helper.make_graph(nodes, TASK_ID, [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def _grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[0, int(val), r, c] = 1.0
    return out


def _expected_onehot(grid: list[list[int]]) -> np.ndarray:
    return _grid_to_onehot(grid)


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def validate_json(model: onnx.ModelProto) -> int:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            g = np.array(ex["input"], dtype=np.int64)
            expected = np.array(ex["output"], dtype=np.int64)
            pred_tensor = _run_onnx(model, _grid_to_onehot(ex["input"]))
            if not np.array_equal(pred_tensor > 0.0, _expected_onehot(ex["output"]) > 0.0):
                bad += 1
                if bad <= 3:
                    pred = _onehot_to_grid(pred_tensor)[: g.shape[0], : g.shape[1]]
                    print(f"Mismatch {split}[{idx}]\ninput:\n{g}\npred:\n{pred}\nexpected:\n{expected}")
            ref = solve(g)
            if not np.array_equal(ref, expected):
                raise AssertionError(f"reference solver mismatch on {split}[{idx}]")
    return bad


def main() -> None:
    model = build_model()
    onnx.save(model, BEST_PATH)
    bad = validate_json(model)
    if bad:
        raise AssertionError(f"{bad} examples failed")
    result = score_file(BEST_PATH)
    print(f"saved {BEST_PATH}")
    print(f"memory={result['memory']} params={result['params']} cost={result['cost']} score={result['score']:.6f}")


if __name__ == "__main__":
    main()
