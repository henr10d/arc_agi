"""Minimal ONNX for ARC task017: repair an erased periodic 21x21 pattern.

Task rule: the input is a 21x21 square whose nonzero cells come from one
periodic color tile. Some cells have been erased to color 0. Find the smallest
period in {2, 4, 5, 6, 7, 8, 9} whose residue classes contain no conflicting
nonzero colors, use the known colors to reconstruct that tile, repeat it over
the 21x21 region, and leave the padded area outside the task grid all-zero.

ONNX approach: convert the one-hot input to a compact int32 color grid. For
each supported period, gather all cells belonging to each residue class, reject
periods whose known nonzero colors conflict, and use the per-residue maximum as
the reconstructed tile color. The graph creates the final one-hot float tensor
only at the output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task017.onnx"
DATA_PATH = OUT_DIR.parent / "data" / "task017.json"

C = 10
GH = GW = 21
H = W = 30
NCELL = GH * GW
PERIODS: Tuple[int, ...] = (2, 4, 5, 6, 7, 8, 9)
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def _init(inits: List[onnx.TensorProto], arr, name: str, dtype) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=dtype), name=name))
    return name


def _i64(inits: List[onnx.TensorProto], vals: Sequence[int], name: str) -> str:
    return _init(inits, vals, name, np.int64)


def _i32(inits: List[onnx.TensorProto], arr, name: str) -> str:
    return _init(inits, arr, name, np.int32)


def tile_index_grid(period: int) -> np.ndarray:
    idx = np.zeros((1, GH, GW), dtype=np.int64)
    for r in range(GH):
        for c in range(GW):
            idx[0, r, c] = (r % period) * period + (c % period)
    return idx


def _grid_period(grid: np.ndarray) -> int:
    for period in PERIODS:
        if all(grid[r, c] == grid[r % period, c % period] for r in range(GH) for c in range(GW)):
            return period
    raise ValueError("task017 grid does not match a supported period")


def residue_index_table(period: int) -> np.ndarray:
    """Flat 21x21 source indices grouped by residue modulo the period."""
    rows: List[List[int]] = []
    max_k = 0
    for rr in range(period):
        for cc in range(period):
            row: List[int] = []
            for r in range(rr, GH, period):
                for c in range(cc, GW, period):
                    row.append(r * GW + c)
            rows.append(row)
            max_k = max(max_k, len(row))

    table = np.zeros((period * period, max_k), dtype=np.int64)
    for row_idx, row in enumerate(rows):
        table[row_idx, : len(row)] = row
        if len(row) < max_k:
            table[row_idx, len(row) :] = row[-1]
    return table


def build_reference_numpy(grid: np.ndarray) -> np.ndarray:
    """Fill the smallest non-conflicting square period in the task period set."""
    g = np.asarray(grid, dtype=np.int64)
    if g.ndim == 4:
        g = g[0].argmax(axis=0)
    elif g.ndim == 3:
        g = g.argmax(axis=0)

    h, w = g.shape
    for period in PERIODS:
        tmpl = np.zeros((period, period), dtype=np.int64)
        ok = True
        for r in range(h):
            for c in range(w):
                v = int(g[r, c])
                if v == 0:
                    continue
                rr, cc = r % period, c % period
                if tmpl[rr, cc] == 0:
                    tmpl[rr, cc] = v
                elif tmpl[rr, cc] != v:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return np.fromfunction(
                lambda r, c: tmpl[(r.astype(np.int64) % period), (c.astype(np.int64) % period)],
                (h, w),
                dtype=np.int64,
            ).astype(np.int64)
    return g.copy()


def _grid_to_onehot(grid: List[List[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros((1, C, H, W), dtype=np.float32)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            color = int(arr[r, c])
            if 0 <= color < C:
                out[0, color, r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def _period_candidate(
    nodes: List[onnx.NodeProto],
    inits: List[onnx.TensorProto],
    flat_colors: str,
    period: int,
    *,
    need_valid: bool,
) -> Tuple[str, str | None]:
    idx = _i64(inits, residue_index_table(period), f"p{period}_src_idx")
    gathered = f"p{period}_g"
    maxv = f"p{period}_max"
    nodes.extend(
        [
            helper.make_node("Gather", [flat_colors, idx], [gathered], axis=0),
            helper.make_node("ReduceMax", [gathered], [maxv], axes=[1], keepdims=0),
        ]
    )

    valid = None
    if need_valid:
        nodes.extend(
            [
                helper.make_node("Equal", [gathered, "zero_i"], [f"p{period}_is_zero"]),
                helper.make_node("Where", [f"p{period}_is_zero", "ten_i", gathered], [f"p{period}_nz_or_10"]),
                helper.make_node("ReduceMin", [f"p{period}_nz_or_10"], [f"p{period}_min"], axes=[1], keepdims=0),
                helper.make_node("Equal", [maxv, f"p{period}_min"], [f"p{period}_same"]),
                helper.make_node("Greater", [maxv, "zero_i"], [f"p{period}_has_known"]),
                helper.make_node("Not", [f"p{period}_same"], [f"p{period}_diff"]),
                helper.make_node("And", [f"p{period}_has_known", f"p{period}_diff"], [f"p{period}_res_bad"]),
                helper.make_node("Cast", [f"p{period}_res_bad"], [f"p{period}_bad_i"], to=TensorProto.INT32),
                helper.make_node("ReduceMax", [f"p{period}_bad_i"], [f"p{period}_bad"], axes=[0], keepdims=0),
                helper.make_node("Equal", [f"p{period}_bad", "zero_i"], [f"p{period}_valid"]),
            ]
        )
        valid = f"p{period}_valid"

    return maxv, valid


def build_onnx_model() -> onnx.ModelProto:
    """Build the compact color-index implementation for task017."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    shape = [1, C, H, W]
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, shape)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, shape)

    _i64(inits, [1, 2], "ax_hw")
    _i64(inits, [0, 0], "st_hw")
    _i64(inits, [GH, GW], "en_hw")
    _i64(inits, [NCELL], "flat_shape")
    _i32(inits, 0, "zero_i")
    _i32(inits, 10, "ten_i")
    _i32(inits, np.arange(C, dtype=np.int32).reshape(1, C, 1, 1), "colors")

    nodes.extend(
        [
            helper.make_node("ArgMax", [IN_NAME], ["arg"], axis=1, keepdims=0),
            helper.make_node("Slice", ["arg", "st_hw", "en_hw", "ax_hw"], ["arg21"]),
            helper.make_node("Cast", ["arg21"], ["colors21"], to=TensorProto.INT32),
            helper.make_node("Reshape", ["colors21", "flat_shape"], ["flat"]),
        ]
    )

    candidates: List[Tuple[str, str | None]] = []
    for period in PERIODS:
        candidates.append(_period_candidate(nodes, inits, "flat", period, need_valid=(period != PERIODS[-1])))

    maps: List[str] = []
    offset = 0
    for period, (template, _) in zip(PERIODS, candidates):
        maps.append(_i32(inits, tile_index_grid(period) + offset, f"p{period}_out_idx"))
        offset += period * period

    nodes.append(helper.make_node("Concat", [template for template, _ in candidates], ["all_templates"], axis=0))

    selected_map = maps[-1]
    for index_map, (_, valid) in reversed(list(zip(maps[:-1], candidates[:-1]))):
        assert valid is not None
        nodes.append(helper.make_node("Where", [valid, index_map, selected_map], [f"{index_map}_sel"]))
        selected_map = f"{index_map}_sel"

    nodes.append(helper.make_node("Gather", ["all_templates", selected_map], ["selected"], axis=0))

    row_pad = _i32(inits, np.full((1, H - GH, GW), -1, dtype=np.int32), "row_pad")
    col_pad = _i32(inits, np.full((1, H, W - GW), -1, dtype=np.int32), "col_pad")
    nodes.extend(
        [
            helper.make_node("Concat", ["selected", "row_pad"], ["out30h"], axis=1),
            helper.make_node("Concat", ["out30h", "col_pad"], ["outidx"], axis=2),
            helper.make_node("Equal", ["outidx", "colors"], ["outb"]),
            helper.make_node("Cast", ["outb"], [OUT_NAME], to=TensorProto.FLOAT),
        ]
    )

    graph = helper.make_graph(nodes, "g", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def save_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    model = build_onnx_model()
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return model


def model_stats(model: onnx.ModelProto) -> Dict[str, int]:
    params = sum(int(np.prod(list(t.dims))) if t.dims else 1 for t in model.graph.initializer)
    return {"nodes": len(model.graph.node), "params": params, "inits": len(model.graph.initializer)}


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]
    except ImportError:
        pass

    from onnx import reference

    sess = reference.ReferenceEvaluator(model)
    return sess.run(None, {IN_NAME: x.astype(np.float32)})[0]


def test() -> None:
    model = build_onnx_model()
    stats = model_stats(model)
    print(f"nodes={stats['nodes']} params={stats['params']} inits={stats['inits']}")

    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    bad = 0
    for split in ("train", "test", "arc-gen"):
        for index, ex in enumerate(data[split]):
            inp = np.array(ex["input"], dtype=np.int64)
            exp = np.array(ex["output"], dtype=np.int64)
            ref = build_reference_numpy(inp)
            if not np.array_equal(ref, exp):
                print(f"reference mismatch: {split} {index}")
                bad += 1
                continue
            pred = _onehot_to_grid(_run_onnx(model, _grid_to_onehot(inp))[0])[: inp.shape[0], : inp.shape[1]]
            if not np.array_equal(pred, exp):
                print(f"onnx mismatch: {split} {index}")
                bad += 1
    print(f"task017.json: {'PASS' if bad == 0 else f'FAIL ({bad} wrong)'}")


def main() -> None:
    save_model()
    test()
    print(f"saved {BEST_PATH}")


if __name__ == "__main__":
    main()
