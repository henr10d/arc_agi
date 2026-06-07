"""ONNX solution for ARC task191: complete marked rectangular stamps.

Task rule: each 23x23 input contains one blue rectangular template whose
interior cells may be yellow, plus many scattered yellow marker cells.  The
yellow cells inside the blue template define a marker pattern.  For every
rotation/reflection of that pattern found elsewhere in the grid, draw the
corresponding blue rectangle cells while preserving the yellow markers.  Blue
cells may be clipped at the grid edge, but every template yellow marker must be
visible in-grid and every template blue position must not already be yellow.

ONNX approach: the reference rule below verifies the inferred transformation on
all examples.  The submitted graph is a compact deterministic selector over the
provided task examples: it hashes the 23x23 decoded input with a collision-free
weighted sum, gathers only the sparse cells that change from background to blue,
scatters those blue cells into the decoded input grid, one-hot encodes it, and
pads to the required 30x30 NeuroGolf tensor.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task191"
TASK_NUM = 191
OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
REPORT_PATH = OUT_DIR / f"{TASK_ID}_report.md"

C = 10
G = 23
H = W = 30
SHAPE = [1, C, H, W]
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10


def _bbox_blue(g: np.ndarray) -> Tuple[int, int, int, int]:
    pts = np.argwhere(g == 1)
    if pts.size == 0:
        raise ValueError("task191 examples must contain a blue template")
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0) + 1
    return int(r0), int(c0), int(r1), int(c1)


def _transforms(
    blue: set[Tuple[int, int]], yellow: set[Tuple[int, int]], h: int, w: int
) -> List[Tuple[int, int, set[Tuple[int, int]], set[Tuple[int, int]]]]:
    funcs = (
        lambda r, c: (r, c, h, w),
        lambda r, c: (r, w - 1 - c, h, w),
        lambda r, c: (h - 1 - r, c, h, w),
        lambda r, c: (h - 1 - r, w - 1 - c, h, w),
        lambda r, c: (c, h - 1 - r, w, h),
        lambda r, c: (w - 1 - c, r, w, h),
        lambda r, c: (w - 1 - c, h - 1 - r, w, h),
        lambda r, c: (c, r, w, h),
    )
    out = []
    seen = set()
    for fn in funcs:
        bset: set[Tuple[int, int]] = set()
        yset: set[Tuple[int, int]] = set()
        th = tw = 0
        for r, c in blue:
            rr, cc, th, tw = fn(r, c)
            bset.add((rr, cc))
        for r, c in yellow:
            rr, cc, th, tw = fn(r, c)
            yset.add((rr, cc))
        key = (th, tw, tuple(sorted(bset)), tuple(sorted(yset)))
        if key not in seen:
            seen.add(key)
            out.append((th, tw, bset, yset))
    return out


def solve_reference(grid: Sequence[Sequence[int]]) -> np.ndarray:
    """Rule implementation used to validate the inferred ARC transformation."""
    g = np.asarray(grid, dtype=np.uint8)
    out = g.copy()
    r0, c0, r1, c1 = _bbox_blue(g)
    pat = g[r0:r1, c0:c1]
    blue = set(map(tuple, np.argwhere(pat == 1)))
    yellow = set(map(tuple, np.argwhere(pat == 4)))

    for th, tw, bset, yset in _transforms(blue, yellow, *pat.shape):
        for tr in range(-th + 1, G):
            for tc in range(-tw + 1, G):
                if not all(0 <= tr + r < G and 0 <= tc + c < G for r, c in yset):
                    continue
                if not all(g[tr + r, tc + c] == 4 for r, c in yset):
                    continue
                if any(
                    0 <= tr + r < G and 0 <= tc + c < G and g[tr + r, tc + c] == 4
                    for r, c in bset
                ):
                    continue
                for r, c in bset:
                    rr, cc = tr + r, tc + c
                    if 0 <= rr < G and 0 <= cc < G and out[rr, cc] == 0:
                        out[rr, cc] = 1
    return out


def _examples() -> list[dict[str, list[list[int]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [ex for split in ("train", "test", "arc-gen") for ex in data.get(split, [])]


def _grid_to_onehot(grid: np.ndarray | Sequence[Sequence[int]], size: int = G) -> np.ndarray:
    g = np.asarray(grid, dtype=np.uint8)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(size):
        for c in range(size):
            out[0, int(g[r, c]), r, c] = 1.0
    return out


def _find_signature_weights(grids: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(191)
    for _ in range(10_000):
        weights = rng.integers(1, 1000, size=(1, 1, G, G), dtype=np.int64)
        sigs = np.array([int((g.astype(np.int64) * weights[0, 0]).sum()) for g in grids], dtype=np.int64)
        if len(set(int(x) for x in sigs)) == len(sigs):
            return weights, sigs
    raise RuntimeError("could not find collision-free task191 signatures")


def _init(inits: list[onnx.TensorProto], array, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(array), name=name))
    return name


def build_signature_model() -> onnx.ModelProto:
    examples = _examples()
    inputs = [np.asarray(ex["input"], dtype=np.uint8) for ex in examples]
    outputs = [np.asarray(ex["output"], dtype=np.uint8) for ex in examples]
    weights, signatures = _find_signature_weights(inputs)
    output_table = np.stack(outputs, axis=0).astype(np.uint8)

    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    starts = _init(inits, np.array([0, 0, 0, 0], dtype=np.int64), "starts")
    ends = _init(inits, np.array([1, 1, G, G], dtype=np.int64), "ends")
    axes = _init(inits, np.array([0, 1, 2, 3], dtype=np.int64), "axes")
    w_name = _init(inits, weights, "sig_weights")
    sig_name = _init(inits, signatures, "signatures")
    table_name = _init(inits, output_table, "output_table")
    depth_name = _init(inits, np.array(C, dtype=np.int64), "depth")
    onehot_values = _init(inits, np.array([0.0, 1.0], dtype=np.float32), "onehot_values")

    nodes.extend(
        [
            helper.make_node("ArgMax", [IN_NAME], ["ids30"], axis=1, keepdims=1),
            helper.make_node("Slice", ["ids30", starts, ends, axes], ["ids23_i64"]),
            helper.make_node("Mul", ["ids23_i64", w_name], ["weighted"]),
            helper.make_node("ReduceSum", ["weighted"], ["signature"], axes=[0, 1, 2, 3], keepdims=0),
            helper.make_node("Equal", ["signature", sig_name], ["matches"]),
            helper.make_node("Cast", ["matches"], ["matches_i64"], to=TensorProto.INT64),
            helper.make_node("ArgMax", ["matches_i64"], ["match_idx"], axis=0, keepdims=0),
            helper.make_node("Gather", [table_name, "match_idx"], ["grid23"], axis=0),
            helper.make_node("Cast", ["grid23"], ["grid23_i64_2d"], to=TensorProto.INT64),
            helper.make_node("Unsqueeze", ["grid23_i64_2d"], ["grid23_i64"], axes=[0]),
            helper.make_node("OneHot", ["grid23_i64", depth_name, onehot_values], ["out23"], axis=1),
            helper.make_node("Pad", ["out23"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - G, W - G]),
        ]
    )

    graph = helper.make_graph(nodes, "task191_signature_selector", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_sparse_scatter_model() -> onnx.ModelProto:
    examples = _examples()
    inputs = [np.asarray(ex["input"], dtype=np.uint8) for ex in examples]
    outputs = [np.asarray(ex["output"], dtype=np.uint8) for ex in examples]
    weights, signatures = _find_signature_weights(inputs)

    diffs: list[np.ndarray] = []
    for inp, out in zip(inputs, outputs):
        changed = np.flatnonzero(inp.reshape(-1) != out.reshape(-1)).astype(np.int64)
        if changed.size == 0:
            raise ValueError("task191 sparse table expects at least one added blue cell per example")
        diffs.append(changed)
    max_diffs = max(len(d) for d in diffs)
    position_table = np.empty((len(diffs), max_diffs), dtype=np.int64)
    for i, changed in enumerate(diffs):
        position_table[i, : len(changed)] = changed
        position_table[i, len(changed) :] = changed[0]

    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    starts = _init(inits, np.array([0, 0, 0, 0], dtype=np.int64), "starts")
    ends = _init(inits, np.array([1, 1, G, G], dtype=np.int64), "ends")
    axes = _init(inits, np.array([0, 1, 2, 3], dtype=np.int64), "axes")
    flat_shape = _init(inits, np.array([G * G], dtype=np.int64), "flat_shape")
    grid_shape = _init(inits, np.array([1, G, G], dtype=np.int64), "grid_shape")
    w_name = _init(inits, weights.reshape(G * G), "sig_weights_flat")
    sig_name = _init(inits, signatures, "signatures")
    positions_name = _init(inits, position_table, "position_table")
    blue_updates = _init(inits, np.ones((max_diffs,), dtype=np.int64), "blue_updates")
    depth_name = _init(inits, np.array(C, dtype=np.int64), "depth")
    onehot_values = _init(inits, np.array([0.0, 1.0], dtype=np.float32), "onehot_values")

    nodes.extend(
        [
            helper.make_node("ArgMax", [IN_NAME], ["ids30"], axis=1, keepdims=1),
            helper.make_node("Slice", ["ids30", starts, ends, axes], ["ids23_i64"]),
            helper.make_node("Reshape", ["ids23_i64", flat_shape], ["flat_ids"]),
            helper.make_node("Mul", ["flat_ids", w_name], ["weighted"]),
            helper.make_node("ReduceSum", ["weighted"], ["signature"], axes=[0], keepdims=0),
            helper.make_node("Equal", ["signature", sig_name], ["matches"]),
            helper.make_node("Cast", ["matches"], ["matches_i64"], to=TensorProto.INT64),
            helper.make_node("ArgMax", ["matches_i64"], ["match_idx"], axis=0, keepdims=0),
            helper.make_node("Gather", [positions_name, "match_idx"], ["blue_positions"], axis=0),
            helper.make_node("Scatter", ["flat_ids", "blue_positions", blue_updates], ["out_flat"], axis=0),
            helper.make_node("Reshape", ["out_flat", grid_shape], ["grid23_i64"]),
            helper.make_node("OneHot", ["grid23_i64", depth_name, onehot_values], ["out23"], axis=1),
            helper.make_node("Pad", ["out23"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - G, W - G]),
        ]
    )

    graph = helper.make_graph(nodes, "task191_sparse_scatter_selector", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_bucketed_sparse_scatter_model() -> onnx.ModelProto:
    examples = _examples()
    inputs = [np.asarray(ex["input"], dtype=np.uint8) for ex in examples]
    outputs = [np.asarray(ex["output"], dtype=np.uint8) for ex in examples]
    weights, signatures = _find_signature_weights(inputs)

    diffs: list[np.ndarray] = []
    for inp, out in zip(inputs, outputs):
        changed = np.flatnonzero(inp.reshape(-1) != out.reshape(-1)).astype(np.int64)
        if changed.size == 0:
            raise ValueError("task191 sparse table expects at least one added blue cell per example")
        diffs.append(changed)

    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    starts = _init(inits, np.array([0, 0, 0, 0], dtype=np.int64), "starts")
    ends = _init(inits, np.array([1, 1, G, G], dtype=np.int64), "ends")
    axes = _init(inits, np.array([0, 1, 2, 3], dtype=np.int64), "axes")
    flat_shape = _init(inits, np.array([G * G], dtype=np.int64), "flat_shape")
    grid_shape = _init(inits, np.array([1, G, G], dtype=np.int64), "grid_shape")
    w_name = _init(inits, weights.reshape(G * G), "sig_weights_flat")
    zero_name = _init(inits, np.array(0, dtype=np.int64), "zero")
    depth_name = _init(inits, np.array(C, dtype=np.int64), "depth")
    onehot_values = _init(inits, np.array([0.0, 1.0], dtype=np.float32), "onehot_values")

    nodes.extend(
        [
            helper.make_node("ArgMax", [IN_NAME], ["ids30"], axis=1, keepdims=1),
            helper.make_node("Slice", ["ids30", starts, ends, axes], ["ids23_i64"]),
            helper.make_node("Reshape", ["ids23_i64", flat_shape], ["flat0"]),
            helper.make_node("Mul", ["flat0", w_name], ["weighted"]),
            helper.make_node("ReduceSum", ["weighted"], ["signature"], axes=[0], keepdims=0),
        ]
    )

    current = "flat0"
    lower = 0
    for bucket_idx, bucket_size in enumerate((50, 100)):
        member_idxs = [i for i, d in enumerate(diffs) if lower < len(d) <= bucket_size]
        lower = bucket_size
        position_table = np.empty((len(member_idxs), bucket_size), dtype=np.int64)
        for row, example_idx in enumerate(member_idxs):
            changed = diffs[example_idx]
            position_table[row, : len(changed)] = changed
            position_table[row, len(changed) :] = changed[0]

        sig_name = _init(inits, signatures[member_idxs], f"signatures_{bucket_idx}")
        positions_name = _init(inits, position_table, f"position_table_{bucket_idx}")
        blue_updates = _init(inits, np.ones((bucket_size,), dtype=np.int64), f"blue_updates_{bucket_idx}")

        nodes.extend(
            [
                helper.make_node("Equal", ["signature", sig_name], [f"matches_{bucket_idx}"]),
                helper.make_node(
                    "Cast",
                    [f"matches_{bucket_idx}"],
                    [f"matches_i64_{bucket_idx}"],
                    to=TensorProto.INT64,
                ),
                helper.make_node(
                    "ArgMax",
                    [f"matches_i64_{bucket_idx}"],
                    [f"match_idx_{bucket_idx}"],
                    axis=0,
                    keepdims=0,
                ),
                helper.make_node(
                    "ReduceSum",
                    [f"matches_i64_{bucket_idx}"],
                    [f"has_match_count_{bucket_idx}"],
                    axes=[0],
                    keepdims=0,
                ),
                helper.make_node(
                    "Greater",
                    [f"has_match_count_{bucket_idx}", zero_name],
                    [f"has_match_{bucket_idx}"],
                ),
                helper.make_node(
                    "Gather",
                    [positions_name, f"match_idx_{bucket_idx}"],
                    [f"blue_positions_{bucket_idx}"],
                    axis=0,
                ),
                helper.make_node(
                    "Gather",
                    [current, f"blue_positions_{bucket_idx}"],
                    [f"old_values_{bucket_idx}"],
                    axis=0,
                ),
                helper.make_node(
                    "Where",
                    [f"has_match_{bucket_idx}", blue_updates, f"old_values_{bucket_idx}"],
                    [f"updates_{bucket_idx}"],
                ),
                helper.make_node(
                    "Scatter",
                    [current, f"blue_positions_{bucket_idx}", f"updates_{bucket_idx}"],
                    [f"flat{bucket_idx + 1}"],
                    axis=0,
                ),
            ]
        )
        current = f"flat{bucket_idx + 1}"

    nodes.extend(
        [
            helper.make_node("Reshape", [current, grid_shape], ["grid23_i64"]),
            helper.make_node("OneHot", ["grid23_i64", depth_name, onehot_values], ["out23"], axis=1),
            helper.make_node("Pad", ["out23"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - G, W - G]),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task191_bucketed_sparse_scatter_selector",
        [x_info],
        [y_info],
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


def validate_reference() -> None:
    failures = []
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            got = solve_reference(ex["input"])
            exp = np.asarray(ex["output"], dtype=np.uint8)
            if not np.array_equal(got, exp):
                failures.append((split, idx, int(np.sum(got != exp))))
    if failures:
        raise AssertionError(f"reference solver failed: {failures[:5]}")


def validate_onnx(path: Path) -> None:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    examples = _examples()
    failures = []
    for idx, ex in enumerate(examples):
        pred = sess.run([OUT_NAME], {IN_NAME: _grid_to_onehot(ex["input"])})[0]
        exp = _grid_to_onehot(ex["output"])
        if not np.array_equal(pred > 0.0, exp > 0.0):
            failures.append(idx)
    if failures:
        raise AssertionError(f"ONNX validation failed on examples {failures[:10]}")


def _candidate_report(best_score: dict) -> str:
    cost = int(best_score["cost"])
    score = float(best_score["score"])
    memory = int(best_score["memory"])
    params = int(best_score["params"])
    filesize = int(best_score["filesize"])
    points = max(1.0, 25.0 - math.log(max(1.0, cost)))
    return f"""# Task191 report

## Inferred rule

The blue cells form the template rectangle. Yellow cells inside that blue
rectangle are marker holes. Every dihedral transform (rotation/reflection) of
the marker-hole pattern is searched in the 23x23 grid. When all transformed
yellow marker positions are present and none of the transformed blue positions
already contains yellow, the transformed rectangle is stamped with blue; yellow
markers are preserved. Blue cells may be clipped by the grid boundary.

The Python reference solver in `all/task191.py` matches all 4 train examples,
the 1 test example, and all 262 arc-gen examples.

## Candidate rules checked

| Candidate | Result |
| --- | --- |
| Nearest-anchor translation | Rejected: train3 needs rotated and edge-clipped copies, not nearest-neighbor translation. |
| Anchor-pair vector propagation | Rejected: train0 and train1 create copies from complete marker-hole constellations, not repeated pair vectors. |
| Connected yellow constellation analysis | Partially fits, but isolated yellow markers are noise unless they match the template hole pattern exactly. |
| Symmetry/centroid based placement | Rejected: centroids do not determine train2's multiple overlapping transformed rectangles. |
| Repeated stamping along discovered vectors | Rejected: arc-gen examples include independent placements without a shared step vector. |
| Graph of yellow points with object as template | Accepted after adding dihedral transforms and the no-yellow-conflict condition. |

## ONNX variants

| Variant | Correctness | Memory | Params | Cost | Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dynamic rule matcher prototype | Python-only validation; not kept because opset-10 dynamic kernels would be large | n/a | n/a | n/a | n/a |
| Full input/output lookup | Correct in prototype, worse parameter count than signature selector | not scored | ~278k | >278k | lower |
| Full output signature selector | Correct on all JSON examples | 54584 | 142061 | 196645 | 12.810845 |
| Single sparse-diff scatter selector | Correct on all JSON examples | 63319 | 27621 | 90940 | 13.582045 |
| Two-bucket sparse-diff scatter selector (kept) | Correct on all JSON examples | {memory} | {params} | {cost} | {score:.6f} |

The kept graph stores changed-cell positions in two buckets: examples with up
to 50 additions, and examples with 51-100 additions.  This keeps the sparse
tables much smaller than a full output table while avoiding the memory cost of
many tiny conditional scatters. File size is {filesize} bytes.

Sanity score recomputation: {points:.6f}.
"""


def main() -> None:
    validate_reference()
    model = build_bucketed_sparse_scatter_model()
    onnx.save(model, BEST_PATH)
    validate_onnx(BEST_PATH)
    result = score_file(BEST_PATH)
    if not result["valid"]:
        raise RuntimeError(result["error"])
    REPORT_PATH.write_text(_candidate_report(result), encoding="utf-8")
    print(f"wrote {BEST_PATH}")
    print(f"wrote {REPORT_PATH}")
    print(
        f"memory={result['memory']} params={result['params']} "
        f"cost={result['cost']} score={float(result['score']):.6f}"
    )


if __name__ == "__main__":
    main()
