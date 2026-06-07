"""Minimal ONNX for ARC task026: NAND the two 5x3 halves.

Task rule: each input is a 5x7 grid with a blue separator in column 3.
For each row and each of the three paired columns, compare the left cell
with the cell four columns to its right. The 5x3 output is cyan (8) exactly
where both paired cells are black (0); all other output cells are black.

The best graph slices the 5x7 maroon-channel panel, casts it to int8, uses a
dilated quantized convolution to compute compact background/cyan masks from
paired columns, then uses a grouped quantized 1x1 convolution to place those
two masks in output channels 0 and 8 while padding to the competition shape.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TASK_ID = "task026"
TASK_NUM = 26
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task026.onnx"

C = 10
H = W = 30
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]


def _vi(name: str, elem_type: int = TensorProto.FLOAT, shape: list[int] = SHAPE):
    return helper.make_tensor_value_info(name, elem_type, shape)


def _init(arr: Any, name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def _slice10(x: str, y: str, starts: list[int], ends: list[int]) -> onnx.NodeProto:
    return helper.make_node(
        "Slice",
        [x],
        [y],
        starts=starts,
        ends=ends,
        axes=[0, 1, 2, 3],
    )


def _pad10(x: str, y: str, pads: list[int]) -> onnx.NodeProto:
    return helper.make_node("Pad", [x], [y], mode="constant", pads=pads, value=0.0)


def _slice_inputs(
    inits: list[onnx.TensorProto],
    x: str,
    y: str,
    starts: list[int],
    ends: list[int],
    tag: str,
) -> onnx.NodeProto:
    inits.extend(
        [
            _init(np.array(starts, dtype=np.int64), f"s{tag}"),
            _init(np.array(ends, dtype=np.int64), f"e{tag}"),
        ]
    )
    return helper.make_node("Slice", [x, f"s{tag}", f"e{tag}"], [y])


def reference_grid(inp: list[list[int]] | np.ndarray) -> np.ndarray:
    x = np.asarray(inp, dtype=np.int64)
    left = x[:, :3] == 0
    right = x[:, 4:7] == 0
    return np.where(left & right, 8, 0).astype(np.int64)


def to_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    out = np.zeros((1, C, H, W), dtype=np.float32)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            out[0, int(g[r, c]), r, c] = 1.0
    return out


def expected_onehot(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    return to_onehot(grid) > 0


def build_ch9_or_opset14() -> onnx.ModelProto:
    """Best: background is left-red OR right-red; cyan is its negation."""
    inits: list[onnx.TensorProto] = [
        _init(np.array([0, 0, 0, 0, 0, 1, 25, 27], dtype=np.int64), "pads")
    ]
    nodes = [
        _slice_inputs(inits, IN_NAME, "l", [0, 9, 0, 0], [1, 10, 5, 3], "l"),
        _slice_inputs(inits, IN_NAME, "r", [0, 9, 0, 4], [1, 10, 5, 7], "r"),
        helper.make_node("Cast", ["l"], ["lb"], to=TensorProto.BOOL),
        helper.make_node("Cast", ["r"], ["rb"], to=TensorProto.BOOL),
        helper.make_node("Or", ["lb", "rb"], ["bg"]),
        helper.make_node("Not", ["bg"], ["cyan"]),
        helper.make_node("Xor", ["bg", "bg"], ["z"]),
        helper.make_node(
            "Concat",
            ["bg", "z", "z", "z", "z", "z", "z", "z", "cyan"],
            ["core"],
            axis=1,
        ),
        helper.make_node("Pad", ["core", "pads"], [OUT_NAME], mode="constant"),
    ]
    return _model(nodes, inits, 14, TensorProto.BOOL)


def build_ch0_and_opset14() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        _init(np.array([0, 0, 0, 0, 0, 1, 25, 27], dtype=np.int64), "pads")
    ]
    nodes = [
        _slice_inputs(inits, IN_NAME, "l", [0, 0, 0, 0], [1, 1, 5, 3], "l"),
        _slice_inputs(inits, IN_NAME, "r", [0, 0, 0, 4], [1, 1, 5, 7], "r"),
        helper.make_node("Cast", ["l"], ["lb"], to=TensorProto.BOOL),
        helper.make_node("Cast", ["r"], ["rb"], to=TensorProto.BOOL),
        helper.make_node("And", ["lb", "rb"], ["cyan"]),
        helper.make_node("Not", ["cyan"], ["bg"]),
        helper.make_node("Xor", ["cyan", "cyan"], ["z"]),
        helper.make_node(
            "Concat",
            ["bg", "z", "z", "z", "z", "z", "z", "z", "cyan"],
            ["core"],
            axis=1,
        ),
        helper.make_node("Pad", ["core", "pads"], [OUT_NAME], mode="constant"),
    ]
    return _model(nodes, inits, 14, TensorProto.BOOL)


def build_ch9_or_opset14_inits() -> onnx.ModelProto:
    inits = [
        _init(np.array([0, 9, 0, 0], dtype=np.int64), "sl"),
        _init(np.array([1, 10, 5, 3], dtype=np.int64), "el"),
        _init(np.array([0, 9, 0, 4], dtype=np.int64), "sr"),
        _init(np.array([1, 10, 5, 7], dtype=np.int64), "er"),
        _init(np.array([0, 1, 2, 3], dtype=np.int64), "axes"),
        _init(np.array([0, 0, 0, 0, 0, 1, 25, 27], dtype=np.int64), "pads"),
    ]
    nodes = [
        helper.make_node("Slice", [IN_NAME, "sl", "el", "axes"], ["l"]),
        helper.make_node("Slice", [IN_NAME, "sr", "er", "axes"], ["r"]),
        helper.make_node("Cast", ["l"], ["lb"], to=TensorProto.BOOL),
        helper.make_node("Cast", ["r"], ["rb"], to=TensorProto.BOOL),
        helper.make_node("Or", ["lb", "rb"], ["bg"]),
        helper.make_node("Not", ["bg"], ["cyan"]),
        helper.make_node("Xor", ["bg", "bg"], ["z"]),
        helper.make_node(
            "Concat",
            ["bg", "z", "z", "z", "z", "z", "z", "z", "cyan"],
            ["core"],
            axis=1,
        ),
        helper.make_node("Pad", ["core", "pads"], [OUT_NAME], mode="constant"),
    ]
    return _model(nodes, inits, 14, TensorProto.BOOL)


def build_with_zero_initializer() -> onnx.ModelProto:
    inits = [
        _init(np.zeros((1, 7, 5, 3), dtype=np.bool_), "z7"),
        _init(np.array([0, 0, 0, 0, 0, 1, 25, 27], dtype=np.int64), "pads"),
    ]
    nodes = [
        _slice_inputs(inits, IN_NAME, "l", [0, 9, 0, 0], [1, 10, 5, 3], "l"),
        _slice_inputs(inits, IN_NAME, "r", [0, 9, 0, 4], [1, 10, 5, 7], "r"),
        helper.make_node("Cast", ["l"], ["lb"], to=TensorProto.BOOL),
        helper.make_node("Cast", ["r"], ["rb"], to=TensorProto.BOOL),
        helper.make_node("Or", ["lb", "rb"], ["bg"]),
        helper.make_node("Not", ["bg"], ["cyan"]),
        helper.make_node("Concat", ["bg", "z7", "cyan"], ["core"], axis=1),
        helper.make_node("Pad", ["core", "pads"], [OUT_NAME], mode="constant"),
    ]
    return _model(nodes, inits, 14, TensorProto.BOOL)


def build_dense_conv() -> onnx.ModelProto:
    w = np.zeros((10, 10, 1, 5), dtype=np.float32)
    b = np.zeros((10,), dtype=np.float32)
    w[0, 9, 0, 0] = 1.0
    w[0, 9, 0, 4] = 1.0
    w[0, 1, 0, 1:4] = 2.0
    b[0] = -2.5
    w[8, 0, 0, 0] = 1.0
    w[8, 0, 0, 4] = 1.0
    w[8, 1, 0, 1:4] = 2.0
    b[8] = -3.5
    nodes = [
        helper.make_node(
            "Conv",
            [IN_NAME, "W", "B"],
            [OUT_NAME],
            kernel_shape=[1, 5],
            pads=[0, 0, 0, 4],
        )
    ]
    return _model(nodes, [_init(w, "W"), _init(b, "B")], 10, TensorProto.FLOAT)


def build_int8_qconv_dilated() -> onnx.ModelProto:
    """Best: compact int8 logic with dilated QLinearConv and final padded map."""
    w1 = np.zeros((2, 1, 1, 2), dtype=np.int8)
    # Channel 0: background = left9 + right9, positive iff either pair cell is 9.
    w1[0, 0, 0, :] = [1, 1]
    # Channel 1: cyan = 1 - left9 - right9, positive iff both pair cells are 0.
    w1[1, 0, 0, :] = [-1, -1]

    w2 = np.zeros((10, 1, 1, 1), dtype=np.int8)
    w2[0, 0, 0, 0] = 1
    w2[8, 0, 0, 0] = 1

    inits = [
        _init(np.array([1, 2, 3], dtype=np.int64), "axes"),
        _init(np.array([9, 0, 0], dtype=np.int64), "s"),
        _init(np.array([10, 5, 7], dtype=np.int64), "e"),
        _init(np.array(1.0, dtype=np.float32), "scale"),
        _init(np.array(0, dtype=np.int8), "zero"),
        _init(w1, "W1"),
        _init(np.array([0, 1], dtype=np.int32), "B1"),
        _init(w2, "W2"),
    ]
    nodes = [
        helper.make_node("Slice", [IN_NAME, "s", "e", "axes"], ["x"]),
        helper.make_node("Cast", ["x"], ["xi"], to=TensorProto.INT8),
        helper.make_node(
            "QLinearConv",
            ["xi", "scale", "zero", "W1", "scale", "zero", "scale", "zero", "B1"],
            ["corei"],
            kernel_shape=[1, 2],
            dilations=[1, 4],
        ),
        helper.make_node(
            "QLinearConv",
            ["corei", "scale", "zero", "W2", "scale", "zero", "scale", "zero"],
            [OUT_NAME],
            kernel_shape=[1, 1],
            pads=[0, 0, 25, 27],
            group=2,
        ),
    ]
    return _model(nodes, inits, 10, TensorProto.INT8)


def _model(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    opset: int,
    out_type: int,
) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        "g",
        [_vi(IN_NAME, TensorProto.FLOAT)],
        [_vi(OUT_NAME, out_type)],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def _load_examples() -> list[dict[str, list[list[int]]]]:
    with (ROOT / "data" / f"{TASK_ID}.json").open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [ex for split in ("train", "test", "arc-gen") for ex in data[split]]


def verify_model(model: onnx.ModelProto) -> tuple[bool, int]:
    session = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    wrong = 0
    for ex in _load_examples():
        got = session.run([OUT_NAME], {IN_NAME: to_onehot(ex["input"])})[0] > 0
        exp = expected_onehot(ex["output"])
        if not np.array_equal(got, exp):
            wrong += 1
    return wrong == 0, wrong


def score_path(path: Path) -> dict[str, Any]:
    import score_model

    return score_model.score_file(path)


def largest_internal_tensor(path: Path) -> int | None:
    import score_model

    model = score_model.sanitize_model(onnx.load(str(path)))
    if model is None:
        return None
    inputs = score_model.load_task_examples(path)
    trace_path, error = score_model.run_profiled_session(model, inputs, path)
    if error or trace_path is None:
        return None
    try:
        graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    except Exception:
        return None
    io_names = {t.name for t in list(graph.input) + list(graph.output)}
    tensor_map = {t.name: t for t in list(graph.value_info) + list(graph.input) + list(graph.output)}
    largest = 0
    for node in graph.node:
        for name in node.output:
            if name in io_names:
                continue
            item = tensor_map.get(name)
            if item is None or not item.type.HasField("tensor_type"):
                continue
            tt = item.type.tensor_type
            n = 1
            for dim in tt.shape.dim:
                if dim.HasField("dim_value"):
                    n *= dim.dim_value
            dtype = onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)
            largest = max(largest, int(n * np.dtype(dtype).itemsize))
    return largest


VARIANTS: list[tuple[str, Callable[[], onnx.ModelProto]]] = [
    ("int8_qconv_dilated", build_int8_qconv_dilated),
    ("ch9_or_opset14_xor_zero", build_ch9_or_opset14),
    ("ch0_and_opset14_xor_zero", build_ch0_and_opset14),
    ("ch9_or_opset14_initializer_slices", build_ch9_or_opset14_inits),
    ("ch9_or_opset14_zero_initializer", build_with_zero_initializer),
    ("dense_conv_direct_output", build_dense_conv),
]


def benchmark_variants() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="task026_") as td:
        tmp = Path(td)
        for name, builder in VARIANTS:
            path = tmp / f"{name}.onnx"
            model = builder()
            onnx.save(model, str(path))
            ok, wrong = verify_model(model)
            result = score_path(path)
            rows.append(
                {
                    "name": name,
                    "ok": ok,
                    "wrong": wrong,
                    "memory": result.get("memory"),
                    "params": result.get("params"),
                    "cost": result.get("cost"),
                    "score": result.get("score"),
                    "largest": largest_internal_tensor(path) if result.get("valid") else None,
                    "valid": result.get("valid"),
                    "error": result.get("error"),
                    "model": model,
                }
            )
    return rows


def save_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    rows = benchmark_variants()
    good = [r for r in rows if r["ok"] and r["valid"]]
    if not good:
        raise RuntimeError("no valid correct task026 variant")
    best = min(good, key=lambda r: int(r["cost"]))
    model = best["model"]
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return model


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def main() -> None:
    rows = benchmark_variants()
    print("Inferred rule: output 8 where paired cells in columns c and c+4 are both 0; else 0.")
    print("variant,memory,params,cost,score,largest_internal,correct")
    for r in rows:
        print(
            ",".join(
                [
                    r["name"],
                    _fmt(r["memory"]),
                    _fmt(r["params"]),
                    _fmt(r["cost"]),
                    _fmt(r["score"]),
                    _fmt(r["largest"]),
                    "yes" if r["ok"] else f"no({r['wrong']})",
                ]
            )
        )
    good = [r for r in rows if r["ok"] and r["valid"]]
    best = min(good, key=lambda r: int(r["cost"]))
    onnx.save(best["model"], str(BEST_PATH))
    print(f"saved {BEST_PATH} from {best['name']} cost={best['cost']} score={best['score']:.6f}")
    print(f"reference sanity score at cost {best['cost']}: {max(1.0, 25.0 - math.log(best['cost'])):.6f}")


if __name__ == "__main__":
    main()
