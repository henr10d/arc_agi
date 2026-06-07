"""Convert each complete 3x3 blue hollow square into a red plus.

Task rule: inputs are 10x10 grids with background 0 and blue 1.  Every
complete 3x3 blue frame with an empty center is replaced by a red 3x3 plus;
the four blue corners are erased to background and all other blue objects are
preserved.  The ONNX model keeps detection on compact 8x8/10x10 masks and only
pads to the required Kaggle one-hot [1,10,30,30] output at the final node.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_model import score_file

TASK_ID = "task160"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / "task160.onnx"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"

IN_NAME = "input"
OUT_NAME = "output"
C = 10
H = W = 30
N = 10
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def solve(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    x = np.asarray(grid, dtype=np.int64)
    out = x.copy()
    for r in range(N - 2):
        for c in range(N - 2):
            patch = x[r : r + 3, c : c + 3]
            if (
                patch[1, 1] == 0
                and np.all(patch[[0, 0, 0, 1, 1, 2, 2, 2], [0, 1, 2, 0, 2, 0, 1, 2]] == 1)
            ):
                out[r : r + 3, c : c + 3] = 0
                out[r, c + 1] = 2
                out[r + 1, c : c + 3] = 2
                out[r + 2, c + 1] = 2
    return out


def _init(inits: list[onnx.TensorProto], name: str, arr: np.ndarray | list[int] | list[float]) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _i64(inits: list[onnx.TensorProto], name: str, vals: list[int]) -> str:
    return _init(inits, name, np.asarray(vals, dtype=np.int64))


def _f32(inits: list[onnx.TensorProto], name: str, vals: np.ndarray | list[float]) -> str:
    return _init(inits, name, np.asarray(vals, dtype=np.float32))


def _slice(nodes: list[onnx.NodeProto], data: str, out: str, starts: str, ends: str, axes: str | None = None) -> str:
    inputs = [data, starts, ends] if axes is None else [data, starts, ends, axes]
    nodes.append(helper.make_node("Slice", inputs, [out]))
    return out


def _pad10(nodes: list[onnx.NodeProto], data: str, out: str, r: int, c: int) -> str:
    nodes.append(
        helper.make_node(
            "Pad",
            [data],
            [out],
            pads=[0, 0, r, c, 0, 0, N - 8 - r, N - 8 - c],
        )
    )
    return out


def _or_chain(nodes: list[onnx.NodeProto], names: list[str], prefix: str) -> str:
    cur = names[0]
    for i, name in enumerate(names[1:], 1):
        nxt = f"{prefix}{i}"
        nodes.append(helper.make_node("Or", [cur, name], [nxt]))
        cur = nxt
    return cur


def _add_chain(nodes: list[onnx.NodeProto], names: list[str], prefix: str) -> str:
    cur = names[0]
    for i, name in enumerate(names[1:], 1):
        nxt = f"{prefix}{i}"
        nodes.append(helper.make_node("Add", [cur, name], [nxt]))
        cur = nxt
    return cur


def _and_chain(nodes: list[onnx.NodeProto], names: list[str], prefix: str) -> str:
    cur = names[0]
    for i, name in enumerate(names[1:], 1):
        nxt = f"{prefix}{i}"
        nodes.append(helper.make_node("And", [cur, name], [nxt]))
        cur = nxt
    return cur


def _finish(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], plus: str, square: str, c2: str) -> None:
    s0 = _i64(inits, "f_s0", [0, 0, 0, 0])
    e_c0 = _i64(inits, "f_e_c0", [1, 1, N, N])
    s_c1 = _i64(inits, "f_s_c1", [0, 1, 0, 0])
    e_c1 = _i64(inits, "f_e_c1", [1, 2, N, N])
    zero = _f32(inits, "f_zero", [0.0])
    one = _f32(inits, "f_one", [1.0])

    _slice(nodes, IN_NAME, "in0", s0, e_c0)
    _slice(nodes, IN_NAME, "in1", s_c1, e_c1)
    nodes.append(helper.make_node("Where", [square, one, "in0"], ["bg_sq"]))
    nodes.append(helper.make_node("Where", [plus, zero, "bg_sq"], ["c0"]))
    nodes.append(helper.make_node("Where", [square, zero, "in1"], ["c1"]))
    nodes.append(helper.make_node("Concat", ["c0", "c1", c2], ["c012"], axis=1))
    nodes.append(
        helper.make_node(
            "Pad",
            ["c012"],
            [OUT_NAME],
            pads=[0, 0, 0, 0, 0, C - 3, H - N, W - N],
        )
    )


def _finish_bool(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    plus: str,
    square: str,
    c2: str,
    blue: str,
) -> None:
    nodes.append(helper.make_node("Not", [square], ["not_square"]))
    nodes.append(helper.make_node("And", [blue, "not_square"], ["c1b"]))
    nodes.append(helper.make_node("Or", ["c1b", plus], ["occupied"]))
    nodes.append(helper.make_node("Not", ["occupied"], ["c0b"]))
    nodes.append(helper.make_node("Cast", ["c0b"], ["c0"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Cast", ["c1b"], ["c1"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Concat", ["c0", "c1", c2], ["c012"], axis=1))
    nodes.append(
        helper.make_node(
            "Pad",
            ["c012"],
            [OUT_NAME],
            pads=[0, 0, 0, 0, 0, C - 3, H - N, W - N],
        )
    )


def _finish_arith(
    nodes: list[onnx.NodeProto],
    plus: str,
    squaref: str,
    c2: str,
    bluef: str,
    zero: str,
) -> None:
    nodes.append(helper.make_node("Sub", [bluef, squaref], ["c1"]))
    nodes.append(helper.make_node("Greater", ["c1", zero], ["c1pos"]))
    nodes.append(helper.make_node("Or", ["c1pos", plus], ["occupied"]))
    nodes.append(helper.make_node("Not", ["occupied"], ["c0b"]))
    nodes.append(helper.make_node("Cast", ["c0b"], ["c0"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Concat", ["c0", "c1", c2], ["c012"], axis=1))
    nodes.append(
        helper.make_node(
            "Pad",
            ["c012"],
            [OUT_NAME],
            pads=[0, 0, 0, 0, 0, C - 3, H - N, W - N],
        )
    )


def build_conv_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    _slice(nodes, IN_NAME, "b", _i64(inits, "s_b", [0, 1, 0, 0]), _i64(inits, "e_b", [1, 2, N, N]))

    k = np.array([[[[1, 1, 1], [1, -8, 1], [1, 1, 1]]]], dtype=np.float32)
    _f32(inits, "k", k)
    nodes.append(helper.make_node("Conv", ["b", "k"], ["score"]))
    nodes.append(helper.make_node("Greater", ["score", _f32(inits, "seven5", [7.5])], ["sq"]))
    nodes.append(helper.make_node("Cast", ["sq"], ["sqf"], to=TensorProto.FLOAT))

    plus_k = np.array([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], dtype=np.float32)
    square_k = np.array([[[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]], dtype=np.float32)
    _f32(inits, "pk", plus_k)
    _f32(inits, "sk", square_k)
    nodes.append(helper.make_node("ConvTranspose", ["sqf", "pk"], ["plusf"]))
    nodes.append(helper.make_node("ConvTranspose", ["sqf", "sk"], ["squaref"]))
    zero = _f32(inits, "zf", [0.0])
    nodes.append(helper.make_node("Greater", ["plusf", zero], ["plus"]))
    _finish_arith(nodes, "plus", "squaref", "plusf", "b", zero)

    graph = helper.make_graph(nodes, "task160_conv", [x_info], [y_info], initializer=inits)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", OPSET)])
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_conv_f16_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    _slice(nodes, IN_NAME, "b32", _i64(inits, "s_b16", [0, 1, 0, 0]), _i64(inits, "e_b16", [1, 2, N, N]))
    nodes.append(helper.make_node("Cast", ["b32"], ["b"], to=TensorProto.FLOAT16))

    k = np.array([[[[1, 1, 1], [1, -8, 1], [1, 1, 1]]]], dtype=np.float16)
    _init(inits, "k16", k)
    nodes.append(helper.make_node("Conv", ["b", "k16"], ["score"]))
    _init(inits, "seven5_16", np.asarray([7.5], dtype=np.float16))
    nodes.append(helper.make_node("Greater", ["score", "seven5_16"], ["sq"]))
    nodes.append(helper.make_node("Cast", ["sq"], ["sqf"], to=TensorProto.FLOAT16))

    plus_k = np.array([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], dtype=np.float16)
    square_k = np.array([[[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]], dtype=np.float16)
    _init(inits, "pk16", plus_k)
    _init(inits, "sk16", square_k)
    nodes.append(helper.make_node("ConvTranspose", ["sqf", "pk16"], ["plusf"]))
    nodes.append(helper.make_node("ConvTranspose", ["sqf", "sk16"], ["squaref"]))
    _init(inits, "one16", np.asarray([1.0], dtype=np.float16))
    nodes.append(helper.make_node("Sub", ["b", "squaref"], ["c1"]))
    nodes.append(helper.make_node("Sub", ["one16", "c1"], ["c0a"]))
    nodes.append(helper.make_node("Sub", ["c0a", "plusf"], ["c0"]))
    nodes.append(helper.make_node("Concat", ["c0", "c1", "plusf"], ["c012h"], axis=1))
    nodes.append(helper.make_node("Cast", ["c012h"], ["c012"], to=TensorProto.FLOAT))
    nodes.append(
        helper.make_node(
            "Pad",
            ["c012"],
            [OUT_NAME],
            pads=[0, 0, 0, 0, 0, C - 3, H - N, W - N],
        )
    )

    graph = helper.make_graph(nodes, "task160_conv_f16", [x_info], [y_info], initializer=inits)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", OPSET)])
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_conv_f16_bool_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    _slice(nodes, IN_NAME, "b32", _i64(inits, "s_b16b", [0, 1, 0, 0]), _i64(inits, "e_b16b", [1, 2, N, N]))
    nodes.append(helper.make_node("Cast", ["b32"], ["b"], to=TensorProto.FLOAT16))

    _init(inits, "zero16b", np.asarray([0.0], dtype=np.float16))
    nodes.append(helper.make_node("Greater", ["b", "zero16b"], ["bb"]))

    k = np.array([[[[1, 1, 1], [1, -8, 1], [1, 1, 1]]]], dtype=np.float16)
    _init(inits, "k16b", k)
    nodes.append(helper.make_node("Conv", ["b", "k16b"], ["score"]))
    _init(inits, "seven5_16b", np.asarray([7.5], dtype=np.float16))
    nodes.append(helper.make_node("Greater", ["score", "seven5_16b"], ["sq8"]))
    nodes.append(helper.make_node("Cast", ["sq8"], ["sqf"], to=TensorProto.FLOAT16))

    plus_k = np.array([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], dtype=np.float16)
    square_k = np.array([[[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]], dtype=np.float16)
    _init(inits, "pk16b", plus_k)
    _init(inits, "sk16b", square_k)
    nodes.append(helper.make_node("ConvTranspose", ["sqf", "pk16b"], ["plusf"]))
    nodes.append(helper.make_node("ConvTranspose", ["sqf", "sk16b"], ["squaref"]))
    nodes.append(helper.make_node("Greater", ["plusf", "zero16b"], ["plus"]))
    nodes.append(helper.make_node("Greater", ["squaref", "zero16b"], ["square"]))
    nodes.append(helper.make_node("Not", ["square"], ["not_square"]))
    nodes.append(helper.make_node("And", ["bb", "not_square"], ["c1b"]))
    nodes.append(helper.make_node("Or", ["c1b", "plus"], ["occupied"]))
    nodes.append(helper.make_node("Not", ["occupied"], ["c0b"]))
    nodes.append(helper.make_node("Concat", ["c0b", "c1b", "plus"], ["c012b"], axis=1))
    nodes.append(helper.make_node("Cast", ["c012b"], ["c012"], to=TensorProto.FLOAT))
    nodes.append(
        helper.make_node(
            "Pad",
            ["c012"],
            [OUT_NAME],
            pads=[0, 0, 0, 0, 0, C - 3, H - N, W - N],
        )
    )

    graph = helper.make_graph(nodes, "task160_conv_f16_bool", [x_info], [y_info], initializer=inits)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", OPSET)])
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def build_slice_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    _slice(nodes, IN_NAME, "b_f", _i64(inits, "s_b", [0, 1, 0, 0]), _i64(inits, "e_b", [1, 2, N, N]))
    nodes.append(helper.make_node("Greater", ["b_f", _f32(inits, "half", [0.5])], ["b"]))

    required = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
    terms: list[str] = []
    for i, (r, c) in enumerate(required):
        terms.append(
            _slice(
                nodes,
                "b",
                f"t{i}",
                _i64(inits, f"s{i}", [0, 0, r, c]),
                _i64(inits, f"e{i}", [1, 1, r + 8, c + 8]),
            )
        )
    center = _slice(nodes, "b", "center", _i64(inits, "s_c", [0, 0, 1, 1]), _i64(inits, "e_c", [1, 1, 9, 9]))
    nodes.append(helper.make_node("Not", [center], ["empty"]))
    sq = _and_chain(nodes, terms + ["empty"], "and")
    nodes.append(helper.make_node("Cast", [sq], ["sqf"], to=TensorProto.FLOAT))

    border = required
    plus_offsets = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
    squaref = _add_chain(nodes, [_pad10(nodes, "sqf", f"sb{i}", r, c) for i, (r, c) in enumerate(border)], "sa")
    plusf = _add_chain(nodes, [_pad10(nodes, "sqf", f"pb{i}", r, c) for i, (r, c) in enumerate(plus_offsets)], "pa")
    zero = _f32(inits, "zmask", [0.0])
    nodes.append(helper.make_node("Greater", [plusf, zero], ["plus"]))
    _finish_arith(nodes, "plus", squaref, plusf, "b_f", zero)

    graph = helper.make_graph(nodes, "task160_slice", [x_info], [y_info], initializer=inits)
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", OPSET)])
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def _grid_to_onehot(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _decode(onehot: np.ndarray) -> np.ndarray:
    return (onehot[0, :, :N, :N] > 0.0).argmax(axis=0).astype(np.int64)


def validate(model: onnx.ModelProto) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    data = json.loads(DATA_PATH.read_text())
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data[split]):
            got = session.run([OUT_NAME], {IN_NAME: _grid_to_onehot(ex["input"])})[0]
            expected = _grid_to_onehot(ex["output"])
            if not np.array_equal(got > 0.0, expected > 0.0):
                raise AssertionError(f"{split}[{idx}] failed")
            if not np.array_equal(_decode(got), np.asarray(ex["output"], dtype=np.int64)):
                raise AssertionError(f"{split}[{idx}] decode failed")
            if not np.array_equal(solve(ex["input"]), np.asarray(ex["output"], dtype=np.int64)):
                raise AssertionError(f"{split}[{idx}] reference rule failed")


def _write_score(model: onnx.ModelProto, name: str) -> tuple[Path, dict[str, object]]:
    tmp = Path(tempfile.mkdtemp(prefix=f"ng_{TASK_ID}_{name}_")) / f"{TASK_ID}.onnx"
    onnx.save(model, tmp)
    return tmp, score_file(tmp)


def main() -> None:
    variants: dict[str, Callable[[], onnx.ModelProto]] = {
        "slice": build_slice_model,
        "conv": build_conv_model,
        "conv_f16": build_conv_f16_model,
        "conv_f16_bool": build_conv_f16_bool_model,
    }
    results: list[tuple[str, onnx.ModelProto, dict[str, object], Path]] = []
    for name, builder in variants.items():
        model = builder()
        validate(model)
        path, report = _write_score(model, name)
        if not report["valid"]:
            raise RuntimeError(f"{name} score invalid: {report['error']}")
        results.append((name, model, report, path))

    best_name, best_model, best_report, _ = min(results, key=lambda item: int(item[2]["cost"]))
    onnx.save(best_model, BEST_PATH)

    print(f"wrote {BEST_PATH} ({best_name})")
    for name, _, report, path in results:
        print(
            f"{name:5s} path={path} memory={report['memory']} params={report['params']} "
            f"cost={report['cost']} score={float(report['score']):.6f}"
        )
    print(
        f"best  memory={best_report['memory']} params={best_report['params']} "
        f"cost={best_report['cost']} score={float(best_report['score']):.6f}"
    )


if __name__ == "__main__":
    main()
