"""ARC task055: fill regions between cyan hashtag separators with fixed colors.

Task rule: input is black (0) plus cyan separator lines (8) forming a 2×2 grid.
Output keeps all cyan lines and fills the five interior cross regions:
above→2, center→6, left→4, right→3, below→1; corners stay black.

ONNX: detect two full horizontal and two full vertical separator lines, paint
the fixed 3×3 quadrant palette [[0,2,0],[4,6,3],[0,1,0]], preserve ch8.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402

BEST_PATH = OUT_DIR / "task055.onnx"
DATA_PATH = ROOT / "data" / "task055.json"

C = 10
H = W = 30
FRAME = 8
PATTERN = np.array([[0, 2, 0], [4, 6, 3], [0, 1, 0]], dtype=np.int64)
IN_NAME = "input"
OUT_NAME = "output"
OPSET = 10
IR_VERSION = 10

TOY_INPUT = [
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 8, 1, 1, 1, 1, 1, 1, 8, 0, 0, 0, 0],
    [0, 0, 8, 1, 1, 1, 1, 1, 1, 8, 0, 0, 0, 0],
    [0, 0, 8, 1, 1, 1, 1, 1, 1, 8, 0, 0, 0, 0],
    [0, 0, 8, 1, 1, 1, 1, 1, 1, 8, 0, 0, 0, 0],
]
TOY_OUTPUT = [
    [0, 0, 8, 2, 2, 2, 2, 2, 2, 8, 0, 0, 0, 0],
    [0, 0, 8, 2, 2, 2, 2, 2, 2, 8, 0, 0, 0, 0],
    [0, 0, 8, 2, 2, 2, 2, 2, 2, 8, 0, 0, 0, 0],
    [0, 0, 8, 2, 2, 2, 2, 2, 2, 8, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [4, 4, 8, 6, 6, 6, 6, 6, 6, 8, 3, 3, 3, 3],
    [4, 4, 8, 6, 6, 6, 6, 6, 6, 8, 3, 3, 3, 3],
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 8, 1, 1, 1, 1, 1, 1, 8, 0, 0, 0, 0],
    [0, 0, 8, 1, 1, 1, 1, 1, 1, 8, 0, 0, 0, 0],
    [0, 0, 8, 1, 1, 1, 1, 1, 1, 8, 0, 0, 0, 0],
    [0, 0, 8, 1, 1, 1, 1, 1, 1, 8, 0, 0, 0, 0],
]


def _i64(inits: List[onnx.TensorProto], vals, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name))
    return name


def _f32(inits: List[onnx.TensorProto], arr, name: str) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name))
    return name


def build_reference_numpy(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    if g.ndim == 4:
        g = g[0].argmax(axis=0)
    elif g.ndim == 3:
        g = g.argmax(axis=0)
    h, w = g.shape
    out = g.copy()
    hlines = sorted(r for r in range(h) if np.all(g[r, :] == FRAME))
    vlines = sorted(c for c in range(w) if np.all(g[:, c] == FRAME))
    hr0, hr1 = hlines
    vc0, vc1 = vlines
    row_ranges = [(0, hr0), (hr0 + 1, hr1), (hr1 + 1, h)]
    col_ranges = [(0, vc0), (vc0 + 1, vc1), (vc1 + 1, w)]
    for ri, (r0, r1) in enumerate(row_ranges):
        for ci, (c0, c1) in enumerate(col_ranges):
            color = int(PATTERN[ri, ci])
            if color:
                out[r0:r1, c0:c1] = color
    return out


def _grid_to_onehot(grid: List[List[int]]) -> np.ndarray:
    out = np.zeros((1, C, H, W), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            color = int(val)
            if 0 <= color < C:
                out[0, color, r, c] = 1.0
    return out


def _onehot_to_grid(onehot: np.ndarray) -> np.ndarray:
    return onehot.reshape(C, H, W).argmax(axis=0).astype(np.int64)


def build_onnx_model(*, compact_coords: bool = False, mask_padding: bool = True) -> onnx.ModelProto:
    """Detect two full rows/cols of color 8, paint fixed 3×3 quadrant colors."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    shape = [1, C, H, W]
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, shape)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, shape)

    o = _f32(inits, 1.0, "o")
    frame = _i64(inits, FRAME, "frame")
    zi = _i64(inits, 0, "zi")
    big = _f32(inits, 999.0, "big")
    neg = _f32(inits, -1.0, "neg")
    ax = _i64(inits, [0, 1, 2, 3], "ax")
    idx = _f32(inits, np.arange(W, dtype=np.float32), "idx")
    rows = np.arange(H, dtype=np.float32).reshape(1, H, 1)
    cols = np.arange(W, dtype=np.float32).reshape(1, 1, W)
    _f32(inits, rows, "Y1")
    _f32(inits, cols, "X1")
    if compact_coords:
        y_band, x_band = "Y1", "X1"
    else:
        y_band = "Y3"
        x_band = "X3"
        _f32(inits, np.broadcast_to(rows, (1, H, W)), y_band)
        _f32(inits, np.broadcast_to(cols, (1, H, W)), x_band)
    zz = _f32(inits, np.zeros((1, 1, H, W), dtype=np.float32), "zz")

    sa8 = _i64(inits, [0, FRAME, 0, 0], "sa8")
    ea8 = _i64(inits, [1, FRAME + 1, H, W], "ea8")

    nodes.extend(
        [
            helper.make_node("ArgMax", [IN_NAME], ["g"], axis=1, keepdims=0),
            helper.make_node("Greater", ["g", zi], ["nz"]),
            helper.make_node("Cast", ["nz"], ["m"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMax", ["m"], ["row_any"], axes=[2], keepdims=0),
            helper.make_node("ReduceMax", ["m"], ["col_any"], axes=[1], keepdims=0),
            helper.make_node("Mul", ["row_any", idx], ["row_w"]),
            helper.make_node("ReduceMax", ["row_w"], ["hh"], axes=[1], keepdims=0),
            helper.make_node("Add", ["hh", o], ["hh1"]),
            helper.make_node("Mul", ["col_any", idx], ["col_w"]),
            helper.make_node("ReduceMax", ["col_w"], ["ww"], axes=[1], keepdims=0),
            helper.make_node("Add", ["ww", o], ["ww1"]),
            helper.make_node("Equal", ["g", frame], ["eq8"]),
            helper.make_node("Cast", ["eq8"], ["e8"], to=TensorProto.FLOAT),
            helper.make_node("Less", ["X1", "ww1"], ["cm1"]),
            helper.make_node("Cast", ["cm1"], ["cm"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["e8", "cm"], ["e8c"]),
            helper.make_node("ReduceSum", ["e8c"], ["rc"], axes=[2], keepdims=0),
            helper.make_node("Cast", ["rc"], ["rci"], to=TensorProto.INT64),
            helper.make_node("Cast", ["ww1"], ["wwi"], to=TensorProto.INT64),
            helper.make_node("Equal", ["rci", "wwi"], ["hl"]),
            helper.make_node("Less", ["Y1", "hh1"], ["rm1"]),
            helper.make_node("Cast", ["rm1"], ["rm"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["e8", "rm"], ["e8r"]),
            helper.make_node("ReduceSum", ["e8r"], ["cc"], axes=[1], keepdims=0),
            helper.make_node("Cast", ["cc"], ["cci"], to=TensorProto.INT64),
            helper.make_node("Cast", ["hh1"], ["hhi"], to=TensorProto.INT64),
            helper.make_node("Equal", ["cci", "hhi"], ["vl"]),
            helper.make_node("Cast", ["hl"], ["hlf"], to=TensorProto.FLOAT),
            helper.make_node("Where", ["hl", idx, big], ["hr_pick"]),
            helper.make_node("ReduceMin", ["hr_pick"], ["hr0"], axes=[1], keepdims=0),
            helper.make_node("Where", ["hl", idx, neg], ["hr_pick2"]),
            helper.make_node("ReduceMax", ["hr_pick2"], ["hr1"], axes=[1], keepdims=0),
            helper.make_node("Cast", ["vl"], ["vlf"], to=TensorProto.FLOAT),
            helper.make_node("Where", ["vl", idx, big], ["vc_pick"]),
            helper.make_node("ReduceMin", ["vc_pick"], ["vc0"], axes=[1], keepdims=0),
            helper.make_node("Where", ["vl", idx, neg], ["vc_pick2"]),
            helper.make_node("ReduceMax", ["vc_pick2"], ["vc1"], axes=[1], keepdims=0),
            helper.make_node("Less", [y_band, "hr0"], ["br0"]),
            helper.make_node("Greater", [y_band, "hr0"], ["gt0"]),
            helper.make_node("Less", [y_band, "hr1"], ["lt1"]),
            helper.make_node("And", ["gt0", "lt1"], ["br1"]),
            helper.make_node("Greater", [y_band, "hr1"], ["gt1"]),
            helper.make_node("Less", [y_band, "hh1"], ["lt2"]),
            helper.make_node("And", ["gt1", "lt2"], ["br2"]),
            helper.make_node("Less", [x_band, "vc0"], ["bc0"]),
            helper.make_node("Greater", [x_band, "vc0"], ["gc0"]),
            helper.make_node("Less", [x_band, "vc1"], ["lc1"]),
            helper.make_node("And", ["gc0", "lc1"], ["bc1"]),
            helper.make_node("Greater", [x_band, "vc1"], ["gc1"]),
            helper.make_node("Less", [x_band, "ww1"], ["lc2"]),
            helper.make_node("And", ["gc1", "lc2"], ["bc2"]),
            helper.make_node("Slice", [IN_NAME, sa8, ea8, ax], ["ch8"]),
        ]
    )

    band_r = ["br0", "br1", "br2"]
    band_c = ["bc0", "bc1", "bc2"]
    planes: List[str] = ["zz"] * C
    planes[FRAME] = "ch8"

    for ri in range(3):
        for ci in range(3):
            color = int(PATTERN[ri, ci])
            if not color:
                continue
            name = f"p{color}"
            nodes.append(helper.make_node("And", [band_r[ri], band_c[ci]], [name]))
            nodes.append(helper.make_node("Cast", [name], [f"{name}f"], to=TensorProto.FLOAT))
            plane4 = f"{name}4"
            nodes.append(helper.make_node("Unsqueeze", [f"{name}f"], [plane4], axes=[1]))
            if planes[color] == "zz":
                planes[color] = plane4
            else:
                merged = f"m{color}"
                nodes.append(helper.make_node("Max", [planes[color], plane4], [merged]))
                planes[color] = merged

    used = planes[1]
    for c in range(2, C):
        if planes[c] == "zz":
            continue
        nxt = f"u{c}"
        nodes.append(helper.make_node("Add", [used, planes[c]], [nxt]))
        used = nxt
    nodes.append(helper.make_node("Sub", [o, used], ["bg0"]))
    nodes.append(helper.make_node("Max", ["bg0", zz], ["bg_raw"]))
    if mask_padding:
        nodes.append(helper.make_node("Less", ["Y1", "hh1"], ["ar1"]))
        nodes.append(helper.make_node("Less", ["X1", "ww1"], ["ac1"]))
        nodes.append(helper.make_node("And", ["ar1", "ac1"], ["active2"]))
        nodes.append(helper.make_node("Cast", ["active2"], ["activef"], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Unsqueeze", ["activef"], ["active4"], axes=[1]))
        nodes.append(helper.make_node("Mul", ["bg_raw", "active4"], ["bg"]))
    else:
        nodes.append(helper.make_node("Identity", ["bg_raw"], ["bg"]))
    planes[0] = "bg"
    nodes.append(helper.make_node("Concat", planes, [OUT_NAME], axis=1))

    graph = helper.make_graph(nodes, "task055", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return model


def build_bool_onnx_model() -> onnx.ModelProto:
    """Channel-8-only detector with boolean output planes until final Cast."""
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    shape = [1, C, H, W]
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, shape)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, shape)

    o = _f32(inits, 1.0, "o")
    half = _f32(inits, 0.5, "half")
    big = _f32(inits, 999.0, "big")
    neg = _f32(inits, -1.0, "neg")
    idx = _f32(inits, np.arange(W, dtype=np.float32).reshape(1, 1, W), "idx")
    _f32(inits, np.arange(H, dtype=np.float32).reshape(1, H, 1), "Y1")
    _f32(inits, np.arange(W, dtype=np.float32).reshape(1, 1, W), "X1")

    ax = _i64(inits, [0, 1, 2, 3], "ax")
    sa8 = _i64(inits, [0, FRAME, 0, 0], "sa8")
    ea8 = _i64(inits, [1, FRAME + 1, H, W], "ea8")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, sa8, ea8, ax], ["ch8"]),
            helper.make_node("ReduceMax", ["ch8"], ["row_any"], axes=[3], keepdims=0),
            helper.make_node("ReduceMax", ["ch8"], ["col_any"], axes=[2], keepdims=0),
            helper.make_node("Mul", ["row_any", idx], ["row_w"]),
            helper.make_node("ReduceMax", ["row_w"], ["hh"], axes=[2], keepdims=0),
            helper.make_node("Add", ["hh", o], ["hh1"]),
            helper.make_node("Mul", ["col_any", idx], ["col_w"]),
            helper.make_node("ReduceMax", ["col_w"], ["ww"], axes=[2], keepdims=0),
            helper.make_node("Add", ["ww", o], ["ww1"]),
            helper.make_node("ReduceSum", ["ch8"], ["rc"], axes=[3], keepdims=0),
            helper.make_node("Sub", ["ww1", half], ["wwm"]),
            helper.make_node("Greater", ["rc", "wwm"], ["hl"]),
            helper.make_node("ReduceSum", ["ch8"], ["cc"], axes=[2], keepdims=0),
            helper.make_node("Sub", ["hh1", half], ["hhm"]),
            helper.make_node("Greater", ["cc", "hhm"], ["vl"]),
            helper.make_node("Where", ["hl", idx, big], ["hr_pick"]),
            helper.make_node("ReduceMin", ["hr_pick"], ["hr0"], axes=[2], keepdims=0),
            helper.make_node("Where", ["hl", idx, neg], ["hr_pick2"]),
            helper.make_node("ReduceMax", ["hr_pick2"], ["hr1"], axes=[2], keepdims=0),
            helper.make_node("Where", ["vl", idx, big], ["vc_pick"]),
            helper.make_node("ReduceMin", ["vc_pick"], ["vc0"], axes=[2], keepdims=0),
            helper.make_node("Where", ["vl", idx, neg], ["vc_pick2"]),
            helper.make_node("ReduceMax", ["vc_pick2"], ["vc1"], axes=[2], keepdims=0),
            helper.make_node("Less", ["Y1", "hr0"], ["br0"]),
            helper.make_node("Greater", ["Y1", "hr0"], ["gt0"]),
            helper.make_node("Less", ["Y1", "hr1"], ["lt1"]),
            helper.make_node("And", ["gt0", "lt1"], ["br1"]),
            helper.make_node("Greater", ["Y1", "hr1"], ["gt1"]),
            helper.make_node("Less", ["Y1", "hh1"], ["lt2"]),
            helper.make_node("And", ["gt1", "lt2"], ["br2"]),
            helper.make_node("Less", ["X1", "vc0"], ["bc0"]),
            helper.make_node("Greater", ["X1", "vc0"], ["gc0"]),
            helper.make_node("Less", ["X1", "vc1"], ["lc1"]),
            helper.make_node("And", ["gc0", "lc1"], ["bc1"]),
            helper.make_node("Greater", ["X1", "vc1"], ["gc1"]),
            helper.make_node("Less", ["X1", "ww1"], ["lc2"]),
            helper.make_node("And", ["gc1", "lc2"], ["bc2"]),
            helper.make_node("Unsqueeze", ["br0"], ["br0u"], axes=[1]),
            helper.make_node("Unsqueeze", ["br1"], ["br1u"], axes=[1]),
            helper.make_node("Unsqueeze", ["br2"], ["br2u"], axes=[1]),
            helper.make_node("Unsqueeze", ["bc0"], ["bc0u"], axes=[1]),
            helper.make_node("Unsqueeze", ["bc1"], ["bc1u"], axes=[1]),
            helper.make_node("Unsqueeze", ["bc2"], ["bc2u"], axes=[1]),
        ]
    )

    band_r = ["br0u", "br1u", "br2u"]
    band_c = ["bc0u", "bc1u", "bc2u"]
    planes: List[str | None] = [None] * C

    for ri in range(3):
        for ci in range(3):
            color = int(PATTERN[ri, ci])
            if not color:
                continue
            p = f"p{color}"
            nodes.append(helper.make_node("And", [band_r[ri], band_c[ci]], [p]))
            planes[color] = p

    nodes.extend(
        [
            helper.make_node("Cast", ["ch8"], ["ch8b"], to=TensorProto.BOOL),
            helper.make_node("Greater", ["ch8", o], ["false4"]),
        ]
    )
    planes[FRAME] = "ch8b"

    nodes.extend(
        [
            helper.make_node("And", ["br0u", "bc0u"], ["corner0"]),
            helper.make_node("And", ["br0u", "bc2u"], ["corner1"]),
            helper.make_node("And", ["br2u", "bc0u"], ["corner2"]),
            helper.make_node("And", ["br2u", "bc2u"], ["corner3"]),
            helper.make_node("Or", ["corner0", "corner1"], ["corners01"]),
            helper.make_node("Or", ["corner2", "corner3"], ["corners23"]),
            helper.make_node("Or", ["corners01", "corners23"], ["bg"]),
        ]
    )
    planes[0] = "bg"

    for color, plane in enumerate(planes):
        if plane is None:
            planes[color] = "false4"

    nodes.append(helper.make_node("Concat", [p for p in planes if p is not None], ["yb"], axis=1))
    nodes.append(helper.make_node("Cast", ["yb"], [OUT_NAME], to=TensorProto.FLOAT))

    graph = helper.make_graph(nodes, "task055_bool", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    onnx.checker.check_model(model)
    return model


BUILDERS: List[Tuple[str, Callable[[], onnx.ModelProto]]] = [
    ("bool", build_bool_onnx_model),
    ("compact", lambda: build_onnx_model(compact_coords=True, mask_padding=True)),
]


def _run_onnx(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    return sess.run([OUT_NAME], {IN_NAME: x.astype(np.float32)})[0]


def verify_model(model: onnx.ModelProto) -> Tuple[bool, str]:
    inp = np.array(TOY_INPUT, dtype=np.int64)
    exp = np.array(TOY_OUTPUT, dtype=np.int64)
    if not np.array_equal(build_reference_numpy(inp), exp):
        return False, "reference solver mismatch on toy"

    toy_in = _grid_to_onehot(TOY_INPUT)
    pred = _onehot_to_grid(_run_onnx(model, toy_in)[0])[: inp.shape[0], : inp.shape[1]]
    if not np.array_equal(pred, exp):
        return False, "toy example FAIL"

    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for split in ("train", "test", "arc-gen"):
        for i, ex in enumerate(data[split]):
            bench_in = convert_to_numpy(ex, "input")
            bench_out = convert_to_numpy(ex, "output")
            if bench_in is None or bench_out is None:
                continue
            user = _run_onnx(model, bench_in)
            if not np.array_equal(user, bench_out):
                return False, f"{split}[{i}] FAIL"
    return True, "PASS"


def pick_best() -> Tuple[onnx.ModelProto, str, dict]:
    best_model: onnx.ModelProto | None = None
    best_name = ""
    best_result: dict = {"cost": None}

    for name, builder in BUILDERS:
        try:
            model = builder()
        except Exception as exc:
            print(f"  {name}: build error {exc}")
            continue

        ok, msg = verify_model(model)
        if not ok:
            print(f"  {name}: verify FAIL ({msg})")
            continue

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            onnx.save(model, str(tmp_path))
        result = score_file(tmp_path)
        tmp_path.unlink(missing_ok=True)

        print(
            f"  {name}: mem={result.get('memory')} params={result.get('params')} "
            f"cost={result.get('cost')} score={result.get('score')}"
        )
        cost = result.get("cost")
        if result.get("valid") and cost is not None and (
            best_result.get("cost") is None or cost < best_result["cost"]
        ):
            best_model = model
            best_name = name
            best_result = result

    if best_model is None:
        raise RuntimeError("no valid variant found")
    print(
        f"\nBest: {best_name}  memory={best_result.get('memory')}  "
        f"params={best_result.get('params')}  cost={best_result.get('cost')}  "
        f"score={best_result.get('score')}"
    )
    return best_model, best_name, best_result


def save_model(path: Path = BEST_PATH) -> onnx.ModelProto:
    model, _, _ = pick_best()
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return model


def main() -> None:
    print("Variant benchmark:")
    model = save_model()
    ok, msg = verify_model(model)
    print(f"verify: {msg}")
    if not ok:
        raise SystemExit(1)

    result = score_file(BEST_PATH)
    print(
        f"saved {BEST_PATH} ({BEST_PATH.stat().st_size} bytes)\n"
        f"memory={result.get('memory')} params={result.get('params')} "
        f"cost={result.get('cost')} score={result.get('score')}"
    )


if __name__ == "__main__":
    main()
