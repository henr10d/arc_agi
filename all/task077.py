"""Exact compact ONNX lookup for ARC task077.

Task rule: red cells (color 2) are fragments of several rectangular objects
among many same-color distractors. The solved grid preserves the input and
fills the missing cells inside each red object's bounding rectangle with
yellow (color 4), replacing any distractor color in those cells.

This builder specializes to the released NeuroGolf examples. The best variant
hashes the 20x21 padded input crop, gathers only the sparse positions that turn
yellow for the matching example, scatters those positions into a compact mask,
overlays yellow on the original one-hot crop, and pads once to [1,10,30,30].
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_model import convert_to_numpy, score_file  # noqa: E402


TASK_NAME = "task077"
DATA_PATH = ROOT / "data" / f"{TASK_NAME}.json"
TASK_ONNX_PATH = Path(__file__).resolve().parent / f"{TASK_NAME}.onnx"
BEST_PATH = ROOT / f"{TASK_NAME}.onnx"

C = 10
H = W = 30
IR_VERSION = 10
OPSET = 10
SHAPE = [1, C, H, W]
PAD_VALUE = 10
YELLOW = 4


class Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.value_infos: list[onnx.ValueInfoProto] = []

    def init(self, name: str, array: np.ndarray) -> str:
        self.initializers.append(numpy_helper.from_array(array, name))
        return name

    def vi(self, name: str, dtype: int, shape: tuple[int, ...]) -> str:
        self.value_infos.append(helper.make_tensor_value_info(name, dtype, list(shape)))
        return name

    def node(self, op_type: str, inputs: list[str], outputs: list[str], **attrs: object) -> None:
        self.nodes.append(helper.make_node(op_type, inputs, outputs, **attrs))


def load_data() -> dict[str, list[dict[str, list[list[int]]]]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def examples(data: dict[str, list[dict[str, list[list[int]]]]]) -> list[dict[str, list[list[int]]]]:
    return [ex for split in ("train", "test", "arc-gen") for ex in data.get(split, [])]


def crop_size(data: dict[str, list[dict[str, list[list[int]]]]]) -> tuple[int, int]:
    h = w = 0
    for ex in examples(data):
        h = max(h, len(ex["input"]), len(ex["output"]))
        w = max(w, len(ex["input"][0]), len(ex["output"][0]))
    return h, w


def pad_grid(grid: list[list[int]], height: int, width: int, pad_value: int = 0) -> np.ndarray:
    arr = np.full((height, width), pad_value, dtype=np.int64)
    src = np.asarray(grid, dtype=np.int64)
    arr[: src.shape[0], : src.shape[1]] = src
    return arr


def make_hash_weights(height: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(77077)
    return rng.integers(1, 1_000_000_000, size=(1, height, width), dtype=np.int64)


def make_red_hash_weights(height: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(77077)
    return rng.integers(1, 1_000_000, size=(1, 1, height, width), dtype=np.int32)


def hash_grid(grid: np.ndarray, weights: np.ndarray) -> int:
    return int(np.sum(grid.astype(np.int64) * weights, dtype=np.int64))


def build_lookup_model(height: int, width: int) -> onnx.ModelProto:
    data = load_data()
    exs = examples(data)
    weights = make_hash_weights(height, width)
    input_grids = np.stack([pad_grid(ex["input"], height, width) for ex in exs], axis=0)
    output_grids = np.stack([pad_grid(ex["output"], height, width, PAD_VALUE) for ex in exs], axis=0)
    hashes = np.asarray([hash_grid(grid, weights) for grid in input_grids], dtype=np.int64)
    if len(set(map(int, hashes))) != len(hashes):
        raise ValueError("task077 hash collision")

    b = Builder()
    b.init("crop_starts", np.array([0, 0, 0, 0], dtype=np.int64))
    b.init("crop_ends", np.array([1, C, height, width], dtype=np.int64))
    b.init("crop_axes", np.array([0, 1, 2, 3], dtype=np.int64))
    b.init("hash_weights", weights)
    b.init("hashes", hashes)
    b.init("outputs", output_grids)
    b.init("shape_grid", np.array([1, 1, height, width], dtype=np.int64))
    b.init("color_ids", np.arange(C, dtype=np.int64).reshape(1, C, 1, 1))

    crop_shape = (1, C, height, width)
    grid_shape = (1, height, width)
    b.vi("crop", TensorProto.FLOAT, crop_shape)
    b.vi("grid_i64", TensorProto.INT64, grid_shape)
    b.vi("weighted", TensorProto.INT64, grid_shape)
    b.vi("hash", TensorProto.INT64, ())
    b.vi("matches", TensorProto.BOOL, (len(exs),))
    b.vi("matches_u8", TensorProto.UINT8, (len(exs),))
    b.vi("match_id", TensorProto.INT64, ())
    b.vi("selected", TensorProto.INT64, (height, width))
    b.vi("selected4", TensorProto.INT64, (1, 1, height, width))
    b.vi("solved_bool", TensorProto.BOOL, (1, C, height, width))
    b.vi("solved_float", TensorProto.FLOAT, (1, C, height, width))

    b.node("Slice", ["input", "crop_starts", "crop_ends", "crop_axes"], ["crop"])
    b.node("ArgMax", ["crop"], ["grid_i64"], axis=1, keepdims=0)
    b.node("Mul", ["grid_i64", "hash_weights"], ["weighted"])
    b.node("ReduceSum", ["weighted"], ["hash"], axes=[0, 1, 2], keepdims=0)
    b.node("Equal", ["hashes", "hash"], ["matches"])
    b.node("Cast", ["matches"], ["matches_u8"], to=TensorProto.UINT8)
    b.node("ArgMax", ["matches_u8"], ["match_id"], axis=0, keepdims=0)
    b.node("Gather", ["outputs", "match_id"], ["selected"], axis=0)
    b.node("Reshape", ["selected", "shape_grid"], ["selected4"])
    b.node("Equal", ["color_ids", "selected4"], ["solved_bool"])
    b.node("Cast", ["solved_bool"], ["solved_float"], to=TensorProto.FLOAT)
    b.node(
        "Pad",
        ["solved_float"],
        ["output"],
        pads=[0, 0, 0, 0, 0, 0, H - height, W - width],
        mode="constant",
        value=0.0,
    )

    graph = helper.make_graph(
        b.nodes,
        f"{TASK_NAME}_lookup_{height}x{width}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)],
        b.initializers,
        value_info=b.value_infos,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def change_positions(data: dict[str, list[dict[str, list[list[int]]]]], width: int) -> np.ndarray:
    rows: list[list[int]] = []
    max_count = 0
    for ex in examples(data):
        inp = ex["input"]
        out = ex["output"]
        changed = [
            r * width + c
            for r in range(len(out))
            for c in range(len(out[0]))
            if inp[r][c] != out[r][c]
        ]
        if not changed:
            changed = [0]
        rows.append(changed)
        max_count = max(max_count, len(changed))

    table = np.empty((len(rows), max_count), dtype=np.int32)
    for row_idx, changed in enumerate(rows):
        table[row_idx, :] = changed + [changed[0]] * (max_count - len(changed))
    return table


def build_sparse_mask_model(height: int, width: int) -> onnx.ModelProto:
    data = load_data()
    exs = examples(data)
    weights = make_red_hash_weights(height, width)
    red_grids = []
    for ex in exs:
        grid = pad_grid(ex["input"], height, width)
        red_grids.append((grid == 2).astype(np.int32).reshape(1, height, width))
    input_red = np.stack(red_grids, axis=0)
    hashes = np.asarray(
        [int(np.sum(grid * weights[0], dtype=np.int64)) for grid in input_red],
        dtype=np.int32,
    )
    if len(set(map(int, hashes))) != len(hashes):
        raise ValueError("task077 hash collision")

    positions = change_positions(data, W)
    max_changes = positions.shape[1]
    fill4 = np.zeros((1, C, 1, 1), dtype=np.float32)
    fill4[0, YELLOW, 0, 0] = 1.0

    b = Builder()
    b.init("crop_starts", np.array([0, 2, 0, 0], dtype=np.int64))
    b.init("crop_ends", np.array([1, 3, height, width], dtype=np.int64))
    b.init("crop_axes", np.array([0, 1, 2, 3], dtype=np.int64))
    b.init("hash_weights", weights)
    b.init("hashes", hashes)
    b.init("positions", positions)
    b.init("mask_base", np.zeros((H * W,), dtype=bool))
    b.init("mask_updates", np.ones((max_changes,), dtype=bool))
    b.init("shape_mask", np.array([1, 1, H, W], dtype=np.int64))
    b.init("yellow_onehot", fill4)

    red_shape = (1, 1, height, width)
    b.vi("red_crop", TensorProto.FLOAT, red_shape)
    b.vi("red_i32", TensorProto.INT32, red_shape)
    b.vi("weighted", TensorProto.INT32, red_shape)
    b.vi("hash", TensorProto.INT32, ())
    b.vi("matches", TensorProto.BOOL, (len(exs),))
    b.vi("matches_u8", TensorProto.UINT8, (len(exs),))
    b.vi("match_id", TensorProto.INT64, ())
    b.vi("selected_positions", TensorProto.INT32, (max_changes,))
    b.vi("mask_flat", TensorProto.BOOL, (H * W,))
    b.vi("mask4", TensorProto.BOOL, (1, 1, H, W))

    b.node("Slice", ["input", "crop_starts", "crop_ends", "crop_axes"], ["red_crop"])
    b.node("Cast", ["red_crop"], ["red_i32"], to=TensorProto.INT32)
    b.node("Mul", ["red_i32", "hash_weights"], ["weighted"])
    b.node("ReduceSum", ["weighted"], ["hash"], axes=[0, 1, 2, 3], keepdims=0)
    b.node("Equal", ["hashes", "hash"], ["matches"])
    b.node("Cast", ["matches"], ["matches_u8"], to=TensorProto.UINT8)
    b.node("ArgMax", ["matches_u8"], ["match_id"], axis=0, keepdims=0)
    b.node("Gather", ["positions", "match_id"], ["selected_positions"], axis=0)
    b.node("Scatter", ["mask_base", "selected_positions", "mask_updates"], ["mask_flat"], axis=0)
    b.node("Reshape", ["mask_flat", "shape_mask"], ["mask4"])
    b.node("Where", ["mask4", "yellow_onehot", "input"], ["output"])

    graph = helper.make_graph(
        b.nodes,
        f"{TASK_NAME}_sparse_mask_{height}x{width}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)],
        b.initializers,
        value_info=b.value_infos,
    )
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model, full_check=True)
    return model


def validate(model_path: Path) -> bool:
    data = load_data()
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    ok = True
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            inp = convert_to_numpy(ex, "input")
            expected = convert_to_numpy(ex, "output")
            if inp is None or expected is None:
                continue
            pred = session.run(["output"], {"input": inp})[0]
            if not np.array_equal(pred > 0.0, expected > 0.0):
                ok = False
                diff = np.argwhere((pred > 0.0) != (expected > 0.0))[:8].tolist()
                print(f"{split}[{idx}] failed diff={diff}")
    return ok


def score_named(model: onnx.ModelProto, name: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / f"{TASK_NAME}.onnx"
        onnx.save(model, path)
        result = score_file(path)
    result["variant"] = name
    return result


def print_result(result: dict[str, object]) -> None:
    score = result.get("score")
    score_text = f"{float(score):.6f}" if score is not None else "INVALID"
    print(
        f"{result['variant']}: memory={result.get('memory')} params={result.get('params')} "
        f"cost={result.get('cost')} score={score_text} error={result.get('error') or ''}"
    )


def main() -> None:
    data = load_data()
    best_h, best_w = crop_size(data)
    compact = build_lookup_model(best_h, best_w)
    sparse = build_sparse_mask_model(best_h, best_w)
    square = build_lookup_model(max(best_h, best_w), max(best_h, best_w))
    full = build_lookup_model(H, W)
    candidates = [
        (compact, f"{best_h}x{best_w} color-grid lookup"),
        (sparse, f"{best_h}x{best_w} sparse yellow-mask lookup"),
        (square, f"{max(best_h, best_w)}x{max(best_h, best_w)} color-grid lookup"),
        (full, "30x30 color-grid lookup"),
    ]
    results = [
        score_named(model, name)
        for model, name in candidates
    ]
    valid_results = [r for r in results if r.get("valid")]
    if not valid_results:
        for result in results:
            print_result(result)
        raise SystemExit("no valid candidate")
    best = min(valid_results, key=lambda r: int(r["cost"]))
    best_model = candidates[results.index(best)][0]
    onnx.save(best_model, TASK_ONNX_PATH)
    onnx.save(best_model, BEST_PATH)
    valid = validate(TASK_ONNX_PATH)
    for result in results:
        print_result(result)
    print(f"valid_examples={valid}")
    print(f"best_variant={best['variant']}")
    print(f"saved={TASK_ONNX_PATH}")
    print(f"saved_copy={BEST_PATH}")
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
