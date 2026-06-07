"""Minimal ONNX for ARC task236: XOR of separated blue and red masks.

Task rule: the 9x4 input contains a 4x4 blue mask in rows 0..3, a yellow
separator row, and a 4x4 red mask in rows 5..8. The 4x4 output is green where
exactly one of the corresponding blue/red mask cells is present, and black
where both masks agree. The output is padded to the fixed 30x30 NeuroGolf
one-hot tensor shape.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
sys.path.insert(0, str(ROOT))

from score_model import score_file  # noqa: E402

TASK_ID = "task236"
BEST_PATH = OUT_DIR / "task236.onnx"
DATA_PATH = ROOT / "data" / "task236.json"
CANDIDATE_DIR = OUT_DIR / "task236_candidates"

C = 10
H = W = 30
OH = OW = 4
PAD = H - OH
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


def solve(grid: np.ndarray) -> np.ndarray:
    """Reference solver: green for blue/red XOR, black otherwise."""
    g = np.asarray(grid, dtype=np.int64)
    blue = g[:OH, :OW] == 1
    red = g[OH + 1 : OH + 1 + OH, :OW] == 2
    out = np.zeros((OH, OW), dtype=np.int64)
    out[np.logical_xor(blue, red)] = 3
    return out


def _load_data() -> dict:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _grid_to_onehot(grid: List[List[int]] | np.ndarray) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.float32)
    arr = np.asarray(grid, dtype=np.int64)
    for r in range(min(arr.shape[0], H)):
        for c in range(min(arr.shape[1], W)):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def _expected_onehot(grid: List[List[int]] | np.ndarray) -> np.ndarray:
    return _grid_to_onehot(grid)


def _candidate_masks(grid: np.ndarray) -> Dict[str, np.ndarray]:
    blue = grid[:OH, :OW] == 1
    red = grid[OH + 1 : OH + 1 + OH, :OW] == 2
    return {
        "and": blue & red,
        "or": blue | red,
        "xor": np.logical_xor(blue, red),
        "xnor": np.logical_not(np.logical_xor(blue, red)),
        "b": blue,
        "r": red,
        "not_b": np.logical_not(blue),
        "not_r": np.logical_not(red),
    }


def validate_candidate_rule(name: str, examples: List[dict]) -> bool:
    for ex in examples:
        grid = np.asarray(ex["input"], dtype=np.int64)
        expected = np.asarray(ex["output"], dtype=np.int64) == 3
        if not np.array_equal(_candidate_masks(grid)[name], expected):
            return False
    return True


def _logical_nodes(name: str, nodes: List[onnx.NodeProto]) -> str:
    if name == "and":
        nodes.append(helper.make_node("And", ["b", "r"], ["g"]))
        return "g"
    if name == "or":
        nodes.append(helper.make_node("Or", ["b", "r"], ["g"]))
        return "g"
    if name == "xor":
        nodes.append(helper.make_node("Xor", ["b", "r"], ["g"]))
        return "g"
    if name == "xnor":
        nodes.append(helper.make_node("Xor", ["b", "r"], ["x"]))
        nodes.append(helper.make_node("Not", ["x"], ["g"]))
        return "g"
    if name == "b":
        return "b"
    if name == "r":
        return "r"
    if name == "not_b":
        nodes.append(helper.make_node("Not", ["b"], ["g"]))
        return "g"
    if name == "not_r":
        nodes.append(helper.make_node("Not", ["r"], ["g"]))
        return "g"
    raise ValueError(name)


def build_model(rule: str = "xor") -> onnx.ModelProto:
    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)

    axes = _i64(inits, [1, 2, 3], "axes")
    b_st = _i64(inits, [1, 0, 0], "b_st")
    b_en = _i64(inits, [2, OH, OW], "b_en")
    r_st = _i64(inits, [2, OH + 1, 0], "r_st")
    r_en = _i64(inits, [3, OH + 1 + OH, OW], "r_en")

    nodes.extend(
        [
            helper.make_node("Slice", [IN_NAME, b_st, b_en, axes], ["bf"]),
            helper.make_node("Slice", [IN_NAME, r_st, r_en, axes], ["rf"]),
            helper.make_node("Cast", ["bf"], ["b"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["rf"], ["r"], to=TensorProto.BOOL),
        ]
    )

    green = _logical_nodes(rule, nodes)
    nodes.extend(
        [
            helper.make_node("Not", [green], ["black"]),
            helper.make_node("And", [green, "black"], ["false"]),
            helper.make_node("Concat", ["black", "false", "false", green], ["out4b"], axis=1),
            helper.make_node("Cast", ["out4b"], ["out4"], to=TensorProto.FLOAT),
            helper.make_node("Pad", ["out4"], [OUT_NAME], pads=[0, 0, 0, 0, 0, C - 4, PAD, PAD]),
        ]
    )

    graph = helper.make_graph(nodes, f"{TASK_ID}_{rule}", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def verify_model(path: Path, examples: List[dict]) -> bool:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for ex in examples:
        pred = sess.run([OUT_NAME], {IN_NAME: _grid_to_onehot(ex["input"])})[0]
        expected = _expected_onehot(ex["output"])
        if not np.array_equal(pred > 0.0, expected > 0.0):
            return False
    return True


def _score_as_task236(model_path: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="task236_score_") as tmp:
        tmp_path = Path(tmp) / "task236.onnx"
        shutil.copyfile(model_path, tmp_path)
        return score_file(tmp_path)


def main() -> None:
    data = _load_data()
    examples = [ex for split in ("train", "test", "arc-gen") for ex in data.get(split, [])]
    train = data["train"]

    CANDIDATE_DIR.mkdir(exist_ok=True)

    rows: List[Tuple[str, bool, bool, int | None, int | None, float | None, Path]] = []
    for rule in ("and", "or", "xor", "xnor", "b", "r", "not_b", "not_r"):
        model = build_model(rule)
        path = CANDIDATE_DIR / f"{TASK_ID}_{rule}.onnx"
        onnx.save(model, path)
        train_ok = validate_candidate_rule(rule, train)
        all_ok = train_ok and verify_model(path, examples)
        result = _score_as_task236(path)
        rows.append(
            (
                rule,
                train_ok,
                all_ok,
                result.get("memory"),
                result.get("params"),
                result.get("score"),
                path,
            )
        )

    correct = [row for row in rows if row[2] and row[5] is not None]
    if not correct:
        raise RuntimeError("no correct candidate model found")
    best = max(correct, key=lambda row: float(row[5]))
    shutil.copyfile(best[6], BEST_PATH)

    print("rule       train  all    memory  params  score      path")
    for rule, train_ok, all_ok, memory, params, score, path in rows:
        score_s = "INVALID" if score is None else f"{score:.6f}"
        print(f"{rule:<10} {str(train_ok):<6} {str(all_ok):<6} {str(memory):>6}  {str(params):>6}  {score_s:<10} {path}")
    print(f"best: {best[0]} -> {BEST_PATH}")


if __name__ == "__main__":
    main()
