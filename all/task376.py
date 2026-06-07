"""ONNX solution for ARC task376: extend sampled rows as a vertical bounce lattice.

Task rule: the input is a 17-column grid with k non-empty sampled rows at the
top. The output reuses those exact row patterns in a reflected cycle
``0, 1, ..., k-1, k-2, ..., 1`` and repeats that cycle through height
``4*k - 3``. Black/background cells inside the visible output remain channel-0
one-hot; cells beyond the visible output rectangle stay all-zero.

ONNX approach: slice the compact 7x17 row source, compute ``k - 3`` from the
one-hot activity in rows 3..5 at column 0, gather the matching int32
length-21 row-index vector from a small table, and use input row 6 as the
all-zero padding row for shorter outputs.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_all_onnx import verify_correctness  # noqa: E402
from score_model import print_report, score_file  # noqa: E402

TASK_ID = "task376"
DATA_PATH = ROOT / "data" / f"{TASK_ID}.json"
OUT_DIR = Path(__file__).resolve().parent
BEST_PATH = OUT_DIR / f"{TASK_ID}.onnx"
ROOT_PATH = ROOT / f"{TASK_ID}.onnx"

C = 10
H = W = 30
VISIBLE_W = 17
ZERO_ROW = 6
COMPACT_ROWS = 21
IN_NAME = "input"
OUT_NAME = "output"
SHAPE = [1, C, H, W]
OPSET = 10
IR_VERSION = 10


def _init(inits: list[onnx.TensorProto], name: str, arr: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def _slice(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    source: str,
    out: str,
    starts: list[int],
    ends: list[int],
    axes: list[int] | None = None,
) -> str:
    s = _init(inits, f"{out}_s", np.asarray(starts, dtype=np.int64))
    e = _init(inits, f"{out}_e", np.asarray(ends, dtype=np.int64))
    inputs = [source, s, e]
    if axes is not None:
        inputs.append(_init(inits, f"{out}_a", np.asarray(axes, dtype=np.int64)))
    nodes.append(helper.make_node("Slice", inputs, [out]))
    return out


def _row_exists(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    core_name: str,
    row: int,
    zero_f: str,
) -> str:
    cell = _slice(nodes, inits, core_name, f"row{row}_cell", [0, 0, row, 0], [1, C, row + 1, 1], [0, 1, 2, 3])
    nodes.append(helper.make_node("ReduceMax", [cell], [f"row{row}_max"], axes=[0, 1, 2, 3], keepdims=0))
    nodes.append(helper.make_node("Greater", [f"row{row}_max", zero_f], [f"has_row{row}"]))
    return f"has_row{row}"


def _bounce_indices(k: int) -> np.ndarray:
    cycle = list(range(k)) + list(range(k - 2, 0, -1))
    height = 4 * k - 3
    seq = [cycle[i % len(cycle)] for i in range(height)]
    seq.extend([ZERO_ROW] * (COMPACT_ROWS - len(seq)))
    return np.asarray(seq, dtype=np.int64)


def _append_index_selection(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    core_name: str,
) -> str:
    zero_f = _init(inits, "zero_f", np.asarray([0.0], dtype=np.float32))
    has3 = _row_exists(nodes, inits, core_name, 3, zero_f)
    has4 = _row_exists(nodes, inits, core_name, 4, zero_f)
    has5 = _row_exists(nodes, inits, core_name, 5, zero_f)

    idx3 = _init(inits, "idx3", _bounce_indices(3))
    idx4 = _init(inits, "idx4", _bounce_indices(4))
    idx5 = _init(inits, "idx5", _bounce_indices(5))
    idx6 = _init(inits, "idx6", _bounce_indices(6))

    nodes.append(helper.make_node("Where", [has3, idx4, idx3], ["idx34"]))
    nodes.append(helper.make_node("Where", [has4, idx5, "idx34"], ["idx345"]))
    nodes.append(helper.make_node("Where", [has5, idx6, "idx345"], ["idx"]))
    return "idx"


def _append_index_table_selection(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
) -> str:
    """Return the output-row gather indices using one compact k probe."""
    probe = _slice(nodes, inits, IN_NAME, "k_probe", [0, 0, 3, 0], [1, C, 6, 1])
    nodes.append(helper.make_node("ReduceSum", [probe], ["k_float"], axes=[0, 1, 2, 3], keepdims=0))
    nodes.append(helper.make_node("Cast", ["k_float"], ["k_idx"], to=TensorProto.INT32))

    table = _init(inits, "idx_table", np.asarray([_bounce_indices(k) for k in (3, 4, 5, 6)], dtype=np.int32))
    nodes.append(helper.make_node("Gather", [table, "k_idx"], ["idx"], axis=0))
    return "idx"


def build_index_table_float_model() -> onnx.ModelProto:
    """Gather compact float one-hot rows with a single table-selected index vector."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    core = _slice(nodes, inits, IN_NAME, "core", [0, 0, 0, 0], [1, C, ZERO_ROW + 1, VISIBLE_W])
    idx = _append_index_table_selection(nodes, inits)
    nodes.append(helper.make_node("Gather", [core, idx], ["gathered"], axis=2))
    nodes.append(
        helper.make_node(
            "Pad",
            ["gathered"],
            [OUT_NAME],
            pads=[0, 0, 0, 0, 0, 0, H - COMPACT_ROWS, W - VISIBLE_W],
        )
    )

    return _make_model(nodes, inits, "index_table_float")


def build_index_float_model() -> onnx.ModelProto:
    """Gather compact float one-hot rows, then pad directly to graph output."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    core = _slice(nodes, inits, IN_NAME, "core", [0, 0, 0, 0], [1, C, ZERO_ROW + 1, VISIBLE_W], [0, 1, 2, 3])
    idx = _append_index_selection(nodes, inits, core)
    nodes.append(helper.make_node("Gather", [core, idx], ["gathered"], axis=2))
    nodes.append(
        helper.make_node(
            "Pad",
            ["gathered"],
            [OUT_NAME],
            pads=[0, 0, 0, 0, 0, 0, H - COMPACT_ROWS, W - VISIBLE_W],
        )
    )

    return _make_model(nodes, inits, "index_float")


def build_index_full_float_model() -> onnx.ModelProto:
    """Gather selected rows directly from the full-width input, then pad height."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    zero_f = _init(inits, "zero_f", np.asarray([0.0], dtype=np.float32))
    has3 = _row_exists(nodes, inits, IN_NAME, 3, zero_f)
    has4 = _row_exists(nodes, inits, IN_NAME, 4, zero_f)
    has5 = _row_exists(nodes, inits, IN_NAME, 5, zero_f)

    idx3 = _init(inits, "idx3", _bounce_indices(3))
    idx4 = _init(inits, "idx4", _bounce_indices(4))
    idx5 = _init(inits, "idx5", _bounce_indices(5))
    idx6 = _init(inits, "idx6", _bounce_indices(6))
    nodes.append(helper.make_node("Where", [has3, idx4, idx3], ["idx34"]))
    nodes.append(helper.make_node("Where", [has4, idx5, "idx34"], ["idx345"]))
    nodes.append(helper.make_node("Where", [has5, idx6, "idx345"], ["idx"]))
    nodes.append(helper.make_node("Gather", [IN_NAME, "idx"], ["gathered"], axis=2))
    nodes.append(helper.make_node("Pad", ["gathered"], [OUT_NAME], pads=[0, 0, 0, 0, 0, 0, H - COMPACT_ROWS, 0]))

    return _make_model(nodes, inits, "index_full_float")


def build_branch_float_model() -> onnx.ModelProto:
    """Build every k-specific float candidate and select the matching one."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    core = _slice(nodes, inits, IN_NAME, "core", [0, 0, 0, 0], [1, C, ZERO_ROW + 1, VISIBLE_W], [0, 1, 2, 3])
    zero_f = _init(inits, "zero_f", np.asarray([0.0], dtype=np.float32))
    has3 = _row_exists(nodes, inits, core, 3, zero_f)
    has4 = _row_exists(nodes, inits, core, 4, zero_f)
    has5 = _row_exists(nodes, inits, core, 5, zero_f)

    candidate_names: dict[int, str] = {}
    for k in (3, 4, 5, 6):
        idx = _init(inits, f"bidx{k}", _bounce_indices(k))
        nodes.append(helper.make_node("Gather", [core, idx], [f"cand{k}"], axis=2))
        candidate_names[k] = f"cand{k}"

    nodes.append(helper.make_node("Where", [has3, candidate_names[4], candidate_names[3]], ["sel34"]))
    nodes.append(helper.make_node("Where", [has4, candidate_names[5], "sel34"], ["sel345"]))
    nodes.append(helper.make_node("Where", [has5, candidate_names[6], "sel345"], ["selected_f"]))
    nodes.append(
        helper.make_node(
            "Pad",
            ["selected_f"],
            [OUT_NAME],
            pads=[0, 0, 0, 0, 0, 0, H - COMPACT_ROWS, W - VISIBLE_W],
        )
    )

    return _make_model(nodes, inits, "branch_float")


def _make_model(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
    x_info = helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, SHAPE)
    y_info = helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, SHAPE)
    graph = helper.make_graph(nodes, f"{TASK_ID}_{name}", [x_info], [y_info], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = ""
    onnx.checker.check_model(model)
    return model


def solve_grid(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    rows = np.asarray(grid, dtype=np.uint8)
    k = rows.shape[0]
    cycle = list(range(k)) + list(range(k - 2, 0, -1))
    height = 4 * k - 3
    return np.asarray([rows[cycle[r % len(cycle)]] for r in range(height)], dtype=np.uint8)


def hypothesis_a(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    rows = [np.asarray(row, dtype=np.uint8) for row in grid if np.any(row)]
    width = len(grid[0])
    return np.asarray([rows[r % len(rows)] for r in range(width)], dtype=np.uint8)


def onehot(grid: np.ndarray | list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros(SHAPE, dtype=np.float32)
    for r in range(min(H, arr.shape[0])):
        for c in range(min(W, arr.shape[1])):
            out[0, int(arr[r, c]), r, c] = 1.0
    return out


def validate_hypotheses() -> tuple[str, str]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    total = 0
    hyp_a = 0
    bounce = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data.get(split, []):
            total += 1
            expected = np.asarray(ex["output"], dtype=np.uint8)
            try:
                hyp_a += int(np.array_equal(hypothesis_a(ex["input"]), expected))
            except ValueError:
                pass
            bounce += int(np.array_equal(solve_grid(ex["input"]), expected))
    return f"{hyp_a}/{total}", f"{bounce}/{total}"


def validate_reference() -> tuple[bool, str]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data.get(split, []):
            total += 1
            expected = np.asarray(ex["output"], dtype=np.uint8)
            actual = solve_grid(ex["input"])
            if np.array_equal(actual, expected):
                passed += 1
    return passed == total, f"{passed}/{total}"


def validate_model(model: onnx.ModelProto) -> tuple[bool, str]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        session = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        return False, f"load failed: {exc}"

    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    passed = 0
    total = 0
    for split in ("train", "test", "arc-gen"):
        for ex in data.get(split, []):
            total += 1
            expected = onehot(ex["output"]) > 0.0
            pred = session.run([OUT_NAME], {IN_NAME: onehot(ex["input"])})[0] > 0.0
            if np.array_equal(pred, expected):
                passed += 1
            else:
                return False, f"{split} case {total} failed ({passed}/{total})"
    return passed == total, f"{passed}/{total}"


def internal_tensor_count(model: onnx.ModelProto) -> int:
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    return sum(1 for node in graph.node for out in node.output if out and out != OUT_NAME)


def save_candidate(model: onnx.ModelProto, label: str, root: Path) -> Path:
    path = root / label / f"{TASK_ID}.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    return path


def main() -> None:
    hyp_a_summary, bounce_summary = validate_hypotheses()
    ref_ok, ref_summary = validate_reference()
    if not ref_ok:
        raise SystemExit(f"reference solver mismatch: {ref_summary}")

    builders = {
        "index_table_float": build_index_table_float_model,
        "index_float": build_index_float_model,
        "index_full_float": build_index_full_float_model,
        "branch_float": build_branch_float_model,
    }

    candidate_root = Path(tempfile.gettempdir()) / f"{TASK_ID}_candidates"
    rows: list[dict[str, Any]] = []
    for label, builder in builders.items():
        model = builder()
        model_ok, model_summary = validate_model(model)
        path = save_candidate(model, label, candidate_root)
        correctness_ok, correctness, _passed, _total = verify_correctness(path)
        result = score_file(path)
        rows.append(
            {
                "label": label,
                "model": model,
                "path": path,
                "model_ok": model_ok,
                "model_summary": model_summary,
                "correctness_ok": correctness_ok,
                "correctness": correctness,
                "score": result,
                "tensors": internal_tensor_count(model),
            }
        )

    valid_rows = [row for row in rows if row["model_ok"] and row["correctness_ok"] and row["score"]["valid"]]
    if not valid_rows:
        raise SystemExit("no valid task376 candidates")
    best = min(valid_rows, key=lambda row: int(row["score"]["cost"]))

    onnx.save(best["model"], BEST_PATH)
    shutil.copy2(BEST_PATH, ROOT_PATH)

    print(f"hypothesis A cyclic rows: {hyp_a_summary}")
    print(f"validated bounce rule:    {bounce_summary}")
    print(f"reference:                {ref_summary}")
    print()
    print("candidate         local     correct  tensors  memory  params  cost    score")
    print("----------------  --------  -------  -------  ------  ------  ------  --------")
    for row in rows:
        result = row["score"]
        if result["valid"]:
            print(
                f"{row['label']:<16}  {row['model_summary']:<8}  {row['correctness']:<7}  "
                f"{row['tensors']:>7}  {result['memory']:>6}  {result['params']:>6}  "
                f"{result['cost']:>6}  {result['score']:.6f}"
            )
        else:
            print(
                f"{row['label']:<16}  {row['model_summary']:<8}  {row['correctness']:<7}  "
                f"{row['tensors']:>7}  INVALID {str(result.get('error') or '').splitlines()[0]}"
            )

    print()
    print(f"best candidate: {best['label']}")
    print_report(score_file(BEST_PATH))
    print(f"candidate dir:  {candidate_root}")
    print(f"copied root model: {ROOT_PATH}")


if __name__ == "__main__":
    main()
