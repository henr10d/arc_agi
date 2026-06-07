# NeuroGolf 2026 — Agent Guide

This document describes the [IJCAI-ECAI 2026 NeuroGolf Championship](https://www.kaggle.com/competitions/neurogolf-2026) rules, scoring, and optimization targets. Use it when designing or building ONNX solutions for ARC-style grid tasks in this repo.

**Source of truth:** `data/neurogolf_utils/neurogolf_utils.py` (official Kaggle utilities). Local scoring mirror: `score_model.py`.

---

## Competition goal

For each ARC task, submit a **single ONNX model** that:

1. Maps an input grid to the correct output grid on **all** examples (train, test, and arc-gen).
2. Minimizes **cost = memory + params** under the official measurement rules.

Correctness is mandatory. A model that fails any example scores **0 for that task**. Only fully correct models receive points.

---

## Task data format

Each task is a JSON file (`task001.json`, …) with three splits:

| Split | Purpose |
|-------|---------|
| `train` | Training examples shown to competitors |
| `test` | Held-out test examples (must pass at submission) |
| `arc-gen` | Generated generalization examples (must pass at submission) |

Each example has `"input"` and `"output"`: 2D integer grids where each cell is a color index **0–9**.

Grids may be **smaller than 30×30** (ragged rows allowed). Examples with any dimension **> 30** are ignored during verification.

---

## Tensor I/O contract

Every submitted model must use exactly this interface:

| Tensor | Name | Dtype | Shape |
|--------|------|-------|-------|
| Input | `input` | `float32` | `[1, 10, 30, 30]` |
| Output | `output` | `float32` | `[1, 10, 30, 30]` |

### One-hot encoding

Grids are converted to one-hot tensors along the channel axis:

- Cell color `c` → `input[0, c, row, col] = 1.0`, all other channels 0.
- Unused cells (padding beyond the grid) stay all-zero across channels.

### Decoding output

A prediction is correct when, after thresholding at 0:

```python
(user_output > 0.0) == expected_one_hot
```

**Encoding errors** (competitor-side debugging only):

- **10** — no color activated (empty one-hot)
- **11** — more than one channel active (invalid one-hot)

Always produce exactly one active channel per output cell inside the task grid.

---

## Scoring formula

Per task (only if **100% correct** on train + test + arc-gen):

```
points = max(1.0, 25.0 - log(max(1.0, memory + params)))
```

| Cost (`memory + params`) | Points |
|--------------------------|--------|
| 1 | 25.000 |
| 10 | 22.697 |
| 100 | 20.394 |
| 1,000 | 18.092 |
| 10,000 | 15.789 |
| 100,000 | 13.486 |

- **Lower cost → higher score.** The relationship is logarithmic: early savings matter more than marginal tweaks at high cost.
- **Zero-cost networks** (memory=0 and params=0) earn the full **25 points**.
- **Minimum score** for a valid correct model is **1.0** (even at extremely high cost).
- **MACs (multiply-accumulates) do NOT affect the score.**

Total competition score = sum of per-task points across all tasks.

---

## Memory measurement

Memory counts **internal activation tensor bytes only**. Input and output tensors are **excluded**.

### How memory is computed

1. Run **strict ONNX shape inference** — all shapes must be fully static (no symbolic dimensions).
2. For each internal tensor, compute `num_elements × dtype_itemsize` from inferred shapes.
3. Run the model under **ONNX Runtime profiling** on all task examples (train + test + arc-gen).
4. For each tensor, take the **maximum** byte size seen across all profiled runs (handles shape-varying intermediates).
5. **Sum** all internal tensor maxima → total memory.

### Memory optimization levers

| Strategy | Why it helps |
|----------|--------------|
| Use **bool** (`uint8`) intermediates | 1 byte/elem vs 4 for `float32` |
| Work on **small sub-regions** | e.g. 9×9 core instead of full 30×30×10 |
| **Defer Cast to float** until the final pad/reshape before `output` | Avoids large float tensors; output itself is not scored |
| **Fuse / eliminate** intermediate tensors | Fewer tensors in the sum |
| Avoid broadcasting to full grid early | `[1,10,30,30]` float tensor ≈ 36,000 bytes per tensor |
| Prefer ops that preserve compact shapes | Reshape/Slice before expensive ops |

Reference: `all/task001.py` documents bool-tensor and compact-shape patterns that cut official memory.

---

## Parameter measurement

Params = total element count across:

- All **initializers** (weights, constants in `graph.initializer`)
- **Constant** node tensor attributes (`value`, `sparse_value`)
- **Scalar** Constant attributes: `value_floats`, `value_ints`, `value_strings` — each scalar counts as **1 param**

Params do **not** include input/output tensor data (those are runtime inputs).

### Parameter optimization levers

| Strategy | Why it helps |
|----------|--------------|
| Inline small constants as **scalar** attrs when possible | 1 param each vs full tensor |
| Avoid huge weight tensors | Conv `[10,10,k,k]` scales as `100×k²` |
| Remove redundant Constant nodes | Duplicate initializers still count |
| Prefer structural ops (Slice, Pad, Where) over learned weights | Zero or few params |

---

## ONNX model constraints

Violating any rule → model is **invalid** (no score).

### Required

- **IR version:** 10
- **Opset:** 10 (`ai.onnx` domain only — no custom domains)
- **Exactly 1 input, 1 output** named `input` / `output`
- **Static positive shapes** on all tensors (no `dim_param`, no dim ≤ 0)
- **File size ≤ 1.44 MB** (1,440 KiB)

### Forbidden

| Category | Details |
|----------|---------|
| Op types | `LOOP`, `SCAN`, `NONZERO`, `UNIQUE`, `SCRIPT`, `FUNCTION`, `COMPRESS`, any op with `Sequence` in the name |
| Graph features | Custom functions, subgraph attributes, multi-domain opsets |
| Tensor types | Sequence types |
| Naming | Tensor names containing `kernel_time`; duplicate names in input/value_info/output; name collision between tensors and initializers |
| Values | Negative memory or param counts |

### Runtime settings (official measurement)

- ONNX Runtime **profiling enabled**
- Graph optimization **disabled** (`ORT_DISABLE_ALL`)
- Constant folding is applied during validation (affects param count)

---

## Correctness checklist

Before treating a model as submission-ready:

- [ ] Task script has a **top-of-file docstring** describing the input→output rule
- [ ] Passes **all** `train` examples
- [ ] Passes **all** `test` examples
- [ ] Passes **all** `arc-gen` examples
- [ ] Output is valid one-hot (exactly one channel > 0 per cell in the output grid)
- [ ] Model loads under ONNX Runtime with opset 10 / IR 10
- [ ] Passes official sanitization (see `sanitize_model` in `neurogolf_utils.py`)
- [ ] File size under 1.44 MB

Use `verify_network(model, task_num, examples)` from the official utils, or locally:

```bash
python score_model.py onnx/task001.onnx
python score_model.py all/   # score all ONNX files in a directory
```

---

## Optimization workflow for agents

When building or improving a task solution, follow this order:

### 1. Understand and document the task

Read the task JSON examples and write a clear top-of-file docstring before implementing. State the input→output rule so the file is self-explanatory.

### 2. Correctness first

Implement the transformation logic and verify on train, test, and arc-gen. No amount of cost reduction matters if any example fails.

### 3. Profile cost

```bash
python score_model.py path/to/taskNNN.onnx
```

Read `memory`, `params`, `cost`, and `score`. Identify which dominates.

### 4. Attack the dominant term

- **High memory:** shrink tensor shapes, use bool/int8 intermediates, reduce number of live tensors, avoid full-grid materialization.
- **High params:** replace weight-heavy ops with structural logic, shrink constant tensors, deduplicate initializers.

### 5. Iteration targets

Aim for meaningful cost drops because scoring is logarithmic:

| Reduce cost from → to | Point gain (approx.) |
|-----------------------|----------------------|
| 10,000 → 1,000 | +2.3 |
| 1,000 → 100 | +2.3 |
| 100 → 10 | +2.3 |

Diminishing returns appear above ~10⁴–10⁵ cost; prioritize tasks with the worst (highest) cost first when optimizing a full submission zip.

### 6. Validate again

Re-run correctness and scoring after every structural change. Shape inference failures and sanitization errors are common when refactoring.

---

## Task script conventions

Every task solution file (e.g. `all/taskNNN.py`, `tests/customNNN.py`) **must** start with a top-of-file comment describing what the task does.

Use a module docstring as the first statement in the file. It should explain:

- The **ARC transformation** in plain language (input → output rule)
- Any **non-obvious constraints** (grid size, colors used, symmetry, etc.)
- Optionally, the **ONNX approach** if it is not obvious from the rule alone

Example (`all/task001.py`):

```python
"""Minimal ONNX for ARC task001 using Kaggle one-hot I/O.

Task rule: read the top-left 3×3 core; for each non-zero cell (i, j),
tile a copy of the full core into output block (i, j) of a 9×9 grid,
then pad to 30×30 for competition I/O.
"""
```

Do not leave task files with generic placeholders like `"Task NNN solution"` — the header should let another agent understand the puzzle without opening the JSON examples.

---

## Repo tooling map

| Path | Role |
|------|------|
| `data/neurogolf_utils/neurogolf_utils.py` | Official rules, scoring, verification |
| `score_model.py` | Local official-style scorer |
| `all/taskNNN.py` | Hand-built ONNX generators per task (must include top-of-file task description) |
| `tests/` | Test scripts and reference ONNX files |
| `backend/` | Visual graph editor + ONNX export |
| `onnx/` | Exported models from the editor |

Supported ops in the local editor (subset of ONNX): `Constant`, `Cast`, `Identity`, `Equal`, `Greater`, `Less`, `Not`, `And`, `Or`, `Where`, `ReduceSum`, `ArgMax`, `Slice`, `Pad`, `Reshape`, `Concat`, `Gather`, `Unsqueeze`, `Squeeze`.

Hand-built graphs can use any **permitted** ONNX op at opset 10 — the editor list is not exhaustive.

---

## Quick reference

```
INPUT:  float32 [1, 10, 30, 30]  one-hot grid
OUTPUT: float32 [1, 10, 30, 30]  one-hot grid

CORRECTNESS: all train + test + arc-gen must match exactly (threshold > 0)

COST     = memory_bytes + param_count
POINTS   = max(1, 25 - log(max(1, cost)))     # per task, if correct
MEMORY   = sum of max internal tensor bytes across profiled runs (excl. I/O)
PARAMS   = sum(initializer elements) + Constant tensor elements + scalar constants

TARGET:   minimize cost while preserving 100% accuracy
```
