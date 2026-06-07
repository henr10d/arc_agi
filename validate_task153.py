from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from score_model import convert_to_numpy

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "task153.onnx"
TASK = ROOT / "data" / "task153.json"

data = json.loads(TASK.read_text(encoding="utf-8"))
session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
passed = 0
total = 0

for example in data["train"]:
    inp = convert_to_numpy(example, "input")
    expected = convert_to_numpy(example, "output")
    pred = session.run(["output"], {"input": inp})[0]
    total += 1
    passed += int(np.array_equal(pred > 0.0, expected > 0.0))

print(f"task153 train exact-match accuracy: {passed}/{total} ({passed / total:.1%})")
