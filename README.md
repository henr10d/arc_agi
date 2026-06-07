# NeuroGolf Lab

Local visual editor for creating tiny ONNX-style computation graphs for ARC/NeuroGolf grid tasks.

## Layout

- `backend/` FastAPI API, NumPy graph runner, ONNX compiler/exporter.
- `frontend/` React + TypeScript + Vite app using React Flow.
- `data/` local ARC-style task JSON files named `task001.json`, `task002.json`, etc.
- `projects/` saved graph projects. Includes `projects/identity.json`.
- `onnx/` exported ONNX models from the app.

## Backend

```bash
cd /home/filip/Desktop/neuro_golf
python -m pip install -r backend/requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

For backend hot reload, restrict the watched directory so Uvicorn does not recursively watch large task/model folders:

```bash
uvicorn backend.main:app --reload --reload-dir backend --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /tasks`
- `GET /tasks/{task_id}`
- `GET /projects`
- `GET /projects/{name}`
- `POST /projects/{name}`
- `POST /run`
- `POST /compile`
- `POST /export`

## Frontend

```bash
cd /home/filip/Desktop/neuro_golf/frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

`npm run dev` uses polling to avoid Linux `EMFILE: too many open files` watcher failures in this large repo. If your system watch limits are high enough and you want native file watching, run:

```bash
npm run dev:native-watch
```

## Workflow

1. Start the backend.
2. Start the frontend.
3. Select a task from the left list.
4. Use Train/Test/Extra tabs to inspect examples.
5. Add and connect graph nodes in the editor.
6. Select a node to edit inputs, constants, dtype, axes, shapes, keepdims, and attributes.
7. Click `Run` to execute the graph on the selected example.
8. Click `Compile` to validate ONNX conversion in memory.
9. Click `Export ONNX` to write `onnx/taskXXX.onnx`.
10. Click `Save` to store a project JSON in `projects/`.
11. Pick a saved project in the inspector and click `Load Selected`.

## Graph JSON

Projects store a graph like:

```json
{
  "nodes": [
    {
      "id": "identity_1",
      "op": "Identity",
      "inputs": ["input"],
      "attrs": {}
    }
  ],
  "constants": {},
  "output": "identity_1"
}
```

`Constant` editor nodes are saved into the `constants` object. Other node outputs are referenced by their node IDs. The reserved input tensor is named `input`.

## Supported Ops

`Constant`, `Cast`, `Identity`, `Equal`, `Greater`, `Less`, `Not`, `And`, `Or`, `Where`, `ReduceSum`, `ArgMax`, `Slice`, `Pad`, `Reshape`, `Concat`, `Gather`, `Unsqueeze`, `Squeeze`.
