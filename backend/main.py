from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .graph_engine import run_graph
from .onnx_builder import compile_graph, export_graph
from .storage import export_path, list_projects, list_tasks, load_project, load_task, save_project

app = FastAPI(title="NeuroGolf Lab API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProjectPayload(BaseModel):
    name: str | None = None
    taskId: str | None = None
    graph: dict[str, Any]
    layout: dict[str, Any] = Field(default_factory=dict)


class RunRequest(BaseModel):
    graph: dict[str, Any]
    input: list[list[int]]
    expected: list[list[int]] | None = None


class CompileRequest(BaseModel):
    graph: dict[str, Any]
    input: list[list[int]] | None = None


class ExportRequest(BaseModel):
    taskId: str
    graph: dict[str, Any]
    input: list[list[int]] | None = None


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "NeuroGolf Lab API",
        "frontend": "Open http://127.0.0.1:5173 for the visual editor.",
        "docs": "Open http://127.0.0.1:8000/docs for API docs.",
    }


@app.get("/tasks")
def get_tasks() -> list[dict[str, Any]]:
    return list_tasks()


@app.get("/tasks/{task_id}")
def get_task(task_id: str) -> dict[str, Any]:
    try:
        return load_task(task_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Task not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/projects")
def get_projects() -> list[str]:
    return list_projects()


@app.get("/projects/{name}")
def get_project(name: str) -> dict[str, Any]:
    try:
        return load_project(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Project not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/projects/{name}")
def post_project(name: str, payload: ProjectPayload) -> dict[str, Any]:
    try:
        data = payload.model_dump()
        data["name"] = name
        return save_project(name, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/run")
def post_run(payload: RunRequest) -> dict[str, Any]:
    try:
        return run_graph(payload.graph, payload.input, payload.expected)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/compile")
def post_compile(payload: CompileRequest) -> dict[str, Any]:
    try:
        return compile_graph(payload.graph, payload.input)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/export")
def post_export(payload: ExportRequest) -> dict[str, Any]:
    try:
        return export_graph(export_path(payload.taskId), payload.graph, payload.input)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
