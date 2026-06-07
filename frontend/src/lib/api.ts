import type { ArcTask, GraphSpec, ProjectSpec, RunResult, TaskSummary, Grid } from '../types';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
    ...init
  });
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      message = body.detail || message;
    } catch {
      // Keep HTTP message.
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export const api = {
  tasks: () => request<TaskSummary[]>('/tasks'),
  task: (id: string) => request<ArcTask>(`/tasks/${id}`),
  projects: () => request<string[]>('/projects'),
  project: (name: string) => request<ProjectSpec>(`/projects/${name}`),
  saveProject: (name: string, project: ProjectSpec) =>
    request<{ name: string; path: string }>(`/projects/${name}`, {
      method: 'POST',
      body: JSON.stringify(project)
    }),
  run: (graph: GraphSpec, input: Grid, expected?: Grid) =>
    request<RunResult>('/run', {
      method: 'POST',
      body: JSON.stringify({ graph, input, expected })
    }),
  compile: (graph: GraphSpec, input?: Grid) =>
    request<Record<string, unknown>>('/compile', {
      method: 'POST',
      body: JSON.stringify({ graph, input })
    }),
  export: (taskId: string, graph: GraphSpec, input?: Grid) =>
    request<Record<string, unknown>>('/export', {
      method: 'POST',
      body: JSON.stringify({ taskId, graph, input })
    })
};
