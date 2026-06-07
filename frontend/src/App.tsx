import { useCallback, useEffect, useMemo, useState } from 'react';
import { ReactFlowProvider, Connection, Edge, Node, applyEdgeChanges, applyNodeChanges, NodeChange, EdgeChange } from 'reactflow';
import ExamplePanel, { getExamples } from './components/ExamplePanel';
import GraphEditor, { edgesFromNodes, makeNode, OPS } from './components/GraphEditor';
import Inspector from './components/Inspector';
import TaskList from './components/TaskList';
import { api } from './lib/api';
import type { ArcTask, GraphSpec, LabNodeData, ProjectSpec, RunResult, TaskSummary } from './types';

const inputNode: Node<LabNodeData> = {
  id: 'input',
  type: 'lab',
  position: { x: 40, y: 140 },
  data: { op: 'Input', inputs: [], attrs: {} },
  draggable: true
};

function defaultNodes(): Node<LabNodeData>[] {
  const id = makeNode('identity_1', 'Identity', 300, 140);
  id.data.inputs = ['input'];
  return [inputNode, id];
}

function nodeIdFor(op: string, nodes: Node<LabNodeData>[]): string {
  const base = op.toLowerCase();
  let index = nodes.length;
  let id = `${base}_${index}`;
  const ids = new Set(nodes.map((node) => node.id));
  while (ids.has(id)) {
    index += 1;
    id = `${base}_${index}`;
  }
  return id;
}

function graphFromNodes(nodes: Node<LabNodeData>[], output: string): GraphSpec {
  const constants: GraphSpec['constants'] = {};
  const specs: GraphSpec['nodes'] = [];
  for (const node of nodes) {
    if (node.id === 'input') continue;
    if (node.data.op === 'Constant') {
      constants[node.id] = { dtype: node.data.dtype ?? 'int64', value: node.data.value ?? 0 };
    } else {
      specs.push({ id: node.id, op: node.data.op, inputs: node.data.inputs || [], attrs: node.data.attrs || {} });
    }
  }
  return { nodes: specs, constants, output };
}

function nodesFromProject(project: ProjectSpec): Node<LabNodeData>[] {
  const layout = project.layout ?? {};
  const nodes: Node<LabNodeData>[] = [{ ...inputNode, position: layout.input ?? inputNode.position }];
  let row = 0;
  for (const [id, spec] of Object.entries(project.graph.constants || {})) {
    nodes.push({
      id,
      type: 'lab',
      position: layout[id] ?? { x: 280, y: 60 + row * 110 },
      data: { op: 'Constant', inputs: [], attrs: {}, dtype: spec.dtype, value: spec.value }
    });
    row += 1;
  }
  for (const spec of project.graph.nodes || []) {
    nodes.push({
      id: spec.id,
      type: 'lab',
      position: layout[spec.id] ?? { x: 520, y: 80 + row * 110 },
      data: { op: spec.op, inputs: spec.inputs || [], attrs: spec.attrs || {} }
    });
    row += 1;
  }
  return nodes;
}

function layoutFromNodes(nodes: Node<LabNodeData>[]): ProjectSpec['layout'] {
  return Object.fromEntries(nodes.map((node) => [node.id, node.position]));
}

export default function App() {
  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [taskId, setTaskId] = useState('task001');
  const [task, setTask] = useState<ArcTask>();
  const [tab, setTab] = useState('Train');
  const [exampleIndex, setExampleIndex] = useState(0);
  const [projectName, setProjectName] = useState('identity');
  const [projects, setProjects] = useState<string[]>([]);
  const [nodes, setNodes] = useState<Node<LabNodeData>[]>(defaultNodes);
  const [edges, setEdges] = useState<Edge[]>(edgesFromNodes(defaultNodes()));
  const [output, setOutput] = useState('identity_1');
  const [selectedNodeId, setSelectedNodeId] = useState<string | undefined>();
  const [runResult, setRunResult] = useState<RunResult>();
  const [status, setStatus] = useState('Ready.');

  const selectedNode = useMemo(() => nodes.find((node) => node.id === selectedNodeId), [nodes, selectedNodeId]);
  const allIds = useMemo(() => nodes.map((node) => node.id), [nodes]);
  const examples = getExamples(task, tab);
  const selectedExample = examples[exampleIndex] ?? examples[0];
  const graph = useMemo(() => graphFromNodes(nodes, output), [nodes, output]);

  const refreshProjects = useCallback(async () => {
    setProjects(await api.projects());
  }, []);

  useEffect(() => {
    api.tasks().then((loaded) => {
      setTasks(loaded);
      if (loaded.length && !loaded.some((item) => item.id === taskId)) setTaskId(loaded[0].id);
    }).catch((error) => setStatus(`Task list error: ${error.message}`));
    refreshProjects().catch((error) => setStatus(`Project list error: ${error.message}`));
  }, [refreshProjects]);

  useEffect(() => {
    api.task(taskId)
      .then((loaded) => {
        setTask(loaded);
        setExampleIndex(0);
        setRunResult(undefined);
      })
      .catch((error) => setStatus(`Task load error: ${error.message}`));
  }, [taskId]);

  const syncEdges = useCallback((nextNodes: Node<LabNodeData>[]) => {
    setEdges(edgesFromNodes(nextNodes));
  }, []);

  const setGraphNodes = useCallback((updater: (current: Node<LabNodeData>[]) => Node<LabNodeData>[]) => {
    setNodes((current) => {
      const next = updater(current);
      setEdges(edgesFromNodes(next));
      return next;
    });
  }, []);

  const onNodesChange = useCallback((changes: NodeChange[]) => {
    setNodes((current) => applyNodeChanges(changes, current));
  }, []);

  const onEdgesChange = useCallback((changes: EdgeChange[]) => {
    setEdges((current) => applyEdgeChanges(changes, current));
  }, []);

  const onConnect = useCallback((connection: Connection) => {
    if (!connection.source || !connection.target) return;
    const index = Number((connection.targetHandle ?? 'in-0').replace('in-', '')) || 0;
    setGraphNodes((current) => current.map((node) => {
      if (node.id !== connection.target) return node;
      const inputs = [...(node.data.inputs || [])];
      inputs[index] = connection.source!;
      return { ...node, data: { ...node.data, inputs } };
    }));
  }, [setGraphNodes]);

  const addNode = useCallback((op: string, position: { x: number; y: number }) => {
    if (!OPS.includes(op)) return;
    setGraphNodes((current) => [...current, makeNode(nodeIdFor(op, current), op, position.x, position.y)]);
  }, [setGraphNodes]);

  const updateNode = useCallback((id: string, data: Partial<LabNodeData>) => {
    setGraphNodes((current) => current.map((node) => node.id === id ? { ...node, data: { ...node.data, ...data } } : node));
  }, [setGraphNodes]);

  const deleteNode = useCallback((id: string) => {
    setGraphNodes((current) => current
      .filter((node) => node.id !== id)
      .map((node) => ({ ...node, data: { ...node.data, inputs: (node.data.inputs || []).filter((input) => input !== id) } })));
    if (output === id) setOutput('input');
    if (selectedNodeId === id) setSelectedNodeId(undefined);
  }, [output, selectedNodeId, setGraphNodes]);

  async function loadProject(name: string) {
    try {
      const project = await api.project(name);
      const loadedNodes = nodesFromProject(project);
      setProjectName(name);
      if (project.taskId) setTaskId(project.taskId);
      setNodes(loadedNodes);
      syncEdges(loadedNodes);
      setOutput(project.graph.output || loadedNodes[loadedNodes.length - 1]?.id || 'input');
      setRunResult(undefined);
      setStatus(`Loaded project ${name}.`);
    } catch (error) {
      setStatus(`Load failed: ${(error as Error).message}`);
    }
  }

  async function saveProject() {
    try {
      const project: ProjectSpec = { name: projectName, taskId, graph, layout: layoutFromNodes(nodes) };
      await api.saveProject(projectName, project);
      await refreshProjects();
      setStatus(`Saved ${projectName}.`);
    } catch (error) {
      setStatus(`Save failed: ${(error as Error).message}`);
    }
  }

  async function runGraph() {
    if (!selectedExample) return;
    try {
      const result = await api.run(graph, selectedExample.input, selectedExample.output);
      setRunResult(result);
      const verdict = result.matches === undefined ? '' : result.matches ? ' Match.' : ` ${result.mismatch_count ?? 'Shape'} mismatches.`;
      setStatus(`Run completed.${verdict}`);
    } catch (error) {
      setStatus(`Run failed: ${(error as Error).message}`);
    }
  }

  async function compileGraph() {
    try {
      const result = await api.compile(graph, selectedExample?.input);
      setStatus(`Compiled: ${JSON.stringify(result)}`);
    } catch (error) {
      setStatus(`Compile failed: ${(error as Error).message}`);
    }
  }

  async function exportGraph() {
    try {
      const result = await api.export(taskId, graph, selectedExample?.input);
      setStatus(`Exported: ${JSON.stringify(result)}`);
    } catch (error) {
      setStatus(`Export failed: ${(error as Error).message}`);
    }
  }

  return (
    <ReactFlowProvider>
      <div className="app-shell">
        <TaskList tasks={tasks} selected={taskId} onSelect={setTaskId} />
        <main className="main-column">
          <header className="topbar">
            <div>
              <h1>NeuroGolf Lab</h1>
              <p>Local ARC grid editor for tiny ONNX-style computation graphs.</p>
            </div>
            <div className="actions">
              <button onClick={runGraph}>Run</button>
              <button onClick={compileGraph}>Compile</button>
              <button onClick={exportGraph}>Export ONNX</button>
              <button onClick={saveProject}>Save</button>
              <button onClick={() => loadProject(projectName)}>Load Selected</button>
            </div>
          </header>
          <ExamplePanel
            task={task}
            tab={tab}
            exampleIndex={exampleIndex}
            predicted={runResult?.predicted}
            onTab={(nextTab) => { setTab(nextTab); setExampleIndex(0); setRunResult(undefined); }}
            onExample={(index) => { setExampleIndex(index); setRunResult(undefined); }}
          />
          <GraphEditor
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onAddNode={addNode}
            onSelection={setSelectedNodeId}
          />
          <div className="status">{status}</div>
        </main>
        <Inspector
          node={selectedNode}
          allIds={allIds}
          output={output}
          onUpdate={updateNode}
          onDelete={deleteNode}
          onOutput={setOutput}
          projectName={projectName}
          projects={projects}
          onProjectName={setProjectName}
          onLoadProject={loadProject}
        />
      </div>
    </ReactFlowProvider>
  );
}
