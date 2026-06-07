import React, { useMemo } from 'react';
import ReactFlow, {
  Background,
  Controls,
  Edge,
  EdgeChange,
  Handle,
  MiniMap,
  Node,
  NodeProps,
  Position,
  Connection,
  useReactFlow
} from 'reactflow';
import type { LabNodeData } from '../types';

export const OPS = [
  'Constant', 'Cast', 'Identity', 'Equal', 'Greater', 'Less', 'Not', 'And', 'Or', 'Where',
  'ReduceSum', 'ArgMax', 'Slice', 'Pad', 'Reshape', 'Concat', 'Gather', 'Unsqueeze', 'Squeeze'
];

const BINARY = new Set(['Equal', 'Greater', 'Less', 'And', 'Or', 'Gather']);
const UNARY = new Set(['Cast', 'Identity', 'Not', 'ReduceSum', 'ArgMax', 'Squeeze', 'Unsqueeze']);

export function inputCount(op: string): number {
  if (op === 'Input' || op === 'Constant') return 0;
  if (op === 'Where') return 3;
  if (op === 'Slice') return 5;
  if (op === 'Pad' || op === 'Reshape') return 2;
  if (op === 'Concat') return 4;
  if (BINARY.has(op)) return 2;
  if (UNARY.has(op)) return 1;
  return 1;
}

export function defaultAttrs(op: string): Record<string, unknown> {
  switch (op) {
    case 'Cast': return { dtype: 'int64' };
    case 'ReduceSum': return { axes: [0], keepdims: 1 };
    case 'ArgMax': return { axis: 0, keepdims: 1 };
    case 'Slice': return { starts: [0], ends: [1], axes: [0], steps: [1] };
    case 'Pad': return { pads: [0, 0, 0, 0], mode: 'constant', constant_value: 0 };
    case 'Reshape': return { shape: [] };
    case 'Concat': return { axis: 0 };
    case 'Gather': return { axis: 0 };
    case 'Unsqueeze': return { axes: [0] };
    case 'Squeeze': return { axes: [0] };
    default: return {};
  }
}

function LabNode({ data, selected }: NodeProps<LabNodeData>) {
  const count = inputCount(data.op);
  const targets = Array.from({ length: count }, (_, index) => index);
  return (
    <div className={selected ? 'lab-node selected' : 'lab-node'}>
      {targets.map((index) => (
        <Handle
          key={index}
          id={`in-${index}`}
          type="target"
          position={Position.Left}
          style={{ top: `${((index + 1) / (targets.length + 1)) * 100}%` }}
        />
      ))}
      <div className="node-op">{data.op}</div>
      <div className="node-id">{data.op === 'Constant' ? `${data.dtype ?? 'int64'} = ${JSON.stringify(data.value ?? 0)}` : data.inputs.join(', ') || 'no inputs'}</div>
      <Handle id="out" type="source" position={Position.Right} />
    </div>
  );
}

const nodeTypes = { lab: LabNode };

export function makeNode(id: string, op: string, x: number, y: number): Node<LabNodeData> {
  return {
    id,
    type: 'lab',
    position: { x, y },
    data: {
      op,
      inputs: [],
      attrs: defaultAttrs(op),
      dtype: op === 'Constant' ? 'int64' : undefined,
      value: op === 'Constant' ? 0 : undefined
    }
  };
}

export function edgesFromNodes(nodes: Node<LabNodeData>[]): Edge[] {
  const ids = new Set(nodes.map((node) => node.id));
  return nodes.flatMap((node) =>
    (node.data.inputs || [])
      .map((source, index) => ({ source, index }))
      .filter(({ source }) => ids.has(source))
      .map(({ source, index }) => ({
        id: `${source}-${node.id}-${index}`,
        source,
        target: node.id,
        sourceHandle: 'out',
        targetHandle: `in-${index}`,
        animated: false
      }))
  );
}

export default function GraphEditor({
  nodes,
  edges,
  onNodesChange,
  onEdgesChange,
  onConnect,
  onAddNode,
  onSelection
}: {
  nodes: Node<LabNodeData>[];
  edges: Edge[];
  onNodesChange: (changes: import('reactflow').NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection) => void;
  onAddNode: (op: string, position: { x: number; y: number }) => void;
  onSelection: (id?: string) => void;
}) {
  const rf = useReactFlow<LabNodeData>();
  const toolbar = useMemo(() => OPS, []);
  return (
    <section className="graph-pane">
      <div className="op-palette">
        {toolbar.map((op) => (
          <button
            key={op}
            onClick={() => {
              const center = rf.screenToFlowPosition({ x: window.innerWidth * 0.55, y: window.innerHeight * 0.55 });
              onAddNode(op, center);
            }}
          >
            + {op}
          </button>
        ))}
      </div>
      <div className="flow-wrap">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={(connection) => onConnect(connection)}
          onNodeClick={(_, node) => onSelection(node.id)}
          onPaneClick={() => onSelection(undefined)}
          fitView
        >
          <Background color="#40505b" gap={22} />
          <MiniMap pannable zoomable />
          <Controls />
        </ReactFlow>
      </div>
    </section>
  );
}
