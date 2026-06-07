import type { Node } from 'reactflow';
import type { LabNodeData } from '../types';

function parseJson(value: string): unknown {
  if (!value.trim()) return {};
  return JSON.parse(value);
}

export default function Inspector({
  node,
  allIds,
  output,
  onUpdate,
  onDelete,
  onOutput,
  projectName,
  projects,
  onProjectName,
  onLoadProject
}: {
  node?: Node<LabNodeData>;
  allIds: string[];
  output: string;
  onUpdate: (id: string, data: Partial<LabNodeData>) => void;
  onDelete: (id: string) => void;
  onOutput: (id: string) => void;
  projectName: string;
  projects: string[];
  onProjectName: (name: string) => void;
  onLoadProject: (name: string) => void;
}) {
  return (
    <aside className="inspector">
      <div className="side-title">Inspector</div>
      <label className="field">
        <span>Project</span>
        <input value={projectName} onChange={(event) => onProjectName(event.target.value)} />
      </label>
      <label className="field">
        <span>Load saved</span>
        <select value="" onChange={(event) => event.target.value && onLoadProject(event.target.value)}>
          <option value="">Select project...</option>
          {projects.map((name) => <option key={name} value={name}>{name}</option>)}
        </select>
      </label>
      <label className="field">
        <span>Graph output</span>
        <select value={output} onChange={(event) => onOutput(event.target.value)}>
          {allIds.map((id) => <option key={id} value={id}>{id}</option>)}
        </select>
      </label>

      {node ? (
        <div className="node-inspector">
          <h3>{node.data.op}</h3>
          <small>{node.id}</small>
          {node.id !== 'input' && (
            <label className="field">
              <span>Input connections</span>
              <input
                value={(node.data.inputs || []).join(', ')}
                placeholder="input, constant_1"
                onChange={(event) => onUpdate(node.id, { inputs: event.target.value.split(',').map((v) => v.trim()).filter(Boolean) })}
              />
            </label>
          )}
          {node.data.op === 'Constant' && (
            <>
              <label className="field">
                <span>Dtype</span>
                <select value={node.data.dtype ?? 'int64'} onChange={(event) => onUpdate(node.id, { dtype: event.target.value })}>
                  {['int64', 'int32', 'float32', 'float64', 'bool', 'uint8'].map((dtype) => <option key={dtype}>{dtype}</option>)}
                </select>
              </label>
              <label className="field">
                <span>Value JSON</span>
                <textarea
                  value={JSON.stringify(node.data.value ?? 0)}
                  onChange={(event) => {
                    try { onUpdate(node.id, { value: JSON.parse(event.target.value) }); } catch { /* Wait for valid JSON. */ }
                  }}
                />
              </label>
            </>
          )}
          {node.id !== 'input' && node.data.op !== 'Constant' && (
            <label className="field">
              <span>Attributes JSON</span>
              <textarea
                value={JSON.stringify(node.data.attrs ?? {}, null, 2)}
                onChange={(event) => {
                  try { onUpdate(node.id, { attrs: parseJson(event.target.value) as Record<string, unknown> }); } catch { /* Wait for valid JSON. */ }
                }}
              />
            </label>
          )}
          <div className="available">
            <span>Available sources</span>
            <code>{allIds.join(', ')}</code>
          </div>
          {node.id !== 'input' && <button className="danger" onClick={() => onDelete(node.id)}>Delete node</button>}
        </div>
      ) : (
        <div className="empty-state">Select a node to edit inputs, constants, axes, shapes, dtype, and attributes.</div>
      )}
    </aside>
  );
}
