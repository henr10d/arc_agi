export type Grid = number[][];

export interface ArcExample {
  input: Grid;
  output?: Grid;
}

export interface ArcTask {
  train?: ArcExample[];
  test?: ArcExample[];
  [key: string]: ArcExample[] | undefined;
}

export interface TaskSummary {
  id: string;
  name: string;
  train: number;
  test: number;
  extra: number;
}

export interface GraphNodeSpec {
  id: string;
  op: string;
  inputs: string[];
  attrs: Record<string, unknown>;
}

export interface ConstantSpec {
  dtype: string;
  value: unknown;
}

export interface GraphSpec {
  nodes: GraphNodeSpec[];
  constants: Record<string, ConstantSpec>;
  output: string;
}

export interface ProjectSpec {
  name: string;
  taskId?: string;
  graph: GraphSpec;
  layout?: Record<string, { x: number; y: number }>;
}

export interface RunResult {
  predicted: Grid;
  matches?: boolean;
  mismatch_count?: number | null;
  shape: number[];
  tensors: Record<string, number[]>;
}

export interface LabNodeData {
  op: string;
  inputs: string[];
  attrs: Record<string, unknown>;
  dtype?: string;
  value?: unknown;
}
