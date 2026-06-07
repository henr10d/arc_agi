import type { ArcExample, ArcTask, Grid } from '../types';
import GridView from './GridView';

function extraExamples(task?: ArcTask): ArcExample[] {
  if (!task) return [];
  return Object.entries(task)
    .filter(([key, value]) => key !== 'train' && key !== 'test' && Array.isArray(value))
    .flatMap(([, value]) => value ?? []);
}

export function getExamples(task: ArcTask | undefined, tab: string): ArcExample[] {
  if (!task) return [];
  if (tab === 'Train') return task.train ?? [];
  if (tab === 'Test') return task.test ?? [];
  return extraExamples(task);
}

export default function ExamplePanel({
  task,
  tab,
  exampleIndex,
  predicted,
  onTab,
  onExample
}: {
  task?: ArcTask;
  tab: string;
  exampleIndex: number;
  predicted?: Grid;
  onTab: (tab: string) => void;
  onExample: (index: number) => void;
}) {
  const examples = getExamples(task, tab);
  const example = examples[exampleIndex] ?? examples[0];
  return (
    <section className="examples">
      <div className="tabs">
        {['Train', 'Test', 'Extra'].map((name) => (
          <button key={name} className={tab === name ? 'tab active' : 'tab'} onClick={() => onTab(name)}>{name}</button>
        ))}
      </div>
      <div className="example-strip">
        {examples.map((_, index) => (
          <button key={index} className={index === exampleIndex ? 'pill active' : 'pill'} onClick={() => onExample(index)}>
            {tab.toLowerCase()} {index + 1}
          </button>
        ))}
      </div>
      {example ? (
        <>
          <div className="dims-line">
            {example.input.length}x{example.input[0]?.length ?? 0} -&gt; {example.output ? `${example.output.length}x${example.output[0]?.length ?? 0}` : 'unknown'}
          </div>
          <div className="grid-row">
            <GridView grid={example.input} title="Input" />
            <GridView grid={example.output} title="Expected" />
            <GridView grid={predicted} compareTo={example.output} title="Predicted" />
          </div>
        </>
      ) : (
        <div className="empty-state">No examples for this tab.</div>
      )}
    </section>
  );
}
