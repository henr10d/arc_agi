import type { TaskSummary } from '../types';

export default function TaskList({ tasks, selected, onSelect }: {
  tasks: TaskSummary[];
  selected?: string;
  onSelect: (id: string) => void;
}) {
  return (
    <aside className="task-list">
      <div className="side-title">Tasks</div>
      <div className="task-scroll">
        {tasks.map((task) => (
          <button
            key={task.id}
            className={task.id === selected ? 'task-row active' : 'task-row'}
            onClick={() => onSelect(task.id)}
          >
            <span>{task.name}</span>
            <small>{task.train}/{task.test}{task.extra ? ` +${task.extra}` : ''}</small>
          </button>
        ))}
      </div>
    </aside>
  );
}
