import type { Grid } from '../types';

const COLORS = [
  '#050505',
  '#1f5eff',
  '#ef3038',
  '#22a447',
  '#f2d33d',
  '#8b8f96',
  '#f46ab6',
  '#ff972f',
  '#35c9d0',
  '#7a1414'
];

function dims(grid?: Grid): string {
  if (!grid || grid.length === 0) return '0x0';
  return `${grid.length}x${grid[0]?.length ?? 0}`;
}

export default function GridView({
  grid,
  title,
  compareTo,
  compact = false
}: {
  grid?: Grid;
  title: string;
  compareTo?: Grid;
  compact?: boolean;
}) {
  if (!grid) {
    return (
      <div className="grid-card empty">
        <div className="grid-head"><span>{title}</span><small>missing</small></div>
      </div>
    );
  }
  const cols = grid[0]?.length ?? 0;
  const sameShape = compareTo && compareTo.length === grid.length && compareTo[0]?.length === cols;
  return (
    <div className={compact ? 'grid-card compact' : 'grid-card'}>
      <div className="grid-head"><span>{title}</span><small>{dims(grid)}</small></div>
      <div className="arc-grid" style={{ gridTemplateColumns: `repeat(${cols}, minmax(8px, 1fr))` }}>
        {grid.flatMap((row, r) =>
          row.map((color, c) => {
            const mismatch = sameShape && compareTo?.[r]?.[c] !== color;
            return (
              <div
                className={mismatch ? 'arc-cell mismatch' : 'arc-cell'}
                key={`${r}-${c}`}
                title={`${r},${c}: ${color}`}
                style={{ background: COLORS[color] ?? '#ffffff' }}
              />
            );
          })
        )}
      </div>
    </div>
  );
}
