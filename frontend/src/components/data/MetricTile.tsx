interface MetricTileProps {
  label: string
  value: string | number
  unit?: string
  color?: 'up' | 'down' | 'accent' | 'warn' | 'muted' | 'text'
  size?: 'sm' | 'md'
}

const colorMap = {
  up:     'text-up',
  down:   'text-down',
  accent: 'text-accent',
  warn:   'text-warn',
  muted:  'text-muted',
  text:   'text-text',
}

export default function MetricTile({ label, value, unit, color = 'text', size = 'md' }: MetricTileProps) {
  return (
    <div className="flex flex-col gap-0.5 p-2 border-b border-border">
      <span className="text-2xs text-dim">{label}</span>
      <div className="flex items-baseline gap-1">
        <span className={`tabnum font-medium ${size === 'md' ? 'text-base' : 'text-sm'} ${colorMap[color]}`}>
          {value}
        </span>
        {unit && <span className="text-2xs text-muted">{unit}</span>}
      </div>
    </div>
  )
}
