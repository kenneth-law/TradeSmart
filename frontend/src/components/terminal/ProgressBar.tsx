export default function ProgressBar({ value }: { value: number }) {
  const pct = Math.min(100, Math.max(0, value))
  return (
    <div className="w-full h-px bg-s2 relative overflow-hidden" style={{ height: 2 }}>
      <div
        className="absolute left-0 top-0 h-full bg-accent"
        style={{ width: `${pct}%` }}
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`Progress: ${pct}%`}
      />
    </div>
  )
}
