function scoreColor(score: number) {
  if (score >= 70) return 'bg-up'
  if (score >= 50) return 'bg-accent'
  if (score >= 35) return 'bg-warn'
  return 'bg-down'
}

export default function ScoreBar({ score }: { score: number }) {
  const pct = Math.min(100, Math.max(0, score))
  return (
    <div className="flex items-center gap-1.5 w-full">
      <div className="flex-1 h-px bg-border relative overflow-hidden" style={{ height: 2 }}>
        <div
          className={`absolute left-0 top-0 h-full ${scoreColor(score)}`}
          style={{ width: `${pct}%` }}
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>
      <span className="tabnum text-2xs text-muted w-8 text-right shrink-0">{pct.toFixed(0)}</span>
    </div>
  )
}
