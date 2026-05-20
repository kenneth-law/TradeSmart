import { useAppStore } from '../../store/useAppStore'

function fmt(d: Date | null) {
  if (!d) return '—'
  return d.toTimeString().slice(0, 8)
}

export default function StatusBar() {
  const { connection, lastUpdated, analysisResult } = useAppStore()

  const dotColor = { connected: 'bg-up', degraded: 'bg-warn', offline: 'bg-down' }[connection]
  const statusLabel = { connected: 'live', degraded: 'degraded', offline: 'offline' }[connection]

  return (
    <div
      className="app-chrome app-chrome-border-t fixed bottom-0 left-0 right-0 z-50 flex items-center gap-4 border-t px-3 select-none"
      style={{ height: 'var(--statusbar-height)' }}
      aria-label="Status bar"
    >
      {/* Connection */}
      <div className="flex items-center gap-1">
        <div className={`w-1.5 h-1.5 rounded-full ${dotColor}`} />
        <span className="text-2xs text-dim">{statusLabel}</span>
      </div>

      <span className="text-border">|</span>

      {/* Last update */}
      <span className="text-2xs text-dim">
        updated <span className="tabnum">{fmt(lastUpdated)}</span>
      </span>

      {analysisResult && (
        <>
          <span className="text-border">|</span>
          <span className="text-2xs text-dim tabnum">
            {analysisResult.ranked_stocks.length} stocks · session {analysisResult.session_id}
          </span>
        </>
      )}

      {/* Keyboard hint — pushed right */}
      <span className="ml-auto text-2xs text-dim">
        <kbd className="text-dim">Ctrl+K</kbd> command · <kbd className="text-dim">Alt+1–7</kbd> nav
      </span>
    </div>
  )
}
