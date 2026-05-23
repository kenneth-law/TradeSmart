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
    <footer
      className="app-chrome app-chrome-border-t fixed bottom-0 left-0 right-0 z-50 flex items-center gap-3 border-t px-3 select-none"
      style={{ height: 'var(--statusbar-height)' }}
      aria-label="Legal disclaimer and status"
    >
      {/* Connection */}
      <div className="flex shrink-0 items-center gap-1">
        <div className={`w-1.5 h-1.5 rounded-full ${dotColor}`} />
        <span className="text-2xs text-dim">{statusLabel}</span>
      </div>

      <span className="shrink-0 text-border">|</span>

      {/* Last update */}
      <span className="shrink-0 text-2xs text-dim">
        updated <span className="tabnum">{fmt(lastUpdated)}</span>
      </span>

      {analysisResult && (
        <>
          <span className="hidden shrink-0 text-border md:inline">|</span>
          <span className="hidden shrink-0 text-2xs text-dim tabnum md:inline">
            {analysisResult.ranked_stocks.length} stocks · session {analysisResult.session_id}
          </span>
        </>
      )}

      <span className="min-w-0 flex-1 truncate text-left text-2xs text-muted">
        Legal disclaimer: research tool only; not financial advice.
      </span>

      {/* Keyboard hint */}
      <span className="hidden shrink-0 text-2xs text-dim sm:inline">
        <kbd className="text-dim">Ctrl+K</kbd> command · <kbd className="text-dim">Alt+1-8</kbd> nav
      </span>

      <span className="hidden shrink-0 text-border sm:inline">|</span>

      <span className="hidden shrink-0 text-2xs text-dim lg:inline">© Kenneth Law 2026 | TradeSmart Analytics</span>
    </footer>
  )
}
