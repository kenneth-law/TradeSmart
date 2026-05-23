import { BrainCircuit, Database, Radio, Server } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useMarketDataStatus } from '../../hooks/useMarketDataStatus'
import { useAppStore } from '../../store/useAppStore'

function fmt(d: Date | null) {
  if (!d) return '—'
  return d.toTimeString().slice(0, 8)
}

function fmtIso(value?: string) {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '—'
  return fmt(date)
}

function streamTone(status: 'online' | 'warn' | 'offline') {
  return {
    online: 'bg-up text-up',
    warn: 'bg-warn text-warn',
    offline: 'bg-down text-down',
  }[status]
}

function StreamRow({
  icon: Icon,
  name,
  status,
  detail,
  tone,
}: {
  icon: LucideIcon
  name: string
  status: string
  detail: string
  tone: 'online' | 'warn' | 'offline'
}) {
  const [dotClass, textClass] = streamTone(tone).split(' ')

  return (
    <div className="grid grid-cols-[18px_minmax(80px,1fr)_auto] items-start gap-2 border-t border-border px-3 py-2 first:border-t-0">
      <Icon size={14} className="mt-0.5 text-muted" aria-hidden="true" />
      <div className="min-w-0">
        <div className="truncate text-2xs font-medium text-text">{name}</div>
        <div className="truncate text-2xs text-dim" title={detail}>{detail}</div>
      </div>
      <div className="flex items-center gap-1.5 pl-2">
        <span className={`h-1.5 w-1.5 rounded-full ${dotClass}`} aria-hidden="true" />
        <span className={`whitespace-nowrap text-2xs ${textClass}`}>{status}</span>
      </div>
    </div>
  )
}

export default function StatusBar() {
  const { lastUpdated, analysisResult, openaiKey } = useAppStore()
  const [open, setOpen] = useState(false)
  const panelRef = useRef<HTMLDivElement>(null)
  const { data, loading } = useMarketDataStatus(openaiKey)

  const dotColor = { live: 'bg-up', delayed: 'bg-warn', offline: 'bg-down' }[data.state]
  const textColor = { live: 'text-up', delayed: 'text-warn', offline: 'text-down' }[data.state]
  const alpaca = data.streams.alpaca
  const yfinance = data.streams.yfinance
  const openai = data.streams.openai
  const backend = data.streams.tradesmart_backend

  const alpacaOnline = alpaca.connected && alpaca.authenticated
  const alpacaStatus = alpacaOnline ? 'connected' : alpaca.configured ? 'standby' : 'not configured'
  const alpacaTone = alpacaOnline ? 'online' : alpaca.configured ? 'warn' : 'offline'
  const alpacaDetail = alpaca.error
    ? alpaca.error
    : alpaca.missing?.length
      ? `Missing ${alpaca.missing.join(', ')}`
      : `Feed ${alpaca.feed || 'default'} · checked ${fmtIso(data.timestamp)}`

  const yfinanceStatus = yfinance.reachable ? 'delayed' : 'unreachable'
  const yfinanceDetail = yfinance.error
    ? yfinance.error
    : `Yahoo Finance · ${yfinance.latency_ms ?? '—'} ms · checked ${fmtIso(yfinance.checked_at)}`

  const openaiStatus = openai.configured ? 'configured' : 'not configured'
  const openaiDetail = openai.configured
    ? `Browser key saved · checked ${fmtIso(openai.checked_at)}`
    : 'Add an API key in Settings'

  const backendStatus = backend.reachable ? 'online' : 'offline'
  const backendDetail = backend.error
    ? backend.error
    : `Flask API · checked ${fmtIso(backend.checked_at)}`

  useEffect(() => {
    if (!open) return

    function onPointerDown(event: MouseEvent) {
      if (!panelRef.current?.contains(event.target as Node)) setOpen(false)
    }

    function onKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') setOpen(false)
    }

    document.addEventListener('mousedown', onPointerDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('mousedown', onPointerDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [open])

  return (
    <footer
      className="app-chrome app-chrome-border-t fixed bottom-0 left-0 right-0 z-50 flex items-center gap-3 overflow-x-auto overflow-y-hidden border-t px-3 select-none"
      style={{ height: 'var(--statusbar-height)' }}
      aria-label="Legal disclaimer and status"
    >
      <div ref={panelRef} className="relative flex shrink-0 items-center">
        <button
          type="button"
          onClick={() => setOpen(value => !value)}
          className="app-chrome-hover -ml-1 flex h-6 items-center gap-1 rounded-sm px-1.5 text-2xs focus-visible:outline focus-visible:outline-1 focus-visible:outline-accent"
          title="Market data streams"
          aria-label={`${data.label}. Open data stream status.`}
          aria-expanded={open}
          aria-controls="market-data-status-panel"
        >
          <span className={`h-1.5 w-1.5 rounded-full ${dotColor}`} aria-hidden="true" />
          <span className={`whitespace-nowrap ${loading ? 'text-dim' : textColor}`}>
            {loading ? 'market data: checking' : data.label}
          </span>
        </button>

        {open && (
          <div
            id="market-data-status-panel"
            className="app-chrome absolute bottom-full left-0 mb-2 w-[min(92vw,380px)] overflow-hidden rounded-[8px] border border-border-strong"
            role="dialog"
            aria-label="Market data streams"
          >
            <div className="flex items-center justify-between px-3 py-2">
              <span className="text-xs font-medium text-text">Data streams</span>
              <span className="text-2xs text-dim">updated {fmtIso(data.timestamp)}</span>
            </div>
            <StreamRow
              icon={Radio}
              name="Alpaca"
              status={alpacaStatus}
              detail={alpacaDetail}
              tone={alpacaTone}
            />
            <StreamRow
              icon={Database}
              name="YFinance"
              status={yfinanceStatus}
              detail={yfinanceDetail}
              tone={yfinance.reachable ? 'warn' : 'offline'}
            />
            <StreamRow
              icon={BrainCircuit}
              name="OpenAI"
              status={openaiStatus}
              detail={openaiDetail}
              tone={openai.configured ? 'online' : 'warn'}
            />
            <StreamRow
              icon={Server}
              name="TradeSmart backend"
              status={backendStatus}
              detail={backendDetail}
              tone={backend.reachable ? 'online' : 'offline'}
            />
          </div>
        )}
      </div>

      <span className="shrink-0 text-border">|</span>

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

      <span className="hidden shrink-0 text-2xs text-dim sm:inline">
        <kbd className="text-dim">Ctrl+K</kbd> command · <kbd className="text-dim">Alt+1-8</kbd> nav
      </span>

      <span className="hidden shrink-0 text-border sm:inline">|</span>

      <span className="hidden shrink-0 text-2xs text-dim lg:inline">© Kenneth Law 2026 | TradeSmart Analytics</span>
    </footer>
  )
}
