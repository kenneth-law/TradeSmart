import { useState } from 'react'
import { api } from '../lib/api'
import { useSSE } from '../hooks/useSSE'
import { useQuery } from '@tanstack/react-query'
import TerminalLog from '../components/terminal/TerminalLog'
import ProgressBar from '../components/terminal/ProgressBar'
import MetricTile from '../components/data/MetricTile'
import DataTable from '../components/data/DataTable'
import type { Column } from '../components/data/DataTable'
import SignalBadge from '../components/data/SignalBadge'
import ScoreBar from '../components/data/ScoreBar'
import type { StockResult } from '../types'

const ASX200 = 'CBA.AX, BHP.AX, CSL.AX, NAB.AX, WBC.AX, WES.AX, ANZ.AX, MQG.AX, GMG.AX, TLS.AX, FMG.AX, RIO.AX, TCL.AX, WDS.AX, ALL.AX, WOW.AX, QBE.AX, REA.AX, WTC.AX, BXB.AX'

const SIGNAL_COLUMNS: Column<StockResult & Record<string, unknown>>[] = [
  { key: 'ticker',              label: 'TICKER',  width: 80 },
  { key: 'day_trading_score',   label: 'SCORE',   width: 110, align: 'right',
    render: r => <ScoreBar score={r.day_trading_score as number} />
  },
  { key: 'day_trading_strategy', label: 'SIGNAL', width: 110,
    render: r => <SignalBadge signal={r.day_trading_strategy as StockResult['day_trading_strategy']} />
  },
  { key: 'current_price', label: 'PRICE', width: 80, align: 'right',
    render: r => <span className="tabnum">{(r.current_price as number).toFixed(2)}</span>
  },
  { key: 'return_1d', label: 'CHG%', width: 72, align: 'right',
    render: r => {
      const v = r.return_1d as number
      return <span className={`tabnum ${v >= 0 ? 'text-up' : 'text-down'}`}>{v >= 0 ? '+' : ''}{v.toFixed(2)}%</span>
    }
  },
  { key: 'macd_trend', label: 'MACD', width: 72,
    render: r => {
      const v = r.macd_trend as string
      return <span className={v === 'bullish' ? 'text-up' : v === 'bearish' ? 'text-down' : 'text-muted'}>{v.toUpperCase()}</span>
    }
  },
]

type Phase = 'config' | 'running' | 'results'

export default function IntegratedSystem() {
  const [phase, setPhase] = useState<Phase>('config')
  const [tickerInput, setTickerInput] = useState('')
  const [useMl, setUseMl] = useState(true)
  const [executeTrades, setExecuteTrades] = useState(false)
  const [systemId, setSystemId] = useState<string | null>(null)
  const [sseUrl, setSseUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const { messages, progress, status, lastMessage } = useSSE(sseUrl)

  const { data: results, isLoading } = useQuery({
    queryKey: ['integratedResults', systemId],
    queryFn: () => api.getIntegratedResults(systemId!),
    enabled: phase === 'results' && !!systemId,
    retry: 2,
  })

  if (status === 'complete' && lastMessage?.system_id && phase === 'running') {
    setPhase('results')
    setSystemId(lastMessage.system_id)
    setSseUrl(null)
  }

  if (status === 'error' && phase === 'running') {
    setError('System run failed. Check the log above.')
    setPhase('config')
    setSseUrl(null)
  }

  async function startSystem() {
    const tickers = tickerInput.trim()
    if (!tickers) return
    setError(null)
    setPhase('running')
    try {
      const resp = await api.runIntegrated({
        tickers: tickers.split(',').map(t => t.trim()).filter(Boolean),
        use_ml: useMl,
        execute_trades: executeTrades,
      })
      setSystemId(resp.system_id)
      setSseUrl(`/integrated_progress_stream?system_id=${resp.system_id}`)
    } catch (e: unknown) {
      setError(`Failed to start: ${e instanceof Error ? e.message : String(e)}`)
      setPhase('config')
    }
  }

  function reset() {
    setPhase('config')
    setSystemId(null)
    setSseUrl(null)
    setError(null)
  }

  if (phase === 'config') {
    return (
      <div className="p-4 max-w-2xl mx-auto">
        <h1 className="text-sm text-text font-medium uppercase tracking-wide mb-4">Integrated Trading System</h1>

        <div className="border border-border bg-s1">
          <div className="flex items-start gap-2 p-2 border-b border-border">
            <span className="text-muted text-sm pt-px select-none shrink-0">{'>'}</span>
            <textarea
              value={tickerInput}
              onChange={e => setTickerInput(e.target.value)}
              placeholder="AAPL, MSFT, GOOGL  —  comma separated"
              rows={3}
              className="flex-1 bg-transparent text-text text-sm placeholder:text-dim resize-none outline-none"
              aria-label="Ticker symbols"
            />
          </div>

          <div className="grid grid-cols-2 gap-px bg-border">
            <div className="bg-s1 p-2 flex items-center justify-between">
              <label className="text-2xs text-dim uppercase tracking-wide">ML Signals</label>
              <button
                onClick={() => setUseMl(v => !v)}
                className={`text-2xs px-2 py-1 border ${useMl ? 'border-accent text-accent' : 'border-border text-muted hover:text-text'}`}
              >
                {useMl ? 'ON' : 'OFF'}
              </button>
            </div>
            <div className="bg-s1 p-2 flex items-center justify-between">
              <label className="text-2xs text-dim uppercase tracking-wide">Execute Trades</label>
              <button
                onClick={() => setExecuteTrades(v => !v)}
                className={`text-2xs px-2 py-1 border ${executeTrades ? 'border-down text-down' : 'border-border text-muted hover:text-text'}`}
              >
                {executeTrades ? 'ON' : 'OFF'}
              </button>
            </div>
          </div>

          {executeTrades && (
            <div className="px-3 py-2 border-t border-border bg-s2">
              <p className="text-2xs text-warn">
                Trade execution is enabled. System will attempt live order placement.
              </p>
            </div>
          )}

          <div className="flex items-center gap-2 px-2 py-1.5">
            <button
              onClick={() => setTickerInput(ASX200)}
              className="text-2xs text-muted border border-border-strong px-2 py-1 hover:text-text"
            >
              Load ASX sample
            </button>
            <button
              onClick={startSystem}
              disabled={!tickerInput.trim()}
              className="ml-auto text-sm text-bg bg-accent px-3 py-1 hover:opacity-90 disabled:opacity-40 font-medium"
            >
              Run System [↵]
            </button>
          </div>
        </div>

        {error && <p className="mt-2 text-2xs text-down" role="alert">{error}</p>}
      </div>
    )
  }

  if (phase === 'running') {
    return (
      <div className="p-4 max-w-2xl mx-auto">
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm text-text font-medium uppercase tracking-wide">System running…</span>
          <span className="text-2xs text-dim tabnum">{progress}%</span>
        </div>
        <ProgressBar value={progress} />
        <div className="mt-3">
          <TerminalLog messages={messages} height={280} />
        </div>
      </div>
    )
  }

  if (isLoading || !results) {
    return <div className="p-4 text-muted text-sm tabnum">Loading results…</div>
  }

  const { signals, portfolio_summary, tickers } = results
  const signalData = (signals as unknown as Array<StockResult & Record<string, unknown>>)

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-3 px-4 py-2 border-b border-border bg-s1 shrink-0">
        <button onClick={reset} className="text-2xs text-dim hover:text-text">
          ← New run
        </button>
        <span className="text-border">|</span>
        <span className="text-2xs text-muted tabnum">{tickers.length} tickers processed</span>
      </div>

      <div className="flex flex-1 overflow-hidden">
        <div className="flex-1 overflow-auto">
          <DataTable
            columns={SIGNAL_COLUMNS}
            data={signalData}
            rowKey={r => r.ticker}
            virtualRows={signalData.length > 50}
            emptyMessage="No signals generated."
          />
        </div>

        <div className="w-52 shrink-0 border-l border-border overflow-y-auto">
          <div className="px-3 py-2 border-b border-border">
            <p className="text-2xs text-dim uppercase tracking-wide">Portfolio</p>
          </div>
          <MetricTile label="Total Value" value={`$${portfolio_summary.total_value.toLocaleString()}`} />
          <MetricTile label="Cash" value={`$${portfolio_summary.cash.toLocaleString()}`} />
          <MetricTile label="Invested" value={`$${portfolio_summary.invested.toLocaleString()}`} />
          <MetricTile
            label="Total P&L"
            value={`${portfolio_summary.total_pnl >= 0 ? '+' : ''}$${portfolio_summary.total_pnl.toLocaleString()}`}
            color={portfolio_summary.total_pnl >= 0 ? 'up' : 'down'}
          />
          <MetricTile
            label="Total P&L %"
            value={`${portfolio_summary.total_pnl_pct >= 0 ? '+' : ''}${portfolio_summary.total_pnl_pct.toFixed(2)}`}
            unit="%"
            color={portfolio_summary.total_pnl_pct >= 0 ? 'up' : 'down'}
          />
          <MetricTile label="Positions" value={portfolio_summary.num_positions} />
          {portfolio_summary.portfolio_beta !== undefined && (
            <MetricTile label="Beta" value={portfolio_summary.portfolio_beta.toFixed(2)} />
          )}
          {portfolio_summary.max_drawdown !== undefined && (
            <MetricTile label="Max DD" value={portfolio_summary.max_drawdown.toFixed(2)} unit="%" color="down" />
          )}
        </div>
      </div>
    </div>
  )
}
