import { useState } from 'react'
import { api } from '../lib/api'
import { useSSE } from '../hooks/useSSE'
import { useQuery } from '@tanstack/react-query'
import TerminalLog from '../components/terminal/TerminalLog'
import ProgressBar from '../components/terminal/ProgressBar'
import MetricTile from '../components/data/MetricTile'
import DataTable from '../components/data/DataTable'
import type { Column } from '../components/data/DataTable'
import type { Trade } from '../types'

const STRATEGIES = [
  'Combined Strategy',
  'RSI Mean Reversion',
  'MACD Momentum',
  'Bollinger Breakout',
  'Volume Surge',
]

const ASX200 = 'CBA.AX, BHP.AX, CSL.AX, NAB.AX, WBC.AX, WES.AX, ANZ.AX, MQG.AX, GMG.AX, TLS.AX, FMG.AX, RIO.AX, TCL.AX, WDS.AX, ALL.AX, WOW.AX, QBE.AX, REA.AX, WTC.AX, BXB.AX'

const TRADE_COLUMNS: Column<Trade & Record<string, unknown>>[] = [
  { key: 'date',       label: 'DATE',       width: 100 },
  { key: 'ticker',     label: 'TICKER',     width: 80 },
  { key: 'type',       label: 'TYPE',       width: 60,
    render: r => (
      <span className={(r.type as string) === 'BUY' ? 'text-up' : 'text-down'}>
        {r.type as string}
      </span>
    )
  },
  { key: 'price',      label: 'PRICE',      width: 80, align: 'right',
    render: r => <span className="tabnum">{(r.price as number).toFixed(2)}</span>
  },
  { key: 'shares',     label: 'SHARES',     width: 72, align: 'right',
    render: r => <span className="tabnum">{r.shares !== undefined ? String(r.shares) : '—'}</span>
  },
  { key: 'return_pct', label: 'RETURN%',    width: 80, align: 'right',
    render: r => {
      const v = r.return_pct as number | undefined
      if (v === undefined) return <span className="text-dim">—</span>
      return <span className={`tabnum ${v >= 0 ? 'text-up' : 'text-down'}`}>{v >= 0 ? '+' : ''}{v.toFixed(2)}%</span>
    }
  },
]

type Phase = 'config' | 'running' | 'results'

export default function Backtest() {
  const [phase, setPhase] = useState<Phase>('config')
  const [tickerInput, setTickerInput] = useState('')
  const [strategy, setStrategy] = useState(STRATEGIES[0])
  const [days, setDays] = useState('365')
  const [txCost, setTxCost] = useState('0.001')
  const [backtestId, setBacktestId] = useState<string | null>(null)
  const [sseUrl, setSseUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const { messages, progress, status, lastMessage } = useSSE(sseUrl)

  const { data: results, isLoading: resultsLoading } = useQuery({
    queryKey: ['backtestResults', backtestId],
    queryFn: () => api.getBacktestResults(backtestId!),
    enabled: phase === 'results' && !!backtestId,
    retry: 2,
  })

  if (status === 'complete' && lastMessage?.backtest_id && phase === 'running') {
    setPhase('results')
    setBacktestId(lastMessage.backtest_id)
    setSseUrl(null)
  }

  if (status === 'error' && phase === 'running') {
    setError('Backtest failed. Check the log above.')
    setPhase('config')
    setSseUrl(null)
  }

  async function startBacktest() {
    const tickers = tickerInput.trim()
    if (!tickers) return
    setError(null)
    setPhase('running')
    try {
      const resp = await api.runBacktest({
        tickers: tickers.split(',').map(t => t.trim()).filter(Boolean),
        strategy,
        days: parseInt(days) || 365,
        custom_transaction_cost: parseFloat(txCost) || 0.001,
      })
      setBacktestId(resp.backtest_id)
      setSseUrl(`/backtest_progress_stream?backtest_id=${resp.backtest_id}`)
    } catch (e: unknown) {
      setError(`Failed to start backtest: ${e instanceof Error ? e.message : String(e)}`)
      setPhase('config')
    }
  }

  function reset() {
    setPhase('config')
    setBacktestId(null)
    setSseUrl(null)
    setError(null)
  }

  if (phase === 'config') {
    return (
      <div className="p-4 max-w-2xl mx-auto">
        <h1 className="text-sm text-text font-medium uppercase tracking-wide mb-4">Backtest</h1>

        <div className="border border-border bg-s1">
          {/* Tickers */}
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

          {/* Config fields */}
          <div className="grid grid-cols-2 gap-px bg-border">
            <div className="bg-s1 p-2">
              <label className="text-2xs text-dim uppercase tracking-wide block mb-1">Strategy</label>
              <select
                value={strategy}
                onChange={e => setStrategy(e.target.value)}
                className="w-full bg-s2 text-text text-sm border border-border px-2 py-1 outline-none focus:border-accent"
              >
                {STRATEGIES.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
            <div className="bg-s1 p-2">
              <label className="text-2xs text-dim uppercase tracking-wide block mb-1">Lookback (days)</label>
              <input
                type="number"
                value={days}
                onChange={e => setDays(e.target.value)}
                className="w-full bg-s2 text-text text-sm border border-border px-2 py-1 outline-none focus:border-accent tabnum"
              />
            </div>
            <div className="bg-s1 p-2">
              <label className="text-2xs text-dim uppercase tracking-wide block mb-1">Transaction Cost</label>
              <input
                type="number"
                step="0.0001"
                value={txCost}
                onChange={e => setTxCost(e.target.value)}
                className="w-full bg-s2 text-text text-sm border border-border px-2 py-1 outline-none focus:border-accent tabnum"
              />
            </div>
          </div>

          {/* Toolbar */}
          <div className="flex items-center gap-2 px-2 py-1.5">
            <button
              onClick={() => setTickerInput(ASX200)}
              className="text-2xs text-muted border border-border-strong px-2 py-1 hover:text-text"
            >
              Load ASX sample
            </button>
            <button
              onClick={startBacktest}
              disabled={!tickerInput.trim()}
              className="ml-auto text-sm text-bg bg-accent px-3 py-1 hover:opacity-90 disabled:opacity-40 font-medium"
            >
              Run Backtest [↵]
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
          <span className="text-sm text-text font-medium uppercase tracking-wide">Backtest running…</span>
          <span className="text-2xs text-dim tabnum">{progress}%</span>
        </div>
        <ProgressBar value={progress} />
        <div className="mt-3">
          <TerminalLog messages={messages} height={280} />
        </div>
      </div>
    )
  }

  // results phase
  if (resultsLoading || !results) {
    return <div className="p-4 text-muted text-sm tabnum">Loading results…</div>
  }

  const { metrics, trades, equity_curve } = results
  const tradeData = (trades as unknown as Array<Trade & Record<string, unknown>>)

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-border bg-s1 shrink-0">
        <button onClick={reset} className="text-2xs text-dim hover:text-text">
          ← New backtest
        </button>
        <span className="text-border">|</span>
        <span className="text-2xs text-muted tabnum">
          {results.strategy} · {results.tickers.length} tickers · {results.start_date} → {results.end_date}
        </span>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Trades table */}
        <div className="flex-1 overflow-auto">
          <DataTable
            columns={TRADE_COLUMNS}
            data={tradeData}
            rowKey={r => `${r.date}-${r.ticker}-${r.type}`}
            emptyMessage="No trades recorded."
          />
        </div>

        {/* Metrics sidebar */}
        <div className="w-52 shrink-0 border-l border-border overflow-y-auto">
          <div className="px-3 py-2 border-b border-border">
            <p className="text-2xs text-dim uppercase tracking-wide mb-1">Performance</p>
          </div>
          <MetricTile
            label="Total Return"
            value={`${metrics.total_return >= 0 ? '+' : ''}${metrics.total_return.toFixed(2)}`}
            unit="%"
            color={metrics.total_return >= 0 ? 'up' : 'down'}
          />
          {metrics.annualized_return !== undefined && (
            <MetricTile
              label="Ann. Return"
              value={`${metrics.annualized_return >= 0 ? '+' : ''}${metrics.annualized_return.toFixed(2)}`}
              unit="%"
              color={metrics.annualized_return >= 0 ? 'up' : 'down'}
            />
          )}
          <MetricTile label="Sharpe" value={metrics.sharpe_ratio.toFixed(2)} color={metrics.sharpe_ratio >= 1 ? 'up' : metrics.sharpe_ratio >= 0 ? 'warn' : 'down'} />
          <MetricTile label="Max Drawdown" value={metrics.max_drawdown.toFixed(2)} unit="%" color="down" />
          <MetricTile label="Win Rate" value={`${(metrics.win_rate * 100).toFixed(1)}`} unit="%" color={metrics.win_rate >= 0.5 ? 'up' : 'down'} />
          <MetricTile label="Trades" value={metrics.num_trades} />
          {metrics.profit_factor !== undefined && (
            <MetricTile label="Profit Factor" value={metrics.profit_factor.toFixed(2)} color={metrics.profit_factor >= 1 ? 'up' : 'down'} />
          )}

          {equity_curve && equity_curve.length > 0 && (
            <div className="px-3 py-2 border-t border-border">
              <p className="text-2xs text-dim uppercase tracking-wide mb-2">Equity Curve</p>
              <div className="h-24 flex items-end gap-px">
                {equity_curve.slice(-40).map((pt, i) => {
                  const min = Math.min(...equity_curve.map(p => p.value))
                  const max = Math.max(...equity_curve.map(p => p.value))
                  const range = max - min || 1
                  const h = ((pt.value - min) / range) * 88
                  return (
                    <div
                      key={i}
                      className="flex-1 bg-accent opacity-70"
                      style={{ height: `${h}%`, minHeight: 1 }}
                    />
                  )
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
