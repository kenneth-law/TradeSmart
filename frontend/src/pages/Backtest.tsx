import { useState } from 'react'
import { useLocation } from 'react-router-dom'
import { api } from '../lib/api'
import { useSSE } from '../hooks/useSSE'
import { useQueries, useQuery } from '@tanstack/react-query'
import TerminalLog from '../components/terminal/TerminalLog'
import ProgressBar from '../components/terminal/ProgressBar'
import MetricTile from '../components/data/MetricTile'
import DataTable from '../components/data/DataTable'
import type { Column } from '../components/data/DataTable'
import EquityChart from '../components/charts/EquityChart'
import type { PriceOverlay } from '../components/charts/EquityChart'
import SignalBadge from '../components/data/SignalBadge'
import ScoreBar from '../components/data/ScoreBar'
import type { PriceHistory, StockResult, Trade } from '../types'

const STRATEGIES = [
  { label: 'ML Strategy', value: 'ml' },
  { label: 'Technical Strategy', value: 'technical' },
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

const SIGNAL_COLUMNS: Column<StockResult & Record<string, unknown>>[] = [
  { key: 'ticker', label: 'TICKER', width: 100 },
  { key: 'day_trading_score', label: 'SCORE', width: 130, align: 'right',
    render: r => <ScoreBar score={r.day_trading_score as number} />
  },
  { key: 'day_trading_strategy', label: 'SIGNAL', width: 140,
    render: r => <SignalBadge signal={r.day_trading_strategy as StockResult['day_trading_strategy']} />
  },
  { key: 'current_price', label: 'PRICE', width: 100, align: 'right',
    render: r => <span className="tabnum">{(r.current_price as number).toFixed(2)}</span>
  },
  { key: 'return_1d', label: 'CHG%', width: 90, align: 'right',
    render: r => {
      const v = r.return_1d as number
      return <span className={`tabnum ${v >= 0 ? 'text-up' : 'text-down'}`}>{v >= 0 ? '+' : ''}{v.toFixed(2)}%</span>
    }
  },
  { key: 'macd_trend', label: 'MACD', width: 90,
    render: r => {
      const v = r.macd_trend as string
      return <span className={v === 'bullish' ? 'text-up' : v === 'bearish' ? 'text-down' : 'text-muted'}>{v.toUpperCase()}</span>
    }
  },
]

type Phase = 'config' | 'running' | 'results'
type RunMode = 'simulation' | 'workflow'
type TransactionCostType = 'fixed' | 'percent'

type RawTrade = Omit<Trade, 'type'> & {
  action?: 'BUY' | 'SELL'
  type?: 'BUY' | 'SELL'
}

type RawEquityPoint = {
  date: string
  value?: number
  equity?: number
}

const OVERLAY_COLORS = ['#E89B2C', '#3B82F6', '#A78BFA', '#14B8A6', '#F472B6', '#FACC15', '#94A3B8']

function normalizeTrade(trade: RawTrade): Trade {
  return {
    ...trade,
    type: trade.type ?? trade.action ?? 'BUY',
  }
}

function normalizeEquityPoint(point: RawEquityPoint) {
  return {
    date: point.date,
    value: Number(point.value ?? point.equity ?? 0),
  }
}

function backtestLookbackDays(startDate?: string, endDate?: string) {
  if (!startDate || !endDate) return 365

  const start = new Date(`${startDate}T00:00:00`)
  const end = new Date(`${endDate}T00:00:00`)
  const diffMs = end.getTime() - start.getTime()

  if (!Number.isFinite(diffMs) || diffMs <= 0) return 365
  return Math.ceil(diffMs / 86_400_000) + 10
}

function buildPriceOverlay(
  ticker: string,
  color: string,
  history: PriceHistory | undefined,
  startDate: string,
  endDate: string,
): PriceOverlay | null {
  if (!history?.dates?.length) return null

  const points = history.dates
    .map((date, i) => ({
      date,
      value: Number(history.close?.[i] ?? history.prices?.[i]),
    }))
    .filter(point => (
      point.date >= startDate &&
      point.date <= endDate &&
      Number.isFinite(point.value) &&
      point.value > 0
    ))

  if (!points.length) return null

  return { ticker, color, points }
}

export default function Backtest() {
  const location = useLocation()
  const initialMode: RunMode = location.pathname.includes('integrated') ? 'workflow' : 'simulation'
  const [phase, setPhase] = useState<Phase>('config')
  const [runMode, setRunMode] = useState<RunMode>(initialMode)
  const [resultMode, setResultMode] = useState<RunMode>(initialMode)
  const [tickerInput, setTickerInput] = useState('')
  const [strategy, setStrategy] = useState(STRATEGIES[0].value)
  const [days, setDays] = useState('365')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [txCostType, setTxCostType] = useState<TransactionCostType>('percent')
  const [txCost, setTxCost] = useState('0.10')
  const [simulateTrading, setSimulateTrading] = useState(true)
  const [useMl, setUseMl] = useState(true)
  const [executeTrades, setExecuteTrades] = useState(false)
  const [backtestId, setBacktestId] = useState<string | null>(null)
  const [systemId, setSystemId] = useState<string | null>(null)
  const [sseUrl, setSseUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selectedOverlayTickers, setSelectedOverlayTickers] = useState<string[]>([])

  const { messages, progress, status, lastMessage } = useSSE(sseUrl)

  const { data: results, isLoading: resultsLoading } = useQuery({
    queryKey: ['backtestResults', backtestId],
    queryFn: () => api.getBacktestResults(backtestId!),
    enabled: phase === 'results' && !!backtestId,
    retry: 2,
  })

  const { data: workflowResults, isLoading: workflowLoading } = useQuery({
    queryKey: ['integratedResults', systemId],
    queryFn: () => api.getIntegratedResults(systemId!),
    enabled: phase === 'results' && resultMode === 'workflow' && !!systemId,
    retry: 2,
  })

  const overlayDays = backtestLookbackDays(results?.start_date, results?.end_date)
  const overlayQueries = useQueries({
    queries: selectedOverlayTickers.map(ticker => ({
      queryKey: ['backtestPriceOverlay', ticker, overlayDays],
      queryFn: () => api.getPriceHistory(ticker, overlayDays),
      enabled: phase === 'results' && !!results,
      staleTime: 300_000,
      retry: 1,
    })),
  })

  if (status === 'complete' && lastMessage?.backtest_id && phase === 'running') {
    setPhase('results')
    setResultMode('simulation')
    setBacktestId(lastMessage.backtest_id)
    setSseUrl(null)
  }

  if (status === 'complete' && lastMessage?.system_id && phase === 'running') {
    setPhase('results')
    setResultMode('workflow')
    setSystemId(lastMessage.system_id)
    setSseUrl(null)
  }

  if (status === 'error' && phase === 'running') {
    setError('System run failed. Check the log above.')
    setPhase('config')
    setSseUrl(null)
  }

  async function startRun() {
    const tickers = tickerInput.trim()
    if (!tickers) return
    setError(null)
    setSelectedOverlayTickers([])
    setPhase('running')
    try {
      const tickerList = tickers.split(',').map(t => t.trim()).filter(Boolean)
      if (runMode === 'workflow') {
        const resp = await api.runIntegrated({
          tickers: tickerList,
          use_ml: useMl,
          execute_trades: executeTrades,
        })
        setSystemId(resp.system_id)
        setBacktestId(null)
        setSseUrl(`/integrated_progress_stream?system_id=${resp.system_id}`)
        return
      }

      const resp = await api.runBacktest({
        tickers: tickerList,
        strategy,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
        days: parseInt(days) || 365,
        custom_transaction_cost: Number.isFinite(parseFloat(txCost)) ? parseFloat(txCost) : undefined,
        transaction_cost_type: txCostType,
      })
      setBacktestId(resp.backtest_id)
      setSystemId(null)
      setSseUrl(`/backtest_progress_stream?backtest_id=${resp.backtest_id}`)
    } catch (e: unknown) {
      setError(`Failed to start system run: ${e instanceof Error ? e.message : String(e)}`)
      setPhase('config')
    }
  }

  function reset() {
    setPhase('config')
    setBacktestId(null)
    setSystemId(null)
    setSseUrl(null)
    setError(null)
    setSelectedOverlayTickers([])
  }

  function toggleOverlayTicker(ticker: string) {
    setSelectedOverlayTickers(current =>
      current.includes(ticker)
        ? current.filter(t => t !== ticker)
        : [...current, ticker]
    )
  }

  if (phase === 'config') {
    return (
      <div className="p-4 max-w-4xl mx-auto">
        <h1 className="text-sm text-text font-medium mb-4">System</h1>

        <div className="flex gap-px mb-3 border border-border bg-border">
          {([
            ['simulation', 'Simulation'],
            ['workflow', 'Integrated'],
          ] as Array<[RunMode, string]>).map(([mode, label]) => (
            <button
              key={mode}
              onClick={() => setRunMode(mode)}
              className={[
                'flex-1 bg-s1 px-3 py-2 text-sm',
                runMode === mode ? 'text-accent' : 'text-muted hover:text-text',
              ].join(' ')}
            >
              {label}
            </button>
          ))}
        </div>

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
              <label className="text-2xs text-dim block mb-1">Strategy</label>
              <select
                value={strategy}
                onChange={e => setStrategy(e.target.value)}
                disabled={runMode !== 'simulation'}
                className="w-full bg-s2 text-text text-sm border border-border px-2 py-1 outline-none focus:border-accent"
              >
                {STRATEGIES.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
              </select>
            </div>
            <div className="bg-s1 p-2">
              <label className="text-2xs text-dim block mb-1">Lookback (days)</label>
              <input
                type="number"
                value={days}
                onChange={e => setDays(e.target.value)}
                disabled={runMode !== 'simulation' || !!startDate || !!endDate}
                className="w-full bg-s2 text-text text-sm border border-border px-2 py-1 outline-none focus:border-accent tabnum"
              />
            </div>
            <div className="bg-s1 p-2">
              <label className="text-2xs text-dim block mb-1">Start Date</label>
              <input
                type="date"
                value={startDate}
                onChange={e => setStartDate(e.target.value)}
                disabled={runMode !== 'simulation'}
                className="w-full bg-s2 text-text text-sm border border-border px-2 py-1 outline-none focus:border-accent tabnum"
              />
            </div>
            <div className="bg-s1 p-2">
              <label className="text-2xs text-dim block mb-1">End Date</label>
              <input
                type="date"
                value={endDate}
                onChange={e => setEndDate(e.target.value)}
                disabled={runMode !== 'simulation'}
                className="w-full bg-s2 text-text text-sm border border-border px-2 py-1 outline-none focus:border-accent tabnum"
              />
            </div>
            <div className="bg-s1 p-2">
              <label className="text-2xs text-dim block mb-1">Transaction Cost</label>
              <input
                type="number"
                step="0.0001"
                value={txCost}
                onChange={e => setTxCost(e.target.value)}
                disabled={runMode !== 'simulation'}
                className="w-full bg-s2 text-text text-sm border border-border px-2 py-1 outline-none focus:border-accent tabnum"
              />
            </div>
            <div className="bg-s1 p-2">
              <label className="text-2xs text-dim block mb-1">Cost Mode</label>
              <div className="flex gap-px border border-border bg-border">
                {(['percent', 'fixed'] as TransactionCostType[]).map(mode => (
                  <button
                    key={mode}
                    onClick={() => {
                      setTxCostType(mode)
                      setTxCost(mode === 'percent' ? '0.10' : '10')
                    }}
                    disabled={runMode !== 'simulation'}
                    className={[
                      'flex-1 bg-s2 px-2 py-1 text-2xs',
                      txCostType === mode ? 'text-accent' : 'text-muted hover:text-text',
                    ].join(' ')}
                  >
                    {mode === 'percent' ? '% Notional' : 'Fixed $'}
                  </button>
                ))}
              </div>
            </div>
            <div className="bg-s1 p-2 flex items-center justify-between">
              <label className="text-2xs text-dim">Simulate Trading</label>
              <button
                onClick={() => setSimulateTrading(v => !v)}
                disabled={runMode !== 'simulation'}
                className={`text-2xs px-2 py-1 border ${simulateTrading ? 'border-accent text-accent' : 'border-border text-muted hover:text-text'}`}
              >
                {simulateTrading ? 'ON' : 'OFF'}
              </button>
            </div>
            <div className="bg-s1 p-2 flex items-center justify-between">
              <label className="text-2xs text-dim">ML Signals</label>
              <button
                onClick={() => setUseMl(v => !v)}
                disabled={runMode !== 'workflow'}
                className={`text-2xs px-2 py-1 border ${useMl ? 'border-accent text-accent' : 'border-border text-muted hover:text-text'}`}
              >
                {useMl ? 'ON' : 'OFF'}
              </button>
            </div>
            <div className="bg-s1 p-2 flex items-center justify-between">
              <label className="text-2xs text-dim">Execute Trades</label>
              <button
                onClick={() => setExecuteTrades(v => !v)}
                disabled={runMode !== 'workflow'}
                className={`text-2xs px-2 py-1 border ${executeTrades ? 'border-down text-down' : 'border-border text-muted hover:text-text'}`}
              >
                {executeTrades ? 'ON' : 'OFF'}
              </button>
            </div>
          </div>

          {runMode === 'simulation' && !simulateTrading && (
            <div className="px-3 py-2 border-t border-border bg-s2">
              <p className="text-2xs text-warn">Simulation trading is off. Turn it on to run a trade-level backtest.</p>
            </div>
          )}

          {runMode === 'workflow' && executeTrades && (
            <div className="px-3 py-2 border-t border-border bg-s2">
              <p className="text-2xs text-down">Live execution is enabled for the integrated workflow.</p>
            </div>
          )}

          {/* Toolbar */}
          <div className="flex items-center gap-2 px-2 py-1.5">
            <button
              onClick={() => setTickerInput(ASX200)}
              className="text-2xs text-muted border border-border-strong px-2 py-1 hover:text-text"
            >
              Load ASX sample
            </button>
            <button
              onClick={startRun}
              disabled={!tickerInput.trim() || (runMode === 'simulation' && !simulateTrading)}
              className="ml-auto text-sm text-bg bg-accent px-3 py-1 hover:opacity-90 disabled:opacity-40 font-medium"
            >
              Run {runMode === 'simulation' ? 'Simulation' : 'Integrated'} [↵]
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
          <span className="text-sm text-text font-medium">System running…</span>
          <span className="text-2xs text-dim tabnum">{progress}%</span>
        </div>
        <ProgressBar value={progress} />
        <div className="mt-3">
          <TerminalLog messages={messages} height={280} />
        </div>
      </div>
    )
  }

  if (resultMode === 'workflow') {
    if (workflowLoading || !workflowResults) {
      return <div className="p-4 text-muted text-sm tabnum">Loading results…</div>
    }

    const { signals, tickers } = workflowResults
    const portfolio_summary = {
      total_value: Number(workflowResults.portfolio_summary?.total_value ?? 0),
      cash: Number(workflowResults.portfolio_summary?.cash ?? 0),
      invested: Number(workflowResults.portfolio_summary?.invested ?? 0),
      total_pnl: Number(workflowResults.portfolio_summary?.total_pnl ?? 0),
      total_pnl_pct: Number(workflowResults.portfolio_summary?.total_pnl_pct ?? 0),
      num_positions: Number(workflowResults.portfolio_summary?.num_positions ?? 0),
      portfolio_beta: workflowResults.portfolio_summary?.portfolio_beta != null ? Number(workflowResults.portfolio_summary.portfolio_beta) : undefined,
      max_drawdown: workflowResults.portfolio_summary?.max_drawdown != null ? Number(workflowResults.portfolio_summary.max_drawdown) : undefined,
    }
    const signalData = (signals as unknown as Array<StockResult & Record<string, unknown>>)

    return (
      <div className="flex flex-col h-full">
        <div className="flex items-center gap-3 px-4 py-2 border-b border-border bg-s1 shrink-0">
          <button onClick={reset} className="text-2xs text-dim hover:text-text">← New system run</button>
          <span className="text-border">|</span>
          <span className="text-2xs text-muted tabnum">Integrated · {tickers.length} tickers processed</span>
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
              <p className="text-2xs text-dim">Portfolio</p>
            </div>
            <MetricTile label="Total Value" value={`$${portfolio_summary.total_value.toLocaleString()}`} />
            <MetricTile label="Cash" value={`$${portfolio_summary.cash.toLocaleString()}`} />
            <MetricTile label="Invested" value={`$${portfolio_summary.invested.toLocaleString()}`} />
            <MetricTile label="Total P&L" value={`${portfolio_summary.total_pnl >= 0 ? '+' : ''}$${portfolio_summary.total_pnl.toLocaleString()}`} color={portfolio_summary.total_pnl >= 0 ? 'up' : 'down'} />
            <MetricTile label="Total P&L %" value={`${portfolio_summary.total_pnl_pct >= 0 ? '+' : ''}${portfolio_summary.total_pnl_pct.toFixed(2)}`} unit="%" color={portfolio_summary.total_pnl_pct >= 0 ? 'up' : 'down'} />
            <MetricTile label="Positions" value={portfolio_summary.num_positions} />
            {portfolio_summary.portfolio_beta !== undefined && <MetricTile label="Beta" value={portfolio_summary.portfolio_beta.toFixed(2)} />}
            {portfolio_summary.max_drawdown !== undefined && <MetricTile label="Max DD" value={portfolio_summary.max_drawdown.toFixed(2)} unit="%" color="down" />}
          </div>
        </div>
      </div>
    )
  }

  if (resultsLoading || !results) {
    return <div className="p-4 text-muted text-sm tabnum">Loading results…</div>
  }

  const raw = results
  const metrics = {
    total_return:      Number(raw.metrics?.total_return      ?? 0),
    annualized_return: raw.metrics?.annualized_return != null ? Number(raw.metrics.annualized_return) : undefined,
    sharpe_ratio:      Number(raw.metrics?.sharpe_ratio      ?? 0),
    max_drawdown:      Number(raw.metrics?.max_drawdown      ?? 0),
    win_rate:          Number(raw.metrics?.win_rate          ?? 0),
    num_trades:        Number(raw.metrics?.num_trades        ?? 0),
    profit_factor:     raw.metrics?.profit_factor != null ? Number(raw.metrics.profit_factor) : undefined,
  }
  const trades = ((raw.trades ?? []) as RawTrade[]).map(normalizeTrade)
  const equity_curve = ((raw.equity_curve ?? []) as RawEquityPoint[])
    .map(normalizeEquityPoint)
    .filter(point => point.date && Number.isFinite(point.value))
  const tradeData = (trades as unknown as Array<Trade & Record<string, unknown>>)
  const overlayTickers = raw.tickers.map(ticker => ticker.toUpperCase())
  const overlayLoading = overlayQueries.some(q => q.isLoading || q.isFetching)
  const overlays = selectedOverlayTickers
    .map((ticker, i) => buildPriceOverlay(
      ticker,
      OVERLAY_COLORS[i % OVERLAY_COLORS.length],
      overlayQueries[i]?.data,
      raw.start_date,
      raw.end_date,
    ))
    .filter((overlay): overlay is PriceOverlay => overlay !== null)

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-border bg-s1 shrink-0">
        <button onClick={reset} className="text-2xs text-dim hover:text-text">
          ← New system run
        </button>
        <span className="text-border">|</span>
        <span className="text-2xs text-muted tabnum">
          Simulation · {results.strategy} · {results.tickers.length} tickers · {results.start_date} → {results.end_date}
        </span>
      </div>

      {/* Equity curve with trade markers */}
      {equity_curve && equity_curve.length > 0 && (
        <div className="shrink-0 border-b border-border">
          <div className="flex items-center gap-2 px-3 py-1 bg-s1 border-b border-border">
            <span className="text-2xs text-dim">Portfolio Equity</span>
            <span className="text-border mx-1">|</span>
            <span className="text-2xs text-muted">
              <span className="text-up">▲ BUY</span>
              <span className="mx-2 text-border">·</span>
              <span className="text-down">▼ SELL</span>
            </span>
            {overlayTickers.length > 0 && (
              <>
                <span className="text-border mx-1">|</span>
                <div className="flex min-w-0 flex-wrap items-center gap-1">
                  <span className="text-2xs text-dim mr-1">Overlay</span>
                  {overlayTickers.map((ticker, i) => {
                    const selected = selectedOverlayTickers.includes(ticker)
                    const color = OVERLAY_COLORS[selectedOverlayTickers.indexOf(ticker) % OVERLAY_COLORS.length] ?? OVERLAY_COLORS[i % OVERLAY_COLORS.length]
                    return (
                      <button
                        key={ticker}
                        onClick={() => toggleOverlayTicker(ticker)}
                        className={[
                          'text-2xs tabnum border px-2 py-0.5 transition-colors',
                          selected
                            ? 'border-border-strong bg-s2 text-text'
                            : 'border-border text-muted hover:text-text',
                        ].join(' ')}
                        style={selected ? { color, borderColor: color } : undefined}
                        aria-pressed={selected}
                      >
                        {ticker}
                      </button>
                    )
                  })}
                  {overlayLoading && <span className="text-2xs text-dim ml-1">loading...</span>}
                </div>
              </>
            )}
          </div>
          <EquityChart equityCurve={equity_curve} trades={trades} overlays={overlays} height={240} />
        </div>
      )}

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
            <p className="text-2xs text-dim mb-1">Performance</p>
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
        </div>
      </div>
    </div>
  )
}
