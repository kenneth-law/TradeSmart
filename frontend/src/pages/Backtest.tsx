import { useState } from 'react'
import type { ReactNode } from 'react'
import { useLocation } from 'react-router-dom'
import { Activity, Database, Download, FileJson, Table2 } from 'lucide-react'
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

const DEFAULT_BUY_THRESHOLD = '50'
const DEFAULT_SELL_THRESHOLD = '40'

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
    render: r => {
      const v = Number(r.price)
      return Number.isFinite(v) ? <span className="tabnum">{v.toFixed(2)}</span> : <span className="text-dim">—</span>
    }
  },
  { key: 'shares',     label: 'SHARES',     width: 72, align: 'right',
    render: r => <span className="tabnum">{r.shares !== undefined ? String(r.shares) : '—'}</span>
  },
  { key: 'cost',       label: 'COST',       width: 80, align: 'right',
    render: r => {
      const v = Number(r.cost)
      return Number.isFinite(v) ? <span className="tabnum">${v.toFixed(2)}</span> : <span className="text-dim">—</span>
    }
  },
  { key: 'pnl',        label: 'P&L',        width: 86, align: 'right',
    render: r => {
      if (r.pnl == null) return <span className="text-dim">OPEN</span>
      const v = Number(r.pnl)
      if (!Number.isFinite(v)) return <span className="text-dim">—</span>
      return <span className={`tabnum ${v >= 0 ? 'text-up' : 'text-down'}`}>{v >= 0 ? '+' : ''}${v.toFixed(2)}</span>
    }
  },
  { key: 'return_pct', label: 'RETURN%',    width: 80, align: 'right',
    render: r => {
      if (r.return_pct == null) return <span className="text-dim">OPEN</span>
      const v = Number(r.return_pct)
      if (!Number.isFinite(v)) return <span className="text-dim">—</span>
      return <span className={`tabnum ${v >= 0 ? 'text-up' : 'text-down'}`}>{v >= 0 ? '+' : ''}{v.toFixed(2)}%</span>
    }
  },
]

const ROUND_TRIP_COLUMNS: Column<RoundTripRow & Record<string, unknown>>[] = [
  { key: 'ticker',      label: 'TICKER', width: 82 },
  { key: 'entry_date',  label: 'ENTRY',  width: 104 },
  { key: 'exit_date',   label: 'EXIT',   width: 104 },
  { key: 'entry_price', label: 'IN',     width: 76, align: 'right',
    render: r => <span className="tabnum">{Number(r.entry_price ?? 0).toFixed(2)}</span>
  },
  { key: 'exit_price',  label: 'OUT',    width: 76, align: 'right',
    render: r => <span className="tabnum">{Number(r.exit_price ?? 0).toFixed(2)}</span>
  },
  { key: 'shares',      label: 'QTY',    width: 64, align: 'right' },
  { key: 'pnl',         label: 'P&L',    width: 86, align: 'right',
    render: r => {
      const v = Number(r.pnl ?? 0)
      return <span className={`tabnum ${v >= 0 ? 'text-up' : 'text-down'}`}>{v >= 0 ? '+' : ''}${v.toFixed(2)}</span>
    }
  },
  { key: 'return_pct',  label: 'RET%',   width: 78, align: 'right',
    render: r => {
      const v = Number(r.return_pct ?? 0)
      return <span className={`tabnum ${v >= 0 ? 'text-up' : 'text-down'}`}>{v >= 0 ? '+' : ''}{v.toFixed(2)}%</span>
    }
  },
]

const TRAINING_COLUMNS: Column<TrainingPreviewRow & Record<string, unknown>>[] = [
  { key: 'ticker', label: 'TICKER', width: 82 },
  { key: 'feature_date', label: 'FEATURE DATE', width: 120 },
  { key: 'label_date', label: 'LABEL DATE', width: 110 },
  { key: 'future_return_pct', label: 'FWD RET%', width: 92, align: 'right',
    render: r => {
      const v = Number(r.future_return_pct ?? 0)
      return <span className={`tabnum ${v >= 0 ? 'text-up' : 'text-down'}`}>{v >= 0 ? '+' : ''}{v.toFixed(2)}%</span>
    }
  },
  { key: 'future_signal', label: 'LABEL', width: 72, align: 'right',
    render: r => <span className={Number(r.future_signal ?? 0) === 1 ? 'text-up' : 'text-down'}>{Number(r.future_signal ?? 0) === 1 ? 'UP' : 'DOWN'}</span>
  },
  { key: 'rsi14', label: 'RSI14', width: 72, align: 'right',
    render: r => <span className="tabnum">{Number(r.features?.rsi14 ?? 0).toFixed(1)}</span>
  },
  { key: 'atr_pct', label: 'ATR%', width: 72, align: 'right',
    render: r => <span className="tabnum">{Number(r.features?.atr_pct ?? 0).toFixed(2)}</span>
  },
  { key: 'volume_ratio', label: 'VOL R', width: 76, align: 'right',
    render: r => <span className="tabnum">{Number(r.features?.volume_ratio ?? 0).toFixed(2)}</span>
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
type ResultTab = 'trades' | 'research' | 'training' | 'export'
type ExitSizingMode = 'fixed_tranche' | 'remaining_fraction'

type RawTrade = Omit<Trade, 'type'> & {
  action?: 'BUY' | 'SELL'
  type?: 'BUY' | 'SELL'
}

type RawEquityPoint = {
  date: string
  value?: number
  equity?: number
}

type KeyValue = {
  label: string
  value: string | number
  color?: 'up' | 'down' | 'accent' | 'warn' | 'muted' | 'text'
}

type TrainingPreviewRow = {
  ticker?: string
  feature_date?: string
  label_date?: string
  prediction_horizon_days?: number
  future_return_pct?: number
  future_signal?: number
  features?: Record<string, unknown>
}

type RoundTripRow = {
  ticker?: string
  entry_date?: string
  exit_date?: string
  entry_price?: number
  exit_price?: number
  shares?: number
  pnl?: number
  return_pct?: number
}

type DrawdownPoint = {
  date: string
  drawdown: number
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

function formatMoney(value: unknown, digits = 0) {
  const number = Number(value ?? 0)
  return `$${number.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })}`
}

function formatPct(value: unknown, digits = 2) {
  const number = Number(value ?? 0)
  return `${number >= 0 ? '+' : ''}${number.toFixed(digits)}%`
}

function formatNumber(value: unknown, digits = 2) {
  return Number(value ?? 0).toFixed(digits)
}

function numberOrFallback(value: unknown, fallback: number) {
  const number = Number(value)
  return Number.isFinite(number) && number !== 0 ? number : fallback
}

function downloadJson(payload: unknown, filename: string) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

function buildFineTuningPayload(
  raw: Record<string, unknown>,
  metrics: Record<string, unknown>,
  trades: Trade[],
  equityCurve: Array<{ date: string; value: number }>,
) {
  const training = (raw.training_context ?? {}) as Record<string, unknown>
  const runMetadata = (raw.run_metadata ?? {}) as Record<string, unknown>
  const previewRows = (training.preview_rows ?? []) as TrainingPreviewRow[]
  const roundTrips = (raw.round_trips ?? []) as RoundTripRow[]

  return {
    schema_version: 'tradesmart.backtest.finetune.v1',
    generated_at: new Date().toISOString(),
    task: 'trade_strategy_research_and_model_fine_tuning',
    run: {
      backtest_id: runMetadata.backtest_id,
      strategy: raw.strategy,
      tickers: raw.tickers,
      start_date: raw.start_date,
      end_date: raw.end_date,
      metadata: runMetadata,
    },
    model_training: training,
    performance: metrics,
    risk_summary: raw.risk_summary ?? {},
    supervised_examples: previewRows.map(row => ({
      input: {
        ticker: row.ticker,
        as_of_date: row.feature_date,
        prediction_horizon_days: row.prediction_horizon_days,
        features: row.features,
      },
      output: {
        future_return_pct: row.future_return_pct,
        future_signal: row.future_signal,
        label_date: row.label_date,
      },
    })),
    trade_outcomes: roundTrips.map(trade => ({
      input: {
        ticker: trade.ticker,
        entry_date: trade.entry_date,
        entry_price: trade.entry_price,
        shares: trade.shares,
        strategy: raw.strategy,
      },
      output: {
        exit_date: trade.exit_date,
        exit_price: trade.exit_price,
        pnl: trade.pnl,
        return_pct: trade.return_pct,
      },
    })),
    raw: {
      trades,
      equity_curve: equityCurve,
      drawdown_curve: raw.drawdown_curve ?? [],
      daily_returns: raw.daily_returns ?? [],
    },
  }
}

function buildRoundTripsFromTrades(trades: Trade[]): RoundTripRow[] {
  const lots = new Map<string, Array<{ price: number; shares: number; remaining: number; cost: number; date: string }>>()
  const roundTrips: RoundTripRow[] = []

  for (const trade of trades) {
    const shares = Number(trade.shares ?? 0)
    const price = Number(trade.price ?? 0)
    const cost = Number(trade.cost ?? 0)
    if (!shares || !price) continue

    if (trade.type === 'BUY') {
      const tickerLots = lots.get(trade.ticker) ?? []
      tickerLots.push({ price, shares, remaining: shares, cost, date: trade.date })
      lots.set(trade.ticker, tickerLots)
      continue
    }

    let sharesLeft = shares
    const tickerLots = lots.get(trade.ticker) ?? []
    while (sharesLeft > 0 && tickerLots.length) {
      const lot = tickerLots[0]
      const matched = Math.min(sharesLeft, lot.remaining)
      const buyCost = lot.cost * (matched / lot.shares)
      const sellCost = cost * (matched / shares)
      const grossBuy = lot.price * matched
      const pnl = (price - lot.price) * matched - buyCost - sellCost
      roundTrips.push({
        ticker: trade.ticker,
        entry_date: lot.date,
        exit_date: trade.date,
        entry_price: lot.price,
        exit_price: price,
        shares: matched,
        pnl,
        return_pct: grossBuy ? (pnl / grossBuy) * 100 : 0,
      })
      lot.remaining -= matched
      sharesLeft -= matched
      if (lot.remaining <= 0) tickerLots.shift()
    }
  }

  return roundTrips
}

function StatStrip({ items }: { items: KeyValue[] }) {
  const colors = {
    up: 'text-up',
    down: 'text-down',
    accent: 'text-accent',
    warn: 'text-warn',
    muted: 'text-muted',
    text: 'text-text',
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 border-b border-border bg-s1">
      {items.map(item => (
        <div key={item.label} className="border-r border-border px-3 py-2 last:border-r-0">
          <p className="text-2xs text-dim">{item.label}</p>
          <p className={`tabnum text-sm font-medium ${colors[item.color ?? 'text']}`}>{item.value}</p>
        </div>
      ))}
    </div>
  )
}

function ResearchPanel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="border-b border-border">
      <div className="flex items-center justify-between bg-s1 px-3 py-2 border-b border-border">
        <h2 className="text-2xs text-dim font-medium uppercase tracking-[0.18em]">{title}</h2>
      </div>
      {children}
    </section>
  )
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
      date: String(date),
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
  const [buyThreshold, setBuyThreshold] = useState(DEFAULT_BUY_THRESHOLD)
  const [sellThreshold, setSellThreshold] = useState(DEFAULT_SELL_THRESHOLD)
  const [partialExitPct, setPartialExitPct] = useState('25')
  const [exitSizingMode, setExitSizingMode] = useState<ExitSizingMode>('fixed_tranche')
  const [reentryCooldownDays, setReentryCooldownDays] = useState('10')
  const [reentryDiscountPct, setReentryDiscountPct] = useState('1.0')
  const [allowPyramiding, setAllowPyramiding] = useState(false)
  const [simulateTrading, setSimulateTrading] = useState(true)
  const [useMl, setUseMl] = useState(true)
  const [executeTrades, setExecuteTrades] = useState(false)
  const [backtestId, setBacktestId] = useState<string | null>(null)
  const [systemId, setSystemId] = useState<string | null>(null)
  const [sseUrl, setSseUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selectedOverlayTickers, setSelectedOverlayTickers] = useState<string[]>([])
  const [activeResultTab, setActiveResultTab] = useState<ResultTab>('trades')

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
    setActiveResultTab('trades')
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
        buy_threshold: parseInt(buyThreshold) || Number(DEFAULT_BUY_THRESHOLD),
        sell_threshold: parseInt(sellThreshold) || Number(DEFAULT_SELL_THRESHOLD),
        partial_exit_fraction: (parseFloat(partialExitPct) || 25) / 100,
        exit_sizing_mode: exitSizingMode,
        reentry_cooldown_days: parseInt(reentryCooldownDays) || 0,
        min_reentry_discount_pct: Number.isFinite(parseFloat(reentryDiscountPct)) ? parseFloat(reentryDiscountPct) : 1,
        allow_pyramiding: allowPyramiding,
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
    setActiveResultTab('trades')
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
            <div className="bg-s1 p-2">
              <label className="text-2xs text-dim block mb-1">Buy Threshold</label>
              <input
                type="number"
                min="0"
                max="100"
                value={buyThreshold}
                onChange={e => setBuyThreshold(e.target.value)}
                disabled={runMode !== 'simulation' || strategy !== 'ml'}
                className="w-full bg-s2 text-text text-sm border border-border px-2 py-1 outline-none focus:border-accent tabnum"
              />
            </div>
            <div className="bg-s1 p-2">
              <label className="text-2xs text-dim block mb-1">Sell Threshold</label>
              <input
                type="number"
                min="0"
                max="100"
                value={sellThreshold}
                onChange={e => setSellThreshold(e.target.value)}
                disabled={runMode !== 'simulation' || strategy !== 'ml'}
                className="w-full bg-s2 text-text text-sm border border-border px-2 py-1 outline-none focus:border-accent tabnum"
              />
            </div>
            <div className="bg-s1 p-2">
              <label className="text-2xs text-dim block mb-1">Partial Exit %</label>
              <input
                type="number"
                min="5"
                max="100"
                step="5"
                value={partialExitPct}
                onChange={e => setPartialExitPct(e.target.value)}
                disabled={runMode !== 'simulation'}
                className="w-full bg-s2 text-text text-sm border border-border px-2 py-1 outline-none focus:border-accent tabnum"
              />
            </div>
            <div className="bg-s1 p-2">
              <label className="text-2xs text-dim block mb-1">Exit Sizing</label>
              <div className="flex gap-px border border-border bg-border">
                {([
                  ['fixed_tranche', 'Fixed Tranche'],
                  ['remaining_fraction', 'Half-Life'],
                ] as Array<[ExitSizingMode, string]>).map(([mode, label]) => (
                  <button
                    key={mode}
                    onClick={() => setExitSizingMode(mode)}
                    disabled={runMode !== 'simulation'}
                    className={[
                      'flex-1 bg-s2 px-2 py-1 text-2xs',
                      exitSizingMode === mode ? 'text-accent' : 'text-muted hover:text-text',
                    ].join(' ')}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
            <div className="bg-s1 p-2">
              <label className="text-2xs text-dim block mb-1">Re-entry Cooldown</label>
              <input
                type="number"
                min="0"
                value={reentryCooldownDays}
                onChange={e => setReentryCooldownDays(e.target.value)}
                disabled={runMode !== 'simulation'}
                className="w-full bg-s2 text-text text-sm border border-border px-2 py-1 outline-none focus:border-accent tabnum"
              />
            </div>
            <div className="bg-s1 p-2">
              <label className="text-2xs text-dim block mb-1">Re-entry Discount %</label>
              <input
                type="number"
                min="0"
                step="0.25"
                value={reentryDiscountPct}
                onChange={e => setReentryDiscountPct(e.target.value)}
                disabled={runMode !== 'simulation'}
                className="w-full bg-s2 text-text text-sm border border-border px-2 py-1 outline-none focus:border-accent tabnum"
              />
            </div>
            <div className="bg-s1 p-2 flex items-center justify-between">
              <label className="text-2xs text-dim">Allow Pyramiding</label>
              <button
                onClick={() => setAllowPyramiding(v => !v)}
                disabled={runMode !== 'simulation'}
                className={`text-2xs px-2 py-1 border ${allowPyramiding ? 'border-warn text-warn' : 'border-border text-muted hover:text-text'}`}
              >
                {allowPyramiding ? 'ON' : 'OFF'}
              </button>
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
  const trades = ((raw.trades ?? []) as RawTrade[]).map(normalizeTrade)
  const equity_curve = ((raw.equity_curve ?? []) as RawEquityPoint[])
    .map(normalizeEquityPoint)
    .filter(point => point.date && Number.isFinite(point.value))
  const inferredInitialCapital = equity_curve[0]?.value ?? 0
  const inferredFinalCapital = equity_curve[equity_curve.length - 1]?.value ?? inferredInitialCapital
  const inferredTradeValue = trades.reduce((sum, trade) => sum + Number(trade.value ?? (trade.price * (trade.shares ?? 0)) ?? 0), 0)
  const inferredCosts = trades.reduce((sum, trade) => sum + Number(trade.cost ?? 0), 0)
  const inferredBuyCount = trades.filter(trade => trade.type === 'BUY').length
  const inferredSellCount = trades.filter(trade => trade.type === 'SELL').length
  const metrics = {
    total_return:      Number(raw.metrics?.total_return      ?? 0),
    annualized_return: raw.metrics?.annualized_return != null ? Number(raw.metrics.annualized_return) : undefined,
    sharpe_ratio:      Number(raw.metrics?.sharpe_ratio      ?? 0),
    max_drawdown:      Number(raw.metrics?.max_drawdown      ?? 0),
    win_rate:          Number(raw.metrics?.win_rate          ?? 0),
    num_trades:        numberOrFallback(raw.metrics?.num_trades, trades.length),
    profit_factor:     raw.metrics?.profit_factor != null ? Number(raw.metrics.profit_factor) : undefined,
    initial_capital:   numberOrFallback(raw.metrics?.initial_capital, inferredInitialCapital),
    final_capital:     numberOrFallback(raw.metrics?.final_capital, inferredFinalCapital),
    total_transaction_costs: numberOrFallback(raw.metrics?.total_transaction_costs, inferredCosts),
    transaction_cost_percentage: Number(raw.metrics?.transaction_cost_percentage ?? 0),
    total_trade_value: numberOrFallback(raw.metrics?.total_trade_value, inferredTradeValue),
    avg_trade_value:   numberOrFallback(raw.metrics?.avg_trade_value, trades.length ? inferredTradeValue / trades.length : 0),
    buy_count:         numberOrFallback(raw.metrics?.buy_count, inferredBuyCount),
    sell_count:        numberOrFallback(raw.metrics?.sell_count, inferredSellCount),
    avg_daily_return:  Number(raw.metrics?.avg_daily_return  ?? 0),
    daily_volatility:  Number(raw.metrics?.daily_volatility  ?? 0),
    annualized_volatility: Number(raw.metrics?.annualized_volatility ?? 0),
    best_day:          Number(raw.metrics?.best_day          ?? 0),
    worst_day:         Number(raw.metrics?.worst_day         ?? 0),
  }
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
  const resultRecord = raw as unknown as Record<string, unknown>
  const trainingContext = (resultRecord.training_context ?? {}) as Record<string, unknown>
  const riskSummaryRaw = (resultRecord.risk_summary ?? {}) as Record<string, unknown>
  const runMetadata = (resultRecord.run_metadata ?? {}) as Record<string, unknown>
  const trainingRows = ((trainingContext.preview_rows ?? []) as TrainingPreviewRow[])
    .map(row => ({ ...row, ...row.features }))
  const featureImportance = Object.entries((trainingContext.feature_importance ?? {}) as Record<string, number>)
    .slice(0, 12)
  const apiRoundTrips = ((resultRecord.round_trips ?? []) as RoundTripRow[])
    .map(row => ({ ...row }))
  const roundTrips = apiRoundTrips.length ? apiRoundTrips : buildRoundTripsFromTrades(trades)
  const riskSummary: Record<string, unknown> = {
    ...riskSummaryRaw,
    completed_round_trips: numberOrFallback(riskSummaryRaw.completed_round_trips, roundTrips.length),
    winning_round_trips: numberOrFallback(riskSummaryRaw.winning_round_trips, roundTrips.filter(trade => Number(trade.pnl ?? 0) > 0).length),
    losing_round_trips: numberOrFallback(riskSummaryRaw.losing_round_trips, roundTrips.filter(trade => Number(trade.pnl ?? 0) <= 0).length),
    turnover_to_initial_capital: numberOrFallback(riskSummaryRaw.turnover_to_initial_capital, metrics.initial_capital ? metrics.total_trade_value / metrics.initial_capital : 0),
    average_transaction_cost: numberOrFallback(riskSummaryRaw.average_transaction_cost, trades.length ? metrics.total_transaction_costs / trades.length : 0),
  }
  const completedRoundTrips = Number(riskSummary.completed_round_trips ?? 0)
  const drawdownCurve = ((resultRecord.drawdown_curve ?? []) as DrawdownPoint[])
  const fineTuningPayload = buildFineTuningPayload(resultRecord, metrics, trades, equity_curve)
  const exportFilename = `tradesmart-backtest-${raw.strategy || 'strategy'}-${raw.start_date || 'start'}-${raw.end_date || 'end'}.json`
  const topStats: KeyValue[] = [
    { label: 'Total Return', value: formatPct(metrics.total_return), color: metrics.total_return >= 0 ? 'up' : 'down' },
    { label: 'Final Equity', value: formatMoney(metrics.final_capital), color: metrics.final_capital >= metrics.initial_capital ? 'up' : 'down' },
    { label: 'Sharpe', value: formatNumber(metrics.sharpe_ratio), color: metrics.sharpe_ratio >= 1 ? 'up' : metrics.sharpe_ratio >= 0 ? 'warn' : 'down' },
    { label: 'Max Drawdown', value: formatPct(-Math.abs(metrics.max_drawdown)), color: 'down' },
    { label: 'Trades', value: metrics.num_trades, color: 'text' },
    { label: 'Training Samples', value: Number(trainingContext.sample_count ?? 0).toLocaleString(), color: Number(trainingContext.sample_count ?? 0) > 0 ? 'accent' : 'muted' },
  ]

  const resultTabs: Array<{ key: ResultTab; label: string; icon: typeof Table2 }> = [
    { key: 'trades', label: 'Trades', icon: Table2 },
    { key: 'research', label: 'Research', icon: Activity },
    { key: 'training', label: 'Training Data', icon: Database },
    { key: 'export', label: 'Fine-Tune JSON', icon: FileJson },
  ]

  return (
    <div className="flex flex-col h-full">
      <div className="flex flex-wrap items-center gap-3 px-4 py-2 border-b border-border bg-s1 shrink-0">
        <button onClick={reset} className="text-2xs text-dim hover:text-text">
          ← New system run
        </button>
        <span className="text-border">|</span>
        <span className="text-2xs text-muted tabnum">
          Simulation · {results.strategy} · {results.tickers.length} tickers · {results.start_date} → {results.end_date}
        </span>
        <button
          onClick={() => downloadJson(fineTuningPayload, exportFilename)}
          className="ml-auto inline-flex items-center gap-1.5 border border-accent bg-accent px-2.5 py-1 text-2xs font-medium text-bg hover:opacity-90"
        >
          <Download size={13} aria-hidden="true" />
          Export JSON
        </button>
      </div>

      <StatStrip items={topStats} />

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
          <EquityChart equityCurve={equity_curve} trades={trades} overlays={overlays} height={220} />
        </div>
      )}

      <div className="flex items-center gap-px border-b border-border bg-border shrink-0">
        {resultTabs.map(tab => {
          const Icon = tab.icon
          return (
            <button
              key={tab.key}
              onClick={() => setActiveResultTab(tab.key)}
              className={[
                'inline-flex items-center gap-1.5 bg-s1 px-3 py-2 text-2xs',
                activeResultTab === tab.key ? 'text-accent' : 'text-muted hover:text-text',
              ].join(' ')}
            >
              <Icon size={13} aria-hidden="true" />
              {tab.label}
            </button>
          )
        })}
      </div>

      <div className="flex flex-1 overflow-hidden">
        <div className="flex-1 overflow-auto">
          {activeResultTab === 'trades' && (
            <DataTable
              columns={TRADE_COLUMNS}
              data={tradeData}
              rowKey={r => `${r.date}-${r.ticker}-${r.type}-${r.price}-${r.shares}`}
              emptyMessage="No trades recorded."
            />
          )}

          {activeResultTab === 'research' && (
            <div>
              <ResearchPanel title="Execution Diagnostics">
                <div className="grid grid-cols-2 md:grid-cols-4">
                  <MetricTile label="Initial Capital" value={formatMoney(metrics.initial_capital)} size="sm" />
                  <MetricTile label="Final Capital" value={formatMoney(metrics.final_capital)} color={metrics.final_capital >= metrics.initial_capital ? 'up' : 'down'} size="sm" />
                  <MetricTile label="Total Notional" value={formatMoney(metrics.total_trade_value)} size="sm" />
                  <MetricTile label="Avg Trade" value={formatMoney(metrics.avg_trade_value)} size="sm" />
                  <MetricTile label="Buys" value={metrics.buy_count} color="up" size="sm" />
                  <MetricTile label="Sells" value={metrics.sell_count} color="down" size="sm" />
                  <MetricTile label="Round Trips" value={Number(riskSummary.completed_round_trips ?? 0)} size="sm" />
                  <MetricTile label="Open Lots" value={Number(riskSummary.open_positions_estimate ?? 0)} color="warn" size="sm" />
                </div>
              </ResearchPanel>

              <ResearchPanel title="Risk And Costs">
                <div className="grid grid-cols-2 md:grid-cols-4">
                  <MetricTile label="Avg Daily Return" value={formatPct(metrics.avg_daily_return)} color={metrics.avg_daily_return >= 0 ? 'up' : 'down'} size="sm" />
                  <MetricTile label="Daily Vol" value={formatNumber(metrics.daily_volatility)} unit="%" color="warn" size="sm" />
                  <MetricTile label="Ann. Vol" value={formatNumber(metrics.annualized_volatility)} unit="%" color="warn" size="sm" />
                  <MetricTile label="Best Day" value={formatPct(metrics.best_day)} color="up" size="sm" />
                  <MetricTile label="Worst Day" value={formatPct(metrics.worst_day)} color="down" size="sm" />
                  <MetricTile label="Total Costs" value={formatMoney(metrics.total_transaction_costs, 2)} color="warn" size="sm" />
                  <MetricTile label="Cost Drag" value={formatNumber(metrics.transaction_cost_percentage)} unit="%" color="warn" size="sm" />
                  <MetricTile label="Turnover" value={formatNumber(Number(riskSummary.turnover_to_initial_capital ?? 0))} unit="x" size="sm" />
                </div>
              </ResearchPanel>

              <ResearchPanel title="Completed Round Trips">
                <DataTable
                  columns={ROUND_TRIP_COLUMNS}
                  data={roundTrips as Array<RoundTripRow & Record<string, unknown>>}
                  rowKey={r => `${r.ticker}-${r.entry_date}-${r.exit_date}-${r.shares}`}
                  emptyMessage="No completed round trips yet."
                />
              </ResearchPanel>

              <ResearchPanel title="Drawdown Tail">
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3">
                  {drawdownCurve.slice(-12).map(point => (
                    <div key={`${point.date}-${point.drawdown}`} className="flex items-center justify-between border-b border-r border-border px-3 py-2">
                      <span className="text-2xs text-muted tabnum">{point.date}</span>
                      <span className="text-2xs text-down tabnum">-{Math.abs(Number(point.drawdown ?? 0)).toFixed(2)}%</span>
                    </div>
                  ))}
                  {!drawdownCurve.length && <p className="p-4 text-sm text-muted">No drawdown series returned.</p>}
                </div>
              </ResearchPanel>
            </div>
          )}

          {activeResultTab === 'training' && (
            <div>
              <ResearchPanel title="Point-In-Time Training Context">
                <div className="grid grid-cols-2 md:grid-cols-4">
                  <MetricTile label="Model Trained" value={trainingContext.model_trained ? 'Yes' : 'No'} color={trainingContext.model_trained ? 'up' : 'warn'} size="sm" />
                  <MetricTile label="Samples" value={Number(trainingContext.sample_count ?? 0).toLocaleString()} color="accent" size="sm" />
                  <MetricTile label="Feature Window" value={`${trainingContext.min_feature_date ?? '—'} → ${trainingContext.max_feature_date ?? '—'}`} size="sm" />
                  <MetricTile label="Label Cutoff" value={String(trainingContext.training_end_date_exclusive ?? '—')} color="warn" size="sm" />
                  <MetricTile label="Horizon" value={Number(trainingContext.prediction_horizon_days ?? 0)} unit="days" size="sm" />
                  <MetricTile label="Positive Labels" value={Number(trainingContext.positive_samples ?? 0).toLocaleString()} color="up" size="sm" />
                  <MetricTile label="Negative Labels" value={Number(trainingContext.negative_samples ?? 0).toLocaleString()} color="down" size="sm" />
                  <MetricTile label="Persisted" value={runMetadata.model_trained ? 'Memory' : 'None'} color="muted" size="sm" />
                </div>
              </ResearchPanel>

              <ResearchPanel title="Feature Importance">
                <div className="grid grid-cols-1 md:grid-cols-2">
                  {featureImportance.map(([feature, importance]) => (
                    <div key={feature} className="flex items-center gap-3 border-b border-r border-border px-3 py-2">
                      <span className="w-44 truncate text-2xs text-muted">{feature}</span>
                      <div className="h-1.5 flex-1 bg-s2">
                        <div className="h-full bg-accent" style={{ width: `${Math.min(100, Number(importance ?? 0) * 100)}%` }} />
                      </div>
                      <span className="w-14 text-right text-2xs tabnum text-text">{Number(importance ?? 0).toFixed(3)}</span>
                    </div>
                  ))}
                  {!featureImportance.length && <p className="p-4 text-sm text-muted">Feature importance is only available when the ML model trains successfully.</p>}
                </div>
              </ResearchPanel>

              <ResearchPanel title="Training Rows Preview">
                <DataTable
                  columns={TRAINING_COLUMNS}
                  data={trainingRows as Array<TrainingPreviewRow & Record<string, unknown>>}
                  rowKey={r => `${r.ticker}-${r.feature_date}-${r.label_date}`}
                  emptyMessage="No training rows were captured for this run."
                />
              </ResearchPanel>
            </div>
          )}

          {activeResultTab === 'export' && (
            <div className="h-full overflow-auto">
              <ResearchPanel title="Fine-Tuning Export">
                <div className="flex flex-wrap items-center gap-2 px-3 py-3 border-b border-border">
                  <button
                    onClick={() => downloadJson(fineTuningPayload, exportFilename)}
                    className="inline-flex items-center gap-1.5 bg-accent px-3 py-1.5 text-2xs font-medium text-bg hover:opacity-90"
                  >
                    <Download size={13} aria-hidden="true" />
                    Download JSON
                  </button>
                  <span className="text-2xs text-muted tabnum">{exportFilename}</span>
                </div>
                <pre className="m-0 max-h-[520px] overflow-auto bg-bg p-3 text-2xs leading-5 text-muted tabnum">
                  {JSON.stringify(fineTuningPayload, null, 2)}
                </pre>
              </ResearchPanel>
            </div>
          )}
        </div>

        <div className="w-56 shrink-0 border-l border-border overflow-y-auto">
          <div className="px-3 py-2 border-b border-border">
            <p className="text-2xs text-dim mb-1">Performance Stack</p>
          </div>
          <MetricTile label="Total Return" value={formatPct(metrics.total_return)} color={metrics.total_return >= 0 ? 'up' : 'down'} />
          {metrics.annualized_return !== undefined && (
            <MetricTile label="Ann. Return" value={formatPct(metrics.annualized_return)} color={metrics.annualized_return >= 0 ? 'up' : 'down'} />
          )}
          <MetricTile label="Sharpe" value={metrics.sharpe_ratio.toFixed(2)} color={metrics.sharpe_ratio >= 1 ? 'up' : metrics.sharpe_ratio >= 0 ? 'warn' : 'down'} />
          <MetricTile label="Max Drawdown" value={metrics.max_drawdown.toFixed(2)} unit="%" color="down" />
          <MetricTile
            label="Win Rate"
            value={completedRoundTrips > 0 ? `${(metrics.win_rate * 100).toFixed(1)}` : '—'}
            unit={completedRoundTrips > 0 ? '%' : undefined}
            color={completedRoundTrips > 0 ? (metrics.win_rate >= 0.5 ? 'up' : 'down') : 'muted'}
          />
          <MetricTile label="Trades" value={metrics.num_trades} />
          <MetricTile label="Training Rows" value={Number(trainingContext.sample_count ?? 0).toLocaleString()} color="accent" />
          {metrics.profit_factor !== undefined && (
            <MetricTile label="Profit Factor" value={completedRoundTrips > 0 ? metrics.profit_factor.toFixed(2) : '—'} color={completedRoundTrips > 0 ? (metrics.profit_factor >= 1 ? 'up' : 'down') : 'muted'} />
          )}
        </div>
      </div>
    </div>
  )
}
