import { useNavigate } from 'react-router-dom'
import { useAppStore } from '../store/useAppStore'
import DataTable from '../components/data/DataTable'
import type { Column } from '../components/data/DataTable'
import SignalBadge from '../components/data/SignalBadge'
import ScoreBar from '../components/data/ScoreBar'
import DistributionChart from '../components/charts/DistributionChart'
import type { StockResult } from '../types'

const STRATEGY_ORDER = ['Strong Buy', 'Buy', 'Neutral/Watch', 'Sell', 'Strong Sell']

const COLUMNS: Column<StockResult & Record<string, unknown>>[] = [
  { key: 'rank',                label: 'RANK',      width: 52,  align: 'right', sortable: false },
  { key: 'ticker',              label: 'TICKER',    width: 80 },
  {
    key: 'day_trading_score',
    label: 'SCORE',
    width: 120,
    align: 'right',
    render: (r) => <ScoreBar score={r.day_trading_score as number} />,
  },
  {
    key: 'day_trading_strategy',
    label: 'SIGNAL',
    width: 110,
    render: (r) => <SignalBadge signal={r.day_trading_strategy as StockResult['day_trading_strategy']} />,
  },
  {
    key: 'current_price',
    label: 'PRICE',
    width: 80,
    align: 'right',
    render: (r) => <span className="tabnum">{(r.current_price as number).toFixed(2)}</span>,
  },
  {
    key: 'return_1d',
    label: 'CHG%',
    width: 72,
    align: 'right',
    render: (r) => {
      const v = r.return_1d as number
      return (
        <span className={`tabnum ${v >= 0 ? 'text-up' : 'text-down'}`}>
          {v >= 0 ? '+' : ''}{v.toFixed(2)}%
        </span>
      )
    },
  },
  {
    key: 'rsi7',
    label: 'RSI',
    width: 60,
    align: 'right',
    render: (r) => <span className="tabnum">{(r.rsi7 as number).toFixed(1)}</span>,
  },
  {
    key: 'atr_pct',
    label: 'ATR%',
    width: 64,
    align: 'right',
    render: (r) => <span className="tabnum">{(r.atr_pct as number).toFixed(2)}</span>,
  },
  {
    key: 'volume_ratio',
    label: 'VOL/AVG',
    width: 76,
    align: 'right',
    render: (r) => <span className="tabnum">{(r.volume_ratio as number).toFixed(1)}x</span>,
  },
  {
    key: 'macd_trend',
    label: 'MACD',
    width: 72,
    render: (r) => {
      const v = r.macd_trend as string
      return (
        <span className={v === 'bullish' ? 'text-up' : v === 'bearish' ? 'text-down' : 'text-muted'}>
          {v.toUpperCase()}
        </span>
      )
    },
  },
  {
    key: 'news_sentiment_label',
    label: 'SENTIMENT',
    width: 90,
    render: (r) => <span className="text-muted text-2xs">{String(r.news_sentiment_label)}</span>,
  },
]

export default function Results() {
  const navigate = useNavigate()
  const result = useAppStore(s => s.analysisResult)
  const setTickerContext = useAppStore(s => s.setTickerContext)

  if (!result) {
    return (
      <div className="p-4">
        <p className="text-muted text-sm">
          No analysis results. Return to{' '}
          <button onClick={() => navigate('/')} className="text-accent hover:underline">
            Dashboard
          </button>{' '}
          and run an analysis first.
        </p>
      </div>
    )
  }

  const { ranked_stocks, failed_tickers, session_id, csv_url } = result
  const tableData = ranked_stocks.map((s, i) => ({ ...s, rank: i + 1 } as StockResult & Record<string, unknown>))

  const distData = STRATEGY_ORDER.map(name => ({
    name,
    value: ranked_stocks.filter(s => s.day_trading_strategy === name).length,
  }))

  function onRowClick(row: StockResult & Record<string, unknown>) {
    setTickerContext({
      ticker: row.ticker,
      price: row.current_price as number,
      change: row.return_1d as number,
      score: row.day_trading_score as number,
    })
    navigate(`/stock/${row.ticker}`)
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header bar */}
      <div className="flex items-center gap-4 px-4 py-2 border-b border-border bg-s1 shrink-0">
        <button
          onClick={() => navigate('/')}
          className="text-2xs text-dim hover:text-text"
        >
          ← New analysis
        </button>
        <span className="text-border">|</span>
        <span className="text-2xs text-muted tabnum">
          {ranked_stocks.length} stocks · session {session_id}
        </span>
        {failed_tickers.length > 0 && (
          <>
            <span className="text-border">|</span>
            <span className="text-2xs text-warn tabnum">
              {failed_tickers.length} failed: {failed_tickers.slice(0, 5).join(', ')}
              {failed_tickers.length > 5 ? '…' : ''}
            </span>
          </>
        )}
        {csv_url && (
          <a
            href={csv_url}
            download
            className="ml-auto text-2xs text-muted border border-border-strong px-2 py-1 hover:text-text"
          >
            Export CSV
          </a>
        )}
      </div>

      {/* Two-column: table + sidebar */}
      <div className="flex flex-1 overflow-hidden">
        {/* Main table */}
        <div className="flex-1 overflow-auto">
          <DataTable
            columns={COLUMNS}
            data={tableData}
            rowKey={r => r.ticker}
            onRowClick={onRowClick}
            virtualRows={ranked_stocks.length > 50}
            emptyMessage="No stocks in results."
          />
        </div>

        {/* Sidebar */}
        <div className="w-56 shrink-0 border-l border-border overflow-y-auto">
          {/* Summary counts */}
          <div className="px-3 py-2 border-b border-border">
            <p className="text-2xs text-dim uppercase tracking-wide mb-2">Signal breakdown</p>
            {STRATEGY_ORDER.map(name => {
              const count = ranked_stocks.filter(s => s.day_trading_strategy === name).length
              return (
                <div key={name} className="flex justify-between items-center py-0.5">
                  <span className="text-2xs text-muted">{name}</span>
                  <span className="text-2xs tabnum text-text">{count}</span>
                </div>
              )
            })}
          </div>

          {/* Distribution chart */}
          <div className="px-3 py-2 border-b border-border">
            <p className="text-2xs text-dim uppercase tracking-wide mb-2">Distribution</p>
            <DistributionChart data={distData} height={120} />
          </div>

          {/* Keyboard hints */}
          <div className="px-3 py-2">
            <p className="text-2xs text-dim uppercase tracking-wide mb-2">Keyboard</p>
            <div className="text-2xs text-dim space-y-0.5">
              <div>↑↓ navigate rows</div>
              <div>Enter open detail</div>
              <div>/ focus search</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
