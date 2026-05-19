import { useParams, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import PriceChart from '../components/charts/PriceChart'
import MetricTile from '../components/data/MetricTile'
import SignalBadge from '../components/data/SignalBadge'
import ScoreBar from '../components/data/ScoreBar'
import type { StockResult } from '../types'

export default function StockDetail() {
  const { ticker } = useParams<{ ticker: string }>()
  const navigate = useNavigate()

  const { data: stock, isLoading: stockLoading, error: stockError } = useQuery({
    queryKey: ['stock', ticker],
    queryFn: () => api.getStock(ticker!),
    enabled: !!ticker,
  })

  const { data: history, isLoading: histLoading } = useQuery({
    queryKey: ['priceHistory', ticker],
    queryFn: () => api.getPriceHistory(ticker!),
    enabled: !!ticker,
  })

  const { data: peers } = useQuery({
    queryKey: ['peers', ticker],
    queryFn: () => api.getIndustryPeers(ticker!),
    enabled: !!ticker,
  })

  if (!ticker) {
    return <div className="p-4 text-muted text-sm">No ticker specified.</div>
  }

  if (stockLoading) {
    return (
      <div className="p-4">
        <div className="text-muted text-sm tabnum">Loading {ticker}…</div>
      </div>
    )
  }

  if (stockError || !stock) {
    return (
      <div className="p-4">
        <p className="text-down text-sm">Failed to load {ticker}.</p>
        <button onClick={() => navigate(-1)} className="mt-2 text-2xs text-muted hover:text-text">
          ← Back
        </button>
      </div>
    )
  }

  const s = stock as StockResult

  const chartData = history && !histLoading ? {
    dates: history.dates,
    open: history.open,
    high: history.high,
    low: history.low,
    close: history.close,
    ma5: history.ma5,
    ma20: history.ma20,
  } : null

  const change1d = s.return_1d
  const priceColor = change1d >= 0 ? 'up' : 'down'

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-border bg-s1 shrink-0">
        <button
          onClick={() => navigate(-1)}
          className="text-2xs text-dim hover:text-text"
        >
          ← Back
        </button>
        <span className="text-border">|</span>
        <span className="text-sm text-text font-medium tabnum">{ticker}</span>
        {s.company_name && (
          <span className="text-2xs text-muted">{s.company_name}</span>
        )}
        <span className="text-border">|</span>
        <span className="text-2xs text-muted tabnum">{s.sector}</span>
        <div className="ml-auto flex items-center gap-4">
          <span className="tabnum text-sm text-text">{s.current_price.toFixed(2)}</span>
          <span className={`tabnum text-sm ${change1d >= 0 ? 'text-up' : 'text-down'}`}>
            {change1d >= 0 ? '+' : ''}{change1d.toFixed(2)}%
          </span>
          <SignalBadge signal={s.day_trading_strategy} />
        </div>
      </div>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: Chart */}
        <div className="flex-1 overflow-hidden flex flex-col">
          <div className="flex-1 min-h-0">
            <PriceChart data={chartData} height={undefined} />
          </div>
          {/* Technical indicators strip */}
          {s.metrics && (
            <div className="border-t border-border overflow-auto shrink-0" style={{ maxHeight: 220 }}>
              {Object.entries(s.metrics).map(([category, vals]) => (
                <div key={category} className="border-b border-border">
                  <div className="px-3 py-1 bg-s2">
                    <span className="text-2xs text-dim uppercase tracking-wide">{category}</span>
                  </div>
                  <div className="grid grid-cols-4">
                    {Object.entries(vals).map(([k, v]) => (
                      <div key={k} className="p-2 border-b border-r border-border">
                        <div className="text-2xs text-dim uppercase tracking-wide">{k}</div>
                        <div className="text-sm tabnum text-text mt-0.5">{v}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right: Metrics sidebar */}
        <div className="w-52 shrink-0 border-l border-border overflow-y-auto">
          {/* Score */}
          <div className="p-3 border-b border-border">
            <div className="text-2xs text-dim uppercase tracking-wide mb-1">Day Trading Score</div>
            <ScoreBar score={s.day_trading_score} />
            <div className="tabnum text-base font-medium text-accent mt-1">{s.day_trading_score.toFixed(1)}</div>
          </div>

          <MetricTile label="Price" value={s.current_price.toFixed(2)} color={priceColor} />
          <MetricTile label="1D Change" value={`${change1d >= 0 ? '+' : ''}${change1d.toFixed(2)}`} unit="%" color={priceColor} />
          <MetricTile label="RSI (7)" value={s.rsi7.toFixed(1)} color={s.rsi7 >= 70 ? 'down' : s.rsi7 <= 30 ? 'up' : 'text'} />
          <MetricTile label="ATR %" value={s.atr_pct.toFixed(2)} unit="%" />
          <MetricTile label="Vol / Avg" value={`${s.volume_ratio.toFixed(1)}x`} color={s.volume_ratio > 1.5 ? 'up' : 'muted'} />
          <MetricTile label="MACD Trend" value={s.macd_trend.toUpperCase()} color={s.macd_trend === 'bullish' ? 'up' : s.macd_trend === 'bearish' ? 'down' : 'muted'} />
          <MetricTile label="Above MA5" value={s.above_ma5 ? 'YES' : 'NO'} color={s.above_ma5 ? 'up' : 'down'} />
          <MetricTile label="Above MA20" value={s.above_ma20 ? 'YES' : 'NO'} color={s.above_ma20 ? 'up' : 'down'} />
          <MetricTile label="Sentiment" value={s.news_sentiment_label} />
          <MetricTile label="Sent. Score" value={s.news_sentiment_score.toFixed(3)} />

          {/* Peers */}
          {peers && peers.length > 0 && (
            <div className="border-t border-border mt-2">
              <div className="px-3 py-2">
                <p className="text-2xs text-dim uppercase tracking-wide mb-2">Industry Peers</p>
                {peers.slice(0, 8).map(p => (
                  <div
                    key={p.ticker}
                    className="flex justify-between items-center py-0.5 cursor-pointer hover:text-text"
                    onClick={() => navigate(`/stock/${p.ticker}`)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={e => e.key === 'Enter' && navigate(`/stock/${p.ticker}`)}
                  >
                    <span className="text-2xs text-muted tabnum">{p.ticker}</span>
                    <span className="text-2xs tabnum text-text">{p.day_trading_score.toFixed(0)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {s.strategy_details && (
            <div className="border-t border-border px-3 py-2">
              <p className="text-2xs text-dim uppercase tracking-wide mb-1">Strategy</p>
              <p className="text-2xs text-muted leading-relaxed">{s.strategy_details}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
