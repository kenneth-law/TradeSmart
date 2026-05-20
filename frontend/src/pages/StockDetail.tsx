import { useState, useEffect } from 'react'
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
  const [tickerInput, setTickerInput] = useState(ticker?.toUpperCase() ?? '')

  useEffect(() => {
    if (ticker) setTickerInput(ticker.toUpperCase())
  }, [ticker])

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

  function loadTicker() {
    const t = tickerInput.trim().toUpperCase()
    if (t && t !== ticker?.toUpperCase()) navigate(`/stock/${t}`)
  }

  // Reusable header with back button + ticker search
  function Header({ children }: { children?: React.ReactNode }) {
    return (
      <div className="flex items-center gap-3 px-4 py-2 border-b border-border bg-s1 shrink-0">
        <button
          onClick={() => navigate(-1)}
          className="text-2xs text-dim hover:text-text"
        >
          {'<-'} Back
        </button>
        <span className="text-border">|</span>

        {/* Ticker search */}
        <div className="flex items-center gap-1 border border-border-strong bg-bg px-2 py-0.5">
          <span className="text-muted text-xs select-none">{'>'}</span>
          <input
            value={tickerInput}
            onChange={e => setTickerInput(e.target.value.toUpperCase())}
            onKeyDown={e => e.key === 'Enter' && loadTicker()}
            placeholder="TICKER"
            className="w-20 bg-transparent text-text text-xs outline-none placeholder:text-dim tabnum"
            spellCheck={false}
          />
        </div>
        <button
          onClick={loadTicker}
          className="text-2xs text-bg bg-accent px-2 py-0.5 hover:opacity-90 font-medium"
        >
          GO
        </button>

        {children}
      </div>
    )
  }

  if (!ticker) {
    return (
      <div className="flex flex-col h-full">
        <Header />
        <div className="p-4 text-muted text-sm">No ticker specified. Type one above.</div>
      </div>
    )
  }

  if (stockLoading) {
    return (
      <div className="flex flex-col h-full">
        <Header />
        <div className="p-4 text-muted text-sm tabnum">Loading {ticker}...</div>
      </div>
    )
  }

  if (stockError || !stock) {
    return (
      <div className="flex flex-col h-full">
        <Header />
        <div className="p-4">
          <p className="text-down text-sm">Failed to load {ticker}.</p>
        </div>
      </div>
    )
  }

  const s = stock as StockResult

  const chartData = history && !histLoading ? {
    dates: history.dates,
    open:  history.open,
    high:  history.high,
    low:   history.low,
    close: history.close,
    ma5:   history.ma5,
    ma20:  history.ma20,
  } : null

  const change1d   = s.return_1d
  const priceColor = change1d >= 0 ? 'up' : 'down'

  return (
    <div className="flex flex-col h-full">
      <Header>
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
      </Header>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        <div className="flex-1 overflow-hidden flex flex-col">
          <div className="flex-1 min-h-0">
            <PriceChart data={chartData} height={undefined} />
          </div>

          {s.metrics && (
            <div className="border-t border-border overflow-auto shrink-0" style={{ maxHeight: 220 }}>
              {Object.entries(s.metrics).map(([category, vals]) => (
                <div key={category} className="border-b border-border">
                  <div className="px-3 py-1 bg-s2">
                    <span className="text-2xs text-dim">{category}</span>
                  </div>
                  <div className="grid grid-cols-4">
                    {Object.entries(vals).map(([k, v]) => (
                      <div key={k} className="p-2 border-b border-r border-border">
                        <div className="text-2xs text-dim">{k}</div>
                        <div className="text-sm tabnum text-text mt-0.5">{v}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="w-52 shrink-0 border-l border-border overflow-y-auto">
          <div className="p-3 border-b border-border">
            <div className="text-2xs text-dim mb-1">Day Trading Score</div>
            <ScoreBar score={s.day_trading_score} />
            <div className="tabnum text-base font-medium text-accent mt-1">{s.day_trading_score.toFixed(1)}</div>
          </div>

          <MetricTile label="Price"      value={s.current_price.toFixed(2)} color={priceColor} />
          <MetricTile label="1D Change"  value={`${change1d >= 0 ? '+' : ''}${change1d.toFixed(2)}`} unit="%" color={priceColor} />
          <MetricTile label="RSI (7)"    value={s.rsi7.toFixed(1)} color={s.rsi7 >= 70 ? 'down' : s.rsi7 <= 30 ? 'up' : 'text'} />
          <MetricTile label="ATR %"      value={s.atr_pct.toFixed(2)} unit="%" />
          <MetricTile label="Vol / Avg"  value={`${s.volume_ratio.toFixed(1)}x`} color={s.volume_ratio > 1.5 ? 'up' : 'muted'} />
          <MetricTile label="MACD Trend" value={s.macd_trend.toUpperCase()} color={s.macd_trend === 'bullish' ? 'up' : s.macd_trend === 'bearish' ? 'down' : 'muted'} />
          <MetricTile label="Above MA5"  value={s.above_ma5  ? 'YES' : 'NO'} color={s.above_ma5  ? 'up' : 'down'} />
          <MetricTile label="Above MA20" value={s.above_ma20 ? 'YES' : 'NO'} color={s.above_ma20 ? 'up' : 'down'} />
          <MetricTile label="Sentiment"  value={s.news_sentiment_label} />
          <MetricTile label="Sent. Score" value={s.news_sentiment_score.toFixed(3)} />

          {peers && peers.length > 0 && (
            <div className="border-t border-border mt-2">
              <div className="px-3 py-2">
                <p className="text-2xs text-dim mb-2">Industry Peers</p>
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
              <p className="text-2xs text-dim mb-1">Strategy</p>
              <p className="text-2xs text-muted leading-relaxed">{s.strategy_details}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
