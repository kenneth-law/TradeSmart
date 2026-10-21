import { useState, useEffect, useRef } from 'react'
import { useParams, useNavigate, useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Bookmark, Network } from 'lucide-react'
import { api } from '../lib/api'
import PriceChart from '../components/charts/PriceChart'
import SignalBadge from '../components/data/SignalBadge'
import ScoreBar from '../components/data/ScoreBar'
import StrategyBrief from '../components/strategy/StrategyBrief'
import ChatPanel from '../components/strategy/ChatPanel'
import Financials from '../components/financials/Financials'
import type { StockResult, DetailedMetrics } from '../types'
import { addRecentSearch, isInWatchlist, toggleWatchlist } from '../lib/userPrefs'
import { useAppStore } from '../store/useAppStore'
import PaperTradeTicket from '../components/trading/PaperTradeTicket'
import { quotePrice, useLiveQuotes } from '../hooks/useLiveQuotes'
import SupplyChain from './SupplyChain'
import EquityResearchWorkspace, { type ResearchCategory } from '../components/research/EquityResearchWorkspace'

type DetailTab = 'overview' | 'supply-chain' | ResearchCategory

const DETAIL_TABS: Array<[DetailTab, string]> = [
  ['overview', 'Overview'],
  ['supply-chain', 'TradeSmart Intelligence'],
  ['markets', 'Market View'],
  ['screening', 'Screening'],
  ['fundamentals', 'Fundamentals'],
  ['estimates', 'Estimates'],
  ['valuation', 'Valuation'],
  ['ownership', 'Ownership'],
  ['research', 'Research'],
  ['charts', 'Charts'],
  ['testing', 'Testing'],
]

function n(v: unknown, dp = 2): string {
  return typeof v === 'number' ? v.toFixed(dp) : '-'
}
function pct(v: unknown, dp = 2): string {
  if (typeof v !== 'number') return '-'
  return `${v >= 0 ? '+' : ''}${v.toFixed(dp)}%`
}
function yn(v: unknown): string { return v ? 'YES' : 'NO' }
function rsiColor(v: unknown): string {
  if (typeof v !== 'number') return 'text-text'
  return v >= 70 ? 'text-down' : v <= 30 ? 'text-up' : 'text-text'
}
function retColor(v: unknown): string {
  if (typeof v !== 'number') return 'text-text'
  return v >= 0 ? 'text-up' : 'text-down'
}
function KV({ label, value, cls = 'text-text' }: { label: string; value: string; cls?: string }) {
  return (
    <div>
      <div className="text-2xs text-dim mb-0.5">{label}</div>
      <div className={`tabnum text-xs font-medium ${cls}`}>{value}</div>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="flex-1 min-w-0 p-3">
      <div className="text-2xs uppercase tracking-[0.2em] text-dim mb-3 pb-1.5 border-b border-border">
        {title}
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-2.5">
        {children}
      </div>
    </div>
  )
}

function TickerHeader({
  tickerInput,
  onTickerInputChange,
  onLoadTicker,
  onBack,
  children,
}: {
  tickerInput: string
  onTickerInputChange: (value: string) => void
  onLoadTicker: () => void
  onBack: () => void
  children?: React.ReactNode
}) {
  return (
    <div className="flex items-center gap-3 px-4 py-2 border-b border-border bg-s1/60 shrink-0 backdrop-blur-sm">
      <button
        onClick={onBack}
        className="text-2xs text-dim hover:text-text transition-colors"
      >
        {'<-'} Back
      </button>
      <span className="text-border">|</span>
      <div className="flex items-center gap-1 border border-border-strong bg-bg/80 px-2 py-0.5">
        <span className="text-dim text-xs select-none">{'>'}</span>
        <input
          value={tickerInput}
          onChange={e => onTickerInputChange(e.target.value.toUpperCase())}
          onKeyDown={e => e.key === 'Enter' && onLoadTicker()}
          placeholder="TICKER"
          className="w-20 bg-transparent text-text text-xs outline-none placeholder:text-dim tabnum"
          spellCheck={false}
        />
      </div>
      <button
        onClick={onLoadTicker}
        className="text-2xs text-bg bg-accent px-2 py-0.5 hover:opacity-90 font-semibold"
      >
        GO
      </button>
      {children}
    </div>
  )
}

export default function StockDetail() {
  const { ticker } = useParams<{ ticker: string }>()
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const settings = useAppStore(s => s.settings)
  const [tickerInput, setTickerInput] = useState(ticker?.toUpperCase() ?? '')
  const [watched, setWatched] = useState(false)
  const [chatOpen, setChatOpen] = useState(false)
  const [bottomTab, setBottomTab] = useState<'technicals' | 'financials'>('technicals')
  const [stripHeight, setStripHeight] = useState<number | null>(null)
  const stripRef = useRef<HTMLDivElement | null>(null)
  const requestedTab = searchParams.get('tab')
  const detailTab = DETAIL_TABS.some(([tab]) => tab === requestedTab) ? requestedTab as DetailTab : 'overview'

  function setDetailTab(tab: DetailTab) {
    const next = new URLSearchParams(searchParams)
    if (tab === 'overview') next.delete('tab')
    else next.set('tab', tab)
    setSearchParams(next, { replace: true })
  }

  function startStripDrag(e: React.MouseEvent) {
    e.preventDefault()
    const startY = e.clientY
    const startHeight = stripRef.current?.getBoundingClientRect().height ?? 140
    function onMove(ev: MouseEvent) {
      const delta = startY - ev.clientY
      const next = Math.max(80, Math.min(800, startHeight + delta))
      setStripHeight(next)
    }
    function onUp() {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

  useEffect(() => {
    if (ticker) {
      setTickerInput(ticker.toUpperCase())
      setWatched(isInWatchlist(ticker.toUpperCase()))
      addRecentSearch(ticker.toUpperCase())
      setChatOpen(false)
    }
  }, [ticker])

  const { data: stock, isLoading: stockLoading, error: stockError } = useQuery({
    queryKey: ['stock', ticker],
    queryFn: () => api.getStock(ticker!),
    enabled: !!ticker,
    refetchInterval: settings.marketRefreshSeconds * 1000,
  })
  const live = useLiveQuotes([ticker], settings.fastPaperRefresh && !!ticker)

  const { data: peers } = useQuery({
    queryKey: ['peers', ticker],
    queryFn: () => api.getIndustryPeers(ticker!),
    enabled: !!ticker,
  })

  function loadTicker() {
    const t = tickerInput.trim().toUpperCase()
    if (t && t !== ticker?.toUpperCase()) navigate(`/stock/${t}`)
  }

  if (!ticker) {
    return (
      <div className="flex flex-col h-full">
        <TickerHeader
          tickerInput={tickerInput}
          onTickerInputChange={setTickerInput}
          onLoadTicker={loadTicker}
          onBack={() => navigate(-1)}
        />
        <div className="p-6 text-muted text-sm">No ticker specified. Type one above.</div>
      </div>
    )
  }

  if (stockLoading) {
    return (
      <div className="flex flex-col h-full">
        <TickerHeader
          tickerInput={tickerInput}
          onTickerInputChange={setTickerInput}
          onLoadTicker={loadTicker}
          onBack={() => navigate(-1)}
        />
        <div className="p-6 text-muted text-sm tabnum">Loading {ticker}...</div>
      </div>
    )
  }

  if (stockError || !stock) {
    return (
      <div className="flex flex-col h-full">
        <TickerHeader
          tickerInput={tickerInput}
          onTickerInputChange={setTickerInput}
          onLoadTicker={loadTicker}
          onBack={() => navigate(-1)}
        />
        <div className="p-6 text-down text-sm">Failed to load {ticker}.</div>
      </div>
    )
  }

  const s = stock as StockResult
  const m = (s.metrics ?? {}) as DetailedMetrics
  const peerTickers = peers?.map(p => p.ticker) ?? []
  const liveTicker = ticker.toUpperCase()
  const livePrice = quotePrice(live.quotes[liveTicker])
  const currentPrice = livePrice ?? s.current_price
  const tradableStock: StockResult = livePrice != null ? { ...s, current_price: currentPrice } : s
  const change1d = s.return_1d
  const updown = change1d >= 0 ? 'text-up' : 'text-down'

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <TickerHeader
        tickerInput={tickerInput}
        onTickerInputChange={setTickerInput}
        onLoadTicker={loadTicker}
        onBack={() => navigate(-1)}
      >
        <span className="text-border">|</span>
        <span className="text-sm font-semibold tabnum text-text">{ticker}</span>
        {s.company_name && (
          <span className="text-2xs text-muted truncate max-w-48">{s.company_name}</span>
        )}
        {s.sector && (
          <>
            <span className="text-border hidden sm:block">|</span>
            <span className="text-2xs text-dim hidden sm:block">{s.sector}</span>
          </>
        )}
        <div className="ml-auto flex items-center gap-5">
          <div className="flex items-baseline gap-2">
            <span className="tabnum text-base font-semibold text-text">
              ${currentPrice.toFixed(2)}
            </span>
            <span className={`tabnum text-sm font-medium ${updown}`}>
              {change1d >= 0 ? '+' : ''}{change1d.toFixed(2)}%
            </span>
            {settings.fastPaperRefresh && (
              <span className={`text-2xs tabnum ${livePrice != null ? 'text-accent' : 'text-muted'}`}>
                {live.configured ? (livePrice != null ? `alpaca ${live.feed || 'live'}` : 'alpaca waiting') : 'alpaca not configured'}
              </span>
            )}
          </div>
          <SignalBadge signal={s.day_trading_strategy} />
          <button
            onClick={() => setDetailTab('supply-chain')}
            title="Research suppliers, customers and competitors"
            className="flex items-center gap-1 text-2xs text-dim transition-colors hover:text-accent"
          >
            <Network size={13} strokeWidth={1.8} />
            <span>TradeSmart Intelligence</span>
          </button>
          <button
            onClick={() => setWatched(toggleWatchlist(ticker!.toUpperCase()))}
            title={watched ? 'Remove from watchlist' : 'Add to watchlist'}
            className={`flex items-center gap-1 text-2xs transition-colors ${watched ? 'text-accent' : 'text-dim hover:text-text'}`}
          >
            <Bookmark size={13} fill={watched ? 'currentColor' : 'none'} strokeWidth={1.8} />
            <span>{watched ? 'Watching' : 'Watch'}</span>
          </button>
        </div>
      </TickerHeader>

      <div className="flex shrink-0 items-center overflow-x-auto border-b border-border bg-s1/40 px-3">
        {DETAIL_TABS.map(([tab, label]) => (
          <button
            key={tab}
            type="button"
            onClick={() => setDetailTab(tab)}
            className={`-mb-px border-b-2 px-4 py-2 text-2xs uppercase tracking-[0.18em] transition-colors ${detailTab === tab ? 'border-accent text-text' : 'border-transparent text-dim hover:text-text'}`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Body */}
      {detailTab === 'supply-chain' ? (
        <div className="min-h-0 flex-1 overflow-hidden">
          <SupplyChain embedded tickerOverride={ticker.toUpperCase()} />
        </div>
      ) : detailTab !== 'overview' ? (
        <div className="min-h-0 flex-1 overflow-hidden">
          <EquityResearchWorkspace category={detailTab} ticker={ticker.toUpperCase()} stock={tradableStock} peerTickers={peerTickers} onOpenIntelligence={intelligenceView => {
            const next = new URLSearchParams(searchParams)
            next.set('tab', 'supply-chain')
            if (intelligenceView) next.set('intel', intelligenceView)
            setSearchParams(next, { replace: true })
          }} />
        </div>
      ) : <div className="flex flex-1 overflow-hidden">
        {/* Left: chart + metrics strip */}
        <div className="flex-1 overflow-hidden flex flex-col min-w-0">
          <div className="flex-1 min-h-0">
            <PriceChart ticker={ticker!} overlayPeers={peerTickers} />
          </div>

          {/* Bottom strip: drag handle + tabs (Technicals / Financials) */}
          <div
            onMouseDown={startStripDrag}
            title="Drag to resize"
            className="shrink-0 h-1.5 cursor-row-resize border-t border-border bg-border/40 hover:bg-accent/50 transition-colors"
          />
          <div className="shrink-0 flex items-center gap-0 border-b border-border bg-s1/60 px-2">
            {(['technicals', 'financials'] as const).map(t => (
              <button
                key={t}
                type="button"
                onClick={() => setBottomTab(t)}
                className={[
                  'px-3 py-1.5 text-2xs uppercase tracking-[0.2em] border-b-2 -mb-px transition-colors',
                  bottomTab === t
                    ? 'border-accent text-text'
                    : 'border-transparent text-dim hover:text-text',
                ].join(' ')}
              >
                {t}
              </button>
            ))}
          </div>
          <div
            ref={stripRef}
            className="shrink-0 bg-s1/50 overflow-hidden flex flex-col"
            style={{ height: stripHeight ?? 200 }}
          >
            {bottomTab === 'technicals' ? (
              <div className="flex-1 min-h-0 overflow-y-auto flex divide-x divide-border">
                <Section title="Technical">
                  <KV label="RSI (7)"    value={n(m.technical?.rsi7, 1)}    cls={rsiColor(m.technical?.rsi7)} />
                  <KV label="RSI (14)"   value={n(m.technical?.rsi14, 1)}   cls={rsiColor(m.technical?.rsi14)} />
                  <KV label="MACD"       value={(m.technical?.macd_trend ?? '-').toUpperCase()} cls={m.technical?.macd_trend === 'bullish' ? 'text-up' : m.technical?.macd_trend === 'bearish' ? 'text-down' : 'text-muted'} />
                  <KV label="BB Pos"     value={n(m.technical?.bb_position, 2)} />
                  <KV label="Above MA5"  value={yn(m.technical?.above_ma5)}  cls={m.technical?.above_ma5  ? 'text-up' : 'text-down'} />
                  <KV label="Above MA10" value={yn(m.technical?.above_ma10)} cls={m.technical?.above_ma10 ? 'text-up' : 'text-down'} />
                  <KV label="Above MA20" value={yn(m.technical?.above_ma20)} cls={m.technical?.above_ma20 ? 'text-up' : 'text-down'} />
                </Section>

                <Section title="Volatility">
                  <KV label="ATR"         value={n(m.volatility?.atr)} />
                  <KV label="ATR %"       value={`${n(m.volatility?.atr_pct)}%`} />
                  <KV label="Intraday Rng" value={`${n(m.volatility?.avg_intraday_range)}%`} />
                  <KV label="Gaps Up 5D"  value={n(m.volatility?.gap_ups_5d, 0)} cls="text-up" />
                  <KV label="Gaps Dn 5D"  value={n(m.volatility?.gap_downs_5d, 0)} cls="text-down" />
                </Section>

                <Section title="Momentum">
                  <KV label="Return 1D" value={pct(m.momentum?.return_1d)} cls={retColor(m.momentum?.return_1d)} />
                  <KV label="Return 3D" value={pct(m.momentum?.return_3d)} cls={retColor(m.momentum?.return_3d)} />
                  <KV label="Return 5D" value={pct(m.momentum?.return_5d)} cls={retColor(m.momentum?.return_5d)} />
                </Section>

                <Section title="Volume / Sentiment">
                  <KV label="Vol / Avg"   value={`${n(m.volume?.volume_ratio, 1)}x`} cls={(m.volume?.volume_ratio ?? 0) > 1.5 ? 'text-up' : 'text-text'} />
                  <KV label="Sentiment"   value={m.sentiment?.news_sentiment_label ?? '-'} />
                  <KV label="Sent. Score" value={n(m.sentiment?.news_sentiment_score, 3)} />
                </Section>
              </div>
            ) : (
              <Financials ticker={ticker!} />
            )}
          </div>
        </div>

        {/* Sidebar */}
        <div className="w-[390px] xl:w-[430px] shrink-0 border-l border-border flex flex-col bg-s1/40 min-h-0 overflow-hidden">
          {/* Scrollable upper area: score, quick stats, peers, brief */}
          <div className={`${chatOpen ? 'shrink-0 basis-1/2' : 'flex-1'} min-h-0 overflow-y-auto`}>
            {/* Score */}
            <div className="p-4 border-b border-border">
              <div className="text-2xs uppercase tracking-[0.2em] text-dim mb-2">Trading Score</div>
              <ScoreBar score={s.day_trading_score} />
              <div className="tabnum text-2xl font-bold text-accent mt-2 mb-3">
                {s.day_trading_score.toFixed(1)}
              </div>
              <SignalBadge signal={s.day_trading_strategy} />
            </div>

            {/* Quick stats */}
            <div className="p-3 border-b border-border">
              <div className="text-2xs uppercase tracking-[0.2em] text-dim mb-2">Quick Stats</div>
              <div className="flex flex-col gap-2">
                {[
                  { label: 'Price',    value: `$${currentPrice.toFixed(2)}`, cls: 'text-text' },
                  { label: '1D',       value: `${change1d >= 0 ? '+' : ''}${change1d.toFixed(2)}%`, cls: updown },
                  { label: 'ATR %',    value: `${s.atr_pct.toFixed(2)}%`, cls: 'text-text' },
                  { label: 'Vol/Avg',  value: `${s.volume_ratio.toFixed(1)}x`, cls: s.volume_ratio > 1.5 ? 'text-up' : 'text-text' },
                ].map(({ label, value, cls }) => (
                  <div key={label} className="flex justify-between items-center">
                    <span className="text-2xs text-dim">{label}</span>
                    <span className={`tabnum text-xs font-medium ${cls}`}>{value}</span>
                  </div>
                ))}
              </div>
            </div>

            <PaperTradeTicket stock={tradableStock} />

            {/* Peers */}
            {peers && peers.length > 0 && (
              <div className="p-3 border-b border-border">
                <div className="text-2xs uppercase tracking-[0.2em] text-dim mb-2">Industry Peers</div>
                <div className="flex flex-col">
                  {peers.slice(0, 8).map(p => (
                    <div
                      key={p.ticker}
                      className="flex justify-between items-center py-1 cursor-pointer group"
                      onClick={() => navigate(`/stock/${p.ticker}`)}
                      role="button"
                      tabIndex={0}
                      onKeyDown={e => e.key === 'Enter' && navigate(`/stock/${p.ticker}`)}
                    >
                      <span className="text-2xs text-muted group-hover:text-accent transition-colors tabnum">
                        {p.ticker}
                      </span>
                      <span className="text-2xs tabnum text-dim">
                        {p.day_trading_score.toFixed(0)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Strategy brief (AI) */}
            <StrategyBrief
              stock={s}
              chatOpen={chatOpen}
              onOpenChat={() => setChatOpen(true)}
            />
          </div>

          {/* Chat panel (mounted only when open) */}
          {chatOpen && (
            <ChatPanel stock={s} onClose={() => setChatOpen(false)} />
          )}
        </div>
      </div>}
    </div>
  )
}
