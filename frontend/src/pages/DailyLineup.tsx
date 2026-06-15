import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer, ReferenceLine } from 'recharts'
import { RefreshCw, Settings, Sparkles } from 'lucide-react'
import { api } from '../lib/api'
import { OpenAIError, streamDailyLineup, type Citation } from '../lib/openai'
import { getWatchlist } from '../lib/userPrefs'
import { useAppStore } from '../store/useAppStore'
import MetricTile from '../components/data/MetricTile'
import TickerSearchChart from '../components/charts/TickerSearchChart'
import Markdown from '../components/strategy/Markdown'

type Period = '1d' | '1w' | '1m'

const PERIOD_KEY: Record<Period, 'return_1d' | 'return_1w' | 'return_1m'> = {
  '1d': 'return_1d',
  '1w': 'return_1w',
  '1m': 'return_1m',
}

function uniqueTickers(values: string[]) {
  return Array.from(new Set(values.map(v => v.trim().toUpperCase()).filter(Boolean))).sort()
}

export default function DailyLineup() {
  const [period, setPeriod] = useState<Period>('1d')
  const [watchlist, setWatchlist] = useState<string[]>([])
  const [lineup, setLineup] = useState('')
  const [citations, setCitations] = useState<Citation[]>([])
  const [aiError, setAiError] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [isSearching, setIsSearching] = useState(false)
  const autoStarted = useRef(false)
  const abortRef = useRef<AbortController | null>(null)

  const settings = useAppStore(s => s.settings)
  const openaiKey = useAppStore(s => s.openaiKey)
  const account = useAppStore(s => s.paperAccount)
  const marketRefreshSeconds = settings.marketRefreshSeconds

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['dailyLineupMarketOverview'],
    queryFn: api.getMarketOverview,
    refetchInterval: marketRefreshSeconds * 1000,
  })

  useEffect(() => {
    setWatchlist(getWatchlist())
  }, [])

  const positions = useMemo(() => account.positions.map(p => ({
    ticker: p.ticker,
    kind: p.kind,
    quantity: p.quantity,
    costBasis: p.costBasis,
    companyName: p.companyName,
    sector: p.sector,
    optionType: p.optionType,
    strike: p.strike,
    expiry: p.expiry,
  })), [account.positions])

  const focusTickers = useMemo(
    () => uniqueTickers([...watchlist, ...positions.map(p => p.ticker)]),
    [positions, watchlist],
  )

  async function generateLineup() {
    if (!openaiKey || !data) return
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    setLineup('')
    setCitations([])
    setAiError('')
    setIsSearching(false)
    setIsGenerating(true)
    try {
      await streamDailyLineup({
        apiKey: openaiKey,
        model: settings.openaiModel,
        watchlist,
        positions,
        marketOverview: data,
        signal: controller.signal,
        temperature: settings.openaiTemperature,
        extraSystemPrompt: settings.openaiSystemPrompt,
        onSearchStart: () => setIsSearching(true),
        onCitation: citation => setCitations(prev => (
          prev.some(c => c.url === citation.url) ? prev : [...prev, citation]
        )),
        onDelta: chunk => setLineup(prev => prev + chunk),
      })
    } catch (e) {
      if (controller.signal.aborted) return
      const message = e instanceof OpenAIError || e instanceof Error
        ? e.message
        : 'Unable to generate the Daily Lineup.'
      setAiError(message)
    } finally {
      if (!controller.signal.aborted) {
        setIsGenerating(false)
        setIsSearching(false)
      }
    }
  }

  useEffect(() => {
    if (autoStarted.current || !openaiKey || !data) return
    autoStarted.current = true
    void generateLineup()
    return () => abortRef.current?.abort()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [openaiKey, data])

  if (isLoading) {
    return <div className="p-4 text-muted text-sm tabnum">Loading daily market data...</div>
  }

  if (error || !data) {
    return (
      <div className="p-4">
        <p className="text-down text-sm">Failed to load Daily Lineup market overview.</p>
        <button onClick={() => refetch()} className="mt-2 text-2xs text-muted hover:text-text">Retry</button>
      </div>
    )
  }

  const { sectors, market_trend, advances, declines } = data
  const extData = data as typeof data & { avg_day_change?: number; avg_week_change?: number; avg_month_change?: number }
  const avg_day_change = extData.avg_day_change
  const avg_week_change = extData.avg_week_change
  const avg_month_change = extData.avg_month_change

  const key = PERIOD_KEY[period]
  const getValue = (s: (typeof sectors)[0]) =>
    key === 'return_1d' ? s.return_1d : key === 'return_1w' ? (s.return_1w ?? 0) : (s.return_1m ?? 0)

  const chartData = [...sectors]
    .sort((a, b) => getValue(b) - getValue(a))
    .map(s => ({
      name: s.name.replace('Consumer ', 'Cons. ').replace('Communication Services', 'Comm.'),
      value: getValue(s),
      fullName: s.name,
      trend: s.trend,
    }))

  const trendColor = (market_trend as string).toLowerCase().includes('bull') ? 'up'
    : (market_trend as string).toLowerCase().includes('bear') ? 'down'
    : 'muted'

  const avgChange = period === '1d' ? avg_day_change : period === '1w' ? avg_week_change : avg_month_change

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-3 px-4 py-2 border-b border-border bg-s1 shrink-0">
        <Sparkles size={16} className="text-accent" aria-hidden="true" />
        <span className="text-sm text-text font-medium">Daily Lineup</span>
        <span className="text-2xs text-muted">AI watchlist from market tape, saved names, and paper positions</span>
        <button onClick={() => refetch()} className="ml-auto text-2xs text-muted hover:text-text">
          Refresh market
        </button>
      </div>

      <div className="flex flex-1 overflow-hidden">
        <div className="flex-1 overflow-auto p-4">
          <div className="mb-5">
            <TickerSearchChart defaultTicker="SPY" />
          </div>

          <div className="grid grid-cols-4 border border-border mb-5">
            <MetricTile label="Market Trend" value={(market_trend as string).toUpperCase()} color={trendColor} />
            <MetricTile label="Advancing" value={advances ?? 0} color="up" />
            <MetricTile label="Declining" value={declines ?? 0} color="down" />
            <MetricTile
              label={`Avg ${period.toUpperCase()}`}
              value={`${Number(avgChange ?? 0) >= 0 ? '+' : ''}${Number(avgChange ?? 0).toFixed(2)}`}
              unit="%"
              color={Number(avgChange ?? 0) >= 0 ? 'up' : 'down'}
            />
          </div>

          <div className="border border-border mb-5">
            <div className="flex items-center gap-3 bg-s2 px-3 py-1 border-b border-border">
              <span className="text-2xs text-dim">
                Market Overview - Sector Performance
              </span>
              <div className="ml-auto flex items-center gap-px">
                {(['1d', '1w', '1m'] as Period[]).map(p => (
                  <button
                    key={p}
                    onClick={() => setPeriod(p)}
                    className={[
                      'px-2 py-0.5 text-2xs border',
                      period === p
                        ? 'border-accent text-accent'
                        : 'border-border text-muted hover:text-text',
                    ].join(' ')}
                  >
                    {p.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
            <div className="px-3 pb-3 pt-2">
              <ResponsiveContainer width="99%" height={240}>
                <BarChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 40 }}>
                  <XAxis
                    dataKey="name"
                    tick={{ fill: '#5A5A63', fontSize: 10, fontFamily: 'Segoe UI' }}
                    angle={-35}
                    textAnchor="end"
                    interval={0}
                    tickLine={false}
                    axisLine={{ stroke: '#1F1F23' }}
                  />
                  <YAxis
                    tick={{ fill: '#5A5A63', fontSize: 10, fontFamily: 'Segoe UI' }}
                    tickFormatter={v => `${v > 0 ? '+' : ''}${v.toFixed(1)}%`}
                    tickLine={false}
                    axisLine={false}
                    width={52}
                  />
                  <ReferenceLine y={0} stroke="#2A2A30" strokeWidth={1} />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (!active || !payload?.length) return null
                      const d = payload[0].payload
                      const v = d.value as number
                      return (
                        <div className="bg-s1 border border-border-strong px-2 py-1.5 text-2xs">
                          <div className="text-text mb-0.5">{d.fullName}</div>
                          <div className={v >= 0 ? 'text-up tabnum' : 'text-down tabnum'}>
                            {v >= 0 ? '+' : ''}{v.toFixed(2)}%
                          </div>
                          {d.trend && <div className="text-dim mt-0.5">{d.trend}</div>}
                        </div>
                      )
                    }}
                    cursor={{ fill: 'rgba(255,255,255,0.03)' }}
                  />
                  <Bar dataKey="value" radius={0} maxBarSize={32}>
                    {chartData.map((entry, i) => (
                      <Cell
                        key={i}
                        fill={entry.value >= 0 ? '#1FBF75' : '#E5484D'}
                        fillOpacity={0.8}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="border border-border">
            <div className="bg-s2 px-3 py-1 border-b border-border">
              <span className="text-2xs text-dim">Sector Detail</span>
            </div>
            <table className="w-full table-fixed">
              <thead>
                <tr className="border-b border-border bg-s2">
                  <th className="px-3 py-1 text-2xs text-muted text-left">Sector</th>
                  <th className="px-3 py-1 text-2xs text-muted text-right w-20">1D%</th>
                  <th className="px-3 py-1 text-2xs text-muted text-right w-20">1W%</th>
                  <th className="px-3 py-1 text-2xs text-muted text-right w-20">1M%</th>
                  <th className="px-3 py-1 text-2xs text-muted text-left w-36">Trend</th>
                </tr>
              </thead>
              <tbody>
                {[...sectors].sort((a, b) => b.return_1d - a.return_1d).map((s) => {
                  const r1d = s.return_1d
                  const r1w = s.return_1w ?? 0
                  const r1m = s.return_1m ?? 0
                  return (
                    <tr key={s.name as string} className="border-b border-border hover:bg-s2">
                      <td className="px-3 py-1.5 text-sm text-text">{s.name as string}</td>
                      <td className={`px-3 py-1.5 text-sm tabnum text-right ${r1d >= 0 ? 'text-up' : 'text-down'}`}>
                        {r1d >= 0 ? '+' : ''}{r1d.toFixed(2)}%
                      </td>
                      <td className={`px-3 py-1.5 text-sm tabnum text-right ${r1w >= 0 ? 'text-up' : 'text-down'}`}>
                        {r1w >= 0 ? '+' : ''}{r1w.toFixed(2)}%
                      </td>
                      <td className={`px-3 py-1.5 text-sm tabnum text-right ${r1m >= 0 ? 'text-up' : 'text-down'}`}>
                        {r1m >= 0 ? '+' : ''}{r1m.toFixed(2)}%
                      </td>
                      <td className="px-3 py-1.5 text-2xs text-muted">{(s.trend as string) ?? '-'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>

        <aside className="w-[26rem] shrink-0 border-l border-border bg-s1/70 overflow-y-auto">
          <div className="flex items-center gap-2 border-b border-border px-4 py-3">
            <Sparkles size={15} className="text-accent" aria-hidden="true" />
            <div className="min-w-0">
              <p className="text-sm font-medium text-text">AI Daily Lineup</p>
              <p className="text-2xs text-dim tabnum">
                {focusTickers.length ? `Tracking ${focusTickers.join(', ')}` : 'Broad market scan'}
              </p>
            </div>
            <button
              type="button"
              onClick={() => void generateLineup()}
              disabled={!openaiKey || isGenerating}
              className="ml-auto flex h-8 items-center gap-1 border border-border bg-s2 px-2 text-2xs text-muted hover:border-border-strong hover:text-text disabled:opacity-40"
            >
              <RefreshCw size={13} className={isGenerating ? 'animate-spin' : ''} aria-hidden="true" />
              Run
            </button>
          </div>

          {!openaiKey ? (
            <div className="m-4 border border-accent/40 bg-accent/10 p-4">
              <div className="flex items-start gap-3">
                <Settings size={16} className="mt-0.5 shrink-0 text-accent" aria-hidden="true" />
                <div>
                  <p className="text-sm font-medium text-text">OpenAI key required</p>
                  <p className="mt-1 text-2xs leading-relaxed text-muted">
                    Add your OpenAI API key in Settings to generate the Daily Lineup with fresh market context.
                  </p>
                  <Link
                    to="/settings"
                    className="mt-3 inline-flex h-8 items-center border border-accent/50 px-3 text-2xs font-medium text-accent hover:bg-accent/10"
                  >
                    Open Settings
                  </Link>
                </div>
              </div>
            </div>
          ) : (
            <div className="p-4">
              <div className="mb-3 flex flex-wrap gap-2">
                {focusTickers.slice(0, 10).map(ticker => (
                  <Link
                    key={ticker}
                    to={`/stock/${ticker}`}
                    className="border border-border px-2 py-1 text-2xs tabnum text-muted hover:border-accent/50 hover:text-text"
                  >
                    {ticker}
                  </Link>
                ))}
              </div>

              {isSearching && (
                <p className="mb-3 text-2xs text-accent">Searching latest market catalysts...</p>
              )}
              {aiError && (
                <div className="mb-3 border border-down/40 bg-down/10 p-3 text-2xs text-down">
                  {aiError}
                </div>
              )}
              {lineup ? (
                <Markdown text={lineup} />
              ) : (
                <div className="border border-border bg-bg/50 p-4 text-2xs leading-relaxed text-dim">
                  {isGenerating
                    ? 'Building the lineup from current market news, your watchlist, and paper positions...'
                    : 'Run the AI lineup to see what is worth watching today.'}
                </div>
              )}

              {citations.length > 0 && (
                <div className="mt-5 border-t border-border pt-3">
                  <p className="mb-2 text-2xs uppercase tracking-[0.2em] text-dim">Sources</p>
                  <div className="space-y-1.5">
                    {citations.slice(0, 6).map(c => (
                      <a
                        key={c.url}
                        href={c.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="block truncate text-2xs text-muted hover:text-accent"
                      >
                        {c.title}
                      </a>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </aside>
      </div>
    </div>
  )
}
