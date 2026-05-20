import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer, ReferenceLine } from 'recharts'
import { api } from '../lib/api'
import { useAppStore } from '../store/useAppStore'
import MetricTile from '../components/data/MetricTile'
import TickerSearchChart from '../components/charts/TickerSearchChart'

type Period = '1d' | '1w' | '1m'

const PERIOD_KEY: Record<Period, 'return_1d' | 'return_1w' | 'return_1m'> = {
  '1d': 'return_1d',
  '1w': 'return_1w',
  '1m': 'return_1m',
}

export default function MarketOverview() {
  const [period, setPeriod] = useState<Period>('1d')
  const marketRefreshSeconds = useAppStore(s => s.settings.marketRefreshSeconds)

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['marketOverview'],
    queryFn: api.getMarketOverview,
    refetchInterval: marketRefreshSeconds * 1000,
  })

  if (isLoading) {
    return <div className="p-4 text-muted text-sm tabnum">Loading market data…</div>
  }

  if (error || !data) {
    return (
      <div className="p-4">
        <p className="text-down text-sm">Failed to load market overview.</p>
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
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-border bg-s1 shrink-0">
        <span className="text-sm text-text font-medium">Market Overview</span>
        <span className="text-border">|</span>
        {/* Period toggle */}
        <div className="flex items-center gap-px">
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
        <button onClick={() => refetch()} className="ml-auto text-2xs text-muted hover:text-text">
          Refresh
        </button>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Main content */}
        <div className="flex-1 overflow-auto p-4">
          {/* Ticker chart search */}
          <div className="mb-5">
            <TickerSearchChart defaultTicker="SPY" />
          </div>

          {/* Summary tiles */}
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

          {/* Bar chart */}
          <div className="border border-border mb-5">
            <div className="bg-s2 px-3 py-1 border-b border-border">
              <span className="text-2xs text-dim">
                Sector Performance — {period.toUpperCase()}
              </span>
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

          {/* Detail table */}
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
                      <td className="px-3 py-1.5 text-2xs text-muted">{(s.trend as string) ?? '—'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}
