import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import MetricTile from '../components/data/MetricTile'
import type { SectorData } from '../types'

export default function MarketOverview() {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['marketOverview'],
    queryFn: api.getMarketOverview,
    refetchInterval: 300_000,
  })

  if (isLoading) {
    return <div className="p-4 text-muted text-sm tabnum">Loading market data…</div>
  }

  if (error || !data) {
    return (
      <div className="p-4">
        <p className="text-down text-sm">Failed to load market overview.</p>
        <button onClick={() => refetch()} className="mt-2 text-2xs text-muted hover:text-text">
          Retry
        </button>
      </div>
    )
  }

  const { sectors, market_trend, advances, declines, market_health } = data

  const sortedSectors = [...sectors].sort((a, b) => b.return_1d - a.return_1d)

  const trendColor = market_trend === 'bullish' ? 'up'
    : market_trend === 'bearish' ? 'down'
    : 'muted'

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-border bg-s1 shrink-0">
        <span className="text-sm text-text font-medium uppercase tracking-wide">Market Overview</span>
        <span className="text-border ml-auto">|</span>
        <button onClick={() => refetch()} className="text-2xs text-muted hover:text-text">
          Refresh
        </button>
      </div>

      <div className="flex flex-1 overflow-auto">
        <div className="flex-1 p-4">
          {/* Market Summary tiles */}
          <div className="grid grid-cols-4 border border-border mb-6">
            <MetricTile label="Market Trend" value={market_trend.toUpperCase()} color={trendColor} />
            {advances !== undefined && (
              <MetricTile label="Advances" value={advances} color="up" />
            )}
            {declines !== undefined && (
              <MetricTile label="Declines" value={declines} color="down" />
            )}
            {market_health && (
              <MetricTile label="Health" value={market_health.toUpperCase()} />
            )}
          </div>

          {/* Sector heatmap-style table */}
          <div className="border border-border">
            <div className="bg-s2 px-3 py-1 border-b border-border">
              <span className="text-2xs text-dim uppercase tracking-wide">Sector Performance (1D)</span>
            </div>
            <table className="w-full table-fixed">
              <thead>
                <tr className="border-b border-border bg-s2">
                  <th className="px-3 py-1 text-2xs text-muted uppercase tracking-wide text-left w-48">Sector</th>
                  <th className="px-3 py-1 text-2xs text-muted uppercase tracking-wide text-right w-24">1D%</th>
                  <th className="px-3 py-1 text-2xs text-muted uppercase tracking-wide text-left">Trend</th>
                  <th className="px-3 py-1 text-2xs text-muted uppercase tracking-wide text-right w-48">Relative</th>
                </tr>
              </thead>
              <tbody>
                {sortedSectors.map((sector: SectorData) => {
                  const pct = sector.return_1d
                  const isPos = pct >= 0
                  const absMax = Math.max(...sortedSectors.map(s => Math.abs(s.return_1d)), 1)
                  const barWidth = (Math.abs(pct) / absMax) * 100

                  return (
                    <tr key={sector.name} className="border-b border-border hover:bg-s2">
                      <td className="px-3 py-1.5 text-sm text-text">{sector.name}</td>
                      <td className={`px-3 py-1.5 text-sm tabnum text-right ${isPos ? 'text-up' : 'text-down'}`}>
                        {isPos ? '+' : ''}{pct.toFixed(2)}%
                      </td>
                      <td className="px-3 py-1.5 text-2xs text-muted">
                        {sector.trend?.toUpperCase() ?? '—'}
                      </td>
                      <td className="px-3 py-1.5">
                        <div className="flex items-center justify-end">
                          <div
                            className={`h-1.5 ${isPos ? 'bg-up' : 'bg-down'} opacity-70`}
                            style={{ width: `${barWidth}%`, minWidth: 2, maxWidth: '100%' }}
                          />
                        </div>
                      </td>
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
