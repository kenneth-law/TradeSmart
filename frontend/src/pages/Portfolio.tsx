import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import MetricTile from '../components/data/MetricTile'
import DataTable from '../components/data/DataTable'
import type { Column } from '../components/data/DataTable'
import type { PortfolioPosition } from '../types'

const COLUMNS: Column<PortfolioPosition & Record<string, unknown>>[] = [
  { key: 'ticker',             label: 'TICKER',   width: 80 },
  { key: 'sector',             label: 'SECTOR',   width: 120 },
  { key: 'shares',             label: 'SHARES',   width: 72, align: 'right',
    render: r => <span className="tabnum">{r.shares as number}</span>
  },
  { key: 'cost_basis',         label: 'COST',     width: 80, align: 'right',
    render: r => <span className="tabnum">{(r.cost_basis as number).toFixed(2)}</span>
  },
  { key: 'current_price',      label: 'PRICE',    width: 80, align: 'right',
    render: r => <span className="tabnum">{(r.current_price as number).toFixed(2)}</span>
  },
  { key: 'market_value',       label: 'MKT VAL', width: 90, align: 'right',
    render: r => <span className="tabnum">${(r.market_value as number).toLocaleString()}</span>
  },
  { key: 'unrealized_pnl',     label: 'P&L',      width: 90, align: 'right',
    render: r => {
      const v = r.unrealized_pnl as number
      return <span className={`tabnum ${v >= 0 ? 'text-up' : 'text-down'}`}>{v >= 0 ? '+' : ''}${v.toLocaleString()}</span>
    }
  },
  { key: 'unrealized_pnl_pct', label: 'P&L%',     width: 72, align: 'right',
    render: r => {
      const v = r.unrealized_pnl_pct as number
      return <span className={`tabnum ${v >= 0 ? 'text-up' : 'text-down'}`}>{v >= 0 ? '+' : ''}{v.toFixed(2)}%</span>
    }
  },
  { key: 'entry_date',         label: 'ENTRY',    width: 90 },
]

export default function Portfolio() {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['portfolio'],
    queryFn: api.getPortfolio,
    refetchInterval: 300_000,
  })

  if (isLoading) {
    return <div className="p-4 text-muted text-sm tabnum">Loading portfolio…</div>
  }

  if (error || !data) {
    return (
      <div className="p-4">
        <p className="text-down text-sm">Failed to load portfolio.</p>
        <button onClick={() => refetch()} className="mt-2 text-2xs text-muted hover:text-text">
          Retry
        </button>
      </div>
    )
  }

  const { positions, summary } = data
  const tableData = positions as unknown as Array<PortfolioPosition & Record<string, unknown>>

  // Sector aggregation
  const bySector: Record<string, number> = {}
  for (const p of positions) {
    const sec = p.sector ?? 'Unknown'
    bySector[sec] = (bySector[sec] ?? 0) + p.market_value
  }
  const totalInvested = summary.invested || 1
  const sectorEntries = Object.entries(bySector).sort((a, b) => b[1] - a[1])

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-border bg-s1 shrink-0">
        <span className="text-sm text-text font-medium uppercase tracking-wide">Portfolio</span>
        <span className="text-border ml-auto">|</span>
        <span className="text-2xs text-muted tabnum">{positions.length} positions</span>
        <button onClick={() => refetch()} className="text-2xs text-muted hover:text-text ml-2">
          Refresh
        </button>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Main table */}
        <div className="flex-1 overflow-auto">
          <DataTable
            columns={COLUMNS}
            data={tableData}
            rowKey={r => r.ticker}
            emptyMessage="No positions."
            virtualRows={positions.length > 50}
          />
        </div>

        {/* Sidebar */}
        <div className="w-52 shrink-0 border-l border-border overflow-y-auto">
          <div className="px-3 py-2 border-b border-border">
            <p className="text-2xs text-dim uppercase tracking-wide">Summary</p>
          </div>
          <MetricTile label="Total Value" value={`$${summary.total_value.toLocaleString()}`} />
          <MetricTile label="Cash" value={`$${summary.cash.toLocaleString()}`} />
          <MetricTile label="Invested" value={`$${summary.invested.toLocaleString()}`} />
          <MetricTile
            label="Total P&L"
            value={`${summary.total_pnl >= 0 ? '+' : ''}$${summary.total_pnl.toLocaleString()}`}
            color={summary.total_pnl >= 0 ? 'up' : 'down'}
          />
          <MetricTile
            label="P&L %"
            value={`${summary.total_pnl_pct >= 0 ? '+' : ''}${summary.total_pnl_pct.toFixed(2)}`}
            unit="%"
            color={summary.total_pnl_pct >= 0 ? 'up' : 'down'}
          />
          {summary.portfolio_beta !== undefined && (
            <MetricTile label="Beta" value={summary.portfolio_beta.toFixed(2)} />
          )}
          {summary.max_drawdown !== undefined && (
            <MetricTile label="Max Drawdown" value={summary.max_drawdown.toFixed(2)} unit="%" color="down" />
          )}

          {/* Sector allocation */}
          {sectorEntries.length > 0 && (
            <div className="border-t border-border mt-1">
              <div className="px-3 py-2">
                <p className="text-2xs text-dim uppercase tracking-wide mb-2">Sector Allocation</p>
                {sectorEntries.map(([name, value]) => {
                  const pct = (value / totalInvested) * 100
                  return (
                    <div key={name} className="mb-1.5">
                      <div className="flex justify-between text-2xs mb-0.5">
                        <span className="text-muted truncate pr-1">{name}</span>
                        <span className="tabnum text-text shrink-0">{pct.toFixed(1)}%</span>
                      </div>
                      <div className="h-1 bg-border">
                        <div className="h-full bg-accent opacity-70" style={{ width: `${pct}%` }} />
                      </div>
                    </div>
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
