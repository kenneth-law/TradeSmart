import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQueries } from '@tanstack/react-query'
import { RotateCcw, Save, WalletCards } from 'lucide-react'
import { api } from '../lib/api'
import { DEFAULT_PAPER_COST_MODEL, useAppStore, type PaperOrder, type PaperPosition } from '../store/useAppStore'
import MetricTile from '../components/data/MetricTile'
import DataTable, { type Column } from '../components/data/DataTable'
import { quotePrice, useLiveQuotes } from '../hooks/useLiveQuotes'

type PaperRow = PaperPosition & Record<string, unknown> & {
  label: string
  description: string
  multiplier: number
  lastPrice: number
  marketValue: number
  costValue: number
  unrealizedPnl: number
  unrealizedPnlPct: number
  underlyingPrice?: number
}

const POSITION_COLUMNS: Column<PaperRow>[] = [
  { key: 'label', label: 'INSTRUMENT', width: 150,
    render: r => (
      <div>
        <div className="text-sm text-text tabnum">{r.label}</div>
        <div className="text-2xs text-dim truncate">{r.description}</div>
      </div>
    )
  },
  { key: 'quantity', label: 'QTY', width: 64, align: 'right',
    render: r => <span className="tabnum">{r.quantity}</span>
  },
  { key: 'costBasis', label: 'AVG COST', width: 86, align: 'right',
    render: r => <span className="tabnum">${r.costBasis.toFixed(2)}</span>
  },
  { key: 'lastPrice', label: 'LAST', width: 86, align: 'right',
    render: r => <span className="tabnum">${r.lastPrice.toFixed(2)}</span>
  },
  { key: 'marketValue', label: 'VALUE', width: 100, align: 'right',
    render: r => <span className="tabnum">${Math.round(r.marketValue).toLocaleString()}</span>
  },
  { key: 'unrealizedPnl', label: 'P&L', width: 96, align: 'right',
    render: r => (
      <span className={`tabnum ${r.unrealizedPnl >= 0 ? 'text-up' : 'text-down'}`}>
        {r.unrealizedPnl >= 0 ? '+' : ''}${Math.round(r.unrealizedPnl).toLocaleString()}
      </span>
    )
  },
  { key: 'unrealizedPnlPct', label: 'P&L%', width: 76, align: 'right',
    render: r => (
      <span className={`tabnum ${r.unrealizedPnlPct >= 0 ? 'text-up' : 'text-down'}`}>
        {r.unrealizedPnlPct >= 0 ? '+' : ''}{r.unrealizedPnlPct.toFixed(2)}%
      </span>
    )
  },
]

const ORDER_COLUMNS: Column<PaperOrder & Record<string, unknown>>[] = [
  { key: 'timestamp', label: 'TIME', width: 132,
    render: r => <span className="tabnum">{new Date(r.timestamp).toLocaleString([], { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' })}</span>
  },
  { key: 'action', label: 'ACTION', width: 92,
    render: r => <span className={r.action.startsWith('BUY') ? 'text-up' : 'text-down'}>{r.action.replace('_', ' ')}</span>
  },
  { key: 'ticker', label: 'TICKER', width: 80 },
  { key: 'quantity', label: 'QTY', width: 60, align: 'right',
    render: r => <span className="tabnum">{r.quantity}</span>
  },
  { key: 'price', label: 'PRICE', width: 76, align: 'right',
    render: r => <span className="tabnum">${r.price.toFixed(2)}</span>
  },
  { key: 'notional', label: 'NOTIONAL', width: 96, align: 'right',
    render: r => <span className="tabnum">${Math.round(r.notional).toLocaleString()}</span>
  },
  { key: 'fee', label: 'COST', width: 74, align: 'right',
    render: r => {
      const v = r.fee ?? 0
      return v > 0 ? <span className="tabnum text-accent">${v.toFixed(2)}</span> : <span className="text-dim">-</span>
    }
  },
  { key: 'realizedPnl', label: 'REALIZED', width: 92, align: 'right',
    render: r => {
      const v = r.realizedPnl
      if (v == null) return <span className="text-dim">-</span>
      return <span className={`tabnum ${v >= 0 ? 'text-up' : 'text-down'}`}>{v >= 0 ? '+' : ''}${Math.round(v).toLocaleString()}</span>
    }
  },
]

function money(v: number) {
  return `$${Math.round(v).toLocaleString()}`
}

export default function Portfolio() {
  const navigate = useNavigate()
  const settings = useAppStore(s => s.settings)
  const account = useAppStore(s => s.paperAccount)
  const setPaperCash = useAppStore(s => s.setPaperCash)
  const setPaperCostModel = useAppStore(s => s.setPaperCostModel)
  const resetPaperAccount = useAppStore(s => s.resetPaperAccount)
  const [cashDraft, setCashDraft] = useState(String(account.cash))
  const [costDraft, setCostDraft] = useState({
    stockPerShare: String(account.costModel.stockPerShare),
    optionPerContract: String(account.costModel.optionPerContract),
    slippageBps: String(account.costModel.slippageBps),
  })

  useEffect(() => {
    setCashDraft(String(account.cash))
  }, [account.cash])

  useEffect(() => {
    setCostDraft({
      stockPerShare: String(account.costModel.stockPerShare),
      optionPerContract: String(account.costModel.optionPerContract),
      slippageBps: String(account.costModel.slippageBps),
    })
  }, [account.costModel])

  const tickers = useMemo(
    () => Array.from(new Set(account.positions.map(p => p.ticker))).sort(),
    [account.positions]
  )
  const live = useLiveQuotes(tickers, settings.fastPaperRefresh && tickers.length > 0)
  const refreshMs = settings.marketRefreshSeconds * 1000
  const priceQueries = useQueries({
    queries: tickers.map(ticker => ({
      queryKey: ['paper-position-stock', ticker],
      queryFn: () => api.getStock(ticker),
      refetchInterval: refreshMs,
      enabled: tickers.length > 0,
    })),
  })
  const priceByTicker = useMemo(() => {
    const map = new Map<string, { price?: number; sector?: string; name?: string; live?: boolean }>()
    tickers.forEach((ticker, i) => {
      const data = priceQueries[i]?.data
      const livePrice = quotePrice(live.quotes[ticker])
      if (data || livePrice != null) {
        map.set(ticker, {
          price: livePrice ?? data?.current_price,
          sector: data?.sector,
          name: data?.company_name,
          live: livePrice != null,
        })
      }
    })
    return map
  }, [live.quotes, priceQueries, tickers])

  const rows = useMemo<PaperRow[]>(() => account.positions.map(position => {
    const quote = priceByTicker.get(position.ticker)
    const multiplier = position.kind === 'option' ? 100 : 1
    const lastPrice = position.kind === 'stock'
      ? quote?.price ?? position.costBasis
      : position.costBasis
    const marketValue = lastPrice * position.quantity * multiplier
    const costValue = position.costBasis * position.quantity * multiplier
    const unrealizedPnl = marketValue - costValue
    const unrealizedPnlPct = costValue > 0 ? (unrealizedPnl / costValue) * 100 : 0
    const optionLabel = position.kind === 'option'
      ? `${position.expiry ?? '-'} ${position.strike?.toFixed(2) ?? '-'} ${position.optionType?.toUpperCase() ?? ''}`
      : quote?.name ?? position.companyName ?? position.sector ?? 'Stock'
    return {
      ...position,
      sector: quote?.sector ?? position.sector,
      companyName: quote?.name ?? position.companyName,
      label: position.kind === 'option' ? `${position.ticker} OPT` : position.ticker,
      description: optionLabel,
      multiplier,
      lastPrice,
      marketValue,
      costValue,
      unrealizedPnl,
      unrealizedPnlPct,
      underlyingPrice: quote?.price,
    }
  }), [account.positions, priceByTicker])

  const invested = rows.reduce((sum, row) => sum + row.marketValue, 0)
  const totalValue = account.cash + invested
  const totalPnl = totalValue - account.initialCash
  const totalPnlPct = account.initialCash > 0 ? (totalPnl / account.initialCash) * 100 : 0
  const costInvested = rows.reduce((sum, row) => sum + row.costValue, 0)

  const byType = rows.reduce<Record<string, number>>((acc, row) => {
    const key = row.kind === 'option' ? 'Options' : 'Stocks'
    acc[key] = (acc[key] ?? 0) + row.marketValue
    return acc
  }, {})
  const allocation = Object.entries(byType).sort((a, b) => b[1] - a[1])

  function saveCash() {
    const next = Number(cashDraft)
    if (Number.isFinite(next) && next >= 0) setPaperCash(next)
  }

  function saveCosts() {
    setPaperCostModel({
      stockPerShare: Number(costDraft.stockPerShare),
      optionPerContract: Number(costDraft.optionPerContract),
      slippageBps: Number(costDraft.slippageBps),
    })
  }

  function resetCosts() {
    setCostDraft({
      stockPerShare: String(DEFAULT_PAPER_COST_MODEL.stockPerShare),
      optionPerContract: String(DEFAULT_PAPER_COST_MODEL.optionPerContract),
      slippageBps: String(DEFAULT_PAPER_COST_MODEL.slippageBps),
    })
    setPaperCostModel(DEFAULT_PAPER_COST_MODEL)
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-3 px-4 py-2 border-b border-border bg-s1 shrink-0">
        <WalletCards size={16} className="text-accent" aria-hidden="true" />
        <span className="text-sm text-text font-medium">Paper Portfolio</span>
        <span className="text-2xs text-muted tabnum">{rows.length} positions</span>
        {settings.fastPaperRefresh && (
          <span className={`text-2xs tabnum ${live.connected || live.configured ? 'text-accent' : 'text-muted'}`}>
            {live.configured ? `alpaca ${live.feed || 'live'}` : 'alpaca not configured'}
          </span>
        )}
        <div className="ml-auto flex items-center gap-2">
          <input
            type="number"
            min={0}
            value={cashDraft}
            onChange={event => setCashDraft(event.target.value)}
            className="h-8 w-32 border border-border bg-s2 px-2 text-xs text-text outline-none focus-visible:border-accent tabnum"
            aria-label="Paper starting cash"
          />
          <button
            type="button"
            onClick={saveCash}
            className="flex h-8 items-center gap-1 border border-border bg-s2 px-2 text-2xs text-muted hover:border-border-strong hover:text-text"
          >
            <Save size={13} aria-hidden="true" />
            Set Cash
          </button>
          <button
            type="button"
            onClick={() => {
              const value = Number(cashDraft)
              resetPaperAccount(Number.isFinite(value) ? value : undefined)
            }}
            className="flex h-8 items-center gap-1 border border-border bg-s2 px-2 text-2xs text-muted hover:border-border-strong hover:text-down"
          >
            <RotateCcw size={13} aria-hidden="true" />
            Reset
          </button>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        <div className="flex-1 overflow-auto">
          <DataTable
            columns={POSITION_COLUMNS}
            data={rows}
            rowKey={r => r.id}
            onRowClick={r => navigate(`/stock/${r.ticker}`)}
            emptyMessage="No paper positions. Open a stock detail page to place a paper order."
            virtualRows={rows.length > 50}
          />
          <div className="border-t border-border">
            <div className="px-4 py-2 text-2xs uppercase tracking-[0.2em] text-dim">Recent Orders</div>
            <DataTable
              columns={ORDER_COLUMNS}
              data={account.orders as Array<PaperOrder & Record<string, unknown>>}
              rowKey={r => r.id}
              onRowClick={r => navigate(`/stock/${r.ticker}`)}
              emptyMessage="No paper orders yet."
              virtualRows={account.orders.length > 30}
            />
          </div>
        </div>

        <div className="w-56 shrink-0 border-l border-border overflow-y-auto">
          <div className="px-3 py-2 border-b border-border">
            <p className="text-2xs text-dim">Account</p>
          </div>
          <MetricTile label="Total Value" value={money(totalValue)} />
          <MetricTile label="Cash" value={money(account.cash)} />
          <MetricTile label="Starting Cash" value={money(account.initialCash)} />
          <MetricTile label="Market Value" value={money(invested)} />
          <MetricTile label="Cost Basis" value={money(costInvested)} />
          <MetricTile
            label="Total P&L"
            value={`${totalPnl >= 0 ? '+' : ''}${money(totalPnl)}`}
            color={totalPnl >= 0 ? 'up' : 'down'}
          />
          <MetricTile
            label="P&L %"
            value={`${totalPnlPct >= 0 ? '+' : ''}${totalPnlPct.toFixed(2)}`}
            unit="%"
            color={totalPnlPct >= 0 ? 'up' : 'down'}
          />
          <div className="border-t border-border mt-1 px-3 py-3">
            <p className="text-2xs text-dim mb-2">Transaction Costs</p>
            <div className="space-y-2">
              <label className="block">
                <span className="block text-2xs text-muted mb-1">Stock $/share</span>
                <input
                  type="number"
                  min={0}
                  step={0.001}
                  value={costDraft.stockPerShare}
                  onChange={event => setCostDraft(d => ({ ...d, stockPerShare: event.target.value }))}
                  className="h-8 w-full border border-border bg-bg px-2 text-xs text-text outline-none focus-visible:border-accent tabnum"
                />
              </label>
              <label className="block">
                <span className="block text-2xs text-muted mb-1">Option $/contract</span>
                <input
                  type="number"
                  min={0}
                  step={0.01}
                  value={costDraft.optionPerContract}
                  onChange={event => setCostDraft(d => ({ ...d, optionPerContract: event.target.value }))}
                  className="h-8 w-full border border-border bg-bg px-2 text-xs text-text outline-none focus-visible:border-accent tabnum"
                />
              </label>
              <label className="block">
                <span className="block text-2xs text-muted mb-1">Slippage bps</span>
                <input
                  type="number"
                  min={0}
                  step={1}
                  value={costDraft.slippageBps}
                  onChange={event => setCostDraft(d => ({ ...d, slippageBps: event.target.value }))}
                  className="h-8 w-full border border-border bg-bg px-2 text-xs text-text outline-none focus-visible:border-accent tabnum"
                />
              </label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={saveCosts}
                  className="h-8 border border-border bg-s2 px-2 text-2xs text-muted hover:border-border-strong hover:text-text"
                >
                  Apply
                </button>
                <button
                  type="button"
                  onClick={resetCosts}
                  className="h-8 border border-border bg-s2 px-2 text-2xs text-muted hover:border-border-strong hover:text-text"
                >
                  Clear
                </button>
              </div>
            </div>
          </div>
          {allocation.length > 0 && (
            <div className="border-t border-border mt-1">
              <div className="px-3 py-2">
                <p className="text-2xs text-dim mb-2">Allocation</p>
                {allocation.map(([name, value]) => {
                  const pct = invested > 0 ? (value / invested) * 100 : 0
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
