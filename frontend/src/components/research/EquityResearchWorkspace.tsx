import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { api } from '../../lib/api'
import type { EquityResearchResponse, EquityResearchSection, StockResult } from '../../types'
import Financials from '../financials/Financials'
import PriceChart from '../charts/PriceChart'
import { CartesianGrid, Legend as ChartLegend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

export type ResearchCategory = 'markets' | 'screening' | 'fundamentals' | 'estimates' | 'valuation' | 'ownership' | 'research' | 'charts' | 'testing'

type FunctionDef = { code: string; label: string }

const FUNCTIONS: Record<ResearchCategory, FunctionDef[]> = {
  markets: [
    { code: 'WEI', label: 'Market Overview' }, { code: 'CCB', label: 'Classification' },
    { code: 'EVTS', label: 'Events' }, { code: 'CACT', label: 'Corporate Actions' }, { code: 'BMAP', label: 'Company Map' },
  ],
  screening: [
    { code: 'SECF', label: 'Security Finder' }, { code: 'EQS', label: 'Equity Screen' },
  ],
  fundamentals: [
    { code: 'DES', label: 'Description' }, { code: 'BQ', label: 'Market Quote' }, { code: 'FA', label: 'Financials' }, { code: 'DVD', label: 'Dividends' },
    { code: 'CAST', label: 'Capital Structure' }, { code: 'CRPR', label: 'Credit Profile' },
  ],
  estimates: [
    { code: 'ANR', label: 'Recommendations' }, { code: 'EE', label: 'Estimates' }, { code: 'EEO', label: 'Estimate Overview' },
    { code: 'EEG', label: 'Estimate Graph' }, { code: 'ERN', label: 'Earnings History' }, { code: 'EM', label: 'Trends' },
  ],
  valuation: [
    { code: 'RV', label: 'Relative Value' }, { code: 'RVG', label: 'Value Graph' }, { code: 'RVC', label: 'Correlation' },
    { code: 'WACC', label: 'Cost of Capital' }, { code: 'DDM', label: 'Dividend Model' }, { code: 'BETA', label: 'Beta' },
  ],
  ownership: [
    { code: 'HDS', label: 'Holders' }, { code: 'OWN', label: 'Ownership' }, { code: 'GPTR', label: 'Insider Transactions' },
    { code: 'RELS', label: 'Related Securities' },
  ],
  research: [
    { code: 'CN', label: 'Company News' }, { code: 'CF', label: 'Filings' }, { code: 'BRC', label: 'Rating Changes' },
    { code: 'BI', label: 'Industry Research' }, { code: 'EVTS', label: 'Events' }, { code: 'CACT', label: 'Corporate Actions' },
  ],
  charts: [
    { code: 'GP', label: 'Price Graph' }, { code: 'GPO', label: 'OHLC Graph' }, { code: 'GF', label: 'Fundamentals' },
    { code: 'TRA', label: 'Total Return' }, { code: 'COMP', label: 'Comparable Return' }, { code: 'HP', label: 'Price Table' },
    { code: 'TECH', label: 'Technical Studies' },
  ],
  testing: [
    { code: 'FTST', label: 'Factor Test' }, { code: 'EQBT', label: 'Equity Backtest' }, { code: 'BT', label: 'Technical Backtest' },
  ],
}

const SECTION_FOR_CATEGORY: Partial<Record<ResearchCategory, EquityResearchSection>> = {
  markets: 'research', fundamentals: 'profile', estimates: 'estimates', valuation: 'valuation', ownership: 'ownership', research: 'research',
}

const CATEGORY_LABELS: Record<ResearchCategory, string> = {
  markets: 'Equity Markets', screening: 'Screening', fundamentals: 'Fundamentals', estimates: 'Estimates',
  valuation: 'Valuation', ownership: 'Ownership', research: 'Research', charts: 'Charts', testing: 'Testing',
}

export default function EquityResearchWorkspace({
  category,
  ticker,
  stock,
  peerTickers,
  onOpenIntelligence,
}: {
  category: ResearchCategory
  ticker: string
  stock: StockResult
  peerTickers: string[]
  onOpenIntelligence: (view?: 'network' | 'facilities' | 'related' | 'documents' | 'news') => void
}) {
  const definitions = FUNCTIONS[category]
  const [selection, setSelection] = useState<Partial<Record<ResearchCategory, string>>>({})
  const activeCode = selection[category] ?? definitions[0].code
  const section = SECTION_FOR_CATEGORY[category]
  const query = useQuery({
    queryKey: ['equity-research', ticker, section],
    queryFn: () => api.getEquityResearch(ticker, section!),
    enabled: !!section,
    staleTime: 30 * 60 * 1000,
    retry: 1,
  })

  return (
    <div className="flex h-full min-h-0 flex-col bg-bg/80">
      <div className="flex shrink-0 items-center gap-3 border-b border-border bg-s1/60 px-4 py-2">
        <span className="text-xs font-semibold uppercase tracking-[0.18em] text-text">{CATEGORY_LABELS[category]}</span>
        <span className="text-2xs text-dim">{definitions.find(item => item.code === activeCode)?.label}</span>
        {query.isFetching && <span className="ml-auto text-2xs text-dim">Loading {ticker} research…</span>}
        {query.error && <span className="ml-auto text-2xs text-down">Live research feed could not be loaded.</span>}
        {query.data?.as_of && <span className="ml-auto text-[9px] tabnum text-dim">AS OF {new Date(query.data.as_of).toLocaleString()}</span>}
      </div>

      <div className="flex shrink-0 items-center overflow-x-auto border-b border-border bg-s1/90 px-2">
        {definitions.map(item => (
          <button key={item.code} onClick={() => setSelection(current => ({ ...current, [category]: item.code }))} className={`shrink-0 border-r border-border px-4 py-2 text-[10px] font-medium transition-colors ${activeCode === item.code ? 'bg-accent text-bg' : 'text-muted hover:bg-s2 hover:text-text'}`}>
            {item.label}
          </button>
        ))}
      </div>

      <div className="min-h-0 flex-1 overflow-hidden">
        <FunctionView code={activeCode} ticker={ticker} stock={stock} peerTickers={peerTickers} data={query.data} onOpenIntelligence={onOpenIntelligence} />
      </div>
    </div>
  )
}

function FunctionView({ code, ticker, stock, peerTickers, data, onOpenIntelligence }: { code: string; ticker: string; stock: StockResult; peerTickers: string[]; data?: EquityResearchResponse; onOpenIntelligence: (view?: 'network' | 'facilities' | 'related' | 'documents' | 'news') => void }) {
  const profile = data?.profile ?? {}
  if (code === 'WEI') return <MarketOverviewPanel />
  if (code === 'CCB') return <IndustryPanel stock={stock} />
  if (code === 'BMAP') return <ActionPanel title="Company map" message="Manufacturing, distribution, warehousing, research and office locations are available in TradeSmart Intelligence Facilities." action="Open Facilities" onAction={() => onOpenIntelligence('facilities')} />
  if (code === 'SECF') return <NavigationAction title="Security finder" message="Open Equities lookup to resolve and load an equity security." action="Open Equities lookup" path="/market" />
  if (code === 'EQS') return <NavigationAction title="Equity screening" message="Open the TradeSmart technical scanner to rank and filter an equity universe." action="Open equity scanner" path="/technical" />
  if (code === 'FA') return <Financials ticker={ticker} />
  if (code === 'DES') return <DescriptionPanel ticker={ticker} stock={stock} profile={profile} />
  if (code === 'BQ') return <QuotePanel stock={stock} profile={profile} />
  if (code === 'DVD') return <DividendsPanel dividends={data?.dividends ?? []} splits={data?.splits ?? []} profile={profile} />
  if (code === 'CAST') return <CapitalPanel profile={profile} />
  if (code === 'CRPR') return <CreditPanel profile={profile} />

  if (code === 'ANR') return <AnalystPanel data={data} stock={stock} />
  if (code === 'EE') return <EstimatesPanel earnings={data?.earnings_estimate ?? []} revenue={data?.revenue_estimate ?? []} />
  if (code === 'EEO') return <EstimateOverview data={data} stock={stock} />
  if (code === 'EEG') return <EstimateBars data={data} />
  if (code === 'ERN') return <EarningsHistoryPanel history={data?.earnings_history ?? []} dates={data?.earnings_dates ?? []} />
  if (code === 'EM') return <GrowthTrendsPanel rows={data?.growth_estimates ?? []} />

  if (code === 'RV') return <RelativeValue profile={profile} peers={data?.peers ?? []} />
  if (code === 'RVG') return <RelativeValueBars profile={profile} peers={data?.peers ?? []} />
  if (code === 'RVC') return <CorrelationPanel ticker={ticker} peers={(data?.peers ?? []).map(peer => stringValue(peer.symbol)).filter(Boolean) as string[]} fallbackPeers={peerTickers} />
  if (code === 'WACC') return <WaccCalculator profile={profile} />
  if (code === 'DDM') return <DdmCalculator profile={profile} stock={stock} />
  if (code === 'BETA') return <BetaPanel profile={profile} stock={stock} />

  if (code === 'HDS') return <HoldersPanel institutions={data?.institutional_holders ?? []} funds={data?.mutualfund_holders ?? []} />
  if (code === 'OWN') return <OwnershipPanel profile={profile} major={data?.major_holders ?? []} institutions={data?.institutional_holders ?? []} />
  if (code === 'GPTR') return <InsiderPanel transactions={data?.insider_transactions ?? []} purchases={data?.insider_purchases ?? []} />
  if (code === 'RELS') return <ActionPanel title="Related securities" message="Ultimate parents, subsidiaries, investors, related instruments and state-ownership risks are mapped in TradeSmart Intelligence." action="Open TradeSmart Intelligence" onAction={() => onOpenIntelligence('related')} />

  if (code === 'CN') return <NewsPanel rows={[...(data?.news ?? []), ...(data?.press_releases ?? [])]} />
  if (code === 'CF') return <FilingsPanel rows={data?.sec_filings ?? []} />
  if (code === 'BRC') return <RatingChangesPanel rows={data?.upgrades_downgrades ?? []} />
  if (code === 'BI') return <IndustryPanel stock={stock} />
  if (code === 'EVTS') return <EventsPanel calendar={data?.calendar} />
  if (code === 'CACT') return <CorporateActionsPanel rows={data?.actions ?? []} />

  if (code === 'GP') return <PriceChart ticker={ticker} overlayPeers={peerTickers} initialChartType="line" />
  if (code === 'GPO') return <PriceChart ticker={ticker} overlayPeers={peerTickers} initialChartType="candlestick" />
  if (code === 'GF') return <FundamentalTrendPanel ticker={ticker} />
  if (code === 'TRA') return <ReturnComparisonPanel ticker={ticker} peers={[]} />
  if (code === 'COMP') return <ReturnComparisonPanel ticker={ticker} peers={peerTickers.slice(0, 5)} />
  if (code === 'HP') return <HistoricalPriceTable ticker={ticker} />
  if (code === 'TECH') return <TechnicalPanel stock={stock} />

  if (code === 'FTST' || code === 'EQBT' || code === 'BT') return <BacktestLink code={code} ticker={ticker} />
  return <EmptyState message="This function is not available." />
}

function DescriptionPanel({ ticker, stock, profile }: { ticker: string; stock: StockResult; profile: Record<string, unknown> }) {
  return <div className="h-full overflow-auto p-4"><div className="grid gap-4 xl:grid-cols-[1fr_380px]"><section className="border border-border bg-s1/50 p-5"><p className="text-2xs uppercase tracking-[0.18em] text-accent">{ticker} · {stringValue(profile.exchange)} · {stringValue(profile.currency)}</p><h2 className="mt-3 text-2xl font-medium text-text">{stringValue(profile.longName) || stock.company_name}</h2><p className="mt-1 text-xs text-muted">{stringValue(profile.sector) || stock.sector} / {stringValue(profile.industry) || stock.industry}</p><p className="mt-5 max-w-5xl text-xs leading-6 text-muted">{stringValue(profile.longBusinessSummary) || 'No business description was returned.'}</p>{stringValue(profile.website) && <a href={stringValue(profile.website)} target="_blank" rel="noreferrer" className="mt-4 inline-block text-2xs text-accent hover:underline">{stringValue(profile.website)}</a>}</section><section className="border border-border bg-s1/30 p-4"><Definition label="Headquarters" value={[stringValue(profile.city), stringValue(profile.state), stringValue(profile.country)].filter(Boolean).join(', ')} /><Definition label="Employees" value={integer(profile.fullTimeEmployees)} /><Definition label="Market capitalization" value={money(profile.marketCap)} /><Definition label="Enterprise value" value={money(profile.enterpriseValue)} /><Definition label="52-week range" value={`${number(profile.fiftyTwoWeekLow)} – ${number(profile.fiftyTwoWeekHigh)}`} /></section></div></div>
}

function QuotePanel({ stock, profile }: { stock: StockResult; profile: Record<string, unknown> }) {
  const dayLow = numeric(profile.dayLow), dayHigh = numeric(profile.dayHigh), yearLow = numeric(profile.fiftyTwoWeekLow), yearHigh = numeric(profile.fiftyTwoWeekHigh)
  return <div className="h-full overflow-auto p-4"><div className="grid gap-4 xl:grid-cols-[360px_1fr]"><section className="border border-border bg-s1/60 p-6"><p className="text-2xs uppercase tracking-[0.18em] text-dim">Last price</p><p className="mt-3 text-5xl font-semibold tabnum text-text">{money(stock.current_price, false)}</p><p className={`mt-2 text-lg tabnum ${stock.return_1d >= 0 ? 'text-up' : 'text-down'}`}>{stock.return_1d >= 0 ? '+' : ''}{stock.return_1d.toFixed(2)}%</p><div className="mt-8 grid grid-cols-2 gap-4"><Figure label="Bid" value={money(profile.bid, false)} /><Figure label="Ask" value={money(profile.ask, false)} /><Figure label="Open" value={money(profile.open, false)} /><Figure label="Previous" value={money(profile.previousClose, false)} /></div></section><section className="space-y-4"><RangeCard label="Session range" low={dayLow} high={dayHigh} value={stock.current_price} /><RangeCard label="52-week range" low={yearLow} high={yearHigh} value={stock.current_price} /><div className="grid grid-cols-2 gap-4"><section className="border border-border bg-s1/40 p-4"><p className="text-2xs uppercase text-dim">Volume</p><p className="mt-2 text-xl tabnum text-text">{integer(profile.volume)}</p></section><section className="border border-border bg-s1/40 p-4"><p className="text-2xs uppercase text-dim">Average volume</p><p className="mt-2 text-xl tabnum text-text">{integer(profile.averageVolume)}</p></section></div></section></div></div>
}

function MarketOverviewPanel() {
  const query = useQuery({ queryKey: ['equity-research-market-overview'], queryFn: api.getMarketOverview, staleTime: 5 * 60 * 1000 })
  const data = query.data
  if (!data) return <EmptyState message={query.isLoading ? 'Loading market overview…' : 'Market overview is unavailable.'} />
  const total = (data.advances ?? 0) + (data.declines ?? 0) || 1
  return <div className="h-full overflow-auto p-4"><div className="flex items-stretch border border-border bg-s1/50"><div className="min-w-56 p-4"><p className="text-2xs uppercase text-dim">Market regime</p><p className="mt-2 text-2xl font-medium text-text">{data.market_trend}</p></div><div className="flex-1 border-l border-border p-4"><div className="flex justify-between text-2xs"><span className="text-up">{data.advances ?? 0} advancing</span><span className="text-down">{data.declines ?? 0} declining</span></div><div className="mt-3 flex h-4 overflow-hidden bg-s2"><div className="bg-up" style={{ width: `${(data.advances ?? 0) / total * 100}%` }} /><div className="bg-down" style={{ width: `${(data.declines ?? 0) / total * 100}%` }} /></div></div></div><div className="mt-4 overflow-auto border border-border"><table className="w-full text-left text-2xs"><thead className="bg-s2 text-[9px] uppercase tracking-wider text-dim"><tr><th className="px-3 py-2">Sector</th><th className="px-3 py-2 text-right">1D</th><th className="px-3 py-2 text-right">1W</th><th className="px-3 py-2 text-right">1M</th><th className="px-3 py-2">Trend</th></tr></thead><tbody className="divide-y divide-border">{data.sectors.map(sector => <tr key={sector.name} className="bg-bg/30"><td className="px-3 py-2 text-text">{sector.name}</td><td className={`px-3 py-2 text-right tabnum ${sector.return_1d >= 0 ? 'text-up' : 'text-down'}`}>{sector.return_1d.toFixed(2)}%</td><td className="px-3 py-2 text-right tabnum text-muted">{number(sector.return_1w)}%</td><td className="px-3 py-2 text-right tabnum text-muted">{number(sector.return_1m)}%</td><td className="px-3 py-2 text-muted">{sector.trend ?? '—'}</td></tr>)}</tbody></table></div></div>
}

function CapitalPanel({ profile }: { profile: Record<string, unknown> }) {
  const equity = Math.max(numeric(profile.marketCap) ?? 0, 0), debt = Math.max(numeric(profile.totalDebt) ?? 0, 0), total = equity + debt || 1
  return <div className="h-full overflow-auto p-4"><section className="border border-border bg-s1/50 p-5"><div className="flex items-center justify-between"><div><p className="text-2xs uppercase tracking-[0.18em] text-dim">Capital mix</p><p className="mt-2 text-2xl tabnum text-text">{money(equity + debt)}</p></div><div className="text-right"><p className="text-2xs text-dim">Enterprise value</p><p className="mt-1 text-xl tabnum text-accent">{money(profile.enterpriseValue)}</p></div></div><div className="mt-6 flex h-8 overflow-hidden"><div className="flex items-center bg-blue-500/70 px-3 text-2xs text-white" style={{ width: `${equity / total * 100}%` }}>Equity {(equity / total * 100).toFixed(1)}%</div><div className="flex items-center justify-end bg-accent/80 px-3 text-2xs text-bg" style={{ width: `${debt / total * 100}%` }}>Debt {(debt / total * 100).toFixed(1)}%</div></div><div className="mt-5 grid grid-cols-2 gap-4 lg:grid-cols-4"><Figure label="Cash" value={money(profile.totalCash)} /><Figure label="Debt / equity" value={percent(profile.debtToEquity)} /><Figure label="Shares outstanding" value={integer(profile.sharesOutstanding)} /><Figure label="Book value / share" value={number(profile.bookValue)} /></div></section></div>
}

function CreditPanel({ profile }: { profile: Record<string, unknown> }) {
  const debt = Math.max(numeric(profile.totalDebt) ?? 0, 0), cash = Math.max(numeric(profile.totalCash) ?? 0, 0), coverage = debt ? Math.min(100, cash / debt * 100) : 100
  return <div className="h-full overflow-auto p-4"><div className="grid gap-4 lg:grid-cols-2"><section className="border border-border bg-s1/50 p-5"><p className="text-2xs uppercase tracking-[0.18em] text-dim">Liquidity against debt</p><div className="mt-5 h-5 bg-s2"><div className="h-full bg-up" style={{ width: `${coverage}%` }} /></div><div className="mt-3 flex justify-between text-xs"><span className="text-muted">Cash {money(cash)}</span><span className="text-muted">Debt {money(debt)}</span></div><p className="mt-5 text-3xl tabnum text-text">{coverage.toFixed(1)}%</p><p className="text-2xs text-dim">cash coverage of reported debt</p></section><section className="border border-border bg-s1/30 p-5"><Definition label="Debt / equity" value={percent(profile.debtToEquity)} /><Definition label="Free cash flow" value={money(profile.freeCashflow)} /><Definition label="Operating cash flow" value={money(profile.operatingCashflow)} /><Definition label="Overall governance risk" value={number(profile.overallRisk)} /><Definition label="Audit / board risk" value={`${number(profile.auditRisk)} / ${number(profile.boardRisk)}`} /></section></div></div>
}

function DividendsPanel({ dividends, splits, profile }: { dividends: Array<Record<string, unknown>>; splits: Array<Record<string, unknown>>; profile: Record<string, unknown> }) {
  const recent = dividends.slice(-16).reverse()
  const max = Math.max(...recent.map(row => numeric(getField(row, ['value', 'Dividends'])) ?? 0), 1)
  return <div className="h-full overflow-auto p-4"><div className="grid gap-4 lg:grid-cols-[280px_1fr]"><section className="border border-border bg-s1/50 p-5"><p className="text-2xs uppercase tracking-[0.18em] text-dim">Current distribution</p><p className="mt-4 text-3xl font-semibold tabnum text-accent">{money(profile.dividendRate, false)}</p><p className="mt-1 text-xs text-muted">annual dividend per share</p><div className="mt-6 flex justify-between border-t border-border pt-3 text-xs"><span className="text-dim">Yield</span><span className="tabnum text-text">{percentageRatio(profile.dividendYield)}</span></div><div className="mt-2 flex justify-between text-xs"><span className="text-dim">Payout ratio</span><span className="tabnum text-text">{percentageRatio(profile.payoutRatio)}</span></div><div className="mt-2 flex justify-between text-xs"><span className="text-dim">Recorded splits</span><span className="tabnum text-text">{splits.length}</span></div></section><section className="border border-border bg-s1/30 p-4"><p className="mb-4 text-2xs uppercase tracking-[0.18em] text-dim">Recent dividend history</p><div className="space-y-2">{recent.map((row, index) => { const value = numeric(getField(row, ['value', 'Dividends'])) ?? 0; return <div key={index} className="grid grid-cols-[120px_1fr_70px] items-center gap-3 text-2xs"><span className="truncate tabnum text-dim">{shortDate(getField(row, ['Date', 'index']))}</span><div className="h-3 bg-s2"><div className="h-full bg-accent" style={{ width: `${Math.max(2, value / max * 100)}%` }} /></div><span className="text-right tabnum text-text">{money(value, false)}</span></div>})}{!recent.length && <EmptyState message="No dividend history was returned." />}</div></section></div></div>
}

function AnalystPanel({ data, stock }: { data?: EquityResearchResponse; stock: StockResult }) {
  const summary = data?.recommendations_summary?.[0] ?? {}
  const ratings = [['Strong buy', numeric(getField(summary, ['strongBuy'])) ?? 0, '#10B981'], ['Buy', numeric(getField(summary, ['buy'])) ?? 0, '#34D399'], ['Hold', numeric(getField(summary, ['hold'])) ?? 0, '#E89B2C'], ['Sell', numeric(getField(summary, ['sell'])) ?? 0, '#F97316'], ['Strong sell', numeric(getField(summary, ['strongSell'])) ?? 0, '#E5484D']] as const
  const total = ratings.reduce((sum, [, value]) => sum + value, 0)
  const targets = asObject(data?.analyst_price_targets)
  const low = numeric(targets.low), mean = numeric(targets.mean), high = numeric(targets.high)
  const range = low != null && high != null && high > low ? high - low : 1
  return <div className="h-full overflow-auto p-4"><div className="grid gap-4 xl:grid-cols-2"><section className="border border-border bg-s1/50 p-5"><div className="flex items-center justify-between"><p className="text-2xs uppercase tracking-[0.18em] text-dim">Recommendation distribution</p><span className="text-xs tabnum text-muted">{total} ratings</span></div><div className="mt-5 flex h-7 overflow-hidden bg-s2">{ratings.map(([label, value, color]) => value > 0 && <div key={label} title={`${label}: ${value}`} style={{ width: `${value / total * 100}%`, backgroundColor: color }} />)}</div><div className="mt-4 grid grid-cols-5 gap-2">{ratings.map(([label, value, color]) => <div key={label}><span className="block h-1" style={{ backgroundColor: color }} /><span className="mt-1 block text-[9px] text-dim">{label}</span><span className="block text-sm tabnum text-text">{value}</span></div>)}</div></section><section className="border border-border bg-s1/50 p-5"><p className="text-2xs uppercase tracking-[0.18em] text-dim">Target price range</p><div className="mt-5 grid grid-cols-4 text-center"><Figure label="Low" value={money(low, false)} /><Figure label="Mean" value={money(mean, false)} accent /><Figure label="High" value={money(high, false)} /><Figure label="Current" value={money(stock.current_price, false)} /></div>{low != null && high != null && <div className="relative mt-8 h-2 bg-gradient-to-r from-down via-accent to-up"><span className="absolute -top-1 h-4 w-0.5 bg-white" style={{ left: `${Math.max(0, Math.min(100, (stock.current_price - low) / range * 100))}%` }} /></div>}<p className="mt-4 text-center text-xs text-muted">Mean target implies <span className={mean != null && mean >= stock.current_price ? 'text-up' : 'text-down'}>{mean == null ? '—' : `${((mean / stock.current_price - 1) * 100).toFixed(1)}%`}</span></p></section></div></div>
}

function EstimatesPanel({ earnings, revenue }: { earnings: Array<Record<string, unknown>>; revenue: Array<Record<string, unknown>> }) {
  const [mode, setMode] = useState<'earnings' | 'revenue'>('earnings')
  const rows = mode === 'earnings' ? earnings : revenue
  return <div className="flex h-full min-h-0 flex-col"><div className="flex shrink-0 border-b border-border bg-s1/40 px-3">{(['earnings', 'revenue'] as const).map(item => <button key={item} onClick={() => setMode(item)} className={`border-b-2 px-4 py-2 text-2xs uppercase ${mode === item ? 'border-accent text-accent' : 'border-transparent text-dim'}`}>{item}</button>)}</div><div className="min-h-0 flex-1 overflow-auto"><table className="w-full text-left text-xs"><thead className="sticky top-0 bg-s2 text-[9px] uppercase tracking-wider text-dim"><tr><th className="px-4 py-2">Period</th><th className="px-4 py-2 text-right">Low</th><th className="px-4 py-2 text-right">Average</th><th className="px-4 py-2 text-right">High</th><th className="px-4 py-2 text-right">Year ago</th><th className="px-4 py-2 text-right">Analysts</th><th className="px-4 py-2">Range</th></tr></thead><tbody className="divide-y divide-border">{rows.map((row, index) => { const low = numeric(getField(row, ['low', 'lowEstimate'])); const avg = numeric(getField(row, ['avg', 'avgEstimate'])); const high = numeric(getField(row, ['high', 'highEstimate'])); const yearAgo = numeric(getField(row, ['yearAgo', 'yearAgoEps', 'yearAgoRevenue'])); return <tr key={index} className="bg-bg/30"><td className="px-4 py-3 font-medium text-text">{display(getField(row, ['period', 'index']))}</td><td className="px-4 py-3 text-right tabnum text-muted">{compactMaybe(low)}</td><td className="px-4 py-3 text-right tabnum font-medium text-accent">{compactMaybe(avg)}</td><td className="px-4 py-3 text-right tabnum text-muted">{compactMaybe(high)}</td><td className="px-4 py-3 text-right tabnum text-dim">{compactMaybe(yearAgo)}</td><td className="px-4 py-3 text-right tabnum text-muted">{integer(getField(row, ['numberOfAnalysts', 'numberOfAnalysts']))}</td><td className="w-56 px-4 py-3"><RangeBar low={low} value={avg} high={high} /></td></tr>})}</tbody></table>{!rows.length && <EmptyState message="No estimate series was returned." />}</div></div>
}

function EarningsHistoryPanel({ history, dates }: { history: Array<Record<string, unknown>>; dates: Array<Record<string, unknown>> }) {
  const rows = history.length ? history : dates
  const surprises = rows.map(row => numeric(getField(row, ['surprisePercent', 'surprise(%)', 'surprise'])) ?? 0)
  const max = Math.max(...surprises.map(Math.abs), 1)
  return <div className="h-full overflow-auto p-4"><div className="space-y-2">{rows.map((row, index) => { const surprise = surprises[index]; const estimate = getField(row, ['epsEstimate', 'estimate']); const actual = getField(row, ['epsActual', 'reportedEPS', 'actual']); return <div key={index} className="grid grid-cols-[130px_90px_90px_1fr_70px] items-center gap-3 border-b border-border py-2 text-2xs"><span className="tabnum text-dim">{shortDate(getField(row, ['quarter', 'Earnings Date', 'index']))}</span><span className="text-right tabnum text-muted">Est {compactMaybe(numeric(estimate))}</span><span className="text-right tabnum text-text">Act {compactMaybe(numeric(actual))}</span><div className="relative h-4 bg-s2"><span className="absolute left-1/2 top-0 h-full w-px bg-border-strong" /><span className={`absolute top-1 h-2 ${surprise >= 0 ? 'left-1/2 bg-up' : 'right-1/2 bg-down'}`} style={{ width: `${Math.abs(surprise) / max * 50}%` }} /></div><span className={`text-right tabnum ${surprise >= 0 ? 'text-up' : 'text-down'}`}>{surprise.toFixed(1)}%</span></div>})}{!rows.length && <EmptyState message="No earnings history was returned." />}</div></div>
}

function GrowthTrendsPanel({ rows }: { rows: Array<Record<string, unknown>> }) {
  const points = rows.flatMap(row => Object.entries(row).filter(([key, value]) => key !== 'index' && typeof value === 'number').map(([key, value]) => ({ label: `${display(getField(row, ['index', 'period']))} · ${key}`, value: value as number }))).slice(0, 24)
  const max = Math.max(...points.map(point => Math.abs(point.value)), 0.01)
  return <div className="h-full overflow-auto p-4"><div className="grid gap-3 lg:grid-cols-2">{points.map((point, index) => <div key={index} className="border border-border bg-s1/40 p-3"><div className="flex justify-between text-2xs"><span className="text-muted">{point.label}</span><span className={point.value >= 0 ? 'text-up' : 'text-down'}>{percentageRatio(point.value)}</span></div><div className="mt-2 h-2 bg-s2"><div className={point.value >= 0 ? 'h-full bg-up' : 'h-full bg-down'} style={{ width: `${Math.abs(point.value) / max * 100}%` }} /></div></div>)}</div>{!points.length && <EmptyState message="No growth trend series was returned." />}</div>
}

function EstimateOverview({ data, stock }: { data?: EquityResearchResponse; stock: StockResult }) {
  const targets = asObject(data?.analyst_price_targets)
  const mean = numeric(targets.mean)
  return <div className="h-full overflow-auto p-4"><div className="grid gap-4 lg:grid-cols-[1fr_1.5fr]"><section className="border border-border bg-s1/50 p-5"><p className="text-2xs uppercase tracking-[0.18em] text-dim">Consensus target</p><p className="mt-4 text-4xl font-semibold tabnum text-accent">{money(mean, false)}</p><p className={`mt-2 text-sm ${mean != null && mean >= stock.current_price ? 'text-up' : 'text-down'}`}>{mean == null ? 'No target' : `${((mean / stock.current_price - 1) * 100).toFixed(1)}% implied return`}</p><div className="mt-6 grid grid-cols-3 gap-3"><Figure label="Low" value={money(targets.low, false)} /><Figure label="Median" value={money(targets.median, false)} /><Figure label="High" value={money(targets.high, false)} /></div></section><section className="min-h-[360px] border border-border bg-s1/30"><EstimatesPanel earnings={data?.earnings_estimate ?? []} revenue={data?.revenue_estimate ?? []} /></section></div></div>
}

function EstimateBars({ data }: { data?: EquityResearchResponse }) {
  const rows = [...(data?.earnings_estimate ?? []), ...(data?.revenue_estimate ?? [])]
  const points = rows.flatMap(row => Object.entries(row).filter(([, value]) => typeof value === 'number').map(([key, value]) => ({ label: `${stringValue(row.index) || stringValue(row.period)} ${key}`, value: value as number }))).slice(0, 16)
  const max = Math.max(...points.map(point => Math.abs(point.value)), 1)
  return <div className="h-full overflow-auto p-4"><div className="space-y-2">{points.map((point, index) => <div key={`${point.label}-${index}`} className="grid grid-cols-[220px_1fr_100px] items-center gap-3 text-2xs"><span className="truncate text-muted">{point.label}</span><div className="h-3 bg-s2"><div className="h-full bg-accent/80" style={{ width: `${Math.max(1, Math.abs(point.value) / max * 100)}%` }} /></div><span className="text-right tabnum text-text">{compactNumber(point.value)}</span></div>)}</div>{!points.length && <EmptyState message="No numeric estimate series are available." />}</div>
}

const VALUATION_FIELDS = [
  ['trailingPE', 'P/E'], ['forwardPE', 'Fwd P/E'], ['priceToBook', 'P/B'], ['priceToSalesTrailing12Months', 'P/S'], ['enterpriseToRevenue', 'EV/Revenue'], ['enterpriseToEbitda', 'EV/EBITDA'], ['pegRatio', 'PEG'],
] as const

function RelativeValue({ profile, peers }: { profile: Record<string, unknown>; peers: Array<Record<string, unknown>> }) {
  const rows = [profile, ...peers]
  return <div className="h-full overflow-auto"><table className="w-full min-w-[900px] text-left text-2xs"><thead className="sticky top-0 bg-s2 text-[9px] uppercase tracking-wider text-dim"><tr><th className="px-3 py-2">Security</th>{VALUATION_FIELDS.map(([, label]) => <th key={label} className="px-3 py-2 text-right">{label}</th>)}</tr></thead><tbody className="divide-y divide-border">{rows.map((row, index) => <tr key={`${stringValue(row.symbol)}-${index}`} className={index === 0 ? 'bg-accent/5 text-accent' : 'bg-bg/30 text-muted'}><td className="px-3 py-2.5 font-medium tabnum">{stringValue(row.symbol) || stringValue(row.shortName) || '—'}</td>{VALUATION_FIELDS.map(([key]) => <td key={key} className="px-3 py-2.5 text-right tabnum">{number(row[key])}</td>)}</tr>)}</tbody></table></div>
}

function RelativeValueBars({ profile, peers }: { profile: Record<string, unknown>; peers: Array<Record<string, unknown>> }) {
  const rows = [profile, ...peers].map(row => ({ symbol: stringValue(row.symbol) || '—', value: numeric(row.forwardPE) ?? numeric(row.trailingPE) ?? 0 }))
  const max = Math.max(...rows.map(row => row.value), 1)
  return <div className="h-full overflow-auto p-4"><p className="mb-4 text-2xs uppercase tracking-[0.18em] text-dim">Forward P/E comparison</p><div className="space-y-3">{rows.map((row, index) => <div key={`${row.symbol}-${index}`} className="grid grid-cols-[80px_1fr_70px] items-center gap-3"><span className="font-mono text-xs text-text">{row.symbol}</span><div className="h-5 bg-s2"><div className={`h-full ${index === 0 ? 'bg-accent' : 'bg-blue-500/60'}`} style={{ width: `${Math.max(1, row.value / max * 100)}%` }} /></div><span className="text-right text-xs tabnum text-muted">{number(row.value)}x</span></div>)}</div></div>
}

function CorrelationPanel({ ticker, peers, fallbackPeers }: { ticker: string; peers: string[]; fallbackPeers: string[] }) {
  const symbols = [ticker, ...(peers.length ? peers : fallbackPeers).slice(0, 5)]
  const query = useQuery({ queryKey: ['correlation-history', symbols.join('|')], queryFn: async () => Promise.all(symbols.map(symbol => api.getPriceHistory(symbol, 365))), staleTime: 10 * 60 * 1000 })
  const values = useMemo(() => {
    const histories = query.data ?? []
    if (histories.length < 2) return []
    const base = returns(histories[0].close)
    return symbols.slice(1).map((symbol, index) => ({ symbol, correlation: correlation(base, returns(histories[index + 1].close)) }))
  }, [query.data, symbols.join('|')])
  return <div className="h-full overflow-auto p-4"><div className="space-y-3">{values.map(row => <div key={row.symbol} className="grid grid-cols-[100px_1fr_80px] items-center gap-3"><span className="font-mono text-xs text-text">{row.symbol}</span><div className="relative h-5 bg-s2"><span className="absolute left-1/2 h-full w-px bg-border-strong" /><span className={`absolute top-1 h-3 ${row.correlation >= 0 ? 'left-1/2 bg-up' : 'right-1/2 bg-down'}`} style={{ width: `${Math.abs(row.correlation) * 50}%` }} /></div><span className={`text-right tabnum ${row.correlation >= 0 ? 'text-up' : 'text-down'}`}>{row.correlation.toFixed(3)}</span></div>)}</div>{query.isLoading && <EmptyState message="Calculating one-year daily return correlations…" />}</div>
}

function WaccCalculator({ profile }: { profile: Record<string, unknown> }) {
  const beta = numeric(profile.beta) ?? 1
  const debt = Math.max(numeric(profile.totalDebt) ?? 0, 0)
  const equity = Math.max(numeric(profile.marketCap) ?? 0, 0)
  const [riskFree, setRiskFree] = useState(4.25)
  const [premium, setPremium] = useState(5.0)
  const [debtCost, setDebtCost] = useState(6.0)
  const [taxRate, setTaxRate] = useState(25.0)
  const total = debt + equity || 1
  const costEquity = riskFree + beta * premium
  const wacc = equity / total * costEquity + debt / total * debtCost * (1 - taxRate / 100)
  return <Calculator title="Weighted average cost of capital" result={`${wacc.toFixed(2)}%`}><Input label="Risk-free rate %" value={riskFree} onChange={setRiskFree} /><Input label="Equity risk premium %" value={premium} onChange={setPremium} /><Input label="Pre-tax debt cost %" value={debtCost} onChange={setDebtCost} /><Input label="Tax rate %" value={taxRate} onChange={setTaxRate} /><ReadOnly label="Beta" value={beta.toFixed(2)} /><ReadOnly label="Cost of equity" value={`${costEquity.toFixed(2)}%`} /></Calculator>
}

function DdmCalculator({ profile, stock }: { profile: Record<string, unknown>; stock: StockResult }) {
  const [growth, setGrowth] = useState(2.5)
  const [required, setRequired] = useState(9.0)
  const dividend = numeric(profile.dividendRate) ?? 0
  const value = required > growth ? dividend * (1 + growth / 100) / ((required - growth) / 100) : null
  return <Calculator title="Gordon growth dividend model" result={value == null ? 'Invalid spread' : money(value, false)}><ReadOnly label="Annual dividend / share" value={money(dividend, false)} /><Input label="Long-term growth %" value={growth} onChange={setGrowth} /><Input label="Required return %" value={required} onChange={setRequired} /><ReadOnly label="Current price" value={money(stock.current_price, false)} /><ReadOnly label="Implied upside" value={value == null ? '—' : `${((value / stock.current_price - 1) * 100).toFixed(1)}%`} /></Calculator>
}

function BetaPanel({ profile, stock }: { profile: Record<string, unknown>; stock: StockResult }) {
  const beta = Math.max(0, Math.min(2, numeric(profile.beta) ?? 1))
  return <div className="h-full overflow-auto p-4"><div className="mx-auto max-w-3xl border border-border bg-s1/50 p-6"><p className="text-2xs uppercase tracking-[0.18em] text-dim">Market sensitivity</p><p className="mt-4 text-5xl font-semibold tabnum text-accent">{beta.toFixed(3)}</p><div className="relative mt-8 h-4 bg-gradient-to-r from-blue-500 via-accent to-down"><span className="absolute -top-2 h-8 w-0.5 bg-white" style={{ left: `${beta / 2 * 100}%` }} /></div><div className="mt-2 flex justify-between text-[9px] text-dim"><span>0 · defensive</span><span>1 · market</span><span>2 · high sensitivity</span></div><div className="mt-8 grid grid-cols-3 gap-4"><Figure label="ATR / price" value={`${stock.atr_pct.toFixed(2)}%`} /><Figure label="1D return" value={`${stock.return_1d.toFixed(2)}%`} /><Figure label="TradeSmart score" value={stock.day_trading_score.toFixed(1)} accent /></div></div></div>
}

function HoldersPanel({ institutions, funds }: { institutions: Array<Record<string, unknown>>; funds: Array<Record<string, unknown>> }) {
  const [mode, setMode] = useState<'institutions' | 'funds'>('institutions')
  const rows = mode === 'institutions' ? institutions : funds
  const ranked = rows.map(row => ({ row, pct: normalizePct(getField(row, ['pctHeld', '% Out', 'percentHeld'])), value: numeric(getField(row, ['Value', 'value'])) ?? 0 })).sort((a, b) => b.pct - a.pct)
  const max = Math.max(...ranked.map(item => item.pct), 0.01)
  return <div className="flex h-full min-h-0 flex-col"><div className="flex shrink-0 border-b border-border bg-s1/40 px-3">{(['institutions', 'funds'] as const).map(item => <button key={item} onClick={() => setMode(item)} className={`border-b-2 px-4 py-2 text-2xs uppercase ${mode === item ? 'border-accent text-accent' : 'border-transparent text-dim'}`}>{item}</button>)}</div><div className="min-h-0 flex-1 overflow-auto p-4"><div className="space-y-2">{ranked.map(({ row, pct, value }, index) => <div key={index} className="grid grid-cols-[minmax(180px,1fr)_220px_80px_110px] items-center gap-3 border-b border-border py-2"><div><p className="truncate text-xs font-medium text-text">{display(getField(row, ['Holder', 'holder', 'Organization']))}</p><p className="text-[9px] tabnum text-dim">{shortDate(getField(row, ['Date Reported', 'dateReported']))}</p></div><div className="h-2 bg-s2"><div className="h-full bg-accent" style={{ width: `${pct / max * 100}%` }} /></div><span className="text-right text-xs tabnum text-accent">{pct.toFixed(2)}%</span><span className="text-right text-xs tabnum text-muted">{money(value)}</span></div>)}{!ranked.length && <EmptyState message="No holder records were returned." />}</div></div></div>
}

function OwnershipPanel({ profile, major, institutions }: { profile: Record<string, unknown>; major: Array<Record<string, unknown>>; institutions: Array<Record<string, unknown>> }) {
  const insiders = normalizePct(profile.heldPercentInsiders)
  const institutional = normalizePct(profile.heldPercentInstitutions)
  const publicShare = Math.max(0, 100 - insiders - institutional)
  return <div className="h-full overflow-auto p-4"><div className="grid gap-4 xl:grid-cols-[360px_1fr]"><section className="border border-border bg-s1/50 p-5"><p className="text-2xs uppercase tracking-[0.18em] text-dim">Ownership composition</p><div className="mx-auto mt-6 h-52 w-52 rounded-full" style={{ background: `conic-gradient(#E89B2C 0 ${insiders}%, #3B82F6 ${insiders}% ${insiders + institutional}%, #33343A ${insiders + institutional}% 100%)` }}><div className="relative left-10 top-10 flex h-32 w-32 items-center justify-center rounded-full bg-bg text-center"><span><strong className="block text-2xl tabnum text-text">{integer(profile.sharesOutstanding)}</strong><small className="text-[9px] uppercase text-dim">shares</small></span></div></div><div className="mt-5 space-y-2 text-xs"><Legend color="#E89B2C" label="Insiders" value={`${insiders.toFixed(2)}%`} /><Legend color="#3B82F6" label="Institutions" value={`${institutional.toFixed(2)}%`} /><Legend color="#33343A" label="Other / public" value={`${publicShare.toFixed(2)}%`} /></div></section><section className="border border-border bg-s1/30 p-4"><p className="mb-3 text-2xs uppercase tracking-[0.18em] text-dim">Concentration summary</p><div className="grid grid-cols-2 gap-px bg-border"><Figure label="Institution records" value={String(institutions.length)} /><Figure label="Major-holder rows" value={String(major.length)} /><Figure label="Float shares" value={integer(profile.floatShares)} /><Figure label="Market capitalization" value={money(profile.marketCap)} accent /></div><div className="mt-4 space-y-2">{major.slice(0, 8).map((row, index) => <div key={index} className="flex justify-between border-b border-border py-2 text-xs"><span className="text-muted">{display(getField(row, ['Breakdown', 'index', 'category']))}</span><span className="tabnum text-text">{display(getField(row, ['Value', 'value']))}</span></div>)}</div></section></div></div>
}

function InsiderPanel({ transactions, purchases }: { transactions: Array<Record<string, unknown>>; purchases: Array<Record<string, unknown>> }) {
  const rows = transactions.length ? transactions : purchases
  return <div className="h-full overflow-auto"><table className="w-full min-w-[900px] text-left text-2xs"><thead className="sticky top-0 bg-s2 text-[9px] uppercase tracking-wider text-dim"><tr><th className="px-4 py-2">Date</th><th className="px-4 py-2">Insider</th><th className="px-4 py-2">Position</th><th className="px-4 py-2">Transaction</th><th className="px-4 py-2 text-right">Shares</th><th className="px-4 py-2 text-right">Value</th><th className="px-4 py-2">Ownership</th></tr></thead><tbody className="divide-y divide-border">{rows.map((row, index) => { const action = display(getField(row, ['Text', 'Transaction', 'transactionText'])); const buy = /buy|purchase|acquisition/i.test(action); return <tr key={index} className="bg-bg/30"><td className="px-4 py-3 tabnum text-dim">{shortDate(getField(row, ['Start Date', 'startDate', 'Date']))}</td><td className="px-4 py-3 font-medium text-text">{display(getField(row, ['Insider', 'insider', 'Name']))}</td><td className="px-4 py-3 text-muted">{display(getField(row, ['Position', 'position']))}</td><td className={`px-4 py-3 ${buy ? 'text-up' : 'text-accent'}`}>{action}</td><td className="px-4 py-3 text-right tabnum text-muted">{integer(getField(row, ['Shares', 'shares']))}</td><td className="px-4 py-3 text-right tabnum text-text">{money(getField(row, ['Value', 'value']))}</td><td className="px-4 py-3 text-muted">{display(getField(row, ['Ownership', 'ownership']))}</td></tr>})}</tbody></table>{!rows.length && <EmptyState message="No insider transactions were returned." />}</div>
}

function RatingChangesPanel({ rows }: { rows: Array<Record<string, unknown>> }) {
  return <div className="h-full overflow-auto"><table className="w-full text-left text-xs"><thead className="sticky top-0 bg-s2 text-[9px] uppercase tracking-wider text-dim"><tr><th className="px-4 py-2">Date</th><th className="px-4 py-2">Firm</th><th className="px-4 py-2">Action</th><th className="px-4 py-2">From</th><th className="px-4 py-2">To</th></tr></thead><tbody className="divide-y divide-border">{rows.map((row, index) => { const action = display(getField(row, ['Action', 'action'])); return <tr key={index} className="bg-bg/30"><td className="px-4 py-3 tabnum text-dim">{shortDate(getField(row, ['GradeDate', 'date', 'index']))}</td><td className="px-4 py-3 font-medium text-text">{display(getField(row, ['Firm', 'firm']))}</td><td className={`px-4 py-3 ${/up/i.test(action) ? 'text-up' : /down/i.test(action) ? 'text-down' : 'text-accent'}`}>{action}</td><td className="px-4 py-3 text-muted">{display(getField(row, ['FromGrade', 'fromGrade']))}</td><td className="px-4 py-3 font-medium text-text">{display(getField(row, ['ToGrade', 'toGrade']))}</td></tr>})}</tbody></table>{!rows.length && <EmptyState message="No rating changes were returned." />}</div>
}

function EventsPanel({ calendar }: { calendar: unknown }) {
  const rows = objectRows(calendar)
  return <div className="h-full overflow-auto p-4"><div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">{rows.map((row, index) => <article key={index} className="border border-border bg-s1/50 p-4"><p className="text-[9px] uppercase tracking-[0.16em] text-dim">{display(row.field)}</p><p className="mt-3 text-sm font-medium tabnum text-text">{display(row.value)}</p></article>)}</div>{!rows.length && <EmptyState message="No scheduled company events were returned." />}</div>
}

function CorporateActionsPanel({ rows }: { rows: Array<Record<string, unknown>> }) {
  return <div className="h-full overflow-auto p-4"><div className="relative ml-3 border-l border-border pl-6">{rows.slice().reverse().map((row, index) => { const dividend = numeric(getField(row, ['Dividends', 'dividends'])) ?? 0; const split = numeric(getField(row, ['Stock Splits', 'splits'])) ?? 0; return <article key={index} className="relative mb-4 border border-border bg-s1/50 p-4"><span className="absolute -left-[31px] top-5 h-2.5 w-2.5 rounded-full bg-accent" /><p className="text-2xs tabnum text-dim">{shortDate(getField(row, ['Date', 'index']))}</p><div className="mt-2 flex gap-6"><Figure label="Dividend" value={money(dividend, false)} accent={dividend > 0} /><Figure label="Split ratio" value={split ? `${split.toFixed(3)}x` : '—'} /></div></article>})}</div>{!rows.length && <EmptyState message="No corporate actions were returned." />}</div>
}

function NewsPanel({ rows }: { rows: Array<Record<string, unknown>> }) {
  const items = rows.map(flattenNews).filter(item => item.title)
  return <div className="h-full overflow-auto divide-y divide-border">{items.map((item, index) => <a key={`${item.url}-${index}`} href={item.url || '#'} target="_blank" rel="noreferrer" className="block px-4 py-3 hover:bg-s1"><div className="flex items-center gap-3"><span className="text-xs font-medium text-text">{item.title}</span><span className="ml-auto shrink-0 text-[9px] tabnum text-dim">{item.date}</span></div><p className="mt-1 line-clamp-2 text-2xs leading-relaxed text-muted">{item.summary}</p><span className="mt-1 block text-[9px] text-accent">{item.publisher}</span></a>)}{!items.length && <EmptyState message="No current company news was returned." />}</div>
}

function FilingsPanel({ rows }: { rows: Array<Record<string, unknown>> }) {
  return <div className="h-full overflow-auto p-4"><div className="grid gap-3 lg:grid-cols-2">{rows.map((row, index) => { const title = display(getField(row, ['title', 'type', 'formType'])); const url = stringValue(getField(row, ['edgarUrl', 'url', 'link'])); return <article key={index} className="border border-border bg-s1/50 p-4"><div className="flex items-start justify-between gap-3"><div><p className="text-xs font-medium text-text">{title}</p><p className="mt-1 text-[9px] tabnum text-dim">{shortDate(getField(row, ['date', 'filingDate', 'epochDate']))}</p></div>{url && <a href={url} target="_blank" rel="noreferrer" className="text-2xs text-accent hover:underline">Open document</a>}</div><p className="mt-3 line-clamp-3 text-2xs leading-relaxed text-muted">{display(getField(row, ['description', 'summary', 'type']))}</p></article>})}</div>{!rows.length && <EmptyState message="No company filings were returned." />}</div>
}

function IndustryPanel({ stock }: { stock: StockResult }) {
  const sentiment = Math.max(-1, Math.min(1, stock.news_sentiment_score))
  return <div className="h-full overflow-auto p-4"><div className="grid gap-4 lg:grid-cols-[1.2fr_1fr]"><section className="border border-border bg-s1/50 p-5"><p className="text-2xs uppercase tracking-[0.18em] text-accent">Classification</p><h3 className="mt-3 text-2xl font-medium text-text">{stock.sector}</h3><p className="mt-1 text-sm text-muted">{stock.industry}</p><div className="mt-6 flex items-end gap-3"><span className="text-4xl font-semibold tabnum text-accent">{stock.day_trading_score.toFixed(1)}</span><span className="pb-1 text-xs text-muted">{stock.day_trading_strategy}</span></div></section><section className="border border-border bg-s1/50 p-5"><p className="text-2xs uppercase tracking-[0.18em] text-dim">News sentiment</p><div className="mt-5 h-3 overflow-hidden bg-gradient-to-r from-down via-border to-up"><span className="block h-full w-0.5 bg-white" style={{ marginLeft: `${(sentiment + 1) * 50}%` }} /></div><div className="mt-3 flex justify-between text-2xs text-dim"><span>Negative</span><span className="text-text">{stock.news_sentiment_label} · {stock.news_sentiment_score.toFixed(3)}</span><span>Positive</span></div></section></div></div>
}

function FundamentalTrendPanel({ ticker }: { ticker: string }) {
  const query = useQuery({ queryKey: ['fundamental-trends', ticker], queryFn: () => api.getFinancials(ticker, 'annual'), staleTime: 60 * 60 * 1000 })
  const statement = query.data?.income_statement
  const row = (names: string[]) => names.map(name => statement?.rows[name]).find(Boolean) ?? {}
  const revenue = row(['Total Revenue', 'Operating Revenue'])
  const ebitda = row(['EBITDA', 'Normalized EBITDA'])
  const income = row(['Net Income', 'Net Income Common Stockholders'])
  const data = (statement?.periods ?? []).slice().reverse().map(period => ({ period: period.slice(0, 4), Revenue: revenue[period], EBITDA: ebitda[period], 'Net Income': income[period] }))
  return <div className="h-full min-h-0 p-4"><div className="h-full border border-border bg-s1/30 p-3">{data.length ? <ResponsiveContainer width="100%" height="100%"><LineChart data={data}><CartesianGrid stroke="#1F1F23" strokeDasharray="2 4" /><XAxis dataKey="period" stroke="#666" tick={{ fontSize: 9 }} /><YAxis stroke="#666" tick={{ fontSize: 9 }} tickFormatter={value => new Intl.NumberFormat('en', { notation: 'compact' }).format(Number(value))} /><Tooltip contentStyle={{ background: '#101014', border: '1px solid #2A2A30', fontSize: 11 }} formatter={value => money(value)} /><ChartLegend wrapperStyle={{ fontSize: 10 }} /><Line dataKey="Revenue" stroke="#3B82F6" strokeWidth={2} dot={false} connectNulls /><Line dataKey="EBITDA" stroke="#E89B2C" strokeWidth={2} dot={false} connectNulls /><Line dataKey="Net Income" stroke="#10B981" strokeWidth={2} dot={false} connectNulls /></LineChart></ResponsiveContainer> : <EmptyState message={query.isLoading ? 'Loading financial trends…' : 'Financial trend history is unavailable.'} />}</div></div>
}

function ReturnComparisonPanel({ ticker, peers }: { ticker: string; peers: string[] }) {
  const symbols = [ticker, ...peers]
  const query = useQuery({ queryKey: ['return-comparison', symbols.join('|')], queryFn: () => Promise.all(symbols.map(symbol => api.getPriceHistory(symbol, 365))), staleTime: 5 * 60 * 1000 })
  const chartData = useMemo(() => {
    const histories = query.data ?? []
    const base = histories[0]
    if (!base?.dates?.length) return []
    return base.dates.map((date, index) => {
      const point: Record<string, string | number> = { date: String(date).slice(0, 10) }
      histories.forEach((history, seriesIndex) => {
        const first = history.close.find(value => Number.isFinite(value) && value > 0)
        const value = history.close[index]
        if (first && Number.isFinite(value)) point[symbols[seriesIndex]] = (value / first - 1) * 100
      })
      return point
    })
  }, [query.data, symbols.join('|')])
  const colors = ['#E89B2C', '#3B82F6', '#10B981', '#A78BFA', '#F472B6', '#FACC15']
  return <div className="h-full min-h-0 p-4"><div className="h-full border border-border bg-s1/30 p-3">{chartData.length ? <ResponsiveContainer width="100%" height="100%"><LineChart data={chartData}><CartesianGrid stroke="#1F1F23" strokeDasharray="2 4" /><XAxis dataKey="date" stroke="#666" tick={{ fontSize: 9 }} minTickGap={35} /><YAxis stroke="#666" tick={{ fontSize: 9 }} tickFormatter={value => `${Number(value).toFixed(0)}%`} /><Tooltip contentStyle={{ background: '#101014', border: '1px solid #2A2A30', fontSize: 11 }} formatter={value => `${Number(value).toFixed(2)}%`} /><ChartLegend wrapperStyle={{ fontSize: 10 }} />{symbols.map((symbol, index) => <Line key={symbol} dataKey={symbol} type="monotone" dot={false} stroke={colors[index % colors.length]} strokeWidth={index === 0 ? 2 : 1} connectNulls />)}</LineChart></ResponsiveContainer> : <EmptyState message={query.isLoading ? 'Calculating normalized returns…' : 'Return history is unavailable.'} />}</div></div>
}

function HistoricalPriceTable({ ticker }: { ticker: string }) {
  const query = useQuery({ queryKey: ['historical-table', ticker], queryFn: () => api.getPriceHistory(ticker, 365), staleTime: 60_000 })
  const rows = (query.data?.dates ?? []).map((date, index) => ({ date, open: query.data?.open[index], high: query.data?.high[index], low: query.data?.low[index], close: query.data?.close[index], volume: (query.data?.volume ?? query.data?.volumes ?? [])[index] })).reverse()
  return <div className="h-full overflow-auto"><table className="w-full text-left text-xs"><thead className="sticky top-0 bg-s2 text-[9px] uppercase tracking-wider text-dim"><tr><th className="px-4 py-2">Date</th><th className="px-4 py-2 text-right">Open</th><th className="px-4 py-2 text-right">High</th><th className="px-4 py-2 text-right">Low</th><th className="px-4 py-2 text-right">Close</th><th className="px-4 py-2 text-right">Change</th><th className="px-4 py-2 text-right">Volume</th></tr></thead><tbody className="divide-y divide-border">{rows.map((row, index) => { const prior = rows[index + 1]?.close; const change = typeof prior === 'number' && prior ? ((row.close ?? 0) / prior - 1) * 100 : null; return <tr key={String(row.date)} className="bg-bg/30"><td className="px-4 py-2.5 tabnum text-dim">{shortDate(row.date)}</td><td className="px-4 py-2.5 text-right tabnum text-muted">{number(row.open)}</td><td className="px-4 py-2.5 text-right tabnum text-up">{number(row.high)}</td><td className="px-4 py-2.5 text-right tabnum text-down">{number(row.low)}</td><td className="px-4 py-2.5 text-right tabnum font-medium text-text">{number(row.close)}</td><td className={`px-4 py-2.5 text-right tabnum ${change == null ? 'text-dim' : change >= 0 ? 'text-up' : 'text-down'}`}>{change == null ? '—' : `${change.toFixed(2)}%`}</td><td className="px-4 py-2.5 text-right tabnum text-muted">{integer(row.volume)}</td></tr>})}</tbody></table></div>
}

function TechnicalPanel({ stock }: { stock: StockResult }) {
  const metrics = stock.metrics ?? {}
  const indicators = [['RSI 7', stock.rsi7, 100], ['RSI 14', metrics.technical?.rsi14 ?? 50, 100], ['ATR %', stock.atr_pct, Math.max(10, stock.atr_pct)], ['Volume / average', stock.volume_ratio, Math.max(3, stock.volume_ratio)]] as const
  return <div className="h-full overflow-auto p-4"><div className="grid gap-4 lg:grid-cols-[1.5fr_1fr]"><section className="space-y-3 border border-border bg-s1/40 p-5">{indicators.map(([label, value, max]) => <div key={label}><div className="flex justify-between text-xs"><span className="text-muted">{label}</span><span className="tabnum text-text">{Number(value).toFixed(2)}</span></div><div className="mt-2 h-2 bg-s2"><div className="h-full bg-accent" style={{ width: `${Math.min(100, Number(value) / Number(max) * 100)}%` }} /></div></div>)}</section><section className="border border-border bg-s1/50 p-5"><p className="text-2xs uppercase tracking-[0.18em] text-dim">Composite signal</p><p className="mt-4 text-2xl font-medium text-accent">{stock.day_trading_strategy}</p><p className="mt-2 text-4xl font-semibold tabnum text-text">{stock.day_trading_score.toFixed(1)}</p><div className="mt-6 space-y-2 text-xs"><Legend color={stock.above_ma5 ? '#10B981' : '#E5484D'} label="Price above MA5" value={stock.above_ma5 ? 'Yes' : 'No'} /><Legend color={stock.above_ma20 ? '#10B981' : '#E5484D'} label="Price above MA20" value={stock.above_ma20 ? 'Yes' : 'No'} /><Legend color={stock.macd_trend === 'bullish' ? '#10B981' : '#E5484D'} label="MACD trend" value={stock.macd_trend} /></div></section></div></div>
}

function BacktestLink({ code, ticker }: { code: string; ticker: string }) {
  const navigate = useNavigate()
  const descriptions: Record<string, string> = { FTST: 'Test factor and model-driven signals with time-aware training.', EQBT: 'Run the integrated equity strategy simulator with costs and benchmark diagnostics.', BT: 'Backtest technical threshold strategies and inspect trades, drawdowns and risk.' }
  return <ActionPanel title={FUNCTIONS.testing.find(item => item.code === code)?.label ?? 'Strategy testing'} message={`${descriptions[code]} The existing TradeSmart Quant workspace will open with ${ticker} available for configuration.`} action="Open Quant backtesting" onAction={() => navigate('/system')} />
}

function NavigationAction({ title, message, action, path }: { title: string; message: string; action: string; path: string }) {
  const navigate = useNavigate()
  return <ActionPanel title={title} message={message} action={action} onAction={() => navigate(path)} />
}

function Figure({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) {
  return <div><span className="block text-[9px] uppercase tracking-[0.12em] text-dim">{label}</span><span className={`mt-1 block text-sm font-medium tabnum ${accent ? 'text-accent' : 'text-text'}`}>{value || '—'}</span></div>
}

function Definition({ label, value }: { label: string; value: string }) {
  return <div className="flex items-start justify-between gap-4 border-b border-border py-3 text-xs"><span className="text-dim">{label}</span><span className="text-right tabnum text-text">{value || '—'}</span></div>
}

function Legend({ color, label, value }: { color: string; label: string; value: string }) {
  return <div className="flex items-center gap-2"><span className="h-2 w-2" style={{ backgroundColor: color }} /><span className="text-muted">{label}</span><span className="ml-auto tabnum text-text">{value}</span></div>
}

function RangeBar({ low, value, high }: { low: number | null; value: number | null; high: number | null }) {
  if (low == null || value == null || high == null || high <= low) return <span className="text-dim">—</span>
  return <div className="relative h-2 bg-gradient-to-r from-down via-accent to-up"><span className="absolute -top-1 h-4 w-0.5 bg-white" style={{ left: `${Math.max(0, Math.min(100, (value - low) / (high - low) * 100))}%` }} /></div>
}

function RangeCard({ label, low, value, high }: { label: string; low: number | null; value: number; high: number | null }) {
  return <section className="border border-border bg-s1/40 p-5"><div className="flex justify-between"><p className="text-2xs uppercase tracking-[0.16em] text-dim">{label}</p><p className="text-xs tabnum text-text">{money(low, false)} – {money(high, false)}</p></div><div className="mt-5"><RangeBar low={low} value={value} high={high} /></div></section>
}

function Calculator({ title, result, children }: { title: string; result: string; children: React.ReactNode }) {
  return <div className="h-full overflow-auto p-4"><div className="mx-auto max-w-3xl border border-border bg-s1/50"><div className="flex items-center justify-between border-b border-border px-4 py-3"><span className="text-xs uppercase tracking-[0.16em] text-muted">{title}</span><span className="text-xl font-semibold tabnum text-accent">{result}</span></div><div className="grid grid-cols-2 gap-px bg-border">{children}</div><p className="px-4 py-3 text-[10px] text-dim">Editable research model. Inputs are assumptions, not market recommendations.</p></div></div>
}

function Input({ label, value, onChange }: { label: string; value: number; onChange: (value: number) => void }) {
  return <label className="bg-bg p-3"><span className="block text-[9px] uppercase text-dim">{label}</span><input type="number" step="0.1" value={value} onChange={event => onChange(Number(event.target.value))} className="mt-2 w-full border border-border bg-s1 px-2 py-1 text-sm tabnum text-text outline-none focus:border-accent" /></label>
}

function ReadOnly({ label, value }: { label: string; value: string }) {
  return <div className="bg-bg p-3"><span className="block text-[9px] uppercase text-dim">{label}</span><span className="mt-2 block text-sm tabnum text-text">{value}</span></div>
}

function ActionPanel({ title, message, action, onAction }: { title: string; message: string; action: string; onAction: () => void }) {
  return <div className="flex h-full items-center justify-center p-6"><div className="max-w-xl border border-border bg-s1/60 p-5"><h3 className="text-sm font-medium text-text">{title}</h3><p className="mt-2 text-xs leading-relaxed text-muted">{message}</p><button onClick={onAction} className="mt-4 bg-accent px-3 py-2 text-2xs font-semibold text-bg">{action}</button></div></div>
}

function EmptyState({ message }: { message: string }) { return <div className="flex min-h-32 items-center justify-center p-6 text-xs text-dim">{message}</div> }
function objectRows(value: unknown): Array<Record<string, unknown>> { return value && typeof value === 'object' ? Object.entries(value as Record<string, unknown>).map(([field, item]) => ({ field, value: item })) : [] }
function asObject(value: unknown): Record<string, unknown> { return value && !Array.isArray(value) && typeof value === 'object' ? value as Record<string, unknown> : {} }
function getField(row: Record<string, unknown>, aliases: string[]): unknown {
  const keys = Object.keys(row)
  for (const alias of aliases) {
    const match = keys.find(key => key.toLowerCase() === alias.toLowerCase())
    if (match) return row[match]
  }
  return null
}
function display(value: unknown): string {
  if (value == null) return '—'
  if (typeof value === 'number') return Number.isFinite(value) ? compactNumber(value) : '—'
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}
function stringValue(value: unknown): string { return typeof value === 'string' ? value : value == null ? '' : String(value) }
function numeric(value: unknown): number | null { const parsed = Number(value); return Number.isFinite(parsed) ? parsed : null }
function number(value: unknown, digits = 2): string { const parsed = numeric(value); return parsed == null ? '—' : parsed.toFixed(digits) }
function integer(value: unknown): string { const parsed = numeric(value); return parsed == null ? '—' : Math.round(parsed).toLocaleString() }
function compactNumber(value: number): string { return Math.abs(value) >= 1_000_000 ? new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 2 }).format(value) : value.toLocaleString(undefined, { maximumFractionDigits: 3 }) }
function money(value: unknown, compact = true): string { const parsed = numeric(value); if (parsed == null) return '—'; return new Intl.NumberFormat('en', { style: 'currency', currency: 'USD', notation: compact ? 'compact' : 'standard', maximumFractionDigits: compact ? 2 : 2 }).format(parsed) }
function percent(value: unknown): string { const parsed = numeric(value); return parsed == null ? '—' : `${parsed.toFixed(2)}%` }
function normalizePct(value: unknown): number { const parsed = numeric(value) ?? 0; return Math.abs(parsed) <= 1 ? parsed * 100 : parsed }
function percentageRatio(value: unknown): string { const parsed = numeric(value); return parsed == null ? '—' : `${(Math.abs(parsed) <= 1 ? parsed * 100 : parsed).toFixed(2)}%` }
function compactMaybe(value: number | null): string { return value == null ? '—' : compactNumber(value) }
function shortDate(value: unknown): string { const raw = stringValue(value); const parsed = Date.parse(raw); return Number.isFinite(parsed) ? new Date(parsed).toLocaleDateString() : raw || '—' }
function returns(prices: number[]): number[] { return prices.slice(1).map((price, index) => prices[index] ? price / prices[index] - 1 : 0).filter(Number.isFinite) }
function correlation(a: number[], b: number[]): number { const length = Math.min(a.length, b.length); if (length < 3) return 0; const x = a.slice(-length), y = b.slice(-length); const mx = x.reduce((sum, value) => sum + value, 0) / length, my = y.reduce((sum, value) => sum + value, 0) / length; const numerator = x.reduce((sum, value, index) => sum + (value - mx) * (y[index] - my), 0); const dx = Math.sqrt(x.reduce((sum, value) => sum + (value - mx) ** 2, 0)); const dy = Math.sqrt(y.reduce((sum, value) => sum + (value - my) ** 2, 0)); return dx && dy ? numerator / (dx * dy) : 0 }
function flattenNews(row: Record<string, unknown>) { const content = asObject(row.content); const canonical = asObject(content.canonicalUrl); const provider = asObject(content.provider); return { title: stringValue(content.title || row.title), summary: stringValue(content.summary || content.description || row.summary), url: stringValue(canonical.url || row.link || row.url), publisher: stringValue(provider.displayName || row.publisher), date: stringValue(content.pubDate || row.providerPublishTime || row.published_at) } }
