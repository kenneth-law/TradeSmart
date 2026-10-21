import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { api } from '../../lib/api'
import type { EquityResearchResponse, EquityResearchSection, StockResult } from '../../types'
import Financials from '../financials/Financials'
import PriceChart from '../charts/PriceChart'
import { Bar, CartesianGrid, Cell, ComposedChart, Legend as ChartLegend, Line, LineChart, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

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
  if (code === 'DVD') return <DividendsPanel ticker={ticker} dividends={data?.dividends ?? []} splits={data?.splits ?? []} profile={profile} />
  if (code === 'CAST') return <CapitalPanel ticker={ticker} profile={profile} />
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
  if (code === 'EVTS') return <EventsPanel ticker={ticker} calendar={data?.calendar} earningsDates={data?.earnings_dates ?? []} actions={data?.actions ?? []} />
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

function CapitalPanel({ ticker, profile }: { ticker: string; profile: Record<string, unknown> }) {
  const financials = useQuery({ queryKey: ['capital-financials', ticker], queryFn: () => api.getFinancials(ticker, 'annual'), staleTime: 30 * 60 * 1000, retry: 1 })
  const statements = financials.data
  const periods = statements?.balance_sheet.periods ?? []
  const currentPeriod = periods[0] ?? ''
  const field = (statement: 'balance_sheet' | 'income_statement' | 'cash_flow', aliases: string[], period = currentPeriod) => financialStatementValue(statements?.[statement].rows, aliases, period)
  const currency = stringValue(profile.currency) || 'USD'
  const marketEquity = Math.max(numeric(profile.marketCap) ?? 0, 0)
  const totalDebt = Math.max(numeric(profile.totalDebt) ?? field('balance_sheet', ['Total Debt']) ?? 0, 0)
  const preferredEquity = Math.max(field('balance_sheet', ['Preferred Stock Equity', 'Preferred Shares Number']) ?? 0, 0)
  const cash = Math.max(numeric(profile.totalCash) ?? field('balance_sheet', ['Cash Cash Equivalents And Short Term Investments', 'Cash And Cash Equivalents', 'Cash Financial']) ?? 0, 0)
  const capital = marketEquity + totalDebt + preferredEquity || 1
  const equityWeight = marketEquity / capital
  const debtWeight = totalDebt / capital
  const preferredWeight = preferredEquity / capital
  const beta = Math.max(numeric(profile.beta) ?? 1, 0)
  const riskFree = 4.25
  const equityRiskPremium = 5.0
  const costEquity = riskFree + beta * equityRiskPremium
  const interestExpense = Math.abs(field('income_statement', ['Interest Expense', 'Interest Expense Non Operating']) ?? 0)
  const reportedTaxRate = field('income_statement', ['Tax Rate For Calcs'])
  const taxProvision = field('income_statement', ['Tax Provision'])
  const pretaxIncome = field('income_statement', ['Pretax Income'])
  const taxRate = Math.max(0, Math.min(50, reportedTaxRate != null ? normalizePct(reportedTaxRate) : pretaxIncome && taxProvision != null ? Math.abs(taxProvision / pretaxIncome * 100) : 25))
  const costDebt = totalDebt && interestExpense ? Math.max(1, Math.min(20, interestExpense / totalDebt * 100)) : 6
  const costPreferred = numeric(profile.dividendYield) != null ? normalizePct(profile.dividendYield) : 7
  const weightedEquity = equityWeight * costEquity
  const weightedDebt = debtWeight * costDebt * (1 - taxRate / 100)
  const weightedPreferred = preferredWeight * costPreferred
  const wacc = weightedEquity + weightedDebt + weightedPreferred
  const ebit = field('income_statement', ['EBIT', 'Operating Income']) ?? ((pretaxIncome ?? 0) + interestExpense)
  const nopat = ebit * (1 - taxRate / 100)
  const investedCapital = Math.max(field('balance_sheet', ['Invested Capital']) ?? (totalDebt + (field('balance_sheet', ['Stockholders Equity', 'Common Stock Equity']) ?? marketEquity) - cash), 0)
  const capitalCharge = investedCapital * wacc / 100
  const eva = nopat - capitalCharge
  const roic = investedCapital ? nopat / investedCapital * 100 : null
  const spread = roic == null ? null : roic - wacc
  const history = [...periods].slice(0, 6).reverse().map(period => {
    const debt = Math.max(field('balance_sheet', ['Total Debt'], period) ?? 0, 0)
    const equity = Math.max(field('balance_sheet', ['Stockholders Equity', 'Common Stock Equity', 'Total Equity Gross Minority Interest'], period) ?? 0, 0)
    const periodCash = Math.max(field('balance_sheet', ['Cash Cash Equivalents And Short Term Investments', 'Cash And Cash Equivalents', 'Cash Financial'], period) ?? 0, 0)
    return { period: period.slice(0, 4), debt, equity, netDebt: debt - periodCash, leverage: equity ? debt / equity * 100 : 0 }
  })
  const capitalRows = [
    { label: 'Common equity', value: marketEquity, weight: equityWeight, cost: costEquity, weighted: weightedEquity, color: '#2557d6' },
    { label: 'Interest-bearing debt', value: totalDebt, weight: debtWeight, cost: costDebt * (1 - taxRate / 100), weighted: weightedDebt, color: '#e89b2c' },
    { label: 'Preferred equity', value: preferredEquity, weight: preferredWeight, cost: costPreferred, weighted: weightedPreferred, color: '#10b981' },
  ]
  return <div className="relative flex h-full min-h-0 flex-col overflow-auto bg-[#08090b]">
    <div className="flex h-9 shrink-0 items-center border-b border-[#5b1018] bg-[#31080d] px-3"><span className="text-xs font-semibold uppercase tracking-[0.16em] text-[#f0d7d9]">Capital Structure & Cost of Capital</span><span className="ml-3 text-[10px] tabnum text-[#c99ca1]">{ticker}</span><span className="ml-auto text-[9px] text-[#c99ca1]">MARKET VALUE BASIS · {currency}</span></div>
    <div className="flex h-7 shrink-0 items-center gap-5 border-b border-[#5b3b0a] bg-[#e89b2c] px-2 text-[10px] tabnum text-black"><strong>WACC {wacc.toFixed(2)}%</strong><span>Cost of equity {costEquity.toFixed(2)}%</span><span>After-tax debt {costDebt.toFixed(2)}% / {(costDebt * (1 - taxRate / 100)).toFixed(2)}%</span><span>Tax rate {taxRate.toFixed(1)}%</span><span className="ml-auto">Statement period {currentPeriod || 'Unavailable'}</span></div>
    <div className="grid min-h-[285px] shrink-0 border-b border-border xl:grid-cols-[1.35fr_1fr]">
      <section className="border-r border-border"><div className="border-b border-border bg-[#111318] px-3 py-1.5 text-[9px] uppercase tracking-wider text-accent">Weighted cost of capital</div><table className="w-full text-[11px]"><thead className="bg-[#17191d] text-[9px] uppercase text-dim"><tr><th className="px-3 py-1.5 text-left">Source</th><th className="px-3 py-1.5 text-right">Market value</th><th className="px-3 py-1.5 text-right">Weight</th><th className="px-3 py-1.5 text-right">Cost</th><th className="px-3 py-1.5 text-right">Contribution</th></tr></thead><tbody>{capitalRows.map((row, index) => <tr key={row.label} className={index % 2 ? 'bg-[#0d0f12]' : 'bg-[#090a0c]'}><td className="border-b border-border px-3 py-2 text-text"><span className="mr-2 inline-block h-2 w-2" style={{ background: row.color }} />{row.label}</td><td className="border-b border-border px-3 py-2 text-right tabnum text-muted">{capitalMoney(row.value, currency)}</td><td className="border-b border-border px-3 py-2 text-right tabnum text-accent">{(row.weight * 100).toFixed(2)}%</td><td className="border-b border-border px-3 py-2 text-right tabnum text-muted">{row.cost.toFixed(2)}%</td><td className="border-b border-border px-3 py-2 text-right tabnum text-text">{row.weighted.toFixed(2)}%</td></tr>)}</tbody><tfoot className="bg-[#15171b] font-semibold"><tr><td className="px-3 py-2 text-text">Total / WACC</td><td className="px-3 py-2 text-right tabnum text-text">{capitalMoney(capital, currency)}</td><td className="px-3 py-2 text-right tabnum text-text">100.00%</td><td /><td className="px-3 py-2 text-right tabnum text-accent">{wacc.toFixed(2)}%</td></tr></tfoot></table><div className="grid grid-cols-3 border-t border-border bg-[#0c0d10] px-3 py-2 text-[9px] text-dim"><span>Risk-free: <b className="text-muted">{riskFree.toFixed(2)}%</b></span><span>Equity premium: <b className="text-muted">{equityRiskPremium.toFixed(2)}%</b></span><span>Beta: <b className="text-muted">{beta.toFixed(2)}</b></span></div></section>
      <section><div className="border-b border-border bg-[#111318] px-3 py-1.5 text-[9px] uppercase tracking-wider text-accent">Capital structure</div><div className="relative h-[calc(100%-27px)] min-h-[245px] p-3"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={capitalRows.filter(row => row.value > 0)} dataKey="value" nameKey="label" cx="50%" cy="46%" innerRadius={48} outerRadius={86} paddingAngle={2} stroke="#08090b" strokeWidth={2}>{capitalRows.filter(row => row.value > 0).map(row => <Cell key={row.label} fill={row.color} />)}</Pie><Tooltip contentStyle={{ background: '#111318', border: '1px solid #34373d', borderRadius: 0, fontSize: 11 }} formatter={(value: number | string) => capitalMoney(value, currency)} /></PieChart></ResponsiveContainer><div className="pointer-events-none absolute inset-0 flex items-center justify-center pb-6"><div className="text-center"><span className="block text-[9px] uppercase text-dim">Total capital</span><span className="block text-sm font-semibold tabnum text-text">{capitalMoney(capital, currency)}</span></div></div><div className="absolute inset-x-3 bottom-2 flex justify-center gap-5">{capitalRows.filter(row => row.value > 0).map(row => <span key={row.label} className="text-[9px] text-muted"><i className="mr-1.5 inline-block h-2 w-2" style={{ background: row.color }} />{row.label} {(row.weight * 100).toFixed(1)}%</span>)}</div></div></section>
    </div>
    <div className="grid min-h-[310px] flex-1 xl:grid-cols-[1.45fr_1fr]">
      <section className="min-h-[300px] border-r border-border"><div className="border-b border-border bg-[#111318] px-3 py-1.5 text-[9px] uppercase tracking-wider text-accent">Reported capital history</div><div className="h-[calc(100%-27px)] min-h-[270px] p-3"><ResponsiveContainer width="100%" height="100%"><ComposedChart data={history} margin={{ top: 8, right: 16, bottom: 2, left: 8 }}><CartesianGrid stroke="#202329" vertical={false} /><XAxis dataKey="period" stroke="#676b75" tick={{ fill: '#777b85', fontSize: 9 }} /><YAxis yAxisId="capital" stroke="#676b75" tick={{ fill: '#777b85', fontSize: 9 }} tickFormatter={value => compactNumber(Number(value))} width={58} /><YAxis yAxisId="leverage" orientation="right" stroke="#676b75" tick={{ fill: '#777b85', fontSize: 9 }} tickFormatter={value => `${Number(value).toFixed(0)}%`} width={42} /><Tooltip contentStyle={{ background: '#111318', border: '1px solid #34373d', borderRadius: 0, fontSize: 11 }} formatter={(value: number | string, name: string) => [name === 'Debt / equity' ? `${Number(value).toFixed(2)}%` : capitalMoney(value, currency), name]} /><ChartLegend wrapperStyle={{ fontSize: 9 }} /><Line yAxisId="capital" type="monotone" dataKey="equity" stroke="#3b6eea" strokeWidth={2.5} dot={{ r: 3, fill: '#3b6eea' }} name="Book equity" /><Line yAxisId="capital" type="monotone" dataKey="debt" stroke="#e89b2c" strokeWidth={2.5} dot={{ r: 3, fill: '#e89b2c' }} name="Debt" /><Line yAxisId="leverage" type="monotone" dataKey="leverage" stroke="#10b981" strokeWidth={2} strokeDasharray="5 3" dot={{ r: 3, fill: '#10b981' }} name="Debt / equity" /></ComposedChart></ResponsiveContainer></div></section>
      <section><div className="border-b border-border bg-[#111318] px-3 py-1.5 text-[9px] uppercase tracking-wider text-accent">Economic value & returns</div><div className="grid grid-cols-2 gap-x-5 p-4"><Definition label="EBIT (derived where needed)" value={capitalMoney(ebit, currency)} /><Definition label="NOPAT" value={capitalMoney(nopat, currency)} /><Definition label="Invested capital" value={capitalMoney(investedCapital, currency)} /><Definition label="Capital charge" value={capitalMoney(capitalCharge, currency)} /><Definition label="Economic value added" value={capitalMoney(eva, currency)} /><Definition label="Free cash flow" value={capitalMoney(profile.freeCashflow, currency)} /></div><div className="mx-4 border-t border-border pt-4"><div className="grid grid-cols-3 text-center"><Figure label="ROIC" value={roic == null ? '—' : `${roic.toFixed(2)}%`} accent={roic != null && roic > wacc} /><Figure label="WACC" value={`${wacc.toFixed(2)}%`} /><Figure label="Value spread" value={spread == null ? '—' : `${spread >= 0 ? '+' : ''}${spread.toFixed(2)}%`} accent={spread != null && spread > 0} /></div>{spread != null && <div className="relative mt-5 h-3 bg-[#17191d]"><span className="absolute left-1/2 top-0 h-full w-px bg-white/40" /><span className={`absolute top-0 h-full ${spread >= 0 ? 'left-1/2 bg-up' : 'right-1/2 bg-down'}`} style={{ width: `${Math.min(50, Math.abs(spread) / 20 * 50)}%` }} /></div>}<p className="mt-3 text-[9px] leading-relaxed text-dim">WACC is an indicative calculation using a {riskFree.toFixed(2)}% risk-free rate and {equityRiskPremium.toFixed(2)}% equity risk premium. Reported statement values are used where available.</p></div></section>
    </div>
    {financials.isLoading && <div className="absolute inset-0 flex items-center justify-center bg-black/60 text-xs text-accent">Loading capital statements…</div>}
  </div>
}

function CreditPanel({ profile }: { profile: Record<string, unknown> }) {
  const debt = Math.max(numeric(profile.totalDebt) ?? 0, 0), cash = Math.max(numeric(profile.totalCash) ?? 0, 0), coverage = debt ? Math.min(100, cash / debt * 100) : 100
  return <div className="h-full overflow-auto p-4"><div className="grid gap-4 lg:grid-cols-2"><section className="border border-border bg-s1/50 p-5"><p className="text-2xs uppercase tracking-[0.18em] text-dim">Liquidity against debt</p><div className="mt-5 h-5 bg-s2"><div className="h-full bg-up" style={{ width: `${coverage}%` }} /></div><div className="mt-3 flex justify-between text-xs"><span className="text-muted">Cash {money(cash)}</span><span className="text-muted">Debt {money(debt)}</span></div><p className="mt-5 text-3xl tabnum text-text">{coverage.toFixed(1)}%</p><p className="text-2xs text-dim">cash coverage of reported debt</p></section><section className="border border-border bg-s1/30 p-5"><Definition label="Debt / equity" value={percent(profile.debtToEquity)} /><Definition label="Free cash flow" value={money(profile.freeCashflow)} /><Definition label="Operating cash flow" value={money(profile.operatingCashflow)} /><Definition label="Overall governance risk" value={number(profile.overallRisk)} /><Definition label="Audit / board risk" value={`${number(profile.auditRisk)} / ${number(profile.boardRisk)}`} /></section></div></div>
}

function DividendsPanel({ ticker, dividends, splits, profile }: { ticker: string; dividends: Array<Record<string, unknown>>; splits: Array<Record<string, unknown>>; profile: Record<string, unknown> }) {
  const [range, setRange] = useState<'5Y' | '10Y' | 'ALL'>('10Y')
  const [showSplits, setShowSplits] = useState(true)
  const history = useMemo(() => {
    const rows = dividends.map((row, index) => {
      const dt = eventDateTime(getField(row, ['Date', 'date', 'index']))
      return { id: `${dt.date}-${index}`, date: dt.date, dividend: numeric(getField(row, ['value', 'Dividends', 'dividend'])) ?? 0 }
    }).filter(row => row.date && row.dividend > 0).sort((a, b) => a.date.localeCompare(b.date))
    return rows.map((row, index) => {
      const timestamp = Date.parse(row.date)
      const trailing12m = rows.filter(item => { const itemTime = Date.parse(item.date); return itemTime <= timestamp && itemTime > timestamp - 366 * 86400000 }).reduce((sum, item) => sum + item.dividend, 0)
      const previous = rows[index - 1]?.dividend
      return { ...row, trailing12m, change: previous ? (row.dividend / previous - 1) * 100 : null }
    })
  }, [dividends])
  const splitHistory = useMemo(() => splits.map((row, index) => { const dt = eventDateTime(getField(row, ['Date', 'date', 'index'])); return { id: `${dt.date}-${index}`, date: dt.date, ratio: numeric(getField(row, ['value', 'Stock Splits', 'split'])) ?? 0 } }).filter(row => row.date && row.ratio > 0).sort((a, b) => b.date.localeCompare(a.date)), [splits])
  const latestDate = history.at(-1)?.date ?? new Date().toISOString().slice(0, 10)
  const cutoffYear = range === 'ALL' ? 0 : Number(latestDate.slice(0, 4)) - (range === '5Y' ? 5 : 10)
  const visible = history.filter(row => !cutoffYear || Number(row.date.slice(0, 4)) >= cutoffYear)
  const annual = useMemo(() => {
    const totals = new Map<number, number>()
    history.forEach(row => { const year = Number(row.date.slice(0, 4)); totals.set(year, (totals.get(year) ?? 0) + row.dividend) })
    return Array.from(totals, ([year, total]) => ({ year, total })).sort((a, b) => a.year - b.year)
  }, [history])
  const latest = history.at(-1)
  const latestTime = latest ? Date.parse(latest.date) : Date.now()
  const currentTtm = history.filter(row => { const time = Date.parse(row.date); return time <= latestTime && time > latestTime - 366 * 86400000 }).reduce((sum, row) => sum + row.dividend, 0)
  const priorTtm = history.filter(row => { const time = Date.parse(row.date); return time <= latestTime - 366 * 86400000 && time > latestTime - 732 * 86400000 }).reduce((sum, row) => sum + row.dividend, 0)
  const ttmGrowth = priorTtm ? (currentTtm / priorTtm - 1) * 100 : null
  const recentPayments = history.filter(row => { const time = Date.parse(row.date); return time <= latestTime && time > latestTime - 366 * 86400000 }).length
  const frequency = recentPayments >= 10 ? 'Monthly' : recentPayments >= 4 ? 'Quarterly' : recentPayments === 2 ? 'Semiannual' : recentPayments === 1 ? 'Annual' : `${recentPayments} / year`
  const cagr = (years: number) => { const end = annual.at(-1); const start = annual.filter(item => end && item.year <= end.year - years).at(-1); return start && end && start.total > 0 ? (Math.pow(end.total / start.total, 1 / (end.year - start.year)) - 1) * 100 : null }
  const annualCagr = cagr(5)
  const chartRows = visible.map(row => ({ ...row, label: row.date.slice(0, 7) }))
  const tableRows = [...visible].reverse()
  const profileAnnual = numeric(profile.dividendRate)
  return <div className="flex h-full min-h-0 flex-col bg-[#08090b]">
    <div className="flex h-9 shrink-0 items-center border-b border-[#5b1018] bg-[#31080d] px-3"><span className="text-xs font-semibold uppercase tracking-[0.16em] text-[#f0d7d9]">Dividend Analysis</span><span className="ml-3 text-[10px] tabnum text-[#c99ca1]">{ticker}</span><button onClick={() => exportDividendsExcel(tableRows, splitHistory, ticker)} disabled={!tableRows.length} className="ml-auto h-full border-x border-black/50 bg-[#8c111d] px-5 text-2xs font-medium text-white hover:bg-[#a91522] disabled:opacity-40">Export Excel</button></div>
    <div className="flex h-7 shrink-0 items-center gap-5 overflow-x-auto border-b border-[#5b3b0a] bg-[#e89b2c] px-2 text-[10px] tabnum text-black"><strong>{history.length} payments</strong><span>TTM {money(currentTtm, false)}</span><span>TTM growth {ttmGrowth == null ? '—' : `${ttmGrowth.toFixed(1)}%`}</span><span>Frequency {frequency}</span><span className="ml-auto">Latest ex-date {latest?.date ?? '—'}</span></div>
    <div className="flex min-h-0 flex-1">
      <aside className="w-60 shrink-0 overflow-y-auto border-r border-border bg-[#0c0d10] p-3">
        <p className="text-[9px] uppercase tracking-wider text-accent">History range</p><div className="mt-2 grid grid-cols-3 gap-px bg-border">{(['5Y', '10Y', 'ALL'] as const).map(item => <button key={item} onClick={() => setRange(item)} className={`h-7 text-[10px] ${range === item ? 'bg-accent text-black' : 'bg-[#111318] text-muted hover:text-text'}`}>{item}</button>)}</div>
        <label className="mt-3 flex items-center gap-2 text-2xs text-muted"><input type="checkbox" checked={showSplits} onChange={event => setShowSplits(event.target.checked)} className="accent-amber-500" />Show split history</label>
        <div className="mt-4 border-t border-border pt-3"><p className="mb-2 text-[9px] uppercase tracking-wider text-accent">Distribution</p><Definition label="Indicated annual" value={money(profileAnnual ?? currentTtm, false)} /><Definition label="Trailing 12 months" value={money(currentTtm, false)} /><Definition label="Latest payment" value={money(latest?.dividend, false)} /><Definition label="Dividend yield" value={percentageRatio(profile.dividendYield)} /><Definition label="Payout ratio" value={percentageRatio(profile.payoutRatio)} /></div>
        <div className="mt-3 border-t border-border pt-3"><p className="mb-2 text-[9px] uppercase tracking-wider text-accent">Growth & dates</p><Definition label="TTM growth" value={ttmGrowth == null ? '—' : `${ttmGrowth.toFixed(2)}%`} /><Definition label="5-year CAGR" value={annualCagr == null ? '—' : `${annualCagr.toFixed(2)}%`} /><Definition label="5Y average yield" value={percentageRatio(profile.fiveYearAvgDividendYield)} /><Definition label="Next ex-dividend" value={profileDate(profile.exDividendDate)} /><Definition label="Payment frequency" value={frequency} /></div>
        {showSplits && <div className="mt-3 border-t border-border pt-3"><p className="mb-2 text-[9px] uppercase tracking-wider text-accent">Stock splits ({splitHistory.length})</p>{splitHistory.slice(0, 5).map(row => <div key={row.id} className="flex justify-between border-b border-border/60 py-1.5 text-[10px]"><span className="tabnum text-dim">{row.date}</span><span className="tabnum text-up">{row.ratio.toFixed(4)}x</span></div>)}{!splitHistory.length && <p className="text-[10px] text-dim">No split records.</p>}</div>}
      </aside>
      <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <section className="h-[46%] min-h-[245px] shrink-0 border-b border-border bg-[#090a0c] p-3"><div className="mb-1 flex items-center"><div><p className="text-[10px] font-medium uppercase tracking-wider text-text">Distribution timeline</p><p className="text-[9px] text-dim">Cash dividend per share and rolling 12-month distribution</p></div><div className="ml-auto flex gap-4 text-[9px]"><span className="text-accent">■ Payment</span><span className="text-up">━ Rolling 12M</span></div></div><div className="h-[calc(100%-34px)]"><ResponsiveContainer width="100%" height="100%"><ComposedChart data={chartRows} margin={{ top: 8, right: 18, bottom: 2, left: 0 }}><CartesianGrid stroke="#202329" vertical={false} /><XAxis dataKey="label" stroke="#676b75" tick={{ fill: '#777b85', fontSize: 9 }} minTickGap={35} /><YAxis yAxisId="payment" stroke="#676b75" tick={{ fill: '#777b85', fontSize: 9 }} tickFormatter={value => `$${Number(value).toFixed(2)}`} width={48} /><YAxis yAxisId="ttm" orientation="right" stroke="#676b75" tick={{ fill: '#777b85', fontSize: 9 }} tickFormatter={value => `$${Number(value).toFixed(2)}`} width={48} /><Tooltip contentStyle={{ background: '#111318', border: '1px solid #34373d', borderRadius: 0, fontSize: 11 }} labelStyle={{ color: '#d7dae0' }} formatter={(value: number | string, name: string) => [money(value, false), name === 'dividend' ? 'Payment' : 'Rolling 12M']} /><Bar yAxisId="payment" dataKey="dividend" fill="#e89b2c" maxBarSize={14} /><Line yAxisId="ttm" type="stepAfter" dataKey="trailing12m" stroke="#10b981" strokeWidth={2} dot={false} /></ComposedChart></ResponsiveContainer></div></section>
        <section className="min-h-0 flex-1 overflow-auto"><table className="w-full min-w-[780px] border-collapse text-left text-[11px]"><thead className="sticky top-0 z-10 bg-[#17191d] text-[9px] uppercase tracking-wider text-[#8c8f98]"><tr><th className="border-r border-border px-3 py-1.5">Ex-date</th><th className="border-r border-border px-3 py-1.5 text-right">Dividend / share</th><th className="border-r border-border px-3 py-1.5 text-right">Change</th><th className="border-r border-border px-3 py-1.5 text-right">Trailing 12M</th><th className="border-r border-border px-3 py-1.5 text-right">Indicated annual</th><th className="px-3 py-1.5">Cadence</th></tr></thead><tbody>{tableRows.map((row, index) => <tr key={row.id} className={index % 2 ? 'bg-[#0d0f12]' : 'bg-[#090a0c]'}><td className="border-b border-r border-border px-3 py-1.5 tabnum text-accent">{row.date}</td><td className="border-b border-r border-border px-3 py-1.5 text-right tabnum text-text">{money(row.dividend, false)}</td><td className={`border-b border-r border-border px-3 py-1.5 text-right tabnum ${row.change == null ? 'text-dim' : row.change >= 0 ? 'text-up' : 'text-down'}`}>{row.change == null ? '—' : `${row.change >= 0 ? '+' : ''}${row.change.toFixed(2)}%`}</td><td className="border-b border-r border-border px-3 py-1.5 text-right tabnum text-up">{money(row.trailing12m, false)}</td><td className="border-b border-r border-border px-3 py-1.5 text-right tabnum text-muted">{money(row.dividend * Math.max(recentPayments, 1), false)}</td><td className="border-b border-border px-3 py-1.5 text-muted">{frequency}</td></tr>)}</tbody></table>{!tableRows.length && <EmptyState message="No dividend history was returned." />}</section>
      </main>
    </div>
  </div>
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

type CalendarEventRow = { id: string; security: string; date: string; time: string; period: string; type: string; actual: number | null; estimate: number | null; surprise: number | null; guidance: string }

function EventsPanel({ ticker, calendar, earningsDates, actions }: { ticker: string; calendar: unknown; earningsDates: Array<Record<string, unknown>>; actions: Array<Record<string, unknown>> }) {
  const allRows = useMemo(() => buildCalendarEvents(ticker, calendar, earningsDates, actions), [ticker, calendar, earningsDates, actions])
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<'daily' | 'monthly'>('daily')
  const [hiddenTypes, setHiddenTypes] = useState<Set<string>>(new Set())
  const types = Array.from(new Set(allRows.map(row => row.type)))
  const filtered = allRows.filter(row => (!startDate || row.date >= startDate) && (!endDate || row.date <= endDate) && !hiddenTypes.has(row.type) && (!query || `${row.security} ${row.type} ${row.period}`.toLowerCase().includes(query.toLowerCase())))
  const displayRows = mode === 'daily' ? filtered : monthlyEvents(filtered)

  function toggleType(type: string) {
    setHiddenTypes(current => { const next = new Set(current); if (next.has(type)) next.delete(type); else next.add(type); return next })
  }

  return <div className="flex h-full min-h-0 flex-col bg-[#08090b]"><div className="flex h-9 shrink-0 items-center border-b border-[#5b1018] bg-[#31080d] px-3"><span className="text-xs font-semibold uppercase tracking-[0.16em] text-[#f0d7d9]">Events Calendar</span><div className="ml-auto flex h-full items-center"><button onClick={() => exportEventsExcel(displayRows, ticker)} disabled={!displayRows.length} className="h-full border-x border-black/50 bg-[#8c111d] px-5 text-2xs font-medium text-white hover:bg-[#a91522] disabled:opacity-40">Export Excel</button></div></div><div className="flex h-7 shrink-0 items-center border-b border-[#5b3b0a] bg-[#e89b2c] px-2 text-[10px] text-black"><strong>{displayRows.length} events</strong><span className="ml-4">Filter by security, date range and event type</span><span className="ml-auto">View: {mode}</span></div><div className="flex min-h-0 flex-1"><aside className="w-56 shrink-0 overflow-y-auto border-r border-border bg-[#0c0d10] p-3"><label className="block text-[9px] uppercase tracking-wider text-dim">Security</label><input value={query} onChange={event => setQuery(event.target.value)} placeholder={ticker} className="mt-1 h-7 w-full border border-border bg-bg px-2 text-2xs text-text outline-none focus:border-accent" /><div className="mt-5 border-t border-border pt-3"><p className="text-[9px] uppercase tracking-wider text-accent">Date range</p><label className="mt-2 block text-[9px] text-dim">Start<input type="date" value={startDate} onChange={event => setStartDate(event.target.value)} className="mt-1 h-7 w-full border border-border bg-bg px-1 text-2xs tabnum text-text" /></label><label className="mt-2 block text-[9px] text-dim">End<input type="date" value={endDate} onChange={event => setEndDate(event.target.value)} className="mt-1 h-7 w-full border border-border bg-bg px-1 text-2xs tabnum text-text" /></label><button onClick={() => { setStartDate(''); setEndDate('') }} className="mt-2 text-[9px] text-accent hover:underline">Clear date range</button></div><div className="mt-5 border-t border-border pt-3"><p className="mb-2 text-[9px] uppercase tracking-wider text-accent">Event types</p><div className="space-y-2">{types.map(type => <label key={type} className="flex cursor-pointer items-center gap-2 text-2xs text-muted"><input type="checkbox" checked={!hiddenTypes.has(type)} onChange={() => toggleType(type)} className="accent-amber-500" /><span>{type}</span></label>)}</div></div></aside><main className="flex min-w-0 flex-1 flex-col"><div className="flex h-8 shrink-0 items-center border-b border-border bg-s1 px-3"><div className="flex items-center gap-4">{(['daily', 'monthly'] as const).map(item => <label key={item} className="flex cursor-pointer items-center gap-1.5 text-2xs"><input type="radio" checked={mode === item} onChange={() => setMode(item)} className="accent-amber-500" /><span className={mode === item ? 'text-accent' : 'text-dim'}>{item[0].toUpperCase() + item.slice(1)}</span></label>)}</div><span className="ml-auto text-[9px] text-dim">Group by: {mode === 'daily' ? 'event' : 'month'}</span></div><div className="min-h-0 flex-1 overflow-auto"><table className="w-full min-w-[900px] border-collapse text-left text-[11px]"><thead className="sticky top-0 z-10 bg-[#17191d] text-[9px] uppercase tracking-wider text-[#8c8f98]"><tr><th className="border-r border-border px-2 py-1.5">Security</th><th className="border-r border-border px-2 py-1.5">Date</th><th className="border-r border-border px-2 py-1.5">Time</th><th className="border-r border-border px-2 py-1.5">Event</th><th className="border-r border-border px-2 py-1.5">Period</th><th className="border-r border-border px-2 py-1.5 text-right">Actual</th><th className="border-r border-border px-2 py-1.5 text-right">Estimate</th><th className="border-r border-border px-2 py-1.5 text-right">Surprise</th><th className="px-2 py-1.5">Guidance</th></tr></thead><tbody>{displayRows.map((row, index) => <tr key={row.id} className={index % 2 ? 'bg-[#0d0f12]' : 'bg-[#090a0c]'}><td className="border-b border-r border-border px-2 py-1 font-medium tabnum text-accent">{row.security}</td><td className="border-b border-r border-border px-2 py-1 tabnum text-accent">{row.date || '—'}</td><td className="border-b border-r border-border px-2 py-1 tabnum text-muted">{row.time || '—'}</td><td className="border-b border-r border-border px-2 py-1 text-text">{row.type}</td><td className="border-b border-r border-border px-2 py-1 tabnum text-accent">{row.period || '—'}</td><td className={`border-b border-r border-border px-2 py-1 text-right tabnum ${row.actual == null ? 'text-dim' : 'text-up'}`}>{compactMaybe(row.actual)}</td><td className="border-b border-r border-border px-2 py-1 text-right tabnum text-accent">{compactMaybe(row.estimate)}</td><td className={`border-b border-r border-border px-2 py-1 text-right tabnum ${row.surprise == null ? 'text-dim' : row.surprise >= 0 ? 'text-up' : 'text-down'}`}>{row.surprise == null ? '—' : `${row.surprise.toFixed(2)}%`}</td><td className="border-b border-border px-2 py-1 text-muted">{row.guidance || '—'}</td></tr>)}</tbody></table>{!displayRows.length && <EmptyState message="No events match the current filters." />}</div></main></div></div>
}

function CorporateActionsPanel({ rows }: { rows: Array<Record<string, unknown>> }) {
  const actions = useMemo(() => rows.map((row, index) => { const rawDate = getField(row, ['Date', 'date', 'index']); const dt = eventDateTime(rawDate); const dividend = numeric(getField(row, ['Dividends', 'dividends'])) ?? 0; const split = numeric(getField(row, ['Stock Splits', 'splits'])) ?? 0; return { id: `${dt.date}-${index}`, date: dt.date, dividend, split, type: dividend > 0 && split > 0 ? 'Dividend / Split' : dividend > 0 ? 'Dividend' : split > 0 ? 'Stock Split' : 'Other' } }).filter(row => row.dividend > 0 || row.split > 0).sort((a, b) => b.date.localeCompare(a.date)), [rows])
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [showDividends, setShowDividends] = useState(true)
  const [showSplits, setShowSplits] = useState(true)
  const filtered = actions.filter(row => (!startDate || row.date >= startDate) && (!endDate || row.date <= endDate) && ((showDividends && row.dividend > 0) || (showSplits && row.split > 0)))
  const totalDividends = filtered.reduce((sum, row) => sum + row.dividend, 0)
  return <div className="flex h-full min-h-0 flex-col bg-[#08090b]"><div className="flex h-9 shrink-0 items-center border-b border-[#5b1018] bg-[#31080d] px-3"><span className="text-xs font-semibold uppercase tracking-[0.16em] text-[#f0d7d9]">Corporate Actions</span><button onClick={() => exportCorporateActionsExcel(filtered)} disabled={!filtered.length} className="ml-auto h-full border-x border-black/50 bg-[#8c111d] px-5 text-2xs font-medium text-white hover:bg-[#a91522] disabled:opacity-40">Export Excel</button></div><div className="flex h-7 shrink-0 items-center border-b border-[#5b3b0a] bg-[#e89b2c] px-2 text-[10px] text-black"><strong>{filtered.length} actions</strong><span className="ml-4">Total dividends: {money(totalDividends, false)}</span><span className="ml-auto">Sorted by most recent</span></div><div className="flex min-h-0 flex-1"><aside className="w-56 shrink-0 overflow-y-auto border-r border-border bg-[#0c0d10] p-3"><p className="text-[9px] uppercase tracking-wider text-accent">Date range</p><label className="mt-2 block text-[9px] text-dim">Start<input type="date" value={startDate} onChange={event => setStartDate(event.target.value)} className="mt-1 h-7 w-full border border-border bg-bg px-1 text-2xs tabnum text-text" /></label><label className="mt-2 block text-[9px] text-dim">End<input type="date" value={endDate} onChange={event => setEndDate(event.target.value)} className="mt-1 h-7 w-full border border-border bg-bg px-1 text-2xs tabnum text-text" /></label><button onClick={() => { setStartDate(''); setEndDate('') }} className="mt-2 text-[9px] text-accent hover:underline">Clear date range</button><div className="mt-5 border-t border-border pt-3"><p className="mb-2 text-[9px] uppercase tracking-wider text-accent">Action types</p><label className="flex items-center gap-2 text-2xs text-muted"><input type="checkbox" checked={showDividends} onChange={event => setShowDividends(event.target.checked)} className="accent-amber-500" />Dividends</label><label className="mt-2 flex items-center gap-2 text-2xs text-muted"><input type="checkbox" checked={showSplits} onChange={event => setShowSplits(event.target.checked)} className="accent-amber-500" />Stock splits</label></div><div className="mt-5 border-t border-border pt-3"><Definition label="Visible actions" value={String(filtered.length)} /><Definition label="Dividend total" value={money(totalDividends, false)} /><Definition label="Split events" value={String(filtered.filter(row => row.split > 0).length)} /></div></aside><main className="min-w-0 flex-1 overflow-auto"><table className="w-full min-w-[760px] border-collapse text-left text-[11px]"><thead className="sticky top-0 z-10 bg-[#17191d] text-[9px] uppercase tracking-wider text-[#8c8f98]"><tr><th className="border-r border-border px-3 py-1.5">Date</th><th className="border-r border-border px-3 py-1.5">Action Type</th><th className="border-r border-border px-3 py-1.5 text-right">Dividend</th><th className="border-r border-border px-3 py-1.5 text-right">Split Ratio</th><th className="px-3 py-1.5">Description</th></tr></thead><tbody>{filtered.map((row, index) => <tr key={row.id} className={index % 2 ? 'bg-[#0d0f12]' : 'bg-[#090a0c]'}><td className="border-b border-r border-border px-3 py-1.5 tabnum text-accent">{row.date}</td><td className="border-b border-r border-border px-3 py-1.5 text-text">{row.type}</td><td className={`border-b border-r border-border px-3 py-1.5 text-right tabnum ${row.dividend > 0 ? 'text-accent' : 'text-dim'}`}>{row.dividend > 0 ? money(row.dividend, false) : '—'}</td><td className={`border-b border-r border-border px-3 py-1.5 text-right tabnum ${row.split > 0 ? 'text-up' : 'text-dim'}`}>{row.split > 0 ? `${row.split.toFixed(4)}x` : '—'}</td><td className="border-b border-border px-3 py-1.5 text-muted">{row.dividend > 0 ? `Cash distribution of ${money(row.dividend, false)} per share` : row.split > 0 ? `${row.split.toFixed(4)}-for-1 stock split` : '—'}</td></tr>)}</tbody></table>{!filtered.length && <EmptyState message="No corporate actions match the current filters." />}</main></div></div>
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
function buildCalendarEvents(ticker: string, calendar: unknown, earningsDates: Array<Record<string, unknown>>, actions: Array<Record<string, unknown>>): CalendarEventRow[] {
  const rows: CalendarEventRow[] = []
  earningsDates.forEach((item, index) => { const dt = eventDateTime(getField(item, ['Earnings Date', 'date', 'index'])); const rawSurprise = getField(item, ['Surprise(%)', 'surprisePercent', 'surprise']); rows.push({ id: `earnings-${index}-${dt.date}`, security: ticker, date: dt.date, time: dt.time, period: display(getField(item, ['quarter', 'period'])), type: 'Earnings Release', actual: numeric(getField(item, ['Reported EPS', 'epsActual', 'actual'])), estimate: numeric(getField(item, ['EPS Estimate', 'epsEstimate', 'estimate'])), surprise: rawSurprise == null ? null : normalizePct(rawSurprise), guidance: '' }) })
  actions.forEach((item, index) => { const dt = eventDateTime(getField(item, ['Date', 'date', 'index'])); const dividend = numeric(getField(item, ['Dividends', 'dividends'])) ?? 0; const split = numeric(getField(item, ['Stock Splits', 'splits'])) ?? 0; if (dividend > 0) rows.push({ id: `dividend-${index}-${dt.date}`, security: ticker, date: dt.date, time: dt.time, period: '', type: 'Dividend', actual: dividend, estimate: null, surprise: null, guidance: '' }); if (split > 0) rows.push({ id: `split-${index}-${dt.date}`, security: ticker, date: dt.date, time: dt.time, period: '', type: 'Stock Split', actual: split, estimate: null, surprise: null, guidance: `${split.toFixed(3)}x ratio` }) })
  objectRows(calendar).forEach((item, index) => { const field = display(item.field); if (!/date|event/i.test(field)) return; const values = Array.isArray(item.value) ? item.value : [item.value]; values.forEach((value, valueIndex) => { const dt = eventDateTime(value); if (dt.date) rows.push({ id: `calendar-${index}-${valueIndex}-${dt.date}`, security: ticker, date: dt.date, time: dt.time, period: '', type: /earn/i.test(field) ? 'Earnings Release' : field, actual: null, estimate: null, surprise: null, guidance: '' }) }) })
  const deduped = new Map<string, CalendarEventRow>(); rows.forEach(row => { const key = `${row.date}|${row.type}|${row.security}`; if (!deduped.has(key)) deduped.set(key, row) })
  return Array.from(deduped.values()).filter(row => row.date).sort((a, b) => a.date.localeCompare(b.date))
}
function monthlyEvents(rows: CalendarEventRow[]): CalendarEventRow[] { const groups = new Map<string, CalendarEventRow[]>(); rows.forEach(row => { const key = row.date.slice(0, 7); groups.set(key, [...(groups.get(key) ?? []), row]) }); return Array.from(groups.entries()).map(([month, items]) => ({ id: `month-${month}`, security: items[0]?.security ?? '', date: month, time: '', period: `${items.length} events`, type: Array.from(new Set(items.map(item => item.type))).join(', '), actual: null, estimate: null, surprise: average(items.map(item => item.surprise).filter((value): value is number => value != null)), guidance: '' })) }
function exportEventsExcel(rows: CalendarEventRow[], ticker: string) { const headers: Array<[keyof CalendarEventRow, string]> = [['security', 'Security'], ['date', 'Date'], ['time', 'Time'], ['type', 'Event Type'], ['period', 'Period'], ['actual', 'Actual'], ['estimate', 'Estimate'], ['surprise', 'Surprise %'], ['guidance', 'Guidance']]; const cells = (values: unknown[]) => values.map(value => `<Cell><Data ss:Type="${typeof value === 'number' ? 'Number' : 'String'}">${xmlEscape(value == null ? '' : String(value))}</Data></Cell>`).join(''); const xml = `<?xml version="1.0"?><Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet" xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet"><Worksheet ss:Name="Events"><Table><Row>${cells(headers.map(([, label]) => label))}</Row>${rows.map(row => `<Row>${cells(headers.map(([key]) => row[key]))}</Row>`).join('')}</Table></Worksheet></Workbook>`; const url = URL.createObjectURL(new Blob([xml], { type: 'application/vnd.ms-excel;charset=utf-8' })); const anchor = document.createElement('a'); anchor.href = url; anchor.download = `${ticker.replace(/[^A-Za-z0-9.-]/g, '_')}-events.xls`; anchor.click(); URL.revokeObjectURL(url) }
function exportCorporateActionsExcel(rows: Array<{ date: string; type: string; dividend: number; split: number }>) { const cells = (values: unknown[]) => values.map(value => `<Cell><Data ss:Type="${typeof value === 'number' ? 'Number' : 'String'}">${xmlEscape(String(value ?? ''))}</Data></Cell>`).join(''); const xml = `<?xml version="1.0"?><Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet" xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet"><Worksheet ss:Name="Corporate Actions"><Table><Row>${cells(['Date', 'Action Type', 'Dividend', 'Split Ratio'])}</Row>${rows.map(row => `<Row>${cells([row.date, row.type, row.dividend || '', row.split || ''])}</Row>`).join('')}</Table></Worksheet></Workbook>`; const url = URL.createObjectURL(new Blob([xml], { type: 'application/vnd.ms-excel;charset=utf-8' })); const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'corporate-actions.xls'; anchor.click(); URL.revokeObjectURL(url) }
function exportDividendsExcel(rows: Array<{ date: string; dividend: number; change: number | null; trailing12m: number }>, splits: Array<{ date: string; ratio: number }>, ticker: string) { const cells = (values: unknown[]) => values.map(value => `<Cell><Data ss:Type="${typeof value === 'number' ? 'Number' : 'String'}">${xmlEscape(value == null ? '' : String(value))}</Data></Cell>`).join(''); const xml = `<?xml version="1.0"?><Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet" xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet"><Worksheet ss:Name="Dividends"><Table><Row>${cells(['Ex-Date', 'Dividend / Share', 'Change %', 'Trailing 12M'])}</Row>${rows.map(row => `<Row>${cells([row.date, row.dividend, row.change, row.trailing12m])}</Row>`).join('')}</Table></Worksheet><Worksheet ss:Name="Stock Splits"><Table><Row>${cells(['Date', 'Split Ratio'])}</Row>${splits.map(row => `<Row>${cells([row.date, row.ratio])}</Row>`).join('')}</Table></Worksheet></Workbook>`; const url = URL.createObjectURL(new Blob([xml], { type: 'application/vnd.ms-excel;charset=utf-8' })); const anchor = document.createElement('a'); anchor.href = url; anchor.download = `${ticker.replace(/[^A-Za-z0-9.-]/g, '_')}-dividend-history.xls`; anchor.click(); URL.revokeObjectURL(url) }
function eventDateTime(value: unknown): { date: string; time: string } { const raw = stringValue(value); const parsed = Date.parse(raw); if (!Number.isFinite(parsed)) return { date: /^\d{4}-\d{2}/.test(raw) ? raw.slice(0, 10) : '', time: '' }; const date = new Date(parsed); return { date: date.toISOString().slice(0, 10), time: date.toISOString().slice(11, 16) === '00:00' ? '' : date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) } }
function average(values: number[]): number | null { return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null }
function xmlEscape(value: string): string { return value.replace(/[<>&'"]/g, character => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', "'": '&apos;', '"': '&quot;' })[character] ?? character) }
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
function capitalMoney(value: unknown, currency: string): string { const parsed = numeric(value); if (parsed == null) return '—'; try { return new Intl.NumberFormat('en', { style: 'currency', currency, notation: 'compact', maximumFractionDigits: 2 }).format(parsed) } catch { return `${currency} ${compactNumber(parsed)}` } }
function financialStatementValue(rows: Record<string, Record<string, number | null>> | undefined, aliases: string[], period: string): number | null { if (!rows || !period) return null; const normalize = (value: string) => value.toLowerCase().replace(/[^a-z0-9]/g, ''); const entries = Object.entries(rows); for (const alias of aliases) { const exact = entries.find(([label]) => normalize(label) === normalize(alias)); if (exact) return numeric(exact[1]?.[period]) } for (const alias of aliases) { const partial = entries.find(([label]) => normalize(label).includes(normalize(alias))); if (partial) return numeric(partial[1]?.[period]) } return null }
function percent(value: unknown): string { const parsed = numeric(value); return parsed == null ? '—' : `${parsed.toFixed(2)}%` }
function normalizePct(value: unknown): number { const parsed = numeric(value) ?? 0; return Math.abs(parsed) <= 1 ? parsed * 100 : parsed }
function percentageRatio(value: unknown): string { const parsed = numeric(value); return parsed == null ? '—' : `${(Math.abs(parsed) <= 1 ? parsed * 100 : parsed).toFixed(2)}%` }
function compactMaybe(value: number | null): string { return value == null ? '—' : compactNumber(value) }
function shortDate(value: unknown): string { const raw = stringValue(value); const parsed = Date.parse(raw); return Number.isFinite(parsed) ? new Date(parsed).toLocaleDateString() : raw || '—' }
function profileDate(value: unknown): string { const timestamp = numeric(value); if (timestamp != null && timestamp > 0) return new Date(timestamp < 1e12 ? timestamp * 1000 : timestamp).toLocaleDateString(); return shortDate(value) }
function returns(prices: number[]): number[] { return prices.slice(1).map((price, index) => prices[index] ? price / prices[index] - 1 : 0).filter(Number.isFinite) }
function correlation(a: number[], b: number[]): number { const length = Math.min(a.length, b.length); if (length < 3) return 0; const x = a.slice(-length), y = b.slice(-length); const mx = x.reduce((sum, value) => sum + value, 0) / length, my = y.reduce((sum, value) => sum + value, 0) / length; const numerator = x.reduce((sum, value, index) => sum + (value - mx) * (y[index] - my), 0); const dx = Math.sqrt(x.reduce((sum, value) => sum + (value - mx) ** 2, 0)); const dy = Math.sqrt(y.reduce((sum, value) => sum + (value - my) ** 2, 0)); return dx && dy ? numerator / (dx * dy) : 0 }
function flattenNews(row: Record<string, unknown>) { const content = asObject(row.content); const canonical = asObject(content.canonicalUrl); const provider = asObject(content.provider); return { title: stringValue(content.title || row.title), summary: stringValue(content.summary || content.description || row.summary), url: stringValue(canonical.url || row.link || row.url), publisher: stringValue(provider.displayName || row.publisher), date: stringValue(content.pubDate || row.providerPublishTime || row.published_at) } }
