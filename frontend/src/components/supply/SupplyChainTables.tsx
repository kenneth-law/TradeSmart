import { useMemo, useState } from 'react'
import type { SupplyChainDocument, SupplyChainNewsItem, SupplyChainRelatedEntity, SupplyChainResearch } from '../../types'

function textMatch(values: Array<string | null | undefined>, query: string) {
  const needle = query.trim().toLowerCase()
  return !needle || values.filter(Boolean).join(' ').toLowerCase().includes(needle)
}

function riskTone(value: string) {
  if (value === 'high') return 'border-down/40 bg-down/10 text-down'
  if (value === 'medium') return 'border-accent/40 bg-accent/10 text-accent'
  if (value === 'low') return 'border-up/40 bg-up/10 text-up'
  return 'border-border text-dim'
}

export function RelatedSecuritiesTable({ data, onOpenTicker }: { data: SupplyChainResearch; onOpenTicker: (ticker: string) => void }) {
  const [query, setQuery] = useState('')
  const [security, setSecurity] = useState('all')
  const rows = useMemo(() => (data.related_entities ?? []).filter(item =>
    (security === 'all' || item.security_type === security) &&
    textMatch([item.entity_name, item.ticker, item.relationship_type, item.country, item.state_owner_country, item.description], query)
  ), [data.related_entities, query, security])

  return (
    <TableShell
      title="Related securities and ownership"
      caption="Ultimate parents, subsidiaries, major investors and related instruments across the mapped supply chain."
      controls={<><SearchInput value={query} onChange={setQuery} placeholder="Find entity, ticker or country" /><select value={security} onChange={event => setSecurity(event.target.value)} className="h-7 border border-border bg-bg px-2 text-2xs text-muted outline-none"><option value="all">All instruments</option><option value="debt">Debt</option><option value="equity">Equity</option><option value="fund">Fund</option><option value="joint">Joint</option><option value="none">Entity only</option></select></>}
    >
      <div className="grid shrink-0 grid-cols-[minmax(0,1fr)_auto] gap-4 border-b border-border bg-s1/30 px-4 py-3">
        <div><p className="text-[9px] uppercase tracking-[0.18em] text-accent">Issuer overview</p><p className="mt-1 text-xs font-medium text-text">{data.company.name}{data.company.ticker ? ` · ${data.company.ticker}` : ''}</p><p className="mt-1 max-w-4xl text-2xs leading-relaxed text-muted">{data.company.description}</p></div>
        <div className="grid grid-cols-4 gap-px self-start bg-border text-center"><IssuerCount label="Debt" value={(data.related_entities ?? []).filter(item => item.security_type === 'debt').length} /><IssuerCount label="Equity" value={(data.related_entities ?? []).filter(item => item.security_type === 'equity').length} /><IssuerCount label="Funds" value={(data.related_entities ?? []).filter(item => item.security_type === 'fund').length} /><IssuerCount label="Joint" value={(data.related_entities ?? []).filter(item => item.security_type === 'joint').length} /></div>
      </div>
      <table className="w-full min-w-[980px] border-collapse text-left text-2xs">
        <thead className="sticky top-0 z-10 bg-s2 text-[9px] uppercase tracking-[0.16em] text-dim"><tr><th className="px-3 py-2">Entity</th><th className="px-3 py-2">Ticker</th><th className="px-3 py-2">Instrument</th><th className="px-3 py-2">Relationship</th><th className="px-3 py-2 text-right">Ownership</th><th className="px-3 py-2">Country / state owner</th><th className="px-3 py-2">Risk</th><th className="px-3 py-2 text-right">Confidence</th></tr></thead>
        <tbody className="divide-y divide-border">{rows.map(item => <RelatedRow key={item.id} item={item} onOpenTicker={onOpenTicker} />)}</tbody>
      </table>
      {!rows.length && <Empty label="No related entities match these filters." />}
    </TableShell>
  )
}

function IssuerCount({ label, value }: { label: string; value: number }) {
  return <div className="min-w-16 bg-bg px-3 py-2"><span className="block text-[9px] uppercase text-dim">{label}</span><span className="mt-0.5 block text-sm tabnum text-text">{value}</span></div>
}

function RelatedRow({ item, onOpenTicker }: { item: SupplyChainRelatedEntity; onOpenTicker: (ticker: string) => void }) {
  return <tr className="bg-bg/35 hover:bg-s1/80"><td className="max-w-[360px] px-3 py-2.5"><div className="font-medium text-text" style={{ paddingLeft: Math.min(item.hierarchy_level, 6) * 16 }}>{item.hierarchy_level > 0 ? '└ ' : ''}{item.entity_name}</div><div className="mt-0.5 truncate text-[10px] text-dim" style={{ paddingLeft: Math.min(item.hierarchy_level, 6) * 16 }}>{item.description}</div></td><td className="px-3 py-2.5">{item.ticker ? <button onClick={() => onOpenTicker(item.ticker!)} className="tabnum text-accent hover:underline">{item.ticker}</button> : <span className="text-dim">—</span>}</td><td className="px-3 py-2.5 uppercase text-muted">{item.security_type}</td><td className="px-3 py-2.5 capitalize text-muted">{item.relationship_type.replaceAll('_', ' ')}</td><td className="px-3 py-2.5 text-right tabnum text-text">{item.ownership_pct == null ? '—' : `${item.ownership_pct.toFixed(1)}%`}</td><td className="px-3 py-2.5 text-muted">{item.country ?? '—'}{item.government_owned ? <span className="mt-0.5 block text-[9px] text-accent">State interest: {item.state_owner_country ?? 'disclosed'}</span> : null}</td><td className="px-3 py-2.5"><span className={`border px-1.5 py-0.5 text-[9px] uppercase ${riskTone(item.risk_level)}`}>{item.risk_level}</span></td><td className="px-3 py-2.5 text-right tabnum text-muted">{Math.round(item.confidence)}%</td></tr>
}

export function DocumentSearchTable({ data, onOpenTicker }: { data: SupplyChainResearch; onOpenTicker: (ticker: string) => void }) {
  const [query, setQuery] = useState('')
  const [type, setType] = useState('all')
  const rows = useMemo(() => (data.documents ?? []).filter(item =>
    (type === 'all' || item.document_type === type) && textMatch([item.title, item.company_name, item.company_ticker, item.summary, item.disruption_relevance], query)
  ), [data.documents, query, type])
  return <TableShell title="Company document search" caption="Filings and primary documents that help explain dependencies, capacity, contracts and disruption exposure." controls={<><SearchInput value={query} onChange={setQuery} placeholder="Search documents and disclosures" /><select value={type} onChange={event => setType(event.target.value)} className="h-7 border border-border bg-bg px-2 text-2xs text-muted outline-none"><option value="all">All document types</option>{['annual_report', 'regulatory_filing', 'earnings', 'procurement', 'contract', 'presentation', 'policy', 'other'].map(value => <option key={value} value={value}>{value.replaceAll('_', ' ')}</option>)}</select></>}><table className="w-full min-w-[920px] border-collapse text-left text-2xs"><thead className="sticky top-0 z-10 bg-s2 text-[9px] uppercase tracking-[0.16em] text-dim"><tr><th className="px-3 py-2">Date</th><th className="px-3 py-2">Company</th><th className="px-3 py-2">Type</th><th className="px-3 py-2">Document</th><th className="px-3 py-2">Supply-chain relevance</th></tr></thead><tbody className="divide-y divide-border">{rows.map(item => <DocumentRow key={item.id} item={item} onOpenTicker={onOpenTicker} />)}</tbody></table>{!rows.length && <Empty label="No documents match this search." />}</TableShell>
}

function DocumentRow({ item, onOpenTicker }: { item: SupplyChainDocument; onOpenTicker: (ticker: string) => void }) {
  return <tr className="bg-bg/35 align-top hover:bg-s1/80"><td className="whitespace-nowrap px-3 py-2.5 tabnum text-dim">{item.published_at ?? '—'}</td><td className="px-3 py-2.5"><span className="text-text">{item.company_name}</span>{item.company_ticker && <button onClick={() => onOpenTicker(item.company_ticker!)} className="ml-2 tabnum text-accent hover:underline">{item.company_ticker}</button>}</td><td className="px-3 py-2.5 capitalize text-muted">{item.document_type.replaceAll('_', ' ')}</td><td className="max-w-[360px] px-3 py-2.5"><a href={item.url} target="_blank" rel="noreferrer" className="font-medium text-text hover:text-accent hover:underline">{item.title}</a><p className="mt-1 line-clamp-2 text-[10px] leading-relaxed text-dim">{item.summary}</p></td><td className="max-w-[360px] px-3 py-2.5 leading-relaxed text-muted">{item.disruption_relevance}</td></tr>
}

export function CompanyNewsTable({ data, onOpenTicker }: { data: SupplyChainResearch; onOpenTicker: (ticker: string) => void }) {
  const [query, setQuery] = useState('')
  const [severity, setSeverity] = useState('all')
  const rows = useMemo(() => (data.news ?? []).filter(item =>
    (severity === 'all' || item.severity === severity) && textMatch([item.headline, item.company_name, item.company_ticker, item.publisher, item.event_type, item.summary, item.location], query)
  ), [data.news, query, severity])
  return <TableShell title="Company news and disruptions" caption="Current events across the mapped network that may explain where and why supply is being disrupted." controls={<><SearchInput value={query} onChange={setQuery} placeholder="Search events, companies or locations" /><select value={severity} onChange={event => setSeverity(event.target.value)} className="h-7 border border-border bg-bg px-2 text-2xs text-muted outline-none"><option value="all">All severities</option><option value="high">High</option><option value="medium">Medium</option><option value="low">Low</option></select></>}><table className="w-full min-w-[940px] border-collapse text-left text-2xs"><thead className="sticky top-0 z-10 bg-s2 text-[9px] uppercase tracking-[0.16em] text-dim"><tr><th className="px-3 py-2">Date</th><th className="px-3 py-2">Severity</th><th className="px-3 py-2">Company</th><th className="px-3 py-2">Event</th><th className="px-3 py-2">Type / location</th><th className="px-3 py-2">Publisher</th></tr></thead><tbody className="divide-y divide-border">{rows.map(item => <NewsRow key={item.id} item={item} onOpenTicker={onOpenTicker} />)}</tbody></table>{!rows.length && <Empty label="No news items match these filters." />}</TableShell>
}

function NewsRow({ item, onOpenTicker }: { item: SupplyChainNewsItem; onOpenTicker: (ticker: string) => void }) {
  return <tr className="bg-bg/35 align-top hover:bg-s1/80"><td className="whitespace-nowrap px-3 py-2.5 tabnum text-dim">{item.published_at ?? '—'}</td><td className="px-3 py-2.5"><span className={`border px-1.5 py-0.5 text-[9px] uppercase ${riskTone(item.severity)}`}>{item.severity}</span></td><td className="px-3 py-2.5"><span className="text-text">{item.company_name}</span>{item.company_ticker && <button onClick={() => onOpenTicker(item.company_ticker!)} className="ml-2 tabnum text-accent hover:underline">{item.company_ticker}</button>}</td><td className="max-w-[440px] px-3 py-2.5"><a href={item.url} target="_blank" rel="noreferrer" className="font-medium text-text hover:text-accent hover:underline">{item.headline}</a><p className="mt-1 line-clamp-2 text-[10px] leading-relaxed text-dim">{item.summary}</p></td><td className="px-3 py-2.5 capitalize text-muted">{item.event_type.replaceAll('_', ' ')}{item.location && <span className="mt-0.5 block text-[9px] text-dim">{item.location}</span>}</td><td className="px-3 py-2.5 text-muted">{item.publisher}</td></tr>
}

function TableShell({ title, caption, controls, children }: { title: string; caption: string; controls: React.ReactNode; children: React.ReactNode }) {
  return <div className="flex h-full min-h-0 flex-col"><div className="flex shrink-0 flex-wrap items-center gap-3 border-b border-border bg-s1/50 px-4 py-2"><div className="mr-auto"><h3 className="text-xs font-semibold text-text">{title}</h3><p className="mt-0.5 text-[10px] text-dim">{caption}</p></div>{controls}</div><div className="min-h-0 flex-1 overflow-auto">{children}</div></div>
}

function SearchInput({ value, onChange, placeholder }: { value: string; onChange: (value: string) => void; placeholder: string }) {
  return <input value={value} onChange={event => onChange(event.target.value)} placeholder={placeholder} className="h-7 w-60 border border-border bg-bg px-2 text-2xs text-text outline-none placeholder:text-dim focus:border-accent/60" />
}

function Empty({ label }: { label: string }) {
  return <div className="p-8 text-center text-xs text-dim">{label}</div>
}
