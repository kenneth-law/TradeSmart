import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { AlertTriangle, ArrowLeft, ExternalLink, Network, RefreshCw, Search, ShieldCheck, X } from 'lucide-react'
import { api } from '../lib/api'
import { normalizeSupplyChainShares, OpenAIError, researchSupplyChain } from '../lib/openai'
import { useAppStore } from '../store/useAppStore'
import FacilityMap, { FACILITY_STYLE } from '../components/supply/FacilityMap'
import { CompanyNewsTable, DocumentSearchTable, RelatedSecuritiesTable } from '../components/supply/SupplyChainTables'
import type { SupplyChainFacility, SupplyChainFacilityType, SupplyChainRelationship, SupplyChainRelationshipKind, SupplyChainResearch } from '../types'

const CACHE_PREFIX = 'tradesmart_supply_chain_v4:'
const CACHE_TTL = 24 * 60 * 60 * 1000

type KindFilter = 'all' | SupplyChainRelationshipKind

const KIND_STYLE: Record<SupplyChainRelationshipKind, { stroke: string; fill: string; label: string }> = {
  supplier: { stroke: '#3B82F6', fill: 'rgba(59,130,246,0.13)', label: 'SUPPLIER' },
  customer: { stroke: '#10B981', fill: 'rgba(16,185,129,0.13)', label: 'CUSTOMER' },
  competitor: { stroke: '#E89B2C', fill: 'rgba(232,155,44,0.13)', label: 'COMPETITOR' },
}

function safeUrl(value: string) {
  try {
    const url = new URL(value)
    return url.protocol === 'https:' || url.protocol === 'http:' ? url.toString() : null
  } catch {
    return null
  }
}

function cacheKey(query: string) {
  return `${CACHE_PREFIX}${query.trim().toUpperCase()}`
}

function loadCached(query: string): { data: SupplyChainResearch; savedAt: number } | null {
  if (!query || typeof window === 'undefined') return null
  try {
    const parsed = JSON.parse(window.localStorage.getItem(cacheKey(query)) ?? 'null')
    if (!parsed?.data || !Array.isArray(parsed.data.relationships)) return null
    return { ...parsed, data: normalizeSupplyChainShares(parsed.data) }
  } catch {
    return null
  }
}

function saveCached(query: string, data: SupplyChainResearch) {
  try {
    window.localStorage.setItem(cacheKey(query), JSON.stringify({ data, savedAt: Date.now() }))
  } catch {
    // Research remains usable when storage is unavailable.
  }
}

function compact(value: string, length = 22) {
  return value.length > length ? `${value.slice(0, length - 1)}…` : value
}

function confidenceTone(confidence: number) {
  if (confidence >= 85) return 'text-up'
  if (confidence >= 65) return 'text-accent'
  return 'text-down'
}

function exposureLabel(relationship: SupplyChainRelationship) {
  if (relationship.kind === 'customer' && relationship.revenue_pct != null) return `REV ${relationship.revenue_pct.toFixed(1)}%`
  if (relationship.kind === 'supplier' && relationship.cogs_pct != null && relationship.capex_pct != null) return `COGS ${relationship.cogs_pct.toFixed(1)}% · CAPEX ${relationship.capex_pct.toFixed(1)}%`
  if (relationship.kind === 'competitor') return 'Peer / substitute'
  if (relationship.exposure_pct == null) return 'Relationship only'
  return `${relationship.exposure_pct.toFixed(1)}% ${relationship.exposure_basis.toUpperCase()}`
}

const NODE_WIDTH = 168
const NODE_HEIGHT = 62

function GraphNode({ relationship, x, y, compactNode, selected, onSelect }: {
  relationship: SupplyChainRelationship
  x: number
  y: number
  compactNode: boolean
  selected: boolean
  onSelect: () => void
}) {
  const style = KIND_STYLE[relationship.kind]
  const left = x - NODE_WIDTH / 2
  const top = y - NODE_HEIGHT / 2
  return (
    <g onClick={onSelect} className="cursor-pointer" role="button" tabIndex={0}>
      <rect x={left} y={top} width={NODE_WIDTH} height={NODE_HEIGHT} rx={3} fill={selected ? style.stroke : style.fill} fillOpacity={selected ? 0.24 : 1} stroke={style.stroke} strokeWidth={selected ? 2 : 1} />
      {!compactNode && <text x={left + 12} y={top + 16} fill={style.stroke} fontSize="8" fontWeight="700" letterSpacing="1">{style.label}{relationship.ticker ? ` · ${compact(relationship.ticker, 10)}` : ''}</text>}
      {!compactNode && <text x={left + NODE_WIDTH - 10} y={top + 16} textAnchor="end" fill={relationship.confidence >= 85 ? '#10B981' : relationship.confidence >= 65 ? '#E89B2C' : '#E5484D'} fontSize="8">{Math.round(relationship.confidence)}%</text>}
      <text x={left + 12} y={top + (compactNode ? 25 : 35)} fill="var(--color-text)" fontSize="11" fontWeight="600">{compact(relationship.company_name, 22)}</text>
      <text x={left + 12} y={top + (compactNode ? 48 : 52)} fill="var(--color-muted)" fontSize="8.5">{compact(exposureLabel(relationship), 30)}</text>
    </g>
  )
}

function SupplyChainGraph({ data, relationships, selectedId, onSelect }: {
  data: SupplyChainResearch
  relationships: SupplyChainRelationship[]
  selectedId: string | null
  onSelect: (id: string | null) => void
}) {
  const suppliers = relationships.filter(item => item.kind === 'supplier')
  const customers = relationships.filter(item => item.kind === 'customer')
  const competitors = relationships.filter(item => item.kind === 'competitor')
  const mappedSummary = [
    customers.length ? 'REV 100%' : '',
    suppliers.length ? 'COGS 100% · CAPEX 100%' : '',
  ].filter(Boolean).join(' · ')
  const sideCount = Math.max(suppliers.length, customers.length)
  const sideLaneHeight = Math.max(364, Math.max(0, sideCount - 1) * 78)
  const focalY = 78 + sideLaneHeight / 2
  const competitorColumns = Math.min(5, Math.max(1, competitors.length))
  const competitorRows = Math.max(1, Math.ceil(competitors.length / competitorColumns))
  const competitorStartY = 78 + sideLaneHeight + 100
  const graphHeight = competitorStartY + 70 + Math.max(0, competitorRows - 1) * 88
  const verticalPosition = (index: number, count: number) => count <= 1 ? focalY : 78 + index * (sideLaneHeight / (count - 1))
  const competitorPosition = (index: number) => {
    const row = Math.floor(index / competitorColumns)
    const column = index % competitorColumns
    const countInRow = Math.min(competitorColumns, competitors.length - row * competitorColumns)
    const x = countInRow <= 1 ? 500 : 100 + column * (800 / (countInRow - 1))
    return { x, y: competitorStartY + row * 88 }
  }

  const points = [
    ...suppliers.map((relationship, index) => ({ relationship, x: 120, y: verticalPosition(index, suppliers.length), compactNode: index >= 6 })),
    ...customers.map((relationship, index) => ({ relationship, x: 880, y: verticalPosition(index, customers.length), compactNode: index >= 6 })),
    ...competitors.map((relationship, index) => ({ relationship, ...competitorPosition(index), compactNode: index >= 5 })),
  ]

  return (
    <svg viewBox={`0 0 1000 ${graphHeight}`} className="min-h-[520px] min-w-[900px] w-full" style={{ height: graphHeight }} aria-label={`TradeSmart Intelligence graph for ${data.company.name}`}>
      <defs>
        <pattern id="supply-grid" width="24" height="24" patternUnits="userSpaceOnUse">
          <path d="M 24 0 L 0 0 0 24" fill="none" stroke="var(--color-border)" strokeWidth="0.6" opacity="0.45" />
        </pattern>
      </defs>
      <rect width="1000" height={graphHeight} fill="url(#supply-grid)" onClick={() => onSelect(null)} />

      {points.map(({ relationship, x, y }) => {
        const endX = x < 500 ? x + NODE_WIDTH / 2 : x > 500 ? x - NODE_WIDTH / 2 : x
        const endY = relationship.kind === 'competitor' ? y - NODE_HEIGHT / 2 : y
        const startX = x < 500 ? 394 : x > 500 ? 606 : 500
        const startY = relationship.kind === 'competitor' ? focalY + 45 : focalY
        const style = KIND_STYLE[relationship.kind]
        return <path key={`edge-${relationship.id}`} d={`M ${startX} ${startY} C ${(startX + endX) / 2} ${startY}, ${(startX + endX) / 2} ${endY}, ${endX} ${endY}`} fill="none" stroke={style.stroke} strokeWidth={selectedId === relationship.id ? 2 : 1} opacity={selectedId && selectedId !== relationship.id ? 0.22 : 0.72} />
      })}

      <g onClick={() => onSelect(null)} className="cursor-pointer">
        <rect x="394" y={focalY - 45} width="212" height="90" rx="4" fill="rgba(16,185,129,0.13)" stroke="#10B981" strokeWidth="2" />
        <text x="410" y={focalY - 23} fill="#10B981" fontSize="9" fontWeight="700" letterSpacing="1.4">FOCAL COMPANY</text>
        <text x="410" y={focalY} fill="var(--color-text)" fontSize="15" fontWeight="700">{compact(data.company.name, 27)}</text>
        <text x="410" y={focalY + 20} fill="var(--color-muted)" fontSize="10">{data.company.ticker ?? 'PRIVATE'} · {compact(data.company.sector ?? 'Sector unknown', 27)}</text>
        <text x="410" y={focalY + 35} fill="var(--color-dim)" fontSize="9">{mappedSummary ? `${mappedSummary} mapped` : `${competitors.length} competitors mapped`}</text>
      </g>

      {points.map(point => <GraphNode key={point.relationship.id} {...point} selected={selectedId === point.relationship.id} onSelect={() => onSelect(point.relationship.id)} />)}
      {points.length === 0 && <text x="500" y="380" textAnchor="middle" fill="var(--color-dim)" fontSize="12">No relationships match the current filters.</text>}
    </svg>
  )
}

function EvidencePanel({ data, selected, onOpenTicker }: { data: SupplyChainResearch; selected: SupplyChainRelationship | null; onOpenTicker: (ticker: string) => void }) {
  if (!selected) {
    return (
      <div className="space-y-4 p-4">
        <div>
          <p className="text-2xs uppercase tracking-[0.2em] text-accent">Research summary</p>
          <p className="mt-2 text-xs leading-relaxed text-muted">{data.summary}</p>
        </div>
        <div className="border-y border-border py-3">
          <p className="mb-2 text-2xs uppercase tracking-[0.2em] text-dim">Risk signals</p>
          <div className="space-y-2">
            {data.risks.length === 0 && <p className="text-2xs text-dim">No specific risks were supported by the research.</p>}
            {data.risks.map((risk, index) => (
              <div key={`${risk.title}-${index}`} className="border border-border bg-bg/60 p-2">
                <div className="flex items-center justify-between gap-2"><span className="text-xs text-text">{risk.title}</span><span className={`text-[9px] uppercase ${risk.severity === 'high' ? 'text-down' : risk.severity === 'medium' ? 'text-accent' : 'text-up'}`}>{risk.severity}</span></div>
                <p className="mt-1 text-2xs leading-relaxed text-muted">{risk.description}</p>
              </div>
            ))}
          </div>
        </div>
        <div>
          <p className="mb-2 text-2xs uppercase tracking-[0.2em] text-dim">Sources · {data.sources.length}</p>
          <div className="space-y-1.5">
            {data.sources.map((source, index) => {
              const url = safeUrl(source.url)
              return url ? <a key={`${url}-${index}`} href={url} target="_blank" rel="noreferrer" className="block border-l border-border pl-2 text-2xs text-muted hover:border-accent hover:text-accent"><span className="block truncate text-text">{source.title}</span>{source.publisher}{source.published_at ? ` · ${source.published_at}` : ''}</a> : null
            })}
          </div>
        </div>
      </div>
    )
  }

  const relatedSources = data.sources.filter(source => selected.source_urls.includes(source.url))
  return (
    <div className="p-4">
      <div className="flex items-start justify-between gap-2">
        <div><p className="text-2xs uppercase tracking-[0.2em]" style={{ color: KIND_STYLE[selected.kind].stroke }}>{selected.kind}</p><h3 className="mt-1 text-base font-semibold text-text">{selected.company_name}</h3><p className="text-2xs text-dim">{selected.ticker ?? 'Private / unknown ticker'}{selected.country ? ` · ${selected.country}` : ''}</p></div>
        <span className={`text-xs font-semibold tabnum ${confidenceTone(selected.confidence)}`}>{Math.round(selected.confidence)}% confidence</span>
      </div>
      {selected.ticker && (
        <button onClick={() => onOpenTicker(selected.ticker!)} className="mt-3 flex w-full items-center justify-between border border-accent/40 bg-accent/5 px-3 py-2 text-2xs text-accent hover:bg-accent/10">
          <span>Open {selected.ticker} market details</span><ExternalLink size={11} />
        </button>
      )}
      <div className="mt-4 grid grid-cols-2 gap-px bg-border">
        <div className="bg-s1 p-2"><span className="block text-[9px] uppercase text-dim">Mapped share</span><span className="text-xs text-text tabnum">{exposureLabel(selected)}</span></div>
        <div className="bg-s1 p-2"><span className="block text-[9px] uppercase text-dim">Evidence</span><span className="text-xs capitalize text-text">{selected.evidence_class.replace('_', ' ')}</span></div>
      </div>
      <div className="mt-4 space-y-3 text-xs leading-relaxed">
        <div><p className="text-2xs uppercase text-dim">Relationship</p><p className="mt-1 text-muted">{selected.relationship}</p></div>
        <div><p className="text-2xs uppercase text-dim">Evidence basis</p><p className="mt-1 text-muted">{selected.evidence_summary}</p></div>
        <div><p className="text-2xs uppercase text-dim">Last verified</p><p className="mt-1 text-muted tabnum">{selected.last_verified}</p></div>
      </div>
      <div className="mt-5 border-t border-border pt-3">
        <p className="mb-2 text-2xs uppercase tracking-[0.2em] text-dim">Supporting evidence</p>
        <div className="space-y-2">
          {(relatedSources.length ? relatedSources : selected.source_urls.map(url => ({ url, title: url, publisher: '', published_at: null }))).map((source, index) => {
            const url = safeUrl(source.url)
            return url ? <a key={`${url}-${index}`} href={url} target="_blank" rel="noreferrer" className="flex items-start gap-2 border border-border bg-bg p-2 text-2xs text-muted hover:border-accent hover:text-accent"><ExternalLink size={11} className="mt-0.5 shrink-0" /><span><span className="block text-text">{source.title}</span>{source.publisher}</span></a> : null
          })}
        </div>
      </div>
    </div>
  )
}

function FacilityPanel({ data, selected, onOpenTicker }: { data: SupplyChainResearch; selected: SupplyChainFacility | null; onOpenTicker: (ticker: string) => void }) {
  if (!selected) {
    const counts = (data.facilities ?? []).reduce<Partial<Record<SupplyChainFacilityType, number>>>((result, facility) => {
      result[facility.facility_type] = (result[facility.facility_type] ?? 0) + 1
      return result
    }, {})
    return (
      <div className="space-y-4 p-4">
        <div><p className="text-2xs uppercase tracking-[0.2em] text-accent">Company map</p><p className="mt-2 text-xs leading-relaxed text-muted">Evidenced manufacturing, storage, distribution, research and administrative locations for {data.company.name} and identified suppliers. Select a marker for site-level detail.</p></div>
        <div className="space-y-1.5 border-t border-border pt-3">
          {(Object.entries(FACILITY_STYLE) as Array<[SupplyChainFacilityType, { color: string; label: string }]>).map(([type, style]) => (
            <div key={type} className="flex items-center justify-between text-2xs"><span className="flex items-center gap-2 text-muted"><span className="h-2 w-2 rounded-full" style={{ backgroundColor: style.color }} />{style.label}</span><span className="tabnum text-text">{counts[type] ?? 0}</span></div>
          ))}
        </div>
        <p className="border-t border-border pt-3 text-[10px] leading-relaxed text-dim">Marker positions reflect the disclosed site, city, or region. The precision shown in each facility record indicates which level was available.</p>
      </div>
    )
  }

  const style = FACILITY_STYLE[selected.facility_type]
  const relatedSources = data.sources.filter(source => selected.source_urls.includes(source.url))
  const location = [selected.city, selected.region, selected.country].filter(Boolean).join(', ')
  return (
    <div className="p-4">
      <div className="flex items-center gap-2 text-2xs uppercase tracking-[0.18em]" style={{ color: style.color }}><span className="h-2 w-2 rounded-full" style={{ backgroundColor: style.color }} />{style.label}</div>
      <h3 className="mt-2 text-base font-semibold text-text">{selected.facility_name}</h3>
      <p className="mt-1 text-xs text-muted">{selected.company_name}</p>
      <p className="mt-1 text-2xs text-dim">{location}</p>
      {selected.company_ticker && <button onClick={() => onOpenTicker(selected.company_ticker!)} className="mt-3 flex w-full items-center justify-between border border-accent/40 bg-accent/5 px-3 py-2 text-2xs text-accent hover:bg-accent/10"><span>Open {selected.company_ticker} market details</span><ExternalLink size={11} /></button>}
      <div className="mt-4 grid grid-cols-2 gap-px bg-border">
        <div className="bg-s1 p-2"><span className="block text-[9px] uppercase text-dim">Confidence</span><span className={`text-xs tabnum ${confidenceTone(selected.confidence)}`}>{Math.round(selected.confidence)}%</span></div>
        <div className="bg-s1 p-2"><span className="block text-[9px] uppercase text-dim">Map precision</span><span className="text-xs capitalize text-text">{selected.coordinate_precision}</span></div>
      </div>
      <div className="mt-4 space-y-3 text-xs leading-relaxed">
        <div><p className="text-2xs uppercase text-dim">Site activity</p><p className="mt-1 text-muted">{selected.description}</p></div>
        <div><p className="text-2xs uppercase text-dim">Products / activities</p>{selected.products.length ? <div className="mt-1 flex flex-wrap gap-1">{selected.products.map(product => <span key={product} className="border border-border bg-bg px-1.5 py-1 text-2xs text-muted">{product}</span>)}</div> : <p className="mt-1 text-muted">No site-specific products disclosed.</p>}</div>
        <div><p className="text-2xs uppercase text-dim">Last verified</p><p className="mt-1 text-muted tabnum">{selected.last_verified}</p></div>
      </div>
      <div className="mt-5 border-t border-border pt-3"><p className="mb-2 text-2xs uppercase tracking-[0.2em] text-dim">Facility sources</p><div className="space-y-2">{relatedSources.map((source, index) => { const url = safeUrl(source.url); return url ? <a key={`${url}-${index}`} href={url} target="_blank" rel="noreferrer" className="flex items-start gap-2 border border-border bg-bg p-2 text-2xs text-muted hover:border-accent hover:text-accent"><ExternalLink size={11} className="mt-0.5 shrink-0" /><span><span className="block text-text">{source.title}</span>{source.publisher}</span></a> : null })}</div></div>
    </div>
  )
}

export default function SupplyChain({ embedded = false, tickerOverride }: { embedded?: boolean; tickerOverride?: string } = {}) {
  const { ticker: routeTicker } = useParams<{ ticker: string }>()
  const ticker = tickerOverride ?? routeTicker
  const navigate = useNavigate()
  const [pageSearchParams] = useSearchParams()
  const apiKey = useAppStore(state => state.openaiKey)
  const settings = useAppStore(state => state.settings)
  const initialQuery = decodeURIComponent(ticker ?? '').toUpperCase()
  const [input, setInput] = useState(initialQuery)
  const [data, setData] = useState<SupplyChainResearch | null>(null)
  const [savedAt, setSavedAt] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const requestedView = pageSearchParams.get('intel')
  const [view, setView] = useState<'network' | 'facilities' | 'related' | 'documents' | 'news'>(
    requestedView === 'facilities' || requestedView === 'related' || requestedView === 'documents' || requestedView === 'news' ? requestedView : 'network'
  )
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [selectedFacilityId, setSelectedFacilityId] = useState<string | null>(null)
  const [kindFilter, setKindFilter] = useState<KindFilter>('all')
  const [facilityFilter, setFacilityFilter] = useState<'all' | SupplyChainFacilityType>('all')
  const [confidenceFloor, setConfidenceFloor] = useState(35)
  const [relationshipSearch, setRelationshipSearch] = useState('')
  const [facilitySearch, setFacilitySearch] = useState('')
  const controller = useRef<AbortController | null>(null)

  useEffect(() => {
    setInput(initialQuery)
    const cached = loadCached(initialQuery)
    setData(cached?.data ?? null)
    setSavedAt(cached?.savedAt ?? null)
    setSelectedId(null)
    setSelectedFacilityId(null)
    setError(null)
  }, [initialQuery])

  useEffect(() => () => controller.current?.abort(), [])

  const filteredRelationships = useMemo(() => {
    const needle = relationshipSearch.trim().toLowerCase()
    const matches = (data?.relationships ?? []).filter(relationship =>
      (kindFilter === 'all' || relationship.kind === kindFilter) &&
      relationship.confidence >= confidenceFloor &&
      (!needle || `${relationship.company_name} ${relationship.ticker ?? ''} ${relationship.sector ?? ''}`.toLowerCase().includes(needle))
    )
    if (!data) return matches
    return normalizeSupplyChainShares({
      ...data,
      relationships: matches.map(relationship => ({ ...relationship })),
    }).relationships
  }, [confidenceFloor, data, kindFilter, relationshipSearch])
  const selected = filteredRelationships.find(relationship => relationship.id === selectedId) ?? null
  const filteredFacilities = useMemo(() => {
    const needle = facilitySearch.trim().toLowerCase()
    return (data?.facilities ?? []).filter(facility =>
      (facilityFilter === 'all' || facility.facility_type === facilityFilter) &&
      facility.confidence >= confidenceFloor &&
      (!needle || `${facility.facility_name} ${facility.company_name} ${facility.city ?? ''} ${facility.country} ${facility.products.join(' ')}`.toLowerCase().includes(needle))
    )
  }, [confidenceFloor, data, facilityFilter, facilitySearch])
  const selectedFacility = filteredFacilities.find(facility => facility.id === selectedFacilityId) ?? null

  function go() {
    const query = input.trim().toUpperCase()
    if (!query) return
    if (query !== initialQuery) navigate(embedded ? `/stock/${encodeURIComponent(query)}?tab=supply-chain` : `/supply-chain/${encodeURIComponent(query)}`)
    void runResearch(query)
  }

  async function runResearch(query = initialQuery) {
    if (!query || loading) return
    if (!apiKey) {
      setError('Add an OpenAI API key in Settings before running web research.')
      return
    }
    controller.current?.abort()
    controller.current = new AbortController()
    setLoading(true)
    setError(null)
    setSelectedId(null)
    setSelectedFacilityId(null)
    try {
      let companyHint: string | undefined
      let sectorHint: string | undefined
      try {
        const stock = await api.getStock(query)
        companyHint = stock.company_name
        sectorHint = stock.sector
      } catch {
        // The research model can still resolve the entity from the query.
      }
      const result = await researchSupplyChain({ apiKey, model: settings.openaiModel, query, companyHint, sectorHint, signal: controller.current.signal })
      setData(result)
      setSavedAt(Date.now())
      saveCached(query, result)
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === 'AbortError') return
      setError(cause instanceof OpenAIError || cause instanceof Error ? cause.message : 'Supply-chain research failed.')
    } finally {
      setLoading(false)
    }
  }

  const isStale = savedAt != null && Date.now() - savedAt > CACHE_TTL

  return (
    <div className="flex h-full min-h-0 flex-col bg-bg/80">
      <div className="flex shrink-0 flex-wrap items-center gap-3 border-b border-border bg-s1/80 px-4 py-2">
        {!embedded && <button onClick={() => navigate(-1)} className="text-dim hover:text-text" title="Back"><ArrowLeft size={14} /></button>}
        <div className="flex items-center gap-2"><Network size={15} className="text-accent" /><span className="text-sm font-medium text-text">TradeSmart Intelligence</span><span className="text-2xs text-dim">Company network research</span></div>
        <div className="ml-auto flex min-w-0 flex-1 items-center justify-end gap-2">
          {!embedded && <div className="flex h-8 w-full max-w-sm items-center border border-border-strong bg-bg px-2"><Search size={12} className="text-dim" /><input value={input} onChange={event => setInput(event.target.value.toUpperCase())} onKeyDown={event => event.key === 'Enter' && go()} placeholder="Ticker, e.g. LUV" className="min-w-0 flex-1 bg-transparent px-2 text-xs text-text outline-none tabnum" /></div>}
          {loading ? <button onClick={() => controller.current?.abort()} className="flex h-8 items-center gap-1 border border-down px-3 text-2xs text-down"><X size={12} />Cancel</button> : <button onClick={go} disabled={!input.trim()} className="flex h-8 items-center gap-1 bg-accent px-3 text-2xs font-semibold text-bg disabled:opacity-40"><RefreshCw size={12} />{data ? 'Refresh' : 'Research'}</button>}
        </div>
      </div>

      {error && <div className="flex shrink-0 items-center gap-2 border-b border-down/40 bg-down/10 px-4 py-2 text-2xs text-down"><AlertTriangle size={12} />{error}</div>}

      {!data ? (
        <div className="flex flex-1 items-center justify-center overflow-auto p-8">
          <div className="border border-border bg-s1/80 p-6 text-center">
            <p className="text-sm text-muted">{loading ? `Researching ${initialQuery || 'company'}…` : `No TradeSmart Intelligence research is cached for ${initialQuery || 'this company'}.`}</p>
            {!apiKey && <button onClick={() => navigate('/settings')} className="mt-5 border border-accent px-3 py-2 text-2xs text-accent hover:bg-accent/10">Configure OpenAI key</button>}
            {apiKey && !loading && <button onClick={go} className="mt-5 bg-accent px-4 py-2 text-2xs font-semibold text-bg hover:opacity-90">Research {initialQuery}</button>}
            {loading && <div className="mx-auto mt-4 h-1 w-48 overflow-hidden bg-border"><div className="h-full w-1/2 animate-pulse bg-accent" /></div>}
          </div>
        </div>
      ) : (
        <>
          <div className="flex shrink-0 flex-wrap items-center gap-3 border-b border-border px-4 py-2">
            <div><span className="text-sm font-semibold text-text">{data.company.name}</span><span className="ml-2 text-2xs text-accent tabnum">{data.company.ticker}</span></div>
            <span className={`border px-1.5 py-0.5 text-[9px] uppercase ${data.data_quality === 'strong' ? 'border-up/40 text-up' : data.data_quality === 'moderate' ? 'border-accent/40 text-accent' : 'border-down/40 text-down'}`}>{data.data_quality} evidence</span>
            <span className="text-2xs text-dim">{data.relationships.length} relationships · {(data.facilities ?? []).length} facilities · {data.sources.length} sources{isStale ? ' · cache stale' : ''}</span>
            <div className="ml-auto flex items-center gap-1">{([
              ['network', 'Flow'],
              ['facilities', 'Facilities'],
              ['related', 'Related'],
              ['documents', 'Documents'],
              ['news', 'News'],
            ] as const).map(([key, label]) => <button key={key} onClick={() => setView(key)} className={`border px-3 py-1 text-[10px] uppercase ${view === key ? 'border-accent bg-accent/10 text-accent' : 'border-border text-dim hover:text-text'}`}>{label}</button>)}</div>
          </div>
          {(view === 'network' || view === 'facilities') && <div className="flex shrink-0 flex-wrap items-center gap-4 border-b border-border bg-s1/30 px-4 py-1.5">
            <label className="flex items-center gap-2 text-2xs text-dim">Confidence ≥ <span className="w-7 text-right text-text tabnum">{confidenceFloor}</span><input type="range" min={0} max={95} step={5} value={confidenceFloor} onChange={event => setConfidenceFloor(Number(event.target.value))} className="w-28 accent-amber-500" /></label>
            {view === 'network' ? <>
              <div className="flex items-center border border-border bg-bg px-2"><Search size={10} className="text-dim" /><input value={relationshipSearch} onChange={event => setRelationshipSearch(event.target.value)} placeholder="Filter relationships" className="h-6 w-40 bg-transparent px-2 text-2xs text-text outline-none" /></div>
              <div className="flex items-center gap-1">{(['all', 'supplier', 'customer', 'competitor'] as const).map(kind => <button key={kind} onClick={() => setKindFilter(kind)} className={`border px-2 py-1 text-[10px] uppercase ${kindFilter === kind ? 'border-accent text-accent' : 'border-border text-dim hover:text-text'}`}>{kind}</button>)}</div>
              <div className="ml-auto flex items-center gap-1 text-[10px] text-dim"><ShieldCheck size={11} className="text-up" />Visible REV, COGS and CAPEX shares are normalized to 100%; modeled allocations are estimates.</div>
            </> : <>
              <div className="flex items-center border border-border bg-bg px-2"><Search size={10} className="text-dim" /><input value={facilitySearch} onChange={event => setFacilitySearch(event.target.value)} placeholder="Filter sites or products" className="h-6 w-44 bg-transparent px-2 text-2xs text-text outline-none" /></div>
              <select value={facilityFilter} onChange={event => setFacilityFilter(event.target.value as 'all' | SupplyChainFacilityType)} className="h-7 border border-border bg-bg px-2 text-2xs text-muted outline-none"><option value="all">All facility types</option>{(Object.entries(FACILITY_STYLE) as Array<[SupplyChainFacilityType, { color: string; label: string }]>).map(([type, style]) => <option key={type} value={type}>{style.label}</option>)}</select>
              <div className="ml-auto flex flex-wrap items-center justify-end gap-3">{(Object.entries(FACILITY_STYLE) as Array<[SupplyChainFacilityType, { color: string; label: string }]>).map(([type, style]) => <span key={type} className="flex items-center gap-1 text-[9px] text-dim"><span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: style.color }} />{style.label}</span>)}</div>
            </>}
          </div>}
          {view === 'network' ? (
            <div className="flex min-h-0 flex-1 overflow-hidden">
              <div className="min-w-0 flex-1 overflow-auto"><SupplyChainGraph data={data} relationships={filteredRelationships} selectedId={selectedId} onSelect={setSelectedId} /></div>
              <aside className="w-[340px] shrink-0 overflow-y-auto border-l border-border bg-s1/70"><EvidencePanel data={data} selected={selected} onOpenTicker={relatedTicker => navigate(`/stock/${encodeURIComponent(relatedTicker)}`)} /></aside>
            </div>
          ) : view === 'facilities' ? (
            <div className="flex min-h-0 flex-1 overflow-hidden">
              <div className="min-w-0 flex-1"><FacilityMap facilities={filteredFacilities} selectedId={selectedFacilityId} onSelect={setSelectedFacilityId} /></div>
              <aside className="w-[340px] shrink-0 overflow-y-auto border-l border-border bg-s1/80"><FacilityPanel data={data} selected={selectedFacility} onOpenTicker={relatedTicker => navigate(`/stock/${encodeURIComponent(relatedTicker)}`)} /></aside>
            </div>
          ) : view === 'related' ? (
            <div className="min-h-0 flex-1 overflow-hidden"><RelatedSecuritiesTable data={data} onOpenTicker={relatedTicker => navigate(`/stock/${encodeURIComponent(relatedTicker)}`)} /></div>
          ) : view === 'documents' ? (
            <div className="min-h-0 flex-1 overflow-hidden"><DocumentSearchTable data={data} onOpenTicker={relatedTicker => navigate(`/stock/${encodeURIComponent(relatedTicker)}`)} /></div>
          ) : (
            <div className="min-h-0 flex-1 overflow-hidden"><CompanyNewsTable data={data} onOpenTicker={relatedTicker => navigate(`/stock/${encodeURIComponent(relatedTicker)}`)} /></div>
          )}
        </>
      )}
    </div>
  )
}
