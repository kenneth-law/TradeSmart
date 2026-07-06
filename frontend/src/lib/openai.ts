import type { DetailedMetrics, MarketOverview, StockResult } from '../types'

const CHAT_ENDPOINT = 'https://api.openai.com/v1/chat/completions'
const RESPONSES_ENDPOINT = 'https://api.openai.com/v1/responses'

export interface Citation {
  url: string
  title: string
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
  citations?: Citation[]
}

export interface StockContext {
  ticker: string
  company_name?: string
  sector?: string
  industry?: string
  current_price: number
  return_1d: number
  day_trading_score: number
  day_trading_strategy: string
  atr_pct: number
  volume_ratio: number
  rsi7?: number
  rsi14?: number
  macd_trend?: string
  bb_position?: number
  above_ma5?: boolean
  above_ma10?: boolean
  above_ma20?: boolean
  intraday_range_pct?: number
  return_3d?: number
  return_5d?: number
  news_sentiment_label?: string
  news_sentiment_score?: number
  gap_ups_5d?: number
  gap_downs_5d?: number
}

export function buildStockContext(s: StockResult): StockContext {
  const m: DetailedMetrics = s.metrics ?? {}
  return {
    ticker: s.ticker,
    company_name: s.company_name,
    sector: s.sector,
    industry: s.industry,
    current_price: s.current_price,
    return_1d: s.return_1d,
    day_trading_score: s.day_trading_score,
    day_trading_strategy: s.day_trading_strategy,
    atr_pct: s.atr_pct,
    volume_ratio: s.volume_ratio,
    rsi7: m.technical?.rsi7 ?? s.rsi7,
    rsi14: m.technical?.rsi14,
    macd_trend: m.technical?.macd_trend ?? s.macd_trend,
    bb_position: m.technical?.bb_position,
    above_ma5: m.technical?.above_ma5 ?? s.above_ma5,
    above_ma10: m.technical?.above_ma10,
    above_ma20: m.technical?.above_ma20 ?? s.above_ma20,
    intraday_range_pct: m.volatility?.avg_intraday_range,
    return_3d: m.momentum?.return_3d,
    return_5d: m.momentum?.return_5d,
    news_sentiment_label: m.sentiment?.news_sentiment_label ?? s.news_sentiment_label,
    news_sentiment_score: m.sentiment?.news_sentiment_score ?? s.news_sentiment_score,
    gap_ups_5d: m.volatility?.gap_ups_5d,
    gap_downs_5d: m.volatility?.gap_downs_5d,
  }
}

function contextBlock(ctx: StockContext): string {
  const lines: string[] = []
  lines.push(`Ticker: ${ctx.ticker}${ctx.company_name ? ` (${ctx.company_name})` : ''}`)
  if (ctx.sector) lines.push(`Sector: ${ctx.sector}${ctx.industry ? ` / ${ctx.industry}` : ''}`)
  lines.push(`Price: $${ctx.current_price.toFixed(2)} | 1D: ${ctx.return_1d.toFixed(2)}%`)
  if (ctx.return_3d != null) lines.push(`3D return: ${ctx.return_3d.toFixed(2)}% | 5D: ${ctx.return_5d?.toFixed(2) ?? '-'}%`)
  lines.push(`Day-trading score: ${ctx.day_trading_score.toFixed(1)} (${ctx.day_trading_strategy})`)
  lines.push(`ATR%: ${ctx.atr_pct.toFixed(2)} | Vol/Avg: ${ctx.volume_ratio.toFixed(2)}x | Intraday range: ${ctx.intraday_range_pct?.toFixed(2) ?? '-'}%`)
  lines.push(`RSI(7): ${ctx.rsi7?.toFixed(1) ?? '-'} | RSI(14): ${ctx.rsi14?.toFixed(1) ?? '-'} | MACD: ${ctx.macd_trend ?? '-'}`)
  lines.push(`BB position: ${ctx.bb_position?.toFixed(2) ?? '-'} | Above MA5/10/20: ${ctx.above_ma5 ? 'Y' : 'N'}/${ctx.above_ma10 ? 'Y' : 'N'}/${ctx.above_ma20 ? 'Y' : 'N'}`)
  if (ctx.gap_ups_5d != null || ctx.gap_downs_5d != null) {
    lines.push(`Gaps 5D: up ${ctx.gap_ups_5d ?? 0} / down ${ctx.gap_downs_5d ?? 0}`)
  }
  if (ctx.news_sentiment_label) {
    lines.push(`News sentiment: ${ctx.news_sentiment_label}${ctx.news_sentiment_score != null ? ` (${ctx.news_sentiment_score.toFixed(3)})` : ''}`)
  }
  return lines.join('\n')
}

const BRIEF_SYSTEM = `You are a concise day-trading strategist. Given a snapshot of one stock's technicals, sentiment, and price action, write a ~150-word actionable brief covering: (1) the dominant signal and what is driving it, (2) the specific deltas/levels that matter today (RSI, MACD, MA position, volume, ATR), (3) one realistic intraday playbook with an entry condition, a stop, and a target framing. Be direct, no disclaimers. You may use light markdown formatting (bold for key levels, short bullet lists where they aid scanning) but keep the total under ~150 words.`

const CHAT_SYSTEM = `You are a trading copilot embedded in a stock analysis tool. The user is looking at one specific stock and has its technicals, sentiment, and price action in front of them. Answer their questions about this stock concisely and concretely. Reference the actual numbers from the context block. If they ask about entries, stops, or targets, give specific levels grounded in the data shown. Use markdown formatting where it helps readability (bold for key levels, bullet lists for steps or comparisons, tables for side-by-side data). Keep responses tight - a few sentences unless they ask for depth. No disclaimers.`

export class OpenAIError extends Error {
  status?: number
  constructor(message: string, status?: number) {
    super(message)
    this.status = status
    this.name = 'OpenAIError'
  }
}

interface StreamOpts {
  apiKey: string
  model: string
  messages: ChatMessage[]
  signal?: AbortSignal
  onDelta: (textChunk: string) => void
  temperature?: number
}

async function streamChat(opts: StreamOpts): Promise<string> {
  if (!opts.apiKey) throw new OpenAIError('No OpenAI API key set. Add one in Settings.')

  const body: Record<string, unknown> = {
    model: opts.model,
    messages: opts.messages,
    stream: true,
  }
  if (typeof opts.temperature === 'number') body.temperature = opts.temperature

  const res = await fetch(CHAT_ENDPOINT, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${opts.apiKey}`,
    },
    body: JSON.stringify(body),
    signal: opts.signal,
  })

  if (!res.ok || !res.body) {
    let detail = ''
    try {
      const errJson = await res.json()
      detail = errJson?.error?.message ?? ''
    } catch {
      // ignore parse failure
    }
    throw new OpenAIError(detail || `OpenAI request failed (${res.status})`, res.status)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let full = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed.startsWith('data:')) continue
      const payload = trimmed.slice(5).trim()
      if (payload === '[DONE]') return full
      try {
        const json = JSON.parse(payload)
        const delta: string | undefined = json?.choices?.[0]?.delta?.content
        if (delta) {
          full += delta
          opts.onDelta(delta)
        }
      } catch {
        // skip non-JSON keepalive lines
      }
    }
  }
  return full
}

function withCustomPrompt(base: string, extra?: string): string {
  const trimmed = extra?.trim()
  if (!trimmed) return base
  return `${base}\n\nAdditional user instructions:\n${trimmed}`
}

// ---------------------------------------------------------------------------
// Financials context injection
// ---------------------------------------------------------------------------

export interface FinancialsSnapshot {
  annual?: import('../types').FinancialsResponse | null
  quarterly?: import('../types').FinancialsResponse | null
}

export type FinancialsDepthMode = 'concise' | 'full'

// Key line items to include in the concise view (lowercased substring match).
const CONCISE_KEYS = [
  'total revenue',
  'gross profit',
  'operating income',
  'net income',
  'diluted eps',
  'basic eps',
  'total assets',
  'total liabilities',
  'total equity',
  'total debt',
  'cash and cash equivalents',
  'operating cash flow',
  'free cash flow',
  'capital expenditure',
]

function abbreviateNumber(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '-'
  const abs = Math.abs(v)
  if (abs >= 1e12) return `${(v / 1e12).toFixed(2)}T`
  if (abs >= 1e9) return `${(v / 1e9).toFixed(2)}B`
  if (abs >= 1e6) return `${(v / 1e6).toFixed(2)}M`
  if (abs >= 1e3) return `${(v / 1e3).toFixed(2)}K`
  return v.toFixed(2)
}

function pickRows(statement: import('../types').FinancialsStatement, depth: FinancialsDepthMode): string[] {
  const allRows = Object.keys(statement.rows)
  if (depth === 'full') return allRows
  // concise: keep only rows whose name matches a key substring, preserving statement order
  const lowered = allRows.map(r => r.toLowerCase())
  const selected: string[] = []
  for (let i = 0; i < allRows.length; i++) {
    if (CONCISE_KEYS.some(k => lowered[i].includes(k))) selected.push(allRows[i])
  }
  return selected
}

function renderStatement(
  title: string,
  statement: import('../types').FinancialsStatement | undefined,
  depth: FinancialsDepthMode,
): string {
  if (!statement || statement.periods.length === 0) return `${title}: no data.`
  const rows = pickRows(statement, depth)
  if (rows.length === 0) return `${title}: no relevant rows.`
  const header = `${title} (${statement.periods.join(' | ')})`
  const lines = rows.map(name => {
    const r = statement.rows[name]
    const vals = statement.periods.map(p => abbreviateNumber(r[p]))
    return `  ${name}: ${vals.join(' | ')}`
  })
  return `${header}\n${lines.join('\n')}`
}

function renderPeriodSet(
  label: string,
  resp: import('../types').FinancialsResponse | null | undefined,
  depth: FinancialsDepthMode,
): string {
  if (!resp) return ''
  const parts = [
    renderStatement('Income Statement', resp.income_statement, depth),
    renderStatement('Balance Sheet', resp.balance_sheet, depth),
    renderStatement('Cash Flow', resp.cash_flow, depth),
  ]
  return `--- ${label.toUpperCase()} ---\n${parts.join('\n\n')}`
}

export function financialsBlock(snap: FinancialsSnapshot | undefined, depth: FinancialsDepthMode): string {
  if (!snap) return ''
  const annual = renderPeriodSet('Annual', snap.annual, depth)
  const quarterly = renderPeriodSet('Quarterly', snap.quarterly, depth)
  const filled = [annual, quarterly].filter(s => s)
  if (filled.length === 0) return ''
  return `\n\nFinancial statements (${depth}):\n${filled.join('\n\n')}`
}

export function streamBrief(params: {
  apiKey: string
  model: string
  context: StockContext
  onDelta: (chunk: string) => void
  signal?: AbortSignal
  temperature?: number
  extraSystemPrompt?: string
  financials?: FinancialsSnapshot
  financialsDepth?: FinancialsDepthMode
}): Promise<string> {
  const briefSystemWithFinancials = BRIEF_SYSTEM + financialsBlock(params.financials, params.financialsDepth ?? 'concise')
  return streamChat({
    apiKey: params.apiKey,
    model: params.model,
    signal: params.signal,
    onDelta: params.onDelta,
    temperature: params.temperature,
    messages: [
      { role: 'system', content: withCustomPrompt(briefSystemWithFinancials, params.extraSystemPrompt) },
      {
        role: 'user',
        content: `Stock snapshot:\n${contextBlock(params.context)}\n\nWrite the ~150-word strategy brief now.`,
      },
    ],
  })
}

export function streamConversation(params: {
  apiKey: string
  model: string
  context: StockContext
  history: ChatMessage[]
  onDelta: (chunk: string) => void
  signal?: AbortSignal
  temperature?: number
  extraSystemPrompt?: string
  financials?: FinancialsSnapshot
  financialsDepth?: FinancialsDepthMode
}): Promise<string> {
  const baseSystem = `${CHAT_SYSTEM}\n\nCurrent stock context:\n${contextBlock(params.context)}` + financialsBlock(params.financials, params.financialsDepth ?? 'concise')
  const systemMsg: ChatMessage = {
    role: 'system',
    content: withCustomPrompt(baseSystem, params.extraSystemPrompt),
  }
  return streamChat({
    apiKey: params.apiKey,
    model: params.model,
    signal: params.signal,
    onDelta: params.onDelta,
    temperature: params.temperature,
    messages: [systemMsg, ...params.history],
  })
}

// ---------------------------------------------------------------------------
// Responses API path (with web_search tool)
// ---------------------------------------------------------------------------

export interface WebStreamResult {
  text: string
  citations: Citation[]
}

interface WebStreamOpts {
  apiKey: string
  model: string
  input: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>
  signal?: AbortSignal
  onDelta: (chunk: string) => void
  onCitation?: (c: Citation) => void
  onSearchStart?: () => void
  temperature?: number
}

async function streamResponses(opts: WebStreamOpts): Promise<WebStreamResult> {
  if (!opts.apiKey) throw new OpenAIError('No OpenAI API key set. Add one in Settings.')

  const body: Record<string, unknown> = {
    model: opts.model,
    input: opts.input,
    tools: [{ type: 'web_search' }],
    stream: true,
  }
  if (typeof opts.temperature === 'number') body.temperature = opts.temperature

  const res = await fetch(RESPONSES_ENDPOINT, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${opts.apiKey}`,
    },
    body: JSON.stringify(body),
    signal: opts.signal,
  })

  if (!res.ok || !res.body) {
    let detail = ''
    try {
      const errJson = await res.json()
      detail = errJson?.error?.message ?? ''
    } catch {
      // ignore
    }
    throw new OpenAIError(detail || `OpenAI request failed (${res.status})`, res.status)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let full = ''
  const citations: Citation[] = []
  const seenUrls = new Set<string>()

  function addCitation(url: unknown, title: unknown) {
    if (typeof url !== 'string' || !url) return
    if (seenUrls.has(url)) return
    seenUrls.add(url)
    const cit: Citation = { url, title: typeof title === 'string' && title ? title : url }
    citations.push(cit)
    opts.onCitation?.(cit)
  }

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed.startsWith('data:')) continue
      const payload = trimmed.slice(5).trim()
      if (!payload || payload === '[DONE]') continue
      let evt: Record<string, unknown>
      try {
        evt = JSON.parse(payload)
      } catch {
        continue
      }
      const t = evt.type as string | undefined
      if (!t) continue

      if (t === 'response.output_text.delta') {
        const delta = (evt.delta as string | undefined) ?? ''
        if (delta) {
          full += delta
          opts.onDelta(delta)
        }
      } else if (t === 'response.output_text.annotation.added') {
        const ann = evt.annotation as Record<string, unknown> | undefined
        if (ann && ann.type === 'url_citation') {
          addCitation(ann.url, ann.title)
        }
      } else if (t === 'response.output_item.added') {
        const item = evt.item as Record<string, unknown> | undefined
        if (item && item.type === 'web_search_call') {
          opts.onSearchStart?.()
        }
      } else if (t === 'response.completed') {
        // Final sweep: the completed response may carry annotations we did not catch streaming.
        const response = evt.response as Record<string, unknown> | undefined
        const output = response?.output as Array<Record<string, unknown>> | undefined
        if (Array.isArray(output)) {
          for (const item of output) {
            if (item.type !== 'message') continue
            const content = item.content as Array<Record<string, unknown>> | undefined
            if (!Array.isArray(content)) continue
            for (const part of content) {
              const anns = part.annotations as Array<Record<string, unknown>> | undefined
              if (!Array.isArray(anns)) continue
              for (const a of anns) {
                if (a.type === 'url_citation') addCitation(a.url, a.title)
              }
            }
          }
        }
      } else if (t === 'response.failed' || t === 'error') {
        const err = (evt.error as { message?: string } | undefined)?.message
        throw new OpenAIError(err || 'OpenAI streaming error')
      }
    }
  }

  return { text: full, citations }
}

export function streamConversationWeb(params: {
  apiKey: string
  model: string
  context: StockContext
  history: ChatMessage[]
  onDelta: (chunk: string) => void
  onCitation?: (c: Citation) => void
  onSearchStart?: () => void
  signal?: AbortSignal
  temperature?: number
  extraSystemPrompt?: string
  financials?: FinancialsSnapshot
  financialsDepth?: FinancialsDepthMode
}): Promise<WebStreamResult> {
  const baseSystem = `${CHAT_SYSTEM}\n\nWhen the user asks about anything that may have changed recently (earnings, news, filings, guidance, analyst actions, macro), use the web_search tool to find the latest information and cite sources. For purely technical/price questions, answer from the context block without searching.\n\nCurrent stock context:\n${contextBlock(params.context)}` + financialsBlock(params.financials, params.financialsDepth ?? 'concise')
  const systemMsg = {
    role: 'system' as const,
    content: withCustomPrompt(baseSystem, params.extraSystemPrompt),
  }
  const input = [systemMsg, ...params.history.map(m => ({ role: m.role, content: m.content }))]
  return streamResponses({
    apiKey: params.apiKey,
    model: params.model,
    input,
    signal: params.signal,
    onDelta: params.onDelta,
    onCitation: params.onCitation,
    onSearchStart: params.onSearchStart,
    temperature: params.temperature,
  })
}

export function streamBriefWeb(params: {
  apiKey: string
  model: string
  context: StockContext
  onDelta: (chunk: string) => void
  onCitation?: (c: Citation) => void
  onSearchStart?: () => void
  signal?: AbortSignal
  temperature?: number
  extraSystemPrompt?: string
  financials?: FinancialsSnapshot
  financialsDepth?: FinancialsDepthMode
}): Promise<WebStreamResult> {
  const baseSystem = BRIEF_SYSTEM + '\n\nUse the web_search tool to incorporate the latest news, earnings, or filings relevant to today\'s setup. Cite sources.' + financialsBlock(params.financials, params.financialsDepth ?? 'concise')
  const input = [
    { role: 'system' as const, content: withCustomPrompt(baseSystem, params.extraSystemPrompt) },
    {
      role: 'user' as const,
      content: `Stock snapshot:\n${contextBlock(params.context)}\n\nWrite the ~150-word strategy brief now, factoring in any recent news from the web.`,
    },
  ]
  return streamResponses({
    apiKey: params.apiKey,
    model: params.model,
    input,
    signal: params.signal,
    onDelta: params.onDelta,
    onCitation: params.onCitation,
    onSearchStart: params.onSearchStart,
    temperature: params.temperature,
  })
}

function renderLineupMarket(
  market: import('../types').MarketOverview | null | undefined,
): string {
  if (!market) return 'Market overview: unavailable.'
  const sectors = [...(market.sectors ?? [])]
    .sort((a, b) => b.return_1d - a.return_1d)
    .slice(0, 12)
    .map(s => {
      const parts = [
        `${s.name}: 1D ${s.return_1d >= 0 ? '+' : ''}${s.return_1d.toFixed(2)}%`,
        s.return_1w == null ? '' : `1W ${s.return_1w >= 0 ? '+' : ''}${s.return_1w.toFixed(2)}%`,
        s.return_1m == null ? '' : `1M ${s.return_1m >= 0 ? '+' : ''}${s.return_1m.toFixed(2)}%`,
        s.trend ? `trend ${s.trend}` : '',
      ].filter(Boolean)
      return `- ${parts.join(' | ')}`
    })
  return [
    `Market trend: ${market.market_trend}`,
    market.advances == null ? '' : `Advancers: ${market.advances}`,
    market.declines == null ? '' : `Decliners: ${market.declines}`,
    market.market_health ? `Market health: ${market.market_health}` : '',
    'Sector snapshot:',
    sectors.join('\n') || '- No sector data.',
  ].filter(Boolean).join('\n')
}

function renderLineupPositions(positions: Array<{
  ticker: string
  kind?: string
  quantity?: number
  costBasis?: number
  companyName?: string
  sector?: string
  optionType?: string
  strike?: number
  expiry?: string
}>): string {
  if (positions.length === 0) return 'Paper portfolio positions: none.'
  return [
    'Paper portfolio positions:',
    ...positions.map(p => {
      const descriptor = p.kind === 'option'
        ? `${p.optionType?.toUpperCase() ?? 'OPTION'} ${p.strike ?? '-'} ${p.expiry ?? ''}`.trim()
        : 'stock'
      const details = [
        p.companyName,
        p.sector,
        `qty ${p.quantity ?? '-'}`,
        p.costBasis == null ? '' : `avg cost ${p.costBasis.toFixed(2)}`,
      ].filter(Boolean)
      return `- ${p.ticker} (${descriptor}): ${details.join(' | ')}`
    }),
  ].join('\n')
}

export interface DailyLineupPosition {
  ticker: string
  kind?: string
  quantity?: number
  costBasis?: number
  companyName?: string
  sector?: string
  optionType?: string
  strike?: number
  expiry?: string
}

export function streamDailyLineup(params: {
  apiKey: string
  model: string
  watchlist: string[]
  positions: DailyLineupPosition[]
  marketOverview?: MarketOverview | null
  onDelta: (chunk: string) => void
  onCitation?: (c: Citation) => void
  onSearchStart?: () => void
  signal?: AbortSignal
  temperature?: number
  extraSystemPrompt?: string
}): Promise<WebStreamResult> {
  const today = new Date().toLocaleDateString(undefined, {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })
  const system = withCustomPrompt(
    `You are TradeSmart Daily Lineup, a concise market desk assistant. Use web_search for fresh market news, earnings, macro events, sector movers, and company-specific catalysts. Build a practical watchlist for today's session from the provided market overview, the user's watchlist, and paper portfolio positions. Prioritize what is worth watching, why it matters today, and what would invalidate the setup. Output tight markdown with these sections: Market Tape, Worth Watching, Portfolio Focus, Risk Calendar. Include specific tickers when useful. Cite fresh sources. No generic disclaimers.`,
    params.extraSystemPrompt,
  )
  const input = [
    { role: 'system' as const, content: system },
    {
      role: 'user' as const,
      content: [
        `Date: ${today}`,
        `User watchlist: ${params.watchlist.length ? params.watchlist.join(', ') : 'none'}`,
        renderLineupPositions(params.positions),
        renderLineupMarket(params.marketOverview),
        'Create the daily lineup now. Keep it scan-friendly and focus on names or themes worth watching today.',
      ].join('\n\n'),
    },
  ]
  return streamResponses({
    apiKey: params.apiKey,
    model: params.model,
    input,
    signal: params.signal,
    onDelta: params.onDelta,
    onCitation: params.onCitation,
    onSearchStart: params.onSearchStart,
    temperature: params.temperature,
  })
}

export function streamDailyLineupConversation(params: {
  apiKey: string
  model: string
  lineup: string
  history: ChatMessage[]
  watchlist: string[]
  positions: DailyLineupPosition[]
  marketOverview?: MarketOverview | null
  onDelta: (chunk: string) => void
  onCitation?: (c: Citation) => void
  onSearchStart?: () => void
  signal?: AbortSignal
  temperature?: number
  extraSystemPrompt?: string
}): Promise<WebStreamResult> {
  const system = withCustomPrompt(
    `You are the TradeSmart Daily Lineup follow-up assistant. Answer questions about the generated lineup, current market conditions, watchlist names, portfolio positions, catalysts, risks, and possible scenarios. Use web_search for every user follow-up so current claims are verified and cited. Clearly distinguish fresh web findings from the supplied market snapshot. Be concise, concrete, and scan-friendly. Do not claim that delayed market data is live.`,
    params.extraSystemPrompt,
  )
  const suppliedContext = [
    `User watchlist: ${params.watchlist.length ? params.watchlist.join(', ') : 'none'}`,
    renderLineupPositions(params.positions),
    renderLineupMarket(params.marketOverview),
    `Generated Daily Lineup:\n${params.lineup}`,
  ].join('\n\n')
  const input = [
    { role: 'system' as const, content: `${system}\n\nSupplied research context:\n${suppliedContext}` },
    ...params.history.map(message => ({ role: message.role, content: message.content })),
  ]
  return streamResponses({
    apiKey: params.apiKey,
    model: params.model,
    input,
    signal: params.signal,
    onDelta: params.onDelta,
    onCitation: params.onCitation,
    onSearchStart: params.onSearchStart,
    temperature: params.temperature,
  })
}

// ---------------------------------------------------------------------------
// Citation-backed supply-chain research
// ---------------------------------------------------------------------------

const SUPPLY_CHAIN_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['company', 'summary', 'generated_at', 'data_quality', 'relationships', 'facilities', 'related_entities', 'documents', 'news', 'risks', 'sources'],
  properties: {
    company: {
      type: 'object',
      additionalProperties: false,
      required: ['name', 'ticker', 'country', 'sector', 'description'],
      properties: {
        name: { type: 'string' },
        ticker: { type: ['string', 'null'] },
        country: { type: ['string', 'null'] },
        sector: { type: ['string', 'null'] },
        description: { type: 'string' },
      },
    },
    summary: { type: 'string' },
    generated_at: { type: 'string' },
    data_quality: { type: 'string', enum: ['limited', 'moderate', 'strong'] },
    relationships: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'kind', 'company_name', 'ticker', 'country', 'sector', 'relationship', 'exposure_pct', 'exposure_basis', 'revenue_pct', 'cogs_pct', 'capex_pct', 'evidence_class', 'confidence', 'evidence_summary', 'source_urls', 'last_verified'],
        properties: {
          id: { type: 'string' },
          kind: { type: 'string', enum: ['supplier', 'customer', 'competitor'] },
          company_name: { type: 'string' },
          ticker: { type: ['string', 'null'] },
          country: { type: ['string', 'null'] },
          sector: { type: ['string', 'null'] },
          relationship: { type: 'string' },
          exposure_pct: { type: ['number', 'null'], minimum: 0, maximum: 100 },
          exposure_basis: { type: 'string', enum: ['revenue', 'cogs', 'capex', 'procurement', 'none'] },
          revenue_pct: { type: ['number', 'null'], minimum: 0, maximum: 100 },
          cogs_pct: { type: ['number', 'null'], minimum: 0, maximum: 100 },
          capex_pct: { type: ['number', 'null'], minimum: 0, maximum: 100 },
          evidence_class: { type: 'string', enum: ['reported', 'calculated', 'estimated', 'relationship_only'] },
          confidence: { type: 'number', minimum: 0, maximum: 100 },
          evidence_summary: { type: 'string' },
          source_urls: { type: 'array', items: { type: 'string' } },
          last_verified: { type: 'string' },
        },
      },
    },
    facilities: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'company_name', 'company_ticker', 'facility_name', 'facility_type', 'city', 'region', 'country', 'latitude', 'longitude', 'coordinate_precision', 'products', 'description', 'confidence', 'source_urls', 'last_verified'],
        properties: {
          id: { type: 'string' },
          company_name: { type: 'string' },
          company_ticker: { type: ['string', 'null'] },
          facility_name: { type: 'string' },
          facility_type: { type: 'string', enum: ['manufacturing', 'distribution', 'warehouse', 'research', 'office', 'headquarters', 'other'] },
          city: { type: ['string', 'null'] },
          region: { type: ['string', 'null'] },
          country: { type: 'string' },
          latitude: { type: 'number', minimum: -90, maximum: 90 },
          longitude: { type: 'number', minimum: -180, maximum: 180 },
          coordinate_precision: { type: 'string', enum: ['exact', 'city', 'region'] },
          products: { type: 'array', items: { type: 'string' } },
          description: { type: 'string' },
          confidence: { type: 'number', minimum: 0, maximum: 100 },
          source_urls: { type: 'array', items: { type: 'string' } },
          last_verified: { type: 'string' },
        },
      },
    },
    related_entities: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'parent_id', 'hierarchy_level', 'entity_name', 'ticker', 'relationship_type', 'security_type', 'ownership_pct', 'country', 'government_owned', 'state_owner_country', 'risk_level', 'description', 'confidence', 'source_urls', 'last_verified'],
        properties: {
          id: { type: 'string' },
          parent_id: { type: ['string', 'null'] },
          hierarchy_level: { type: 'integer', minimum: 0, maximum: 8 },
          entity_name: { type: 'string' },
          ticker: { type: ['string', 'null'] },
          relationship_type: { type: 'string', enum: ['ultimate_parent', 'parent', 'subsidiary', 'major_investor', 'joint_venture', 'debt_issuer', 'related_entity', 'other'] },
          security_type: { type: 'string', enum: ['debt', 'equity', 'fund', 'joint', 'none'] },
          ownership_pct: { type: ['number', 'null'], minimum: 0, maximum: 100 },
          country: { type: ['string', 'null'] },
          government_owned: { type: 'boolean' },
          state_owner_country: { type: ['string', 'null'] },
          risk_level: { type: 'string', enum: ['low', 'medium', 'high', 'unknown'] },
          description: { type: 'string' },
          confidence: { type: 'number', minimum: 0, maximum: 100 },
          source_urls: { type: 'array', items: { type: 'string' } },
          last_verified: { type: 'string' },
        },
      },
    },
    documents: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'company_name', 'company_ticker', 'title', 'document_type', 'published_at', 'summary', 'disruption_relevance', 'url'],
        properties: {
          id: { type: 'string' },
          company_name: { type: 'string' },
          company_ticker: { type: ['string', 'null'] },
          title: { type: 'string' },
          document_type: { type: 'string', enum: ['annual_report', 'regulatory_filing', 'earnings', 'procurement', 'contract', 'presentation', 'policy', 'other'] },
          published_at: { type: ['string', 'null'] },
          summary: { type: 'string' },
          disruption_relevance: { type: 'string' },
          url: { type: 'string' },
        },
      },
    },
    news: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'company_name', 'company_ticker', 'headline', 'publisher', 'published_at', 'event_type', 'severity', 'summary', 'location', 'url'],
        properties: {
          id: { type: 'string' },
          company_name: { type: 'string' },
          company_ticker: { type: ['string', 'null'] },
          headline: { type: 'string' },
          publisher: { type: 'string' },
          published_at: { type: ['string', 'null'] },
          event_type: { type: 'string', enum: ['supply_disruption', 'plant_event', 'labour', 'logistics', 'sanctions', 'geopolitical', 'weather', 'cyber', 'regulatory', 'other'] },
          severity: { type: 'string', enum: ['low', 'medium', 'high'] },
          summary: { type: 'string' },
          location: { type: ['string', 'null'] },
          url: { type: 'string' },
        },
      },
    },
    risks: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['title', 'severity', 'description', 'affected_relationship_ids'],
        properties: {
          title: { type: 'string' },
          severity: { type: 'string', enum: ['low', 'medium', 'high'] },
          description: { type: 'string' },
          affected_relationship_ids: { type: 'array', items: { type: 'string' } },
        },
      },
    },
    sources: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['url', 'title', 'publisher', 'published_at'],
        properties: {
          url: { type: 'string' },
          title: { type: 'string' },
          publisher: { type: 'string' },
          published_at: { type: ['string', 'null'] },
        },
      },
    },
  },
} as const

function responseOutputPayload(response: Record<string, unknown>): { text: string; citations: Citation[] } {
  const output = response.output as Array<Record<string, unknown>> | undefined
  if (!Array.isArray(output)) return { text: '', citations: [] }
  const citations: Citation[] = []
  for (const item of output) {
    if (item.type !== 'message') continue
    const content = item.content as Array<Record<string, unknown>> | undefined
    if (!Array.isArray(content)) continue
    for (const part of content) {
      if (part.type === 'output_text' && typeof part.text === 'string') {
        const annotations = part.annotations as Array<Record<string, unknown>> | undefined
        if (Array.isArray(annotations)) {
          for (const annotation of annotations) {
            if (annotation.type === 'url_citation' && typeof annotation.url === 'string') {
              citations.push({
                url: annotation.url,
                title: typeof annotation.title === 'string' ? annotation.title : annotation.url,
              })
            }
          }
        }
        return { text: part.text, citations }
      }
      if (part.type === 'refusal' && typeof part.refusal === 'string') {
        throw new OpenAIError(part.refusal)
      }
    }
  }
  return { text: '', citations }
}

export async function researchSupplyChain(params: {
  apiKey: string
  model: string
  query: string
  companyHint?: string
  sectorHint?: string
  signal?: AbortSignal
}): Promise<import('../types').SupplyChainResearch> {
  if (!params.apiKey) throw new OpenAIError('No OpenAI API key set. Add one in Settings.')
  const today = new Date().toISOString().slice(0, 10)
  const system = `You are an equity-research supply-chain analyst. Build a defensible map of a public company's material suppliers, customers, and competitors using web search.

Research rules:
- Prefer primary sources: regulatory filings, annual reports, investor relations releases, procurement disclosures, government contracts, and direct company statements. Use reputable reporting only as secondary evidence.
- Never invent a relationship. If a relationship is plausible but cannot be evidenced, omit it.
- Build normalized mapped-composition estimates as well as sourced exposure figures. For customers, set revenue_pct on every returned customer and make the customer revenue_pct values sum to exactly 100. For suppliers, set cogs_pct and capex_pct on every returned supplier and make each supplier metric sum to exactly 100. These are shares of the returned map, not claims that the named companies cover all actual company revenue or spend.
- Use disclosed values and transparent calculations first. Allocate the undisclosed remainder using defensible materiality signals such as contract size, purchase volume, segment exposure, strategic importance, and evidence strength. Mark model-allocated relationships estimated and explain that basis. Use 0 where a supplier is not relevant to CAPEX or COGS.
- Competitors must have null revenue_pct, cogs_pct, and capex_pct. Keep exposure_pct for a sourced company-specific figure only; otherwise set it to null.
- evidence_class must be reported, calculated, estimated, or relationship_only. Explain the basis briefly.
- Every relationship must have at least one source_urls entry copied from an actual web source used during research. URLs must also appear in sources.
- Return up to 14 suppliers, 14 customers, and 8 competitors. Include a broader evidenced long tail; concise relationship and evidence summaries are fine for less material nodes.
- Build a CMAP-style facilities map in facilities. Include evidenced manufacturing plants, assembly sites, distribution centres, warehouses, R&D sites, headquarters, and material offices belonging to the focal company and identified suppliers. Return up to 40 facilities, prioritizing manufacturing, distribution, storage, and R&D coverage across geographies.
- Never invent a facility. Each facility needs a supporting source URL. Coordinates may use the disclosed street/site location (coordinate_precision exact), the named city's centre (city), or a named region's centre (region); do not imply greater precision than the evidence supports.
- List disclosed products, components, or activities performed at a site in products. Use an empty array when none are disclosed. Keep descriptions concise for long-tail locations.
- Build a RELS-style issuer map in related_entities for the focal company and material suppliers/customers. Identify ultimate parents, direct parents, material subsidiaries, joint ventures, major investors, related listed issuers, and meaningful debt/equity/fund/joint instruments. Use parent_id and hierarchy_level to form a readable tree. Return up to 40 entries.
- Investigate beneficial and state ownership. Set government_owned and state_owner_country only when supported by evidence. Flag high risk only for a concrete sanctions, export-control, conflict, coercion, or concentrated state-control concern; do not label a country or investor hostile without specific sourced grounds.
- Build a DSCO-style documents result in documents with up to 25 useful primary documents across the focal company and mapped suppliers/customers: filings, annual reports, earnings materials, procurement disclosures, contracts, presentations, and policies. Use direct document URLs where possible and explain disruption relevance.
- Build a CN-style current news result in news with up to 25 recent, source-backed events that could explain supply disruption: plant incidents, strikes, logistics constraints, sanctions, geopolitical events, severe weather, cyber incidents, and regulatory actions. Prefer recent reporting, include publication dates, and avoid duplicating the same event.
- confidence measures evidence strength, not business importance. Primary explicit disclosure may score 90+, multiple credible secondary sources 70-89, and indirect evidence below 70.
- Use stable lowercase IDs such as supplier-tsmc. Dates use ISO YYYY-MM-DD.
- If evidence is sparse, return fewer relationships and data_quality limited. Empty arrays are preferable to speculation.
- Summarize concentration, single-source dependencies, geography, and competitive substitution risks without investment advice.`
  const user = [
    `Research target: ${params.query.trim()}`,
    params.companyHint ? `Known company name: ${params.companyHint}` : '',
    params.sectorHint ? `Known sector: ${params.sectorHint}` : '',
    `Research date: ${today}`,
    'Search the web thoroughly, resolve the correct listed entity, and return the structured supply-chain map.',
  ].filter(Boolean).join('\n')

  const res = await fetch(RESPONSES_ENDPOINT, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${params.apiKey}`,
    },
    body: JSON.stringify({
      model: params.model,
      store: false,
      tools: [{ type: 'web_search', search_context_size: 'high' }],
      include: ['web_search_call.action.sources'],
      input: [
        { role: 'system', content: system },
        { role: 'user', content: user },
      ],
      text: {
        format: {
          type: 'json_schema',
          name: 'supply_chain_research',
          strict: true,
          schema: SUPPLY_CHAIN_SCHEMA,
        },
      },
    }),
    signal: params.signal,
  })

  if (!res.ok) {
    let detail = ''
    try {
      const payload = await res.json()
      detail = payload?.error?.message ?? ''
    } catch {
      // Ignore non-JSON error bodies.
    }
    throw new OpenAIError(detail || `OpenAI research request failed (${res.status})`, res.status)
  }

  const response = await res.json() as Record<string, unknown>
  const { text, citations } = responseOutputPayload(response)
  if (!text) throw new OpenAIError('OpenAI returned no structured supply-chain research.')
  try {
    const research = JSON.parse(text) as import('../types').SupplyChainResearch
    const knownUrls = new Set(research.sources.map(source => source.url))
    for (const citation of citations) {
      if (knownUrls.has(citation.url)) continue
      knownUrls.add(citation.url)
      let publisher = 'Web source'
      try { publisher = new URL(citation.url).hostname.replace(/^www\./, '') } catch { /* keep fallback */ }
      research.sources.push({ url: citation.url, title: citation.title, publisher, published_at: null })
    }
    return normalizeSupplyChainShares(research)
  } catch {
    throw new OpenAIError('The supply-chain response could not be parsed.')
  }
}

function normalizeMetric(
  relationships: import('../types').SupplyChainRelationship[],
  field: 'revenue_pct' | 'cogs_pct' | 'capex_pct',
) {
  if (!relationships.length) return
  const weights = relationships.map(item => {
    const legacyValue = field === 'revenue_pct' && item.exposure_basis === 'revenue'
      ? item.exposure_pct
      : field === 'cogs_pct' && (item.exposure_basis === 'cogs' || item.exposure_basis === 'procurement')
        ? item.exposure_pct
        : field === 'capex_pct' && item.exposure_basis === 'capex'
          ? item.exposure_pct
          : null
    const value = Number(item[field] ?? legacyValue)
    return Number.isFinite(value) && value >= 0 ? value : 0
  })
  const total = weights.reduce((sum, value) => sum + value, 0)
  const source = total > 0 ? weights : weights.map(() => 1)
  const sourceTotal = source.reduce((sum, value) => sum + value, 0)
  const exactTenths = source.map(value => value / sourceTotal * 1000)
  const tenths = exactTenths.map(Math.floor)
  let remainder = 1000 - tenths.reduce((sum, value) => sum + value, 0)
  const order = exactTenths.map((value, index) => ({ index, fraction: value - Math.floor(value) })).sort((a, b) => b.fraction - a.fraction)
  for (let i = 0; i < remainder; i += 1) tenths[order[i % order.length].index] += 1
  relationships.forEach((item, index) => { item[field] = tenths[index] / 10 })
}

export function normalizeSupplyChainShares(research: import('../types').SupplyChainResearch) {
  const customers = research.relationships.filter(item => item.kind === 'customer')
  const suppliers = research.relationships.filter(item => item.kind === 'supplier')
  normalizeMetric(customers, 'revenue_pct')
  normalizeMetric(suppliers, 'cogs_pct')
  normalizeMetric(suppliers, 'capex_pct')
  return research
}
