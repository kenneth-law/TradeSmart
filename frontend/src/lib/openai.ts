import type { DetailedMetrics, StockResult } from '../types'

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

export function streamDailyLineup(params: {
  apiKey: string
  model: string
  watchlist: string[]
  positions: Array<{
    ticker: string
    kind?: string
    quantity?: number
    costBasis?: number
    companyName?: string
    sector?: string
    optionType?: string
    strike?: number
    expiry?: string
  }>
  marketOverview?: import('../types').MarketOverview | null
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
