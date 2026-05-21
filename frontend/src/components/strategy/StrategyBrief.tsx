import { useEffect, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Globe, MessageSquare, RotateCw } from 'lucide-react'
import type { StockResult } from '../../types'
import { isReasoningModel, useAppStore } from '../../store/useAppStore'
import { api } from '../../lib/api'
import {
  buildStockContext,
  OpenAIError,
  streamBrief,
  streamBriefWeb,
  type Citation,
  type FinancialsSnapshot,
} from '../../lib/openai'
import Markdown from './Markdown'

interface Props {
  stock: StockResult
  onOpenChat: () => void
  chatOpen: boolean
}

interface CachedBrief {
  text: string
  citations: Citation[]
}

const briefCache = new Map<string, CachedBrief>()

function hashString(s: string): string {
  let h = 0
  for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0
  return h.toString(36)
}

function cacheKey(
  stock: StockResult,
  model: string,
  web: boolean,
  temp: number | undefined,
  prompt: string,
  depth: string,
  finSig: string,
): string {
  const tempPart = temp == null ? 'def' : temp.toFixed(2)
  const promptPart = prompt ? hashString(prompt) : 'np'
  return `${stock.ticker}|${model}|${web ? 'web' : 'plain'}|${tempPart}|${promptPart}|${depth}|${finSig}|${stock.day_trading_score.toFixed(1)}|${stock.current_price.toFixed(2)}`
}

function financialsSig(snap: FinancialsSnapshot, depth: string): string {
  // Cheap signature: period counts + first/last periods per statement.
  function part(resp: import('../../types').FinancialsResponse | null | undefined) {
    if (!resp) return 'x'
    const p = resp.income_statement.periods
    return `${p.length}:${p[0] ?? ''}:${p[p.length - 1] ?? ''}`
  }
  return `${depth}|${part(snap.annual)}|${part(snap.quarterly)}`
}

function hostFromUrl(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, '')
  } catch {
    return url
  }
}

export default function StrategyBrief({ stock, onOpenChat, chatOpen }: Props) {
  const apiKey = useAppStore(s => s.openaiKey)
  const model = useAppStore(s => s.settings.openaiModel)
  const briefWebSearch = useAppStore(s => s.settings.briefWebSearch)
  const temperatureSetting = useAppStore(s => s.settings.openaiTemperature)
  const customPrompt = useAppStore(s => s.settings.openaiSystemPrompt)
  const financialsDepth = useAppStore(s => s.settings.financialsDepth)

  const { data: annualFinancials } = useQuery({
    queryKey: ['financials', stock.ticker, 'annual'],
    queryFn: () => api.getFinancials(stock.ticker, 'annual'),
    enabled: !!apiKey,
    staleTime: 60 * 60 * 1000,
  })
  const { data: quarterlyFinancials } = useQuery({
    queryKey: ['financials', stock.ticker, 'quarterly'],
    queryFn: () => api.getFinancials(stock.ticker, 'quarterly'),
    enabled: !!apiKey && financialsDepth === 'full',
    staleTime: 60 * 60 * 1000,
  })
  const [text, setText] = useState('')
  const [citations, setCitations] = useState<Citation[]>([])
  const [loading, setLoading] = useState(false)
  const [searching, setSearching] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const autoRunKeyRef = useRef<string | null>(null)

  function run(force = false) {
    if (!apiKey) {
      abortRef.current?.abort()
      setText('')
      setCitations([])
      setError(null)
      return
    }
    const temperature = isReasoningModel(model) ? undefined : temperatureSetting
    const snap: FinancialsSnapshot = {
      annual: annualFinancials ?? null,
      quarterly: financialsDepth === 'full' ? (quarterlyFinancials ?? null) : null,
    }
    const finSig = financialsSig(snap, financialsDepth)
    const key = cacheKey(stock, model, briefWebSearch, temperature, customPrompt, financialsDepth, finSig)
    if (!force) {
      const cached = briefCache.get(key)
      if (cached) {
        setText(cached.text)
        setCitations(cached.citations)
        setLoading(false)
        setSearching(false)
        setError(null)
        return
      }
    }

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setText('')
    setCitations([])
    setError(null)
    setLoading(true)
    setSearching(false)

    if (briefWebSearch) {
      const cits: Citation[] = []
      streamBriefWeb({
        apiKey,
        model,
        context: buildStockContext(stock),
        signal: controller.signal,
        onDelta: chunk => setText(prev => prev + chunk),
        onCitation: c => {
          if (cits.some(x => x.url === c.url)) return
          cits.push(c)
          setCitations([...cits])
        },
        onSearchStart: () => setSearching(true),
        temperature,
        extraSystemPrompt: customPrompt,
        financials: snap,
        financialsDepth,
      })
        .then(result => {
          briefCache.set(key, { text: result.text, citations: result.citations })
          setLoading(false)
          setSearching(false)
        })
        .catch(err => {
          if (controller.signal.aborted) return
          setLoading(false)
          setSearching(false)
          setError(err instanceof OpenAIError ? err.message : 'Failed to generate brief.')
        })
    } else {
      streamBrief({
        apiKey,
        model,
        context: buildStockContext(stock),
        signal: controller.signal,
        onDelta: chunk => setText(prev => prev + chunk),
        temperature,
        extraSystemPrompt: customPrompt,
        financials: snap,
        financialsDepth,
      })
        .then(final => {
          briefCache.set(key, { text: final, citations: [] })
          setLoading(false)
        })
        .catch(err => {
          if (controller.signal.aborted) return
          setLoading(false)
          setError(err instanceof OpenAIError ? err.message : 'Failed to generate brief.')
        })
    }
  }

  useEffect(() => {
    const autoRunKey = `${stock.ticker}|${apiKey ? 'key' : 'no-key'}`
    if (autoRunKeyRef.current === autoRunKey) return
    autoRunKeyRef.current = autoRunKey
    run(false)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stock.ticker, apiKey])

  useEffect(() => {
    return () => abortRef.current?.abort()
  }, [])

  return (
    <div className="p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1.5">
          <div className="text-2xs uppercase tracking-[0.2em] text-dim">Strategy Brief</div>
          {briefWebSearch && apiKey && (
            <span title="Brief uses web search">
              <Globe size={10} className="text-accent" />
            </span>
          )}
        </div>
        {apiKey && (
          <button
            type="button"
            onClick={() => run(true)}
            disabled={loading}
            title="Regenerate brief"
            className="text-dim hover:text-text disabled:opacity-40"
          >
            <RotateCw size={11} className={loading ? 'animate-spin' : ''} />
          </button>
        )}
      </div>

      {!apiKey && (
        <div className="text-2xs text-muted leading-relaxed">
          {stock.strategy_details ? (
            <p className="mb-2">{stock.strategy_details}</p>
          ) : (
            <p className="mb-2 text-dim italic">No strategy details available.</p>
          )}
          <p className="text-dim">
            Add your OpenAI API key in Settings to generate an AI strategy brief and open a chat about this stock.
          </p>
        </div>
      )}

      {apiKey && (
        <>
          {error && (
            <div className="text-2xs text-down leading-relaxed mb-2">
              {error}
            </div>
          )}
          {!error && searching && !text && (
            <div className="flex items-center gap-1.5 text-dim mb-2">
              <Globe size={11} className="animate-pulse" />
              <span className="text-2xs">Searching the web...</span>
            </div>
          )}
          {!error && (
            <div className="mb-2 min-h-[60px]">
              {text ? (
                <Markdown text={text} />
              ) : (
                loading && !searching ? (
                  <span className="inline-block w-1.5 h-3 bg-accent align-middle animate-pulse" />
                ) : null
              )}
            </div>
          )}

          {citations.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mb-3 pt-2 border-t border-border/50">
              {citations.map((c, idx) => (
                <a
                  key={c.url}
                  href={c.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  title={c.title}
                  className="inline-flex items-center gap-1 text-[10px] text-accent hover:underline border border-border bg-s1/60 px-1.5 py-0.5 tabnum"
                >
                  <span className="text-dim">[{idx + 1}]</span>
                  <span className="truncate max-w-[140px]">{hostFromUrl(c.url)}</span>
                </a>
              ))}
            </div>
          )}

          <button
            type="button"
            onClick={onOpenChat}
            disabled={chatOpen}
            className="w-full flex items-center justify-center gap-1.5 border border-accent bg-accent text-bg px-3 py-1.5 text-2xs font-semibold hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <MessageSquare size={12} strokeWidth={2.2} />
            {chatOpen ? 'Chat open' : 'Chat'}
          </button>
        </>
      )}
    </div>
  )
}
