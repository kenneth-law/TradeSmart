import { useEffect, useRef, useState } from 'react'
import { Globe, Send, Square, X } from 'lucide-react'
import type { StockResult } from '../../types'
import { useQuery } from '@tanstack/react-query'
import { isReasoningModel, useAppStore } from '../../store/useAppStore'
import { api } from '../../lib/api'
import {
  buildStockContext,
  OpenAIError,
  streamConversation,
  streamConversationWeb,
  type ChatMessage,
  type Citation,
  type FinancialsSnapshot,
} from '../../lib/openai'
import Markdown from './Markdown'

interface Props {
  stock: StockResult
  onClose: () => void
}

export default function ChatPanel({ stock, onClose }: Props) {
  const apiKey = useAppStore(s => s.openaiKey)
  const model = useAppStore(s => s.settings.openaiModel)
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
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [searching, setSearching] = useState(false)
  const [useWeb, setUseWeb] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const scrollerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    setMessages([])
    setError(null)
    setSearching(false)
    abortRef.current?.abort()
  }, [stock.ticker])

  useEffect(() => {
    const el = scrollerRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages, streaming, searching])

  useEffect(() => () => abortRef.current?.abort(), [])

  function appendToAssistant(chunk: string) {
    setMessages(curr => {
      const last = curr[curr.length - 1]
      if (!last || last.role !== 'assistant') return curr
      const updated = [...curr]
      updated[updated.length - 1] = { ...last, content: last.content + chunk }
      return updated
    })
  }

  function addCitationToAssistant(c: Citation) {
    setMessages(curr => {
      const last = curr[curr.length - 1]
      if (!last || last.role !== 'assistant') return curr
      const existing = last.citations ?? []
      if (existing.some(e => e.url === c.url)) return curr
      const updated = [...curr]
      updated[updated.length - 1] = { ...last, citations: [...existing, c] }
      return updated
    })
  }

  function send() {
    const trimmed = input.trim()
    if (!trimmed || streaming || !apiKey) return

    const webForThisMessage = useWeb
    const next: ChatMessage[] = [
      ...messages,
      { role: 'user', content: trimmed },
      { role: 'assistant', content: '' },
    ]
    setMessages(next)
    setInput('')
    setError(null)
    setStreaming(true)
    setSearching(false)
    setUseWeb(false)

    const controller = new AbortController()
    abortRef.current = controller

    const history: ChatMessage[] = next
      .slice(0, -1)
      .filter(m => m.content || m.role === 'user')
      .map(({ role, content }) => ({ role, content }))

    const ctx = buildStockContext(stock)
    const temperature = isReasoningModel(model) ? undefined : temperatureSetting
    const snap: FinancialsSnapshot = {
      annual: annualFinancials ?? null,
      quarterly: financialsDepth === 'full' ? (quarterlyFinancials ?? null) : null,
    }

    const promise = webForThisMessage
      ? streamConversationWeb({
          apiKey,
          model,
          context: ctx,
          history,
          signal: controller.signal,
          onDelta: appendToAssistant,
          onCitation: addCitationToAssistant,
          onSearchStart: () => setSearching(true),
          temperature,
          extraSystemPrompt: customPrompt,
          financials: snap,
          financialsDepth,
        }).then(() => undefined)
      : streamConversation({
          apiKey,
          model,
          context: ctx,
          history,
          signal: controller.signal,
          onDelta: appendToAssistant,
          temperature,
          extraSystemPrompt: customPrompt,
          financials: snap,
          financialsDepth,
        }).then(() => undefined)

    promise
      .then(() => {
        setStreaming(false)
        setSearching(false)
      })
      .catch(err => {
        setStreaming(false)
        setSearching(false)
        if (controller.signal.aborted) return
        setError(err instanceof OpenAIError ? err.message : 'Chat request failed.')
        setMessages(curr => {
          const last = curr[curr.length - 1]
          if (last && last.role === 'assistant' && !last.content) {
            return curr.slice(0, -1)
          }
          return curr
        })
      })
  }

  function stop() {
    abortRef.current?.abort()
    setStreaming(false)
    setSearching(false)
  }

  function hostFromUrl(url: string): string {
    try {
      return new URL(url).hostname.replace(/^www\./, '')
    } catch {
      return url
    }
  }

  return (
    <div className="flex flex-col flex-1 min-h-0 border-t border-border">
      <div className="flex items-center justify-between px-3 py-2 border-b border-border bg-s1/60">
        <div className="text-2xs uppercase tracking-[0.2em] text-dim">Chat · {stock.ticker}</div>
        <button
          type="button"
          onClick={onClose}
          title="Close chat"
          className="text-dim hover:text-text"
        >
          <X size={13} />
        </button>
      </div>

      <div ref={scrollerRef} className="flex-1 min-h-0 overflow-y-auto px-3 py-3 flex flex-col gap-3">
        {messages.length === 0 && !error && (
          <div className="text-2xs text-dim leading-relaxed">
            Ask anything about {stock.ticker} - entries, stops, what the signal really means, how today compares to recent action.
            Click the globe to let the AI search the web (earnings, news, filings).
          </div>
        )}

        {messages.map((m, i) => {
          const isLast = i === messages.length - 1
          return (
            <div
              key={i}
              className={[
                'break-words',
                m.role === 'user'
                  ? 'self-end max-w-[90%] border border-border bg-s2 text-text px-2 py-1.5 text-2xs leading-relaxed whitespace-pre-wrap'
                  : 'self-start max-w-full',
              ].join(' ')}
            >
              {m.role === 'assistant' && (
                <div className="text-[10px] uppercase tracking-widest text-accent mb-1">AI</div>
              )}
              {m.role === 'assistant' && searching && isLast && !m.content && (
                <div className="flex items-center gap-1.5 text-dim mb-1">
                  <Globe size={11} className="animate-pulse" />
                  <span className="text-2xs">Searching the web...</span>
                </div>
              )}
              {m.role === 'assistant' ? (
                m.content ? (
                  <Markdown text={m.content} />
                ) : (streaming && isLast && !searching ? (
                  <span className="inline-block w-1.5 h-3 bg-accent align-middle animate-pulse" />
                ) : null)
              ) : (
                m.content
              )}
              {m.citations && m.citations.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mt-2 pt-2 border-t border-border/50">
                  {m.citations.map((c, idx) => (
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
            </div>
          )
        })}

        {error && (
          <div className="text-2xs text-down leading-relaxed border border-down/40 bg-down/10 px-2 py-1.5">
            {error}
          </div>
        )}
      </div>

      <div className="border-t border-border p-2 bg-s1/60">
        <div className="flex items-end gap-1.5">
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                send()
              }
            }}
            placeholder={useWeb ? 'Ask (web search on)...' : 'Ask about this stock...'}
            rows={2}
            disabled={streaming}
            className="flex-1 min-w-0 resize-none border border-border bg-bg/80 px-2 py-1.5 text-2xs text-text outline-none focus-visible:border-accent placeholder:text-dim disabled:opacity-60"
          />
          <button
            type="button"
            onClick={() => setUseWeb(v => !v)}
            disabled={streaming}
            title={useWeb ? 'Web search on (this message)' : 'Enable web search for this message'}
            className={[
              'shrink-0 border px-2 py-1.5 disabled:opacity-40 disabled:cursor-not-allowed',
              useWeb
                ? 'border-accent bg-accent text-bg'
                : 'border-border bg-s2 text-dim hover:text-text hover:border-border-strong',
            ].join(' ')}
          >
            <Globe size={11} />
          </button>
          {streaming ? (
            <button
              type="button"
              onClick={stop}
              title="Stop"
              className="shrink-0 border border-down bg-down/20 text-down px-2 py-1.5 hover:bg-down/30"
            >
              <Square size={11} fill="currentColor" />
            </button>
          ) : (
            <button
              type="button"
              onClick={send}
              disabled={!input.trim()}
              title="Send"
              className="shrink-0 border border-accent bg-accent text-bg px-2 py-1.5 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <Send size={11} />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
