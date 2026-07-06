import { useEffect, useRef, useState } from 'react'
import { Globe, Send, Square, X } from 'lucide-react'
import type { MarketOverview } from '../../types'
import { isReasoningModel, useAppStore } from '../../store/useAppStore'
import {
  OpenAIError,
  streamDailyLineupConversation,
  type ChatMessage,
  type Citation,
  type DailyLineupPosition,
} from '../../lib/openai'
import Markdown from './Markdown'

interface Props {
  lineup: string
  watchlist: string[]
  positions: DailyLineupPosition[]
  marketOverview: MarketOverview
  onClose: () => void
}

export default function DailyLineupChat({ lineup, watchlist, positions, marketOverview, onClose }: Props) {
  const apiKey = useAppStore(s => s.openaiKey)
  const settings = useAppStore(s => s.settings)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [searching, setSearching] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const scrollerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const el = scrollerRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages, streaming, searching])

  useEffect(() => () => abortRef.current?.abort(), [])

  function appendToAssistant(chunk: string) {
    setMessages(current => {
      const last = current[current.length - 1]
      if (!last || last.role !== 'assistant') return current
      const updated = [...current]
      updated[updated.length - 1] = { ...last, content: last.content + chunk }
      return updated
    })
  }

  function addCitationToAssistant(citation: Citation) {
    setMessages(current => {
      const last = current[current.length - 1]
      if (!last || last.role !== 'assistant') return current
      const existing = last.citations ?? []
      if (existing.some(item => item.url === citation.url)) return current
      const updated = [...current]
      updated[updated.length - 1] = { ...last, citations: [...existing, citation] }
      return updated
    })
  }

  function send() {
    const question = input.trim()
    if (!question || streaming || !apiKey) return

    const next: ChatMessage[] = [
      ...messages,
      { role: 'user', content: question },
      { role: 'assistant', content: '' },
    ]
    setMessages(next)
    setInput('')
    setError(null)
    setStreaming(true)
    setSearching(false)

    const controller = new AbortController()
    abortRef.current = controller
    const history = next
      .slice(0, -1)
      .filter(message => message.content || message.role === 'user')
      .map(({ role, content }) => ({ role, content }))

    streamDailyLineupConversation({
      apiKey,
      model: settings.openaiModel,
      lineup,
      history,
      watchlist,
      positions,
      marketOverview,
      signal: controller.signal,
      temperature: isReasoningModel(settings.openaiModel) ? undefined : settings.openaiTemperature,
      extraSystemPrompt: settings.openaiSystemPrompt,
      onDelta: appendToAssistant,
      onCitation: addCitationToAssistant,
      onSearchStart: () => setSearching(true),
    })
      .then(() => {
        setStreaming(false)
        setSearching(false)
      })
      .catch(cause => {
        setStreaming(false)
        setSearching(false)
        if (controller.signal.aborted) return
        setError(cause instanceof OpenAIError ? cause.message : 'Daily Lineup chat request failed.')
        setMessages(current => {
          const last = current[current.length - 1]
          return last?.role === 'assistant' && !last.content ? current.slice(0, -1) : current
        })
      })
  }

  function stop() {
    abortRef.current?.abort()
    setStreaming(false)
    setSearching(false)
  }

  function hostFromUrl(url: string) {
    try {
      return new URL(url).hostname.replace(/^www\./, '')
    } catch {
      return url
    }
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col border-t border-border">
      <div className="flex items-center justify-between border-b border-border bg-s1/60 px-3 py-2">
        <div className="flex items-center gap-1.5 text-2xs uppercase tracking-[0.2em] text-dim">
          Chat · Daily Lineup
          <Globe size={11} className="text-accent" aria-label="Web search enabled" />
        </div>
        <button type="button" onClick={onClose} title="Close chat" className="text-dim hover:text-text">
          <X size={13} />
        </button>
      </div>

      <div ref={scrollerRef} className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-3 py-3">
        {messages.length === 0 && !error && (
          <div className="text-2xs leading-relaxed text-dim">
            Ask follow-up questions about the lineup, catalysts, sectors, portfolio exposure, or what to watch next. Web search is enabled for every answer.
          </div>
        )}

        {messages.map((message, index) => {
          const isLast = index === messages.length - 1
          return (
            <div
              key={index}
              className={[
                'break-words',
                message.role === 'user'
                  ? 'self-end max-w-[90%] whitespace-pre-wrap border border-border bg-s2 px-2 py-1.5 text-2xs leading-relaxed text-text'
                  : 'self-start max-w-full',
              ].join(' ')}
            >
              {message.role === 'assistant' && (
                <div className="mb-1 text-[10px] uppercase tracking-widest text-accent">AI</div>
              )}
              {message.role === 'assistant' && searching && isLast && !message.content && (
                <div className="mb-1 flex items-center gap-1.5 text-dim">
                  <Globe size={11} className="animate-pulse" />
                  <span className="text-2xs">Searching the web...</span>
                </div>
              )}
              {message.role === 'assistant' ? (
                message.content ? <Markdown text={message.content} /> : (
                  streaming && isLast && !searching
                    ? <span className="inline-block h-3 w-1.5 animate-pulse bg-accent align-middle" />
                    : null
                )
              ) : message.content}
              {message.citations && message.citations.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1.5 border-t border-border/50 pt-2">
                  {message.citations.map((citation, citationIndex) => (
                    <a
                      key={citation.url}
                      href={citation.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      title={citation.title}
                      className="inline-flex items-center gap-1 border border-border bg-s1/60 px-1.5 py-0.5 text-[10px] text-accent hover:underline tabnum"
                    >
                      <span className="text-dim">[{citationIndex + 1}]</span>
                      <span className="max-w-[140px] truncate">{hostFromUrl(citation.url)}</span>
                    </a>
                  ))}
                </div>
              )}
            </div>
          )
        })}

        {error && (
          <div className="border border-down/40 bg-down/10 px-2 py-1.5 text-2xs leading-relaxed text-down">
            {error}
          </div>
        )}
      </div>

      <div className="border-t border-border bg-s1/60 p-2">
        <div className="flex items-end gap-1.5">
          <textarea
            value={input}
            onChange={event => setInput(event.target.value)}
            onKeyDown={event => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                send()
              }
            }}
            placeholder="Ask about the lineup (web search on)..."
            rows={2}
            disabled={streaming}
            className="min-w-0 flex-1 resize-none border border-border bg-bg px-2 py-1.5 text-2xs text-text outline-none placeholder:text-dim focus-visible:border-accent disabled:opacity-60"
          />
          <div
            title="Web search enabled for every message"
            className="shrink-0 border border-accent bg-accent px-2 py-1.5 text-bg"
          >
            <Globe size={11} />
          </div>
          {streaming ? (
            <button type="button" onClick={stop} title="Stop" className="shrink-0 border border-down bg-down/20 px-2 py-1.5 text-down hover:bg-down/30">
              <Square size={11} fill="currentColor" />
            </button>
          ) : (
            <button type="button" onClick={send} disabled={!input.trim()} title="Send" className="shrink-0 border border-accent bg-accent px-2 py-1.5 text-bg disabled:cursor-not-allowed disabled:opacity-40">
              <Send size={11} />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
