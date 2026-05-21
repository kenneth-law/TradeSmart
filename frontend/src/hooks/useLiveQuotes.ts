import { useEffect, useMemo, useState } from 'react'
import type { LiveMarketPayload, LiveQuote } from '../types'

type LiveState = {
  quotes: Record<string, LiveQuote | null>
  configured: boolean
  connected: boolean
  authenticated: boolean
  feed: string
  error?: string | null
}

const EMPTY: LiveState = {
  quotes: {},
  configured: false,
  connected: false,
  authenticated: false,
  feed: '',
  error: null,
}

export function useLiveQuotes(symbols: Array<string | undefined | null>, enabled = true) {
  const rawSymbols = symbols
    .map(s => s?.trim().toUpperCase())
    .filter(Boolean)
    .join(',')
  const symbolKey = useMemo(() => (
    Array.from(new Set(symbols.map(s => s?.trim().toUpperCase()).filter(Boolean) as string[]))
      .sort()
      .join(',')
  ), [rawSymbols])
  const [state, setState] = useState<LiveState>(EMPTY)

  useEffect(() => {
    if (!enabled || !symbolKey) {
      setState(EMPTY)
      return
    }

    const es = new EventSource(`/api/live/stream?symbols=${encodeURIComponent(symbolKey)}`)
    es.onmessage = event => {
      try {
        const data = JSON.parse(event.data) as LiveMarketPayload
        setState({
          quotes: data.quotes ?? {},
          configured: data.configured,
          connected: data.connected,
          authenticated: data.authenticated,
          feed: data.feed,
          error: data.error,
        })
      } catch {
        // Ignore malformed frames.
      }
    }
    es.onerror = () => {
      setState(prev => ({ ...prev, connected: false, error: prev.error ?? 'Live quote stream disconnected.' }))
    }

    return () => es.close()
  }, [enabled, symbolKey])

  return state
}

export function quotePrice(quote?: LiveQuote | null): number | null {
  if (!quote) return null
  const values = [quote.price, quote.last, quote.bid && quote.ask ? (quote.bid + quote.ask) / 2 : null]
  for (const value of values) {
    if (typeof value === 'number' && Number.isFinite(value) && value > 0) return value
  }
  return null
}
