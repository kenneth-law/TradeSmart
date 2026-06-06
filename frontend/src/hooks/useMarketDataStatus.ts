import { useEffect, useMemo, useState } from 'react'
import { api } from '../lib/api'
import type { MarketDataStatus } from '../types'

type MarketDataStatusState = {
  data: MarketDataStatus
  loading: boolean
}

function backendOfflineStatus(): MarketDataStatus {
  const checkedAt = new Date().toISOString()

  return {
    provider: 'none',
    label: 'market data: none',
    state: 'offline',
    timestamp: checkedAt,
    streams: {
      alpaca: {
        provider: 'alpaca',
        configured: false,
        connected: false,
        authenticated: false,
        feed: '',
        missing: [],
        error: 'TradeSmart backend is unreachable.',
      },
      yfinance: {
        provider: 'yfinance',
        reachable: false,
        delayed: true,
        latency_ms: null,
        checked_at: checkedAt,
        error: 'TradeSmart backend is unreachable.',
      },
      openai: {
        provider: 'openai',
        status: 'unknown',
        configured: null,
      },
      tradesmart_backend: {
        provider: 'tradesmart_backend',
        reachable: false,
        status: 'offline',
        checked_at: checkedAt,
        error: 'Health request failed.',
      },
    },
  }
}

function withOpenAIStatus(data: MarketDataStatus, openaiKey: string): MarketDataStatus {
  return {
    ...data,
    streams: {
      ...data.streams,
      openai: {
        ...data.streams.openai,
        configured: Boolean(openaiKey),
        status: openaiKey ? 'configured' : 'not_configured',
        checked_at: new Date().toISOString(),
      },
    },
  }
}

export function useMarketDataStatus(openaiKey: string) {
  const [state, setState] = useState<MarketDataStatusState>(() => ({
    data: withOpenAIStatus(backendOfflineStatus(), openaiKey),
    loading: true,
  }))

  useEffect(() => {
    let alive = true

    async function load() {
      try {
        const data = await api.getMarketDataStatus()
        if (alive) setState({ data: withOpenAIStatus(data, openaiKey), loading: false })
      } catch {
        if (alive) setState({ data: withOpenAIStatus(backendOfflineStatus(), openaiKey), loading: false })
      }
    }

    load()
    const id = window.setInterval(load, 15000)
    return () => {
      alive = false
      window.clearInterval(id)
    }
  }, [openaiKey])

  return useMemo(() => state, [state])
}
