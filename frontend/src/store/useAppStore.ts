import { create } from 'zustand'
import type { AnalysisResult, TickerContext } from '../types'

interface AppStore {
  analysisResult: AnalysisResult | null
  setAnalysisResult: (r: AnalysisResult) => void

  tickerContext: TickerContext | null
  setTickerContext: (ctx: TickerContext) => void

  lastUpdated: Date | null
  setLastUpdated: (d: Date) => void

  connection: 'connected' | 'degraded' | 'offline'
  setConnection: (s: 'connected' | 'degraded' | 'offline') => void
}

export const useAppStore = create<AppStore>((set) => ({
  analysisResult: null,
  setAnalysisResult: (r) => set({ analysisResult: r, lastUpdated: new Date() }),

  tickerContext: null,
  setTickerContext: (ctx) => set({ tickerContext: ctx }),

  lastUpdated: null,
  setLastUpdated: (d) => set({ lastUpdated: d }),

  connection: 'connected',
  setConnection: (s) => set({ connection: s }),
}))
