import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'

export function useAnalysisResults(sessionId: string | null) {
  return useQuery({
    queryKey: ['analysis-results', sessionId],
    queryFn: () => api.getAnalysisResults(sessionId!),
    enabled: !!sessionId,
    staleTime: Infinity,
  })
}

export function useStockDetail(ticker: string | undefined) {
  return useQuery({
    queryKey: ['stock', ticker],
    queryFn: () => api.getStock(ticker!),
    enabled: !!ticker,
    staleTime: 60_000,
  })
}

export function usePriceHistory(ticker: string | undefined) {
  return useQuery({
    queryKey: ['price-history', ticker],
    queryFn: () => api.getPriceHistory(ticker!),
    enabled: !!ticker,
    staleTime: 60_000,
  })
}

export function useIndustryPeers(ticker: string | undefined) {
  return useQuery({
    queryKey: ['peers', ticker],
    queryFn: () => api.getIndustryPeers(ticker!),
    enabled: !!ticker,
    staleTime: 120_000,
  })
}
