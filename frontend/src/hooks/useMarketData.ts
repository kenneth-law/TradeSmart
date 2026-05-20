import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import { useAppStore } from '../store/useAppStore'

export function useMarketOverview() {
  const marketRefreshSeconds = useAppStore(s => s.settings.marketRefreshSeconds)

  return useQuery({
    queryKey: ['market-overview'],
    queryFn: () => api.getMarketOverview(),
    staleTime: 120_000,
    refetchInterval: marketRefreshSeconds * 1000,
  })
}

export function usePortfolio() {
  return useQuery({
    queryKey: ['portfolio'],
    queryFn: () => api.getPortfolio(),
    staleTime: 60_000,
  })
}
