import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'

export function useMarketOverview() {
  return useQuery({
    queryKey: ['market-overview'],
    queryFn: () => api.getMarketOverview(),
    staleTime: 120_000,
    refetchInterval: 300_000,
  })
}

export function usePortfolio() {
  return useQuery({
    queryKey: ['portfolio'],
    queryFn: () => api.getPortfolio(),
    staleTime: 60_000,
  })
}
