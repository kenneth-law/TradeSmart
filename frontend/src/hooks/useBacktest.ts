import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'

export function useBacktestResults(backtestId: string | null) {
  return useQuery({
    queryKey: ['backtest-results', backtestId],
    queryFn: () => api.getBacktestResults(backtestId!),
    enabled: !!backtestId,
    staleTime: Infinity,
  })
}

export function useIntegratedResults(systemId: string | null) {
  return useQuery({
    queryKey: ['integrated-results', systemId],
    queryFn: () => api.getIntegratedResults(systemId!),
    enabled: !!systemId,
    staleTime: Infinity,
  })
}
