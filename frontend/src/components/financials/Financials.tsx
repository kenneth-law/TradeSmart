import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../../lib/api'
import type { FinancialsPeriod } from '../../types'
import FinancialsTable from './FinancialsTable'

interface Props {
  ticker: string
}

export default function Financials({ ticker }: Props) {
  const [period, setPeriod] = useState<FinancialsPeriod>('annual')

  const { data, isLoading, error } = useQuery({
    queryKey: ['financials', ticker, period],
    queryFn: () => api.getFinancials(ticker, period),
    enabled: !!ticker,
    staleTime: 60 * 60 * 1000,
  })

  const periodToggle = (
    <div className="flex items-center gap-0 border border-border bg-s1/60">
      {(['annual', 'quarterly'] as const).map(p => (
        <button
          key={p}
          type="button"
          onClick={() => setPeriod(p)}
          className={[
            'px-3 py-1 text-2xs uppercase tracking-wider transition-colors',
            period === p
              ? 'bg-accent text-bg font-semibold'
              : 'text-dim hover:text-text',
          ].join(' ')}
        >
          {p}
        </button>
      ))}
    </div>
  )

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="shrink-0 flex items-center gap-3 px-3 py-1.5 border-b border-border bg-s1/40">
        {periodToggle}
        {isLoading && <span className="text-2xs text-dim">Loading {period}...</span>}
        {error && <span className="text-2xs text-down">Failed to load financials.</span>}
      </div>

      <div className="flex-1 min-h-0 overflow-auto">
        <div className="flex divide-x divide-border" style={{ minWidth: 'max-content', width: '100%' }}>
          <FinancialsTable title="Income Statement" data={data?.income_statement} />
          <FinancialsTable title="Balance Sheet" data={data?.balance_sheet} />
          <FinancialsTable title="Cash Flow" data={data?.cash_flow} />
        </div>
      </div>
    </div>
  )
}
