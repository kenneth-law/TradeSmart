const BASE = import.meta.env.DEV ? '' : ''

async function get<T>(path: string): Promise<T> {
  const r = await fetch(BASE + path)
  if (!r.ok) {
    const text = await r.text()
    throw new Error(`${r.status} ${r.statusText}: ${text}`)
  }
  return r.json()
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) {
    const text = await r.text()
    throw new Error(`${r.status} ${r.statusText}: ${text}`)
  }
  return r.json()
}

export const api = {
  getAnalysisResults: (sessionId: string) =>
    get<import('../types').AnalysisResult>(`/api/analysis_results/${sessionId}`),

  getStock: (ticker: string) =>
    get<import('../types').StockResult>(`/api/stock/${ticker}`),

  getPriceHistory: (ticker: string, days?: number) =>
    get<import('../types').PriceHistory>(`/api/stock_price_history/${ticker}${days ? `?days=${days}` : ''}`),

  getIndustryPeers: (ticker: string) =>
    get<Array<{ ticker: string; day_trading_score: number }>>(`/api/industry_peers/${ticker}`),

  getFinancials: (ticker: string, period: 'annual' | 'quarterly') =>
    get<import('../types').FinancialsResponse>(`/api/financials/${ticker}?period=${period}`),

  getMarketOverview: () =>
    get<import('../types').MarketOverview>('/api/market_overview'),

  getPortfolio: () =>
    get<{ positions: import('../types').PortfolioPosition[]; summary: import('../types').PortfolioSummary }>(
      '/api/portfolio'
    ),

  getLiveSnapshot: (symbols: string[]) =>
    get<import('../types').LiveMarketPayload>(`/api/live/snapshot?symbols=${encodeURIComponent(symbols.join(','))}`),

  runBacktest: (params: {
    tickers: string[]
    strategy: string
    start_date?: string
    end_date?: string
    days?: number
    custom_transaction_cost?: number
    transaction_cost_type?: 'fixed' | 'percent' | 'per_share'
  }) => post<{ backtest_id: string }>('/api/run_backtest', params),

  getBacktestResults: (backtestId: string) =>
    get<import('../types').BacktestResult>(`/api/backtest_results/${backtestId}`),

  runIntegrated: (params: {
    tickers: string[]
    use_ml: boolean
    execute_trades: boolean
  }) => post<{ system_id: string }>('/api/run_integrated', params),

  getIntegratedResults: (systemId: string) =>
    get<{ signals: import('../types').StockResult[]; portfolio_summary: import('../types').PortfolioSummary; tickers: string[] }>(
      `/api/integrated_results/${systemId}`
    ),

  getDocumentation: (docType: string = 'readme') =>
    get<{ title: string; html_content: string }>(`/api/documentation/${docType}`),

  healthcheck: () =>
    get<{ status: string; timestamp: string }>('/healthcheck'),
}
