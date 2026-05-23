export interface DetailedMetrics {
  technical?: {
    rsi7?: number
    rsi14?: number
    macd_trend?: string
    bb_position?: number
    above_ma5?: boolean
    above_ma10?: boolean
    above_ma20?: boolean
  }
  volatility?: {
    atr?: number
    atr_pct?: number
    avg_intraday_range?: number
    gap_ups_5d?: number
    gap_downs_5d?: number
  }
  momentum?: {
    return_1d?: number
    return_3d?: number
    return_5d?: number
  }
  volume?: {
    volume_ratio?: number
  }
  sentiment?: {
    news_sentiment_score?: number
    news_sentiment_label?: string
  }
}

export interface StockResult {
  ticker: string
  company_name: string
  sector: string
  industry: string
  day_trading_score: number
  day_trading_strategy: 'Strong Buy' | 'Buy' | 'Neutral/Watch' | 'Sell' | 'Strong Sell'
  current_price: number
  return_1d: number
  rsi7: number
  atr_pct: number
  volume_ratio: number
  macd_trend: 'bullish' | 'bearish' | 'neutral'
  news_sentiment_label: string
  news_sentiment_score: number
  above_ma5: boolean
  above_ma20: boolean
  metrics?: DetailedMetrics
  strategy_details?: string
}

export interface AnalysisResult {
  ranked_stocks: StockResult[]
  failed_tickers: string[]
  session_id: string
  csv_url?: string
}

export interface SSEMessage {
  progress?: number
  message?: string
  status?: 'initializing' | 'processing' | 'saving' | 'generating_charts' | 'preparing_results' | 'complete' | 'error'
  error?: string
  session_id?: string
  system_id?: string
  backtest_id?: string
  redirect_url?: string
}

export interface BacktestMetrics {
  total_return: number
  annualized_return?: number
  sharpe_ratio: number
  max_drawdown: number
  win_rate: number
  num_trades: number
  profit_factor?: number
  initial_capital?: number
  final_capital?: number
  total_transaction_costs?: number
  transaction_cost_percentage?: number
  total_trade_value?: number
  avg_trade_value?: number
  buy_count?: number
  sell_count?: number
  avg_daily_return?: number
  daily_volatility?: number
  annualized_volatility?: number
  best_day?: number
  worst_day?: number
}

export interface Trade {
  date: string
  type: 'BUY' | 'SELL'
  ticker: string
  price: number
  shares?: number
  return_pct?: number
  cost?: number
  value?: number
  net_value?: number
  pnl?: number | null
}

export interface RoundTrip {
  ticker: string
  entry_date: string
  exit_date: string
  entry_price: number
  exit_price: number
  shares: number
  pnl: number
  return_pct: number
}

export interface PositionRoundTrip {
  ticker?: string
  entry_date?: string
  last_exit_date?: string
  entry_price?: number
  shares_sold?: number
  exit_count?: number
  pnl?: number
  gross_entry_value?: number
  return_pct?: number | null
}

export interface BacktestBenchmark {
  cash_return?: number
  equal_weight_return?: number
  strategy_vs_equal_weight?: number
  strategy_vs_cash?: number
  strategy_vs_technical?: number
  coverage?: number
  technical_rule?: { return?: number; sharpe_ratio?: number; max_drawdown?: number; num_trades?: number }
  best_ticker?: { ticker?: string; return_pct?: number }
  worst_ticker?: { ticker?: string; return_pct?: number }
  ticker_returns?: Array<{ ticker: string; start_price?: number; end_price?: number; return_pct: number }>
}

export interface AlphaVerdict {
  label: string
  color?: 'up' | 'down' | 'accent' | 'warn' | 'muted' | 'text'
  score?: number
  summary?: string
  warnings?: string[]
}

export interface TrainingPreviewRow {
  ticker?: string
  feature_date?: string
  label_date?: string
  prediction_horizon_days?: number
  future_return_pct?: number
  future_signal?: number
  features?: Record<string, unknown>
}

export interface BacktestTrainingContext {
  training_context?: string
  model_trained?: boolean
  sample_count?: number
  min_feature_date?: string
  max_feature_date?: string
  min_label_date?: string
  max_label_date?: string
  training_end_date_exclusive?: string
  prediction_horizon_days?: number
  positive_samples?: number
  negative_samples?: number
  label_balance?: Record<string, number>
  feature_importance?: Record<string, number>
  preview_rows?: TrainingPreviewRow[]
}

export interface BacktestResult {
  metrics: BacktestMetrics
  trades: Trade[]
  equity_curve?: Array<{ date: string; value: number }>
  drawdown_curve?: Array<{ date: string; drawdown: number }>
  daily_returns?: number[]
  round_trips?: RoundTrip[]
  position_diagnostics?: Record<string, unknown>
  benchmarks?: BacktestBenchmark
  concentration?: Record<string, unknown>
  alpha_verdict?: AlphaVerdict
  risk_summary?: Record<string, number>
  training_context?: BacktestTrainingContext
  run_metadata?: Record<string, unknown>
  tickers: string[]
  strategy: string
  start_date: string
  end_date: string
}

export interface PortfolioPosition {
  ticker: string
  shares: number
  cost_basis: number
  current_price: number
  market_value: number
  unrealized_pnl: number
  unrealized_pnl_pct: number
  sector?: string
  entry_date?: string
}

export interface PortfolioSummary {
  total_value: number
  cash: number
  invested: number
  total_pnl: number
  total_pnl_pct: number
  num_positions: number
  portfolio_beta?: number
  max_drawdown?: number
}

export interface SectorData {
  name: string
  return_1d: number
  return_1w?: number
  return_1m?: number
  trend?: string
}

export interface MarketOverview {
  sectors: SectorData[]
  market_trend: string
  advances?: number
  declines?: number
  market_health?: string
}

export interface PriceHistory {
  ticker?: string
  dates: Array<string | number>
  open: number[]
  high: number[]
  low: number[]
  close: number[]
  prices?: number[]
  volume?: number[]
  volumes?: number[]
  ma5?: number[]
  ma20?: number[]
  interval?: string
  source?: string
  live_granularity?: string
  stats?: { last: number; high: number; low: number; avg: number }
}

export type TickerContext = {
  ticker: string
  price: number
  change: number
  score: number
}

export interface LiveQuote {
  symbol: string
  price?: number | null
  bid?: number | null
  ask?: number | null
  last?: number | null
  bid_size?: number | null
  ask_size?: number | null
  last_size?: number | null
  source?: string
  feed?: string
  timestamp?: string
  received_at?: string
}

export interface LiveMarketPayload {
  provider: string
  configured: boolean
  connected: boolean
  authenticated: boolean
  feed: string
  missing?: string[]
  error?: string | null
  quotes: Record<string, LiveQuote | null>
}

export interface LiveMarketStatus {
  provider: string
  configured: boolean
  connected: boolean
  authenticated: boolean
  feed: string
  missing?: string[]
  error?: string | null
}

export type MarketDataProvider = 'alpaca' | 'delayed' | 'none'
export type MarketDataState = 'live' | 'delayed' | 'offline'

export interface YFinanceStatus {
  provider: 'yfinance'
  reachable: boolean
  delayed: boolean
  latency_ms?: number | null
  checked_at?: string
  error?: string | null
}

export interface ClientStreamStatus {
  provider: 'openai'
  status: string
  configured: boolean | null
  checked_at?: string
}

export interface BackendStreamStatus {
  provider: 'tradesmart_backend'
  reachable: boolean
  status: string
  checked_at?: string
  error?: string | null
}

export interface MarketDataStatus {
  provider: MarketDataProvider
  label: string
  state: MarketDataState
  timestamp: string
  streams: {
    alpaca: LiveMarketStatus
    yfinance: YFinanceStatus
    openai: ClientStreamStatus
    tradesmart_backend: BackendStreamStatus
  }
}

export interface ModelArtifactStatus {
  exists: boolean
  size_bytes: number
  modified_at?: string | null
}

export interface TrainedModelRecord {
  id: string
  name: string
  scope: 'global' | 'ticker'
  ticker?: string | null
  model_type: string
  trained_at?: string | null
  sample_count?: number | null
  feature_count: number
  feature_names: string[]
  ready: boolean
  metadata: Record<string, unknown>
  artifacts: Record<string, ModelArtifactStatus>
}

export interface TrainedModelsResponse {
  models: TrainedModelRecord[]
  count: number
  storage: {
    path: string
    exists: boolean
    backend_only?: boolean
  }
}

export interface FinancialsStatement {
  periods: string[]
  rows: Record<string, Record<string, number | null>>
}

export type FinancialsPeriod = 'annual' | 'quarterly'

export interface FinancialsResponse {
  ticker: string
  period: FinancialsPeriod
  income_statement: FinancialsStatement
  balance_sheet: FinancialsStatement
  cash_flow: FinancialsStatement
}
