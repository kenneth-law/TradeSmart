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
}

export interface Trade {
  date: string
  type: 'BUY' | 'SELL'
  ticker: string
  price: number
  shares?: number
  return_pct?: number
}

export interface BacktestResult {
  metrics: BacktestMetrics
  trades: Trade[]
  equity_curve?: Array<{ date: string; value: number }>
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
  dates: string[]
  open: number[]
  high: number[]
  low: number[]
  close: number[]
  prices?: number[]
  volume?: number[]
  volumes?: number[]
  ma5?: number[]
  ma20?: number[]
  stats?: { last: number; high: number; low: number; avg: number }
}

export type TickerContext = {
  ticker: string
  price: number
  change: number
  score: number
}
