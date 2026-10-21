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
  training_lookback_days?: number
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

export type EquityResearchSection = 'profile' | 'estimates' | 'valuation' | 'ownership' | 'research'

export interface EquityResearchResponse {
  ticker: string
  section: EquityResearchSection
  as_of: string
  profile: Record<string, unknown>
  peers?: Array<Record<string, unknown>>
  calendar?: Record<string, unknown> | Array<Record<string, unknown>>
  actions?: Array<Record<string, unknown>>
  dividends?: Array<Record<string, unknown>>
  splits?: Array<Record<string, unknown>>
  sustainability?: Array<Record<string, unknown>> | Record<string, unknown>
  recommendations?: Array<Record<string, unknown>>
  recommendations_summary?: Array<Record<string, unknown>>
  analyst_price_targets?: Record<string, unknown> | Array<Record<string, unknown>>
  earnings_estimate?: Array<Record<string, unknown>>
  revenue_estimate?: Array<Record<string, unknown>>
  earnings_history?: Array<Record<string, unknown>>
  growth_estimates?: Array<Record<string, unknown>>
  earnings_dates?: Array<Record<string, unknown>>
  major_holders?: Array<Record<string, unknown>>
  institutional_holders?: Array<Record<string, unknown>>
  mutualfund_holders?: Array<Record<string, unknown>>
  insider_roster?: Array<Record<string, unknown>>
  insider_transactions?: Array<Record<string, unknown>>
  insider_purchases?: Array<Record<string, unknown>>
  news?: Array<Record<string, unknown>>
  press_releases?: Array<Record<string, unknown>>
  sec_filings?: Array<Record<string, unknown>>
  upgrades_downgrades?: Array<Record<string, unknown>>
}

export type SupplyChainRelationshipKind = 'supplier' | 'customer' | 'competitor'
export type SupplyChainEvidenceClass = 'reported' | 'calculated' | 'estimated' | 'relationship_only'

export interface SupplyChainCompany {
  name: string
  ticker: string | null
  country: string | null
  sector: string | null
  description: string
}

export interface SupplyChainRelationship {
  id: string
  kind: SupplyChainRelationshipKind
  company_name: string
  ticker: string | null
  country: string | null
  sector: string | null
  relationship: string
  exposure_pct: number | null
  exposure_basis: 'revenue' | 'cogs' | 'capex' | 'procurement' | 'none'
  revenue_pct: number | null
  cogs_pct: number | null
  capex_pct: number | null
  evidence_class: SupplyChainEvidenceClass
  confidence: number
  evidence_summary: string
  source_urls: string[]
  last_verified: string
}

export interface SupplyChainRisk {
  title: string
  severity: 'low' | 'medium' | 'high'
  description: string
  affected_relationship_ids: string[]
}

export interface SupplyChainSource {
  url: string
  title: string
  publisher: string
  published_at: string | null
}

export type SupplyChainFacilityType = 'manufacturing' | 'distribution' | 'warehouse' | 'research' | 'office' | 'headquarters' | 'other'

export interface SupplyChainFacility {
  id: string
  company_name: string
  company_ticker: string | null
  facility_name: string
  facility_type: SupplyChainFacilityType
  city: string | null
  region: string | null
  country: string
  latitude: number
  longitude: number
  coordinate_precision: 'exact' | 'city' | 'region'
  products: string[]
  description: string
  confidence: number
  source_urls: string[]
  last_verified: string
}

export type RelatedEntityType = 'ultimate_parent' | 'parent' | 'subsidiary' | 'major_investor' | 'joint_venture' | 'debt_issuer' | 'related_entity' | 'other'
export type RelatedSecurityType = 'debt' | 'equity' | 'fund' | 'joint' | 'none'

export interface SupplyChainRelatedEntity {
  id: string
  parent_id: string | null
  hierarchy_level: number
  entity_name: string
  ticker: string | null
  relationship_type: RelatedEntityType
  security_type: RelatedSecurityType
  ownership_pct: number | null
  country: string | null
  government_owned: boolean
  state_owner_country: string | null
  risk_level: 'low' | 'medium' | 'high' | 'unknown'
  description: string
  confidence: number
  source_urls: string[]
  last_verified: string
}

export interface SupplyChainDocument {
  id: string
  company_name: string
  company_ticker: string | null
  title: string
  document_type: 'annual_report' | 'regulatory_filing' | 'earnings' | 'procurement' | 'contract' | 'presentation' | 'policy' | 'other'
  published_at: string | null
  summary: string
  disruption_relevance: string
  url: string
}

export interface SupplyChainNewsItem {
  id: string
  company_name: string
  company_ticker: string | null
  headline: string
  publisher: string
  published_at: string | null
  event_type: 'supply_disruption' | 'plant_event' | 'labour' | 'logistics' | 'sanctions' | 'geopolitical' | 'weather' | 'cyber' | 'regulatory' | 'other'
  severity: 'low' | 'medium' | 'high'
  summary: string
  location: string | null
  url: string
}

export interface SupplyChainResearch {
  company: SupplyChainCompany
  summary: string
  generated_at: string
  data_quality: 'limited' | 'moderate' | 'strong'
  relationships: SupplyChainRelationship[]
  facilities: SupplyChainFacility[]
  related_entities: SupplyChainRelatedEntity[]
  documents: SupplyChainDocument[]
  news: SupplyChainNewsItem[]
  risks: SupplyChainRisk[]
  sources: SupplyChainSource[]
}
