# Backtesting

The backtesting system simulates strategy behavior over historical windows and returns diagnostics for performance, risk, and implementation quality.

## Inputs

| Input | Description |
| --- | --- |
| Tickers | Strategy universe or watchlist under test |
| Date range / days | Historical window used for simulation |
| Initial capital | Starting portfolio value |
| Buy threshold | Minimum signal score required to enter or add exposure |
| Sell threshold | Signal level that triggers exits or reductions |
| Transaction cost | Fixed, percent, or per-share cost assumptions |
| Training lookback | Historical data used for model fitting where ML is enabled |
| Position cap | Maximum allocation allowed per position |
| Re-entry controls | Cooldown and discount settings for re-entering after exits |

## Execution Flow

<div class="doc-flowchart">
<svg viewBox="0 0 920 260" role="img" aria-label="Backtesting execution flow">
  <defs>
    <marker id="arrow-backtest" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="currentColor" />
    </marker>
  </defs>
  <g class="flow-lane"><rect x="28" y="56" width="140" height="64" rx="6" /><text x="98" y="85">Configure</text><text x="98" y="105">strategy</text></g>
  <g class="flow-lane"><rect x="206" y="56" width="140" height="64" rx="6" /><text x="276" y="85">Load</text><text x="276" y="105">history</text></g>
  <g class="flow-lane"><rect x="384" y="56" width="140" height="64" rx="6" /><text x="454" y="85">Generate</text><text x="454" y="105">signals</text></g>
  <g class="flow-lane"><rect x="562" y="56" width="140" height="64" rx="6" /><text x="632" y="85">Simulate</text><text x="632" y="105">positions</text></g>
  <g class="flow-lane"><rect x="740" y="56" width="140" height="64" rx="6" /><text x="810" y="85">Report</text><text x="810" y="105">diagnostics</text></g>
  <path class="flow-edge" d="M168 88 H206" marker-end="url(#arrow-backtest)" />
  <path class="flow-edge" d="M346 88 H384" marker-end="url(#arrow-backtest)" />
  <path class="flow-edge" d="M524 88 H562" marker-end="url(#arrow-backtest)" />
  <path class="flow-edge" d="M702 88 H740" marker-end="url(#arrow-backtest)" />
  <rect class="flow-note" x="126" y="174" width="668" height="48" rx="6" />
  <text class="flow-note-text" x="460" y="202">Progress is streamed to the React terminal through server-sent events.</text>
</svg>
</div>

## Metrics

- Total return and annualized return
- Equity curve and drawdown curve
- Maximum drawdown
- Sharpe and Sortino ratios
- Win rate and trade count
- Position-level diagnostics
- Passive benchmark comparison
- Alpha verdict for the selected window

## Interpretation

A good backtest is not just a high return. It should show whether the idea survives costs, how much drawdown it requires, whether the result depends on a few trades, and how it compares to a passive benchmark over the same window.
