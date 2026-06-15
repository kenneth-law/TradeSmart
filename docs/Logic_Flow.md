# Logic Flow

The integrated workflow combines ticker research, signal scoring, model context, backtesting, and portfolio analytics into a single research pass.

## End-to-End Workflow

<div class="doc-flowchart">
<svg viewBox="0 0 980 420" role="img" aria-label="Integrated system logic flow">
  <defs>
    <marker id="arrow-logic" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="currentColor" />
    </marker>
  </defs>
  <g class="flow-lane"><rect x="50" y="36" width="160" height="68" rx="6" /><text x="130" y="66">Ticker Set</text><text x="130" y="87">research universe</text></g>
  <g class="flow-lane"><rect x="280" y="36" width="160" height="68" rx="6" /><text x="360" y="66">Data Retrieval</text><text x="360" y="87">price + context</text></g>
  <g class="flow-lane"><rect x="510" y="36" width="160" height="68" rx="6" /><text x="590" y="66">Feature Build</text><text x="590" y="87">technical + risk</text></g>
  <g class="flow-lane"><rect x="740" y="36" width="160" height="68" rx="6" /><text x="820" y="66">Signal Score</text><text x="820" y="87">heuristic + ML</text></g>
  <path class="flow-edge" d="M210 70 H280" marker-end="url(#arrow-logic)" />
  <path class="flow-edge" d="M440 70 H510" marker-end="url(#arrow-logic)" />
  <path class="flow-edge" d="M670 70 H740" marker-end="url(#arrow-logic)" />

  <g class="flow-lane"><rect x="164" y="176" width="170" height="72" rx="6" /><text x="249" y="206">Ranked Signals</text><text x="249" y="228">watchlist candidates</text></g>
  <g class="flow-lane"><rect x="405" y="176" width="170" height="72" rx="6" /><text x="490" y="206">Portfolio Review</text><text x="490" y="228">sizing + exposure</text></g>
  <g class="flow-lane"><rect x="646" y="176" width="170" height="72" rx="6" /><text x="731" y="206">Backtest</text><text x="731" y="228">costs + benchmark</text></g>
  <path class="flow-edge" d="M820 104 C760 146, 626 157, 490 176" marker-end="url(#arrow-logic)" />
  <path class="flow-edge" d="M820 104 C760 142, 735 154, 731 176" marker-end="url(#arrow-logic)" />
  <path class="flow-edge" d="M820 104 C650 146, 400 152, 249 176" marker-end="url(#arrow-logic)" />

  <g class="flow-lane"><rect x="202" y="318" width="190" height="72" rx="6" /><text x="297" y="348">Diagnostics</text><text x="297" y="370">returns, drawdown, risk</text></g>
  <g class="flow-lane"><rect x="588" y="318" width="190" height="72" rx="6" /><text x="683" y="348">Research Output</text><text x="683" y="370">decision context</text></g>
  <path class="flow-edge" d="M249 248 C260 286, 276 300, 297 318" marker-end="url(#arrow-logic)" />
  <path class="flow-edge" d="M490 248 C452 286, 370 300, 297 318" marker-end="url(#arrow-logic)" />
  <path class="flow-edge" d="M731 248 C690 286, 620 300, 683 318" marker-end="url(#arrow-logic)" />
  <path class="flow-edge" d="M392 354 H588" marker-end="url(#arrow-logic)" />
</svg>
</div>

## 1. User Input

The user selects tickers, model options, backtest settings, and portfolio assumptions from the React terminal. Long-running workflows return an ID immediately and stream progress through server-sent events.

## 2. Data Retrieval

The backend retrieves market data, ticker metadata, historical bars, financial context, and provider status. Yahoo Finance is the fallback data source; Alpaca can provide live-data context where credentials are configured.

## 3. Feature Engineering

The technical analysis layer calculates trend, momentum, volatility, volume, return, and risk features. These features are used by both heuristic scoring and model-assisted research.

## 4. Model Context

Where enabled, the GBDT scorer loads trained artifacts, feature metadata, and preprocessing components. Model outputs are treated as research signals and compared with traditional indicator-based scores.

## 5. Signal Ranking

Tickers are ranked into research candidates using score, volatility, technical context, and model output. The result is a watchlist-like research set rather than a guarantee or recommendation.

## 6. Portfolio Review

Portfolio logic checks sizing, concentration, exposure, cash allocation, and risk limits. This step helps answer whether a signal is usable inside a capital allocation framework.

## 7. Backtest and Diagnostics

The backtest engine simulates the selected rules over the chosen historical window. It reports returns, drawdown, Sharpe/Sortino ratios, trade count, benchmarks, and position-level diagnostics.

## 8. Research Output

The terminal presents the combined output as a research artifact: ranked signals, charts, model status, portfolio context, and backtest evidence.
