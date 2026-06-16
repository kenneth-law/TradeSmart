# System Architecture

TradeSmart is structured as a private research terminal: React owns the user experience, Flask owns analytics and data APIs, and the Python modules contain the research logic.

## Platform Flow

<div class="doc-flowchart">
<svg viewBox="0 0 980 360" role="img" aria-label="TradeSmart platform architecture flowchart">
  <defs>
    <marker id="arrow-platform" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="currentColor" />
    </marker>
  </defs>
  <g class="flow-lane">
    <rect x="24" y="36" width="190" height="80" rx="6" />
    <text x="119" y="68">React Terminal</text>
    <text x="119" y="91">routes, shell, charts</text>
  </g>
  <g class="flow-lane">
    <rect x="280" y="36" width="190" height="80" rx="6" />
    <text x="375" y="68">Flask API</text>
    <text x="375" y="91">JSON + SSE endpoints</text>
  </g>
  <g class="flow-lane">
    <rect x="536" y="36" width="190" height="80" rx="6" />
    <text x="631" y="68">Analytics Modules</text>
    <text x="631" y="91">signals, ML, risk</text>
  </g>
  <g class="flow-lane">
    <rect x="792" y="36" width="164" height="80" rx="6" />
    <text x="874" y="68">Data Providers</text>
    <text x="874" y="91">Yahoo, Alpaca, OpenAI</text>
  </g>
  <path class="flow-edge" d="M214 76 H280" marker-end="url(#arrow-platform)" />
  <path class="flow-edge" d="M470 76 H536" marker-end="url(#arrow-platform)" />
  <path class="flow-edge" d="M726 76 H792" marker-end="url(#arrow-platform)" />
  <path class="flow-edge muted" d="M792 104 C660 180, 370 180, 214 104" marker-end="url(#arrow-platform)" />

  <g class="flow-lane">
    <rect x="96" y="224" width="170" height="74" rx="6" />
    <text x="181" y="254">Research Views</text>
    <text x="181" y="276">market, stock, docs</text>
  </g>
  <g class="flow-lane">
    <rect x="314" y="224" width="170" height="74" rx="6" />
    <text x="399" y="254">Backtest Engine</text>
    <text x="399" y="276">costs + diagnostics</text>
  </g>
  <g class="flow-lane">
    <rect x="532" y="224" width="170" height="74" rx="6" />
    <text x="617" y="254">Model Artifacts</text>
    <text x="617" y="276">GBDT + metadata</text>
  </g>
  <g class="flow-lane">
    <rect x="750" y="224" width="170" height="74" rx="6" />
    <text x="835" y="254">Portfolio State</text>
    <text x="835" y="276">positions + risk</text>
  </g>
  <path class="flow-edge" d="M375 116 C330 158, 242 180, 181 224" marker-end="url(#arrow-platform)" />
  <path class="flow-edge" d="M430 116 C428 158, 410 180, 399 224" marker-end="url(#arrow-platform)" />
  <path class="flow-edge" d="M586 116 C590 158, 606 180, 617 224" marker-end="url(#arrow-platform)" />
  <path class="flow-edge" d="M664 116 C714 158, 795 180, 835 224" marker-end="url(#arrow-platform)" />
</svg>
</div>

## Request Lifecycle

<div class="doc-flowchart">
<svg viewBox="0 0 980 300" role="img" aria-label="TradeSmart request lifecycle">
  <defs>
    <marker id="arrow-request" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="currentColor" />
    </marker>
  </defs>
  <g class="flow-step"><circle cx="92" cy="84" r="38" /><text x="92" y="80">1</text><text x="92" y="103">User Action</text></g>
  <g class="flow-step"><circle cx="252" cy="84" r="38" /><text x="252" y="80">2</text><text x="252" y="103">React Hook</text></g>
  <g class="flow-step"><circle cx="412" cy="84" r="38" /><text x="412" y="80">3</text><text x="412" y="103">API Route</text></g>
  <g class="flow-step"><circle cx="572" cy="84" r="38" /><text x="572" y="80">4</text><text x="572" y="103">Module Logic</text></g>
  <g class="flow-step"><circle cx="732" cy="84" r="38" /><text x="732" y="80">5</text><text x="732" y="103">Data + Models</text></g>
  <g class="flow-step"><circle cx="892" cy="84" r="38" /><text x="892" y="80">6</text><text x="892" y="103">UI Render</text></g>
  <path class="flow-edge" d="M130 84 H214" marker-end="url(#arrow-request)" />
  <path class="flow-edge" d="M290 84 H374" marker-end="url(#arrow-request)" />
  <path class="flow-edge" d="M450 84 H534" marker-end="url(#arrow-request)" />
  <path class="flow-edge" d="M610 84 H694" marker-end="url(#arrow-request)" />
  <path class="flow-edge" d="M770 84 H854" marker-end="url(#arrow-request)" />

  <rect class="flow-note" x="80" y="188" width="820" height="64" rx="6" />
  <text class="flow-note-text" x="490" y="214">Long-running jobs use server-sent events for progress updates.</text>
  <text class="flow-note-text" x="490" y="238">Backtest and integrated workflows stream logs before final result retrieval.</text>
</svg>
</div>

## Runtime Components

| Layer | Responsibility | Primary Files |
| --- | --- | --- |
| React terminal | Navigation, screens, charts, onboarding, local preferences | `frontend/src/App.tsx`, `frontend/src/pages`, `frontend/src/components` |
| API boundary | JSON endpoints, SSE streams, SPA serving | `app.py`, `wsgi.py` |
| Market data | Price history, market overview, provider status | `modules/data_retrieval.py`, `modules/market_data.py`, `modules/alpaca_live_data.py` |
| Research logic | Technical indicators, signal scoring, model metadata | `modules/technical_analysis.py`, `modules/ml_scoring.py`, `modules/trading_system.py` |
| Validation | Backtests, cost modeling, benchmarks, diagnostics | `modules/backtesting.py` |
| Portfolio layer | Position state, allocation context, risk controls | `modules/portfolio_management.py` |
| Deployment | Frontend build and Gunicorn runtime | `railpack.json`, `nixpacks.toml`, `Procfile` |

## Deployment Flow

1. Railway/Railpack or Nixpacks installs Python and Node dependencies.
2. `frontend/` is built with Vite.
3. The compiled app is emitted to `static/dist`.
4. Gunicorn starts `wsgi:app`.
5. Flask serves API routes and falls back to the React app for terminal routes.
