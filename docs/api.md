# API Reference

The Flask backend exposes JSON endpoints for the React terminal. In development, Vite proxies `/api`, `/analysis_progress`, `/backtest_progress_stream`, `/integrated_progress_stream`, and `/healthcheck` to the Flask server.

## Market and Research

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/market_overview` | GET | Sector and market overview payload |
| `/api/stock/<ticker>` | GET | Stock-level analysis and signal data |
| `/api/stock_price_history/<ticker>?days=30` | GET | Chart-ready price history |
| `/api/industry_peers/<ticker>` | GET | Peer comparison data |
| `/api/financials/<ticker>?period=annual` | GET | Annual or quarterly financial statements |
| `/api/portfolio` | GET | Portfolio positions and summary |

## Live Data and Status

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/market_data/status` | GET | Combined provider status for UI indicators |
| `/api/live/status` | GET | Alpaca live-data connection/configuration status |
| `/api/live/snapshot?symbols=AAPL,MSFT` | GET | Live or fallback quote snapshot |
| `/api/live/stream` | GET | Streaming quote endpoint where configured |

## Models and Backtesting

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/models/trained` | GET | Trained model artifacts and metadata |
| `/api/run_backtest` | POST | Start a backtest job |
| `/backtest_progress_stream?backtest_id=<id>` | GET | SSE stream for backtest progress |
| `/api/backtest_results/<backtest_id>` | GET | Backtest result payload |
| `/api/run_integrated` | POST | Start integrated strategy workflow |
| `/integrated_progress_stream?system_id=<id>` | GET | SSE stream for integrated workflow progress |
| `/api/integrated_results/<system_id>` | GET | Integrated workflow result payload |

## Documentation

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/documentation` | GET | README documentation |
| `/api/documentation/<type>` | GET | Documentation page HTML |
| `/documentation/<type>` | GET | Legacy redirect to React docs route |

Supported documentation types: `readme`, `architecture`, `logic-flow`, `strategy`, `backtest`, `api`, and `about`.

## Health

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/healthcheck` | GET | Runtime health status |
