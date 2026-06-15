# TradeSmart — Claude Code Guide

## Project Overview

TradeSmart is a Flask web application for stock analysis, day-trading opportunity detection, and algorithmic backtesting. It integrates technical analysis, ML scoring, news sentiment, and portfolio management into a unified trading system.

## Architecture

```
app.py              — Flask web server, SSE endpoints, route handlers
main.py             — CLI entry point for traditional (non-web) analysis
wsgi.py             — WSGI entry point (production)
modules/
  trading_system.py — Integrated TradingSystem orchestrator
  technical_analysis.py — Indicators, scoring, stock data fetching
  analysis_reporting.py — Ranking, CSV export, watchlist generation
  backtesting.py    — Strategy backtesting engine
  ml_scoring.py     — ML-based stock scoring
  news_sentiment.py — News fetching and sentiment analysis
  data_retrieval.py — yfinance wrappers, session/cookie management
  market_data.py    — Sector performance, market breadth, intraday data
  visualization.py  — Chart data preparation (Plotly-ready)
  portfolio_management.py — Portfolio tracking and management
  execution.py      — Trade execution logic
  utils.py          — Logging, message handler
templates/          — Jinja2 HTML templates
static/             — CSS, JS, assets
```

## Running the App

```bash
# Development server
python app.py

# Or via Flask CLI
flask run

# CLI analysis (no web server)
python main.py
```

The Flask app runs on `http://localhost:5000` by default.

## Key Patterns

- **SSE Progress Streaming**: Long-running tasks (analysis, backtesting) use Server-Sent Events. Each task gets a `Queue`, a background `Thread`, and a `/..._progress` SSE endpoint. Signal completion by putting `None` in the queue.
- **yfinance session**: A single `yf_session` is created at startup via `get_yf_session()` and reused across requests. The cookie patch (`yfinance_cookie_patch.py`) is applied at import time.
- **TradingSystem**: Central orchestrator in `modules/trading_system.py`. Instantiate once and reuse; it holds references to all sub-components.
- **Scoring**: `technical_analysis.py` computes a `day_trading_score` (0–100). `ml_scoring.py` adds an ML layer on top.

## Custom Commands

| Command | Purpose |
|---|---|
| `/analyze` | Analyze a list of tickers |
| `/backtest` | Run a backtest on a strategy |
| `/add-module` | Scaffold a new module |
| `/run-server` | Start the dev server |
| `/debug-sse` | Debug SSE endpoint issues |

## Testing

No test suite currently. When adding tests, use `pytest` and place them in `tests/`. Integration tests should hit real yfinance data or recorded cassettes — do not mock yfinance responses.

## Dependencies

Key packages: Flask, pandas, numpy, yfinance, curl_cffi, plotly, scikit-learn, beautifulsoup4, markdown. See `requirements.txt` for pinned versions.

## Deployment

`Procfile` is present for Heroku/Railway-style deployment. `wsgi.py` exposes the `app` object for gunicorn.

## Notes

- `cache/` directory stores intermediate data — safe to delete, will be rebuilt.
- `static/` is in `.gitignore` (built assets only).
- `yfinance_cookie_patch.py` patches `yfinance.data` at import time to avoid rate-limit errors.
