# TradeSmart Analytics

TradeSmart Analytics is a private investment research terminal for systematic equity strategy development, market analysis, backtesting, and portfolio review. It brings together a React research workspace, a Flask analytics API, market data pipelines, Gradient Boosted Decision Tree models, risk analytics, and strategy simulation tools into one platform for exploring equity ideas across technical, fundamental, sentiment, and portfolio contexts.

The project is designed as a personal quantitative research environment rather than a single-purpose stock screener. It supports ticker-level research, strategy scoring, model-driven signal generation, historical performance simulation, portfolio diagnostics, and market status monitoring, with an emphasis on repeatable research workflows and realistic assumptions.

<img width="1728" height="1117" alt="TradeSmart Analytics private research terminal" src="https://github.com/user-attachments/assets/5e62e969-008a-4349-ad9c-30eb7bea8081" />

[System Architecture](https://tradevision.up.railway.app/documentation/system_architecture)

<img width="100%" alt="TradeSmart system architecture" src="https://github.com/user-attachments/assets/e712bc85-26ab-440d-8da8-5ccf1cd5113e" />

## Overview

TradeSmart Analytics is built around the research process used to develop and evaluate systematic equity strategies:

- **Market research terminal**: Search and inspect equities, review price history, fundamentals, peers, market context, watchlists, and recent research activity.
- **Quantitative signal development**: Combine technical indicators, volatility measures, returns, volume behavior, sentiment features, and model outputs into explainable stock-level signals.
- **Machine learning scoring**: Use Gradient Boosted Decision Tree models to identify non-linear relationships across engineered market features and produce regression or classification outputs.
- **Backtesting and strategy validation**: Simulate strategy ideas across historical periods with transaction costs, drawdown analysis, benchmarks, position-level diagnostics, and walk-forward style evaluation.
- **Portfolio analytics**: Review position sizing, concentration, sector exposure, risk metrics, Sharpe/Sortino ratios, VaR-style measures, drawdowns, and portfolio-level context.
- **Execution research**: Model spread, fees, market impact, and participation logic for strategy capacity and implementation-cost analysis.

<img width="1728" height="1117" alt="TradeSmart ticker research page" src="https://github.com/user-attachments/assets/44869f48-bd98-40d7-94ab-9ed53d828e03" />
<img width="1728" height="1117" alt="TradeSmart analytics workspace" src="https://github.com/user-attachments/assets/007ab30e-f6b3-4431-b35f-765c9faf72aa" />

## Core Modules

### Research Terminal

- Search any supported ticker and inspect live market data, charts, financials, peer context, and system-generated signals.
- Maintain a private watchlist and recent-search workflow for ongoing research.
- Monitor backend, market data, model, and live data status from the application shell.

<img width="1728" height="1117" alt="Screenshot 2026-05-21 at 17 55 06" src="https://github.com/user-attachments/assets/2cba66a6-767c-4b83-9ac7-c7c804e25622" />
<img width="1728" height="1117" alt="TradeSmart market research dashboard" src="https://github.com/user-attachments/assets/150dbbac-6aa7-4daa-9f60-3be425d91a35" />


### Signal and Scoring Engine

- Calculates technical, volatility, return, trend, volume, and sentiment features.
- Produces stock-level scores using both heuristic scoring and machine learning models.
- Supports explainable model output through feature importance and stored model metadata.
- Separates training and inference paths to reduce leakage risk in model evaluation.

### Machine Learning Research

The machine learning layer uses Gradient Boosted Decision Trees for alpha signal research and model-based scoring.

- **Model types**: Regression scoring and classification-style signal generation.
- **Feature engineering**: Technical indicators, volatility features, trend metrics, return windows, volume behavior, and derived market features.
- **Preprocessing**: Robust scaling, feature selection, PCA support, and scikit-learn pipelines.
- **Validation**: Time-series-aware train/test separation with expanding-window style evaluation.
- **Model artifacts**: Stored model files, scalers, selected features, PCA objects, metadata, and trained-model status endpoints.
- **Adaptability**: Global and per-ticker model support for comparing broad-market and instrument-specific behavior.

<img width="1728" height="1117" alt="TradeSmart model and signal analytics" src="https://github.com/user-attachments/assets/8620e644-86e5-4d33-a59b-dea40898494d" />

### Backtesting Engine

The backtesting system is designed for strategy research rather than simple historical chart replay.

- Simulates strategy behavior over historical windows.
- Models transaction costs, spreads, market impact, and fees.
- Tracks equity curves, drawdowns, trades, win rate, Sharpe ratio, Sortino ratio, and alpha verdicts.
- Supports passive benchmark comparison over the same test window.
- Streams progress to the frontend through server-sent events.
- Supports ML-driven strategies, technical baselines, and custom strategy functions.

<img width="1728" height="1117" alt="TradeSmart backtesting configuration" src="https://github.com/user-attachments/assets/a339949e-400d-4e28-8552-b56108ed7dfc" />
<img width="1728" height="1117" alt="TradeSmart backtesting results" src="https://github.com/user-attachments/assets/eb64484f-5360-4e13-821a-c32159fcd729" />
<img width="1728" height="1117" alt="TradeSmart strategy diagnostics" src="https://github.com/user-attachments/assets/ee9d3167-4a72-4b6d-bd85-fb492246e2ea" />

### Portfolio and Risk Analytics

- Position sizing based on expected edge, volatility, and risk constraints.
- Sector and single-name concentration checks.
- Drawdown and risk monitoring.
- Sharpe, Sortino, win-rate, and return analysis.
- Research-oriented portfolio context for comparing strategy outputs against capital allocation constraints.

<img width="3444" height="1806" alt="image" src="https://github.com/user-attachments/assets/f7fd92b5-c76e-435b-9bd5-205ae946b7f9" />

### Market Data and Integrations

- Yahoo Finance data retrieval with caching and cookie/session handling.
- Optional Alpaca live market data status and snapshot support.
- OpenAI-supported research and sentiment workflows where configured.
- API endpoints for stock history, financials, peers, market overview, trained models, portfolio data, backtests, and documentation.

<img width="357" height="229" alt="Screenshot 2026-06-16 at 00 50 31" src="https://github.com/user-attachments/assets/81543a0d-40d9-4086-8cf9-d2f8c6da1cbc" />


## Technical Indicators and Features

TradeSmart uses a broad feature set for research and model development:

- **Momentum and trend**: RSI, MACD, stochastic oscillator, moving averages, 30/90-day returns, golden/death cross context.
- **Volatility and range**: ATR, Bollinger Bands, intraday range, historical volatility, drawdown behavior.
- **Volume and liquidity**: Relative volume, on-balance volume, spread and liquidity proxies.
- **Fundamental context**: Financial statement data and ticker-level company information where available.
- **Sentiment and catalyst context**: News sentiment analysis and qualitative context where configured.
- **Portfolio context**: Risk, concentration, exposure, and allocation diagnostics.

## Architecture

TradeSmart is split into a modern React frontend and a Python analytics backend.

- **Frontend**: React, TypeScript, Vite, Tailwind CSS, React Router, React Query, Zustand, Recharts, and lightweight-charts.
- **Backend**: Flask API served by Gunicorn in production.
- **Analytics modules**: Python modules for market data, technical analysis, ML scoring, backtesting, portfolio management, execution research, reporting, and visualization.
- **Deployment**: Railway-compatible configuration with frontend build output served from `static/dist`.

## Project Structure

```text
.
├── app.py                         # Flask API and SPA serving entry point
├── wsgi.py                        # Gunicorn production entry point
├── modules/
│   ├── trading_system.py          # Integrated research and strategy workflow
│   ├── ml_scoring.py              # GBDT scoring and model pipeline
│   ├── backtesting.py             # Historical strategy simulation
│   ├── portfolio_management.py    # Portfolio construction and risk controls
│   ├── execution.py               # Cost and execution research utilities
│   ├── market_data.py             # Market overview data
│   ├── data_retrieval.py          # Price/history retrieval and caching
│   └── technical_analysis.py      # Indicator calculations
├── frontend/
│   ├── src/pages/                 # React terminal pages
│   ├── src/components/            # Layout, charts, data, onboarding, strategy UI
│   ├── src/hooks/                 # API, SSE, live quotes, and analysis hooks
│   └── vite.config.ts             # Builds frontend into static/dist
├── docs/                          # Documentation surfaced in the app
├── templates/                     # Legacy Flask templates and documentation views
├── requirements.txt               # Python dependencies
├── railpack.json                  # Railway/Railpack deployment build config
├── nixpacks.toml                  # Nixpacks deployment build config
└── Procfile                       # Gunicorn startup command
```

## Local Development

### Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

By default, the Flask backend runs on:

```text
http://localhost:8080
```

To use the Vite development proxy, run the backend on port `5001`:

```bash
PORT=5001 python app.py
```

### Frontend

```bash
cd frontend
npm ci
npm run dev
```

The Vite development server runs on:

```text
http://localhost:5173
```

### Production-style Build

```bash
cd frontend
npm ci
npm run build
cd ..
python app.py
```

The frontend build is written to `static/dist`, which the Flask app serves for production-style local testing.

## Environment Variables

- `PORT`: Port used by Flask/Gunicorn.
- `FLASK_ENV`: Set to `production` for production behavior.
- `OPENAI_API_KEY`: Optional, used by sentiment and AI-assisted research workflows where enabled.
- Alpaca credentials may be configured for live market data features if supported by the local environment.

## Deployment

The project is configured for Railway-style deployment.

- `railpack.json` installs Node 20 and builds the React frontend.
- `nixpacks.toml` installs Python and Node dependencies, then builds the frontend.
- `Procfile` starts the app through Gunicorn:

```bash
gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 2 --worker-class gthread --threads 4 --timeout 120
```

The deployed Flask service serves the compiled React app from `static/dist` and exposes the API endpoints used by the terminal.

## Disclaimer

This software is provided for educational, research, and informational purposes only. The author is not a registered investment adviser and this project does not provide financial advice, investment recommendations, or guarantees of market performance.

Market data, model outputs, scores, backtests, and portfolio analytics may be incomplete, delayed, inaccurate, or unsuitable for real investment decisions. Backtested performance is hypothetical and does not guarantee future results. Use this software at your own risk and consult a qualified financial professional before making investment decisions.

This project is not affiliated with the ASX, Yahoo Finance, Alpaca, OpenAI, Railway, or any financial institution.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](https://www.apache.org/licenses/LICENSE-2.0.txt) for details.

## Acknowledgements

- Yahoo Finance for market data access.
- OpenAI for AI and language-model capabilities where configured.
- The Python, Flask, React, Vite, and scikit-learn ecosystems that power the research stack.
