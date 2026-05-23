from flask import Flask, request, jsonify, send_from_directory, Response
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime, timedelta
import os
import math
from curl_cffi import requests
import re

import time
import threading
import queue
from threading import Thread
from queue import Queue

# Import integrated trading system
from modules.trading_system import TradingSystem

# Import necessary modules
from modules.utils import set_message_handler, log_message
from modules.data_retrieval import get_stock_info, get_stock_history, get_yf_session
from modules.technical_analysis import get_stock_data
from modules.visualization import prepare_price_chart_data, get_detailed_stock_metrics
from modules.market_data import get_sector_performance
from modules.alpaca_live_data import alpaca_live_data

from requests.cookies import create_cookie
import yfinance.data as _data


app = Flask(__name__)


def _sanitize(obj):
    """Recursively convert numpy / pandas types to plain Python for jsonify."""
    try:
        import numpy as _np
        if isinstance(obj, _np.integer):  return int(obj)
        if isinstance(obj, _np.floating):
            value = float(obj)
            return value if math.isfinite(value) else None
        if isinstance(obj, _np.bool_):    return bool(obj)
        if isinstance(obj, _np.ndarray):  return [_sanitize(v) for v in obj.tolist()]
    except ImportError:
        pass
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):  return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_sanitize(v) for v in obj]
    if isinstance(obj, tuple): return [_sanitize(v) for v in obj]
    if hasattr(obj, 'isoformat'): return obj.isoformat()
    if hasattr(obj, 'item'):      return obj.item()   # generic numpy scalar
    return obj


analysis_progress = {}
analysis_logs = {}
analysis_queues = {}

# For progress tracking
backtest_queues = {}
integrated_queues = {}

# Configure session at app startup to avoid repeated session creation
yf_session = get_yf_session()
_market_data_status_cache = {"expires_at": 0, "value": None}


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _probe_yfinance_status():
    """Check Yahoo Finance reachability with a short-lived cache for UI status."""
    now = time.time()
    cached = _market_data_status_cache.get("value")
    if cached and now < _market_data_status_cache.get("expires_at", 0):
        return cached

    started = time.time()
    status = {
        "provider": "yfinance",
        "reachable": False,
        "delayed": True,
        "latency_ms": None,
        "checked_at": datetime.now().isoformat(),
        "error": None,
    }

    try:
        response = yf_session.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/SPY",
            params={"range": "1d", "interval": "1d"},
            timeout=4,
        )
        status["latency_ms"] = round((time.time() - started) * 1000)
        status_code = getattr(response, "status_code", None)
        if not getattr(response, "ok", False):
            status["error"] = f"Yahoo Finance returned {status_code}"
        else:
            payload = response.json()
            result = payload.get("chart", {}).get("result") or []
            status["reachable"] = bool(result)
            if not status["reachable"]:
                status["error"] = "Yahoo Finance returned no chart data"
    except Exception as exc:
        status["latency_ms"] = round((time.time() - started) * 1000)
        status["error"] = str(exc)

    _market_data_status_cache["value"] = status
    _market_data_status_cache["expires_at"] = now + 20
    return status


def _market_data_status():
    alpaca = alpaca_live_data.status()
    yfinance = _probe_yfinance_status()
    backend = {
        "provider": "tradesmart_backend",
        "reachable": True,
        "status": "online",
        "checked_at": datetime.now().isoformat(),
    }

    if alpaca.get("configured") and alpaca.get("connected") and alpaca.get("authenticated"):
        provider = "alpaca"
        label = "market data: alpaca"
        state = "live"
    elif yfinance.get("reachable"):
        provider = "delayed"
        label = "market data: delayed"
        state = "delayed"
    else:
        provider = "none"
        label = "market data: none"
        state = "offline"

    return {
        "provider": provider,
        "label": label,
        "state": state,
        "timestamp": datetime.now().isoformat(),
        "streams": {
            "alpaca": alpaca,
            "yfinance": yfinance,
            "openai": {
                "provider": "openai",
                "status": "client_configured",
                "configured": None,
                "checked_at": datetime.now().isoformat(),
            },
            "tradesmart_backend": backend,
        },
    }


def _model_artifact_status(path):
    exists = os.path.exists(path)
    if not exists:
        return {"exists": False, "size_bytes": 0, "modified_at": None}
    stat = os.stat(path)
    return {
        "exists": True,
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def _read_model_metadata(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as exc:
        return {"error": str(exc)}


def _read_feature_names(path):
    if not os.path.exists(path):
        return []
    try:
        import joblib
        features = joblib.load(path)
        return list(features) if features is not None else []
    except Exception:
        return []


def _model_record(model_dir, scope, ticker=None):
    model_path = os.path.join(model_dir, 'stock_scorer_regression.joblib')
    artifact_paths = {
        "model": model_path,
        "feature_scaler": os.path.join(model_dir, 'feature_scaler.joblib'),
        "target_scaler": os.path.join(model_dir, 'target_scaler.joblib'),
        "feature_pca": os.path.join(model_dir, 'feature_pca.joblib'),
        "feature_selector": os.path.join(model_dir, 'feature_selector.joblib'),
        "feature_names": os.path.join(model_dir, 'feature_names.joblib'),
        "training_metadata": os.path.join(model_dir, 'training_metadata.json'),
    }
    artifacts = {name: _model_artifact_status(path) for name, path in artifact_paths.items()}
    model_status = artifacts["model"]
    metadata = _read_model_metadata(artifact_paths["training_metadata"])
    feature_names = _read_feature_names(artifact_paths["feature_names"])
    trained_at = metadata.get("trained_at") or model_status.get("modified_at")

    return {
        "id": ticker or "global-regression",
        "name": f"{ticker} regression scorer" if ticker else "Global regression scorer",
        "scope": scope,
        "ticker": ticker,
        "model_type": metadata.get("model_type", "regression"),
        "trained_at": trained_at,
        "sample_count": metadata.get("sample_count"),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "metadata": metadata,
        "artifacts": artifacts,
        "ready": bool(
            artifacts["model"]["exists"]
            and artifacts["feature_scaler"]["exists"]
            and artifacts["target_scaler"]["exists"]
            and artifacts["feature_pca"]["exists"]
            and artifacts["feature_names"]["exists"]
        ),
    }


def _trained_models():
    model_root = 'cache/models'
    models = []
    if not os.path.exists(model_root):
        return {"models": [], "count": 0, "storage": {"path": model_root, "exists": False}}

    if os.path.exists(os.path.join(model_root, 'stock_scorer_regression.joblib')):
        models.append(_model_record(model_root, "global"))

    for entry in sorted(os.listdir(model_root)):
        model_dir = os.path.join(model_root, entry)
        if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, 'stock_scorer_regression.joblib')):
            models.append(_model_record(model_dir, "ticker", entry))

    return {
        "models": models,
        "count": len(models),
        "storage": {
            "path": model_root,
            "exists": True,
            "backend_only": True,
        },
    }


def _calculate_buy_hold_benchmarks(tickers, start_date, end_date):
    """Calculate passive benchmarks for the exact backtest window."""
    ticker_returns = []
    for ticker in tickers:
        try:
            hist = get_stock_history(ticker, start_date, end_date, "1d")
            if hist is None or hist.empty or 'Close' not in hist:
                continue
            closes = hist['Close'].dropna()
            if len(closes) < 2:
                continue
            start_price = _safe_float(closes.iloc[0])
            end_price = _safe_float(closes.iloc[-1])
            if start_price <= 0:
                continue
            ticker_returns.append({
                "ticker": ticker,
                "start_price": start_price,
                "end_price": end_price,
                "return_pct": ((end_price / start_price) - 1) * 100,
            })
        except Exception as exc:
            log_message(f"Benchmark skipped for {ticker}: {exc}")

    equal_weight_return = (
        sum(row["return_pct"] for row in ticker_returns) / len(ticker_returns)
        if ticker_returns else 0.0
    )
    best = max(ticker_returns, key=lambda row: row["return_pct"], default=None)
    worst = min(ticker_returns, key=lambda row: row["return_pct"], default=None)
    return {
        "cash_return": 0.0,
        "equal_weight_return": equal_weight_return,
        "ticker_returns": ticker_returns,
        "best_ticker": best,
        "worst_ticker": worst,
        "coverage": len(ticker_returns),
    }


@app.route('/analysis_progress')
def analysis_progress_stream():
    """Server-Sent Events (SSE) endpoint for streaming analysis progress"""


    # Add a test message to verify
    print("DEBUG: analysis_progress_stream route called")



    tickers_input = request.args.get('tickers', '')
    tickers = [ticker.strip() for ticker in tickers_input.split(',') if ticker.strip()]

    if not tickers:
        return jsonify({"error": "No tickers provided"}), 400

    # Create a unique ID for this analysis session
    analysis_id = f"analysis_{int(time.time())}"

    # Create a queue for this analysis session
    message_queue = Queue()
    analysis_queues[analysis_id] = message_queue

    # Start the analysis in a background thread
    analysis_thread = Thread(target=run_analysis_with_updates, 
                            args=(tickers, analysis_id, message_queue))
    analysis_thread.daemon = True
    analysis_thread.start()

    # SSE response function
    def generate():
        try:
            while True:
                message = message_queue.get()
                if message is None:  # None is our signal to stop
                    break
                yield f"data: {json.dumps(message)}\n\n"

                # If analysis is complete, stop the stream
                if message.get('status') == 'complete':
                    break
        except GeneratorExit:
            # Client disconnected
            if analysis_id in analysis_queues:
                del analysis_queues[analysis_id]

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


def _parse_symbols(value):
    return [s.strip().upper() for s in (value or '').split(',') if s.strip()]


@app.route('/api/live/status')
def api_live_status():
    return jsonify(_sanitize(alpaca_live_data.status()))


@app.route('/api/market_data/status')
def api_market_data_status():
    return jsonify(_sanitize(_market_data_status()))


@app.route('/api/models/trained')
def api_trained_models():
    return jsonify(_sanitize(_trained_models()))


@app.route('/api/live/snapshot')
def api_live_snapshot():
    symbols = _parse_symbols(request.args.get('symbols', ''))
    if not symbols:
        return jsonify({"error": "No symbols provided"}), 400
    return jsonify(_sanitize(alpaca_live_data.snapshot(symbols)))


LIVE_STREAM_MAX_DURATION_SEC = 300
MAX_QUANT_TICKERS = 3


@app.route('/api/live/stream')
def api_live_stream():
    symbols = _parse_symbols(request.args.get('symbols', ''))
    if not symbols:
        return jsonify({"error": "No symbols provided"}), 400

    def generate():
        alpaca_live_data.acquire(symbols)
        deadline = time.time() + LIVE_STREAM_MAX_DURATION_SEC
        try:
            yield f"data: {json.dumps(_sanitize(alpaca_live_data.snapshot(symbols, hydrate=True)))}\n\n"
            while time.time() < deadline:
                payload = alpaca_live_data.snapshot(symbols, hydrate=False)
                yield f"data: {json.dumps(_sanitize(payload))}\n\n"
                time.sleep(1)
        except GeneratorExit:
            pass
        finally:
            alpaca_live_data.release()

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })

def run_analysis_with_updates(tickers, analysis_id, message_queue):
    """Run stock analysis with progress updates sent to the client using the integrated trading system"""
    try:
        # Define a custom message handler for this analysis
        def custom_message_handler(message):
            # Log to console as well as queue
            print(f"DEBUG: {message}")
            message_queue.put({
                "message": str(message)
            })

        # Set the custom message handler
        set_message_handler(custom_message_handler)

        # Continue with actual analysis...
        total_tickers = len(tickers)

        # Send initial message
        message_queue.put({
            "progress": 0,
            "message": f"Starting analysis of {total_tickers} tickers..."
        })

        # Initialize the trading system
        system = TradingSystem(initial_capital=100000.0)

        # Use the trading system to analyze stocks
        message_queue.put({
            "progress": 10,
            "message": "Initializing trading system..."
        })

        # Analyze stocks
        ranked_stocks, failed_tickers = system.analyze_stocks(tickers, use_ml=False)

        # Generate watchlist
        if ranked_stocks:
            message_queue.put({
                "progress": 80,
                "message": "Generating watchlist..."
            })

            system.generate_watchlist(ranked_stocks)

            # Create session for data storage
            session_id = datetime.now().strftime('%Y%m%d%H%M%S')
            session_folder = os.path.join(app.static_folder, 'sessions', session_id)
            os.makedirs(session_folder, exist_ok=True)

            # Save analysis data
            message_queue.put({
                "progress": 90,
                "message": "Saving analysis results..."
            })

            # Convert to DataFrame and save CSV
            df = pd.DataFrame(ranked_stocks)
            csv_path = os.path.join(session_folder, 'analysis.csv')
            df.to_csv(csv_path, index=False)

            # Create charts
            message_queue.put({
                "progress": 95,
                "message": "Generating visualizations..."
            })

            # Store analysis data in session
            analysis_progress[session_id] = {
                'ranked_stocks': ranked_stocks,
                'failed_tickers': failed_tickers,
                'session_id': session_id
            }

            # Completed
            message_queue.put({
                "progress": 100,
                "message": "Analysis complete! Redirecting to results...",
                "status": "complete",
                "session_id": session_id
            })
        else:
            message_queue.put({
                "error": "No valid data found for any of the provided tickers"
            })

    except Exception as e:
        message_queue.put({
            "error": f"Analysis failed: {str(e)}"
        })

    finally:
        # Reset the message handler to default
        set_message_handler(print)
        message_queue.put(None)
        analysis_queues.pop(analysis_id, None)


@app.route('/')
def index():
    return _spa_index()


@app.route('/api/stock_price_history/<ticker>')
def api_price_history(ticker):
    """API endpoint to get price history for charts"""
    days = request.args.get('days', 30, type=int)
    chart_data = prepare_price_chart_data(ticker, days=days)
    if not chart_data or "error" in chart_data:
        return jsonify({"error": "Failed to retrieve price history"}), 404
    return jsonify(_sanitize(chart_data))

@app.route('/api/industry_peers/<ticker>')
def api_industry_peers(ticker):
    """API endpoint to get industry peers for comparison"""
    try:
        # Get stock data using the technical_analysis module
        stock_data, error = get_stock_data(ticker)
        if error or not stock_data:
            return jsonify({"error": f"Failed to retrieve data for {ticker}"}), 404

        industry = stock_data.get('industry', 'Unknown')
        sector = stock_data.get('sector', 'Unknown')

        # Initialize the trading system to get peer data
        system = TradingSystem(initial_capital=100000.0)

        # Define peer tickers based on sector
        if sector == "Technology":
            peer_tickers = ["AAPL", "MSFT", "GOOGL", "META", "NVDA"]
        elif sector == "Consumer Cyclical":
            peer_tickers = ["AMZN", "TSLA", "HD", "MCD", "NKE"]
        elif sector == "Financial Services":
            peer_tickers = ["JPM", "BAC", "WFC", "C", "GS"]
        else:
            # Default peers
            peer_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

        # Add the current ticker if not already in the list
        if ticker not in peer_tickers:
            peer_tickers.append(ticker)

        # Get data for peers
        peers = []
        for peer_ticker in peer_tickers:
            try:
                if peer_ticker == ticker:
                    # Use the data we already have
                    peer_data = {
                        "ticker": ticker,
                        "day_trading_score": stock_data['day_trading_score']
                    }
                else:
                    # Get data for this peer
                    peer_stock_data, peer_error = get_stock_data(peer_ticker)
                    if not peer_error and peer_stock_data:
                        peer_data = {
                            "ticker": peer_ticker,
                            "day_trading_score": peer_stock_data['day_trading_score']
                        }
                    else:
                        continue

                peers.append(peer_data)
            except Exception:
                continue

        # Sort by score
        peers.sort(key=lambda x: x["day_trading_score"], reverse=True)

        return jsonify(_sanitize(peers))
    except Exception as e:
        return jsonify({"error": f"Error getting peer data: {str(e)}"}), 500

@app.route('/integrated_progress_stream')
def integrated_progress_stream():
    """Server-Sent Events (SSE) endpoint for streaming integrated system progress"""
    system_id = request.args.get('system_id')

    if not system_id or system_id not in integrated_queues:
        return jsonify({"error": "Invalid system ID"}), 400

    message_queue = integrated_queues[system_id]

    # SSE response function
    def generate():
        try:
            while True:
                message = message_queue.get()
                if message is None:  # None is our signal to stop
                    break
                yield f"data: {json.dumps(message)}\n\n"

                # If processing is complete, stop the stream
                if message.get('status') == 'complete':
                    break
        except GeneratorExit:
            # Client disconnected
            pass

    return Response(generate(), mimetype="text/event-stream")

def run_integrated_system_with_updates(tickers, use_ml, execute_trades, system_id, message_queue, session_id):
    """Run the integrated system with progress updates sent to the client"""
    try:
        # Define a custom message handler for this integrated system run
        def custom_message_handler(message):
            # Log to console as well as queue
            print(f"DEBUG: {message}")
            message_queue.put({
                "message": str(message)
            })

        # Set the custom message handler
        set_message_handler(custom_message_handler)

        # Send initial message
        message_queue.put({
            "progress": 0,
            "message": f"Starting integrated system for {', '.join(tickers)}...",
            "status": "initializing"
        })

        # Create session folder
        session_folder = os.path.join(app.static_folder, 'sessions', session_id)
        os.makedirs(session_folder, exist_ok=True)

        # Initialize the trading system
        message_queue.put({
            "progress": 10,
            "message": "Initializing trading system...",
            "status": "initializing"
        })
        system = TradingSystem(initial_capital=100000.0, market_neutral=True)

        # Run the complete workflow
        message_queue.put({
            "progress": 20,
            "message": "Running integrated system workflow...",
            "status": "processing"
        })
        results = system.run_complete_workflow(tickers, use_ml=use_ml, execute_trades=execute_trades)

        # Save results to session folder
        message_queue.put({
            "progress": 80,
            "message": "Saving results...",
            "status": "saving"
        })
        with open(os.path.join(session_folder, 'workflow_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        # Create charts
        message_queue.put({
            "progress": 90,
            "message": "Generating charts...",
            "status": "generating_charts"
        })
        charts = create_trading_system_charts(system, session_folder)

        # Prepare data for template
        template_data = {
            'system': system,
            'results': results,
            'charts': charts,
            'watchlist': system.watchlist,  # All analyzed stocks
            'portfolio_summary': system.portfolio_manager.get_portfolio_summary(),
            'session_id': session_id,
            'tickers': tickers,
            'use_ml': use_ml,
            'execute_trades': execute_trades
        }

        # Store the template data for later use
        app.config[f'integrated_result_{system_id}'] = template_data

        # Send completion message with redirect URL
        message_queue.put({
            "progress": 100,
            "message": "Integrated system processing complete!",
            "status": "complete",
            "system_id": system_id,
            "redirect_url": f"/integrated_results?system_id={system_id}"
        })

    except Exception as e:
        error_message = f"Error running integrated system: {str(e)}"
        print(f"ERROR: {error_message}")
        message_queue.put({
            "progress": 100,
            "message": error_message,
            "status": "error",
            "redirect_url": "/integrated_system"
        })
    finally:
        message_queue.put(None)
        integrated_queues.pop(system_id, None)


@app.route('/backtest_progress_stream')
def backtest_progress_stream():
    """Server-Sent Events (SSE) endpoint for streaming backtest progress"""
    backtest_id = request.args.get('backtest_id')

    if not backtest_id or backtest_id not in backtest_queues:
        return jsonify({"error": "Invalid backtest ID"}), 400

    message_queue = backtest_queues[backtest_id]

    # SSE response function
    def generate():
        try:
            while True:
                message = message_queue.get()
                if message is None:  # None is our signal to stop
                    break
                yield f"data: {json.dumps(message)}\n\n"

                # If backtest is complete, stop the stream
                if message.get('status') == 'complete':
                    break
        except GeneratorExit:
            # Client disconnected
            pass

    return Response(generate(), mimetype="text/event-stream")

def run_backtest_with_updates(tickers, strategy, start_date, end_date, days, custom_transaction_cost,
                              transaction_cost_type, backtest_id, message_queue, session_id,
                              buy_threshold=50, sell_threshold=40, partial_exit_fraction=0.25,
                              exit_sizing_mode='fixed_tranche', reentry_cooldown_days=10,
                              min_reentry_discount_pct=1.0, allow_pyramiding=False):
    """Run a backtest with progress updates sent to the client"""
    try:
        # Define a custom message handler for this backtest
        def custom_message_handler(message):
            # Log to console as well as queue
            print(f"DEBUG: {message}")
            message_queue.put({
                "message": str(message)
            })

        # Set the custom message handler
        set_message_handler(custom_message_handler)

        # Send initial message
        message_queue.put({
            "progress": 0,
            "message": f"Starting backtest for {', '.join(tickers)}...",
            "status": "initializing"
        })

        # Initialize the backtester
        system = TradingSystem(initial_capital=100000.0)

        message_queue.put({
            "progress": 10,
            "message": "Initializing trading system...",
            "status": "initializing"
        })

        # Run the backtest
        result = system.run_backtest(
            tickers=tickers,
            strategy=strategy,
            start_date=start_date if start_date else None,
            end_date=end_date if end_date else None,
            days=days,
            custom_transaction_cost=custom_transaction_cost,
            custom_transaction_cost_type=transaction_cost_type,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            partial_exit_fraction=partial_exit_fraction,
            exit_sizing_mode=exit_sizing_mode,
            reentry_cooldown_days=reentry_cooldown_days,
            min_reentry_discount_pct=min_reentry_discount_pct,
            allow_pyramiding=allow_pyramiding,
        )

        technical_baseline_result = None
        if strategy != 'technical':
            message_queue.put({
                "progress": 72,
                "message": "Running simple technical benchmark...",
                "status": "processing"
            })
            technical_baseline_result = system.run_backtest(
                tickers=tickers,
                strategy='technical',
                start_date=start_date if start_date else None,
                end_date=end_date if end_date else None,
                days=days,
                custom_transaction_cost=custom_transaction_cost,
                custom_transaction_cost_type=transaction_cost_type,
            )

        message_queue.put({
            "progress": 80,
            "message": "Backtest completed. Generating charts...",
            "status": "generating_charts"
        })

        # Create session folder if it doesn't exist
        session_folder = os.path.join(app.static_folder, 'sessions', session_id)
        os.makedirs(session_folder, exist_ok=True)

        # Create charts
        charts = create_backtest_charts(result, session_folder)

        message_queue.put({
            "progress": 95,
            "message": "Charts generated. Preparing results...",
            "status": "preparing_results"
        })

        # Prepare data for template
        template_data = {
            'result': result,
            'technical_baseline_result': technical_baseline_result,
            'charts': charts,
            'session_id': session_id,
            'tickers': tickers,
            'strategy': strategy,
            'start_date': start_date or (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            'end_date': end_date or datetime.now().strftime('%Y-%m-%d')
        }

        # Store the template data for later use
        app.config[f'backtest_result_{backtest_id}'] = template_data

        # Send completion message with redirect URL
        message_queue.put({
            "progress": 100,
            "message": "Backtest complete!",
            "status": "complete",
            "backtest_id": backtest_id,
            "redirect_url": f"/backtest_results?backtest_id={backtest_id}"
        })

    except Exception as e:
        error_message = f"Error running backtest: {str(e)}"
        print(f"ERROR: {error_message}")
        message_queue.put({
            "progress": 100,
            "message": error_message,
            "status": "error",
            "redirect_url": "/backtest"
        })
    finally:
        message_queue.put(None)
        backtest_queues.pop(backtest_id, None)


@app.route('/api/analysis_results/<session_id>')
def api_analysis_results(session_id):
    """Return stored analysis results as JSON for the React frontend"""
    data = analysis_progress.get(session_id)
    if not data:
        return jsonify({"error": "Session not found"}), 404
    ranked_stocks = _sanitize(data.get('ranked_stocks', []))
    failed_tickers = _sanitize(data.get('failed_tickers', []))
    csv_url = f'/static/sessions/{session_id}/analysis.csv'
    return jsonify({
        "ranked_stocks": ranked_stocks,
        "failed_tickers": failed_tickers,
        "session_id": session_id,
        "csv_url": csv_url if os.path.exists(os.path.join(app.static_folder, 'sessions', session_id, 'analysis.csv')) else None
    })


@app.route('/api/stock/<ticker>')
def api_stock_detail(ticker):
    """Return stock detail data as JSON for the React frontend"""
    try:
        stock_data, error = get_stock_data(ticker)
        if error or not stock_data:
            return jsonify({"error": f"Failed to retrieve {ticker}: {error}"}), 404
        stock_data['metrics'] = get_detailed_stock_metrics(stock_data)
        return jsonify(_sanitize(stock_data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# In-memory cache for financial statements. Keyed by (ticker, period) -> (timestamp, payload).
# Financials don't change intraday so a 1-hour TTL is plenty.
_financials_cache = {}
_FINANCIALS_TTL_SECONDS = 3600


def _df_to_periods(df):
    """Convert a yfinance financial-statement DataFrame to a JSON-safe dict.

    Returns a dict shaped as:
        { "periods": ["2024-12-31", "2023-12-31", ...],
          "rows": { "Total Revenue": {"2024-12-31": 123.4, ...}, ... } }
    """
    try:
        import pandas as _pd
    except ImportError:
        return {"periods": [], "rows": {}}
    if df is None or not hasattr(df, 'columns') or df.empty:
        return {"periods": [], "rows": {}}

    period_labels = []
    for col in df.columns:
        try:
            period_labels.append(col.strftime('%Y-%m-%d'))
        except Exception:
            period_labels.append(str(col))

    cleaned = df.where(_pd.notna(df), None)
    rows = {}
    for row_name, series in cleaned.iterrows():
        row_dict = {}
        for col, label in zip(df.columns, period_labels):
            val = series[col]
            if val is None:
                row_dict[label] = None
            else:
                try:
                    fval = float(val)
                    row_dict[label] = fval if math.isfinite(fval) else None
                except (TypeError, ValueError):
                    row_dict[label] = None
        rows[str(row_name)] = row_dict
    return {"periods": period_labels, "rows": rows}


def _fetch_financials(ticker, period):
    """Fetch annual or quarterly financials from yfinance for one ticker."""
    import yfinance as yf
    t = yf.Ticker(ticker, session=yf_session)
    if period == 'quarterly':
        income = t.quarterly_income_stmt
        balance = t.quarterly_balance_sheet
        cashflow = t.quarterly_cashflow
    else:
        income = t.income_stmt
        balance = t.balance_sheet
        cashflow = t.cashflow
    return {
        "income_statement": _df_to_periods(income),
        "balance_sheet": _df_to_periods(balance),
        "cash_flow": _df_to_periods(cashflow),
    }


@app.route('/api/financials/<ticker>')
def api_financials(ticker):
    """Return annual or quarterly 3-statement financials for a ticker."""
    period = request.args.get('period', 'annual')
    if period not in ('annual', 'quarterly'):
        return jsonify({"error": "period must be 'annual' or 'quarterly'"}), 400

    cache_key = (ticker.upper(), period)
    cached = _financials_cache.get(cache_key)
    now = time.time()
    if cached and now - cached[0] < _FINANCIALS_TTL_SECONDS:
        return jsonify(cached[1])

    try:
        payload = _fetch_financials(ticker, period)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch financials for {ticker}: {e}"}), 500

    payload = _sanitize(payload)
    payload['ticker'] = ticker.upper()
    payload['period'] = period
    _financials_cache[cache_key] = (now, payload)
    return jsonify(payload)


@app.route('/api/market_overview')
def api_market_overview():
    """Return market overview data as JSON for the React frontend"""
    try:
        raw = get_sector_performance()
        # get_sector_performance returns a dict with a "sectors" key (list of sector dicts)
        raw_sectors = raw.get("sectors", []) if isinstance(raw, dict) else []
        sectors = []
        for item in raw_sectors:
            if not isinstance(item, dict):
                continue
            sectors.append({
                "name": item.get("sector") or item.get("name", "Unknown"),
                "return_1d":    float(item.get("day_change_pct") or item.get("return_1d") or 0),
                "return_1w":    float(item.get("week_change_pct") or 0),
                "return_1m":    float(item.get("month_change_pct") or 0),
                "trend":        item.get("trend"),
            })
        market_trend = raw.get("market_trend", "Neutral") if isinstance(raw, dict) else "Neutral"
        pos = sum(1 for s in sectors if s["return_1d"] > 0)
        neg = sum(1 for s in sectors if s["return_1d"] < 0)
        return jsonify({
            "sectors": sectors,
            "market_trend": market_trend,
            "advances": pos,
            "declines": neg,
            "market_health": market_trend,
            "avg_day_change":   raw.get("avg_day_change", 0) if isinstance(raw, dict) else 0,
            "avg_week_change":  raw.get("avg_week_change", 0) if isinstance(raw, dict) else 0,
            "avg_month_change": raw.get("avg_month_change", 0) if isinstance(raw, dict) else 0,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/portfolio')
def api_portfolio():
    """Return portfolio data as JSON for the React frontend"""
    try:
        system = TradingSystem.load_system_state()
        summary = system.portfolio_manager.get_portfolio_summary()
        positions_raw = system.portfolio_manager.positions

        positions = []
        for pos in (positions_raw or []):
            if isinstance(pos, dict):
                positions.append(pos)
            elif hasattr(pos, '__dict__'):
                positions.append(pos.__dict__)

        summary_dict = summary if isinstance(summary, dict) else (summary.__dict__ if hasattr(summary, '__dict__') else {})

        return jsonify({"positions": positions, "summary": summary_dict})
    except Exception as e:
        return jsonify({"positions": [], "summary": {
            "total_value": 100000.0,
            "cash": 100000.0,
            "invested": 0.0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "num_positions": 0,
        }, "error": str(e)})


@app.route('/api/run_backtest', methods=['POST'])
def api_run_backtest():
    """Start a backtest and return the backtest_id for SSE progress tracking"""
    data = request.get_json() or {}
    tickers = data.get('tickers', [])
    if not tickers:
        return jsonify({"error": "No tickers provided"}), 400
    if len(tickers) > MAX_QUANT_TICKERS:
        return jsonify({
            "error": "3 tickers is the max for now. Log in to run larger universes because they are computationally heavy."
        }), 403

    strategy = data.get('strategy', 'Combined Strategy')
    start_date = data.get('start_date', '')
    end_date = data.get('end_date', '')
    days = int(data.get('days', 365))
    custom_transaction_cost = data.get('custom_transaction_cost')
    transaction_cost_type = data.get('transaction_cost_type', 'per_share')
    if transaction_cost_type not in ('fixed', 'percent', 'per_share'):
        transaction_cost_type = 'per_share'
    if custom_transaction_cost is not None:
        try:
            custom_transaction_cost = float(custom_transaction_cost)
        except (ValueError, TypeError):
            custom_transaction_cost = None
    try:
        buy_threshold = int(data.get('buy_threshold', 50))
    except (ValueError, TypeError):
        buy_threshold = 50
    try:
        sell_threshold = int(data.get('sell_threshold', 40))
    except (ValueError, TypeError):
        sell_threshold = 40
    try:
        partial_exit_fraction = float(data.get('partial_exit_fraction', 0.25))
    except (ValueError, TypeError):
        partial_exit_fraction = 0.25
    partial_exit_fraction = min(max(partial_exit_fraction, 0.05), 1.0)
    exit_sizing_mode = data.get('exit_sizing_mode', 'fixed_tranche')
    if exit_sizing_mode not in ('fixed_tranche', 'remaining_fraction'):
        exit_sizing_mode = 'fixed_tranche'
    try:
        reentry_cooldown_days = int(data.get('reentry_cooldown_days', 10))
    except (ValueError, TypeError):
        reentry_cooldown_days = 10
    reentry_cooldown_days = max(0, reentry_cooldown_days)
    try:
        min_reentry_discount_pct = float(data.get('min_reentry_discount_pct', 1.0))
    except (ValueError, TypeError):
        min_reentry_discount_pct = 1.0
    min_reentry_discount_pct = max(0.0, min_reentry_discount_pct)
    allow_pyramiding = bool(data.get('allow_pyramiding', False))

    backtest_id = f"backtest_{int(time.time())}"
    session_id = datetime.now().strftime('%Y%m%d%H%M%S')

    message_queue = Queue()
    backtest_queues[backtest_id] = message_queue

    backtest_thread = Thread(
        target=run_backtest_with_updates,
        args=(
            tickers, strategy, start_date, end_date, days, custom_transaction_cost,
            transaction_cost_type, backtest_id, message_queue, session_id,
            buy_threshold, sell_threshold, partial_exit_fraction, exit_sizing_mode,
            reentry_cooldown_days, min_reentry_discount_pct, allow_pyramiding
        )
    )
    backtest_thread.daemon = True
    backtest_thread.start()

    return jsonify({"backtest_id": backtest_id})


@app.route('/api/backtest_results/<backtest_id>')
def api_backtest_results(backtest_id):
    """Return backtest results as JSON for the React frontend"""
    key = f'backtest_result_{backtest_id}'
    if key not in app.config:
        return jsonify({"error": "Backtest results not found"}), 404

    template_data = app.config[key]
    result = template_data.get('result')
    if result is None:
        return jsonify({"error": "No result data"}), 404

    def _get(obj, attr, default):
        return obj.get(attr, default) if isinstance(obj, dict) else getattr(obj, attr, default)

    raw_trades = _get(result, 'trades', [])
    raw_equity = _get(result, 'equity_curve', [])
    raw_drawdown = _get(result, 'drawdown_curve', [])
    daily_returns = _get(result, 'daily_returns', []) or []
    initial_capital = _get(result, 'initial_capital', 0.0) or 0.0
    final_capital = _get(result, 'final_capital', initial_capital) or 0.0
    total_transaction_costs = _get(result, 'total_transaction_costs', 0.0) or 0.0
    transaction_cost_percentage = _get(result, 'transaction_cost_percentage', 0.0) or 0.0
    total_trade_value = sum((_get(trade, 'value', 0.0) or 0.0) for trade in raw_trades)
    if not total_transaction_costs:
        total_transaction_costs = sum((_get(trade, 'cost', 0.0) or 0.0) for trade in raw_trades)
    buy_count = sum(1 for trade in raw_trades if (_get(trade, 'type', None) or _get(trade, 'action', '')) == 'BUY')
    sell_count = sum(1 for trade in raw_trades if (_get(trade, 'type', None) or _get(trade, 'action', '')) == 'SELL')
    avg_daily_return = (sum(daily_returns) / len(daily_returns)) if daily_returns else 0.0
    best_day = max(daily_returns) if daily_returns else 0.0
    worst_day = min(daily_returns) if daily_returns else 0.0
    daily_volatility = 0.0
    if len(daily_returns) > 1:
        mean_return = avg_daily_return
        variance = sum((ret - mean_return) ** 2 for ret in daily_returns) / len(daily_returns)
        daily_volatility = math.sqrt(variance)

    if raw_equity:
        first_equity = _get(raw_equity[0], 'value', None)
        if first_equity is None:
            first_equity = _get(raw_equity[0], 'equity', 0.0)
        last_equity = _get(raw_equity[-1], 'value', None)
        if last_equity is None:
            last_equity = _get(raw_equity[-1], 'equity', 0.0)
        if not initial_capital:
            initial_capital = first_equity or 0.0
        if not final_capital:
            final_capital = last_equity or initial_capital

    metrics = {
        "total_return": (_get(result, 'total_return', 0.0) or 0.0) * 100,
        "annualized_return": (_get(result, 'annualized_return', 0.0) or 0.0) * 100,
        "sharpe_ratio": _get(result, 'sharpe_ratio', 0.0) or 0.0,
        "max_drawdown": (_get(result, 'max_drawdown', 0.0) or 0.0) * 100,
        "win_rate": _get(result, 'win_rate', 0.0) or 0.0,
        "num_trades": len(raw_trades),
        "profit_factor": _get(result, 'profit_factor', 0.0) or 0.0,
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "total_transaction_costs": total_transaction_costs,
        "transaction_cost_percentage": transaction_cost_percentage,
        "total_trade_value": total_trade_value,
        "avg_trade_value": total_trade_value / len(raw_trades) if raw_trades else 0.0,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "avg_daily_return": avg_daily_return * 100,
        "daily_volatility": daily_volatility * 100,
        "annualized_volatility": daily_volatility * math.sqrt(252) * 100,
        "best_day": best_day * 100,
        "worst_day": worst_day * 100,
    }

    trades = []
    open_lots = {}
    completed_round_trips = []
    for trade in raw_trades:
        action = _get(trade, 'type', None) or _get(trade, 'action', '')
        ticker = _get(trade, 'ticker', '')
        price = _get(trade, 'price', 0.0) or 0.0
        shares = _get(trade, 'shares', 0) or 0
        cost = _get(trade, 'cost', 0.0) or 0.0
        return_pct = None
        pnl = None
        if action == 'BUY':
            open_lots.setdefault(ticker, []).append({
                'price': price,
                'shares': shares,
                'remaining_shares': shares,
                'cost': cost,
                'date': _get(trade, 'date', ''),
            })
        elif action == 'SELL' and open_lots.get(ticker):
            shares_left = shares
            trade_pnl = 0.0
            matched_total = 0
            matched_cost_basis = 0.0
            while shares_left > 0 and open_lots.get(ticker):
                buy_lot = open_lots[ticker][0]
                matched_shares = min(shares_left, buy_lot.get('remaining_shares', buy_lot.get('shares', shares)) or shares)
                gross_buy = buy_lot.get('price', 0.0) * matched_shares
                gross_sell = price * matched_shares
                buy_cost_alloc = buy_lot.get('cost', 0.0) * (matched_shares / buy_lot.get('shares', matched_shares)) if buy_lot.get('shares') else 0.0
                sell_cost_alloc = cost * (matched_shares / shares) if shares else 0.0
                lot_pnl = gross_sell - gross_buy - sell_cost_alloc - buy_cost_alloc
                completed_round_trips.append({
                    'ticker': ticker,
                    'entry_date': buy_lot.get('date', ''),
                    'exit_date': _get(trade, 'date', ''),
                    'entry_price': buy_lot.get('price', 0.0),
                    'exit_price': price,
                    'shares': matched_shares,
                    'pnl': lot_pnl,
                    'return_pct': (lot_pnl / gross_buy * 100) if gross_buy else None,
                })
                trade_pnl += lot_pnl
                matched_total += matched_shares
                matched_cost_basis += gross_buy
                buy_lot['remaining_shares'] -= matched_shares
                shares_left -= matched_shares
                if buy_lot['remaining_shares'] <= 0:
                    open_lots[ticker].pop(0)
            pnl = trade_pnl if matched_total else None
            return_pct = (trade_pnl / matched_cost_basis * 100) if matched_cost_basis else None
        trades.append({
            **(trade if isinstance(trade, dict) else {}),
            "date": _get(trade, 'date', ''),
            "ticker": ticker,
            "type": action,
            "action": action,
            "price": price,
            "shares": shares,
            "cost": cost,
            "value": _get(trade, 'value', 0.0),
            "net_value": _get(trade, 'net_value', 0.0),
            "pnl": pnl,
            "return_pct": return_pct,
        })

    equity = []
    for point in raw_equity:
        value = _get(point, 'value', None)
        if value is None:
            value = _get(point, 'equity', 0.0)
        equity.append({
            **(point if isinstance(point, dict) else {}),
            "date": _get(point, 'date', ''),
            "value": value,
            "equity": value,
        })

    drawdown = []
    for point in raw_drawdown:
        value = _get(point, 'drawdown', 0.0) or 0.0
        drawdown.append({
            **(point if isinstance(point, dict) else {}),
            "date": _get(point, 'date', ''),
            "drawdown": value * 100,
        })

    risk_summary = {
        "trading_days": len(equity),
        "completed_round_trips": len(completed_round_trips),
        "winning_round_trips": sum(1 for trade in completed_round_trips if (trade.get('pnl') or 0) > 0),
        "losing_round_trips": sum(1 for trade in completed_round_trips if (trade.get('pnl') or 0) <= 0),
        "open_positions_estimate": sum(len(lots) for lots in open_lots.values()),
        "turnover_to_initial_capital": (total_trade_value / initial_capital) if initial_capital else 0.0,
        "average_transaction_cost": total_transaction_costs / len(raw_trades) if raw_trades else 0.0,
    }

    position_map = {}
    ticker_pnl = {}
    for rt in completed_round_trips:
        key = (
            rt.get('ticker', ''),
            rt.get('entry_date', ''),
            round(_safe_float(rt.get('entry_price')), 6),
        )
        position = position_map.setdefault(key, {
            "ticker": rt.get('ticker', ''),
            "entry_date": rt.get('entry_date', ''),
            "last_exit_date": rt.get('exit_date', ''),
            "entry_price": rt.get('entry_price', 0.0),
            "shares_sold": 0,
            "exit_count": 0,
            "pnl": 0.0,
            "gross_entry_value": 0.0,
        })
        shares = int(rt.get('shares') or 0)
        position["shares_sold"] += shares
        position["exit_count"] += 1
        position["last_exit_date"] = rt.get('exit_date', position["last_exit_date"])
        position["pnl"] += _safe_float(rt.get('pnl'))
        position["gross_entry_value"] += _safe_float(rt.get('entry_price')) * shares
        ticker_pnl[rt.get('ticker', '')] = ticker_pnl.get(rt.get('ticker', ''), 0.0) + _safe_float(rt.get('pnl'))

    position_round_trips = []
    for position in position_map.values():
        gross_entry = position.get("gross_entry_value", 0.0)
        position_round_trips.append({
            **position,
            "return_pct": (position["pnl"] / gross_entry * 100) if gross_entry else None,
        })

    winning_positions = sum(1 for row in position_round_trips if _safe_float(row.get('pnl')) > 0)
    losing_positions = sum(1 for row in position_round_trips if _safe_float(row.get('pnl')) <= 0)
    gross_position_profit = sum(_safe_float(row.get('pnl')) for row in position_round_trips if _safe_float(row.get('pnl')) > 0)
    gross_position_loss = abs(sum(_safe_float(row.get('pnl')) for row in position_round_trips if _safe_float(row.get('pnl')) < 0))
    position_profit_factor = (
        gross_position_profit / gross_position_loss if gross_position_loss > 0
        else (gross_position_profit if gross_position_profit > 0 else 0.0)
    )
    position_diagnostics = {
        "completed_positions": len(position_round_trips),
        "winning_positions": winning_positions,
        "losing_positions": losing_positions,
        "position_win_rate": (winning_positions / len(position_round_trips)) if position_round_trips else 0.0,
        "position_profit_factor": position_profit_factor,
        "gross_position_profit": gross_position_profit,
        "gross_position_loss": gross_position_loss,
        "position_round_trips": position_round_trips,
    }

    realized_pnl = sum(_safe_float(row.get('pnl')) for row in completed_round_trips)
    best_trade = max(completed_round_trips, key=lambda row: _safe_float(row.get('pnl')), default=None)
    top_ticker, top_ticker_pnl = (None, 0.0)
    if ticker_pnl:
        top_ticker, top_ticker_pnl = max(ticker_pnl.items(), key=lambda item: item[1])
    concentration = {
        "realized_pnl": realized_pnl,
        "ticker_pnl": [{"ticker": ticker, "pnl": pnl} for ticker, pnl in sorted(ticker_pnl.items(), key=lambda item: item[1], reverse=True)],
        "top_ticker": top_ticker,
        "top_ticker_pnl": top_ticker_pnl,
        "top_ticker_profit_share": (top_ticker_pnl / gross_position_profit) if gross_position_profit > 0 else 0.0,
        "best_trade": best_trade,
        "best_trade_pnl": _safe_float(best_trade.get('pnl')) if best_trade else 0.0,
        "return_without_top_ticker": (((final_capital - top_ticker_pnl) / initial_capital) - 1) * 100 if initial_capital and top_ticker else metrics["total_return"],
        "return_without_best_trade": (((final_capital - (_safe_float(best_trade.get('pnl')) if best_trade else 0.0)) / initial_capital) - 1) * 100 if initial_capital else metrics["total_return"],
    }

    start_for_benchmark = template_data.get('start_date', '')
    end_for_benchmark = template_data.get('end_date', '')
    benchmarks = template_data.get('benchmarks')
    if not isinstance(benchmarks, dict):
        benchmarks = _calculate_buy_hold_benchmarks(template_data.get('tickers', []), start_for_benchmark, end_for_benchmark)
        template_data['benchmarks'] = benchmarks
    benchmarks["strategy_vs_equal_weight"] = metrics["total_return"] - benchmarks.get("equal_weight_return", 0.0)
    benchmarks["strategy_vs_cash"] = metrics["total_return"] - benchmarks.get("cash_return", 0.0)
    technical_baseline_result = template_data.get('technical_baseline_result')
    if technical_baseline_result is not None:
        technical_return = (_get(technical_baseline_result, 'total_return', 0.0) or 0.0) * 100
        benchmarks["technical_rule"] = {
            "return": technical_return,
            "sharpe_ratio": _get(technical_baseline_result, 'sharpe_ratio', 0.0) or 0.0,
            "max_drawdown": (_get(technical_baseline_result, 'max_drawdown', 0.0) or 0.0) * 100,
            "num_trades": len(_get(technical_baseline_result, 'trades', []) or []),
        }
        benchmarks["strategy_vs_technical"] = metrics["total_return"] - technical_return
    elif template_data.get('strategy') == 'technical':
        benchmarks["technical_rule"] = {
            "return": metrics["total_return"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "num_trades": metrics["num_trades"],
        }
        benchmarks["strategy_vs_technical"] = 0.0

    warnings = []
    score = 0
    equal_weight_return = benchmarks.get("equal_weight_return", 0.0)
    if benchmarks.get("coverage", 0) > 0:
        if metrics["total_return"] > equal_weight_return + 2:
            score += 1
        elif metrics["total_return"] < equal_weight_return:
            score -= 1
            warnings.append(f"Underperformed equal-weight buy-and-hold by {abs(metrics['total_return'] - equal_weight_return):.2f} percentage points.")

    if "strategy_vs_technical" in benchmarks:
        if benchmarks["strategy_vs_technical"] > 1:
            score += 1
        elif benchmarks["strategy_vs_technical"] < 0:
            score -= 1
            warnings.append(f"Underperformed the simple technical rule by {abs(benchmarks['strategy_vs_technical']):.2f} percentage points.")

    if metrics["sharpe_ratio"] >= 1:
        score += 2
    elif metrics["sharpe_ratio"] >= 0.5:
        score += 1
    else:
        score -= 1
        warnings.append("Sharpe is below 0.50, so risk-adjusted return is weak.")

    if metrics["max_drawdown"] <= 10:
        score += 1
    else:
        score -= 1
        warnings.append("Maximum drawdown is above 10%.")

    if len(position_round_trips) >= 10:
        score += 1
    else:
        score -= 1
        warnings.append("Too few completed position lifecycles for a reliable conclusion.")

    if risk_summary["open_positions_estimate"] > len(position_round_trips):
        score -= 1
        warnings.append("Many open lots remain, so final equity relies heavily on mark-to-market positions.")

    if concentration["top_ticker_profit_share"] > 0.4:
        score -= 1
        warnings.append(f"Realized profits are concentrated in {top_ticker}.")

    if gross_position_loss < max(100.0, initial_capital * 0.001) and gross_position_profit > 0:
        score -= 1
        warnings.append("Profit factor is unstable because realized losses are tiny.")

    if buy_count and len(template_data.get('tickers', [])) and buy_count >= len(template_data.get('tickers', [])) * 0.8:
        warnings.append("Entry threshold is broad; the strategy bought most of the universe rather than selecting tightly.")

    if score >= 4:
        verdict_label = "Robust Candidate"
        verdict_color = "up"
    elif score >= 2:
        verdict_label = "Promising"
        verdict_color = "accent"
    elif score >= 0:
        verdict_label = "Needs Validation"
        verdict_color = "warn"
    else:
        verdict_label = "Weak"
        verdict_color = "down"

    alpha_verdict = {
        "label": verdict_label,
        "color": verdict_color,
        "score": score,
        "summary": (
            f"{verdict_label}: {metrics['total_return']:.2f}% return, "
            f"{metrics['sharpe_ratio']:.2f} Sharpe, "
            f"{benchmarks.get('strategy_vs_equal_weight', 0.0):+.2f} pp vs equal-weight."
        ),
        "warnings": warnings[:6],
    }

    run_metadata = _get(result, 'run_metadata', {}) or {}
    if isinstance(run_metadata, dict):
        run_metadata = {
            **run_metadata,
            "backtest_id": backtest_id,
            "session_id": template_data.get('session_id', ''),
        }

    return jsonify({
        "metrics":      _sanitize(metrics),
        "trades":       _sanitize(trades),
        "equity_curve": _sanitize(equity),
        "drawdown_curve": _sanitize(drawdown),
        "daily_returns": _sanitize([(ret * 100) for ret in daily_returns]),
        "round_trips": _sanitize(completed_round_trips),
        "position_diagnostics": _sanitize(position_diagnostics),
        "benchmarks": _sanitize(benchmarks),
        "concentration": _sanitize(concentration),
        "alpha_verdict": _sanitize(alpha_verdict),
        "risk_summary": _sanitize(risk_summary),
        "training_context": _sanitize(_get(result, 'training_context', {})),
        "run_metadata": _sanitize(run_metadata),
        "tickers":   template_data.get('tickers', []),
        "strategy":  template_data.get('strategy', ''),
        "start_date": template_data.get('start_date', ''),
        "end_date":   template_data.get('end_date', ''),
    })


@app.route('/api/run_integrated', methods=['POST'])
def api_run_integrated():
    """Start the integrated trading system and return the system_id for SSE tracking"""
    data = request.get_json() or {}
    tickers = data.get('tickers', [])
    if not tickers:
        return jsonify({"error": "No tickers provided"}), 400
    if len(tickers) > MAX_QUANT_TICKERS:
        return jsonify({
            "error": "3 tickers is the max for now. Log in to run larger universes because they are computationally heavy."
        }), 403

    use_ml = bool(data.get('use_ml', True))
    execute_trades = bool(data.get('execute_trades', False))

    system_id = f"system_{int(time.time())}"
    session_id = datetime.now().strftime('%Y%m%d%H%M%S')

    message_queue = Queue()
    integrated_queues[system_id] = message_queue

    system_thread = Thread(
        target=run_integrated_system_with_updates,
        args=(tickers, use_ml, execute_trades, system_id, message_queue, session_id)
    )
    system_thread.daemon = True
    system_thread.start()

    return jsonify({"system_id": system_id})


@app.route('/api/integrated_results/<system_id>')
def api_integrated_results(system_id):
    """Return integrated system results as JSON for the React frontend"""
    key = f'integrated_result_{system_id}'
    if key not in app.config:
        return jsonify({"error": "Integrated results not found"}), 404

    template_data = app.config[key]

    watchlist = _sanitize(template_data.get('watchlist', []))
    portfolio_summary = _sanitize(template_data.get('portfolio_summary', {}))
    tickers = template_data.get('tickers', [])

    signals = [s for s in watchlist if isinstance(s, dict)] if isinstance(watchlist, list) else []

    return jsonify({
        "signals": signals,
        "portfolio_summary": portfolio_summary if isinstance(portfolio_summary, dict) else {
            "total_value": 100000.0, "cash": 100000.0, "invested": 0.0,
            "total_pnl": 0.0, "total_pnl_pct": 0.0, "num_positions": 0,
        },
        "tickers": tickers,
    })


@app.route('/api/documentation/<doc_type>')
def api_documentation(doc_type='readme'):
    """Return documentation as HTML for the React frontend"""
    doc_map = {
        'readme':     ('README.md',                      'README — TradeSmart'),
        'strategy':   (os.path.join('docs', 'strategy.md'), 'Strategy Guide'),
        'backtest':   (os.path.join('docs', 'backtest.md'), 'Backtesting'),
        'integrated': (os.path.join('docs', 'Logic_Flow.md'), 'Integrated System'),
        'api':        (os.path.join('docs', 'api.md'),    'API Reference'),
    }
    if doc_type not in doc_map:
        return jsonify({"error": "Not found"}), 404

    file_path, title = doc_map[doc_type]
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        html_content = markdown.markdown(content, extensions=['tables', 'fenced_code'])
        return jsonify({"title": title, "html_content": html_content})
    except FileNotFoundError:
        return jsonify({"title": title, "html_content": "<p>Documentation not available.</p>"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/healthcheck')
def healthcheck():
    """Simple endpoint to check if the API is functioning"""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

def _spa_index():
    """Serve the React SPA index.html if the build exists."""
    dist = os.path.join(os.path.dirname(__file__), 'static', 'dist')
    if os.path.exists(os.path.join(dist, 'index.html')):
        return send_from_directory(dist, 'index.html')
    return None


@app.route('/<path:path>')
def spa_catchall(path):
    """Serve static assets from the Vite build, or fall back to the SPA for React routes."""
    dist = os.path.join(os.path.dirname(__file__), 'static', 'dist')
    candidate = os.path.join(dist, path)
    if os.path.isfile(candidate):
        return send_from_directory(dist, path)
    return send_from_directory(dist, 'index.html')

def create_analysis_charts(df, session_folder):
    charts = {}

    # 1. Strategy Distribution Pie Chart
    fig = px.pie(df, names='day_trading_strategy', title='Strategy Distribution')
    charts['strategy_dist'] = f"strategy_dist.html"
    fig.write_html(os.path.join(session_folder, charts['strategy_dist']))

    # 2. Trading Score vs. Volatility Scatter Plot
    fig = px.scatter(df, x='atr_pct', y='day_trading_score', 
                    color='day_trading_strategy', hover_name='ticker',
                    title='Trading Score vs. Volatility (ATR%)',
                    labels={'atr_pct': 'ATR %', 'day_trading_score': 'Day Trading Score'})
    charts['score_vs_volatility'] = f"score_vs_volatility.html"
    fig.write_html(os.path.join(session_folder, charts['score_vs_volatility']))

    # 3. RSI Distribution
    fig = px.histogram(df, x='rsi7', color='day_trading_strategy', 
                     title='RSI-7 Distribution',
                     labels={'rsi7': 'RSI-7 Value', 'count': 'Number of Stocks'})
    charts['rsi_dist'] = f"rsi_dist.html"
    fig.write_html(os.path.join(session_folder, charts['rsi_dist']))

    # 4. Sentiment vs Trading Score
    fig = px.scatter(df, x='news_sentiment_score', y='day_trading_score', 
                    color='day_trading_strategy', hover_name='ticker',
                    title='News Sentiment vs. Trading Score',
                    labels={'news_sentiment_score': 'News Sentiment', 'day_trading_score': 'Day Trading Score'})
    charts['sentiment_vs_score'] = f"sentiment_vs_score.html"
    fig.write_html(os.path.join(session_folder, charts['sentiment_vs_score']))

    # 5. Top 15 Stocks by Day Trading Score
    top_15 = df.sort_values('day_trading_score', ascending=False).head(15)
    fig = px.bar(top_15, x='ticker', y='day_trading_score', color='day_trading_strategy',
               title='Top 15 Stocks by Day Trading Score',
               labels={'ticker': 'Ticker Symbol', 'day_trading_score': 'Day Trading Score'})
    charts['top_15'] = f"top_15.html"
    fig.write_html(os.path.join(session_folder, charts['top_15']))

    return charts

def create_stock_detail_charts(stock_data):
    charts = {}

    # 1. Radar Chart of Key Metrics
    categories = ['technical_score', 'volatility_score', 'news_sentiment_score_normalized', 
                'gap_score', 'volume_score']
    values = [stock_data[c] for c in categories]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=['Technical', 'Volatility', 'News Sentiment', 'Gap Potential', 'Volume'],
        fill='toself',
        name=stock_data['ticker']
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title=f"Performance Metrics for {stock_data['ticker']}"
    )
    charts['radar'] = plotly.utils.PlotlyJSONEncoder().encode(fig)

    # 2. Technical Indicators Gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = stock_data['day_trading_score'],
        title = {'text': "Day Trading Score"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 35], 'color': "red"},
                {'range': [35, 45], 'color': "orange"},
                {'range': [45, 60], 'color': "yellow"},
                {'range': [60, 70], 'color': "lightgreen"},
                {'range': [70, 100], 'color': "green"}
            ]
        }
    ))
    charts['score_gauge'] = plotly.utils.PlotlyJSONEncoder().encode(fig)

    return charts

def generate_watchlist_data(df, min_score=65, max_stocks=10):
    """
    Generate a watchlist of the most promising stocks.

    Parameters:
        df (DataFrame): DataFrame containing stock data
        min_score (float): Minimum score to include in watchlist
        max_stocks (int): Maximum number of stocks in the watchlist

    Returns:
        list: Watchlist of stocks
    """
    # Filter stocks by minimum score
    candidates = df[df['day_trading_score'] >= min_score].to_dict('records')

    # If we don't have enough stocks meeting the score threshold, take the top scoring ones
    if len(candidates) < max_stocks:
        additional = df[df['day_trading_score'] < min_score].sort_values('day_trading_score', ascending=False).head(max_stocks-len(candidates)).to_dict('records')
        candidates.extend(additional)

    # Limit to max_stocks
    return candidates[:max_stocks]

def create_trading_system_charts(system, session_folder):
    """Create charts for the integrated trading system results"""
    charts = {}

    # 1. Portfolio Composition Pie Chart
    portfolio_summary = system.portfolio_manager.get_portfolio_summary()

    if portfolio_summary['invested_value'] > 0:
        # Create data for pie chart
        sector_values = []
        sector_labels = []

        for sector, data in portfolio_summary['sector_exposures'].items():
            sector_values.append(data['value'])
            sector_labels.append(f"{sector} (${data['value']:,.0f})")

        # Add cash as a sector
        sector_values.append(portfolio_summary['cash'])
        sector_labels.append(f"Cash (${portfolio_summary['cash']:,.0f})")

        fig = px.pie(
            values=sector_values,
            names=sector_labels,
            title='Portfolio Composition'
        )
        charts['portfolio_composition'] = "portfolio_composition.html"
        fig.write_html(os.path.join(session_folder, charts['portfolio_composition']))

    # 2. Watchlist Scores Bar Chart
    if system.watchlist:
        watchlist_df = pd.DataFrame([
            {
                'ticker': stock['ticker'],
                'score': stock['day_trading_score'],
                'strategy': stock['day_trading_strategy']
            }
            for stock in system.watchlist[:15]  # Top 15
        ])

        fig = px.bar(
            watchlist_df,
            x='ticker',
            y='score',
            color='strategy',
            title='Watchlist Stocks by Score',
            labels={'ticker': 'Ticker', 'score': 'Trading Score', 'strategy': 'Strategy'}
        )
        charts['watchlist_scores'] = "watchlist_scores.html"
        fig.write_html(os.path.join(session_folder, charts['watchlist_scores']))

    # 3. ML Feature Importance (if available)
    if system.is_model_trained:
        feature_importance = system.ml_scorer.get_feature_importance()
        if feature_importance:
            # Convert to DataFrame
            importance_df = pd.DataFrame({
                'feature': list(feature_importance.keys()),
                'importance': list(feature_importance.values())
            }).sort_values('importance', ascending=False).head(15)  # Top 15 features

            fig = px.bar(
                importance_df,
                x='importance',
                y='feature',
                orientation='h',
                title='ML Feature Importance',
                labels={'importance': 'Importance', 'feature': 'Feature'}
            )
            charts['feature_importance'] = "feature_importance.html"
            fig.write_html(os.path.join(session_folder, charts['feature_importance']))

    return charts

def create_backtest_charts(backtest_result, session_folder):
    """Create charts for backtest results"""
    charts = {}

    # 1. Equity Curve
    if backtest_result.equity_curve:
        # Convert to DataFrame
        equity_df = pd.DataFrame(backtest_result.equity_curve)

        fig = px.line(
            equity_df,
            x='date',
            y='equity',
            title='Equity Curve',
            labels={'date': 'Date', 'equity': 'Equity ($)'}
        )
        charts['equity_curve'] = "equity_curve.html"
        fig.write_html(os.path.join(session_folder, charts['equity_curve']))

    # 2. Drawdown Chart
    if backtest_result.drawdown_curve:
        # Convert to DataFrame
        drawdown_df = pd.DataFrame(backtest_result.drawdown_curve)

        fig = px.line(
            drawdown_df,
            x='date',
            y='drawdown',
            title='Drawdown',
            labels={'date': 'Date', 'drawdown': 'Drawdown (%)'}
        )
        # Update y-axis to show as percentage and inverted
        fig.update_layout(
            yaxis=dict(
                tickformat='.0%',
                autorange="reversed"  # Invert y-axis
            )
        )
        charts['drawdown'] = "drawdown.html"
        fig.write_html(os.path.join(session_folder, charts['drawdown']))

    # 3. Trade Analysis
    if backtest_result.trades:
        # Convert to DataFrame
        trades_df = pd.DataFrame(backtest_result.trades)

        # Group by ticker and action
        trade_summary = trades_df.groupby(['ticker', 'action']).agg({
            'value': 'sum',
            'cost': 'sum',
            'shares': 'sum'
        }).reset_index()

        fig = px.bar(
            trade_summary,
            x='ticker',
            y='value',
            color='action',
            title='Trade Analysis by Ticker',
            labels={'ticker': 'Ticker', 'value': 'Trade Value ($)', 'action': 'Action'}
        )
        charts['trade_analysis'] = "trade_analysis.html"
        fig.write_html(os.path.join(session_folder, charts['trade_analysis']))

    return charts

def _wrap_cookie(cookie, session):
    """
    If cookie is just a str (cookie name), look up its value
    in session.cookies and wrap it into a real Cookie object.
    """
    if isinstance(cookie, str):
        value = session.cookies.get(cookie)
        return create_cookie(name=cookie, value=value)
    return cookie

def patch_yfdata_cookie_basic():
    """
    Monkey-patch YfData._get_cookie_basic so that
    it always returns a proper Cookie object,
    even when response.cookies is a simple dict.
    """
    original = _data.YfData._get_cookie_basic

    def _patched(self, timeout=30):
        cookie = original(self, timeout)
        return _wrap_cookie(cookie, self._session)

    _data.YfData._get_cookie_basic = _patched

# Apply the yfinance cookie patch at module level
patch_yfdata_cookie_basic()

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(os.path.join('static', 'sessions'), exist_ok=True)

    # Apply the yfinance cookie patch
    patch_yfdata_cookie_basic()

    # Log successful startup
    print(f"Starting ASX Financial Data Analysis server at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Using modified yfinance with proper rate-limiting protection")

    import os

    # Get environment variables
    port = int(os.environ.get("PORT", 8080))
    debug_mode = os.environ.get("FLASK_ENV", "production") != "production"

    # Run the app with appropriate settings for the environment
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
