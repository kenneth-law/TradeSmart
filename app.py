from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime, timedelta
import os
from curl_cffi import requests
import markdown
import re

import time
import threading
import queue
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response
from threading import Thread
from queue import Queue
import json

# Import integrated trading system
from modules.trading_system import TradingSystem

# Import necessary modules
from modules.utils import set_message_handler, log_message
from modules.data_retrieval import get_stock_info, get_stock_history, get_yf_session
from modules.technical_analysis import get_stock_data
from modules.visualization import prepare_price_chart_data, get_detailed_stock_metrics
from modules.market_data import get_sector_performance

from requests.cookies import create_cookie
import yfinance.data as _data


app = Flask(__name__)


analysis_progress = {}
analysis_logs = {}
analysis_queues = {}

# For progress tracking
backtest_queues = {}
integrated_queues = {}

# Configure session at app startup to avoid repeated session creation
yf_session = get_yf_session()


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

    return Response(generate(), mimetype="text/event-stream")

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

        # Clean up
        if analysis_id in analysis_queues:
            del analysis_queues[analysis_id]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle form submission and display results using the integrated trading system"""
    # Get ticker symbols from form
    ticker_input = request.form.get('tickers', '')
    tickers = [ticker.strip() for ticker in ticker_input.split(',') if ticker.strip()]

    if not tickers:
        return render_template('index.html', error="Please enter at least one ticker symbol")

    # For form submissions without JS, run analysis directly
    # For JS-enabled clients, analysis would have already run via SSE

    # Check if we already have analysis results in progress data
    recent_sessions = sorted(analysis_progress.keys(), reverse=True)
    if recent_sessions:
        session_id = recent_sessions[0]
        data = analysis_progress[session_id]
        ranked_stocks = data['ranked_stocks']
        failed_tickers = data['failed_tickers']

        # Clean up the progress data
        del analysis_progress[session_id]
    else:
        # Fallback to regular analysis using the integrated trading system
        system = TradingSystem(initial_capital=100000.0)
        ranked_stocks, failed_tickers = system.analyze_stocks(tickers, use_ml=False)

        # Create session folder for charts
        session_id = datetime.now().strftime('%Y%m%d%H%M%S')
        session_folder = os.path.join(app.static_folder, 'sessions', session_id)
        os.makedirs(session_folder, exist_ok=True)

    if not ranked_stocks:
        return render_template('index.html', error="No valid data found for the provided tickers")

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(ranked_stocks)

    # Save CSV
    csv_path = os.path.join(app.static_folder, 'sessions', session_id, 'analysis.csv')
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)

    # Create charts
    charts = create_analysis_charts(df, os.path.join(app.static_folder, 'sessions', session_id))

    # Categorize stocks
    categories = {
        "Strong Buy": df[df['day_trading_strategy'] == 'Strong Buy'].to_dict('records'),
        "Buy": df[df['day_trading_strategy'] == 'Buy'].to_dict('records'),
        "Neutral/Watch": df[df['day_trading_strategy'] == 'Neutral/Watch'].to_dict('records'),
        "Sell": df[df['day_trading_strategy'] == 'Sell'].to_dict('records'),
        "Strong Sell": df[df['day_trading_strategy'] == 'Strong Sell'].to_dict('records')
    }

    # Generate watchlist
    watchlist = generate_watchlist_data(df, min_score=65, max_stocks=10)

    return render_template(
        'results.html',
        categories=categories,
        charts=charts,
        watchlist=watchlist,
        failed_tickers=failed_tickers,
        session_id=session_id,
        csv_url=f'/static/sessions/{session_id}/analysis.csv'
    )

@app.route('/stock/<ticker>')
def stock_detail(ticker):
    """Display detailed information for a single stock"""
    try:
        # Get detailed data for a single stock using the technical_analysis module
        stock_data, error = get_stock_data(ticker)

        if error or not stock_data:
            return render_template('error.html', error=f"Error retrieving data for {ticker}: {error}")

        # Get detailed metrics
        stock_data['metrics'] = get_detailed_stock_metrics(stock_data)

        # Create detailed charts for this stock
        charts = create_stock_detail_charts(stock_data)

        return render_template('stock_detail.html', stock=stock_data, charts=charts)
    except Exception as e:
        return render_template('error.html', error=f"Error analyzing {ticker}: {str(e)}")

@app.route('/api/stock_price_history/<ticker>')
def api_price_history(ticker):

    """API endpoint to get price history for charts"""
    chart_data = prepare_price_chart_data(ticker)
    if not chart_data:
        return jsonify({"error": "Failed to retrieve price history"}), 404
    return jsonify(chart_data)

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

        return jsonify(peers)
    except Exception as e:
        return jsonify({"error": f"Error getting peer data: {str(e)}"}), 500

@app.route('/market_overview')
def market_overview():
    """Show market overview with sector performance"""
    sector_data = get_sector_performance()
    return render_template('market_overview.html', sector_data=sector_data)

@app.route('/integrated_system')
def integrated_system():
    """Show the integrated trading system interface"""
    return render_template('integrated_system.html')

@app.route('/integrated_progress')
def integrated_progress():
    """Show the integrated system progress page"""
    system_id = request.args.get('system_id')
    if not system_id or system_id not in integrated_queues:
        return redirect(url_for('integrated_system'))

    return render_template('integrated_progress.html')

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

@app.route('/integrated_results')
def integrated_results():
    """Show the integrated system results page"""
    system_id = request.args.get('system_id')

    if not system_id or f'integrated_result_{system_id}' not in app.config:
        return redirect(url_for('integrated_system'))

    # Get the template data
    template_data = app.config[f'integrated_result_{system_id}']

    # Clean up the queue
    if system_id in integrated_queues:
        del integrated_queues[system_id]

    return render_template('integrated_results.html', **template_data)

@app.route('/run_integrated_system', methods=['POST'])
def run_integrated_system():
    """Run the integrated trading system with the provided tickers"""
    # Get ticker symbols from form
    ticker_input = request.form.get('tickers', '')
    tickers = [ticker.strip() for ticker in ticker_input.split(',') if ticker.strip()]

    if not tickers:
        return render_template('integrated_system.html', error="Please enter at least one ticker symbol")

    # Get options
    use_ml = request.form.get('use_ml', 'true') == 'true'
    execute_trades = request.form.get('execute_trades', 'false') == 'true'

    # Create a unique ID for this integrated system session
    system_id = f"system_{int(time.time())}"
    session_id = datetime.now().strftime('%Y%m%d%H%M%S')

    # Create a queue for this integrated system session
    message_queue = Queue()
    integrated_queues[system_id] = message_queue

    # Start the integrated system in a background thread
    system_thread = Thread(
        target=run_integrated_system_with_updates, 
        args=(tickers, use_ml, execute_trades, system_id, message_queue, session_id)
    )
    system_thread.daemon = True
    system_thread.start()

    # Redirect to the progress page
    return redirect(url_for('integrated_progress', system_id=system_id))

@app.route('/backtest')
def backtest():
    """Show the backtesting interface"""
    return render_template('backtest.html')

@app.route('/backtest_progress')
def backtest_progress():
    """Show the backtest progress page"""
    backtest_id = request.args.get('backtest_id')
    if not backtest_id or backtest_id not in backtest_queues:
        return redirect(url_for('backtest'))

    return render_template('backtest_progress.html')

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

def run_backtest_with_updates(tickers, strategy, start_date, end_date, days, custom_transaction_cost, backtest_id, message_queue, session_id):
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
            custom_transaction_cost=custom_transaction_cost
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

@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    """Run a backtest with the provided parameters"""
    # Get ticker symbols from form
    ticker_input = request.form.get('tickers', '')
    tickers = [ticker.strip() for ticker in ticker_input.split(',') if ticker.strip()]

    if not tickers:
        return render_template('backtest.html', error="Please enter at least one ticker symbol")

    # Get backtest parameters
    strategy = request.form.get('strategy', 'ml')
    start_date = request.form.get('start_date', '')
    end_date = request.form.get('end_date', '')
    days = int(request.form.get('days', '180'))

    # Get custom transaction cost if provided
    custom_transaction_cost = request.form.get('custom_transaction_cost', '')
    if custom_transaction_cost:
        try:
            custom_transaction_cost = float(custom_transaction_cost)
        except ValueError:
            custom_transaction_cost = None
    else:
        custom_transaction_cost = None

    # Create a unique ID for this backtest session
    backtest_id = f"backtest_{int(time.time())}"
    session_id = datetime.now().strftime('%Y%m%d%H%M%S')

    # Create a queue for this backtest session
    message_queue = Queue()
    backtest_queues[backtest_id] = message_queue

    # Start the backtest in a background thread
    backtest_thread = Thread(
        target=run_backtest_with_updates, 
        args=(tickers, strategy, start_date, end_date, days, custom_transaction_cost, backtest_id, message_queue, session_id)
    )
    backtest_thread.daemon = True
    backtest_thread.start()

    # Redirect to the progress page
    return redirect(url_for('backtest_progress', backtest_id=backtest_id))

@app.route('/backtest_results')
def backtest_results():
    """Show the backtest results page"""
    backtest_id = request.args.get('backtest_id')

    if not backtest_id or f'backtest_result_{backtest_id}' not in app.config:
        return redirect(url_for('backtest'))

    # Get the template data
    template_data = app.config[f'backtest_result_{backtest_id}']

    # Clean up the queue
    if backtest_id in backtest_queues:
        del backtest_queues[backtest_id]

    return render_template('backtest_results.html', **template_data)

@app.route('/portfolio')
def portfolio():
    """Show the portfolio management interface"""
    # Try to load existing portfolio
    try:
        system = TradingSystem.load_system_state()
        portfolio_summary = system.portfolio_manager.get_portfolio_summary()
        positions = system.portfolio_manager.positions

        return render_template('portfolio.html', 
                              portfolio_summary=portfolio_summary,
                              positions=positions)
    except Exception as e:
        # No existing portfolio, show empty interface
        return render_template('portfolio.html', error=f"No portfolio data available: {str(e)}")

@app.route('/documentation')
@app.route('/documentation/<doc_type>')
def documentation(doc_type='readme'):
    """Display documentation pages with rich text formatting"""
    if doc_type == 'readme':
        file_path = 'README.md'
        doc_title = 'README - TradeSmart Analytics'
    elif doc_type == 'logic_flow':
        file_path = os.path.join('docs', 'Logic_Flow.md')
        doc_title = 'Integrated Trading System Logic Flow'
    elif doc_type == 'system_architecture':
        # For system architecture, we render a dedicated template
        return render_template('system_architecture.html')
    else:
        return render_template('error.html', error="Documentation not found")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Convert markdown to HTML
        html_content = markdown.markdown(content, extensions=['tables', 'fenced_code'])

        return render_template('documentation.html', 
                              doc_type=doc_type,
                              doc_title=doc_title,
                              doc_content=html_content)
    except Exception as e:
        return render_template('error.html', error=f"Error loading documentation: {str(e)}")

@app.route('/healthcheck')
def healthcheck():
    """Simple endpoint to check if the API is functioning"""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

@app.route('/templates/<path:filename>')
def templates(filename):
    """Serve files from the templates directory"""
    return send_from_directory('templates', filename)

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

@app.template_filter('format_number')
def format_number_filter(value):
    """Format a number with commas as thousands separators"""
    return "{:,}".format(value)

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
