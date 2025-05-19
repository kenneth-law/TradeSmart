"""
Visualization Module

This module contains functions for preparing data for charts and visualizations.
"""

import pandas as pd
from datetime import datetime, timedelta
from modules.data_retrieval import get_stock_history
from modules.technical_analysis import get_stock_data, calculate_score_contribution

def get_historical_data_for_chart(ticker_symbol, days=30):
    """
    Retrieves historical stock data for charting purposes.
    
    Parameters:
        ticker_symbol (str): The stock ticker symbol
        days (int): Number of days of historical data to retrieve
        
    Returns:
        dict: Dictionary containing historical data formatted for charting
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Get historical data
        hist = get_stock_history(ticker_symbol, start_date_str, end_date_str, "1d")
        
        if len(hist) == 0:
            return {"error": "No historical data available"}
        
        # Format data for chart
        dates = [d.strftime('%Y-%m-%d') for d in hist.index]
        prices = hist['Close'].tolist()
        volumes = hist['Volume'].tolist()
        
        # Calculate moving averages
        hist['MA5'] = hist['Close'].rolling(window=5).mean()
        hist['MA20'] = hist['Close'].rolling(window=20).mean()
        
        ma5 = hist['MA5'].tolist()
        ma20 = hist['MA20'].tolist()
        
        return {
            "ticker": ticker_symbol,
            "dates": dates,
            "prices": prices,
            "volumes": volumes,
            "ma5": ma5,
            "ma20": ma20
        }
    except Exception as e:
        return {"error": str(e)}

def get_detailed_stock_metrics(stock_data):
    """
    Extracts and formats detailed metrics from stock data for visualization.
    
    Parameters:
        stock_data (dict): Dictionary containing stock data
        
    Returns:
        dict: Dictionary with formatted metrics for visualization
    """
    if not stock_data:
        return {"error": "No stock data provided"}
    
    # Extract key metrics
    metrics = {
        "ticker": stock_data.get('ticker', ''),
        "company_name": stock_data.get('company_name', ''),
        "current_price": stock_data.get('current_price', 0),
        "day_trading_score": stock_data.get('day_trading_score', 0),
        "strategy": stock_data.get('day_trading_strategy', ''),
        "technical": {
            "rsi7": stock_data.get('rsi7', 0),
            "rsi14": stock_data.get('rsi14', 0),
            "macd": stock_data.get('macd', 0),
            "macd_signal": stock_data.get('macd_signal', 0),
            "macd_hist": stock_data.get('macd_hist', 0),
            "macd_trend": stock_data.get('macd_trend', ''),
            "bb_position": stock_data.get('bb_position', 0),
            "above_ma5": stock_data.get('above_ma5', False),
            "above_ma10": stock_data.get('above_ma10', False),
            "above_ma20": stock_data.get('above_ma20', False),
        },
        "volatility": {
            "atr": stock_data.get('atr', 0),
            "atr_pct": stock_data.get('atr_pct', 0),
            "avg_intraday_range": stock_data.get('avg_intraday_range', 0),
            "gap_ups_5d": stock_data.get('gap_ups_5d', 0),
            "gap_downs_5d": stock_data.get('gap_downs_5d', 0),
        },
        "momentum": {
            "return_1d": stock_data.get('return_1d', 0),
            "return_3d": stock_data.get('return_3d', 0),
            "return_5d": stock_data.get('return_5d', 0),
        },
        "volume": {
            "volume_ratio": stock_data.get('volume_ratio', 0),
        },
        "sentiment": {
            "news_sentiment_score": stock_data.get('news_sentiment_score', 0),
            "news_sentiment_label": stock_data.get('news_sentiment_label', ''),
        }
    }
    
    # Calculate score contributions
    score_contributions = calculate_score_contribution(stock_data)
    metrics["score_contributions"] = score_contributions
    
    return metrics

def prepare_price_chart_data(ticker_symbol, days=30):
    """
    Prepares comprehensive price chart data including technical indicators.
    
    Parameters:
        ticker_symbol (str): The stock ticker symbol
        days (int): Number of days of historical data to retrieve
        
    Returns:
        dict: Dictionary containing chart data with technical indicators
    """
    try:
        # Get basic historical data
        chart_data = get_historical_data_for_chart(ticker_symbol, days)
        
        if "error" in chart_data:
            return chart_data
        
        # Get current stock data for additional metrics
        stock_data, error = get_stock_data(ticker_symbol)
        
        if error:
            chart_data["error_details"] = error
            return chart_data
        
        # Add current metrics
        chart_data["current_metrics"] = {
            "price": stock_data.get('current_price', 0),
            "day_trading_score": stock_data.get('day_trading_score', 0),
            "strategy": stock_data.get('day_trading_strategy', ''),
            "rsi7": stock_data.get('rsi7', 0),
            "macd_trend": stock_data.get('macd_trend', ''),
            "atr_pct": stock_data.get('atr_pct', 0),
            "volume_ratio": stock_data.get('volume_ratio', 0),
            "news_sentiment": stock_data.get('news_sentiment_score', 0),
        }
        
        return chart_data
    except Exception as e:
        return {"error": str(e)}

def get_stock_comparison_data(ticker_symbols, metric='day_trading_score'):
    """
    Retrieves data for comparing multiple stocks based on a specified metric.
    
    Parameters:
        ticker_symbols (list): List of stock ticker symbols to compare
        metric (str): The metric to use for comparison
        
    Returns:
        dict: Dictionary containing comparison data
    """
    if not ticker_symbols:
        return {"error": "No ticker symbols provided"}
    
    comparison_data = {
        "tickers": [],
        "values": [],
        "labels": [],
        "colors": []
    }
    
    for ticker in ticker_symbols:
        stock_data, error = get_stock_data(ticker)
        
        if error:
            continue
            
        # Add data for this ticker
        comparison_data["tickers"].append(ticker)
        
        # Get the metric value
        value = stock_data.get(metric, 0)
        comparison_data["values"].append(value)
        
        # Create label
        label = f"{ticker} - {stock_data.get('company_name', '')}"
        comparison_data["labels"].append(label)
        
        # Determine color based on strategy
        strategy = stock_data.get('day_trading_strategy', '')
        if strategy == "Strong Buy":
            color = "green"
        elif strategy == "Buy":
            color = "lightgreen"
        elif strategy == "Neutral/Watch":
            color = "gray"
        elif strategy == "Sell":
            color = "pink"
        elif strategy == "Strong Sell":
            color = "red"
        else:
            color = "blue"
            
        comparison_data["colors"].append(color)
    
    return comparison_data