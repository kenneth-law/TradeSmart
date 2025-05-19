"""
Market Data Module

This module contains functions for retrieving broader market data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from modules.data_retrieval import get_stock_history, get_stock_info
from modules.utils import log_message

def get_sector_performance():
    """
    Retrieves and analyzes sector performance data.
    
    Returns:
        dict: Dictionary containing sector performance data
    """
    try:
        # Define sector ETFs to track
        sector_etfs = {
            "XLK": "Technology",
            "XLF": "Financials",
            "XLV": "Healthcare",
            "XLP": "Consumer Staples",
            "XLY": "Consumer Discretionary",
            "XLE": "Energy",
            "XLB": "Materials",
            "XLI": "Industrials",
            "XLU": "Utilities",
            "XLRE": "Real Estate",
            "XLC": "Communication Services"
        }
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 30 days of data
        
        # Format dates
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Get data for each sector ETF
        sector_data = []
        
        for ticker, sector_name in sector_etfs.items():
            try:
                # Get historical data
                hist = get_stock_history(ticker, start_date_str, end_date_str, "1d")
                
                if len(hist) > 0:
                    # Calculate returns
                    current_price = hist['Close'].iloc[-1]
                    prev_day_price = hist['Close'].iloc[-2] if len(hist) > 1 else hist['Close'].iloc[0]
                    week_ago_price = hist['Close'].iloc[-5] if len(hist) >= 5 else hist['Close'].iloc[0]
                    month_ago_price = hist['Close'].iloc[0]
                    
                    day_change_pct = ((current_price / prev_day_price) - 1) * 100
                    week_change_pct = ((current_price / week_ago_price) - 1) * 100
                    month_change_pct = ((current_price / month_ago_price) - 1) * 100
                    
                    # Calculate momentum (rate of change)
                    momentum = day_change_pct - (week_change_pct / 5)  # Daily momentum vs 5-day average
                    
                    # Determine trend
                    if month_change_pct > 5 and week_change_pct > 1:
                        trend = "Strong Uptrend"
                    elif month_change_pct > 2 and week_change_pct > 0:
                        trend = "Uptrend"
                    elif month_change_pct < -5 and week_change_pct < -1:
                        trend = "Strong Downtrend"
                    elif month_change_pct < -2 and week_change_pct < 0:
                        trend = "Downtrend"
                    else:
                        trend = "Sideways"
                    
                    sector_data.append({
                        "ticker": ticker,
                        "sector": sector_name,
                        "current_price": current_price,
                        "day_change_pct": day_change_pct,
                        "week_change_pct": week_change_pct,
                        "month_change_pct": month_change_pct,
                        "momentum": momentum,
                        "trend": trend
                    })
            except Exception as e:
                log_message(f"Error retrieving data for sector ETF {ticker}: {str(e)}")
                continue
        
        # Sort sectors by daily performance
        sorted_sectors = sorted(sector_data, key=lambda x: x['day_change_pct'], reverse=True)
        
        # Calculate market breadth based on sector performance
        sectors_up = sum(1 for s in sector_data if s['day_change_pct'] > 0)
        sectors_down = sum(1 for s in sector_data if s['day_change_pct'] < 0)
        
        # Calculate average sector performance
        avg_day_change = np.mean([s['day_change_pct'] for s in sector_data]) if sector_data else 0
        avg_week_change = np.mean([s['week_change_pct'] for s in sector_data]) if sector_data else 0
        avg_month_change = np.mean([s['month_change_pct'] for s in sector_data]) if sector_data else 0
        
        # Determine overall market trend
        if avg_month_change > 3 and avg_week_change > 0.5:
            market_trend = "Bullish"
        elif avg_month_change < -3 and avg_week_change < -0.5:
            market_trend = "Bearish"
        elif avg_day_change > 0.5 and avg_week_change > 0:
            market_trend = "Short-term Bullish"
        elif avg_day_change < -0.5 and avg_week_change < 0:
            market_trend = "Short-term Bearish"
        else:
            market_trend = "Neutral"
        
        return {
            "sectors": sorted_sectors,
            "sectors_up": sectors_up,
            "sectors_down": sectors_down,
            "avg_day_change": avg_day_change,
            "avg_week_change": avg_week_change,
            "avg_month_change": avg_month_change,
            "market_trend": market_trend,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        return {"error": str(e)}

def get_market_breadth():
    """
    Analyzes market breadth using index components and technical indicators.
    
    Returns:
        dict: Dictionary containing market breadth data
    """
    try:
        # Define indices to analyze
        indices = {
            "SPY": "S&P 500",
            "QQQ": "NASDAQ-100",
            "DIA": "Dow Jones Industrial Average",
            "IWM": "Russell 2000"
        }
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Format dates
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Get data for each index
        index_data = []
        
        for ticker, index_name in indices.items():
            try:
                # Get historical data
                hist = get_stock_history(ticker, start_date_str, end_date_str, "1d")
                
                if len(hist) > 0:
                    # Calculate returns
                    current_price = hist['Close'].iloc[-1]
                    prev_day_price = hist['Close'].iloc[-2] if len(hist) > 1 else hist['Close'].iloc[0]
                    week_ago_price = hist['Close'].iloc[-5] if len(hist) >= 5 else hist['Close'].iloc[0]
                    month_ago_price = hist['Close'].iloc[0]
                    
                    day_change_pct = ((current_price / prev_day_price) - 1) * 100
                    week_change_pct = ((current_price / week_ago_price) - 1) * 100
                    month_change_pct = ((current_price / month_ago_price) - 1) * 100
                    
                    # Calculate technical indicators
                    # RSI
                    delta = hist['Close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
                    
                    # Moving averages
                    hist['MA50'] = hist['Close'].rolling(window=50).mean()
                    hist['MA200'] = hist['Close'].rolling(window=200).mean()
                    
                    # Check if price is above key moving averages
                    ma50_available = not pd.isna(hist['MA50'].iloc[-1]) if len(hist['MA50']) > 0 else False
                    ma200_available = not pd.isna(hist['MA200'].iloc[-1]) if len(hist['MA200']) > 0 else False
                    
                    above_ma50 = current_price > hist['MA50'].iloc[-1] if ma50_available else None
                    above_ma200 = current_price > hist['MA200'].iloc[-1] if ma200_available else None
                    
                    # Determine trend
                    if month_change_pct > 5 and week_change_pct > 1:
                        trend = "Strong Uptrend"
                    elif month_change_pct > 2 and week_change_pct > 0:
                        trend = "Uptrend"
                    elif month_change_pct < -5 and week_change_pct < -1:
                        trend = "Strong Downtrend"
                    elif month_change_pct < -2 and week_change_pct < 0:
                        trend = "Downtrend"
                    else:
                        trend = "Sideways"
                    
                    index_data.append({
                        "ticker": ticker,
                        "index": index_name,
                        "current_price": current_price,
                        "day_change_pct": day_change_pct,
                        "week_change_pct": week_change_pct,
                        "month_change_pct": month_change_pct,
                        "rsi": current_rsi,
                        "above_ma50": above_ma50,
                        "above_ma200": above_ma200,
                        "trend": trend
                    })
            except Exception as e:
                log_message(f"Error retrieving data for index {ticker}: {str(e)}")
                continue
        
        # Calculate overall market breadth
        indices_up = sum(1 for idx in index_data if idx['day_change_pct'] > 0)
        indices_down = sum(1 for idx in index_data if idx['day_change_pct'] < 0)
        
        # Calculate average index performance
        avg_day_change = np.mean([idx['day_change_pct'] for idx in index_data]) if index_data else 0
        avg_week_change = np.mean([idx['week_change_pct'] for idx in index_data]) if index_data else 0
        avg_month_change = np.mean([idx['month_change_pct'] for idx in index_data]) if index_data else 0
        avg_rsi = np.mean([idx['rsi'] for idx in index_data]) if index_data else 50
        
        # Determine market health
        if avg_rsi > 70:
            market_health = "Overbought"
        elif avg_rsi < 30:
            market_health = "Oversold"
        elif avg_rsi > 60 and avg_day_change > 0:
            market_health = "Strong"
        elif avg_rsi < 40 and avg_day_change < 0:
            market_health = "Weak"
        else:
            market_health = "Neutral"
        
        return {
            "indices": index_data,
            "indices_up": indices_up,
            "indices_down": indices_down,
            "avg_day_change": avg_day_change,
            "avg_week_change": avg_week_change,
            "avg_month_change": avg_month_change,
            "avg_rsi": avg_rsi,
            "market_health": market_health,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        return {"error": str(e)}

def get_intraday_data(ticker_symbol, interval='30m', days=5):
    """
    Retrieves intraday price data for a given ticker.
    
    Parameters:
        ticker_symbol (str): The stock ticker symbol
        interval (str): Time interval for data points (e.g., '30m', '1h')
        days (int): Number of days of intraday data to retrieve
        
    Returns:
        dict: Dictionary containing intraday price data
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Get intraday data
        intraday = get_stock_history(ticker_symbol, start_date_str, end_date_str, interval)
        
        if len(intraday) == 0:
            return {"error": "No intraday data available"}
        
        # Format data for output
        timestamps = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in intraday.index]
        opens = intraday['Open'].tolist()
        highs = intraday['High'].tolist()
        lows = intraday['Low'].tolist()
        closes = intraday['Close'].tolist()
        volumes = intraday['Volume'].tolist()
        
        # Calculate VWAP (Volume Weighted Average Price)
        intraday['Typical'] = (intraday['High'] + intraday['Low'] + intraday['Close']) / 3
        intraday['TPV'] = intraday['Typical'] * intraday['Volume']
        intraday['Cumulative_TPV'] = intraday['TPV'].cumsum()
        intraday['Cumulative_Volume'] = intraday['Volume'].cumsum()
        intraday['VWAP'] = intraday['Cumulative_TPV'] / intraday['Cumulative_Volume']
        
        vwap = intraday['VWAP'].tolist()
        
        return {
            "ticker": ticker_symbol,
            "interval": interval,
            "timestamps": timestamps,
            "opens": opens,
            "highs": highs,
            "lows": lows,
            "closes": closes,
            "volumes": volumes,
            "vwap": vwap
        }
    except Exception as e:
        return {"error": str(e)}