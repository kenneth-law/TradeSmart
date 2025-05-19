"""
Analysis and Reporting Module

This module contains functions for analyzing stocks and formatting results.
"""

import time
from datetime import datetime
import pandas as pd
from modules.technical_analysis import get_stock_data
from modules.utils import log_message

def analyze_stocks(ticker_list, delay=1):
    """
    Analyzes a list of stock tickers to identify day trading opportunities. This function retrieves stock
    data for the provided tickers, applies a day trading strategy, and ranks them based on their day
    trading score. Tickers that encounter errors during data fetching will be recorded separately.

    Parameters:
    ticker_list: list of str
        A list of stock ticker symbols to analyze.
    delay: int, optional
        The delay between processing each ticker, specified in seconds. Default is 1 second.

    Returns:
    tuple
        A tuple containing two elements:
        - ranked_stocks: list of dict
            A list of dictionaries representing stock data, sorted by their day trading score
            in descending order.
        - failed_tickers: list of str
            A list of ticker symbols for which the fetching of stock data failed.

    Raises:
    None
    """
    stock_analysis = []
    failed_tickers = []

    log_message(f"Analyzing {len(ticker_list)} stocks for day trading opportunities...")

    for i, ticker_symbol in enumerate(ticker_list):
        log_message(f"Processing {i+1}/{len(ticker_list)}: {ticker_symbol}")

        stock_data, error = get_stock_data(ticker_symbol)

        if error:
            log_message(f"!!! {error}")
            failed_tickers.append(ticker_symbol)
            continue

        stock_analysis.append(stock_data)

        # Print brief update
        log_message(f"  {stock_data['ticker']}: {stock_data['day_trading_strategy']} (Score: {stock_data['day_trading_score']:.1f})")

        time.sleep(0.05)  # Prevent API rate limits

    # Sort by day trading score (higher is better)
    ranked_stocks = sorted(stock_analysis, key=lambda x: x['day_trading_score'], reverse=True)

    return ranked_stocks, failed_tickers

def format_results(ranked_stocks, failed_tickers=None, top_n=20):
    """
    Formats the analysis results for ranked stocks showcasing day trading opportunities
    organized by strategy, highest volatility, and strongest news sentiment. The output
    returns a detailed string representation tailored for readability and thorough analysis.
    If no stocks are provided, it returns a fallback message. Optionally highlights any
    tickers for which data retrieval failed.

    Parameters:
        ranked_stocks (list[dict]): A list of dictionaries where each dictionary represents
            a stock with relevant information such as 'ticker', 'company_name',
            'day_trading_strategy', 'current_price', 'day_trading_score', 'atr_pct',
            'rsi7', 'return_1d', 'macd_trend', 'news_sentiment_label',
            'news_sentiment_score', 'volume_ratio', and 'strategy_details'.
        failed_tickers (list[str], optional): A list of string tickers for which data retrieval failed.
            Defaults to None.
        top_n (int, optional): Maximum number of stocks to display per strategy category.
            Defaults to 20.

    Returns:
        str: A formatted string summarizing day trading opportunities, highlighting
        top-ranked stocks by strategy, volatility, and news sentiment, alongside
        any missing data points.
    """
    if not ranked_stocks:
        return "No stocks to display."

    results = "\n======== DAY TRADING OPPORTUNITIES ========\n"
    results += f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

    # Group by strategy
    categories = {
        "Strong Buy": [],
        "Buy": [],
        "Neutral/Watch": [],
        "Sell": [],
        "Strong Sell": []
    }

    for stock in ranked_stocks:
        categories[stock['day_trading_strategy']].append(stock)

    # Print each category (limiting to top_n stocks per category)
    for category, stocks in categories.items():
        if stocks:
            stocks = stocks[:top_n]  # Limit to top_n per category
            results += f"\n--- {category.upper()} ---\n"
            for stock in stocks:
                results += f"- {stock['ticker']} - {stock['company_name']} (${stock['current_price']:.2f})\n"
                results += f"  Day Trading Score: {stock['day_trading_score']:.1f}/100 | ATR: {stock['atr_pct']:.2f}% | RSI7: {stock['rsi7']:.1f}\n"
                results += f"  Technical Score: {stock['technical_score']:.1f} | Long-Term Score: {stock['long_term_score']:.1f}\n"
                results += f"  1-Day Return: {stock['return_1d']:.2f}% | 30-Day: {stock.get('return_30d', 0):.2f}% | 90-Day: {stock.get('return_90d', 0):.2f}%\n"
                results += f"  MACD: {stock['macd_trend'].upper()} | Stochastic: {stock.get('stoch_trend', 'N/A').upper()} | OBV: {stock.get('obv_trend', 'N/A').upper()}\n"
                results += f"  News Sentiment: {stock['news_sentiment_label']} | Volume: {stock['volume_ratio']:.2f}x avg\n"
                results += f"  Strategy: {stock['strategy_details']}\n\n"

    results += f"\n\nTOP 5 HIGHEST VOLATILITY OPPORTUNITIES (ATR%):\n"
    volatile_stocks = sorted(ranked_stocks, key=lambda x: x['atr_pct'], reverse=True)[:5]
    for stock in volatile_stocks:
        results += f"- {stock['ticker']}: ATR {stock['atr_pct']:.2f}% | Score: {stock['day_trading_score']:.1f} | Strategy: {stock['day_trading_strategy']}\n"

    results += f"\n\nTOP 5 STRONGEST NEWS SENTIMENT:\n"
    sentiment_stocks = sorted(ranked_stocks, key=lambda x: abs(x['news_sentiment_score']), reverse=True)[:5]
    for stock in sentiment_stocks:
        sentiment_direction = "POSITIVE" if stock['news_sentiment_score'] > 0 else "NEGATIVE"
        results += f"- {stock['ticker']}: {sentiment_direction} ({stock['news_sentiment_score']:.2f}) | Score: {stock['day_trading_score']:.1f} | Strategy: {stock['day_trading_strategy']}\n"

    if failed_tickers:
        results += f"\n\nFAILED TO RETRIEVE DATA FOR {len(failed_tickers)} TICKERS:\n"
        results += ", ".join(failed_tickers)

    return results

def save_to_csv(ranked_stocks, filename="day_trading_opportunities.csv"):
    """
    Saves the ranked stocks data to a CSV file.

    Parameters:
        ranked_stocks (list): List of dictionaries containing stock data
        filename (str): Name of the CSV file to save

    Returns:
        str: Path to the saved file
    """
    if not ranked_stocks:
        return "No data to save."

    # Convert to DataFrame
    df = pd.DataFrame(ranked_stocks)

    # Save to CSV
    df.to_csv(filename, index=False)

    return f"Data saved to {filename}"

def generate_watchlist(ranked_stocks, min_score=65, max_stocks=10):
    """
    Generates a watchlist of top-ranked stocks that meet a minimum score threshold.

    Parameters:
        ranked_stocks (list): List of dictionaries containing stock data
        min_score (float): Minimum day trading score to include in watchlist
        max_stocks (int): Maximum number of stocks to include in watchlist

    Returns:
        list: List of dictionaries containing watchlist stocks
    """
    if not ranked_stocks:
        return []

    # Filter by minimum score
    watchlist_candidates = [stock for stock in ranked_stocks if stock['day_trading_score'] >= min_score]

    # Sort by score (highest first)
    watchlist_candidates = sorted(watchlist_candidates, key=lambda x: x['day_trading_score'], reverse=True)

    # Limit to max_stocks
    watchlist = watchlist_candidates[:max_stocks]

    return watchlist

def format_stock_for_json(stock_data):
    """
    Formats stock data for JSON serialization by handling non-serializable types.

    Parameters:
        stock_data (dict): Dictionary containing stock data

    Returns:
        dict: Dictionary with JSON-serializable values
    """
    # Create a copy to avoid modifying the original
    result = {}

    for key, value in stock_data.items():
        # Handle numpy types
        if hasattr(value, 'item'):
            result[key] = value.item()
        # Handle other non-serializable types
        elif isinstance(value, (datetime, pd.Timestamp)):
            result[key] = value.isoformat()
        else:
            result[key] = value

    return result
