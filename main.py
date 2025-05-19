"""
Stock Analysis Application

This is the main entry point for the stock analysis application.
It imports and uses the modules to provide stock analysis functionality.
"""

import os
import time
import warnings
import logging
from datetime import datetime

# Import modules
from modules.utils import set_message_handler, log_message
from modules.data_retrieval import patch_yfdata_cookie_basic
from modules.technical_analysis import get_stock_data
from modules.analysis_reporting import analyze_stocks, format_results, save_to_csv, generate_watchlist, format_stock_for_json
from modules.visualization import get_historical_data_for_chart, get_detailed_stock_metrics, prepare_price_chart_data, get_stock_comparison_data
from modules.market_data import get_sector_performance, get_market_breadth, get_intraday_data
from modules.news_sentiment import get_news_sentiment_with_timeframes

# Suppress warnings
warnings.filterwarnings('ignore')

# Apply yfinance patch
patch_yfdata_cookie_basic()

def main():
    """
    Main function to demonstrate the stock analysis functionality.
    """
    print("Stock Analysis Application")
    print("=========================")
    
    # Example usage
    ticker_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    
    print(f"\nAnalyzing {len(ticker_list)} stocks...")
    ranked_stocks, failed_tickers = analyze_stocks(ticker_list)
    
    print("\nResults:")
    results = format_results(ranked_stocks, failed_tickers)
    print(results)
    
    # Save results to CSV
    csv_path = save_to_csv(ranked_stocks)
    print(f"\n{csv_path}")
    
    # Generate watchlist
    watchlist = generate_watchlist(ranked_stocks)
    print(f"\nWatchlist ({len(watchlist)} stocks):")
    for stock in watchlist:
        print(f"- {stock['ticker']}: {stock['day_trading_score']:.1f} - {stock['day_trading_strategy']}")
    
    # Get market data
    print("\nGetting market data...")
    sector_data = get_sector_performance()
    market_breadth = get_market_breadth()
    
    print(f"\nMarket Trend: {sector_data.get('market_trend', 'Unknown')}")
    print(f"Market Health: {market_breadth.get('market_health', 'Unknown')}")
    
    # Example of getting detailed data for a single stock
    if ranked_stocks:
        ticker = ranked_stocks[0]['ticker']
        print(f"\nDetailed analysis for {ticker}:")
        
        # Get chart data
        chart_data = prepare_price_chart_data(ticker)
        print(f"Chart data available: {len(chart_data.get('dates', []))} days")
        
        # Get intraday data
        intraday_data = get_intraday_data(ticker)
        print(f"Intraday data available: {len(intraday_data.get('timestamps', []))} periods")

if __name__ == "__main__":
    main()