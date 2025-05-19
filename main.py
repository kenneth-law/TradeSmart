"""
Stock Analysis Application

This is the main entry point for the stock analysis application.
It imports and uses the modules to provide stock analysis functionality.
This version integrates all components (ML scoring, backtesting, portfolio management,
and execution) into a cohesive trading system.
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

# Import integrated trading system
from modules.trading_system import TradingSystem

# Suppress warnings
warnings.filterwarnings('ignore')

# Apply yfinance patch
patch_yfdata_cookie_basic()

def traditional_analysis():
    """
    Run the traditional stock analysis functionality.
    """
    print("\nTraditional Stock Analysis")
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
    print(f"\nResults saved to: {csv_path}")

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

def integrated_trading_system():
    """
    Run the integrated trading system that combines ML scoring, backtesting,
    portfolio management, and execution.
    """
    print("\nIntegrated Trading System")
    print("=======================")

    # Initialize the trading system
    system = TradingSystem(initial_capital=100000.0, market_neutral=True)

    # Define a list of stocks to analyze
    ticker_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]

    print(f"\nRunning complete workflow for {len(ticker_list)} stocks...")

    # Run the complete workflow
    results = system.run_complete_workflow(ticker_list, use_ml=True, execute_trades=False)

    # Print results
    print("\nTrading System Results:")
    print("=" * 50)
    print(f"Stocks analyzed: {results['tickers_analyzed']}")
    print(f"Stocks ranked: {results['ranked_stocks_count']}")
    print(f"Failed tickers: {results['failed_tickers_count']}")
    print(f"Watchlist size: {results['watchlist_count']}")
    print(f"Portfolio value: ${results['portfolio_value']:,.2f}")
    print(f"Cash: ${results['portfolio_cash']:,.2f}")
    print(f"Buy recommendations: {results['buy_recommendations']}")
    print(f"Sell recommendations: {results['sell_recommendations']}")

    # Print backtest results
    print("\nBacktest Results:")
    print("=" * 50)
    print(f"Return: {results['backtest_return']:.2%}")
    print(f"Sharpe ratio: {results['backtest_sharpe']:.2f}")

    # Print watchlist
    print("\nWatchlist:")
    print("=" * 50)
    for i, stock in enumerate(system.watchlist[:5], 1):  # Show top 5
        print(f"{i}. {stock['ticker']} ({stock['company_name']}): {stock['day_trading_score']:.1f} - {stock['day_trading_strategy']}")

    # Print portfolio summary
    portfolio_summary = system.portfolio_manager.get_portfolio_summary()
    print("\nPortfolio Summary:")
    print("=" * 50)
    print(f"Total value: ${portfolio_summary['total_value']:,.2f}")
    print(f"Cash: ${portfolio_summary['cash']:,.2f} ({portfolio_summary['cash_percentage']:.1f}%)")
    print(f"Invested: ${portfolio_summary['invested_value']:,.2f} ({portfolio_summary['invested_percentage']:.1f}%)")
    print(f"Number of positions: {portfolio_summary['num_positions']}")
    print(f"Portfolio beta: {portfolio_summary['portfolio_beta']:.2f}")

    # Print sector exposures
    print("\nSector Exposures:")
    print("=" * 50)
    for sector, data in portfolio_summary['sector_exposures'].items():
        print(f"{sector}: ${data['value']:,.2f} ({data['percentage']:.1f}%)")

    return system

def main():
    """
    Main function to demonstrate both traditional analysis and the integrated trading system.
    """
    print("Stock Analysis and Trading System")
    print("================================")

    # Ask user which mode to run
    print("\nSelect mode:")
    print("1. Traditional Stock Analysis")
    print("2. Integrated Trading System")
    print("3. Both")

    try:
        choice = input("\nEnter choice (1-3): ").strip()

        if choice == '1':
            traditional_analysis()
        elif choice == '2':
            integrated_trading_system()
        elif choice == '3':
            traditional_analysis()
            integrated_trading_system()
        else:
            print("Invalid choice. Running integrated trading system by default.")
            integrated_trading_system()
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Running integrated trading system by default.")
        integrated_trading_system()

if __name__ == "__main__":
    main()
