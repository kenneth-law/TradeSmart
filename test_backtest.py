"""
Test script to verify that the backtester works with the updated transaction cost model.
"""

from modules.backtesting import Backtester
from datetime import datetime, timedelta

def simple_strategy(data, context):
    """
    A simple buy and hold strategy for testing.
    """
    signals = {}
    
    # Buy on the first day
    if context['day_index'] == 0:
        for ticker in data:
            signals[ticker] = 'BUY'
    
    # Sell on the last day
    elif context['day_index'] == context['total_days'] - 1:
        for ticker in context['portfolio']['positions']:
            signals[ticker] = 'SELL'
            
    return signals

def test_backtester():
    """
    Test the backtester with different transaction cost models.
    """
    # Test parameters
    tickers = ['AAPL', 'MSFT']
    # Use historical data from 2023
    end_date = '2023-12-31'
    start_date = '2023-12-01'
    
    print("Testing Backtester with Different Transaction Cost Models")
    print("=" * 70)
    
    # Test with default transaction cost model
    print("\nTesting with default transaction cost model:")
    print("-" * 50)
    backtester = Backtester(initial_capital=10000.0)
    result = backtester.run_backtest(
        strategy=simple_strategy,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date
    )
    print(result.generate_report())
    
    # Test with fixed transaction cost model
    print("\nTesting with fixed transaction cost model ($10):")
    print("-" * 50)
    backtester = Backtester(initial_capital=10000.0, custom_transaction_cost=10.0)
    result = backtester.run_backtest(
        strategy=simple_strategy,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date
    )
    print(result.generate_report())
    
    # Test with percentage transaction cost model (1%)
    print("\nTesting with percentage transaction cost model (1%):")
    print("-" * 50)
    backtester = Backtester(initial_capital=10000.0, custom_transaction_cost="1%")
    result = backtester.run_backtest(
        strategy=simple_strategy,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date
    )
    print(result.generate_report())

if __name__ == "__main__":
    test_backtester()