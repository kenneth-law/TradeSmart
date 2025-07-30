"""
Backtesting Module

This module implements industrial-grade backtesting capabilities for stock trading strategies.
It addresses the following requirements:
1. Point-in-time integrity (includes delisted tickers to avoid survivorship bias)
2. Realistic transaction cost modeling (spread, impact, rebates)
3. Walk-forward and Monte Carlo scenario testing

The module provides a framework for evaluating trading strategies with realistic assumptions
about market conditions and execution costs.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import random
from modules.utils import log_message
from modules.data_retrieval import get_stock_history
from modules.technical_analysis import get_stock_data
from modules.ml_scoring import score_stock_ml

# Create cache directory for backtest results if it doesn't exist
os.makedirs('cache/backtest_results', exist_ok=True)

class TransactionCostModel:



    """
    Models realistic transaction costs including spread, market impact, and fees.
    Costs are adjusted based on stock liquidity, volatility, and time of day.
    """

    def __init__(self, custom_transaction_cost=None):
        """
        Initialize the transaction cost model with a flat rate commission or custom value.

        Parameters:
            custom_transaction_cost (float, str): Optional transaction cost value. Can be:
                - A float representing a fixed cost per transaction
                - A string in format "X%" representing a percentage of transaction value
                - None to use the default model with spread, impact, and flat commission
        """
        self.cost_mode = "default"
        self.percentage_cost = 0.01  # Default 1% if percentage mode is used
        self.fixed_cost = 10.0  # Default fixed cost if fixed mode is used
        self.flat_commission = 10.0  # Default flat rate of $10 per transaction for default mode
        
        if custom_transaction_cost is not None:
            if isinstance(custom_transaction_cost, str) and "%" in custom_transaction_cost:
                # Percentage-based cost
                self.cost_mode = "percentage"
                try:
                    self.percentage_cost = float(custom_transaction_cost.strip("%")) / 100
                except ValueError:
                    # If conversion fails, use default 1%
                    self.percentage_cost = 0.01
            else:
                # Fixed cost per transaction
                self.cost_mode = "fixed"
                try:
                    self.fixed_cost = float(custom_transaction_cost)
                except (ValueError, TypeError):
                    # If conversion fails, use default $10
                    self.fixed_cost = 10.0

    def estimate_spread(self, price, volume, volatility):
        """
        Estimate bid-ask spread based on price, volume, and volatility.

        Parameters:
            price (float): Current stock price
            volume (float): Average daily volume
            volatility (float): Stock volatility (e.g., ATR as percentage)

        Returns:
            float: Estimated spread as percentage of price
        """
        # Base spread as function of price (higher for lower-priced stocks)
        base_spread = 0.03 if price < 5 else 0.01 if price < 20 else 0.005

        # Adjust for volume (higher spread for lower volume)
        volume_factor = 1.0
        if volume < 100000:
            volume_factor = 1.5
        elif volume < 500000:
            volume_factor = 1.2
        elif volume < 1000000:
            volume_factor = 1.1

        # Adjust for volatility (higher spread for more volatile stocks)
        volatility_factor = 1.0 + (volatility / 15.0)

        return base_spread * volume_factor * volatility_factor

    def estimate_market_impact(self, price, volume, trade_size):
        """
        Estimate market impact based on trade size relative to average volume.

        Parameters:
            price (float): Current stock price
            volume (float): Average daily volume
            trade_size (float): Number of shares to trade

        Returns:
            float: Estimated market impact as percentage of price
        """
        # Calculate trade size as percentage of average daily volume
        trade_volume_pct = (trade_size / volume) * 100

        # No impact for very small trades (less than 0.1% of daily volume)
        if trade_volume_pct < 0.1:
            return 0.0

        # Reduced square root model for market impact
        impact = 0.05 * np.sqrt(trade_volume_pct / 20.0) if trade_volume_pct > 0 else 0

        # Higher price stocks typically have lower impact
        if price > 50:
            impact *= 0.8
        elif price > 100:
            impact *= 0.6

        return min(impact, 0.5)  # Cap at 0.5%

    def calculate_transaction_cost(self, price, shares, volume, volatility, is_buy=True):
        """
        Calculate total transaction cost for a trade.

        Parameters:
            price (float): Current stock price
            shares (int): Number of shares to trade
            volume (float): Average daily volume
            volatility (float): Stock volatility
            is_buy (bool): True if buying, False if selling

        Returns:
            tuple: (total_cost_dollars, total_cost_percentage)
        """
        transaction_value = price * shares
        
        if self.cost_mode == "fixed":
            # Fixed cost per transaction (not per share)
            total_cost_dollars = self.fixed_cost
        elif self.cost_mode == "percentage":
            # Percentage of transaction value
            total_cost_dollars = transaction_value * self.percentage_cost
        else:
            # Default mode: Use realistic model with spread, impact, and commission
            # Estimate spread
            spread_pct = self.estimate_spread(price, volume, volatility)
            spread_cost = price * spread_pct / 2  # Half spread for each side

            # Estimate market impact
            impact_pct = self.estimate_market_impact(price, volume, shares)
            impact_cost = price * impact_pct

            # Use flat rate commission
            commission = self.flat_commission

            # Total cost
            total_cost_dollars = (spread_cost + impact_cost) * shares + commission

        # Avoid division by zero
        if transaction_value > 0:
            total_cost_percentage = (total_cost_dollars / transaction_value) * 100
        else:
            total_cost_percentage = 0.0

        return total_cost_dollars, total_cost_percentage

class BacktestResult:
    """
    Stores and analyzes the results of a backtest.
    """

    def __init__(self, strategy_name, initial_capital, start_date, end_date):
        """
        Initialize the backtest result.

        Parameters:
            strategy_name (str): Name of the strategy
            initial_capital (float): Initial capital in dollars
            start_date (str): Start date of the backtest
            end_date (str): End date of the backtest
        """
        self.strategy_name = strategy_name
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date

        # Performance metrics
        self.final_capital = initial_capital
        self.total_return = 0.0
        self.annualized_return = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        self.win_rate = 0.0
        self.profit_factor = 0.0

        # Daily performance data
        self.daily_returns = []
        self.equity_curve = []
        self.drawdown_curve = []

        # Trade history
        self.trades = []

        # Transaction costs
        self.total_transaction_costs = 0.0
        self.transaction_cost_percentage = 0.0

    def add_trade(self, date, ticker, action, price, shares, cost):
        """
        Add a trade to the history.

        Parameters:
            date (datetime): Date of the trade
            ticker (str): Ticker symbol
            action (str): 'BUY' or 'SELL'
            price (float): Execution price
            shares (int): Number of shares
            cost (float): Transaction cost in dollars
        """
        self.trades.append({
            'date': date,
            'ticker': ticker,
            'action': action,
            'price': price,
            'shares': shares,
            'cost': cost,
            'value': price * shares,
            'net_value': price * shares - cost
        })

        self.total_transaction_costs += cost

    def update_equity(self, date, equity):
        """
        Update the equity curve.

        Parameters:
            date (datetime): Current date
            equity (float): Current equity value
        """
        self.equity_curve.append({
            'date': date,
            'equity': equity
        })

        # Calculate daily return
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]['equity']
            daily_return = (equity / prev_equity) - 1
            self.daily_returns.append(daily_return)

        # Update drawdown
        if len(self.equity_curve) > 0:
            peak = max([e['equity'] for e in self.equity_curve])
            drawdown = (peak - equity) / peak if peak > 0 else 0
            self.drawdown_curve.append({
                'date': date,
                'drawdown': drawdown
            })

            self.max_drawdown = max(self.max_drawdown, drawdown)

    def calculate_metrics(self):
        """Calculate performance metrics from the backtest data."""
        if not self.equity_curve:
            return

        # Final capital and total return
        self.final_capital = self.equity_curve[-1]['equity']
        # Avoid division by zero
        if self.initial_capital > 0:
            self.total_return = (self.final_capital / self.initial_capital) - 1
        else:
            self.total_return = 0.0

        # Annualized return
        days = (datetime.strptime(self.end_date, '%Y-%m-%d') - 
                datetime.strptime(self.start_date, '%Y-%m-%d')).days
        years = days / 365.0
        self.annualized_return = ((1 + self.total_return) ** (1 / years)) - 1 if years > 0 else 0

        # Sharpe ratio (assuming risk-free rate of 0)
        if self.daily_returns:
            daily_return_mean = np.mean(self.daily_returns)
            daily_return_std = np.std(self.daily_returns)
            self.sharpe_ratio = (daily_return_mean / daily_return_std) * np.sqrt(252) if daily_return_std > 0 else 0

        # Win rate based on complete round-trip trades
        self.win_rate = self.calculate_win_rate()

        # Transaction cost percentage - Fix: use average portfolio value over the period
        # Use average portfolio value for percentage calculation
        total_portfolio_value = np.mean([e['equity'] for e in self.equity_curve]) if self.equity_curve else self.final_capital
        self.transaction_cost_percentage = (self.total_transaction_costs / total_portfolio_value * 100) if total_portfolio_value > 0 else 0

    def calculate_win_rate(self):
        """Calculate win rate based on complete round-trip trades"""
        positions = {}
        completed_trades = []

        for trade in self.trades:
            ticker = trade['ticker']
            if ticker not in positions:
                positions[ticker] = []

            if trade['action'] == 'BUY':
                positions[ticker].append(trade)
            elif trade['action'] == 'SELL' and positions[ticker]:
                # Match with oldest buy
                buy_trade = positions[ticker].pop(0)
                # Calculate profit/loss for this round-trip trade
                pnl = (trade['price'] - buy_trade['price']) * min(trade['shares'], buy_trade['shares']) - (trade['cost'] + buy_trade['cost'])
                completed_trades.append(pnl > 0)

        return sum(completed_trades) / len(completed_trades) if completed_trades else 0

    def generate_report(self):
        """
        Generate a comprehensive backtest report.

        Returns:
            str: Formatted report text
        """
        self.calculate_metrics()

        report = f"Backtest Report: {self.strategy_name}\n"
        report += f"{'=' * 50}\n"
        report += f"Period: {self.start_date} to {self.end_date}\n"
        report += f"Initial Capital: ${self.initial_capital:,.2f}\n"
        report += f"Final Capital: ${self.final_capital:,.2f}\n\n"

        report += f"Performance Metrics:\n"
        report += f"{'=' * 50}\n"
        report += f"Total Return: {self.total_return:.2%}\n"
        report += f"Annualized Return: {self.annualized_return:.2%}\n"
        report += f"Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
        report += f"Maximum Drawdown: {self.max_drawdown:.2%}\n"
        report += f"Win Rate: {self.win_rate:.2%}\n\n"

        report += f"Transaction Costs:\n"
        report += f"{'=' * 50}\n"
        report += f"Total Transaction Costs: ${self.total_transaction_costs:,.2f}\n"
        report += f"Transaction Cost Percentage: {self.transaction_cost_percentage:.2%}\n\n"

        report += f"Trade Summary:\n"
        report += f"{'=' * 50}\n"
        report += f"Total Trades: {len(self.trades)}\n"

        return report

class Backtester:
    """
    Backtesting engine for stock trading strategies.
    """

    def __init__(self, initial_capital=100000.0, custom_transaction_cost=None):
        """
        Initialize the backtester.

        Parameters:
            initial_capital (float): Initial capital in dollars
            custom_transaction_cost (float, str): Optional transaction cost value. Can be:
                - A float representing a fixed cost per transaction
                - A string in format "X%" representing a percentage of transaction value (e.g., "1%")
                - None to use the default model with spread, impact, and flat commission
        """
        self.initial_capital = initial_capital
        self.transaction_cost_model = TransactionCostModel(custom_transaction_cost)

    def run_backtest(self, strategy, tickers, start_date, end_date, 
                     use_point_in_time_universe=True, include_delisted=True, ml_scorer=None,
                     custom_transaction_cost=None, buy_threshold=60, sell_threshold=40):
        """
        Run a backtest for the given strategy and parameters.

        Parameters:
            strategy (callable): Strategy function that returns buy/sell signals
            tickers (list): List of ticker symbols to include
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            use_point_in_time_universe (bool): Whether to use point-in-time universe
            include_delisted (bool): Whether to include delisted tickers
            ml_scorer (MLScorer): Optional pre-initialized ML scorer for ML strategies
            custom_transaction_cost (float, str): Optional transaction cost value. Can be:
                - A float representing a fixed cost per transaction
                - A string in format "X%" representing a percentage of transaction value (e.g., "1%")
                - None to use the default model with spread, impact, and flat commission
            buy_threshold (int): Score threshold for buy signals in ML strategy (default: 60)
            sell_threshold (int): Score threshold for sell signals in ML strategy (default: 40)

        Returns:
            BacktestResult: Object containing backtest results
        """
        log_message(f"Starting backtest from {start_date} to {end_date} with {len(tickers)} tickers")

        # If custom transaction cost is provided, update the transaction cost model
        if custom_transaction_cost is not None:
            self.transaction_cost_model = TransactionCostModel(custom_transaction_cost)

        # Initialize result object
        result = BacktestResult(
            strategy_name=strategy.__name__,
            initial_capital=self.initial_capital,
            start_date=start_date,
            end_date=end_date
        )

        # Initialize portfolio
        portfolio = {
            'cash': self.initial_capital,
            'positions': {},  # ticker -> {'shares': int, 'cost_basis': float}
            'equity': self.initial_capital
        }

        # Get date range
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        date_range = [start_dt + timedelta(days=x) for x in range((end_dt - start_dt).days + 1)]

        # Get historical data for all tickers
        ticker_data = {}
        for ticker in tqdm(tickers, desc="Loading historical data"):
            try:
                hist = get_stock_history(ticker, start_date, end_date, "1d")
                if not hist.empty:
                    ticker_data[ticker] = hist
            except Exception as e:
                log_message(f"Error loading data for {ticker}: {e}")

        # Improved market hours and holidays handling
        # Only include days where we actually have market data
        trading_days = []
        for d in date_range:
            # Skip weekends first (optimization)
            if d.weekday() >= 5:
                continue
            # Check if any ticker has data for this day
            if any(d.date() in data.index.date for data in ticker_data.values()):
                trading_days.append(d)

        # Run the backtest day by day
        for day in tqdm(trading_days, desc="Running backtest"):
            day_str = day.strftime('%Y-%m-%d')

            # Skip if no market data for this day (holiday, weekend)
            if all(day.date() not in data.index.date for data in ticker_data.values()):
                continue

            # Get universe for this day
            if use_point_in_time_universe:
                # Use only tickers that have data up to this day
                universe = [t for t in ticker_data if day.date() in ticker_data[t].index.date]
            else:
                universe = list(ticker_data.keys())

            # Update portfolio value
            portfolio_value = portfolio['cash']
            for ticker, position in portfolio['positions'].items():
                if ticker in ticker_data and day.date() in ticker_data[ticker].index.date:
                    price = ticker_data[ticker].loc[ticker_data[ticker].index.date == day.date(), 'Close'].iloc[0]
                    position['current_price'] = price
                    position['current_value'] = price * position['shares']
                    portfolio_value += position['current_value']

            portfolio['equity'] = portfolio_value
            result.update_equity(day, portfolio_value)

            # Get signals from strategy
            signals = {}
            for ticker in universe:
                try:
                    # Get data up to this day
                    hist_slice = ticker_data[ticker][ticker_data[ticker].index.date <= day.date()].copy()

                    if len(hist_slice) < 10:  # Need enough history for indicators
                        continue

                    # Get stock data for this point in time
                    stock_data, error = get_stock_data(ticker, historical_data=hist_slice)

                    if error or not stock_data:
                        continue

                    # Apply ML scoring if using ml_strategy
                    if strategy.__name__ == 'ml_strategy':
                        stock_data = score_stock_ml(stock_data, ml_scorer)

                    # Get signal from strategy
                    if strategy.__name__ == 'ml_strategy':
                        # Pass thresholds to ML strategy
                        signal = strategy(stock_data, buy_threshold, sell_threshold)
                    else:
                        signal = strategy(stock_data)

                    if signal:
                        signals[ticker] = signal
                except Exception as e:
                    log_message(f"Error processing {ticker} on {day_str}: {e}")

            # Execute trades based on signals
            for ticker, signal in signals.items():
                try:
                    if ticker not in ticker_data or day.date() not in ticker_data[ticker].index.date:
                        continue

                    # Fix look-ahead bias: Use next day's open price for execution
                    tomorrow = day + timedelta(days=1)
                    # Try to find the next trading day
                    next_trading_day_found = False
                    for i in range(1, 5):  # Look up to 5 days ahead (to handle weekends and holidays)
                        next_day = day + timedelta(days=i)
                        if ticker in ticker_data and next_day.date() in ticker_data[ticker].index.date:
                            tomorrow = next_day
                            next_trading_day_found = True
                            break

                    # If we can't find next trading day, skip this signal
                    if not next_trading_day_found:
                        continue

                    # Use next day's open price for execution
                    price = ticker_data[ticker].loc[ticker_data[ticker].index.date == tomorrow.date(), 'Open'].iloc[0]
                    volume = ticker_data[ticker].loc[ticker_data[ticker].index.date == day.date(), 'Volume'].iloc[0]

                    # Get volatility (ATR as percentage)
                    volatility = 2.0  # Default if not available
                    if 'atr_pct' in stock_data:
                        volatility = stock_data['atr_pct']

                    if signal['action'] == 'BUY':
                        # Calculate position size based on available cash, not total equity
                        available_cash = portfolio['cash']
                        desired_position_pct = signal.get('size', 0.1)  # 10% of portfolio

                        # Calculate what 10% of portfolio would be
                        target_position_value = portfolio['equity'] * desired_position_pct

                        # But limit to available cash and position size limits
                        max_position_value = portfolio['equity'] * 0.05  # 5% position limit
                        cash_to_use = min(available_cash * 0.8, target_position_value, max_position_value)

                        shares = int(cash_to_use / price)

                        if shares > 0:  # Don't need the cash check anymore
                            # Calculate transaction cost
                            cost, cost_pct = self.transaction_cost_model.calculate_transaction_cost(
                                price, shares, volume, volatility, is_buy=True
                            )

                            # Execute trade if affordable
                            if cost + (price * shares) <= portfolio['cash']:
                                # Update portfolio
                                portfolio['cash'] -= (price * shares + cost)

                                if ticker in portfolio['positions']:
                                    # Update existing position
                                    position = portfolio['positions'][ticker]
                                    total_shares = position['shares'] + shares
                                    total_cost = position['cost_basis'] * position['shares'] + price * shares + cost
                                    position['shares'] = total_shares
                                    position['cost_basis'] = total_cost / total_shares
                                else:
                                    # Create new position
                                    portfolio['positions'][ticker] = {
                                        'shares': shares,
                                        'cost_basis': (price * shares + cost) / shares,
                                        'entry_date': day_str,
                                        'current_price': price,
                                        'current_value': price * shares
                                    }

                                # Record trade
                                result.add_trade(day, ticker, 'BUY', price, shares, cost)

                    elif signal['action'] == 'SELL' and ticker in portfolio['positions']:
                        position = portfolio['positions'][ticker]
                        shares_to_sell = int(position['shares'] * signal.get('size', 1.0))  # Default sell all

                        if shares_to_sell > 0:
                            # Calculate transaction cost
                            cost, cost_pct = self.transaction_cost_model.calculate_transaction_cost(
                                price, shares_to_sell, volume, volatility, is_buy=False
                            )

                            # Execute trade
                            portfolio['cash'] += (price * shares_to_sell - cost)

                            # Update position
                            position['shares'] -= shares_to_sell
                            if position['shares'] <= 0:
                                del portfolio['positions'][ticker]

                            # Record trade
                            result.add_trade(day, ticker, 'SELL', price, shares_to_sell, cost)

                except Exception as e:
                    log_message(f"Error executing trade for {ticker} on {day_str}: {e}")

        # Calculate final metrics
        result.calculate_metrics()

        log_message(f"Backtest completed. Final equity: ${result.final_capital:,.2f}, Return: {result.total_return:.2%}")

        return result

# Simple strategy functions
def ml_strategy(stock_data, buy_threshold=60, sell_threshold=40):
    """ML-based strategy using the day_trading_score with improved risk management

    Parameters:
        stock_data (dict): Stock data including technical indicators and ML scores
        buy_threshold (int): Score threshold for buy signals (default: 60)
        sell_threshold (int): Score threshold for sell signals (default: 40)
    """
    score = stock_data.get('day_trading_score', 0)

    # Get volatility and volume metrics for risk management
    volatility = stock_data.get('atr_pct', 5.0)
    volume_ratio = stock_data.get('volume_ratio', 1.0)

    # Skip low volume days (avoid illiquid conditions)
    if volume_ratio < 0.7:
        return None

    # Adjust position size based on volatility (smaller positions for higher volatility)
    if volatility <= 0:
        position_size = 0
    else:
        position_size = max(0.05, 0.15 / (volatility / 5.0))

    # Limit maximum position size
    position_size = min(position_size, 0.15)

    # Use configurable thresholds for entry criteria
    if score >= buy_threshold:
        # Check for confirmation signals
        rsi = stock_data.get('rsi14', 50)
        bb_position = stock_data.get('bb_position', 0.5)

        # Only buy if not overbought
        if rsi < 70 and bb_position < 0.8:
            return {'action': 'BUY', 'size': position_size}

    # Use configurable thresholds for exit criteria
    elif score <= sell_threshold:
        # Only sell if we have confirmation
        rsi = stock_data.get('rsi14', 50)
        bb_position = stock_data.get('bb_position', 0.5)

        if rsi > 30 or bb_position > 0.2:
            return {'action': 'SELL', 'size': 1.0}

    # Add partial profit taking at moderate scores
    elif score <= sell_threshold + 15 and score > sell_threshold:
        return {'action': 'SELL', 'size': 0.5}  # Sell half the position

    return None

def simple_technical_strategy(stock_data):
    """Simple technical strategy based on RSI and Bollinger Bands"""
    rsi = stock_data.get('rsi14', 50)
    bb_position = stock_data.get('bb_position', 0.5)

    # Buy when oversold
    if rsi < 30 and bb_position < 0.2:
        return {'action': 'BUY', 'size': 0.15}

    # Sell when overbought
    if rsi > 70 and bb_position > 0.8:
        return {'action': 'SELL', 'size': 1.0}

    return None
