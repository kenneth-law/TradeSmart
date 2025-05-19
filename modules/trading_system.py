"""
Trading System Integration Module

This module integrates all components of the trading system:
1. ML-based scoring system
2. Backtesting framework
3. Portfolio management
4. Execution algorithms

It provides a unified interface for using these components together in a cohesive workflow.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import logging
from modules.utils import log_message
from modules.ml_scoring import MLScorer, score_stock_ml, collect_training_data
from modules.backtesting import Backtester, BacktestResult, ml_strategy, simple_technical_strategy
from modules.portfolio_management import PortfolioManager
from modules.execution import ExecutionManager, OrderType, OrderSide
from modules.technical_analysis import get_stock_data
from modules.data_retrieval import get_stock_history, get_stock_info

# Create cache directory for the trading system if it doesn't exist
os.makedirs('cache/trading_system', exist_ok=True)

class TradingSystem:
    """
    Integrated trading system that combines ML scoring, backtesting,
    portfolio management, and execution components.
    """
    
    def __init__(self, initial_capital=100000.0, market_neutral=True):
        """
        Initialize the trading system.
        
        Parameters:
            initial_capital (float): Initial capital in dollars
            market_neutral (bool): Whether to target market neutrality
        """
        self.initial_capital = initial_capital
        self.market_neutral = market_neutral
        
        # Initialize components
        self.ml_scorer = MLScorer(model_type='regression')
        self.backtester = Backtester(initial_capital=initial_capital)
        self.portfolio_manager = PortfolioManager(
            initial_capital=initial_capital,
            market_neutral=market_neutral
        )
        self.execution_manager = ExecutionManager()
        
        # System state
        self.is_model_trained = False
        self.last_backtest_result = None
        self.watchlist = []
        self.market_data = {}
        
        log_message("Trading system initialized")
    
    def train_ml_model(self, tickers=None, force=False):
        """
        Train the ML model using historical data.
        
        Parameters:
            tickers (list): List of ticker symbols to use for training
            force (bool): Force retraining even if interval hasn't elapsed
            
        Returns:
            bool: True if training was successful
        """
        if tickers is None:
            # Default to a list of major stocks if none provided
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]
        
        log_message(f"Training ML model with {len(tickers)} tickers")
        
        # Collect training data
        training_data = collect_training_data(tickers, lookback_days=180, prediction_horizon=5)
        
        if not training_data:
            log_message("No training data collected")
            return False
        
        # Train the model
        success = self.ml_scorer.train(training_data, force=force)
        self.is_model_trained = success
        
        return success
    
    def analyze_stocks(self, tickers, use_ml=True):
        """
        Analyze a list of stocks using either ML-based or traditional scoring.
        
        Parameters:
            tickers (list): List of ticker symbols to analyze
            use_ml (bool): Whether to use ML-based scoring
            
        Returns:
            tuple: (ranked_stocks, failed_tickers)
        """
        log_message(f"Analyzing {len(tickers)} stocks with {'ML' if use_ml else 'traditional'} scoring")
        
        ranked_stocks = []
        failed_tickers = []
        
        for ticker in tickers:
            try:
                # Get stock data
                stock_data, error = get_stock_data(ticker)
                
                if error or not stock_data:
                    log_message(f"Error analyzing {ticker}: {error}")
                    failed_tickers.append(ticker)
                    continue
                
                # Apply ML scoring if requested and model is trained
                if use_ml and self.is_model_trained:
                    stock_data = score_stock_ml(stock_data, self.ml_scorer)
                
                # Add to ranked stocks
                ranked_stocks.append(stock_data)
                
                # Update market data for execution
                self.execution_manager.update_market_data(
                    ticker, 
                    stock_data.get('current_price', 0),
                    stock_data.get('volume_ratio', 1) * 100000  # Rough estimate
                )
                
                # Add to market data for portfolio management
                self.market_data[ticker] = {
                    'price': stock_data.get('current_price', 0),
                    'volume': stock_data.get('volume_ratio', 1) * 100000,
                    'sector': stock_data.get('sector', 'Unknown'),
                    'beta': 1.0,  # Default if not available
                    'volatility': stock_data.get('atr_pct', 5.0)
                }
                
            except Exception as e:
                log_message(f"Error processing {ticker}: {str(e)}")
                failed_tickers.append(ticker)
        
        # Sort by score
        if ranked_stocks:
            ranked_stocks = sorted(ranked_stocks, key=lambda x: x.get('day_trading_score', 0), reverse=True)
        
        return ranked_stocks, failed_tickers
    
    def run_backtest(self, tickers, strategy='ml', start_date=None, end_date=None, days=180):
        """
        Run a backtest on the specified tickers and strategy.
        
        Parameters:
            tickers (list): List of ticker symbols to include in the backtest
            strategy (str): Strategy to use ('ml' or 'technical')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            days (int): Number of days to backtest if start_date not provided
            
        Returns:
            BacktestResult: Object containing backtest results
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        log_message(f"Running {strategy} strategy backtest from {start_date} to {end_date}")
        
        # Select strategy function
        if strategy == 'ml':
            strategy_func = ml_strategy
        else:
            strategy_func = simple_technical_strategy
        
        # Run backtest
        result = self.backtester.run_backtest(
            strategy=strategy_func,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date
        )
        
        self.last_backtest_result = result
        
        # Generate and log report
        report = result.generate_report()
        log_message(f"Backtest completed:\n{report}")
        
        return result
    
    def update_portfolio(self, ticker_prices=None):
        """
        Update portfolio values based on current market prices.
        
        Parameters:
            ticker_prices (dict): Dictionary mapping tickers to current prices
            
        Returns:
            dict: Updated portfolio summary
        """
        if ticker_prices is None:
            ticker_prices = {ticker: data['price'] for ticker, data in self.market_data.items()}
        
        # Update portfolio values
        portfolio_value = self.portfolio_manager.update_portfolio_values(ticker_prices)
        
        # Get portfolio summary
        summary = self.portfolio_manager.get_portfolio_summary()
        
        log_message(f"Portfolio updated: Total value ${summary['total_value']:,.2f}, " +
                   f"Cash ${summary['cash']:,.2f}, Invested ${summary['invested_value']:,.2f}")
        
        return summary
    
    def generate_trade_recommendations(self, ranked_stocks):
        """
        Generate trade recommendations based on stock scores and current portfolio.
        
        Parameters:
            ranked_stocks (list): List of dictionaries with stock scores and data
            
        Returns:
            dict: Recommendations for positions to add, reduce, or close
        """
        # Get recommendations from portfolio manager
        recommendations = self.portfolio_manager.get_position_recommendations(ranked_stocks)
        
        log_message(f"Generated trade recommendations: " +
                   f"{len(recommendations['buy'])} buys, {len(recommendations['sell'])} sells, " +
                   f"{len(recommendations['hold'])} holds, {len(recommendations['rebalance'])} rebalances")
        
        return recommendations
    
    def execute_trades(self, recommendations):
        """
        Execute trades based on recommendations.
        
        Parameters:
            recommendations (dict): Trade recommendations from portfolio manager
            
        Returns:
            list: Executed orders
        """
        executed_orders = []
        
        # Process buy recommendations
        for rec in recommendations.get('buy', []):
            ticker = rec['ticker']
            shares = rec['shares']
            
            # Create and submit order
            order = self.execution_manager.create_order(
                ticker=ticker,
                side=OrderSide.BUY,
                quantity=shares,
                order_type=OrderType.MARKET
            )
            
            self.execution_manager.submit_order(order)
            executed_orders.append(order)
            
            # Update portfolio if order filled
            if order.status.name == 'FILLED':
                price = order.average_fill_price
                cost = order.commission
                
                # Get stock data for sector, beta, etc.
                stock_data = next((s for s in self.watchlist if s['ticker'] == ticker), None)
                if stock_data:
                    sector = stock_data.get('sector', 'Unknown')
                    beta = 1.0  # Default if not available
                    volatility = stock_data.get('atr_pct', 5.0)
                    score = stock_data.get('day_trading_score', 50)
                    
                    self.portfolio_manager.add_position(
                        ticker=ticker,
                        shares=shares,
                        price=price,
                        cost=cost,
                        sector=sector,
                        beta=beta,
                        volatility=volatility,
                        score=score
                    )
        
        # Process sell recommendations
        for rec in recommendations.get('sell', []):
            ticker = rec['ticker']
            shares = rec['shares']
            
            # Create and submit order
            order = self.execution_manager.create_order(
                ticker=ticker,
                side=OrderSide.SELL,
                quantity=shares,
                order_type=OrderType.MARKET
            )
            
            self.execution_manager.submit_order(order)
            executed_orders.append(order)
            
            # Update portfolio if order filled
            if order.status.name == 'FILLED':
                price = order.average_fill_price
                cost = order.commission
                
                self.portfolio_manager.reduce_position(
                    ticker=ticker,
                    shares_to_sell=shares,
                    price=price,
                    cost=cost
                )
        
        # Process orders
        self.execution_manager.process_orders()
        
        log_message(f"Executed {len(executed_orders)} trades")
        
        return executed_orders
    
    def generate_watchlist(self, ranked_stocks, max_stocks=20, min_score=60):
        """
        Generate a watchlist of the most promising stocks.
        
        Parameters:
            ranked_stocks (list): List of dictionaries with stock scores and data
            max_stocks (int): Maximum number of stocks in the watchlist
            min_score (float): Minimum score to include in watchlist
            
        Returns:
            list: Watchlist of stocks
        """
        # Filter by minimum score
        candidates = [s for s in ranked_stocks if s.get('day_trading_score', 0) >= min_score]
        
        # Sort by score
        candidates = sorted(candidates, key=lambda x: x.get('day_trading_score', 0), reverse=True)
        
        # Limit to max_stocks
        watchlist = candidates[:max_stocks]
        
        self.watchlist = watchlist
        
        log_message(f"Generated watchlist with {len(watchlist)} stocks")
        
        return watchlist
    
    def save_system_state(self, filepath=None):
        """
        Save the current system state to a file.
        
        Parameters:
            filepath (str): Path to save the file, or None to use default
            
        Returns:
            str: Path to the saved file
        """
        if filepath is None:
            filepath = f"cache/trading_system/system_state_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Save portfolio state
        portfolio_path = self.portfolio_manager.save_portfolio_state()
        
        # Create system state
        state = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'initial_capital': self.initial_capital,
            'market_neutral': self.market_neutral,
            'is_model_trained': self.is_model_trained,
            'watchlist': [stock['ticker'] for stock in self.watchlist],
            'portfolio_path': portfolio_path,
            'market_data': {k: v for k, v in self.market_data.items() if k in self.watchlist}
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        log_message(f"System state saved to {filepath}")
        
        return filepath
    
    @classmethod
    def load_system_state(cls, filepath=None):
        """
        Load system state from a file.
        
        Parameters:
            filepath (str): Path to the file, or None to use latest
            
        Returns:
            TradingSystem: Loaded trading system
        """
        if filepath is None:
            # Find latest file
            system_dir = "cache/trading_system"
            if not os.path.exists(system_dir):
                return cls()  # Return new instance if no directory
                
            files = [f for f in os.listdir(system_dir) if f.startswith('system_state_')]
            if not files:
                return cls()  # Return new instance if no files
                
            filepath = os.path.join(system_dir, sorted(files)[-1])  # Get latest file
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Create new instance
            system = cls(
                initial_capital=state.get('initial_capital', 100000.0),
                market_neutral=state.get('market_neutral', True)
            )
            
            # Restore state
            system.is_model_trained = state.get('is_model_trained', False)
            
            # Load portfolio
            portfolio_path = state.get('portfolio_path')
            if portfolio_path and os.path.exists(portfolio_path):
                system.portfolio_manager = PortfolioManager.load_portfolio_state(portfolio_path)
            
            # Load market data
            system.market_data = state.get('market_data', {})
            
            # Load watchlist tickers
            watchlist_tickers = state.get('watchlist', [])
            if watchlist_tickers:
                # Get data for watchlist tickers
                watchlist = []
                for ticker in watchlist_tickers:
                    try:
                        stock_data, error = get_stock_data(ticker)
                        if not error and stock_data:
                            watchlist.append(stock_data)
                    except Exception:
                        pass
                
                system.watchlist = watchlist
            
            log_message(f"System state loaded from {filepath}")
            
            return system
            
        except Exception as e:
            log_message(f"Error loading system state: {e}")
            return cls()  # Return new instance on error
    
    def run_complete_workflow(self, tickers, use_ml=True, execute_trades=False):
        """
        Run a complete workflow from analysis to trade execution.
        
        Parameters:
            tickers (list): List of ticker symbols to analyze
            use_ml (bool): Whether to use ML-based scoring
            execute_trades (bool): Whether to execute recommended trades
            
        Returns:
            dict: Workflow results
        """
        results = {
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'tickers_analyzed': len(tickers),
            'ml_scoring_used': use_ml,
            'trades_executed': execute_trades
        }
        
        # Step 1: Train ML model if using ML scoring
        if use_ml and not self.is_model_trained:
            log_message("Training ML model...")
            self.train_ml_model(tickers)
            results['ml_model_trained'] = self.is_model_trained
        
        # Step 2: Analyze stocks
        log_message("Analyzing stocks...")
        ranked_stocks, failed_tickers = self.analyze_stocks(tickers, use_ml=use_ml)
        results['ranked_stocks_count'] = len(ranked_stocks)
        results['failed_tickers_count'] = len(failed_tickers)
        
        # Step 3: Generate watchlist
        log_message("Generating watchlist...")
        watchlist = self.generate_watchlist(ranked_stocks)
        results['watchlist_count'] = len(watchlist)
        
        # Step 4: Update portfolio values
        log_message("Updating portfolio...")
        portfolio_summary = self.update_portfolio()
        results['portfolio_value'] = portfolio_summary['total_value']
        results['portfolio_cash'] = portfolio_summary['cash']
        
        # Step 5: Generate trade recommendations
        log_message("Generating trade recommendations...")
        recommendations = self.generate_trade_recommendations(ranked_stocks)
        results['buy_recommendations'] = len(recommendations['buy'])
        results['sell_recommendations'] = len(recommendations['sell'])
        
        # Step 6: Execute trades if requested
        if execute_trades:
            log_message("Executing trades...")
            executed_orders = self.execute_trades(recommendations)
            results['executed_orders'] = len(executed_orders)
            
            # Update portfolio after trades
            portfolio_summary = self.update_portfolio()
            results['portfolio_value_after_trades'] = portfolio_summary['total_value']
            results['portfolio_cash_after_trades'] = portfolio_summary['cash']
        
        # Step 7: Save system state
        log_message("Saving system state...")
        state_path = self.save_system_state()
        results['state_saved'] = bool(state_path)
        
        # Step 8: Run backtest on watchlist tickers
        log_message("Running backtest on watchlist...")
        backtest_result = self.run_backtest(
            tickers=[stock['ticker'] for stock in watchlist[:10]],  # Limit to top 10
            strategy='ml' if use_ml else 'technical'
        )
        results['backtest_return'] = backtest_result.total_return
        results['backtest_sharpe'] = backtest_result.sharpe_ratio
        
        results['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message(f"Workflow completed in {(datetime.now() - datetime.strptime(results['start_time'], '%Y-%m-%d %H:%M:%S')).total_seconds():.2f} seconds")
        
        return results

# Example usage
def run_demo():
    """Run a demonstration of the trading system"""
    # Initialize the trading system
    system = TradingSystem(initial_capital=100000.0)
    
    # Define a list of stocks to analyze
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]
    
    # Run the complete workflow
    results = system.run_complete_workflow(tickers, use_ml=True, execute_trades=False)
    
    # Print results
    print("\nTrading System Demo Results:")
    print("=" * 50)
    print(f"Stocks analyzed: {results['tickers_analyzed']}")
    print(f"Stocks ranked: {results['ranked_stocks_count']}")
    print(f"Watchlist size: {results['watchlist_count']}")
    print(f"Portfolio value: ${results['portfolio_value']:,.2f}")
    print(f"Buy recommendations: {results['buy_recommendations']}")
    print(f"Sell recommendations: {results['sell_recommendations']}")
    print(f"Backtest return: {results['backtest_return']:.2%}")
    print(f"Backtest Sharpe ratio: {results['backtest_sharpe']:.2f}")
    
    return system, results

if __name__ == "__main__":
    run_demo()