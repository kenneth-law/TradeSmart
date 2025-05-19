"""
Portfolio Management Module

This module implements robust risk management and portfolio construction overlays.
It addresses the following requirements:
1. Position sizing based on expected edge and forecast volatility
2. Sector and single-name concentration limits
3. Market-neutral or beta-targeted exposures
4. Risk-based kill switches for drawdown protection

The module provides a framework for constructing and managing portfolios with proper
risk controls and diversification.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from modules.utils import log_message
from modules.data_retrieval import get_stock_info, get_stock_history

# Create cache directory for portfolio data if it doesn't exist
os.makedirs('cache/portfolio', exist_ok=True)

class PortfolioManager:
    """
    Portfolio manager that implements risk management and position sizing.
    """

    def __init__(self, initial_capital=100000.0, market_neutral=True, 
                 max_position_size=0.05, max_sector_exposure=0.25,
                 max_drawdown_threshold=0.15, beta_target=0.0):
        """
        Initialize the portfolio manager.

        Parameters:
            initial_capital (float): Initial capital in dollars
            market_neutral (bool): Whether to target market neutrality
            max_position_size (float): Maximum position size as fraction of portfolio
            max_sector_exposure (float): Maximum sector exposure as fraction of portfolio
            max_drawdown_threshold (float): Maximum drawdown before kill switch activates
            beta_target (float): Target portfolio beta (0.0 for market neutral)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.market_neutral = market_neutral
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.max_drawdown_threshold = max_drawdown_threshold
        self.beta_target = beta_target

        # Portfolio state
        self.positions = {}  # ticker -> {'shares': int, 'cost_basis': float, 'sector': str, 'beta': float}
        self.sector_exposures = {}  # sector -> dollar exposure
        self.portfolio_beta = 0.0
        self.peak_capital = initial_capital
        self.current_drawdown = 0.0
        self.kill_switch_activated = False

        # Performance tracking
        self.equity_curve = []
        self.position_history = []
        self.trade_history = []

    def calculate_position_size(self, ticker, score, price, volatility, volume, sector, beta):
        """
        Calculate the optimal position size based on expected edge, volatility, and risk limits.

        Parameters:
            ticker (str): Ticker symbol
            score (float): Signal score (0-100)
            price (float): Current stock price
            volatility (float): Stock volatility (e.g., ATR as percentage)
            volume (float): Average daily volume
            sector (str): Stock sector
            beta (float): Stock beta

        Returns:
            tuple: (dollar_amount, shares, reason) or (0, 0, reason) if no position should be taken
        """
        if self.kill_switch_activated:
            return 0, 0, "Kill switch activated due to drawdown"

        # Convert score to expected edge (score of 50 = neutral, 100 = strong buy, 0 = strong sell)
        expected_edge = (score - 50) / 50.0  # Range: -1.0 to 1.0

        # For hedgefund approach, we still consider stocks with small edge
        # but we'll reduce position size for stocks with lower scores
        edge_factor = max(0.2, abs(expected_edge))  # Minimum 0.2 edge factor

        # Base position size on edge and volatility (higher edge = larger position, higher vol = smaller position)
        # Kelly criterion-inspired sizing
        kelly_fraction = edge_factor / (volatility / 100)

        # Apply a safety factor (e.g., half-Kelly)
        kelly_fraction *= 0.5

        # Cap at max position size
        position_fraction = min(abs(kelly_fraction), self.max_position_size)

        # Calculate dollar amount
        dollar_amount = self.current_capital * position_fraction

        # Check sector exposure limits
        current_sector_exposure = self.sector_exposures.get(sector, 0)
        if (current_sector_exposure + dollar_amount) / self.current_capital > self.max_sector_exposure:
            # Scale back to respect sector limit
            available_sector_room = (self.max_sector_exposure * self.current_capital) - current_sector_exposure
            if available_sector_room <= 0:
                return 0, 0, f"Sector {sector} exposure limit reached"
            dollar_amount = min(dollar_amount, available_sector_room)

        # Check liquidity constraints (don't take more than 10% of daily volume)
        max_shares_by_volume = volume * 0.1
        max_dollars_by_volume = max_shares_by_volume * price

        if dollar_amount > max_dollars_by_volume:
            dollar_amount = max_dollars_by_volume

        # Calculate shares
        shares = int(dollar_amount / price)

        # No position if too small
        if shares <= 0:
            return 0, 0, "Position too small"

        # Adjust for market neutrality if needed
        if self.market_neutral and expected_edge > 0:  # Long position
            # Check if adding this position would push portfolio beta too far from target
            new_beta_contribution = beta * (dollar_amount / self.current_capital)
            new_portfolio_beta = self.portfolio_beta + new_beta_contribution

            if abs(new_portfolio_beta - self.beta_target) > 0.2:  # Allow some deviation
                # Need to reduce position size or find a hedge
                beta_adjustment_factor = max(0, 1 - (abs(new_portfolio_beta - self.beta_target) - 0.2) / 0.2)
                dollar_amount *= beta_adjustment_factor
                shares = int(dollar_amount / price)

                if shares <= 0:
                    return 0, 0, "Position eliminated due to beta constraints"

        return dollar_amount, shares, "Position sized based on edge, volatility, and risk limits"

    def add_position(self, ticker, shares, price, cost, sector, beta, volatility, score):
        """
        Add a new position to the portfolio.

        Parameters:
            ticker (str): Ticker symbol
            shares (int): Number of shares
            price (float): Purchase price per share
            cost (float): Transaction cost
            sector (str): Stock sector
            beta (float): Stock beta
            volatility (float): Stock volatility
            score (float): Signal score

        Returns:
            bool: True if position was added successfully
        """
        if ticker in self.positions:
            # Update existing position
            position = self.positions[ticker]
            old_value = position['shares'] * position['current_price']

            # Update sector exposure
            self.sector_exposures[position['sector']] -= old_value

            # Update position
            total_shares = position['shares'] + shares
            total_cost = position['cost_basis'] * position['shares'] + price * shares + cost
            position['shares'] = total_shares
            position['cost_basis'] = total_cost / total_shares
            position['current_price'] = price
            position['current_value'] = price * total_shares
            position['last_update'] = datetime.now().strftime('%Y-%m-%d')

            # Update sector exposure
            new_value = position['shares'] * position['current_price']
            self.sector_exposures[position['sector']] = self.sector_exposures.get(position['sector'], 0) + new_value

            # Update portfolio beta
            self.portfolio_beta += (beta * (new_value - old_value) / self.current_capital)
        else:
            # Create new position
            self.positions[ticker] = {
                'shares': shares,
                'cost_basis': (price * shares + cost) / shares,
                'current_price': price,
                'current_value': price * shares,
                'entry_date': datetime.now().strftime('%Y-%m-%d'),
                'last_update': datetime.now().strftime('%Y-%m-%d'),
                'sector': sector,
                'beta': beta,
                'volatility': volatility,
                'score': score
            }

            # Update sector exposure
            position_value = price * shares
            self.sector_exposures[sector] = self.sector_exposures.get(sector, 0) + position_value

            # Update portfolio beta
            self.portfolio_beta += beta * (position_value / self.current_capital)

        # Update capital
        self.current_capital -= (price * shares + cost)

        # Record trade
        self.trade_history.append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'ticker': ticker,
            'action': 'BUY',
            'shares': shares,
            'price': price,
            'cost': cost,
            'value': price * shares
        })

        # Record position history
        self._record_position_snapshot()

        return True

    def reduce_position(self, ticker, shares_to_sell, price, cost):
        """
        Reduce or close an existing position.

        Parameters:
            ticker (str): Ticker symbol
            shares_to_sell (int): Number of shares to sell
            price (float): Sale price per share
            cost (float): Transaction cost

        Returns:
            bool: True if position was reduced successfully
        """
        if ticker not in self.positions:
            return False

        position = self.positions[ticker]

        if shares_to_sell > position['shares']:
            shares_to_sell = position['shares']  # Can't sell more than we have

        # Calculate values
        old_value = position['shares'] * position['current_price']
        sell_value = shares_to_sell * price

        # Update sector exposure
        self.sector_exposures[position['sector']] -= old_value

        # Update position
        position['shares'] -= shares_to_sell
        position['current_price'] = price
        position['current_value'] = price * position['shares']
        position['last_update'] = datetime.now().strftime('%Y-%m-%d')

        # Update sector exposure if position still exists
        if position['shares'] > 0:
            new_value = position['shares'] * position['current_price']
            self.sector_exposures[position['sector']] = self.sector_exposures.get(position['sector'], 0) + new_value

            # Update portfolio beta
            self.portfolio_beta -= (position['beta'] * (old_value - new_value) / self.current_capital)
        else:
            # Position closed, remove it
            self.portfolio_beta -= (position['beta'] * old_value / self.current_capital)
            del self.positions[ticker]

        # Update capital
        self.current_capital += (sell_value - cost)

        # Record trade
        self.trade_history.append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'ticker': ticker,
            'action': 'SELL',
            'shares': shares_to_sell,
            'price': price,
            'cost': cost,
            'value': sell_value
        })

        # Record position history
        self._record_position_snapshot()

        return True

    def update_portfolio_values(self, ticker_prices):
        """
        Update portfolio values based on current market prices.

        Parameters:
            ticker_prices (dict): Dictionary mapping tickers to current prices

        Returns:
            float: Updated portfolio value
        """
        portfolio_value = self.current_capital

        for ticker, position in list(self.positions.items()):
            if ticker in ticker_prices:
                # Get current price
                price = ticker_prices[ticker]

                # Update position value
                old_value = position['current_value']
                position['current_price'] = price
                position['current_value'] = price * position['shares']
                position['last_update'] = datetime.now().strftime('%Y-%m-%d')

                # Update sector exposure
                sector = position['sector']
                self.sector_exposures[sector] = self.sector_exposures.get(sector, 0) - old_value + position['current_value']

                # Add to portfolio value
                portfolio_value += position['current_value']
            else:
                # Price not available, use last known value
                portfolio_value += position['current_value']

        # Update current capital and check for drawdown
        old_capital = self.current_capital
        self.current_capital = portfolio_value

        # Update peak capital and drawdown
        if portfolio_value > self.peak_capital:
            self.peak_capital = portfolio_value

        self.current_drawdown = 1 - (portfolio_value / self.peak_capital) if self.peak_capital > 0 else 0

        # Check kill switch
        if self.current_drawdown >= self.max_drawdown_threshold and not self.kill_switch_activated:
            self.kill_switch_activated = True
            log_message(f"Kill switch activated: Drawdown of {self.current_drawdown:.2%} exceeded threshold of {self.max_drawdown_threshold:.2%}")

        # Record equity point
        self.equity_curve.append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'equity': portfolio_value,
            'cash': self.current_capital,
            'invested': portfolio_value - self.current_capital,
            'drawdown': self.current_drawdown
        })

        return portfolio_value

    def _record_position_snapshot(self):
        """Record a snapshot of current positions for historical tracking."""
        snapshot = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'positions': {ticker: position.copy() for ticker, position in self.positions.items()},
            'sector_exposures': self.sector_exposures.copy(),
            'portfolio_beta': self.portfolio_beta,
            'cash': self.current_capital,
            'total_value': self.current_capital + sum(p['current_value'] for p in self.positions.values())
        }

        self.position_history.append(snapshot)

    def get_portfolio_summary(self):
        """
        Get a summary of the current portfolio.

        Returns:
            dict: Portfolio summary
        """
        total_value = self.current_capital + sum(p['current_value'] for p in self.positions.values())

        return {
            'total_value': total_value,
            'cash': self.current_capital,
            'cash_percentage': (self.current_capital / total_value) * 100 if total_value > 0 else 0,
            'invested_value': total_value - self.current_capital,
            'invested_percentage': ((total_value - self.current_capital) / total_value) * 100 if total_value > 0 else 0,
            'num_positions': len(self.positions),
            'sector_exposures': {sector: {'value': value, 'percentage': (value / total_value) * 100 if total_value > 0 else 0} 
                                for sector, value in self.sector_exposures.items()},
            'portfolio_beta': self.portfolio_beta,
            'current_drawdown': self.current_drawdown,
            'kill_switch_activated': self.kill_switch_activated
        }

    def get_position_recommendations(self, stock_scores, market_data=None):
        """
        Generate position recommendations based on stock scores and current portfolio.

        Parameters:
            stock_scores (list): List of dictionaries with stock scores and data
            market_data (dict): Optional market data for beta adjustments

        Returns:
            dict: Recommendations for positions to add, reduce, or close
        """
        recommendations = {
            'buy': [],
            'sell': [],
            'hold': [],
            'rebalance': []
        }

        # Check if kill switch is activated
        if self.kill_switch_activated:
            # Recommend selling all positions
            for ticker, position in self.positions.items():
                recommendations['sell'].append({
                    'ticker': ticker,
                    'shares': position['shares'],
                    'current_value': position['current_value'],
                    'reason': "Kill switch activated due to drawdown"
                })
            return recommendations

        # Process existing positions first
        for ticker, position in self.positions.items():
            # Find this ticker in the scores
            ticker_data = next((s for s in stock_scores if s['ticker'] == ticker), None)

            if ticker_data:
                score = ticker_data.get('day_trading_score', 0)

                # Update position score
                position['score'] = score

                # Determine action based on score
                if score < 40:  # Sell signal
                    recommendations['sell'].append({
                        'ticker': ticker,
                        'shares': position['shares'],
                        'current_value': position['current_value'],
                        'reason': f"Score {score} below threshold"
                    })
                elif score < 60:  # Hold
                    recommendations['hold'].append({
                        'ticker': ticker,
                        'shares': position['shares'],
                        'current_value': position['current_value'],
                        'reason': f"Score {score} in neutral range"
                    })
                else:  # Strong position, check if we should add more
                    # Check if position is below target size
                    position_value = position['current_value']
                    position_pct = position_value / (self.current_capital + sum(p['current_value'] for p in self.positions.values()))

                    target_pct = min(self.max_position_size, score / 200)  # Score of 100 -> 0.5 max size

                    if position_pct < target_pct * 0.8:  # More than 20% below target
                        # Calculate additional shares to buy
                        target_value = target_pct * (self.current_capital + sum(p['current_value'] for p in self.positions.values()))
                        additional_value = target_value - position_value
                        additional_shares = int(additional_value / position['current_price'])

                        if additional_shares > 0:
                            recommendations['buy'].append({
                                'ticker': ticker,
                                'shares': additional_shares,
                                'estimated_cost': additional_shares * position['current_price'],
                                'reason': f"Increasing position with score {score}"
                            })
                    else:
                        recommendations['hold'].append({
                            'ticker': ticker,
                            'shares': position['shares'],
                            'current_value': position['current_value'],
                            'reason': f"Position adequately sized with score {score}"
                        })
            else:
                # No current data, recommend holding
                recommendations['hold'].append({
                    'ticker': ticker,
                    'shares': position['shares'],
                    'current_value': position['current_value'],
                    'reason': "No current score data available"
                })

        # Process new position candidates - Hedgefund approach: invest in all analyzed stocks
        # Get all stocks not currently in the portfolio
        candidates = [s for s in stock_scores if s['ticker'] not in self.positions]

        # Calculate available capital for new positions
        available_capital = self.current_capital * 0.9  # Use more capital for diversification

        if candidates:
            # Calculate equal allocation per stock (with a small buffer)
            allocation_per_stock = available_capital / (len(candidates) * 1.1)

            for candidate in candidates:
                ticker = candidate['ticker']
                score = candidate.get('day_trading_score', 0)
                price = candidate.get('current_price', 0)
                volatility = candidate.get('atr_pct', 10)  # Default to 10% if not available
                volume = candidate.get('volume_ratio', 1) * 100000  # Rough estimate if not available
                sector = candidate.get('sector', 'Unknown')
                beta = candidate.get('beta', 1.0)  # Default to market beta if not available

                # Calculate shares based on equal allocation
                target_dollars = allocation_per_stock

                # Apply basic risk management - reduce allocation for very volatile stocks
                if volatility > 15:
                    target_dollars *= (15 / volatility)

                # Check sector exposure limits
                current_sector_exposure = self.sector_exposures.get(sector, 0)
                if (current_sector_exposure + target_dollars) / self.current_capital > self.max_sector_exposure:
                    # Scale back to respect sector limit
                    available_sector_room = (self.max_sector_exposure * self.current_capital) - current_sector_exposure
                    if available_sector_room <= 0:
                        continue  # Skip this stock if sector limit reached
                    target_dollars = min(target_dollars, available_sector_room)

                # Calculate shares
                shares = int(target_dollars / price)

                if shares > 0 and price * shares <= available_capital:
                    recommendations['buy'].append({
                        'ticker': ticker,
                        'shares': shares,
                        'estimated_cost': shares * price,
                        'reason': f"Hedgefund allocation for {ticker} (Score: {score})"
                    })

                    available_capital -= (shares * price)

        # Check for portfolio rebalancing needs
        if self.market_neutral and abs(self.portfolio_beta - self.beta_target) > 0.2:
            # Need to rebalance for beta neutrality
            if self.portfolio_beta > self.beta_target:
                # Too much long exposure, recommend reducing high-beta positions
                high_beta_positions = sorted(
                    [(ticker, pos) for ticker, pos in self.positions.items() if pos['beta'] > 1.0],
                    key=lambda x: x[1]['beta'], reverse=True
                )

                for ticker, position in high_beta_positions[:3]:  # Top 3 highest beta positions
                    recommendations['rebalance'].append({
                        'ticker': ticker,
                        'shares': int(position['shares'] * 0.3),  # Reduce by 30%
                        'current_value': position['current_value'] * 0.3,
                        'reason': f"Reducing high beta position (β={position['beta']:.2f}) for portfolio neutrality"
                    })
            else:
                # Too much short exposure or not enough long exposure
                # Recommend adding low-beta positions or reducing short positions
                recommendations['rebalance'].append({
                    'action': 'adjust_beta',
                    'current_beta': self.portfolio_beta,
                    'target_beta': self.beta_target,
                    'reason': "Portfolio beta below target, consider adding low-beta long positions"
                })

        return recommendations

    def save_portfolio_state(self, filepath=None):
        """
        Save the current portfolio state to a file.

        Parameters:
            filepath (str): Path to save the file, or None to use default

        Returns:
            str: Path to the saved file
        """
        if filepath is None:
            filepath = f"cache/portfolio/portfolio_state_{datetime.now().strftime('%Y%m%d')}.json"

        state = {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'current_drawdown': self.current_drawdown,
            'kill_switch_activated': self.kill_switch_activated,
            'market_neutral': self.market_neutral,
            'max_position_size': self.max_position_size,
            'max_sector_exposure': self.max_sector_exposure,
            'max_drawdown_threshold': self.max_drawdown_threshold,
            'beta_target': self.beta_target,
            'portfolio_beta': self.portfolio_beta,
            'positions': self.positions,
            'sector_exposures': self.sector_exposures,
            'equity_curve': self.equity_curve[-30:] if len(self.equity_curve) > 30 else self.equity_curve,
            'trade_history': self.trade_history[-50:] if len(self.trade_history) > 50 else self.trade_history,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        return filepath

    @classmethod
    def load_portfolio_state(cls, filepath=None):
        """
        Load portfolio state from a file.

        Parameters:
            filepath (str): Path to the file, or None to use latest

        Returns:
            PortfolioManager: Loaded portfolio manager
        """
        if filepath is None:
            # Find latest file
            portfolio_dir = "cache/portfolio"
            if not os.path.exists(portfolio_dir):
                return cls()  # Return new instance if no directory

            files = [f for f in os.listdir(portfolio_dir) if f.startswith('portfolio_state_')]
            if not files:
                return cls()  # Return new instance if no files

            filepath = os.path.join(portfolio_dir, sorted(files)[-1])  # Get latest file

        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            # Create new instance
            portfolio = cls(
                initial_capital=state.get('initial_capital', 100000.0),
                market_neutral=state.get('market_neutral', True),
                max_position_size=state.get('max_position_size', 0.05),
                max_sector_exposure=state.get('max_sector_exposure', 0.25),
                max_drawdown_threshold=state.get('max_drawdown_threshold', 0.15),
                beta_target=state.get('beta_target', 0.0)
            )

            # Restore state
            portfolio.current_capital = state.get('current_capital', portfolio.initial_capital)
            portfolio.peak_capital = state.get('peak_capital', portfolio.initial_capital)
            portfolio.current_drawdown = state.get('current_drawdown', 0.0)
            portfolio.kill_switch_activated = state.get('kill_switch_activated', False)
            portfolio.portfolio_beta = state.get('portfolio_beta', 0.0)
            portfolio.positions = state.get('positions', {})
            portfolio.sector_exposures = state.get('sector_exposures', {})
            portfolio.equity_curve = state.get('equity_curve', [])
            portfolio.trade_history = state.get('trade_history', [])

            return portfolio
        except Exception as e:
            log_message(f"Error loading portfolio state: {e}")
            return cls()  # Return new instance on error
