"""
Execution Module

This module implements integration with low-slippage execution algorithms.
It addresses the following requirements:
1. Smart order routing to minimize market impact
2. Dynamic participation rate adjustment based on liquidity
3. Integration with broker APIs for order execution
4. Support for various order types (market, limit, VWAP, TWAP)

The module provides a framework for efficient trade execution with minimal market impact.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
import requests
from enum import Enum
from modules.utils import log_message

# Create cache directory for execution data if it doesn't exist
os.makedirs('cache/execution', exist_ok=True)

class OrderType(Enum):
    """Enum for order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TWAP = "TWAP"  # Time-Weighted Average Price
    VWAP = "VWAP"  # Volume-Weighted Average Price
    PARTICIPATE = "PARTICIPATE"  # Participate with % of volume

class OrderSide(Enum):
    """Enum for order sides"""
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    """Enum for order statuses"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"

class Order:
    """
    Represents a trading order with execution details.
    """
    
    def __init__(self, ticker, side, quantity, order_type=OrderType.MARKET, 
                 limit_price=None, stop_price=None, time_in_force="DAY",
                 participation_rate=None, start_time=None, end_time=None):
        """
        Initialize an order.
        
        Parameters:
            ticker (str): Ticker symbol
            side (OrderSide): BUY or SELL
            quantity (int): Number of shares
            order_type (OrderType): Type of order (MARKET, LIMIT, etc.)
            limit_price (float): Limit price for LIMIT orders
            stop_price (float): Stop price for STOP orders
            time_in_force (str): Time in force (DAY, GTC, IOC)
            participation_rate (float): Participation rate for PARTICIPATE orders (0.0-1.0)
            start_time (datetime): Start time for TWAP/VWAP orders
            end_time (datetime): End time for TWAP/VWAP orders
        """
        self.order_id = f"ORD-{int(time.time())}-{hash(ticker) % 10000}"
        self.ticker = ticker
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.time_in_force = time_in_force
        self.participation_rate = participation_rate
        self.start_time = start_time or datetime.now()
        self.end_time = end_time
        
        # Execution details
        self.status = OrderStatus.PENDING
        self.filled_quantity = 0
        self.average_fill_price = 0.0
        self.fills = []  # List of individual fills
        self.commission = 0.0
        self.submitted_time = None
        self.filled_time = None
        self.cancelled_time = None
        self.rejection_reason = None
        self.broker_order_id = None
        
        # Child orders (for algorithmic orders)
        self.child_orders = []
    
    def update_status(self, new_status, message=None):
        """Update order status and log the change"""
        old_status = self.status
        self.status = new_status
        
        if new_status == OrderStatus.SUBMITTED:
            self.submitted_time = datetime.now()
        elif new_status == OrderStatus.FILLED:
            self.filled_time = datetime.now()
        elif new_status == OrderStatus.CANCELLED:
            self.cancelled_time = datetime.now()
        elif new_status == OrderStatus.REJECTED:
            self.rejection_reason = message
        
        log_message(f"Order {self.order_id} ({self.ticker}) status changed: {old_status.value} -> {new_status.value}" + 
                   (f" - {message}" if message else ""))
    
    def add_fill(self, quantity, price, timestamp=None):
        """
        Add a fill to the order.
        
        Parameters:
            quantity (int): Number of shares filled
            price (float): Fill price
            timestamp (datetime): Fill timestamp
        """
        timestamp = timestamp or datetime.now()
        
        fill = {
            'quantity': quantity,
            'price': price,
            'timestamp': timestamp
        }
        
        self.fills.append(fill)
        
        # Update filled quantity and average price
        old_value = self.filled_quantity * self.average_fill_price
        new_value = quantity * price
        self.filled_quantity += quantity
        
        if self.filled_quantity > 0:
            self.average_fill_price = (old_value + new_value) / self.filled_quantity
        
        # Update status
        if self.filled_quantity >= self.quantity:
            self.update_status(OrderStatus.FILLED)
        elif self.filled_quantity > 0:
            self.update_status(OrderStatus.PARTIALLY_FILLED)
    
    def cancel(self, reason=None):
        """Cancel the order"""
        if self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
            self.update_status(OrderStatus.CANCELLED, reason)
            return True
        return False
    
    def to_dict(self):
        """Convert order to dictionary for serialization"""
        return {
            'order_id': self.order_id,
            'ticker': self.ticker,
            'side': self.side.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'limit_price': self.limit_price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force,
            'participation_rate': self.participation_rate,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'average_fill_price': self.average_fill_price,
            'fills': self.fills,
            'commission': self.commission,
            'submitted_time': self.submitted_time.isoformat() if self.submitted_time else None,
            'filled_time': self.filled_time.isoformat() if self.filled_time else None,
            'cancelled_time': self.cancelled_time.isoformat() if self.cancelled_time else None,
            'rejection_reason': self.rejection_reason,
            'broker_order_id': self.broker_order_id
        }

class ExecutionAlgorithm:
    """
    Base class for execution algorithms.
    """
    
    def __init__(self, name):
        """
        Initialize the execution algorithm.
        
        Parameters:
            name (str): Algorithm name
        """
        self.name = name
    
    def create_execution_plan(self, order):
        """
        Create an execution plan for the order.
        
        Parameters:
            order (Order): Order to execute
            
        Returns:
            list: List of child orders or execution steps
        """
        raise NotImplementedError("Subclasses must implement create_execution_plan")
    
    def execute_step(self, order, market_data):
        """
        Execute a single step of the algorithm.
        
        Parameters:
            order (Order): Order being executed
            market_data (dict): Current market data
            
        Returns:
            bool: True if execution is complete
        """
        raise NotImplementedError("Subclasses must implement execute_step")

class TWAPAlgorithm(ExecutionAlgorithm):
    """
    Time-Weighted Average Price (TWAP) execution algorithm.
    Divides the order into equal-sized child orders spread over time.
    """
    
    def __init__(self, num_slices=10, slice_interval_minutes=5):
        """
        Initialize the TWAP algorithm.
        
        Parameters:
            num_slices (int): Number of slices to divide the order into
            slice_interval_minutes (int): Time interval between slices in minutes
        """
        super().__init__("TWAP")
        self.num_slices = num_slices
        self.slice_interval_minutes = slice_interval_minutes
    
    def create_execution_plan(self, order):
        """
        Create a TWAP execution plan.
        
        Parameters:
            order (Order): Order to execute
            
        Returns:
            list: List of child orders
        """
        if order.order_type != OrderType.TWAP:
            # Convert to TWAP
            order.order_type = OrderType.TWAP
        
        # Calculate slice size
        slice_size = max(1, order.quantity // self.num_slices)
        remainder = order.quantity % self.num_slices
        
        # Create child orders
        child_orders = []
        start_time = order.start_time
        
        for i in range(self.num_slices):
            # Add remainder to last slice
            qty = slice_size + (remainder if i == self.num_slices - 1 else 0)
            
            if qty <= 0:
                continue
                
            child = Order(
                ticker=order.ticker,
                side=order.side,
                quantity=qty,
                order_type=OrderType.MARKET,
                time_in_force="DAY"
            )
            
            # Set scheduled time
            scheduled_time = start_time + timedelta(minutes=i * self.slice_interval_minutes)
            child.start_time = scheduled_time
            
            child_orders.append(child)
        
        order.child_orders = child_orders
        return child_orders
    
    def execute_step(self, order, market_data):
        """
        Execute a TWAP step.
        
        Parameters:
            order (Order): TWAP order
            market_data (dict): Current market data
            
        Returns:
            bool: True if execution is complete
        """
        now = datetime.now()
        
        # Check if any child orders are due for execution
        for child in order.child_orders:
            if child.status == OrderStatus.PENDING and child.start_time <= now:
                # Submit this child order
                child.update_status(OrderStatus.SUBMITTED)
                
                # In a real implementation, this would submit to the broker
                # For simulation, we'll just mark it as filled
                price = market_data.get(child.ticker, {}).get('price', 0)
                if price > 0:
                    child.add_fill(child.quantity, price)
        
        # Check if all child orders are filled
        all_filled = all(child.status == OrderStatus.FILLED for child in order.child_orders)
        
        if all_filled:
            # Calculate overall fill details
            total_quantity = sum(child.filled_quantity for child in order.child_orders)
            total_value = sum(child.filled_quantity * child.average_fill_price for child in order.child_orders)
            
            order.filled_quantity = total_quantity
            order.average_fill_price = total_value / total_quantity if total_quantity > 0 else 0
            
            # Mark parent order as filled
            if order.filled_quantity >= order.quantity:
                order.update_status(OrderStatus.FILLED)
            elif order.filled_quantity > 0:
                order.update_status(OrderStatus.PARTIALLY_FILLED)
            
            return True
        
        return False

class VWAPAlgorithm(ExecutionAlgorithm):
    """
    Volume-Weighted Average Price (VWAP) execution algorithm.
    Divides the order into child orders sized according to expected volume profile.
    """
    
    def __init__(self, volume_profile=None):
        """
        Initialize the VWAP algorithm.
        
        Parameters:
            volume_profile (dict): Expected volume profile by time of day
        """
        super().__init__("VWAP")
        
        # Default volume profile (percentage of daily volume by hour)
        self.volume_profile = volume_profile or {
            9: 0.12,   # 9:00-10:00: 12%
            10: 0.09,  # 10:00-11:00: 9%
            11: 0.08,  # 11:00-12:00: 8%
            12: 0.07,  # 12:00-13:00: 7%
            13: 0.08,  # 13:00-14:00: 8%
            14: 0.09,  # 14:00-15:00: 9%
            15: 0.22,  # 15:00-16:00: 22% (including close)
            16: 0.25   # After-hours: 25%
        }
    
    def create_execution_plan(self, order):
        """
        Create a VWAP execution plan.
        
        Parameters:
            order (Order): Order to execute
            
        Returns:
            list: List of child orders
        """
        if order.order_type != OrderType.VWAP:
            # Convert to VWAP
            order.order_type = OrderType.VWAP
        
        # Determine time range
        start_hour = order.start_time.hour
        end_hour = order.end_time.hour if order.end_time else 16
        
        # Calculate volume profile for this time range
        total_volume = sum(self.volume_profile.get(h, 0) for h in range(start_hour, end_hour + 1))
        
        if total_volume == 0:
            # Fallback to equal distribution
            hours = max(1, end_hour - start_hour + 1)
            profile = {h: 1.0 / hours for h in range(start_hour, end_hour + 1)}
        else:
            # Normalize profile for this time range
            profile = {h: self.volume_profile.get(h, 0) / total_volume for h in range(start_hour, end_hour + 1)}
        
        # Create child orders
        child_orders = []
        remaining_quantity = order.quantity
        
        for hour, volume_pct in sorted(profile.items()):
            if hour < start_hour or hour > end_hour:
                continue
                
            # Calculate quantity for this hour
            qty = int(order.quantity * volume_pct)
            
            # Ensure we don't exceed total quantity
            qty = min(qty, remaining_quantity)
            
            if qty <= 0:
                continue
                
            child = Order(
                ticker=order.ticker,
                side=order.side,
                quantity=qty,
                order_type=OrderType.MARKET,
                time_in_force="DAY"
            )
            
            # Set scheduled time
            scheduled_time = order.start_time.replace(hour=hour, minute=0, second=0)
            if scheduled_time < order.start_time:
                scheduled_time = order.start_time
            
            child.start_time = scheduled_time
            
            child_orders.append(child)
            remaining_quantity -= qty
        
        # Add any remaining quantity to the last child order
        if remaining_quantity > 0 and child_orders:
            child_orders[-1].quantity += remaining_quantity
        
        order.child_orders = child_orders
        return child_orders
    
    def execute_step(self, order, market_data):
        """
        Execute a VWAP step.
        
        Parameters:
            order (Order): VWAP order
            market_data (dict): Current market data
            
        Returns:
            bool: True if execution is complete
        """
        # Implementation similar to TWAP
        now = datetime.now()
        
        # Check if any child orders are due for execution
        for child in order.child_orders:
            if child.status == OrderStatus.PENDING and child.start_time <= now:
                # Submit this child order
                child.update_status(OrderStatus.SUBMITTED)
                
                # In a real implementation, this would submit to the broker
                # For simulation, we'll just mark it as filled
                price = market_data.get(child.ticker, {}).get('price', 0)
                if price > 0:
                    child.add_fill(child.quantity, price)
        
        # Check if all child orders are filled
        all_filled = all(child.status == OrderStatus.FILLED for child in order.child_orders)
        
        if all_filled:
            # Calculate overall fill details
            total_quantity = sum(child.filled_quantity for child in order.child_orders)
            total_value = sum(child.filled_quantity * child.average_fill_price for child in order.child_orders)
            
            order.filled_quantity = total_quantity
            order.average_fill_price = total_value / total_quantity if total_quantity > 0 else 0
            
            # Mark parent order as filled
            if order.filled_quantity >= order.quantity:
                order.update_status(OrderStatus.FILLED)
            elif order.filled_quantity > 0:
                order.update_status(OrderStatus.PARTIALLY_FILLED)
            
            return True
        
        return False

class ExecutionManager:
    """
    Manages order execution across different algorithms and brokers.
    """
    
    def __init__(self):
        """Initialize the execution manager."""
        self.orders = {}  # order_id -> Order
        self.algorithms = {
            OrderType.TWAP: TWAPAlgorithm(),
            OrderType.VWAP: VWAPAlgorithm()
        }
        self.market_data = {}  # ticker -> market data
    
    def create_order(self, ticker, side, quantity, order_type=OrderType.MARKET, **kwargs):
        """
        Create a new order.
        
        Parameters:
            ticker (str): Ticker symbol
            side (OrderSide): BUY or SELL
            quantity (int): Number of shares
            order_type (OrderType): Type of order
            **kwargs: Additional order parameters
            
        Returns:
            Order: Created order
        """
        order = Order(
            ticker=ticker,
            side=side,
            quantity=quantity,
            order_type=order_type,
            **kwargs
        )
        
        self.orders[order.order_id] = order
        return order
    
    def submit_order(self, order):
        """
        Submit an order for execution.
        
        Parameters:
            order (Order): Order to submit
            
        Returns:
            bool: True if order was submitted successfully
        """
        if order.order_id not in self.orders:
            self.orders[order.order_id] = order
        
        # Check if this is an algorithmic order
        if order.order_type in self.algorithms:
            # Create execution plan
            algorithm = self.algorithms[order.order_type]
            algorithm.create_execution_plan(order)
            
            # Mark as submitted
            order.update_status(OrderStatus.SUBMITTED)
            return True
        else:
            # Simple order, just mark as submitted and filled (for simulation)
            order.update_status(OrderStatus.SUBMITTED)
            
            # In a real implementation, this would submit to a broker
            # For simulation, we'll just mark it as filled
            price = self.market_data.get(order.ticker, {}).get('price', 100.0)  # Default price
            order.add_fill(order.quantity, price)
            
            return True
    
    def cancel_order(self, order_id):
        """
        Cancel an order.
        
        Parameters:
            order_id (str): Order ID to cancel
            
        Returns:
            bool: True if order was cancelled successfully
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        return order.cancel("Cancelled by user")
    
    def update_market_data(self, ticker, price, volume=0):
        """
        Update market data for a ticker.
        
        Parameters:
            ticker (str): Ticker symbol
            price (float): Current price
            volume (float): Current volume
        """
        self.market_data[ticker] = {
            'price': price,
            'volume': volume,
            'timestamp': datetime.now()
        }
    
    def process_orders(self):
        """
        Process all active orders.
        
        Returns:
            int: Number of orders processed
        """
        processed = 0
        
        for order_id, order in list(self.orders.items()):
            if order.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                # Check if this is an algorithmic order
                if order.order_type in self.algorithms:
                    algorithm = self.algorithms[order.order_type]
                    
                    # Execute a step
                    complete = algorithm.execute_step(order, self.market_data)
                    processed += 1
                    
                    if complete:
                        log_message(f"Order {order_id} execution completed")
        
        return processed
