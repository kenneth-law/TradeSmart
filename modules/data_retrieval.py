"""
Data Retrieval Module

This module contains functions for retrieving stock data from external sources.

Supported intervals for historical data:
- Intraday: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h
  Note: 1m data is only available for the last 7 days
  Note: All intraday data (interval < 1d) is only available for the last 60 days
- Daily and above: 1d, 5d, 1wk, 1mo, 3mo
"""

import yfinance as yf
import time
import random
from datetime import datetime, timedelta
from functools import lru_cache
from curl_cffi import requests as curl_requests
from requests.cookies import create_cookie
import yfinance_cookie_patch

def get_yf_session():
    """Create a session for yfinance with Chrome browser fingerprint to avoid rate limiting"""
    # Import curl_cffi requests at the function level to avoid import errors if the library isn't installed
    from curl_cffi import requests as curl_requests

    # Create a session that impersonates Chrome's TLS fingerprint
    session = curl_requests.Session(impersonate="chrome")

    return session

# Use caching to reduce API calls for the same ticker within a time window
@lru_cache(maxsize=100)
def get_stock_info(ticker_symbol, timestamp=None):
    """Get stock info with caching based on ticker symbol

    Args:
        ticker_symbol: The stock ticker
        timestamp: A timestamp that can be used to invalidate cache (changes hourly)

    Returns:
        Stock info dictionary
    """
    # timestamp parameter ensures cache is invalidated hourly
    # even though we don't use it directly in the function

    session = get_yf_session()
    ticker = yf.Ticker(ticker_symbol, session=session)
    return ticker.info

@lru_cache(maxsize=100)
def get_stock_history(ticker_symbol, start_date, end_date, interval="1d", timestamp=None, force_refresh=False):
    """
    Fetches historical stock data for a given ticker symbol over a specified date range.

    This function uses Yahoo Finance to retrieve stock data. It applies a small random
    delay before making requests to mimic human-like behavior. The function is cached
    using LRU cache to store the results for faster access if the same query is made
    within the cache size limit.

    Arguments:
        ticker_symbol (str): The stock ticker symbol of the company.
        start_date (str): The starting date for historical stock data in the format 'YYYY-MM-DD'.
        end_date (str): The ending date for historical stock data in the format 'YYYY-MM-DD'.
        interval (str): The interval for stock data. Defaults to "1d".
        timestamp (datetime, optional): Optional timestamp to indicate request time.
        force_refresh (bool): If True, bypass cache and force a fresh data fetch.

    Returns:
        DataFrame: A Pandas DataFrame containing the historical stock data for the given
        ticker symbol and date range.
    """
    # If force_refresh is True, we'll add the current timestamp to the cache key
    # This effectively bypasses the cache since the key will be unique each time
    if force_refresh:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')

    session = get_yf_session()

    # Add a small random delay to appear more human-like (reduced for performance)
    time.sleep(random.uniform(0.1, 1))

    ticker = yf.Ticker(ticker_symbol, session=session)
    return ticker.history(start=start_date, end=end_date, interval=interval)

def _wrap_cookie(cookie, session):
    """
    Wraps a cookie for use with the yfinance library.

    Parameters:
        cookie: The cookie to wrap
        session: The session to add the cookie to

    Returns:
        The wrapped cookie
    """
    return create_cookie(
        name=cookie.name,
        value=cookie.value,
        domain=cookie.domain,
        path=cookie.path,
        expires=cookie.expires
    )

def get_intraday_data(ticker_symbol, interval="1h", days=5, force_refresh=False):
    """
    Fetches intraday stock data for a given ticker symbol.

    This function is specifically designed for retrieving intraday data with proper
    handling of the limitations imposed by Yahoo Finance:
    - 1m data is only available for the last 7 days
    - All intraday data (interval < 1d) is only available for the last 60 days

    Arguments:
        ticker_symbol (str): The stock ticker symbol of the company.
        interval (str): The interval for stock data. Valid values are:
                        "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"
        days (int): Number of days of data to retrieve (max 60 for intervals > 1m, max 7 for 1m interval)
        force_refresh (bool): If True, bypass cache and force a fresh data fetch.

    Returns:
        tuple: A tuple containing:
            - DataFrame: A Pandas DataFrame containing the intraday stock data
            - str: An error message if data retrieval failed, otherwise None

    Example:
        # Get 5-minute data for BHP.AX for the last 3 days
        data, error = get_intraday_data("BHP.AX", interval="5m", days=3)
        if error:
            print(f"Error: {error}")
        else:
            print(f"Retrieved {len(data)} data points")
            print(data.head())

        # Get 1-minute data for ASX.AX for the last 2 days
        data, error = get_intraday_data("ASX.AX", interval="1m", days=2)
        if error:
            print(f"Error: {error}")
        else:
            print(f"Retrieved {len(data)} data points")
            # Calculate average volume per minute
            avg_volume = data['Volume'].mean()
            print(f"Average volume per minute: {avg_volume:.2f}")
    """
    try:
        # Import log_message here to avoid circular imports
        from modules.utils import log_message

        # Validate interval
        valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]
        if interval not in valid_intervals:
            return None, f"Invalid interval: {interval}. Valid intervals are: {', '.join(valid_intervals)}"

        # Check days limit based on interval
        if interval == "1m" and days > 7:
            log_message(f"Warning: 1m data is only available for the last 7 days. Limiting request to 7 days.")
            days = 7
        elif days > 60:
            log_message(f"Warning: Intraday data is only available for the last 60 days. Limiting request to 60 days.")
            days = 60

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Format dates
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        # Generate timestamp for cache
        cache_timestamp = datetime.now().strftime('%Y%m%d%H%M')
        if force_refresh:
            cache_timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')

        # Get data
        data = get_stock_history(ticker_symbol, start_date_str, end_date_str, interval, cache_timestamp, force_refresh)

        if len(data) == 0:
            return None, f"No intraday data available for {ticker_symbol} with interval {interval}"

        return data, None

    except Exception as e:
        return None, f"Error retrieving intraday data for {ticker_symbol}: {str(e)}"

def patch_yfdata_cookie_basic():
    """
    Patches the yfinance library to use a custom cookie handler.
    """
    def _patched(self, timeout=30):
        return get_yf_session()

