"""
Data Retrieval Module

This module contains functions for retrieving stock data from external sources.
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
def get_stock_history(ticker_symbol, start_date, end_date, interval="1d", timestamp=None):
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

    Returns:
        DataFrame: A Pandas DataFrame containing the historical stock data for the given
        ticker symbol and date range.
    """
    session = get_yf_session()
    
    # Add a small random delay to appear more human-like
    time.sleep(random.uniform(0.5, 2.0))
    
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

def patch_yfdata_cookie_basic():
    """
    Patches the yfinance library to use a custom cookie handler.
    """
    def _patched(self, timeout=30):
        return get_yf_session()
    
    yf.base._YahooFinance._get_session = _patched