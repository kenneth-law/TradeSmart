"""
Technical Analysis Module

This module contains functions for calculating technical indicators and scores for stocks.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from modules.data_retrieval import get_stock_info, get_stock_history
from modules.news_sentiment import get_news_sentiment_with_timeframes
from modules.utils import log_message

def get_stock_data(ticker_symbol, historical_data=None):
    """
    Fetches and computes stock market data and technical indicators for the given ticker symbol. The function
    retrieves basic stock information, historical price data, and intraday data when available. It calculates
    a variety of technical indicators such as moving averages, Bollinger Bands, RSI, ATR, and others to
    support both day trading and longer-term analysis. The function also identifies patterns like gap-ups
    and gap-downs along with volatility measures.

    Parameters:
        ticker_symbol (str): The symbol representing the stock (e.g., 'BHP.AX', 'ASX.AX').
        historical_data (pandas.DataFrame, optional): Pre-loaded historical data to use instead of fetching new data.
                                                     Used primarily for backtesting and ML training.

    Returns:
        tuple: A tuple containing the following elements:
            - dict: A dictionary with the computed stock indicators and statistics.
            - str: An error message if no data is found, otherwise None.
    """
    try:
        # Generate a cache timestamp (changes every minute to ensure fresh data)
        cache_timestamp = datetime.now().strftime('%Y%m%d%H%M')

        # Get stock info with caching
        info = get_stock_info(ticker_symbol, cache_timestamp)

        if not info:
            return None, f"No data found for {ticker_symbol}"

        # Use provided historical data if available, otherwise fetch it
        if historical_data is not None:
            hist = historical_data.copy()  # Create a copy to avoid SettingWithCopyWarning
            # For backtesting/ML training, we don't need intraday data
            intraday = pd.DataFrame()
        else:
            # Get historical data for technical indicators
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)  # Extended from 30 to 180 days for long-term analysis

            # Format dates for caching
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')

            # Get historical data with caching, force refresh to ensure we have the latest data
            hist = get_stock_history(ticker_symbol, start_date_str, end_date_str, "1d", cache_timestamp, force_refresh=True)
            # Create a copy to ensure we're not working with a view
            hist = hist.copy()

            # For intraday patterns, also get hourly data if available
            intraday_start = end_date - timedelta(days=5)
            intraday_start_str = intraday_start.strftime('%Y-%m-%d')
            intraday = get_stock_history(ticker_symbol, intraday_start_str, end_date_str, "1h", cache_timestamp, force_refresh=True)
            # Create a copy to ensure we're not working with a view
            intraday = intraday.copy()

        # Calculate technical indicators
        if len(hist) > 0:
            # Check if we have data for the current trading day
            today = datetime.now().date()
            latest_data_date = hist.index[-1].date() if len(hist) > 0 else None

            # Log if we're using data from a previous day
            if latest_data_date and latest_data_date < today:
                log_message(f"Using data from {latest_data_date} for {ticker_symbol} (current date: {today})")

            # Basic price data
            current_price = hist['Close'].iloc[-1] if not pd.isna(hist['Close'].iloc[-1]) else info.get('currentPrice', 0)
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price

            # Moving averages - both short-term for day trading and long-term for trend analysis
            hist.loc[:, 'MA5'] = hist['Close'].rolling(window=5).mean()
            hist.loc[:, 'MA10'] = hist['Close'].rolling(window=10).mean()
            hist.loc[:, 'MA20'] = hist['Close'].rolling(window=20).mean()
            hist.loc[:, 'MA50'] = hist['Close'].rolling(window=50).mean()  # Medium-term trend
            hist.loc[:, 'MA200'] = hist['Close'].rolling(window=200).mean()  # Long-term trend

            # Check if price is above/below key moving averages
            above_ma5 = current_price > hist['MA5'].iloc[-1] if not pd.isna(hist['MA5'].iloc[-1]) else False
            above_ma10 = current_price > hist['MA10'].iloc[-1] if not pd.isna(hist['MA10'].iloc[-1]) else False
            above_ma20 = current_price > hist['MA20'].iloc[-1] if not pd.isna(hist['MA20'].iloc[-1]) else False
            above_ma50 = current_price > hist['MA50'].iloc[-1] if not pd.isna(hist['MA50'].iloc[-1]) else False
            above_ma200 = current_price > hist['MA200'].iloc[-1] if not pd.isna(hist['MA200'].iloc[-1]) else False

            # Check for golden cross (MA50 crosses above MA200) and death cross (MA50 crosses below MA200)
            golden_cross = False
            death_cross = False
            if len(hist) > 201:  # Need at least 201 days of data
                ma50_yesterday = hist['MA50'].iloc[-2]
                ma200_yesterday = hist['MA200'].iloc[-2]
                ma50_today = hist['MA50'].iloc[-1]
                ma200_today = hist['MA200'].iloc[-1]

                if not pd.isna(ma50_yesterday) and not pd.isna(ma200_yesterday) and not pd.isna(ma50_today) and not pd.isna(ma200_today):
                    golden_cross = ma50_yesterday <= ma200_yesterday and ma50_today > ma200_today
                    death_cross = ma50_yesterday >= ma200_yesterday and ma50_today < ma200_today

            # Volatility measures
            # ATR (Average True Range)
            hist.loc[:, 'high_low'] = hist['High'] - hist['Low']
            hist.loc[:, 'high_close'] = np.abs(hist['High'] - hist['Close'].shift(1))
            hist.loc[:, 'low_close'] = np.abs(hist['Low'] - hist['Close'].shift(1))
            hist.loc[:, 'tr'] = hist[['high_low', 'high_close', 'low_close']].max(axis=1)
            hist.loc[:, 'atr14'] = hist['tr'].rolling(window=14).mean()

            # Bollinger Bands (20-day, 2 standard deviations)
            hist.loc[:, 'bb_middle'] = hist['Close'].rolling(window=20).mean()
            hist.loc[:, 'bb_std'] = hist['Close'].rolling(window=20).std()
            hist.loc[:, 'bb_upper'] = hist['bb_middle'] + 2 * hist['bb_std']
            hist.loc[:, 'bb_lower'] = hist['bb_middle'] - 2 * hist['bb_std']

            # Calculate BB position (0 = at lower band, 1 = at upper band)
            bb_range = hist['bb_upper'].iloc[-1] - hist['bb_lower'].iloc[-1]
            bb_position = (current_price - hist['bb_lower'].iloc[-1]) / bb_range if bb_range > 0 else 0.5

            # RSI (Relative Strength Index) - shorter 7-day for day trading
            delta = hist['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=7).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=7).mean()
            rs = gain / loss
            hist.loc[:, 'RSI7'] = 100 - (100 / (1 + rs))

            # Standard 14-day RSI
            gain14 = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss14 = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs14 = gain14 / loss14
            hist.loc[:, 'RSI14'] = 100 - (100 / (1 + rs14))

            # Momentum indicators
            # Recent and longer-term returns
            hist.loc[:, 'return_1d'] = hist['Close'].pct_change(periods=1) * 100
            hist.loc[:, 'return_3d'] = hist['Close'].pct_change(periods=3) * 100
            hist.loc[:, 'return_5d'] = hist['Close'].pct_change(periods=5) * 100
            hist.loc[:, 'return_30d'] = hist['Close'].pct_change(periods=30) * 100  # Monthly return
            hist.loc[:, 'return_90d'] = hist['Close'].pct_change(periods=90) * 100  # Quarterly return

            # Stochastic Oscillator (14-day)
            hist.loc[:, 'lowest_low'] = hist['Low'].rolling(window=14).min()
            hist.loc[:, 'highest_high'] = hist['High'].rolling(window=14).max()
            hist.loc[:, '%K'] = ((hist['Close'] - hist['lowest_low']) / (hist['highest_high'] - hist['lowest_low'])) * 100
            hist.loc[:, '%D'] = hist['%K'].rolling(window=3).mean()  # 3-day SMA of %K

            # On-Balance Volume (OBV)
            hist.loc[:, 'obv'] = 0
            hist.loc[hist['Close'] > hist['Close'].shift(1), 'obv'] = hist['Volume']
            hist.loc[hist['Close'] < hist['Close'].shift(1), 'obv'] = -hist['Volume']
            hist.loc[hist['Close'] == hist['Close'].shift(1), 'obv'] = 0
            hist.loc[:, 'obv'] = hist['obv'].cumsum()
            hist.loc[:, 'obv_ma20'] = hist['obv'].rolling(window=20).mean()  # 20-day MA of OBV

            # MACD for trend direction
            hist.loc[:, 'ema12'] = hist['Close'].ewm(span=12, adjust=False).mean()
            hist.loc[:, 'ema26'] = hist['Close'].ewm(span=26, adjust=False).mean()
            hist.loc[:, 'macd'] = hist['ema12'] - hist['ema26']
            hist.loc[:, 'signal'] = hist['macd'].ewm(span=9, adjust=False).mean()
            hist.loc[:, 'macd_hist'] = hist['macd'] - hist['signal']

            # Volume analysis
            hist.loc[:, 'volume_ma5'] = hist['Volume'].rolling(window=5).mean()
            volume_ratio = hist['Volume'].iloc[-1] / hist['volume_ma5'].iloc[-1] if hist['volume_ma5'].iloc[-1] > 0 else 1

            # Intraday volatility (if available)
            if len(intraday) > 0:
                # Calculate typical intraday price movement
                intraday_ranges = []
                for day in set(intraday.index.date):
                    day_data = intraday[intraday.index.date == day]
                    if len(day_data) > 1:
                        day_range = day_data['High'].max() - day_data['Low'].min()
                        day_open = day_data.iloc[0]['Open']
                        intraday_ranges.append(day_range / day_open * 100)  # As percentage

                avg_intraday_range = np.mean(intraday_ranges) if intraday_ranges else 0

                # Check for gap up/down patterns
                gap_ups = 0
                gap_downs = 0

                for i in range(1, min(5, len(intraday) // 7)):  # Check last 5 days if available
                    prev_close_idx = i * 7 - 1
                    if prev_close_idx < len(intraday) and prev_close_idx >= 0:
                        try:
                            day_idx = prev_close_idx - 7 + 1
                            if day_idx >= 0:
                                prev_close = intraday['Close'].iloc[-prev_close_idx]
                                next_open = intraday['Open'].iloc[-day_idx]

                                if next_open > prev_close * 1.005:  # 0.5% gap up
                                    gap_ups += 1
                                elif next_open < prev_close * 0.995:  # 0.5% gap down
                                    gap_downs += 1
                        except IndexError:
                            pass
            else:
                avg_intraday_range = 0
                gap_ups = 0
                gap_downs = 0

            # Key stats for day trading
            recent_rsi7 = hist['RSI7'].iloc[-1] if not pd.isna(hist['RSI7'].iloc[-1]) else 50
            recent_rsi14 = hist['RSI14'].iloc[-1] if not pd.isna(hist['RSI14'].iloc[-1]) else 50
            atr = hist['atr14'].iloc[-1] if not pd.isna(hist['atr14'].iloc[-1]) else 0
            atr_pct = (atr / current_price) * 100  # ATR as percentage of price

            # Returns - short and long term
            return_1d = hist['return_1d'].iloc[-1] if not pd.isna(hist['return_1d'].iloc[-1]) else 0
            return_3d = hist['return_3d'].iloc[-1] if not pd.isna(hist['return_3d'].iloc[-1]) else 0
            return_5d = hist['return_5d'].iloc[-1] if not pd.isna(hist['return_5d'].iloc[-1]) else 0
            return_30d = hist['return_30d'].iloc[-1] if not pd.isna(hist['return_30d'].iloc[-1]) else 0
            return_90d = hist['return_90d'].iloc[-1] if not pd.isna(hist['return_90d'].iloc[-1]) else 0

            # Stochastic Oscillator
            stoch_k = hist['%K'].iloc[-1] if not pd.isna(hist['%K'].iloc[-1]) else 50
            stoch_d = hist['%D'].iloc[-1] if not pd.isna(hist['%D'].iloc[-1]) else 50
            stoch_trend = "bullish" if stoch_k > stoch_d else "bearish"
            stoch_overbought = stoch_k > 80
            stoch_oversold = stoch_k < 20

            # On-Balance Volume
            obv = hist['obv'].iloc[-1] if not pd.isna(hist['obv'].iloc[-1]) else 0
            obv_ma20 = hist['obv_ma20'].iloc[-1] if not pd.isna(hist['obv_ma20'].iloc[-1]) else 0
            obv_trend = "bullish" if obv > obv_ma20 else "bearish"

            # MACD
            macd = hist['macd'].iloc[-1] if not pd.isna(hist['macd'].iloc[-1]) else 0
            macd_signal = hist['signal'].iloc[-1] if not pd.isna(hist['signal'].iloc[-1]) else 0
            macd_hist = hist['macd_hist'].iloc[-1] if not pd.isna(hist['macd_hist'].iloc[-1]) else 0
            macd_trend = "bullish" if macd > macd_signal else "bearish"

            # Pre-market indicators
            premarket_available = False
            premarket_change = 0

            # Calculate a score for pre-market movement if available
            if 'preMarketPrice' in info and info.get('preMarketPrice') and prev_close:
                premarket_available = True
                premarket_price = info.get('preMarketPrice')
                premarket_change = ((premarket_price / prev_close) - 1) * 100
        else:
            # Default values if no historical data
            current_price = info.get('currentPrice', 0)
            prev_close = info.get('previousClose', current_price)
            above_ma5 = False
            above_ma10 = False
            above_ma20 = False
            above_ma50 = False
            above_ma200 = False
            golden_cross = False
            death_cross = False
            recent_rsi7 = 50
            recent_rsi14 = 50
            atr = 0
            atr_pct = 0
            bb_position = 0.5
            return_1d = 0
            return_3d = 0
            return_5d = 0
            return_30d = 0
            return_90d = 0
            stoch_k = 50
            stoch_d = 50
            stoch_trend = "neutral"
            stoch_overbought = False
            stoch_oversold = False
            obv = 0
            obv_ma20 = 0
            obv_trend = "neutral"
            macd = 0
            macd_signal = 0
            macd_hist = 0
            macd_trend = "neutral"
            volume_ratio = 1
            avg_intraday_range = 0
            gap_ups = 0
            gap_downs = 0
            premarket_available = False
            premarket_change = 0

        # Get key financial data
        company_name = info.get('shortName', ticker_symbol)
        market_cap = info.get('marketCap', 0)

        # Get news sentiment
        sentiment_score, sentiment_label, sentiment_error, raw_sentiment_score, headlines = get_news_sentiment_with_timeframes(ticker_symbol, company_name)

        # Day Trading Score Components

        # 1. Technical signals (0-100, higher is more bullish)
        technical_score = 0

        # MA crossovers (bullish when shorter MA crosses above longer MA)
        if above_ma5:
            technical_score += 8
        if above_ma10:
            technical_score += 4
        if above_ma20:
            technical_score += 4

        # RSI signals
        if recent_rsi7 < 30:  # Oversold
            technical_score += 12  # Strong buy signal for day trading
        elif recent_rsi7 < 40:
            technical_score += 8
        elif recent_rsi7 > 70:  # Overbought
            technical_score -= 12  # Potential sell signal
        elif recent_rsi7 > 60:
            technical_score -= 8

        # BB position
        if bb_position < 0.2:  # Near lower band - potential bounce
            technical_score += 12
        elif bb_position > 0.8:  # Near upper band - potential reversal
            technical_score -= 12

        # MACD trend
        if macd_trend == "bullish" and macd_hist > 0:
            technical_score += 12
        elif macd_trend == "bearish" and macd_hist < 0:
            technical_score -= 12

        # Stochastic Oscillator
        if stoch_trend == "bullish" and stoch_oversold:
            technical_score += 10  # Bullish reversal from oversold
        elif stoch_trend == "bearish" and stoch_overbought:
            technical_score -= 10  # Bearish reversal from overbought

        # On-Balance Volume trend
        if obv_trend == "bullish":
            technical_score += 8  # Volume confirming price trend
        elif obv_trend == "bearish":
            technical_score -= 8

        # Recent momentum
        if return_1d > 1.5:  # Strong up day
            technical_score += 8
        elif return_1d < -1.5:  # Strong down day
            technical_score -= 8

        if return_3d > 3:  # Strong 3-day momentum
            technical_score += 4
        elif return_3d < -3:
            technical_score -= 4

        # 2. Long-term performance score (0-100, higher is more bullish)
        long_term_score = 50  # Start at neutral

        # Long-term moving averages
        if above_ma50:
            long_term_score += 10
        else:
            long_term_score -= 10

        if above_ma200:
            long_term_score += 15
        else:
            long_term_score -= 15

        # Golden/Death Cross (major long-term signals)
        if golden_cross:
            long_term_score += 20
        elif death_cross:
            long_term_score -= 20

        # Long-term returns
        if return_30d > 8:  # Strong monthly return
            long_term_score += 10
        elif return_30d < -8:
            long_term_score -= 10
        elif return_30d > 4:
            long_term_score += 5
        elif return_30d < -4:
            long_term_score -= 5

        if return_90d > 15:  # Strong quarterly return
            long_term_score += 15
        elif return_90d < -15:
            long_term_score -= 15
        elif return_90d > 8:
            long_term_score += 8
        elif return_90d < -8:
            long_term_score -= 8

        # Cap scores at 0-100
        technical_score = max(0, min(100, technical_score + 50))  # Convert to 0-100 scale
        long_term_score = max(0, min(100, long_term_score))

        # 2. Volatility score (0-100, higher means more volatile)
        volatility_score = 0

        # ATR percentage
        if atr_pct > 3:  # High volatility
            volatility_score = 90
        elif atr_pct > 2:
            volatility_score = 75
        elif atr_pct > 1.5:
            volatility_score = 60
        elif atr_pct > 1:
            volatility_score = 45
        elif atr_pct > 0.7:
            volatility_score = 30
        else:
            volatility_score = 15

        # Adjust for intraday range
        if avg_intraday_range > 2:
            volatility_score += 10

        # Cap at 100
        volatility_score = min(100, volatility_score)

        # 3. Sentiment score (0-100, higher is more positive)
        # Convert sentiment_score from -1 to 1 scale to 0-100
        news_sentiment_score = int((sentiment_score + 1) * 50)

        # 4. Gap potential (0-100, higher means more potential for gap trading)
        gap_score = 0

        # Recent gap frequency
        gap_frequency = (gap_ups + gap_downs) / 5 if gap_ups + gap_downs > 0 else 0
        gap_score += gap_frequency * 30

        # Premarket movement
        if premarket_available:
            if abs(premarket_change) > 3:
                gap_score += 50
            elif abs(premarket_change) > 1.5:
                gap_score += 30
            elif abs(premarket_change) > 0.5:
                gap_score += 15

        # News can cause gaps
        if abs(sentiment_score) > 0.5:
            gap_score += 20

        # Cap at 100
        gap_score = min(100, gap_score)

        # 5. Volume activity (0-100, higher means higher relative volume)
        volume_score = 0

        if volume_ratio > 2:
            volume_score = 90
        elif volume_ratio > 1.5:
            volume_score = 75
        elif volume_ratio > 1.2:
            volume_score = 60
        elif volume_ratio > 1:
            volume_score = 45
        elif volume_ratio > 0.8:
            volume_score = 30
        else:
            volume_score = 15

        # Calculate final day trading score
        # Updated weights: Technical 35%, Long-term 15%, Volatility 20%, News 10%, Gap 10%, Volume 10%
        day_trading_score = (
            0.35 * technical_score +
            0.15 * long_term_score +  # New component for long-term performance
            0.20 * volatility_score +
            0.10 * news_sentiment_score +
            0.10 * gap_score +        # Reduced from 15% to 10%
            0.10 * volume_score
        )

        # Determine trading strategy based on score
        if day_trading_score >= 70:
            strategy = "Strong Buy"
            if long_term_score >= 70:
                strategy_details = "Bullish technicals with strong long-term trend support"
            else:
                strategy_details = "Bullish technicals with good volatility for day trading"
        elif day_trading_score >= 60:
            strategy = "Buy"
            if long_term_score >= 60:
                strategy_details = "Favorable conditions with positive long-term outlook"
            else:
                strategy_details = "Favorable conditions for a long day trade"
        elif day_trading_score >= 45:
            strategy = "Neutral/Watch"
            if long_term_score >= 65:
                strategy_details = "Mixed short-term signals but positive long-term trend"
            elif long_term_score <= 35:
                strategy_details = "Mixed short-term signals with concerning long-term trend"
            else:
                strategy_details = "Mixed signals - monitor for intraday setup"
        elif day_trading_score >= 35:
            strategy = "Sell"
            if long_term_score <= 30:
                strategy_details = "Bearish conditions with negative long-term outlook"
            else:
                strategy_details = "Bearish short-term conditions despite average long-term trend"
        else:
            strategy = "Strong Sell"
            if long_term_score <= 30:
                strategy_details = "Strongly bearish technicals with poor long-term performance"
            else:
                strategy_details = "Strongly bearish technicals or insufficient volatility"

        # Store all relevant data
        stock_data = {
            'ticker': ticker_symbol,
            'company_name': company_name,
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': market_cap,
            'current_price': current_price,
            'prev_close': prev_close,
            'atr': atr,
            'atr_pct': atr_pct,
            'rsi7': recent_rsi7,
            'rsi14': recent_rsi14,
            'bb_position': bb_position,
            # Moving averages
            'above_ma5': above_ma5,
            'above_ma10': above_ma10,
            'above_ma20': above_ma20,
            'above_ma50': above_ma50,
            'above_ma200': above_ma200,
            'golden_cross': golden_cross,
            'death_cross': death_cross,
            # MACD
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist,
            'macd_trend': macd_trend,
            # Stochastic Oscillator
            'stoch_k': stoch_k,
            'stoch_d': stoch_d,
            'stoch_trend': stoch_trend,
            'stoch_overbought': stoch_overbought,
            'stoch_oversold': stoch_oversold,
            # On-Balance Volume
            'obv_trend': obv_trend,
            # Returns - short and long term
            'return_1d': return_1d,
            'return_3d': return_3d,
            'return_5d': return_5d,
            'return_30d': return_30d,
            'return_90d': return_90d,
            # Other indicators
            'volume_ratio': volume_ratio,
            'avg_intraday_range': avg_intraday_range,
            'gap_ups_5d': gap_ups,
            'gap_downs_5d': gap_downs,
            'premarket_change': premarket_change if premarket_available else None,
            # News sentiment
            'news_sentiment_score': sentiment_score,
            'news_sentiment_label': sentiment_label,
            'raw_sentiment_score': raw_sentiment_score,
            'weighted_sentiment_score': sentiment_score,
            'headlines': headlines,
            # Scores
            'technical_score': technical_score,
            'long_term_score': long_term_score,
            'volatility_score': volatility_score,
            'news_sentiment_score_normalized': news_sentiment_score,
            'gap_score': gap_score,
            'volume_score': volume_score,
            'day_trading_score': day_trading_score,
            'day_trading_strategy': strategy,
            'strategy_details': strategy_details
        }

        return stock_data, None

    except Exception as e:
        return None, f"Error retrieving data for {ticker_symbol}: {str(e)}"

def calculate_score_contribution(stock_data):
    """
    Calculates the contribution of each component to the overall day trading score.

    Parameters:
        stock_data (dict): Dictionary containing stock data and scores

    Returns:
        dict: Dictionary with score contributions
    """
    # Get the component scores
    technical_score = stock_data.get('technical_score', 0)
    long_term_score = stock_data.get('long_term_score', 50)
    volatility_score = stock_data.get('volatility_score', 0)
    news_sentiment_score = stock_data.get('news_sentiment_score_normalized', 50)
    gap_score = stock_data.get('gap_score', 0)
    volume_score = stock_data.get('volume_score', 0)

    # Calculate weighted contributions (same weights as in get_stock_data)
    technical_contribution = 0.35 * technical_score
    long_term_contribution = 0.15 * long_term_score
    volatility_contribution = 0.20 * volatility_score
    sentiment_contribution = 0.10 * news_sentiment_score
    gap_contribution = 0.10 * gap_score
    volume_contribution = 0.10 * volume_score

    # Get the total score for percentage calculations
    day_trading_score = stock_data.get('day_trading_score', 1)

    # Return the contributions
    return {
        'technical': technical_contribution,
        'long_term': long_term_contribution,
        'volatility': volatility_contribution,
        'sentiment': sentiment_contribution,
        'gap': gap_contribution,
        'volume': volume_contribution,
        'technical_pct': technical_contribution / day_trading_score * 100,
        'long_term_pct': long_term_contribution / day_trading_score * 100,
        'volatility_pct': volatility_contribution / day_trading_score * 100,
        'sentiment_pct': sentiment_contribution / day_trading_score * 100,
        'gap_pct': gap_contribution / day_trading_score * 100,
        'volume_pct': volume_contribution / day_trading_score * 100
    }
