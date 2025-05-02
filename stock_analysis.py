import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import requests
import openai
import os
import warnings
from bs4 import BeautifulSoup
warnings.filterwarnings('ignore')
from functools import lru_cache  # For caching results
from curl_cffi import requests as curl_requests
from requests.cookies import create_cookie
import yfinance_cookie_patch


# OpenAI API key should be set as environment variable
# os.environ["OPENAI_API_KEY"] = "your-api-key" or in cmd "set OPENAI_API_KEY="OPENAI_API_KEY""

import random
import time


yfinance_cookie_patch.patch_yfdata_cookie_basic()


def get_yf_session():
    """Create a session for yfinance with Chrome browser fingerprint to avoid rate limiting"""
    # Import curl_cffi requests at the function level to avoid import errors if the library isn't installed
    from curl_cffi import requests as curl_requests
    
    # Create a session that impersonates Chrome's TLS fingerprint
    session = curl_requests.Session(impersonate="chrome")
    

    
    return session

def get_news_sentiment(ticker_symbol, company_name):
    """Get news sentiment for a ticker using OpenAI API with BeautifulSoup for HTML parsing"""
    try:
        # Build search parameters
        search_term = f"{company_name} stock news"
        news_url = f"https://www.google.com/search?q={search_term}&tbm=nws&source=lnt&tbs=qdr:d"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(news_url, headers=headers)
        
        if response.status_code != 200:
            return 0, "Neutral", f"Failed to retrieve news: {response.status_code}"
        
        # Use BeautifulSoup to parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract headlines; the selectors here are based on current Google News HTML structures.
        headlines = []
        for tag in soup.find_all("div", class_="BNeawe vvjwJb AP7Wnd"):
            text = tag.get_text(strip=True)
            if text:
                headlines.append(text)
        
        # Fallback: try an alternative search if no headlines were found
        if not headlines:
            headlines = [tag.get_text(strip=True) for tag in soup.find_all("div", {"role": "heading"})]
        
        # If still empty, return a neutral result
        if not headlines:
            return 0, "Neutral", "No headlines found for sentiment analysis"
        
        # Combine the top 5 headlines into a single string for analysis
        combined_headlines = "\n".join(headlines[:5])
        
        # Now, if an OpenAI API key is available, analyze the sentiment using these headlines
        if "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]:
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            
            prompt = f"""
            Analyze the market sentiment for {company_name} ({ticker_symbol}) based on these recent news headlines:
            
            {combined_headlines}
            
            Rate the sentiment on a scale from -1.0 (very negative) to 1.0 (very positive).
            Also provide a summary label (Very Negative, Negative, Slightly Negative, Neutral, Slightly Positive, Positive, Very Positive).
            Format your response as exactly two lines:
            [sentiment_score]
            [sentiment_label]
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.2,
            )
            
            result = response.choices[0].message.content.strip().split('\n')
            print(result)
            
            # Extract sentiment score and label
            if len(result) >= 2:
                try:
                    sentiment_score = float(result[0].strip())
                    sentiment_label = result[1].strip()
                    print(f"The sentiment score given by AI is: {sentiment_score}")
                    return sentiment_score, sentiment_label, None
                except Exception as e:
                    return 0, "Neutral", "Failed to parse sentiment"
            else:
                return 0, "Neutral", "Incomplete sentiment analysis"
        else:
            return 0, "Neutral", "OpenAI API key not set"
    
    except Exception as e:
        return 0, "Neutral", f"Error analyzing news sentiment: {str(e)}"

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
    session = get_yf_session()
    
    # Add a small random delay to appear more human-like
    time.sleep(random.uniform(0.5, 2.0))
    
    ticker = yf.Ticker(ticker_symbol, session=session)
    return ticker.history(start=start_date, end=end_date, interval=interval)

def get_stock_data(ticker_symbol):
    """Get detailed stock data for day trading analysis"""
    try:
        # Generate a cache timestamp (changes every hour)
        cache_timestamp = datetime.now().strftime('%Y%m%d%H')
        
        # Get stock info with caching
        info = get_stock_info(ticker_symbol, cache_timestamp)
        
        if not info:
            return None, f"No data found for {ticker_symbol}"
            
        # Get historical data for technical indicators
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 30 days is enough for short-term analysis
        
        # Format dates for caching
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Get historical data with caching
        hist = get_stock_history(ticker_symbol, start_date_str, end_date_str, "1d", cache_timestamp)
        
        # For intraday patterns, also get hourly data if available
        intraday_start = end_date - timedelta(days=5)
        intraday_start_str = intraday_start.strftime('%Y-%m-%d')
        intraday = get_stock_history(ticker_symbol, intraday_start_str, end_date_str, "1h", cache_timestamp)
        
        # Calculate technical indicators
        if len(hist) > 0:
            # Basic price data
            current_price = hist['Close'].iloc[-1] if not pd.isna(hist['Close'].iloc[-1]) else info.get('currentPrice', 0)
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            
            # Moving averages - shorter periods for day trading
            hist['MA5'] = hist['Close'].rolling(window=5).mean()
            hist['MA10'] = hist['Close'].rolling(window=10).mean()
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            
            # Check if price is above/below key moving averages
            above_ma5 = current_price > hist['MA5'].iloc[-1] if not pd.isna(hist['MA5'].iloc[-1]) else False
            above_ma10 = current_price > hist['MA10'].iloc[-1] if not pd.isna(hist['MA10'].iloc[-1]) else False
            above_ma20 = current_price > hist['MA20'].iloc[-1] if not pd.isna(hist['MA20'].iloc[-1]) else False
            
            # Volatility measures
            # ATR (Average True Range)
            hist['high_low'] = hist['High'] - hist['Low']
            hist['high_close'] = np.abs(hist['High'] - hist['Close'].shift(1))
            hist['low_close'] = np.abs(hist['Low'] - hist['Close'].shift(1))
            hist['tr'] = hist[['high_low', 'high_close', 'low_close']].max(axis=1)
            hist['atr14'] = hist['tr'].rolling(window=14).mean()
            
            # Bollinger Bands (20-day, 2 standard deviations)
            hist['bb_middle'] = hist['Close'].rolling(window=20).mean()
            hist['bb_std'] = hist['Close'].rolling(window=20).std()
            hist['bb_upper'] = hist['bb_middle'] + 2 * hist['bb_std']
            hist['bb_lower'] = hist['bb_middle'] - 2 * hist['bb_std']
            
            # Calculate BB position (0 = at lower band, 1 = at upper band)
            bb_range = hist['bb_upper'].iloc[-1] - hist['bb_lower'].iloc[-1]
            bb_position = (current_price - hist['bb_lower'].iloc[-1]) / bb_range if bb_range > 0 else 0.5
            
            # RSI (Relative Strength Index) - shorter 7-day for day trading
            delta = hist['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=7).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=7).mean()
            rs = gain / loss
            hist['RSI7'] = 100 - (100 / (1 + rs))
            
            # Standard 14-day RSI
            gain14 = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss14 = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs14 = gain14 / loss14
            hist['RSI14'] = 100 - (100 / (1 + rs14))
            
            # Momentum indicators
            # Recent 1, 3, 5 day returns
            hist['return_1d'] = hist['Close'].pct_change(periods=1) * 100
            hist['return_3d'] = hist['Close'].pct_change(periods=3) * 100
            hist['return_5d'] = hist['Close'].pct_change(periods=5) * 100
            
            # MACD for trend direction
            hist['ema12'] = hist['Close'].ewm(span=12, adjust=False).mean()
            hist['ema26'] = hist['Close'].ewm(span=26, adjust=False).mean()
            hist['macd'] = hist['ema12'] - hist['ema26']
            hist['signal'] = hist['macd'].ewm(span=9, adjust=False).mean()
            hist['macd_hist'] = hist['macd'] - hist['signal']
            
            # Volume analysis
            hist['volume_ma5'] = hist['Volume'].rolling(window=5).mean()
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
            
            # Returns
            return_1d = hist['return_1d'].iloc[-1] if not pd.isna(hist['return_1d'].iloc[-1]) else 0
            return_3d = hist['return_3d'].iloc[-1] if not pd.isna(hist['return_3d'].iloc[-1]) else 0
            return_5d = hist['return_5d'].iloc[-1] if not pd.isna(hist['return_5d'].iloc[-1]) else 0
            
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
            recent_rsi7 = 50
            recent_rsi14 = 50
            atr = 0
            atr_pct = 0
            bb_position = 0.5
            return_1d = 0
            return_3d = 0
            return_5d = 0
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
        sentiment_score, sentiment_label, sentiment_error = get_news_sentiment(ticker_symbol, company_name)
        
        # Day Trading Score Components
        
        # 1. Technical signals (0-100, higher is more bullish)
        technical_score = 0
        
        # MA crossovers (bullish when shorter MA crosses above longer MA)
        if above_ma5:
            technical_score += 10
        if above_ma10:
            technical_score += 5
        if above_ma20:
            technical_score += 5
            
        # RSI signals
        if recent_rsi7 < 30:  # Oversold
            technical_score += 15  # Strong buy signal for day trading
        elif recent_rsi7 < 40:
            technical_score += 10
        elif recent_rsi7 > 70:  # Overbought
            technical_score -= 15  # Potential sell signal
        elif recent_rsi7 > 60:
            technical_score -= 10
            
        # BB position
        if bb_position < 0.2:  # Near lower band - potential bounce
            technical_score += 15
        elif bb_position > 0.8:  # Near upper band - potential reversal
            technical_score -= 15
            
        # MACD trend
        if macd_trend == "bullish" and macd_hist > 0:
            technical_score += 15
        elif macd_trend == "bearish" and macd_hist < 0:
            technical_score -= 15
            
        # Recent momentum
        if return_1d > 1.5:  # Strong up day
            technical_score += 10
        elif return_1d < -1.5:  # Strong down day
            technical_score -= 10
            
        if return_3d > 3:  # Strong 3-day momentum
            technical_score += 5
        elif return_3d < -3:
            technical_score -= 5
        
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
        # Weights: Technical 40%, Volatility 20%, News 15%, Gap 15%, Volume 10%
        day_trading_score = (
            0.40 * technical_score +
            0.20 * volatility_score +
            0.15 * news_sentiment_score +
            0.15 * gap_score +
            0.10 * volume_score
        )
        
        # Determine trading strategy based on score
        if day_trading_score >= 70:
            strategy = "Strong Buy"
            strategy_details = "Bullish technicals with good volatility for day trading"
        elif day_trading_score >= 60:
            strategy = "Buy"
            strategy_details = "Favorable conditions for a long day trade"
        elif day_trading_score >= 45:
            strategy = "Neutral/Watch"
            strategy_details = "Mixed signals - monitor for intraday setup"
        elif day_trading_score >= 35:
            strategy = "Sell"
            strategy_details = "Bearish conditions likely"
        else:
            strategy = "Strong Sell"
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
            'above_ma5': above_ma5,
            'above_ma10': above_ma10,
            'above_ma20': above_ma20,
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist,
            'macd_trend': macd_trend,
            'return_1d': return_1d,
            'return_3d': return_3d,
            'return_5d': return_5d,
            'volume_ratio': volume_ratio,
            'avg_intraday_range': avg_intraday_range,
            'gap_ups_5d': gap_ups,
            'gap_downs_5d': gap_downs,
            'premarket_change': premarket_change if premarket_available else None,
            'news_sentiment_score': sentiment_score,
            'news_sentiment_label': sentiment_label,
            'technical_score': technical_score,
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

def analyze_stocks(ticker_list, delay=1):
    """Analyze multiple stocks for day trading opportunities"""
    stock_analysis = []
    failed_tickers = []
    
    print(f"Analyzing {len(ticker_list)} stocks for day trading opportunities...")
    
    for i, ticker_symbol in enumerate(ticker_list):
        print(f"Processing {i+1}/{len(ticker_list)}: {ticker_symbol}")
        
        stock_data, error = get_stock_data(ticker_symbol)
        
        if error:
            print(f"!!! {error}")
            failed_tickers.append(ticker_symbol)
            continue
            
        stock_analysis.append(stock_data)
        
        # Print brief update
        print(f"  {stock_data['ticker']}: {stock_data['day_trading_strategy']} (Score: {stock_data['day_trading_score']:.1f})")
        
        time.sleep(delay)  # Prevent API rate limits
    
    # Sort by day trading score (higher is better)
    ranked_stocks = sorted(stock_analysis, key=lambda x: x['day_trading_score'], reverse=True)
    
    return ranked_stocks, failed_tickers

def format_results(ranked_stocks, failed_tickers=None, top_n=20):
    """Format the analysis results for display with day trading focus"""
    if not ranked_stocks:
        return "No stocks to display."
        
    results = "\n======== DAY TRADING OPPORTUNITIES ========\n"
    results += f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    # Group by strategy
    categories = {
        "Strong Buy": [],
        "Buy": [],
        "Neutral/Watch": [],
        "Sell": [],
        "Strong Sell": []
    }
    
    for stock in ranked_stocks:
        categories[stock['day_trading_strategy']].append(stock)
    
    # Print each category (limiting to top_n stocks per category)
    for category, stocks in categories.items():
        if stocks:
            stocks = stocks[:top_n]  # Limit to top_n per category
            results += f"\n--- {category.upper()} ---\n"
            for stock in stocks:
                results += f"- {stock['ticker']} - {stock['company_name']} (${stock['current_price']:.2f})\n"
                results += f"  Day Trading Score: {stock['day_trading_score']:.1f}/100 | ATR: {stock['atr_pct']:.2f}% | RSI7: {stock['rsi7']:.1f}\n"
                results += f"  1-Day Return: {stock['return_1d']:.2f}% | MACD: {stock['macd_trend'].upper()}\n"
                results += f"  News Sentiment: {stock['news_sentiment_label']} | Volume: {stock['volume_ratio']:.2f}x avg\n"
                results += f"  Strategy: {stock['strategy_details']}\n\n"
    
    results += f"\n\nTOP 5 HIGHEST VOLATILITY OPPORTUNITIES (ATR%):\n"
    volatile_stocks = sorted(ranked_stocks, key=lambda x: x['atr_pct'], reverse=True)[:5]
    for stock in volatile_stocks:
        results += f"- {stock['ticker']}: ATR {stock['atr_pct']:.2f}% | Score: {stock['day_trading_score']:.1f} | Strategy: {stock['day_trading_strategy']}\n"
    
    results += f"\n\nTOP 5 STRONGEST NEWS SENTIMENT:\n"
    sentiment_stocks = sorted(ranked_stocks, key=lambda x: abs(x['news_sentiment_score']), reverse=True)[:5]
    for stock in sentiment_stocks:
        sent_str = f"+{stock['news_sentiment_score']:.2f}" if stock['news_sentiment_score'] > 0 else f"{stock['news_sentiment_score']:.2f}"
        results += f"- {stock['ticker']}: {stock['news_sentiment_label']} ({sent_str}) | Strategy: {stock['day_trading_strategy']}\n"
    
    # Show failed tickers
    if failed_tickers:
        results += "\nFAILED TO RETRIEVE DATA FOR:\n"
        results += ", ".join(failed_tickers)
    
    return results

def save_to_csv(ranked_stocks, filename="day_trading_opportunities.csv"):
    """Save the analysis results to a CSV file"""
    if not ranked_stocks:
        return "No data to save."
    
    try:
        df = pd.DataFrame(ranked_stocks)
        df.to_csv(filename, index=False)
        return f"Results saved to {filename}"
    except Exception as e:
        return f"Error saving to CSV: {str(e)}"

def generate_watchlist(ranked_stocks, min_score=65, max_stocks=10):
    """Generate a focused watchlist for day trading"""
    if not ranked_stocks:
        return "No stocks available for watchlist."
    
    # Filter stocks by minimum score
    candidates = [s for s in ranked_stocks if s['day_trading_score'] >= min_score]
    
    # If we don't have enough stocks meeting the score threshold, take the top scoring ones
    if len(candidates) < max_stocks:
        additional = [s for s in ranked_stocks if s['day_trading_score'] < min_score]
        candidates.extend(additional[:max_stocks-len(candidates)])
    
    # Limit to max_stocks
    watchlist = candidates[:max_stocks]
    
    result = "\n======== DAY TRADING WATCHLIST ========\n"
    result += f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for i, stock in enumerate(watchlist, 1):
        result += f"{i}. {stock['ticker']} - {stock['company_name']}\n"
        result += f"   Price: ${stock['current_price']:.2f} | Strategy: {stock['day_trading_strategy']}\n"
        result += f"   Key Metrics: ATR {stock['atr_pct']:.2f}% | RSI7: {stock['rsi7']:.1f} | MACD: {stock['macd_trend'].upper()}\n"
        result += f"   Trading Note: {stock['strategy_details']}\n\n"
    
    return result

def get_historical_data_for_chart(ticker_symbol, days=30):
    """Get historical price data for charting"""
    try:
        # Generate a cache timestamp (changes every hour)
        cache_timestamp = datetime.now().strftime('%Y%m%d%H')
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for caching
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Get historical data with caching
        hist = get_stock_history(ticker_symbol, start_date_str, end_date_str, "1d", cache_timestamp)
        
        if hist.empty:
            return None
        
        # Reset index to make date a column
        hist = hist.reset_index()
        
        # Convert date to string format
        hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
        
        return hist
    except Exception as e:
        print(f"Error getting historical data: {e}")
        return None

def get_detailed_stock_metrics(stock_data):
    """Extract key metrics from stock data for detailed display"""
    # Format all metrics for UI display
    metrics = {
        'Score Components': {
            'Technical Score': f"{stock_data['technical_score']:.1f}/100",
            'Volatility Score': f"{stock_data['volatility_score']:.1f}/100",
            'News Sentiment': f"{stock_data['news_sentiment_score']:.2f} ({stock_data['news_sentiment_label']})",
            'Gap Score': f"{stock_data['gap_score']:.1f}/100",
            'Volume Score': f"{stock_data['volume_score']:.1f}/100"
        },
        'Technical Indicators': {
            'RSI (7-day)': f"{stock_data['rsi7']:.1f}",
            'RSI (14-day)': f"{stock_data['rsi14']:.1f}",
            'MACD': f"{stock_data['macd']:.4f}",
            'MACD Signal': f"{stock_data['macd_signal']:.4f}",
            'MACD Histogram': f"{stock_data['macd_hist']:.4f}",
            'MACD Trend': stock_data['macd_trend'].upper(),
            'Above 5-day MA': "Yes" if stock_data['above_ma5'] else "No",
            'Above 10-day MA': "Yes" if stock_data['above_ma10'] else "No",
            'Above 20-day MA': "Yes" if stock_data['above_ma20'] else "No",
            'BB Position': f"{stock_data['bb_position']:.2f}"
        },
        'Volatility Metrics': {
            'ATR': f"{stock_data['atr']:.4f}",
            'ATR %': f"{stock_data['atr_pct']:.2f}%",
            'Avg Intraday Range': f"{stock_data['avg_intraday_range']:.2f}%",
            'Gap Ups (5d)': stock_data['gap_ups_5d'],
            'Gap Downs (5d)': stock_data['gap_downs_5d']
        },
        'Performance Metrics': {
            '1-Day Return': f"{stock_data['return_1d']:.2f}%",
            '3-Day Return': f"{stock_data['return_3d']:.2f}%",
            '5-Day Return': f"{stock_data['return_5d']:.2f}%",
            'Volume Ratio': f"{stock_data['volume_ratio']:.2f}x avg"
        }
    }
    
    return metrics

def generate_watchlist_data(df, min_score=65, max_stocks=10):
    """Generate a watchlist data from DataFrame"""
    # Filter stocks by minimum score
    candidates = df[df['day_trading_score'] >= min_score].to_dict('records')
    
    # If we don't have enough stocks meeting the score threshold, take the top scoring ones
    if len(candidates) < max_stocks:
        additional = df[df['day_trading_score'] < min_score].sort_values('day_trading_score', ascending=False).head(max_stocks-len(candidates)).to_dict('records')
        candidates.extend(additional)
    
    # Limit to max_stocks
    return candidates[:max_stocks]

def prepare_price_chart_data(ticker_symbol, days=30):
    """Prepare stock price data for candlestick chart"""
    try:
        # Generate a cache timestamp (changes every hour)
        cache_timestamp = datetime.now().strftime('%Y%m%d%H')
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for caching
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Get historical data with caching
        hist = get_stock_history(ticker_symbol, start_date_str, end_date_str, "1d", cache_timestamp)
        
        if hist.empty:
            return None
        
        # Calculate some moving averages for the chart
        hist['MA5'] = hist['Close'].rolling(window=5).mean()
        hist['MA20'] = hist['Close'].rolling(window=20).mean()
        
        # Format for plotly
        chart_data = {
            'dates': hist.index.strftime('%Y-%m-%d').tolist(),
            'open': hist['Open'].tolist(),
            'high': hist['High'].tolist(),
            'low': hist['Low'].tolist(),
            'close': hist['Close'].tolist(),
            'volume': hist['Volume'].tolist(),
            'ma5': hist['MA5'].tolist(),
            'ma20': hist['MA20'].tolist(),
        }
        
        return chart_data
    except Exception as e:
        print(f"Error preparing price chart data: {e}")
        return None

def calculate_score_contribution(stock_data):
    """Calculate how much each component contributes to the final score"""
    # The weights used in the original calculation
    weights = {
        'technical': 0.40,
        'volatility': 0.20,
        'sentiment': 0.15,
        'gap': 0.15,
        'volume': 0.10
    }
    
    contributions = {
        'Technical': stock_data['technical_score'] * weights['technical'],
        'Volatility': stock_data['volatility_score'] * weights['volatility'],
        'News Sentiment': stock_data['news_sentiment_score_normalized'] * weights['sentiment'],
        'Gap Potential': stock_data['gap_score'] * weights['gap'],
        'Volume Activity': stock_data['volume_score'] * weights['volume']
    }
    
    # Calculate percentage contribution to total score
    total_score = stock_data['day_trading_score']
    for key in contributions:
        contributions[key] = {
            'absolute': contributions[key],
            'percentage': (contributions[key] / total_score * 100) if total_score > 0 else 0
        }
    
    return contributions

def get_stock_comparison_data(ticker_symbols, metric='day_trading_score'):
    """Compare multiple stocks based on a specific metric"""
    comparison_data = []
    
    for ticker in ticker_symbols:
        stock_data, error = get_stock_data(ticker)
        if error or not stock_data:
            continue
            
        comparison_data.append({
            'ticker': stock_data['ticker'],
            'company_name': stock_data['company_name'],
            'value': stock_data.get(metric, 0)
        })
    
    # Sort by the chosen metric
    comparison_data.sort(key=lambda x: x['value'], reverse=True)
    
    return comparison_data

def get_sector_performance():
    """Get performance data for major market sectors"""
    # List of ETFs that represent different sectors
    sector_etfs = {
        'Technology': 'XLK',
        'Healthcare': 'XLV',
        'Financials': 'XLF',
        'Consumer Staples': 'XLP',
        'Consumer Discretionary': 'XLY',
        'Industrials': 'XLI',
        'Energy': 'XLE',
        'Materials': 'XLB',
        'Utilities': 'XLU',
        'Real Estate': 'XLRE',
        'Communications': 'XLC'
    }
    
    sector_data = []
    
    # Generate a cache timestamp (changes every hour)
    cache_timestamp = datetime.now().strftime('%Y%m%d%H')
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    for sector, etf in sector_etfs.items():
        try:
            # Use caching to get data
            hist = get_stock_history(etf, start_date_str, end_date_str, "1d", cache_timestamp)
            
            if hist.empty:
                continue
                
            # Calculate 1-day and 5-day returns
            latest_close = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else latest_close
            first_close = hist['Close'].iloc[0] if len(hist) > 0 else latest_close
            
            one_day_return = ((latest_close / prev_close) - 1) * 100
            five_day_return = ((latest_close / first_close) - 1) * 100
            
            sector_data.append({
                'sector': sector,
                'etf': etf,
                'price': latest_close,
                'one_day_return': one_day_return,
                'five_day_return': five_day_return
            })
        except Exception as e:
            print(f"Error getting data for {sector} ({etf}): {e}")
    
    # Sort by 1-day performance
    sector_data.sort(key=lambda x: x['one_day_return'], reverse=True)
    
    return sector_data

def get_market_breadth():
    """Calculate market breadth using S&P 500 stocks"""
    try:
        # Get S&P 500 tickers (would need to be implemented or use a static list)
        # For demonstration we'll use a smaller subset
        sample_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']
        
        advancing = 0
        declining = 0
        unchanged = 0
        above_ma50 = 0
        above_ma200 = 0
        
        # Generate a cache timestamp (changes every hour)
        cache_timestamp = datetime.now().strftime('%Y%m%d%H')
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=250)  # Enough data for 200-day MA
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        for ticker in sample_tickers:
            try:
                # Use caching to get data
                hist = get_stock_history(ticker, start_date_str, end_date_str, "1d", cache_timestamp)
                
                if len(hist) < 2:
                    continue
                
                # Check if advancing or declining
                if hist['Close'].iloc[-1] > hist['Close'].iloc[-2]:
                    advancing += 1
                elif hist['Close'].iloc[-1] < hist['Close'].iloc[-2]:
                    declining += 1
                else:
                    unchanged += 1
                
                # Check if above moving averages
                ma50 = hist['Close'].rolling(window=50).mean().iloc[-1]
                ma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
                
                if hist['Close'].iloc[-1] > ma50:
                    above_ma50 += 1
                
                if hist['Close'].iloc[-1] > ma200:
                    above_ma200 += 1
                
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue
        
        # Calculate breadth metrics
        total = advancing + declining + unchanged
        if total > 0:
            advance_decline_ratio = advancing / declining if declining > 0 else float('inf')
            percent_advancing = (advancing / total) * 100
            percent_above_ma50 = (above_ma50 / total) * 100
            percent_above_ma200 = (above_ma200 / total) * 100
        else:
            advance_decline_ratio = 0
            percent_advancing = 0
            percent_above_ma50 = 0
            percent_above_ma200 = 0
        
        breadth_data = {
            'advancing': advancing,
            'declining': declining,
            'unchanged': unchanged,
            'advance_decline_ratio': advance_decline_ratio,
            'percent_advancing': percent_advancing,
            'above_ma50': above_ma50,
            'percent_above_ma50': percent_above_ma50,
            'above_ma200': above_ma200,
            'percent_above_ma200': percent_above_ma200,
            'total_analyzed': total
        }
        
        return breadth_data
        
    except Exception as e:
        print(f"Error calculating market breadth: {e}")
        return None

def get_intraday_data(ticker_symbol, interval='30m', days=5):
    """Get intraday price data for a ticker"""
    try:
        # Generate a cache timestamp (changes every hour)
        cache_timestamp = datetime.now().strftime('%Y%m%d%H')
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Use caching to get data
        intraday = get_stock_history(ticker_symbol, start_date_str, end_date_str, interval, cache_timestamp)
        
        if intraday.empty:
            return None
        
        # Format for charts
        intraday_data = {
            'datetime': intraday.index.strftime('%Y-%m-%d %H:%M').tolist(),
            'open': intraday['Open'].tolist(),
            'high': intraday['High'].tolist(),
            'low': intraday['Low'].tolist(),
            'close': intraday['Close'].tolist(),
            'volume': intraday['Volume'].tolist()
        }
        
        return intraday_data
    except Exception as e:
        print(f"Error getting intraday data: {e}")
        return None

def format_stock_for_json(stock_data):
    """Format stock data to ensure it's JSON serializable"""
    # Create a copy to avoid modifying the original
    json_safe = dict(stock_data)
    
    # Convert numpy types to Python native types
    for key, value in json_safe.items():
        if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
            json_safe[key] = int(value)
        elif isinstance(value, (np.float64, np.float32, np.float16)):
            json_safe[key] = float(value)
        elif isinstance(value, np.bool_):
            json_safe[key] = bool(value)
        elif pd.isna(value):
            json_safe[key] = None
            
    return json_safe

from requests.cookies import create_cookie
import yfinance.data as _data

def _wrap_cookie(cookie, session):
    """
    If cookie is just a str (cookie name), look up its value
    in session.cookies and wrap it into a real Cookie object.
    """
    if isinstance(cookie, str):
        value = session.cookies.get(cookie)
        return create_cookie(name=cookie, value=value)
    return cookie

def patch_yfdata_cookie_basic():
    """
    Monkey-patch YfData._get_cookie_basic so that
    it always returns a proper Cookie object,
    even when response.cookies is a simple dict.
    """
    original = _data.YfData._get_cookie_basic

    def _patched(self, timeout=30):
        cookie = original(self, timeout)
        return _wrap_cookie(cookie, self._session)

    _data.YfData._get_cookie_basic = _patched

if __name__ == "__main__":
    # Configure proper user-agent to avoid rate limiting
    session = get_yf_session()
    
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    print("Day Trading Analysis Tool")
    print("------------------------")
    print(f"Analyzing {len(tickers)} stocks...")
    
    ranked_stocks, failed_tickers = analyze_stocks(tickers)
    results = format_results(ranked_stocks, failed_tickers)
    
    print(results)
    print("\n" + generate_watchlist(ranked_stocks))
    print(save_to_csv(ranked_stocks))

