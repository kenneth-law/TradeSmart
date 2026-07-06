"""
Fast technical-only ranking for large ticker universes.

The detailed stock view uses per-ticker metadata, intraday, and sentiment calls.
That is useful for one name, but too slow for screening hundreds of tickers. This
module performs one bulk OHLCV fetch and computes the ranking fields locally.
"""

from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import yfinance_cookie_patch

from modules.data_retrieval import get_yf_session
from modules.utils import log_message


yfinance_cookie_patch.patch_yfdata_cookie_basic()

CACHE_DIR = Path("cache/fast_ranking")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def rank_stocks_fast(tickers: list[str], days: int = 365) -> tuple[list[dict], list[str]]:
    """
    Rank tickers using daily OHLCV data only.

    This intentionally skips per-ticker `Ticker.info`, intraday bars, and news
    sentiment. Those calls dominate runtime and prevent large-universe screens.
    """
    clean_tickers = _clean_tickers(tickers)
    if not clean_tickers:
        return [], []

    start = time.perf_counter()
    history = _download_bulk_history(clean_tickers, days=days)
    ranked: list[dict] = []
    failed: list[str] = []

    for ticker in clean_tickers:
        frame = _frame_for_ticker(history, ticker, len(clean_tickers))
        if frame.empty:
            failed.append(ticker)
            continue

        stock_data = _score_frame(ticker, frame)
        if stock_data is None:
            failed.append(ticker)
            continue
        ranked.append(stock_data)

    ranked.sort(key=lambda row: row.get("day_trading_score", 0), reverse=True)
    elapsed = time.perf_counter() - start
    log_message(
        f"Fast technical ranking completed for {len(ranked)}/{len(clean_tickers)} "
        f"tickers in {elapsed:.2f}s"
    )
    return ranked, failed


def _download_bulk_history(tickers: list[str], days: int) -> pd.DataFrame:
    cache_key = _cache_key(tickers, days)
    cache_path = CACHE_DIR / f"{cache_key}.pkl"
    ttl_seconds = int(os.getenv("TRADESMART_FAST_RANK_CACHE_TTL", "900"))

    if cache_path.exists() and time.time() - cache_path.stat().st_mtime <= ttl_seconds:
        try:
            log_message("Using cached bulk price history for fast ranking")
            return pd.read_pickle(cache_path)
        except Exception as exc:
            log_message(f"Fast ranking cache ignored: {exc}")

    log_message(f"Downloading daily bars for {len(tickers)} tickers in one batch")

    download_kwargs = {
        "tickers": " ".join(tickers),
        "interval": "1d",
        "group_by": "ticker",
        "auto_adjust": False,
        "actions": False,
        "progress": False,
        "threads": True,
        "session": get_yf_session(),
        "timeout": 12,
    }
    if days == 365:
        download_kwargs["period"] = "1y"
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        download_kwargs["start"] = start_date.strftime("%Y-%m-%d")
        download_kwargs["end"] = end_date.strftime("%Y-%m-%d")

    data = yf.download(**download_kwargs)

    try:
        data.to_pickle(cache_path)
    except Exception as exc:
        log_message(f"Fast ranking cache write skipped: {exc}")

    return data


def _score_frame(ticker: str, frame: pd.DataFrame) -> dict | None:
    hist = frame.copy()
    hist = hist.rename(columns={col: str(col).title() for col in hist.columns})
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(hist.columns):
        return None

    hist = hist[list(required)].apply(pd.to_numeric, errors="coerce").dropna(subset=["Close"])
    if len(hist) < 30:
        return None

    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]
    volume = hist["Volume"].fillna(0)

    current_price = _safe_float(close.iloc[-1])
    prev_close = _safe_float(close.iloc[-2]) if len(close) > 1 else current_price
    if current_price <= 0:
        return None

    ma5 = close.rolling(window=5).mean()
    ma10 = close.rolling(window=10).mean()
    ma20 = close.rolling(window=20).mean()
    ma50 = close.rolling(window=50).mean()
    ma200 = close.rolling(window=200).mean()

    above_ma5 = _gt(current_price, ma5.iloc[-1])
    above_ma10 = _gt(current_price, ma10.iloc[-1])
    above_ma20 = _gt(current_price, ma20.iloc[-1])
    above_ma50 = _gt(current_price, ma50.iloc[-1])
    above_ma200 = _gt(current_price, ma200.iloc[-1])

    golden_cross = False
    death_cross = False
    if len(hist) > 201:
        ma50_yesterday = ma50.iloc[-2]
        ma200_yesterday = ma200.iloc[-2]
        ma50_today = ma50.iloc[-1]
        ma200_today = ma200.iloc[-1]
        if all(pd.notna(v) for v in [ma50_yesterday, ma200_yesterday, ma50_today, ma200_today]):
            golden_cross = ma50_yesterday <= ma200_yesterday and ma50_today > ma200_today
            death_cross = ma50_yesterday >= ma200_yesterday and ma50_today < ma200_today

    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr14 = true_range.rolling(window=14).mean()

    bb_middle = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_range = _safe_float(bb_upper.iloc[-1] - bb_lower.iloc[-1])
    bb_position = (current_price - _safe_float(bb_lower.iloc[-1])) / bb_range if bb_range > 0 else 0.5

    delta = close.diff()
    gain7 = delta.where(delta > 0, 0).rolling(window=7).mean()
    loss7 = -delta.where(delta < 0, 0).rolling(window=7).mean()
    rsi7 = 100 - (100 / (1 + (gain7 / loss7)))

    gain14 = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss14 = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rsi14 = 100 - (100 / (1 + (gain14 / loss14)))

    returns = {
        period: close.pct_change(periods=period) * 100
        for period in (1, 3, 5, 30, 90)
    }

    lowest_low = low.rolling(window=14).min()
    highest_high = high.rolling(window=14).max()
    stoch_k_series = ((close - lowest_low) / (highest_high - lowest_low)) * 100
    stoch_d_series = stoch_k_series.rolling(window=3).mean()

    obv_step = np.select(
        [close > close.shift(1), close < close.shift(1)],
        [volume, -volume],
        default=0,
    )
    obv = pd.Series(obv_step, index=hist.index).cumsum()
    obv_ma20 = obv.rolling(window=20).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    volume_ma5 = volume.rolling(window=5).mean()
    volume_ratio = _safe_float(volume.iloc[-1] / volume_ma5.iloc[-1]) if _safe_float(volume_ma5.iloc[-1]) > 0 else 1.0

    recent_rsi7 = _safe_float(rsi7.iloc[-1], 50)
    recent_rsi14 = _safe_float(rsi14.iloc[-1], 50)
    atr = _safe_float(atr14.iloc[-1])
    atr_pct = (atr / current_price) * 100 if current_price else 0
    return_1d = _safe_float(returns[1].iloc[-1])
    return_3d = _safe_float(returns[3].iloc[-1])
    return_5d = _safe_float(returns[5].iloc[-1])
    return_30d = _safe_float(returns[30].iloc[-1])
    return_90d = _safe_float(returns[90].iloc[-1])
    stoch_k = _safe_float(stoch_k_series.iloc[-1], 50)
    stoch_d = _safe_float(stoch_d_series.iloc[-1], 50)
    stoch_trend = "bullish" if stoch_k > stoch_d else "bearish"
    stoch_overbought = stoch_k > 80
    stoch_oversold = stoch_k < 20
    obv_trend = "bullish" if _safe_float(obv.iloc[-1]) > _safe_float(obv_ma20.iloc[-1]) else "bearish"
    macd_value = _safe_float(macd.iloc[-1])
    macd_signal_value = _safe_float(macd_signal.iloc[-1])
    macd_hist_value = _safe_float(macd_hist.iloc[-1])
    macd_trend = "bullish" if macd_value > macd_signal_value else "bearish"

    scores = _score_components(
        above_ma5=above_ma5,
        above_ma10=above_ma10,
        above_ma20=above_ma20,
        above_ma50=above_ma50,
        above_ma200=above_ma200,
        golden_cross=golden_cross,
        death_cross=death_cross,
        recent_rsi7=recent_rsi7,
        bb_position=bb_position,
        macd_trend=macd_trend,
        macd_hist=macd_hist_value,
        stoch_trend=stoch_trend,
        stoch_overbought=stoch_overbought,
        stoch_oversold=stoch_oversold,
        obv_trend=obv_trend,
        return_1d=return_1d,
        return_3d=return_3d,
        return_30d=return_30d,
        return_90d=return_90d,
        atr_pct=atr_pct,
        avg_intraday_range=0,
        sentiment_score=0,
        gap_ups=0,
        gap_downs=0,
        premarket_available=False,
        premarket_change=0,
        volume_ratio=volume_ratio,
    )

    return {
        "ticker": ticker,
        "company_name": ticker,
        "sector": "Unknown",
        "industry": "Unknown",
        "market_cap": 0,
        "current_price": current_price,
        "prev_close": prev_close,
        "atr": atr,
        "atr_pct": atr_pct,
        "rsi7": recent_rsi7,
        "rsi14": recent_rsi14,
        "bb_position": bb_position,
        "above_ma5": above_ma5,
        "above_ma10": above_ma10,
        "above_ma20": above_ma20,
        "above_ma50": above_ma50,
        "above_ma200": above_ma200,
        "golden_cross": golden_cross,
        "death_cross": death_cross,
        "macd": macd_value,
        "macd_signal": macd_signal_value,
        "macd_hist": macd_hist_value,
        "macd_trend": macd_trend,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "stoch_trend": stoch_trend,
        "stoch_overbought": stoch_overbought,
        "stoch_oversold": stoch_oversold,
        "obv_trend": obv_trend,
        "return_1d": return_1d,
        "return_3d": return_3d,
        "return_5d": return_5d,
        "return_30d": return_30d,
        "return_90d": return_90d,
        "volume_ratio": volume_ratio,
        "avg_intraday_range": 0,
        "gap_ups_5d": 0,
        "gap_downs_5d": 0,
        "premarket_change": None,
        "news_sentiment_score": 0,
        "news_sentiment_label": "Neutral",
        "raw_sentiment_score": 0,
        "weighted_sentiment_score": 0,
        "headlines": [],
        "technical_score": scores["technical_score"],
        "long_term_score": scores["long_term_score"],
        "volatility_score": scores["volatility_score"],
        "news_sentiment_score_normalized": 50,
        "gap_score": scores["gap_score"],
        "volume_score": scores["volume_score"],
        "day_trading_score": scores["day_trading_score"],
        "day_trading_strategy": scores["strategy"],
        "strategy_details": scores["strategy_details"],
        "ranking_mode": "fast_technical",
    }


def _score_components(**values) -> dict:
    technical_score = 0

    if values["above_ma5"]:
        technical_score += 8
    if values["above_ma10"]:
        technical_score += 4
    if values["above_ma20"]:
        technical_score += 4

    recent_rsi7 = values["recent_rsi7"]
    if recent_rsi7 < 30:
        technical_score += 12
    elif recent_rsi7 < 40:
        technical_score += 8
    elif recent_rsi7 > 70:
        technical_score -= 12
    elif recent_rsi7 > 60:
        technical_score -= 8

    if values["bb_position"] < 0.2:
        technical_score += 12
    elif values["bb_position"] > 0.8:
        technical_score -= 12

    if values["macd_trend"] == "bullish" and values["macd_hist"] > 0:
        technical_score += 12
    elif values["macd_trend"] == "bearish" and values["macd_hist"] < 0:
        technical_score -= 12

    if values["stoch_trend"] == "bullish" and values["stoch_oversold"]:
        technical_score += 10
    elif values["stoch_trend"] == "bearish" and values["stoch_overbought"]:
        technical_score -= 10

    if values["obv_trend"] == "bullish":
        technical_score += 8
    elif values["obv_trend"] == "bearish":
        technical_score -= 8

    if values["return_1d"] > 1.5:
        technical_score += 8
    elif values["return_1d"] < -1.5:
        technical_score -= 8

    if values["return_3d"] > 3:
        technical_score += 4
    elif values["return_3d"] < -3:
        technical_score -= 4

    long_term_score = 50
    long_term_score += 10 if values["above_ma50"] else -10
    long_term_score += 15 if values["above_ma200"] else -15

    if values["golden_cross"]:
        long_term_score += 20
    elif values["death_cross"]:
        long_term_score -= 20

    if values["return_30d"] > 8:
        long_term_score += 10
    elif values["return_30d"] < -8:
        long_term_score -= 10
    elif values["return_30d"] > 4:
        long_term_score += 5
    elif values["return_30d"] < -4:
        long_term_score -= 5

    if values["return_90d"] > 15:
        long_term_score += 15
    elif values["return_90d"] < -15:
        long_term_score -= 15
    elif values["return_90d"] > 8:
        long_term_score += 8
    elif values["return_90d"] < -8:
        long_term_score -= 8

    technical_score = max(0, min(100, technical_score + 50))
    long_term_score = max(0, min(100, long_term_score))

    atr_pct = values["atr_pct"]
    if atr_pct > 3:
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

    if values["avg_intraday_range"] > 2:
        volatility_score += 10
    volatility_score = min(100, volatility_score)

    sentiment_score = values["sentiment_score"]
    news_sentiment_score = int((sentiment_score + 1) * 50)

    gap_score = 0
    gap_frequency = (values["gap_ups"] + values["gap_downs"]) / 5 if values["gap_ups"] + values["gap_downs"] > 0 else 0
    gap_score += gap_frequency * 30
    if values["premarket_available"]:
        premarket_change = abs(values["premarket_change"])
        if premarket_change > 3:
            gap_score += 50
        elif premarket_change > 1.5:
            gap_score += 30
        elif premarket_change > 0.5:
            gap_score += 15
    if abs(sentiment_score) > 0.5:
        gap_score += 20
    gap_score = min(100, gap_score)

    volume_ratio = values["volume_ratio"]
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

    day_trading_score = (
        0.35 * technical_score
        + 0.15 * long_term_score
        + 0.20 * volatility_score
        + 0.10 * news_sentiment_score
        + 0.10 * gap_score
        + 0.10 * volume_score
    )

    if day_trading_score >= 70:
        strategy = "Strong Buy"
        strategy_details = (
            "Bullish technicals with strong long-term trend support"
            if long_term_score >= 70
            else "Bullish technicals with good volatility for day trading"
        )
    elif day_trading_score >= 60:
        strategy = "Buy"
        strategy_details = (
            "Favorable conditions with positive long-term outlook"
            if long_term_score >= 60
            else "Favorable conditions for a long day trade"
        )
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
        strategy_details = (
            "Bearish conditions with negative long-term outlook"
            if long_term_score <= 30
            else "Bearish short-term conditions despite average long-term trend"
        )
    else:
        strategy = "Strong Sell"
        strategy_details = (
            "Strongly bearish technicals with poor long-term performance"
            if long_term_score <= 30
            else "Strongly bearish technicals or insufficient volatility"
        )

    return {
        "technical_score": technical_score,
        "long_term_score": long_term_score,
        "volatility_score": volatility_score,
        "gap_score": gap_score,
        "volume_score": volume_score,
        "day_trading_score": day_trading_score,
        "strategy": strategy,
        "strategy_details": strategy_details,
    }


def _frame_for_ticker(data: pd.DataFrame, ticker: str, ticker_count: int) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0)
        level1 = data.columns.get_level_values(1)
        if ticker in level0:
            return data[ticker].dropna(how="all")
        if ticker in level1:
            return data.xs(ticker, axis=1, level=1).dropna(how="all")
        return pd.DataFrame()

    if ticker_count == 1:
        return data.dropna(how="all")

    return pd.DataFrame()


def _clean_tickers(tickers: list[str]) -> list[str]:
    seen = set()
    clean = []
    for ticker in tickers:
        value = str(ticker or "").strip().upper()
        if value and value not in seen:
            seen.add(value)
            clean.append(value)
    return clean


def _cache_key(tickers: list[str], days: int) -> str:
    today = datetime.now().strftime("%Y%m%d")
    payload = "|".join(tickers) + f"|{days}|{today}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _gt(left: float, right: float) -> bool:
    return pd.notna(right) and left > _safe_float(right)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except Exception:
        return default
