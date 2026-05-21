"""
Alpaca live market data bridge.

This module keeps the Alpaca integration intentionally small: a background
websocket maintains a latest-quote cache, and Flask endpoints read that cache.
If credentials or websocket-client are missing, callers get a clear disabled
status and the rest of TradeSmart keeps using its existing research data.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from modules.utils import log_message

try:
    import websocket
except Exception:  # pragma: no cover - optional dependency
    websocket = None


class AlpacaLiveData:
    def __init__(self):
        self.api_key = (
            os.getenv("ALPACA_API_KEY_ID")
            or os.getenv("APCA_API_KEY_ID")
            or os.getenv("ALPACA_API_KEY")
        )
        self.secret_key = (
            os.getenv("ALPACA_API_SECRET_KEY")
            or os.getenv("APCA_API_SECRET_KEY")
            or os.getenv("ALPACA_SECRET_KEY")
        )
        self.feed = os.getenv("ALPACA_DATA_FEED", "iex").lower()
        self.websocket_url = f"wss://stream.data.alpaca.markets/v2/{self.feed}"
        self.rest_base = "https://data.alpaca.markets"

        self._quotes: dict[str, dict[str, Any]] = {}
        self._desired_symbols: set[str] = set()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._ws = None
        self._connected = False
        self._authenticated = False
        self._last_error: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.secret_key and websocket is not None)

    def status(self) -> dict[str, Any]:
        missing = []
        if not self.api_key:
            missing.append("ALPACA_API_KEY_ID")
        if not self.secret_key:
            missing.append("ALPACA_API_SECRET_KEY")
        if websocket is None:
            missing.append("websocket-client")
        return {
            "provider": "alpaca",
            "configured": self.configured,
            "connected": self._connected,
            "authenticated": self._authenticated,
            "feed": self.feed,
            "missing": missing,
            "error": self._last_error,
        }

    def ensure_subscribed(self, symbols: list[str]) -> None:
        clean = {s.strip().upper() for s in symbols if s and s.strip()}
        if not clean:
            return

        with self._lock:
            before = set(self._desired_symbols)
            self._desired_symbols.update(clean)
            added = self._desired_symbols - before

        if not self.configured:
            return

        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run_forever, name="alpaca-live-data", daemon=True)
            self._thread.start()
            return

        if added:
            self._send_subscription(sorted(added))

    def snapshot(self, symbols: list[str], hydrate: bool = True) -> dict[str, Any]:
        clean = sorted({s.strip().upper() for s in symbols if s and s.strip()})
        self.ensure_subscribed(clean)
        if hydrate:
            self._hydrate_missing_or_stale(clean)

        with self._lock:
            quotes = {symbol: self._quotes.get(symbol) for symbol in clean}

        return {
            **self.status(),
            "quotes": quotes,
        }

    def historical_bars(self, symbol: str, timeframe: str = "1Min", days: int = 1) -> tuple[list[dict[str, Any]], str | None]:
        clean = symbol.strip().upper()
        if not self.api_key or not self.secret_key:
            return [], "Alpaca credentials are not configured."

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=max(1, days))
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }
        params = {
            "symbols": clean,
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 10000,
            "feed": self.feed,
            "adjustment": "raw",
        }
        try:
            response = requests.get(
                f"{self.rest_base}/v2/stocks/bars",
                headers=headers,
                params=params,
                timeout=6,
            )
            if not response.ok:
                return [], f"Alpaca bars request failed: {response.status_code} {response.text[:160]}"
            payload = response.json()
            return payload.get("bars", {}).get(clean, []) or [], None
        except Exception as exc:
            return [], str(exc)

    def _run_forever(self) -> None:
        while True:
            with self._lock:
                has_symbols = bool(self._desired_symbols)
            if not has_symbols:
                return

            try:
                self._connected = False
                self._authenticated = False
                self._ws = websocket.WebSocketApp(
                    self.websocket_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:
                self._last_error = str(exc)
                log_message(f"Alpaca websocket error: {exc}")

            time.sleep(3)

    def _on_open(self, ws) -> None:
        self._connected = True
        self._last_error = None
        ws.send(json.dumps({
            "action": "auth",
            "key": self.api_key,
            "secret": self.secret_key,
        }))

    def _on_message(self, ws, message: str) -> None:
        try:
            payload = json.loads(message)
            items = payload if isinstance(payload, list) else [payload]
        except Exception:
            return

        for item in items:
            if not isinstance(item, dict):
                continue
            typ = item.get("T")
            if typ == "success" and item.get("msg") == "authenticated":
                self._authenticated = True
                with self._lock:
                    symbols = sorted(self._desired_symbols)
                self._send_subscription(symbols)
            elif typ == "error":
                self._last_error = item.get("msg") or str(item)
                log_message(f"Alpaca stream error: {self._last_error}")
            elif typ == "q":
                self._update_quote(item)
            elif typ == "t":
                self._update_trade(item)
            elif typ == "b":
                self._update_bar(item)

    def _on_error(self, _ws, error) -> None:
        self._last_error = str(error)
        log_message(f"Alpaca websocket callback error: {error}")

    def _on_close(self, _ws, _status_code, _message) -> None:
        self._connected = False
        self._authenticated = False

    def _send_subscription(self, symbols: list[str]) -> None:
        if not symbols or not self._ws or not self._authenticated:
            return
        try:
            self._ws.send(json.dumps({
                "action": "subscribe",
                "trades": symbols,
                "quotes": symbols,
            }))
        except Exception as exc:
            self._last_error = str(exc)

    def _update_quote(self, item: dict[str, Any]) -> None:
        symbol = str(item.get("S", "")).upper()
        if not symbol:
            return
        bid = _num(item.get("bp"))
        ask = _num(item.get("ap"))
        mid = (bid + ask) / 2 if bid and ask else None
        now = _iso_now()
        with self._lock:
            existing = self._quotes.get(symbol, {})
            self._quotes[symbol] = {
                **existing,
                "symbol": symbol,
                "bid": bid,
                "ask": ask,
                "bid_size": _num(item.get("bs")),
                "ask_size": _num(item.get("as")),
                "price": mid or existing.get("last") or existing.get("price"),
                "source": "alpaca",
                "feed": self.feed,
                "timestamp": item.get("t") or now,
                "received_at": now,
            }

    def _update_trade(self, item: dict[str, Any]) -> None:
        symbol = str(item.get("S", "")).upper()
        if not symbol:
            return
        price = _num(item.get("p"))
        now = _iso_now()
        with self._lock:
            existing = self._quotes.get(symbol, {})
            self._quotes[symbol] = {
                **existing,
                "symbol": symbol,
                "last": price,
                "last_size": _num(item.get("s")),
                "price": price or existing.get("price"),
                "source": "alpaca",
                "feed": self.feed,
                "timestamp": item.get("t") or now,
                "received_at": now,
            }

    def _update_bar(self, item: dict[str, Any]) -> None:
        symbol = str(item.get("S", "")).upper()
        close = _num(item.get("c"))
        if not symbol or close is None:
            return
        now = _iso_now()
        with self._lock:
            existing = self._quotes.get(symbol, {})
            self._quotes[symbol] = {
                **existing,
                "symbol": symbol,
                "last": close,
                "price": close or existing.get("price"),
                "source": "alpaca",
                "feed": self.feed,
                "timestamp": item.get("t") or now,
                "received_at": now,
            }

    def _hydrate_missing_or_stale(self, symbols: list[str]) -> None:
        if not self.api_key or not self.secret_key or not symbols:
            return
        now = time.time()
        with self._lock:
            needed = [
                s for s in symbols
                if s not in self._quotes or now - _parse_received(self._quotes[s].get("received_at")) > 15
            ]
        if not needed:
            return

        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }
        params = {"symbols": ",".join(needed), "feed": self.feed}
        try:
            quote_resp = requests.get(
                f"{self.rest_base}/v2/stocks/quotes/latest",
                headers=headers,
                params=params,
                timeout=4,
            )
            if quote_resp.ok:
                for symbol, quote in quote_resp.json().get("quotes", {}).items():
                    self._update_quote({"T": "q", "S": symbol, **quote})

            trade_resp = requests.get(
                f"{self.rest_base}/v2/stocks/trades/latest",
                headers=headers,
                params=params,
                timeout=4,
            )
            if trade_resp.ok:
                for symbol, trade in trade_resp.json().get("trades", {}).items():
                    self._update_trade({"T": "t", "S": symbol, **trade})
        except Exception as exc:
            self._last_error = str(exc)


def _num(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_received(value) -> float:
    if not value:
        return 0
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0


alpaca_live_data = AlpacaLiveData()
