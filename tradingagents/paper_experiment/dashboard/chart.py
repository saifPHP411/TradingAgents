"""Bounded, cached public OHLC history for the dashboard chart.

Display-only data: Binance public Spot candles, no credentials and no order
endpoints. Only completed bars are returned.
"""
from __future__ import annotations

import math
import threading
import time
from datetime import UTC, datetime
from typing import Any

from tradingagents.crypto.symbols import parse_pair

CHART_INTERVAL_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400}
DEFAULT_CHART_INTERVAL = "1h"
DEFAULT_CHART_LIMIT = 160
MIN_CHART_LIMIT = 60
MAX_CHART_LIMIT = 240
DEFAULT_CACHE_TTL_SECONDS = 20.0


class CryptoChartService:
    """Fetch/cache a small history window. Constructing it performs no I/O."""

    def __init__(self, *, client_factory=None, clock=None,
                 ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS) -> None:
        if not math.isfinite(float(ttl_seconds)) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be finite and positive")
        self._client_factory = client_factory
        self._clock = clock or time.monotonic
        self._ttl_seconds = float(ttl_seconds)
        self._lock = threading.Lock()
        self._client: Any | None = None
        self._cache: dict[tuple[str, str, int], tuple[float, dict[str, Any]]] = {}

    def get(self, symbol: str, interval: str = DEFAULT_CHART_INTERVAL,
            limit: int = DEFAULT_CHART_LIMIT) -> dict[str, Any]:
        """Return validated completed OHLC candles for one configured pair."""
        pair = parse_pair(symbol).canonical
        if interval not in CHART_INTERVAL_SECONDS:
            raise ValueError("unsupported chart interval")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("chart limit must be an integer")
        if not MIN_CHART_LIMIT <= limit <= MAX_CHART_LIMIT:
            raise ValueError(
                f"chart limit must be between {MIN_CHART_LIMIT} and {MAX_CHART_LIMIT}"
            )

        key = (pair, interval, limit)
        now = float(self._clock())
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None and now - cached[0] < self._ttl_seconds:
                return _copy_payload(cached[1])
            try:
                if self._client is None:
                    factory = self._client_factory
                    if factory is None:
                        from tradingagents.crypto.binance import BinancePublicClient
                        factory = BinancePublicClient
                    self._client = factory()
                source = self._client.klines(pair, interval, limit=limit)
                candles = _normalise_candles(source, pair, interval, limit)
                if not candles:
                    raise ValueError("no completed candles returned")
                payload = {
                    "symbol": pair,
                    "interval": interval,
                    "provider": "binance_public_spot",
                    "status": "ok",
                    "candles": candles,
                    "refreshed_at": datetime.now(UTC).isoformat(),
                    "detail": None,
                }
                self._cache[key] = (now, payload)
                return _copy_payload(payload)
            except Exception as exc:  # noqa: BLE001 - chart errors are display state
                if cached is not None:
                    stale = _copy_payload(cached[1])
                    stale["status"] = "stale"
                    stale["detail"] = f"refresh failed: {type(exc).__name__}"
                    return stale
                return {
                    "symbol": pair,
                    "interval": interval,
                    "provider": "binance_public_spot",
                    "status": "error",
                    "candles": [],
                    "refreshed_at": None,
                    "detail": f"chart data unavailable: {type(exc).__name__}",
                }


def _normalise_candles(source: Any, symbol: str, interval: str,
                       limit: int) -> list[dict[str, float | int]]:
    """Validate vendor candles and keep only closed, unique bars."""
    if not isinstance(source, (list, tuple)):
        raise ValueError("market-data source did not return a candle list")
    rows: dict[int, dict[str, float | int]] = {}
    now = datetime.now(UTC)
    for candle in source:
        if getattr(candle, "symbol", symbol) not in (symbol, symbol.replace("/", "")):
            raise ValueError("market-data source returned a different symbol")
        if getattr(candle, "interval", interval) != interval:
            raise ValueError("market-data source returned a different interval")
        opened = getattr(candle, "open_time", None)
        closed = getattr(candle, "close_time", None)
        if not isinstance(opened, datetime) or not isinstance(closed, datetime):
            raise ValueError("candle timestamps must be datetimes")
        if opened.tzinfo is None or closed.tzinfo is None:
            raise ValueError("candle timestamps must be timezone-aware")
        if closed > now:
            continue
        values = {
            name: _finite_positive(getattr(candle, name, None), name)
            for name in ("open", "high", "low", "close")
        }
        if values["high"] < max(values["open"], values["close"], values["low"]):
            raise ValueError("candle high is below another OHLC value")
        if values["low"] > min(values["open"], values["close"], values["high"]):
            raise ValueError("candle low is above another OHLC value")
        timestamp = int(opened.timestamp())
        rows[timestamp] = {"time": timestamp, **values}
    return [rows[key] for key in sorted(rows)][-limit:]


def _finite_positive(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"candle {name} is not numeric") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"candle {name} must be finite and positive")
    return result


def _copy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "candles": [dict(row) for row in payload["candles"]]}


__all__ = [
    "CHART_INTERVAL_SECONDS", "CryptoChartService", "DEFAULT_CHART_INTERVAL",
    "DEFAULT_CHART_LIMIT", "MAX_CHART_LIMIT", "MIN_CHART_LIMIT",
]
