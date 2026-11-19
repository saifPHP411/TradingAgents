"""Read-only Bybit V5 public-data adapter for linear USDT perpetual research.

Only public GET market-data endpoints are implemented. There are no credentials,
account endpoints, leverage mutations, or order-submission methods.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from urllib.parse import urlsplit

import pandas as pd

from tradingagents.crypto.perpetuals import LinearPerpetualInstrument
from tradingagents.crypto.transport import fetch_json

ALLOWED_HOSTS = {"api.bybit.com", "api-demo.bybit.com", "api-testnet.bybit.com"}
BASE_URL = "https://api.bybit.com"
INTERVALS = {"4h": ("240", 14_400_000)}
_SYMBOL = re.compile(r"^[A-Z0-9]{3,30}$")


class BybitPublicDataError(RuntimeError):
    pass


class BybitLinearPublicClient:
    """Minimal public V5 reader with bounded pagination and opt-in forming candles."""

    def __init__(self, base_url: str = BASE_URL, transport: Callable | None = None, timeout: float = 15.0):
        url = urlsplit(base_url)
        if url.scheme != "https" or url.hostname not in ALLOWED_HOSTS or url.port not in (None, 443) or url.path not in ("", "/") or url.query or url.fragment or url.username or url.password:
            raise ValueError("only approved Bybit public HTTPS hosts are allowed")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.base_url, self.transport, self.timeout = base_url.rstrip("/"), transport, timeout

    @staticmethod
    def _symbol(symbol: str) -> str:
        value = str(symbol).strip().upper()
        if not _SYMBOL.fullmatch(value):
            raise ValueError("invalid Bybit symbol")
        return value

    def _get(self, endpoint: str, params: Mapping[str, object]) -> dict:
        payload = fetch_json(url=self.base_url + endpoint, params=params, headers={}, timeout=self.timeout, transport=self.transport)
        if not isinstance(payload, dict) or payload.get("retCode") != 0 or not isinstance(payload.get("result"), dict):
            raise BybitPublicDataError("Bybit public API returned an invalid or unsuccessful response")
        return payload

    @staticmethod
    def _number(value: object, name: str, allow_zero: bool = False) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            raise BybitPublicDataError(f"missing or invalid {name}") from None
        if not math.isfinite(parsed) or (parsed < 0 if allow_zero else parsed <= 0):
            raise BybitPublicDataError(f"invalid {name}")
        return parsed

    def instrument_info(self, symbol: str) -> LinearPerpetualInstrument:
        market = self._symbol(symbol)
        result = self._get("/v5/market/instruments-info", {"category": "linear", "symbol": market})["result"]
        items = result.get("list")
        if not isinstance(items, list) or len(items) != 1 or items[0].get("symbol") != market:
            raise BybitPublicDataError("instrument result did not match requested symbol")
        row = items[0]
        lot, price, lev = row.get("lotSizeFilter"), row.get("priceFilter"), row.get("leverageFilter")
        if not all(isinstance(v, dict) for v in (lot, price, lev)):
            raise BybitPublicDataError("instrument response omitted lot, price or leverage filters")
        try:
            funding_minutes = int(row["fundingInterval"])
        except (KeyError, TypeError, ValueError):
            raise BybitPublicDataError("invalid funding interval") from None
        return LinearPerpetualInstrument(
            symbol=market, base_coin=str(row.get("baseCoin", "")).upper(),
            quote_coin=str(row.get("quoteCoin", "")).upper(), settle_coin=str(row.get("settleCoin", "")).upper(),
            status=str(row.get("status", "")), contract_type=str(row.get("contractType", "")),
            min_quantity=self._number(lot.get("minOrderQty"), "minOrderQty"),
            max_market_quantity=self._number(lot.get("maxMktOrderQty"), "maxMktOrderQty"),
            quantity_step=self._number(lot.get("qtyStep"), "qtyStep"),
            min_notional=self._number(lot.get("minNotionalValue"), "minNotionalValue"),
            tick_size=self._number(price.get("tickSize"), "tickSize"),
            min_leverage=self._number(lev.get("minLeverage"), "minLeverage"),
            max_leverage=self._number(lev.get("maxLeverage"), "maxLeverage"),
            leverage_step=self._number(lev.get("leverageStep"), "leverageStep"),
            funding_interval_minutes=funding_minutes,
        )

    def _candles(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        endpoint: str,
        with_volume: bool,
        *,
        include_current: bool = False,
    ) -> pd.DataFrame:
        market = self._symbol(symbol)
        if isinstance(start_ms, bool) or isinstance(end_ms, bool) or not isinstance(start_ms, int) or not isinstance(end_ms, int) or start_ms <= 0 or end_ms <= start_ms:
            raise ValueError("start_ms/end_ms must be positive ordered integers")
        interval, duration = INTERVALS["4h"]
        collected: dict[int, list[float]] = {}
        cursor_end = end_ms
        for _ in range(50):
            payload = self._get(endpoint, {"category": "linear", "symbol": market, "interval": interval, "start": start_ms, "end": cursor_end, "limit": 1000})
            server_ms = int(payload.get("time", 0))
            rows = payload["result"].get("list", [])
            if not isinstance(rows, list) or not rows:
                break
            stamps = []
            for row in rows:
                if not isinstance(row, list) or len(row) < (7 if with_volume else 5):
                    raise BybitPublicDataError("malformed candle row")
                stamp = int(row[0])
                stamps.append(stamp)
                completed = stamp + duration <= server_ms
                forming = include_current and stamp <= server_ms < stamp + duration
                if start_ms <= stamp <= end_ms and (completed or forming):
                    values = [self._number(v, "OHLC price") for v in row[1:5]]
                    if with_volume:
                        values.append(self._number(row[5], "volume", True))
                    collected[stamp] = values
            oldest = min(stamps)
            if oldest <= start_ms:
                break
            cursor_end = oldest - 1
        else:
            raise BybitPublicDataError("candle request exceeded 50 pages")
        if not collected:
            raise BybitPublicDataError("no completed candles returned for the requested range")
        columns = ["open", "high", "low", "close"] + (["volume"] if with_volume else [])
        frame = pd.DataFrame.from_dict(collected, orient="index", columns=columns).sort_index()
        frame.index = pd.to_datetime(frame.index, unit="ms", utc=True)
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise BybitPublicDataError("duplicate or unordered candle timestamps")
        return frame

    def klines(
        self,
        symbol: str,
        *,
        start_ms: int,
        end_ms: int,
        include_current: bool = False,
    ) -> pd.DataFrame:
        """Fetch 4h candles; include the in-progress candle only when explicitly requested."""
        return self._candles(
            symbol,
            start_ms,
            end_ms,
            "/v5/market/kline",
            True,
            include_current=include_current,
        )

    def mark_price_klines(self, symbol: str, *, start_ms: int, end_ms: int) -> pd.DataFrame:
        return self._candles(symbol, start_ms, end_ms, "/v5/market/mark-price-kline", False)

    def funding_rates(self, symbol: str, *, start_ms: int, end_ms: int, interval_minutes: int = 480) -> pd.Series:
        market = self._symbol(symbol)
        if isinstance(start_ms, bool) or isinstance(end_ms, bool) or not isinstance(start_ms, int) or not isinstance(end_ms, int) or start_ms <= 0 or end_ms <= start_ms:
            raise ValueError("start_ms/end_ms must be positive ordered integers")
        if isinstance(interval_minutes, bool) or not isinstance(interval_minutes, int) or interval_minutes < 1:
            raise ValueError("interval_minutes must be a positive integer")
        interval_ms = interval_minutes * 60_000
        events: dict[int, float] = {}
        cursor = start_ms
        for _ in range(50):
            window_end = min(end_ms, cursor + interval_ms * 199)
            payload = self._get("/v5/market/funding/history", {"category": "linear", "symbol": market, "startTime": cursor, "endTime": window_end, "limit": 200})
            rows = payload["result"].get("list", [])
            if not isinstance(rows, list):
                raise BybitPublicDataError("malformed funding response")
            for row in rows:
                try:
                    stamp = int(row["fundingRateTimestamp"])
                    rate = float(row["fundingRate"])
                except (KeyError, TypeError, ValueError):
                    raise BybitPublicDataError("malformed funding event") from None
                if not math.isfinite(rate) or abs(rate) > 0.05:
                    raise BybitPublicDataError("funding rate failed sanity bound")
                if start_ms <= stamp <= end_ms:
                    events[stamp] = rate
            if window_end >= end_ms:
                break
            cursor = window_end + 1
        else:
            raise BybitPublicDataError("funding request exceeded 50 pages")
        if not events:
            raise BybitPublicDataError("no funding events returned")
        stamps = sorted(events)
        index = pd.to_datetime(stamps, unit="ms", utc=True)
        series = pd.Series([events[t] for t in stamps], index=index, name="funding_rate", dtype=float)
        if len(series) > 1 and not (series.index.to_series().diff().dropna() == pd.Timedelta(minutes=interval_minutes)).all():
            raise BybitPublicDataError("funding history contains missing or irregular settlement events")
        return series
