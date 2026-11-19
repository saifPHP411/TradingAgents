"""Read-only Binance USD-M perpetual public-data adapter for cross-venue research.

This adapter supplies completed 4h traded and mark-price candles plus settled
funding rates. It submits no orders and reads no account data. Its output can be
used to stress-test a generic linear-perpetual strategy, but it is not evidence
of Bybit-specific fills, funding, instrument limits or Demo execution.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from urllib.parse import urlsplit

import pandas as pd

from tradingagents.crypto.perpetuals import LinearPerpetualInstrument
from tradingagents.crypto.transport import fetch_json

BASE_URL = "https://fapi.binance.com"
_INTERVAL = "4h"
_INTERVAL_MS = 4 * 60 * 60 * 1000
_PAGE_LIMIT = 1500
_FUNDING_PAGE_LIMIT = 1000
_SYMBOL = re.compile(r"^[A-Z0-9]{3,30}$")


class BinancePerpetualDataError(RuntimeError):
    """Public USD-M data was unavailable, malformed or incomplete."""


class BinanceUSDMPublicClient:
    """Bounded read-only reader for Binance USD-M linear perpetual public data."""

    venue_name = "Binance USD-M perpetual (cross-venue proxy)"

    def __init__(
        self,
        base_url: str = BASE_URL,
        transport: Callable | None = None,
        timeout: float = 15.0,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "fapi.binance.com"
            or parsed.port not in (None, 443)
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            raise ValueError("only the official Binance USD-M Futures public HTTPS host is allowed")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.base_url = base_url.rstrip("/")
        self.transport = transport
        self.timeout = timeout
        self._exchange_info: dict | None = None
        self._funding_info: dict[str, int] | None = None

    @staticmethod
    def _symbol(symbol: str) -> str:
        value = str(symbol).strip().upper()
        if not _SYMBOL.fullmatch(value):
            raise ValueError("invalid Binance USD-M symbol")
        return value

    def _get(self, endpoint: str, params: Mapping[str, object] | None = None):
        payload = fetch_json(
            url=self.base_url + endpoint,
            params=params or {},
            headers={},
            timeout=self.timeout,
            transport=self.transport,
        )
        if isinstance(payload, dict) and "code" in payload and payload.get("code") not in (0, "0"):
            raise BinancePerpetualDataError("Binance USD-M public API returned an error")
        return payload

    @staticmethod
    def _positive(value: object, name: str, *, allow_zero: bool = False) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            raise BinancePerpetualDataError(f"missing or invalid {name}") from None
        if not math.isfinite(parsed) or (parsed < 0 if allow_zero else parsed <= 0):
            raise BinancePerpetualDataError(f"invalid {name}")
        return parsed

    @staticmethod
    def _finite_number(value: object, name: str) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            raise BinancePerpetualDataError(f"missing or invalid {name}") from None
        if not math.isfinite(parsed):
            raise BinancePerpetualDataError(f"invalid {name}")
        return parsed

    def _get_exchange_info(self) -> dict:
        if self._exchange_info is None:
            payload = self._get("/fapi/v1/exchangeInfo")
            if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), list):
                raise BinancePerpetualDataError("Binance exchangeInfo response is malformed")
            self._exchange_info = payload
        return self._exchange_info

    def _get_funding_info(self) -> dict[str, int]:
        if self._funding_info is None:
            payload = self._get("/fapi/v1/fundingInfo")
            if not isinstance(payload, list):
                raise BinancePerpetualDataError("Binance fundingInfo response is malformed")
            intervals: dict[str, int] = {}
            for row in payload:
                if not isinstance(row, dict) or not row.get("symbol"):
                    continue
                try:
                    hours = float(row["fundingIntervalHours"])
                    minutes = int(round(hours * 60))
                except (KeyError, TypeError, ValueError):
                    raise BinancePerpetualDataError("Binance funding interval is malformed") from None
                if not math.isfinite(hours) or minutes < 1 or minutes > 480:
                    raise BinancePerpetualDataError("Binance funding interval is outside supported bounds")
                intervals[str(row["symbol"]).upper()] = minutes
            self._funding_info = intervals
        return self._funding_info

    def instrument_info(self, symbol: str) -> LinearPerpetualInstrument:
        market = self._symbol(symbol)
        rows = self._get_exchange_info()["symbols"]
        matches = [row for row in rows if isinstance(row, dict) and row.get("symbol") == market]
        if len(matches) != 1:
            raise BinancePerpetualDataError("exchangeInfo did not return exactly one requested symbol")
        row = matches[0]
        if (
            row.get("status") != "TRADING"
            or row.get("contractType") != "PERPETUAL"
            or row.get("quoteAsset") != "USDT"
            or row.get("marginAsset") != "USDT"
        ):
            raise BinancePerpetualDataError("symbol is not an active USDT-margined perpetual")

        filters = {
            str(item.get("filterType")): item
            for item in row.get("filters", [])
            if isinstance(item, dict)
        }
        price = filters.get("PRICE_FILTER", {})
        market_lot = filters.get("MARKET_LOT_SIZE") or {}
        try:
            for field in ("minQty", "maxQty", "stepSize"):
                self._positive(market_lot.get(field), "market " + field)
        except BinancePerpetualDataError:
            market_lot = filters.get("LOT_SIZE") or {}
        notional_filter = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {}
        min_notional_value = notional_filter.get("notional", notional_filter.get("minNotional"))
        if not all((price, market_lot)) or min_notional_value is None:
            raise BinancePerpetualDataError("exchangeInfo omitted required price, quantity or notional filters")

        funding_minutes = self._get_funding_info().get(market, 480)
        # The 2x value is a research ceiling only; Binance's account-specific
        # leverage brackets are not fetched by this public-data-only adapter.
        return LinearPerpetualInstrument(
            symbol=market,
            base_coin=str(row.get("baseAsset", "")).upper(),
            quote_coin=str(row.get("quoteAsset", "")).upper(),
            settle_coin=str(row.get("marginAsset", "")).upper(),
            status="Trading",
            contract_type="LinearPerpetual",
            min_quantity=self._positive(market_lot.get("minQty"), "market minQty"),
            max_market_quantity=self._positive(market_lot.get("maxQty"), "market maxQty"),
            quantity_step=self._positive(market_lot.get("stepSize"), "market stepSize"),
            min_notional=self._positive(min_notional_value, "minNotional"),
            tick_size=self._positive(price.get("tickSize"), "tickSize"),
            min_leverage=1.0,
            max_leverage=2.0,
            leverage_step=1.0,
            funding_interval_minutes=funding_minutes,
        )

    def _candles(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        endpoint: str,
        *,
        include_volume: bool,
    ) -> pd.DataFrame:
        market = self._symbol(symbol)
        if (
            isinstance(start_ms, bool)
            or isinstance(end_ms, bool)
            or not isinstance(start_ms, int)
            or not isinstance(end_ms, int)
            or start_ms <= 0
            or end_ms <= start_ms
        ):
            raise ValueError("start_ms/end_ms must be positive ordered integers")

        rows: list[list] = []
        cursor = start_ms
        for _ in range(50):
            payload = self._get(
                endpoint,
                {
                    "symbol": market,
                    "interval": _INTERVAL,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": _PAGE_LIMIT,
                },
            )
            if not isinstance(payload, list):
                raise BinancePerpetualDataError("Binance kline response is not a list")
            if not payload:
                break
            if any(not isinstance(item, list) or len(item) < 7 for item in payload):
                raise BinancePerpetualDataError("Binance kline row is malformed")
            page = sorted(payload, key=lambda item: int(item[0]))
            for item in page:
                opened = int(item[0])
                closed = int(item[6])
                if start_ms <= opened <= end_ms and closed < end_ms:
                    rows.append(item)
            last_open = int(page[-1][0])
            next_cursor = last_open + _INTERVAL_MS
            if next_cursor <= cursor:
                raise BinancePerpetualDataError("Binance kline pagination did not advance")
            cursor = next_cursor
            if len(page) < _PAGE_LIMIT or cursor > end_ms:
                break
        else:
            raise BinancePerpetualDataError("Binance kline pagination exceeded its bounded page limit")

        if not rows:
            raise BinancePerpetualDataError("Binance returned no completed 4-hour candles")
        unique = {int(row[0]): row for row in rows}
        ordered = [unique[key] for key in sorted(unique)]
        index = pd.to_datetime([int(row[0]) for row in ordered], unit="ms", utc=True)
        data = {
            "open": [float(row[1]) for row in ordered],
            "high": [float(row[2]) for row in ordered],
            "low": [float(row[3]) for row in ordered],
            "close": [float(row[4]) for row in ordered],
        }
        if include_volume:
            data["volume"] = [float(row[5]) for row in ordered]
        return pd.DataFrame(data, index=index)

    def klines(self, symbol: str, *, start_ms: int, end_ms: int) -> pd.DataFrame:
        return self._candles(
            symbol, start_ms, end_ms, "/fapi/v1/klines", include_volume=True
        )

    def mark_price_klines(self, symbol: str, *, start_ms: int, end_ms: int) -> pd.DataFrame:
        return self._candles(
            symbol, start_ms, end_ms, "/fapi/v1/markPriceKlines", include_volume=False
        )

    def funding_rates(
        self,
        symbol: str,
        *,
        start_ms: int,
        end_ms: int,
        interval_minutes: int = 480,
    ) -> pd.Series:
        market = self._symbol(symbol)
        if (
            isinstance(start_ms, bool)
            or isinstance(end_ms, bool)
            or not isinstance(start_ms, int)
            or not isinstance(end_ms, int)
            or start_ms <= 0
            or end_ms <= start_ms
        ):
            raise ValueError("start_ms/end_ms must be positive ordered integers")
        if isinstance(interval_minutes, bool) or not isinstance(interval_minutes, int) or not 1 <= interval_minutes <= 480:
            raise ValueError("interval_minutes must be an integer in [1, 480]")

        records: list[dict] = []
        cursor = start_ms
        for _ in range(50):
            payload = self._get(
                "/fapi/v1/fundingRate",
                {
                    "symbol": market,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": _FUNDING_PAGE_LIMIT,
                },
            )
            if not isinstance(payload, list):
                raise BinancePerpetualDataError("Binance funding history response is not a list")
            if not payload:
                break
            if any(
                not isinstance(item, dict) or "fundingTime" not in item or "fundingRate" not in item
                for item in payload
            ):
                raise BinancePerpetualDataError("Binance funding history row is malformed")
            page = sorted(payload, key=lambda item: int(item["fundingTime"]))
            for item in page:
                stamp = int(item["fundingTime"])
                if start_ms <= stamp <= end_ms:
                    records.append(
                        {"timestamp": stamp, "funding_rate": self._finite_number(
                            item["fundingRate"], "fundingRate"
                        )}
                    )
            last_time = int(page[-1]["fundingTime"])
            next_cursor = last_time + 1
            if next_cursor <= cursor:
                raise BinancePerpetualDataError("Binance funding pagination did not advance")
            cursor = next_cursor
            if len(page) < _FUNDING_PAGE_LIMIT or cursor > end_ms:
                break
        else:
            raise BinancePerpetualDataError("Binance funding pagination exceeded its bounded page limit")

        if not records:
            raise BinancePerpetualDataError("Binance returned no settled funding records")
        frame = pd.DataFrame(records).drop_duplicates("timestamp").sort_values("timestamp")
        index = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        rates = pd.Series(frame["funding_rate"].astype(float).to_numpy(), index=index, name="funding_rate")
        if len(rates) > 1:
            gaps = rates.index.to_series().diff().dropna()
            maximum_gap = pd.Timedelta(minutes=interval_minutes)
            if (gaps > maximum_gap).any():
                raise BinancePerpetualDataError("Binance funding history has a missing settlement interval")
        return rates
