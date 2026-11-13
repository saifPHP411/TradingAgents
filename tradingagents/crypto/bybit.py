"""Bybit's public market data: REST for the live tape (demo and live).

Bybit publishes its market data on hosts that need **no API key and no
registration** for public market data. The demo environment uses the same
market data endpoints as the live environment, but with different base URLs.

What it serves, and what each is used for:

- ``/v5/market/tickers`` — the rolling 24-hour ticker (last, bid, ask, volume,
  high, low, change), for the live market view.
- ``/v5/market/instruments-info`` — every instrument, its status and its trading
  rules (min quantity, step size, tick size, min notional, order types). Read
  once and cached; see :mod:`tradingagents.crypto.instruments`.

Everything is normalized into :mod:`tradingagents.crypto.types` at the boundary,
so nothing downstream learns Bybit's field names. The transport is injectable,
so every branch is exercised offline with a mocked response.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

from .instruments import STATUS_UNKNOWN, InstrumentRegistry
from .symbols import SymbolError, parse_pair
from .transport import CryptoDataError, CryptoRateLimitedError, fetch_json
from .types import CANDLE_INTERVALS, Candle, candle_from_bybit_kline

DEMO_REST_HOST = "https://api-demo.bybit.com"
"""Bybit's **demo** REST host: no key needed for public market data."""

LIVE_REST_HOST = "https://api.bybit.com"
"""Bybit's **live** REST host: no key needed for public market data."""

REQUEST_TIMEOUT = 15.0

_BYBIT_KLINE_INTERVALS = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "1h": "60",
    "4h": "240",
}



def _classify_status(status: int, body: str) -> None:
    if status == 429:
        raise CryptoRateLimitedError(f"the venue throttled the request (HTTP 429): {body}")


def bybit_symbol(symbol: str) -> str:
    """The Bybit spelling (``BTCUSDT``) of a canonical ``BTC/USDT`` pair."""
    from .symbols import parse_pair

    return parse_pair(symbol).binance_symbol  # Bybit uses same format as Binance


class BybitPublicClient:
    """Bybit's public market data over REST. No credential, ever.

    The client never sends a key and never reads one: every endpoint it calls is
    public. That is deliberate — a market-data credential is a liability with no
    benefit here, and keeping the data path credential-free means the live view
    cannot accidentally be pointed at a trading endpoint.
    """

    provider_name = "bybit_public"
    kind = "vendor_rest"

    def __init__(
        self,
        *,
        base_url: str = DEMO_REST_HOST,
        transport: Callable[..., Any] | None = None,
        timeout: float = REQUEST_TIMEOUT,
        diagnose: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._transport = transport
        self._timeout = timeout
        self._diagnose = diagnose
        self.last_error: str | None = None
        self.last_diagnostics: list[dict[str, object]] = []
        """The most recent request records, for the dashboard's status view."""

    def describe(self) -> dict[str, Any]:
        return {
            "crypto_data_provider": self.provider_name,
            "crypto_data_kind": self.kind,
            "crypto_data_rest_host": self._base_url,
            "crypto_data_requires_key": False,
            "crypto_data_last_error": self.last_error,
        }

    # -- low level --------------------------------------------------------

    def _record_diagnostic(self, record: dict[str, object]) -> None:
        """Keep the sanitized request record and pass it to the caller's sink."""
        self.last_diagnostics.append(record)
        del self.last_diagnostics[:-20]
        if self._diagnose is not None:
            # A sink that fails must not break the read.
            with contextlib.suppress(Exception):
                self._diagnose(record)

    def _get(self, path: str, params: dict | None = None) -> Any:
        try:
            payload = fetch_json(
                url=f"{self._base_url}{path}",
                params=params or {},
                timeout=self._timeout,
                transport=self._transport,
                classify_status=_classify_status,
                diagnose=self._record_diagnostic,
            )
        except CryptoDataError:
            raise
        except Exception as exc:  # noqa: BLE001 - any failure is reported as data
            raise CryptoDataError(f"the request failed: {type(exc).__name__}") from None
        self.last_error = None
        return payload

    # -- instruments ------------------------------------------------------

    def instruments_info(self, category: str = "spot") -> InstrumentRegistry:
        """Every instrument and its trading rules, as a reloadable registry."""
        payload = self._get("/v5/market/instruments-info", {"category": category})
        if not isinstance(payload, dict):
            raise CryptoDataError("instruments info was not a JSON object")
        result = payload.get("result", {})
        if not isinstance(result, dict):
            raise CryptoDataError("instruments info missing 'result'")
        instrument_list = result.get("list", [])
        if not isinstance(instrument_list, list):
            raise CryptoDataError("instruments info missing 'list'")

        # Bybit's format is different, need to adapt
        rules = self._parse_bybit_instruments(instrument_list)
        registry = InstrumentRegistry(
            rules=rules,
            venue="bybit",
            fetched_at=datetime.now(UTC).isoformat(),
        )
        return registry

    def _parse_bybit_instruments(self, instrument_list: list) -> dict:
        """Parse Bybit instruments into trading rules.

        The venue's Spot ``lotSizeFilter`` carries **no** ``qtyStep``: the step
        is ``basePrecision``, the precision of the base coin (the documented
        BTCUSDT Spot example shows ``basePrecision`` ``0.000001``). The step is
        read from ``qtyStep`` when a non-Spot shape carries it, and from
        ``basePrecision`` otherwise.

        ``minOrderQty`` and ``maxOrderQty`` are **deprecated** for Spot: the
        venue documents that it no longer checks them and checks ``minOrderAmt``
        instead, so ``minOrderAmt`` is the minimum notional that is enforced.
        The deprecated pair is still read, because a venue that does state a
        maximum is worth knowing, but it is not what the order check uses.

        Every bound is the venue's own number. An absent bound is ``None``
        (unstated), never ``0.0`` (a stated zero), so the order check can tell
        "the venue did not say" from "the venue said zero".
        """
        from .instruments import (
            TradingRules,
            _as_float,
            _decimals,
            _fractionable_from_step,
        )

        rules = {}
        for entry in instrument_list:
            if not isinstance(entry, dict):
                continue
            symbol = str(entry.get("symbol", "")).upper()
            if not symbol:
                continue
            base = str(entry.get("baseCoin", "")).upper()
            quote = str(entry.get("quoteCoin", "")).upper()
            if not base or not quote:
                continue
            status = str(entry.get("status", ""))
            lot_filter = entry.get("lotSizeFilter", {})
            price_filter = entry.get("priceFilter", {})

            # The step is the venue's own decimal string: ``basePrecision`` for
            # Spot, ``qtyStep`` when a shape carries one. It is kept as the
            # string the venue sent, so the decimals are the venue's own and
            # not a float's rendering of them.
            step_text = str(
                lot_filter.get("qtyStep") or lot_filter.get("basePrecision") or ""
            )
            tick_text = str(price_filter.get("tickSize") or "")
            rules[f"{base}/{quote}"] = TradingRules(
                symbol=f"{base}/{quote}",
                base_asset=base,
                quote_asset=quote,
                status=status.upper() or STATUS_UNKNOWN,
                min_quantity=_as_float(lot_filter.get("minOrderQty")),
                max_quantity=_as_float(lot_filter.get("maxOrderQty")),
                quantity_step=_as_float(step_text) if step_text else None,
                price_tick=_as_float(tick_text) if tick_text else None,
                min_notional=_as_float(lot_filter.get("minOrderAmt")),
                order_types=("market", "limit"),
                quote_precision=_decimals(tick_text) if tick_text else 0,
                base_precision=_decimals(step_text) if step_text else 0,
                # A Spot pair trades fractional quantities whenever the step
                # is smaller than one unit; an integral step means whole units.
                fractionable=_fractionable_from_step(step_text),
            )
        return rules

    # -- historical candles -----------------------------------------------

    def klines(
        self,
        symbol: str,
        interval: str,
        *,
        limit: int = 1000,
        start: int | None = None,
        end: int | None = None,
    ) -> list[Candle]:
        """Fetch Spot candles, paging backwards at Bybit's 1000-row cap.

        Bybit returns each page newest-first. Pages are walked backwards with an
        exclusive end timestamp, de-duplicated by open time, then returned oldest
        first so every downstream frame has one deterministic order.
        """
        if interval not in CANDLE_INTERVALS or interval not in _BYBIT_KLINE_INTERVALS:
            raise CryptoDataError(f"unsupported Bybit kline interval {interval!r}")
        try:
            wanted = int(limit)
        except (TypeError, ValueError):
            raise CryptoDataError(f"invalid Bybit kline limit {limit!r}") from None
        if wanted < 1:
            raise CryptoDataError("Bybit kline limit must be positive")
        lower = int(start) if start is not None else None
        upper = int(end) if end is not None else None
        cursor_end = upper
        market = bybit_symbol(symbol)
        collected: dict[datetime, Candle] = {}

        while len(collected) < wanted:
            page_limit = min(1000, wanted - len(collected))
            params: dict[str, Any] = {
                "category": "spot",
                "symbol": market,
                "interval": _BYBIT_KLINE_INTERVALS[interval],
                "limit": page_limit,
            }
            if lower is not None:
                params["start"] = lower
            if cursor_end is not None:
                params["end"] = cursor_end
            payload = self._get("/v5/market/kline", params)
            if not isinstance(payload, dict):
                raise CryptoDataError("Bybit kline response was not a JSON object")
            ret_code = payload.get("retCode", 0)
            if ret_code not in (0, "0", None):
                ret_message = str(payload.get("retMsg") or "unknown venue error")
                raise CryptoDataError(
                    f"Bybit rejected the kline request ({ret_code}): {ret_message}"
                )
            result = payload.get("result", {})
            rows = result.get("list", []) if isinstance(result, dict) else None
            if not isinstance(rows, list):
                raise CryptoDataError("Bybit kline response is missing result.list")
            if not rows:
                break

            parsed = [
                candle
                for row in rows
                if (candle := candle_from_bybit_kline(symbol, interval, row)) is not None
            ]
            for candle in parsed:
                open_ms = int(candle.open_time.timestamp() * 1000)
                if lower is not None and open_ms < lower:
                    continue
                if upper is not None and open_ms > upper:
                    continue
                collected[candle.open_time] = candle

            if len(rows) < page_limit or not parsed:
                break
            oldest_ms = min(int(c.open_time.timestamp() * 1000) for c in parsed)
            next_end = oldest_ms - 1
            if lower is not None and next_end < lower:
                break
            if cursor_end is not None and next_end >= cursor_end:
                break
            cursor_end = next_end

        ordered = [collected[key] for key in sorted(collected)]
        return ordered[-wanted:]

    # -- live views -------------------------------------------------------

    def tickers(self, symbols: Iterable[str]) -> dict[str, dict]:
        """The rolling 24-hour ticker for each symbol.

        The result is keyed by both spellings of every pair: the canonical
        ``BTC/USDT`` and the venue's own ``BTCUSDT``. A caller that asks for a
        ticker by either spelling must find it, because the two spellings name
        the same instrument.

        The request itself is a single batched call.
        """
        markets = [bybit_symbol(s) for s in symbols]
        if not markets:
            return {}

        params = {"category": "spot", "symbol": ",".join(markets)}
        payload = self._get("/v5/market/tickers", params)

        if not isinstance(payload, dict):
            raise CryptoDataError("the tickers response was not a JSON object")

        result = payload.get("result", {})
        if not isinstance(result, dict):
            raise CryptoDataError("the tickers response missing 'result'")

        ticker_list = result.get("list", [])
        if not isinstance(ticker_list, list):
            raise CryptoDataError("the tickers response missing 'list'")

        # Bybit V5 puts the observation timestamp on the response envelope
        # ("time"), not on each item in result.list. The execution-price
        # freshness guard consumes a ticker-shaped mapping, so preserve that
        # venue timestamp on each copied row. Do not substitute the local clock:
        # if Bybit omits its timestamp, the downstream guard must still fail closed.
        response_time = payload.get("time")
        out: dict[str, dict] = {}
        for row in ticker_list:
            if not isinstance(row, dict):
                continue
            sym = str(row.get("symbol", "")).upper()
            if not sym:
                continue
            normalized_row = dict(row)
            if normalized_row.get("time") in (None, "") and response_time not in (None, ""):
                normalized_row["time"] = response_time
            # Key by both canonical and venue spelling. Both aliases point to
            # the same normalized row, carrying the same response observation.
            out[sym] = normalized_row
            # Also try to parse canonical
            try:
                pair = parse_pair(sym)
                out[pair.canonical] = normalized_row
            except SymbolError:
                pass

        return out


__all__ = [
    "REQUEST_TIMEOUT",
    "DEMO_REST_HOST",
    "LIVE_REST_HOST",
    "BybitPublicClient",
    "bybit_symbol",
]
