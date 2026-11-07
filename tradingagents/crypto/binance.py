"""Binance's public market data: REST for history, WebSocket for the live tape.

Binance publishes its market data on hosts that need **no API key and no
registration**, and — importantly for this project — on *data-only* mirrors
(``data-api.binance.vision`` for REST, ``data-stream.binance.vision`` for the
stream) that are served even where the main ``api.binance.com`` host answers HTTP
451. That is why this module is the crypto market-data source: it is genuinely
free, needs no credential, and its data host is reachable.

What it serves, and what each is used for:

- ``/api/v3/exchangeInfo`` — every instrument, its status and its trading rules
  (min quantity, step size, tick size, min notional, order types). Read once and
  cached; see :mod:`tradingagents.crypto.instruments`.
- ``/api/v3/klines`` — OHLCV bars at 1m/5m/15m/1h/4h, with deep history, for the
  backtest and for the strategy's point-in-time window.
- ``/api/v3/ticker/24hr`` — the rolling 24-hour ticker (last, bid, ask, volume,
  high, low, change), for the live market view.
- ``/api/v3/depth`` — an order-book snapshot.
- the ``<symbol>@kline_<interval>`` and ``<symbol>@trade`` streams — the live tape.

Everything is normalized into :mod:`tradingagents.crypto.types` at the boundary,
so nothing downstream learns Binance's field names. The transport is injectable,
so every branch is exercised offline with a mocked response.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from .instruments import InstrumentRegistry, parse_exchange_info
from .transport import CryptoDataError, CryptoRateLimitedError, fetch_json
from .types import (
    BINANCE_INTERVALS,
    Candle,
    OrderBook,
    Ticker,
    candle_from_binance_kline,
)

REST_HOST = "https://data-api.binance.vision"
"""Binance's public **data** REST host: no key, reachable where the main host is not."""

WS_HOST = "wss://data-stream.binance.vision"
"""Binance's public **data** WebSocket host: no key, same reachability note."""

REQUEST_TIMEOUT = 15.0
_MAX_KLINES = 1000
"""Bars per kline request; the venue's own cap. A longer window is paged."""

_WS_INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
}


def _classify_status(status: int, body: str) -> None:
    if status == 429:
        raise CryptoRateLimitedError(f"the venue throttled the request (HTTP 429): {body}")
    if status in (418,):
        raise CryptoRateLimitedError(f"the venue rate-limited the IP (HTTP {status}): {body}")


def binance_symbol(symbol: str) -> str:
    """The Binance spelling (``BTCUSDT``) of a canonical ``BTC/USDT`` pair."""
    from .symbols import parse_pair

    return parse_pair(symbol).binance_symbol


class BinancePublicClient:
    """Binance's public market data over REST. No credential, ever.

    The client never sends a key and never reads one: every endpoint it calls is
    public. That is deliberate — a market-data credential is a liability with no
    benefit here, and keeping the data path credential-free means the live view
    cannot accidentally be pointed at a trading endpoint.
    """

    provider_name = "binance_public"
    kind = "vendor_rest"

    def __init__(
        self,
        *,
        base_url: str = REST_HOST,
        transport: Callable[..., Any] | None = None,
        timeout: float = REQUEST_TIMEOUT,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._transport = transport
        self._timeout = timeout
        self.last_error: str | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "crypto_data_provider": self.provider_name,
            "crypto_data_kind": self.kind,
            "crypto_data_rest_host": self._base_url,
            "crypto_data_requires_key": False,
            "crypto_data_last_error": self.last_error,
        }

    # -- low level --------------------------------------------------------

    def _get(self, path: str, params: dict | None = None) -> Any:
        try:
            payload = fetch_json(
                url=f"{self._base_url}{path}",
                params=params or {},
                timeout=self._timeout,
                transport=self._transport,
                classify_status=_classify_status,
            )
        except CryptoDataError:
            raise
        except Exception as exc:  # noqa: BLE001 - any failure is reported as data
            raise CryptoDataError(f"the request failed: {type(exc).__name__}") from None
        self.last_error = None
        return payload

    # -- instruments ------------------------------------------------------

    def exchange_info(self) -> InstrumentRegistry:
        """Every instrument and its trading rules, as a reloadable registry."""
        payload = self._get("/api/v3/exchangeInfo")
        if not isinstance(payload, dict):
            raise CryptoDataError("exchange info was not a JSON object")
        registry = InstrumentRegistry(
            rules=parse_exchange_info(payload),
            venue="binance",
            fetched_at=datetime.now(UTC).isoformat(),
        )
        return registry

    # -- bars -------------------------------------------------------------

    def klines(
        self,
        symbol: str,
        interval: str,
        *,
        limit: int = _MAX_KLINES,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Candle]:
        """OHLCV bars for ``symbol`` at ``interval``, oldest first.

        ``limit`` is capped at the venue's own maximum (1000); a longer window is
        paged internally so the caller gets the bars it asked for without knowing
        the cap. Paging goes *backward* from ``end`` (or now) because the venue
        returns the most recent bars when no ``startTime`` is given — paging forward
        from the newest bar would ask for bars that have not happened yet and get an
        empty page. A partial or malformed bar is dropped rather than guessed at.
        """
        if interval not in BINANCE_INTERVALS:
            raise CryptoDataError(
                f"unsupported interval {interval!r}; choose one of {sorted(BINANCE_INTERVALS)}"
            )
        market = binance_symbol(symbol)
        remaining = max(int(limit), 1)
        cursor_end = end
        collected: list[Candle] = []
        while remaining > 0:
            page = min(remaining, _MAX_KLINES)
            params: dict[str, Any] = {
                "symbol": market,
                "interval": BINANCE_INTERVALS[interval],
                "limit": page,
            }
            if start is not None:
                params["startTime"] = int(start.timestamp() * 1000)
            if cursor_end is not None:
                params["endTime"] = int(cursor_end.timestamp() * 1000)
            payload = self._get("/api/v3/klines", params)
            if not isinstance(payload, list) or not payload:
                break
            for row in payload:
                candle = candle_from_binance_kline(symbol, interval, row)
                if candle is not None:
                    collected.append(candle)
            remaining -= len(payload)
            if len(payload) < page:
                break
            # Move the window back one bar behind the oldest bar seen, so the next
            # page continues the history rather than repeating this one.
            oldest = min(c.open_time for c in collected)
            cursor_end = oldest - timedelta(milliseconds=1)
        deduped: dict[datetime, Candle] = {}
        for candle in collected:
            deduped.setdefault(candle.open_time, candle)
        return sorted(deduped.values(), key=lambda c: c.open_time)[-limit:]

    # -- live views -------------------------------------------------------

    def ticker_24h(self, symbols: Iterable[str]) -> dict[str, Ticker]:
        """The rolling 24-hour ticker for each symbol, keyed by canonical symbol."""
        from .symbols import parse_pair

        markets = [binance_symbol(s) for s in symbols]
        if not markets:
            return {}
        payload = self._get("/api/v3/ticker/24hr", {"symbols": _json_symbols(markets)})
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            raise CryptoDataError("the 24h ticker was not a JSON array")
        out: dict[str, Ticker] = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            ticker = Ticker.from_binance(row)
            pair = parse_pair(ticker.symbol)
            out[pair.canonical] = ticker
        return out

    def depth(self, symbol: str, *, limit: int = 20) -> OrderBook:
        """An order-book snapshot for ``symbol``."""
        payload = self._get(
            "/api/v3/depth", {"symbol": binance_symbol(symbol), "limit": int(limit)}
        )
        if not isinstance(payload, dict):
            raise CryptoDataError("the order book was not a JSON object")
        return OrderBook.from_binance(symbol, payload)

    def server_time(self) -> datetime:
        """The venue's own clock, for a skew check against the local clock."""
        payload = self._get("/api/v3/time")
        if not isinstance(payload, dict) or "serverTime" not in payload:
            raise CryptoDataError("the venue did not report its server time")
        return datetime.fromtimestamp(int(payload["serverTime"]) / 1000.0, tz=UTC)


def _json_symbols(markets: list[str]) -> str:
    """Binance wants a JSON array of symbols as a query-string value."""
    import json

    return json.dumps(markets, separators=(",", ":"))


def kline_stream_names(symbols: Iterable[str], interval: str) -> list[str]:
    """The ``<market>@kline_<interval>`` stream names for a set of symbols."""
    return [f"{binance_symbol(s).lower()}@kline_{BINANCE_INTERVALS[interval]}" for s in symbols]


def trade_stream_names(symbols: Iterable[str]) -> list[str]:
    """The ``<market>@trade`` stream names for a set of symbols."""
    return [f"{binance_symbol(s).lower()}@trade" for s in symbols]


def ticker_stream_names(symbols: Iterable[str]) -> list[str]:
    """The ``<market>@ticker`` stream names for a set of symbols.

    The ``@trade`` stream carries only a price and a quantity, but the live view
    also needs the bid, the ask and the rolling 24-hour statistics. The
    ``@ticker`` stream is the one that carries them, and it pushes on the venue's
    own cadence, so the dashboard does not have to poll for them.
    """
    return [f"{binance_symbol(s).lower()}@ticker" for s in symbols]


__all__ = [
    "REQUEST_TIMEOUT",
    "REST_HOST",
    "WS_HOST",
    "BinancePublicClient",
    "binance_symbol",
    "kline_stream_names",
    "ticker_stream_names",
    "trade_stream_names",
]
